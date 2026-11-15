"""複数の項目で使う AI・解析の部品。

- DepthEstimator: 奥行き（Depth Anything V2 Small。Apache-2.0）。0〜1 で、1 が手前。
- PersonSegmenter: 人物の切り抜き（MediaPipe の selfie / multiclass）。0〜1 の地図。
- optical_flow / RaftFlow / flow_to_color: 画素ごとの動き（OpenCV の DIS・Farneback、torchvision の RAFT）。
- global_motion: 画面全体の動き（特徴点を追って、移動・回転・拡大を求める）。
- ImageText: 画像と言葉の照合（SigLIP 2。多言語対応で日本語の言葉も使える。Apache-2.0）。
"""

from __future__ import annotations

from functools import cache
from typing import Any

import cv2
import numpy as np

from .core import VfxError
from .models import hf_repo, mp_detect, mp_task, torch_setup

# ── 奥行き ────────────────────────────────────────────────────────────────


class DepthEstimator:
    """Depth Anything V2 Small で、相対的な奥行き（0〜1、1 が手前）を求める。"""

    def __init__(self) -> None:
        self.torch = torch_setup()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        repo = hf_repo("depth")
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.net = AutoModelForDepthEstimation.from_pretrained(repo).eval()

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        inputs = self.proc(images=rgb, return_tensors="pt")
        pred = self.net(**inputs).predicted_depth  # 大きいほど手前（逆深度）
        pred = self.torch.nn.functional.interpolate(
            pred[:, None], size=(h, w), mode="bicubic", align_corners=False
        )
        d = pred[0, 0].numpy().astype(np.float32)
        lo, hi = np.percentile(d, 1), np.percentile(d, 99)
        return np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)


# ── 人物の切り抜き ─────────────────────────────────────────────────────────


class PersonSegmenter:
    """MediaPipe で人物（または髪・顔・服など）の地図を求める。

    kind='selfie': 人物全体（速い）。kind='multiclass': 0 背景 1 髪 2 体の肌 3 顔の肌 4 服 5 その他（小物）。
    kind='hair': 髪だけ。動画は video=True にして、時刻（ミリ秒）つきで渡す。
    """

    def __init__(self, kind: str = "selfie", *, video: bool = False) -> None:
        self.kind = kind
        self.video = video
        multi = kind == "multiclass"
        self.task = mp_task(
            kind,
            video=video,
            output_confidence_masks=not multi,
            output_category_mask=multi,
        )

    def __call__(self, frame: np.ndarray, t_ms: int | None = None) -> np.ndarray:
        # MediaPipe の結果の画素は、次の呼び出しで解放される。写さずに返すと、取っておいた地図が壊れたり
        # 異常終了したりした（2026-11-14 実測）ので、必ず写して返す。
        res = mp_detect(self.task, "segment", frame, t_ms if self.video else None)
        if self.kind == "multiclass":
            m = res.category_mask.numpy_view()
            return np.array(np.squeeze(m), dtype=np.uint8, order="C", copy=True)
        masks = res.confidence_masks
        m = masks[-1].numpy_view() if len(masks) > 1 else masks[0].numpy_view()
        return np.array(np.squeeze(m), dtype=np.float32, order="C", copy=True)


# ── 動き ──────────────────────────────────────────────────────────────────


@cache
def _dis() -> Any:
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def optical_flow(
    prev_gray: np.ndarray, gray: np.ndarray, method: str = "dis"
) -> np.ndarray:
    """前のコマ → 今のコマの、画素ごとの動き（H×W×2、px）。method は dis / farneback。"""
    if method == "dis":
        return _dis().calc(prev_gray, gray, None)
    if method == "farneback":
        return cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 4, 21, 3, 5, 1.1, 0
        )
    raise VfxError(f"知らない方法です: {method}（dis / farneback / raft）")


class RaftFlow:
    """torchvision の RAFT（精度が高いが CPU では遅い）。

    重みは small・large とも C_T_V2（FlyingChairs と FlyingThings3D だけで学習）。large の torchvision の既定
    （C_T_SKHT_V2）は KITTI（データは CC BY-NC-SA 3.0）でも学習しているので使わない。
    """

    def __init__(self, large: bool = False) -> None:
        self.torch = torch_setup()
        from torchvision.models import optical_flow as of

        if large:
            self.weights = of.Raft_Large_Weights.C_T_V2
            self.net = of.raft_large(weights=self.weights).eval()
        else:
            self.weights = of.Raft_Small_Weights.DEFAULT
            self.net = of.raft_small(weights=self.weights).eval()

    def __call__(self, prev_bgr: np.ndarray, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        h8, w8 = (h + 7) // 8 * 8, (w + 7) // 8 * 8

        def prep(img: np.ndarray) -> Any:
            rgb = cv2.cvtColor(cv2.resize(img[:, :, :3], (w8, h8)), cv2.COLOR_BGR2RGB)
            t = self.torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
            return t * 2 - 1

        flows = self.net(prep(prev_bgr), prep(bgr))
        f = flows[-1][0].permute(1, 2, 0).numpy()
        f = cv2.resize(f, (w, h))
        f[..., 0] *= w / w8
        f[..., 1] *= h / h8
        return f.astype(np.float32)


def flow_to_color(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """動きを色で表す（色相 = 向き、明るさ = 速さ）。BGR を返す。"""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    top = max_mag or float(np.percentile(mag, 99)) or 1.0
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / top * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def track_points(
    prev_gray: np.ndarray, gray: np.ndarray, pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """点を前のコマから今のコマへ追う（往復で確かめ、ずれの大きい点は捨てる）。(新しい位置, 有効か)"""
    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk)
    fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
    return p1.reshape(-1, 2), ok


def global_motion(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    *,
    max_points: int = 400,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, int]:
    """画面全体の動き（2×3 の相似変換: 移動・回転・拡大）と、使えた点の数を返す。求まらなければ None。"""
    pts = cv2.goodFeaturesToTrack(
        prev_gray, max_points, 0.01, 8, mask=mask, blockSize=7
    )
    if pts is None or len(pts) < 8:
        return None, 0
    p1, ok = track_points(prev_gray, gray, pts.reshape(-1, 2))
    if ok.sum() < 8:
        return None, int(ok.sum())
    m, inl = cv2.estimateAffinePartial2D(
        pts.reshape(-1, 2)[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    return m, int(inl.sum()) if inl is not None else 0


def decompose_similarity(m: np.ndarray) -> tuple[float, float, float, float]:
    """2×3 の相似変換を (dx, dy, 回転[度], 拡大率) に分ける。"""
    dx, dy = float(m[0, 2]), float(m[1, 2])
    angle = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    return dx, dy, angle, scale


# ── 画像と言葉 ────────────────────────────────────────────────────────────


class ImageText:
    """SigLIP 2（多言語）で、画像と言葉の近さを測る。日本語の言葉もそのまま使える。"""

    def __init__(self) -> None:
        self.torch = torch_setup()
        from transformers import AutoImageProcessor, AutoModel, GemmaTokenizerFast

        repo = hf_repo("clip")
        # transformers 5.2.0 の AutoProcessor / AutoTokenizer はこのモデルで失敗する（2026-11-14 実測）ので、個別に読む
        self.img_proc = AutoImageProcessor.from_pretrained(repo)
        self.tok = GemmaTokenizerFast.from_pretrained(repo)
        self.net = AutoModel.from_pretrained(repo).eval()

    def embed_images(self, frames: list[np.ndarray], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(frames), batch):
            rgb = [
                cv2.cvtColor(f[:, :, :3], cv2.COLOR_BGR2RGB)
                for f in frames[i : i + batch]
            ]
            inputs = self.img_proc(images=rgb, return_tensors="pt")
            e = self.net.get_image_features(**inputs)
            e = e if isinstance(e, self.torch.Tensor) else e.pooler_output
            out.append(self.torch.nn.functional.normalize(e, dim=-1).numpy())
        return np.concatenate(out)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # SigLIP は 64 トークンに詰めて学習しているので、同じ形で渡す
        inputs = self.tok(
            texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        e = self.net.get_text_features(**inputs)
        e = e if isinstance(e, self.torch.Tensor) else e.pooler_output
        return self.torch.nn.functional.normalize(e, dim=-1).numpy()

    def probs(self, img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
        """画像×言葉の「当てはまる確率」（0〜1。SigLIP は言葉ごとに独立に判定する）。"""
        scale = float(self.net.logit_scale.exp())
        bias = float(self.net.logit_bias)
        logits = img_emb @ txt_emb.T * scale + bias
        return 1 / (1 + np.exp(-logits))
