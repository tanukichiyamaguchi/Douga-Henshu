"""複数の項目で使う検出器: 物体（人物を含む）・顔。

- 物体: torchvision の Faster R-CNN v2（精度重視、CPU で 1 コマ数秒）か、MediaPipe の EfficientDet-Lite2（速さ重視）。
  どちらも COCO の 80 種類（人・車・カップ・本など）。
- 顔: OpenCV の YuNet（顔の枠と、目・鼻・口の端の 5 点）。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import cv2
import numpy as np

from .models import fetch, mp_task, torch_setup

# COCO の 80 種類の日本語名
COCO_JA = {
    "person": "人", "bicycle": "自転車", "car": "車", "motorcycle": "バイク", "airplane": "飛行機", "bus": "バス",
    "train": "電車", "truck": "トラック", "boat": "船", "traffic light": "信号機", "fire hydrant": "消火栓",
    "stop sign": "止まれの標識", "parking meter": "パーキングメーター", "bench": "ベンチ", "bird": "鳥", "cat": "猫",
    "dog": "犬", "horse": "馬", "sheep": "羊", "cow": "牛", "elephant": "象", "bear": "熊", "zebra": "シマウマ",
    "giraffe": "キリン", "backpack": "リュック", "umbrella": "傘", "handbag": "ハンドバッグ", "tie": "ネクタイ",
    "suitcase": "スーツケース", "frisbee": "フリスビー", "skis": "スキー板", "snowboard": "スノーボード",
    "sports ball": "ボール", "kite": "凧", "baseball bat": "バット", "baseball glove": "グローブ",
    "skateboard": "スケートボード", "surfboard": "サーフボード", "tennis racket": "ラケット", "bottle": "ボトル",
    "wine glass": "ワイングラス", "cup": "カップ", "fork": "フォーク", "knife": "ナイフ", "spoon": "スプーン",
    "bowl": "ボウル", "banana": "バナナ", "apple": "リンゴ", "sandwich": "サンドイッチ", "orange": "オレンジ",
    "broccoli": "ブロッコリー", "carrot": "ニンジン", "hot dog": "ホットドッグ", "pizza": "ピザ", "donut": "ドーナツ",
    "cake": "ケーキ", "chair": "椅子", "couch": "ソファ", "potted plant": "鉢植え", "bed": "ベッド",
    "dining table": "テーブル", "toilet": "トイレ", "tv": "テレビ", "laptop": "ノートパソコン", "mouse": "マウス",
    "remote": "リモコン", "keyboard": "キーボード", "cell phone": "スマートフォン", "microwave": "電子レンジ",
    "oven": "オーブン", "toaster": "トースター", "sink": "流し台", "refrigerator": "冷蔵庫", "book": "本",
    "clock": "時計", "vase": "花瓶", "scissors": "はさみ", "teddy bear": "ぬいぐるみ", "hair drier": "ドライヤー",
    "toothbrush": "歯ブラシ",
}  # fmt: skip


@dataclass
class Det:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2（px）
    score: float
    label: str  # 英語（COCO の名前）

    @property
    def ja(self) -> str:
        return COCO_JA.get(self.label, self.label)

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.box
        return round(x1), round(y1), round(x2 - x1), round(y2 - y1)


class ObjectDetector:
    """物体検出。model='accurate'（Faster R-CNN v2）か 'fast'（EfficientDet-Lite2）。"""

    def __init__(
        self,
        model: str = "accurate",
        score: float = 0.5,
        classes: list[str] | None = None,
    ) -> None:
        self.model = model
        self.score = score
        self.classes = set(classes) if classes else None
        if model == "fast":
            self.task = mp_task(
                "objects", video=False, score_threshold=min(score, 0.3), max_results=100
            )
        else:
            torch = torch_setup()
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
            )

            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.net = fasterrcnn_resnet50_fpn_v2(
                weights=weights, box_score_thresh=min(score, 0.3)
            ).eval()
            self.names = weights.meta["categories"]
            self.torch = torch

    def __call__(self, frame: np.ndarray) -> list[Det]:
        dets: list[Det] = []
        if self.model == "fast":
            from .models import mp_image

            res = self.task.detect(mp_image(frame))
            for d in res.detections:
                c = d.categories[0]
                b = d.bounding_box
                dets.append(
                    Det(
                        (
                            b.origin_x,
                            b.origin_y,
                            b.origin_x + b.width,
                            b.origin_y + b.height,
                        ),
                        c.score,
                        c.category_name,
                    )
                )
        else:
            rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
            x = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            out = self.net([x])[0]
            for b, s, lab in zip(
                out["boxes"].tolist(),
                out["scores"].tolist(),
                out["labels"].tolist(),
                strict=True,
            ):
                dets.append(Det(tuple(b), s, self.names[lab]))
        return [
            d
            for d in dets
            if d.score >= self.score
            and (self.classes is None or d.label in self.classes)
        ]


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h
    score: float
    points: (
        np.ndarray
    )  # 5×2: 右目・左目・鼻先・口の右端・口の左端（画像の向きで見た左右）
    raw: np.ndarray  # YuNet の出力 1 行（顔の特徴量の計算に使う）


class FaceDetector:
    """YuNet で顔を見つける。"""

    def __init__(self, score: float = 0.7) -> None:
        self.net = cv2.FaceDetectorYN.create(
            str(fetch("yunet")), "", (320, 320), score, 0.3, 5000
        )
        self.size: tuple[int, int] | None = None

    def __call__(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        if self.size != (w, h):
            self.net.setInputSize((w, h))
            self.size = (w, h)
        _, faces = self.net.detect(np.ascontiguousarray(frame[:, :, :3]))
        out = []
        for f in faces if faces is not None else []:
            out.append(
                Face(
                    tuple(f[:4].tolist()),
                    float(f[14]),
                    f[4:14].reshape(5, 2).copy(),
                    f.copy(),
                )
            )
        return out


@cache
def face_recognizer() -> Any:
    """SFace（顔の特徴量。同じ人かどうかを比べるのに使う）。"""
    return cv2.FaceRecognizerSF.create(str(fetch("sface")), "")


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """枠（x1,y1,x2,y2）どうしの重なりの割合。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def color_for(i: int) -> tuple[int, int, int]:
    """番号ごとに見分けやすい色（BGR）。"""
    hue = (i * 47) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)
