"""AI モデルの取得とキャッシュ、読み込み。

- モデルは初めて使う時に取得し、~/.cache/douga-vfx/（Hugging Face のものは ~/.cache/huggingface/）に置く。
  クラウド環境はセッションごとに作り直されるので、新しいセッションでは取り直しになる。
- 通信は curl で行う（このクラウド環境のプロキシと証明書の設定をそのまま使うため。fetch_fonts.py と同じ）。
- ここに載せたのは、商用で使えるライセンスのものだけ（2026-11-14 に各配布元で確認）。
"""

from __future__ import annotations

import atexit
import hashlib
import os
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .core import VfxError, log

CACHE = Path(os.environ.get("VFX_CACHE", Path.home() / ".cache" / "douga-vfx"))

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MP = "https://storage.googleapis.com/mediapipe-models/"
GH = "https://github.com/"


@dataclass(frozen=True)
class ModelFile:
    url: str
    license: str
    note: str = ""
    sha256: str | None = None


# 単体のモデルファイル（キー → 取得先）。sha256 は取得した物と照合する（配布元で中身が変わったら止まる）。
FILES: dict[str, ModelFile] = {
    # OpenCV 公式の小さなモデル（OpenCV の dnn で動く）
    "yunet": ModelFile(
        ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx", "MIT", "顔検出"
    ),
    "sface": ModelFile(
        ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "Apache-2.0",
        "顔の特徴量",
    ),
    "lpd_yunet": ModelFile(
        ZOO
        + "license_plate_detection_yunet/license_plate_detection_lpd_yunet_2023mar.onnx",
        "Apache-2.0",
        "ナンバープレート検出",
    ),
    "vittrack": ModelFile(
        ZOO + "object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx",
        "Apache-2.0",
        "追跡",
    ),
    "youtu_reid": ModelFile(
        ZOO + "person_reid_youtureid/person_reid_youtu_2021nov.onnx",
        "Apache-2.0",
        "人物の特徴量",
    ),
    # MediaPipe（Google。Apache-2.0）
    "mp_face_detector": ModelFile(
        MP
        + "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "Apache-2.0",
    ),
    "mp_face_landmarker": ModelFile(
        MP + "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "Apache-2.0",
    ),
    "mp_hand_landmarker": ModelFile(
        MP + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "Apache-2.0",
    ),
    "mp_gesture": ModelFile(
        MP + "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        "Apache-2.0",
    ),
    "mp_pose_lite": ModelFile(
        MP + "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "Apache-2.0",
    ),
    "mp_pose_full": ModelFile(
        MP + "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "Apache-2.0",
    ),
    "mp_pose_heavy": ModelFile(
        MP
        + "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "Apache-2.0",
    ),
    "mp_selfie": ModelFile(
        MP + "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        "Apache-2.0",
    ),
    "mp_multiclass": ModelFile(
        MP
        + "image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        "Apache-2.0",
    ),
    "mp_hair": ModelFile(
        MP + "image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
        "Apache-2.0",
    ),
    "mp_objects": ModelFile(
        MP + "object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite",
        "Apache-2.0",
    ),
    "mp_embedder": ModelFile(
        MP + "image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite",
        "Apache-2.0",
    ),
    # GitHub のリリースにある重み
    "lama": ModelFile(
        GH + "Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "Apache-2.0",
        "LaMa（穴埋め）",
    ),
    "realesrgan": ModelFile(
        GH + "xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "BSD-3-Clause",
        "Real-ESRGAN（超解像）",
    ),
    "gfpgan": ModelFile(
        GH + "TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "Apache-2.0",
        "GFPGAN（顔の復元）",
    ),
    "birefnet_lite": ModelFile(
        GH
        + "danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "MIT",
        "BiRefNet（切り抜き）",
    ),
    "birefnet_portrait": ModelFile(
        GH
        + "danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx",
        "MIT",
        "BiRefNet（人物）",
    ),
}

# Hugging Face のモデル（transformers で読む）。キー → (リポジトリ, ライセンス)
HF: dict[str, tuple[str, str]] = {
    "depth": ("depth-anything/Depth-Anything-V2-Small-hf", "Apache-2.0"),
    "clip": ("google/siglip2-base-patch16-224", "Apache-2.0"),
    "xclip": ("microsoft/xclip-base-patch32", "MIT"),
    "dino": ("facebook/dinov2-small", "Apache-2.0"),
    "fer": ("trpakov/vit-face-expression", "Apache-2.0"),
    "florence": ("florence-community/Florence-2-base", "MIT"),
    "gdino": ("IDEA-Research/grounding-dino-tiny", "Apache-2.0"),
    "panoptic": ("facebook/detr-resnet-50-panoptic", "Apache-2.0"),
    "ade": ("openmmlab/upernet-convnext-small", "MIT"),
    "sam2": ("facebook/sam2.1-hiera-tiny", "Apache-2.0"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str) -> Path:
    """FILES に載せたモデルファイルを（無ければ取得して）キャッシュの場所を返す。"""
    if key not in FILES:
        raise VfxError(f"知らないモデルです: {key}")
    return fetch_spec(FILES[key])


def fetch_spec(spec: ModelFile) -> Path:
    """ModelFile を（無ければ取得して）キャッシュの場所を返す。各モジュールが自分で ModelFile を持ってもよい。"""
    name = Path(urlparse(spec.url).path).name
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    CACHE.mkdir(parents=True, exist_ok=True)
    # 同時に同じモデルを取りに行っても壊れないよう、作業用の名前はプロセスごとに変える
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    log(f"モデルを取得します: {name}（{spec.license}）")
    cmd = ["curl", "-sS", "-L", "--fail", "--retry", "3", "-o", str(tmp), spec.url]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise VfxError(f"モデルを取得できません: {spec.url}\n{result.stderr.strip()}")
    if spec.sha256 and _sha256(tmp) != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise VfxError(
            f"取得したモデルの中身が想定と違います（配布元で変わった可能性）: {spec.url}"
        )
    tmp.replace(dest)
    return dest


# ── PyTorch ───────────────────────────────────────────────────────────────


@cache
def torch_setup() -> Any:
    """PyTorch を CPU の全コアで使う。"""
    import torch

    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_grad_enabled(False)
    return torch


def hf_repo(key: str) -> str:
    return HF[key][0]


# ── MediaPipe ─────────────────────────────────────────────────────────────


def mp_image(frame_bgr: np.ndarray) -> Any:
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def mp_task(kind: str, *, video: bool, **options: Any) -> Any:
    """MediaPipe の作業（顔・手・姿勢・物体・切り抜きなど）を作る。video=True なら動画用（時刻つき）。"""
    from mediapipe.tasks.python import BaseOptions, vision

    table = {
        "face_detector": (
            "mp_face_detector",
            vision.FaceDetector,
            vision.FaceDetectorOptions,
        ),
        "face_landmarker": (
            "mp_face_landmarker",
            vision.FaceLandmarker,
            vision.FaceLandmarkerOptions,
        ),
        "hand_landmarker": (
            "mp_hand_landmarker",
            vision.HandLandmarker,
            vision.HandLandmarkerOptions,
        ),
        "gesture": (
            "mp_gesture",
            vision.GestureRecognizer,
            vision.GestureRecognizerOptions,
        ),
        "pose_lite": (
            "mp_pose_lite",
            vision.PoseLandmarker,
            vision.PoseLandmarkerOptions,
        ),
        "pose_full": (
            "mp_pose_full",
            vision.PoseLandmarker,
            vision.PoseLandmarkerOptions,
        ),
        "pose_heavy": (
            "mp_pose_heavy",
            vision.PoseLandmarker,
            vision.PoseLandmarkerOptions,
        ),
        "selfie": ("mp_selfie", vision.ImageSegmenter, vision.ImageSegmenterOptions),
        "multiclass": (
            "mp_multiclass",
            vision.ImageSegmenter,
            vision.ImageSegmenterOptions,
        ),
        "hair": ("mp_hair", vision.ImageSegmenter, vision.ImageSegmenterOptions),
        "objects": ("mp_objects", vision.ObjectDetector, vision.ObjectDetectorOptions),
        "embedder": ("mp_embedder", vision.ImageEmbedder, vision.ImageEmbedderOptions),
    }
    if kind not in table:
        raise VfxError(f"知らない MediaPipe の作業です: {kind}")
    key, task_cls, opt_cls = table[kind]
    base = BaseOptions(model_asset_path=str(fetch(key)))
    mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
    task = task_cls.create_from_options(
        opt_cls(base_options=base, running_mode=mode, **options)
    )
    _OPEN_TASKS.append(task)
    return task


# MediaPipe の作業は、終了時に Python の後片付けより先に閉じる（閉じないと終了時に無害なエラーが出る）
_OPEN_TASKS: list[Any] = []


@atexit.register
def _close_tasks() -> None:
    while _OPEN_TASKS:
        task = _OPEN_TASKS.pop()
        close = getattr(task, "close", None)
        if close is not None:
            close()


def mp_detect(task: Any, method: str, frame_bgr: np.ndarray, t_ms: int | None) -> Any:
    """MediaPipe の作業に 1 コマ渡す（動画なら時刻つき）。"""
    image = mp_image(frame_bgr)
    if t_ms is None:
        return getattr(task, method)(image)
    return getattr(task, method + "_for_video")(image, t_ms)
