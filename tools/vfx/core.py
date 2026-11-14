"""tools/vfx の共通部分: 動画・画像の読み書き、ffmpeg の実行、JSON、日本語の文字入れ、一覧画像。

動画の読み書きは ffmpeg のパイプで行う（どの形式でも読め、透過つき WebM も扱えるため）。
フレームは OpenCV と同じ BGR（透過つきは BGRA）の numpy 配列（高さ×幅×色、uint8）。
"""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# 既定の出力先（Git に入れない out/ の下）
DEFAULT_OUT_DIR = REPO_ROOT / "out" / "vfx"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
# 日本語を書ける書体（上から順に探す）
JP_FONTS = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
# HyperFrames で途中から再生しても重くならないよう、キーフレームを細かく入れる（CLAUDE.md）
MP4_ARGS = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
MP4_ARGS += ["-g", "6", "-bf", "0", "-movflags", "+faststart"]


class VfxError(RuntimeError):
    """利用者に伝えるべき失敗（入力の誤り、モデルが取れない等）。"""


def log(msg: str) -> None:
    print(msg, flush=True)


# ── ffmpeg ───────────────────────────────────────────────────────────────


def run(cmd: list[str], *, quiet: bool = True) -> subprocess.CompletedProcess[str]:
    """コマンドを実行する。失敗したらコマンドとエラー出力を添えて止める。"""
    result = subprocess.run(cmd, capture_output=quiet, text=True, check=False)
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-15:]
        raise VfxError(
            "コマンドが失敗しました:\n  "
            + shlex.join(cmd)
            + ("\n" + "\n".join(tail) if tail else "")
        )
    return result


def ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    return run(["ffmpeg", "-hide_banner", "-nostdin", "-y", *args])


def ffmpeg_log(args: list[str]) -> str:
    """ffmpeg を実行し、解析フィルターの出力（標準エラー）を返す。"""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-15:]
        raise VfxError(
            "ffmpeg が失敗しました:\n  " + shlex.join(cmd) + "\n" + "\n".join(tail)
        )
    return result.stderr


# ── 入力の情報 ────────────────────────────────────────────────────────────


@dataclass
class MediaInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    frames: int
    has_audio: bool
    has_alpha: bool
    is_image: bool
    codec: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def _rate(text: str) -> float:
    if "/" in text:
        num, den = text.split("/")
        return float(num) / float(den) if float(den) else 0.0
    return float(text or 0)


def probe(path: str | Path) -> MediaInfo:
    """ffprobe で大きさ・コマ数・長さ・音声と透過の有無を調べる。"""
    path = Path(path)
    if not path.exists():
        raise VfxError(f"ファイルがありません: {path}")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(run(cmd).stdout)
    video = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video is None:
        raise VfxError(f"映像が入っていません: {path}")
    width, height = int(video["width"]), int(video["height"])
    # スマートフォンの縦動画など、回転の情報がある時は縦横を入れ替える（ffmpeg は自動で回して読む）
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    if abs(rotation) % 180 == 90:
        width, height = height, width
    image = is_image(path)
    fps = _rate(video.get("avg_frame_rate") or "0") or _rate(
        video.get("r_frame_rate") or "0"
    )
    duration = float(
        video.get("duration") or data.get("format", {}).get("duration") or 0
    )
    frames = int(video.get("nb_frames") or 0)
    if image:
        fps, duration, frames = 1.0, 0.0, 1
    elif not frames and fps and duration:
        frames = round(fps * duration)
    tags = video.get("tags", {}) or {}
    has_alpha = str(
        tags.get("alpha_mode") or tags.get("ALPHA_MODE") or "0"
    ) == "1" or video.get("pix_fmt", "") in {
        "yuva420p",
        "rgba",
        "bgra",
        "argb",
        "ya8",
        "yuva444p",
    }
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    return MediaInfo(
        path=path,
        width=width,
        height=height,
        fps=fps or 30.0,
        duration=duration,
        frames=frames,
        has_audio=has_audio,
        has_alpha=has_alpha,
        is_image=image,
        codec=video.get("codec_name", ""),
        extra={"rotation": rotation, "streams": data.get("streams", [])},
    )


def scaled_size(
    width: int, height: int, scale: float | None = None, max_side: int | None = None
) -> tuple[int, int]:
    """倍率か長辺の上限から、偶数の大きさを求める（動画の書き出しは偶数でないと失敗するため）。"""
    factor = 1.0
    if scale:
        factor = scale
    if max_side and max(width, height) * factor > max_side:
        factor = max_side / max(width, height)
    w = max(2, round(width * factor / 2) * 2)
    h = max(2, round(height * factor / 2) * 2)
    return w, h


# ── 読み込み ──────────────────────────────────────────────────────────────


def load_image(path: str | Path, *, alpha: bool = False) -> np.ndarray:
    """画像を BGR（alpha=True なら BGRA）で読む。WebP なども読める。"""
    flag = cv2.IMREAD_UNCHANGED if alpha else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise VfxError(f"画像を読めません: {path}")
    if alpha:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def save_image(path: str | Path, img: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = (
        [cv2.IMWRITE_JPEG_QUALITY, 92]
        if path.suffix.lower() in {".jpg", ".jpeg"}
        else []
    )
    if not cv2.imwrite(str(path), img, params):
        raise VfxError(f"画像を保存できません: {path}")
    return path


def read_frames(
    path: str | Path,
    *,
    start: float = 0.0,
    duration: float | None = None,
    max_frames: int | None = None,
    scale: float | None = None,
    max_side: int | None = None,
    fps: float | None = None,
    alpha: bool = False,
    size: tuple[int, int] | None = None,
) -> Iterator[np.ndarray]:
    """動画（または画像 1 枚）のコマを順に返す。start/duration で範囲、scale/max_side で縮小。"""
    info = probe(path)
    if info.is_image:
        img = load_image(path, alpha=alpha)
        w, h = size or scaled_size(info.width, info.height, scale, max_side)
        if (w, h) != (img.shape[1], img.shape[0]):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        yield img
        return
    w, h = size or scaled_size(info.width, info.height, scale, max_side)
    channels = 4 if alpha else 3
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.4f}"]
    if info.has_alpha and info.codec == "vp9":
        cmd += ["-c:v", "libvpx-vp9"]  # 標準の VP9 デコーダーは透過を捨てるため
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", f"{duration:.4f}"]
    filters = []
    if fps:
        filters.append(f"fps={fps}")
    if (w, h) != (info.width, info.height):
        filters.append(f"scale={w}:{h}:flags=area")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-an", "-f", "rawvideo", "-pix_fmt", "bgra" if alpha else "bgr24", "-"]
    frame_bytes = w * h * channels
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    count = 0
    try:
        while max_frames is None or count < max_frames:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, channels).copy()
            count += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if count == 0:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise VfxError(f"コマを読めませんでした: {path}\n{err[-800:]}")


def read_all(path: str | Path, **kwargs: Any) -> list[np.ndarray]:
    return list(read_frames(path, **kwargs))


def frame_at(path: str | Path, t: float, **kwargs: Any) -> np.ndarray:
    """指定した秒のコマを 1 枚読む。"""
    return next(read_frames(path, start=t, max_frames=1, **kwargs))


# ── 書き出し ──────────────────────────────────────────────────────────────


class VideoWriter:
    """コマを順に受け取り、ffmpeg で MP4（H.264）か透過つき WebM（VP9）に書き出す。

    audio_from を渡すと、その音声（同じ範囲）を入れる。
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        *,
        alpha: bool = False,
        audio_from: str | Path | None = None,
        audio_start: float = 0.0,
        audio_duration: float | None = None,
        crf: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.alpha = alpha or self.path.suffix.lower() == ".webm"
        self.audio_from = Path(audio_from) if audio_from else None
        self.audio_start = audio_start
        self.audio_duration = audio_duration
        self.crf = crf
        self.proc: subprocess.Popen[bytes] | None = None
        self.size: tuple[int, int] | None = None
        self.count = 0

    def _open(self, w: int, h: int) -> None:
        if w % 2 or h % 2:
            raise VfxError(f"動画の大きさは偶数にしてください（{w}×{h}）")
        pix = "bgra" if self.alpha else "bgr24"
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error"]
        cmd += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix,
            "-s",
            f"{w}x{h}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
        ]
        audio = self.audio_from is not None and probe(self.audio_from).has_audio
        if audio:
            if self.audio_start:
                cmd += ["-ss", f"{self.audio_start:.4f}"]
            if self.audio_duration:
                cmd += ["-t", f"{self.audio_duration:.4f}"]
            cmd += [
                "-i",
                str(self.audio_from),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-shortest",
            ]
        if self.path.suffix.lower() == ".webm":
            cmd += [
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuva420p" if self.alpha else "yuv420p",
            ]
            cmd += [
                "-b:v",
                "0",
                "-crf",
                str(self.crf or 24),
                "-row-mt",
                "1",
                "-auto-alt-ref",
                "0",
            ]
            cmd += ["-c:a", "libopus"] if audio else []
        else:
            args = list(MP4_ARGS)
            if self.crf is not None:
                args[args.index("-crf") + 1] = str(self.crf)
            cmd += args + (["-c:a", "aac", "-b:a", "192k"] if audio else [])
        cmd.append(str(self.path))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.size = (w, h)

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self.alpha and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        if not self.alpha and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        h, w = frame.shape[:2]
        if self.proc is None:
            self._open(w, h)
        elif (w, h) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.count += 1

    def close(self) -> Path:
        if self.proc is None:
            raise VfxError(f"書き出すコマがありません: {self.path}")
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = (
            self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        )
        if self.proc.wait() != 0:
            raise VfxError(f"書き出しに失敗しました: {self.path}\n{err[-800:]}")
        return self.path

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None and exc[0] is None:
            self.close()
        elif self.proc is not None:
            self.proc.kill()


def map_video(
    src: str | Path,
    out: str | Path,
    fn: Callable[[np.ndarray, int], np.ndarray],
    *,
    start: float = 0.0,
    duration: float | None = None,
    max_frames: int | None = None,
    scale: float | None = None,
    max_side: int | None = None,
    keep_audio: bool = True,
    alpha_out: bool = False,
    alpha_in: bool = False,
    progress: bool = True,
) -> Path:
    """動画のコマを 1 枚ずつ fn(コマ, 番号) で変えて書き出す。画像なら画像で書き出す。"""
    info = probe(src)
    out = Path(out)
    if info.is_image:
        img = next(read_frames(src, scale=scale, max_side=max_side, alpha=alpha_in))
        return save_image(out, fn(img, 0))
    total = max_frames or (
        round((duration or max(info.duration - start, 0)) * info.fps) or None
    )
    writer = VideoWriter(
        out,
        info.fps,
        alpha=alpha_out,
        audio_from=src if keep_audio else None,
        audio_start=start,
        audio_duration=duration,
    )
    frames = read_frames(
        src,
        start=start,
        duration=duration,
        max_frames=max_frames,
        scale=scale,
        max_side=max_side,
        alpha=alpha_in,
    )
    with writer:
        for i, frame in enumerate(frames):
            writer.write(fn(frame, i))
            if progress and total and (i + 1) % max(1, total // 10) == 0:
                log(f"  {i + 1}/{total} コマ")
    return out


# ── 出力 ──────────────────────────────────────────────────────────────────


def default_out(src: str | Path, tag: str, ext: str) -> Path:
    """出力先を指定しなかった時の既定: out/vfx/<元の名前>-<処理>.<拡張子>"""
    return DEFAULT_OUT_DIR / f"{Path(src).stem}-{tag}{ext}"


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return round(float(o), 5)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"JSON にできない値: {type(o)}")

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1, default=conv), encoding="utf-8"
    )
    return path


def t_of(i: int, fps: float, start: float = 0.0) -> float:
    """コマ番号から秒（元の動画の中での時刻）を求める。"""
    return round(start + i / fps, 4)


# ── 描画 ──────────────────────────────────────────────────────────────────

_FONT_CACHE: dict[int, Any] = {}


def _font(size: int) -> Any:
    from PIL import ImageFont

    if size not in _FONT_CACHE:
        for f in JP_FONTS:
            if Path(f).exists():
                _FONT_CACHE[size] = ImageFont.truetype(f, size)
                break
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    size: int = 28,
    color: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] | None = (0, 0, 0),
    pad: int = 4,
    anchor: str = "lt",
) -> np.ndarray:
    """日本語も書ける文字入れ（色は BGR）。img を書き換えて返す。"""
    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    d = ImageDraw.Draw(pil)
    font = _font(size)
    box = d.textbbox(xy, text, font=font, anchor=anchor)
    if bg is not None:
        d.rectangle(
            (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=bg[::-1]
        )
    d.text(xy, text, font=font, fill=color[::-1], anchor=anchor)
    img[:, :, :3] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def heatmap(
    values: np.ndarray, *, lo: float | None = None, hi: float | None = None
) -> np.ndarray:
    """0〜1 などの数値の地図を、色分けした画像（BGR）にする。"""
    v = values.astype(np.float32)
    lo = float(np.nanmin(v)) if lo is None else lo
    hi = float(np.nanmax(v)) if hi is None else hi
    norm = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def overlay(base: np.ndarray, top: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    """top を base に重ねる（alpha は 0〜1 の数か、画素ごとの地図）。"""
    a = alpha if isinstance(alpha, float) else alpha.astype(np.float32)[..., None]
    out = (
        base[:, :, :3].astype(np.float32) * (1 - a)
        + top[:, :, :3].astype(np.float32) * a
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def contact_sheet(
    images: list[np.ndarray],
    labels: list[str] | None = None,
    *,
    cols: int = 4,
    cell_width: int = 360,
    bg: tuple[int, int, int] = (24, 24, 24),
) -> np.ndarray:
    """画像を格子状に並べ、左上に見出しを付けた 1 枚の画像を作る（確認用）。"""
    if not images:
        raise VfxError("並べる画像がありません")
    cells = []
    for i, im in enumerate(images):
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        im = im[:, :, :3]
        h = round(im.shape[0] * cell_width / im.shape[1])
        cell = cv2.resize(im, (cell_width, h), interpolation=cv2.INTER_AREA)
        if labels and i < len(labels) and labels[i]:
            draw_text(cell, labels[i], (6, 6), size=max(14, cell_width // 22))
        cells.append(cell)
    cell_h = max(c.shape[0] for c in cells)
    rows = math.ceil(len(cells) / cols)
    sheet = np.full(
        (rows * cell_h, min(cols, len(cells)) * cell_width, 3), bg, np.uint8
    )
    for i, c in enumerate(cells):
        y, x = (i // cols) * cell_h, (i % cols) * cell_width
        sheet[y : y + c.shape[0], x : x + c.shape[1]] = c
    return sheet


def side_by_side(
    before: np.ndarray, after: np.ndarray, labels: tuple[str, str] = ("前", "後")
) -> np.ndarray:
    """前後を左右に並べる（大きさは前に合わせる）。"""
    after = cv2.resize(
        after[:, :, :3],
        (before.shape[1], before.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    sheet = contact_sheet(
        [before, after], list(labels), cols=2, cell_width=min(960, before.shape[1])
    )
    return sheet


def ensure_even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def parse_box(text: str) -> tuple[int, int, int, int]:
    """'x,y,w,h' を数の組にする。"""
    parts = [int(float(v)) for v in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise VfxError(f"範囲は x,y,幅,高さ の 4 つの数で指定してください: {text}")
    return parts[0], parts[1], parts[2], parts[3]


def parse_points(text: str) -> np.ndarray:
    """'x1,y1 x2,y2 ...' を点の配列にする。"""
    pts = [[float(v) for v in p.split(",")] for p in text.split()]
    return np.array(pts, np.float32)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise VfxError(msg)


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
