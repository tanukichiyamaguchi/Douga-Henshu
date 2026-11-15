"""ナレーションの文と文の間（無音と息継ぎ）を詰めて、尺を短くする。

ElevenLabs の eleven_v4 は話す速さ（speed）の設定が効かない（2026-11-14 実測）。
尺を合わせたい時は、この道具で長い間だけを詰める。言葉そのものには手を加えない。

使い方（tools/ ディレクトリで実行）:
  uv run python tighten_pauses.py ../out/narration.mp3 --out ../out/narration-tight.wav
  uv run python tighten_pauses.py in.mp3 --max-pause 0.22 --out out.wav    # 間を 0.22 秒に
  uv run python tighten_pauses.py in.mp3 --tempo 1.04 --out out.wav        # さらに 4% 速く（音の高さは保つ）

仕組み:
  - 10 ミリ秒ごとの音量で「声のある所」と「間」を分ける。--threshold（既定 -40 dB）以下を間とみなす。
    ElevenLabs の声の息継ぎは -45 dB 前後なので、既定では間に含まれ、一緒に詰まる（2026-11-14 実測）。
  - --min-gap（既定 0.30 秒）より長い間だけを --max-pause の長さに縮める。読点程度の短い間はそのまま。
  - 縮めた間のうち --keep-head（既定 0.10 秒）は次の言葉の直前に残す。「ス」「シ」など
    小さな音で始まる言葉の出だしを切らないため。切り口は 10 ミリ秒でなめらかにつなぐ。
  - --tempo を付けると、最後に ffmpeg の rubberband で速さだけを変える（音の高さは保つ）。
  - 出力は WAV（44.1kHz・モノラル・16bit）。詰めた後は transcribe.py で内容とタイミングを確かめる。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

SR = 44100
HOP = 0.01  # 音量を測る区切り（秒）
FADE = 0.01  # 切り口をつなぐ長さ（秒）


def decode(path: Path) -> np.ndarray:
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(pcm, dtype=np.float32).copy()


def loudness_db(x: np.ndarray) -> np.ndarray:
    hop = int(SR * HOP)
    n = len(x) // hop
    frames = x[: n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)


def find_gaps(db: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    """声のない区間を (始まり秒, 終わり秒) で返す。先頭と末尾の無音も含む。"""
    quiet = db <= threshold
    gaps: list[tuple[float, float]] = []
    start = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            gaps.append((start * HOP, i * HOP))
            start = None
    if start is not None:
        gaps.append((start * HOP, len(quiet) * HOP))
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description="ナレーションの長い間（無音・息継ぎ）を詰める")
    parser.add_argument("input", type=Path, help="音声ファイル（mp3 / wav など）")
    parser.add_argument("--out", type=Path, required=True, help="保存先（.wav）")
    parser.add_argument("--max-pause", type=float, default=0.22, help="詰めた後の間の長さ（秒、既定 0.22）")
    parser.add_argument("--min-gap", type=float, default=0.30, help="これより長い間だけを詰める（秒、既定 0.30）")
    parser.add_argument("--threshold", type=float, default=-40.0, help="これ以下の音量を間とみなす（dB、既定 -40）")
    parser.add_argument("--keep-head", type=float, default=0.10, help="次の言葉の直前に残す長さ（秒、既定 0.10）")
    parser.add_argument("--lead", type=float, default=0.05, help="最初の言葉の前に残す無音（秒、既定 0.05）")
    parser.add_argument("--tail", type=float, default=0.15, help="最後の言葉の後に残す長さ（秒、既定 0.15）")
    parser.add_argument("--tempo", type=float, help="最後に速さだけを変える倍率（例: 1.04。音の高さは保つ）")
    args = parser.parse_args()

    if args.keep_head >= args.max_pause:
        raise SystemExit("--keep-head は --max-pause より短くしてください。")

    x = decode(args.input)
    total = len(x) / SR
    db = loudness_db(x)
    if not (db > args.threshold).any():
        raise SystemExit("声が見つかりませんでした（--threshold を確認してください）。")
    gaps = find_gaps(db, args.threshold)

    # 残す区間を (始まり秒, 終わり秒) で並べる。
    first_voice = gaps[0][1] if gaps[0][0] == 0.0 else 0.0
    last_voice = gaps[-1][0] if abs(gaps[-1][1] - total) < HOP * 1.5 else total
    keep: list[list[float]] = [[max(0.0, first_voice - args.lead), 0.0]]
    print(f"元の長さ: {total:.2f} 秒（声は {first_voice:.2f}〜{last_voice:.2f} 秒）")
    for g0, g1 in gaps:
        if g0 <= first_voice or g1 >= last_voice:
            continue  # 先頭・末尾の無音は別に扱う
        length = g1 - g0
        if length < args.min_gap:
            continue
        cut_from = g0 + (args.max_pause - args.keep_head)
        cut_to = g1 - args.keep_head
        keep[-1][1] = cut_from
        keep.append([cut_to, 0.0])
        print(f"  間 {g0:6.2f}〜{g1:6.2f} 秒（{length:.2f} 秒）→ {args.max_pause:.2f} 秒")
    keep[-1][1] = min(total, last_voice + args.tail)

    # つなぐ（切り口は短いクロスフェード）。
    fade = int(SR * FADE)
    out = x[int(keep[0][0] * SR) : int(keep[0][1] * SR)].copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    for a, b in keep[1:]:
        piece = x[int(a * SR) : int(b * SR)]
        if len(out) >= fade and len(piece) >= fade:
            out[-fade:] = out[-fade:] * (1 - ramp) + piece[:fade] * ramp
            out = np.concatenate([out, piece[fade:]])
        else:
            out = np.concatenate([out, piece])
    out[-fade * 5 :] *= np.linspace(1.0, 0.0, len(out[-fade * 5 :]), dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    filters = ["-af", f"rubberband=tempo={args.tempo}"] if args.tempo else []
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", *filters,
         "-c:a", "pcm_s16le", str(args.out)],
        input=out.astype(np.float32).tobytes(),
        check=True,
    )
    result = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(args.out)],
            capture_output=True, text=True, check=True,
        ).stdout
    )
    print(f"保存しました: {args.out}（{total:.2f} 秒 → {result:.2f} 秒）")


if __name__ == "__main__":
    main()
