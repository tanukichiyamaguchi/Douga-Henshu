"""音声・動画ファイルから字幕（SRT）と単語ごとのタイミング（JSON）を作る。faster-whisper を CPU で使う。

使い方（tools/ ディレクトリで実行）:
  uv run python transcribe.py ../out/narration.wav
  uv run python transcribe.py input.mp4 --model small --out-dir ../out

出力:
  <名前>.srt           … 字幕ファイル（多くの編集ソフト・HyperFrames・Remotion で読める）
  <名前>.captions.json … 単語（トークン）ごとのタイミング。Remotion の Caption 型
                         （text / startMs / endMs / timestampMs / confidence）と同じ形。

注意:
  - 初回は選んだモデルを Hugging Face から取得する（small 約 0.5GB、large-v3-turbo 約 1.6GB）。
  - GPU がないため CPU（int8）で動かす。精度と速さはモデルの大きさで決まる。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from faster_whisper import WhisperModel


def srt_time(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="faster-whisper で字幕を作る")
    parser.add_argument("input", type=Path, help="音声または動画ファイル（wav / mp3 / mp4 など）")
    parser.add_argument("--model", default="large-v3-turbo", help="モデル名（既定: large-v3-turbo。速さ優先なら small）")
    parser.add_argument("--lang", default="ja", help="言語コード（既定: ja。自動判定にするなら auto）")
    parser.add_argument("--out-dir", type=Path, help="出力先フォルダ（既定: 入力ファイルと同じ場所）")
    parser.add_argument("--no-vad", action="store_true", help="無音区間の自動除去を使わない")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"ファイルが見つかりません: {args.input}")
    out_dir = args.out_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    started = time.perf_counter()
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    print(f"モデル読み込み: {time.perf_counter() - started:.1f} 秒（{args.model}）")

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        str(args.input),
        language=None if args.lang == "auto" else args.lang,
        word_timestamps=True,
        vad_filter=not args.no_vad,
    )

    srt_lines: list[str] = []
    captions: list[dict] = []
    for index, seg in enumerate(segments, 1):
        text = seg.text.strip()
        srt_lines += [str(index), f"{srt_time(seg.start)} --> {srt_time(seg.end)}", text, ""]
        print(f"[{srt_time(seg.start)}] {text}")
        for word in seg.words or []:
            captions.append(
                {
                    "text": word.word,
                    "startMs": round(word.start * 1000),
                    "endMs": round(word.end * 1000),
                    "timestampMs": round((word.start + word.end) / 2 * 1000),
                    "confidence": round(word.probability, 4),
                }
            )

    srt_path = out_dir / f"{stem}.srt"
    json_path = out_dir / f"{stem}.captions.json"
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    json_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - t0
    print(f"言語: {info.language}（確度 {info.language_probability:.2f}）/ 音声 {info.duration:.1f} 秒 / 処理 {elapsed:.1f} 秒")
    print(f"出力: {srt_path}")
    print(f"出力: {json_path}")


if __name__ == "__main__":
    main()
