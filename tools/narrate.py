"""テキストからナレーション音声（WAV）を作る。Chatterbox（多言語モデル）を CPU で使う。

使い方（tools/ ディレクトリで実行）:
  uv run python narrate.py --text "こんにちは。動画編集の練習です。" --out ../out/narration.wav
  uv run python narrate.py --text-file script.txt --out ../out/narration.wav --voice my_voice.wav

注意:
  - 初回はモデル（約3GB）を Hugging Face から取得するため時間がかかる。
  - 生成した音声には Resemble AI の電子透かし（PerTh）が必ず入る。
  - --voice に他人の声を使う場合は、本人の許可を得ること。
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

# 生成中の進捗バーは大量のログになるため既定で止める（TQDM_DISABLE=0 で表示）。
os.environ.setdefault("TQDM_DISABLE", "1")

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# 一度に読ませる文字数の目安。
# 長すぎると発音が崩れ、短すぎる（「こんにちは。」だけ等）と意味のない音が
# 生成されやすい（実測で確認）ため、MIN_CHARS 未満の文は次の文とつなげる。
MAX_CHARS = 120
MIN_CHARS = 20
PAUSE_SEC = 0.35


def split_sentences(text: str) -> list[str]:
    """句点・感嘆符・疑問符・改行で区切り、短い文は次とつなげ、長すぎる文は読点で分ける。"""
    parts = [p.strip() for p in re.split(r"(?<=[。．！？!?])|\n+", text) if p and p.strip()]
    pieces: list[str] = []
    for part in parts:
        while len(part) > MAX_CHARS:
            cut = part.rfind("、", 0, MAX_CHARS)
            cut = cut + 1 if cut > 0 else MAX_CHARS
            pieces.append(part[:cut])
            part = part[cut:]
        if part:
            pieces.append(part)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        if buffer and len(buffer) + len(piece) > MAX_CHARS:
            chunks.append(buffer)
            buffer = ""
        buffer += piece
        if len(buffer) >= MIN_CHARS:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        if chunks and len(chunks[-1]) + len(buffer) <= MAX_CHARS:
            chunks[-1] += buffer
        else:
            chunks.append(buffer)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatterbox でナレーション音声を作る")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="読み上げる文章")
    src.add_argument("--text-file", type=Path, help="読み上げる文章が入ったテキストファイル（UTF-8）")
    parser.add_argument("--out", type=Path, required=True, help="出力する WAV ファイル")
    parser.add_argument("--lang", default="ja", help="言語コード（既定: ja）")
    parser.add_argument("--voice", type=Path, help="声を真似る参考音声（WAV、10秒前後）")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="抑揚の強さ（既定: 0.5）")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="話す速さ・安定度（既定: 0.5）")
    parser.add_argument("--seed", type=int, default=0, help="乱数の種（同じ値なら同じ結果に近づく）")
    args = parser.parse_args()

    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    if not sentences:
        raise SystemExit("読み上げる文章が空です。")

    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    print(f"モデル読み込み: {time.perf_counter() - started:.1f} 秒")

    pause = torch.zeros(1, int(model.sr * PAUSE_SEC))
    pieces: list[torch.Tensor] = []
    gen_started = time.perf_counter()
    for i, sentence in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}] {sentence}")
        wav = model.generate(
            sentence,
            language_id=args.lang,
            audio_prompt_path=str(args.voice) if args.voice else None,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
        )
        pieces.extend([wav, pause])

    audio = torch.cat(pieces[:-1], dim=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(args.out), audio, model.sr)
    seconds = audio.shape[1] / model.sr
    elapsed = time.perf_counter() - gen_started
    print(f"出力: {args.out}（音声 {seconds:.1f} 秒 / 生成 {elapsed:.1f} 秒）")


if __name__ == "__main__":
    main()
