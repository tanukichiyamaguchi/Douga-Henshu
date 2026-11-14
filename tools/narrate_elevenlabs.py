"""ElevenLabs の音声合成で、文章からナレーション（MP3）を作る。

使い方（tools/ ディレクトリで実行）:
  uv run python narrate_elevenlabs.py --list-voices            # 使える声の一覧（日本語の声を優先表示）
  uv run python narrate_elevenlabs.py --list-models            # 使えるモデルと対応言語
  uv run python narrate_elevenlabs.py --voice-id <声のID> \
      --text "新しいお店がオープンしました。" --out ../out/narration.mp3
  uv run python narrate_elevenlabs.py --voice-id <ID> --text-file script.txt --out ../out/narration.mp3

前提:
  - API キーはクラウド環境の「ネットワークシークレット」として保存されており、
    通信時にプロキシが自動で付ける。このスクリプトにキーを書かない・渡さない。
  - 声（--voice-id）とモデル（--model）の標準は、キー登録後に API の一覧で確認して決める。
    環境変数 ELEVENLABS_VOICE_ID / ELEVENLABS_TTS_MODEL でも指定できる。
  - 実在の人物の声を真似る場合は、本人の許可が必要。
  - 文字数に応じた従量課金。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from elevenlabs.client import ElevenLabs

DUMMY_KEY = "injected-by-proxy"
BASE_URL = os.environ.get("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")
# 日本語に対応した多言語モデル。キー登録後に --list-models で確認して見直す。
DEFAULT_MODEL = "eleven_multilingual_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="ElevenLabs でナレーションを作る")
    parser.add_argument("--text", help="読み上げる文章")
    parser.add_argument("--text-file", type=Path, help="読み上げる文章のファイル（UTF-8）")
    parser.add_argument("--out", type=Path, help="保存先（.mp3）")
    parser.add_argument("--voice-id", default=os.environ.get("ELEVENLABS_VOICE_ID"), help="声の ID（--list-voices で確認）")
    parser.add_argument("--model", default=os.environ.get("ELEVENLABS_TTS_MODEL", DEFAULT_MODEL), help=f"モデル（既定: {DEFAULT_MODEL}）")
    parser.add_argument("--lang", help="言語コード（例: ja。モデルが対応している場合だけ指定する）")
    parser.add_argument("--seed", type=int, help="乱数の種")
    parser.add_argument("--list-voices", action="store_true", help="使える声の一覧を表示する")
    parser.add_argument("--list-models", action="store_true", help="使えるモデルと対応言語を表示する")
    args = parser.parse_args()

    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", DUMMY_KEY), base_url=BASE_URL)

    if args.list_models:
        for model in client.models.list():
            if not getattr(model, "can_do_text_to_speech", True):
                continue
            languages = [lang.language_id for lang in (model.languages or [])]
            mark = "日本語対応" if "ja" in languages else "日本語なし"
            print(f"{model.model_id:<28} {mark}  {model.name}")
        return

    if args.list_voices:
        voices = client.voices.search(page_size=100).voices
        for voice in sorted(voices, key=lambda v: "ja" not in str(getattr(v, "labels", {}) or {})):
            labels = getattr(voice, "labels", None) or {}
            print(f"{voice.voice_id}  {voice.name}  {labels}")
        return

    if not args.voice_id:
        sys.exit("声の ID を --voice-id か環境変数 ELEVENLABS_VOICE_ID で指定してください（--list-voices で確認）。")
    if not args.out:
        sys.exit("--out で保存先を指定してください。")
    text = args.text if args.text is not None else (args.text_file.read_text(encoding="utf-8") if args.text_file else "")
    if not text.strip():
        sys.exit("--text か --text-file で読み上げる文章を指定してください。")

    options = {key: value for key, value in {"language_code": args.lang, "seed": args.seed}.items() if value is not None}
    audio = client.text_to_speech.convert(
        voice_id=args.voice_id,
        text=text,
        model_id=args.model,
        output_format="mp3_44100_128",
        **options,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with args.out.open("wb") as f:
        for chunk in audio:
            f.write(chunk)
            size += len(chunk)
    print(f"保存しました: {args.out}（{size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
