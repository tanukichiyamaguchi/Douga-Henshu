"""ElevenLabs Music で、文章から BGM（音楽）を生成して保存する。

使い方（tools/ ディレクトリで実行）:
  uv run python gen_music.py --prompt "明るく爽やかなアコースティックギター、120BPM" \
      --seconds 30 --out ../out/bgm.mp3
  uv run python gen_music.py --prompt "..." --seconds 60 --vocals --out ../out/song.mp3

前提:
  - API キーはクラウド環境の「ネットワークシークレット」として保存されており、
    通信時にプロキシが自動で付ける。このスクリプトにキーを書かない・渡さない。
  - 音楽 API は有料プランが必要（公式ドキュメントの記載）。使える範囲（SNS・広告など）は
    契約プランの規約で確認する。
  - 既定は歌なし（インスト）。--vocals で歌ありにできる。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from elevenlabs.client import ElevenLabs

DUMMY_KEY = "injected-by-proxy"
BASE_URL = os.environ.get("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")


def main() -> None:
    parser = argparse.ArgumentParser(description="ElevenLabs Music で BGM を生成する")
    parser.add_argument("--prompt", required=True, help="曲の雰囲気・楽器・テンポなどの説明")
    parser.add_argument("--seconds", type=float, default=30.0, help="曲の長さ（秒、既定: 30）")
    parser.add_argument("--vocals", action="store_true", help="歌ありにする（既定は歌なし）")
    parser.add_argument("--seed", type=int, help="乱数の種")
    parser.add_argument("--out", type=Path, required=True, help="保存先（.mp3）")
    args = parser.parse_args()

    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", DUMMY_KEY), base_url=BASE_URL)
    options = {"seed": args.seed} if args.seed is not None else {}
    audio = client.music.compose(
        prompt=args.prompt,
        music_length_ms=int(args.seconds * 1000),
        force_instrumental=not args.vocals,
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
