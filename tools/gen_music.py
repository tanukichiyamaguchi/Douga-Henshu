"""ElevenLabs Music で、文章から BGM（音楽）を生成して保存する。

使い方（tools/ ディレクトリで実行）:
  uv run python gen_music.py --prompt "明るく爽やかなアコースティックギター、120BPM" \
      --seconds 30 --out ../out/bgm.mp3
  uv run python gen_music.py --prompt "..." --seconds 60 --vocals --out ../out/song.mp3
  uv run python gen_music.py --plan plan.json --out ../out/bgm.mp3
      # 区間ごとの長さと雰囲気を指定する（例: 6 秒で溜めて、6 秒目でドロップ）
      # 注意: 区間の雰囲気の指定は守られないことがある（2026-11-14 実測: 6 秒目のドロップを指定して
      #       14 秒付近になった）。山の位置を確実に合わせたい時は、編集で作る（CLAUDE.md 参照）。

--plan の JSON（区間の長さの合計が曲の長さになる。歌なしは lines を空にする）:
  {
    "positive_global_styles": ["chic dance pop", "120 bpm", "instrumental"],
    "negative_global_styles": ["vocals"],
    "sections": [
      {"section_name": "build", "duration_ms": 6000, "lines": [],
       "positive_local_styles": ["sparse, rising tension"], "negative_local_styles": ["full drums"]},
      {"section_name": "drop", "duration_ms": 8000, "lines": [],
       "positive_local_styles": ["big euphoric drop on the first beat"], "negative_local_styles": []}
    ]
  }

前提:
  - API キーはクラウド環境の「ネットワークシークレット」として保存されており、
    通信時にプロキシが自動で付ける。このスクリプトにキーを書かない・渡さない。
  - 音楽 API は有料プランが必要（公式ドキュメントの記載）。使える範囲（SNS・広告など）は
    契約プランの規約で確認する。
  - 既定は歌なし（インスト）。--vocals で歌ありにできる。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from elevenlabs.client import ElevenLabs
from elevenlabs.types import MusicPrompt

DUMMY_KEY = "injected-by-proxy"
BASE_URL = os.environ.get("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")


def main() -> None:
    parser = argparse.ArgumentParser(description="ElevenLabs Music で BGM を生成する")
    parser.add_argument("--prompt", help="曲の雰囲気・楽器・テンポなどの説明（--plan を使う時は不要）")
    parser.add_argument("--plan", type=Path, help="区間ごとの構成を書いた JSON（使い方は先頭の説明を参照）")
    parser.add_argument("--seconds", type=float, default=30.0, help="曲の長さ（秒、既定: 30。--plan の時は区間の合計）")
    parser.add_argument("--vocals", action="store_true", help="歌ありにする（既定は歌なし。--plan の時は lines で決まる）")
    parser.add_argument("--seed", type=int, help="乱数の種")
    parser.add_argument("--out", type=Path, required=True, help="保存先（.mp3）")
    args = parser.parse_args()
    if not args.prompt and not args.plan:
        parser.error("--prompt か --plan を指定してください。")

    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", DUMMY_KEY), base_url=BASE_URL)
    options = {"seed": args.seed} if args.seed is not None else {}
    if args.plan:
        plan = MusicPrompt(**json.loads(args.plan.read_text(encoding="utf-8")))
        total = sum(section.duration_ms for section in plan.sections)
        print(f"構成: {len(plan.sections)} 区間・合計 {total / 1000:.1f} 秒（区間の長さを守る指定で生成）")
        audio = client.music.compose(composition_plan=plan, respect_sections_durations=True, **options)
    else:
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
