"""ElevenLabs の効果音生成で、文章から短い効果音を作って保存する。

使い方（tools/ ディレクトリで実行）:
  uv run python gen_sfx.py --prompt "short airy whoosh for a fast camera whip pan" --seconds 0.6 --out ../out/whoosh.mp3
  uv run python gen_sfx.py --prompt "soft cute pop" --seconds 0.4 --takes 3 --out ../out/pop.mp3
      # --takes 3 で pop-1.mp3, pop-2.mp3, pop-3.mp3 を作る（同じ音の使い回しを避けるため）

前提:
  - API キーはクラウド環境の「ネットワークシークレット」として保存されており、
    通信時にプロキシが自動で付ける。このスクリプトにキーを書かない・渡さない。
  - 契約プランのクレジットから引かれる。
  - 効果音は「画面上に原因がある所」だけに使う（転換・札が出る瞬間など）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from elevenlabs.client import ElevenLabs

DUMMY_KEY = "injected-by-proxy"
BASE_URL = os.environ.get("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")


def main() -> None:
    parser = argparse.ArgumentParser(description="ElevenLabs で効果音を生成する")
    parser.add_argument("--prompt", required=True, help="効果音の説明（英語の方が意図が伝わりやすい）")
    parser.add_argument("--seconds", type=float, help="長さ（秒）。省略するとモデルが決める")
    parser.add_argument("--influence", type=float, help="説明文にどれだけ忠実にするか（0〜1）")
    parser.add_argument("--takes", type=int, default=1, help="作る本数（2 以上なら名前に -1, -2 … を付ける）")
    parser.add_argument("--out", type=Path, required=True, help="保存先（.mp3）")
    args = parser.parse_args()

    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", DUMMY_KEY), base_url=BASE_URL)
    options = {
        key: value
        for key, value in {"duration_seconds": args.seconds, "prompt_influence": args.influence}.items()
        if value is not None
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for take in range(1, args.takes + 1):
        path = args.out if args.takes == 1 else args.out.with_name(f"{args.out.stem}-{take}{args.out.suffix}")
        audio = client.text_to_sound_effects.convert(text=args.prompt, output_format="mp3_44100_128", **options)
        size = 0
        with path.open("wb") as f:
            for chunk in audio:
                f.write(chunk)
                size += len(chunk)
        print(f"保存しました: {path}（{size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
