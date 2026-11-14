"""OpenAI の画像生成（GPT Image）で、文章から画像を作って保存する。

使い方（tools/ ディレクトリで実行）:
  uv run python gen_image_openai.py --prompt "朝日に照らされた木造のカフェの外観、写真風" \
      --size 1536x1024 --out ../out/cafe.png
  uv run python gen_image_openai.py --prompt "..." --model gpt-image-2.5-sunburst --quality high --out ../out/a.png

モデル（2026-11-14 時点、公式 SDK openai 3.51.0 の定義で確認）:
  - gpt-image-2.5-flare    … 既定。速くて高品質（OpenAI が多くの用途の標準として推奨）
  - gpt-image-2.5-sunburst … 時間はかかるが、細かい編集・仕上げ向き
  - gpt-image-2.6-flare / gpt-image-2.6-sunburst … 2026-11-04 公開の新版

前提:
  - API キーはクラウド環境の「ネットワークシークレット」として保存されており、
    通信時にプロキシが自動で付ける。このスクリプトにキーを書かない・渡さない。
  - ChatGPT の月額プランとは別に、OpenAI API の契約（従量課金）が必要。
  - 料金は画像の大きさ・品質で変わる（トークン単位の課金）。
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from openai import OpenAI

DUMMY_KEY = "injected-by-proxy"
DEFAULT_MODEL = "gpt-image-2.5-flare"


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI の GPT Image で画像を生成する")
    parser.add_argument("--prompt", required=True, help="作りたい画像の説明")
    parser.add_argument("--out", type=Path, required=True, help="保存先（.png / .jpg / .webp）")
    parser.add_argument("--model", default=os.environ.get("OPENAI_IMAGE_MODEL", DEFAULT_MODEL), help=f"モデル（既定: {DEFAULT_MODEL}）")
    parser.add_argument("--size", default="auto", help="大きさ（例: 1536x1024 横長 / 1024x1536 縦長 / 1024x1024 正方形 / auto）")
    parser.add_argument("--quality", help="品質（例: low / medium / high / auto）")
    parser.add_argument("--background", choices=["transparent", "opaque", "auto"], help="背景（transparent で透過 PNG）")
    parser.add_argument("--n", type=int, default=1, help="枚数（既定: 1）")
    args = parser.parse_args()

    suffix = args.out.suffix.lower().lstrip(".")
    output_format = {"jpg": "jpeg", "jpeg": "jpeg", "webp": "webp"}.get(suffix, "png")
    options = {key: value for key, value in {"quality": args.quality, "background": args.background}.items() if value}

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", DUMMY_KEY))
    response = client.images.generate(
        model=args.model,
        prompt=args.prompt,
        size=args.size,
        n=args.n,
        output_format=output_format,
        **options,
    )

    images = response.data or []
    if not images:
        sys.exit("画像が返ってきませんでした。")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        path = args.out if index == 0 else args.out.with_name(f"{args.out.stem}_{index + 1}{args.out.suffix}")
        path.write_bytes(base64.b64decode(image.b64_json))
        print(f"保存しました: {path}")


if __name__ == "__main__":
    main()
