"""OpenAI の画像生成（GPT Image）で、文章から画像を作って保存する。

使い方（tools/ ディレクトリで実行）:
  uv run python gen_image_openai.py --prompt "朝日に照らされた木造のカフェの外観、写真風" --out ../out/cafe.png
      # 既定（2026-11-17 から）: gpt-image-2.6-sunburst・品質 max・縦長 2160x3840
  uv run python gen_image_openai.py --prompt "..." --size 3840x2160 --out ../out/wide.png   # 横長
  uv run python gen_image_openai.py --prompt "..." --model gpt-image-2.5-flare --quality high --size 1024x1536 --out ../out/a.png   # 速く・安く試す時

モデル（2026-11-14 時点、公式 SDK openai 3.51.0 の定義で確認）:
  - gpt-image-2.5-flare    … 速くて高品質（OpenAI が多くの用途の標準として推奨）。2026-11-16 までの既定
  - gpt-image-2.5-sunburst … 時間はかかるが、細かい編集・仕上げ向き
  - gpt-image-2.6-flare / gpt-image-2.6-sunburst … 2026-11-04 公開の新版

既定の設定（2026-11-17 ユーザーの指示「今後は、2.6sunburstでMAXの品質で出して欲しい。今後は最高峰のレベルで出して。」）:
  - モデル gpt-image-2.6-sunburst・品質 max・大きさ 2160x3840（縦長）。大きさも、いちばん上にするのは Claude の解釈。
  - 品質は low / medium / high / xhigh / max / auto。SDK（openai 3.51.0）の説明では xhigh・max は 2.5 の flare と sunburst だけだが、
    2.6 sunburst でも max は通り、high の 4 倍のトークンを使った（2026-11-17 に 1440x2560 で実測: max 7,370・high 1,843）。
  - 大きさは「横x縦」。縦横とも 16 の倍数・縦横比 1:3〜3:1・最大 3840x2160。2560x1440 を超える大きさは、SDK の説明で「試験的」。
    2160x3840 は 1 枚約 2 分・PNG 約 9.5MB・出力 13,342 トークン（2026-11-17 の 1 回の実測）。

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
DEFAULT_MODEL = "gpt-image-2.6-sunburst"
DEFAULT_QUALITY = "max"
DEFAULT_SIZE = "2160x3840"


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI の GPT Image で画像を生成する")
    parser.add_argument("--prompt", required=True, help="作りたい画像の説明")
    parser.add_argument("--out", type=Path, required=True, help="保存先（.png / .jpg / .webp）")
    parser.add_argument("--model", default=os.environ.get("OPENAI_IMAGE_MODEL", DEFAULT_MODEL), help=f"モデル（既定: {DEFAULT_MODEL}）")
    parser.add_argument("--size", default=DEFAULT_SIZE, help=f"大きさ（既定: {DEFAULT_SIZE} 縦長。横長は 3840x2160。縦横とも 16 の倍数・最大 3840x2160・auto も可）")
    parser.add_argument("--quality", default=DEFAULT_QUALITY, help=f"品質（既定: {DEFAULT_QUALITY}。low / medium / high / xhigh / max / auto）")
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
