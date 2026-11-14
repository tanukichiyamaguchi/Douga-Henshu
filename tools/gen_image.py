"""BytePlus ModelArk の Seedream で、文章（と参考画像）から画像を生成して保存する。

使い方（tools/ ディレクトリで実行）:
  uv run python gen_image.py --model <SeedreamのモデルID> \
      --prompt "朝日に照らされた木造のカフェの外観、写真風" --out ../out/cafe.png
  uv run python gen_image.py --model <ID> --prompt "..." --size 2K --out ../out/a.png

前提:
  - API キーはクラウド環境の「ネットワークシークレット」として保存されており、
    通信時にプロキシが自動で付ける。このスクリプトにキーを書かない・渡さない。
  - 画像はデータ（base64）として受け取るので、画像配信用の別ホストへの通信は不要。
  - 参考画像（--image）は公開 URL のみ。
  - 生成は従量課金。
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from byteplussdkarkruntime import Ark

DUMMY_KEY = "injected-by-proxy"
BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seedream で画像を生成する")
    parser.add_argument("--model", default=os.environ.get("SEEDREAM_MODEL"), help="Seedream のモデル ID（環境変数 SEEDREAM_MODEL でも可）")
    parser.add_argument("--prompt", required=True, help="作りたい画像の説明")
    parser.add_argument("--out", type=Path, required=True, help="保存先（.png / .jpg）")
    parser.add_argument("--size", help="大きさ（例: 2K、または 1920x1080）")
    parser.add_argument("--seed", type=int, help="乱数の種")
    parser.add_argument("--image", action="append", default=[], help="参考画像の URL（複数可）")
    args = parser.parse_args()

    if not args.model:
        sys.exit("モデル ID を --model か環境変数 SEEDREAM_MODEL で指定してください。")

    options = {key: value for key, value in {"size": args.size, "seed": args.seed}.items() if value is not None}
    if args.image:
        options["image"] = args.image if len(args.image) > 1 else args.image[0]

    client = Ark(api_key=os.environ.get("ARK_API_KEY", DUMMY_KEY), base_url=BASE_URL)
    response = client.images.generate(
        model=args.model,
        prompt=args.prompt,
        response_format="b64_json",
        watermark=False,
        **options,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    images = response.data or []
    if not images:
        sys.exit("画像が返ってきませんでした。")
    for index, image in enumerate(images):
        path = args.out if index == 0 else args.out.with_name(f"{args.out.stem}_{index + 1}{args.out.suffix}")
        path.write_bytes(base64.b64decode(image.b64_json))
        print(f"保存しました: {path}（{image.size}）")


if __name__ == "__main__":
    main()
