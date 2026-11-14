"""BytePlus ModelArk の Seedance で、文章（と参考画像）から映像を生成して MP4 を保存する。

使い方（tools/ ディレクトリで実行）:
  uv run python gen_video.py --model <SeedanceのモデルID> \
      --prompt "夕暮れの海辺を歩く人、ゆっくり横移動するカメラ" --out ../out/shot1.mp4
  uv run python gen_video.py --model <ID> --prompt "..." --ratio 9:16 --duration 5 --out ../out/v.mp4
  uv run python gen_video.py --model <ID> --prompt "画像1の商品を回転させる" \
      --image https://example.com/product.png --out ../out/product.mp4

前提:
  - API キーはクラウド環境の「ネットワークシークレット」として保存されており、
    通信時にプロキシが自動で付ける。このスクリプトにキーを書かない・渡さない。
  - モデル ID は BytePlus の管理画面（ModelArk）で有効にしたものを使う。
  - 参考画像（--image）は公開 URL か、BytePlus の素材ライブラリの asset:// ID のみ。
    手元のファイルを直接渡すことはできない。
  - 生成は従量課金。試す時は短い秒数・低い解像度から始める。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx
from byteplussdkarkruntime import Ark

# 本物のキーはプロキシが付けるため、SDK には形式上の値だけを渡す。
DUMMY_KEY = "injected-by-proxy"
BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seedance で映像を生成する")
    parser.add_argument("--model", default=os.environ.get("SEEDANCE_MODEL"), help="Seedance のモデル ID（環境変数 SEEDANCE_MODEL でも可）")
    parser.add_argument("--prompt", required=True, help="作りたい映像の説明")
    parser.add_argument("--out", type=Path, required=True, help="保存先の MP4")
    parser.add_argument("--ratio", help="縦横比（例: 16:9, 9:16, 1:1）")
    parser.add_argument("--duration", type=int, help="秒数")
    parser.add_argument("--resolution", help="解像度（例: 720p, 1080p）")
    parser.add_argument("--seed", type=int, help="乱数の種（同じ値で結果を近づける）")
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=None, help="音声も生成するか（モデルが対応している場合）")
    parser.add_argument("--image", action="append", default=[], help="参考画像の URL または asset:// ID（複数可）")
    parser.add_argument("--image-role", default="reference_image", help="参考画像の役割（既定: reference_image）")
    parser.add_argument("--poll", type=float, default=5.0, help="結果を確認する間隔（秒）")
    parser.add_argument("--timeout", type=float, default=1800.0, help="待つ上限（秒）")
    args = parser.parse_args()

    if not args.model:
        sys.exit("モデル ID を --model か環境変数 SEEDANCE_MODEL で指定してください。")

    content: list[dict] = [{"type": "text", "text": args.prompt}]
    for url in args.image:
        content.append({"type": "image_url", "image_url": {"url": url}, "role": args.image_role})

    options = {
        key: value
        for key, value in {
            "ratio": args.ratio,
            "duration": args.duration,
            "resolution": args.resolution,
            "seed": args.seed,
            "generate_audio": args.audio,
        }.items()
        if value is not None
    }

    client = Ark(api_key=os.environ.get("ARK_API_KEY", DUMMY_KEY), base_url=BASE_URL)
    created = client.content_generation.tasks.create(model=args.model, content=content, **options)
    print(f"生成を依頼しました: タスク {created.id}")

    started = time.monotonic()
    while True:
        task = client.content_generation.tasks.get(task_id=created.id)
        if task.status == "succeeded":
            break
        if task.status in ("failed", "cancelled"):
            error = getattr(task, "error", None)
            detail = f"{error.code}: {error.message}" if error else "詳細なし"
            sys.exit(f"生成に失敗しました（{task.status}）: {detail}")
        if time.monotonic() - started > args.timeout:
            sys.exit(f"{args.timeout:.0f} 秒待っても終わりませんでした。タスク {created.id} を後で確認してください。")
        print(f"  状態: {task.status}（{time.monotonic() - started:.0f} 秒経過）")
        time.sleep(args.poll)

    url = task.content.video_url
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=300, follow_redirects=True) as response:
        response.raise_for_status()
        with args.out.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    print(f"保存しました: {args.out}（{time.monotonic() - started:.0f} 秒）")
    print(f"  ダウンロード元のホスト: {httpx.URL(url).host}")


if __name__ == "__main__":
    main()
