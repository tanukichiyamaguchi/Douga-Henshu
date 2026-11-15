"""BytePlus ModelArk の Seedance で、文章（と参考画像）から映像を生成して MP4 を保存する。

使い方（tools/ ディレクトリで実行）:
  uv run python gen_video.py --model <SeedanceのモデルID（省略可）> \
      --prompt "夕暮れの海辺を歩く人、ゆっくり横移動するカメラ" --out ../out/shot1.mp4
  uv run python gen_video.py --model <ID> --prompt "..." --ratio 9:16 --duration 5 --out ../out/v.mp4
  uv run python gen_video.py --model <ID> --prompt "画像1の商品を回転させる" \
      --image https://example.com/product.png --out ../out/product.mp4
  uv run python gen_video.py --prompt "画像1の店内で、スタッフがスマホで自撮りをして手を振る" \
      --image ../work/<作品>/assets/interior.jpg --ratio 9:16 --resolution 720p --duration 5 --no-audio --out ../out/selfie.mp4
      # 手元の画像を「雰囲気の参考」として渡す（役割 reference_image）
  uv run python gen_video.py --prompt "カメラは固定。カーテンがゆっくり揺れる" \
      --image ../work/<作品>/assets/first.png --image-role first_frame --resolution 720p --duration 5 --no-audio --out ../out/shot.mp4
      # 手元の画像を「最初のコマ」にして動かす（縦横比は画像に従うので --ratio は付けない）

前提:
  - API キーはクラウド環境の「ネットワークシークレット」として保存されており、
    通信時にプロキシが自動で付ける。このスクリプトにキーを書かない・渡さない。
  - モデル ID は BytePlus の管理画面（ModelArk）で有効にしたものを使う。
  - 参考画像（--image）は、公開 URL・BytePlus の素材ライブラリの asset:// ID・手元の画像ファイルのどれか。
    手元のファイルは、長い辺を 1920px までに縮めた JPEG を base64 のデータ URI にして送る
    （2026-11-15 に Seedance 2.5 で、役割 reference_image と first_frame の両方で生成を確認）。
  - 人物の顔が写った画像は、提供元の検査で断られることがある
    （InputImageSensitiveContentDetected.PrivacyInformation）。回避しない。
  - 生成は従量課金。試す時は短い秒数・低い解像度から始める。
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

import httpx
from byteplussdkarkruntime import Ark

# 本物のキーはプロキシが付けるため、SDK には形式上の値だけを渡す。
DUMMY_KEY = "injected-by-proxy"
BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3")
# 2026-11-14 に API のモデル一覧（GET /models）で確認した最新版。
DEFAULT_MODEL = "dreamina-seedance-2-5-260628"
# 手元の画像を送る時の長い辺の上限（px）
MAX_IMAGE_SIDE = 1920


def image_reference(value: str) -> str:
    """--image の値を API に渡す形にする。URL と asset:// はそのまま、手元のファイルはデータ URI にする。"""
    if value.startswith(("http://", "https://", "asset://", "data:")):
        return value
    path = Path(value)
    if not path.is_file():
        sys.exit(f"--image のファイルが見つかりません: {value}（URL なら http:// か https:// から書く）")
    # Pillow は他の依存関係（transformers など）と一緒に入っている。手元の画像を使う時だけ読み込む。
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=93)
    print(f"  画像 {path.name} を {image.width}×{image.height} の JPEG（{buffer.tell() / 1024:.0f}KB）にして送ります")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seedance で映像を生成する")
    parser.add_argument("--model", default=os.environ.get("SEEDANCE_MODEL", DEFAULT_MODEL), help=f"Seedance のモデル ID（既定: {DEFAULT_MODEL}）")
    parser.add_argument("--prompt", help="作りたい映像の説明（--task-id を使う時は不要）")
    parser.add_argument("--task-id", help="生成済み（または生成中）の依頼のタスク ID。新しく生成せず、結果の取得だけ行う")
    parser.add_argument("--out", type=Path, required=True, help="保存先の MP4")
    parser.add_argument("--ratio", help="縦横比（例: 16:9, 9:16, 1:1）")
    parser.add_argument("--duration", type=int, help="秒数")
    parser.add_argument("--resolution", help="解像度（例: 720p, 1080p）")
    parser.add_argument("--seed", type=int, help="乱数の種（同じ値で結果を近づける）")
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=None, help="音声も生成するか（モデルが対応している場合）")
    parser.add_argument("--image", action="append", default=[], help="参考画像の URL・asset:// ID・手元の画像ファイル（複数可）")
    parser.add_argument(
        "--image-role",
        default="reference_image",
        help="画像の役割（既定: reference_image = 雰囲気の参考。first_frame = 最初のコマ。first_frame の時は --ratio を付けない）",
    )
    parser.add_argument("--poll", type=float, default=5.0, help="結果を確認する間隔（秒）")
    parser.add_argument("--timeout", type=float, default=1800.0, help="待つ上限（秒）")
    args = parser.parse_args()

    if not args.model:
        sys.exit("モデル ID を --model か環境変数 SEEDANCE_MODEL で指定してください。")
    if not args.prompt and not args.task_id:
        sys.exit("--prompt（新しく生成）か --task-id（生成済みの結果を取得）を指定してください。")

    if args.image_role == "first_frame" and args.ratio:
        # 最初のコマを渡す時、縦横比は画像に従う。--ratio を一緒に送ると API に断られた（2026-11-15 実測）。
        print(f"  first_frame では縦横比が画像に従うため、--ratio {args.ratio} は送りません")
        args.ratio = None

    content: list[dict] = [{"type": "text", "text": args.prompt or ""}]
    if not args.task_id:
        for value in args.image:
            content.append({"type": "image_url", "image_url": {"url": image_reference(value)}, "role": args.image_role})

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
    if args.task_id:
        task_id = args.task_id
        print(f"生成済みのタスク {task_id} の結果を取得します（新しい生成はしないので料金はかからない）")
    else:
        try:
            task_id = client.content_generation.tasks.create(model=args.model, content=content, **options).id
        except Exception as error:  # API が断った理由をそのまま見せる（画像の検査で断られた時など）
            sys.exit(f"生成の依頼が断られました: {type(error).__name__}: {str(error)[:700]}")
        print(f"生成を依頼しました: タスク {task_id}（ダウンロードに失敗しても --task-id {task_id} で取り直せる）")

    started = time.monotonic()
    while True:
        task = client.content_generation.tasks.get(task_id=task_id)
        if task.status == "succeeded":
            break
        if task.status in ("failed", "cancelled"):
            error = getattr(task, "error", None)
            detail = f"{error.code}: {error.message}" if error else "詳細なし"
            sys.exit(f"生成に失敗しました（{task.status}）: {detail}")
        if time.monotonic() - started > args.timeout:
            sys.exit(f"{args.timeout:.0f} 秒待っても終わりませんでした。後で --task-id {task_id} で取得してください。")
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
    usage = getattr(task, "usage", None)
    if usage is not None:
        print(f"  使用量: {usage}")


if __name__ == "__main__":
    main()
