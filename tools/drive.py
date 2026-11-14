"""Google Drive API で、完成動画の保存（大きなファイルも可）と素材の取り込みを行う。

使い方（tools/ ディレクトリで実行）:
  uv run python drive.py whoami                          # どのアカウントにつながるか確認
  uv run python drive.py upload ../out/final.mp4         # 02_完成動画 に保存
  uv run python drive.py list                            # 01_素材 の中身を一覧
  uv run python drive.py download 台本.txt --out ../out/input/   # 01_素材 から取り込む

対象のフォルダ:
  マイドライブ直下の「Douga-Henshu」フォルダの中だけを使う。
    01_素材      … 取り込み元（list / download の既定）
    02_完成動画  … 保存先（upload の既定）
  それ以外のフォルダは見ない・触らない。

前提（認証情報）:
  クラウド環境の設定画面で、次の環境変数をユーザーが登録する（値をこのスクリプトや
  リポジトリに書かない）。新しいセッションから使えるようになる。
    GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET / GOOGLE_OAUTH_REFRESH_TOKEN
  通信先: oauth2.googleapis.com（認証）と www.googleapis.com（Drive 本体）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

TOKEN_URL = os.environ.get("GOOGLE_OAUTH_TOKEN_URL", "https://oauth2.googleapis.com/token")
API = os.environ.get("DRIVE_API_BASE", "https://www.googleapis.com")
ROOT_FOLDER = "Douga-Henshu"
INPUT_FOLDER = "01_素材"
OUTPUT_FOLDER = "02_完成動画"
CHUNK = 8 * 1024 * 1024  # 分割アップロードの 1 回分（256KiB の倍数である必要がある）
MIME = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".png": "image/png",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".mp3": "audio/mpeg", ".wav": "audio/wav",
        ".srt": "application/x-subrip", ".txt": "text/plain", ".json": "application/json"}


def access_token() -> str:
    names = ("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_OAUTH_REFRESH_TOKEN")
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit("認証情報の環境変数がありません: " + ", ".join(missing)
                 + "\nクラウド環境の設定画面で登録し、新しいセッションで実行してください。")
    response = httpx.post(TOKEN_URL, data={
        "client_id": os.environ["GOOGLE_OAUTH_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_OAUTH_CLIENT_SECRET"],
        "refresh_token": os.environ["GOOGLE_OAUTH_REFRESH_TOKEN"],
        "grant_type": "refresh_token",
    }, timeout=30)
    if response.status_code != 200:
        sys.exit(f"認証に失敗しました（HTTP {response.status_code}）: {response.text[:300]}")
    return response.json()["access_token"]


class Drive:
    def __init__(self) -> None:
        self.client = httpx.Client(base_url=API, headers={"Authorization": f"Bearer {access_token()}"}, timeout=120)

    def _get(self, path: str, **params) -> dict:
        response = self.client.get(path, params=params)
        if response.status_code != 200:
            sys.exit(f"Drive API エラー（HTTP {response.status_code}）: {response.text[:300]}")
        return response.json()

    def find_folder(self, name: str, parent: str) -> str:
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        query = (f"name = '{escaped}' and '{parent}' in parents and trashed = false "
                 "and mimeType = 'application/vnd.google-apps.folder'")
        files = self._get("/drive/v3/files", q=query, fields="files(id,name)", pageSize=10)["files"]
        if not files:
            where = "マイドライブ直下" if parent == "root" else "Douga-Henshu の中"
            sys.exit(f"{where}に「{name}」フォルダが見つかりません。認証情報のアカウントや権限の範囲が"
                     "違う可能性があります（whoami で確認できます）。")
        return files[0]["id"]

    def folder(self, name: str) -> str:
        return self.find_folder(name, self.find_folder(ROOT_FOLDER, "root"))

    def upload(self, path: Path, folder_name: str, name: str | None) -> dict:
        size = path.stat().st_size
        mime = MIME.get(path.suffix.lower(), "application/octet-stream")
        start = self.client.post(
            "/upload/drive/v3/files",
            params={"uploadType": "resumable", "fields": "id,name,size,webViewLink"},
            headers={"X-Upload-Content-Type": mime, "X-Upload-Content-Length": str(size)},
            json={"name": name or path.name, "parents": [self.folder(folder_name)]},
        )
        if start.status_code != 200:
            sys.exit(f"アップロードを開始できませんでした（HTTP {start.status_code}）: {start.text[:300]}")
        session_url = start.headers["Location"]
        sent = 0
        with path.open("rb") as f:
            while True:
                chunk = f.read(CHUNK)
                end = sent + len(chunk) - 1
                headers = {"Content-Length": str(len(chunk)), "Content-Range": f"bytes {sent}-{end}/{size}"}
                response = self.client.put(session_url, content=chunk, headers=headers)
                sent += len(chunk)
                if response.status_code in (200, 201):
                    return response.json()
                if response.status_code != 308:
                    sys.exit(f"アップロードに失敗しました（HTTP {response.status_code}）: {response.text[:300]}")
                print(f"  送信中: {sent / size:.0%}")

    def list(self, folder_name: str) -> list[dict]:
        query = f"'{self.folder(folder_name)}' in parents and trashed = false"
        return self._get("/drive/v3/files", q=query, pageSize=200, orderBy="name",
                         fields="files(id,name,size,mimeType,modifiedTime)")["files"]

    def download(self, folder_name: str, target: str, out_dir: Path) -> Path:
        files = self.list(folder_name)
        match = next((f for f in files if target in (f["name"], f["id"])), None)
        if match is None:
            sys.exit(f"「{folder_name}」に「{target}」が見つかりません。list で名前を確認してください。")
        if match["mimeType"].startswith("application/vnd.google-apps."):
            sys.exit(f"「{match['name']}」は Google ドキュメント等の形式なので、このコマンドでは取り込めません。")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / match["name"]
        with self.client.stream("GET", f"/drive/v3/files/{match['id']}", params={"alt": "media"}) as response:
            if response.status_code != 200:
                sys.exit(f"取り込みに失敗しました（HTTP {response.status_code}）")
            with out.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Drive の Douga-Henshu フォルダとやり取りする")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("whoami", help="つながるアカウントとフォルダを確認する")
    up = sub.add_parser("upload", help="ファイルを保存する（既定: 02_完成動画）")
    up.add_argument("file", type=Path)
    up.add_argument("--to", default=OUTPUT_FOLDER, help=f"保存先フォルダ（既定: {OUTPUT_FOLDER}）")
    up.add_argument("--name", help="Drive 上のファイル名（既定: 元の名前）")
    ls = sub.add_parser("list", help="フォルダの中身を一覧する（既定: 01_素材）")
    ls.add_argument("--in", dest="folder", default=INPUT_FOLDER)
    dl = sub.add_parser("download", help="ファイルを取り込む（既定: 01_素材 から）")
    dl.add_argument("target", help="ファイル名またはファイル ID")
    dl.add_argument("--from", dest="folder", default=INPUT_FOLDER)
    dl.add_argument("--out", type=Path, default=Path("../out/input"))
    args = parser.parse_args()

    drive = Drive()
    if args.command == "whoami":
        user = drive._get("/drive/v3/about", fields="user(emailAddress,displayName)")["user"]
        print(f"アカウント: {user.get('emailAddress')}（{user.get('displayName')}）")
        for name in (INPUT_FOLDER, OUTPUT_FOLDER):
            drive.folder(name)
            print(f"フォルダ確認 OK: {ROOT_FOLDER}/{name}")
    elif args.command == "upload":
        if not args.file.exists():
            sys.exit(f"ファイルが見つかりません: {args.file}")
        result = drive.upload(args.file, args.to, args.name)
        print(f"保存しました: {result['name']}（{int(result.get('size', 0)) / 1024 / 1024:.1f}MB）")
        print(f"  Drive で開く: {result.get('webViewLink')}")
    elif args.command == "list":
        for f in drive.list(args.folder):
            size = f"{int(f['size']) / 1024 / 1024:.1f}MB" if "size" in f else "-"
            print(f"{f['name']}\t{size}\t{f['modifiedTime'][:16]}\t{f['id']}")
    elif args.command == "download":
        print(f"取り込みました: {drive.download(args.folder, args.target, args.out)}")


if __name__ == "__main__":
    main()
