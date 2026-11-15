"""HyperFrames の作品で使う文字だけを切り出した Google Fonts の書体を取得する。

使い方（tools/ で実行）:
  uv run python fetch_fonts.py ../work/<作品フォルダ> \
      --font "Shippori Mincho B1:500,600,700,800" --font "Zen Kaku Gothic New:700,900"
  uv run python fetch_fonts.py ../work/<作品フォルダ> --font "Cormorant Garamond:600,600i"   # i を付けると斜体

- 作品の index.html と compositions/*.html の本文の文字（＋英数字・よく使う記号）を集め、
  Google Fonts の CSS API の text= で、その文字だけを含むファイル（woff2）を取得して
  <作品>/assets/fonts/ に置く。1 ファイル 20〜50KB 程度。
- 最後に、使う場面の <style> に貼る @font-face を表示する（HyperFrames は同梱の 18 書体以外、
  @font-face が無いと lint の font_family_without_font_face で止まる）。
- 文字を変えたら取り直す（ファイルに無い文字は、font-family で次に書いた書体で描かれる）。
- Google Fonts の書体は OFL などの商用利用できるライセンス。書体ごとの条件は各書体のページで確認する。
- 通信は curl で行う（このクラウド環境のプロキシと証明書の設定をそのまま使うため）。
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

# Chrome の UA を付けて、woff2 の CSS を返してもらう
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
EXTRA = "、。・「」（）＋？！ー〜／％ 　"


def collect_text(project: Path) -> str:
    files = [project / "index.html", *sorted((project / "compositions").glob("*.html"))]
    chars: set[str] = set()
    for f in files:
        if not f.exists():
            continue
        s = f.read_text(encoding="utf-8")
        for pattern in (r"<style>.*?</style>", r"<script>.*?</script>", r"<!--.*?-->"):
            s = re.sub(pattern, "", s, flags=re.DOTALL)
        s = html.unescape(re.sub(r"<[^>]+>", " ", s))
        chars |= {c for c in s if not c.isspace()}
    chars |= {chr(c) for c in range(0x21, 0x7F)}
    chars |= set(EXTRA)
    return "".join(sorted(chars))


def curl(args: list[str]) -> str:
    result = subprocess.run(
        ["curl", "-sS", "--fail", "-A", UA, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"取得に失敗しました: {' '.join(args[-1:])}\n{result.stderr.strip()}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="作品で使う文字だけの Google Fonts を取得する"
    )
    parser.add_argument(
        "project", type=Path, help="HyperFrames の作品フォルダ（index.html がある所）"
    )
    parser.add_argument(
        "--font",
        action="append",
        required=True,
        help='"書体名:太さ,太さ" の形（例: "Zen Kaku Gothic New:700,900"）。太さに i を付けると斜体（600i）。複数指定できる',
    )
    args = parser.parse_args()

    text = collect_text(args.project)
    out_dir = args.project / "assets" / "fonts"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"文字の数: {len(text)}")

    faces = []
    for spec in args.font:
        family, _, weights = spec.partition(":")
        family = family.strip()
        for spec_weight in [
            w.strip() for w in (weights or "400").split(",") if w.strip()
        ]:
            italic = spec_weight.endswith("i")
            weight = spec_weight[:-1] if italic else spec_weight
            axis = f"ital,wght@1,{weight}" if italic else f"wght@{weight}"
            url = (
                "https://fonts.googleapis.com/css2?family="
                + urllib.parse.quote_plus(family)
                + f":{axis}&text="
                + urllib.parse.quote(text)
            )
            css = curl([url])
            sources = re.findall(r"src: url\((https://[^)]+)\) format\('woff2'\)", css)
            if len(sources) != 1:
                sys.exit(
                    f"{family} {spec_weight}: 想定外の応答です（書体名・太さを確認してください）\n{css[:300]}"
                )
            name = f"{family.replace(' ', '')}-{spec_weight}.woff2"
            curl(["-o", str(out_dir / name), sources[0]])
            size = (out_dir / name).stat().st_size
            print(f"{family} {spec_weight}: assets/fonts/{name}（{size:,} バイト）")
            style = " font-style: italic;" if italic else ""
            faces.append(
                f'@font-face {{ font-family: "{family}"; font-weight: {weight};{style} '
                f'src: url("assets/fonts/{name}") format("woff2"); }}'
            )

    print("\n使う場面の <style> に貼る:")
    print("\n".join(faces))


if __name__ == "__main__":
    main()
