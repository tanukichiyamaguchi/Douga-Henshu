"""コマンドの入口。

使い方（tools/ で実行）:
  uv run python -m vfx list                 # 100 項目と状態の一覧
  uv run python -m vfx info 76              # 1 項目の説明と引数
  uv run python -m vfx scenes in.mp4        # 名前で実行
  uv run python -m vfx 76 in.mp4            # 番号でも実行できる
  uv run python -m vfx catalog --markdown   # SKILL.md 用の一覧表（Markdown）
  uv run python -m vfx catalog --reference 1  # 分類ごとの詳しい説明（Markdown）
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from . import catalog
from .core import VfxError, eprint
from .registry import COMMANDS, by_no

# テクニックを実装したモジュール（担当ごとに 1 ファイル）
MODULES = [
    "flow", "track", "mot", "stabilize", "camera", "body", "face", "understand", "restore",
    "tone", "enhance", "optics", "matte", "keying", "compose", "generate", "scenes", "find",
]  # fmt: skip

# 読み込めなかったモジュール（名前 → 理由）
LOAD_ERRORS: dict[str, str] = {}


def _load() -> None:
    for m in MODULES:
        try:
            mod = importlib.import_module(f"vfx.{m}")
        except Exception as e:  # noqa: BLE001  1 つのモジュールが壊れていても、他のコマンドは使えるようにする
            LOAD_ERRORS[m] = f"{type(e).__name__}: {e}"
            continue
        for no, n in getattr(mod, "NOTES", {}).items():
            catalog.NOTES[no] = n


def _status(no: int) -> str:
    return catalog.note(no, "status") or "未"


def cmd_list(cat: int | None) -> None:
    for c, title in catalog.CATEGORIES.items():
        if cat and c != cat:
            continue
        print(f"\n■ {c}/4 {title}")
        for t in catalog.TECHNIQUES:
            if t.cat == c:
                mark = "" if t.cmd in COMMANDS else "（未実装）"
                print(
                    f"  {_status(t.no)} {t.no:3d} {t.name:<14}{t.cmd:<18}{t.what}{mark}"
                )
    print(
        "\n記号: ◎ 実装して確認済み / ○ 簡易な方法で実装・確認済み / △ 条件付き"
        " / × この環境では不可 / 未 まだ作っていない（記号は作った AI の試験による）"
    )


def cmd_info(key: str) -> None:
    t = catalog.BY_NO.get(int(key)) if key.isdigit() else catalog.BY_CMD.get(key)
    if t is None:
        raise VfxError(f"見つかりません: {key}")
    print(
        f"{t.no}. {t.name}（{catalog.CATEGORIES[t.cat].split('（')[0]}）  状態: {_status(t.no)}"
    )
    print(f"  何をする: {t.what}")
    print(f"  頼み方の例: 「{t.example}」")
    print(f"  コマンド: uv run python -m vfx {t.cmd} ...")
    for field, label in catalog.NOTE_KEYS.items():
        if field != "status" and catalog.note(t.no, field):
            print(f"  {label.split('（')[0]}: {catalog.note(t.no, field)}")
    if t.cmd in COMMANDS:
        print()
        build_parser().parse_args([t.cmd, "--help"])


def _md(text: str) -> str:
    """表の中に入れても崩れないように整える。"""
    return text.replace("|", "／").replace("\n", " ")


def cmd_catalog(markdown: bool, reference: int | None) -> None:
    if reference:
        c = reference
        print(f"# {c}/4 {catalog.CATEGORIES[c]}\n")
        print(
            "`uv run python -m vfx <コマンド> --help` で引数の詳細が出る（tools/ で実行）。\n"
        )
        for t in catalog.TECHNIQUES:
            if t.cat != c:
                continue
            print(f"## {t.no}. {t.name}（`{t.cmd}`） {_status(t.no)}\n")
            print(f"- 何をする: {t.what}")
            print(f"- 頼み方の例: 「{t.example}」")
            for field, label in catalog.NOTE_KEYS.items():
                value = catalog.note(t.no, field)
                if field != "status" and value:
                    print(f"- {label.split('（')[0]}: {value}")
            if t.cmd in COMMANDS:
                usage = (
                    build_parser()
                    ._subparsers._group_actions[0]
                    .choices[t.cmd]
                    .format_usage()
                )
                print(
                    f"\n```bash\ncd tools && uv run {usage.replace('usage: ', '').strip()}\n```"
                )
            print()
        return
    rows = []
    for t in catalog.TECHNIQUES:
        rows.append(
            (t.no, t.name, t.cmd, _status(t.no), t.what, catalog.note(t.no, "how"))
        )
    if not markdown:
        for r in rows:
            print("\t".join(str(x) for x in r))
        return
    for c, title in catalog.CATEGORIES.items():
        print(f"\n### {c}/4 {title}\n")
        print("| 番号 | 名前 | コマンド | 状態 | 何をする |")
        print("|---|---|---|---|---|")
        for no, name, cmd, st, what, _how in rows:
            if (no - 1) // 25 + 1 == c:
                print(f"| {no} | {name} | `{cmd}` | {st} | {_md(what)} |")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vfx", description="映像処理テクニック 100 選（tools/vfx）"
    )
    sub = parser.add_subparsers(dest="command", metavar="<コマンド>")
    p = sub.add_parser("list", help="100 項目と状態の一覧")
    p.add_argument("--cat", type=int, choices=[1, 2, 3, 4], help="分類で絞る")
    p = sub.add_parser("info", help="1 項目の説明")
    p.add_argument("key", help="番号かコマンド名")
    p = sub.add_parser("catalog", help="一覧表を出す")
    p.add_argument("--markdown", action="store_true", help="Markdown の表で出す")
    p.add_argument(
        "--reference",
        type=int,
        choices=[1, 2, 3, 4],
        help="分類ごとの詳しい説明を Markdown で出す",
    )
    for name, c in sorted(COMMANDS.items(), key=lambda kv: kv[1].no):
        t = catalog.BY_NO.get(c.no)
        title = f"{c.no}. {t.name}: {t.what}" if t else c.help
        sp = sub.add_parser(
            name,
            help=title,
            description=f"{title}\n{c.help}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        for flags, kwargs in c.args:
            sp.add_argument(*flags, **kwargs)
    return parser


def main(argv: list[str] | None = None) -> int:
    _load()
    for m, err in LOAD_ERRORS.items():
        eprint(
            f"警告: モジュール {m} を読み込めませんでした（そのコマンドは使えません）: {err}"
        )
    argv = list(sys.argv[1:] if argv is None else argv)
    # 番号でも呼べる（例: python -m vfx 76 in.mp4）
    if argv and argv[0].isdigit():
        c = by_no().get(int(argv[0]))
        if c is None:
            eprint(f"番号 {argv[0]} のコマンドはまだありません")
            return 2
        argv[0] = c.name
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "list"):
            cmd_list(getattr(args, "cat", None))
        elif args.command == "info":
            cmd_info(args.key)
        elif args.command == "catalog":
            cmd_catalog(args.markdown, args.reference)
        else:
            t0 = time.time()
            COMMANDS[args.command].func(args)
            print(f"（{time.time() - t0:.1f} 秒）")
    except VfxError as e:
        eprint(f"エラー: {e}")
        return 1
    return 0
