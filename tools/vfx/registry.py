"""コマンドの登録。各テクニックの関数に @command(番号, 名前, 引数...) を付けると CLI に並ぶ。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Arg = tuple[tuple[str, ...], dict[str, Any]]


@dataclass
class Command:
    no: int
    name: str
    func: Callable[[argparse.Namespace], Any]
    args: list[Arg] = field(default_factory=list)
    help: str = ""


COMMANDS: dict[str, Command] = {}


def arg(*flags: str, **kwargs: Any) -> Arg:
    return flags, kwargs


# よく使う引数のひな形
SRC = arg("src", help="入力（動画か画像）")
OUT = arg("-o", "--out", help="出力先（省略時は out/vfx/<元の名前>-<処理>.<拡張子>）")
JSON = arg(
    "--json",
    help="結果の数値を書き出す JSON の場所（省略時は出力先と同じ名前の .json）",
)
RANGE = [
    arg("--start", type=float, default=0.0, help="処理を始める秒（既定 0）"),
    arg("--duration", type=float, help="処理する長さ（秒。省略時は最後まで）"),
    arg("--max-frames", type=int, help="処理するコマ数の上限"),
]
SIZE = [
    arg(
        "--max-side",
        type=int,
        help="処理する時の長辺の上限（px。速くしたい時に小さくする）",
    )
]


def command(
    no: int, name: str, *args: Arg | list[Arg], help: str = ""
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    flat: list[Arg] = []
    for a in args:
        if isinstance(a, list):
            flat.extend(a)
        else:
            flat.append(a)

    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in COMMANDS:
            raise ValueError(f"同じ名前のコマンドがあります: {name}")
        COMMANDS[name] = Command(
            no, name, func, flat, help or (func.__doc__ or "").strip().splitlines()[0]
        )
        return func

    return deco


def by_no() -> dict[int, Command]:
    return {c.no: c for c in COMMANDS.values()}
