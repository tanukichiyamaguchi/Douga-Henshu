"""動画のテンポを測る: 絵の動き（0.1 秒ごとのコマの差）と、話す速さ・文の間。

2026-11-16 に「直近2つはテンポが悪い。今後はかなりテンポよく、見る人が飽きないようなテンポにして。」と言われた時に
使った測り方を、毎回同じ物差しで使えるようにした物（docs/notes/rules-history.md の「毎回守ること」のテンポの行と、
docs/notes/measurements.md の「テンポと声の指摘で測ったこと」。2026-11-18 より前は CLAUDE.md にあった）。
下の「目安」は、その時の 2 本と参考動画を測って Claude が決めた数字で、ユーザーが数字で指定した物ではない。

使い方（tools/ ディレクトリで実行）:
  uv run python tempo_check.py ../work/<作品>/out.mp4                      # 絵のテンポ
  uv run python tempo_check.py out.mp4 --lines lines.json                  # 話す速さと、文の間も
  uv run python tempo_check.py out.mp4 --json tempo.json                   # 数字を JSON にも保存
  uv run python tempo_check.py --frames-dir コマのフォルダ                   # 0.1 秒ごとの画像（jpg・png）から測る

lines.json の形（文ごとの読みと、動画の中の時刻）:
  [{"kana": "まつげぱーま、すぐとれちゃうひとみて", "start": 0.12, "end": 2.62}, ...]
  kana は、その文の読み（ひらがな・カタカナ）。数字・英字・漢字は、読みに直して書く
  （漢字は、pykakasi が入っていれば読みに直すが、数字と英字は直らない）。

測る物:
  絵: 0.1 秒ごとのコマ（161×349 の白黒に縮めた物）で、となりのコマとの差の平均（0〜255）を出す。
      3 より大きい = 動いている／9 より大きい = 大きな動き（文字が飛び込む・絵が入れ替わる）。
      続けて起きた物は 1 回に数える。
  カット: となりのコマとの似かた（明るさの平均を引いた相関）が 0.6 より小さい所を、絵がまるごと替わった切り替えとする
      （0.5 秒以内に続いた物は 1 回）。暗い場面どうしの切り替えも拾える。手前で文字や図形だけが替わる所
      （背景が同じ）は、カットには数えない。
  声: モーラの数（ゃ・ゅ・ょ などの小さい字は数えない。ー・っ・ん は 1 つ）を、話している時間で割る。

2026-11-16 に測った数字（比べる時の物差し）:
  参考動画（14.5 秒）: 大きな動き 0.90 回/秒・動いている時間 74%・大きな動きの無い最長 2.7 秒
  指摘された 2 本（30 秒）: 大きな動き 0.43 回/秒・動いている時間 39〜40%・大きな動きの無い最長 3.3〜6.7 秒
"""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

W, H = 161, 349  # 2026-11-16 の測定と同じ大きさ（比べられるように、変えない）
STEP = 0.1  # コマの間隔（秒）
MOVING, BIG = 3.0, 9.0  # コマの差のしきい値（0〜255）
CUT_CORR = 0.6  # となりのコマとの相関が、これより小さい = 絵がまるごと替わった
CUT_MERGE = 0.5  # この秒数の内に続いた切り替えは、1 回に数える

# 目安（Claude が決めた数字。参考で、決まりではない。docs/notes/rules-history.md の「毎回守ること」のテンポの行と同じ）
GOAL_BIG_PER_SEC = 1.0  # 大きな動きは 1 秒に 1 回以上
GOAL_MOVING = 0.70  # 画面が動いている時間は 7 割以上
GOAL_QUIET_MAX = 1.5  # 大きな動きの無い時間は 1.5 秒まで
GOAL_SHOT_MAX = 2.0  # 同じ絵は長くて 2 秒
GOAL_SHOT_MEAN = (1.0, 1.5)  # 同じ絵の長さの平均
GOAL_GAP_MAX = 0.15  # 文の間は 0.15 秒まで
GOAL_SILENCE = 0.05  # 声の無い時間は全体の 5% まで

SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")


def frames_from_video(path: Path) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={round(1 / STEP)},scale={W}:{H},format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)


def frames_from_dir(folder: Path) -> np.ndarray:
    files = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not files:
        raise SystemExit(f"画像が見つからない: {folder}")
    return np.stack(
        [
            cv2.resize(
                cv2.imread(str(p), cv2.IMREAD_GRAYSCALE),
                (W, H),
                interpolation=cv2.INTER_AREA,
            )
            for p in files
        ]
    ).astype(np.float32)


def starts(mask: np.ndarray) -> np.ndarray:
    """続けて起きた物を 1 回にまとめた時の、始まりの位置。"""
    if mask.size == 0:
        return np.zeros(0, dtype=int)
    first = np.concatenate([[bool(mask[0])], mask[1:] & ~mask[:-1]])
    return np.where(first)[0]


def similarity(frames: np.ndarray) -> np.ndarray:
    """となりのコマとの相関（明るさの平均を引いてから）。1 に近いほど、同じ絵。"""
    x = frames.reshape(len(frames), -1)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.sqrt((x * x).sum(axis=1)) + 1e-9
    return (x[:-1] * x[1:]).sum(axis=1) / (norm[:-1] * norm[1:])


def visual(frames: np.ndarray) -> dict:
    diff = np.abs(np.diff(frames, axis=0)).mean(
        axis=(1, 2)
    )  # 0.1 秒ごとの、画面全体の変化の大きさ
    dur = len(frames) * STEP
    big = starts(diff > BIG)
    cut = []
    for i in starts(similarity(frames) < CUT_CORR):
        if not cut or (i - cut[-1]) * STEP > CUT_MERGE:
            cut.append(int(i))
    big_idx = np.where(diff > BIG)[0]
    quiet = (
        np.diff(np.concatenate([[-1], big_idx, [len(diff)]])) * STEP
    )  # 大きな動きの無い時間（始めと終わりも入れる）
    cut_times = [round(float((i + 1) * STEP), 1) for i in cut]
    edges = [0.0, *cut_times, dur]
    shots = [round(b - a, 1) for a, b in pairwise(edges) if b - a > 1e-6]
    return {
        "duration": round(dur, 1),
        "change_mean": round(float(diff.mean()), 1),
        "change_median": round(float(np.median(diff)), 1),
        "big_moves": len(big),
        "big_moves_per_sec": round(len(big) / dur, 2),
        "moving_ratio": round(float((diff > MOVING).mean()), 2),
        "longest_without_big_move": round(float(quiet.max()), 1),
        "cuts": len(cut),
        "cut_times": cut_times,
        "shot_count": len(shots),
        "shot_mean": round(float(np.mean(shots)), 2),
        "shot_max": round(float(np.max(shots)), 1),
        "shots": shots,
    }


def to_kana(text: str) -> str:
    """漢字が混じっていたら、pykakasi があれば読みに直す。"""
    if not any("一" <= c <= "鿿" for c in text):
        return text
    try:
        import pykakasi
    except ImportError:
        raise SystemExit(
            f"漢字が入っている（{text}）。読みをひらがなで書くか、pykakasi を入れる"
        ) from None
    return "".join(item["hira"] for item in pykakasi.kakasi().convert(text))


def morae(kana: str) -> int:
    return sum(
        1
        for c in kana
        if c not in SMALL_KANA and ("ぁ" <= c <= "ゖ" or "ァ" <= c <= "ヺ" or c == "ー")
    )


def speech(lines: list[dict], dur: float) -> dict:
    rows = []
    for ln in lines:
        kana = to_kana(ln["kana"])
        left = [c for c in kana if c.isascii() and c.isalnum()]
        if left:
            raise SystemExit(
                f"数字・英字が残っている（{ln['kana']}）。読みをひらがなで書く"
            )
        rows.append(
            {
                "kana": kana,
                "morae": morae(kana),
                "start": float(ln["start"]),
                "end": float(ln["end"]),
            }
        )
    talk = sum(r["end"] - r["start"] for r in rows)
    total = sum(r["morae"] for r in rows)
    raw_gaps = [b["start"] - a["end"] for a, b in pairwise(rows)]
    gaps = [round(g, 2) for g in raw_gaps]
    return {
        "lines": len(rows),
        "morae": total,
        "morae_per_sec_talking": round(total / talk, 1),
        "morae_per_sec_overall": round(total / dur, 1),
        "per_line": [round(r["morae"] / (r["end"] - r["start"]), 1) for r in rows],
        "gaps": gaps,
        "gap_mean": round(float(np.mean(raw_gaps)), 2) if gaps else 0.0,
        "gap_max": max(gaps) if gaps else 0.0,
        "first_word_at": round(rows[0]["start"], 2),
        "silence_ratio": round((dur - talk) / dur, 2),
        "line_mean_sec": round(talk / len(rows), 1),
    }


def mark(ok: bool) -> str:
    return "○" if ok else "×"


def report(v: dict, s: dict | None) -> None:
    print(f"長さ {v['duration']} 秒")
    print("絵の動き")
    print(
        f"  {mark(v['big_moves_per_sec'] >= GOAL_BIG_PER_SEC)} 大きな動き {v['big_moves']} 回（1 秒に {v['big_moves_per_sec']} 回。目安 {GOAL_BIG_PER_SEC} 回以上）"
    )
    print(
        f"  {mark(v['moving_ratio'] >= GOAL_MOVING)} 画面が動いている時間 {v['moving_ratio'] * 100:.0f}%（目安 {GOAL_MOVING * 100:.0f}% 以上）"
    )
    print(
        f"  {mark(v['longest_without_big_move'] <= GOAL_QUIET_MAX)} 大きな動きの無い、いちばん長い時間 {v['longest_without_big_move']} 秒（目安 {GOAL_QUIET_MAX} 秒まで）"
    )
    print("カット（絵がまるごと替わる切り替えで区切った、同じ絵が続く長さ）")
    print(
        f"  {mark(v['shot_max'] <= GOAL_SHOT_MAX)} いちばん長いカット {v['shot_max']} 秒（目安 {GOAL_SHOT_MAX} 秒まで）"
    )
    lo, hi = GOAL_SHOT_MEAN
    print(
        f"  {mark(v['shot_mean'] <= hi)} カット {v['shot_count']} 個・平均 {v['shot_mean']} 秒（目安 {lo}〜{hi} 秒）"
    )
    print(
        f"    切り替えの時刻（秒）: {' '.join(str(t) for t in v['cut_times']) or 'なし'}"
    )
    print(f"    カットの長さ（秒）: {' '.join(str(x) for x in v['shots'])}")
    if s:
        print("声")
        print(
            f"    {s['lines']} 文・{s['morae']} モーラ。話している間で 1 秒に {s['morae_per_sec_talking']} モーラ（動画全体では {s['morae_per_sec_overall']}）"
        )
        print(f"    文ごと（モーラ/秒）: {' '.join(str(x) for x in s['per_line'])}")
        print(
            f"  {mark(s['gap_max'] <= GOAL_GAP_MAX)} 文の間 平均 {s['gap_mean']} 秒・最長 {s['gap_max']} 秒（目安 {GOAL_GAP_MAX} 秒まで。見せ場の間を 1 か所だけ残すのはよい）"
        )
        print(
            f"  {mark(s['silence_ratio'] <= GOAL_SILENCE)} 声の無い時間 {s['silence_ratio'] * 100:.0f}%（目安 {GOAL_SILENCE * 100:.0f}% まで）"
        )
        print(
            f"  {mark(s['first_word_at'] <= 0.3)} 最初の言葉 {s['first_word_at']} 秒（目安 0.3 秒以内）"
        )
    print(
        "（目安は、2026-11-16 のテンポの指摘の後に Claude が決めた数字。参考で、決まりではない。docs/notes/rules-history.md）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="動画のテンポ（絵の動き・話す速さ）を測る"
    )
    parser.add_argument("video", type=Path, nargs="?", help="動画ファイル（mp4 など）")
    parser.add_argument(
        "--frames-dir", type=Path, help="0.1 秒ごとの画像のフォルダ（動画の代わり）"
    )
    parser.add_argument(
        "--lines", type=Path, help="文ごとの読みと時刻の JSON（上の説明）"
    )
    parser.add_argument("--json", type=Path, help="数字を JSON で保存する先")
    args = parser.parse_args()
    if (args.video is None) == (args.frames_dir is None):
        parser.error("動画ファイルか --frames-dir の、どちらか 1 つを指定する")
    frames = (
        frames_from_video(args.video)
        if args.video
        else frames_from_dir(args.frames_dir)
    )
    if len(frames) < 3:
        raise SystemExit("コマが少なすぎる")
    v = visual(frames)
    s = (
        speech(json.loads(args.lines.read_text(encoding="utf-8")), v["duration"])
        if args.lines
        else None
    )
    report(v, s)
    if args.json:
        args.json.write_text(
            json.dumps({"visual": v, "speech": s}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"保存: {args.json}")


if __name__ == "__main__":
    main()
