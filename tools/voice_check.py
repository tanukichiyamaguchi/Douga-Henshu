"""声の音色（こもり）と、加工で声が変わっていないかを測る。

2026-11-16 に「声がAIっぽい。(こもっている感じ)もっと自然にして欲しい。」と言われた時に使った測り方を、毎回同じ
物差しで使えるようにした物（docs/notes/rules-history.md の「毎回守ること」の声の行と、docs/notes/measurements.md の
「テンポと声の指摘で測ったこと」。2026-11-18 より前は CLAUDE.md にあった）。
Claude は音を聞けないので、数字で分かることだけを見る。自然に聞こえるかどうかは、この数字では分からない
（ユーザーに聞き比べで選んでもらう）。

使い方（tools/ ディレクトリで実行）:
  uv run python voice_check.py 声1.mp3 声2.mp3                              # 声ごとの数字（候補を比べる）
  uv run python voice_check.py --before 受け取った声.mp3 --after 加工後.wav    # 加工の前後で比べる
  uv run python voice_check.py 声.mp3 --json voice.json                     # 数字を JSON にも保存

測る物（声のある所だけ。300〜4000Hz の大きさが、大きい方から 25dB 以内のコマ）:
  帯域の分け前: 80Hz〜16kHz の全体を 0dB とした時の、帯域ごとの大きさ（dB）
  音のある上の限り: 声の中心の帯域より 60dB 小さくなる周波数
  大きさの幅: 50 ミリ秒ごとの大きさ（dB）の、上から 5% の点と下から 10% の点の差（小さいほど、ならしてある）

2026-11-16 に測った数字（比べる時の物差し。同じ声でも、文によって 2dB ほど変わる）:
  Harune・eleven_v4（聞き比べでユーザーが選んだ標準の声）: 200〜500Hz −3.5〜−5.2・2〜5kHz −11.8〜−15.0・上の限り 約 12.6kHz
  こもった音色だった声（Chihiro）: 200〜500Hz −1.2〜−1.4・2〜5kHz −17.3〜−19.7・上の限り 約 10.7kHz
  口の動きに合わせる伸び縮み（WSOLA）の後: 5〜12kHz が約 1〜3.5dB 下がった
  強い圧縮（2.5:1・立ち上がり 5ms）の後: 大きさの幅が約 20dB → 約 15dB

下の「目安」は、5 つの声から Claude が決めた仮の線。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import stft

SR = 48000
BANDS = [
    (80, 200),
    (200, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 5000),
    (5000, 8000),
    (8000, 12000),
    (12000, 16000),
]
BOXY = -2.5  # 200〜500Hz の分け前がこれより大きい = 低い所に偏っている（こもり気味）
WEAK = -16.0  # 2〜5kHz の分け前がこれより小さい = 言葉がはっきりしにくい
HIGH_DROP = 1.0  # 加工の後、5〜12kHz がこれより下がったら知らせる（dB）
RANGE_DROP = 2.0  # 加工の後、大きさの幅がこれより狭まったら知らせる（dB）


def band_name(lo: int, hi: int) -> str:
    def f(v: int) -> str:
        return f"{v // 1000}k" if v >= 1000 else str(v)

    return f"{f(lo)}-{f(hi)}"


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def file_format(path: Path) -> str:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,bit_rate",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return out or "?"


def measure(path: Path) -> dict:
    x = decode(path)
    if len(x) < SR // 2:
        raise SystemExit(f"短すぎる: {path}")
    f, _, z = stft(x, fs=SR, nperseg=2048, noverlap=1536, window="hann")
    p = np.abs(z) ** 2
    core = (f >= 300) & (f < 4000)
    e = p[core].sum(axis=0)
    active = e > np.percentile(e, 95) * 10 ** (-25 / 10)  # 声のあるコマ
    ltas = p[:, active].mean(axis=1)
    total = ltas[(f >= 80) & (f < 16000)].sum()
    shares = {
        band_name(lo, hi): round(
            float(10 * np.log10(ltas[(f >= lo) & (f < hi)].sum() / total + 1e-20)), 1
        )
        for lo, hi in BANDS
    }
    # 音のある上の限り: ならした大きさが、声の中心の帯域の山より 60dB 小さくなる所
    dens = 10 * np.log10(ltas + 1e-20)
    smooth = np.convolve(dens, np.ones(15) / 15, mode="same")
    below = np.where((f > 3000) & (smooth < smooth[core].max() - 60))[0]
    top = float(f[below[0]]) if len(below) else SR / 2
    # 大きさの幅と、山と平均の差
    n = SR // 20
    fr = x[: len(x) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((fr**2).mean(axis=1) + 1e-20))
    loud = db > db.max() - 35
    xt = x[: len(fr) * n]
    crest = 20 * np.log10(
        np.abs(xt).max() / np.sqrt((xt[np.repeat(loud, n)] ** 2).mean())
    )
    return {
        "file": str(path),
        "format": file_format(path),
        "seconds": round(len(x) / SR, 2),
        "shares": shares,
        "top_khz": round(top / 1000, 1),
        "range_db": round(
            float(np.percentile(db[loud], 95) - np.percentile(db[loud], 10)), 1
        ),
        "crest_db": round(float(crest), 1),
    }


def notes(m: dict) -> list[str]:
    out = []
    if m["shares"]["200-500"] > BOXY:
        out.append(
            f"200〜500Hz に偏っている（{m['shares']['200-500']}。目安 {BOXY} より小さく）= こもり気味"
        )
    if m["shares"]["2k-5k"] < WEAK:
        out.append(
            f"2〜5kHz が弱い（{m['shares']['2k-5k']}。目安 {WEAK} より大きく）= はっきりしにくい"
        )
    return out


def show(m: dict) -> None:
    print(f"{Path(m['file']).name}（{m['format']}・{m['seconds']} 秒）")
    print("  " + " ".join(k.rjust(7) for k in m["shares"]))
    print("  " + " ".join(f"{v:7.1f}" for v in m["shares"].values()))
    print(
        f"  音のある上の限り {m['top_khz']}kHz・大きさの幅 {m['range_db']}dB・山と平均の差 {m['crest_db']}dB"
    )
    for line in notes(m) or ["帯域の偏りは、目安の内"]:
        print(f"  → {line}")


def compare(before: dict, after: dict) -> list[str]:
    out = []
    for key in ("5k-8k", "8k-12k"):
        d = round(after["shares"][key] - before["shares"][key], 1)
        if d < -HIGH_DROP:
            out.append(
                f"高い音（{key}）が {-d}dB 下がっている（目安 {HIGH_DROP}dB まで）"
            )
    d = round(after["range_db"] - before["range_db"], 1)
    if d < -RANGE_DROP:
        out.append(
            f"大きさの幅が {-d}dB 狭まっている（目安 {RANGE_DROP}dB まで）= ならしすぎ"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="声の音色（こもり）と、加工の前後の変化を測る"
    )
    parser.add_argument(
        "files", type=Path, nargs="*", help="声のファイル（mp3・wav・mp4 など）"
    )
    parser.add_argument("--before", type=Path, help="加工する前の声")
    parser.add_argument("--after", type=Path, help="加工した後の声")
    parser.add_argument("--json", type=Path, help="数字を JSON で保存する先")
    args = parser.parse_args()
    if bool(args.before) != bool(args.after):
        parser.error("--before と --after は、両方を指定する")
    if not args.files and not args.before:
        parser.error("声のファイルか、--before と --after を指定する")
    result: dict = {"files": [measure(p) for p in args.files]}
    for m in result["files"]:
        show(m)
    if args.before:
        b, a = measure(args.before), measure(args.after)
        print("加工の前")
        show(b)
        print("加工の後")
        show(a)
        diff = compare(b, a)
        print("加工の前後の差")
        for line in diff or ["高い音と大きさの幅は、目安の内（声は、ほぼそのまま）"]:
            print(f"  → {line}")
        result["before"], result["after"], result["changes"] = b, a, diff
    print(
        "（目安は Claude が決めた仮の線。自然に聞こえるかどうかは、この数字では分からない）"
    )
    if args.json:
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"保存: {args.json}")


if __name__ == "__main__":
    main()
