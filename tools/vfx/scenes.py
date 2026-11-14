"""自動で見つける（時間の流れの解析）。

担当するテクニック:
   76 シーン検出（scenes）
   83 キーフレーム抽出（keyframes）
   88 黒コマ検出（black-frames）
   89 重複フレーム検出（duplicates）
   92 フェード検出（fades）
   93 ピンぼけ検出（blur-detect）
   99 フラッシュ検出（flashes）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from . import core
from .registry import JSON, OUT, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {
    88: {
        "status": "◎",
        "how": "ffmpeg の blackdetect（画素の明るさが pix_th 以下の画素が pic_th 以上を占めるコマが、d 秒以上続く区間）。"
        "区間の始め・中・終わりのコマを並べた確認画像も作る",
        "measured": "黒を差し込んだ合成映像（30fps）で、0〜0.5 秒・1.0〜1.5 秒・1.5〜1.6 秒・2.6〜2.667 秒（2 コマ）を"
        "すべてコマ単位で正しく検出（2 コマの黒は --min-duration 0.03 で検出）。OpenCV の Megamind.avi（先頭の 1 コマだけ黒）は"
        " 1 コマ分の区間（0.042〜0.083 秒と表示）として検出。黒の無い広告 v4 の完成動画（20 秒）は 0 件。20 秒の動画で約 1 秒",
        "limits": "暗い夜景や黒い服のアップも黒と判定されることがある（--pix-th と --pic-th で調整）。"
        "完全な黒でない暗転は blackdetect では拾えないので、フェード検出（92）を使う",
        "tested": "合成した試験映像（色の画面の途中に黒）、OpenCV の Megamind.avi、広告 v4 の完成動画",
    },
}

_BLACK = re.compile(
    r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)"
)


@command(
    88,
    "black-frames",
    SRC,
    OUT,
    JSON,
    arg(
        "--min-duration",
        type=float,
        default=0.05,
        help="黒が続く最短の長さ（秒。既定 0.05 ≒ 2 コマ）",
    ),
    arg(
        "--pix-th",
        type=float,
        default=0.10,
        help="黒とみなす画素の明るさの上限（0〜1。既定 0.10）",
    ),
    arg(
        "--pic-th",
        type=float,
        default=0.98,
        help="黒い画素が画面に占める割合の下限（0〜1。既定 0.98）",
    ),
)
def black_frames(a: argparse.Namespace) -> None:
    """真っ黒な区間を探し、時刻を JSON に、前後のコマを並べた確認画像を JPG に書く。"""
    info = core.probe(a.src)
    core.require(not info.is_image, "動画を指定してください")
    vf = f"blackdetect=d={a.min_duration}:pix_th={a.pix_th}:pic_th={a.pic_th}"
    text = core.ffmpeg_log(["-i", str(a.src), "-vf", vf, "-an", "-f", "null", "-"])
    segments = [
        {"start": float(s), "end": float(e), "duration": float(d)}
        for s, e, d in _BLACK.findall(text)
    ]
    out = Path(a.out) if a.out else core.default_out(a.src, "black-frames", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "duration": info.duration,
            "params": {
                "min_duration": a.min_duration,
                "pix_th": a.pix_th,
                "pic_th": a.pic_th,
            },
            "segments": segments,
        },
    )
    core.log(f"黒い区間: {len(segments)} 件")
    for s in segments:
        core.log(f"  {s['start']:.3f}〜{s['end']:.3f} 秒（{s['duration']:.3f} 秒）")
    # 確認画像: 区間ごとに「直前・中・直後」の 3 コマ（最大 8 区間）
    images, labels = [], []
    for s in segments[:8]:
        for label, t in (
            ("直前", s["start"] - 0.1),
            ("中", (s["start"] + s["end"]) / 2),
            ("直後", s["end"] + 0.05),
        ):
            t = min(max(t, 0.0), max(info.duration - 0.05, 0.0))
            images.append(core.frame_at(a.src, t, max_side=480))
            labels.append(f"{label} {t:.2f}秒")
    if images:
        core.save_image(out, core.contact_sheet(images, labels, cols=3, cell_width=320))
        core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")
