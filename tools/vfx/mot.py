"""動きの解析と自動検出（複数の人の検出・追跡）。

担当するテクニック:
   12 背景差分（bg-subtract）
   15 複数物体追跡（multi-track）
   85 人物の再識別（reid）
   95 人数カウント（count-people）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import core
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {
    12: {
        "status": "◎",
        "how": "OpenCV の背景モデル（既定 KNN。MOG2 も選べる）か、範囲全体から取ったコマの画素ごとの中央値（median）、"
        "または背景だけの画像（--bg）との色の差で前景を求める。始める前に範囲全体から 30 コマ取り、中央値から離れた所"
        "（動く物）を中央値で置き換えたコマを覚えさせる（下準備）。影（明るさだけ下がり色味が同じ所）は背景として分け、"
        "開く・閉じる処理と小さな塊の除去で整える。--hold 秒より長く止まった物は背景に溶ける（秒数から覚えるコマ数を決める）。"
        "出力: 白黒の地図か元の映像に重ねた表示の MP4、--cutout で前景だけの透過 WebM（画像は PNG）、背景の推定 -bg.jpg"
        "（動く物を消した画）、コマごとの前景の割合と塊の枠の JSON。外部のモデルは使わない",
        "measured": "正解の地図つきの合成映像（vtest の背景＋AI 生成の人物 2 人と赤い玉＋足元の影＋ノイズ σ3。768×576・"
        "150 コマ。最初の 10 コマを除く）の IoU: knn 0.970（適合率 0.986・再現率 0.983）・mog2 0.970・median 0.954・"
        "--bg 0.949、静止画＋--bg 0.956。影を前景と誤った割合 3.0%（--shadows keep では 98%）。下準備なしでは knn 0.21・"
        "mog2 0.86（取ったコマをそのまま覚えさせた最初の版は通った人まで背景になり 0.62）。止まった四角が背景に溶けるまで: "
        "--hold 2/5/10 で knn 2.0/4.8/10.8 秒、mog2 2.0/5.3 秒。vtest（実写の固定カメラ、30 秒）では歩く人だけが赤く、"
        "足元の影が青く分かれ、建物や道に誤検出が無いことを目で確認（下準備を直す前の mog2 は建物・道に大きな誤検出が出ていた）。"
        "カメラを動かした試験（vtest を 1 コマ 1px・3px ずつ横に流した切り出し）では、前景の割合が固定の 1.6% から "
        "12.7%・23.5% に増えた（背景の誤検出）。速さ: 768×576・300 コマで約 10〜14 秒（白黒の地図。下準備を含む。"
        "CPU を他の作業と分け合った状態）",
        "limits": "固定カメラ専用（カメラが動くと背景を前景と誤る。上の試験で 1 コマ 1px のパンでも前景の割合が 8 倍。先に手ぶれ補正で"
        "止めるか、人物の切り抜きなど別の方法を使う）。"
        "範囲の半分以上の時間同じ所にいる物は、下準備の中央値で始めから背景になる（--warmup 0 で避けられるが、その場合は誤検出が長く続く。合成映像で knn の IoU 0.21・mog2 0.86）。--hold より長く止まった人は消える。服の色が背景に近い所は抜ける（クリーム色のニットと灰色の道で"
        "再現率 0.98）。照明が急に変わる映像は試していない（背景との差が広く出るはず）。影の判定は「暗くなっただけで色味が"
        "同じ」ことによる（明るさが 0.35 倍未満のごく濃い影は判定の外）ので、"
        "色の付いた照明の影や、ごく濃い影は前景に残ることがある。画像 1 枚には --bg（背景だけの画像）が要る",
        "tested": "合成映像と正解の地図（vfx-tests/mot/bg_synth.py・bg_eval.py）、止まる四角の合成映像（bg_stop.py）、"
        "横に流したカメラの動き（bg_pan.py）、"
        "vtest.mp4 の 30 秒（bg_vtest_review.py。重ねた表示・白黒の地図・背景の推定・透過 WebM を画像で確認）",
    },
    15: {
        "status": "○",
        "how": "人の検出は共有の detectors（accurate: torchvision〔BSD-3-Clause〕の Faster R-CNN v2 / fast: MediaPipe の "
        "EfficientDet-Lite2〔Apache-2.0〕。どちらも COCO の 80 種類で、--classes で車なども追える）。追跡は ByteTrack と同じ"
        "考え方を自作: 等速のカルマンフィルター（枠の中心・縦横比・高さと速さ）で次の位置を予測し、ハンガリアン法（scipy）"
        "で IoU を照合する。1 段目は確信度 0.5 以上の検出を追跡中・見失い中の人へ、2 段目は 0.1〜0.5 の検出（隠れかけた人）を"
        "残った追跡中の人へ結ぶ。新しい番号は 0.6 以上の検出から始め、次のコマでも結べたら確定。見失っても --keep 秒は"
        "同じ番号で待つ。カットの変わり目（縮小したコマの HSV の差が 27 以上）では追跡をやり直す。出力: 番号つきの枠と"
        "足元の軌跡を描いた MP4、番号ごとの時刻と枠（元の大きさの px）の JSON、12 コマを並べた確認画像",
        "measured": "追跡だけの試験（正解の枠に 2px のゆらぎ・見落とし 10%・誤検出 5%・重なると確信度が下がる、6 人・200 コマ・"
        "10 通りの平均）: IDF1 0.847・MOTA 0.801・入れ替わり 1.3 回・番号 7.0 個（正解 6 人）。重なりで確信度が下がらない"
        "場合は IDF1 0.942・入れ替わり 0 回・番号 6.0 個。検出から通した合成映像（同じ服の AI 生成の人物 4 人が交差し、"
        "手前の人が奥の人を隠す。768×576・10fps・120 コマ）: accurate で MOTA 0.895・IDF1 0.772・見落とし 33/382・"
        "誤検出 5・番号 4 個（正解 4 人）・入れ替わり 1 組（2 人が交差して番号を取り違えた）。入れ子の枠を除かない"
        "（--nested 1.1）と MOTA 0.929・IDF1 0.776・見落とし 18・番号 6 個（短い余分な番号が出る）。fast は MOTA 0.775・"
        "IDF1 0.610・入れ替わり 5 回。vtest（実写の固定カメラ、0〜25 秒・250 コマ。入れ子の除去を入れる前の版）: accurate で"
        "番号 20 個、fast で 17 個。標識のそばで 3 人が重なった所（1.5〜4.2 秒）を拡大して見ると、標識に隠れた人が 3→6→3、"
        "黒いコートの人が 2→6 に入れ替わり、黒い上着の人は 1 のままだった。story540（AI 生成の広告、6 コマ/秒）では"
        "カットごとに 1 人に 1 つの番号（洗面所・日差し・顔の寄り・カフェで番号 3〜6）で、同じ人に 2 つの番号が付くことは"
        "無かった（目の寄りのカットだけ、顔の上下に 2 つの枠と番号）。カットの変わり目: story540 で 3.042・5.792・8.875・"
        "12.375 秒（正解 3.0・5.79・8.88・12.38 秒。1 コマ以内）、固定カメラの vtest では誤りなし（差の最大 4.6）。"
        "速さ: accurate は 1 コマ約 2〜2.5 秒（CPU が空いている時）、他の作業と分け合った状態では 4.8 秒（768×576）〜"
        "15 秒（540×960）。fast は 1 コマ 0.13 秒",
        "limits": "見た目の特徴を使わず、位置と動きだけで結ぶ（ByteTrack と同じ）。そのため、似た服の人が重なって隠れたり、"
        "体の一部だけが見えて枠の形が変わったりすると、番号が入れ替わる・新しい番号になることがある。本格的な追跡"
        "（BoT-SORT・StrongSORT など）が使う、見た目の特徴での照合とカメラの動きの補正は入れていない。カメラが速く動くと IoU が合わず番号が切れやすいはず（試していない）。カットの変わり目では番号を付け直すので、別のカットや長く隠れた"
        "後の同じ人をつなぐのは 85（reid）を使う。同じ人の頭だけの枠などを除く処理（--nested）は、重なって隠れた人も除く"
        "ことがある。accurate は遅いので、長い映像は --fps 10 などに下げるか --model fast（小さい人・重なった人を見落としやすい）",
        "tested": "追跡だけの試験（vfx-tests/mot/tracker_unit.py）、合成映像と正解の枠（mot_synth.py・mot_eval.py）、"
        "vtest.mp4 の 25 秒（accurate と fast。人が重なった所を元のコマと並べて拡大: mt_review.py・mt_vtest_review.py）、"
        "カットの変わり目（cuts_check.py）",
    },
    85: {
        "status": "○",
        "how": "カットの変わり目（15 と同じ判定）ごとに、人を検出して（共有 detectors。既定 accurate）15 と同じ追跡で「区間」"
        "（同じカットで続けて追えた 1 人分）にまとめ、区間ごとに顔の特徴量（YuNet〔MIT〕で顔を見つけ、SFace〔Apache-2.0〕"
        "で 128 次元）と体の特徴量（OpenCV Zoo の YouTu ReID〔Apache-2.0〕で 768 次元。服装・体つき）を平均する。体は他の人"
        "と重ならないコマを優先する。区間どうしの似ている度合い（コサイン）を、顔は 0.363（OpenCV の推奨値）、体は 0.60 を"
        "基準に点にし（両方ある時は顔 0.75・体 0.25 の重み）、同じコマの別の場所に映った区間は同じ人にしない条件つきの"
        "平均連結クラスタリングで「人物 1, 2, …」にまとめる。人として見つからない顔の寄りは、顔の周りを人の枠にする。"
        "複数の動画・画像をまたいでもつなぐ。出力: 人物ごとに場面の切り抜きを並べた一覧 JPG、人物ごとの出てくる時刻と枠"
        "の JSON（区間どうしの顔・体の類似度も）、--video で通しの番号を書いた MP4。DINOv2 small（Apache-2.0）の特徴量も"
        "比べたが、見分けが悪かったので使わない",
        "measured": "体の特徴量（vtest を 10fps で追った 13 人を前半・後半に分けた組）: YouTu ReID は後半から同じ人を 1 番に"
        "選べた割合 0.923・AUC 0.985、DINOv2 small は 0.769・0.941。切り抜きを目で見て、同じ黒いコートの人と分かった番号"
        "（2・8・19）をまとめ、途中で別の人に入れ替わっていた番号（1・6）を除いて数え直すと、同じ人 23 組は 0.655〜"
        "（中央値 0.817）、別の人 208 組は中央値 0.278・最大 0.676（AUC 1.000）。しきい値 0.60 で同じ人 23/23・別の人を"
        "同じと誤る 5/208、0.65 で 23/23・1/208。story540（AI 生成の広告、カット 5 つ、4 コマ/秒で調べて 228〜262 秒）: "
        "カットの変わり目 4 か所はすべて 1 コマ以内。主人公が映るカットのうち、洗面所・日差し・カフェの 3 つを 1 人に"
        "まとめた（顔の類似度 0.69〜0.76）。顔の極端な寄りのカットは顔の類似度 0.31〜0.38 で別の人になった（顔を見つけられた"
        "のは 14 コマ中 3 コマ）。目の寄りのカットは顔の上と下に枠が 2 つ出て 2 人になった。画像 4 枚（主人公の 3 カットの"
        "静止画と vtest の 1 コマ）: 主人公の 3 枚を 1 人にまとめ、vtest の 7 人はすべて別の人（13 秒）。--no-face（体だけ）"
        "では、洗面所とカフェの静止画の体の類似度が 0.456 で、同じ人にできなかった。vtest の 25 秒"
        "（4 コマ/秒・574 秒）: 22 区間 → 12 人。10fps の追跡の番号を基準にすると、別の人を別にできた組 142/146・同じ人を"
        "同じにできた組 4/7。目で見直すと、標識の陰を出入りして基準でも分かれていた黒いコートの人を 4 区間つないだのは正しく、"
        "誤りは並んで歩く 2 人の取り違え（1 区間）と、交差した人が混ざった区間を別の人につないだ（1 区間）。同じ人なのに"
        "別のままの区間が 1 つ。OpenCV 4.14 の dnn で YouTu ReID に入れる枚数を途中で変えると出力が壊れる（値が発散し"
        "最後は NaN）ことを見つけ、1 枚ずつ通すようにした（同じ枚数なら作り直した網と完全に一致）",
        "limits": "同じ人かは、顔（36px 以上で見つかった時）と服装・体つきで判断する。そのため、着替えた同じ人は顔が見えないと"
        "つながらず、似た服の別人（黒い上着どうしなど）は体だけだと似る（vtest の別人で最大 0.68）。逆に、同じ服でも照明や写り方が"
        "大きく違うカットどうしは体だけでは似ないことがある（story540 の洗面所と他のカットで 0.46〜0.54。日差しとカフェは 0.76）ので、カットをまたいでつなぐには顔が見えている方がよい。顔の極端な寄りでは顔を見つけにくく、"
        "類似度も下がってつながらないことがある（story540 の顔の寄り。横顔・うつむきは試していない）。人が交差した所で追跡が入れ替わると、"
        "その区間に 2 人分が混ざって誤ってつながる。--margin で厳しさを変えられる（負の値でまとまりやすく、正の値で分かれ"
        "やすい）。AI 生成の人物はカットごとに顔立ちが少し変わることがあり、正解そのものがあいまいなことがある。用途は"
        "自分の映像の中で同じ人をつなぐことに限る（実在の人の特定には使わない）。accurate の検出が重く、15 秒で約 4 分",
        "tested": "story540.mp4（目で決めた正解と比べた。reid_story.py）、vtest.mp4 の 25 秒（10fps の追跡の番号を基準に"
        "数え、一覧を目で見た。reid_vtest.py）、vtest の 13 人の前半・後半の組（reid_calib.py）、画像 4 枚（reid_images.py）、"
        "YouTu ReID の枚数の不具合（youtu_batch.py）。すべて vfx-tests/mot/ に置いた",
    },
    95: {
        "status": "◎",
        "how": "共有の detectors で人を見つけ（accurate: Faster R-CNN v2 / fast: EfficientDet-Lite2）、確信度の下限"
        "（accurate 0.6・fast 0.3）以上の枠のうち、大きい枠の中に面積の 65% 以上が入る枠（同じ人の頭・上半身だけの"
        "枠など）を除いて数える。動画は --fps（既定 2 コマ/秒）で間引いて数え、中央値でならした人数（--smooth 3）も出す。"
        "出力: 時刻ごとの人数と枠の JSON、人数の時間変化のグラフ（PNG）、枠と人数を書いた確認用の MP4（数えたコマだけ）"
        "と 12 コマの一覧 JPG。画像 1 枚なら、人数を書いた JPG と JSON",
        "measured": "vtest（実写の固定カメラ）の 8 コマを目で数えた人数（計 49 人）と比べて、accurate は 8/8 コマで一致、"
        "fast は 5/8 コマ（計 46 人。少なめ）。下限と入れ子の除去はこの 8 コマと story540 で決め、その後に目で数えた"
        "別の 6 コマ（計 36 人）では accurate 5/6 コマ一致（平均誤差 0.17 人）・fast 5/6 コマ（0.33 人）。入れ子の除去を"
        "入れる前（下限 0.5）は、標識の陰の人に頭だけの枠が重なり 12 秒で 3 人多く数え、story540 の 3〜13 秒（正解 1 人）"
        "で 1 人と数えたのは 12/20 コマだった（入れた後は 20/20）。人のいない写真（mp-cats_and_dogs.jpg）は 0 人。"
        "速さ: accurate は 1 コマ約 2〜2.5 秒（CPU が空いている時。分け合った状態では 540×960 の 30 コマで 142 秒）、"
        "fast は約 0.1 秒",
        "limits": "大きく写った人の枠の中に入る別の人（端に少し映る人・手前を横切る人など）は、入れ子とみなして数えない"
        "ことがある（story540 の 14.5 秒で、右端にぼけて映る 2 人目を数えなかった。--nested 1.1 で除くのをやめられる）。"
        "大きく隠れた人や遠くの小さな人は見落とすことがある（vtest の 78 秒で 1 人少ない）。fast は小さい人・重なった人を"
        "見落としやすい。目の寄りなど顔の一部だけのカットは 1 人とも 2 人とも数える（story540 の 0〜3 秒）。数えるのは"
        "各コマに映っている人数で、通った人の延べ人数ではない（延べ人数の目安は 15 multi-track の番号の数）。下限は"
        "1 つの固定カメラの 14 コマと 1 本の広告で決めた値なので、映像によっては --score の調整が要る",
        "tested": "vtest.mp4 の 14 コマ（8 コマで決め、6 コマで確かめた。標識の陰で半分隠れた人も 1 人、腕だけ・頭だけが"
        "見える人は数えないとして目で数えた。vfx-tests/mot/count_eval.py・count_thresh.py）、mp-cats_and_dogs.jpg、"
        "story540.mp4 の動画（count_story.py。グラフと確認用の映像を目で確認）",
    },
}


# ── 共通の部品 ─────────────────────────────────────────────────────────────

MODEL = arg(
    "--model",
    choices=["accurate", "fast"],
    default="accurate",
    help="人の検出器（accurate: Faster R-CNN v2。正確だが CPU で 1 コマ約 2〜3 秒 / "
    "fast: EfficientDet-Lite2。1 コマ約 0.1 秒だが、小さい人や重なった人を見落としやすい）",
)


# 入れ子の枠を除く割合と、人数を数える時の確信度の下限（vtest の 8 コマと story540 を目で数えた人数に合わせて決め、
# vtest の別の 6 コマで確かめた。NOTES[95] を参照）
NESTED = 0.65
COUNT_SCORE = {"accurate": 0.6, "fast": 0.3}


def _paths(
    a: argparse.Namespace, src: str | Path, tag: str, ext: str
) -> tuple[Path, Path]:
    """出力先と JSON の場所（-o / --json を省略した時の既定も決める）。"""
    out = Path(a.out) if a.out else core.default_out(src, tag, ext)
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    return out, json_path


def _sibling(path: Path, suffix: str) -> Path:
    """出力の隣に置く付属のファイル（例: a.mp4 と '-graph.png' → a-graph.png）。"""
    return path.with_name(f"{path.stem}{suffix}")


def _span(info: core.MediaInfo, a: argparse.Namespace, fps: float) -> float:
    """処理する長さ（秒）。--duration と --max-frames も考える。"""
    span = a.duration or max(info.duration - a.start, 0.0)
    if getattr(a, "max_frames", None):
        span = min(span, a.max_frames / fps)
    return span


@cache
def _font(size: int) -> Any:
    from PIL import ImageFont

    for f in core.JP_FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


Label = tuple[
    str, tuple[int, int], int, tuple[int, int, int], tuple[int, int, int] | None
]


def _paint(img: np.ndarray, items: list[Label]) -> np.ndarray:
    """文字（日本語も可）をまとめて書く。1 コマに多くの札を書く時に、画像の変換を 1 回で済ませる。

    items: (文字, 左上の位置, 文字の大きさ, 文字の色 BGR, 地の色 BGR か None)
    """
    if not items:
        return img
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    h, w = img.shape[:2]
    for text, (x, y), size, color, bg in items:
        font = _font(size)
        box = d.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        x = int(min(max(x, 3), max(w - tw - 4, 3)))
        y = int(min(max(y, 3), max(h - th - 4, 3)))
        box = d.textbbox((x, y), text, font=font)
        if bg is not None:
            d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=bg[::-1])
        d.text((x, y), text, font=font, fill=color[::-1])
    img[:, :, :3] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def _label_size(h: int) -> int:
    return max(16, round(h / 30))


def _classes(text: str) -> list[str]:
    """'person,car' や '人,車' を COCO の英語名の一覧にする。"""
    from .detectors import COCO_JA

    ja_to_en = {v: k for k, v in COCO_JA.items()}
    names: list[str] = []
    for raw in text.replace("、", ",").split(","):
        name = raw.strip()
        if not name:
            continue
        if name in COCO_JA:
            names.append(name)
        elif name in ja_to_en:
            names.append(ja_to_en[name])
        else:
            raise core.VfxError(
                f"知らない種類です: {name}（COCO の 80 種類の英語名か日本語名。例: person / 人、car / 車）"
            )
    core.require(bool(names), "--classes に種類を書いてください（例: person）")
    return names


def _xywh(box: np.ndarray | tuple[float, ...], sx: float, sy: float) -> list[float]:
    """x1,y1,x2,y2（処理した大きさの px）→ 元の大きさの x,y,幅,高さ。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    return [
        round(x1 * sx, 1),
        round(y1 * sy, 1),
        round((x2 - x1) * sx, 1),
        round((y2 - y1) * sy, 1),
    ]


def _fit(img: np.ndarray, w: int, h: int, bg: tuple[int, int, int] = (40, 40, 40)):
    """縦横比を保って w×h の枠に収める（余りは bg の色）。"""
    cell = np.full((h, w, 3), bg, np.uint8)
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return cell
    f = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * f)), max(1, round(ih * f))
    small = cv2.resize(img[:, :, :3], (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    cell[y0 : y0 + nh, x0 : x0 + nw] = small
    return cell


def _crop(frame: np.ndarray, box: np.ndarray, margin: float = 0.08) -> np.ndarray:
    """枠（x1,y1,x2,y2）の周りを少し広げて切り出す。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    xa, ya = max(0, int(x1 - mx)), max(0, int(y1 - my))
    xb, yb = min(w, math.ceil(x2 + mx)), min(h, math.ceil(y2 + my))
    if xb <= xa or yb <= ya:
        return np.zeros((2, 2, 3), np.uint8)
    return frame[ya:yb, xa:xb, :3].copy()


def _drop_nested(dets: list[Any], ratio: float = 0.65) -> list[Any]:
    """大きい枠の中にほぼ収まる小さい枠（同じ人の頭・上半身だけの枠など）を除く。

    大きい順に見て、面積の ratio 以上が先に残した枠と重なる枠は数えない（ratio が 1 を超えると何も除かない）。
    """
    kept: list[Any] = []
    for d in sorted(dets, key=lambda d: -(d.box[2] - d.box[0]) * (d.box[3] - d.box[1])):
        x1, y1, x2, y2 = d.box
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        inside = False
        for k in kept:
            ix = max(0.0, min(x2, k.box[2]) - max(x1, k.box[0]))
            iy = max(0.0, min(y2, k.box[3]) - max(y1, k.box[1]))
            if ix * iy >= ratio * area:
                inside = True
                break
        if not inside:
            kept.append(d)
    return kept


def _clean_dets(dets: list[Any], high: float, ratio: float) -> list[Any]:
    """確信度 high 以上の検出から、入れ子の枠（同じ人の頭・上半身だけの枠など）を除く。低い検出はそのまま残す
    （隠れかけた人を 2 段目の照合でつなぐのに使うため）。"""
    sure = [d for d in dets if d.score >= high]
    return _drop_nested(sure, ratio) + [d for d in dets if d.score < high]


# ── カットの変わり目 ───────────────────────────────────────────────────────


class _Cuts:
    """カットの変わり目を見つける（縮小したコマの HSV の差の平均。PySceneDetect の ContentDetector と同じ考え方）。

    差（0〜255）が threshold 以上で、前の変わり目から min_gap コマ以上たっていれば変わり目とする。
    """

    def __init__(self, threshold: float = 27.0, min_gap: int = 6) -> None:
        self.threshold = threshold
        self.min_gap = min_gap
        self.prev: np.ndarray | None = None
        self.since = 0
        self.last_score = 0.0

    def __call__(self, frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        size = (64, max(1, round(64 * h / w)))
        small = cv2.resize(frame[:, :, :3], size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16)
        cut = False
        if self.prev is not None:
            self.last_score = float(np.abs(hsv - self.prev).mean())
            cut = self.last_score >= self.threshold and self.since >= self.min_gap
        self.prev = hsv
        self.since = 0 if cut else self.since + 1
        return cut


# ── 追跡（ByteTrack と同じ考え方: カルマンフィルター＋ハンガリアン法の IoU 照合を 2 段で） ──


def _to_xyah(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], float)


def _to_box(mean: np.ndarray) -> np.ndarray:
    cx, cy, ar, h = mean[:4]
    w = ar * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], float)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """枠（x1,y1,x2,y2）の組どうしの重なりの割合（N×M）。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _assign(
    cost: np.ndarray, limit: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """ハンガリアン法で、費用が limit 以下の組だけを結ぶ。(組, 余った行, 余った列)"""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    from scipy.optimize import linear_sum_assignment

    c = np.where(cost > limit, limit + 1e4, cost)
    rows, cols = linear_sum_assignment(c)
    pairs = [(int(r), int(k)) for r, k in zip(rows, cols) if cost[r, k] <= limit]
    used_r, used_c = {r for r, _ in pairs}, {k for _, k in pairs}
    return (
        pairs,
        [r for r in range(n) if r not in used_r],
        [k for k in range(m) if k not in used_c],
    )


class _Kalman:
    """等速のカルマンフィルター。状態は 枠の中心 x, y・縦横比・高さ と、それぞれの速さ（8 次元）。

    ゆらぎの大きさは枠の高さに比例させる（位置 1/20、速さ 1/160。ByteTrack・SORT 系と同じ設定）。
    """

    POS, VEL = 1 / 20, 1 / 160

    def __init__(self) -> None:
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = z[3]
        std = [2 * self.POS * h, 2 * self.POS * h, 1e-2, 2 * self.POS * h]
        std += [10 * self.VEL * h, 10 * self.VEL * h, 1e-5, 10 * self.VEL * h]
        return np.r_[z, np.zeros(4)], np.diag(np.square(std))

    def predict(
        self, mean: np.ndarray, cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self.POS * h, self.POS * h, 1e-2, self.POS * h]
        std += [self.VEL * h, self.VEL * h, 1e-5, self.VEL * h]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(std))
        return mean, cov

    def update(
        self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        r = np.diag(np.square([self.POS * h, self.POS * h, 1e-1, self.POS * h]))
        s = self.H @ cov @ self.H.T + r
        gain = np.linalg.solve(s, self.H @ cov).T  # cov H^T S^-1（S と cov は対称）
        mean = mean + gain @ (z - self.H @ mean)
        cov = cov - gain @ s @ gain.T
        return mean, cov


class _Track:
    def __init__(
        self, box: np.ndarray, score: float, label: str, frame: int, kf: _Kalman
    ) -> None:
        self.mean, self.cov = kf.initiate(_to_xyah(box))
        self.score = score
        self.label = label
        self.id = 0  # 確定した時に番号を付ける
        self.state = "new"  # new（未確定）/ tracked / lost / removed
        self.start = frame
        self.last = frame  # 最後に検出と結べたコマ
        self.det = -1  # このコマで結んだ検出の番号（結べなければ -1）

    @property
    def box(self) -> np.ndarray:
        return _to_box(self.mean)

    def predict(self, kf: _Kalman) -> None:
        if self.state != "tracked":
            self.mean[7] = 0.0  # 見失い中は高さの変化を止める
        self.mean, self.cov = kf.predict(self.mean, self.cov)

    def update(
        self,
        kf: _Kalman,
        box: np.ndarray,
        score: float,
        label: str,
        frame: int,
        det: int,
    ) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, _to_xyah(box))
        self.score, self.label, self.last, self.det = score, label, frame, det
        if self.state in ("tracked", "lost"):
            self.state = "tracked"


class _ByteTracker:
    """ByteTrack 風の追跡。

    1 段目: 確信度の高い検出（high 以上）を、追跡中と見失い中の人に IoU で結ぶ（確信度を掛けた重なり > 0.2）。
    2 段目: 残った追跡中の人を、確信度の低い検出（low 以上 high 未満。隠れかけた人など）に結ぶ（IoU > 0.5）。
    新しい人は high+0.1 以上の検出から始め、次のコマでも結べたら番号を確定する（最初のコマは即確定）。
    keep コマのあいだ結べなければ消す。番号はこの追跡器の中で通し番号（reset しても続く）。
    """

    def __init__(
        self,
        high: float = 0.5,
        low: float = 0.1,
        keep: int = 30,
        ids: list[int] | None = None,
    ) -> None:
        self.high, self.low, self.keep = high, low, keep
        self.new_thresh = high + 0.1
        self.kf = _Kalman()
        self.tracks: list[_Track] = []
        # 次に付ける番号（種類ごとに追跡器を分ける時は、同じ入れ物を渡して番号を通しにする）
        self.ids = ids if ids is not None else [1]
        self.fresh = True  # 始めたばかり（最初のコマの検出はすぐ確定する）

    def reset(self) -> None:
        """カットが変わった時など、追跡中の人をすべて終わらせる（番号は続きから）。"""
        self.tracks = []
        self.fresh = True

    def _confirm(self, t: _Track) -> None:
        t.id = self.ids[0]
        self.ids[0] += 1
        t.state = "tracked"

    def update(self, dets: list[Any], frame: int) -> list[_Track]:
        """このコマの検出（box, score, label を持つ物）を受け取り、番号の付いた人の一覧を返す。"""
        boxes = np.array([d.box for d in dets], float).reshape(-1, 4)
        scores = np.array([d.score for d in dets], float)
        labels = [d.label for d in dets]
        hi = [i for i in range(len(dets)) if scores[i] >= self.high]
        lo = [i for i in range(len(dets)) if self.low <= scores[i] < self.high]
        for t in self.tracks:
            t.det = -1
            t.predict(self.kf)
        unconfirmed = [t for t in self.tracks if t.state == "new"]
        pool = [t for t in self.tracks if t.state in ("tracked", "lost")]

        def boxes_of(ts: list[_Track]) -> np.ndarray:
            return np.array([t.box for t in ts], float).reshape(-1, 4)

        # 1 段目
        sim = _iou_matrix(boxes_of(pool), boxes[hi]) * scores[hi][None, :]
        pairs, left_t, left_d = _assign(1 - sim, 0.8)
        for ti, di in pairs:
            d = hi[di]
            pool[ti].update(self.kf, boxes[d], scores[d], labels[d], frame, d)
        # 2 段目
        rest = [pool[i] for i in left_t if pool[i].state == "tracked"]
        iou = _iou_matrix(boxes_of(rest), boxes[lo])
        pairs, left_r, _ = _assign(1 - iou, 0.5)
        for ti, di in pairs:
            d = lo[di]
            rest[ti].update(self.kf, boxes[d], scores[d], labels[d], frame, d)
        for i in left_r:
            rest[i].state = "lost"
        # 未確定の人
        hi_left = [hi[i] for i in left_d]
        sim = (
            _iou_matrix(boxes_of(unconfirmed), boxes[hi_left])
            * (scores[hi_left][None, :])
        )
        pairs, left_u, left_d = _assign(1 - sim, 0.7)
        for ui, di in pairs:
            d = hi_left[di]
            unconfirmed[ui].update(self.kf, boxes[d], scores[d], labels[d], frame, d)
            self._confirm(unconfirmed[ui])
        for i in left_u:
            unconfirmed[i].state = "removed"
        # 新しい人
        for di in left_d:
            d = hi_left[di]
            if scores[d] < self.new_thresh:
                continue
            t = _Track(boxes[d], scores[d], labels[d], frame, self.kf)
            t.det = d
            if self.fresh:
                self._confirm(t)
            self.tracks.append(t)
        # 長く見失った人を消す
        for t in self.tracks:
            if t.state == "lost" and frame - t.last > self.keep:
                t.state = "removed"
        self.tracks = [t for t in self.tracks if t.state != "removed"]
        self._drop_duplicates()
        self.fresh = False
        return [t for t in self.tracks if t.state == "tracked" and t.last == frame]

    def _drop_duplicates(self) -> None:
        """追跡中と見失い中で同じ所を指す 2 本（IoU > 0.85）は、短い方を消す。"""
        tracked = [t for t in self.tracks if t.state == "tracked"]
        lost = [t for t in self.tracks if t.state == "lost"]
        if not tracked or not lost:
            return
        iou = _iou_matrix(
            np.array([t.box for t in tracked]), np.array([t.box for t in lost])
        )
        drop: set[int] = set()
        for p, q in zip(*np.nonzero(iou > 0.85)):
            a, b = tracked[p], lost[q]
            drop.add(id(b) if a.last - a.start > b.last - b.start else id(a))
        self.tracks = [t for t in self.tracks if id(t) not in drop]


# ── 12 背景差分 ─────────────────────────────────────────────────────────────


def _plate_diff(
    frame: np.ndarray, plate: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """背景の画像との差で前景を求める。影（明るさだけ下がり、色味が同じ所）は分けて返す。(前景, 影)"""
    f = cv2.GaussianBlur(frame[:, :, :3], (3, 3), 0).astype(np.float32)
    diff = np.sqrt(((f - plate) ** 2).sum(axis=2))
    fg = diff > threshold
    yf, yb = f.sum(axis=2) + 1.0, plate.sum(axis=2) + 1.0
    ratio = yf / yb
    chroma = np.abs(f / yf[..., None] - plate / yb[..., None]).max(axis=2)
    shadow = fg & (yb > 60) & (ratio >= 0.35) & (ratio <= 0.92) & (chroma < 0.04)
    return fg & ~shadow, shadow


def _clean_mask(
    fg: np.ndarray, k_open: int, k_close: int, min_px: float
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """小さな点を消し（開く）、穴を埋め（閉じる）、小さな塊を消す。(地図 0/255, 塊の枠 x,y,w,h)"""
    m = fg.astype(np.uint8) * 255
    if k_open > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    if k_close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    blobs = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_px:
            keep[k] = True
            blobs.append(tuple(int(v) for v in stats[k, :4]))
    return np.where(keep[labels], 255, 0).astype(np.uint8), blobs


def _bg_overlay(frame: np.ndarray, m: np.ndarray, shadow: np.ndarray) -> np.ndarray:
    """元のコマに、前景を赤・影（背景とした所）を青で重ねる。"""
    vis = frame[:, :, :3].copy()
    fg = m > 0
    red = np.array([40, 40, 255], np.float32)
    vis[fg] = (vis[fg] * 0.45 + red * 0.55).astype(np.uint8)
    sh = shadow & ~fg
    if sh.any():
        blue = np.array([255, 120, 0], np.float32)
        vis[sh] = (vis[sh] * 0.6 + blue * 0.4).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _cutout(frame: np.ndarray, m: np.ndarray, feather: float) -> np.ndarray:
    """前景だけを残した BGRA（縁を feather px ぼかす）。"""
    alpha = m
    if feather > 0:
        alpha = cv2.GaussianBlur(m, (0, 0), feather)
    return np.dstack([frame[:, :, :3], alpha])


@command(
    12,
    "bg-subtract",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=["knn", "mog2", "median"],
        default="knn",
        help="背景の覚え方（knn: 画素ごとに過去の色を覚え、近い色が少なければ前景。既定 / "
        "mog2: 画素ごとの混合ガウス / median: 範囲全体から取ったコマの中央値を背景にする。"
        "どれも固定カメラ向き）",
    ),
    arg(
        "--bg",
        help="背景だけが写った画像（クリーンプレート）。指定するとこの画像との差で前景を求める（画像 1 枚にも使える）",
    ),
    arg(
        "--view",
        choices=["mask", "overlay"],
        default="mask",
        help="出力の見せ方（mask: 白が前景の白黒の地図 / overlay: 元の映像に前景を赤・影を青で重ねる）",
    ),
    arg(
        "--cutout",
        action="store_true",
        help="前景だけを残した透過つきの WebM（画像なら PNG）も書き出す",
    ),
    arg(
        "--shadows",
        choices=["remove", "keep"],
        default="remove",
        help="影の扱い（remove: 影は背景とする。既定 / keep: 影も前景に含める）",
    ),
    arg(
        "--threshold",
        type=float,
        help="前景とみなす差の大きさ（mog2: 既定 16〔マハラノビス距離の 2 乗〕/ knn: 既定 400〔距離の 2 乗〕/ "
        "median・--bg: 既定 30〔色の差 0〜441〕）",
    ),
    arg(
        "--hold",
        type=float,
        default=5.0,
        help="止まった物を前景として保つ目安（秒。mog2・knn。これより長く止まると背景に溶ける。既定 5）",
    ),
    arg(
        "--warmup",
        type=int,
        default=30,
        help="範囲全体から等間隔に取るコマ数。その中央値（動く物のいない背景の推定）を、mog2・knn では始める前に"
        "覚えさせ、median では背景そのものに使う（mog2・knn で 0 にすると、最初のコマから覚え始める。既定 30）",
    ),
    arg(
        "--open", type=int, help="小さな点を消す強さ（px。既定は大きさに合わせて自動）"
    ),
    arg("--close", type=int, help="小さな穴を埋める強さ（px。既定は自動）"),
    arg(
        "--min-area",
        type=float,
        default=0.0002,
        help="これより小さい塊は消す（画面の面積に対する割合。既定 0.0002）",
    ),
    arg("--feather", type=float, default=1.0, help="透過の縁をぼかす幅（px。既定 1）"),
)
def bg_subtract(a: argparse.Namespace) -> None:
    """背景を覚えて、動いている所（前景）だけを取り出す。白黒の地図か重ねた映像の MP4、前景の割合の JSON を書き出す。"""
    info = core.probe(a.src)
    plate = core.load_image(a.bg) if a.bg else None
    if info.is_image:
        core.require(
            plate is not None,
            "画像 1 枚では背景を学べません。--bg で背景だけが写った画像を指定してください",
        )
    w, h = core.scaled_size(info.width, info.height, None, a.max_side)
    sx, sy = info.width / w, info.height / h
    s = min(w, h) / 540
    k_open = a.open if a.open is not None else max(1, round(3 * s))
    k_close = a.close if a.close is not None else max(1, round(9 * s))
    min_px = a.min_area * w * h
    method = "plate" if plate is not None else a.method
    ext = (".png" if a.view == "mask" else ".jpg") if info.is_image else ".mp4"
    out, json_path = _paths(a, a.src, "bg-subtract", ext)
    cut_path = _sibling(out, "-cutout" + (".png" if info.is_image else ".webm"))
    bg_path = _sibling(out, "-bg.jpg")

    def finish(fg: np.ndarray, shadow: np.ndarray, frame: np.ndarray) -> tuple:
        if a.shadows == "keep":
            fg, shadow = fg | shadow, np.zeros_like(shadow)
        m, blobs = _clean_mask(fg, k_open, k_close, min_px)
        view = (
            cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            if a.view == "mask"
            else _bg_overlay(frame, m, shadow)
        )
        rec = {
            "ratio": round(float((m > 0).mean()), 5),
            "shadow": round(float((shadow & (m == 0)).mean()), 5),
            "blobs": [_xywh((x, y, x + bw, y + bh), sx, sy) for x, y, bw, bh in blobs],
        }
        return m, view, rec

    # ── 背景の準備 ──
    # 止まった物が背景に溶けるまでのコマ数は、覚えるコマ数（history）の約 0.18 倍（knn）・約 0.1 倍（mog2）
    # だった（止まる四角の合成映像で実測）。--hold の秒数から history を決める
    fps_in = info.fps if not info.is_image else 30.0
    history = max(10, round(a.hold * fps_in / (0.18 if method == "knn" else 0.1)))
    sub = None
    bg_saved = False
    plate_f = None
    threshold = a.threshold
    if plate is not None:
        plate_f = cv2.resize(plate[:, :, :3], (w, h), interpolation=cv2.INTER_AREA)
        threshold = threshold or 30.0
    elif method == "median" or a.warmup > 0:
        span = _span(info, a, info.fps)
        n = max(3, a.warmup or 25)
        times = [a.start + span * (k + 0.5) / n for k in range(n)]
        samples = [
            core.frame_at(a.src, t, max_side=a.max_side)[:, :, :3] for t in times
        ]
        # 画素ごとの中央値 = 動く物が半分より短い時間しか通らない所では、背景そのもの
        median = np.median(np.stack(samples), axis=0).astype(np.uint8)
        # 人などを消した背景（クリーンプレート）として書き出す
        core.save_image(bg_path, median)
        bg_saved = True
        if method == "median":
            plate_f = median
            threshold = threshold or 30.0
    if method in ("mog2", "knn"):
        if method == "mog2":
            sub = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=threshold or 16.0, detectShadows=True
            )
        else:
            sub = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=threshold or 400.0, detectShadows=True
            )
        if a.warmup > 0:
            # 取ったコマをそのまま覚えさせると、通った人まで背景になる（合成映像の試験で IoU 0.62）。
            # 中央値だけを何度も覚えさせると、ゆらぎ（分散）が小さくなりすぎて、実写のノイズを前景と誤る。
            # そこで、取ったコマのうち中央値から離れた所（動く物）だけを中央値で置き換えて覚えさせる
            med = median.astype(np.int16)
            for img in samples:
                far = np.abs(img.astype(np.int16) - med).max(axis=2) > 25
                sub.apply(np.where(far[..., None], median, img), learningRate=-1)
    plate_blur = (
        cv2.GaussianBlur(plate_f, (3, 3), 0).astype(np.float32)
        if plate_f is not None
        else None
    )
    lr = 1.0 / history

    def segment(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if sub is not None:
            raw = sub.apply(frame[:, :, :3], learningRate=lr)
            return raw == 255, raw == 127
        assert plate_blur is not None
        return _plate_diff(frame, plate_blur, threshold)

    # ── 画像 1 枚 ──
    if info.is_image:
        frame = next(core.read_frames(a.src, max_side=a.max_side))
        m, view, rec = finish(*segment(frame), frame)
        core.save_image(out, view)
        if a.cutout:
            core.save_image(cut_path, _cutout(frame, m, a.feather))
            core.log(f"保存: {cut_path}")
        core.write_json(
            json_path,
            {
                "source": str(a.src),
                "background": str(a.bg),
                "size": [info.width, info.height],
                "params": {"threshold": threshold, "shadows": a.shadows},
                **rec,
            },
        )
        core.log(f"前景の割合: {rec['ratio'] * 100:.1f}%  塊: {len(rec['blobs'])} 個")
        core.log(f"保存: {out}")
        core.log(f"保存: {json_path}")
        return

    # ── 動画 ──
    span = _span(info, a, info.fps)
    total = max(1, round(span * info.fps))
    writer = core.VideoWriter(
        out,
        info.fps,
        audio_from=a.src if a.view == "overlay" else None,
        audio_start=a.start,
        audio_duration=a.duration,
    )
    cut_writer = core.VideoWriter(cut_path, info.fps, alpha=True) if a.cutout else None
    records = []
    frames = core.read_frames(
        a.src,
        start=a.start,
        duration=a.duration,
        max_frames=a.max_frames,
        max_side=a.max_side,
    )
    with writer:
        for i, frame in enumerate(frames):
            m, view, rec = finish(*segment(frame), frame)
            writer.write(view)
            if cut_writer is not None:
                cut_writer.write(_cutout(frame, m, a.feather))
            records.append({"t": core.t_of(i, info.fps, a.start), **rec})
            if (i + 1) % max(1, total // 10) == 0:
                core.log(f"  {i + 1}/{total} コマ")
    if cut_writer is not None:
        cut_writer.close()
    if sub is not None and not bg_saved:
        # 中央値を取らなかった時だけ、覚えた背景を書き出す（knn の背景は、人が通った所が黒く抜けることがある）
        bgimg = sub.getBackgroundImage()
        if bgimg is not None:
            core.save_image(bg_path, bgimg)
            bg_saved = True
    ratios = np.array([r["ratio"] for r in records])
    peak = int(ratios.argmax())
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": info.fps,
            "size": [info.width, info.height],
            "params": {
                "method": method,
                "threshold": threshold,
                "hold": a.hold,
                "history": history,
                "warmup": a.warmup,
                "shadows": a.shadows,
                "open": k_open,
                "close": k_close,
                "min_area": a.min_area,
            },
            "summary": {
                "frames": len(records),
                "mean_ratio": round(float(ratios.mean()), 5),
                "max_ratio": round(float(ratios.max()), 5),
                "max_at": records[peak]["t"],
                "moving_frames": int((ratios > 0.001).sum()),
            },
            "frames": records,
        },
    )
    core.log(
        f"{len(records)} コマ: 前景の割合 平均 {ratios.mean() * 100:.1f}%・最大 {ratios.max() * 100:.1f}%"
        f"（{records[peak]['t']:.2f} 秒）"
    )
    if cut_writer is not None:
        core.log(f"保存: {cut_path}")
    if bg_saved:
        core.log(f"保存: {bg_path}（背景の推定。動く物を消した画）")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 15 複数物体追跡 ─────────────────────────────────────────────────────────


@command(
    15,
    "multi-track",
    arg("src", help="入力（動画）"),
    OUT,
    JSON,
    RANGE,
    SIZE,
    MODEL,
    arg(
        "--classes",
        default="person",
        help="追う種類（COCO の 80 種類の英語名か日本語名。カンマ区切り。既定 person＝人）",
    ),
    arg(
        "--fps",
        type=float,
        help="1 秒あたり何コマ処理するか（省略時は全部のコマ。重い時は 10 などに下げる）",
    ),
    arg(
        "--high",
        type=float,
        default=0.5,
        help="確信度の高い検出の下限（0〜1。1 段目の照合と、新しい番号を作るのに使う。既定 0.5）",
    ),
    arg(
        "--low",
        type=float,
        default=0.1,
        help="確信度の低い検出の下限（0〜1。2 段目の照合で、隠れかけた人をつなぐ。既定 0.1）",
    ),
    arg(
        "--keep",
        type=float,
        default=1.0,
        help="見失ってから同じ番号で待つ秒数（既定 1.0）",
    ),
    arg("--trail", type=float, default=2.0, help="軌跡を描く長さ（秒。0 で描かない）"),
    arg(
        "--nested",
        type=float,
        default=NESTED,
        help="確信度の高い検出のうち、大きい枠の中に面積のこの割合以上が収まる小さい枠（同じ人の頭・上半身だけの枠"
        "など）を使わない（既定 0.65。1 より大きくすると使う。重なって隠れた人を拾いたい時に）",
    ),
    arg(
        "--no-cuts",
        action="store_true",
        help="カットの変わり目で追跡をやり直さない（既定は、変わり目で全員の追跡を終える）",
    ),
)
def multi_track(a: argparse.Namespace) -> None:
    """人などを検出して番号を付けて追い、番号つきの枠と軌跡の MP4・番号ごとの JSON・確認画像を書き出す。"""
    from . import detectors

    info = core.probe(a.src)
    core.require(
        not info.is_image,
        "動画を指定してください（画像 1 枚では追えません。人数を数えるなら count-people）",
    )
    classes = _classes(a.classes)
    fps = a.fps if a.fps and a.fps < info.fps else info.fps
    w, h = core.scaled_size(info.width, info.height, None, a.max_side)
    sx, sy = info.width / w, info.height / h
    n_est = max(1, int(_span(info, a, fps) * fps))
    out, json_path = _paths(a, a.src, "multi-track", ".mp4")
    sheet_path = out.with_suffix(".jpg")
    det = detectors.ObjectDetector(a.model, score=min(a.low, a.high), classes=classes)
    # 種類ごとに追跡器を分ける（人の番号が車の検出に移らないように）。番号は種類をまたいで通し
    ids = [1]
    trackers = {
        c: _ByteTracker(a.high, a.low, keep=max(1, round(a.keep * fps)), ids=ids)
        for c in classes
    }
    cuts = None if a.no_cuts else _Cuts(min_gap=max(2, round(0.25 * fps)))
    sheet_at = set(np.linspace(0, n_est - 1, min(12, n_est)).round().astype(int))
    trail_frames = round(a.trail * fps)
    size = _label_size(h)
    multi = len(classes) > 1
    writer = core.VideoWriter(
        out,
        fps,
        audio_from=a.src if fps == info.fps else None,
        audio_start=a.start,
        audio_duration=a.duration,
    )
    frames = core.read_frames(
        a.src,
        start=a.start,
        duration=a.duration,
        max_frames=a.max_frames,
        max_side=a.max_side,
        fps=fps if fps != info.fps else None,
    )
    records: dict[int, dict[str, Any]] = {}
    trails: dict[int, list[tuple[int, int, int]]] = {}
    cut_times: list[float] = []
    sheet, sheet_labels = [], []
    t_det, t_begin, n = 0.0, time.time(), 0
    with writer:
        for i, frame in enumerate(frames):
            t = core.t_of(i, fps, a.start, snap=fps == info.fps)
            if cuts is not None and cuts(frame) and i > 0:
                for tracker in trackers.values():
                    tracker.reset()
                cut_times.append(t)
            t0 = time.time()
            found = det(frame)
            t_det += time.time() - t0
            active = []
            for c, tracker in trackers.items():
                dets = _clean_dets([d for d in found if d.label == c], a.high, a.nested)
                active += tracker.update(dets, i)
            vis = frame[:, :, :3].copy()
            items: list[Label] = []
            for tr in sorted(active, key=lambda x: x.id):
                col = detectors.color_for(tr.id)
                x1, y1, x2, y2 = tr.box
                ja = detectors.COCO_JA.get(tr.label, tr.label)
                rec = records.setdefault(
                    tr.id,
                    {"id": tr.id, "label": tr.label, "label_ja": ja, "path": []},
                )
                rec["path"].append(
                    {"t": t, "box": _xywh(tr.box, sx, sy), "score": round(tr.score, 3)}
                )
                pts = trails.setdefault(tr.id, [])
                pts.append((i, round((x1 + x2) / 2), round(y2)))
                if trail_frames > 0:
                    line = [(x, y) for k, x, y in pts if i - k <= trail_frames]
                    if len(line) > 1:
                        cv2.polylines(
                            vis, [np.array(line, np.int32)], False, col, 2, cv2.LINE_AA
                        )
                cv2.rectangle(
                    vis,
                    (round(x1), round(y1)),
                    (round(x2), round(y2)),
                    col,
                    2,
                    cv2.LINE_AA,
                )
                text = f"{ja} {tr.id}" if multi else str(tr.id)
                items.append(
                    (text, (round(x1), round(y1) - size - 6), size, (0, 0, 0), col)
                )
            items.append(
                (
                    f"{t:.1f}秒  追跡中 {len(active)}",
                    (8, 8),
                    size,
                    (255, 255, 255),
                    (0, 0, 0),
                )
            )
            _paint(vis, items)
            writer.write(vis)
            if i in sheet_at:
                sheet.append(vis)
                sheet_labels.append("")
            n = i + 1
            if n % max(1, n_est // 10) == 0:
                core.log(f"  {n}/{n_est} コマ（番号 {ids[0] - 1} まで）")
    elapsed = time.time() - t_begin
    tracks = []
    for r in sorted(records.values(), key=lambda r: r["id"]):
        path = r.pop("path")
        tracks.append(
            {
                **r,
                "start": path[0]["t"],
                "end": path[-1]["t"],
                "frames": len(path),
                "path": path,
            }
        )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": fps,
            "size": [info.width, info.height],
            "params": {
                "model": a.model,
                "classes": classes,
                "high": a.high,
                "low": a.low,
                "keep": a.keep,
                "cuts": not a.no_cuts,
                "nested": a.nested,
                "max_side": a.max_side,
            },
            "cuts": cut_times,
            "summary": {
                "tracks": len(tracks),
                "frames": n,
                "sec_per_frame": round(elapsed / max(n, 1), 3),
                "detect_sec_per_frame": round(t_det / max(n, 1), 3),
            },
            "tracks": tracks,
        },
    )
    if sheet:
        core.save_image(
            sheet_path, core.contact_sheet(sheet, sheet_labels, cols=4, cell_width=400)
        )
    core.log(
        f"{n} コマ: 番号 {len(tracks)} 個・カットの変わり目 {len(cut_times)} か所・"
        f"1 コマ {elapsed / max(n, 1):.2f} 秒（検出 {t_det / max(n, 1):.2f} 秒）"
    )
    for r in sorted(tracks, key=lambda r: -r["frames"])[:8]:
        core.log(
            f"  {r['label_ja']} {r['id']}: {r['start']:.1f}〜{r['end']:.1f} 秒（{r['frames']} コマ）"
        )
    if sheet:
        core.log(f"保存: {sheet_path}")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 85 人物の再識別 ─────────────────────────────────────────────────────────

# 同じ人とみなす目安（試験で決めた値。NOTES[85] を参照）
FACE_SAME = 0.363  # SFace のコサイン類似度（OpenCV の推奨値）
BODY_SAME = 0.60  # YouTu ReID のコサイン類似度
FACE_WEIGHT = 0.75  # 顔と体の両方がある時の、顔の重み


class _BodyEmbed:
    """人の体の見た目の特徴量（YouTu ReID。OpenCV Zoo、Apache-2.0。128×256 の切り抜き → 768 次元）。"""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self) -> None:
        from .models import fetch

        self.net = cv2.dnn.readNet(str(fetch("youtu_reid")))

    def __call__(self, crops: list[np.ndarray]) -> list[np.ndarray | None]:
        """切り抜きごとの特徴量（長さ 1）。求まらなかった切り抜きは None。

        1 枚ずつ通す。OpenCV 4.14 の dnn では、このモデルに入れる枚数を途中で変える（1 枚 → 2 枚など）と、
        その後の出力がすべて壊れる（値が発散する）ことを実測で確かめた。同じ枚数で通し続ければ、毎回作り直した
        網と完全に一致する。
        """
        out: list[np.ndarray | None] = []
        for c in crops:
            small = cv2.resize(c[:, :, :3], (128, 256), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = ((rgb - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
            self.net.setInput(np.ascontiguousarray(x))
            f = self.net.forward().reshape(-1).astype(np.float64)
            norm = float(np.linalg.norm(f))
            out.append(f / norm if np.isfinite(norm) and norm > 0 else None)
        return out


class _FaceEmbed:
    """顔を見つけて（YuNet、MIT）、顔の特徴量（SFace、Apache-2.0。128 次元）を求める。"""

    def __init__(self, score: float = 0.8, min_size: int = 36) -> None:
        from . import detectors

        self.det = detectors.FaceDetector(score=score)
        self.rec = detectors.face_recognizer()
        self.min_size = min_size

    def __call__(self, frame: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
        img = np.ascontiguousarray(frame[:, :, :3])
        out = []
        for f in self.det(img):
            x, y, w, h = f.box
            if min(w, h) < self.min_size:
                continue
            feat = self.rec.feature(self.rec.alignCrop(img, f.raw)).reshape(-1)
            feat = feat / (np.linalg.norm(feat) + 1e-9)
            out.append((np.array([x, y, x + w, y + h], float), feat, f.score))
        return out


@dataclass
class _PseudoDet:
    box: tuple[float, float, float, float]
    score: float
    label: str = "person"


def _match_faces(
    person_boxes: list[np.ndarray], faces: list[tuple[np.ndarray, np.ndarray, float]]
) -> tuple[dict[int, int], list[int]]:
    """顔を人の枠に割り当てる（顔の中心が枠の上 65% にあり、顔の 6 割以上が枠の中）。(人→顔, 余った顔)"""
    owner: dict[int, int] = {}
    loose = []
    for fi, (fb, _, _) in enumerate(faces):
        cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
        best, best_area = -1, math.inf
        for pi, pb in enumerate(person_boxes):
            if pi in owner:
                continue
            inside = pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[1] + 0.65 * (
                pb[3] - pb[1]
            )
            ix = max(0.0, min(fb[2], pb[2]) - max(fb[0], pb[0]))
            iy = max(0.0, min(fb[3], pb[3]) - max(fb[1], pb[1]))
            cover = ix * iy / max((fb[2] - fb[0]) * (fb[3] - fb[1]), 1e-6)
            area = (pb[2] - pb[0]) * (pb[3] - pb[1])
            if inside and cover >= 0.6 and area < best_area:
                best, best_area = pi, area
        if best >= 0:
            owner[best] = fi
        else:
            loose.append(fi)
    return owner, loose


@dataclass
class _Tracklet:
    """1 つのカットの中で続けて追えた 1 人分（再識別の単位）。"""

    file: int
    shot: int
    tid: int
    frames: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    body: list[np.ndarray] = field(default_factory=list)
    body_overlap: list[np.ndarray] = field(
        default_factory=list
    )  # 他の人と重なったコマ（混ざりやすい）
    face: list[np.ndarray] = field(default_factory=list)
    crop: np.ndarray | None = None
    crop_rank: float = -1.0
    body_mean: np.ndarray | None = None
    face_mean: np.ndarray | None = None
    person: int = 0


def _mean_unit(vs: list[np.ndarray]) -> np.ndarray | None:
    if not vs:
        return None
    m = np.mean(np.stack(vs), axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def _reid_scan(
    path: str | Path,
    file_idx: int,
    a: argparse.Namespace,
    det: Any,
    faces: _FaceEmbed | None,
    body: _BodyEmbed | None,
) -> tuple[list[_Tracklet], dict[str, Any]]:
    """1 本の動画（か画像）から、カットごとに人を追い、追えた区間ごとの特徴量を集める。"""
    info = core.probe(path)
    w, h = core.scaled_size(info.width, info.height, None, a.max_side)
    sx, sy = info.width / w, info.height / h
    step = 1.0 / a.fps if a.fps and not info.is_image else 0.0
    # コマの番号は元の動画の番号で数える（間引いても、見失ってから待つ長さは 1.5 秒）
    tracker = _ByteTracker(0.5, 0.1, keep=max(1, round(1.5 * info.fps)))
    cuts = _Cuts(min_gap=max(2, round(0.25 * info.fps)))
    tracklets: dict[int, _Tracklet] = {}
    shot, cut_times, next_t, sampled = 0, [], -1.0, 0
    frames = core.read_frames(
        path,
        start=a.start,
        duration=a.duration,
        max_frames=a.max_frames,
        max_side=a.max_side,
    )
    for i, frame in enumerate(frames):
        t = core.t_of(i, info.fps, a.start)
        if cuts(frame) and i > 0:
            shot += 1
            cut_times.append(t)
            tracker.reset()
            next_t = -1.0
        if t + 1e-6 < next_t:
            continue
        next_t = t + step if next_t < 0 else next_t + step
        sampled += 1
        dets: list[Any] = _clean_dets(det(frame), 0.5, NESTED)
        found = faces(frame) if faces is not None else []
        # 顔は確信度 0.3 以上の人の枠にだけ割り当てる（低い枠は追跡に使われないことがあるため）
        sure = [k for k, d in enumerate(dets) if d.score >= 0.3]
        owner, loose = _match_faces([np.array(dets[k].box) for k in sure], found)
        face_of = {sure[p]: fi for p, fi in owner.items()}
        # 人として見つからなかった顔の寄りは、顔の周りを人の枠として使う
        for fi in loose:
            fb, _, score = found[fi]
            fw, fh = fb[2] - fb[0], fb[3] - fb[1]
            box = (
                max(0.0, fb[0] - 0.4 * fw),
                max(0.0, fb[1] - 0.5 * fh),
                min(float(w), fb[2] + 0.4 * fw),
                min(float(h), fb[3] + 0.9 * fh),
            )
            face_of[len(dets)] = fi
            dets.append(_PseudoDet(box, float(score)))
        active = tracker.update(dets, i) if not info.is_image else []
        if info.is_image:  # 画像は 1 枚ごとに、見つけた人をそのまま 1 区間とする
            active = []
            for k, d in enumerate(dets):
                if d.score >= 0.5:
                    tr = _Track(np.array(d.box), d.score, "person", 0, tracker.kf)
                    tr.id, tr.det, tr.state = k + 1, k, "tracked"
                    active.append(tr)
        body_crops, body_owner, body_mixed = [], [], []
        # 他の人の枠と重なる切り抜きには別の人が混ざるので、重ならないコマがあればそちらを使う
        sure_boxes = np.array([d.box for d in dets if d.score >= 0.5], float).reshape(
            -1, 4
        )
        for tr in active:
            if tr.det < 0:
                continue
            d = dets[tr.det]
            tl = tracklets.setdefault(tr.id, _Tracklet(file_idx, shot, tr.id))
            box = np.array(d.box, float)
            tl.frames.append(i)
            tl.times.append(t)
            tl.boxes.append(_xywh(box, sx, sy))
            fi = face_of.get(tr.det)
            if fi is not None:
                tl.face.append(found[fi][1])
            if not isinstance(d, _PseudoDet):
                body_crops.append(_crop(frame, box, 0.05))
                body_owner.append(tl)
                ious = _iou_matrix(box[None], sure_boxes)[0]
                body_mixed.append(bool(((ious > 0.1) & (ious < 0.99)).any()))
            rank = d.score * math.sqrt((box[2] - box[0]) * (box[3] - box[1]))
            rank *= 1.5 if fi is not None else 1.0
            if rank > tl.crop_rank:
                tl.crop_rank = rank
                c = _crop(frame, box, 0.08)
                f = min(1.0, 360 / max(c.shape[:2]))
                tl.crop = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if body is not None and body_crops:
            feats = body(body_crops)
            for tl, v, mixed in zip(body_owner, feats, body_mixed, strict=True):
                if v is not None:
                    (tl.body_overlap if mixed else tl.body).append(v)
    meta = {
        "file": str(path),
        "fps": info.fps,
        "size": [info.width, info.height],
        "image": info.is_image,
        "cuts": cut_times,
        "shots": shot + 1,
        "sampled": sampled,
    }
    return list(tracklets.values()), meta


def _apart(a: _Tracklet, b: _Tracklet) -> bool:
    """同じコマに別々の場所で映った 2 区間か（同じ人ではありえない）。

    同じコマに映っていても、枠がほぼ重なっている（IoU の平均 0.5 以上）なら、人が重なった所で同じ人に枠が
    2 つ付いただけの見込みが高いので、別人とは決めつけない。
    """
    shared = set(a.frames) & set(b.frames)
    if not shared:
        return False
    ba = {f: x for f, x in zip(a.frames, a.boxes, strict=True)}
    bb = {f: x for f, x in zip(b.frames, b.boxes, strict=True)}

    def xyxy(v: list[float]) -> list[float]:
        return [v[0], v[1], v[0] + v[2], v[1] + v[3]]

    ious = [
        _iou_matrix(np.array([xyxy(ba[f])]), np.array([xyxy(bb[f])]))[0, 0]
        for f in shared
    ]
    return float(np.mean(ious)) < 0.5


def _reid_scores(tls: list[_Tracklet]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """区間どうしの「同じ人らしさ」（0 より大きければ同じ人寄り）と、顔・体の類似度。"""
    n = len(tls)
    z = np.full((n, n), -np.inf)
    sf = np.full((n, n), np.nan)
    sb = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            zs = []
            fi, fj = tls[i].face_mean, tls[j].face_mean
            bi, bj = tls[i].body_mean, tls[j].body_mean
            zf = zb = None
            if fi is not None and fj is not None:
                sf[i, j] = sf[j, i] = float(fi @ fj)
                zf = (sf[i, j] - FACE_SAME) / 0.1
            if bi is not None and bj is not None:
                sb[i, j] = sb[j, i] = float(bi @ bj)
                zb = (sb[i, j] - BODY_SAME) / 0.1
            if zf is not None and zb is not None:
                zs = [FACE_WEIGHT * zf + (1 - FACE_WEIGHT) * zb]
            elif zf is not None:
                zs = [zf]
            elif zb is not None:
                zs = [zb]
            if zs:
                z[i, j] = z[j, i] = zs[0]
    return z, sf, sb


def _cluster(z: np.ndarray, cannot: np.ndarray, margin: float) -> list[int]:
    """平均連結の凝集型クラスタリング（同じコマに一緒に映った区間は同じ人にしない）。区間ごとの組の番号。"""
    groups = [[i] for i in range(len(z))]
    while True:
        best = None
        for p in range(len(groups)):
            for q in range(p + 1, len(groups)):
                ix = np.ix_(groups[p], groups[q])
                if cannot[ix].any():
                    continue
                vals = z[ix]
                if not np.isfinite(vals).all():
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                s = float(vals.mean())
                if s > margin and (best is None or s > best[0]):
                    best = (s, p, q)
        if best is None:
            break
        _, p, q = best
        groups[p] += groups.pop(q)
    label = [0] * len(z)
    for g, members in enumerate(groups):
        for i in members:
            label[i] = g
    return label


@command(
    85,
    "reid",
    arg(
        "src",
        nargs="+",
        help="入力（動画か画像。複数を渡すと、ファイルをまたいで同じ人をつなぐ）",
    ),
    OUT,
    JSON,
    RANGE,
    SIZE,
    MODEL,
    arg(
        "--fps",
        type=float,
        default=4.0,
        help="1 秒あたり何コマ調べるか（既定 4。人の検出が重いので間引く）",
    ),
    arg(
        "--margin",
        type=float,
        default=0.0,
        help="同じ人とみなす厳しさ（0 が標準。正の値ほど厳しく〔分かれやすく〕、負の値ほど緩く〔まとまりやすく〕なる）",
    ),
    arg(
        "--no-face",
        action="store_true",
        help="顔の特徴量を使わず、体（服装など）の見た目だけで比べる",
    ),
    arg(
        "--min-samples",
        type=int,
        default=2,
        help="調べたコマのうち、これより少なくしか映っていない区間は除く（動画のみ。既定 2）",
    ),
    arg(
        "--video",
        action="store_true",
        help="通しの番号を書いた確認用の MP4 も書き出す（動画ごと）",
    ),
)
def reid(a: argparse.Namespace) -> None:
    """カットや場面をまたいで同じ人をつなぎ、通しの番号を付ける。番号ごとの切り抜きを並べた一覧（JPG）と JSON を書き出す。"""
    from . import detectors

    srcs = [Path(s) for s in a.src]
    det = detectors.ObjectDetector(a.model, score=0.1, classes=["person"])
    faces = None if a.no_face else _FaceEmbed()
    body = _BodyEmbed()
    out, json_path = _paths(a, srcs[0], "reid", ".jpg")
    t_begin = time.time()
    tls: list[_Tracklet] = []
    metas = []
    for k, src in enumerate(srcs):
        core.log(f"調べています: {src}")
        found, meta = _reid_scan(src, k, a, det, faces, body)
        need = 1 if meta["image"] else a.min_samples
        tls += [tl for tl in found if len(tl.frames) >= need]
        metas.append(meta)
    core.require(bool(tls), "人が見つかりませんでした")
    for tl in tls:
        tl.body_mean = _mean_unit(tl.body or tl.body_overlap)
        tl.face_mean = _mean_unit(tl.face)
    z, sf, sb = _reid_scores(tls)
    n = len(tls)
    cannot = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            if tls[i].file == tls[j].file and _apart(tls[i], tls[j]):
                cannot[i, j] = cannot[j, i] = True
    labels = _cluster(z, cannot, a.margin)
    # 最初に出た順に 1, 2, 3… と番号を付け直す
    order: dict[int, int] = {}
    for i in sorted(range(n), key=lambda i: (tls[i].file, tls[i].times[0])):
        order.setdefault(labels[i], len(order) + 1)
    for i, tl in enumerate(tls):
        tl.person = order[labels[i]]
    persons: dict[int, list[_Tracklet]] = {}
    for tl in sorted(tls, key=lambda t: (t.file, t.times[0])):
        persons.setdefault(tl.person, []).append(tl)
    multi_file = len(srcs) > 1

    def where(tl: _Tracklet) -> str:
        span = (
            f"{tl.times[0]:.1f}秒"
            if len(tl.times) == 1
            else f"{tl.times[0]:.1f}〜{tl.times[-1]:.1f}秒"
        )
        name = f"{srcs[tl.file].stem[:10]} " if multi_file else ""
        return name + span + (" 顔" if tl.face else "")

    # 一覧の画像: 1 行に 1 人（左端に番号、右に出てきた場面ごとの切り抜き）
    cw, ch, per_row = 200, 300, 6
    cells, cell_labels = [], []
    for pid, items in persons.items():
        rows = [items[k : k + per_row] for k in range(0, len(items), per_row)]
        for r, row in enumerate(rows):
            card = np.full((ch, cw, 3), detectors.color_for(pid), np.uint8)
            if r == 0:
                _paint(card, [(f"人物 {pid}", (12, ch // 2 - 30), 30, (0, 0, 0), None)])
                _paint(
                    card,
                    [(f"{len(items)} 場面", (12, ch // 2 + 14), 20, (0, 0, 0), None)],
                )
            cells.append(card)
            cell_labels.append("")
            for tl in row:
                cells.append(_fit(tl.crop if tl.crop is not None else card, cw, ch))
                cell_labels.append(where(tl))
            for _ in range(per_row - len(row)):
                cells.append(np.full((ch, cw, 3), 24, np.uint8))
                cell_labels.append("")
    core.save_image(
        out, core.contact_sheet(cells, cell_labels, cols=per_row + 1, cell_width=cw)
    )
    elapsed = time.time() - t_begin
    keys = [f"{tl.file}:{tl.shot}:{tl.tid}" for tl in tls]

    def rnd(m: np.ndarray) -> list[list[float | None]]:
        return [
            [None if not np.isfinite(v) else round(float(v), 3) for v in row]
            for row in m
        ]

    core.write_json(
        json_path,
        {
            "sources": [str(s) for s in srcs],
            "params": {
                "model": a.model,
                "fps": a.fps,
                "margin": a.margin,
                "face": not a.no_face,
                "face_same": FACE_SAME,
                "body_same": BODY_SAME,
                "face_weight": FACE_WEIGHT,
            },
            "files": metas,
            "summary": {
                "persons": len(persons),
                "appearances": n,
                "seconds": round(elapsed, 1),
            },
            "persons": [
                {
                    "id": pid,
                    "appearances": [
                        {
                            "file": str(srcs[tl.file]),
                            "shot": tl.shot,
                            "start": tl.times[0],
                            "end": tl.times[-1],
                            "samples": len(tl.frames),
                            "face": bool(tl.face),
                            "box": tl.boxes[len(tl.boxes) // 2],
                            "key": f"{tl.file}:{tl.shot}:{tl.tid}",
                            "path": [
                                {"t": t, "box": b}
                                for t, b in zip(tl.times, tl.boxes, strict=True)
                            ],
                        }
                        for tl in items
                    ],
                }
                for pid, items in persons.items()
            ],
            "pairs": {"keys": keys, "face": rnd(sf), "body": rnd(sb), "score": rnd(z)},
        },
    )
    core.log(f"{n} 区間 → {len(persons)} 人（{elapsed:.0f} 秒）")
    for pid, items in persons.items():
        core.log(f"  人物 {pid}: " + " / ".join(where(tl) for tl in items))
    if a.video:
        for k, src in enumerate(srcs):
            if metas[k]["image"]:
                continue
            vpath = (
                _sibling(out, f"-{src.stem}.mp4")
                if multi_file
                else out.with_suffix(".mp4")
            )
            _reid_video(src, vpath, [tl for tl in tls if tl.file == k], a)
            core.log(f"保存: {vpath}")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


def _reid_video(
    src: Path, out: Path, tls: list[_Tracklet], a: argparse.Namespace
) -> None:
    """通しの番号を書いた確認用の動画（調べたコマの間は枠を直線で補う）。"""
    from . import detectors

    info = core.probe(src)
    w, h = core.scaled_size(info.width, info.height, None, a.max_side)
    sx, sy = info.width / w, info.height / h
    size = _label_size(h)
    spans = []
    for tl in tls:
        boxes = np.array(
            [
                [b[0] / sx, b[1] / sy, (b[0] + b[2]) / sx, (b[1] + b[3]) / sy]
                for b in tl.boxes
            ]
        )
        spans.append((np.array(tl.frames), boxes, tl.person))
    frames = core.read_frames(
        src,
        start=a.start,
        duration=a.duration,
        max_frames=a.max_frames,
        max_side=a.max_side,
    )
    with core.VideoWriter(
        out, info.fps, audio_from=src, audio_start=a.start, audio_duration=a.duration
    ) as writer:
        for i, frame in enumerate(frames):
            vis = frame[:, :, :3].copy()
            items: list[Label] = []
            for fr, boxes, pid in spans:
                if i < fr[0] or i > fr[-1]:
                    continue
                k = int(np.searchsorted(fr, i))
                if fr[k] == i:
                    box = boxes[k]
                else:
                    f0, f1 = fr[k - 1], fr[k]
                    u = (i - f0) / (f1 - f0)
                    box = boxes[k - 1] * (1 - u) + boxes[k] * u
                col = detectors.color_for(pid)
                p1, p2 = (round(box[0]), round(box[1])), (round(box[2]), round(box[3]))
                cv2.rectangle(vis, p1, p2, col, 2, cv2.LINE_AA)
                items.append(
                    (f"人物 {pid}", (p1[0], p1[1] - size - 6), size, (0, 0, 0), col)
                )
            _paint(vis, items)
            writer.write(vis)


# ── 95 人数カウント ─────────────────────────────────────────────────────────


def _count_graph(
    times: list[float], raw: list[int], smooth: list[float], title: str
) -> np.ndarray:
    """人数の時間変化のグラフ（点 = 数えた人数、線 = ならした人数）。"""
    gw, gh = 1000, 420
    ml, mr, mt, mb = 70, 24, 56, 56
    img = np.full((gh, gw, 3), 255, np.uint8)
    top = max(1, int(max(raw)), math.ceil(max(smooth))) + 1
    t0 = times[0]
    t1 = times[-1] if times[-1] > times[0] else times[0] + 1

    def px(t: float, c: float) -> tuple[int, int]:
        x = ml + (t - t0) / (t1 - t0) * (gw - ml - mr)
        y = gh - mb - c / top * (gh - mt - mb)
        return round(x), round(y)

    items: list[Label] = [(title, (ml, 14), 22, (30, 30, 30), None)]
    ystep = max(1, math.ceil(top / 8))
    for c in range(0, top + 1, ystep):
        y = px(t0, c)[1]
        cv2.line(img, (ml, y), (gw - mr, y), (225, 225, 225), 1)
        items.append((str(c), (ml - 34, y - 10), 16, (90, 90, 90), None))
    span = t1 - t0
    tstep = next(
        s for s in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1e9) if span / s <= 12
    )
    tk = math.ceil(t0 / tstep) * tstep
    while tk <= t1 + 1e-6:
        x = px(tk, 0)[0]
        cv2.line(img, (x, mt), (x, gh - mb), (235, 235, 235), 1)
        items.append((f"{tk:g}", (x - 10, gh - mb + 8), 16, (90, 90, 90), None))
        tk += tstep
    cv2.rectangle(img, (ml, mt), (gw - mr, gh - mb), (150, 150, 150), 1)
    pts = [px(t, c) for t, c in zip(times, smooth)]
    if len(pts) > 1:
        cv2.polylines(
            img, [np.array(pts, np.int32)], False, (200, 120, 30), 3, cv2.LINE_AA
        )
    for t, c in zip(times, raw):
        cv2.circle(img, px(t, c), 4, (60, 60, 230), -1, cv2.LINE_AA)
    items.append(("秒", (gw - mr - 20, gh - 26), 16, (90, 90, 90), None))
    items.append(("人", (ml - 40, mt - 30), 16, (90, 90, 90), None))
    items.append(("● 数えた人数", (gw - 330, 16), 16, (230, 60, 60)[::-1], None))
    items.append(("━ ならした人数", (gw - 180, 16), 16, (30, 120, 200)[::-1], None))
    return _paint(img, items)


def _count_view(frame: np.ndarray, people: list[Any], t: float | None) -> np.ndarray:
    vis = frame[:, :, :3].copy()
    h = vis.shape[0]
    size = _label_size(h)
    items: list[Label] = []
    for k, d in enumerate(sorted(people, key=lambda d: d.box[0])):
        x1, y1, x2, y2 = (round(v) for v in d.box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 220, 80), 2, cv2.LINE_AA)
        items.append(
            (
                f"{k + 1}: {d.score:.2f}",
                (x1, y1 - size - 4),
                max(12, size - 4),
                (0, 0, 0),
                (80, 220, 80),
            )
        )
    head = f"人数 {len(people)}" + (f"  {t:.1f}秒" if t is not None else "")
    items.append((head, (10, 10), round(size * 1.6), (255, 255, 255), (0, 0, 0)))
    return _paint(vis, items)


@command(
    95,
    "count-people",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    MODEL,
    arg(
        "--score",
        type=float,
        help="人とみなす確信度の下限（0〜1。既定は accurate 0.6・fast 0.3。目で数えた人数に合わせて決めた値）",
    ),
    arg(
        "--nested",
        type=float,
        default=NESTED,
        help="大きい枠の中に面積のこの割合以上が収まる小さい枠（同じ人の頭・上半身だけの枠など）を数えない"
        "（既定 0.65。1 より大きくすると除かない）",
    ),
    arg(
        "--fps",
        type=float,
        default=2.0,
        help="1 秒あたり何コマ数えるか（既定 2。0 で全部のコマ）。確認用の MP4 も数えたコマだけで、このコマ数/秒になる",
    ),
    arg(
        "--smooth",
        type=int,
        default=3,
        help="人数のゆらぎをならす幅（数えたコマの数。中央値。1 でならさない。既定 3）",
    ),
    arg(
        "--min-height",
        type=float,
        default=0.0,
        help="この高さ（画面の高さに対する割合）より小さい人は数えない（既定 0 = すべて数える）",
    ),
)
def count_people(a: argparse.Namespace) -> None:
    """人を検出して人数を数え、時間ごとの人数の JSON・グラフ・人数を書いた確認用の映像（画像）を書き出す。"""
    from . import detectors

    info = core.probe(a.src)
    score = a.score if a.score is not None else COUNT_SCORE[a.model]
    det = detectors.ObjectDetector(a.model, score=score, classes=["person"])

    def people_in(frame: np.ndarray) -> list[Any]:
        min_h = a.min_height * frame.shape[0]
        people = [d for d in det(frame) if d.box[3] - d.box[1] >= min_h]
        return _drop_nested(people, a.nested)

    w, h = core.scaled_size(info.width, info.height, None, a.max_side)
    sx, sy = info.width / w, info.height / h
    params = {
        "model": a.model,
        "score": score,
        "nested": a.nested,
        "min_height": a.min_height,
    }
    if info.is_image:
        out, json_path = _paths(a, a.src, "count-people", ".jpg")
        frame = next(core.read_frames(a.src, max_side=a.max_side))
        people = people_in(frame)
        core.save_image(out, _count_view(frame, people, None))
        core.write_json(
            json_path,
            {
                "source": str(a.src),
                "size": [info.width, info.height],
                "params": params,
                "count": len(people),
                "people": [
                    {"box": _xywh(d.box, sx, sy), "score": round(d.score, 3)}
                    for d in people
                ],
            },
        )
        core.log(f"人数: {len(people)}")
        core.log(f"保存: {out}")
        core.log(f"保存: {json_path}")
        return
    fps = a.fps if a.fps and a.fps < info.fps else info.fps
    out, json_path = _paths(a, a.src, "count-people", ".mp4")
    graph_path = _sibling(out, "-graph.png")
    sheet_path = out.with_suffix(".jpg")
    n_est = max(1, int(_span(info, a, fps) * fps))
    sheet_at = set(np.linspace(0, n_est - 1, min(12, n_est)).round().astype(int))
    frames = core.read_frames(
        a.src,
        start=a.start,
        duration=a.duration,
        max_frames=a.max_frames,
        max_side=a.max_side,
        fps=fps if fps != info.fps else None,
    )
    records, sheet = [], []
    t_begin = time.time()
    with core.VideoWriter(out, fps) as writer:
        for i, frame in enumerate(frames):
            t = core.t_of(i, fps, a.start, snap=fps == info.fps)
            people = people_in(frame)
            vis = _count_view(frame, people, t)
            writer.write(vis)
            if i in sheet_at:
                sheet.append(vis)
            records.append(
                {
                    "t": t,
                    "count": len(people),
                    "people": [
                        {"box": _xywh(d.box, sx, sy), "score": round(d.score, 3)}
                        for d in people
                    ],
                }
            )
            if (i + 1) % max(1, n_est // 10) == 0:
                core.log(f"  {i + 1}/{n_est} コマ")
    elapsed = time.time() - t_begin
    raw = [r["count"] for r in records]
    k = max(1, a.smooth) // 2
    smooth = [float(np.median(raw[max(0, i - k) : i + k + 1])) for i in range(len(raw))]
    for r, s in zip(records, smooth):
        r["smooth"] = s
    times = [r["t"] for r in records]
    title = f"人数の時間変化（{Path(a.src).name}・{fps:g} コマ/秒で数えた）"
    core.save_image(graph_path, _count_graph(times, raw, smooth, title))
    if sheet:
        core.save_image(
            sheet_path, core.contact_sheet(sheet, None, cols=4, cell_width=400)
        )
    peak = int(np.argmax(raw))
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": fps,
            "size": [info.width, info.height],
            "params": {**params, "smooth": a.smooth},
            "summary": {
                "samples": len(records),
                "max": max(raw),
                "max_at": times[peak],
                "mean": round(float(np.mean(raw)), 3),
                "mean_smooth": round(float(np.mean(smooth)), 3),
                "histogram": {str(c): raw.count(c) for c in sorted(set(raw))},
                "sec_per_frame": round(elapsed / len(records), 3),
            },
            "frames": records,
        },
    )
    core.log(
        f"{len(records)} コマを数えた: 最大 {max(raw)} 人（{times[peak]:.1f} 秒）・平均 {np.mean(raw):.2f} 人・"
        f"1 コマ {elapsed / len(records):.2f} 秒"
    )
    core.log(f"保存: {graph_path}")
    if sheet:
        core.log(f"保存: {sheet_path}")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")
