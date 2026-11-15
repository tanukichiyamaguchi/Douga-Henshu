"""動きの解析（揺れの補正）。

担当するテクニック:
    3 手ぶれ補正（stabilize）
   11 ローリングシャッター補正（rolling-shutter）
   22 カメラパスの平滑化（smooth-path）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。

共通の考え方:
- 隣り合うコマの「画面全体の動き」（相似変換: 移動・回転・拡大）を、格子状に散らした特徴点の追跡（往復で確かめる
  Lucas-Kanade）と RANSAC で求め、つないでカメラの通り道にする。計算は画面の中心を原点にした座標で行う
  （回転・拡大の中心を画面の真ん中にするため）。
- 動きが求まらない所（カット・速すぎる動き）では通り道を切り、前後を別の場面として扱う。
"""

from __future__ import annotations

import argparse
import itertools
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import ai, core
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/stabilize/（make_*.py で正解つきの試験映像を作り、test_*.py で測る。
# run_all.sh でまとめて動く）
NOTES: dict[int, dict[str, str]] = {
    3: {
        "status": "◎",
        "how": "隣り合うコマの画面全体の動き（相似変換: 移動・回転・拡大）を、4×4 の升目ごとに選んだ特徴点（計 480 点）を"
        "往復で確かめる Lucas-Kanade（ai.track_points）で追って RANSAC で求め、つないでカメラの通り道にする（画面の中心が"
        "原点）。人物と背景が別々に動く時は、前の組で選んだ層の点を引き継いで同じ層を選び続ける。人物の判定（MediaPipe の"
        " selfie。Apache-2.0）が場面を通して安定している場面だけ、人物を除いた背景で測り直す（2 回読む）。往復で追える点が"
        "ほぼ 0 の所（カット・白いフラッシュ）で通り道を切る。既定の l1 は通り道を L1 最適化でならす（Grundmann ら 2011 の"
        "考え方を、横・縦・回転・拡大の数値ごとに scipy の HiGHS で解く。1〜3 階の差の重み 1・1・100、生の道から離れる重み 1、"
        "余白は画面の ±6%・回転 ±2 度・拡大 ±5%。端は 1 秒ぶん直線の延長で延ばして解く）→ 止まる・等速・なめらかな加減速の"
        "区間に分かれる。全コマの四隅が元の絵で埋まる最小の拡大率（全コマ共通、上限 1.25 倍）を計算で求めて端の黒を隠し、"
        "明るさだけ軽く締める（unsharp 0.6）。--method gauss（前後 0.5 秒のガウス）、--method vidstab（ffmpeg の"
        " vidstabdetect・vidstabtransform。mincontrast 0.1・optalgo gauss・optzoom 1・bicubic）、--tripod（完全に止める）も"
        "使える。出力: MP4（音声は残す）・JSON（揺れの前後・拡大率・切れ目）・確認画像（約 1 秒ぶんのコマの平均の前後・"
        "前後のコマ・コマ間の動きのグラフ）。scipy・matplotlib は他の依存関係（easyocr・mediapipe など）が入れている物を使う",
        "measured": "正解つきの試験（揺らす前の映像に既知の揺れ（1.5〜6 Hz、各軸 5〜6px・回転 0.4 度）を加えて補正し、"
        "各コマを元のコマに SIFT で合わせて、位置の細かい揺れの残りを測る。測り方は、揺れた映像そのものを測って"
        " 8.15/7.46/7.48px（正解 8.16/7.46/7.49px）と確かめた）: 固定カメラで人が歩く vtest（768×576・80 コマ）"
        "8.16→0.03px（100% 減）、洗面所で髪を結ぶ人（540×960・64 コマ）7.46→0.22px（97% 減）、寄りながら髪がなびく人"
        "（70 コマ）7.49→0.71px（90% 減）。回転の残り 0.002・0.025・0.053 度。拡大 1.060・1.087・1.082 倍、補正で端に黒が"
        "出たコマ 0。ほかの方法: gauss 82・90・88% 減、vidstab 83・72・82% 減（vidstab は髪を結ぶ場面で回転が 0.42→0.25 度"
        "しか減らない）。人物を除かずに測ると（--with-people）髪を結ぶ場面 1.60px・寄りの場面 2.19px（人物の動きまで"
        "消してしまう）。--tripod は固定カメラの vtest を元の画角に 0.16px で止めた。AI の広告 story540（15 秒・361 コマ）:"
        " カット 4 か所を 73・139・213・297 コマ目（3.042・5.792・8.875・12.375 秒）でコマ単位で正しく切り、コマ間の揺れ"
        " 0.44→0.25px/コマ（拡大 1.034 倍。204 コマを背景で測り直した）。目・まつげ・眉の大写し（a-eye・b-lashlift・"
        "c-brow）でも 75・65・86% 減（拡大 1.01 倍）。速さ: 540×960・64〜70 コマで 6 秒前後、768×576・80 コマで 7.1 秒、"
        "story540 で 17.9 秒（他の作業と CPU を分け合った状態の壁時計）",
        "limits": "画面全体を 1 つの動き（相似変換）として直すので、手前と奥の視差が大きい場面や、画面の大部分が別々に"
        "動く場面では残りが出る（寄りながら人が動く場面で 0.71px）。補正の分だけ画角が狭くなる（拡大 1.01〜1.09 倍）。"
        "l1 は余白（--margin、既定 ±6%）を超える大きな揺れを残す。0.5 Hz より遅い揺れは意図した動きと区別できずに残る。"
        "目や顔の大写しのように人物の判定がコマごとに乱れる場面は、人物を除かずに測る（自動）。最初の版は人物の印を"
        "コマごとに使い分けて、まつげの施術の大写しで揺れを増やした（0.50→0.99px/コマ）ので、場面ごとに決めるように"
        "直した。手や道具が画面の大部分で動く大写しでは、その動きに引かれることがある。JSON とグラフの「後」の揺れは同じ"
        "方法で測った目安で、大写しの場面の初めなどで数コマ乱れることがある（story540 の 0〜0.2 秒の山は SIFT で測り直し、"
        "出力が揺れていないことを確かめた）。vidstab はカットで区切らない。ローリングシャッターのゆがみ（ぐにゃぐにゃ）は"
        "直らないので先に 11 を使う。白いフラッシュの各コマは切れ目として扱う",
        "tested": "make_shaky.py（vtest と story540 の洗面所・日差しの場面に既知の揺れ）・test_stabilize.py（3 方法を SIFT の"
        "位置合わせで正解と比べる）、explore_cuts.py（カットの判定: story540 の 4 か所と、カットの無い 9 本の追える点の"
        "割合）、tune_smooth.py・tune_l1.py・tune_vidstab.py（設定の比較）、check_layers.py・check_blash.py（層の選び方・"
        "大写し）。実際の映像 story540・a-eye・b-lashlift・c-brow・v4（範囲・縮小・音声）の確認画像を目で確認",
    },
    11: {
        "status": "○",
        "how": "近似。隣のコマとの画面全体の動き（03 と同じ測り方）から、各コマの撮影時点の速さ（前後の組の動きの平均。"
        "移動・回転・拡大）を出し、行ごとの撮影時刻のずれ τ = 読み出し × (行 − 中央) / 高さ（1 コマの時間に対する割合）"
        "だけ中身が動いた位置から cv2.remap で読み直す（中央の行の時刻にそろえる。縦の動きでは τ が読む行で変わるので"
        " 3 回くり返して合わせる）。--scan right/left で列ごと（横長のセンサーを縦に構えた縦動画）。--readout auto（既定）"
        "は、隣のコマとの 6 自由度のアフィンのせん断・伸び縮みと、速さの変化（加速）の関係から、読み出し時間と向きを"
        "最小二乗で推定する（当てはまり R² 0.3 未満・加速 0.5px/コマ² 未満・推定 1.5 超なら補正しない）。画像 1 枚は、"
        "縦（左右に読む時は横）に近い線分（OpenCV の LSD）の傾きを長さで重み付けした中央値から斜めのずれ（せん断）を"
        "求めて戻す。出力: MP4（音声は残す）・JSON（読み出し・向き・推定の当てはまり・コマごとの速さとずれ）・確認画像"
        "（補正の大きいコマの前後。まっすぐか見やすい補助線つき）",
        "measured": "正解つきの試験（building.jpg を 2.5 倍にした縦の線の多い絵を、行ごとに時刻をずらして撮った合成映像。"
        "同じ時刻に全部の行を撮った映像が正解）: 右へ 40px/コマの速いパン（読み出し 0.8・640×480）を --readout 0.8 で"
        "直すと PSNR 20.45→39.09dB、縦の線の傾きの正解との差 3.74→0.08 度（auto は「揺れが小さく推定できない」として"
        "補正しない）。ゆっくりした揺れ（0.5〜1.5 Hz、15px）＋パン（読み出し 0.7）: auto が 0.709 と推定（R² 0.997）、"
        "PSNR 29.73→38.73dB、傾きの差 1.03→0.11 度。速い手の揺れ（2〜8 Hz、6px、読み出し 0.7）: auto 1.19（実効の値。"
        "R² 0.95）で PSNR 28.98→32.54dB（正解の 0.7 を手で与えると 32.94dB）、傾きの差 0.90→0.51 度。縦動画（列ごと、"
        "読み出し 0.6）: 向き right を正しく推定（1.01）、PSNR 29.47→33.50dB。ローリングシャッターの無い 5 本（固定カメラの"
        " vtest、揺れだけの合成 3 本、AI の広告 story540）はどれも補正しなかった（R² 0.007〜0.03）。画像: せん断 +3.0・"
        "−1.5 度を測った値と正解の差 0.02・0.01 度、直した後の傾き 0.00・0.04 度。速さ: 640×480・72 コマで 4.0 秒",
        "limits": "画面全体が 1 つの動きをする前提の近似。本格的な方法（ジャイロの記録を使う、行ごとの動きを連続した"
        "曲線として最適化する）と違い、手前の人物だけが速く動く・視差が大きい場面のゆがみは直らない。1 コマを撮る間に"
        "速さが変わる速い揺れ（ぐにゃぐにゃ、およそ 4 Hz 以上）は一部しか直らない（速い揺れの試験で PSNR +3.6dB、線の"
        "傾きの差は約半分）。揺れ（速さの変化）の無い等速のパンでは読み出し時間を推定できないので --readout で指定する"
        "（スマートフォンでおよそ 0.3〜0.9）。auto の値はこの補正と組の実効の値で、速い揺れでは実際より大きく出る"
        "（0.7 → 1.19、0.6 → 1.01。速さの求め方が速い揺れで小さく出る分とつり合う）。補正で読み取り位置を動かした分、"
        "画面の端は端の画素をのばして埋める（最大のずれは JSON の max_shift_px）。レンズのゆがみは考えない。画像 1 枚では、"
        "建物の縦の線がもともと傾いている分（あおりなど）も一緒にまっすぐにする。AI の生成映像にはふつうローリング"
        "シャッターは無い（story540 では補正しなかった）",
        "tested": "make_rs.py（等速の速いパン・速い揺れ・ゆっくりした揺れ・縦動画の 4 種と、正解のグローバルシャッター）、"
        "test_rs.py（推定値・PSNR・線の傾き、ローリングシャッターの無い 5 本、画像のせん断 2 種）、確認画像を目で確認",
    },
    22: {
        "status": "◎",
        "how": "03 と同じ方法でカメラの通り道（横・縦・回転・拡大）を推定してならし、生の線とならした線のグラフ（PNG。"
        "カットで区切り点線で示す）、各コマの数値（JSON。カメラの向きで右・下・時計回り・寄るが +、場面の先頭が 0）、"
        "ならした道で揺れを消した動画（MP4）を書き出す。既定の l1 は止まる・等速・なめらかな加減速の区間に分かれる"
        "（三脚やドリーのような動き。余白 ±6% の中）。--method gauss は前後 --smooth 秒の重み付き平均（端は直線の延長で"
        "延ばし、等速の動きを保つ）。--no-video でグラフと JSON だけ",
        "measured": "正解つきの試験（building.jpg を 2 倍にした絵の上を、1 秒止まる → 4 秒で右へ 520px・下へ 60px"
        "（なめらかに加減速）→ 1 秒止まるカメラ。640×360・144 コマ。揺れ 7.07px・回転 0.40 度）: 測った生の道と正解の差"
        " 2.35px（最大 7.29px。6 秒でたまるずれ）・回転 0.042 度。ならした道と意図した道の差（一定のずれを除く）"
        " l1 1.50px・0.037 度（gauss 1.41px・0.045 度）。補正の大きさ 5.19・5.83px・0.398 度（加えた揺れ 5・5px・0.4 度）。"
        "書き出した動画を揺れの無い理想の映像に合わせると、細かい揺れの残り l1 0.36px（入力 7.14px、95% 減。gauss"
        " 0.23px）、全体のずれ 1.67px（gauss 1.58px）。数値の列だけの比較（tune_l1.py。止まる・等速・加減速・動いて止まる"
        "×速い・遅い揺れ、6 通りの乱数の平均）: l1 0.43〜2.16px、gauss 0.60〜2.90px（1 Hz の遅い揺れで l1 が良い）。論文の"
        "重み 10・1・100 はパンを 9〜21px 遅らせたので使わない。story540（カット 4）: 5 つの場面に分けて描き、5.8〜8.9 秒の"
        "右への約 480px のパンと 8.9〜12.4 秒の −13 度の回転を残した。速さ: 144 コマで 3.1 秒（動画なし）、story540"
        "（361 コマ、動画あり）で 18.6 秒",
        "limits": "通り道は 1 コマずつの動きをつないで作るので、長い場面ではずれがたまる（6 秒で最大 7px。グラフは場面の"
        "先頭からの相対の位置）。0.5 Hz より遅い揺れは意図した動きとして残る。l1 は余白（±6%）を超える揺れを残し、"
        "場面の端の 1 秒ほどは道が生の値に少し引かれることがある。カットをまたいだ通り道はつながない。そのほかは 03 と同じ",
        "tested": "make_pan.py・test_smooth_path.py（意図した道と揺れが分かっている合成映像で、生の道・ならした道・出力の"
        "動画を正解と比べる）、tune_l1.py（ならし方の比較）、story540（カットのある実際の映像）のグラフを目で確認",
    },
}

_ANALYSIS_SIDE = 960  # 動きを測る時の長辺の上限（px）
# 「揺れ」とみなす速さの変化を取り出す時の、ならす範囲（前後の秒）
_JITTER_WINDOW = 0.25

# ── 相似変換の小道具（3×3。画面の中心が原点の座標） ──────────────────────────


def _m3(m: np.ndarray) -> np.ndarray:
    out = np.eye(3)
    out[:2] = m[:2]
    return out


def _shift(dx: float, dy: float) -> np.ndarray:
    m = np.eye(3)
    m[0, 2], m[1, 2] = dx, dy
    return m


def _center(w: int, h: int) -> tuple[float, float]:
    return (w - 1) / 2, (h - 1) / 2


def _to_centered(m_px: np.ndarray, w: int, h: int) -> np.ndarray:
    """画素の座標の変換（2×3）→ 画面の中心が原点の座標の変換（3×3）。"""
    cx, cy = _center(w, h)
    return _shift(-cx, -cy) @ _m3(m_px) @ _shift(cx, cy)


def _to_pixel(m_c: np.ndarray, w: int, h: int) -> np.ndarray:
    """中心が原点の座標の変換（3×3）→ 画素の座標の変換（2×3。cv2.warpAffine に渡す形）。"""
    cx, cy = _center(w, h)
    return (_shift(cx, cy) @ m_c @ _shift(-cx, -cy))[:2]


def _rescale(m_c: np.ndarray, kx: float, ky: float) -> np.ndarray:
    """測った大きさの変換を、別の大きさ（横 kx 倍・縦 ky 倍）の画面の変換に直す。"""
    s = np.diag([kx, ky, 1.0])
    return s @ m_c @ np.diag([1 / kx, 1 / ky, 1.0])


def _angle(m: np.ndarray) -> float:
    return math.degrees(math.atan2(m[1, 0], m[0, 0]))


def _log_scale(m: np.ndarray) -> float:
    return math.log(max(math.hypot(m[0, 0], m[1, 0]), 1e-9))


def _from_params(tx: float, ty: float, a_deg: float, ls: float) -> np.ndarray:
    r, s = math.radians(a_deg), math.exp(ls)
    c, n = s * math.cos(r), s * math.sin(r)
    return np.array([[c, -n, tx], [n, c, ty], [0.0, 0.0, 1.0]])


# ── 動きを測る ────────────────────────────────────────────────────────────


def _features(
    gray: np.ndarray,
    max_points: int = 480,
    grid: int = 4,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """特徴点を画面全体に散らして選ぶ（格子のます目ごとに選ぶ。模様の多い所だけに集まらないように）。
    mask（0 以外の所だけから選ぶ）を渡すと、人物などを除ける。"""
    h, w = gray.shape[:2]
    per = max(8, max_points // (grid * grid))
    found = []
    for gy in range(grid):
        for gx in range(grid):
            x0, x1 = w * gx // grid, w * (gx + 1) // grid
            y0, y1 = h * gy // grid, h * (gy + 1) // grid
            sub = None if mask is None else mask[y0:y1, x0:x1]
            if sub is not None and not sub.any():
                continue
            p = cv2.goodFeaturesToTrack(
                gray[y0:y1, x0:x1], per, 0.01, 8, mask=sub, blockSize=7
            )
            if p is not None:
                found.append(p.reshape(-1, 2) + np.float32([x0, y0]))
    if not found:
        return np.zeros((0, 2), np.float32)
    return np.concatenate(found).astype(np.float32)


_RANSAC = {
    "method": cv2.RANSAC,
    "ransacReprojThreshold": 1.5,
    "maxIters": 2000,
    "confidence": 0.995,
    "refineIters": 10,
}
_PRIOR_POINTS = 150  # 次の組に引き継ぐ点の数の上限


@dataclass
class _Pair:
    """隣り合う 2 コマの動き。sim は相似変換（2×3、画素の座標。前のコマ → 今のコマの中身の動き）。"""

    sim: np.ndarray | None
    aff: np.ndarray | None  # 6 自由度のアフィン（full=True の時だけ）
    inliers: int  # RANSAC で当てはまった点
    tracked: int  # 往復で追えた点
    points: int  # 選んだ特徴点
    layer: np.ndarray | None = (
        None  # 選んだ動きに乗っていた点の、今のコマでの位置（次の組に引き継ぐ）
    )

    def usable(self, w: int, h: int) -> bool:
        """動きを使ってよいか。カットでは往復で追える点がほぼ 0 になる（story540 の 4 か所とも 0 点。
        カットの無い映像では最も少ない組でも 47%）。速すぎる動き・大きすぎる回転や拡大も除く。"""
        m = self.sim
        if m is None or self.inliers < 12:
            return False
        if self.tracked < max(20, 0.2 * self.points):
            return False
        return (
            math.hypot(m[0, 2], m[1, 2]) < 0.25 * math.hypot(w, h)
            and abs(_angle(m)) < 10
            and abs(_log_scale(m)) < 0.15
        )


def _pair_motion(
    prev: np.ndarray,
    cur: np.ndarray,
    *,
    full: bool = False,
    prior: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> _Pair:
    """隣り合う 2 コマ（グレー）の画面全体の動き。

    人物と背景が別々に動く（カメラが人を追い、背景が流れる）時は、当てはまる点の多い方を選ぶと、コマによって
    選ぶ層が入れ替わり、通り道が飛ぶ。そこで、前の組で選んだ層の点（prior。prev での位置）を一緒に追い、
    同じくらいの大きさの 2 つ目の層がある時は、prior の点が多く乗っている層を選び続ける。
    """
    p0 = _features(prev, mask=mask)
    if mask is not None and len(p0) < 60:
        p0 = _features(prev)  # 背景に模様が少ない時は画面全体で測る
    n_prior = 0
    if prior is not None and len(prior) and mask is not None:
        # 引き継ぐ点も、人物の上にある物は除く（前のコマで人物の層を選んでいても、背景へ移れるように）
        xi = np.clip(np.round(prior[:, 0]).astype(int), 0, mask.shape[1] - 1)
        yi = np.clip(np.round(prior[:, 1]).astype(int), 0, mask.shape[0] - 1)
        prior = prior[mask[yi, xi] > 0]
    if prior is not None and len(prior):
        p0 = np.concatenate([p0, prior.astype(np.float32)])
        n_prior = len(prior)
    if len(p0) < 12:
        return _Pair(None, None, 0, 0, len(p0))
    p1, ok = ai.track_points(prev, cur, p0)
    tracked = int(ok.sum())
    if tracked < 10:
        return _Pair(None, None, 0, tracked, len(p0))
    a, b = p0[ok], p1[ok]
    from_prior = (np.arange(len(p0)) >= len(p0) - n_prior)[ok]
    sim, mask = cv2.estimateAffinePartial2D(a, b, **_RANSAC)
    if sim is None or mask is None:
        return _Pair(None, None, 0, tracked, len(p0))
    inl = mask.ravel().astype(bool)
    rest = np.flatnonzero(~inl)
    if from_prior.sum() >= 10 and len(rest) >= 30:
        sim2, mask2 = cv2.estimateAffinePartial2D(a[rest], b[rest], **_RANSAC)
        if sim2 is not None and mask2 is not None:
            inl2 = np.zeros_like(inl)
            inl2[rest[mask2.ravel().astype(bool)]] = True
            if (
                inl2.sum() >= 0.35 * inl.sum()
                and (inl2 & from_prior).sum() > (inl & from_prior).sum()
            ):
                sim, inl = sim2, inl2
    aff = None
    if full:
        aff, _ = cv2.estimateAffine2D(a[inl], b[inl], **_RANSAC)
    layer = b[inl]
    if len(layer) > _PRIOR_POINTS:
        layer = layer[np.linspace(0, len(layer) - 1, _PRIOR_POINTS).astype(int)]
    return _Pair(sim, aff, int(inl.sum()), tracked, len(p0), layer)


@dataclass
class _Motion:
    """動画の各コマの、前のコマからの動き（測った大きさの、中心が原点の座標）。"""

    fps: float
    size: tuple[int, int]  # 動きを測った大きさ（幅, 高さ）
    steps: list[np.ndarray | None] = field(
        default_factory=list
    )  # 3×3。先頭と切れ目は None
    affine: list[np.ndarray | None] = field(default_factory=list)  # 3×3（full の時）
    inliers: list[int] = field(default_factory=list)
    points: list[int] = field(default_factory=list)
    people: bool = False  # 人物を除く設定で測ったか
    people_frames: int = 0  # 人物を除いて測ったコマの数
    masks: dict[int, np.ndarray] = field(
        default_factory=dict
    )  # コマ番号 → 人物の印（1/4）

    @property
    def n(self) -> int:
        return len(self.steps)

    @property
    def breaks(self) -> list[int]:
        """通り道が切れる所（新しい場面が始まるコマ。0 は含めない）。"""
        return [i for i in range(1, self.n) if self.steps[i] is None]

    def segments(self) -> list[tuple[int, int]]:
        """切れ目で分けた区間（始め, 終わりの次）。"""
        edges = [0, *self.breaks, self.n]
        return [(s, e) for s, e in itertools.pairwise(edges) if e > s]


def _measure(
    src: str | Path,
    *,
    rng: dict[str, Any],
    size: tuple[int, int],
    fps: float,
    full: bool,
    label: str,
    people: _PersonMasks | None = None,
    masks: dict[int, np.ndarray] | None = None,
    breaks: set[int] | None = None,
) -> _Motion:
    """動画を 1 回読み、隣り合うコマの動きを測る。

    people を渡すと、全コマの人物の印も求めて覚える。masks（コマ番号 → 1/4 の大きさの人物の印）を渡すと、
    印のあるコマからは背景の特徴点だけを選び、印の無いコマの組は測らない（2 回目に、使う場面だけ測り直すため）。
    breaks を渡すと、その所で必ず通り道を切る（1 回目と切れ目をそろえるため）。
    """
    w, h = size
    motion = _Motion(fps=fps, size=size)
    prev = prior = None
    total = rng["max_frames"] or round(
        (rng["duration"] or max(core.probe(src).duration - rng["start"], 0)) * fps
    )
    for i, frame in enumerate(core.read_frames(src, **rng, size=size)):
        gray = core.to_gray(frame)
        if people is not None:
            people.add(frame)
        step = aff = None
        inl = pts = 0
        skip = masks is not None and i - 1 not in masks
        if prev is not None and not skip and not (breaks and i in breaks):
            mask = _background(masks[i - 1], size) if masks else None
            pair = _pair_motion(prev, gray, full=full, prior=prior, mask=mask)
            inl, pts = pair.inliers, pair.points
            prior = None
            if pair.usable(w, h):
                step = _to_centered(pair.sim, w, h)
                prior = pair.layer
                if pair.aff is not None:
                    aff = _to_centered(pair.aff, w, h)
        else:
            prior = None
        motion.steps.append(step)
        motion.affine.append(aff)
        motion.inliers.append(inl)
        motion.points.append(pts)
        prev = gray
        if total and (i + 1) % max(1, total // 5) == 0:
            core.log(f"  {label}: {i + 1}/{total} コマ")
    core.require(motion.n >= 2, "コマが 2 枚以上ある動画を指定してください")
    return motion


def _analyze(
    src: str | Path,
    *,
    start: float = 0.0,
    duration: float | None = None,
    max_frames: int | None = None,
    size: tuple[int, int],
    fps: float,
    full: bool = False,
    people: bool = False,
    label: str = "動きを測っています",
) -> _Motion:
    """動画を読み、隣り合うコマの動きを測る。

    people=True なら、人物の印が安定している場面（_PersonMasks.masks）だけ、人物を除いた背景で測り直す
    （2 回読む。1 回目は画面全体で測りながら人物の印を求め、切れ目で分けた場面ごとに使うかを決める）。
    """
    rng = {"start": start, "duration": duration, "max_frames": max_frames}
    pm = _PersonMasks(size) if people else None
    motion = _measure(
        src, rng=rng, size=size, fps=fps, full=full, label=label, people=pm
    )
    if pm is None:
        return motion
    motion.people = True
    masks = pm.masks(motion.segments())
    if not masks:
        return motion
    core.log(f"  人物の動きを除くため、背景で測り直します（{len(masks)} コマ）")
    again = _measure(
        src,
        rng=rng,
        size=size,
        fps=fps,
        full=full,
        label="背景で測っています",
        masks=masks,
        breaks=set(motion.breaks),
    )
    for i in range(1, min(motion.n, again.n)):
        # 背景で測れた組だけ置き換える（測れなかった組は画面全体の値を残し、切れ目を増やさない）
        if i - 1 in masks and again.steps[i] is not None:
            motion.steps[i] = again.steps[i]
            motion.affine[i] = again.affine[i]
            motion.inliers[i] = again.inliers[i]
            motion.points[i] = again.points[i]
    motion.people_frames = len(masks)
    motion.masks = masks
    return motion


class _PersonMasks:
    """人物（MediaPipe の selfie。Apache-2.0）を少し太らせた印を全コマで求め、場面ごとに使うかを決める。

    人物がカメラと別に動く（頭を振る・髪がなびく・腕を動かす）と、その動きまで揺れとして消してしまうため、
    カメラの動きは背景で測りたい。ただし目や顔の大写しでは人物の判定がコマごとに大きく揺れ、残った「背景」が
    道具（ビューラー・手袋の手）だけになって誤る。そこで、切れ目で分けた場面ごとに、人物の印が前のコマと
    安定して重なり（重なりの割合 IoU ≥ 0.8）、人物が画面の 5〜85% のコマが 85% 以上あり、IoU の中央値が
    0.9 以上の場面だけ印を使う（場面の中で使ったり使わなかったりすると、測る層が入れ替わって通り道が飛ぶ）。
    実測: 上半身の場面の IoU の中央値 0.95〜0.98（0.8 未満のコマ 0〜5%）、目の大写し 0.71（65%）、
    まつげの施術の大写し 0.83（43%）。
    """

    def __init__(self, size: tuple[int, int]) -> None:
        self.seg = ai.PersonSegmenter("selfie")
        self.size = size
        self.small: list[np.ndarray] = []  # 1/4 の大きさの人物の印（bool）
        self.frac: list[float] = []
        self.iou: list[float] = []  # 前のコマとの重なり（最初のコマは nan）

    def add(self, frame: np.ndarray) -> None:
        w, h = self.size
        prob = cv2.resize(
            self.seg(frame),
            (max(1, w // 4), max(1, h // 4)),
            interpolation=cv2.INTER_AREA,
        )
        # 測る大きさで画面の長辺の 2% だけ太らせる（1/4 の大きさで行う。髪の端なども除くため）
        r = max(1, round(0.02 * max(w, h) / 4))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        small = cv2.dilate((prob > 0.5).astype(np.uint8), kernel) > 0
        iou = math.nan
        if self.small:
            union = int((small | self.small[-1]).sum())
            iou = (small & self.small[-1]).sum() / union if union else 1.0
        self.small.append(small)
        self.frac.append(float(small.mean()))
        self.iou.append(float(iou))

    def masks(self, segments: list[tuple[int, int]]) -> dict[int, np.ndarray]:
        """使う場面のコマの、人物の印（1/4 の大きさ。True が人物）。"""
        out: dict[int, np.ndarray] = {}
        for s, e in segments:
            if e - s < 3:
                continue
            frac = np.array(self.frac[s:e])
            iou = np.array(self.iou[s:e])
            iou[0] = 1.0  # 場面の先頭は前の場面と比べない
            good = (frac >= 0.05) & (frac <= 0.85) & (iou >= 0.8)
            if good.mean() < 0.85 or np.median(iou[1:]) < 0.9:
                continue
            last = self.small[s]
            for k in range(s, e):
                if good[k - s]:
                    last = self.small[k]  # 判定が乱れたコマは、直前の安定した印を使う
                out[k] = last
        return out


def _background(person_small: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """1/4 の大きさの人物の印 → 測る大きさの背景の印（255 が背景。cv2 の mask の形）。"""
    big = cv2.resize(
        person_small.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    )
    return np.where(big > 0, 0, 255).astype(np.uint8)


# ── 通り道をつなぐ・ならす ──────────────────────────────────────────────────


def _path(motion: _Motion) -> tuple[list[np.ndarray], np.ndarray]:
    """各コマの累積の変換（場面の先頭のコマ → そのコマ。3×3）と、その数値 (横, 縦, 回転[度], 拡大率の対数)。"""
    acc: list[np.ndarray] = []
    params = np.zeros((motion.n, 4))
    for i, step in enumerate(motion.steps):
        if step is None:
            acc.append(np.eye(3))
            continue
        c = step @ acc[-1]
        acc.append(c)
        # 回転と拡大は足し合わせで求める（±180 度で折り返さないように）
        params[i] = (
            c[0, 2],
            c[1, 2],
            params[i - 1, 2] + _angle(step),
            params[i - 1, 3] + _log_scale(step),
        )
    return acc, params


def _pad_ends(x: np.ndarray, pad: int, fit: int) -> np.ndarray:
    """両端を pad 個ずつ延ばす。端の近く fit 個に当てはめた直線の端の値を中心に点対称に折り返す
    （等速の動きはそのまま続く。端の 1 コマの揺れを中心にすると、その揺れが 2 倍になって延びるため）。"""
    n = len(x)
    pad, fit = min(pad, n - 1), max(2, min(fit, n))
    k = np.arange(fit)
    left = np.polyval(np.polyfit(k, x[:fit], 1), 0)
    right = np.polyval(np.polyfit(k, x[n - fit :], 1), fit - 1)
    return np.concatenate(
        [2 * left - x[1 : pad + 1][::-1], x, 2 * right - x[n - 1 - pad : n - 1][::-1]]
    )


def _gauss(x: np.ndarray, radius: int) -> np.ndarray:
    """前後 radius コマの重み付き平均（vid.stab と同じ形のガウス）。端は _pad_ends で延ばし、等速の動きを保つ。"""
    n = len(x)
    if n < 3 or radius < 1:
        return x.copy()
    radius = min(radius, n - 1)
    d = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-((d / (radius / 2)) ** 2))
    k /= k.sum()
    return np.convolve(_pad_ends(x, radius, radius), k, mode="valid")


def _l1(
    c: np.ndarray,
    margin: float,
    weights: tuple[float, float, float] = (1.0, 1.0, 100.0),
    stay: float = 1.0,
    pad: int = 0,
) -> np.ndarray:
    """L1 最適化でならす（Grundmann ら 2011 の考え方を、数値ごとに分けて解く）。

    1〜3 階の差（速さ・加速度・加加速度）の絶対値の重み付きの和を最小にする → 止まる・等速・なめらかな加減速の
    区間に分かれた通り道になる。生の通り道から ±margin より離れない（切り取る余白の中に収める）。
    生の道からの離れ方にも重み stay を付ける（同じなめらかさの解が余白の端に寄るのを防ぎ、意図したパンを遅らせない）。
    重みは論文の 10・1・100 だと、数値ごとに解くこの形では速さの項が「動く量を減らす」方へ効き、等速・加減速・
    動いて止まるパンが 9〜21px ずれた（試験 tune_l1.py。止まった画は 0px）。1・1・100・stay 1 で、どの道も
    速い揺れ（3 Hz）では前後 0.5 秒のガウスと同じくらい（0.4〜2.0px）、遅い揺れ（1 Hz）ではガウスより小さく
    （止まった画 0.79 対 2.55px）揺れだけを消せた。
    端では差の項が少なく、道が生の値に引かれる。pad コマぶん両端を _pad_ends で延ばしてから解き、延ばした所を
    捨てる（ガウスと同じ扱い）。
    """
    from scipy import sparse
    from scipy.optimize import linprog

    if len(c) < 5 or margin <= 0:
        return c.copy()
    pad = min(pad, len(c) - 1)
    if pad > 0:
        full = _l1(_pad_ends(c, pad, pad), margin, weights, stay)
        return full[pad:-pad]
    n = len(c)
    ds = [
        sparse.diags([-1.0, 1.0], [0, 1], shape=(n - 1, n)),
        sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)),
        sparse.diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(n - 3, n)),
        sparse.identity(n, format="csr"),  # 生の道からの離れ方（p − c）
    ]
    sizes = [d.shape[0] for d in ds]
    total = n + sum(sizes)
    rows, rhs, cost = [], [], [np.zeros(n)]
    off = n
    for j, (d, wgt, m) in enumerate(zip(ds, (*weights, stay), sizes, strict=True)):
        e = sparse.csr_matrix(
            (np.ones(m), (np.arange(m), off + np.arange(m))), shape=(m, total)
        )
        dp = sparse.hstack([d, sparse.csr_matrix((m, total - n))])
        rows += [dp - e, -dp - e]  # |D p − b| <= e
        b = c if j == 3 else np.zeros(m)
        rhs += [b, -b]
        cost.append(np.full(m, wgt))
        off += m
    a_ub = sparse.vstack(rows).tocsr()
    bounds = [(float(v - margin), float(v + margin)) for v in c]
    bounds += [(0.0, None)] * sum(sizes)
    res = linprog(
        np.concatenate(cost),
        A_ub=a_ub,
        b_ub=np.concatenate(rhs),
        bounds=bounds,
        method="highs",
    )
    if res.status != 0 or res.x is None:
        core.log(
            f"  L1 の最適化が解けなかったので、ガウスでならします（{res.message}）"
        )
        return _gauss(c, 12)
    return np.asarray(res.x[:n])


@dataclass
class _Plan:
    """補正の計画（各コマにかける変換）。"""

    motion: _Motion
    acc: list[np.ndarray]
    raw: np.ndarray  # 生の通り道の数値（コマ数×4）
    smooth: np.ndarray  # ならした通り道の数値
    warps: list[np.ndarray]  # 各コマにかける補正（測った大きさ、中心が原点の 3×3）
    zoom: float  # 端の黒を隠すための拡大率（全コマ共通）
    zoom_needed: float  # 上限で切る前の拡大率
    uncovered: int  # 拡大の上限のため、端に黒が出るコマの数


def _needed_zoom(warp: np.ndarray, w: int, h: int) -> float:
    """補正をかけた後、画面の四隅まで元の絵で埋まるのに要る拡大率（画面の中心を基準に拡大）。"""
    inv = np.linalg.inv(warp)
    a, b = inv[:2, :2], inv[:2, 2]
    hx, hy = (w - 1) / 2, (h - 1) / 2
    u = 1.0  # 1 / 拡大率
    for q in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        k = a @ np.array(q)
        for i, half in ((0, hx), (1, hy)):
            if abs(b[i]) >= half:
                return math.inf
            if k[i] > 1e-12:
                u = min(u, (half - b[i]) / k[i])
            elif k[i] < -1e-12:
                u = min(u, (-half - b[i]) / k[i])
    return 1.0 / max(u, 1e-6)


def _plan(
    motion: _Motion,
    *,
    method: str,
    smooth_s: float,
    margin: float,
    max_zoom: float,
    tripod: bool = False,
) -> _Plan:
    """通り道をならし、各コマにかける補正と、端を隠す拡大率を決める。"""
    w, h = motion.size
    acc, raw = _path(motion)
    smooth = raw.copy()
    radius = max(1, round(smooth_s * motion.fps))
    # 余白（L1 の上限と、ガウスで補正が大きすぎる時に抑える上限）: 横・縦は画面の margin 倍、回転と拡大は小さめ
    limits = (margin * w, margin * h, 2.0 * margin / 0.06, 0.05 * margin / 0.06)
    for s, e in motion.segments():
        for j in range(4):
            seg = raw[s:e, j]
            if tripod:
                smooth[s:e, j] = np.median(seg)
            elif method == "l1":
                smooth[s:e, j] = _l1(seg, limits[j], pad=round(motion.fps))
            else:
                smooth[s:e, j] = _gauss(seg, radius)
    if method != "l1" and not tripod:
        # ガウスでならした道が生の道から離れすぎる所は抑える（拡大しすぎて画質が落ちるのを防ぐ）
        lim = np.array(limits) * 1.5
        smooth = raw + np.clip(smooth - raw, -lim, lim)
    warps = [_from_params(*smooth[i]) @ np.linalg.inv(acc[i]) for i in range(motion.n)]
    needs = np.array([_needed_zoom(m, w, h) for m in warps])
    needed = float(np.max(needs))
    zoom = float(min(needed, max_zoom))
    return _Plan(
        motion,
        acc,
        raw,
        smooth,
        warps,
        zoom,
        needed,
        int(np.sum(needs > zoom + 1e-6)),
    )


# ── 揺れの大きさを測る ─────────────────────────────────────────────────────


def _jitter(motion: _Motion, k: float = 1.0) -> dict[str, float]:
    """コマ間の動きのうち、細かく変わる成分（揺れ）の大きさ。k は px を書き出す大きさに直す倍率。"""
    radius = max(1, round(_JITTER_WINDOW * motion.fps))
    hi_t, hi_a, all_t = [], [], []
    run: list[tuple[float, float, float]] = []

    def flush() -> None:
        if len(run) >= 3:
            v = np.array(run)
            for j in range(3):
                v[:, j] -= _gauss(v[:, j], radius)
            hi_t.extend(np.hypot(v[:, 0], v[:, 1]) * k)
            hi_a.extend(np.abs(v[:, 2]))
        run.clear()

    for step in motion.steps:
        if step is None:
            flush()
            continue
        run.append((step[0, 2], step[1, 2], _angle(step)))
        all_t.append(math.hypot(step[0, 2], step[1, 2]) * k)
    flush()

    def rms(v: list[float]) -> float:
        return float(np.sqrt(np.mean(np.square(v)))) if v else 0.0

    return {
        "shake_px": round(rms(hi_t), 3),
        "shake_deg": round(rms(hi_a), 4),
        "motion_px": round(rms(all_t), 3),
        "pairs": len(all_t),
    }


def _reduction(before: dict[str, float], after: dict[str, float]) -> float:
    b = before["shake_px"]
    return round(100 * (1 - after["shake_px"] / b), 1) if b > 0 else 0.0


# ── 書き出し ──────────────────────────────────────────────────────────────


def _unsharp(img: np.ndarray, amount: float) -> np.ndarray:
    """明るさだけを軽く締める（拡大・補間で甘くなる分を戻す。ffmpeg の unsharp=5:5 と同じ考え方）。"""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0].astype(np.float32)
    blur = cv2.GaussianBlur(y, (5, 5), 0)
    ycc[:, :, 0] = np.clip(y + amount * (y - blur), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def _render(
    src: str,
    out: Path,
    plan: _Plan,
    *,
    rng: dict[str, Any],
    size: tuple[int, int],
    sharpen: float,
) -> _Motion:
    """補正をかけて書き出す。書き出したコマの動きも測って返す（揺れがどれだけ減ったかを見るため）。"""
    w, h = size
    aw, ah = plan.motion.size
    kx, ky = w / aw, h / ah
    zoom = np.diag([plan.zoom, plan.zoom, 1.0])
    after = _Motion(fps=plan.motion.fps, size=plan.motion.size)
    breaks = set(plan.motion.breaks)
    masks = plan.motion.masks
    prev = prior = prev_mask = None
    duration = rng["duration"]
    if rng["max_frames"] and not duration:
        duration = rng["max_frames"] / plan.motion.fps
    writer = core.VideoWriter(
        out,
        plan.motion.fps,
        audio_from=src,
        audio_start=rng["start"],
        audio_duration=duration,
    )
    with writer:
        for i, frame in enumerate(core.read_frames(src, **rng, size=size)):
            if i >= plan.motion.n:
                break
            m = _to_pixel(zoom @ _rescale(plan.warps[i], kx, ky), w, h)
            res = cv2.warpAffine(
                frame,
                m,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )
            if sharpen > 0:
                res = _unsharp(res, sharpen)
            writer.write(res)
            small = core.to_gray(
                cv2.resize(res, (aw, ah), interpolation=cv2.INTER_AREA)
            )
            mask = None
            # 入力の背景の印を、同じ補正で動かして使う（前後で同じ所を測るため）
            if i in masks:
                mask = cv2.warpAffine(
                    _background(masks[i], (aw, ah)),
                    _to_pixel(zoom @ plan.warps[i], aw, ah),
                    (aw, ah),
                    flags=cv2.INTER_NEAREST,
                    borderValue=0,
                )
            step = None
            if prev is not None and i not in breaks:
                pair = _pair_motion(prev, small, prior=prior, mask=prev_mask)
                prior = None
                if pair.usable(aw, ah):
                    step = _to_centered(pair.sim, aw, ah)
                    prior = pair.layer
            after.steps.append(step)
            prev, prev_mask = small, mask
            if (i + 1) % max(1, plan.motion.n // 5) == 0:
                core.log(f"  書き出し: {i + 1}/{plan.motion.n} コマ")
    return after


def _check_sheet(
    src: str,
    out_video: Path,
    *,
    rng: dict[str, Any],
    size: tuple[int, int],
    fps: float,
    n: int,
    title: str,
    graph: np.ndarray | None = None,
) -> np.ndarray:
    """確認用の画像: 約 1 秒ぶんのコマを重ねた平均（揺れが大きいほどぼやける）と、前後のコマ。"""
    span = max(2, min(n, round(fps)))
    first = max(0, (n - span) // 2)
    t0 = rng["start"] + first / fps
    before = core.read_all(src, start=t0, max_frames=span, size=size)
    after = core.read_all(out_video, start=first / fps, max_frames=span)
    cells = [
        np.mean(before, axis=0).astype(np.uint8),
        np.mean(after, axis=0).astype(np.uint8),
        before[len(before) // 2],
        after[len(after) // 2],
    ]
    labels = [
        f"前: {span} コマの平均",
        f"後: {span} コマの平均",
        f"前 {t0 + span / 2 / fps:.2f}秒",
        "後（同じ時刻）",
    ]
    sheet = core.contact_sheet(cells, labels, cols=4, cell_width=300)
    head = np.full((44, sheet.shape[1], 3), 24, np.uint8)
    core.draw_text(head, title, (8, 8), size=22, bg=None)
    parts = [head, sheet]
    if graph is not None:
        g = cv2.resize(
            graph,
            (sheet.shape[1], round(graph.shape[0] * sheet.shape[1] / graph.shape[1])),
            interpolation=cv2.INTER_AREA,
        )
        parts.append(g)
    return np.vstack(parts)


# ── グラフ ────────────────────────────────────────────────────────────────


def _plt() -> Any:
    """日本語の書体を設定した matplotlib（画面なしで描く）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for f in core.JP_FONTS:
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=f
            ).get_name()
            break
    return plt


def _fig_to_bgr(plt: Any, fig: Any) -> np.ndarray:
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _camera_view(params: np.ndarray, kx: float, ky: float) -> np.ndarray:
    """中身の動きの数値 → カメラの動き（右・下へ動く・時計回り・寄るで増える）。横縦は書き出しの px、拡大は %。"""
    return np.column_stack(
        [
            -params[:, 0] * kx,
            -params[:, 1] * ky,
            -params[:, 2],
            np.exp(params[:, 3]) * 100,
        ]
    )


def _plot_paths(
    times: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    segments: list[tuple[int, int]],
    title: str,
) -> np.ndarray:
    """カメラの通り道（横・縦・回転・拡大）の生の線と、ならした線のグラフ。"""
    plt = _plt()
    names = [
        "横（px、右へ +）",
        "縦（px、下へ +）",
        "回転（度、時計回り +）",
        "拡大（%、寄ると +）",
    ]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True, dpi=100)
    for j, ax in enumerate(axes):
        for k, (s, e) in enumerate(segments):
            ax.plot(
                times[s:e],
                raw[s:e, j],
                color="#8a9099",
                lw=1.1,
                label="測った通り道（生）" if k == 0 else None,
            )
            ax.plot(
                times[s:e],
                smooth[s:e, j],
                color="#d93025",
                lw=2.2,
                label="ならした通り道" if k == 0 else None,
            )
            if s > 0:
                ax.axvline(times[s], color="#3c4043", ls="--", lw=0.9)
        ax.set_ylabel(names[j], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=10)
    axes[-1].set_xlabel("時刻（秒）　点線は場面の切れ目（カットなど）")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _fig_to_bgr(plt, fig)


def _plot_shake(
    times: np.ndarray, before: _Motion, after: _Motion, k: float, title: str
) -> np.ndarray:
    """コマ間の横・縦の動き（px/コマ）の前後を重ねたグラフ。"""
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, dpi=100)
    for j, (ax, name) in enumerate(zip(axes, ["横の動き", "縦の動き"], strict=True)):
        for m, color, lab in ((before, "#8a9099", "前"), (after, "#d93025", "後")):
            v = np.array(
                [np.nan if s is None else s[j, 2] * k for s in m.steps], dtype=float
            )
            ax.plot(times[: len(v)], v, color=color, lw=1.4, label=lab)
        ax.set_ylabel(f"{name}\n（px/コマ）", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=10)
    axes[-1].set_xlabel("時刻（秒）")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _fig_to_bgr(plt, fig)


# ── 共通: 範囲・大きさ ─────────────────────────────────────────────────────


def _video(src: str, what: str) -> core.MediaInfo:
    info = core.probe(src)
    core.require(
        not info.is_image,
        f"動画を指定してください（{what}は、コマとコマの間の動きを使うため画像 1 枚では使えません）",
    )
    return info


def _rng(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": a.start or 0.0,
        "duration": a.duration,
        "max_frames": a.max_frames,
    }


def _sizes(
    info: core.MediaInfo, max_side: int | None
) -> tuple[tuple[int, int], tuple[int, int]]:
    """(書き出す大きさ, 動きを測る大きさ)。"""
    out = core.scaled_size(info.width, info.height, max_side=max_side)
    ana = core.scaled_size(
        info.width, info.height, max_side=min(_ANALYSIS_SIDE, max_side or 10**9)
    )
    return out, ana


def _times(n: int, fps: float, start: float) -> np.ndarray:
    return start + np.arange(n) / fps


# ── 03 手ぶれ補正 ─────────────────────────────────────────────────────────

_STAB_METHODS = {
    "l1": "特徴点の追跡＋L1 最適化（止まる・等速・なめらかな加減速の区間に分ける。既定）",
    "gauss": "特徴点の追跡＋前後 --smooth 秒のガウスでならす",
    "vidstab": "ffmpeg の vid.stab（2 回に分けて処理。カットで区切らない）",
}
_PEOPLE = arg(
    "--with-people",
    action="store_true",
    help="人物も含めて画面全体で動きを測る（既定は、人物の判定が安定している場面だけ、人物を除いた背景で"
    "測る。顔や目の大写しなど判定が乱れる場面は自動で画面全体を使う）",
)


def _vidstab(
    src: str,
    out: Path,
    *,
    rng: dict[str, Any],
    size: tuple[int, int],
    info: core.MediaInfo,
    smooth_s: float,
    shakiness: int,
    tripod: bool,
    sharpen: float,
) -> dict[str, Any]:
    """ffmpeg の vidstabdetect → vidstabtransform で補正する。"""
    pre = ["-ss", f"{rng['start']:.4f}"] if rng["start"] else []
    lim = ["-t", f"{rng['duration']:.4f}"] if rng["duration"] else []
    if rng["max_frames"]:
        lim += ["-frames:v", str(rng["max_frames"])]
    scale = (
        f"scale={size[0]}:{size[1]}:flags=area,"
        if size != (info.width, info.height)
        else ""
    )
    smoothing = max(1, round(smooth_s * info.fps))
    with tempfile.TemporaryDirectory(prefix="vfx-vidstab-") as tmp:
        trf = Path(tmp) / "motion.trf"
        detect = (
            f"{scale}vidstabdetect=shakiness={shakiness}:accuracy=15:stepsize=6"
            ":mincontrast=0.1"
            f":result={trf}" + (":tripod=1" if tripod else "")
        )
        core.log("  vid.stab: 動きを測っています")
        core.ffmpeg([*pre, "-i", src, *lim, "-vf", detect, "-an", "-f", "null", "-"])
        transform = (
            f"{scale}vidstabtransform=input={trf}:smoothing={smoothing}:optalgo=gauss"
            ":optzoom=1:interpol=bicubic:crop=black" + (":tripod=1" if tripod else "")
        )
        if sharpen > 0:
            transform += f",unsharp=5:5:{sharpen:.2f}:3:3:{sharpen / 2:.2f}"
        audio = (
            ["-map", "0:a?", "-c:a", "aac", "-b:a", "192k"] if info.has_audio else []
        )
        core.log("  vid.stab: 補正して書き出しています")
        core.ffmpeg(
            [
                *pre,
                "-i",
                src,
                *lim,
                "-vf",
                transform,
                "-map",
                "0:v:0",
                *audio,
                *core.MP4_ARGS,
                "-shortest",
                str(out),
            ]
        )
    return {"smoothing_frames": smoothing, "shakiness": shakiness}


@command(
    3,
    "stabilize",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_STAB_METHODS),
        default="l1",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _STAB_METHODS.items()),
    ),
    arg(
        "--smooth",
        type=float,
        default=0.5,
        help="gauss・vidstab でならす範囲（前後それぞれの秒。既定 0.5。大きいほどなめらかだが、"
        "意図した動きも鈍り、拡大も増える）",
    ),
    arg(
        "--margin",
        type=float,
        default=0.06,
        help="l1 で通り道を動かしてよい幅（画面の幅・高さに対する割合。既定 0.06 ≒ 拡大 1.13 倍まで）",
    ),
    arg(
        "--max-zoom",
        type=float,
        default=1.25,
        help="端の黒を隠す拡大の上限（gauss・l1。既定 1.25。超える所は端に黒が出る）",
    ),
    arg(
        "--shakiness",
        type=int,
        default=6,
        help="vidstab の揺れの強さの見込み（1〜10。既定 6。速い揺れなら上げる）",
    ),
    arg(
        "--tripod",
        action="store_true",
        help="三脚のように完全に止める（意図したパンも消える。拡大が大きくなる）",
    ),
    arg(
        "--sharpen",
        type=float,
        default=0.6,
        help="補正後に明るさを締める強さ（0 で無し。既定 0.6）",
    ),
    _PEOPLE,
)
def stabilize(a: argparse.Namespace) -> None:
    """カメラの細かい揺れを抑えた動画を書き出す（揺れの減り方を JSON と確認画像に出す）。"""
    info = _video(a.src, "手ぶれ補正")
    rng = _rng(a)
    size, ana = _sizes(info, a.max_side)
    out = Path(a.out) if a.out else core.default_out(a.src, "stabilize", ".mp4")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    check_path = out.with_name(out.stem + "-check.jpg")
    k = size[0] / ana[0]
    people = not a.with_people
    before = _analyze(a.src, **rng, size=ana, fps=info.fps, people=people)
    data: dict[str, Any] = {
        "source": str(a.src),
        "method": a.method,
        "people_masked_frames": before.people_frames,
        "fps": info.fps,
        "size": size,
        "start": rng["start"],
        "frames": before.n,
        "breaks_sec": [core.t_of(i, info.fps, rng["start"]) for i in before.breaks],
    }
    if a.method == "vidstab":
        data["params"] = _vidstab(
            a.src,
            out,
            rng=rng,
            size=size,
            info=info,
            smooth_s=a.smooth,
            shakiness=a.shakiness,
            tripod=a.tripod,
            sharpen=a.sharpen,
        )
        after = _analyze(
            out,
            size=ana,
            fps=info.fps,
            people=people,
            label="補正後の動きを測っています",
        )
        data["zoom"] = "vid.stab が自動で決める（optzoom=1）"
    else:
        plan = _plan(
            before,
            method=a.method,
            smooth_s=a.smooth,
            margin=a.margin,
            max_zoom=a.max_zoom,
            tripod=a.tripod,
        )
        core.log(
            f"  端を隠す拡大: {plan.zoom:.3f} 倍"
            + (
                f"（必要 {plan.zoom_needed:.3f} 倍。上限のため {plan.uncovered} コマで端に黒が出ます）"
                if plan.uncovered
                else ""
            )
        )
        after = _render(a.src, out, plan, rng=rng, size=size, sharpen=a.sharpen)
        data["params"] = {
            "smooth_sec": a.smooth,
            "margin": a.margin,
            "max_zoom": a.max_zoom,
            "tripod": a.tripod,
        }
        data["zoom"] = round(plan.zoom, 4)
        data["zoom_needed"] = round(plan.zoom_needed, 4)
        data["frames_with_border"] = plan.uncovered
    jb, ja = _jitter(before, k), _jitter(after, k)
    data["shake_before"], data["shake_after"] = jb, ja
    data["shake_reduction_percent"] = _reduction(jb, ja)
    data["note"] = (
        "shake_px は、隣のコマとの動き（px/コマ、書き出しの大きさ）から、前後 0.25 秒でならした"
        "ゆっくりした動きを引いた残り（揺れ）の二乗平均。motion_px は動き全体の二乗平均"
    )
    core.write_json(json_path, data)
    core.log(
        f"揺れ（px/コマ）: 前 {jb['shake_px']:.2f} → 後 {ja['shake_px']:.2f}"
        f"（{data['shake_reduction_percent']:.0f}% 減）／回転 {jb['shake_deg']:.3f} → "
        f"{ja['shake_deg']:.3f} 度/コマ"
    )
    if before.breaks:
        core.log(f"場面の切れ目: {len(before.breaks)} か所 {data['breaks_sec']}")
    times = _times(before.n, info.fps, rng["start"])
    graph = _plot_shake(times, before, after, k, "コマ間の動き（灰: 前、赤: 後）")
    sheet = _check_sheet(
        a.src,
        out,
        rng=rng,
        size=size,
        fps=info.fps,
        n=before.n,
        title=f"手ぶれ補正（{a.method}）: 揺れ {jb['shake_px']:.2f} → {ja['shake_px']:.2f} px/コマ",
        graph=graph,
    )
    core.save_image(check_path, sheet)
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")
    core.log(f"保存: {check_path}")


# ── 22 カメラパスの平滑化 ──────────────────────────────────────────────────


@command(
    22,
    "smooth-path",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=["l1", "gauss"],
        default="l1",
        help="ならし方。l1: 止まる・等速・なめらかな加減速の区間に分ける（三脚やドリーのような動き。既定）"
        " ／ gauss: 前後 --smooth 秒の重み付き平均",
    ),
    arg(
        "--smooth",
        type=float,
        default=0.5,
        help="gauss でならす範囲（前後それぞれの秒。既定 0.5）",
    ),
    arg(
        "--margin",
        type=float,
        default=0.06,
        help="l1 で通り道を動かしてよい幅（画面の幅・高さに対する割合。既定 0.06）",
    ),
    arg(
        "--max-zoom",
        type=float,
        default=1.25,
        help="端の黒を隠す拡大の上限（既定 1.25。超える所は端に黒が出る）",
    ),
    arg(
        "--no-video",
        action="store_true",
        help="グラフと JSON だけを作る（動画を書き出さない）",
    ),
    arg(
        "--sharpen",
        type=float,
        default=0.6,
        help="補正後に明るさを締める強さ（0 で無し。既定 0.6）",
    ),
    _PEOPLE,
)
def smooth_path(a: argparse.Namespace) -> None:
    """カメラの通り道（横・縦・回転・拡大）を推定してならし、グラフ・JSON・揺れを消した動画を書き出す。"""
    info = _video(a.src, "カメラパスの平滑化")
    rng = _rng(a)
    size, ana = _sizes(info, a.max_side)
    out = Path(a.out) if a.out else core.default_out(a.src, "smooth-path", ".mp4")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    graph_path = out.with_name(out.stem + "-path.png")
    kx, ky = size[0] / ana[0], size[1] / ana[1]
    motion = _analyze(a.src, **rng, size=ana, fps=info.fps, people=not a.with_people)
    plan = _plan(
        motion,
        method=a.method,
        smooth_s=a.smooth,
        margin=a.margin,
        max_zoom=a.max_zoom,
    )
    times = _times(motion.n, info.fps, rng["start"])
    cam_raw = _camera_view(plan.raw, kx, ky)
    cam_smooth = _camera_view(plan.smooth, kx, ky)
    segs = motion.segments()
    title = (
        f"カメラの通り道（{'L1 最適化' if a.method == 'l1' else f'ガウス ±{a.smooth} 秒'}）"
        f"　端を隠す拡大 {plan.zoom:.3f} 倍"
    )
    core.save_image(graph_path, _plot_paths(times, cam_raw, cam_smooth, segs, title))
    corr = cam_smooth - cam_raw
    data: dict[str, Any] = {
        "source": str(a.src),
        "method": a.method,
        "fps": info.fps,
        "size": size,
        "start": rng["start"],
        "frames": motion.n,
        "breaks_sec": [core.t_of(i, info.fps, rng["start"]) for i in motion.breaks],
        "params": {
            "smooth_sec": a.smooth,
            "margin": a.margin,
            "max_zoom": a.max_zoom,
        },
        "zoom": round(plan.zoom, 4),
        "zoom_needed": round(plan.zoom_needed, 4),
        "frames_with_border": plan.uncovered,
        "columns": {
            "x_px": "カメラの横の位置（右へ動くと増える。場面の先頭のコマが 0。書き出しの大きさの px）",
            "y_px": "縦の位置（下へ動くと増える）",
            "angle_deg": "回転（時計回りで増える）",
            "zoom_percent": "拡大（寄ると増える。場面の先頭が 100）",
        },
        "correction_rms": {
            "x_px": round(float(np.sqrt(np.mean(corr[:, 0] ** 2))), 3),
            "y_px": round(float(np.sqrt(np.mean(corr[:, 1] ** 2))), 3),
            "angle_deg": round(float(np.sqrt(np.mean(corr[:, 2] ** 2))), 4),
            "zoom_percent": round(float(np.sqrt(np.mean(corr[:, 3] ** 2))), 4),
        },
        "path": [
            {
                "t": round(float(times[i]), 4),
                "raw": [round(float(v), 3) for v in cam_raw[i]],
                "smooth": [round(float(v), 3) for v in cam_smooth[i]],
            }
            for i in range(motion.n)
        ],
    }
    core.log(
        f"補正の大きさ（二乗平均）: 横 {data['correction_rms']['x_px']:.2f} px・縦 "
        f"{data['correction_rms']['y_px']:.2f} px・回転 {data['correction_rms']['angle_deg']:.3f} 度"
        f"／端を隠す拡大 {plan.zoom:.3f} 倍"
    )
    if not a.no_video:
        after = _render(a.src, out, plan, rng=rng, size=size, sharpen=a.sharpen)
        jb, ja = _jitter(motion, kx), _jitter(after, kx)
        data["shake_before"], data["shake_after"] = jb, ja
        data["shake_reduction_percent"] = _reduction(jb, ja)
        core.log(
            f"揺れ（px/コマ）: 前 {jb['shake_px']:.2f} → 後 {ja['shake_px']:.2f}"
            f"（{data['shake_reduction_percent']:.0f}% 減）"
        )
        core.log(f"保存: {out}")
    core.write_json(json_path, data)
    core.log(f"保存: {graph_path}")
    core.log(f"保存: {json_path}")


# ── 11 ローリングシャッター補正 ────────────────────────────────────────────

_SCANS = {
    "down": "上の行から下の行へ（横長に構えたカメラ・スマートフォンの横動画）",
    "up": "下から上へ",
    "right": "左の列から右の列へ（横長のセンサーを縦に構えた縦動画で多い）",
    "left": "右から左へ",
}


def _velocity(motion: _Motion) -> list[np.ndarray | None]:
    """各コマの撮影時点の、画面の中身の速さ（2×3 = 1 コマあたりの動き − 単位行列）。前後のコマとの動きの平均。"""
    eye = np.eye(3)[:2]
    out: list[np.ndarray | None] = []
    for i in range(motion.n):
        gs = []
        if motion.steps[i] is not None:
            gs.append(motion.steps[i][:2] - eye)
        if i + 1 < motion.n and motion.steps[i + 1] is not None:
            gs.append(motion.steps[i + 1][:2] - eye)
        out.append(np.mean(gs, axis=0) if gs else None)
    return out


def _estimate_readout(motion: _Motion) -> dict[str, Any]:
    """読み出し時間を、コマごとのゆがみ（せん断）の変化と、動きの速さの変化の関係から推定する。

    ローリングシャッターでは、速さが v から v' に変わると、隣のコマとの動きに (v' − v) × 読み出し時間 / 高さ
    のせん断が加わる。これを全コマで集め、最小二乗で読み出し時間（と向き。上下か左右か、その逆か）を求める。
    速さの変化は隣の組の動きの差 (d[i+1] − d[i−1]) / 2 で求めるので、速い揺れ（1 コマの位相 ω）では
    ω / sin ω 倍に大きく出る。補正で使う速さ（前後の組の平均）は逆に sin ω / ω 倍に小さく出るので、掛けた
    補正の量はつり合う（推定値は実際の読み出し時間ではなく、この補正と組の「実効の値」）。
    """
    w, h = motion.size
    acc, shear = {"down": [], "right": []}, {"down": [], "right": []}
    for i in range(1, motion.n - 1):
        a, s0, s1 = motion.affine[i], motion.steps[i - 1], motion.steps[i + 1]
        if a is None or s0 is None or s1 is None or motion.steps[i] is None:
            continue
        ax = (s1[0, 2] - s0[0, 2]) / 2
        ay = (s1[1, 2] - s0[1, 2]) / 2
        # 上下に読む: 横の加速 → 横のせん断（A01 + A10）、縦の加速 → 縦の伸び縮み（A11 − A00）
        acc["down"] += [ax / h, ay / h]
        shear["down"] += [a[0, 1] + a[1, 0], a[1, 1] - a[0, 0]]
        # 左右に読む: 横の加速 → 横の伸び縮み（A00 − A11）、縦の加速 → 縦のせん断（A10 + A01）
        acc["right"] += [ax / w, ay / w]
        shear["right"] += [a[0, 0] - a[1, 1], a[1, 0] + a[0, 1]]
    result: dict[str, Any] = {"pairs": len(acc["down"]) // 2}
    best = None
    for scan in ("down", "right"):
        x, y = np.array(acc[scan]), np.array(shear[scan])
        if len(x) < 10 or float(np.sum(x * x)) <= 0:
            continue
        # 外れ値に強くするため、残差の大きい点を除いて 2 回解く
        r = float(np.sum(x * y) / np.sum(x * x))
        for _ in range(2):
            res = np.abs(y - r * x)
            keep = res <= max(3 * np.median(res), 1e-9)
            r = float(np.sum(x[keep] * y[keep]) / max(np.sum(x[keep] ** 2), 1e-12))
        sst = float(np.sum(y[keep] ** 2))
        r2 = 1 - float(np.sum((y[keep] - r * x[keep]) ** 2)) / sst if sst > 0 else 0.0
        # 加速の大きさ（px/コマ²、測った大きさ）: 小さいと推定できない
        acc_px = float(np.sqrt(np.mean((x * (h if scan == "down" else w)) ** 2)))
        name = scan if r >= 0 else {"down": "up", "right": "left"}[scan]
        result[name] = {"readout": round(abs(r), 4), "r2": round(r2, 3)}
        if best is None or r2 > best[2]:
            best = (name, abs(r), r2, acc_px)
    if best is None:
        result.update(ok=False, reason="動きが測れたコマが足りません")
        return result
    name, r, r2, acc_px = best
    result.update(
        scan=name, readout=round(r, 4), r2=round(r2, 3), accel_px=round(acc_px, 3)
    )
    if acc_px < 0.5:
        result.update(ok=False, reason="揺れ（速さの変化）が小さすぎて推定できません")
    elif r2 < 0.3:
        result.update(ok=False, reason="ゆがみと速さの変化の関係がはっきりしません")
    elif r > 1.5:
        result.update(
            ok=False,
            reason=f"推定値 {r:.2f} が大きすぎます（1 コマの時間を大きく超える）",
        )
    else:
        result.update(ok=True)
    return result


def _rs_maps(
    vel: np.ndarray, readout: float, scan: str, w: int, h: int, grid: tuple
) -> tuple[np.ndarray, np.ndarray, float]:
    """ローリングシャッターを戻す読み取り位置（cv2.remap 用）と、最大のずれ（px）。

    行（列）ごとの撮影時刻のずれ τ（1 コマの時間に対する割合。画面の中央の行が 0）だけ中身は vel × τ 動いて
    写っているので、その位置から読み取る。縦方向は τ 自体が読む行で変わるので、3 回くり返して合わせる。
    """
    qx, qy = grid
    dx = vel[0, 0] * qx + vel[0, 1] * qy + vel[0, 2]
    dy = vel[1, 0] * qx + vel[1, 1] * qy + vel[1, 2]
    px, py = qx, qy
    for _ in range(3):
        if scan in ("down", "up"):
            tau = readout * py / h * (1 if scan == "down" else -1)
        else:
            tau = readout * px / w * (1 if scan == "right" else -1)
        px, py = qx + tau * dx, qy + tau * dy
    cx, cy = _center(w, h)
    shift = float(np.max(np.hypot(px - qx, py - qy)))
    return (px + cx).astype(np.float32), (py + cy).astype(np.float32), shift


def _line_skew(gray: np.ndarray, scan: str) -> tuple[float, int]:
    """画像の、縦（左右に読む時は横）に近い線分の傾きの中央値（tan）と、使った線分の数。"""
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]
    if lines is None:
        return 0.0, 0
    seg = lines.reshape(-1, 4).astype(np.float64)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    if scan in ("right", "left"):
        dx, dy = dy, dx  # 横の線を、縦の線と同じように扱う
    sign = np.where(dy < 0, -1.0, 1.0)
    dx, dy = dx * sign, dy * sign
    length = np.hypot(dx, dy)
    tilt = np.arctan2(dx, dy)
    keep = (np.abs(tilt) < math.radians(15)) & (length > 0.04 * max(gray.shape))
    if keep.sum() < 3:
        return 0.0, int(keep.sum())
    t, wgt = np.tan(tilt[keep]), length[keep]
    order = np.argsort(t)
    cum = np.cumsum(wgt[order])
    return float(t[order][np.searchsorted(cum, cum[-1] / 2)]), int(keep.sum())


def _guides(img: np.ndarray, scan: str, n: int = 6) -> np.ndarray:
    """まっすぐか見比べるための補助線（上下に読む時は縦線、左右に読む時は横線）。"""
    out = img.copy()
    h, w = out.shape[:2]
    for k in range(1, n):
        if scan in ("down", "up"):
            x = round(w * k / n)
            cv2.line(out, (x, 0), (x, h - 1), (0, 220, 255), 1, cv2.LINE_AA)
        else:
            y = round(h * k / n)
            cv2.line(out, (0, y), (w - 1, y), (0, 220, 255), 1, cv2.LINE_AA)
    return out


def _rs_image(a: argparse.Namespace, info: core.MediaInfo) -> None:
    """画像 1 枚: 縦の線の傾きから斜めのゆがみ（せん断）を測り、まっすぐに戻す。"""
    img = next(core.read_frames(a.src, max_side=a.max_side))
    h, w = img.shape[:2]
    gray = core.to_gray(img)
    k, used = _line_skew(gray, a.scan)
    if a.readout not in ("auto", "") and a.skew is None:
        core.log(
            "  画像には読み出し時間（--readout）は使えないので、線の傾きから測ります"
        )
    if a.skew is not None:
        k = math.tan(math.radians(a.skew))
    core.require(
        used >= 3 or a.skew is not None,
        "まっすぐな線が見つからないので、ゆがみを測れません（--skew で角度を指定してください）",
    )
    cx, cy = _center(w, h)
    qy, qx = np.mgrid[0:h, 0:w].astype(np.float32)
    if a.scan in ("down", "up"):
        mx, my = qx + k * (qy - cy), qy
    else:
        mx, my = qx, qy + k * (qx - cx)
    res = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    k_after, _ = _line_skew(core.to_gray(res), a.scan)
    out = Path(a.out) if a.out else core.default_out(a.src, "rolling-shutter", ".png")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    core.save_image(out, res)
    deg, deg_after = math.degrees(math.atan(k)), math.degrees(math.atan(k_after))
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "mode": "image",
            "scan": a.scan,
            "skew_deg": round(deg, 3),
            "lines": used,
            "skew_after_deg": round(deg_after, 3),
        },
    )
    check = out.with_name(out.stem + "-check.jpg")
    core.save_image(
        check,
        core.side_by_side(
            _guides(img, a.scan),
            _guides(res, a.scan),
            (f"前（線の傾き {deg:+.2f} 度）", f"後（{deg_after:+.2f} 度）"),
        ),
    )
    core.log(f"線の傾き: {deg:+.2f} 度（{used} 本）→ 補正後 {deg_after:+.2f} 度")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")
    core.log(f"保存: {check}")


@command(
    11,
    "rolling-shutter",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--readout",
        default="auto",
        help="読み出し時間（1 コマの時間に対する割合。目安: スマートフォン 0.3〜0.9。0〜1.5）。"
        "auto（既定）は揺れのある映像から推定し（速い揺れでは実効の値として 1 を超えることがある）、"
        "推定できない時は補正しない",
    ),
    arg(
        "--scan",
        choices=list(_SCANS),
        help="読み出しの向き（省略時: auto なら推定、数値なら down）。"
        + " ／ ".join(f"{k}: {v}" for k, v in _SCANS.items()),
    ),
    arg(
        "--skew",
        type=float,
        help="画像 1 枚の時: 直す傾き（度。省略時は縦の線の傾きから測る）",
    ),
    _PEOPLE,
)
def rolling_shutter(a: argparse.Namespace) -> None:
    """速いパンや揺れで斜め・ぐにゃぐにゃにゆがんだ映像を、行ごとの撮影時刻のずれを戻して直す。"""
    info = core.probe(a.src)
    if info.is_image:
        a.scan = a.scan or "down"
        _rs_image(a, info)
        return
    rng = _rng(a)
    size, ana = _sizes(info, a.max_side)
    w, h = size
    out = Path(a.out) if a.out else core.default_out(a.src, "rolling-shutter", ".mp4")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    motion = _analyze(
        a.src, **rng, size=ana, fps=info.fps, full=True, people=not a.with_people
    )
    est = _estimate_readout(motion)
    if a.readout == "auto":
        if a.scan:
            # 向きを指定された時は、その向きの推定を使う（当てはまりが悪ければ補正しない）
            cand = est.get(a.scan)
            if cand is None or cand["r2"] < 0.3 or cand["readout"] > 1.5:
                est.update(ok=False, reason=f"{a.scan} の向きでは当てはまりません")
            elif est.get("accel_px", 0.0) >= 0.5:
                est.update(ok=True, scan=a.scan, readout=cand["readout"], r2=cand["r2"])
        scan = a.scan or est.get("scan", "down")
        readout = est["readout"] if est.get("ok") else 0.0
        if not est.get("ok"):
            core.log(
                f"  読み出し時間を推定できませんでした（{est.get('reason')}）。"
                "補正しません。--readout 0.5 などで指定できます"
            )
        else:
            core.log(
                f"  推定: 読み出し {readout:.2f} コマ・向き {scan}（当てはまり R² {est['r2']:.2f}）"
            )
    else:
        try:
            readout = float(a.readout)
        except ValueError as e:
            raise core.VfxError(
                f"--readout は 0〜1 の数か auto です: {a.readout}"
            ) from e
        core.require(0 <= readout <= 1.5, "--readout は 0〜1 の数にしてください")
        scan = a.scan or "down"
    vel = _velocity(motion)
    kx, ky = w / ana[0], h / ana[1]
    cx, cy = _center(w, h)
    qy, qx = np.mgrid[0:h, 0:w].astype(np.float32)
    grid = (qx - cx, qy - cy)
    duration = rng["duration"]
    if rng["max_frames"] and not duration:
        duration = rng["max_frames"] / info.fps
    shifts: list[float] = []
    speeds: list[float] = []
    with core.VideoWriter(
        out,
        info.fps,
        audio_from=a.src,
        audio_start=rng["start"],
        audio_duration=duration,
    ) as writer:
        for i, frame in enumerate(core.read_frames(a.src, **rng, size=size)):
            v = vel[i] if i < len(vel) else None
            if v is None or readout <= 0:
                writer.write(frame)
                shifts.append(0.0)
                speeds.append(0.0)
                continue
            vo = (
                _rescale(np.vstack([v, [0, 0, 0]]) + np.eye(3), kx, ky)[:2]
                - np.eye(3)[:2]
            )
            mx, my, shift = _rs_maps(vo, readout, scan, w, h, grid)
            writer.write(
                cv2.remap(
                    frame, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )
            )
            shifts.append(shift)
            speeds.append(float(math.hypot(vo[0, 2], vo[1, 2])))
            if (i + 1) % max(1, motion.n // 5) == 0:
                core.log(f"  書き出し: {i + 1}/{motion.n} コマ")
    data = {
        "source": str(a.src),
        "fps": info.fps,
        "size": size,
        "start": rng["start"],
        "frames": motion.n,
        "readout": readout,
        "scan": scan,
        "readout_given": a.readout,
        "estimate": est,
        "max_shift_px": round(max(shifts), 2),
        "per_frame": [
            {
                "t": core.t_of(i, info.fps, rng["start"]),
                "speed_px": round(speeds[i], 2),
                "shift_px": round(shifts[i], 2),
            }
            for i in range(len(shifts))
        ],
        "note": "speed_px は画面の中心の速さ（px/コマ）、shift_px はその コマで読み取り位置を動かした最大の量",
    }
    core.write_json(json_path, data)
    check = out.with_name(out.stem + "-check.jpg")
    if shifts:
        # 補正の大きいコマを、0.5 秒以上離して 3 つまで並べる（まっすぐか見やすいよう補助線を引く）
        picked: list[int] = []
        for i in sorted(range(len(shifts)), key=lambda i: -shifts[i]):
            if all(abs(i - j) >= 0.5 * info.fps for j in picked):
                picked.append(i)
            if len(picked) == 3:
                break
        cells, labels = [], []
        for i in sorted(picked):
            t = core.t_of(i, info.fps, rng["start"])
            for img in (
                core.frame_at(a.src, t, size=size),
                core.frame_at(out, t - rng["start"]),
            ):
                cells.append(_guides(img, scan))
            labels += [f"前 {t:.2f}秒", f"後（最大 {shifts[i]:.1f}px 動かした）"]
        core.save_image(
            check, core.contact_sheet(cells, labels, cols=2, cell_width=480)
        )
        core.log(f"保存: {check}")
    core.log(f"読み出し {readout:.2f}・向き {scan}・最大のずれ {max(shifts):.1f} px")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")
