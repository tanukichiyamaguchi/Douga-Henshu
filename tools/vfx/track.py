"""動きの解析（追跡・はめ込み）。

担当するテクニック:
    2 モーショントラッキング（track）
    6 平面トラッキング（planar-track）
    7 特徴点追跡（feature-track）
   25 メッシュワープ（mesh-warp）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import core
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/track/（make_*.py で正解つきの試験映像を作り、eval_*.py で測り、review_*.py で確認画像を作る）
NOTES: dict[int, dict[str, str]] = {
    2: {
        "status": "◎",
        "how": "OpenCV の VitTrack（opencv_zoo の object_tracking_vittrack_2023sep.onnx。Apache-2.0）と CSRT（opencv-contrib）を"
        "並べて走らせる hybrid が既定（--tracker vit / csrt で片方だけも可）。2 つの枠が重なる（IoU 0.5 以上）時は平均、食い違う時は"
        "「直前の動きから予想した位置」と「直近 1 秒の枠の大きさ」に近い方を使い、使わなかった方は見失った時はすぐ、食い違いが 1 秒"
        "続いたら使った方の枠でやり直させる。最初の枠は --box か --auto（face = YuNet、それ以外は detectors の COCO の物体検出で"
        "大きい順の --pick 番目）。カット（色の分布の Bhattacharyya 距離と縮めた白黒の画の相関。0.25 秒以内に戻る白いフラッシュは除く）"
        "で追跡を止め、--redetect なら探し直す（同じ場面では直前の位置に近い物、カットの後は大きい順の --pick 番目）。軌跡は前後を見る"
        "ガウスでならし（σ 0.1 秒、端は点対称に延ばす。大きさは 2 倍の σ）、0.5 秒までの見失いは直線でつなぐ。出力: 文字（core.draw_text）"
        "を付けた確認 MP4（音声は残す。--clean で枠と軌跡を消す）、コマごとの JSON（生の枠・ならした中心と大きさ・確からしさ・"
        "使った追跡器。元の映像の px）、HyperFrames 用の .js（window.__vfxTracks[出力の名前] に作品の fps（既定 30）の時刻つきで x・y・w・h・"
        "見えているか v。作品の大きさ（既定 1080×1920）に映像を object-fit: cover で置いた時の px。window.vfxFollow(tl, 要素, データ, "
        "{start, mediaStart, anchor, dx, dy}) が作品のコマごとに tl.set を並べる）",
        "measured": "正解つきの合成映像（starry_night の上を魚（最初の枠 112×93px）がリサージュで動く。1 コマの動き 中央値 11px・最大 17px、"
        "大きさ ±25%・傾き ±12°・"
        "背景もパン。540×960・180 コマ）: hybrid の中心の誤差 平均 4.80px（最大 12.4）・IoU 0.848（vit 6.07px・0.796、csrt 5.34px・"
        "0.833）。がたつき（誤差の 2 階差分の RMS）は生 3.38px → ならし後 0.77px（ならし後の平均誤差は 5.53px。速い曲がりを少し丸める）。"
        "柱（幅 60px）が魚の 60〜73% を 2 回隠す合成映像: hybrid 8.01px（最大 29.7）・IoU 0.727・見失い 0、csrt 7.74px・0.699、vit 163px・"
        "0.144（隠れると枠が見えている所まで縮み、その後に画面いっぱいへ広がって戻らない）。顔（毎コマの YuNet の枠と比べる）: "
        "plate-stand 9.5px（顔の幅の 5.7%）・IoU 0.81、plate-laugh 5.7px（3.3%）・0.91、story 5.8〜8.8 秒 8.6px（5.2%）・0.85"
        "（vit は 8.9・8.1・6.0px、csrt は 10.4・6.8・12.2px）。vtest の人（--auto person、10 秒）: 前を別の人が横切る前（5.5 秒まで、"
        "検出をつないだ枠と比べて）4.3px・IoU 0.71、横切った後も同じ人を追った（目で確認。検出をつないだ方が別の人に移った）。カット: "
        "story を 5.8 秒から追うと本当のカットのコマ（元のコマ 213 = 8.875 秒。測った時の表示は 8.84 秒で、--start がコマの境目でない分"
        " 1/3 コマ早く出ていた。2026-11-15 に core.t_of を直し、今は最初のコマの本当の時刻から数える）で止まり、--redetect では"
        " 0.25 秒後から次の場面の顔を追った。HyperFrames 0.9.43 の作品（plate-laugh を data-start 0.5・data-media-start 0.2 で全面に置き、vfxFollow で"
        "札を付ける）で check 合格、snapshot の 0.6・0.85・1.0・1.4・1.75 秒で札の位置は .js の値と 1.0〜1.9px 差、その画で YuNet が"
        "見つけた顔の上端の中央と横に 0.2〜15px（作品の px）。素材の 0.6 秒からの追跡（--start 0.6）を付けた 2 つ目の札は、作品の 0.9 秒まで"
        "隠れ、その後は .js の値と 0.7〜0.9px 差。速さ: 540×960・180 コマで hybrid 7.1 秒・vit 3.9 秒・csrt 6.3 秒（読み込み 2 回と"
        "書き出し込み）",
        "limits": "hybrid は vit の約 2 倍の時間。ならし（--smooth 0.1 秒）は速く曲がる動きを少し丸める（合成映像で平均 +0.7px。0 で無効）。"
        "大きく長い隠れでは見失い、0.5 秒（--max-gap）より長いと文字を消す。--redetect は --auto の時だけ。人が交差すると別の人に"
        "移ることがある（vtest の交差 1 回では移らなかった）。カットの見分けは強い照明の変化で誤るかもしれない（試していない）。"
        ".js は映像を作品の画面いっぱいに cover で置く前提（違う置き方なら JSON の元の映像の px から換算する）",
        "tested": "make_track_synth.py の合成映像 2 本（遮り無し・柱で遮る）を eval_track_synth.py で正解と比べた。plate-stand・"
        "plate-laugh・story540 の顔と vtest の人を eval_track_real.py で検出と比べ、確認画像で交差・カットを目で見た。"
        "hf_check.py で HyperFrames の作品の snapshot から札の位置を測った",
    },
    6: {
        "status": "◎",
        "how": "--ref（平面を正面から写した写真）: SIFT（4000 点）＋FLANN（比 0.75）＋USAC_MAGSAC のホモグラフィーで最初のコマ（と見失った"
        "時・--every コマごと）に面を探す。--corners: 最初のコマの四隅を渡す。追跡は面の中の Shi-Tomasi の点（最大 600）を手本に、毎コマ "
        "(1) 予想: 前のコマから LK（往復の確かめつき）＋RANSAC、(2) 補正 2 回: 予想で今のコマを手本の見え方に戻し、面の中の明るさの平均と"
        "散らばりを合わせてから手本の点を LK で照らし RANSAC（1px）で残りのずれを直す。最初のコマと照らせる点が 10% を下回ったら"
        "（寄り・向きが大きく変わった時）0.5 秒ごとに取り替える「近ごろのコマ」の手本で補正する。はめ込み: 画像か動画（短ければ繰り返す。"
        "透過つき PNG・WebM 可）を四隅に透視変換し、縁を内側に --feather px ぼかす。JSON: コマごとの四隅（元の映像の px）とホモグラフィー"
        "（写真の px か最初のコマの px → そのコマ）。画像 1 枚にも使える（--ref で写真の面を探してはめ込む）。外部のモデルは使わない",
        "measured": "正解つきの合成映像（graf1 の板 400×320 を針穴カメラで首振り ±30°・上下 ±15°・傾き ±8°・距離 900〜1500 で動かし、背景も"
        "パン。540×960・150 コマ）: 四隅の誤差 --corners 平均 0.06px（最大 0.16）、--ref（graf1.png 800×640）0.22px（最大 0.36）。"
        "幅 70px の棒が横切り明るさ 0.75〜1.2 倍・ぼけ σ0.8 の映像: --corners 0.18px（最大 1.44。隠れている間 0.40）、--ref 0.32px"
        "（最大 2.23）。--max-side 480（270×480 で処理）でも JSON は元の px で 0.45px・0.65px（座標の換算を確認）。実際の映像（AI 生成の"
        "広告の洗面所、鏡の下の大理石の壁。カメラが上を向きつつ 1.1 倍に寄る 62 コマ）: 62 コマすべて追えて、最初のコマの形に戻した時の"
        "面の中の残りの動き（DIS のフロー）平均 0.79px（最大 2.89）、追わない時は 29.1px。box_in_scene.png で box.png を探すと、合った"
        "特徴 74、四隅は別の方法（AKAZE＋RANSAC）の答えと平均 0.7px（最大 0.93）差。透過 PNG と動画のはめ込みも目で確認。"
        "速さ: 540×960・150 コマ（はめ込みつき）で 11〜13 秒、270×480 で処理すると 5.3 秒（他の作業と CPU を分け合った状態）",
        "limits": "手前の物の隠れは考えない（はめ込んだ物が手前の物の上に出る。box_in_scene では手前の箱の角に掛かった。切り抜き 51〜75 と"
        "組み合わせる）。無地の面は追えない（点が 8 未満ならエラー）。曲がった面は 25（mesh-warp）を使う。写真（--ref）と映像の"
        "見え方（照明・向き）が大きく違うと SIFT で見つからないことがある。最初のコマと照らせなくなって近ごろの手本に切り替えると、"
        "わずかなずれが残る。はめ込む物の明るさは場面に合わせない。面の大半が画面の外に出ると見失う",
        "tested": "make_planar_synth.py の合成映像 2 本（そのまま・棒と明るさとぼけ）を eval_planar.py で正解と比べた（--corners・--ref・"
        "縮めて処理）。eval_planar_real.py で story540 の洗面所の場面の残りの動きを測った。check_box_akaze.py で box_in_scene の四隅を"
        "別の方法と比べた。test_planar_inserts.py で透過 PNG と動画のはめ込みを確かめた",
    },
    7: {
        "status": "◎",
        "how": "Shi-Tomasi の角（goodFeaturesToTrack。強さ 0.01・間隔 8px・最大 300 点。サブピクセルまで詰める）を、ai.track_points"
        "（ピラミッド LK、21×21・3 段、往復のずれ 1px 未満だけ残す）で前のコマから追う。点が上限の 70% を下回ったら、今の点から離れた所に"
        "足す（--refill）。カット（02 と同じ見分け方）で今の点を終え、新しい場面で探し直す（白いフラッシュと見分けるため 0.25 秒待つ）。"
        "出力: 色分けした軌跡（0.6 秒）の確認 MP4、点ごとの JSON（[秒, x, y] の並び。元の映像の px）。画像 1 枚なら見つけた角だけ",
        "measured": "正解つきの合成映像（空撮の写真 aero1 を回転 ±20°・拡大 0.8〜1.2・移動 ±40/±60px で動かす。1 コマの動き 中央値 4.9px・"
        "最大 15px。540×960・120 コマ）: "
        "追い始めた所を正解の変換で移した位置との差 平均 1.18px・95% 2.78px・最大 5.7px。最初の点が 1 秒後に 75%・3 秒後に 62% 残る。"
        "1 コマごとの誤差は平均 0.08px で、誤差は最初のコマからの回転が大きい時に増え（20° で中央値 約 1.7px）、回転が戻ると減る"
        "（たまっていくずれではない）。2.5 倍速・ノイズ σ4・ぼけ: 平均 1.45px・95% 3.13px。vtest（固定カメラ・10 秒）: 5 秒以上追えた"
        "背景の点 214 本の位置のぶれ（標準偏差）中央値 0.03px・95% 0.15px、歩く人の点も 14 本が 5 秒以上続いた。story540（カット 4 つ）:"
        " カットをまたいだ点 0 本、どのカットも 0.29 秒後に新しい点。速さ: 540×960・120 コマで 3.0 秒、story 361 コマで 8.8 秒",
        "limits": "窓（21×21）の平行移動だけを見るので、最初のコマから回転・拡大すると点が少し滑る（1° あたり約 0.08px）。模様の無い所には"
        "点が出ない。カットの後 約 0.25 秒は点が無い。速い動きやぼけでは点が早く消える（1 コマの動き 中央値 12px・最大 38px の映像で"
        "3 秒後に 57%）",
        "tested": "make_feature_synth.py の合成映像 2 本を eval_feature.py で正解と比べた。eval_feature_real.py で vtest の背景の点のぶれと "
        "story540 のカットでの扱いを測り、確認画像で軌跡を目で見た。画像（home.jpg）でも角だけ出ることを確かめた",
    },
    25: {
        "status": "◎",
        "how": "grid（既定）: --box（省略時は画面全体）を升目（既定 約 32px）に分けた網目を、1 升ずつ外まで広げて張る。網目の中の Shi-Tomasi の点"
        "（最大 2000）を手本に、毎コマ (1) 予想: 前のコマから LK で追った点で網目を解く、(2) 補正 2 回: 網目で今のコマを最初のコマの形に戻し"
        "（三角形ごとのアフィン）、明るさをならして手本の点を LK で照らし（往復 1.5px 未満、6px より大きいずれは繰り返し模様の取り違えとして"
        "捨てる）、網目を解き直す。網目は頂点の最小二乗（点の重心座標の式＋網目の線に沿った 2 階差分のなめらかさ（アフィンの動きはゆがめない。"
        "--stiff）＋弱い引き寄せ。Huber の重みで外れた点を弱める）。カットで止める。face: MediaPipe FaceLandmarker（Apache-2.0）の 478 点のうち"
        "網目の 468 点と 852 個の三角形（目と口の中は穴）を使い、最初のコマの点を休みの形にする。貼る位置は --box か --place（頬・額・鼻・"
        "あご）と --size（顔の幅に対する割合）。点は場面ごとに時間方向にならす（σ 0.04 秒、カットで区切る）。奥の三角形から描き、向こうを"
        "向いた三角形・最初のコマで横向きだった三角形（向きの cos 0.2 未満）・そのコマで面積が全体の 3 倍を超えて伸びた三角形は描かない。"
        "貼り方: 今の網目を三角形の番号の画に塗り、三角形ごとの逆のアフィンで貼る物を引く（cv2.remap）。--blend multiply で肌の陰影を残す。"
        "出力: MP4（音声は残す）と JSON（grid は毎コマの頂点、face は毎コマの 468 点と貼った物の四隅・中心）。face は画像 1 枚にも貼れる",
        "measured": "grid: 正弦波でゆがめたチェッカーボード（振幅 0→18→0px・波長 260/300px、全体も ±30/±40px 動く。540×960・120 コマ。"
        "貼る範囲 400×640、網目 12×20 を 14×22 に広げて張る）に starry_night を貼り、範囲の中の点（8px おき）を正解と比べた誤差: 平均 0.21px・"
        "最悪のコマの平均 0.44px・99% 0.97px・最大 2.7px。模様を重ねて明るさが 0.8〜1.15 倍に揺れる映像: 平均 0.25px・99% 0.98px・最大 10.2px"
        "（範囲の右上の角。この角は 120 コマ中 56 コマで 3px を超えた）。最初の版（網目を貼る範囲だけに張り、補正のずれの上限なし）は 0.76px で、角の頂点が繰り返し模様で 1〜2 升"
        "ずれた（最大 88px）。6px を超えるずれを捨て、網目を 1 升外まで広げて上の値になった。face: plate-laugh の顔を既知の動き（回転 ±12°・拡大 ±15%・移動 ±40/±60px）で動かした 90 コマで、シールの中の点の誤差 "
        "平均 0.78px（最悪のコマ 1.60）、時間方向にならさないと 0.81px（2.47）。シールの中心の揺れ（2 階差分）は 0.25px（ならし無し 0.86）、"
        "実際に笑って動く plate-laugh では 0.44px（ならし無し 2.57）。plate-stand・plate-laugh の頬のハート、story の顔の寄り（8.9〜12.3 秒）の"
        "額の模様（乗算）が顔の動きに沿うことを目で確認。story のカット 5.79・8.88 秒を見分けた。速さ: grid は 540×960・120 コマで 59〜135 秒"
        "（他の作業と CPU を分け合った状態）、face は 45 コマで 2.1 秒・90 コマで 3.5 秒",
        "limits": "grid: 模様の無い所は網目を決められない（周りからなめらかにつなぐだけ）。網目の外周（貼る範囲の外の 1 升）は暴れることがあり、"
        "貼る範囲の角は中ほどよりずれやすい（mesh-b の右上の角で最大 10px）。"
        "手前を横切る物の隠れは考えない。予想から 6px を超えて外れる動きは補正で捨てられ、追いつけないことがある。三角形ごとの"
        "アフィンなので升目より細かいゆがみは表せない（試験の波では、頂点を正解の位置に置いても網目の中で平均 0.83px ずれる。点に合わせて"
        "解いた網目の方が小さかった。升目を細かくすると減る）。速くはない（1 コマ 0.5〜1.1 秒）。face: 貼る物は最初のコマの見え方の上に置くので、最初のコマで斜めに見えていた所（頬の外側・鼻の横など）に"
        "掛かった部分は欠けたり縮んだりする（正面を向いたコマを最初にし、--place で頬の中ほどに置くとよい）。目と口の中は穴なので貼れない。"
        "手や髪の隠れは考えない。横顔など MediaPipe が顔を見つけられないコマは貼らない。顔は 1 つ（最初は大きい顔、その後は近い顔）。"
        "カットの後も、顔の同じ点の番号の所に貼り続ける",
        "tested": "make_mesh_synth.py の合成映像 2 本を eval_mesh.py で正解と比べた（試し: try_mesh*.py で升目の数・硬さ・手本の点の選び方を"
        "比べた）。eval_face_mesh.py で既知の動きの合成映像の誤差と、plate-laugh の揺れを測った。plate-laugh・plate-stand・story540 の顔で"
        "シールと模様を貼り、確認画像で目で見た",
    },
}


# ── 共通の部品 ────────────────────────────────────────────────────────────


def _rng(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": a.start or 0.0,
        "duration": a.duration,
        "max_frames": a.max_frames,
    }


def _proc_size(info: core.MediaInfo, max_side: int | None) -> tuple[int, int, float]:
    """処理する大きさ（偶数）と、元の大きさに対する倍率。"""
    w, h = core.scaled_size(info.width, info.height, None, max_side)
    return w, h, w / info.width


def _gauss_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """1 次元（列ごと）のガウスでならす（前後を見るので遅れが出ない）。

    端は点対称に延ばす（2·端 − 内側）。端の値をそのまま延ばすと、動いている所で端のコマが動きの向きへ引っ張られ、
    顔の点で最初のコマが 1〜2px ずれた（face-rigid の試験で確認）。
    """
    if sigma <= 0 or len(x) < 3:
        return x.astype(np.float64).copy()
    r = max(1, round(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    x2 = x.reshape(len(x), -1).astype(np.float64)
    m = min(r, len(x2) - 1)
    left = 2 * x2[:1] - x2[1 : m + 1][::-1]
    right = 2 * x2[-1:] - x2[-m - 1 : -1][::-1]
    if m < r:  # 短い区間は足りない分を端の値で埋める
        left = np.concatenate([np.repeat(left[:1], r - m, 0), left])
        right = np.concatenate([right, np.repeat(right[-1:], r - m, 0)])
    pad = np.concatenate([left, x2, right])
    out = np.stack(
        [np.convolve(pad[:, j], k, mode="valid") for j in range(x2.shape[1])], axis=1
    )
    return out.reshape(x.shape)


def _fill_smooth(
    values: np.ndarray, valid: np.ndarray, sigma: float, max_gap: int
) -> tuple[np.ndarray, np.ndarray]:
    """見失ったコマを前後から直線でつなぎ（max_gap コマまで）、区間ごとにならす。

    戻り値: (ならした値, 使える値か)。max_gap より長く見失った所と、最初に見つかる前・最後に見失った後は使えない。
    """
    n = len(values)
    out = np.full(values.shape, np.nan, np.float64)
    usable = np.zeros(n, bool)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out, usable
    # 区間に分ける（長い空白で切る）
    segs: list[tuple[int, int]] = []
    s = idx[0]
    for a, b in itertools.pairwise(idx):
        if b - a - 1 > max_gap:
            segs.append((s, a))
            s = b
    segs.append((s, idx[-1]))
    for s, e in segs:
        sl = slice(s, e + 1)
        v = valid[sl]
        seg = values[sl].astype(np.float64).reshape(e - s + 1, -1)
        xs = np.arange(e - s + 1)
        filled = np.stack(
            [np.interp(xs, xs[v], seg[v, j]) for j in range(seg.shape[1])], axis=1
        )
        out[sl] = _gauss_smooth(filled, sigma).reshape(out[sl].shape)
        usable[sl] = True
    return out, usable


def _cover(src_w: int, src_h: int, comp: tuple[int, int]) -> tuple[float, float, float]:
    """映像を作品の画面いっぱいに置いた時（object-fit: cover）の倍率とずれ。"""
    cw, ch = comp
    s = max(cw / src_w, ch / src_h)
    return s, (cw - src_w * s) / 2, (ch - src_h * s) / 2


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().replace("×", "x").split("x"))
    except ValueError as e:
        raise core.VfxError(f"大きさは 幅x高さ で指定してください: {text}") from e
    return w, h


def _resample(t_src: np.ndarray, values: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """時刻 t_src の値（列ごと）を、時刻 t_new に直線で補う。"""
    v = values.reshape(len(values), -1)
    out = np.stack(
        [np.interp(t_new, t_src, v[:, j]) for j in range(v.shape[1])], axis=1
    )
    return out.reshape((len(t_new), *values.shape[1:]))


def _js_array(values: np.ndarray, digits: int = 1) -> str:
    return (
        "["
        + ",".join(f"{float(v):.{digits}f}".rstrip("0").rstrip(".") for v in values)
        + "]"
    )


# HyperFrames で使う時の手助け（各 .js に入れる）。tl.set を作品のコマごとに並べるだけなので、
# どの時刻に飛んでも同じ絵になる（HyperFrames の決まり: 決まった結果になる処理だけ）。
_HF_FOLLOW = r"""
// 要素 el を物に付けて動かす。tl: 作品の GSAP タイムライン、track: このファイルのデータ。
// opt.start: 映像の data-start（秒）。opt.mediaStart: 映像の data-media-start（素材の何秒目から使うか。既定 0）。
// opt.anchor: "top"（物の上端の中央。既定）・"center"・"bottom"。opt.dx・opt.dy: ずらす量（作品の px）。
// el は CSS で position:absolute; left:0; top:0 にしておく（位置は x・y の移動だけで決める）。
window.vfxFollow = window.vfxFollow || function (tl, el, track, opt) {
  opt = opt || {};
  const start = opt.start || 0, mediaStart = opt.mediaStart || 0;
  const dx = opt.dx || 0, dy = opt.dy || 0, anchor = opt.anchor || "top";
  const s0 = (track.meta && track.meta.sourceStart) || 0;
  // 追跡した範囲が映像の頭より後から始まる時は、それまで隠しておく
  if (start + s0 + track.t[0] - mediaStart > start + 1e-6) tl.set(el, { autoAlpha: 0 }, start);
  for (let i = 0; i < track.t.length; i++) {
    const at = start + s0 + track.t[i] - mediaStart;
    if (at < start - 1e-6) continue;
    const half = anchor === "top" ? -track.h[i] / 2 : anchor === "bottom" ? track.h[i] / 2 : 0;
    tl.set(el, {
      x: track.x[i] + dx, y: track.y[i] + half + dy,
      xPercent: -50, yPercent: anchor === "top" ? -100 : anchor === "bottom" ? 0 : -50,
      autoAlpha: track.v[i],
    }, at);
  }
};
"""


def _write_hf_js(
    path: Path,
    key: str,
    *,
    fps: float,
    comp: tuple[int, int],
    t: np.ndarray,
    cols: dict[str, np.ndarray],
    meta: dict[str, Any],
    usage: str,
    helper: str = _HF_FOLLOW,
) -> Path:
    """HyperFrames の作品に <script src> で読み込む位置の配列（window.__vfxTracks[key]）。"""
    body = [f"  fps: {fps:g}, width: {comp[0]}, height: {comp[1]},"]
    body.append(f"  meta: {json.dumps(meta, ensure_ascii=False)},")
    body.append(f"  t: {_js_array(t, 4)},")
    for name, v in cols.items():
        digits = 0 if name == "v" else 1
        body.append(f"  {name}: {_js_array(v, digits)},")
    text = (
        "// tools/vfx の追跡結果（HyperFrames 用）。\n"
        + "".join(f"// {line}\n" for line in usage.strip().splitlines())
        + "window.__vfxTracks = window.__vfxTracks || {};\n"
        + f"window.__vfxTracks[{json.dumps(key, ensure_ascii=False)}] = {{\n"
        + "\n".join(body)
        + "\n};\n"
        + helper
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _put_label(
    img: np.ndarray,
    text: str,
    box: tuple[float, float, float, float],
    pos: str,
    size: int,
) -> None:
    """枠（中心 x, 中心 y, 幅, 高さ）に付けて文字を描く。"""
    cx, cy, _, h = box
    gap = max(4, size // 4)
    if pos == "above":
        xy, anchor = (cx, cy - h / 2 - gap), "md"
    elif pos == "below":
        xy, anchor = (cx, cy + h / 2 + gap), "ma"
    else:
        xy, anchor = (cx, cy), "mm"
    core.draw_text(
        img,
        text,
        (round(xy[0]), round(xy[1])),
        size=size,
        color=(255, 255, 255),
        bg=(30, 30, 30),
        pad=max(3, size // 6),
        anchor=anchor,
    )


# ── 02 モーショントラッキング ─────────────────────────────────────────────


def _make_tracker(kind: str, fps: float = 30.0) -> Any:
    if kind == "vit":
        from . import models

        p = cv2.TrackerVit.Params()
        p.net = str(models.fetch("vittrack"))
        return cv2.TrackerVit.create(p)
    if kind == "csrt":
        return cv2.TrackerCSRT.create()
    if kind == "hybrid":
        return _Hybrid(fps)
    raise core.VfxError(f"知らない追跡の方法です: {kind}（hybrid / vit / csrt）")


def _ibox(b: tuple[float, ...]) -> tuple[int, int, int, int]:
    return tuple(round(v) for v in b[:4])  # type: ignore[return-value]


class _Hybrid:
    """VitTrack と CSRT を並べて走らせ、つじつまの合う方を使う。

    VitTrack は形の変化に強いが、物の一部が隠れると枠が見えている所まで縮み、その後に画面いっぱいまで
    広がって戻らないことがある（合成映像で確認）。CSRT は隠れても枠の大きさを保つが、形が変わるとずれやすい。
    2 つが重なる（IoU 0.5 以上）時は 2 つの平均を使う（独立した 2 つの推定を均すと誤差が減る）。食い違う時は、
    「直前までの動きから予想した位置」と「直近 1 秒の枠の大きさ（中央値）」に近い方を使う。
    使われなかった方は、見失った時はすぐに、食い違いが 1 秒続いた時は、使った方の枠でやり直させる
    （すぐにやり直させると、人が交差する場面で 2 つとも背景に移ることがあった）。
    """

    def __init__(self, fps: float = 30.0) -> None:
        self.vit = _make_tracker("vit")
        self.csrt = _make_tracker("csrt")
        self.hist: list[
            tuple[float, float, float, float]
        ] = []  # 使った枠（中心 x, 中心 y, 幅, 高さ）
        self.keep = max(5, round(fps))
        self.patience = max(5, round(fps))
        self.bad = {"vit": 0, "csrt": 0}
        self.csrt_alive = True
        self.score = 1.0
        self.used = "both"

    def init(self, frame: np.ndarray, box: tuple[int, ...]) -> None:
        self.vit.init(frame, box)
        self.csrt.init(frame, box)
        self.csrt_alive = True
        self.bad = {"vit": 0, "csrt": 0}
        self.hist = [(box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3])]

    def getTrackingScore(self) -> float:
        return self.score

    def _cost(self, b: tuple[float, ...]) -> float:
        """予想との食い違い（大きいほど怪しい）。位置は枠の大きさで割り、大きさは比の対数で測る。"""
        h = np.array(self.hist[-self.keep :])
        cx, cy = h[-1, :2]
        vx, vy = (h[-1, :2] - h[-2, :2]) if len(h) >= 2 else (0.0, 0.0)
        ref_w, ref_h = float(np.median(h[:, 2])), float(np.median(h[:, 3]))
        bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
        d = np.hypot(bx - cx - vx, by - cy - vy) / max(ref_w, ref_h, 1.0)
        s = abs(np.log(max(b[2], 1) / ref_w)) + abs(np.log(max(b[3], 1) / ref_h))
        return float(d + s)

    def _use(
        self, b: tuple[float, ...], score: float, who: str
    ) -> tuple[bool, tuple[float, ...]]:
        self.score, self.used = score, who
        self.hist = [
            *self.hist[-self.keep :],
            (b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]),
        ]
        return True, b

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[float, ...]]:
        ok_v, bv = self.vit.update(frame)
        sv = float(self.vit.getTrackingScore())
        bv = tuple(float(v) for v in bv)
        ok_v = bool(ok_v) and bv[2] > 1 and bv[3] > 1
        ok_c, bc = False, bv
        if self.csrt_alive:
            ok_c, bc = self.csrt.update(frame)
            bc = tuple(float(v) for v in bc)
            ok_c = bool(ok_c) and bc[2] > 1 and bc[3] > 1
            self.csrt_alive = (
                ok_c  # 見失った後の CSRT の枠は当てにならないので、やり直すまで使わない
            )
        if ok_v and ok_c and _iou_xywh(bv, bc) >= 0.5:
            self.bad = {"vit": 0, "csrt": 0}
            mean = tuple((p + q) / 2 for p, q in zip(bv, bc, strict=True))
            return self._use(mean, sv, "both")
        cands = []
        if ok_v:
            cands.append((self._cost(bv), "vit", bv))
        if ok_c:
            cands.append((self._cost(bc), "csrt", bc))
        if not cands or min(c[0] for c in cands) > 1.5:
            self.score, self.used = sv, "none"
            return False, bv
        _, who, b = min(cands)
        other = "csrt" if who == "vit" else "vit"
        other_ok = ok_c if other == "csrt" else ok_v
        self.bad[who] = 0
        self.bad[other] = self.bad[other] + 1 if other_ok else self.patience
        if self.bad[other] >= self.patience:
            # 使わなかった方を、使った方の枠でやり直させる
            if other == "csrt":
                self.csrt = _make_tracker("csrt")
                self.csrt.init(frame, _ibox(b))
                self.csrt_alive = True
            else:
                self.vit = _make_tracker("vit")
                self.vit.init(frame, _ibox(b))
            self.bad[other] = 0
        return self._use(b, sv if who == "vit" else min(sv, 0.5), who)


class _CutWatch:
    """隣り合うコマの違いからカット（場面の切り替わり）を見分ける。

    色の分布（HSV）の Bhattacharyya 距離が 0.35 より大きく、かつ 32×32 に縮めた白黒の画の相関が 0.6 より
    小さい時を「大きな変化」とする（試験素材で、カットは距離 0.52 以上・相関 0.32 以下、ふつうの動きは
    距離 0.09 以下・相関 0.88 以上だった）。0.25 秒以内に元の画に戻ったら白いフラッシュなどとして続ける。
    戻り値: "same"（続き）・"pending"（大きな変化の直後で様子見。このコマは追わない）・"cut"（カット）。
    """

    def __init__(self, fps: float) -> None:
        self.prev: tuple[np.ndarray, np.ndarray] | None = None
        self.ref: tuple[np.ndarray, np.ndarray] | None = None
        self.wait = 0
        self.window = max(1, round(0.25 * fps))

    @staticmethod
    def _feat(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        small = cv2.resize(frame[:, :, :3], (160, 160), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        g = cv2.resize(core.to_gray(small), (32, 32), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        return hist, (g - g.mean()) / (g.std() + 1e-6)

    @staticmethod
    def _differs(
        a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
    ) -> bool:
        bh = cv2.compareHist(a[0], b[0], cv2.HISTCMP_BHATTACHARYYA)
        corr = float((a[1] * b[1]).mean())
        return bh > 0.35 and corr < 0.6

    def __call__(self, frame: np.ndarray) -> str:
        f = self._feat(frame)
        if self.ref is not None:
            if not self._differs(self.ref, f):
                self.ref, self.prev = None, f
                return "same"
            self.wait += 1
            self.prev = f
            if self.wait > self.window:
                self.ref = None
                return "cut"
            return "pending"
        if self.prev is not None and self._differs(self.prev, f):
            self.ref, self.wait, self.prev = self.prev, 1, f
            return "pending"
        self.prev = f
        return "same"


class _Finder:
    """--auto で物を探す（顔は YuNet、それ以外は COCO の物体検出）。"""

    def __init__(self, what: str) -> None:
        from . import detectors

        self.what = what
        if what in ("face", "顔"):
            self.face = detectors.FaceDetector(score=0.6)
            self.obj = None
        else:
            names = {v: k for k, v in detectors.COCO_JA.items()}
            label = names.get(what, what)
            core.require(
                label in detectors.COCO_JA,
                f"--auto には face（顔）か COCO の物の名前（person・人・car・cup など）を指定してください: {what}",
            )
            self.face = None
            self.obj = detectors.ObjectDetector("accurate", score=0.5, classes=[label])

    def __call__(
        self, frame: np.ndarray
    ) -> list[tuple[float, float, float, float, float]]:
        """見つかった枠 (x, y, 幅, 高さ, 確からしさ) を大きい順に。"""
        if self.face is not None:
            found = [(*f.box, f.score) for f in self.face(frame)]
        else:
            assert self.obj is not None
            found = [(*d.xywh, d.score) for d in self.obj(frame)]
        return sorted(found, key=lambda b: -b[2] * b[3])


def _iou_xywh(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _nearest(
    cands: list[tuple[float, float, float, float, float]],
    last: tuple[float, ...],
) -> tuple[float, float, float, float, float] | None:
    """探し直した候補から、最後に見えていた枠に近い物（位置と大きさ）を選ぶ。似た物が無ければ None。"""
    lx, ly, lw, lh = last[:4]
    best, best_s = None, 0.5
    for c in cands:
        d = np.hypot(c[0] + c[2] / 2 - lx - lw / 2, c[1] + c[3] / 2 - ly - lh / 2)
        near = max(0.0, 1 - d / (1.5 * max(lw, lh) + 1e-6))
        size = min(c[2] * c[3], lw * lh) / max(c[2] * c[3], lw * lh, 1e-6)
        s = 0.6 * near + 0.4 * size + _iou_xywh(c, last)
        if near > 0 and s > best_s:
            best, best_s = c, s
    return best


@command(
    2,
    "track",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg("--box", help="追う物の枠 x,y,幅,高さ（最初のコマで。元の映像の px）"),
    arg(
        "--auto",
        help="最初のコマで物を自動で選ぶ: face（顔）か COCO の物の名前（person・人・car・cup など）。大きい順",
    ),
    arg(
        "--pick",
        type=int,
        default=0,
        help="--auto で見つかった中の何番目を追うか（大きい順、0 から。既定 0）",
    ),
    arg(
        "--tracker",
        choices=["hybrid", "vit", "csrt"],
        default="hybrid",
        help="追跡の方法: hybrid（既定。VitTrack と CSRT を並べ、重なれば平均、食い違えば動きと大きさのつじつまが合う方を使う）・"
        "vit（VitTrack だけ。速いが、一部が隠れると外れやすい）・csrt（OpenCV の CSRT だけ。モデル不要）",
    ),
    arg(
        "--redetect",
        action="store_true",
        help="見失った時とカットの後に --auto の検出で探し直す（同じ場面では直前の位置に近い物、カットの後は大きい順の --pick 番目）",
    ),
    arg("--text", default="", help="物に付けて動かす文字（日本語可）"),
    arg(
        "--text-pos",
        choices=["above", "below", "center"],
        default="above",
        help="文字の位置: above（物の上。既定）・below（下）・center（中央）",
    ),
    arg(
        "--text-size",
        type=int,
        help="文字の大きさ（処理する映像の px。省略時は高さの 3.5%%）",
    ),
    arg(
        "--smooth",
        type=float,
        default=0.1,
        help="軌跡をならす強さ（秒。ガウスの σ。0 でならさない。既定 0.1）",
    ),
    arg(
        "--max-gap",
        type=float,
        default=0.5,
        help="見失ってもつないで補う長さ（秒。これより長く見失ったら文字を消す。既定 0.5）",
    ),
    arg("--clean", action="store_true", help="確認用の枠と軌跡を描かない（文字だけ）"),
    arg(
        "--fps",
        type=float,
        default=30,
        help="HyperFrames 用の位置の配列のコマの速さ（作品の data-fps。既定 30）",
    ),
    arg(
        "--comp-size",
        default="1080x1920",
        help="HyperFrames の作品の大きさ（幅x高さ。映像を cover で全面に置く前提。既定 1080x1920）",
    ),
)
def track(a: argparse.Namespace) -> None:
    """1 つの物を追い、文字を付けて動かした確認 MP4・コマごとの JSON・HyperFrames 用の位置の配列（.js）を書く。"""
    info = core.probe(a.src)
    core.require(
        not info.is_image, "動画を指定してください（画像 1 枚では追跡できません）"
    )
    core.require(
        bool(a.box) != bool(a.auto), "--box か --auto のどちらか 1 つを指定してください"
    )
    rng = _rng(a)
    _, ph, k = _proc_size(info, a.max_side)
    frames = core.read_frames(a.src, max_side=a.max_side, **rng)
    first = next(frames)
    finder = _Finder(a.auto) if a.auto else None
    if finder is not None:
        found = finder(first)
        core.require(
            len(found) > a.pick,
            f"最初のコマに {a.auto} が {len(found)} 件しか見つかりません（--pick {a.pick}）。--box で指定してください",
        )
        box0 = tuple(float(v) for v in found[a.pick][:4])
        core.log(
            f"最初のコマで {a.auto} を {len(found)} 件見つけ、{a.pick} 番目を追います"
        )
    else:
        bx, by, bw, bh = core.parse_box(a.box)
        box0 = (bx * k, by * k, bw * k, bh * k)
    core.require(box0[2] >= 4 and box0[3] >= 4, f"枠が小さすぎます: {box0}")
    tracker = _make_tracker(a.tracker, info.fps)
    tracker.init(first, _ibox(box0))
    cuts = _CutWatch(info.fps)
    cuts(first)
    boxes = [box0]
    oks = [True]
    scores = [1.0]
    used = ["init"]
    events: list[dict[str, Any]] = []
    last_good = box0
    lost_run = 0
    alive = True  # カットの後（探し直すまで）は追跡器を使わない
    since = 0  # 大きな変化が始まったコマ
    for i, frame in enumerate(frames, start=1):
        t_i = core.t_of(i, info.fps, rng["start"])
        scene = cuts(frame)
        ok, b, score, who = False, last_good, 0.0, "none"
        if scene == "pending" and (not oks or used[-1] != "pending"):
            since = i
        if scene == "cut":
            alive = False
            t_cut = core.t_of(since, info.fps, rng["start"])
            events.append({"i": since, "t": t_cut, "event": "cut"})
            core.log(
                f"  {t_cut:.2f} 秒でカット（場面の切り替わり）を見つけ、追跡を止めました"
            )
        if alive and scene == "same":
            ok, b = tracker.update(frame)
            b = tuple(float(v) for v in b)
            ok = bool(ok) and b[2] > 1 and b[3] > 1
            score = (
                float(tracker.getTrackingScore()) if a.tracker != "csrt" else float(ok)
            )
            who = tracker.used if isinstance(tracker, _Hybrid) else a.tracker
        if ok:
            last_good, lost_run = b, 0
        elif scene != "pending":
            lost_run += 1
            # 探し直す: 見失った直後・3 コマ目・その後は 10 コマごと。カットの後は新しい場面で --pick 番目を選ぶ
            if (
                a.redetect
                and finder is not None
                and (lost_run in (1, 3) or lost_run % 10 == 0)
            ):
                cands = finder(frame)
                if alive:
                    best = _nearest(cands, last_good)
                else:
                    best = cands[a.pick] if len(cands) > a.pick else None
                if best is not None:
                    tracker = _make_tracker(a.tracker, info.fps)
                    tracker.init(frame, _ibox(best))
                    b, ok, score, who = (
                        tuple(float(v) for v in best[:4]),
                        True,
                        float(best[4]),
                        "redetect",
                    )
                    last_good, lost_run, alive = b, 0, True
                    events.append({"i": i, "t": t_i, "event": "redetect", "box": b})
        boxes.append(b)
        oks.append(ok)
        scores.append(score)
        used.append("pending" if scene == "pending" else who)
    n = len(boxes)
    raw = np.array(boxes, np.float64)  # x, y, w, h（処理する大きさの px）
    valid = np.array(oks)
    cxcywh = np.stack(
        [raw[:, 0] + raw[:, 2] / 2, raw[:, 1] + raw[:, 3] / 2, raw[:, 2], raw[:, 3]],
        axis=1,
    )
    sigma = a.smooth * info.fps
    max_gap = round(a.max_gap * info.fps)
    pos, usable_p = _fill_smooth(cxcywh[:, :2], valid, sigma, max_gap)
    size, _ = _fill_smooth(cxcywh[:, 2:], valid, 2 * sigma, max_gap)
    smooth = np.concatenate([pos, size], axis=1)
    usable = usable_p
    lost = int((~valid).sum())
    core.log(
        f"{n} コマを追跡（{a.tracker}）。見失ったコマ {lost}（{lost / n:.0%}）、"
        f"文字を出せないコマ {int((~usable).sum())}"
    )
    # 確認用の MP4
    out = Path(a.out) if a.out else core.default_out(a.src, "track", ".mp4")
    text_size = a.text_size or max(14, round(ph * 0.035))
    trail: list[tuple[int, int]] = []
    writer = core.VideoWriter(
        out,
        info.fps,
        audio_from=a.src,
        audio_start=rng["start"],
        audio_duration=n / info.fps,
    )
    # コマを持ち続けないよう、描く時にもう一度読む（長い映像でもメモリを使いすぎない）
    again = core.read_frames(a.src, max_side=a.max_side, **{**rng, "max_frames": n})
    with writer:
        for i, img in enumerate(again):
            if i >= n:
                break
            if usable[i]:
                cx, cy, w, h = smooth[i]
                if not a.clean:
                    trail.append((round(cx), round(cy)))
                    trail = trail[-int(info.fps) :]
                    for p, q in itertools.pairwise(trail):
                        cv2.line(img, p, q, (0, 220, 255), 2, cv2.LINE_AA)
                    x, y, bw, bh = raw[i]
                    color = (80, 220, 80) if valid[i] else (60, 60, 230)
                    cv2.rectangle(
                        img,
                        (int(x), int(y)),
                        (int(x + bw), int(y + bh)),
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                if a.text:
                    _put_label(img, a.text, (cx, cy, w, h), a.text_pos, text_size)
            elif not a.clean:
                trail = []
                core.draw_text(
                    img,
                    "見失った",
                    (8, 8),
                    size=max(14, text_size * 2 // 3),
                    bg=(40, 40, 200),
                )
            writer.write(img)
    # コマごとの JSON（元の映像の px）
    inv = 1.0 / k
    rows = []
    for i in range(n):
        row: dict[str, Any] = {
            "i": i,
            "t": core.t_of(i, info.fps, rng["start"]),
            "ok": bool(valid[i]),
            "score": round(scores[i], 4),
            "box": [round(float(v) * inv, 1) for v in raw[i]],
        }
        if len(used) == n:
            row["used"] = used[i]
        if usable[i]:
            row["smooth"] = {
                "cx": round(float(smooth[i, 0]) * inv, 2),
                "cy": round(float(smooth[i, 1]) * inv, 2),
                "w": round(float(smooth[i, 2]) * inv, 2),
                "h": round(float(smooth[i, 3]) * inv, 2),
            }
        rows.append(row)
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": info.fps,
            "size": [info.width, info.height],
            "start": rng["start"],
            "tracker": a.tracker,
            "init_box": [round(v * inv, 1) for v in box0],
            "params": {
                "smooth": a.smooth,
                "max_gap": a.max_gap,
                "redetect": a.redetect,
            },
            "lost_frames": lost,
            "redetects": events,
            "note": "box は追跡の生の枠（x, y, 幅, 高さ）、smooth はならした中心と大きさ。どちらも元の映像の px。t は元の映像の中の秒",
            "frames": rows,
        },
    )
    # HyperFrames 用（作品の fps と大きさに合わせる）
    comp = _parse_size(a.comp_size)
    s, ox, oy = _cover(info.width, info.height, comp)
    t_src = np.arange(n) / info.fps
    dur = (n - 1) / info.fps
    t_new = np.arange(0, dur + 1e-9, 1 / a.fps)
    filled = np.where(usable[:, None], smooth, np.nan)
    # 使えないコマは直前の値で埋めて（見えなくする）、補間で値が跳ばないようにする
    for j in range(4):
        col = filled[:, j]
        good = np.flatnonzero(~np.isnan(col))
        if len(good):
            col[: good[0]] = col[good[0]]
            for gi in range(1, n):
                if np.isnan(col[gi]):
                    col[gi] = col[gi - 1]
    rs = _resample(t_src, filled * inv, t_new)
    vis = usable[np.clip(np.round(t_new * info.fps).astype(int), 0, n - 1)].astype(
        float
    )
    js_path = out.with_suffix(".js")
    key = (
        js_path.stem
    )  # 出力の名前（同じ素材で別の物を追っても、作品の中で名前がぶつからないように）
    usage = f"""
使い方（HyperFrames の作品）: このファイルを作品の assets/ に置き、<script src="assets/{js_path.name}"></script> で読む。
  const tr = window.__vfxTracks["{key}"];
  window.vfxFollow(tl, document.querySelector("#label"), tr, {{ start: 映像の data-start, anchor: "top" }});
t は追跡した範囲の頭からの秒（素材の {rng["start"]:g} 秒目が 0。meta.sourceStart）。x・y は物の中心、w・h は枠の大きさ（作品の px。
映像を {comp[0]}x{comp[1]} の画面いっぱいに object-fit: cover で置いた時の位置）。v は 1 = 見えている、0 = 見失った。
"""
    _write_hf_js(
        js_path,
        key,
        fps=a.fps,
        comp=comp,
        t=t_new,
        cols={
            "x": rs[:, 0] * s + ox,
            "y": rs[:, 1] * s + oy,
            "w": rs[:, 2] * s,
            "h": rs[:, 3] * s,
            "v": vis,
        },
        meta={
            "source": Path(a.src).name,
            "sourceStart": rng["start"],
            "sourceSize": [info.width, info.height],
            "fit": "cover",
        },
        usage=usage,
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")
    core.log(
        f"保存: {js_path}（HyperFrames 用。{a.fps:g}fps・{comp[0]}x{comp[1]}・{len(t_new)} 点）"
    )


# ── 06 平面トラッキング ───────────────────────────────────────────────────


def _poly_mask(shape: tuple[int, int], quad: np.ndarray, shrink: int = 0) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    if shrink > 0:
        mask = cv2.erode(mask, np.ones((2 * shrink + 1, 2 * shrink + 1), np.uint8))
    return mask


def _lk(
    prev: np.ndarray,
    cur: np.ndarray,
    pts: np.ndarray,
    init: np.ndarray | None = None,
    fb_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """ai.track_points と同じ往復の確かめつきの LK（往復のずれ fb_max px 未満を有効）。init を渡すとそこから探す。"""
    from .ai import track_points

    if init is None and fb_max == 1.0:
        return track_points(prev, cur, pts)
    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    flags = 0
    p1 = None
    if init is not None:
        p1 = init.reshape(-1, 1, 2).astype(np.float32).copy()
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, p1, flags=flags, **lk)
    p0r = p0.copy()
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(
        cur, prev, p1, p0r, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk
    )
    fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_max)
    return p1.reshape(-1, 2), ok


def _good_h(h: np.ndarray | None) -> bool:
    return h is not None and np.all(np.isfinite(h)) and abs(np.linalg.det(h)) > 1e-8


class _Tpl:
    """面の手本（あるコマの白黒の画・面の中の特徴点・面の範囲と、最初のコマ → そのコマのホモグラフィー）。"""

    def __init__(
        self, g: np.ndarray, p: np.ndarray, mask: np.ndarray, hk: np.ndarray
    ) -> None:
        self.g, self.p, self.mask, self.hk, self.n0 = g, p, mask, hk, len(p)


class _PlaneTracker:
    """平らな面を追う。面の中の特徴点を手本にし、毎コマ 2 段で求める。

    1. 予想: 前のコマの点を LK で今のコマへ追い、RANSAC でホモグラフィーを求めて前の結果に掛ける。
    2. 補正: 今のコマを予想で手本のコマの見え方に戻し（正面に直し）、手本の点を LK で照らして残りのずれを
       求める（2 回。明るさの違いは面の中の平均と散らばりを合わせてならす）。毎コマ手本と照らすので、
       コマを重ねてもずれがたまらない。
    手本は最初のコマ。寄り・引きや向きが大きく変わって、最初のコマと照らせる点が手本の点の 10% を下回ったら、
    0.5 秒ごとに取り替えている「近ごろのコマ」の手本で補正する（その時のわずかな誤差は残るが、見失わない）。
    H は最初のコマの座標 → 今のコマの座標。
    """

    def __init__(
        self,
        gray0: np.ndarray,
        quad0: np.ndarray,
        fps: float = 30.0,
        max_points: int = 600,
    ) -> None:
        self.h, self.w = gray0.shape[:2]
        self.quad0 = quad0.astype(np.float32)
        self.max_points = max_points
        self.H = np.eye(3)
        self.prev = gray0
        first = self._tpl(gray0, np.eye(3))
        core.require(
            first is not None,
            "面の中に目印になる模様が足りません（無地の面は追えません）。四隅を広げるか、模様のある面を選んでください",
        )
        assert first is not None
        self.first: _Tpl = first
        self.recent: _Tpl | None = None
        self.every = max(2, round(fps / 2))
        self.good = 0
        self.inliers = first.n0
        self.used = "first"
        self.p0 = first.p

    def _tpl(self, gray: np.ndarray, hk: np.ndarray) -> _Tpl | None:
        """gray を手本にする（hk は最初のコマ → このコマ）。面の中に点が足りなければ None。"""
        quad = cv2.perspectiveTransform(self.quad0.reshape(-1, 1, 2), hk).reshape(4, 2)
        mask = _poly_mask(gray.shape[:2], quad, shrink=2)
        pts = cv2.goodFeaturesToTrack(
            gray, self.max_points, 0.003, 5, mask=mask, blockSize=7
        )
        if pts is None or len(pts) < 8:
            return None
        return _Tpl(gray, pts.reshape(-1, 2).astype(np.float32), mask, hk)

    def _inside(self, pts: np.ndarray, margin: float = 2.0) -> np.ndarray:
        return (
            (pts[:, 0] >= margin)
            & (pts[:, 1] >= margin)
            & (pts[:, 0] < self.w - margin)
            & (pts[:, 1] < self.h - margin)
        )

    def _correct(
        self, gray: np.ndarray, m: np.ndarray, tpl: _Tpl
    ) -> tuple[np.ndarray | None, int]:
        """今のコマを m（手本 → 今）で手本の見え方に戻し、手本の点を LK で照らして残りのずれを求める。"""
        rect = cv2.warpPerspective(
            gray, m, (self.w, self.h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )
        cover = cv2.warpPerspective(
            np.full(gray.shape[:2], 255, np.uint8),
            m,
            (self.w, self.h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        )
        # 明るさの違いをならす（面の中の平均と散らばりを手本に合わせる。LK は明るさが同じ前提のため）
        both = (tpl.mask > 0) & (cover > 0)
        if both.sum() > 100:
            m0, s0 = float(tpl.g[both].mean()), float(tpl.g[both].std()) + 1e-3
            m1, s1 = float(rect[both].mean()), float(rect[both].std()) + 1e-3
            rect = np.clip(
                (rect.astype(np.float32) - m1) * (s0 / s1) + m0, 0, 255
            ).astype(np.uint8)
        seen = (
            cover[
                np.clip(tpl.p[:, 1].round().astype(int), 0, self.h - 1),
                np.clip(tpl.p[:, 0].round().astype(int), 0, self.w - 1),
            ]
            > 0
        )
        if seen.sum() < 8:
            return None, 0
        q, ok = _lk(tpl.g, rect, tpl.p[seen], fb_max=1.5)
        if ok.sum() < 8:
            return None, 0
        dh, inl = cv2.findHomography(tpl.p[seen][ok], q[ok], cv2.RANSAC, 1.0)
        n = int(inl.sum()) if inl is not None else 0
        if not _good_h(dh) or n < 8:
            return None, n
        return dh, n

    def _refine(
        self, gray: np.ndarray, h: np.ndarray, tpl: _Tpl
    ) -> tuple[np.ndarray, int]:
        """h（最初のコマ → 今）を手本 tpl と照らして 2 回直す。直せなければ h のまま（点の数 0）。"""
        m = h @ np.linalg.inv(tpl.hk)
        n_in = 0
        for _ in range(2):
            dh, n = self._correct(gray, m, tpl)
            if dh is None:
                break
            m, n_in = m @ dh, n
        out = m @ tpl.hk
        return out / out[2, 2], n_in

    def update(self, gray: np.ndarray) -> tuple[np.ndarray, int]:
        """今のコマのホモグラフィー（最初のコマ → 今）と、補正に使えた点の数。"""
        tpl = (
            self.recent
            if self.used == "recent" and self.recent is not None
            else self.first
        )
        m = self.H @ np.linalg.inv(tpl.hk)  # 手本 → 前のコマ
        cur_prev = cv2.perspectiveTransform(tpl.p.reshape(-1, 1, 2), m).reshape(-1, 2)
        use = self._inside(cur_prev)
        h_pred = self.H
        if use.sum() >= 8:
            p1, ok = _lk(self.prev, gray, cur_prev[use])
            if ok.sum() >= 8:
                h_inc, _ = cv2.findHomography(
                    cur_prev[use][ok], p1[ok], cv2.RANSAC, 2.0
                )
                if _good_h(h_inc):
                    h_pred = h_inc @ self.H
        h_new, n_in = self._refine(gray, h_pred, self.first)
        self.used = "first"
        if n_in < max(12, 0.1 * self.first.n0) and self.recent is not None:
            h2, n2 = self._refine(gray, h_pred, self.recent)
            if n2 >= 8:
                h_new, n_in, self.used = h2, n2, "recent"
        if n_in < 8:
            h_new = h_pred
        self.H = h_new
        self.prev = gray
        self.inliers = n_in
        # 近ごろのコマの手本を 0.5 秒ごとに取り替える（よく照らせたコマだけ）
        base = (
            self.recent
            if self.used == "recent" and self.recent is not None
            else self.first
        )
        self.good = (
            self.good + 1
            if n_in >= max(12, 0.3 * base.n0) or self.used == "recent" and n_in >= 12
            else 0
        )
        if self.good >= self.every:
            fresh = self._tpl(gray, self.H)
            if fresh is not None:
                self.recent = fresh
            self.good = 0
        return self.H, n_in


class _RefFinder:
    """平面の写真（--ref）を、SIFT の特徴の照合とホモグラフィーでコマの中に探す。"""

    def __init__(self, ref_gray: np.ndarray) -> None:
        self.sift = cv2.SIFT_create(nfeatures=4000)
        self.ref = ref_gray
        self.kr, self.dr = self.sift.detectAndCompute(ref_gray, None)
        core.require(
            self.dr is not None and len(self.kr) >= 10,
            "平面の写真（--ref）に特徴が足りません。模様のはっきりした写真にしてください",
        )
        self.matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5}, {"checks": 64}
        )

    def find(self, gray: np.ndarray) -> tuple[np.ndarray | None, int]:
        """写真 → コマ のホモグラフィーと、合った特徴の数。見つからなければ (None, 数)。"""
        kf, df = self.sift.detectAndCompute(gray, None)
        if df is None or len(kf) < 10:
            return None, 0
        pairs = self.matcher.knnMatch(self.dr, df, k=2)
        good = [
            p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance
        ]
        if len(good) < 10:
            return None, len(good)
        src = np.float32([self.kr[m.queryIdx].pt for m in good])
        dst = np.float32([kf[m.trainIdx].pt for m in good])
        h, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
        n = int(inl.sum()) if inl is not None else 0
        if not _good_h(h) or n < 10:
            return None, n
        quad = cv2.perspectiveTransform(
            _rect_corners(self.ref.shape[1], self.ref.shape[0]).reshape(-1, 1, 2), h
        )
        if (
            not cv2.isContourConvex(np.round(quad).astype(np.int32))
            or cv2.contourArea(quad) < 100
        ):
            return None, n
        return h, n


def _rect_corners(w: float, h: float) -> np.ndarray:
    """長方形の四隅（左上・右上・右下・左下）。"""
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def _parse_quad(text: str) -> np.ndarray:
    pts = core.parse_points(text.replace(";", " "))
    core.require(
        pts.shape == (4, 2),
        f"四隅は 'x1,y1 x2,y2 x3,y3 x4,y4'（左上・右上・右下・左下）で指定してください: {text}",
    )
    return pts


class _Insert:
    """はめ込む画像か動画（動画は 1 コマずつ進め、短ければ繰り返す）。透過つき PNG・WebM の透過も使う。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.info = core.probe(path)
        self.it: Any = None
        self.img: np.ndarray | None = None
        if self.info.is_image:
            self.img = core.load_image(path, alpha=True)

    def next(self) -> np.ndarray:
        if self.img is not None:
            return self.img
        for _ in range(2):
            if self.it is None:
                self.it = core.read_frames(self.path, alpha=True)
            frame = next(self.it, None)
            if frame is not None:
                return frame
            self.it = None  # 終わったら頭から
        raise core.VfxError(f"はめ込む動画を読めません: {self.path}")


def _composite_quad(
    frame: np.ndarray, ins: np.ndarray, quad: np.ndarray, feather: float, opacity: float
) -> np.ndarray:
    """ins（BGRA）を四角形 quad（左上・右上・右下・左下）に透視変換して重ねる。縁は feather px ぼかす。"""
    h, w = frame.shape[:2]
    ih, iw = ins.shape[:2]
    m = cv2.getPerspectiveTransform(_rect_corners(iw, ih), quad.astype(np.float32))
    color = cv2.warpPerspective(ins[:, :, :3], m, (w, h), flags=cv2.INTER_LINEAR)
    alpha_src = ins[:, :, 3] if ins.shape[2] == 4 else np.full((ih, iw), 255, np.uint8)
    alpha = (
        cv2.warpPerspective(alpha_src, m, (w, h), flags=cv2.INTER_LINEAR).astype(
            np.float32
        )
        / 255
    )
    if feather > 0:
        # 縁だけを内側へぼかす（外へははみ出さない）
        inner = cv2.erode(
            (alpha > 0.5).astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=max(1, int(feather)),
        )
        soft = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), feather)
        alpha = alpha * np.clip(soft * 1.5, 0, 1)
    return core.overlay(frame, color, alpha * opacity)


def _quad_ok(quad: np.ndarray, w: int, h: int) -> bool:
    """四隅が使える形か（凸で、ある程度の大きさがあり、画面から大きく外れていない）。"""
    q = np.round(quad).astype(np.int32)
    area = cv2.contourArea(quad.astype(np.float32))
    inside = (
        (quad[:, 0] > -2 * w)
        & (quad[:, 0] < 3 * w)
        & (quad[:, 1] > -2 * h)
        & (quad[:, 1] < 3 * h)
    )
    return bool(cv2.isContourConvex(q) and area > 64 and inside.all())


def _h_to_src(h: np.ndarray, k: float, ref_mode: bool) -> np.ndarray:
    """処理した大きさのホモグラフィーを元の映像の px に直す（ref の時は写真の px → 元の映像の px）。"""
    s_inv = np.diag([1 / k, 1 / k, 1.0])
    out = s_inv @ h if ref_mode else s_inv @ h @ np.diag([k, k, 1.0])
    return out / out[2, 2]


@command(
    6,
    "planar-track",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--ref",
        help="平面の写真（看板・ポスター・画面などを正面から写した物）。SIFT で照らしてコマの中に探す",
    ),
    arg(
        "--corners",
        help="最初のコマでの面の四隅 'x1,y1 x2,y2 x3,y3 x4,y4'（左上・右上・右下・左下の順。元の映像の px）",
    ),
    arg(
        "--insert",
        help="はめ込む画像か動画（四隅に合わせて引き伸ばす。透過つき PNG・WebM も可）",
    ),
    arg(
        "--feather",
        type=float,
        default=1.5,
        help="はめ込みの縁を内側にぼかす幅（px。既定 1.5）",
    ),
    arg(
        "--opacity", type=float, default=1.0, help="はめ込みの不透明度（0〜1。既定 1）"
    ),
    arg(
        "--outline",
        action="store_true",
        help="追った面の四隅を線で描く（確かめる用。--insert が無い時は常に描く）",
    ),
    arg(
        "--every",
        type=int,
        default=0,
        help="--ref の時、何コマごとに写真と照らし直すか（0 は見失った時だけ。既定 0）",
    ),
)
def planar_track(a: argparse.Namespace) -> None:
    """平らな面（看板・画面・壁など）を追い、画像や動画を透視変換ではめ込む。毎コマのホモグラフィーを JSON に書く。"""
    info = core.probe(a.src)
    core.require(
        bool(a.ref) != bool(a.corners),
        "--ref か --corners のどちらか 1 つを指定してください",
    )
    rng = _rng(a)
    pw, ph, k = _proc_size(info, a.max_side)
    ref_mode = bool(a.ref)
    finder = None
    ref_quad = None
    if ref_mode:
        ref = core.to_gray(core.load_image(a.ref))
        finder = _RefFinder(ref)
        ref_quad = _rect_corners(ref.shape[1], ref.shape[0])
    insert = _Insert(a.insert) if a.insert else None
    suffix = Path(a.src).suffix.lower() if info.is_image else ".mp4"
    out = (
        Path(a.out)
        if a.out
        else core.default_out(
            a.src,
            "planar-track",
            suffix if suffix in (".png", ".jpg", ".mp4") else ".png",
        )
    )
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    outline = a.outline or insert is None

    def draw(frame: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
        img = frame[:, :, :3].copy()
        # はめ込む動画は、面を見失ったコマでも進める（時間がずれないように）
        ins = insert.next() if insert is not None else None
        if quad is None:
            return img
        if ins is not None:
            img = _composite_quad(img, ins, quad, a.feather, a.opacity)
        if outline:
            cv2.polylines(
                img,
                [np.round(quad).astype(np.int32)],
                True,
                (60, 220, 60),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                img, tuple(int(v) for v in quad[0]), 5, (60, 60, 230), -1, cv2.LINE_AA
            )
        return img

    rows: list[dict[str, Any]] = []

    def record(
        i: int, h: np.ndarray | None, quad: np.ndarray | None, n: int, how: str
    ) -> None:
        row: dict[str, Any] = {
            "i": i,
            "t": core.t_of(i, info.fps, rng["start"]) if not info.is_image else 0.0,
        }
        row["ok"] = h is not None
        row["points"] = n
        row["how"] = how
        if h is not None and quad is not None:
            row["corners"] = [
                [round(float(x) / k, 2), round(float(y) / k, 2)] for x, y in quad
            ]
            row["H"] = [
                [round(float(v), 8) for v in r] for r in _h_to_src(h, k, ref_mode)
            ]
        rows.append(row)

    if info.is_image:
        img = core.load_image(a.src)
        if (pw, ph) != (img.shape[1], img.shape[0]):
            img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
        if ref_mode:
            assert finder is not None and ref_quad is not None
            h, n = finder.find(core.to_gray(img))
            core.require(
                h is not None, f"画像の中に平面の写真が見つかりません（合った特徴 {n}）"
            )
            quad = cv2.perspectiveTransform(ref_quad.reshape(-1, 1, 2), h).reshape(4, 2)
        else:
            quad = _parse_quad(a.corners) * k
            h, n = np.eye(3), 0
        record(0, h, quad, n, "ref" if ref_mode else "corners")
        core.save_image(out, draw(img, quad))
        core.write_json(
            json_path,
            {
                "source": str(a.src),
                "mode": "ref" if ref_mode else "corners",
                "frames": rows,
            },
        )
        core.log(f"面の四隅: {rows[0]['corners']}（合った特徴 {n}）")
        core.log(f"保存: {out}")
        core.log(f"保存: {json_path}")
        return

    frames = core.read_frames(a.src, max_side=a.max_side, **rng)
    writer = core.VideoWriter(
        out,
        info.fps,
        audio_from=a.src,
        audio_start=rng["start"],
        audio_duration=a.duration,
    )
    tracker: _PlaneTracker | None = None
    base = np.eye(3)  # 手本のコマの座標 → 面の座標（ref の時は写真 → 手本のコマ）
    quad0 = None if ref_mode else _parse_quad(a.corners) * k
    weak = 0
    n_ok = 0
    n_refind = 0
    with writer:
        for i, frame in enumerate(frames):
            gray = core.to_gray(frame)
            h_full = None
            n = 0
            how = "track"
            if tracker is None:
                if ref_mode:
                    assert finder is not None
                    hr, n = finder.find(gray)
                    if hr is not None:
                        assert ref_quad is not None
                        q = cv2.perspectiveTransform(
                            ref_quad.reshape(-1, 1, 2), hr
                        ).reshape(4, 2)
                        tracker = _PlaneTracker(gray, q, info.fps)
                        base, h_full, how = hr, hr, "ref"
                        n_refind += 1
                else:
                    assert quad0 is not None
                    tracker = _PlaneTracker(gray, quad0, info.fps)
                    base, h_full, how, n = (
                        np.eye(3),
                        np.eye(3),
                        "corners",
                        len(tracker.p0),
                    )
            else:
                ht, n = tracker.update(gray)
                h_full = ht @ base
                weak = weak + 1 if n < 12 else 0
                due = ref_mode and (weak >= 3 or (a.every > 0 and i % a.every == 0))
                if due:
                    assert finder is not None
                    hr, nr = finder.find(gray)
                    if hr is not None:
                        assert ref_quad is not None
                        q = cv2.perspectiveTransform(
                            ref_quad.reshape(-1, 1, 2), hr
                        ).reshape(4, 2)
                        tracker = _PlaneTracker(gray, q, info.fps)
                        base, h_full, n, how, weak = hr, hr, nr, "ref", 0
                        n_refind += 1
                if weak >= 3 and how == "track":
                    h_full = None  # 補正に使える点が続けて足りない時は見失ったとする
            quad = None
            if h_full is not None:
                # 写真の四隅（ref）か最初のコマの四隅（corners）を、今のコマへ移す
                src_quad = ref_quad if ref_mode else quad0
                assert src_quad is not None
                q = cv2.perspectiveTransform(
                    src_quad.reshape(-1, 1, 2), h_full
                ).reshape(4, 2)
                if _quad_ok(q, pw, ph):
                    quad = q
                else:
                    h_full = None
            n_ok += quad is not None
            record(i, h_full, quad, n, how)
            writer.write(draw(frame, quad))
            if (i + 1) % 50 == 0:
                core.log(f"  {i + 1} コマ")
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": info.fps,
            "size": [info.width, info.height],
            "start": rng["start"],
            "mode": "ref" if ref_mode else "corners",
            "ref": str(a.ref) if a.ref else None,
            "note": "corners は面の四隅（左上・右上・右下・左下。元の映像の px）。H は "
            + ("写真（--ref）の px" if ref_mode else "最初のコマの px")
            + " → そのコマの px のホモグラフィー（3×3）。points は最初のコマとの照合に使えた点の数",
            "frames": rows,
        },
    )
    core.log(
        f"{len(rows)} コマ中 {n_ok} コマで面を追えました（写真と照らした回数 {n_refind}）"
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 07 特徴点追跡 ─────────────────────────────────────────────────────────


def _region_mask(
    shape: tuple[int, int], box: str | None, k: float
) -> np.ndarray | None:
    if not box:
        return None
    x, y, w, h = core.parse_box(box)
    mask = np.zeros(shape, np.uint8)
    mask[
        max(0, round(y * k)) : round((y + h) * k),
        max(0, round(x * k)) : round((x + w) * k),
    ] = 255
    return mask


def _detect(
    gray: np.ndarray,
    n: int,
    quality: float,
    min_dist: float,
    mask: np.ndarray | None,
    avoid: np.ndarray | None = None,
) -> np.ndarray:
    """Shi-Tomasi の角（コーナー）を n 個まで探す（avoid の点の近くは除く）。サブピクセルまで詰める。"""
    if n <= 0:
        return np.empty((0, 2), np.float32)
    m = np.full(gray.shape, 255, np.uint8) if mask is None else mask.copy()
    if avoid is not None:
        for x, y in avoid:
            cv2.circle(m, (round(x), round(y)), max(1, int(min_dist)), 0, -1)
    pts = cv2.goodFeaturesToTrack(gray, n, quality, min_dist, mask=m, blockSize=7)
    if pts is None:
        return np.empty((0, 2), np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    pts = cv2.cornerSubPix(gray, pts.astype(np.float32), (5, 5), (-1, -1), crit)
    return pts.reshape(-1, 2)


@command(
    7,
    "feature-track",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--max-points", type=int, default=300, help="同時に追う点の数の上限（既定 300）"
    ),
    arg(
        "--quality",
        type=float,
        default=0.01,
        help="角の強さの下限（いちばん強い角に対する割合。既定 0.01）",
    ),
    arg(
        "--min-distance",
        type=float,
        default=8,
        help="点どうしの最小の間隔（処理する映像の px。既定 8）",
    ),
    arg("--box", help="点を探す範囲 x,y,幅,高さ（元の映像の px。省略時は画面全体）"),
    arg(
        "--refill",
        type=float,
        default=0.7,
        help="点が上限のこの割合を下回ったら新しい点を足す（0 で足さない。既定 0.7）",
    ),
    arg(
        "--trail",
        type=float,
        default=0.6,
        help="確認用の映像に描く軌跡の長さ（秒。既定 0.6）",
    ),
    arg(
        "--min-length",
        type=int,
        default=3,
        help="JSON に書く点の最短のコマ数（既定 3）",
    ),
)
def feature_track(a: argparse.Namespace) -> None:
    """目印になる点（角）を見つけて追い続ける（Shi-Tomasi＋往復で確かめる LK）。軌跡の MP4 と点ごとの JSON を書く。"""
    info = core.probe(a.src)
    rng = _rng(a)
    pw, ph, k = _proc_size(info, a.max_side)
    inv = 1.0 / k
    if info.is_image:
        img = core.load_image(a.src)
        if (pw, ph) != (img.shape[1], img.shape[0]):
            img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
        gray = core.to_gray(img)
        pts = _detect(
            gray,
            a.max_points,
            a.quality,
            a.min_distance,
            _region_mask(gray.shape, a.box, k),
        )
        out = Path(a.out) if a.out else core.default_out(a.src, "feature-track", ".jpg")
        vis = img.copy()
        for x, y in pts:
            cv2.circle(vis, (round(x), round(y)), 3, (0, 220, 255), -1, cv2.LINE_AA)
        core.save_image(out, vis)
        json_path = Path(a.json) if a.json else out.with_suffix(".json")
        core.write_json(
            json_path,
            {
                "source": str(a.src),
                "note": "画像 1 枚なので、見つけた点（角）だけ。元の画像の px",
                "points": [
                    [round(float(x) * inv, 2), round(float(y) * inv, 2)] for x, y in pts
                ],
            },
        )
        core.log(f"点（角）: {len(pts)} 個")
        core.log(f"保存: {out}")
        core.log(f"保存: {json_path}")
        return
    from .ai import track_points

    out = Path(a.out) if a.out else core.default_out(a.src, "feature-track", ".mp4")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    region = _region_mask((ph, pw), a.box, k)
    tracks: dict[int, list[tuple[int, float, float]]] = {}
    alive_ids: list[int] = []
    alive_pts = np.empty((0, 2), np.float32)
    next_id = 0
    prev = None
    cuts = _CutWatch(info.fps)
    trail_n = max(2, round(a.trail * info.fps))
    counts: list[int] = []
    added = 0
    ended_at_cut = 0
    writer = core.VideoWriter(
        out,
        info.fps,
        audio_from=a.src,
        audio_start=rng["start"],
        audio_duration=a.duration,
    )

    def color(tid: int) -> tuple[int, int, int]:
        from .detectors import color_for

        return color_for(tid)

    with writer:
        for i, frame in enumerate(core.read_frames(a.src, max_side=a.max_side, **rng)):
            gray = core.to_gray(frame)
            scene = cuts(frame)
            if scene == "cut":
                # カット: 今の点はここで終える（新しい場面で探し直す）
                ended_at_cut += len(alive_ids)
                alive_ids, alive_pts, prev = [], np.empty((0, 2), np.float32), None
            elif scene == "same" and prev is not None and len(alive_pts):
                p1, ok = track_points(prev, gray, alive_pts)
                ok &= (
                    (p1[:, 0] >= 0)
                    & (p1[:, 1] >= 0)
                    & (p1[:, 0] < pw)
                    & (p1[:, 1] < ph)
                )
                alive_ids = [t for t, o in zip(alive_ids, ok, strict=True) if o]
                alive_pts = p1[ok]
                for t, (x, y) in zip(alive_ids, alive_pts, strict=True):
                    tracks[t].append((i, float(x), float(y)))
            # 最初・カットの後・点が減った時に、今の点から離れた所へ新しい点を足す（様子見のコマでは足さない）
            if scene == "same" and (
                prev is None
                or not alive_ids
                or (a.refill > 0 and len(alive_pts) < a.refill * a.max_points)
            ):
                new = _detect(
                    gray,
                    a.max_points - len(alive_pts),
                    a.quality,
                    a.min_distance,
                    region,
                    alive_pts,
                )
                for x, y in new:
                    tracks[next_id] = [(i, float(x), float(y))]
                    alive_ids.append(next_id)
                    next_id += 1
                if i > 0:
                    added += len(new)
                alive_pts = np.concatenate([alive_pts, new]) if len(alive_pts) else new
            if scene == "same":
                prev = gray  # 白いフラッシュなど様子見のコマは、追う元にしない
            counts.append(len(alive_ids) if scene == "same" else 0)
            vis = frame.copy()
            # 様子見のコマ（カットかフラッシュか分からない間）は点を描かない（前の場面の点が残って見えるため）
            for t in alive_ids if scene == "same" else []:
                tr = tracks[t][-trail_n:]
                c = color(t)
                if len(tr) >= 2:
                    pts = np.array([[x, y] for _, x, y in tr], np.float32)
                    cv2.polylines(
                        vis, [np.round(pts).astype(np.int32)], False, c, 1, cv2.LINE_AA
                    )
                _, x, y = tr[-1]
                cv2.circle(vis, (round(x), round(y)), 2, c, -1, cv2.LINE_AA)
            writer.write(vis)
    rows = []
    lengths = []
    for t, tr in tracks.items():
        if len(tr) < a.min_length:
            continue
        lengths.append(len(tr))
        rows.append(
            {
                "id": t,
                "start": core.t_of(tr[0][0], info.fps, rng["start"]),
                "end": core.t_of(tr[-1][0], info.fps, rng["start"]),
                "points": [
                    [
                        core.t_of(f, info.fps, rng["start"]),
                        round(x * inv, 2),
                        round(y * inv, 2),
                    ]
                    for f, x, y in tr
                ],
            }
        )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": info.fps,
            "size": [info.width, info.height],
            "start": rng["start"],
            "params": {
                "max_points": a.max_points,
                "quality": a.quality,
                "min_distance": a.min_distance,
                "refill": a.refill,
            },
            "note": "points は [秒, x, y] の並び（元の映像の中の秒と px）。白いフラッシュなどで追わなかったコマは抜ける",
            "points_per_frame": counts,
            "tracks": rows,
        },
    )
    n = len(counts)
    core.log(
        f"{n} コマ: 点の数 平均 {np.mean(counts):.0f}（最少 {min(counts)}）、追った点 {len(rows)} 本"
        f"（{a.min_length} コマ以上）、足した点 {added}、カットで終えた点 {ended_at_cut}"
    )
    if lengths:
        core.log(
            f"点の寿命: 中央値 {np.median(lengths) / info.fps:.2f} 秒・最長 {max(lengths) / info.fps:.2f} 秒"
        )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 25 メッシュワープ ─────────────────────────────────────────────────────


def _grid_mesh(
    box: tuple[float, float, float, float], cols: int, rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """box（x, y, 幅, 高さ）を cols×rows の升目に分けた網目の頂点（(cols+1)(rows+1)×2）と三角形（升目ごとに 2 つ）。"""
    x, y, w, h = box
    xs = np.linspace(x, x + w, cols + 1)
    ys = np.linspace(y, y + h, rows + 1)
    verts = np.array([[vx, vy] for vy in ys for vx in xs], np.float64)
    tris = []
    for r in range(rows):
        for c in range(cols):
            i = r * (cols + 1) + c
            tris += [(i, i + 1, i + cols + 2), (i, i + cols + 2, i + cols + 1)]
    return verts, np.array(tris, np.int32)


def _tri_affines(src: np.ndarray, dst: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """三角形ごとに、src の三角形 → dst の三角形 のアフィン変換（M×2×3）。つぶれた三角形は NaN。"""
    a = src[tris]  # M×3×2
    b = dst[tris]
    ones = np.ones((len(tris), 3, 1))
    sa = np.concatenate([a, ones], axis=2)  # M×3×3（行 = 頂点）
    out = np.full((len(tris), 2, 3), np.nan)
    det = np.linalg.det(sa)
    ok = np.abs(det) > 1e-9
    if ok.any():
        # sa @ X = b  →  X（3×2）= 各三角形のアフィンの転置
        x = np.linalg.solve(sa[ok], b[ok])
        out[ok] = np.transpose(x, (0, 2, 1))
    return out


def _raster(
    verts: np.ndarray,
    tris: np.ndarray,
    shape: tuple[int, int],
    order: np.ndarray | None = None,
) -> np.ndarray:
    """三角形の番号を画素に塗った画像（どの三角形にも入らない画素は -1）。order の順に塗る（後の物が上）。"""
    ids = np.full(shape, -1, np.int32)
    pts = np.round(verts * 16).astype(np.int32)  # 1/16 px の精度で塗る
    for k in order if order is not None else range(len(tris)):
        cv2.fillConvexPoly(ids, pts[tris[k]], int(k), cv2.LINE_8, 4)
    return ids


def _pwa_map(
    ids: np.ndarray, aff: np.ndarray, roi: tuple[int, int, int, int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """塗った三角形の番号と三角形ごとのアフィンから、画素ごとの移し先（map_x, map_y）と入っているか（mask）。"""
    h, w = ids.shape
    x0, y0, x1, y1 = roi or (0, 0, w, h)
    sub = ids[y0:y1, x0:x1]
    mask = sub >= 0
    mx = np.full(sub.shape, -1.0, np.float32)
    my = np.full(sub.shape, -1.0, np.float32)
    yy, xx = np.nonzero(mask)
    k = sub[yy, xx]
    a = aff[k]
    gx, gy = xx + x0, yy + y0
    mx[yy, xx] = a[:, 0, 0] * gx + a[:, 0, 1] * gy + a[:, 0, 2]
    my[yy, xx] = a[:, 1, 0] * gx + a[:, 1, 1] * gy + a[:, 1, 2]
    bad = ~np.isfinite(mx) | ~np.isfinite(my)
    mx[bad], my[bad] = -1, -1
    return mx, my, mask & ~bad


def _bbox(
    pts: np.ndarray, w: int, h: int, pad: int = 2
) -> tuple[int, int, int, int] | None:
    x0, y0 = np.floor(pts.min(0)).astype(int) - pad
    x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return (int(x0), int(y0), int(x1), int(y1)) if x1 > x0 and y1 > y0 else None


class _MeshSolver:
    """網目の頂点を、特徴点の対応（残りの頂点はなめらかさ）から最小二乗で求める。

    式: 特徴点 j が休み（最初のコマ）の網目の三角形の中で重心座標 λ を持つ時、Σ λ_k V_k = 今の位置。
    なめらかさ: 網目の線に沿った 2 階差分 V[i-1] − 2V[i] + V[i+1] = 0（アフィンの動きでは 0 なので、移動・回転・
    拡大はゆがめない）。弱い引き寄せ: V = 予想（点の無い所の頂点を決めるため）。外れた対応は Huber の重みで弱める。
    """

    def __init__(
        self,
        rest: np.ndarray,
        tris: np.ndarray,
        cols: int,
        rows: int,
        shape: tuple[int, int],
        smooth: float,
    ) -> None:
        self.rest = rest
        self.tris = tris
        self.n = len(rest)
        self.shape = shape
        lines = []
        for r in range(rows + 1):
            for c in range(1, cols):
                i = r * (cols + 1) + c
                lines.append((i - 1, i, i + 1))
        for r in range(1, rows):
            for c in range(cols + 1):
                i = r * (cols + 1) + c
                lines.append((i - cols - 1, i, i + cols + 1))
        self.lines = np.array(lines, np.int32).reshape(-1, 3)
        self.smooth = smooth
        self.rest_ids = _raster(rest, tris, shape)
        # 休みの網目の中の点 → 重心座標（三角形ごとの [x, y, 1] → λ の逆行列）
        a = rest[tris]
        self.tri_inv = np.linalg.inv(
            np.concatenate([a, np.ones((len(tris), 3, 1))], axis=2)
        )  # M×3×3

    def bary(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """休みの座標の点が入る三角形と重心座標（λ, 3 つ）。網目の外の点は ok=False。"""
        h, w = self.shape
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        k = self.rest_ids[yi, xi]
        ok = k >= 0
        lam = np.zeros((len(pts), 3))
        if ok.any():
            p = np.concatenate([pts[ok], np.ones((ok.sum(), 1))], axis=1)  # P×3
            lam[ok] = np.einsum("pi,pij->pj", p, self.tri_inv[k[ok]])
        return np.where(ok, k, 0), lam, ok

    def apply(self, verts: np.ndarray, k: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """網目 verts の上で、三角形 k・重心座標 lam の点の位置。"""
        return np.einsum("pj,pjc->pc", lam, verts[self.tris[k]])

    def solve(
        self,
        k: np.ndarray,
        lam: np.ndarray,
        target: np.ndarray,
        prior: np.ndarray,
        iters: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """対応（三角形 k・重心座標 lam → target）を満たす頂点。戻り値: (頂点, 各対応の残り px)。

        正規方程式（頂点の数 × 頂点の数）を解く。データの行は 1 点につき 3 つの頂点にしか掛からないので、足し込んで作る。
        """
        n = self.n
        if not hasattr(self, "_reg"):
            a_s = np.zeros((len(self.lines), n))
            r = np.arange(len(self.lines))
            a_s[r, self.lines[:, 0]] = 1
            a_s[r, self.lines[:, 1]] = -2
            a_s[r, self.lines[:, 2]] = 1
            self._reg = (self.smooth**2) * (a_s.T @ a_s)
        prior_w2 = 0.02**2
        t = self.tris[k]  # P×3
        rows = np.repeat(t[:, :, None], 3, axis=2).ravel()
        cols = np.repeat(t[:, None, :], 3, axis=1).ravel()
        ll = lam[:, :, None] * lam[:, None, :]  # P×3×3
        # 点の数で重みをならす（点の多い映像でも少ない映像でも、硬さの効き方が同じになるように）
        base = n / max(len(k), 1)
        w2 = np.full(len(k), base)
        verts = prior.copy()
        res = np.zeros(len(k))
        for _ in range(iters):
            m = self._reg + prior_w2 * np.eye(n)
            np.add.at(m, (rows, cols), (w2[:, None, None] * ll).ravel())
            rhs = prior_w2 * prior.copy()
            np.add.at(
                rhs,
                t.ravel(),
                ((w2[:, None] * lam)[:, :, None] * target[:, None, :]).reshape(-1, 2),
            )
            verts = np.linalg.solve(m, rhs)
            res = np.linalg.norm(self.apply(verts, k, lam) - target, axis=1)
            w2 = (
                base / np.maximum(1.0, res / 1.5) ** 2
            )  # Huber（1.5px より外れた対応を弱める）
        return verts, res


class _MeshTracker:
    """格子の網目を追う。網目の中の特徴点（最初のコマ）を手本にし、毎コマ 2 段で求める（平面の追跡と同じ考え方）。

    1. 予想: 手本の点の前のコマでの位置を LK で今のコマへ追い、網目を解く。
    2. 補正（2 回）: 予想の網目で今のコマを最初のコマの形に戻し、手本の点を LK で照らして、残りのずれから網目を解き直す。
       毎コマ最初のコマと照らすので、ずれがたまらない。
    """

    QUALITY = 0.01  # 手本の点の角の強さの下限（弱い模様の点は位置がぶれ、網目を乱す。mesh-b の試験で 0.003 より良かった）
    MAX_SHIFT = 6.0  # 補正で受け入れる、戻した画の上でのずれの上限（px）

    def __init__(
        self,
        gray0: np.ndarray,
        box: tuple[float, float, float, float],
        cols: int,
        rows: int,
        smooth: float,
    ) -> None:
        self.h, self.w = gray0.shape[:2]
        self.rest, self.tris = _grid_mesh(box, cols, rows)
        self.solver = _MeshSolver(
            self.rest, self.tris, cols, rows, gray0.shape[:2], smooth
        )
        mask = (self.solver.rest_ids >= 0).astype(np.uint8) * 255
        self.mask = mask
        pts = cv2.goodFeaturesToTrack(
            gray0, 2000, self.QUALITY, 4, mask=mask, blockSize=7
        )
        core.require(
            pts is not None and len(pts) >= 12,
            "網目の範囲に目印になる模様が足りません（無地の所は追えません）。--box を広げてください",
        )
        p0 = pts.reshape(-1, 2).astype(np.float64)
        k, lam, ok = self.solver.bary(p0)
        self.p0, self.k0, self.lam0 = p0[ok], k[ok], lam[ok]
        self.g0 = gray0
        self.V = self.rest.copy()
        self.prev = gray0
        # 最初のコマの形に戻すための、網目の中の画素（休みの座標）と重心座標
        ys, xs = np.nonzero(self.solver.rest_ids >= 0)
        pk, plam, _ = self.solver.bary(np.stack([xs, ys], 1).astype(np.float64))
        self.pix = (ys, xs, pk, plam)
        self.used = len(self.p0)

    def _rectify(
        self, gray: np.ndarray, verts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """今のコマを、網目 verts で最初のコマの形に戻す（網目の外は 0）。戻り値: (画, 写った所)。"""
        ys, xs, pk, plam = self.pix
        pos = self.solver.apply(verts, pk, plam)
        mx = np.full((self.h, self.w), -1.0, np.float32)
        my = np.full((self.h, self.w), -1.0, np.float32)
        mx[ys, xs], my[ys, xs] = pos[:, 0], pos[:, 1]
        rect = cv2.remap(
            gray,
            mx,
            my,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        cover = (mx >= 0) & (my >= 0) & (mx <= self.w - 1) & (my <= self.h - 1)
        return rect, cover

    def update(self, gray: np.ndarray) -> tuple[np.ndarray, int]:
        cur_prev = self.solver.apply(self.V, self.k0, self.lam0)
        use = (
            (cur_prev[:, 0] > 1)
            & (cur_prev[:, 1] > 1)
            & (cur_prev[:, 0] < self.w - 2)
            & (cur_prev[:, 1] < self.h - 2)
        )
        v_pred = self.V
        if use.sum() >= 12:
            p1, ok = _lk(self.prev, gray, cur_prev[use].astype(np.float32))
            if ok.sum() >= 12:
                v_pred, _ = self.solver.solve(
                    self.k0[use][ok], self.lam0[use][ok], p1[ok], self.V
                )
        v_new, n_used = v_pred, 0
        for _ in range(2):
            rect, cover = self._rectify(gray, v_new)
            both = cover & (self.mask > 0)
            if both.sum() > 100:
                m0, s0 = float(self.g0[both].mean()), float(self.g0[both].std()) + 1e-3
                m1, s1 = float(rect[both].mean()), float(rect[both].std()) + 1e-3
                rect = np.clip(
                    (rect.astype(np.float32) - m1) * (s0 / s1) + m0, 0, 255
                ).astype(np.uint8)
            seen = cover[
                self.p0[:, 1].round().astype(int).clip(0, self.h - 1),
                self.p0[:, 0].round().astype(int).clip(0, self.w - 1),
            ]
            if seen.sum() < 12:
                break
            q, ok = _lk(self.g0, rect, self.p0[seen].astype(np.float32), fb_max=1.5)
            # 戻した画の上では、正しい対応のずれは数 px のはず。大きいずれは繰り返し模様の取り違えとして捨てる
            ok &= np.linalg.norm(q - self.p0[seen], axis=1) < self.MAX_SHIFT
            if ok.sum() < 12:
                break
            kq, lamq, okq = self.solver.bary(q[ok].astype(np.float64))
            target = self.solver.apply(v_new, kq, lamq)
            sel = np.flatnonzero(seen)[ok][okq]
            v_new, res = self.solver.solve(
                self.k0[sel], self.lam0[sel], target[okq], v_new
            )
            n_used = int((res < 1.5).sum())
        self.V = v_new
        self.prev = gray
        self.used = n_used
        return v_new, n_used


def _face_tris() -> np.ndarray:
    """MediaPipe の顔の網目（468 点を結ぶ 852 個の三角形。目と口の中は穴）。"""
    from mediapipe.tasks.python import vision

    t = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    seen: dict[tuple[int, ...], tuple[int, int, int]] = {}
    for i in range(0, len(t), 3):
        tri = (t[i].start, t[i + 1].start, t[i + 2].start)  # 並びの向き（表・裏）を保つ
        seen.setdefault(tuple(sorted(tri)), tri)
    return np.array(list(seen.values()), np.int32)


# --place の目印（MediaPipe の顔の点の番号と重み。左右は画面で見た向き）。頬は頬の点（205・425）から鼻（4）の方へ
# 15% 寄せる（頬の外側は斜めを向いていて、貼る物の端が欠けやすいため）
_FACE_PLACES: dict[str, tuple[tuple[int, float], ...]] = {
    "cheek-left": ((205, 0.85), (4, 0.15)),  # 画面の左の頬
    "cheek-right": ((425, 0.85), (4, 0.15)),  # 画面の右の頬
    "forehead": ((151, 1.0),),  # 額の中ほど
    "nose": ((4, 1.0),),  # 鼻先の少し上
    "chin": ((199, 1.0),),  # あご
}


def _face_landmarks(
    src: str, rng: dict[str, Any], max_side: int | None, fps: float
) -> tuple[list[np.ndarray | None], list[int]]:
    """全コマの顔の点（468×3: x, y は px、z は奥行き（小さいほど手前）。px 単位に直した値）。顔が無いコマは None。

    顔が複数の時は、最初は大きい顔、その後は前のコマの顔に近い顔を選ぶ。カット（場面の切り替わり）のコマ番号も返す。
    """
    from . import models

    task = models.mp_task("face_landmarker", video=True, num_faces=4)
    out: list[np.ndarray | None] = []
    cuts: list[int] = []
    watch = _CutWatch(fps)
    pending_from: int | None = None
    prev_c = None
    w = h = 0
    for i, frame in enumerate(core.read_frames(src, max_side=max_side, **rng)):
        h, w = frame.shape[:2]
        scene = watch(frame)
        if scene == "pending":
            pending_from = i if pending_from is None else pending_from
        elif scene == "cut":
            cuts.append(
                pending_from if pending_from is not None else i
            )  # 大きな変化が始まったコマ
            pending_from, prev_c = None, None
        else:
            pending_from = None
        res = models.mp_detect(task, "detect", frame, round(i * 1000 / fps))
        faces = []
        for lm in res.face_landmarks or []:
            arr = np.array([[p.x * w, p.y * h, p.z * w] for p in lm[:468]], np.float64)
            faces.append(arr)
        if not faces:
            out.append(None)
            continue
        if prev_c is None:
            best = max(faces, key=lambda f: np.ptp(f[:, 0]) * np.ptp(f[:, 1]))
        else:
            best = min(faces, key=lambda f: np.linalg.norm(f[:, :2].mean(0) - prev_c))
        prev_c = best[:, :2].mean(0)
        out.append(best)
    if pending_from is not None:
        cuts.append(
            pending_from
        )  # 大きな変化のまま終わった（確かめられないので、カットとして区切る）
    return out, cuts


def _cross2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """2 次元のベクトルの外積（z 成分）。三角形の向き（表・裏）を見るのに使う。"""
    return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]


def _render_overlay(
    frame: np.ndarray,
    overlay: np.ndarray,
    rest: np.ndarray,
    cur: np.ndarray,
    tris: np.ndarray,
    sel: np.ndarray,
    *,
    order_z: np.ndarray | None = None,
    blend: str = "normal",
    opacity: float = 1.0,
) -> tuple[np.ndarray, int]:
    """休みの網目の上に置いた overlay（BGRA。画面と同じ大きさ）を、今の網目 cur に合わせて三角形ごとのアフィンで
    ゆがめて frame に重ねる。sel は描く三角形。order_z を渡すと奥から順に描き、裏返った三角形（向こう向き）は描かない。
    戻り値: (画, 描いた三角形の数)。
    """
    h, w = frame.shape[:2]
    out = frame[:, :, :3].copy()
    if len(sel) == 0:
        return out, 0
    tri_r = rest[tris[sel]]
    tri_c = cur[tris[sel]]
    cross_r = _cross2(tri_r[:, 1] - tri_r[:, 0], tri_r[:, 2] - tri_r[:, 0])
    cross_c = _cross2(tri_c[:, 1] - tri_c[:, 0], tri_c[:, 2] - tri_c[:, 0])
    # はっきり裏返った三角形（向こうを向いた面）だけを描かない。縁で細くなって向きが揺れる三角形まで抜くと、
    # 貼った物に穴があき、切れ端が離れて見えた（plate-laugh で確認）
    flipped = (np.sign(cross_r) != np.sign(cross_c)) & (
        np.abs(cross_c) > 0.2 * np.abs(cross_r)
    )
    keep = ~flipped if order_z is not None else np.ones(len(sel), bool)
    if order_z is not None and len(sel) >= 4:
        # 最初のコマで斜めに見えていた三角形が正面を向くと、貼る物の細い帯が大きく引き伸ばされて切れ端のように見える。
        # 面積の伸びが全体（中央値）の 3 倍を超える三角形は、そのコマでは描かない
        ratio = np.abs(cross_c) / (np.abs(cross_r) + 1e-9)
        keep &= ratio <= 3.0 * float(np.median(ratio[keep])) if keep.any() else True
    sel = sel[keep]
    if order_z is not None and len(sel):
        sel = sel[np.argsort(-order_z[tris[sel]].mean(axis=1))]  # 奥（z が大きい）から
    roi = _bbox(cur[tris[sel]].reshape(-1, 2), w, h) if len(sel) else None
    if roi is None:
        return out, 0
    aff = np.full((len(tris), 2, 3), np.nan)
    aff[sel] = _tri_affines(cur, rest, tris[sel])  # 今の三角形 → 休みの三角形
    ids = _raster(cur, tris, (h, w), order=sel)
    mx, my, mask = _pwa_map(ids, aff, roi)
    warped = cv2.remap(
        overlay, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    x0, y0, x1, y1 = roi
    a = warped[:, :, 3].astype(np.float32) / 255 * mask * opacity
    base = out[y0:y1, x0:x1].astype(np.float32)
    top = warped[:, :, :3].astype(np.float32)
    if blend == "multiply":
        top = base * top / 255
    out[y0:y1, x0:x1] = np.clip(
        base * (1 - a[..., None]) + top * a[..., None], 0, 255
    ).astype(np.uint8)
    return out, len(sel)


def _place_overlay(
    img: np.ndarray, box: tuple[float, float, float, float], shape: tuple[int, int]
) -> np.ndarray:
    """貼る画像（BGRA）を、画面と同じ大きさの透明な画の box（x, y, 幅, 高さ）に引き伸ばして置く。"""
    h, w = shape
    x, y, bw, bh = box
    src = img if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    m = cv2.getAffineTransform(
        np.float32([[0, 0], [src.shape[1], 0], [0, src.shape[0]]]),
        np.float32([[x, y], [x + bw, y], [x, y + bh]]),
    )
    return cv2.warpAffine(
        src,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _draw_mesh(
    img: np.ndarray, verts: np.ndarray, tris: np.ndarray, sel: np.ndarray | None = None
) -> None:
    lines = set()
    for t in tris if sel is None else tris[sel]:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            lines.add((min(a, b), max(a, b)))
    for a, b in lines:
        p, q = verts[a], verts[b]
        cv2.line(
            img,
            (int(p[0]), int(p[1])),
            (int(q[0]), int(q[1])),
            (80, 230, 80),
            1,
            cv2.LINE_AA,
        )


@command(
    25,
    "mesh-warp",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--mode",
        choices=["grid", "face"],
        default="grid",
        help="grid（既定。格子の網目を映像の動きに合わせてゆがめる）・face（MediaPipe の顔の網目 468 点で、顔の動きに合わせる）",
    ),
    arg(
        "--overlay",
        help="貼る画像か動画（透過つき PNG・WebM 可）。省略時は網目だけを描く",
    ),
    arg(
        "--box",
        help="grid: 網目を張る範囲＝貼る範囲 x,y,幅,高さ（最初のコマ。元の映像の px。省略時は画面全体）／"
        "face: 貼る範囲（省略時は --place と --size で顔に合わせて決める）",
    ),
    arg(
        "--grid",
        default="auto",
        help="grid: 升目の数 列x行（例 12x16。既定 auto = 升目が約 32px になる数）",
    ),
    arg(
        "--stiff",
        type=float,
        default=0.04,
        help="grid: 網目の硬さ（大きいほどなめらか、小さいほど細かい動きに合う。既定 0.04）",
    ),
    arg(
        "--place",
        choices=list(_FACE_PLACES),
        default="cheek-left",
        help="face: 貼る場所（cheek-left・cheek-right は画面で見た左右の頬、forehead 額、nose 鼻、chin あご。既定 cheek-left）",
    ),
    arg(
        "--size",
        type=float,
        default=0.25,
        help="face: 貼る物の幅（顔の幅に対する割合。既定 0.25）",
    ),
    arg(
        "--temporal",
        type=float,
        default=0.04,
        help="face: 顔の点を時間方向にならす強さ（秒。0 でならさない。既定 0.04）",
    ),
    arg(
        "--blend",
        choices=["normal", "multiply"],
        default="normal",
        help="重ね方: normal（既定）・multiply（乗算。肌の陰影が残る）",
    ),
    arg("--opacity", type=float, default=1.0, help="不透明度（0〜1。既定 1）"),
    arg(
        "--show-mesh",
        action="store_true",
        help="網目の線も描く（確かめる用。--overlay が無い時は常に描く）",
    ),
)
def mesh_warp(a: argparse.Namespace) -> None:
    """網目（メッシュ）を動きに合わせてゆがめ、最初のコマに置いた画像を三角形ごとのアフィン変換で貼り付ける。"""
    info = core.probe(a.src)
    rng = _rng(a)
    pw, ph, k = _proc_size(info, a.max_side)
    inv = 1.0 / k
    ins = _Insert(a.overlay) if a.overlay else None
    show = a.show_mesh or ins is None
    out = (
        Path(a.out)
        if a.out
        else core.default_out(a.src, "mesh-warp", ".jpg" if info.is_image else ".mp4")
    )
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    if a.mode == "grid":
        core.require(
            not info.is_image, "grid は動画で使います（画像 1 枚では動きがありません）"
        )
        _mesh_grid(a, info, rng, pw, ph, k, inv, ins, show, out, json_path)
    else:
        _mesh_face(a, info, rng, pw, ph, k, inv, ins, show, out, json_path)


def _mesh_grid(
    a: argparse.Namespace,
    info: core.MediaInfo,
    rng: dict[str, Any],
    pw: int,
    ph: int,
    k: float,
    inv: float,
    ins: _Insert | None,
    show: bool,
    out: Path,
    json_path: Path,
) -> None:
    if a.box:
        bx, by, bw, bh = core.parse_box(a.box)
        box = (bx * k, by * k, bw * k, bh * k)
    else:
        box = (1.0, 1.0, pw - 3.0, ph - 3.0)
    if a.grid == "auto":
        cols, rows = (int(min(40, max(2, round(v / 32)))) for v in box[2:])
    else:
        try:
            cols, rows = (int(v) for v in a.grid.lower().split("x"))
        except ValueError as e:
            raise core.VfxError(
                f"--grid は 列x行（例 12x16）か auto で指定してください: {a.grid}"
            ) from e
    core.require(
        1 <= cols <= 60 and 1 <= rows <= 60, "--grid の列・行は 1〜60 にしてください"
    )
    # 網目は貼る範囲より 1 升ずつ外まで広げて張る（縁の頂点も両側の模様で決まり、縁で暴れにくい。画面の外へは広げない）
    cw, ch = box[2] / cols, box[3] / rows
    add_l = 1 if box[0] - cw >= 1 else 0
    add_t = 1 if box[1] - ch >= 1 else 0
    add_r = 1 if box[0] + box[2] + cw <= pw - 2 else 0
    add_b = 1 if box[1] + box[3] + ch <= ph - 2 else 0
    grid_box = (
        box[0] - add_l * cw,
        box[1] - add_t * ch,
        box[2] + (add_l + add_r) * cw,
        box[3] + (add_t + add_b) * ch,
    )
    g_cols, g_rows = cols + add_l + add_r, rows + add_t + add_b
    frames = core.read_frames(a.src, max_side=a.max_side, **rng)
    writer = core.VideoWriter(
        out,
        info.fps,
        audio_from=a.src,
        audio_start=rng["start"],
        audio_duration=a.duration,
    )
    tracker: _MeshTracker | None = None
    cuts = _CutWatch(info.fps)
    rows_json: list[dict[str, Any]] = []
    sel = np.empty(0, np.int32)
    used = []
    stopped_at = None
    with writer:
        for i, frame in enumerate(frames):
            gray = core.to_gray(frame)
            scene = cuts(frame)
            if tracker is None:
                tracker = _MeshTracker(gray, grid_box, g_cols, g_rows, a.stiff)
                sel = np.arange(len(tracker.tris))
                verts, n = tracker.rest, len(tracker.p0)
            elif stopped_at is None and scene == "cut":
                stopped_at = i
                core.log(
                    f"  {core.t_of(i, info.fps, rng['start']):.2f} 秒あたりでカットを見つけ、そこで止めました"
                )
            if stopped_at is not None or scene == "pending":
                writer.write(frame)
                rows_json.append(
                    {"i": i, "t": core.t_of(i, info.fps, rng["start"]), "ok": False}
                )
                continue
            if i > 0:
                verts, n = tracker.update(gray)
            used.append(n)
            img = frame[:, :, :3]
            if ins is not None:
                ov = _place_overlay(ins.next(), box, (ph, pw))
                img, _ = _render_overlay(
                    img,
                    ov,
                    tracker.rest,
                    verts,
                    tracker.tris,
                    sel,
                    blend=a.blend,
                    opacity=a.opacity,
                )
            if show:
                img = img.copy()
                _draw_mesh(img, verts, tracker.tris)
            writer.write(img)
            rows_json.append(
                {
                    "i": i,
                    "t": core.t_of(i, info.fps, rng["start"]),
                    "ok": True,
                    "points": n,
                    "verts": [
                        [round(float(x) * inv, 2), round(float(y) * inv, 2)]
                        for x, y in verts
                    ],
                }
            )
    assert tracker is not None
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "mode": "grid",
            "fps": info.fps,
            "size": [info.width, info.height],
            "start": rng["start"],
            "grid": [g_cols, g_rows],
            "overlay_box": [round(v * inv, 2) for v in box],
            "note": "verts は網目の頂点（(列+1)×(行+1) 個を行ごとに左から。元の映像の px）。tris は三角形（頂点の番号 3 つ）。"
            "rest は最初のコマの頂点。網目は貼る範囲（overlay_box）より 1 升ずつ外まで広げてある。points は網目を解くのに使えた点の数",
            "rest": [
                [round(float(x) * inv, 2), round(float(y) * inv, 2)]
                for x, y in tracker.rest
            ],
            "tris": tracker.tris.tolist(),
            "frames": rows_json,
        },
    )
    core.log(
        f"{len(rows_json)} コマ: 網目 {g_cols}×{g_rows}（貼る範囲は {cols}×{rows}。頂点 {len(tracker.rest)}）、手本の点 {len(tracker.p0)}、"
        f"網目を解くのに使えた点 中央値 {int(np.median(used)) if used else 0}（最少 {min(used) if used else 0}）"
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


def _mesh_face(
    a: argparse.Namespace,
    info: core.MediaInfo,
    rng: dict[str, Any],
    pw: int,
    ph: int,
    k: float,
    inv: float,
    ins: _Insert | None,
    show: bool,
    out: Path,
    json_path: Path,
) -> None:
    tris = _face_tris()
    if info.is_image:
        from . import models

        img = core.load_image(a.src)
        if (pw, ph) != (img.shape[1], img.shape[0]):
            img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
        task = models.mp_task("face_landmarker", video=False, num_faces=1)
        res = models.mp_detect(task, "detect", img, None)
        core.require(bool(res.face_landmarks), "顔が見つかりません")
        lms = [
            np.array(
                [[p.x * pw, p.y * ph, p.z * pw] for p in res.face_landmarks[0][:468]]
            )
        ]
        frames_iter: Any = iter([img])
    else:
        core.log("顔の点を読み取っています（MediaPipe FaceLandmarker）…")
        lms, cuts = _face_landmarks(a.src, rng, a.max_side, info.fps)
        frames_iter = core.read_frames(a.src, max_side=a.max_side, **rng)
    n = len(lms)
    have = np.array([lm is not None for lm in lms])
    core.require(have.any(), "顔が見つかりません")
    arr = np.stack(
        [lm if lm is not None else np.zeros((468, 3)) for lm in lms]
    )  # N×468×3
    first = int(np.flatnonzero(have)[0])
    if not info.is_image and a.temporal > 0:
        # カットをまたいでならさない（場面ごとにならす）。カットの後も、顔の同じ場所（点の番号）に貼り続ける
        bounds = [0, *cuts, n]
        sm_all = arr.copy()
        use_all = np.zeros(n, bool)
        for s0, s1 in itertools.pairwise(bounds):
            if s1 <= s0:
                continue
            sm, usable = _fill_smooth(
                arr[s0:s1].reshape(s1 - s0, -1),
                have[s0:s1],
                a.temporal * info.fps,
                round(0.2 * info.fps),
            )
            sm_all[s0:s1] = np.where(
                usable[:, None, None], sm.reshape(s1 - s0, 468, 3), arr[s0:s1]
            )
            use_all[s0:s1] = usable
        arr, have = sm_all, use_all
        for c in cuts:
            core.log(
                f"  {core.t_of(c, info.fps, rng['start']):.2f} 秒でカットを見つけました（ならしはここで区切る）"
            )
    rest = arr[first, :, :2]
    face_w = float(np.linalg.norm(rest[234] - rest[454]))
    if a.box:
        bx, by, bw, bh = core.parse_box(a.box)
        box = (bx * k, by * k, bw * k, bh * k)
    else:
        anchor = sum(w * rest[i] for i, w in _FACE_PLACES[a.place])
        bw = a.size * face_w
        aspect = 1.0
        if ins is not None:
            probe_img = ins.next()
            aspect = probe_img.shape[0] / probe_img.shape[1]
            ins = _Insert(a.overlay)  # 動画なら頭から
        bh = bw * aspect
        box = (anchor[0] - bw / 2, anchor[1] - bh / 2, bw, bh)
    # 貼る範囲に掛かる三角形だけを描く。最初のコマで向こう向き・横向き（画面の上でつぶれている）の三角形は除く
    # （つぶれた三角形を通すと貼る物の一部が大きく引き伸ばされ、切れ端が離れて見えた。plate-laugh で確認）
    tri_r = rest[tris]
    x0, y0, bw, bh = box
    hit = (
        (tri_r[:, :, 0].max(1) >= x0)
        & (tri_r[:, :, 0].min(1) <= x0 + bw)
        & (tri_r[:, :, 1].max(1) >= y0)
        & (tri_r[:, :, 1].min(1) <= y0 + bh)
    )
    cr = _cross2(tri_r[:, 1] - tri_r[:, 0], tri_r[:, 2] - tri_r[:, 0])
    front = float(np.sign(np.median(cr)))
    # 3 次元の面積に対する画面の上の面積（＝面の向きの cos）。0.2 未満（78° より横を向いた面）は、最初のコマで
    # 斜めから見ていて、後で正面を向くと貼る物の細い帯が大きく引き伸ばされる（plate-laugh の頬の外側で確認）ので使わない
    t3 = arr[first][tris]
    n3 = np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0])
    cos_view = np.abs(cr) / (np.linalg.norm(n3, axis=1) + 1e-9)
    facing = (front * cr > 2.0) & (cos_view >= 0.2)
    sel = np.flatnonzero(hit & facing).astype(np.int32)
    corners = np.array(
        [
            [x0, y0],
            [x0 + bw, y0],
            [x0 + bw, y0 + bh],
            [x0, y0 + bh],
            [x0 + bw / 2, y0 + bh / 2],
        ]
    )
    # 貼る物の四隅と中心が、休みの網目のどの三角形のどこにあるか（今のコマでの位置を JSON に書くため）
    tri_inv = np.linalg.inv(np.concatenate([tri_r, np.ones((len(tris), 3, 1))], axis=2))
    lam_all = np.einsum("pi,mij->pmj", np.c_[corners, np.ones(len(corners))], tri_inv)
    inside = (lam_all >= -1e-6).all(axis=2)
    corner_tri = [
        int(np.flatnonzero(inside[j])[0]) if inside[j].any() else -1
        for j in range(len(corners))
    ]
    writer = (
        None
        if info.is_image
        else core.VideoWriter(
            out,
            info.fps,
            audio_from=a.src,
            audio_start=rng["start"],
            audio_duration=a.duration,
        )
    )
    rows_json: list[dict[str, Any]] = []
    drawn = []
    result = None
    for i, frame in enumerate(frames_iter):
        if i >= n:
            break
        img = frame[:, :, :3]
        row: dict[str, Any] = {
            "i": i,
            "t": 0.0 if info.is_image else core.t_of(i, info.fps, rng["start"]),
            "ok": bool(have[i]),
        }
        if have[i]:
            cur = arr[i, :, :2]
            if ins is not None:
                ov = _place_overlay(ins.next(), box, (ph, pw))
                img, nd = _render_overlay(
                    img,
                    ov,
                    rest,
                    cur,
                    tris,
                    sel,
                    order_z=arr[i, :, 2],
                    blend=a.blend,
                    opacity=a.opacity,
                )
                drawn.append(nd)
            if show:
                img = img.copy()
                _draw_mesh(img, cur, tris)
            pos = []
            for j, kt in enumerate(corner_tri):
                if kt < 0:
                    pos.append(None)
                    continue
                lam = lam_all[j, kt]
                p = lam @ cur[tris[kt]]
                pos.append([round(float(p[0]) * inv, 2), round(float(p[1]) * inv, 2)])
            row["overlay_corners"] = pos[:4]
            row["overlay_center"] = pos[4]
            row["landmarks"] = [
                [round(float(x) * inv, 1), round(float(y) * inv, 1)] for x, y in cur
            ]
        if writer is not None:
            writer.write(img)
        else:
            result = img
        rows_json.append(row)
    if writer is not None:
        writer.close()
    else:
        assert result is not None
        core.save_image(out, result)
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "mode": "face",
            "fps": info.fps,
            "size": [info.width, info.height],
            "start": rng["start"],
            "box": [round(v * inv, 2) for v in box],
            "note": "landmarks は MediaPipe の顔の 468 点（元の映像の px）。overlay_corners は貼った物の四隅（左上・右上・右下・"
            "左下）、overlay_center は中心の、そのコマでの位置（網目の三角形の上で動かした位置。目・口の穴の上は null）",
            "tris": tris.tolist(),
            "frames": rows_json,
        },
    )
    core.log(
        f"{n} コマ中 {int(have.sum())} コマで顔を見つけました。顔の幅 {face_w * inv:.0f}px、貼る範囲 {[round(v * inv) for v in box]}、"
        f"掛かる三角形 {len(sel)}（描いた数 中央値 {int(np.median(drawn)) if drawn else 0}）"
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")
