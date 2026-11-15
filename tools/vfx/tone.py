"""画質を上げる（明るさと色）。

担当するテクニック:
   32 逆トーンマッピング（inverse-tonemap）
   33 トーンマッピング（tonemap）
   35 ホワイトバランス（white-balance）
   40 低照度補正（low-light）
   47 周辺減光の補正（devignette）
   48 ガンマ補正（gamma）
   49 ヒストグラム平坦化（equalize）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
どのコマンドも、処理した映像（動画は MP4、画像は画像）と数値の JSON、前後を並べた確認画像
（<出力>-compare.jpg）を書く。32 は HDR（動画は HEVC 10 bit の MP4、画像は AVIF）を書く。
色の行列・範囲などの印（タグ）は元の動画のものを引き継ぐ（印の無い動画は ffmpeg の既定 BT.601 で読み書きする）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np

from . import core
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/tone/（既知の劣化・変換を加えた素材を作り、元との差を測る）
NOTES: dict[int, dict[str, str]] = {
    32: {
        "status": "○",
        "how": "SDR を zscale（libzimg）で線形の光・BT.2020 の色（SDR の白 = 1）にし（動画は BT.1886 の 2.4 乗、画像は sRGB。"
        "印の無い動画は BT.601・TV の範囲として読む）、明るさ Y だけを曲線で広げて 3 色に同じ比を掛ける（色相・彩度を保つ）。"
        "曲線: SDR の信号 --knee（既定 0.75 ≒ 102 nit）までは ITU-R BT.2408 の割り当て（SDR の白 = --white 203 nit）のまま、"
        "それより上は y + A(y − knee)²（つなぎ目で傾きが続く 2 次）で広げ、SDR の白を --peak（既定 1000 nit）にする"
        "（--peak = --white で広げない）。zscale で PQ（SMPTE ST 2084）か HLG（BT.2100、表示 1000 nit 基準）・BT.2020 の行列・"
        "TV の範囲にし、動画は x265 の 10 bit（hvc1。HDR10 は P3-D65・0.0001〜1000 nit のマスタリング表示と、1 回目の読み込みで"
        "測った MaxCLL・MaxFALL を入れる）、画像は AV1 の AVIF（10 bit・4:4:4、nclx の印）。音声は残す。AI の逆トーンマッピングではない",
        "measured": "既知の信号の試験図（灰色 18 段・色の札 8 枚・明るい所の段階）を変換し、書き出した HDR を zscale で nit に戻して"
        "設計の曲線と比べた: 1 nit 以上の灰色 17 段の差は平均 0.21〜0.43%・最大 0.94%（10 bit の符号値で最大 0.70 段。"
        "動画・画像 × PQ・HLG）、SDR の白 → 1000.0〜1004.2 nit、黒 → 0 nit。広げない所の色の札の色度 xy の差は最大 0.0033。"
        "広げない時の SDR の白は HLG 75.00%（BT.2408 の基準 75%）・PQ 58.11%（203 nit = 58.07%）。ffprobe で hevc・yuv420p10le・"
        "bt2020・smpte2084（arib-std-b67）・bt2020nc・tv と、マスタリング表示・MaxCLL/MaxFALL（試験図 1000/113、story540 1000/84）"
        "を確認。33 で SDR に戻すと元との差 38.6 dB（PQ）・39.6 dB（HLG）。試験の途中で直した所: swscale で 16 bit の RGB に"
        "すると 8 bit の 255 が 65280 になって白が 0.9% 暗く、広げると最大 5.3% ずれた → YUV → RGB も zscale にした。YUV の外れ値が"
        " 1 を超えて MaxCLL が 2165 nit になった → 広げる前に 0〜1 に切る。速さ: story540（540×960・361 コマ）全体で 25.5 秒"
        "（MaxCLL を測る 1 回目を含む）、2 秒の動画で 2.2〜2.8 秒、画像 1 枚 1 秒",
        "limits": "HDR の画面で目で見る確認はできていない（この環境に HDR の表示が無い。確認画像は「HDR に対応しない画面での見え方」と"
        "明るさの色分け）。明るい所を広げる簡易な曲線で、学習した逆トーンマッピング（白飛びした所の復元など）ではない。8 bit の"
        " SDR を広げるので、明るい所のなめらかな階調では SDR の 1 段が HDR の 10 bit で平均 2.5 段（上の端で 3 段、最大 5 段）に"
        "広がり、空などで縞が見えることがある（--peak を 400〜600 に下げると緩む）。--knee より明るい白い服・壁も明るくなる"
        "（白い服を 203 nit 付近に保つなら --peak を下げるか --knee を上げる）。HLG は表示 1000 nit 基準で、--peak は 1000 まで。"
        "Dolby Vision などの動的な情報は入れない。x265 の圧縮で明るい輪郭がわずかに行き過ぎる（story540 で最大 1093 nit）",
        "tested": "test_itm.py（試験図の動画・画像 × PQ・HLG、広げない時の基準の白、ffprobe の印と HDR10 の情報、明るい所の段階、"
        "story540 全体）、debug_itm.py・debug_itm2.py（誤差の切り分け）、test_tonemap.py（33 で戻す往復）、確認画像 "
        "32-inverse-tonemap.jpg・32-inverse-tonemap-chart.jpg を目で確認",
    },
    33: {
        "status": "◎",
        "how": "HDR（PQ・HLG。color_transfer の印で判断）→ SDR: ffmpeg の zscale で表示の光に戻し（--white 203 nit = 1。BT.2408 の"
        "基準の白を SDR の白に）、BT.709 の色にして tonemap（mobius・hable・reinhard・clip・linear。3 色の最大で縮めて色相を保つ）、"
        "BT.1886 の 2.4 乗（画像は sRGB）で 8 bit に（誤差拡散のディザ）。最大の明るさは MaxCLL → マスタリング表示 → 1000 nit の順"
        "（HLG は 1000）。mobius の既定は --param 0.5（約 101 nit までそのまま。ffmpeg の既定は 0.3）。1 回の ffmpeg で書き出し、"
        "BT.709 の印を付ける。AVIF などの HDR の画像も読む。SDR の入力: local（既定）= L* を輪郭を保ってぼかした土台（ガイデッド"
        "フィルター、半径 = 長辺の 3%、eps 0.01）と細部に分け、土台を b + 強さ·(0.5 − b)·4b(1 − b)（黒と白の端は動かさず単調）で"
        "中間に寄せ、細部を --detail 倍して戻す。明るさの比を線形の光の 3 色に掛ける（色相を保つ）。fusion = 露出 ±強さ×2 段の"
        " 3 枚を OpenCV の MergeMertens で合成。clahe = 明るさの CLAHE（clip 1.5）",
        "measured": "32 で広げた HDR（最大 1000 nit、story540 6〜8 秒）を SDR に戻して元と比べた（48 コマ）: mobius 38.58 dB・"
        "ΔE00 1.79（--param 0.3 では 34.78 dB）、clip 31.84、reinhard 29.77、hable 20.06（暗くなる）、linear 15.17。HLG の mobius "
        "39.55 dB。広げない HDR を clip で戻す: 39.54 dB・ΔE00 1.72（HDR を crf 0 にすると 43.42 dB）、画像（AVIF 経由）38.01 dB。"
        "zscale で直接作った HDR10（0.1〜4000 nit の段階、MaxCLL 4000）: どの方法も単調。mobius は 100 → 190・203 → 227・"
        "1000 → 250・4000 → 255（255 は 2640 nit から）、hable は 203 → 149・1000 → 220、reinhard 203 → 196、clip は 203 nit から"
        " 255、linear 203 → 73。SDR の local（既定の強さ 0.4・細部 1.2）、7 枚: 白飛び（Y' 250 以上）0.34 → 0.03%、黒つぶれ"
        "（5 以下）0.27 → 0.04%、明るい所の細部（高周波の標準偏差）14.40 → 13.19、暗い所 9.08 → 8.71（強さ 0.5・細部 1.0 では"
        " 11.67・7.99 まで弱まったので既定を選び直した）。fusion: 0.02%・0.50%・13.08・9.78。動画の local でちらつきは増えない"
        "（明るさの平均のコマ間の変化 1.150 → 0.873）。速さ: 540×960・48 コマで HDR → SDR 2.3〜2.7 秒、local 3.6 秒",
        "limits": "実際のカメラの HDR（iPhone の HLG など）では試していない（手元に無い。試したのは 32 で作った HDR と、zscale で"
        "直接作った合成の HDR10）。Dolby Vision の動的な情報は使わない（プロファイル 5 のように HDR10 の土台の無い物は色が正しく"
        "ならない）。hable と linear は全体が暗くなる（hable は 203 nit が 58%）。SDR の局所調整は、完全に白飛びした所（255）の"
        "模様は戻せない（端は動かさない作り）。強くすると平板・かすんだ見た目になる（0.6 以上は様子を見て）。fusion は暗い所を"
        "沈めることがある（building.jpg で黒つぶれ 1.75 → 3.83%）。clahe は 49 と同じ方法",
        "tested": "test_tonemap.py（往復・合成の HDR10 の段階・SDR の 7 枚と動画）、exp_local_tm.py（既定の強さと細部を並べて"
        "目で選んだ: 33-tonemap-local-settings.jpg）、確認画像 33-tonemap.jpg を目で確認",
    },
    35: {
        "status": "◎",
        "how": "線形の光（sRGB を戻した値）で照明の色を推定し、灰色の物の明るさを変えない倍率（von Kries）を掛ける。推定（--method）: "
        "shades（既定。Shades of Gray、6 乗平均）・grayworld・greyedge（σ2 でぼかした 1 次微分の 6 乗平均）・whitepatch（上位 0.5%）・"
        "learning（OpenCV の LearningBasedWB。opencv_contrib 同梱の学習済みモデル、Apache-2.0。線形の 16 bit で渡す）。白飛び"
        "（3 色のどれかが 0.92 以上）と暗すぎる画素は推定に使わず、3 色とも明るい所は白のまま残す。動画は長辺 256px の全コマで"
        "推定し、Lab の差で場面（カット）に分けて場面ごとの中央値の倍率を掛ける（場面の中ではちらつかない）。--temperature: CIE の"
        "昼光（4000K 以上）・プランク（未満）の軌跡の色度から Bradford の色順応（3×3）で --target（既定 6500K）に移す。"
        "--gains: 倍率を直接。--strength で弱める",
        "measured": "線形の光で既知の色かぶり（R・G・B の倍率 4 種類、加えた差 ΔE00 平均 8.37）を 9 枚に加え、元と色かぶりの両方を"
        "補正して比べた（36 例）: 照明の比の推定の角度の誤差 shades 中央値 0.03°・最大 0.34°、grayworld 0.02°・1.35°、"
        "whitepatch 0.13°・0.77°、greyedge 0.12°・2.37°、learning 5.42°・13.22°（8 bit の sRGB で渡すと中央値 8.23°）。明るさを"
        "そろえた補正後どうしの差 ΔE00: shades 平均 0.47・最大 0.63、grayworld 0.52、whitepatch 0.51、greyedge 0.60、learning 6.11、"
        "比べた ffmpeg の grayworld 0.89・greyedge 1.11。色温度の色かぶり（Bradford で D65 → 3000K・4500K・9000K）は自動の対角の"
        "倍率では残る: shades の明るさをそろえた ΔE00 0.73〜4.79（3000K で 2.80〜4.79）。--temperature で逆に戻すと 48.4〜55.0 dB"
        "（ΔE00 0.24〜0.49。色かぶりを加えた時に色域の外に出た fruits の 3000K だけ 31.3 dB）。--gains で逆の倍率: 57.5〜61.6 dB。"
        "動画（story540 全体に暖色の色かぶり）: 場面の区切り 5/5 がコマ単位で正解と一致、場面ごとの推定の誤差 0.06〜1.00°、"
        "補正後どうしの差 ΔE00 1.55（明るさをそろえて。加えた差 6.42）。速さ: 540×960・48 コマで 3.4 秒、361 コマで 19 秒",
        "limits": "自動の方法は「画面の平均（など）は灰色」という仮定なので、夕日・電球・カフェの暖かい色など意図した色も中立に寄せる"
        "（--strength 0.6〜0.8 で雰囲気を残す、--per clip で動画全体を 1 つの倍率に）。ほぼ一色の画面（芝生のアップなど）は大きく"
        "外れる。色温度による色かぶりは対角の倍率では完全には戻らない（光の色温度が分かれば --temperature）。learning は試験で"
        "推定がぶれたので既定にしない。場面の区切りは簡易（ゆっくりした暗転では区切らない）",
        "tested": "test_wb.py（9 枚 × 4 種類の色かぶり、色温度の色かぶり 3 枚 × 3 段、--gains、動画の場面ごとの推定、ffmpeg の"
        " grayworld・greyedge との比較）、exp_learning_wb.py（learning の入力の比較）、確認画像 35-white-balance.jpg を目で確認",
    },
    40: {
        "status": "○",
        "how": "exposure（既定）: 明るさの中央値（線形の光）を --target（sRGB の信号 0.45）にする倍率（1〜--max-gain 16）を線形の"
        "光の 3 色に掛け、明るい所は 3 色の最大で 0.8 から白に向けて滑らかに丸める（色相を保つ）。動画は場面ごと（Lab の差で区切る）"
        "に 1 つの倍率。ノイズ除去: 明るくした後のコマでノイズの強さ σ（2×2 Haar の対角成分の中央値）を測り、画像は OpenCV の"
        "非局所平均（h = 2σ）、動画は ffmpeg の hqdn3d（空間 2σ・時間 3σ）を書き出しの時にかける。lime: 照明の地図（R・G・B の"
        "最大をガイデッドフィルターでならした物）の --strength 乗で割る（LIME 風）。gamma: 48 の自動と同じ。AI モデルは使わない",
        "measured": "普通の明るさの画像 7 枚を線形の光で k 倍に暗くし、センサーのようなノイズ（分散 x/2000 + 0.0015²）を加えて元と"
        "比べた（k 0.1・0.04 の 14 例の平均。PSNR / SSIM / 明るさの平均を元に合わせた PSNR）: exposure（既定）21.72 dB / 0.736 / "
        "27.61、ノイズ除去なし 20.14 / 0.392 / 22.99、lime 0.6 16.89 / 0.672 / 21.87、gamma 17.98 / 0.585 / 19.38、暗くした倍率を"
        "知っている理想の逆（ノイズ除去なし）22.99 / 0.383 / 23.01。最初の既定 lime 0.8 は 16.00 / 0.291（明るくしすぎ、ノイズと"
        "色が強く出た）だったので exposure に替えた。動画（story540 6〜8 秒・12.4〜14.4 秒、k 0.06、コマごとに別のノイズ、48 コマ）: "
        "exposure 28.54・29.69 dB / SSIM 0.642・0.748（ノイズ除去なし 25.26・25.99 / 0.421・0.478）、lime 16.95・17.73、gamma "
        "18.13・20.51。hqdn3d の強さの較正（σ の 0〜4 倍、4 例）: 2 倍が最良か最良に近い（例 28.47 → 30.91 dB）。画像の非局所平均は"
        " 2〜3 倍が最良（k 0.1 で 24.14・24.47 dB）。速さ: 540×960・48 コマで 4.2 秒（hqdn3d を含む）、画像 1 枚 0.2〜0.3 秒",
        "limits": "光のむらが大きい場面（片側だけ照らされた夜など。照明の地図 0.015〜0.35 の試験）はどの方法も戻せない（exposure "
        "14.05 dB・lime 13.99・gamma 15.13、理想の逆 19.16）。exposure は明るい所を丸めるので、照らされた側は白っぽくなる。"
        "ノイズ除去で細かい模様（ニットの編み目など）が柔らかくなる（--denoise-strength 1〜1.5 で残す）。倍率の上限 16 を超える"
        "暗さは暗いまま。学習した低照度補正のモデル（Zero-DCE・RetinexFormer など）は入れていない。hqdn3d は動きを補償しないので、"
        "速く動く所はノイズが残る",
        "tested": "test_lowlight.py（画像 7 枚 × 一様 2 段・むら、動画 2 区間）、exp_lowlight.py（方法とノイズ除去の強さの比較）、"
        "calib_lowlight_video.py（hqdn3d の較正）、確認画像 40-low-light.jpg を目で確認（lime の色ノイズ、exposure の柔らかさ）",
    },
    47: {
        "status": "△",
        "how": "信号の値（ガンマのかかった R・G・B）に、中心からの距離（四隅で 1。ffmpeg の vignette と同じ測り方）だけで決まる倍率"
        " 1/V を掛ける（上限 --max-gain）。V の求め方: --angle = ffmpeg の vignette と同じ cos⁴(角度 × 距離)。--reference = 無地の面を"
        "撮った画像・動画の明るさの対数に log V = a r² + b r⁴ + c r⁶ を当てはめる。自動（既定）= 明るさの対数の外向きの傾きを"
        "半径ごとに上下 20% を除いて平均し（自然な画像では 0 に近いという仮定）、log V = a r² + b r⁴ の傾きに当てはめる（動画は"
        " 32 コマを合わせて）。自動は、外に向かって単調に暗くならない推定や四隅の倍率 1.02 未満では補正しない。上限は自動 2 倍・"
        "ほか 4 倍",
        "measured": "--angle: ffmpeg の vignette（RGB、ディザなし）で 30・45 度の減光を加えた 4 枚を同じ角度で戻すと 47.78〜50.38 dB"
        "（ffmpeg 自身の mode=backward は 44.70〜46.45 dB）。動画（story540 6〜8 秒・36 度）17.68 → 39.89 dB。--reference: 照明の"
        "むら ±8%・ノイズ σ2.5・JPEG 品質 90 の無地の面（画像と 24 コマの動画、540×960 と 1280×720）から測った四隅の明るさの誤差"
        " 0.66〜2.31%（cos⁴40° の 0.3444 を 0.3365〜0.3392、別の型の 0.4066 を 0.3977〜0.4039）、同じ減光を加えた画像を戻すと"
        " 48.94〜53.99 dB（最初は 3% のぼかしで四隅が +9% 明るく出たので、ぼかしを 1px にした）。自動: 減光の無い画像 11 枚の推定は"
        "四隅 0.313〜1.000（0.9〜1.1 に入ったのは 4/11。building 0.313・fruits 0.444・vtest 0.470 は写っている物の暗さを減光と"
        "見なし、上限の 2 倍で明るくした）。cos⁴30°（0.563）を加えた時の元の推定との比の誤差は log で中央値 0.0010（加えた分は"
        "正確に測れる）が、正解との差は中央値 0.334・最大 1.075。動画の自動（story540・32 コマ）: 元は補正なし、cos⁴30° を加えると"
        " 0.508（正解 0.563）。速さ: 540×960・48 コマで 2.1 秒、story540 全体の自動 4.4〜8.2 秒",
        "limits": "条件: 自動の推定は、四隅に暗い物（芝生・木・暗い背景）や明るい物がある構図で大きく外れる（減光の無い画像で四隅"
        " 0.31〜1.0 と推定）。確実なのは --reference（同じカメラ・レンズ・ズームで無地の壁や白い紙を撮る）か --angle（ffmpeg の"
        " vignette で付けた減光を外す時）。自動が正確なのは四隅まで空や壁が続く画像（home・aero1 で 1.000）。作品の演出として"
        "付けた周辺減光も消す。中心のずれた減光・左右非対称な減光・色ごとに違う減光は扱わない。四隅の明るい所は白飛びすることが"
        "ある。試して採らなかった方法: カメラの動きで同じ物が別の距離に写ることを使う推定（DIS のオプティカルフロー）は、減光の"
        "無い短い場面で 0.77〜1.18、加えた減光の比で ±20% ぶれた。方向（扇形 8 つ）ごとの推定の多数決は扇形ごとの推定が不安定",
        "tested": "test_devignette.py（--angle と ffmpeg、--reference の合成の無地の面、自動の 11 枚、動画）、exp_vignette.py・"
        "exp_vignette_motion.py・exp_vignette_sector.py（推定方法の比較）、確認画像 47-devignette.jpg を目で確認（building の誤補正を見た）",
    },
    48: {
        "status": "◎",
        "how": "出力 = 入力^(1/ガンマ) の 256 段の表（cv2.LUT）を R・G・B にかける（--channel rgb、既定）か、YCbCr（BT.601・全範囲、"
        "float で計算）の Y' だけにかける（luma）。黒（0）と白（255）は動かない。--gamma を省くと、範囲の小さなコマ（最大 64 枚）の"
        "明るさ Y' の中央値を --target（0.45）にするガンマ = log(中央値) / log(目標) を 1 つ選ぶ（1/3〜3）。動画は全コマ同じ表なので"
        "ちらつかない",
        "measured": "既知のガンマ γ0（0.5・0.7・1.5・2.2）を独立な実装（float で計算して 8 bit に丸める）でかけた 4 枚を --gamma 1/γ0 で"
        "戻すと 40.60〜59.43 dB、独立な実装で戻した物との差は全画素で 0（誤差は暗くした時の 8 bit の丸めだけ。γ0 0.5 で最大 11 段・"
        "平均 0.29〜1.05）。luma: 28.51〜57.04 dB（fruits の γ0 0.6 は、彩度の高い色が明るくした時に色域の外に出た分が戻らない）。"
        "自動の推定: 元の中央値を目標にすると、かけたガンマを相対誤差 平均 1.79%・最大 4.46% で当てた（10 例）。動画（story540 "
        "6〜8 秒、ffmpeg の lutrgb で γ0 0.6 → 戻す）: 16.73 → 42.21 dB（ガンマ 1.0 で書き直しただけで 46.02 dB）。ffmpeg の "
        "eq=gamma=2 は TV の範囲の Y を 0〜255 のまま計算するため、黒 0 → 55・16 → 82 に持ち上がる（この道具は 0 → 0・16 → 64、"
        "式 63.9）。速さ: 540×960・48 コマで 1.3 秒（luma 1.5 秒）",
        "limits": "中間だけを動かすので、白飛び・黒つぶれは直らない（33・40 を使う）。rgb で大きく明るくすると色が少し薄く、luma では"
        "彩度の高い色が色域の外に出て切れる。自動は中央値だけを見るので、夜の雰囲気なども明るくする（--target を下げる）。8 bit で"
        "暗くした映像を戻しても、消えた暗部の段差は戻らない",
        "tested": "test_gamma.py（画像 4 枚 × 4 段、luma、自動の推定 10 例、動画、ffmpeg eq との比較）、確認画像 48-gamma.jpg を目で確認",
    },
    49: {
        "status": "◎",
        "how": "Lab の L（明るさ）だけを平坦化し、a・b（色）は保つ（変えた量を元の L に足すので端数の精度が残る）。clahe（既定）: OpenCV の"
        " CLAHE（--clip 2.0・--tiles 8×8）。global: OpenCV の equalizeHist と同じ式（累積分布）を、端数の L で補間して当てる。動画の"
        " global は分布を時間でならし（--smooth 0.5 秒の指数移動平均）、場面の切れ目（Lab の差）でやり直す。--strength で元と混ぜる",
        "measured": "平坦化は明るさの順位だけで決まるので、コントラストを下げた画像 C（76〜178 に縮めた物と、ガンマ + 縮小）を平坦化"
        "すると、元 I を平坦化した物と同じになるはず: L の PSNR(eq(C), eq(I)) は global 27.87〜42.69 dB（平均 38.43、7 枚 × 2 種類）、"
        "比べた ffmpeg の histeq 24.81〜32.99（平均 28.56）。CLAHE は伸ばす量に上限があるので一致しない（14.38〜21.83）が、明るさの"
        "標準偏差は C の 13〜33 → 27〜52（元 41〜82）に戻る（global は 72〜75 まで伸ばす）。OpenCV の関数を L に直接かけた物との差: "
        "L の平均 0.56〜1.38 段・最大 12 段（色域の外に出た色を戻す時の切り捨て）。平らな所の細かい揺れ（高周波の標準偏差）: "
        "story-face 0.55 → global 1.68・clahe 1.42、aero1 3.59 → 5.60・8.10。動画のちらつき（固定カメラの vtest を低コントラストに"
        "した 200 コマ、背景の明るさのコマ間の変化）: global のならしなし 0.061 → 既定 0.5 秒 0.046 → 2 秒 0.039、clahe 0.044"
        "（元 0.020）。story540 の場面の切れ目 4/4 をコマ単位で検出。速さ: 540×960・48 コマで 1.7〜2.1 秒、768×576・200 コマで"
        " 4.8〜6.2 秒",
        "limits": "明るさだけを変えるので、薄くなった色（彩度）は戻らない（色を含めた PSNR は global で 19.55〜34.51 dB）。global は"
        "全体を強く伸ばし、空や肌のなめらかな所のノイズ・圧縮の段差が目立つ（--strength 0.5〜0.7）。CLAHE は区画ごとなので、"
        "大きな明暗の差がある所の境に軽いむらが出ることがある。動画の global はならしてもわずかにちらつく",
        "tested": "test_equalize.py（7 枚 × 2 種類の低コントラスト、OpenCV との照合、平らな所の揺れ、vtest のちらつき、story540 の"
        "切れ目、ffmpeg の histeq との比較）、確認画像 49-equalize.jpg を目で確認",
    },
}

# ── 共通の部品 ────────────────────────────────────────────────────────────

# 色の変換を正確にする swscale の設定（色を最近傍で広げて読み、2×2 の平均で戻すと、手を加えない画素はほぼ元に戻る）
_SWS = "accurate_rnd+full_chroma_int+full_chroma_inp"
# ffprobe の color_space → swscale の out_color_matrix の名前
_SWS_MATRIX = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt470",
    "fcc": "fcc",
    "smpte240m": "smpte240m",
}
_HDR_TRC = {"smpte2084": "pq", "arib-std-b67": "hlg"}
# 明るさの重み（線形の光。BT.709 / sRGB）。並びは OpenCV と同じ B, G, R
_Y709 = np.array([0.0722, 0.7152, 0.2126], np.float32)


def _stream(info: core.MediaInfo) -> dict[str, Any]:
    for s in info.extra.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    return {}


def _tags(info: core.MediaInfo) -> dict[str, str]:
    """元の映像の色の印（分かっているものだけ）。"""
    s = _stream(info)
    keys = ("color_space", "color_primaries", "color_transfer", "color_range")
    return {
        k: str(s[k])
        for k in keys
        if s.get(k) and s[k] not in ("unknown", "unspecified", "reserved")
    }


def _hdr_kind(info: core.MediaInfo) -> str | None:
    return _HDR_TRC.get(_tags(info).get("color_transfer", ""))


def _span(a: argparse.Namespace, info: core.MediaInfo) -> float:
    """処理する長さ（秒）。"""
    dur = max(info.duration - (a.start or 0.0), 0.0)
    if a.duration:
        dur = min(dur, a.duration)
    if a.max_frames:
        dur = min(dur, a.max_frames / info.fps)
    return dur


def _n_frames(a: argparse.Namespace, info: core.MediaInfo) -> int:
    if info.is_image:
        return 1
    n = max(1, round(_span(a, info) * info.fps))
    return min(n, a.max_frames) if a.max_frames else n


def _ff_in(a: argparse.Namespace, info: core.MediaInfo) -> list[str]:
    """ffmpeg の入力の指定（範囲つき）。"""
    args: list[str] = []
    if not info.is_image and a.start:
        args += ["-ss", f"{a.start:.4f}"]
    args += ["-i", str(a.src)]
    if not info.is_image and (a.duration or a.max_frames):
        args += ["-t", f"{_span(a, info):.4f}"]
    return args


def _proc_size(info: core.MediaInfo, max_side: int | None) -> tuple[int, int]:
    """処理する大きさ。動画は偶数にそろえる。画像は元の大きさのまま（縮める時だけ丸める）。"""
    if info.is_image:
        f = min(1.0, max_side / max(info.width, info.height)) if max_side else 1.0
        return max(1, round(info.width * f)), max(1, round(info.height * f))
    return core.scaled_size(info.width, info.height, max_side=max_side)


def _paths(
    a: argparse.Namespace,
    info: core.MediaInfo,
    tag: str,
    *,
    video_ext: str = ".mp4",
    image_ext: str = ".png",
) -> tuple[Path, Path, Path]:
    """(出力, JSON, 確認画像) の場所。透過つきの動画（SDR の処理）は透過を残すため .webm に書く。"""
    if not info.is_image and info.has_alpha and video_ext == ".mp4":
        video_ext = ".webm"
    ext = image_ext if info.is_image else video_ext
    out = Path(a.out) if a.out else core.default_out(a.src, tag, ext)
    if info.is_image and image_ext in core.IMAGE_EXTS:
        core.require(
            out.suffix.lower() in core.IMAGE_EXTS,
            f"画像の出力先は画像の拡張子（.png など）にしてください: {out}",
        )
    elif not info.is_image:
        core.require(
            out.suffix.lower() == video_ext,
            f"動画の出力先は {video_ext} にしてください: {out}",
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    return out, json_path, out.with_name(out.stem + "-compare.jpg")


def _pipe_frames(
    cmd: list[str],
    shape: tuple[int, ...],
    dtype: Any,
    max_frames: int | None,
    src: str | Path,
) -> Iterator[np.ndarray]:
    """ffmpeg が書き出す生のコマを順に受け取る。"""
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    count = 0
    try:
        while max_frames is None or count < max_frames:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, dtype).reshape(shape).copy()
            count += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if count == 0:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise core.VfxError(f"コマを読めませんでした: {src}\n{err[-800:]}")


# OpenCV では読めず ffmpeg で読む画像（HDR の画像によく使われる形式）
_FF_IMAGE_EXTS = {".avif", ".heic", ".heif", ".jxl"}


def _probe(path: str | Path) -> core.MediaInfo:
    """core.probe と同じ。ただし AVIF・HEIC などの 1 枚の画像も画像として扱う。"""
    info = core.probe(path)
    if Path(path).suffix.lower() in _FF_IMAGE_EXTS and info.frames <= 1:
        info = dataclasses.replace(info, is_image=True, fps=1.0, duration=0.0, frames=1)
    return info


def _load_image(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    """画像を BGR（透過があれば BGRA）で読み、指定の大きさにする（OpenCV で読めない形式は ffmpeg で読む）。"""
    if Path(path).suffix.lower() in _FF_IMAGE_EXTS:
        info = core.probe(path)
        w, h = info.width, info.height
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path)]
        cmd += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgra", "-"]
        r = subprocess.run(cmd, capture_output=True, check=False)
        core.require(
            r.returncode == 0 and len(r.stdout) >= w * h * 4,
            f"画像を読めません: {path}",
        )
        img = np.frombuffer(r.stdout[: w * h * 4], np.uint8).reshape(h, w, 4).copy()
    else:
        img = core.load_image(path, alpha=True)
    if not (img[:, :, 3] < 255).any():
        img = img[:, :, :3]
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def _read_bgr(
    a: argparse.Namespace, info: core.MediaInfo, size: tuple[int, int]
) -> Iterator[np.ndarray]:
    """処理するコマを BGR（8 bit）で順に返す（画像は 1 枚。透過は捨てる）。"""
    if info.is_image:
        yield _load_image(a.src, size)[:, :, :3].copy()
        return
    w, h = size
    vf = [f"scale={w}:{h}:flags=area"] if (w, h) != (info.width, info.height) else []
    vf += [f"scale=flags=neighbor+{_SWS}", "format=bgr24"]
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *_ff_in(a, info)]
    cmd += ["-an", "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    yield from _pipe_frames(cmd, (h, w, 3), np.uint8, a.max_frames, a.src)


def _small_frames(
    a: argparse.Namespace, info: core.MediaInfo, side: int, max_count: int | None = None
) -> list[np.ndarray]:
    """統計を取る用に、範囲のコマを小さくして読む（max_count を超える時は間引く）。"""
    if info.is_image:
        w, h = core.scaled_size(info.width, info.height, max_side=side)
        return [_load_image(a.src, (w, h))[:, :, :3].copy()]
    span = _span(a, info)
    fps = None
    if max_count and span * info.fps > max_count:
        fps = max_count / max(span, 1e-3)
    frames = core.read_all(
        a.src,
        start=a.start or 0.0,
        duration=span,
        max_frames=a.max_frames,
        max_side=side,
        fps=fps,
    )
    return frames[:max_count] if max_count else frames


class _Writer:
    """Python で作ったコマ（BGR）を MP4（H.264）に書く。元の音声（同じ範囲）と色の印を引き継ぐ。

    色の変換は正確な設定で行う（core.VideoWriter の既定の変換より劣化が少ない）。vf_post は
    YUV に変えた後にかけるフィルター（ノイズ除去など）。
    """

    def __init__(
        self,
        a: argparse.Namespace,
        info: core.MediaInfo,
        out: Path,
        vf_post: list[str] | None = None,
    ) -> None:
        self.a, self.info, self.out = a, info, out
        self.vf_post = vf_post or []
        self.proc: subprocess.Popen[bytes] | None = None
        self.count = 0

    def _open(self, w: int, h: int) -> None:
        a, info = self.a, self.info
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}"]
        cmd += ["-r", f"{info.fps:.6f}", "-i", "-"]
        if info.has_audio:
            if a.start:
                cmd += ["-ss", f"{a.start:.4f}"]
            if a.duration or a.max_frames:
                cmd += ["-t", f"{_span(a, info):.4f}"]
            cmd += ["-i", str(a.src), "-map", "0:v", "-map", "1:a", "-shortest"]
        tags = _tags(info)
        matrix = _SWS_MATRIX.get(tags.get("color_space", ""))
        conv = f"scale=flags=area+{_SWS}"
        if matrix:
            conv += f":out_color_matrix={matrix}:out_range=tv"
        cmd += ["-vf", ",".join([conv, "format=yuv420p", *self.vf_post])]
        cmd += core.MP4_ARGS
        if matrix:
            cmd += ["-colorspace", tags["color_space"], "-color_range", "tv"]
            if tags.get("color_primaries"):
                cmd += ["-color_primaries", tags["color_primaries"]]
            if tags.get("color_transfer") and tags["color_transfer"] not in _HDR_TRC:
                cmd += ["-color_trc", tags["color_transfer"]]
        cmd += ["-c:a", "aac", "-b:a", "192k"] if info.has_audio else []
        cmd.append(str(self.out))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame[:, :, :3])
        h, w = frame.shape[:2]
        if self.proc is None:
            core.require(
                w % 2 == 0 and h % 2 == 0, f"大きさは偶数にしてください（{w}×{h}）"
            )
            self._open(w, h)
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(frame.tobytes())
        self.count += 1

    def close(self) -> None:
        if self.proc is None:
            raise core.VfxError(f"書き出すコマがありません: {self.out}")
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = (
            self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        )
        if self.proc.wait() != 0:
            raise core.VfxError(f"書き出しに失敗しました: {self.out}\n{err[-800:]}")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None and exc[0] is None:
            self.close()
        elif self.proc is not None:
            self.proc.kill()


def _keep_idx(n: int, k: int = 3) -> list[int]:
    """確認画像に使うコマの番号（等間隔に k 枚）。"""
    return sorted({min(n - 1, int((j + 0.5) * n / k)) for j in range(k)})


def _process(
    a: argparse.Namespace,
    info: core.MediaInfo,
    out: Path,
    fn: Callable[[np.ndarray, int], np.ndarray],
    *,
    vf_post: list[str] | None = None,
    keep: int = 3,
) -> tuple[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    """コマを 1 枚ずつ fn(コマ, 番号) で変えて書き出す。画像は画像で保存する（透過は残す）。

    戻り値: (処理したコマ数, [(番号, 前, 後)]) — 確認画像用に、等間隔の数コマの前後を覚えておく。
    """
    if _hdr_kind(info):
        raise core.VfxError(
            "入力は HDR（PQ・HLG）です。この処理は SDR 用なので、先に 33 tonemap で SDR にしてください"
        )
    size = _proc_size(info, a.max_side)
    if not info.is_image and info.has_alpha:
        return _process_alpha(a, info, out, fn, size, keep, bool(vf_post))
    if info.is_image:
        img = _load_image(a.src, size)
        res = fn(np.ascontiguousarray(img[:, :, :3]), 0)
        saved = np.dstack([res, img[:, :, 3]]) if img.shape[2] == 4 else res
        if img.shape[2] == 4 and out.suffix.lower() in {".jpg", ".jpeg"}:
            saved = res
        core.save_image(out, saved)
        return 1, [(0, img[:, :, :3].copy(), res)]
    total = _n_frames(a, info)
    want = set(_keep_idx(total, keep))
    kept: list[tuple[int, np.ndarray, np.ndarray]] = []
    with _Writer(a, info, out, vf_post) as w:
        for i, frame in enumerate(_read_bgr(a, info, size)):
            res = fn(frame, i)
            w.write(res)
            if i in want:
                kept.append((i, frame, res))
            if total >= 20 and (i + 1) % max(1, total // 5) == 0:
                core.log(f"  {i + 1}/{total} コマ")
        count = w.count
    if not kept:
        kept = [(count - 1, frame, res)]
    return count, kept


def _process_alpha(
    a: argparse.Namespace,
    info: core.MediaInfo,
    out: Path,
    fn: Callable[[np.ndarray, int], np.ndarray],
    size: tuple[int, int],
    keep: int,
    had_post: bool,
) -> tuple[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    """透過つきの動画（WebM）: 色だけを処理し、透過はそのまま残して WebM（VP9）に書く。"""
    if had_post:
        core.log(
            "  注意: 透過つきの動画では、書き出しの時のノイズ除去（hqdn3d）は使いません"
        )
    total = _n_frames(a, info)
    want = set(_keep_idx(total, keep))
    kept: list[tuple[int, np.ndarray, np.ndarray]] = []
    frames = core.read_frames(
        a.src,
        start=a.start or 0.0,
        duration=a.duration,
        max_frames=a.max_frames,
        size=size,
        alpha=True,
    )
    writer = core.VideoWriter(
        out,
        info.fps,
        alpha=True,
        audio_from=a.src if info.has_audio else None,
        audio_start=a.start or 0.0,
        audio_duration=a.duration,
    )
    count = 0
    with writer:
        for i, frame in enumerate(frames):
            bgr = np.ascontiguousarray(frame[:, :, :3])
            res = fn(bgr, i)
            writer.write(np.dstack([res, frame[:, :, 3]]))
            if i in want:
                kept.append((i, bgr, res))
            count += 1
    if not kept:
        raise core.VfxError(f"コマを読めませんでした: {a.src}")
    return count, kept


def _finish(
    out: Path,
    json_path: Path,
    sheet_path: Path,
    sheet: np.ndarray | None,
    data: dict[str, Any],
) -> None:
    core.write_json(json_path, data)
    if sheet is not None:
        core.save_image(sheet_path, sheet)
    core.log(f"保存: {out}")
    if sheet is not None:
        core.log(f"保存: {sheet_path}")
    core.log(f"保存: {json_path}")


# ── 色の計算 ──────────────────────────────────────────────────────────────


@cache
def _dec_lut() -> np.ndarray:
    """8 bit の sRGB の値 → 線形の光（0〜1）。"""
    x = np.arange(256, dtype=np.float64) / 255
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


@cache
def _enc_lut() -> np.ndarray:
    """線形の光を 65536 段に分けた表 → 8 bit の sRGB の値。"""
    x = np.arange(65536, dtype=np.float64) / 65535
    v = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.clip(np.round(v * 255), 0, 255).astype(np.uint8)


def _to_lin(img: np.ndarray) -> np.ndarray:
    return _dec_lut()[img[:, :, :3]]


def _to_u8(lin: np.ndarray) -> np.ndarray:
    idx = np.clip(lin * 65535 + 0.5, 0, 65535).astype(np.uint16)
    return _enc_lut()[idx]


def _luma8(img: np.ndarray) -> np.ndarray:
    """BGR → 明るさ Y'（BT.601 の重み。0〜255 の float32）。"""
    f = img[:, :, :3].astype(np.float32)
    return f[..., 0] * 0.114 + f[..., 1] * 0.587 + f[..., 2] * 0.299


def _noise_sigma(img: np.ndarray) -> float:
    """明るさのノイズの強さ（0〜255 の標準偏差）を 1 枚から推定する（2×2 Haar の対角成分の中央値 ÷ 0.6745、
    16px の区画ごとに求めて下位 25%。白飛び・黒つぶれの区画は除く）。"""
    y = _luma8(img)
    h, w = y.shape
    y = y[: h - h % 2, : w - w % 2]
    a, b, c, d = y[0::2, 0::2], y[0::2, 1::2], y[1::2, 0::2], y[1::2, 1::2]
    hh = np.abs(a - b - c + d) / 2
    blk = 8
    hb, wb = hh.shape[0] // blk, hh.shape[1] // blk
    if hb < 2 or wb < 2:
        return float(np.median(hh) / 0.6745)
    t = hh[: hb * blk, : wb * blk].reshape(hb, blk, wb, blk).transpose(0, 2, 1, 3)
    est = np.median(t.reshape(hb * wb, -1), axis=1) / 0.6745
    m = ((a + b + c + d) / 4)[: hb * blk, : wb * blk]
    m = m.reshape(hb, blk, wb, blk).mean(axis=(1, 3)).ravel()
    ok = (m > 12) & (m < 240)
    if ok.sum() < 8:
        ok[:] = True
    return float(np.percentile(est[ok], 25))


def _cuts(smalls: list[np.ndarray], fps: float) -> list[int]:
    """場面の切れ目（新しい場面の最初のコマの番号）を、縮めたコマの Lab の差で簡易に探す。

    前後のコマとの差（32×32 の Lab の平均の差）が、まわりの差の中央値の 3 倍 + 4 と 10 の大きい方を超えたら切れ目。
    0.3 秒以内に続いた切れ目は最初の 1 つにまとめる（白いフラッシュなど）。
    """
    if len(smalls) < 2:
        return []
    labs = [
        cv2.cvtColor(
            cv2.resize(f[:, :, :3], (32, 32), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2Lab,
        ).astype(np.float32)
        for f in smalls
    ]
    d = np.array([np.abs(labs[i] - labs[i - 1]).mean() for i in range(1, len(labs))])
    win = max(3, round(fps / 2))
    cuts: list[int] = []
    for i in range(1, len(labs)):
        lo, hi = max(0, i - 1 - win), min(len(d), i + win)
        neigh = np.delete(d[lo:hi], i - 1 - lo)
        base = float(np.median(neigh)) if len(neigh) else 0.0
        if d[i - 1] > max(10.0, 3.0 * base + 4.0) and (
            not cuts or i - cuts[-1] > 0.3 * fps
        ):
            cuts.append(i)
    return cuts


def _segments(n: int, cuts: list[int]) -> list[tuple[int, int]]:
    """切れ目で区切った場面 [(始め, 終わり + 1)]。"""
    edges = [0, *[c for c in cuts if 0 < c < n], n]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]


# ── 確認画像の部品 ────────────────────────────────────────────────────────


def _plot(
    series: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], str]],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    size: tuple[int, int] = (560, 400),
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    logy: bool = False,
    vlines: list[tuple[float, tuple[int, int, int], str]] | None = None,
    fill: bool = False,
) -> np.ndarray:
    """折れ線のグラフ（白地の BGR 画像）。series は (x, y, 色, 名前)。"""
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    left, right, top, bottom = 66, 14, 38, 44
    pw, ph = w - left - right, h - top - bottom

    def ty(v: Any) -> np.ndarray:
        v = np.asarray(v, np.float64)
        return np.log10(np.clip(v, 1e-6, None)) if logy else v

    ylo, yhi = float(ty(ylim[0])), float(ty(ylim[1]))

    def px(x: Any) -> np.ndarray:
        return left + (np.asarray(x, np.float64) - xlim[0]) / (xlim[1] - xlim[0]) * pw

    def py(y: Any) -> np.ndarray:
        v = top + ph - (ty(y) - ylo) / (yhi - ylo) * ph
        return np.clip(v, top, top + ph)

    for k in range(6):
        x = xlim[0] + (xlim[1] - xlim[0]) * k / 5
        xx = round(float(px(x)))
        cv2.line(img, (xx, top), (xx, top + ph), (232, 232, 232), 1)
        core.draw_text(
            img, f"{x:g}", (xx, top + ph + 5), size=13, color=(80, 80, 80), bg=None,
            anchor="mt",
        )  # fmt: skip
    yticks = (
        [10**e for e in range(math.ceil(ylo), math.floor(yhi) + 1)]
        if logy
        else [ylim[0] + (ylim[1] - ylim[0]) * k / 4 for k in range(5)]
    )
    for v in yticks:
        yy = round(float(py(v)))
        cv2.line(img, (left, yy), (left + pw, yy), (232, 232, 232), 1)
        core.draw_text(
            img, f"{v:g}", (left - 6, yy), size=13, color=(80, 80, 80), bg=None,
            anchor="rm",
        )  # fmt: skip
    for x, color, name in vlines or []:
        xx = round(float(px(x)))
        for y0 in range(top, top + ph, 10):
            cv2.line(img, (xx, y0), (xx, min(y0 + 5, top + ph)), color, 1)
        if name:
            core.draw_text(img, name, (xx + 3, top + 3), size=12, color=color, bg=None)
    for xs, ys, color, _name in series:
        pts = np.stack([px(xs), py(ys)], axis=1).round().astype(np.int32)
        if fill:
            poly = np.vstack([[pts[0, 0], top + ph], pts, [pts[-1, 0], top + ph]])
            ov = img.copy()
            cv2.fillPoly(ov, [poly], color)
            img = cv2.addWeighted(ov, 0.25, img, 0.75, 0)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (left, top), (left + pw, top + ph), (120, 120, 120), 1)
    core.draw_text(img, title, (left, 9), size=16, color=(20, 20, 20), bg=None)
    if xlabel:
        core.draw_text(
            img, xlabel, (left + pw, h - 4), size=13, color=(60, 60, 60), bg=None,
            anchor="rb",
        )  # fmt: skip
    if ylabel:
        core.draw_text(
            img, ylabel, (4, top - 4), size=13, color=(60, 60, 60), bg=None,
            anchor="lb",
        )  # fmt: skip
    y = top + 6
    for _xs, _ys, color, name in series:
        if not name:
            continue
        cv2.line(img, (left + 8, y + 8), (left + 30, y + 8), color, 3)
        core.draw_text(
            img, name, (left + 36, y), size=13, color=(40, 40, 40), bg=(255, 255, 255)
        )
        y += 22
    return img


def _hist_plot(
    imgs: list[np.ndarray], names: list[str], title: str = "明るさの分布"
) -> np.ndarray:
    """明るさ（Y'）の分布を重ねたグラフ。"""
    colors = [(150, 150, 150), (40, 40, 220), (200, 120, 20), (40, 160, 40)]
    series = []
    top = 1e-9
    hists = []
    for im in imgs:
        hist = np.bincount(
            np.clip(_luma8(im) + 0.5, 0, 255).astype(np.uint8).ravel(), minlength=256
        ).astype(np.float64)
        hist = cv2.GaussianBlur(hist.reshape(1, -1), (0, 0), 1.5).ravel()
        hist /= hist.sum()
        hists.append(hist)
        top = max(top, float(np.percentile(hist, 99.5)))
    xs = np.arange(256)
    for k, (hist, name) in enumerate(zip(hists, names, strict=True)):
        series.append((xs, np.minimum(hist / top, 1.0), colors[k % 4], name))
    return _plot(
        series,
        title=title,
        xlabel="明るさ（0〜255）",
        xlim=(0, 255),
        ylim=(0, 1),
        fill=True,
        size=(560, 300),
    )


def _rows(blocks: list[np.ndarray], width: int = 1600) -> np.ndarray:
    """画像を縦に並べる（幅をそろえる）。"""
    rows = []
    for im in blocks:
        im = im[:, :, :3]
        if im.shape[1] != width:
            h = max(1, round(im.shape[0] * width / im.shape[1]))
            im = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
        rows += [im, np.full((6, width, 3), 24, np.uint8)]
    return np.vstack(rows[:-1])


def _hcat(blocks: list[np.ndarray], height: int) -> np.ndarray:
    """画像を横に並べる（高さをそろえる）。"""
    out = []
    for im in blocks:
        im = im[:, :, :3]
        w = max(1, round(im.shape[1] * height / im.shape[0]))
        out += [
            cv2.resize(im, (w, height), interpolation=cv2.INTER_AREA),
            np.full((height, 6, 3), 24, np.uint8),
        ]
    return np.hstack(out[:-1])


def _pairs_sheet(
    kept: list[tuple[int, np.ndarray, np.ndarray]],
    fps: float,
    start: float,
    labels: tuple[str, str],
    is_image: bool,
) -> np.ndarray:
    """前後のコマを並べた画像（動画は数コマ分）。"""
    images, names = [], []
    for i, before, after in kept:
        t = "" if is_image else f" {start + i / fps:.2f}秒"
        images += [before, after]
        names += [labels[0] + t, labels[1] + t]
    portrait = kept[0][1].shape[0] > kept[0][1].shape[1]
    cols = min(len(images), 6 if portrait else 4)
    return core.contact_sheet(
        images, names, cols=cols, cell_width=320 if portrait else 400
    )


# ── 48 ガンマ補正 ─────────────────────────────────────────────────────────

_GAMMA_CH = {
    "rgb": "R・G・B の各色に同じ曲線をかける（写真の「レベル補正」の中間と同じ。明るくすると少し色が薄く見える）",
    "luma": "明るさ（Y'）だけにかけ、色差（Cb・Cr）はそのまま",
}


def _gamma_lut(g: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float64) / 255
    return np.clip(np.round(255 * x ** (1 / g)), 0, 255).astype(np.uint8)


def _apply_gamma(img: np.ndarray, g: float, channel: str) -> np.ndarray:
    if channel == "rgb":
        return cv2.LUT(img, _gamma_lut(g))
    f = img.astype(np.float32) / 255
    ycc = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = np.power(np.clip(ycc[..., 0], 0, 1), 1 / g)
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def _auto_gamma(smalls: list[np.ndarray], target: float) -> tuple[float, float, float]:
    """明るさ（Y'、0〜1）の中央値を target にするガンマ。(ガンマ, 前の中央値, 後の中央値の見込み)"""
    y = np.concatenate([_luma8(f).ravel() for f in smalls]) / 255
    med = float(np.clip(np.median(y), 0.02, 0.98))
    g = math.log(med) / math.log(target)
    g = float(np.clip(g, 1 / 3, 3.0))
    return g, med, med ** (1 / g)


@command(
    48,
    "gamma",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--gamma",
        type=float,
        help="ガンマ値（出力 = 入力^(1/ガンマ)。1 より大きいと中間が明るく、小さいと暗くなる。"
        "黒と白は動かない。目安 0.6〜2.2）。省略時は自動（--target）",
    ),
    arg(
        "--target",
        type=float,
        default=0.45,
        help="自動の時、明るさ（Y'、0〜1）の中央値をこの値にするガンマを選ぶ（既定 0.45。範囲全体で 1 つの値）",
    ),
    arg(
        "--channel",
        choices=list(_GAMMA_CH),
        default="rgb",
        help="かける所。" + " ／ ".join(f"{k}: {v}" for k, v in _GAMMA_CH.items()),
    ),
)
def gamma(a: argparse.Namespace) -> None:
    """中間の明るさだけを上げ下げする（黒と白は動かさない）。省略時は明るさの中央値から自動で決める。"""
    info = _probe(a.src)
    core.require(a.gamma is None or a.gamma > 0, "--gamma は 0 より大きくしてください")
    core.require(0 < a.target < 1, "--target は 0 と 1 の間にしてください")
    out, json_path, sheet_path = _paths(a, info, "gamma")
    auto: dict[str, float] | None = None
    if a.gamma is None:
        g, med, med_after = _auto_gamma(_small_frames(a, info, 256, 64), a.target)
        auto = {
            "median_before": med,
            "target": a.target,
            "median_after_expected": med_after,
        }
        core.log(
            f"自動: 明るさの中央値 {med:.3f} → {med_after:.3f}（目標 {a.target}）にするガンマ {g:.3f}"
        )
    else:
        g = a.gamma
    count, kept = _process(a, info, out, lambda f, _i: _apply_gamma(f, g, a.channel))
    lut = _gamma_lut(g)
    core.log(
        f"ガンマ {g:.3f}（{a.channel}）: 64 → {lut[64]}、128 → {lut[128]}、192 → {lut[192]}（{count} コマ）"
    )
    xs = np.linspace(0, 1, 256)
    curve = _plot(
        [
            (xs, xs, (170, 170, 170), "そのまま"),
            (xs, xs ** (1 / g), (40, 40, 220), f"ガンマ {g:.2f}"),
        ],
        title="入力 → 出力",
        xlabel="入力",
        size=(420, 400),
    )
    _i, before, after = kept[len(kept) // 2]
    hist = _hist_plot([before, after], ["元", "補正後"])
    pairs = _pairs_sheet(
        kept, info.fps, a.start or 0.0, ("元", "補正後"), info.is_image
    )
    sheet = _rows([pairs, _hcat([curve, hist], 300)])
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "gamma": round(g, 4),
            "channel": a.channel,
            "auto": auto,
            "lut_samples": {str(v): int(lut[v]) for v in (16, 32, 64, 128, 192, 240)},
            "frames": count,
        },
    )


# ── 49 ヒストグラム平坦化 ─────────────────────────────────────────────────

_EQ = {
    "clahe": "CLAHE（区画ごとに平坦化し、伸ばしすぎを --clip で抑える。既定。行き過ぎにくい）",
    "global": "画面全体の平坦化（OpenCV の equalizeHist と同じ式。動画は分布を --smooth 秒でならしてちらつきを抑える）",
}


class _Equalizer:
    """Lab の L（明るさ）だけを平坦化し、色（a・b）は保つ。変えた量を元の L に足すので、端数の精度が残る。"""

    def __init__(
        self,
        method: str,
        clip: float,
        tiles: int,
        strength: float,
        alpha: float,
        cuts: set[int],
    ) -> None:
        self.method, self.strength, self.alpha, self.cuts = (
            method,
            strength,
            alpha,
            cuts,
        )
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
        self.hist: np.ndarray | None = None

    def lut(self, l8: np.ndarray, i: int) -> np.ndarray:
        """画面全体の平坦化の表（0〜255 → 0〜255 の float）。分布は時間でならす。"""
        hist = np.bincount(l8.ravel(), minlength=256).astype(np.float64)
        hist /= hist.sum()
        if self.hist is None or i in self.cuts or self.alpha <= 0:
            self.hist = hist
        else:
            self.hist = self.alpha * self.hist + (1 - self.alpha) * hist
        cdf = np.cumsum(self.hist)
        c0 = cdf[np.nonzero(self.hist)[0][0]] if self.hist.any() else 0.0
        return np.clip((cdf - c0) / max(1e-9, 1 - c0) * 255, 0, 255)

    def __call__(self, img: np.ndarray, i: int) -> np.ndarray:
        lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        lf = lab[..., 0] * 2.55  # 0〜255
        l8 = np.clip(lf + 0.5, 0, 255).astype(np.uint8)
        if self.method == "clahe":
            new = self.clahe.apply(l8).astype(np.float32)
            delta = new - l8.astype(np.float32)
        else:
            lut = self.lut(l8, i)
            delta = np.interp(lf, np.arange(256), lut).astype(np.float32) - lf
        lab[..., 0] = np.clip(lab[..., 0] + delta / 2.55 * self.strength, 0, 100)
        out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def _contrast_stats(img: np.ndarray) -> dict[str, float]:
    """明るさ（Y'）の標準偏差（RMS コントラスト）と、1〜99 パーセンタイルの幅。"""
    y = _luma8(img)
    p1, p99 = np.percentile(y, [1, 99])
    return {"rms": float(y.std()), "p1": float(p1), "p99": float(p99)}


@command(
    49,
    "equalize",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_EQ),
        default="clahe",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _EQ.items()),
    ),
    arg(
        "--clip",
        type=float,
        default=2.0,
        help="CLAHE の伸ばしすぎの上限（既定 2.0。1.5 で控えめ、3〜4 で強め）",
    ),
    arg(
        "--tiles",
        type=int,
        default=8,
        help="CLAHE の区画の数（縦横それぞれ。既定 8）",
    ),
    arg(
        "--strength",
        type=float,
        default=1.0,
        help="効き具合（0〜1。元と混ぜる割合。global は 0.5〜0.7 にすると自然）",
    ),
    arg(
        "--smooth",
        type=float,
        default=0.5,
        help="global の動画で、明るさの分布をならす時間（秒。既定 0.5。0 で毎コマ別に計算）",
    ),
)
def equalize(a: argparse.Namespace) -> None:
    """明るさの偏りをならしてコントラストを整える（色は保ち、明るさ L だけを変える）。"""
    info = _probe(a.src)
    core.require(0 <= a.strength <= 1, "--strength は 0〜1 にしてください")
    core.require(
        a.clip > 0 and a.tiles >= 1, "--clip と --tiles は正の数にしてください"
    )
    out, json_path, sheet_path = _paths(a, info, "equalize")
    cuts: list[int] = []
    alpha = 0.0
    if not info.is_image and a.method == "global" and a.smooth > 0:
        cuts = _cuts(_small_frames(a, info, 96), info.fps)
        alpha = math.exp(-1 / (info.fps * a.smooth))
    eq = _Equalizer(a.method, a.clip, a.tiles, a.strength, alpha, set(cuts))
    count, kept = _process(a, info, out, eq)
    before = [_contrast_stats(b) for _i, b, _o in kept]
    after = [_contrast_stats(o) for _i, _b, o in kept]
    rms_b = float(np.mean([s["rms"] for s in before]))
    rms_a = float(np.mean([s["rms"] for s in after]))
    core.log(
        f"{a.method}: 明るさの標準偏差（コントラスト） {rms_b:.1f} → {rms_a:.1f}（{count} コマ）"
    )
    _i, b0, o0 = kept[len(kept) // 2]
    pairs = _pairs_sheet(
        kept, info.fps, a.start or 0.0, ("元", "平坦化後"), info.is_image
    )
    sheet = _rows([pairs, _hist_plot([b0, o0], ["元", "平坦化後"])])
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "clip": a.clip,
            "tiles": a.tiles,
            "strength": a.strength,
            "smooth_seconds": a.smooth if a.method == "global" else None,
            "cuts": [round((a.start or 0.0) + c / info.fps, 4) for c in cuts],
            "frames": count,
            "check_frames": [
                {"index": i, "before": sb, "after": sa}
                for (i, _b, _o), sb, sa in zip(kept, before, after, strict=True)
            ],
        },
    )


# ── 35 ホワイトバランス ───────────────────────────────────────────────────

_WB = {
    "grayworld": "グレーワールド（画面の平均が灰色になるように。線形の光で、白飛び・暗すぎる画素は除く）",
    "shades": "シェード・オブ・グレイ（6 乗の平均。灰色の仮定と白の仮定の中間。既定）",
    "greyedge": "グレーエッジ（輪郭の色の平均が灰色になるように。1 次の微分の 6 乗平均、σ2 のぼかし）",
    "whitepatch": "ホワイトパッチ（各色の上位 0.5%% の明るさが白になるように。白い物が写っている時に強い）",
    "learning": "OpenCV の LearningBasedWB（色の分布の特徴から照明を推定する学習済みの小さなモデル。opencv_contrib 同梱、Apache-2.0。線形の光で渡す。試験では他の方法より推定がぶれた）",
}

# 線形 sRGB（D65）↔ XYZ と、Bradford の色順応の行列（並びは R, G, B）
_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)


def _cct_xy(t: float) -> tuple[float, float]:
    """色温度（K）→ 色度 xy。4000K 以上は CIE の昼光の軌跡、未満はプランクの軌跡（Kim らの近似式）。"""
    if t >= 4000:
        if t <= 7000:
            x = -4.6070e9 / t**3 + 2.9678e6 / t**2 + 0.09911e3 / t + 0.244063
        else:
            x = -2.0064e9 / t**3 + 1.9018e6 / t**2 + 0.24748e3 / t + 0.237040
        return x, -3.000 * x * x + 2.870 * x - 0.275
    x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    if t < 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    else:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    return x, y


def _cat_matrix(src_k: float, dst_k: float) -> np.ndarray:
    """src_k の光の白を dst_k の白に移す、線形 sRGB（R, G, B）の 3×3 の行列（Bradford）。"""

    def white(k: float) -> np.ndarray:
        x, y = _cct_xy(k)
        return np.array([x / y, 1.0, (1 - x - y) / y])

    rs, rd = _BRADFORD @ white(src_k), _BRADFORD @ white(dst_k)
    cat = np.linalg.inv(_BRADFORD) @ np.diag(rd / rs) @ _BRADFORD
    return np.linalg.inv(_RGB2XYZ) @ cat @ _RGB2XYZ


def _wb_mask(lin: np.ndarray) -> np.ndarray:
    mx = lin.max(axis=2)
    return (mx < 0.92) & (lin.min(axis=2) > 0.003) & (mx > 0.02)


def _estimate_illum(img: np.ndarray, method: str) -> np.ndarray | None:
    """照明の色（線形の光の B, G, R。合計 1）を推定する。推定できなければ None。"""
    lin = _to_lin(img)
    m = _wb_mask(lin)
    if m.sum() < 64:
        return None
    if method == "grayworld":
        e = lin[m].mean(axis=0)
    elif method == "shades":
        e = np.power(np.mean(np.power(lin[m], 6), axis=0), 1 / 6)
    elif method == "greyedge":
        b = cv2.GaussianBlur(lin, (0, 0), 2.0)
        gx = cv2.Sobel(b, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(b, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
        m2 = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if m2.sum() < 64:
            return None
        e = np.power(np.mean(np.power(mag[m2], 6), axis=0), 1 / 6)
    elif method == "whitepatch":
        e = np.percentile(lin[m], 99.5, axis=0)
    elif method == "learning":
        # 学習に使われた写真（Gehler-Shi）と同じ線形の光で渡す（8 bit の sRGB で渡すと推定の一貫性が約半分に落ちた）
        wb = cv2.xphoto.createLearningBasedWB()
        wb.setRangeMaxVal(65535)
        lin16 = np.clip(lin * 65535 + 0.5, 0, 65535).astype(np.uint16)
        bal = wb.balanceWhite(lin16).astype(np.float64)
        gains = bal[m].mean(axis=0) / np.maximum(
            lin16[m].astype(np.float64).mean(axis=0), 1e-6
        )
        e = 1 / np.maximum(gains, 1e-6)
    else:
        raise core.VfxError(f"知らない方法です: {method}")
    e = np.maximum(np.asarray(e, np.float64), 1e-6)
    return e / e.sum()


def _gains_from_illum(e: np.ndarray, strength: float) -> np.ndarray:
    """照明の色 e（B, G, R）を灰色に戻す倍率。灰色の物の明るさ（Y）は変えない。"""
    gray = float(_Y709 @ e)
    g = gray / e
    return np.power(g, strength)


def _wb_apply(img: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """線形の光で色の行列（B, G, R の 3×3）をかける。3 色とも白飛びに近い所は白のまま残す。"""
    lin = _to_lin(img)
    out = lin @ mat.T.astype(np.float32)
    w = np.clip((lin.min(axis=2) - 0.80) / 0.15, 0, 1)[..., None]
    out = out * (1 - w) + lin * w
    return _to_u8(out)


def _bgr_matrix(rgb_mat: np.ndarray) -> np.ndarray:
    """R, G, B の並びの行列 → B, G, R の並び。"""
    p = [2, 1, 0]
    return rgb_mat[np.ix_(p, p)]


def _scene_illums(
    smalls: list[np.ndarray], segs: list[tuple[int, int]], method: str
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """コマごとと場面ごと（コマの推定の中央値）の照明の色。"""
    per = [_estimate_illum(f, method) for f in smalls]
    scenes: list[np.ndarray | None] = []
    for s, e in segs:
        vals = [v for v in per[s:e] if v is not None]
        if not vals:
            scenes.append(None)
            continue
        med = np.median(np.array(vals), axis=0)
        scenes.append(med / med.sum())
    return per, scenes


@command(
    35,
    "white-balance",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_WB),
        default="shades",
        help="自動の方法。" + " ／ ".join(f"{k}: {v}" for k, v in _WB.items()),
    ),
    arg(
        "--temperature",
        type=float,
        help="撮影した時の光の色温度（K）を指定して直す（例: 電球 2700〜3200、蛍光灯 4000、曇り 7000）。"
        "指定すると自動の推定は使わない",
    ),
    arg(
        "--target",
        type=float,
        default=6500,
        help="--temperature の時に合わせる白の色温度（K。既定 6500 = D65。5500 にすると少し暖かく残る）",
    ),
    arg(
        "--gains",
        help="倍率を直接指定する 'R,G,B'（線形の光にかける。例: 0.9,1,1.2）",
    ),
    arg(
        "--strength",
        type=float,
        default=1.0,
        help="直す強さ（0〜1。0.6〜0.8 にすると夕日や電球の雰囲気を少し残す）",
    ),
    arg(
        "--per",
        choices=["scene", "clip"],
        default="scene",
        help="動画で、自動の推定を場面（カット）ごとに分けるか（scene・既定）、範囲全体で 1 つにするか（clip）",
    ),
)
def white_balance(a: argparse.Namespace) -> None:
    """色かぶりを直して白い物を白にする（自動の推定・色温度の指定・倍率の指定）。線形の光で直す。"""
    info = _probe(a.src)
    core.require(0 <= a.strength <= 1, "--strength は 0〜1 にしてください")
    out, json_path, sheet_path = _paths(a, info, "white-balance")
    data: dict[str, Any] = {"source": str(a.src), "output": str(out)}
    mats: list[np.ndarray]
    segs: list[tuple[int, int]]
    n = _n_frames(a, info)
    if a.temperature or a.gains:
        if a.temperature:
            core.require(
                1667 <= a.temperature <= 25000 and 1667 <= a.target <= 25000,
                "色温度は 1667〜25000K で指定してください",
            )
            rgb = _cat_matrix(a.temperature, a.target)
            rgb = np.eye(3) * (1 - a.strength) + rgb * a.strength
            data.update(
                {
                    "mode": "temperature",
                    "temperature": a.temperature,
                    "target": a.target,
                }
            )
            core.log(
                f"色温度 {a.temperature:.0f}K の光 → {a.target:.0f}K の白に合わせる"
            )
        else:
            try:
                g = np.array([float(v) for v in a.gains.split(",")], np.float64)
            except ValueError as e:
                raise core.VfxError(
                    f"--gains は 'R,G,B' の 3 つの数で指定してください: {a.gains}"
                ) from e
            core.require(
                len(g) == 3 and (g > 0).all(), "--gains は正の数を 3 つ指定してください"
            )
            rgb = np.diag(np.power(g, a.strength))
            data.update({"mode": "gains", "gains_rgb": g.tolist()})
        mats, segs = [_bgr_matrix(rgb)], [(0, n)]
        data["matrix_rgb"] = np.round(rgb, 5).tolist()
    else:
        smalls = _small_frames(a, info, 256)
        cuts = _cuts(smalls, info.fps) if not info.is_image and a.per == "scene" else []
        segs = _segments(len(smalls), cuts)
        _per, scenes = _scene_illums(smalls, segs, a.method)
        mats, rows = [], []
        for (s, e), ill in zip(segs, scenes, strict=True):
            if ill is None:
                core.log(
                    f"  コマ {s}〜{e - 1}: 推定できる画素が少ないので、そのままにします"
                )
                mats.append(np.eye(3))
                rows.append(
                    {
                        "start": s,
                        "end": e,
                        "illuminant_rgb": None,
                        "gains_rgb": [1, 1, 1],
                    }
                )
                continue
            g = _gains_from_illum(ill, a.strength)
            mats.append(np.diag(g))
            rows.append(
                {
                    "start_frame": s,
                    "end_frame": e,
                    "start": round((a.start or 0.0) + s / info.fps, 4),
                    "illuminant_rgb": np.round(ill[::-1], 5).tolist(),
                    "gains_rgb": np.round(g[::-1], 5).tolist(),
                }
            )
            core.log(
                f"  {'画像' if info.is_image else f'場面 {len(rows)}（{rows[-1]["start"]:.2f} 秒〜）'}: "
                f"倍率 R {g[2]:.3f}・G {g[1]:.3f}・B {g[0]:.3f}"
            )
        # 統計を取った小さなコマの数と、処理するコマの数の比で、場面の境目を処理するコマの番号に直す
        scale = n / max(1, len(smalls))
        segs = [(round(s * scale), round(e * scale)) for s, e in segs]
        data.update({"mode": "auto", "method": a.method, "per": a.per, "scenes": rows})
    data["strength"] = a.strength
    bounds = [s for s, _e in segs]

    def fn(img: np.ndarray, i: int) -> np.ndarray:
        k = max(0, int(np.searchsorted(bounds, i, side="right")) - 1)
        return _wb_apply(img, mats[min(k, len(mats) - 1)])

    count, kept = _process(a, info, out, fn)
    data["frames"] = count
    for i, b, o in kept:
        lb, lo = _to_lin(b), _to_lin(o)
        mb, mo = lb[_wb_mask(lb)], lo[_wb_mask(lo)]
        if len(mb) and len(mo):
            cb, co = mb.mean(axis=0), mo.mean(axis=0)
            core.log(
                f"  確認したコマ {i}: 平均の色（R:G:B）{cb[2] / cb[1]:.3f}:1:{cb[0] / cb[1]:.3f} → "
                f"{co[2] / co[1]:.3f}:1:{co[0] / co[1]:.3f}"
            )
    pairs = _pairs_sheet(
        kept, info.fps, a.start or 0.0, ("元", "補正後"), info.is_image
    )
    _finish(out, json_path, sheet_path, pairs, data)


# ── 40 低照度補正 ─────────────────────────────────────────────────────────

_LOW = {
    "exposure": "露出を上げる（線形の光で全体に同じ倍率を掛け、明るさの中央値を --target にする。明るい所は白に向けて"
    "滑らかに丸めて白飛びを防ぐ。動画は場面ごとに 1 つの倍率でちらつかない。既定）",
    "lime": "照明の地図（各画素の R・G・B の最大を、輪郭を保ってぼかした物）で割る LIME 風の方法。暗い所ほど強く"
    "持ち上げる（光のむらが大きい夜景向け。ノイズと色が強く出やすい）",
    "gamma": "ガンマ曲線で中間を持ち上げる（明るさの中央値を --target に。48 の自動と同じ）",
}


def _rolloff(lin: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """1 を超える明るさを、R・G・B の最大で白に向けて滑らかに丸める（色の比は保つ。knee までは変えない）。"""
    m = lin.max(axis=2, keepdims=True)
    k = knee
    m2 = np.where(m <= k, m, k + (1 - k) * (1 - np.exp(-(m - k) / (1 - k))))
    return lin * (m2 / np.maximum(m, 1e-6))


def _srgb_dec(v: float) -> float:
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


class _LowLight:
    def __init__(
        self,
        method: str,
        strength: float,
        max_gain: float,
        g_auto: float,
        cuts: set[int],
        smooth: float,
        gains: list[tuple[int, float]] | None = None,
    ) -> None:
        self.method, self.strength, self.max_gain = method, strength, max_gain
        self.g_auto, self.cuts, self.smooth = g_auto, cuts, smooth
        self.gains = gains or [(0, 1.0)]
        self.prev: np.ndarray | None = None
        self.last_illum: np.ndarray | None = None

    def gain_at(self, i: int) -> float:
        starts = [s for s, _g in self.gains]
        k = max(0, int(np.searchsorted(starts, i, side="right")) - 1)
        return self.gains[k][1]

    def illum(self, f: np.ndarray, i: int) -> np.ndarray:
        """照明の地図（0〜1）。輪郭は保ち、模様はならす（ガイデッドフィルター）。動画は前のコマと混ぜる。"""
        t0 = np.ascontiguousarray(f.max(axis=2))
        r = max(2, round(0.02 * max(f.shape[:2])))
        t = cv2.ximgproc.guidedFilter(t0, t0, r, 0.01)
        if self.prev is not None and i not in self.cuts and self.smooth > 0:
            t = self.smooth * self.prev + (1 - self.smooth) * t
        self.prev = t
        return np.clip(t, 0, 1)

    def __call__(self, img: np.ndarray, i: int) -> np.ndarray:
        if self.method == "exposure":
            return _to_u8(_rolloff(_to_lin(img) * self.gain_at(i)))
        if self.method == "gamma":
            return _apply_gamma(img, self.g_auto, "rgb")
        f = img.astype(np.float32) / 255
        t = self.illum(f, i)
        floor = self.max_gain ** (-1 / self.strength)
        gain = 1 / np.power(np.maximum(t, floor), self.strength)
        self.last_illum = t
        return np.clip(f * gain[..., None] * 255 + 0.5, 0, 255).astype(np.uint8)


def _exposure_gains(
    smalls: list[np.ndarray],
    cuts: list[int],
    n: int,
    target: float,
    max_gain: float,
) -> list[tuple[int, float, float]]:
    """場面ごとの露出の倍率。(処理するコマでの始まり, 倍率, 明るさの中央値（線形）)"""
    tl = _srgb_dec(target)
    meds = [float(np.median(_to_lin(f) @ _Y709)) for f in smalls]
    scale = n / max(1, len(smalls))
    out = []
    for s, e in _segments(len(smalls), cuts):
        med = max(float(np.median(meds[s:e])), 1e-4)
        out.append((round(s * scale), float(np.clip(tl / med, 1.0, max_gain)), med))
    return out


def _hqdn3d(sigma: float, c: float) -> str:
    """動画のノイズ除去（ffmpeg の hqdn3d。空間 c·σ・時間 1.5c·σ）。"""
    ls = float(np.clip(c * sigma, 0.5, 20))
    return (
        f"hqdn3d=luma_spatial={ls:.2f}:chroma_spatial={0.8 * ls:.2f}:"
        f"luma_tmp={1.5 * ls:.2f}:chroma_tmp={1.2 * ls:.2f}"
    )


@command(
    40,
    "low-light",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_LOW),
        default="exposure",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _LOW.items()),
    ),
    arg(
        "--target",
        type=float,
        default=0.45,
        help="exposure・gamma: 明るさ（sRGB の信号、0〜1）の中央値をこの値にする（既定 0.45。夜らしさを残すなら 0.3〜0.35）",
    ),
    arg(
        "--strength",
        type=float,
        default=0.6,
        help="lime の強さ（0.3〜1。照明の地図を何乗して割るか。1 で暗さをすべて取り除く。既定 0.6）",
    ),
    arg(
        "--max-gain",
        type=float,
        default=16.0,
        help="明るくする倍率の上限（既定 16 = 4 段。真っ暗な所のノイズを持ち上げすぎない）",
    ),
    arg(
        "--denoise",
        default="auto",
        help="ノイズ除去: auto（明るくした後のノイズを測って決める。既定）／ none ／ 数値（ノイズの強さ σ、0〜255）",
    ),
    arg(
        "--denoise-strength",
        type=float,
        default=2.0,
        help="ノイズ除去の強さ（測ったノイズ σ の何倍で消すか。既定 2。細部を残したい時は 1〜1.5）",
    ),
)
def low_light(a: argparse.Namespace) -> None:
    """暗く写った場面を、明るく見えるようにする（露出を上げて明るい所は丸め、増えたノイズを消す）。"""
    info = _probe(a.src)
    core.require(0.1 <= a.strength <= 1.0, "--strength は 0.1〜1 にしてください")
    core.require(a.max_gain >= 1, "--max-gain は 1 以上にしてください")
    core.require(0 < a.target < 1, "--target は 0 と 1 の間にしてください")
    out, json_path, sheet_path = _paths(a, info, "low-light")
    smalls = _small_frames(a, info, 256)
    n = _n_frames(a, info)
    cuts = _cuts(smalls, info.fps) if not info.is_image else []
    g_auto, med, _ = _auto_gamma(smalls, a.target)
    gains = _exposure_gains(smalls, cuts, n, a.target, a.max_gain)
    if a.method == "exposure":
        for s, g, m in gains:
            where = (
                "画像"
                if info.is_image
                else f"{(a.start or 0.0) + s / info.fps:.2f} 秒〜"
            )
            core.log(
                f"  {where}: 明るさの中央値（線形）{m:.4f} → 倍率 {g:.2f}（{math.log2(g):.2f} 段）"
            )
    elif a.method == "gamma":
        core.log(f"  ガンマ {g_auto:.3f}（明るさの中央値 {med:.3f} → {a.target}）")
    proc = _LowLight(
        a.method,
        a.strength,
        a.max_gain,
        g_auto,
        {round(c * n / max(1, len(smalls))) for c in cuts},
        0.5,
        [(s, g) for s, g, _m in gains],
    )
    # ノイズの強さ: 明るくした後のコマで測る（動画は 5 コマの中央値）
    if a.denoise == "none":
        sigma = 0.0
    elif a.denoise == "auto":
        probe_frames = _small_frames(a, info, max(_proc_size(info, a.max_side)), 5)
        tmp = _LowLight(
            a.method, a.strength, a.max_gain, g_auto, set(), 0.0, [(0, gains[0][1])]
        )
        vals = []
        for k, f in enumerate(probe_frames):
            tmp.gains = [(0, proc.gain_at(round(k * n / max(1, len(probe_frames)))))]
            vals.append(_noise_sigma(tmp(f, 0)))
        sigma = float(np.median(vals))
    else:
        try:
            sigma = float(a.denoise)
        except ValueError as e:
            raise core.VfxError(
                f"--denoise は auto・none・数値のどれかです: {a.denoise}"
            ) from e
    use_dn = sigma >= 1.0 and a.denoise_strength > 0
    vf_post = (
        [_hqdn3d(sigma, a.denoise_strength)] if use_dn and not info.is_image else None
    )
    h_nlm = float(np.clip(a.denoise_strength * sigma, 1, 40))

    def fn(img: np.ndarray, i: int) -> np.ndarray:
        res = proc(img, i)
        if use_dn and info.is_image:
            res = cv2.fastNlMeansDenoisingColored(res, None, h_nlm, h_nlm, 7, 21)
        return res

    count, kept = _process(a, info, out, fn, vf_post=vf_post)
    if vf_post and not info.is_image:
        # 確認画像の「後」は、ノイズ除去まで済んだ書き出し後のコマを使う
        kept = [
            (
                i,
                b,
                core.frame_at(
                    out, max(0.0, (i - 0.5) / info.fps), size=(b.shape[1], b.shape[0])
                ),
            )
            for i, b, _o in kept
        ]
    yb = float(np.mean([np.median(_luma8(b)) for _i, b, _o in kept]))
    ya = float(np.mean([np.median(_luma8(o)) for _i, _b, o in kept]))
    nb = float(np.mean([_noise_sigma(b) for _i, b, _o in kept]))
    na = float(np.mean([_noise_sigma(o) for _i, _b, o in kept]))
    dn = (
        "なし"
        if not use_dn
        else (f"hqdn3d（σ {sigma:.1f}）" if vf_post else f"非局所平均（h {h_nlm:.1f}）")
    )
    core.log(
        f"{a.method}: 明るさの中央値 {yb:.1f} → {ya:.1f}、ノイズ σ {nb:.2f} → {na:.2f}"
        f"（ノイズ除去 {dn}、{count} コマ）"
    )
    pairs = _pairs_sheet(
        kept, info.fps, a.start or 0.0, ("元", "補正後"), info.is_image
    )
    _i, b0, o0 = kept[len(kept) // 2]
    sheet = _rows([pairs, _hist_plot([b0, o0], ["元", "補正後"])])
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "target": a.target,
            "strength": a.strength if a.method == "lime" else None,
            "max_gain": a.max_gain,
            "exposure_gains": [
                {
                    "start": round((a.start or 0.0) + s / info.fps, 4),
                    "gain": round(g, 4),
                    "median_linear": round(m, 5),
                }
                for s, g, m in gains
            ]
            if a.method == "exposure"
            else None,
            "auto_gamma": round(g_auto, 4) if a.method == "gamma" else None,
            "noise_sigma_after_enhance": round(sigma, 3),
            "denoise": (
                vf_post[0]
                if vf_post
                else (f"nlmeans h={h_nlm:.2f}" if use_dn else None)
            ),
            "cuts": [round((a.start or 0.0) + c / info.fps, 4) for c in cuts],
            "frames": count,
            "check": {
                "median_luma_before": round(yb, 2),
                "median_luma_after": round(ya, 2),
                "noise_before": round(nb, 3),
                "noise_after": round(na, 3),
            },
        },
    )


# ── 47 周辺減光の補正 ─────────────────────────────────────────────────────


def _radial_grid(h: int, w: int) -> np.ndarray:
    """中心（w/2, h/2）からの距離 ÷ 対角の半分（四隅で 1）。ffmpeg の vignette と同じ測り方。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.hypot(xx - w / 2, yy - h / 2) / math.hypot(w / 2, h / 2)


def _radial_samples(img: np.ndarray, side: int = 360) -> tuple[np.ndarray, np.ndarray]:
    """明るさの対数の、中心から外へ向かう向きの傾き（半径 1 あたり）を画素ごとに求める。(半径, 傾き)"""
    h0, w0 = img.shape[:2]
    f = min(1.0, side / max(h0, w0))
    w, h = max(8, round(w0 * f)), max(8, round(h0 * f))
    small = cv2.resize(img[:, :, :3], (w, h), interpolation=cv2.INTER_AREA)
    y = _luma8(small) / 255
    lg = np.log(np.maximum(y, 1 / 255))
    lg = cv2.GaussianBlur(lg, (0, 0), 1.0)
    gx = cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3) / 8
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx + 0.5 - w / 2, yy + 0.5 - h / 2
    rr = np.hypot(dx, dy)
    big = math.hypot(w / 2, h / 2)
    g = (gx * dx + gy * dy) / np.maximum(rr, 1e-6) * big
    r = rr / big
    m = (y > 0.04) & (y < 0.96) & (r > 0.08)
    m[:2, :] = m[-2:, :] = False
    m[:, :2] = m[:, -2:] = False
    return r[m], g[m]


def _fit_vignette(
    r: np.ndarray, g: np.ndarray, bins: int = 16, trim: float = 0.2
) -> tuple[np.ndarray, dict[str, Any]]:
    """傾きの分布の中心（上下 trim を除いた平均）を半径ごとに求め、log V(r) = a r² + b r⁴ の傾き 2ar + 4br³ に合わせる。

    自然な画像では、外向きの傾きは平均すると 0 に近い（明るくなる所と暗くなる所が同じくらい）。周辺減光が
    あると、外に向かって暗くなる分だけ負にずれる。戻り値: (係数 [a, b], 半径ごとの値)
    """
    edges = np.linspace(0.08, 1.0, bins + 1)
    rows, xs, ys, ws = [], [], [], []
    for k in range(bins):
        sel = (r >= edges[k]) & (r < edges[k + 1])
        n = int(sel.sum())
        if n < 50:
            continue
        v = np.sort(g[sel])
        lo, hi = int(n * trim), int(n * (1 - trim))
        c = float(v[lo:hi].mean())
        rc = float(r[sel].mean())
        xs.append(rc)
        ys.append(c)
        ws.append(math.sqrt(n))
        rows.append({"r": round(rc, 4), "slope": round(c, 5), "count": n})
    if len(xs) < 4:
        raise core.VfxError(
            "周辺減光を推定できる画素が足りません（画面が一色に近い等）"
        )
    x, y, wt = np.array(xs), np.array(ys), np.array(ws)
    design = np.stack([2 * x, 4 * x**3], axis=1) * wt[:, None]
    coef, *_ = np.linalg.lstsq(design, y * wt, rcond=None)
    return coef, {"bins": rows}


def _logv(r: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """log V(r) = a r² + b r⁴ (+ c r⁶)。"""
    out = coef[0] * r**2 + coef[1] * r**4
    if len(coef) > 2:
        out = out + coef[2] * r**6
    return out


def _reference_profile(
    path: str | Path, frames: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """無地の壁・白い紙など、平らな物を撮った参照（画像か動画）から、周辺減光 log V = a r² + b r⁴ + c r⁶ を求める。

    コマを平均した明るさ（Y'）の対数を、中心からの距離だけの式に当てはめる
    （照明のむらが片側に寄っていても、半径ごとに平均されて打ち消し合う）。
    """
    info = _probe(path)
    if info.is_image:
        w, h = core.scaled_size(info.width, info.height, max_side=360)
        imgs = [_load_image(path, (w, h))[:, :, :3]]
    else:
        fps = min(info.fps, frames / max(info.duration, 1e-3))
        imgs = core.read_all(path, max_side=360, fps=fps)[:frames]
    y = np.mean([_luma8(im) for im in imgs], axis=0) / 255
    h, w = y.shape
    core.require(
        float(np.percentile(y, 99)) < 0.98,
        f"参照が白飛びしています（明るさの 99% 点が 0.98 以上）。露出を下げて撮った物を使ってください: {path}",
    )
    core.require(
        float(np.percentile(y, 1)) > 0.03,
        f"参照に暗すぎる所があります（無地の明るい面を画面いっぱいに撮った物を使ってください）: {path}",
    )
    # 軽くぼかすだけにする（強くぼかすと、端で折り返した値が混ざって四隅が明るめに出た: 3% のぼかしで +9%）。
    # ノイズは、全画素で半径の式に当てはめる時に平均されて消える
    y = cv2.GaussianBlur(y, (0, 0), 1.0)
    r = _radial_grid(h, w).ravel()
    lg = np.log(np.maximum(y, 1e-4)).ravel()
    design = np.stack([np.ones_like(r), r**2, r**4, r**6], axis=1)
    sol, *_ = np.linalg.lstsq(design, lg, rcond=None)
    resid = lg - design @ sol
    return sol[1:], {
        "frames_used": len(imgs),
        "fit_rms_log": round(float(np.sqrt(np.mean(resid**2))), 5),
    }


def _gain_map(
    h: int, w: int, coef: np.ndarray, strength: float, max_gain: float
) -> np.ndarray:
    r = _radial_grid(h, w)
    gain = np.exp(-strength * _logv(r, coef))
    return np.clip(gain, 1 / max_gain, max_gain).astype(np.float32)


@command(
    47,
    "devignette",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--angle",
        type=float,
        help="手動: ffmpeg の vignette と同じ型（明るさ = cos⁴(角度 × 中心からの距離)、距離は四隅で 1）の減光を、"
        "この角度（度。ffmpeg の既定は 36）として打ち消す",
    ),
    arg(
        "--reference",
        help="参照: 同じカメラ・レンズ・ズームで、無地の壁や白い紙を画面いっぱいに撮った画像か動画。ここから減光を"
        "測って打ち消す（いちばん確実）",
    ),
    arg(
        "--strength",
        type=float,
        default=1.0,
        help="打ち消す強さ（0〜1.5。既定 1）",
    ),
    arg(
        "--max-gain",
        type=float,
        help="四隅を明るくする倍率の上限（既定: 自動 2、--angle・--reference 4）",
    ),
    arg(
        "--frames",
        type=int,
        default=32,
        help="動画（と参照の動画）で推定に使うコマの数（等間隔に選ぶ。既定 32）",
    ),
    arg(
        "--min-gain",
        type=float,
        default=1.02,
        help="自動で推定した四隅の倍率がこれ未満なら、補正しない（既定 1.02）",
    ),
)
def devignette(a: argparse.Namespace) -> None:
    """四隅が暗くなる周辺減光を打ち消す（参照の映像・ffmpeg の vignette と同じ型の角度・映像からの自動の推定）。"""
    info = _probe(a.src)
    core.require(0 < a.strength <= 1.5, "--strength は 0〜1.5 にしてください")
    core.require(
        a.angle is None or a.reference is None,
        "--angle と --reference はどちらか 1 つにしてください",
    )
    out, json_path, sheet_path = _paths(a, info, "devignette")
    data: dict[str, Any] = {"source": str(a.src), "output": str(out)}
    coef: np.ndarray | None = None
    if a.angle is not None:
        core.require(0 < a.angle < 90, "--angle は 0〜90 度にしてください")
        mode = "angle"
        data["angle_deg"] = a.angle
        corner = math.cos(math.radians(a.angle)) ** 4
        core.log(f"指定: 角度 {a.angle:g} 度（四隅の明るさ {corner:.3f} 倍）を打ち消す")
    elif a.reference:
        mode = "reference"
        coef, detail = _reference_profile(a.reference, a.frames)
        corner = math.exp(float(_logv(np.array(1.0), coef)))
        data.update(
            {
                "reference": str(a.reference),
                "coef_log_r2_r4_r6": np.round(coef, 5).tolist(),
                **detail,
            }
        )
        core.log(
            f"参照から: 四隅の明るさは中心の {corner:.3f} 倍（{detail['frames_used']} コマ、"
            f"式との差 {detail['fit_rms_log']:.4f}）"
        )
    else:
        mode = "auto"
        smalls = _small_frames(a, info, 360, a.frames)
        rs, gs = zip(*[_radial_samples(f) for f in smalls], strict=True)
        coef, detail = _fit_vignette(np.concatenate(rs), np.concatenate(gs))
        corner = math.exp(float(coef[0] + coef[1]))
        rr = np.linspace(0, 1, 51)
        prof = _logv(rr, coef)
        monotone = bool(np.all(np.diff(prof) <= 1e-4))
        data.update(
            {
                "frames_used": len(smalls),
                "coef_log_r2_r4": np.round(coef, 5).tolist(),
                "monotone": monotone,
                "radial": detail["bins"],
            }
        )
        core.log(
            f"自動の推定: 四隅の明るさは中心の {corner:.3f} 倍（log V = {coef[0]:.4f}·r² + {coef[1]:.4f}·r⁴、"
            f"{len(smalls)} コマ）"
        )
        core.log(
            "  注意: 自動の推定は、写っている物の明るさの偏り（中央が明るく周りが暗い構図など）でも"
            "「減光」と見なします。確実なのは --reference（無地の面を撮った物）か --angle です"
        )
        if not monotone:
            core.log(
                "  推定した明るさが外に向かって単調に下がらないので、周辺減光ではないと見なします"
            )
            corner = 1.0
    max_gain = a.max_gain or (2.0 if mode == "auto" else 4.0)
    skip = mode == "auto" and (1 / corner) < a.min_gain
    if skip:
        core.log(
            f"四隅の倍率 {1 / corner:.3f} が --min-gain {a.min_gain} 未満なので、補正しません"
        )
    gain_corner = min(max_gain, (1 / corner) ** a.strength) if not skip else 1.0
    if not skip and (1 / corner) ** a.strength > max_gain:
        core.log(
            f"  四隅の倍率 {(1 / corner) ** a.strength:.2f} を上限 {max_gain:g} で止めます（--max-gain）"
        )
    data.update(
        {
            "mode": mode,
            "corner_brightness": round(corner, 4),
            "corner_gain": round(gain_corner, 4),
            "max_gain": max_gain,
            "strength": a.strength,
            "applied": not skip,
        }
    )
    gain_cache: dict[tuple[int, int], np.ndarray] = {}

    def gain_for(h: int, w: int) -> np.ndarray:
        if (h, w) not in gain_cache:
            if coef is None:
                r = _radial_grid(h, w)
                v = (
                    np.cos(np.clip(math.radians(a.angle) * r, 0, math.pi / 2 - 1e-3))
                    ** 4
                )
                gm = np.clip(np.power(1 / v, a.strength), 1 / max_gain, max_gain)
                gain_cache[(h, w)] = gm.astype(np.float32)
            else:
                gain_cache[(h, w)] = _gain_map(h, w, coef, a.strength, max_gain)
        return gain_cache[(h, w)]

    def fn(img: np.ndarray, _i: int) -> np.ndarray:
        if skip:
            return img
        gm = gain_for(*img.shape[:2])
        return np.clip(img.astype(np.float32) * gm[..., None] + 0.5, 0, 255).astype(
            np.uint8
        )

    count, kept = _process(a, info, out, fn)
    data["frames"] = count

    def ring_ratio(img: np.ndarray) -> float:
        h, w = img.shape[:2]
        r = _radial_grid(h, w)
        y = _luma8(img)
        return float(np.median(y[r > 0.85]) / max(1e-6, np.median(y[r < 0.3])))

    rb = float(np.mean([ring_ratio(b) for _i, b, _o in kept]))
    ra = float(np.mean([ring_ratio(o) for _i, _b, o in kept]))
    clip_pct = float(
        np.mean(
            [
                (o.max(axis=2) >= 255).mean() - (b.max(axis=2) >= 255).mean()
                for _i, b, o in kept
            ]
        )
        * 100
    )
    data["corner_center_ratio"] = {"before": round(rb, 4), "after": round(ra, 4)}
    data["new_clipped_pct"] = round(clip_pct, 3)
    core.log(
        f"四隅（半径 0.85 以上）÷ 中央（0.3 未満）の明るさの中央値: {rb:.3f} → {ra:.3f}（{count} コマ）"
    )
    pairs = _pairs_sheet(
        kept, info.fps, a.start or 0.0, ("元", "補正後"), info.is_image
    )
    rr = np.linspace(0, 1, 101)
    if coef is None:
        prof = np.cos(np.clip(math.radians(a.angle) * rr, 0, math.pi / 2)) ** 4
    else:
        prof = np.exp(_logv(rr, coef))
    names = {
        "angle": "指定した減光",
        "reference": "参照から測った減光",
        "auto": "自動で推定した減光",
    }
    curve = _plot(
        [
            (rr, prof, (40, 40, 220), names[mode]),
            (
                rr,
                np.minimum(1 / np.maximum(prof, 1e-6), max_gain) ** a.strength
                / max_gain,
                (40, 150, 40),
                f"かける倍率 ÷ {max_gain:g}",
            ),
        ],
        title="中心からの距離と明るさ（中心 = 1、四隅の距離 = 1）",
        xlabel="距離",
        ylim=(0, 1.2),
        size=(560, 360),
    )
    _finish(out, json_path, sheet_path, _rows([pairs, curve], width=1400), data)


# ── HDR の共通部分（32・33） ───────────────────────────────────────────────

# zscale の名前（ffprobe の印 → zscale の値）
_Z_PRIM = {"bt709": "709", "smpte170m": "170m", "bt470bg": "bt470bg", "bt2020": "2020"}
_Z_TRC_SDR = {
    "iec61966-2-1": "iec61966-2-1"
}  # それ以外の SDR の伝達特性は 709（BT.1886 の 2.4 乗）で読む


def _side_data(path: str | Path) -> dict[str, float | None]:
    """最初のコマの HDR の情報（MaxCLL・MaxFALL・マスタリング表示の最大の明るさ）。"""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals"]
    cmd += ["%+#1", "-show_frames", "-show_entries", "frame=side_data_list"]
    cmd += ["-of", "json", str(path)]
    res: dict[str, float | None] = {
        "max_cll": None,
        "max_fall": None,
        "mastering_max": None,
    }
    try:
        frames = json.loads(core.run(cmd).stdout).get("frames", [])
    except (core.VfxError, json.JSONDecodeError):
        return res
    for sd in frames[0].get("side_data_list", []) if frames else []:
        kind = sd.get("side_data_type", "")
        if kind == "Content light level metadata":
            res["max_cll"] = float(sd.get("max_content") or 0) or None
            res["max_fall"] = float(sd.get("max_average") or 0) or None
        elif kind == "Mastering display metadata" and sd.get("max_luminance"):
            num, _, den = str(sd["max_luminance"]).partition("/")
            res["mastering_max"] = float(num) / float(den or 1)
    return res


def _pq(nits: np.ndarray) -> np.ndarray:
    """明るさ（nit）→ PQ の信号（0〜1。SMPTE ST 2084）。"""
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    y = np.clip(np.asarray(nits, np.float64) / 10000, 0, 1) ** m1
    return ((c1 + c2 * y) / (1 + c3 * y)) ** m2


def _hlg(nits: np.ndarray) -> np.ndarray:
    """表示の明るさ（灰色、nit。表示の最大 1000 nit）→ HLG の信号（BT.2100。OOTF のガンマ 1.2 を戻してから OETF）。"""
    e = np.clip(np.asarray(nits, np.float64) / 1000, 0, 1) ** (1 / 1.2)
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    return np.where(
        e <= 1 / 12, np.sqrt(3 * e), a * np.log(np.maximum(12 * e - b, 1e-12)) + c
    )


def _decode_nits(
    path: str | Path, t: float, size: tuple[int, int] | None, is_image: bool
) -> np.ndarray:
    """HDR の映像の 1 コマを、表示の明るさ（nit。R, G, B の順、BT.2020）で読む（zscale で変換）。"""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
    if not is_image and t > 0:
        cmd += ["-ss", f"{t:.4f}"]
    cmd += ["-i", str(path), "-frames:v", "1"]
    vf = "zscale=t=linear:npl=1:p=2020,format=gbrpf32le"
    info = core.probe(path)
    w, h = size or (info.width, info.height)
    if (w, h) != (info.width, info.height):
        vf += f",zscale=w={w}:h={h}:f=bilinear"
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-"]
    r = subprocess.run(cmd, capture_output=True, check=False)
    core.require(
        r.returncode == 0 and len(r.stdout) >= w * h * 12,
        f"HDR のコマを読めません: {path}\n{r.stderr.decode()[-600:]}",
    )
    g, b, rr = np.frombuffer(r.stdout[: w * h * 12], np.float32).reshape(3, h, w)
    return np.dstack([rr, g, b])


def _nits_map(rgb_nits: np.ndarray) -> np.ndarray:
    """明るさ（nit）を色分けした画像: 100 未満は灰色の濃淡、100〜203 緑、203〜400 黄、400〜600 橙、600〜1000 赤、1000 以上 紫。"""
    y = rgb_nits @ np.array([0.2627, 0.6780, 0.0593], np.float32)
    gray = np.clip(np.log10(np.maximum(y, 0.1)) / 3, 0, 1)
    out = np.dstack([gray * 200] * 3).astype(np.float32)
    bands = [
        (100, 203, (60, 190, 60)),
        (203, 400, (30, 220, 240)),
        (400, 600, (20, 140, 250)),
        (600, 1000, (40, 40, 230)),
        (1000, 1e9, (200, 60, 200)),
    ]
    for lo, hi, color in bands:
        m = (y >= lo) & (y < hi)
        out[m] = out[m] * 0.35 + np.array(color, np.float32) * 0.65
    return np.clip(out, 0, 255).astype(np.uint8)


def _nits_legend(width: int) -> np.ndarray:
    img = np.full((40, width, 3), 255, np.uint8)
    items = [
        ("〜100 nit（濃淡）", (150, 150, 150)),
        ("100〜203", (60, 190, 60)),
        ("203〜400", (30, 220, 240)),
        ("400〜600", (20, 140, 250)),
        ("600〜1000", (40, 40, 230)),
        ("1000〜", (200, 60, 200)),
    ]
    x = 8
    for name, color in items:
        cv2.rectangle(img, (x, 12), (x + 18, 30), color, -1)
        core.draw_text(img, name, (x + 24, 11), size=15, color=(30, 30, 30), bg=None)
        x += 30 + int(len(name) * 11.5)
    return img


# ── 33 トーンマッピング ───────────────────────────────────────────────────

_TM_HDR = {
    "mobius": "メビウス曲線（ffmpeg の tonemap。--param までの暗い所はそのまま、それより明るい所を滑らかに縮める。HDR の既定）",
    "hable": "Hable（映画風のフィルム曲線。全体に暗めで、明るい所の粘りが強い）",
    "reinhard": "Reinhard（x/(x+--param)。素直に縮める）",
    "clip": "そのまま切る（--white より明るい所は白になる。SDR を入れただけの HDR を戻す時に）",
    "linear": "全体を同じ割合で暗くして最大の明るさを白に合わせる",
}
_TM_SDR = {
    "local": "局所的な調整（明るさを輪郭を保ってぼかした「土台」と細部に分け、土台だけを中間の灰色に寄せる曲線で縮めて"
    "細部を戻す。黒と白の端は動かさない。SDR の既定）",
    "fusion": "露出合成（明るさを ±強さ×2 段変えた 3 枚を、OpenCV の Mertens の方法でよく写った所を選んで合成する）",
    "clahe": "明るさの CLAHE（clip 1.5。49 と同じ方法の控えめな設定）",
}


def _tm_hdr_chain(
    kind: str,
    method: str,
    white: float,
    peak: float,
    param: float | None,
    desat: float,
    size: tuple[int, int] | None,
    image: bool,
) -> list[str]:
    """HDR → SDR の ffmpeg のフィルターの並び（zscale で光の強さに戻し、BT.709 にして tonemap、SDR の信号へ）。"""
    chain = [f"zscale=t=linear:npl={white:g}", "format=gbrpf32le"]
    if size is not None:
        chain.append(f"zscale=w={size[0]}:h={size[1]}:f=bicubic")
    chain.append("zscale=p=709")
    tm = f"tonemap=tonemap={method}:desat={desat:g}:peak={peak / white:.5f}"
    if param is not None:
        tm += f":param={param:g}"
    chain.append(tm)
    if image:
        chain += ["zscale=t=iec61966-2-1:d=error_diffusion", "format=gbrp"]
    else:
        chain += ["zscale=t=709:m=709:r=tv:d=error_diffusion", "format=yuv420p"]
    return chain


class _LocalTM:
    """SDR の局所的なトーンマッピング（土台と細部に分けて、土台だけを縮める）。"""

    def __init__(self, strength: float, detail: float) -> None:
        self.strength, self.detail = strength, detail

    @staticmethod
    def lstar(y: np.ndarray) -> np.ndarray:
        """線形の明るさ Y（0〜1）→ L*（0〜1 に縮めた値）。"""
        d = 6 / 29
        f = np.where(y > d**3, np.cbrt(np.maximum(y, 0)), y / (3 * d * d) + 4 / 29)
        return (116 * f - 16) / 100

    @staticmethod
    def lstar_inv(lv: np.ndarray) -> np.ndarray:
        d = 6 / 29
        f = (lv * 100 + 16) / 116
        return np.where(f > d, f**3, 3 * d * d * (f - 4 / 29))

    def curve(self, b: np.ndarray) -> np.ndarray:
        """土台の明るさ b（L*、0〜1）を中間（0.5）に寄せる。0 と 1 は動かさず、単調に増える（強さ 1 まで）。"""
        return b + self.strength * (0.5 - b) * 4 * b * (1 - b)

    def __call__(self, img: np.ndarray, _i: int) -> np.ndarray:
        lin = _to_lin(img)
        y = lin @ _Y709
        lv = self.lstar(y).astype(np.float32)
        r = max(2, round(0.03 * max(img.shape[:2])))
        base = cv2.ximgproc.guidedFilter(lv, lv, r, 0.01)
        new = np.clip(self.curve(np.clip(base, 0, 1)) + (lv - base) * self.detail, 0, 1)
        y2 = self.lstar_inv(new)
        ratio = ((y2 + 1e-4) / (y + 1e-4)).astype(np.float32)
        out = lin * ratio[..., None]
        mx = out.max(axis=2, keepdims=True)
        out = np.where(mx > 1, out / np.maximum(mx, 1e-6), out)
        return _to_u8(out)


class _FusionTM:
    def __init__(self, strength: float) -> None:
        self.ev = 2.0 * strength
        self.merge = cv2.createMergeMertens()

    def __call__(self, img: np.ndarray, _i: int) -> np.ndarray:
        lin = _to_lin(img)
        stack = [_to_u8(lin * 2.0**ev) for ev in (-self.ev, 0.0, self.ev)]
        fused = self.merge.process(stack)
        return np.clip(fused * 255 + 0.5, 0, 255).astype(np.uint8)


def _clip_stats(img: np.ndarray) -> dict[str, float]:
    """白飛び（Y' 250 以上）・黒つぶれ（5 以下）の画素の割合（%）と、細部の強さ（ラプラシアンの絶対値の平均）。"""
    y = _luma8(img)
    lap = np.abs(cv2.Laplacian(y, cv2.CV_32F))
    return {
        "white_pct": round(float((y >= 250).mean() * 100), 3),
        "black_pct": round(float((y <= 5).mean() * 100), 3),
        "detail": round(float(lap.mean()), 3),
        "mean": round(float(y.mean()), 2),
    }


@command(
    33,
    "tonemap",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=["auto", *_TM_HDR, *_TM_SDR],
        default="auto",
        help="方法（auto: HDR の入力は mobius、SDR は local）。HDR → SDR: "
        + " ／ ".join(f"{k}: {v}" for k, v in _TM_HDR.items())
        + "。SDR の白飛び・黒つぶれを抑える: "
        + " ／ ".join(f"{k}: {v}" for k, v in _TM_SDR.items()),
    ),
    arg(
        "--white",
        type=float,
        default=203.0,
        help="HDR → SDR: SDR の白（100%%）にする HDR の明るさ（nit。既定 203 = ITU-R BT.2408 の基準の白）",
    ),
    arg(
        "--peak",
        type=float,
        help="HDR → SDR: 入力の最大の明るさ（nit）。省略時は映像の情報（MaxCLL・マスタリング表示）、無ければ 1000",
    ),
    arg(
        "--param",
        type=float,
        help="HDR → SDR の曲線の調整値（mobius: そのまま残す明るさの上限、--white = 1 として。既定 0.5 = 約 100 nit。"
        "ffmpeg の既定は 0.3。reinhard: 既定 0.5）",
    ),
    arg(
        "--desat",
        type=float,
        default=0.0,
        help="HDR → SDR: 明るい所の色を薄める強さ（ffmpeg の tonemap の desat。既定 0 = 薄めない）",
    ),
    arg(
        "--strength",
        type=float,
        default=0.4,
        help="SDR: 白飛び・黒つぶれを抑える強さ（0〜1。既定 0.4）",
    ),
    arg(
        "--detail",
        type=float,
        default=1.2,
        help="SDR の local: 細部の強さ（既定 1.2。土台を縮めると細部も弱く見えるので少し強める。1.0 でそのまま）",
    ),
)
def tonemap(a: argparse.Namespace) -> None:
    """明るさの幅を、ふつうの画面で見やすく収める（HDR の映像を SDR に変換／SDR の白飛び・黒つぶれを局所的に抑える）。"""
    info = _probe(a.src)
    kind = _hdr_kind(info)
    method = a.method
    if method == "auto":
        method = "mobius" if kind else "local"
    if kind is None and method in _TM_HDR:
        raise core.VfxError(
            f"--method {method} は HDR（PQ・HLG）の入力用です。この入力は SDR なので local・fusion・clahe を使ってください"
        )
    if kind is not None and method in _TM_SDR:
        raise core.VfxError(
            f"入力は HDR（{kind.upper()}）です。--method {method} は SDR 用なので、mobius・hable などで SDR にしてから使ってください"
        )
    out, json_path, sheet_path = _paths(a, info, "tonemap")
    data: dict[str, Any] = {"source": str(a.src), "output": str(out), "method": method}
    if kind is not None:
        _tonemap_hdr(a, info, kind, method, out, json_path, sheet_path, data)
        return
    core.require(0 <= a.strength <= 1, "--strength は 0〜1 にしてください")
    if method == "local":
        fn: Callable[[np.ndarray, int], np.ndarray] = _LocalTM(a.strength, a.detail)
    elif method == "fusion":
        fn = _FusionTM(a.strength)
    else:
        fn = _Equalizer("clahe", 1.5, 8, 1.0, 0.0, set())  # 49 と同じ部品
    count, kept = _process(a, info, out, fn)
    before = [_clip_stats(b) for _i, b, _o in kept]
    after = [_clip_stats(o) for _i, _b, o in kept]

    def avg(rows: list[dict[str, float]], k: str) -> float:
        return float(np.mean([r[k] for r in rows]))

    core.log(
        f"{method}（強さ {a.strength}）: 白飛び（Y' 250 以上）{avg(before, 'white_pct'):.2f}% → {avg(after, 'white_pct'):.2f}%、"
        f"黒つぶれ（5 以下）{avg(before, 'black_pct'):.2f}% → {avg(after, 'black_pct'):.2f}%、"
        f"細部の強さ {avg(before, 'detail'):.2f} → {avg(after, 'detail'):.2f}（{count} コマ）"
    )
    data.update(
        {
            "strength": a.strength,
            "detail": a.detail if method == "local" else None,
            "frames": count,
            "check_frames": [
                {"index": i, "before": sb, "after": sa}
                for (i, _b, _o), sb, sa in zip(kept, before, after, strict=True)
            ],
        }
    )
    marks = []
    for _i, b, o in kept:
        for im in (b, o):
            y = _luma8(im)
            mk = im.copy()
            mk[y >= 250] = (40, 40, 255)
            mk[y <= 5] = (255, 120, 0)
            marks.append(mk)
    pairs = _pairs_sheet(
        kept, info.fps, a.start or 0.0, ("元", "補正後"), info.is_image
    )
    kept_marks = [
        (i, marks[2 * k], marks[2 * k + 1]) for k, (i, _b, _o) in enumerate(kept)
    ]
    mark_sheet = _pairs_sheet(
        kept_marks,
        info.fps,
        a.start or 0.0,
        ("元: 白飛び赤・黒つぶれ青", "後: 白飛び赤・黒つぶれ青"),
        info.is_image,
    )
    _i, b0, o0 = kept[len(kept) // 2]
    sheet = _rows([pairs, mark_sheet, _hist_plot([b0, o0], ["元", "補正後"])])
    _finish(out, json_path, sheet_path, sheet, data)


def _tonemap_hdr(
    a: argparse.Namespace,
    info: core.MediaInfo,
    kind: str,
    method: str,
    out: Path,
    json_path: Path,
    sheet_path: Path,
    data: dict[str, Any],
) -> None:
    """HDR（PQ・HLG）→ SDR（BT.709）。ffmpeg の zscale と tonemap で、1 回の ffmpeg で書き出す。"""
    core.require(a.white > 0, "--white は正の数にしてください")
    side = _side_data(a.src)
    if a.peak:
        peak, peak_from = a.peak, "指定"
    elif kind == "hlg":
        peak, peak_from = 1000.0, "HLG の基準の表示（1000 nit）"
    elif side["max_cll"]:
        peak, peak_from = float(side["max_cll"]), "MaxCLL"
    elif side["mastering_max"]:
        peak, peak_from = float(side["mastering_max"]), "マスタリング表示の最大"
    else:
        peak, peak_from = 1000.0, "情報が無いので 1000 nit と仮定"
    peak = max(peak, a.white * 1.01)
    param = a.param
    if param is None and method == "mobius":
        # ffmpeg の既定 0.3（--white の 3 割 ≒ 61 nit）より上を縮めると、肌などの中間まで変わる。0.5（≒ 101 nit）までは
        # そのまま残す（32 で広げた HDR を戻す試験で 34.8 → 38.6 dB）
        param = 0.5
    w, h = _proc_size(info, a.max_side)
    size = (w, h) if (w, h) != (info.width, info.height) else None
    chain = _tm_hdr_chain(
        kind, method, a.white, peak, param, a.desat, size, info.is_image
    )
    core.log(
        f"HDR（{kind.upper()}）→ SDR: {method}、白 {a.white:g} nit、最大 {peak:g} nit（{peak_from}）"
    )
    if info.is_image:
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(a.src)]
        cmd += [
            "-frames:v",
            "1",
            "-vf",
            ",".join(chain),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        r = subprocess.run(cmd, capture_output=True, check=False)
        core.require(
            r.returncode == 0 and len(r.stdout) >= w * h * 3,
            f"変換に失敗しました:\n{r.stderr.decode()[-800:]}",
        )
        res = np.frombuffer(r.stdout[: w * h * 3], np.uint8).reshape(h, w, 3)
        core.save_image(out, res)
        count = 1
    else:
        args = [
            *_ff_in(a, info),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            ",".join(chain),
        ]
        if a.max_frames:
            args += ["-frames:v", str(a.max_frames)]
        args += [*core.MP4_ARGS, "-colorspace", "bt709", "-color_primaries", "bt709"]
        args += [
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(out),
        ]
        core.ffmpeg(args)
        count = core.probe(out).frames
    # 確認画像: HDR をそのまま（変換せずに）表示した時の見え方・明るさの地図・変換後
    n = _n_frames(a, info)
    rows = []
    for i in _keep_idx(n, 3 if not info.is_image else 1):
        t_src = (a.start or 0.0) + (i + 0.5) / info.fps if not info.is_image else 0.0
        t_out = (i + 0.5) / info.fps
        sz = (w, h)
        naive = (
            core.frame_at(a.src, t_src, size=sz)
            if not info.is_image
            else _naive_image(a.src, sz)
        )
        nits = _decode_nits(a.src, t_src, sz, info.is_image)
        after = (
            core.load_image(out)
            if info.is_image
            else core.frame_at(out, t_out, size=sz)
        )
        lab = "" if info.is_image else f" {t_src:.2f}秒"
        rows.append(
            core.contact_sheet(
                [naive, _nits_map(nits), after],
                [
                    f"HDR を変換せずに表示{lab}",
                    f"明るさ（nit）{lab}",
                    f"SDR に変換（{method}）{lab}",
                ],
                cols=3,
                cell_width=360 if h > w else 480,
            )
        )
        y = nits @ np.array([0.2627, 0.6780, 0.0593], np.float32)
        data.setdefault("check_frames", []).append(
            {
                "time": round(t_src, 4),
                "hdr_nits_p50": round(float(np.percentile(y, 50)), 2),
                "hdr_nits_p99": round(float(np.percentile(y, 99)), 2),
                "hdr_nits_max": round(float(y.max()), 2),
                "sdr_white_pct": _clip_stats(after)["white_pct"],
            }
        )
    sheet = _rows([*rows, _nits_legend(1600)])
    data.update(
        {
            "input": kind,
            "white_nits": a.white,
            "peak_nits": peak,
            "peak_from": peak_from,
            "side_data": side,
            "param": param,
            "desat": a.desat,
            "filters": ",".join(chain),
            "frames": count,
        }
    )
    _finish(out, json_path, sheet_path, sheet, data)


def _naive_image(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    """HDR の画像を、伝達特性を変換せずに RGB にした物（HDR に対応しない表示での見え方）。"""
    w, h = size
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        "1",
    ]
    cmd += ["-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    r = subprocess.run(cmd, capture_output=True, check=False)
    core.require(r.returncode == 0, f"画像を読めません: {path}")
    return np.frombuffer(r.stdout[: w * h * 3], np.uint8).reshape(h, w, 3)


# ── 32 逆トーンマッピング ─────────────────────────────────────────────────

_ITM_FMT = {
    "pq": "HDR10（BT.2020・PQ。MaxCLL・MaxFALL・マスタリング表示の情報つき。既定）",
    "hlg": "HLG（BT.2020・ARIB STD-B67。表示の最大 1000 nit を基準に変換。--peak は 1000 まで）",
}
# HDR10 のマスタリング表示（P3-D65 の色域）。x265 の master-display の書式（色度は 0.00002 単位）
_P3D65 = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)"


def _expand(y: np.ndarray, knee_lin: float, ratio: float) -> np.ndarray:
    """SDR の線形の明るさ y（白 = 1）→ HDR の明るさ（--white = 1）。

    knee_lin まではそのまま（ITU-R BT.2408 の「SDR の白 = 203 nit」の割り当て）。それより明るい所は
    y + A·(y − knee)²（つなぎ目で傾きが続く 2 次の曲線）で広げ、y = 1 で ratio（= 最大 ÷ 白）にする。
    """
    if ratio <= 1.0 + 1e-6:
        return y
    a = (ratio - 1) / (1 - knee_lin) ** 2
    d = np.maximum(y - knee_lin, 0)
    return y + a * d * d


def _itm_planes(planes: np.ndarray, knee_lin: float, ratio: float) -> np.ndarray:
    """線形の BT.2020（G, B, R の面。SDR の白 = 1）の明るさを広げる。色の比は保つ（明るさの比を 3 色に掛ける）。"""
    # SDR は白（1）より明るくならない。YUV の外れ値や色の補間の行き過ぎで 1 を超えた所は切る
    # （切らないと 2 次の曲線がそのまま伸びて、最大の明るさ --peak を超えた: story540 で 2165 nit）
    planes = np.clip(planes, 0, 1)
    g, b, r = planes
    y = 0.2627 * r + 0.6780 * g + 0.0593 * b
    ye = _expand(y, knee_lin, ratio)
    scale = np.where(y > 1e-7, ye / np.maximum(y, 1e-7), 1.0).astype(np.float32)
    return planes * scale[None]


def _itm_reader(
    a: argparse.Namespace, info: core.MediaInfo, size: tuple[int, int]
) -> Iterator[np.ndarray]:
    """SDR を、線形の光・BT.2020 の色（G, B, R の float の面。SDR の白 = 1）で順に返す（zscale で変換）。

    YUV → RGB も zscale で行う（swscale で 16 bit の RGB にすると 8 bit の 255 が 65280 になり、白が 0.9% 暗くなった。
    明るい所を広げると誤差も広がるため）。色の印の無い動画は ffmpeg の既定と同じ BT.601・TV の範囲として読む。
    """
    w, h = size
    tags = _tags(info)
    pin = _Z_PRIM.get(tags.get("color_primaries", ""), "709")
    default_trc = "iec61966-2-1" if info.is_image else "709"
    tin = _Z_TRC_SDR.get(tags.get("color_transfer", ""), default_trc)
    out_lin = "t=linear:p=2020:m=gbr:r=full"
    if info.is_image:
        img = _load_image(a.src, size)[:, :, :3]
        to_lin = f"zscale=tin={tin}:pin={pin}:min=gbr:rin=full:{out_lin}"
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-f", "rawvideo"]
        cmd += ["-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-i", "-"]
        cmd += ["-vf", f"{to_lin},format=gbrpf32le"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-"]
        r = subprocess.run(
            cmd,
            input=np.ascontiguousarray(img).tobytes(),
            capture_output=True,
            check=False,
        )
        core.require(
            r.returncode == 0, f"変換に失敗しました:\n{r.stderr.decode()[-800:]}"
        )
        yield np.frombuffer(r.stdout[: w * h * 12], np.float32).reshape(3, h, w).copy()
        return
    zin = [f"tin={tin}", f"pin={pin}"]
    if "color_space" not in tags:
        zin.append("min=170m")
    if "color_range" not in tags:
        zin.append("rin=limited")
    vf = [f"zscale={':'.join(zin)}:{out_lin}:f=bicubic", "format=gbrpf32le"]
    if (w, h) != (info.width, info.height):
        vf.append(f"zscale=w={w}:h={h}:f=bicubic")
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *_ff_in(a, info)]
    cmd += ["-an", "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-"]
    yield from _pipe_frames(cmd, (3, h, w), np.float32, a.max_frames, a.src)


def _itm_encode_cmd(
    a: argparse.Namespace,
    info: core.MediaInfo,
    out: Path,
    size: tuple[int, int],
    fmt: str,
    maxcll: tuple[int, int],
) -> list[str]:
    w, h = size
    trc = "smpte2084" if fmt == "pq" else "arib-std-b67"
    enc = f"zscale=tin=linear:pin=2020:min=gbr:rin=full:t={trc}:p=2020:m=2020_ncl:r=tv:npl={a.white:g}:d=ordered"
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-s", f"{w}x{h}"]
    tags = [
        "-color_primaries",
        "bt2020",
        "-color_trc",
        trc,
        "-colorspace",
        "bt2020nc",
        "-color_range",
        "tv",
    ]
    if info.is_image:
        cmd += ["-i", "-", "-vf", f"{enc},format=yuv444p10le", "-frames:v", "1"]
        cmd += [
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            str(a.crf),
            "-cpu-used",
            "6",
        ]
        return [*cmd, *tags, str(out)]
    cmd += ["-r", f"{info.fps:.6f}", "-i", "-"]
    if info.has_audio:
        if a.start:
            cmd += ["-ss", f"{a.start:.4f}"]
        if a.duration or a.max_frames:
            cmd += ["-t", f"{_span(a, info):.4f}"]
        cmd += ["-i", str(a.src), "-map", "0:v", "-map", "1:a", "-shortest"]
    cmd += ["-vf", f"{enc},format=yuv420p10le", "-c:v", "libx265", "-preset", a.preset]
    cmd += ["-crf", str(a.crf), "-tag:v", "hvc1", *tags]
    params = ["log-level=error", "repeat-headers=1", "info=0", "colorprim=bt2020"]
    params += [f"transfer={trc}", "colormatrix=bt2020nc", "range=limited"]
    if fmt == "pq":
        lum = max(a.peak, 1000.0)
        params += ["hdr10=1", f"master-display={_P3D65}L({round(lum * 10000)},1)"]
        params += [f"max-cll={maxcll[0]},{maxcll[1]}"]
    cmd += ["-x265-params", ":".join(params), "-movflags", "+faststart"]
    cmd += ["-c:a", "aac", "-b:a", "192k"] if info.has_audio else []
    return [*cmd, str(out)]


@command(
    32,
    "inverse-tonemap",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--format",
        choices=list(_ITM_FMT),
        default="pq",
        help="書き出す HDR の形式。"
        + " ／ ".join(f"{k}: {v}" for k, v in _ITM_FMT.items()),
    ),
    arg(
        "--peak",
        type=float,
        default=1000.0,
        help="SDR のいちばん明るい白を何 nit にするか（既定 1000。400〜1000 が目安。--white と同じにすると広げない）",
    ),
    arg(
        "--white",
        type=float,
        default=203.0,
        help="中間までの明るさの基準: SDR の白を広げない時に何 nit に置くか（既定 203 = ITU-R BT.2408）",
    ),
    arg(
        "--knee",
        type=float,
        default=0.75,
        help="広げ始める SDR の信号の高さ（0〜1。既定 0.75。これより暗い所は --white の割り当てのまま）",
    ),
    arg(
        "--crf",
        type=int,
        default=18,
        help="画質（x265・AV1 の crf。小さいほど高画質。既定 18）",
    ),
    arg(
        "--preset",
        default="medium",
        help="x265 の速さの設定（既定 medium。ultrafast〜veryslow）",
    ),
)
def inverse_tonemap(a: argparse.Namespace) -> None:
    """SDR の映像を HDR（HDR10 か HLG）に広げる。中間までは BT.2408 の割り当て、明るい所だけを --peak まで広げる。"""
    info = _probe(a.src)
    core.require(_hdr_kind(info) is None, "入力はすでに HDR です（PQ・HLG）")
    core.require(
        0 < a.white <= a.peak <= 10000, "--white ≤ --peak ≤ 10000 にしてください"
    )
    core.require(0.3 <= a.knee < 1, "--knee は 0.3〜1 未満にしてください")
    if a.format == "hlg":
        core.require(a.peak <= 1000, "HLG の --peak は 1000 nit までです")
    out, json_path, sheet_path = _paths(
        a, info, "hdr-" + a.format, video_ext=".mp4", image_ext=".avif"
    )
    if info.is_image:
        core.require(
            out.suffix.lower() == ".avif", f"画像の HDR は .avif で書き出します: {out}"
        )
    size = _proc_size(info, a.max_side)
    if not info.is_image:
        core.require(
            size[0] % 2 == 0 and size[1] % 2 == 0, "動画の大きさは偶数にしてください"
        )
    # 2.4 乗（BT.1886）か sRGB で SDR の信号 → 線形。広げ始める所を線形の明るさで
    knee_lin = a.knee**2.4
    ratio = a.peak / a.white
    core.log(
        f"SDR → HDR（{a.format.upper()}）: 信号 {a.knee:g} まで {a.white:g} nit 基準のまま、"
        f"SDR の白 → {a.peak:g} nit に広げる"
    )
    # 1 回目: HDR10 の MaxCLL・MaxFALL（3 色の最大の明るさの、画素の最大・コマの平均の最大）を測る
    maxcll = (0, 0)
    if a.format == "pq" and not info.is_image:
        mx, fall = 0.0, 0.0
        for planes in _itm_reader(a, info, size):
            m = _itm_planes(planes, knee_lin, ratio).max(axis=0) * a.white
            mx, fall = max(mx, float(m.max())), max(fall, float(m.mean()))
        maxcll = (math.ceil(mx), math.ceil(fall))
        core.log(f"  MaxCLL {maxcll[0]} nit・MaxFALL {maxcll[1]} nit")
    cmd = _itm_encode_cmd(a, info, out, size, a.format, maxcll)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    count = 0
    total = _n_frames(a, info)
    try:
        for planes in _itm_reader(a, info, size):
            proc.stdin.write(
                np.ascontiguousarray(
                    _itm_planes(planes, knee_lin, ratio), np.float32
                ).tobytes()
            )
            count += 1
            if total >= 20 and count % max(1, total // 5) == 0:
                core.log(f"  {count}/{total} コマ")
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise core.VfxError(f"書き出しに失敗しました: {out}\n{err[-800:]}")
    # 結果の確かめ: 印と、SDR の信号の高さごとの明るさ（式）
    st = _stream(_probe(out))
    sig = np.array([0.0, 0.25, 0.5, 0.6, 0.7, a.knee, 0.8, 0.85, 0.9, 0.95, 1.0])
    nits = _expand(sig**2.4, knee_lin, ratio) * a.white
    code = _pq(nits) if a.format == "pq" else _hlg(nits)
    table = [
        {
            "sdr_signal": round(float(s), 3),
            "nits": round(float(v), 2),
            "hdr_signal": round(float(c), 4),
            "code10_limited": round(64 + 876 * float(c)),
        }
        for s, v, c in zip(sig, nits, code, strict=True)
    ]
    core.log(
        "  SDR の信号 → nit: "
        + "、".join(
            f"{r['sdr_signal']:g} → {r['nits']:.0f}"
            for r in table
            if r["sdr_signal"] in (0.5, a.knee, 0.9, 1.0)
        )
    )
    core.log(
        f"  書き出した印: {st.get('codec_name')}・{st.get('pix_fmt')}・{st.get('color_primaries')}・"
        f"{st.get('color_transfer')}・{st.get('color_space')}（{count} コマ）"
    )
    # 確認画像: 元・変換せずに表示した HDR・明るさの地図・曲線
    w, h = size
    n = total
    rows = []
    stats = []
    for i in _keep_idx(n, 3 if not info.is_image else 1):
        t_src = (a.start or 0.0) + (i + 0.5) / info.fps if not info.is_image else 0.0
        t_out = (i + 0.5) / info.fps
        before = (
            _load_image(a.src, size)[:, :, :3]
            if info.is_image
            else core.frame_at(a.src, t_src, size=size)
        )
        naive = (
            _naive_image(out, size)
            if info.is_image
            else core.frame_at(out, t_out, size=size)
        )
        nits_img = _decode_nits(out, t_out, size, info.is_image)
        y = nits_img @ np.array([0.2627, 0.6780, 0.0593], np.float32)
        stats.append(
            {
                "time": round(t_src, 4),
                "nits_p50": round(float(np.percentile(y, 50)), 2),
                "nits_p99": round(float(np.percentile(y, 99)), 2),
                "nits_max": round(float(y.max()), 2),
                "pct_over_203": round(float((y > 203).mean() * 100), 3),
            }
        )
        lab = "" if info.is_image else f" {t_src:.2f}秒"
        rows.append(
            core.contact_sheet(
                [before, naive, _nits_map(nits_img)],
                [
                    f"元（SDR）{lab}",
                    f"HDR を変換せずに表示{lab}",
                    f"HDR の明るさ（nit）{lab}",
                ],
                cols=3,
                cell_width=360 if h > w else 480,
            )
        )
    xs = np.linspace(0, 1, 201)
    curve = _plot(
        [
            (
                xs,
                np.maximum(_expand(xs**2.4, knee_lin, ratio) * a.white, 0.05),
                (40, 40, 220),
                f"広げる（最大 {a.peak:g} nit）",
            ),
            (
                xs,
                np.maximum(xs**2.4 * a.white, 0.05),
                (160, 160, 160),
                f"広げない（白 = {a.white:g} nit）",
            ),
        ],
        title="SDR の信号 → HDR の明るさ（nit）",
        xlabel="SDR の信号",
        ylim=(0.1, 10000),
        logy=True,
        size=(560, 400),
        vlines=[(a.knee, (0, 140, 0), f"広げ始め {a.knee:g}")],
    )
    sheet = _rows([*rows, _nits_legend(1600), curve], width=1600)
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "format": a.format,
            "white_nits": a.white,
            "peak_nits": a.peak,
            "knee_signal": a.knee,
            "max_cll": maxcll[0] if a.format == "pq" and not info.is_image else None,
            "max_fall": maxcll[1] if a.format == "pq" and not info.is_image else None,
            "stream": {
                k: st.get(k)
                for k in (
                    "codec_name",
                    "pix_fmt",
                    "color_primaries",
                    "color_transfer",
                    "color_space",
                    "color_range",
                )
            },
            "side_data": _side_data(out) if not info.is_image else None,
            "curve": table,
            "check_frames": stats,
            "frames": count,
        },
    )
