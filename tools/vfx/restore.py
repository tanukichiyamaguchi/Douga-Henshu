"""画質を上げる（ノイズ・ちらつき・縞・ゴミ）。

担当するテクニック:
   28 ノイズ除去（denoise）
   30 圧縮ノイズ除去（deblock）
   31 時間方向のノイズ除去（temporal-denoise）
   36 フリッカー除去（deflicker）
   41 デインターレース（deinterlace）
   45 バンディング除去（deband）
   46 ゴミ・傷の除去（dust-removal）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
どのコマンドも、処理した映像（動画は MP4、画像は画像）と、数値の JSON、前後を並べた確認画像
（<出力>-compare.jpg。全体と、変化の大きい所の拡大）を書く。
"""

from __future__ import annotations

import argparse
import itertools
import re
import statistics
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np

from . import core
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/restore/（run_all.sh。既知の劣化を加えた素材を作り、元との差を測る）。
# PSNR は元（劣化を加える前）との差。動画の結果は H.264（crf 16）に書き出した物で測った。速さは他の作業と CPU を
# 分け合った状態の壁時計（起動・確認画像を含む）。
NOTES: dict[int, dict[str, str]] = {
    28: {
        "status": "◎",
        "how": "ffmpeg の dctdnoiz（既定。8×8、強いノイズでは 16×16 の DCT の係数をしきい値で切る）・nlmeans（非局所平均）・"
        "hqdn3d（空間だけ）。強さは自動: 明るさ Y の 2×2 Haar ウェーブレットの対角成分の |値| の中央値 ÷ 0.6745 を 32px の"
        "区画ごとに求め、白飛び・黒つぶれの区画を除いた下位 25% をノイズの強さ σ とする（動画は 5 コマの中央値）。"
        "dctdnoiz の sigma = 1.6σ、nlmeans の s = 0.75σ、hqdn3d の空間 = 2σ（既知のノイズで PSNR が最良の倍率を較正）。"
        "dctdnoiz は RGB でしか動かないので、動画は色を最近傍で広げて RGB にし、2×2 の平均で戻す（ffmpeg の自動の変換は"
        "何もしなくても 45.8 dB に劣化、この方法は 99 dB）。画像は OpenCV で読んだ画素を渡す。AI モデルは使わない",
        "measured": "ノイズの推定: RGB 各色に σ5/10/20 を加えた 8 枚で、明るさの真値 3.34/6.68/13.36 に対し 3.22〜3.83／"
        "6.37〜6.99／11.79〜13.12（油絵の starry_night だけは筆の模様をノイズと見て、ノイズ無しで 6.67）。"
        "画像（story540 の顔の寄り／building.jpg、自動の強さ）: σ5 34.1 → 43.95／41.62 dB、σ10 28.1 → 41.00／37.61 dB、"
        "σ20 22.2 → 38.26／33.42 dB（nlmeans は 42.34／38.63・39.87／34.62・36.99／30.85、hqdn3d は 1〜2 dB しか上がらない）。"
        "動画（story540 6〜8 秒・48 コマに σ10、yuv420p）: 29.97 → 37.00 dB（nlmeans 36.39、hqdn3d 32.54）。"
        "ノイズの無い story540 にかけると入力との差 44.37 dB（書き出し直しただけで 45.94 dB）。"
        "強いノイズで 16×16 にすると 8×8 より +1.3〜1.7 dB（σ20・30）。"
        "速さ: 540×960・48 コマで 7〜20 秒（dct）、3 秒（hqdn3d）、25 秒（nlmeans）。画像 1 枚 1〜2 秒",
        "limits": "画素ごとに独立のノイズで較正した。実際のカメラのノイズ（にじんだ色のむら・暗い所ほど強い）は弱めに推定"
        "されることがあるので、残る時は --strength 1.3〜1.5。細かい模様の多い絵（油絵など）は模様をノイズと見て強く"
        "かかる（--sigma で指定）。髪・ニットの細かい模様は σ が大きいほど柔らかくなる（--strength 0.6〜0.8 で残す）。"
        "1 コマずつの処理なので、動画では残ったノイズが毎コマ変わってちらつく（31 を使う）。学習モデルのノイズ除去"
        "（DRUNet・SCUNet など）は入れていない",
        "tested": "test_denoise.py（既知のガウスノイズ: 画像 2 枚×3 段階・動画 48 コマ・ノイズ無しの対照）、test_noise_est.py"
        "（推定の精度、8 枚×5 段階）、calib_denoise.py・calib_denoise_luma.py（倍率の較正: 画像 3 枚×4 段階と動画 3 段階）、"
        "確認画像 28-denoise-images.jpg・28-denoise-video.jpg を目で確認（σ20 の 8×8 の四角いむらを見て 16×16 に変えた）",
    },
    30: {
        "status": "◎",
        "how": "ffmpeg の spp（既定。8×8 の DCT を少しずつずらしてかけ、柔らかいしきい値で量子化の段差をならす。quality 3）・"
        "fspp・pp7。強さ qp は自動: 明るさの隣どうしの差を 8px 周期の位置ごとに平均し、最も大きい位置（ブロックの境目）の"
        "超過分 e とそれ以外の平均 m から目立ち方 x = e²/m、qp = round(1.5 + 0.35x)（1〜12。x < 0.1 は 1）。縮小の前にかける。"
        "色を半分の細かさで持つ JPEG（4:2:0）は yuv420p で処理（色のブロックも 8×8 になる）。画像は OpenCV で読んだ画素を"
        "渡す（ffmpeg の JPEG の読み込みは 0.1〜1 dB 悪かった）",
        "measured": "JPEG の品質 5/10/20/50 で圧縮した 5 枚（20 例、自動の強さ）: すべて改善、平均 +1.00 dB（最小 +0.00、"
        "最大 +1.76。例: story の顔 品質 10 で 31.59 → 33.35 dB、目立ち方 17.4 → 0.05）。式の qp と最良の qp の差は 34 例で"
        "最大 0.16 dB（較正）。方法の比較（品質 10・3 枚）: spp +0.68〜+1.76、fspp +0.50〜+1.78、pp7 +0.27〜+1.24 dB。"
        "動画（48 コマ）: MPEG-4 qscale 20 で 33.34 → 34.07 dB、qscale 31 で 31.60 → 32.51 dB、ビットレートを絞った H.264"
        "（250kbps）は +0.12 dB だけ。ブロックの無い画像は qp 1 になり、入力との差 43〜46 dB。実物の MPEG-4（Megamind.avi）"
        "は x = 0.18 → qp 2、輪郭のまわりのもやもやが減った（目で確認）。速さ: 画像 1 枚 約 1 秒、540×960・48 コマで 2.4 秒",
        "limits": "8×8 の四角の段差（JPEG・MPEG-1/2/4・H.263）向け。H.264/H.265 は保存の前に自分でブロックをならすので、"
        "絞った H.264 では +0.1 dB ほどしか変わらない（にじみ・消えた細部は戻らない）。縮小・拡大された映像は四角が 8px で"
        "なくなり目立ち方を測れない（--qp で指定）。強くかけると細かい模様もならされる。ffmpeg の pp（libpostproc）と "
        "deblock も比べたが、元の量子化の値が無いと +0.05〜0.7 dB で spp より弱かった。映像に記録された QP を spp に渡すと"
        "強すぎてかえって悪化し、H.264 では動かなかった",
        "tested": "test_deblock.py（JPEG 20 例・方法の比較・MPEG-4 と H.264 の動画・ブロックの無い画像・Megamind.avi）、"
        "calib_deblock.py・calib_deblock2.py（qp の曲線と式の較正）、確認画像 30-deblock-images.jpg・"
        "30-deblock-mpeg4-q31.jpg・30-deblock-megamind.jpg",
    },
    31: {
        "status": "◎",
        "how": "mc（既定）: 28 の dct を 0.8 倍の強さで 1 コマずつかけた後、今のコマから前後 2 コマへの動きを OpenCV の DIS "
        "オプティカルフローで求めて重ね、5×5 の区画の明るさの差の二乗 d² から重み exp(−d² / (8・(0.8σ/3)²)) を付けて平均"
        "する（合わない所＝動きの読み違い・隠れていた所・カットの前後は使わない）。fftdnoiz（ffmpeg。前後 1 コマを含む"
        " 3 次元 FFT、sigma 2σ）・hqdn3d（空間 σ・時間 2σ）も選べる。σ は 28 と同じ自動推定。コマは ffmpeg のパイプで"
        "読み書きする（一時ファイルなし）",
        "measured": "既知のノイズ σ10（コマごとに別の乱数）を加えた 48 コマ。story540（髪が動く）: 28 の dct 37.01 dB → "
        "mc 37.31、fftdnoiz 37.62、hqdn3d 32.82。vtest（固定カメラ、10fps で人が歩く）: dct 34.17 → mc 35.32、"
        "--radius 3 で 35.41、fftdnoiz 34.58、hqdn3d 33.68。動く所の PSNR（vtest）: mc 32.34（dct 31.93）、fftdnoiz 30.21"
        "（ノイズのままの 30.20 と同じ＝速い動きでは効かない）。ちらつき（動かない所の誤差の時間方向の標準偏差）: vtest で"
        " dct 2.54 → mc 2.01（−21%）、--radius 3 で 1.88。重ねたコマの数（重みの合計の平均）3.2〜3.6（最大 5）。"
        "速さ: 540×960・48 コマで 14〜22 秒（mc）、5〜8 秒（fftdnoiz）、5〜7 秒（hqdn3d）",
        "limits": "28 より PSNR が上がるのは 0.3〜1.2 dB と小さい（空間で消えた細部は時間の平均では戻らない）。効くのは、"
        "動かない所に残ったノイズのちらつき。fftdnoiz は動きの遅い映像では mc より PSNR が高い（story で +0.3 dB）が、"
        "速い動きでは動く所がノイズのまま残る。動きを読み違えやすい映像（煙・水面・激しいカメラの動き）では重みが下がり"
        "28 に近くなる。1080p は 540p の約 4 倍の時間（--max-side で縮めると速い）",
        "tested": "test_temporal.py（既知のノイズ σ10・2 本×48 コマ、mc・--radius 3・fftdnoiz・hqdn3d と 28 を比較）、"
        "proto_temporal.py・tune_mc.py（方法・空間の強さ・重みの比較。ノイズのあるコマを平均してから dct をかける順は"
        " 33 dB 台に落ちたので不採用）、確認画像 31-temporal-denoise-story.jpg・-vtest.jpg（時間の断面つき）・"
        "-vtest-moving.jpg（歩く人の拡大で残像が無いことを確認）",
    },
    36: {
        "status": "◎",
        "how": "gain（既定）: 幅 64px の灰色の小さなコマで、場面ごと（カット = 各コマを平均で割った差が、前後 5 コマの差の"
        "中央値の 4 倍かつ 8 以上）に、前後 11 コマ以上（既定 0.5 秒）の画素ごとの中央値（対数の明るさ）を基準とし、"
        "そのコマとの差の、画素についての中央値を揺れとする（白飛び・黒つぶれは除く。3 回繰り返す）。各コマに exp(−揺れ) を"
        "掛ける。直す量が ±30% を超えるコマを含むひと続きは演出（白いフラッシュ・フェード）とみなして直さない"
        "（--max-change）。ffmpeg の deflicker（前後のコマの平均の明るさに合わせる）も選べる。確認画像に明るさのグラフと"
        "時間の断面を付ける",
        "measured": "既知の揺れ（乱数 ±10% と 3Hz の ±5% の掛け算）を加えた動画。story540 1〜7 秒（144 コマ・カット 2）: "
        "コマごとの明るさの誤差 7.17% → 2.03%、PSNR 33.83 → 37.98 dB、カット直後 3 コマ 34.8 → 42.2 dB、カット 2/2 をコマ"
        "単位で検出（ffmpeg の deflicker は 5.92%・35.20 dB。カットをまたいでならす）。vtest 10〜20 秒（100 コマ）: "
        "6.93% → 1.99%、32.27 → 38.03 dB（ffmpeg は 1.73%・36.21 dB）。揺れの無い story540: 掛けた値 0.985〜1.012、"
        "入力との差 46.5 dB（最初に作った「平均の明るさを合わせる」方式は、物が動くだけで 0.93〜1.06 を掛けたので変えた）。"
        "v4 の演出の白いフラッシュ（2.5・6.0 秒）: そのまま残した（×1.0。この保護が無いと灰色のコマになった）。"
        "速さ: 540×960・48 コマで 2.1 秒",
        "limits": "画面全体の明るさの揺れ（照明の明滅・タイムラプス・生成映像の明るさの波）向け。画面の一部だけの明滅"
        "（蛍光灯が 1 本だけ・木漏れ日）や色のちらつき（色温度の揺れ）は直さない。明るくするコマの白飛びした所は戻らない。"
        "0.5 秒（少なくとも 11 コマ）より遅い変化は残す（--window）。±30% を超える揺れは演出とみなして残すので、強い"
        "ちらつきは --max-change 1。カットの見つけ方は簡易で、溶けるつなぎ目の途中はならされることがある",
        "tested": "test_deflicker.py（既知の揺れ・2 本、揺れの無い対照、v4 の演出のフラッシュ）、窓の長さと繰り返しの回数の"
        "比較、確認画像 36-deflicker-story.jpg（明るさのグラフ・カットの印・時間の断面）・-vtest.jpg・-v4-flash.jpg",
    },
    41: {
        "status": "◎",
        "how": "ffmpeg の idet でインターレースか（上・下どちらのフィールドが先か）を最大 300 コマ数えて報告し、bwdif（動画の"
        "既定）・yadif・estdif（画像 1 枚の既定）・w3fdif で解除する。--rate field で 1 フィールドを 1 コマに（コマ数 2 倍）。"
        "フィールドの順は idet の結果（--parity で指定も可）。縮小の前にかける。--detect-only は判定だけ",
        "measured": "プログレッシブの 48 コマから 2 コマずつ上下のフィールドを織り込んだ動画（4:4:4 可逆）: idet は TFF 24/24・"
        "BFF 24/24 を正しく判定し、元の story540・v4 は 305/305 コマをプログレッシブと判定。frame（元の 2n 番目と比較）: "
        "story540（髪が動く）織り込んだまま 25.92 → bwdif 41.31・yadif 41.50・estdif 41.89・w3fdif 40.75 dB、vtest（人が"
        "歩く。フィールド間の動きが大きい）29.36 → bwdif 38.16・yadif 37.79・w3fdif 33.61・estdif 30.89 dB。field（48 コマ）: "
        "41.31／37.90 dB、フィールドの順を逆に指定すると 22.93／26.26 dB（織り込んだままより悪い）。画像 1 枚: 22.66 → "
        "estdif 47.06 dB（bwdif は前後のコマが無いと何もしない）。速さ: 540×960・48 コマで 1.2 秒（判定を含む）",
        "limits": "フィールドの順を間違えると動きが前後に揺れて逆効果（自動判定を使う。途中で順が変わる素材は区切って"
        " --parity）。estdif は同じフィールドの中だけで補うので、動かない細かい模様の多い映像では落ちる（vtest で 30.89 dB）。"
        "テレシネ（24 コマを 60i にした物）の逆変換（fieldmatch・decimate）はしていない。インターレースの 4:2:0 の色の"
        "ずれは試していない。プログレッシブの素材にかけると縦の細かさが少し落ちる（判定を見てから）。nnedi は重みファイルが"
        "無いので使えない",
        "tested": "test_deinterlace.py（TFF・BFF×2 本・4 方法・frame と field・逆の順・画像 1 枚・プログレッシブの判定）、"
        "確認画像 41-deinterlace-story.jpg・-vtest.jpg（歩く人のくし状のギザギザが消えたのを確認）・-image.jpg",
    },
    45: {
        "status": "○",
        "how": "project（既定）: 面ごと（画像は B・G・R、動画は Y・U・V のまま）に段の高さ q（隣どうしの 0 でない差の最頻値）"
        "を測り、量子化の幅 [元 − q/2, 元 + q/2] を守ったまま、大きな箱のぼかし → 幅に戻す、を粗い所から細かい所へ繰り返して、"
        "いちばんなめらかな画を求める（射影法）。直すのは平らな台地（5×5 が同じ値）の近くだけで、値が変わった所に ±1 の"
        "三角分布の点（ディザー）を混ぜて丸める。動画の色（U・V）は段が 2 以上の時だけ直す。gradfun（強さ 0.75q・半径 4q）・"
        "deband（しきい値 1.5q・範囲 6q）も選べる",
        "measured": "なめらかな正解（合成の空のグラデーション 540×960）を段 4・8 で量子化: PSNR 46.45 → 49.49、40.79 → 46.15 dB"
        "（gradfun 46.90・44.10、deband 46.81・42.66）。明暗 4 倍の拡大で帯が消えたのは project だけ（gradfun・deband は"
        "境目がぼけるだけで平らな帯が残る）。σ1 の粒子あり: 段 8 で 40.76 → 42.41（gradfun 43.85 の方が良い）、段 4 は粒子が"
        "段を隠していて変化なし。動画（明るさを段 6 で量子化した動く空 48 コマ）: 43.01 → 45.29 dB、誤差の時間のばらつき"
        " 1.47 → 0.83（点のちらつきは増えない）。段の高さの推定: 段 3/4/6/8 を粒子の有無とも正しく推定。実写を段 4〜6 にした"
        "物では、なめらかな所の PSNR が home.jpg の空 43.20 → 43.27、story の顔 43.26 → 42.27、カフェの動画 44.89 → 42.12 と"
        "変わらないか下がった。速さ: 540×960 の画像 0.4〜0.6 秒、48 コマ 4〜5 秒",
        "limits": "なめらかなグラデーション（空・壁・ぼけた背景）の縞向けで、そこでは確かめた。実写の段 4〜6 では、段より"
        "小さな模様・肌のきめが量子化で失われていて戻らず、ならされて PSNR が 0〜2.8 dB 下がった（見た目は縞が減って"
        "なめらか）。粒子の多い映像では project が帯のまわりに輪のような境目を作ることがあり、--method gradfun の方が良かった。"
        "書き出しの H.264 で細かい点が消え、1 段の縞が戻ることがある（試験の空では戻らなかった）。HyperFrames などで最後に"
        "書き出し直す時にも縞が戻り得る",
        "tested": "test_deband.py（合成の空 4 例・実写 2 枚・動画 2 本）、calib_deband.py（gradfun・deband の較正）、"
        "proto_deband.py・proto_dequant.py（方法の比較。ガウスより箱のぼかしの方が速くて 2.6 dB 良かった）、確認画像 "
        "45-deband-images.jpg（明暗 4 倍）・45-deband-video-sky.jpg・45-deband-video-story.jpg",
    },
    46: {
        "status": "◎",
        "how": "detect（既定）: 前と後のコマ（最初・最後のコマは後・前の 2 コマ）を DIS のオプティカルフローで今のコマに重ね、"
        "明るさで「今 − 前」「今 − 後」が同じ符号でどちらも 25（--threshold）を超え、重ねた前と後の差が 12.5 未満の画素を"
        "候補にする。1px 広げた塊ごとに、面積 12px 未満・画面の 0.3% 超・まわり 9×9 で前後が合っていない（差の平均 6.25 以上"
        "＝動きの読み違い）ものを除き、残りを前後のコマの平均で埋める（縁はなじませる）。それ以外の画素は変えない。"
        "コマは 3 枚ずつ読んで進む。tmedian（ffmpeg。前後 1 コマとの画素ごとの中央値。両端に同じコマを足してずれを防ぐ）"
        "も選べる",
        "measured": "1 コマだけのゴミ（黒・白のにじんだ点。半径 1.5〜6px、明るさ ±60〜120）を毎コマ 4 つと、細い傷（幅 1〜2px・"
        "長さ 20〜60px）を入れた 48 コマ（4:4:4 可逆）。story540（髪が動く）: 点 152/192・傷 13/16 が消え、ゴミの中の PSNR"
        " 17.48 → 31.80 dB、ゴミの外で変えた画素 0.002%、動く所 41.82 dB。vtest（人が歩く）: 点 175/192・傷 9/9、33.63 dB、"
        "0.006%、動く所 37.78 dB。tmedian は点 145/192・176/192 だが、ゴミの外の 3.8%・0.5% の画素を変え、動く所が 26.15・"
        "20.28 dB に崩れた。塊を除く条件を入れる前は、vtest で 1 コマだけ見える手足などを 3,356 か所消して動く所が 27.68 dB"
        "だった（入れた後は 258 か所。入れたゴミは 201）。速さ: 540×960・48 コマで 5.7 秒",
        "limits": "1 コマだけに出る物向け。何コマも同じ所に残る傷（フィルムの縦の線）や、前後のコマの同じ所にもあるゴミは"
        "消えない。動く物の上のゴミは動きの読み違いと区別できずに残すことが多い（story で 2 割残った）。速く動く細い物"
        "（1 コマだけ見える手足・飛ぶ鳥・雨粒・火花）をゴミと見て消すことがある（vtest で 25px 以下の小さな変更が 48 コマで"
        "約 140 か所。拡大して目で見ると道路の白線や髪の毛先が少し変わる程度）。薄いゴミも消すなら --threshold 15、誤りを"
        "減らすなら 40。画像 1 枚には使えない（場所を指定して消す穴埋めを使う）",
        "tested": "test_dust.py（既知の場所のゴミ・傷、2 本×48 コマ、detect と tmedian）、diag_dust.py（本物と誤りの塊の"
        "特徴を比べて条件を決めた: まわりの前後の差は本物の中央値 0.7〜1.7、誤り 18〜19）、確認画像 "
        "46-dust-removal-story.jpg・-vtest.jpg・-story-zoom.jpg・-vtest-zoom.jpg・-false.jpg（誤って変えた所の拡大）",
    },
}


# ── 共通の部品 ────────────────────────────────────────────────────────────


def _paths(
    a: argparse.Namespace, info: core.MediaInfo, tag: str
) -> tuple[Path, Path, Path]:
    """(出力, JSON, 確認画像) の場所。画像は画像（既定 PNG）、動画は MP4 に書く。"""
    out = (
        Path(a.out)
        if a.out
        else core.default_out(a.src, tag, ".png" if info.is_image else ".mp4")
    )
    if info.is_image:
        core.require(
            out.suffix.lower() in core.IMAGE_EXTS,
            f"画像の出力先は画像の拡張子（.png など）にしてください: {out}",
        )
    else:
        core.require(
            out.suffix.lower() == ".mp4",
            f"動画の出力先は .mp4 にしてください: {out}",
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    return out, json_path, out.with_name(out.stem + "-compare.jpg")


def _video_only(info: core.MediaInfo, hint: str = "") -> None:
    core.require(
        not info.is_image,
        "動画を指定してください（前後のコマを使う処理なので、画像 1 枚では使えません）"
        + (f"。{hint}" if hint else ""),
    )


def _proc_size(info: core.MediaInfo, max_side: int | None) -> tuple[int, int]:
    """処理する大きさ。動画は偶数にそろえる（H.264 の都合）。画像は元の大きさのまま（縮める時だけ丸める）。"""
    if info.is_image:
        f = min(1.0, max_side / max(info.width, info.height)) if max_side else 1.0
        return max(1, round(info.width * f)), max(1, round(info.height * f))
    return core.scaled_size(info.width, info.height, max_side=max_side)


def _span(a: argparse.Namespace, info: core.MediaInfo) -> float:
    """処理する長さ（秒）。"""
    dur = max(info.duration - (a.start or 0.0), 0.0)
    if a.duration:
        dur = min(dur, a.duration)
    if a.max_frames:
        dur = min(dur, a.max_frames / info.fps)
    return dur


# RGB でしか動かないフィルター（dctdnoiz）を動画にかける時の色の変換。ffmpeg の自動の変換（双三次で色を
# 広げて戻す）は、何もしなくても 45.8 dB まで劣化した。色を最近傍で広げ、2×2 の平均で戻すと 99 dB（ほぼ無劣化）
_SWS = "accurate_rnd+full_chroma_int+full_chroma_inp"
_TO_RGB = f"scale=flags=neighbor+{_SWS},format=rgb24"
_TO_YUV = f"scale=flags=area+{_SWS},format=yuv420p"


def _ffmpeg_apply(
    a: argparse.Namespace,
    info: core.MediaInfo,
    filters: list[str],
    out: Path,
    *,
    scale_first: bool = True,
    image_fmt: str = "yuv444p",
    rgb: bool = False,
    frames_factor: int = 1,
) -> None:
    """ffmpeg のフィルターをかけて書き出す。動画は H.264（音声は残す）、画像は画像のまま。

    画像は OpenCV で読んだ画素を ffmpeg に渡す（ffmpeg の JPEG の読み込みは色のにじみの戻し方が粗く、
    OpenCV より 0.1〜1 dB 悪かった）。フルレンジの YUV（image_fmt）で処理して戻す（往復の誤差は最大 1）。
    透過のある画像は、透過をそのまま残す。
    scale_first=False は、縮小の前にフィルターをかける（圧縮のブロックの大きさを保つため）。
    rgb=True は RGB でしか動かないフィルター用（動画でも色を劣化させずに RGB へ変換して戻す）。
    frames_factor は、フィルターでコマ数が何倍になるか（デインターレースの field で 2）。
    """
    w, h = _proc_size(info, a.max_side)
    scale = [f"scale={w}:{h}:flags=area"] if (w, h) != (info.width, info.height) else []
    if rgb:
        image_fmt = "rgb24"
        if not info.is_image:
            filters = [_TO_RGB, *filters, _TO_YUV]
    chain = scale + filters if scale_first else filters + scale
    if info.is_image:
        img = core.load_image(a.src, alpha=True)
        ih, iw = img.shape[:2]
        if image_fmt == "yuv420p" and (iw % 2 or ih % 2):
            image_fmt = "yuv444p"
        if image_fmt == "yuv420p":
            # 色を 2×2 の平均で半分にして処理し、双三次で戻す（4:2:0 の JPEG の色のブロックを 8×8 に戻すため）
            first = [f"scale=out_range=full:flags=area+{_SWS}", "format=yuv420p"]
            last = [f"scale=flags=bicubic+{_SWS}", "format=bgr24"]
        else:
            first = ["scale=out_range=full", f"format={image_fmt}"]
            last = ["format=bgr24"]
        vf = ",".join([*first, *chain, *last])
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{iw}x{ih}", "-i", "-"]
        cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        r = subprocess.run(
            cmd,
            input=np.ascontiguousarray(img[:, :, :3]).tobytes(),
            capture_output=True,
            check=False,
        )
        if r.returncode != 0 or len(r.stdout) < w * h * 3:
            raise core.VfxError(
                "ffmpeg が失敗しました: " + vf + "\n" + r.stderr.decode()[-800:]
            )
        res = np.frombuffer(r.stdout, np.uint8)[: w * h * 3].reshape(h, w, 3)
        if (img[:, :, 3] < 255).any() and out.suffix.lower() not in {".jpg", ".jpeg"}:
            alpha = cv2.resize(img[:, :, 3], (w, h), interpolation=cv2.INTER_AREA)
            res = np.dstack([res, alpha])
        core.save_image(out, res)
        return
    args: list[str] = []
    if a.start:
        args += ["-ss", f"{a.start:.4f}"]
    args += ["-i", str(a.src)]
    if a.duration or a.max_frames:
        args += ["-t", f"{_span(a, info):.4f}"]
    if a.max_frames:
        args += ["-frames:v", str(a.max_frames * frames_factor)]
    args += ["-map", "0:v:0", "-map", "0:a:0?", "-vf", ",".join(chain)]
    args += [*core.MP4_ARGS, "-c:a", "aac", "-b:a", "192k", str(out)]
    core.ffmpeg(args)


def _read_filtered(
    a: argparse.Namespace,
    info: core.MediaInfo,
    rgb_filters: list[str] | None = None,
    yuv_filters: list[str] | None = None,
    pix_fmt: str = "bgr24",
) -> Iterator[np.ndarray]:
    """コマを順に BGR（pix_fmt="yuv444p" なら Y・U・V の 3 面を重ねた配列）で受け取る（一時ファイルを作らない）。

    範囲と大きさは a に従う。yuv_filters は元の色の形式のまま、rgb_filters は RGB に変えてからかける。
    色は最近傍で広げるので、_Writer（2×2 の平均で戻す）と組にすると、手を加えなかった画素は元とほぼ同じに戻る。
    """
    w, h = _proc_size(info, a.max_side)
    chain = [f"scale={w}:{h}:flags=area"] if (w, h) != (info.width, info.height) else []
    if pix_fmt == "yuv444p":
        chain += [
            *(yuv_filters or []),
            f"scale=flags=neighbor+{_SWS}",
            "format=yuv444p",
        ]
    else:
        chain += [*(yuv_filters or []), _TO_RGB, *(rgb_filters or []), "format=bgr24"]
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
    if a.start:
        cmd += ["-ss", f"{a.start:.4f}"]
    cmd += ["-i", str(a.src)]
    if a.duration or a.max_frames:
        cmd += ["-t", f"{_span(a, info):.4f}"]
    cmd += ["-an", "-vf", ",".join(chain), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    size = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    count = 0
    try:
        while a.max_frames is None or count < a.max_frames:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            arr = np.frombuffer(buf, np.uint8)
            if pix_fmt == "yuv444p":
                yield np.ascontiguousarray(arr.reshape(3, h, w).transpose(1, 2, 0))
            else:
                yield arr.reshape(h, w, 3).copy()
            count += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if count == 0:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise core.VfxError(f"コマを読めませんでした: {a.src}\n{err[-800:]}")


class _Writer:
    """Python で作ったコマ（BGR）を MP4 に書く（元の音声の同じ範囲を入れる）。

    H.264 の設定は core.VideoWriter と同じ（core.MP4_ARGS）。色の変換だけを正確にする: core.VideoWriter の既定の
    変換は、手を加えないコマの往復でも 41.2 dB（明るさ 42.6 dB）まで劣化した（正確な変換なら 46.0 dB）。
    """

    def __init__(
        self,
        a: argparse.Namespace,
        info: core.MediaInfo,
        out: Path,
        pix_fmt: str = "bgr24",
    ) -> None:
        self.a, self.info, self.out, self.pix_fmt = a, info, out, pix_fmt
        self.proc: subprocess.Popen[bytes] | None = None
        self.size: tuple[int, int] | None = None
        self.count = 0

    def _open(self, w: int, h: int) -> None:
        a, info = self.a, self.info
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error"]
        cmd += ["-f", "rawvideo", "-pix_fmt", self.pix_fmt, "-s", f"{w}x{h}"]
        cmd += ["-r", f"{info.fps:.6f}", "-i", "-"]
        if info.has_audio:
            if a.start:
                cmd += ["-ss", f"{a.start:.4f}"]
            if a.duration or a.max_frames:
                cmd += ["-t", f"{_span(a, info):.4f}"]
            cmd += ["-i", str(a.src), "-map", "0:v", "-map", "1:a", "-shortest"]
        cmd += ["-vf", _TO_YUV, *core.MP4_ARGS]
        cmd += ["-c:a", "aac", "-b:a", "192k"] if info.has_audio else []
        cmd.append(str(self.out))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.size = (w, h)

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame[:, :, :3])
        h, w = frame.shape[:2]
        if self.proc is None:
            core.require(
                w % 2 == 0 and h % 2 == 0, f"大きさは偶数にしてください（{w}×{h}）"
            )
            self._open(w, h)
        assert self.proc is not None and self.proc.stdin is not None
        if self.pix_fmt == "yuv444p":
            frame = np.ascontiguousarray(frame.transpose(2, 0, 1))
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


def _sample_times(a: argparse.Namespace, info: core.MediaInfo, n: int) -> list[float]:
    """処理する範囲から、等間隔に n 個の時刻（コマの真ん中）を選ぶ。"""
    start, dur = a.start or 0.0, _span(a, info)
    frames = max(1, int(dur * info.fps))
    idx = sorted({int((k + 0.5) * frames / n) for k in range(n)})
    return [start + (i + 0.5) / info.fps for i in idx]


def _frame_pair(
    a: argparse.Namespace, info: core.MediaInfo, out: Path, k: int | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """比べる用に、元と結果の同じコマを読む（元は結果の大きさにそろえる）。(元, 結果, 元の中の秒)

    k は結果の中のコマ番号（省略時は真ん中）。ffmpeg の -ss は「その時刻以降の最初のコマ」を返すので、
    コマの時刻の半コマ前を指定する（処理した範囲の最初のコマ = 元の ceil(開始×fps) 番目にそろえる）。
    """
    if info.is_image:
        before = core.load_image(a.src)
        after = core.load_image(out)
        if before.shape[:2] != after.shape[:2]:
            before = cv2.resize(
                before, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_AREA
            )
        return before, after, 0.0
    oinfo = core.probe(out)
    n = max(1, round(oinfo.duration * oinfo.fps))
    k = max(0, min(n - 1, n // 2 if k is None else k))
    after = core.frame_at(out, max(0.0, (k - 0.5) / oinfo.fps))
    j0 = int(np.ceil((a.start or 0.0) * info.fps - 1e-6))
    before = core.frame_at(
        a.src,
        max(0.0, (j0 + k - 0.5) / info.fps),
        size=(after.shape[1], after.shape[0]),
    )
    return before, after, (j0 + k) / info.fps


def _change_box(before: np.ndarray, after: np.ndarray) -> tuple[int, int, int]:
    """処理で最も変わった所（正方形の左上 x, y と一辺）。確認画像の拡大に使う。"""
    h, w = before.shape[:2]
    s = max(16, min(h, w) // 3)
    diff = cv2.absdiff(before[:, :, :3], after[:, :, :3]).max(axis=2)
    box = cv2.blur(diff.astype(np.float32), (s, s))
    half = s // 2
    inner = box[half : h - s + half + 1, half : w - s + half + 1]
    y, x = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return int(x), int(y), s


def _compare_sheet(
    before: np.ndarray,
    after: np.ndarray,
    labels: tuple[str, str] = ("元", "処理後"),
    box: tuple[int, int, int] | None = None,
    cell_width: int = 480,
    enhance: float = 1.0,
) -> np.ndarray:
    """前後の全体（左右）と、変化の大きい所の拡大（縦長なら右の列に上下、横長なら下の段）を 1 枚に並べる。

    enhance > 1 なら、拡大の明暗の差を強調する（元の拡大の平均を中心に enhance 倍。段差を見やすくする）。
    """
    x, y, s = box or _change_box(before, after)
    full = []
    for im in (before, after):
        im = im[:, :, :3].copy()
        th = max(2, im.shape[1] // 240)
        cv2.rectangle(im, (x, y), (x + s - 1, y + s - 1), (0, 220, 255), th)
        full.append(im)
    top = core.contact_sheet(full, list(labels), cols=2, cell_width=cell_width)
    portrait = before.shape[0] > before.shape[1]
    cs = top.shape[0] // 2 if portrait else cell_width
    zoom = cs / s
    center = before[y : y + s, x : x + s, :3].reshape(-1, 3).mean(axis=0)
    crops = []
    for im, name in zip((before, after), labels, strict=True):
        c = im[y : y + s, x : x + s, :3]
        if enhance != 1.0:
            c = np.clip((c - center) * enhance + center, 0, 255).astype(np.uint8)
        c = cv2.resize(
            c,
            (cs, cs),
            interpolation=cv2.INTER_NEAREST if zoom >= 2 else cv2.INTER_AREA,
        )
        note = f"拡大 ×{zoom:.1f}" + (
            f"・明暗を {enhance:g} 倍に強調" if enhance != 1.0 else ""
        )
        core.draw_text(c, f"{name}（{note}）", (6, 6), size=max(14, cs // 24))
        crops.append(c)
    if portrait:
        col = np.vstack(crops)
        col = np.vstack(
            [col, np.full((top.shape[0] - col.shape[0], cs, 3), 24, np.uint8)]
        )
        return np.hstack([top, col])
    return np.vstack([top, np.hstack(crops)])


def _luma(img: np.ndarray) -> np.ndarray:
    """BGR → 明るさ Y（BT.601 の重み。0〜255 の float32）。"""
    f = img[:, :, :3].astype(np.float32)
    return f[..., 0] * 0.114 + f[..., 1] * 0.587 + f[..., 2] * 0.299


def _noise_sigma(img: np.ndarray) -> float:
    """明るさ（Y）のノイズの強さ（標準偏差。0〜255）を 1 枚の画像から推定する。

    2×2 の Haar ウェーブレットの対角成分（模様の少ない所ではほぼノイズだけ）の絶対値の中央値 ÷ 0.6745 を
    32×32 px の区画ごとに求め、白飛び・黒つぶれの区画を除いた下位 25% の値を使う（模様の多い区画を避ける）。
    """
    y = _luma(img)
    h, w = y.shape
    y = y[: h - h % 2, : w - w % 2]
    a, b, c, d = y[0::2, 0::2], y[0::2, 1::2], y[1::2, 0::2], y[1::2, 1::2]
    hh = np.abs(a - b - c + d) / 2
    blk = 16
    hb, wb = hh.shape[0] // blk, hh.shape[1] // blk
    if hb < 2 or wb < 2:
        return float(np.median(hh) / 0.6745)
    t = hh[: hb * blk, : wb * blk].reshape(hb, blk, wb, blk).transpose(0, 2, 1, 3)
    est = np.median(t.reshape(hb * wb, -1), axis=1) / 0.6745
    mean = (a + b + c + d)[: hb * blk, : wb * blk] / 4
    m = mean.reshape(hb, blk, wb, blk).mean(axis=(1, 3)).ravel()
    ok = (m > 20) & (m < 235)
    if ok.sum() < 8:
        ok[:] = True
    return float(np.percentile(est[ok], 25))


def _estimate_noise(a: argparse.Namespace, info: core.MediaInfo, n: int = 5) -> float:
    """処理する大きさのコマで、ノイズの強さを推定する（動画は n コマの中央値）。"""
    if info.is_image:
        return _noise_sigma(next(core.read_frames(a.src, max_side=a.max_side)))
    vals = [
        _noise_sigma(core.frame_at(a.src, t, max_side=a.max_side))
        for t in _sample_times(a, info, n)
    ]
    return float(statistics.median(vals))


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


# ── 28 ノイズ除去 ─────────────────────────────────────────────────────────

_DENOISE = {
    "dct": "ffmpeg の dctdnoiz（8×8 の DCT の係数を、ノイズの強さに合わせて切る。既定。速くて強い）",
    "nlmeans": "ffmpeg の nlmeans（似た小さな区画どうしを平均する非局所平均。繰り返し模様に強いが約 3 倍遅い）",
    "hqdn3d": "ffmpeg の hqdn3d の空間成分だけ（とても速いが弱い。軽く整える時）",
}


def _denoise_filter(method: str, sigma: float) -> str:
    """明るさのノイズの強さ σ から、フィルターの設定を作る。

    倍率は既知のガウスノイズで PSNR が最も良くなる値（較正: vfx-tests/restore/calib_denoise_luma.py）。
    dct は画像（σ 5〜30 の 12 例）・動画とも 1.6 倍、nlmeans は 0.6〜0.8 倍、hqdn3d は 2 倍が最良だった。
    dct の区画は、強いノイズ（σ 8 以上）では 16×16（8×8 だと肌に四角いむらが残る。PSNR も 1.3〜1.7 dB 良い。
    約 3.5 倍遅い）、それ未満は 8×8（差は 0.3 dB 以下）。
    """
    if method == "dct":
        return f"dctdnoiz=sigma={max(0.1, 1.6 * sigma):.2f}:n={4 if sigma >= 8 else 3}"
    if method == "nlmeans":
        return f"nlmeans=s={min(30.0, max(1.0, 0.75 * sigma)):.2f}:p=7:r=15"
    if method == "hqdn3d":
        ls = max(0.1, 2.0 * sigma)
        return f"hqdn3d=luma_spatial={ls:.2f}:chroma_spatial={0.75 * ls:.2f}:luma_tmp=0:chroma_tmp=0"
    raise core.VfxError(f"知らない方法です: {method}")


@command(
    28,
    "denoise",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_DENOISE),
        default="dct",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _DENOISE.items()),
    ),
    arg(
        "--sigma",
        type=float,
        help="明るさのノイズの強さ（標準偏差、0〜255。目安: 軽い 2〜3、目立つ 6〜7、ひどい 12〜20）。"
        "省略時は映像から推定する",
    ),
    arg(
        "--strength",
        type=float,
        default=1.0,
        help="強さの倍率（既定 1.0。0.6〜0.8 で細部や粒子を多めに残す、1.3〜1.5 で強め）",
    ),
)
def denoise(a: argparse.Namespace) -> None:
    """1 コマずつ（空間だけで）ノイズのザラつきを消す。強さは映像から測ったノイズの量で自動に決める。"""
    info = core.probe(a.src)
    core.require(a.strength > 0, "--strength は 0 より大きくしてください")
    out, json_path, sheet_path = _paths(a, info, "denoise")
    measured = _estimate_noise(a, info)
    sigma = a.sigma if a.sigma is not None else measured
    core.log(
        f"推定したノイズの強さ: σ {measured:.2f}"
        + (f"（指定 σ {a.sigma:.2f} を使う）" if a.sigma is not None else "")
    )
    if a.sigma is None and measured < 1.0:
        core.log(
            "  ノイズはほとんど無いようです（σ 1.0 未満）。そのままに近い結果になります"
        )
    filt = _denoise_filter(a.method, sigma * a.strength)
    core.log(f"フィルター: {filt}")
    _ffmpeg_apply(a, info, [filt], out, rgb=a.method == "dct")
    before, after, t = _frame_pair(a, info, out)
    after_sigma = _noise_sigma(after)
    core.log(
        f"確認したコマ（{t:.2f} 秒）のノイズ: σ {_noise_sigma(before):.2f} → {after_sigma:.2f}"
    )
    sheet = _compare_sheet(
        before, after, ("元", f"ノイズ除去（{a.method}・σ{sigma * a.strength:.1f}）")
    )
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "filter": filt,
            "noise_sigma_measured": round(measured, 3),
            "noise_sigma_used": round(sigma * a.strength, 3),
            "check_frame": {
                "time": round(t, 4),
                "noise_sigma_before": round(_noise_sigma(before), 3),
                "noise_sigma_after": round(after_sigma, 3),
            },
        },
    )


# ── 30 圧縮ノイズ除去 ─────────────────────────────────────────────────────

_DEBLOCK = {
    "spp": "ffmpeg の spp（8×8 の DCT を少しずつずらしてかけ、量子化の段差をならす。既定。試験で最も良かった）",
    "fspp": "ffmpeg の fspp（spp の速い版。JPEG の強い圧縮では spp より 0.3〜0.6 dB 弱い）",
    "pp7": "ffmpeg の pp7（spp の変種。spp より 0.1〜0.4 dB 弱い）",
}


def _block_measure(img: np.ndarray, period: int = 8) -> float:
    """ブロックの目立ち方 x（0 以上。きれいな画像は 0.1 未満、JPEG の品質 50 で 0.3〜1.5、品質 10 で 7〜18）。

    明るさの隣どうしの差を、横（列ごと）と縦（行ごと）に平均し、8 px 周期の位置ごとにまとめる。最も大きい位置
    （ブロックの境目）の超過分 e と、それ以外の位置の平均 m から x = e² / m（横と縦の平均）。
    """
    y = _luma(img)
    xs = []
    for d in (
        np.abs(np.diff(y, axis=1)).mean(axis=0),
        np.abs(np.diff(y, axis=0)).mean(axis=1),
    ):
        n = len(d) // period * period
        if n < period * 2:
            continue
        prof = d[:n].reshape(-1, period).mean(axis=0)
        k = int(np.argmax(prof))
        m = float(np.delete(prof, k).mean())
        xs.append((float(prof[k]) - m) ** 2 / max(m, 1e-6))
    return float(np.mean(xs)) if xs else 0.0


def _blockiness(a: argparse.Namespace, info: core.MediaInfo, n: int = 5) -> float:
    """元の大きさのコマで測ったブロックの目立ち方（動画は n コマの中央値）。"""
    if info.is_image:
        return _block_measure(core.load_image(a.src))
    return float(
        statistics.median(
            _block_measure(core.frame_at(a.src, t)) for t in _sample_times(a, info, n)
        )
    )


def _auto_qp(x: float) -> int:
    """ブロックの目立ち方 x → spp の qp。

    較正（vfx-tests/restore/calib_deblock2.py）: JPEG の品質 5〜50 の 5 枚の画像と、MPEG-4 の qscale 6〜31 の動画の
    34 例で、この式の qp の PSNR は、最良の qp との差が最大 0.16 dB だった。x が 0.1 未満（ブロックがほぼ無い）は 1。
    """
    if x < 0.1:
        return 1
    return int(np.clip(round(1.5 + 0.35 * x), 1, 12))


def _jpeg_420(path: str | Path) -> bool:
    """色を縦横半分の細かさで持つ JPEG（4:2:0。ふつうの JPEG の多く）か。"""
    from PIL import Image

    try:
        with Image.open(path) as im:
            layers = getattr(im, "layer", None) if im.format == "JPEG" else None
    except OSError:
        return False
    if not layers or len(layers) != 3:
        return False
    (_, yh, yv, _), (_, ch, cv, _) = layers[0], layers[1]
    return (yh, yv) == (2 * ch, 2 * cv) == (2, 2)


def _deblock_filter(method: str, qp: int) -> str:
    if method == "spp":
        return f"spp=quality=3:qp={qp}:mode=soft"
    if method == "fspp":
        return f"fspp=quality=4:qp={qp}:strength=0"
    if method == "pp7":
        return f"pp7=qp={qp}:mode=medium"
    raise core.VfxError(f"知らない方法です: {method}")


@command(
    30,
    "deblock",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_DEBLOCK),
        default="spp",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _DEBLOCK.items()),
    ),
    arg(
        "--qp",
        type=int,
        help="強さ（1〜15。目安: 軽い 2、普通 3〜4、強い 6〜8、ひどい 10〜12）。省略時はブロックの目立ち方から決める",
    ),
    arg(
        "--strength",
        type=float,
        default=1.0,
        help="自動で決めた強さの倍率（既定 1.0。細部を残したい時は 0.7、まだ四角が見える時は 1.5）",
    ),
)
def deblock(a: argparse.Namespace) -> None:
    """強い圧縮でできた四角いブロックの段差やにじみ（JPEG・MPEG の 8×8）をならす。強さは段差の目立ち方から自動で決める。"""
    info = core.probe(a.src)
    core.require(a.strength > 0, "--strength は 0 より大きくしてください")
    out, json_path, sheet_path = _paths(a, info, "deblock")
    x = _blockiness(a, info)
    qp = a.qp if a.qp else max(1, round(_auto_qp(x) * a.strength))
    core.log(
        f"ブロックの目立ち方: {x:.3f}（0.1 未満はほぼ無し、1 前後は軽い、7 以上はひどい）→ 強さ qp {qp}"
        + ("（指定）" if a.qp else "")
    )
    if x < 0.1 and not a.qp:
        core.log(
            "  ブロックの段差はほとんど見つかりません。いちばん弱い強さでかけます"
            "（縮小された映像は 8 px の四角が崩れていて測れないことがあります。その時は --qp で指定）"
        )
    filt = _deblock_filter(a.method, qp)
    core.log(f"フィルター: {filt}")
    # 縮小するとブロックの大きさが 8 px でなくなるので、縮小の前にかける。色を半分の細かさで持つ JPEG（4:2:0）は
    # yuv420p で処理する（色のブロックが 8×8 に戻り、試験の 20 例で平均 +0.85 → +1.00 dB）
    fmt = "yuv420p" if info.is_image and _jpeg_420(a.src) else "yuv444p"
    _ffmpeg_apply(a, info, [filt], out, scale_first=False, image_fmt=fmt)
    before, after, t = _frame_pair(a, info, out)
    scaled = _proc_size(info, a.max_side) != (info.width, info.height)
    xb, xa = _block_measure(before), _block_measure(after)
    core.log(
        f"確認したコマ（{t:.2f} 秒）のブロックの目立ち方: {xb:.3f} → {xa:.3f}"
        + ("（縮小後に測ったので参考）" if scaled else "")
    )
    sheet = _compare_sheet(
        before, after, ("元", f"圧縮ノイズ除去（{a.method}・qp {qp}）")
    )
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "filter": filt,
            "qp": qp,
            "blockiness": round(x, 4),
            "check_frame": {
                "time": round(t, 4),
                "blockiness_before": round(xb, 4),
                "blockiness_after": round(xa, 4),
                "measured_after_scaling": scaled,
            },
        },
    )


# ── 45 バンディング除去 ───────────────────────────────────────────────────

_DEBAND = {
    "project": "量子化の幅（元の値 ±段/2）を守ったまま、いちばんなめらかな画を探す（大きな箱でぼかす → 幅に戻す、を"
    "粗い所から細かい所へ繰り返す）。最後に ±1 段の細かい点を混ぜて 8bit に戻す。既定。広い段でも平らな帯が残らない",
    "gradfun": "ffmpeg の gradfun（段の境目を周りの平均へ寄せ、細かい点でならす。速い。帯の幅が 30px を超えると、"
    "境目がぼけるだけで平らな帯が残る）",
    "deband": "ffmpeg の deband（周りの画素と比べて段差を埋める。速い。gradfun と同じく広い帯は残り、粒子をつぶしやすい）",
}


def _step_heights(img: np.ndarray) -> list[int]:
    """色の段差の高さ（0〜255 の段）を面ごとに: 隣どうしの 0 でない差（24 未満）の最頻値。

    8 段で量子化したなめらかなグラデーションなら 8、ふつうの写真は 1。
    """
    out = []
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.int16)
        d = np.concatenate(
            [np.abs(np.diff(ch, axis=1)).ravel(), np.abs(np.diff(ch, axis=0)).ravel()]
        )
        d = d[(d > 0) & (d < 24)]
        out.append(int(np.bincount(d).argmax()) if len(d) >= 50 else 1)
    return out


def _banding_steps(
    a: argparse.Namespace, info: core.MediaInfo, n: int = 5
) -> list[int]:
    """面ごとの段の高さ。画像は B・G・R、動画は Y・U・V（元の形式のまま。RGB に直すと段が崩れる）。

    動画は n コマの中央値。色は最近傍で広げて読む（双三次で広げると段の値が混ざる）。
    """
    if info.is_image:
        return _step_heights(core.load_image(a.src))
    vals = []
    for t in _sample_times(a, info, n):
        one = argparse.Namespace(
            **{
                **vars(a),
                "start": t,
                "duration": None,
                "max_frames": 1,
                "max_side": None,
            }
        )
        vals.append(_step_heights(next(_read_filtered(one, info, pix_fmt="yuv444p"))))
    return [int(statistics.median(v)) for v in zip(*vals, strict=True)]


def _dequantize(
    img: np.ndarray, steps: list[float], rng: np.random.Generator
) -> np.ndarray:
    """縞を戻す（射影法）。面ごとに、量子化の幅 [元 − 段/2, 元 + 段/2] を守ったまま、箱のぼかしを大きい順に
    かけて幅に戻すことを繰り返し、いちばんなめらかな画を求める。段が 0 の面はそのまま。

    直すのは、平らな台地（5×5 が同じ値）の近く（9×9 に広げ、ぼかしてなじませる）だけ。粒子や模様で値が細かく
    入れ替わる所は縞ではないので変えない（変えると模様が消え、試験で PSNR が 4.5 dB 下がった）。
    値が変わった所だけに、三角分布の ±1 段の点（ディザー）を混ぜて 8bit に丸める（丸めで新しい縞を作らないため）。
    """
    f = img[:, :, :3].astype(np.float32)
    out = img[:, :, :3].copy()
    h, w = f.shape[:2]
    k = max(3, (min(h, w) // 5) | 1)
    sizes = []
    while k >= 3:
        sizes.append(k)
        k = (k // 2) | 1 if k > 3 else 1
    box5, box9 = np.ones((5, 5), np.uint8), np.ones((9, 9), np.uint8)
    for c in range(3):
        if steps[c] <= 0:
            continue
        ch = img[:, :, c]
        flat = (cv2.dilate(ch, box5) - cv2.erode(ch, box5)) == 0
        mask = cv2.GaussianBlur(
            cv2.dilate(flat.astype(np.uint8), box9).astype(np.float32), (0, 0), 2
        )
        if mask.max() < 0.01:
            continue
        x = f[:, :, c]
        lo, hi = x - steps[c] / 2, x + steps[c] / 2
        sm = x.copy()
        for k in sizes:
            for _ in range(3):
                sm = cv2.blur(sm, (k, k))
                np.clip(sm, lo, hi, out=sm)
        changed = np.abs(sm - x) > 0.01
        d = rng.random((h, w), np.float32) - rng.random((h, w), np.float32)
        sm = np.where(changed, sm + d, sm)
        res = mask * sm + (1 - mask) * x
        out[:, :, c] = np.clip(np.round(res), 0, 255).astype(np.uint8)
    return out


def _deband_filter(method: str, step: float) -> str:
    """段の高さ → ffmpeg のフィルターの設定（較正: vfx-tests/restore/calib_deband.py。合成のグラデーションを段 3〜8 で量子化）。

    gradfun は強さ 0.75×段・半径 4×段（16〜32）が、粒子の有無・段 3〜8 のどれでも最良か最良に近かった。
    deband は、しきい値 1.5×段・範囲 6×段（16〜64）。
    """
    if method == "gradfun":
        st = float(np.clip(0.75 * step, 0.51, 64))
        return f"gradfun=strength={st:.2f}:radius={int(np.clip(4 * step, 16, 32))}"
    if method == "deband":
        thr = float(np.clip(1.5 * step / 255, 0.003, 0.5))
        t = f"{thr:.4f}"
        return f"deband=1thr={t}:2thr={t}:3thr={t}:4thr={t}:range={int(np.clip(6 * step, 16, 64))}"
    raise core.VfxError(f"知らない方法です: {method}")


@command(
    45,
    "deband",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_DEBAND),
        default="project",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _DEBAND.items()),
    ),
    arg(
        "--step",
        type=float,
        help="段の高さ（0〜255 の段。目安: うっすら 1〜2、はっきり 4〜6、ひどい 8）。省略時は映像から測る",
    ),
    arg(
        "--strength",
        type=float,
        default=1.0,
        help="強さの倍率（既定 1.0。段が残る時は 1.5、細かい模様がつぶれる時は 0.7）",
    ),
)
def deband(a: argparse.Namespace) -> None:
    """空や壁のなめらかなグラデーションに出る色の段（縞）をならす。強さは段の高さを測って自動で決める。"""
    info = core.probe(a.src)
    core.require(a.strength > 0, "--strength は 0 より大きくしてください")
    out, json_path, sheet_path = _paths(a, info, "deband")
    measured = _banding_steps(a, info)
    names = "B・G・R" if info.is_image else "Y（明るさ）・U・V（色）"
    if a.step:
        steps = [float(a.step)] * 3
    elif info.is_image:
        # 段がある色だけ直す（どの色も 1 なら、1 段の縞として全部）
        steps = [float(v) if v >= 2 or max(measured) <= 1 else 0.0 for v in measured]
    else:
        # 動画: 明るさ（Y）は 1 段でも直す。色（U・V）は段が 2 以上の時だけ（色に点を混ぜないため）
        steps = [float(max(1, measured[0]))] + [
            float(v) if v >= 2 else 0.0 for v in measured[1:]
        ]
    steps = [v * a.strength for v in steps]
    main = max(steps) if max(steps) > 0 else 1.0
    core.log(
        f"段の高さ（{names}）: {' / '.join(map(str, measured))}（1〜2 は段がほぼ見えない、4 以上ははっきり）"
        + (f" → 指定 {a.step:g}" if a.step else "")
    )
    if max(measured) <= 2 and not a.step:
        core.log(
            "  はっきりした段は見つかりません（粒子やノイズで隠れているか、段が 1〜2 と小さい）。"
            "1〜2 段の幅でならして細かい点を混ぜます"
        )
    if a.method == "project":
        filt = (
            "射影法（各面の段 "
            + " / ".join(f"{v:g}" for v in steps)
            + " の幅の中でならす）"
        )
        core.log(f"方法: {filt}")
        rng = np.random.default_rng(0)
        if info.is_image:
            img = core.load_image(a.src, alpha=True)
            res = _dequantize(img, steps, rng)
            if a.max_side:
                res = cv2.resize(
                    res, _proc_size(info, a.max_side), interpolation=cv2.INTER_AREA
                )
            if (img[:, :, 3] < 255).any() and out.suffix.lower() not in {
                ".jpg",
                ".jpeg",
            }:
                alpha = cv2.resize(
                    img[:, :, 3],
                    (res.shape[1], res.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                res = np.dstack([res, alpha])
            core.save_image(out, res)
        else:
            # Y・U・V のまま直す（RGB に変えると段が崩れる）。縮小は直した後
            full = argparse.Namespace(**{**vars(a), "max_side": None})
            size = _proc_size(info, a.max_side)
            total = max(1, round(_span(a, info) * info.fps))
            with _Writer(a, info, out, pix_fmt="yuv444p") as w:
                frames = _read_filtered(full, info, pix_fmt="yuv444p")
                for i, f in enumerate(frames):
                    r = _dequantize(f, steps, rng)
                    if size != (r.shape[1], r.shape[0]):
                        r = cv2.resize(r, size, interpolation=cv2.INTER_AREA)
                    w.write(r)
                    if (i + 1) % max(1, total // 10) == 0:
                        core.log(f"  {i + 1}/{total} コマ")
    else:
        filt = _deband_filter(a.method, max(main, 1.0))
        core.log(f"フィルター: {filt}")
        # 縮小すると段の境目がなめらかに混ざって段の高さが測れなくなるので、縮小の前にかける
        _ffmpeg_apply(a, info, [filt], out, scale_first=False)
    before, after, t = _frame_pair(a, info, out)
    sheet = _compare_sheet(
        before, after, ("元", f"バンディング除去（{a.method}）"), enhance=4.0
    )
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "filter": filt,
            "step_measured": measured,
            "step_planes": names,
            "step_used": [round(v, 3) for v in steps],
            "check_frame": {"time": round(t, 4)},
        },
    )


# ── 時間の断面（確認画像の部品） ──────────────────────────────────────────


def _time_slices(
    a: argparse.Namespace, info: core.MediaInfo, out: Path, y: int, limit: int = 240
) -> tuple[np.ndarray, np.ndarray]:
    """元と結果の、横 1 行（高さ y）をコマの順に縦に積んだ画（縦が時間）。ちらつきは縦の筋のむらに見える。"""
    w, h = _proc_size(info, a.max_side)
    n = max(1, min(limit, round(_span(a, info) * info.fps)))
    y = int(np.clip(y, 0, h - 1))
    before = [
        f[y]
        for f in core.read_frames(
            a.src, start=a.start or 0.0, max_frames=n, size=(w, h)
        )
    ]
    after = [f[y] for f in core.read_frames(out, max_frames=n)]
    m = min(len(before), len(after))
    return np.stack(before[:m]), np.stack(after[:m])


def _slices_row(
    before: np.ndarray,
    after: np.ndarray,
    width: int,
    labels: tuple[str, str] = ("元", "処理後"),
) -> np.ndarray:
    """時間の断面 2 枚を左右に並べた帯（幅 width）。"""
    half = width // 2
    cells = []
    for sl, name in zip((before, after), labels, strict=True):
        hh = max(160, sl.shape[0] * max(1, round(160 / max(1, sl.shape[0]))))
        c = cv2.resize(sl, (half, hh), interpolation=cv2.INTER_NEAREST)
        core.draw_text(
            c, f"{name}: 時間の断面（横 1 行を縦に積む。下へ時間）", (6, 6), size=15
        )
        cells.append(c)
    hmax = max(c.shape[0] for c in cells)
    cells = [
        np.vstack([c, np.full((hmax - c.shape[0], half, 3), 24, np.uint8)])
        for c in cells
    ]
    row = np.hstack(cells)
    if row.shape[1] < width:
        row = np.hstack(
            [row, np.full((row.shape[0], width - row.shape[1], 3), 24, np.uint8)]
        )
    return row


# ── 31 時間方向のノイズ除去 ───────────────────────────────────────────────

_TEMPORAL = {
    "mc": "28 の dct で 1 コマずつ消した後、前後のコマを動きに合わせて（DIS のオプティカルフロー）重ね、"
    "合わない所（動きの読み違い・隠れた所・カットの前後）は使わずに平均する（既定。動く所でも残像が出にくい）",
    "fftdnoiz": "ffmpeg の fftdnoiz（前後 1 コマを含む 3 次元 FFT。速い。動きの遅い映像向けで、速い動きでは効きが落ちる）",
    "hqdn3d": "ffmpeg の hqdn3d（空間と時間。とても速いが弱く、動く所に残像が出やすい）",
}


def _mc_average(
    frames: Iterator[np.ndarray],
    sigma: float,
    radius: int = 2,
    c: float = 8.0,
    patch: int = 5,
) -> Iterator[tuple[np.ndarray, float]]:
    """動き補償つきの時間平均。frames は（空間で軽く消した）コマの列。(平均したコマ, 重みの合計の平均) を順に返す。

    今のコマから前後 radius コマへの動き（DIS）を求めて重ね、5×5 の区画の明るさの差の二乗の平均 d² から
    重み exp(-d² / (c·(σ/3)²)) を付ける（σ は元のノイズの強さ。空間で消した後に残る差の目安）。
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    s2 = max(sigma / 3.0, 0.5) ** 2
    store: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    grid: list[np.ndarray] = []

    def blend(t: int, lo: int, hi: int) -> tuple[np.ndarray, float]:
        cur, g8, gf = store[t]
        h, w = g8.shape
        if not grid or grid[0].shape != (h, w):
            grid[:] = np.meshgrid(
                np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
            )
        acc = cur.astype(np.float32)
        wsum = np.ones((h, w), np.float32)
        for j in range(lo, hi + 1):
            if j == t or j not in store:
                continue
            other, o8, of = store[j]
            fl = dis.calc(g8, o8, None)
            mx, my = grid[0] + fl[..., 0], grid[1] + fl[..., 1]
            warped = cv2.remap(
                other, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            wl = cv2.remap(
                of, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            d2 = cv2.blur((wl - gf) ** 2, (patch, patch))
            wgt = np.exp(-d2 / (c * s2))
            acc += warped.astype(np.float32) * wgt[..., None]
            wsum += wgt
        out = np.clip(acc / wsum[..., None] + 0.5, 0, 255).astype(np.uint8)
        return out, float(wsum.mean())

    n = 0
    for k, f in enumerate(frames):
        g8 = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        store[k] = (f, g8, g8.astype(np.float32))
        n = k + 1
        t = k - radius
        if t >= 0:
            yield blend(t, t - radius, k)
            store.pop(t - radius, None)
    for t in range(max(0, n - radius), n):
        yield blend(t, t - radius, n - 1)


@command(
    31,
    "temporal-denoise",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_TEMPORAL),
        default="mc",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _TEMPORAL.items()),
    ),
    arg(
        "--sigma",
        type=float,
        help="明るさのノイズの強さ（標準偏差、0〜255。目安: 軽い 2〜3、目立つ 6〜7、ひどい 12〜20）。"
        "省略時は映像から推定する",
    ),
    arg(
        "--strength",
        type=float,
        default=1.0,
        help="強さの倍率（既定 1.0。0.6〜0.8 で細部や粒子を多めに残す、1.3〜1.5 で強め）",
    ),
    arg(
        "--radius",
        type=int,
        default=2,
        help="mc で前後に使うコマの数（既定 2 = 前後 2 コマずつ、計 5 コマ。1〜4）",
    ),
)
def temporal_denoise(a: argparse.Namespace) -> None:
    """前後のコマも使ってノイズのザラつきを消す（動画だけ）。1 コマずつの処理で残るノイズのちらつきも抑える。"""
    info = core.probe(a.src)
    _video_only(info, "画像は denoise（28）を使ってください")
    core.require(a.strength > 0, "--strength は 0 より大きくしてください")
    core.require(1 <= a.radius <= 4, "--radius は 1〜4 にしてください")
    out, json_path, sheet_path = _paths(a, info, "temporal-denoise")
    measured = _estimate_noise(a, info)
    sigma = (a.sigma if a.sigma is not None else measured) * a.strength
    core.log(
        f"推定したノイズの強さ: σ {measured:.2f}（使う強さ σ {sigma:.2f}、方法 {a.method}）"
    )
    extra: dict[str, Any] = {}
    if a.method == "fftdnoiz":
        filt = f"fftdnoiz=sigma={max(0.1, 2.0 * sigma):.2f}:prev=1:next=1"
        _ffmpeg_apply(a, info, [filt], out)
    elif a.method == "hqdn3d":
        s = max(0.1, sigma)
        filt = f"hqdn3d={s:.2f}:{0.75 * s:.2f}:{2 * s:.2f}:{1.5 * s:.2f}"
        _ffmpeg_apply(a, info, [filt], out)
    else:
        # 空間の dct は 28 より弱め（0.8 倍）にして細部を残し、残りのノイズは時間の平均で減らす
        # （試験で 1.0 倍・重み c4 より PSNR が 0.2〜0.7 dB 良かった。vfx-tests/restore/tune_mc.py）
        filt = _denoise_filter("dct", 0.8 * sigma)
        frames = _read_filtered(a, info, [filt])
        total = max(1, round(_span(a, info) * info.fps))
        weights = []
        with _Writer(a, info, out) as w:
            for i, (f, ws) in enumerate(_mc_average(frames, 0.8 * sigma, a.radius)):
                w.write(f)
                weights.append(ws)
                if (i + 1) % max(1, total // 10) == 0:
                    core.log(f"  {i + 1}/{total} コマ")
        extra = {
            "radius": a.radius,
            "frames": len(weights),
            "mean_frames_averaged": round(float(np.mean(weights)), 3),
        }
        core.log(
            f"重ねたコマの数（重みの合計の平均）: {extra['mean_frames_averaged']:.2f}"
            f"（最大 {2 * a.radius + 1}）"
        )
        filt = f"{filt} → 動き補償つきの時間平均（前後 {a.radius} コマ）"
    core.log(f"フィルター: {filt}")
    before, after, t = _frame_pair(a, info, out)
    box = _change_box(before, after)
    sl_b, sl_a = _time_slices(a, info, out, box[1] + box[2] // 2)
    top = _compare_sheet(
        before, after, ("元", f"時間方向のノイズ除去（{a.method}）"), box
    )
    sheet = np.vstack([top, _slices_row(sl_b, sl_a, top.shape[1])])
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "filter": filt,
            "noise_sigma_measured": round(measured, 3),
            "noise_sigma_used": round(sigma, 3),
            **extra,
            "check_frame": {
                "time": round(t, 4),
                "noise_sigma_before": round(_noise_sigma(before), 3),
                "noise_sigma_after": round(_noise_sigma(after), 3),
            },
        },
    )


def _line_chart(
    t: np.ndarray,
    series: list[tuple[np.ndarray, tuple[int, int, int], str]],
    title: str,
    width: int,
    height: int = 260,
    marks: list[float] | None = None,
) -> np.ndarray:
    """時間のグラフ（横軸は秒）。series は (値, 色 BGR, 名前)。白地の画像を返す。"""
    img = np.full((height, width, 3), 255, np.uint8)
    left, right, top, bottom = 60, 16, 34, 26
    pw, ph = width - left - right, height - top - bottom
    vals = np.concatenate([np.asarray(v, np.float64) for v, _, _ in series])
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    t0, t1 = float(t[0]), float(t[-1]) if len(t) > 1 else float(t[0]) + 1
    t1 = max(t1, t0 + 1e-3)

    def px(tt: Any) -> np.ndarray:
        return np.round(left + (np.asarray(tt, np.float64) - t0) / (t1 - t0) * pw)

    def py(v: Any) -> np.ndarray:
        return np.round(top + ph - (np.asarray(v, np.float64) - lo) / (hi - lo) * ph)

    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = int(py(v))
        cv2.line(img, (left, y), (left + pw, y), (232, 232, 232), 1)
        core.draw_text(
            img,
            f"{v:.1f}",
            (left - 6, y),
            size=12,
            color=(90, 90, 90),
            bg=None,
            anchor="rm",
        )
    step = next((s for s in (0.5, 1, 2, 5, 10, 30, 60) if (t1 - t0) / s <= 12), 120)
    for k in range(int(np.ceil(t0 / step)), int(t1 / step) + 1):
        x = int(px(k * step))
        cv2.line(img, (x, top), (x, top + ph), (232, 232, 232), 1)
        core.draw_text(
            img, f"{k * step:g}秒", (x, top + ph + 4), size=12, color=(90, 90, 90),
            bg=None, anchor="mt",
        )  # fmt: skip
    for tt in marks or []:
        x = int(px(tt))
        cv2.line(img, (x, top), (x, top + ph), (0, 170, 0), 2)
    x_label = left + 8
    for v, color, name in series:
        pts = np.stack([px(t[: len(v)]), py(v)], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
        core.draw_text(img, name, (x_label, top + 4), size=13, color=color, bg=None)
        x_label += 14 * len(name) + 30
    cv2.rectangle(img, (left, top), (left + pw, top + ph), (150, 150, 150), 1)
    core.draw_text(img, title, (left, 8), size=15, color=(20, 20, 20), bg=None)
    return img


# ── 36 フリッカー除去 ─────────────────────────────────────────────────────

_DEFLICKER = {
    "gain": "コマごとに、前後のコマの画素ごとの中央値との明るさの比（画素についての中央値。動く物に引きずられない）を"
    "求め、掛け算で直す（既定。カットをまたがない。光の強さの揺れは掛け算なので、白飛びの無い所ではほぼ元に戻る）",
    "ffmpeg": "ffmpeg の deflicker（前後 --window コマの平均の明るさに合わせる。カットをまたいで平均してしまう）",
}


def _brightness_series(
    a: argparse.Namespace, info: core.MediaInfo
) -> tuple[np.ndarray, list[np.ndarray]]:
    """コマごとの明るさ（Y の平均。0〜255）と、カットを見つけるための小さな灰色のコマ（幅 64px）。"""
    w = 64
    h = max(2, round(info.height * w / info.width / 2) * 2)
    means, smalls = [], []
    for f in core.read_frames(
        a.src,
        start=a.start or 0.0,
        duration=a.duration,
        max_frames=a.max_frames,
        size=(w, h),
    ):
        y = _luma(f)
        means.append(float(y.mean()))
        smalls.append(y)
    return np.array(means), smalls


def _flicker_cuts(smalls: list[np.ndarray], ratio: float = 4.0) -> list[int]:
    """カット（新しい場面の最初のコマ）。明るさの揺れでは反応しないよう、各コマを平均で割ってから比べる。

    差 d = 平均で割った小さなコマどうしの差の平均×100 が、前後 5 コマの差の中央値の ratio 倍かつ 8 以上ならカット。
    """
    n = len(smalls)
    d = np.zeros(n)
    for i in range(1, n):
        p = smalls[i - 1] / max(float(smalls[i - 1].mean()), 1.0)
        q = smalls[i] / max(float(smalls[i].mean()), 1.0)
        d[i] = float(np.abs(p - q).mean()) * 100
    cuts = []
    for i in range(1, n):
        near = np.concatenate([d[max(1, i - 5) : i], d[i + 1 : i + 6]])
        base = float(np.median(near)) if len(near) else 0.0
        if d[i] >= 8 and d[i] >= ratio * max(base, 0.5):
            cuts.append(i)
    return cuts


def _smooth_target(logm: np.ndarray, cuts: list[int], window: int) -> np.ndarray:
    """場面ごとに、明るさ（対数）を時間方向にならした目標（中央値で外れたコマを除き、移動平均でならす）。"""
    out = np.empty_like(logm)
    bounds = [0, *cuts, len(logm)]
    half = max(1, window // 2)
    for s, e in itertools.pairwise(bounds):
        seg = logm[s:e]
        med = np.array(
            [np.median(seg[max(0, i - half) : i + half + 1]) for i in range(len(seg))]
        )
        k = np.ones(2 * half + 1) / (2 * half + 1)
        pad = np.pad(med, half, mode="edge")
        out[s:e] = np.convolve(pad, k, mode="valid")[: len(seg)]
    return out


def _flicker_gains(
    smalls: list[np.ndarray], cuts: list[int], window: int, iters: int = 3
) -> np.ndarray:
    """コマごとに掛ける明るさの値。動く物（画面の一部）に引きずられないよう、画素ごとに求める。

    場面の中で、前後 window コマの画素ごとの中央値（対数の明るさ）を基準にし、そのコマとの差の、画素についての
    中央値をそのコマの揺れとする（白飛び・黒つぶれの画素は除く）。直した明るさで基準を作り直して 3 回繰り返す。
    基準は前後のコマに追従するので、window より遅い変化（日が陰る等）は残る。
    """
    y = np.stack(smalls).astype(np.float32)
    n = len(y)
    lg = np.log(np.maximum(y, 1.0))
    valid = (y > 8) & (y < 247)
    corr = np.zeros(n, np.float32)
    half = max(1, window // 2)
    bounds = [0, *cuts, n]
    for _ in range(iters):
        lc = lg + corr[:, None, None]
        r = np.zeros(n, np.float32)
        for s, e in itertools.pairwise(bounds):
            for t in range(s, e):
                lo, hi = max(s, t - half), min(e, t + half + 1)
                if hi - lo < 3:
                    continue
                ref = np.median(lc[lo:hi], axis=0)
                d = (lc[t] - ref)[valid[t]]
                if d.size > 50:
                    r[t] = float(np.median(d))
        corr -= r
    return np.exp(corr)


def _protect_events(gains: np.ndarray, max_change: float) -> list[tuple[int, int]]:
    """大きく直す必要のあるコマ（±max_change を超える）を含むひと続き（直す量が 8% を超える間）を、
    演出の変化（白いフラッシュ・フェードなど）とみなして直さない（gains を 1 にする）。(最初, 最後) の一覧を返す。"""
    dev = np.abs(np.log(gains))
    seeds = np.flatnonzero(dev > np.log(1 + max_change))
    events: list[tuple[int, int]] = []
    for i in seeds:
        if events and i <= events[-1][1]:
            continue
        s0 = e0 = int(i)
        while s0 > 0 and dev[s0 - 1] > 0.08:
            s0 -= 1
        while e0 < len(dev) - 1 and dev[e0 + 1] > 0.08:
            e0 += 1
        events.append((s0, e0))
        gains[s0 : e0 + 1] = 1.0
    return events


def _flicker_index(logm: np.ndarray, cuts: list[int], window: int) -> float:
    """ちらつきの強さ（%）: 明るさ（対数）の、ならした目標からの外れの標準偏差 ×100。"""
    if len(logm) < 3:
        return 0.0
    return float(np.std(logm - _smooth_target(logm, cuts, window)) * 100)


@command(
    36,
    "deflicker",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_DEFLICKER),
        default="gain",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _DEFLICKER.items()),
    ),
    arg(
        "--window",
        type=float,
        default=0.5,
        help="ならす時間の幅（秒。既定 0.5。ただし少なくとも 11 コマ）。これより速い明るさの揺れを消し、"
        "遅い変化（日が陰る等）は残す",
    ),
    arg(
        "--max-change",
        type=float,
        default=0.3,
        help="gain の時、これより大きく（±30%%）直す必要のあるコマとその前後は、演出（白いフラッシュ等）とみなして"
        "直さない（既定 0.3。強いちらつきも消したい時は 1）",
    ),
)
def deflicker(a: argparse.Namespace) -> None:
    """コマごとの明るさのちらつき（照明の明滅・タイムラプス・生成映像の明るさの揺れ）を抑える（動画だけ）。"""
    info = core.probe(a.src)
    _video_only(info)
    core.require(a.window > 0, "--window は 0 より大きくしてください")
    out, json_path, sheet_path = _paths(a, info, "deflicker")
    # 中央値の基準を安定させるため、少なくとも 11 コマ（10fps の素材で 5 コマだと誤差が 3.4%、13 コマで 1.9%）
    window = max(11, round(a.window * info.fps)) | 1
    means, smalls = _brightness_series(a, info)
    logm = np.log(np.maximum(means, 1.0))
    cuts = _flicker_cuts(smalls)
    fps, start = info.fps, a.start or 0.0
    core.log(
        f"{len(means)} コマ、カット {len(cuts)} 件"
        + (
            f"（{', '.join(f'{start + c / fps:.2f}' for c in cuts)} 秒）"
            if cuts
            else ""
        )
    )
    before_idx = _flicker_index(logm, cuts, window)
    if a.method == "ffmpeg":
        filt = f"deflicker=size={min(129, window)}:mode=am"
        core.log(f"フィルター: {filt}")
        _ffmpeg_apply(a, info, [filt], out)
        gains = None
        events = []
    else:
        gains = _flicker_gains(smalls, cuts, window)
        filt = f"明るさの掛け算（場面ごとに前後 {window} コマの画素ごとの中央値に合わせる）"
        events = _protect_events(gains, a.max_change)
        if events:
            core.log(
                f"大きな変化（{a.max_change * 100:.0f}% 超）{len(events)} か所は、演出（フラッシュ・フェード等）と"
                "みなして直していません: "
                + ", ".join(
                    f"{start + s0 / fps:.2f}〜{start + (e0 + 1) / fps:.2f} 秒"
                    for s0, e0 in events
                )
            )
        core.log(f"方法: {filt}。掛ける値 {gains.min():.3f}〜{gains.max():.3f}")
        total = len(gains)
        with _Writer(a, info, out) as w:
            for i, f in enumerate(_read_filtered(a, info)):
                g = float(gains[min(i, total - 1)])
                w.write(
                    np.clip(f.astype(np.float32) * g + 0.5, 0, 255).astype(np.uint8)
                )
                if (i + 1) % max(1, total // 10) == 0:
                    core.log(f"  {i + 1}/{total} コマ")
    out_means, _ = _brightness_series(
        argparse.Namespace(src=out, start=0.0, duration=None, max_frames=None),
        core.probe(out),
    )
    out_log = np.log(np.maximum(out_means, 1.0))
    after_idx = _flicker_index(out_log, cuts, window)
    core.log(
        f"ちらつきの強さ（明るさの揺れの標準偏差）: {before_idx:.2f}% → {after_idx:.2f}%"
    )
    # 確認画像: 明るさのグラフと、最も明るさを直したコマの前後
    i_max = (
        int(np.argmax(np.abs(np.log(gains)))) if gains is not None else len(means) // 2
    )
    before, after, _t = _frame_pair(a, info, out, i_max)
    box = _change_box(before, after)
    top = _compare_sheet(before, after, ("元", f"フリッカー除去（{a.method}）"), box)
    ts = start + np.arange(len(means)) / fps
    m = min(len(ts), len(out_means))
    chart = _line_chart(
        ts[:m],
        [
            (means[:m], (60, 60, 230), "元の明るさ"),
            (out_means[:m], (200, 110, 20), "処理後"),
        ],
        f"コマごとの明るさ（Y の平均）  ちらつき {before_idx:.2f}% → {after_idx:.2f}%"
        "  緑の線 = カット",
        top.shape[1],
        marks=[start + c / fps for c in cuts],
    )
    sl_b, sl_a = _time_slices(a, info, out, box[1] + box[2] // 2)
    sheet = np.vstack([top, chart, _slices_row(sl_b, sl_a, top.shape[1])])
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "filter": filt,
            "window_frames": window,
            "max_change": a.max_change,
            "protected_events": [
                [round(start + s0 / fps, 4), round(start + (e0 + 1) / fps, 4)]
                for s0, e0 in (events if gains is not None else [])
            ],
            "cuts": [round(start + c / fps, 4) for c in cuts],
            "flicker_percent_before": round(before_idx, 3),
            "flicker_percent_after": round(after_idx, 3),
            "frames": [
                {
                    "time": round(start + i / fps, 4),
                    "mean_before": round(float(means[i]), 3),
                    "mean_after": round(float(out_means[i]), 3)
                    if i < len(out_means)
                    else None,
                    **(
                        {"gain": round(float(gains[i]), 4)} if gains is not None else {}
                    ),
                }
                for i in range(len(means))
            ],
        },
    )


# ── 41 デインターレース ───────────────────────────────────────────────────

_DEINT = {
    "bwdif": "ffmpeg の bwdif（前後のフィールドと w3fdif の係数で補う。動画の既定。速くて質が良い）",
    "yadif": "ffmpeg の yadif（前後のフィールドで補う。古くからの定番）",
    "estdif": "ffmpeg の estdif（同じフィールドの中の輪郭の傾きに沿って補う。画像 1 枚の既定）",
    "w3fdif": "ffmpeg の w3fdif（BBC の 3 フィールドの係数で補う）",
}
_IDET = re.compile(
    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)"
)
_IDET1 = re.compile(
    r"Single frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)"
)


def _idet(
    a: argparse.Namespace, info: core.MediaInfo, limit: int = 300
) -> dict[str, Any]:
    """ffmpeg の idet で、インターレースか（上のフィールドが先 TFF・下が先 BFF・プログレッシブ）を数える。"""
    args: list[str] = []
    if a.start:
        args += ["-ss", f"{a.start:.4f}"]
    args += ["-i", str(a.src)]
    n = min(limit, max(1, round(_span(a, info) * info.fps)))
    args += ["-frames:v", str(n), "-vf", "idet", "-an", "-f", "null", "-"]
    text = core.ffmpeg_log(args)
    multi = _IDET.findall(text)
    single = _IDET1.findall(text)
    keys = ("tff", "bff", "progressive", "undetermined")
    m = dict(zip(keys, map(int, multi[-1]), strict=True)) if multi else {}
    s1 = dict(zip(keys, map(int, single[-1]), strict=True)) if single else {}
    decided = m.get("tff", 0) + m.get("bff", 0) + m.get("progressive", 0)
    inter = m.get("tff", 0) + m.get("bff", 0)
    verdict = "不明"
    if decided:
        verdict = "インターレース" if inter > 0.5 * decided else "プログレッシブ"
    parity = "tff" if m.get("tff", 0) >= m.get("bff", 0) else "bff"
    return {"frames": n, "multi": m, "single": s1, "verdict": verdict, "parity": parity}


def _deint_filter(method: str, rate: str, parity: str) -> str:
    field = rate == "field"
    if method in ("bwdif", "yadif"):
        p = {"tff": 0, "bff": 1, "auto": -1}[parity]
        return f"{method}=mode={1 if field else 0}:parity={p}:deint=0"
    if method in ("estdif", "w3fdif"):
        return (
            f"{method}=mode={'field' if field else 'frame'}:parity={parity}:deint=all"
        )
    raise core.VfxError(f"知らない方法です: {method}")


@command(
    41,
    "deinterlace",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_DEINT),
        help="方法（省略時は動画 bwdif・画像 estdif）。"
        + " ／ ".join(f"{k}: {v}" for k, v in _DEINT.items()),
    ),
    arg(
        "--rate",
        choices=["frame", "field"],
        default="frame",
        help="frame: コマ数はそのまま（既定） ／ field: 1 フィールドを 1 コマにしてコマ数を 2 倍に（動きがなめらか。"
        "50i → 50p など）",
    ),
    arg(
        "--parity",
        choices=["auto", "tff", "bff"],
        default="auto",
        help="どちらのフィールドが先か（auto: idet で調べた結果。画像は tff）",
    ),
    arg(
        "--detect-only",
        action="store_true",
        help="インターレースかどうかを調べて JSON に書くだけ（映像は作らない）",
    ),
)
def deinterlace(a: argparse.Namespace) -> None:
    """インターレースの横縞（動く物の縁のくし状のギザギザ）を消す。まず idet でインターレースかを調べて報告する。"""
    info = core.probe(a.src)
    out, json_path, sheet_path = _paths(a, info, "deinterlace")
    det: dict[str, Any] = {}
    parity = a.parity
    if info.is_image:
        core.log(
            "画像 1 枚なので、インターレースかどうかは調べず、同じフィールドの中で補います"
        )
        parity = "tff" if parity == "auto" else parity
    else:
        det = _idet(a, info)
        m = det["multi"]
        core.log(
            f"idet（{det['frames']} コマ）: 上が先 {m.get('tff', 0)}・下が先 {m.get('bff', 0)}・"
            f"プログレッシブ {m.get('progressive', 0)}・不明 {m.get('undetermined', 0)} → {det['verdict']}"
            + (
                f"（{'上' if det['parity'] == 'tff' else '下'}のフィールドが先）"
                if det["verdict"] == "インターレース"
                else ""
            )
        )
        if parity == "auto":
            parity = det["parity"]
        if det["verdict"] == "プログレッシブ":
            core.log(
                "  インターレースではないようです。かけると縦の細かさが少し落ちることがあります"
                + ("" if a.detect_only else "（そのままかけます）")
            )
    if a.detect_only:
        core.require(not info.is_image, "--detect-only は動画だけです")
        core.write_json(json_path, {"source": str(a.src), "idet": det})
        core.log(f"保存: {json_path}")
        return
    method = a.method or ("estdif" if info.is_image else "bwdif")
    core.require(
        not (info.is_image and method == "bwdif"),
        "画像 1 枚には bwdif は使えません（前後のフィールドが無いため。estdif・yadif・w3fdif を使ってください）",
    )
    filt = _deint_filter(method, a.rate if not info.is_image else "frame", parity)
    core.log(f"フィルター: {filt}")
    # 縮小すると 2 つのフィールドが混ざるので、縮小の前にかける
    field = not info.is_image and a.rate == "field"
    _ffmpeg_apply(
        a, info, [filt], out, scale_first=False, frames_factor=2 if field else 1
    )
    if field:
        # 2 倍のコマ数になる: 結果の 2k 番目が、元の k 番目の先のフィールドの時刻
        oinfo = core.probe(out)
        k = round(oinfo.duration * oinfo.fps) // 4
        before, _, t = _frame_pair(a, info, out, k)
        after = core.frame_at(out, max(0.0, (2 * k - 0.5) / oinfo.fps))
    else:
        before, after, t = _frame_pair(a, info, out)
    sheet = _compare_sheet(before, after, ("元", f"デインターレース（{method}）"))
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": method,
            "filter": filt,
            "rate": a.rate,
            "parity": parity,
            "idet": det,
            "check_frame": {"time": round(t, 4)},
        },
    )


# ── 46 ゴミ・傷の除去 ─────────────────────────────────────────────────────

_DUST = {
    "detect": "前後のコマを動きに合わせて（DIS のオプティカルフロー）重ね、その 2 コマが互いに合っているのに、このコマだけが"
    "同じ向きに大きく違う所（1 コマだけのゴミ・傷）を見つけ、前後のコマの平均で埋める（既定。それ以外の画素は変えない）",
    "tmedian": "ffmpeg の tmedian（前後 1 コマとの画素ごとの中央値。速いが、動きに合わせないので動く所が崩れる）",
}


class _Flow:
    """DIS のオプティカルフローで、別のコマを今のコマに重ねる。"""

    def __init__(self) -> None:
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.grid: tuple[np.ndarray, np.ndarray] | None = None

    def warp(
        self, cur_g: np.ndarray, other_g: np.ndarray, other: np.ndarray
    ) -> np.ndarray:
        h, w = cur_g.shape
        if self.grid is None or self.grid[0].shape != (h, w):
            self.grid = tuple(
                np.meshgrid(
                    np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
                )
            )  # type: ignore[assignment]
        assert self.grid is not None
        fl = self.dis.calc(cur_g, other_g, None)
        return cv2.remap(
            other,
            self.grid[0] + fl[..., 0],
            self.grid[1] + fl[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )


def _dust_fix(
    cur: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    flow: _Flow,
    threshold: float,
    max_area: int,
    min_area: int = 12,
) -> tuple[np.ndarray, np.ndarray, int]:
    """1 コマのゴミ・傷を、別の 2 コマ p・q（ふつうは前と後）を使って見つけて埋める。(結果, 埋めた所, 見つけた数)

    明るさ Y で、今のコマ − 重ねた p と、今のコマ − 重ねた q が、同じ符号でどちらも threshold を超え、しかも重ねた p と q の
    差が threshold の半分未満の画素を候補にする（動きの読み違い・隠れていた物が出てきた所では p と q が合わない）。
    候補を 1 px 広げてまとめ、塊ごとに次を除く: 面積が max_area を超える・min_area 未満（1 画素だけの誤り）・
    まわり 9×9 で重ねた p と q の差の平均が threshold/4 以上（動きの読み違い。速く動く手足が 1 コマだけ見える所など）。
    """
    cg, pg, qg = (
        cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
        for x in (cur, p, q)
    )
    wp = flow.warp(cg, pg, p)
    wq = flow.warp(cg, qg, q)
    yc, yp, yq = _luma(cur), _luma(wp), _luma(wq)
    ep, eq = yc - yp, yc - yq
    cand = (
        (np.abs(ep) > threshold)
        & (np.abs(eq) > threshold)
        & (np.sign(ep) == np.sign(eq))
        & (np.abs(yp - yq) < threshold / 2)
    )
    cand = cv2.dilate(cand.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    # 塊のまわり（9×9）で、重ねた前後の 2 コマが合っているか。動きの読み違いで出た塊は、まわりでも 2 コマが
    # 合っていない（試験: 本物のゴミは中央値 0.7〜1.7、誤りは 18〜19）
    agree = cv2.blur(np.abs(yp - yq), (9, 9))
    area = stats[:, cv2.CC_STAT_AREA]
    ring = np.bincount(lab.ravel(), weights=agree.ravel(), minlength=n) / np.maximum(
        area, 1
    )
    keep = (area <= max_area) & (area >= min_area) & (ring < threshold / 4)
    keep[0] = False
    mask = keep[lab]
    if not mask.any():
        return cur, mask, 0
    fill = (wp.astype(np.float32) + wq.astype(np.float32)) / 2
    # 見つけた所（1 px 広げた）は置き換え、その外側 1〜2 px はなじませる
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)
    soft[mask] = 1.0
    out = cur.astype(np.float32) * (1 - soft[..., None]) + fill * soft[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), mask, int(keep[1:].sum())


@command(
    46,
    "dust-removal",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_DUST),
        default="detect",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _DUST.items()),
    ),
    arg(
        "--threshold",
        type=float,
        default=25.0,
        help="ゴミとみなす明るさの差（0〜255。既定 25。薄いゴミも消すなら 15、動く所の誤りを減らすなら 40）",
    ),
    arg(
        "--max-size",
        type=float,
        default=0.3,
        help="1 つのゴミの面積の上限（画面に対する %%。既定 0.3%%。これより大きい塊は動きの読み違いとみなして残す）",
    ),
)
def dust_removal(a: argparse.Namespace) -> None:
    """1 コマだけに現れるゴミ・傷・白や黒の点を、前後のコマから見つけて消す（動画だけ）。"""
    info = core.probe(a.src)
    _video_only(
        info,
        "画像のゴミは、場所を指定して消す穴埋め（切り抜き・合成の項目）を使ってください",
    )
    core.require(a.threshold > 0, "--threshold は 0 より大きくしてください")
    out, json_path, sheet_path = _paths(a, info, "dust-removal")
    per_frame: list[int] = []
    masks_px: list[int] = []
    if a.method == "tmedian":
        # tmedian は最初と最後のコマを落とし、1 コマずれる（試験で確認）ので、両端に同じコマを 1 つずつ足す
        filt = "tpad=start=1:stop=1:start_mode=clone:stop_mode=clone,tmedian=radius=1"
        core.log(f"フィルター: {filt}")
        _ffmpeg_apply(a, info, [filt], out)
    else:
        w, h = _proc_size(info, a.max_side)
        max_area = max(4, round(w * h * a.max_size / 100))
        filt = f"前後のコマとの比較（しきい値 {a.threshold:g}、1 つの面積 {max_area} px まで）"
        core.log(f"方法: {filt}")
        flow = _Flow()
        total = max(1, round(_span(a, info) * info.fps))

        def fix(cur: np.ndarray, p: np.ndarray, q: np.ndarray) -> None:
            fixed, mask, cnt = _dust_fix(cur, p, q, flow, a.threshold, max_area)
            wr.write(fixed)
            per_frame.append(cnt)
            masks_px.append(int(mask.sum()))
            if len(per_frame) % max(1, total // 10) == 0:
                core.log(f"  {len(per_frame)}/{total} コマ")

        # 前・今・後の 3 コマだけを持って順に進む（最初のコマは後の 2 コマ、最後のコマは前の 2 コマと比べる）
        reader = iter(_read_filtered(a, info))
        f0, f1, f2 = next(reader), next(reader, None), next(reader, None)
        core.require(
            f1 is not None and f2 is not None, "3 コマ以上の動画を指定してください"
        )
        assert f1 is not None and f2 is not None
        with _Writer(a, info, out) as wr:
            fix(f0, f1, f2)
            prev, cur, nxt = f0, f1, f2
            while True:
                fix(cur, prev, nxt)
                new = next(reader, None)
                if new is None:
                    break
                prev, cur, nxt = cur, nxt, new
            fix(nxt, cur, prev)
        core.log(
            f"見つけて埋めた所: 合計 {sum(per_frame)} か所（{sum(1 for c in per_frame if c)} コマ）、"
            f"1 コマの最大 {max(per_frame)} か所"
        )
    k = int(np.argmax(per_frame)) if per_frame and max(per_frame) else None
    before, after, t = _frame_pair(a, info, out, k)
    sheet = _compare_sheet(before, after, ("元", f"ゴミ・傷の除去（{a.method}）"))
    fps, start = info.fps, a.start or 0.0
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "filter": filt,
            "threshold": a.threshold,
            "total_spots": sum(per_frame),
            "frames": [
                {"time": round(start + i / fps, 4), "spots": c, "pixels": px}
                for i, (c, px) in enumerate(zip(per_frame, masks_px, strict=True))
                if c
            ],
            "check_frame": {"time": round(t, 4)},
        },
    )
