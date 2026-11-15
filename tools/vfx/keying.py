"""切り抜き・合成（キーとマット）。

担当するテクニック:
   52 クロマキー（chroma-key）
   57 スピル除去（despill）
   62 ガベージマット（garbage-matte）
   66 フェザー（feather）
   67 ルミナンスキー（luma-key）
   74 差分キー（diff-key）
   75 クリーンプレート（clean-plate）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import core
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/keying/（make_*.py で正解つきの合成素材を作り、test_*.py で照らし、
# review_*.py で確認画像を作る）。合成素材の人物はすべて AI 生成の広告映像（plate-*・cut-*）から取った。
# 正解つきの合成素材（<スクラッチパッド>/vfx-test/synth/keying/）:
#   green-clean / green-raw / green-stand / blue-clean: むらのある緑（青）の上の人物。正解のアルファ・前景の色・
#     スピルの重み・不要物の地図つき（make_green.py）
#   fx-black / ink-white: 黒地の火花と煙・白地の墨。正解のアルファと乗算済みの色つき（make_luma.py）
#   plaza: 固定カメラの広場を人が通る。正解の背景・人のアルファ・影の地図つき（make_plaza.py）
NOTES: dict[int, dict[str, str]] = {
    52: {
        "status": "◎",
        "how": "色差キー（Vlahos の差分マット。緑なら G − max(R, B)）を、場所ごとの背景の色で割る（Nuke の IBK と同じ"
        "考え方）。背景らしい画素（緑の強さと、その G に対する比が代表の色に近い）を種にして、人物の後ろの背景の色を"
        " push-pull（Gortler ら 1996）で埋め、大まかな背景で一度アルファを求めてほぼ透明な所を種にし直す 2 段で作る。"
        "アルファ = 1 − 緑の強さ ÷ その場所の背景の緑の強さ（黒 0.08・白 0.9 で切り詰め）。境目の帯だけ、すぐ内側の"
        "不透明な所の色と背景の色を結ぶ線への射影で求め直す（色差の式は肌・クリーム色の縁で濃く出るため）。前景の色は"
        "背景の色を差し引いて割り戻し（縁の緑が消える）、続けてスピル除去（57 の average）、半透明の縁は「緑を赤・青の"
        "小さい方より下げない色」と内側の色を半々に混ぜる（引きすぎの赤紫止め）。代表の色は画面の端 8% の帯から自動"
        "（緑か青か）か --color。--method chromakey / hsvkey で ffmpeg のフィルター（hsvkey は明るさを無視させる）。"
        "出力: 透過 WebM（VP9）・PNG、--bg で背景（画像・動画）に重ねた MP4・画像、--alpha-out（白黒）、--mask"
        "（62 のマスクで外す）、確認画像、JSON（コマごとの透明・半透明の割合）。外部のモデルは使わない",
        "measured": "正解のアルファと前景の色がある合成グリーンバック（AI 生成の人物、むらのある緑〔明るさ 0.6〜1.15 倍・"
        "しわ・場所による色味〕、縁の強いスピル、ノイズ σ2、H.264 の 4:2:0。540×960・40 コマのうち 8 コマ）: IoU 0.9962"
        "（最小 0.9943）、アルファの平均誤差 0.79/255、境目の帯 29.1、story のカフェに重ねた色の差 1.63（帯 7.68）。"
        "射影なしは 0.9958・1.15・帯 32.0、場所ごとの背景を使わず代表の色 1 つで割ると 0.9886・6.37（背景を前景と誤る"
        " 1806px）。ブルーバック 0.9963・0.73。縁に元の撮影の明るさが残る版（green-raw）0.9949・0.94。ffmpeg chromakey"
        " は最良の 0.05/0.03 で 0.9923・1.83・帯 51.0（暗い隅を前景と誤る 390px）、0.08/0.05 で人物の 3415px が抜け、0.12"
        " で人物がほぼ消える（IoU 0.07）。hsvkey は明るさを使うと最良 0.889、無視させて 0.03/0.02 で 0.9930・1.48。"
        "ブルーバックの 1 コマで chromakey は暗い左側の 13197px を前景と誤った（diff は誤差 0.53/255）。縁の色: 帯の緑の"
        "出っ張りはスピル除去なし 3.23 → 1.47（正解 1.39）。半透明の縁の赤紫（引きすぎ）は +0.34 → −0.04、帯の色の差"
        " 8.12 → 7.68。速さ: 540×960・40 コマで 8.6 秒（WebM の書き出しを含む）",
        "limits": "スピルの強い縁は、透けているのか緑を浴びたのか色だけでは区別できず、アルファが薄く出る（強いスピルを足した"
        "合成で境目の帯の平均 −0.06〜−0.1。正解の前景の色を与えた射影なら誤差 0.7〜3/255 なので、限界は前景の色の推定）。"
        "薄い時は --clip-white を下げる。背景に近い緑・青の服や小物は抜ける（キーの原理）。1 本の髪の毛の細かさは、"
        "背景の色の推定（2px のぼかし）と 4:2:0 の色の解像度で丸まる。画面の端の 5% 未満しか緑・青でないと止まる"
        "（--color で指定）。照明スタンドなど緑でない物は前景に残る（62 で外す）。ffmpeg の方法は前景の色を直さず"
        "（スピル除去だけ掛ける）、明るさのむらに弱く、しきい値の幅も狭い。実写のグリーンバック素材は手元に無く、確かめたのは"
        "合成した素材だけ",
        "tested": "make_green.py の green-clean・green-raw・blue-clean（正解つき）、test_chroma.py（方法と設定の比較）、"
        "diag_band*.py（境目の誤差の出どころ）、test_fringe.py（縁の色）、test_hsvkey.py、review_chroma.py（目で確認: "
        "vfx-review/52-chroma-key.jpg・52-chroma-key-blue.jpg）、CLI で WebM・--bg（story540 の動画）・--alpha-out・画像 1 枚",
    },
    57: {
        "status": "◎",
        "how": "緑なら G を上限 L で切る: average L = (R+B)/2（既定。ffmpeg の despill の既定〔mix 0.5・expand 0〕と同じ式で、"
        "同じ画像で最大 1/255 の差）/ max L = max(R, B)（弱い。黄・水色を守る）/ min L = min(R, B)（強い）。青は R と G で"
        "同じ。--strength で効き具合、--keep-luma で消した分の明るさ（Rec.709 の係数）を 3 色に足し戻す、--mask で場所を"
        "限る。透過つきの入力（WebM・PNG）はアルファを保つ。出力: 入力と同じ形式、確認画像（元・結果・消した量の地図）、JSON",
        "measured": "52 の合成（スピルの重みが分かっている）をスピル除去なしでキーした前景に掛け、正解の色（スピル前）と"
        " Lab の ΔE で比べた（不透明な所）: スピルの強い所（重み 0.15 以上）で なし 9.75 → average 4.15・max 6.80・min 7.46・"
        "average＋--keep-luma 6.04、ほぼ無い所で 3.06 → 2.60・2.89・14.76（min は肌を赤くする）・2.67。ブルーバックの強い所:"
        " 7.49 → 5.42・5.69・4.98・5.50。かぶりの無い色見本の変化（average / max / min）: 黄 48.7 / 0 / 91.9、水色 45.4 / 0 /"
        " 107.6、肌 0 / 0 / 14〜21、葉の緑 66.5 / 58.6 / 75.0、クリーム 4.7 / 0 / 16.4、白・灰・赤・紫は 0。速さ: 540×960 の"
        "透過 WebM 12 コマで 2.1 秒",
        "limits": "「緑が赤と青より多い」所を一律に直すので、本当に緑の物（葉・緑の服）は灰色になる（--mask で場所を限る）。"
        "average は黄色をオレンジに、水色を青に寄せる（黄・水色が大事なら max）。緑の光で一緒に足された赤・青の分は"
        "消えないので、かぶった所は少し明るいまま。--keep-luma はこの試験では正解から遠くなった（明るさを足すため）。"
        "実写のかぶりは試していない（合成した素材と色見本だけ）",
        "tested": "test_despill.py（make_green.py の合成・色見本・ffmpeg despill との比較）、review_despill.py（目で確認: "
        "vfx-review/57-despill.jpg・57-despill-chart.jpg）、CLI で透過 WebM・普通の MP4（--method max）・画像",
    },
    62: {
        "status": "◎",
        "how": "多角形・四角（元の映像の px か %）を残す（--keep）・外す（--drop）範囲にし、4×4 の細かさで塗って縮めた 0〜1 の"
        "マスクにする（縁の画素は半分の値。多角形は 1 つずつ塗って重ねる）。'@秒 ...; @秒 ...' で時間とともに直線で"
        "動かせる（前後は端の形のまま）。--spec の JSON でも指定できる。--feather で縁をぼかす（66 と同じ）、--invert。"
        "透過つきの入力はアルファに掛け、普通の映像は外した所を透明にした WebM・PNG。--mask-only でマスク（白 = 残す）"
        "だけ、--mask-out でマスクも書く（52・67・74 の --mask に渡せる。灰色のまま書くので白が 255 のまま）。"
        "確認画像は範囲の線（緑 = 残す・赤 = 外す）・マスク・結果",
        "measured": "合成グリーンバックに不要物（左上の黒い旗と支柱〔動かない〕、右から左へ一定の速さで動くブームマイク）を"
        "置いた 40 コマ: 52 のキーだけでは不要物が前景に残り、人物との IoU 0.9328（不要物の残り 1 コマ平均 22314px）。"
        "動かない多角形 1 つと、2 つの時刻で動かす四角 1 つで外すと IoU 0.9964・アルファの平均誤差 0.83/255・不要物の"
        "残り 0px（全コマ）。マスクを先に作って 52 の --mask に渡しても 0.9964・0.84。塗りの正確さ（面積の差）: 端数の"
        "位置の 100×50 の四角 +0.013%、三角 +0.002%、30 度回した正方形 +0.042%、3.5×2.5px の小さな四角 +0.07px²。"
        "動く四角の左端（4 つの時刻）は期待より +0.13px。試験の途中で直した不具合: 多角形をまとめて塗ると重なった所が"
        "穴になり不要物の 9% が残った、マスクの MP4 の白が 250 になり人物が 2% 透けた。速さ: 540×960・40 コマの"
        " WebM で 3.3 秒",
        "limits": "形は手で指定する（自動では追わない。動く物は 2 つ以上の時刻で形を決め、間は直線で動く。曲がった動きは"
        "時刻を増やす）。どの時刻も同じ数の点にする。外す範囲に人物が入ると人物も消える（52 の確認画像を見て範囲を"
        "決める）。物に沿って自動で追うなら 60（mask-track）",
        "tested": "make_green.py の green-stand（不要物の地図つき）、test_garbage.py（不要物の残り・人物の IoU・塗りの面積・"
        "動く形）、review_garbage.py（目で確認: vfx-review/62-garbage-matte.jpg）、画像 1 枚（--drop-box の %）",
    },
    66: {
        "status": "◎",
        "how": "アルファの縁を明るさのモルフォロジー（楕円の最大・最小。端数は前後の 2 つの大きさを混ぜる）で縮める"
        "（--choke）・広げる（--spread）、ガウスでぼかす（--blur は 10〜90% の幅 W。σ = W ÷ 2.563）。--where で置き場所:"
        " center = 元の縁が 50%、inside = 元の縁で 10%（ぼかしを内側に収め、外の背景がはみ出さない）、outside = 元の縁で"
        " 90%。透過つき WebM・PNG はアルファを直し（広げた所の色は内側の色を埋め広げる）、透過の無い入力は白黒のマスク"
        "として直す。--bg で背景に重ねた映像も",
        "measured": "2 値の円（半径 150）・長方形で、元の縁からの距離ごとのアルファの平均から求めた位置（円 / 長方形）: "
        "blur 4・10・20 の 10〜90% の幅 3.70・9.52・19.53 / 4.14・10.05・20.03、50% のずれ 0.13px 以内。inside（blur 10）:"
        " 10% が +0.02 / +0.15px、50% が −4.87 / −4.88（期待 0・−5）。outside: 90% が +0.12 / +0.10、50% が +4.82 / +5.12。"
        "choke 5 → 50% が −4.62 / −4.88、choke 2.5 → −2.38 / −2.38、spread 7 → +6.88 / +7.12、choke 3＋blur 6 → "
        "−2.66 / −2.88・幅 5.97 / 6.09。半透明のある実際のアルファ（AI 生成の人物の切り抜き）: choke 2・spread 3・"
        "choke 1.5 で 0.5 の輪郭が平均 −1.87・+2.87・−1.32px 動いた。速さ: 1080×1920 で choke 10 が 1 コマ 21ms、"
        "blur 20 が 12ms",
        "limits": "楕円の形のモルフォロジーなので、曲がった縁の縮み・広がりは 0.1〜0.4px 小さく出る（円で choke 5 → "
        "4.62）。端数（2.5 など）は前後の大きさを混ぜるだけなので、2 値の縁には 1px の段がつく。縮めると細い所（毛先など）"
        "から先に消える（モルフォロジーの性質）。広げた所の色は内側の色を埋め広げた物で、本当の色ではない",
        "tested": "test_feather.py（2 値の円・長方形の正解の縁で 10・50・90% の位置、cut-stand の正解のアルファ、CLI の PNG・"
        "WebM・--bg、速さ）、review_feather.py（目で確認: vfx-review/66-feather.jpg）",
    },
    67: {
        "status": "◎",
        "how": "黒い背景（既定）: 値 = 3 色の最大（--channel max。色の濃い火花も明るく数える）、アルファ = (値 − low) ÷ "
        "(high − low)（既定 0.04〜1.0。明るさに比例して透ける）。白い背景（--invert）: 値 = 1 − 3 色の最小。前景の色は"
        "背景の黒（白）を差し引いて割り戻す（乗算済みの色として扱う。縁の黒ずみ・白っぽさが出ない）。--keep-color で元の"
        "色のまま、--channel luma で輝度（ffmpeg の lumakey と同じ考え方）。出力は 52 と同じ（透過 WebM・PNG、--bg、"
        "--alpha-out、--mask、確認画像、JSON）",
        "measured": "黒地に手続き的に作った火花（橙。3 色の最大が 1）と煙（明るい灰。最大 0.82）（正解のアルファと乗算済みの"
        "色。ノイズ σ1・H.264。540×960・48 コマのうち 8 コマ）を story のカフェに重ね、正解の合成と比べた色の差（0〜255）:"
        " 既定 火花 8.34・煙 4.85、low 0 で 7.96・3.92、high 0.82（煙の本当の明るさ）で煙 2.65・火花 9.51、luma で火花"
        " 15.84（橙を暗く数える）、--keep-color で 17.65・30.16（煙が灰色に濁り、火花のにじみが 21 暗い）、ffmpeg lumakey"
        "（0/0.04/0.96）で 21.77・32.92（にじみが 22 暗い）。黒が 0.03 浮いてノイズ σ4 の素材では、素材の無い所の背景の"
        "濁りが low 0 で 3.84、0.04 で 0.86、0.06 で 0.14。白地の墨（濃さ 0.15 の灰）: 既定 5.28、low 0・high 0.85"
        "（素材に合わせた値）で 1.11、--keep-color で 30.13。速さ: 540×960・48 コマ（WebM）で 6.8 秒",
        "limits": "明るさから透け具合を決めるので、素材の本当の濃さは分からない（灰色の煙は最大 0.82 の明るさなので、既定"
        "では正解より薄く重なる。濃くしたい時は --high を素材のいちばん明るい所の値に下げる）。黒い煙・影など暗い部分は"
        "黒い背景からは抜けない。low を上げると薄い煙の端が欠ける。白地の素材の明るい部分（白い服など）は透ける。"
        "実写の火花・煙の素材は試していない（手続き的に作った素材だけ）",
        "tested": "make_luma.py の fx-black・ink-white（正解つき）、test_luma.py（方法と設定・ffmpeg lumakey・黒の浮き）、"
        "review_luma.py（目で確認: vfx-review/67-luma-key.jpg）、CLI で WebM・--bg（MP4）・--invert・画像 1 枚",
    },
    74: {
        "status": "◎",
        "how": "プレート（--plate の画像。省略時は 75 と同じ中央値。--exclude-people で人を除いた中央値）とコマを少しぼかし、"
        "差の小さい半分の画素の色の比で露出のゆらぎを合わせ、Lab の色の差（ΔE76）を求める。しきい値 12（前後 8 の幅で"
        " 0〜1）で切り、開く・閉じる処理（短い辺の 1/270）と小さな塊・穴（画面の 0.05% 未満）の除去で整え、縁の ±2px"
        "（短い辺の 1/270）だけ色の差のなだらかな値を使う。影は既定で前景に含め、--shadows remove で Horprasert ら"
        "（1999）の考え方（背景を 0.35〜0.95 倍しただけで、色のずれ 6% 未満）で背景にする。出力は 52 と同じ（透過"
        " WebM・PNG、--bg、--alpha-out、--mask、確認画像、JSON）。プレートを作った時は <出力>-plate.png も",
        "measured": "固定カメラの広場の合成（vtest の中央値〔正解の背景〕に AI 生成の人物 4 人。1 人は 75% のコマで同じ所に"
        "立ち止まる。影 1 つ・露出のゆらぎ ±2.5%・ノイズ σ2・H.264。768×576・100 コマのうち 13 コマ。影の所を除き、"
        "全コマまとめた IoU）: 正解の背景をプレートにして IoU 0.963・アルファの平均誤差 0.72/255、75 の人を除いた中央値で"
        " 0.968・0.63、単純な中央値では立ち止まった人がプレートに入り 0.731（その人の 46% しか抜けない）。縁の作り方:"
        " 色の差のなだらかな値 0.87・帯 24.8（しきい値 14 で比較）、ガイデッドフィルター 1.91・帯 55.9、2 値 0.90。"
        "しきい値 10・12・14 で IoU 0.962・0.963・0.955（10 は背景の誤り、14 は抜けが増える）。--shadows remove: 影の"
        " 99.3% を背景にできたが、灰緑の服の立ち止まった人の 53% も影として消え IoU 0.809（このため既定は keep）。"
        "速さ: 768×576 で 1 コマ約 50ms（キーだけ）、vtest 100 コマ（--plate・--bg・--alpha-out）で 9.2 秒、広場の"
        "合成 100 コマ（--exclude-people でプレートから作る）で 23.7 秒",
        "limits": "固定カメラ専用（カメラが動くとプレートと合わない。先に手ぶれ補正で止める）。背景に近い色の服・物は"
        "抜ける（--threshold を下げる）。影は既定で前景になる（remove は無彩色の暗い服の人も消す）。照明が大きく変わる"
        "所・背景が動く所（木の揺れ・水面・画面）は前景に出る。半透明（毛先）は色の差の幅でしか表せない",
        "tested": "make_plaza.py（正解の背景・人のアルファ・影の地図）、test_diffkey.py・scan_diffkey.py・"
        "test_diffkey_final.py（縁の作り方・しきい値・影・プレートの比較）、exp_shadow.py（影の判定）、vtest の実写"
        "（目で確認: vfx-review/74-diff-key.jpg）、画像 1 枚＋--plate",
    },
    75: {
        "status": "◎",
        "how": "固定カメラの映像の範囲から等間隔に --samples コマ（既定 61）を 1 回の読み込みで取り、画素ごとの中央値を"
        "求める（横長の帯に分けて計算し、メモリを抑える）。--exclude-people: 共有の detectors で人の枠を求め（既定 fast ="
        " MediaPipe の EfficientDet-Lite2〔Apache-2.0〕、accurate = torchvision の Faster R-CNN v2〔BSD-3-Clause〕）、"
        "枠を左右 8%・上 5%・下 12%（足元の影）広げた所を、その画素の中央値から除く（全部のコマで除かれた画素は除かずに"
        "取る）。--exclude-mask で人以外の物も除ける。出力: PNG、確認画像（取ったコマ・プレート・動きの多さの地図。"
        "半分以上のコマでプレートと 25 以上違う画素を赤）、JSON（取った時刻・確かめにくい所の割合）。--video: 74 と"
        "同じ色の差（しきい値 12・影も含める）で動く物を求め、短い辺の 1/120 広げてぼかした所を、明るさを合わせた"
        "プレートで埋めた MP4（人を消した映像）",
        "measured": "広場の合成（正解の背景つき。1 人が 75% のコマで同じ所に立ち止まる。768×576・100 コマ）: 単純な中央値は"
        "正解の背景との差 平均 4.48・10 を超える画素 2.17%・立ち止まった人の所 114.1（人が残る）。--exclude-people"
        "（fast）は 2.19・0.00%・4.4（人が消えた）、accurate も 2.20・0.00%・4.4。正解の背景を書き出し"
        "（core.VideoWriter の MP4。2026-11-15 に色の変換を正確にする前の版）に通しただけで差が 2.20 あるので、書き出しを"
        "通した正解と比べると 1.24（単純な中央値は"
        " 3.54）。vtest（実写の固定カメラ・79 秒）: 元のコマに Faster R-CNN で 6・7・4・6 人（5・20・40・60 秒）→ プレート"
        "は 0 人、確かめにくい所 0.04%。--video（広場の合成）: 人のいた所の、書き出しを通した正解の背景との差 81.1 → 3.14"
        "（人のいない所は 3.09）。速さ: vtest で 3.1 秒、広場の合成で単純な中央値 1.8 秒・fast 11.3 秒（モデルの読み込みを"
        "含む）・accurate 697 秒（他の作業と CPU を分け合った状態で 1 コマ 10 秒以上）",
        "limits": "固定カメラ専用（揺れる映像は先に手ぶれ補正で止める）。範囲の半分以上の時間そこに居続けた物は単純な"
        "中央値に残る（人なら --exclude-people、人以外は --exclude-mask）。--exclude-people の枠の中は使えるコマが減るので"
        "少しノイズが多い（差の地図で薄い四角に見える。正解との差は 10 未満）。--video は差の大きい所を全部埋めるので、"
        "影も埋まる。照明が変わる映像・動く背景（木・水面）は試していない（中央値は平均的な明るさ・ぼけた模様になるはず）",
        "tested": "make_plaza.py（正解の背景つき）、test_cleanplate.py（単純な中央値・--exclude-people の fast と accurate・"
        "vtest の人数・--video）、review_plate_diff.py（目で確認: vfx-review/75-clean-plate.jpg）、画像を渡した時の"
        "エラー",
    },
}

_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv"}

# ── 共通の部品 ────────────────────────────────────────────────────────────


def _rng(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": a.start or 0.0,
        "duration": a.duration,
        "max_frames": a.max_frames,
    }


def _total(info: core.MediaInfo, a: argparse.Namespace) -> int:
    """処理するコマ数の見込み（進み具合の表示と、確認画像に使うコマを選ぶため）。"""
    if info.is_image:
        return 1
    span = a.duration or max(info.duration - (a.start or 0.0), 0.0)
    n = max(1, round(span * info.fps))
    return min(n, a.max_frames) if a.max_frames else n


def _progress(i: int, total: int, t0: float) -> None:
    if total > 1 and ((i + 1) % max(1, total // 10) == 0 or i + 1 == total):
        core.log(f"  {i + 1}/{total} コマ（{time.time() - t0:.0f} 秒）")


def _sibling(path: Path, suffix: str) -> Path:
    """出力の隣に置く別の出力の名前（例: x.webm → x-check.jpg）。"""
    return path.with_name(path.stem + suffix)


def _out_path(
    a: argparse.Namespace, info: core.MediaInfo, tag: str, *, alpha: bool
) -> Path:
    """出力先。alpha=True なら透過を残せる形（動画は .webm、画像は .png）。"""
    if a.out:
        out = Path(a.out)
        ext = out.suffix.lower()
        if info.is_image:
            core.require(
                ext in core.IMAGE_EXTS,
                f"画像の入力には画像の出力先（.png など）を指定してください: {out}",
            )
            core.require(
                not alpha or ext in {".png", ".webp", ".tif", ".tiff"},
                f"透過を残すには .png を指定してください（{ext} は透過を持てません）: {out}",
            )
        else:
            core.require(
                ext in _VIDEO_EXTS,
                f"動画の入力には動画の出力先（.webm・.mp4）を指定してください: {out}",
            )
            core.require(
                not alpha or ext == ".webm",
                f"透過を残すには .webm を指定してください（{ext} は透過を持てません。"
                f"背景に重ねるなら --bg）: {out}",
            )
        return out
    if info.is_image:
        return core.default_out(a.src, tag, ".png")
    return core.default_out(a.src, tag, ".webm" if alpha else ".mp4")


def _json_path(a: argparse.Namespace, out: Path) -> Path:
    return Path(a.json) if getattr(a, "json", None) else out.with_suffix(".json")


def _cover(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """img を size（幅, 高さ）いっぱいに拡大・縮小し、はみ出しを中央で切る（縦横比を保つ）。"""
    w, h = size
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    s = max(w / iw, h / ih)
    nw, nh = max(w, round(iw * s)), max(h, round(ih * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[y0 : y0 + h, x0 : x0 + w]


class _Layer:
    """2 つ目の入力（背景・マスク・プレート）を、処理するコマの大きさで順に返す。

    画像なら同じ物を返し続ける。動画なら 1 コマずつ進み、mode="color" は最後まで行ったら頭から繰り返し、
    mode="mask" は最後のコマを使い続ける。mode="mask" は 0〜1 の地図（透過つきならアルファ、無ければ明るさ）。
    """

    def __init__(
        self, path: str, size: tuple[int, int], mode: str, start: float = 0.0
    ) -> None:
        self.path = path
        self.size = size
        self.mode = mode
        self.start = start
        self.info = core.probe(path)
        self.alpha = mode == "mask" and self.info.has_alpha
        self.const: np.ndarray | None = None
        self.last: np.ndarray | None = None
        self.it: Iterator[np.ndarray] | None = None
        if self.info.is_image:
            img = core.load_image(path, alpha=mode == "mask")
            self.const = self._conv(img)

    def _conv(self, img: np.ndarray) -> np.ndarray:
        img = _cover(img, self.size)
        if self.mode != "mask":
            return img[:, :, :3]
        if img.ndim == 3 and img.shape[2] == 4 and (img[:, :, 3] < 255).any():
            return img[:, :, 3].astype(np.float32) / 255.0
        m = core.to_gray(img[:, :, :3] if img.ndim == 3 else img).astype(np.float32)
        # MP4 のマスクは、書き出しの変換で白が 250 前後・黒が 1〜2 になることがあるので、両端を切り詰める
        # （core の書き出しは 2026-11-15 から 255・0 のまま残るが、前の版や他の道具で作ったマスクのために残す）
        return np.clip((m - 2.0) / 246.0, 0.0, 1.0)

    def _open(self) -> Iterator[np.ndarray]:
        return core.read_frames(self.path, start=self.start, alpha=self.alpha)

    def next(self) -> np.ndarray:
        if self.const is not None:
            return self.const
        if self.it is None:
            self.it = self._open()
        try:
            frame = next(self.it)
        except (StopIteration, core.VfxError):
            if self.mode == "color":
                self.it = core.read_frames(self.path)
                frame = next(self.it)
            elif self.last is not None:
                return self.last
            else:
                raise core.VfxError(f"コマを読めません: {self.path}") from None
        self.last = self._conv(frame)
        return self.last


def _smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """ガウスぼかし。大きなぼかしは縮めた画で行う（速さのため）。"""
    if sigma <= 0:
        return img
    if sigma < 8:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    f = 4.0 / sigma
    small = cv2.resize(
        img,
        (max(4, round(w * f)), max(4, round(h * f))),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(small, (0, 0), 4.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _fill(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """weight が 0 の所を、周りの weight > 0 の所の色で滑らかに埋める（push-pull。Gortler ら 1996）。

    img は H×W×C の float32、weight は 0〜1。weight = 1 の所は元の値のまま。
    """
    w = weight.astype(np.float32)
    x = img.astype(np.float32) * (w[..., None] if img.ndim == 3 else w)
    levels: list[tuple[np.ndarray, np.ndarray]] = []
    while min(w.shape[:2]) > 6:
        levels.append((x, w))
        x, w = cv2.pyrDown(x), cv2.pyrDown(w)
    total = float(w.sum())
    if total <= 1e-6:
        return img.astype(np.float32)
    wx = w[..., None] if x.ndim == 3 else w
    mean = x.reshape(-1, *x.shape[2:]).sum(0) / total
    base = np.where(wx > 1e-4, x / np.maximum(wx, 1e-6), mean).astype(np.float32)
    for xl, wl in reversed(levels):
        up = cv2.pyrUp(base, dstsize=(xl.shape[1], xl.shape[0]))
        wlx = wl[..., None] if xl.ndim == 3 else wl
        conf = np.clip(wlx * 4.0, 0.0, 1.0)
        val = xl / np.maximum(wlx, 1e-6)
        base = (conf * val + (1.0 - conf) * up).astype(np.float32)
    return base


def _checker(h: int, w: int, s: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // s) + (xx // s)) % 2).astype(np.uint8)
    out = np.where(c[..., None] == 1, 205, 150).astype(np.uint8)
    return np.repeat(out, 3, axis=2) if out.shape[2] == 1 else out


def _over(fg: np.ndarray, alpha: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """fg（BGR）をアルファ alpha（0〜1）で bg に重ねる。"""
    a = alpha.astype(np.float32)[..., None]
    out = fg[:, :, :3].astype(np.float32) * a + bg[:, :, :3].astype(np.float32) * (
        1 - a
    )
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _alpha_img(alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)


def _to_u8(img: np.ndarray) -> np.ndarray:
    """0〜1 の float の画を 0〜255 の uint8 にする。"""
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _extend_colors(fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """透明な所の色を、近くの不透明な所の色で埋める（縁の色のにじみ止め。fg は 0〜1 の float）。

    透過つき動画は透明な所にも色が入っていて、拡大・縮小で縁に混ざる。黒や緑のふちが出ないよう、
    アルファ 0.05 未満の所は周りの前景の色にしておく。
    """
    solid = (alpha > 0.5).astype(np.float32)
    if solid.sum() < 1:
        return fg
    ext = _fill(fg, solid)
    keep = (alpha >= 0.05)[..., None]
    return np.where(keep, fg, ext)


class _GrayWriter:
    """白黒の映像（マスク・アルファ）を MP4 に書く。灰色のまま ffmpeg に渡すので、明るさの値が変わらない。

    core.VideoWriter（BGR から変換）は、既定の変換の丸めで白 255 が 250〜253、128 が 123〜126 に下がる
    （2026-11-14 実測）。マスクの白が 250 だと、それを掛けた人物が 2% 透ける。
    """

    def __init__(self, path: str | Path, fps: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.proc: Any = None
        self.size: tuple[int, int] | None = None

    def write(self, m: np.ndarray) -> None:
        g = m if m.dtype == np.uint8 else _to_u8(m)
        if g.ndim == 3:
            g = core.to_gray(g)
        h, w = g.shape
        if self.proc is None:
            core.require(
                w % 2 == 0 and h % 2 == 0,
                f"動画の大きさは偶数にしてください（{w}×{h}）",
            )
            cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error"]
            cmd += ["-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}"]
            cmd += ["-r", f"{self.fps:.6f}", "-i", "-", *core.MP4_ARGS, str(self.path)]
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.size = (w, h)
        self.proc.stdin.write(np.ascontiguousarray(g).tobytes())

    def close(self) -> Path:
        core.require(self.proc is not None, f"書き出すコマがありません: {self.path}")
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise core.VfxError(f"書き出しに失敗しました: {self.path}\n{err[-800:]}")
        return self.path


def _frame_stats(alpha: np.ndarray) -> dict[str, float]:
    return {
        "transparent": round(float((alpha < 0.02).mean()), 5),
        "opaque": round(float((alpha > 0.98).mean()), 5),
        "semi": round(float(((alpha >= 0.02) & (alpha <= 0.98)).mean()), 5),
    }


class _KeyOutput:
    """キーの結果（アルファと前景の色）の書き出し。

    - 既定: 透過つき（動画は WebM・画像は PNG）。--bg があれば、その背景に重ねた MP4・画像。
    - --alpha-out: アルファだけの白黒（MP4 / PNG）。--mask: 白い所だけを残す（ガベージマット）。
    - 確認画像（<出力>-check.jpg）: 最初・中ほど・最後のコマの「元・アルファ・合成」。JSON: コマごとの透明・半透明の割合。
    """

    def __init__(
        self,
        a: argparse.Namespace,
        info: core.MediaInfo,
        tag: str,
        size: tuple[int, int],
    ) -> None:
        self.a = a
        self.info = info
        self.size = size
        self.composite = bool(getattr(a, "bg", None))
        self.out = _out_path(a, info, tag, alpha=not self.composite)
        self.json = _json_path(a, self.out)
        self.check = _sibling(self.out, "-check.jpg")
        self.bg = _Layer(a.bg, size, "color") if self.composite else None
        self.mask = (
            _Layer(a.mask, size, "mask", start=a.start or 0.0)
            if getattr(a, "mask", None)
            else None
        )
        alpha_out = getattr(a, "alpha_out", None)
        self.alpha_out = Path(alpha_out) if alpha_out else None
        if self.alpha_out is not None:
            ok = core.IMAGE_EXTS if info.is_image else {".mp4", ".webm", ".mov", ".mkv"}
            core.require(
                self.alpha_out.suffix.lower() in ok,
                f"--alpha-out の拡張子が入力と合いません: {self.alpha_out}",
            )
        self.writer: core.VideoWriter | None = None
        self.awriter: _GrayWriter | None = None
        if not info.is_image:
            self.writer = core.VideoWriter(
                self.out,
                info.fps,
                alpha=not self.composite,
                audio_from=a.src,
                audio_start=a.start or 0.0,
                audio_duration=a.duration,
            )
            if self.alpha_out is not None:
                self.awriter = _GrayWriter(self.alpha_out, info.fps)
        self.total = _total(info, a)
        self.pick = sorted({0, self.total // 2, max(0, self.total - 1)})
        self.samples: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        self.last: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.stats: list[dict[str, float]] = []
        self.t0 = time.time()

    def write(self, i: int, src: np.ndarray, alpha: np.ndarray, fg: np.ndarray) -> None:
        """alpha は 0〜1 の float、fg は前景の色（BGR。0〜1 の float か uint8）。"""
        if self.mask is not None:
            alpha = alpha * self.mask.next()
        if fg.dtype != np.uint8:
            fg = _to_u8(_extend_colors(fg, alpha))
        if self.composite:
            assert self.bg is not None
            result = _over(fg, alpha, self.bg.next())
        else:
            result = np.dstack([fg[:, :, :3], _to_u8(alpha)])
        if self.writer is not None:
            self.writer.write(result)
        else:
            core.save_image(self.out, result)
        if self.alpha_out is not None:
            if self.awriter is not None:
                self.awriter.write(alpha)
            else:
                core.save_image(self.alpha_out, _to_u8(alpha))
        st = _frame_stats(alpha)
        st["t"] = core.t_of(i, self.info.fps, self.a.start or 0.0)
        self.stats.append(st)
        view = result if self.composite else self._on_checker(fg, alpha)
        rec = (i, src[:, :, :3].copy(), _alpha_img(alpha), view)
        if i in self.pick:
            self.samples.append(rec)
        self.last = rec
        _progress(i, self.total, self.t0)

    @staticmethod
    def _on_checker(fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        h, w = alpha.shape
        return _over(fg, alpha, _checker(h, w, max(8, min(h, w) // 40)))

    def close(
        self, params: dict[str, Any], extra: dict[str, Any] | None = None
    ) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.awriter is not None:
            self.awriter.close()
        samples = list(self.samples)
        if self.last is not None and all(s[0] != self.last[0] for s in samples):
            samples.append(self.last)
        samples = sorted(samples, key=lambda s: s[0])[-3:]
        images, labels = [], []
        for i, src, al, view in samples:
            t = core.t_of(i, self.info.fps, self.a.start or 0.0)
            when = "" if self.info.is_image else f" {t:.2f}秒"
            images += [src, al, view]
            labels += [
                f"元{when}",
                "アルファ（白 = 残す）",
                "合成" if self.composite else "透過（市松模様に重ねた）",
            ]
        if images:
            core.save_image(
                self.check,
                core.contact_sheet(images, labels, cols=3, cell_width=320),
            )
        n = len(self.stats)
        summary = {
            k: round(float(np.mean([s[k] for s in self.stats])), 4) if n else 0.0
            for k in ("transparent", "opaque", "semi")
        }
        data = {
            "source": str(self.a.src),
            "output": str(self.out),
            "frames": n,
            "size": list(self.size),
            "params": params,
            "mean": summary,
            **(extra or {}),
            "per_frame": self.stats if not self.info.is_image else [],
        }
        core.write_json(self.json, data)
        core.log(
            f"{n} コマ: 透明 {summary['transparent'] * 100:.1f}%・半透明 "
            f"{summary['semi'] * 100:.1f}%・不透明 {summary['opaque'] * 100:.1f}%（平均）"
        )
        core.log(f"保存: {self.out}")
        if self.alpha_out is not None:
            core.log(f"保存: {self.alpha_out}")
        core.log(f"保存: {self.check}")
        core.log(f"保存: {self.json}")


def _ffmpeg_frames(
    path: str,
    vf: str,
    size: tuple[int, int],
    *,
    start: float = 0.0,
    duration: float | None = None,
    max_frames: int | None = None,
) -> Iterator[np.ndarray]:
    """ffmpeg のフィルター vf を通したコマを BGRA で順に返す（ffmpeg のキーのフィルターを使う時）。"""
    info = core.probe(path)
    w, h = size
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
    if start and not info.is_image:
        cmd += ["-ss", f"{start:.4f}"]
    if info.has_alpha and info.codec == "vp9":
        cmd += ["-c:v", "libvpx-vp9"]
    cmd += ["-i", str(path)]
    if duration and not info.is_image:
        cmd += ["-t", f"{duration:.4f}"]
    filters = []
    if (w, h) != (info.width, info.height):
        filters.append(f"scale={w}:{h}:flags=area")
    filters += [vf, "format=bgra"]
    cmd += ["-vf", ",".join(filters), "-an", "-f", "rawvideo", "-pix_fmt", "bgra", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    n = 0
    try:
        while max_frames is None or n < max_frames:
            buf = proc.stdout.read(w * h * 4)
            if len(buf) < w * h * 4:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 4).copy()
            n += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if n == 0:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise core.VfxError(f"ffmpeg のフィルターが失敗しました: {vf}\n{err[-800:]}")


# ── スピル（色かぶり）──────────────────────────────────────────────────────

# 消した色の明るさを足し戻す時の重み（Rec.709 の輝度の係数）
_LUMA_W = {"green": 0.7152, "blue": 0.0722}


def _despill(
    f: np.ndarray,
    kind: str,
    method: str = "average",
    strength: float = 1.0,
    keep_luma: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """緑（青）かぶりを消す。f は BGR の 0〜1 の float。(結果, 消した量の地図) を返す。

    緑の上限を average = (R+B)/2（ffmpeg の despill の既定と同じ式）/ max = max(R, B)（弱い・黄色を守る）/
    min = min(R, B)（強い）にする。keep_luma は消した分の明るさを 3 色に足し戻す（暗くならない）。
    """
    ch = 1 if kind == "green" else 0
    o1, o2 = (f[..., 2], f[..., 0]) if kind == "green" else (f[..., 2], f[..., 1])
    c = f[..., ch]
    if method == "average":
        lim = (o1 + o2) * 0.5
    elif method == "max":
        lim = np.maximum(o1, o2)
    elif method == "min":
        lim = np.minimum(o1, o2)
    else:
        raise core.VfxError(f"知らない方法です: {method}（average / max / min）")
    spill = np.maximum(c - lim, 0.0) * float(strength)
    out = f.copy()
    out[..., ch] = c - spill
    if keep_luma:
        out += (spill * _LUMA_W[kind])[..., None]
    return np.clip(out, 0.0, 1.0), spill


# ── クロマキー ────────────────────────────────────────────────────────────


def _dominance(f: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """背景の色の強さ（緑なら G − max(R, B)）と、その色の値（緑なら G）。f は BGR の 0〜1。"""
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    if kind == "green":
        return g - np.maximum(r, b), g
    return b - np.maximum(r, g), b


def _parse_color(text: str) -> np.ndarray:
    """'#RRGGBB' を BGR の 0〜1 にする。"""
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", text.strip())
    core.require(
        m is not None,
        f"色は auto / green / blue / #RRGGBB のどれかで指定してください: {text}",
    )
    assert m is not None
    v = m.group(1)
    r, g, b = (int(v[k : k + 2], 16) / 255.0 for k in (0, 2, 4))
    return np.array([b, g, r], np.float32)


def _detect_screen(frame: np.ndarray, color: str) -> tuple[str, np.ndarray, float]:
    """抜く背景の種類（green / blue）と代表の色（BGR の 0〜1）、画面の端で背景の色だった割合を求める。"""
    f = frame[:, :, :3].astype(np.float32) / 255.0
    if color not in ("auto", "green", "blue"):
        ref = _parse_color(color)
        kind = "green" if ref[1] >= ref[0] else "blue"
        dom, _ = _dominance(ref[None, None], kind)
        core.require(
            float(dom[0, 0]) > 0.05,
            f"指定の色 {color} は緑・青のどちらでもありません（緑か青の背景だけを抜けます）",
        )
        return kind, ref, 1.0
    h, w = f.shape[:2]
    m = max(4, round(min(h, w) * 0.08))
    ring = np.zeros((h, w), bool)
    ring[:m] = ring[-m:] = True
    ring[:, :m] = ring[:, -m:] = True
    best: tuple[str, float, np.ndarray] | None = None
    for kind in ("green", "blue") if color == "auto" else (color,):
        dom, prim = _dominance(f, kind)
        sel = (dom > 0.08) & (dom > 0.25 * prim)
        frac = float(sel[ring].mean())
        if best is None or frac > best[1]:
            best = (kind, frac, sel)
    assert best is not None
    kind, frac, sel = best
    if frac < 0.05:
        frac_all = float(sel.mean())
        core.require(
            frac_all >= 0.05,
            "緑・青の背景が見つかりません（画面の端の 5% 未満）。--color で色を指定してください",
        )
        pick = sel
    else:
        pick = sel & ring
    px = f[pick]
    dom_px, _ = _dominance(px[None], kind)
    strong = px[dom_px[0] >= np.median(dom_px)]
    return kind, np.median(strong, axis=0).astype(np.float32), frac


class _ChromaKeyer:
    """色差キー（Vlahos の差分マット）を、場所ごとの背景の色で割る（Nuke の IBK と同じ考え方）。

    1. 背景らしい画素（緑の強さ G−max(R,B) と、その G に対する比が代表の色に近い）を種にし、縁から少し削る。
    2. 種の色で、人物の後ろの背景の色を周りから滑らかに埋める（push-pull）。大まかなぼかしの背景で一度アルファを
       求め、ほぼ透明な所を種にし直して、細かい背景の色（むら・しわ）を作り直す（2 段）。
    3. アルファ = 1 − (その画素の緑の強さ) ÷ (その場所の背景の緑の強さ)。黒・白の切り詰めで整える。
       境目の帯だけ、前景の色と背景の色を結ぶ線への射影で求め直す（refine）。
    4. 前景の色 = (元の色 − (1−α) × 背景の色) ÷ α（背景の色を差し引く。縁の緑のにじみが消える）。続けてスピル除去。
       半透明の縁は、引きすぎの赤紫を止めた色と内側の色を混ぜる（edge_color）。
    """

    def __init__(
        self,
        kind: str,
        ref: np.ndarray,
        *,
        clip_black: float,
        clip_white: float,
        despill: str,
        short: int,
        refine: bool = True,
    ) -> None:
        self.kind = kind
        self.use_refine = refine
        rdom, rprim = _dominance(ref[None, None], kind)
        self.rdom = float(rdom[0, 0])
        self.rratio = self.rdom / max(float(rprim[0, 0]), 1e-3)
        self.cb = float(clip_black)
        self.cw = float(clip_white)
        core.require(
            0 <= self.cb < self.cw <= 1,
            "--clip-black は --clip-white より小さく、0〜1 にしてください",
        )
        self.despill = despill
        self.erode = max(2, round(short / 240))
        self.sig_small = max(1.0, short / 270)
        self.sig_big = short / 12

    def plate(self, f: np.ndarray) -> np.ndarray:
        """人物の後ろも含めた、場所ごとの背景の色（BGR の 0〜1）。"""
        dom, prim = _dominance(f, self.kind)
        ratio = dom / np.maximum(prim, 1e-3)
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * self.erode + 1, 2 * self.erode + 1)
        )
        seed = (ratio > 0.6 * self.rratio) & (dom > 0.25 * self.rdom)
        seed = cv2.erode(seed.astype(np.uint8), k).astype(np.float32)
        if seed.sum() < 16:
            raise core.VfxError(
                "背景の色の所がほとんどありません（--color を確かめてください）"
            )
        coarse = _smooth(_fill(f, seed), self.sig_big)
        cdom, _ = _dominance(coarse, self.kind)
        a1 = 1.0 - dom / np.maximum(cdom, 0.02)
        seed2 = (a1 < 0.12) & (ratio > 0.4 * self.rratio)
        seed2 = cv2.erode(seed2.astype(np.uint8), k).astype(np.float32)
        if seed2.sum() < 16:
            seed2 = seed
        return _smooth(_fill(f, seed2), self.sig_small)

    def refine(self, f: np.ndarray, plate: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """境目の帯だけ、アルファを「その場所の前景の色と背景の色を結ぶ線」への射影で求め直す。

        色差の式は前景が無彩色（緑の強さ 0）の時だけ正しく、肌やクリーム色の縁では濃く、緑を浴びた黒髪では
        薄く出る。観測 = α×前景 + (1−α)×背景 なので、前景の色（近くの不透明な所の色を埋め広げた物）と背景の色が
        分かれば α = ((観測 − 背景)・(前景 − 背景)) ÷ |前景 − 背景|² で求まる。前景の色は、帯のすぐ内側
        （帯から 3px 以内の不透明な所）の色を埋め広げて使う（内側全体から埋めるより縁の色に近い）。前景と背景の色が
        近い所（差 0.15 未満）は色差の式のまま。
        """
        r = self.erode + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        hard = (alpha > 0.5).astype(np.uint8)
        band = (cv2.dilate(hard, k) > 0) & (cv2.dilate(1 - hard, k) > 0)
        band |= (alpha > 0.02) & (alpha < 0.98)
        if not band.any():
            return alpha
        solid = cv2.erode((alpha > 0.98).astype(np.uint8), np.ones((3, 3), np.uint8))
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        ring = solid & (cv2.dilate(band.astype(np.uint8), k3) > 0)
        if ring.sum() < 16:
            return alpha
        fg_loc = _fill(f, ring.astype(np.float32))
        d = fg_loc - plate
        den = (d * d).sum(-1)
        proj = ((f - plate) * d).sum(-1) / np.maximum(den, 1e-6)
        proj = np.clip((proj - 0.02) / 0.96, 0.0, 1.0)
        use = band & (den > 0.15**2)
        return np.where(use, proj, alpha).astype(np.float32)

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(アルファ 0〜1, 前景の色 BGR 0〜1) を返す。"""
        f = frame[:, :, :3].astype(np.float32) / 255.0
        plate = self.plate(f)
        dom, _ = _dominance(f, self.kind)
        pdom, _ = _dominance(plate, self.kind)
        alpha = 1.0 - dom / np.maximum(pdom, 0.02)
        alpha = np.clip((alpha - self.cb) / (self.cw - self.cb), 0.0, 1.0)
        if self.use_refine:
            alpha = self.refine(f, plate, alpha)
        a3 = alpha[..., None]
        fg = (f - (1.0 - a3) * plate) / np.maximum(a3, 1e-3)
        fg = np.clip(np.where(a3 > 0.004, fg, f), 0.0, 1.0)
        if self.despill != "none":
            fg, _ = _despill(fg, self.kind, self.despill)
        return alpha.astype(np.float32), self.edge_color(fg, alpha)

    def edge_color(self, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """半透明の縁の色を整える。アルファが少し薄く出た所は背景の色を引きすぎて、緑なら赤紫（青なら黄）の
        ふちになるので、(1) 背景の色の成分を、ほかの 2 色の小さい方より下げない色と、(2) すぐ内側の不透明な所の
        色を半分ずつ混ぜる（合成の試験で、縁の色の差が最も小さかった組み合わせ）。"""
        semi = (alpha > 0.004) & (alpha < 0.98)
        if not semi.any():
            return fg.astype(np.float32)
        ch = 1 if self.kind == "green" else 0
        others = [c for c in (0, 1, 2) if c != ch]
        clamp = fg.copy()
        low = np.minimum(fg[..., others[0]], fg[..., others[1]])
        clamp[..., ch] = np.maximum(fg[..., ch], low)
        solid = cv2.erode((alpha > 0.98).astype(np.uint8), np.ones((3, 3), np.uint8))
        inner = _fill(fg, solid.astype(np.float32)) if solid.sum() >= 16 else clamp
        mixed = 0.5 * clamp + 0.5 * inner
        return np.where(semi[..., None], mixed, fg).astype(np.float32)


def _yuv601(bgr: np.ndarray) -> tuple[float, float, float]:
    """BGR（0〜1）→ ffmpeg のフィルターが見る BT.601（テレビの範囲）の Y・U・V（0〜255 の値）。"""
    b, g, r = (float(v) for v in bgr)
    y = 16 + 65.481 * r + 128.553 * g + 24.966 * b
    u = 128 - 37.797 * r - 74.203 * g + 112.0 * b
    v = 128 + 112.0 * r - 93.786 * g - 18.214 * b
    return y, u, v


def _ffmpeg_key_filter(method: str, ref: np.ndarray, sim: float, blend: float) -> str:
    """ffmpeg の chromakey / hsvkey の指定を、代表の色から作る。"""
    if method == "chromakey":
        rgb = "".join(f"{round(float(v) * 255):02X}" for v in ref[::-1])
        return f"chromakey=color=0x{rgb}:similarity={sim}:blend={blend}"
    # hsvkey は U・V の角度を色相、U・V の長さを彩度、Y を明るさとして、円錐の中の距離で比べる（vf_hsvkey.c）。
    # 明るさ（val）は負の値で渡して無視させる（背景のむらで明るさが変わっても抜けるように。試験で IoU 0.889 → 0.993）
    y, u, v = _yuv601(ref)
    uf, vf = u - 127.5, v - 127.5
    hue_rad = float(np.arctan2(uf, vf) + np.pi)
    hue_opt = (526.0 - np.degrees(hue_rad)) % 360.0
    sat = float(np.sqrt((uf * uf + vf * vf) / (127.5 * 127.5 * 2)))
    val = y / 255.0
    return (
        f"hsvkey=hue={hue_opt:.2f}:sat={sat:.4f}:val={-val:.4f}"
        f":similarity={sim}:blend={blend}"
    )


@command(
    52,
    "chroma-key",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--color",
        default="auto",
        help="抜く背景の色: auto（画面の端から緑か青かを決める。既定）/ green / blue / #RRGGBB",
    ),
    arg(
        "--method",
        choices=["diff", "chromakey", "hsvkey"],
        default="diff",
        help="diff: 色差キー（場所ごとの背景の色で割る。むらに強く、縁の色も直す。既定）/ "
        "chromakey: ffmpeg の chromakey（色の近さ）/ hsvkey: ffmpeg の hsvkey（色相・彩度・明るさの近さ）",
    ),
    arg(
        "--clip-black",
        type=float,
        default=0.08,
        help="これより薄いアルファを 0 にする（背景のむら・ノイズを消す。0〜1。既定 0.08。diff のみ）",
    ),
    arg(
        "--clip-white",
        type=float,
        default=0.9,
        help="これより濃いアルファを 1 にする（人物の中の透けを埋める。0〜1。既定 0.9。diff のみ）",
    ),
    arg(
        "--despill",
        choices=["average", "max", "min", "none"],
        default="average",
        help="前景の緑（青）かぶりの消し方（57 と同じ。既定 average）",
    ),
    arg(
        "--similarity",
        type=float,
        help="chromakey・hsvkey: 背景とみなす色の近さ（既定 chromakey 0.05・hsvkey 0.03。"
        "大きすぎると人物まで消える）",
    ),
    arg(
        "--blend",
        type=float,
        help="chromakey・hsvkey: 半透明にする幅（既定 chromakey 0.03・hsvkey 0.02）",
    ),
    arg("--bg", help="差し替える背景（画像か動画）。指定すると重ねた MP4・画像を書く"),
    arg(
        "--mask",
        help="ガベージマット（白 = 残してよい所。黒は必ず透明にする。画像・動画。62 の出力など）",
    ),
    arg("--alpha-out", help="アルファだけの白黒の映像（画像）も書き出す場所"),
)
def chroma_key(a: argparse.Namespace) -> None:
    """緑・青の背景を抜いて、透過つき WebM（画像は PNG）か、--bg の背景に重ねた映像にする。"""
    info = core.probe(a.src)
    size = core.scaled_size(info.width, info.height, max_side=a.max_side)
    first = next(core.read_frames(a.src, start=a.start or 0.0, max_side=a.max_side))
    kind, ref, frac = _detect_screen(first, a.color)
    ref_hex = "#" + "".join(f"{round(float(v) * 255):02X}" for v in ref[::-1])
    core.log(
        f"背景: {'緑' if kind == 'green' else '青'}（代表の色 {ref_hex}。画面の端の {frac * 100:.0f}%）"
    )
    out = _KeyOutput(a, info, "chroma-key", size)
    params: dict[str, Any] = {
        "method": a.method,
        "screen": kind,
        "screen_color": ref_hex,
        "despill": a.despill,
    }
    frames = core.read_frames(a.src, **_rng(a), max_side=a.max_side)
    if a.method == "diff":
        keyer = _ChromaKeyer(
            kind,
            ref,
            clip_black=a.clip_black,
            clip_white=a.clip_white,
            despill=a.despill,
            short=min(size),
        )
        params.update(clip_black=a.clip_black, clip_white=a.clip_white)
        for i, frame in enumerate(frames):
            alpha, fg = keyer(frame)
            out.write(i, frame, alpha, fg)
    else:
        chroma = a.method == "chromakey"
        sim = a.similarity if a.similarity is not None else (0.05 if chroma else 0.03)
        blend = a.blend if a.blend is not None else (0.03 if chroma else 0.02)
        vf = _ffmpeg_key_filter(a.method, ref, sim, blend)
        params.update(similarity=sim, blend=blend, ffmpeg_filter=vf)
        keyed = _ffmpeg_frames(a.src, vf, size, **_rng(a))
        for i, (frame, k) in enumerate(zip(frames, keyed, strict=False)):
            alpha = k[:, :, 3].astype(np.float32) / 255.0
            fg = k[:, :, :3].astype(np.float32) / 255.0
            if a.despill != "none":
                fg, _ = _despill(fg, kind, a.despill)
            out.write(i, frame, alpha, fg)
    out.close(params)


# ── スピル除去 ────────────────────────────────────────────────────────────


@command(
    57,
    "despill",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--color",
        choices=["auto", "green", "blue"],
        default="auto",
        help="消すかぶりの色（auto: 画面全体で緑と青の多い方。既定）",
    ),
    arg(
        "--method",
        choices=["average", "max", "min"],
        default="average",
        help="緑の上限の決め方: average = (赤+青)/2（標準。ffmpeg の despill の既定と同じ）/ "
        "max = 赤と青の大きい方（弱い。黄色・水色を守る）/ min = 小さい方（強い）",
    ),
    arg(
        "--strength",
        type=float,
        default=1.0,
        help="効き具合（0〜1。既定 1）",
    ),
    arg(
        "--keep-luma",
        action="store_true",
        help="消した緑の分の明るさを足し戻す（かぶりの所が暗くならない）",
    ),
    arg("--mask", help="ここだけに掛ける（白 = 掛ける。画像・動画）"),
)
def despill(a: argparse.Namespace) -> None:
    """緑（青）かぶりを消す。透過つきの入力は透過を保ったまま、普通の映像はそのまま色だけ直す。"""
    info = core.probe(a.src)
    size = core.scaled_size(info.width, info.height, max_side=a.max_side)
    alpha_in = info.has_alpha or (
        info.is_image and core.load_image(a.src, alpha=True)[:, :, 3].min() < 255
    )
    if a.out:
        out = Path(a.out)
        core.require(
            (out.suffix.lower() in core.IMAGE_EXTS) == info.is_image,
            f"出力の拡張子が入力（{'画像' if info.is_image else '動画'}）と合いません: {out}",
        )
        if alpha_in and not info.is_image:
            core.require(
                out.suffix.lower() == ".webm",
                f"透過つきの入力は .webm に書き出してください: {out}",
            )
    elif info.is_image:
        ext = ".png" if alpha_in else Path(a.src).suffix.lower()
        out = core.default_out(a.src, "despill", ext)
    else:
        out = core.default_out(a.src, "despill", ".webm" if alpha_in else ".mp4")
    json_path = _json_path(a, out)
    frames = core.read_frames(a.src, **_rng(a), max_side=a.max_side, alpha=alpha_in)
    first = next(core.read_frames(a.src, start=a.start or 0.0, max_side=a.max_side))
    kind = a.color
    if kind == "auto":
        f = first.astype(np.float32) / 255.0
        g = float(np.maximum(_dominance(f, "green")[0], 0).mean())
        b = float(np.maximum(_dominance(f, "blue")[0], 0).mean())
        kind = "green" if g >= b else "blue"
    core.log(f"消すかぶり: {'緑' if kind == 'green' else '青'}（{a.method}）")
    mask = _Layer(a.mask, size, "mask", start=a.start or 0.0) if a.mask else None
    total = _total(info, a)
    writer = (
        None
        if info.is_image
        else core.VideoWriter(
            out,
            info.fps,
            alpha=alpha_in,
            audio_from=a.src,
            audio_start=a.start or 0.0,
            audio_duration=a.duration,
        )
    )
    t0 = time.time()
    stats, picks = [], []
    pick = {0, total // 2, max(0, total - 1)}
    for i, frame in enumerate(frames):
        f = frame[:, :, :3].astype(np.float32) / 255.0
        res, spill = _despill(f, kind, a.method, a.strength, a.keep_luma)
        if mask is not None:
            m = mask.next()[..., None]
            res = res * m + f * (1 - m)
            spill = spill * m[..., 0]
        res8 = _to_u8(res)
        result = np.dstack([res8, frame[:, :, 3]]) if alpha_in else res8
        if writer is not None:
            writer.write(result)
        else:
            core.save_image(out, result)
        vis = frame[:, :, 3] > 127 if alpha_in else np.ones(spill.shape, bool)
        nvis = max(1, int(vis.sum()))
        stats.append(
            {
                "t": core.t_of(i, info.fps, a.start or 0.0),
                # 見えている画素のうち、4/255 以上直した画素の割合と、平均で消した量（0〜255）
                "changed": round(float(((spill > 4 / 255) & vis).sum() / nvis), 5),
                "mean_removed": round(
                    float(spill[vis].mean() * 255) if vis.any() else 0.0, 3
                ),
            }
        )
        if i in pick:
            src, res_v = frame[:, :, :3].copy(), res8
            if alpha_in:  # 透明な所に隠れた色は見せず、市松模様に重ねる
                al = frame[:, :, 3].astype(np.float32) / 255.0
                h, w = al.shape
                board = _checker(h, w, max(8, min(h, w) // 40))
                src, res_v = _over(src, al, board), _over(res8, al, board)
            picks.append((i, src, res_v, spill * vis))
        _progress(i, total, t0)
    if writer is not None:
        writer.close()
    images, labels = [], []
    for i, src, res8, spill in picks[:3]:
        t = "" if info.is_image else f" {core.t_of(i, info.fps, a.start or 0.0):.2f}秒"
        images += [src, res8, core.heatmap(spill, lo=0.0, hi=0.25)]
        labels += [f"元{t}", "スピル除去", "消した量（赤いほど多い。0〜0.25）"]
    check = _sibling(out, "-check.jpg")
    core.save_image(check, core.contact_sheet(images, labels, cols=3, cell_width=320))
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "output": str(out),
            "params": {
                "color": kind,
                "method": a.method,
                "strength": a.strength,
                "keep_luma": a.keep_luma,
            },
            "frames": len(stats),
            "per_frame": stats,
        },
    )
    if stats:
        core.log(
            f"{len(stats)} コマ: かぶりを直した画素 {np.mean([s['changed'] for s in stats]) * 100:.1f}%"
            f"（見えている画素のうち）・消した量の平均 {np.mean([s['mean_removed'] for s in stats]):.1f}/255"
        )
    core.log(f"保存: {out}")
    core.log(f"保存: {check}")
    core.log(f"保存: {json_path}")


# ── 縁の調整（チョーク・スプレッド・フェザー）─────────────────────────────

# 段差をガウスでぼかした時、アルファが 10% から 90% に変わる幅 = 2 × 1.2816σ
_W1090 = 2.5631


def _shift_alpha(alpha: np.ndarray, px: float) -> np.ndarray:
    """アルファの縁を px だけ外へ広げる（負なら内へ縮める）。

    明るさのモルフォロジー（楕円の形の最大・最小）なので、髪などの半透明も形を保って動く。端数は、前後の 2 つの
    大きさの結果を混ぜる（1px より細かくずらせる）。
    """
    if abs(px) < 1e-3:
        return alpha
    op = cv2.dilate if px > 0 else cv2.erode
    r = abs(px)
    r0 = int(np.floor(r))
    frac = r - r0

    def morph(n: int) -> np.ndarray:
        if n == 0:
            return alpha
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
        return op(alpha, k)

    a0 = morph(r0)
    if frac < 1e-3:
        return a0
    return a0 * (1.0 - frac) + morph(r0 + 1) * frac


def _feather(
    alpha: np.ndarray,
    *,
    choke: float = 0.0,
    spread: float = 0.0,
    blur: float = 0.0,
    where: str = "center",
) -> np.ndarray:
    """縁を縮める（choke）・広げる（spread）・ぼかす（blur = アルファが 10% から 90% に変わる幅 px）。

    where: center = 元の縁がぼかしの真ん中（50%）/ inside = ぼかしを元の縁の内側に収める（元の縁で 10%。外の背景が
    はみ出さない）/ outside = 元の縁で 90%（ぼかしが外へ出る）。
    """
    shift = (
        spread
        - choke
        + {"center": 0.0, "inside": -blur / 2, "outside": blur / 2}[where]
    )
    a = _shift_alpha(alpha.astype(np.float32), shift)
    if blur > 0:
        a = cv2.GaussianBlur(a, (0, 0), blur / _W1090, borderType=cv2.BORDER_REPLICATE)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


# ── ガベージマット ─────────────────────────────────────────────────────────

_SHAPE_HELP = (
    "点は元の映像の px（左上が 0,0）か %% で、'x1,y1 x2,y2 x3,y3 ...'。時間で動かす時は "
    "'@0 x1,y1 ...; @2.5 x1,y1 ...'（秒。点の数をそろえる。間は直線で動かし、前後は端の形のまま）"
)


def _num(v: str, full: float) -> float:
    v = v.strip()
    if v.endswith("%"):
        return float(v[:-1]) / 100.0 * full
    return float(v)


def _parse_shape(
    text: str, box: bool, size: tuple[int, int]
) -> list[tuple[float, np.ndarray]]:
    """多角形・四角の指定を (秒, 点の配列〔元の映像の連続座標〕) の並びにする。"""
    w, h = size
    keys: list[tuple[float, np.ndarray]] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        t = 0.0
        m = re.match(r"@\s*([\d.]+)\s*(?:秒|s)?\s+(.*)$", part)
        if m:
            t, part = float(m.group(1)), m.group(2)
        nums = [v for v in re.split(r"[\s,、]+", part.strip()) if v]
        try:
            if box:
                core.require(
                    len(nums) == 4,
                    f"四角は x,y,幅,高さ の 4 つの数で指定してください: {part}",
                )
                x, y = _num(nums[0], w), _num(nums[1], h)
                bw, bh = _num(nums[2], w), _num(nums[3], h)
                pts = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
            else:
                core.require(
                    len(nums) >= 6 and len(nums) % 2 == 0,
                    f"多角形は 3 点以上の x,y で指定してください: {part}",
                )
                pts = [
                    [_num(nums[k], w), _num(nums[k + 1], h)]
                    for k in range(0, len(nums), 2)
                ]
        except ValueError:
            raise core.VfxError(f"数として読めません: {part}") from None
        keys.append((t, np.array(pts, np.float32)))
    core.require(bool(keys), f"範囲の指定が空です: {text!r}")
    keys.sort(key=lambda k: k[0])
    core.require(
        len({len(p) for _, p in keys}) == 1,
        f"時間で動かす時は、どの時刻も同じ数の点にしてください: {text}",
    )
    return keys


def _spec_items(path: str) -> list[tuple[str, str, bool]]:
    """--spec の JSON（{"keep": [...], "drop": [...]}）を (keep/drop, 指定の文字列, 四角か) の並びにする。

    各要素は CLI と同じ書き方の文字列か、{"points": [[x, y], ...]}・{"box": [x, y, w, h]}・
    {"keys": [{"t": 秒, "points": [...]} または {"t": 秒, "box": [...]}, ...]}。
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise core.VfxError(f"--spec を読めません: {path}（{e}）") from None

    def one(item: Any) -> tuple[str, bool]:
        if isinstance(item, str):
            return item, False
        if "keys" in item:
            parts, box = [], False
            for k in item["keys"]:
                text, box = one(k)
                parts.append(f"@{float(k.get('t', 0))} {text}")
            return "; ".join(parts), box
        if "box" in item:
            return ",".join(str(v) for v in item["box"]), True
        if "points" in item:
            return " ".join(f"{p[0]},{p[1]}" for p in item["points"]), False
        raise core.VfxError(f"--spec の要素が読めません: {item}")

    out = []
    for kind in ("keep", "drop"):
        for item in data.get(kind, []):
            text, box = one(item)
            out.append((kind, text, box))
    return out


def _shape_at(keys: list[tuple[float, np.ndarray]], t: float) -> np.ndarray:
    """時刻 t の形（キーの間は直線で動かす）。"""
    if len(keys) == 1 or t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    for (t0, p0), (t1, p1) in itertools.pairwise(keys):
        if t0 <= t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-9)
            return p0 + (p1 - p0) * u
    return keys[-1][1]


def _raster(polys: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """多角形（処理する大きさの連続座標。画素 (0,0) は 0〜1 の正方形）を 4×4 の細かさで塗って縮め、0〜1 にする。"""
    w, h = size
    if not polys:
        return np.zeros((h, w), np.float32)
    s = 4
    big = np.zeros((h * s, w * s), np.uint8)
    for p in polys:
        # 1 つずつ塗る（cv2.fillPoly に複数を渡すと、重なった所が穴になる）
        one = np.zeros_like(big)
        pts = [np.round((p * s - 0.5) * 16).astype(np.int32)]
        cv2.fillPoly(one, pts, 255, lineType=cv2.LINE_8, shift=4)
        # OpenCV は縁が少しでも掛かった画素まで塗るので、縁の画素は半分にする（面積の偏りを消す）
        cv2.polylines(one, pts, True, 128, 1, cv2.LINE_8, shift=4)
        np.maximum(big, one, out=big)
    return (
        cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    )


class _Garbage:
    """残す・外す多角形から、コマごとのマスク（1 = 残す）を作る。動かない時は 1 回だけ作る。"""

    def __init__(
        self,
        shapes: list[tuple[str, list[tuple[float, np.ndarray]]]],
        scale: tuple[float, float],
        size: tuple[int, int],
        *,
        feather: float = 0.0,
        invert: bool = False,
    ) -> None:
        self.shapes = shapes
        self.scale = np.array(scale, np.float32)
        self.size = size
        self.feather = feather
        self.invert = invert
        self.static = all(len(k) == 1 for _, k in shapes)
        self.cache: np.ndarray | None = None

    def polys_at(self, t: float) -> list[tuple[str, np.ndarray]]:
        """時刻 t の多角形（処理する大きさの座標）。"""
        return [(kind, _shape_at(keys, t) * self.scale) for kind, keys in self.shapes]

    def __call__(self, t: float) -> np.ndarray:
        if self.static and self.cache is not None:
            return self.cache
        polys = self.polys_at(t)
        keep = [p for k, p in polys if k == "keep"]
        drop = [p for k, p in polys if k == "drop"]
        w, h = self.size
        m = _raster(keep, self.size) if keep else np.ones((h, w), np.float32)
        if drop:
            m = m * (1.0 - _raster(drop, self.size))
        if self.invert:
            m = 1.0 - m
        if self.feather > 0:
            m = _feather(m, blur=self.feather)
        if self.static:
            self.cache = m
        return m

    def draw(self, img: np.ndarray, t: float) -> np.ndarray:
        """多角形の線を描く（残す = 緑、外す = 赤）。"""
        out = img[:, :, :3].copy()
        th = max(2, round(min(out.shape[:2]) / 270))
        for kind, p in self.polys_at(t):
            color = (60, 220, 60) if kind == "keep" else (40, 40, 240)
            pts = np.round((p - 0.5) * 16).astype(np.int32)
            cv2.polylines(out, [pts], True, color, th, cv2.LINE_AA, shift=4)
        return out


def _alpha_of(frame: np.ndarray, has_alpha: bool) -> tuple[np.ndarray, np.ndarray]:
    """(色 BGR uint8, アルファ 0〜1)。透過の無い入力はアルファ 1。"""
    if has_alpha and frame.ndim == 3 and frame.shape[2] == 4:
        return frame[:, :, :3], frame[:, :, 3].astype(np.float32) / 255.0
    return frame[:, :, :3], np.ones(frame.shape[:2], np.float32)


def _image_has_alpha(path: str) -> bool:
    img = core.load_image(path, alpha=True)
    return bool((img[:, :, 3] < 255).any())


@command(
    62,
    "garbage-matte",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--keep",
        action="append",
        default=[],
        help="残す範囲の多角形（外は透明にする。何度でも指定できる）。" + _SHAPE_HELP,
    ),
    arg(
        "--drop",
        action="append",
        default=[],
        help="外す範囲の多角形（中を透明にする。何度でも）。書き方は --keep と同じ",
    ),
    arg(
        "--keep-box",
        action="append",
        default=[],
        help="残す四角 'x,y,幅,高さ'（'@秒 x,y,幅,高さ; @秒 ...' で動かせる）",
    ),
    arg(
        "--drop-box",
        action="append",
        default=[],
        help="外す四角（書き方は --keep-box と同じ）",
    ),
    arg(
        "--spec",
        help='範囲を JSON で指定（{"keep": [...], "drop": [...]}。要素は上の書き方の文字列か '
        '{"points": [[x, y], ...]}・{"box": [x, y, w, h]}・{"keys": [{"t": 秒, "points": [...]}, ...]}）',
    ),
    arg(
        "--feather",
        type=float,
        default=0.0,
        help="マスクの縁のぼかし（px。10〜90%% の幅。既定 0）",
    ),
    arg("--invert", action="store_true", help="残す・外すを入れ替える"),
    arg(
        "--mask-only",
        action="store_true",
        help="映像に掛けず、マスク（白 = 残す）だけを白黒の MP4・PNG で書き出す",
    ),
    arg(
        "--mask-out",
        help="マスク（白 = 残す）も書き出す場所（52・67・74 の --mask に渡せる）",
    ),
)
def garbage_matte(a: argparse.Namespace) -> None:
    """多角形・四角でいらない所を外す。透過つきの入力はアルファに掛け、普通の映像は外した所を透明にする。"""
    info = core.probe(a.src)
    size = core.scaled_size(info.width, info.height, max_side=a.max_side)
    scale = (size[0] / info.width, size[1] / info.height)
    src_size = (info.width, info.height)
    items = [("keep", t, False) for t in a.keep] + [("drop", t, False) for t in a.drop]
    items += [("keep", t, True) for t in a.keep_box] + [
        ("drop", t, True) for t in a.drop_box
    ]
    if a.spec:
        items += _spec_items(a.spec)
    core.require(
        bool(items),
        "--keep・--drop・--keep-box・--drop-box・--spec のどれかで範囲を指定してください",
    )
    shapes = [(kind, _parse_shape(text, box, src_size)) for kind, text, box in items]
    gm = _Garbage(shapes, scale, size, feather=a.feather, invert=a.invert)
    has_alpha = info.has_alpha or (info.is_image and _image_has_alpha(a.src))
    if a.mask_only:
        if a.out:
            out = Path(a.out)
            ok = core.IMAGE_EXTS if info.is_image else _VIDEO_EXTS
            core.require(
                out.suffix.lower() in ok, f"出力の拡張子が入力と合いません: {out}"
            )
        else:
            out = core.default_out(
                a.src, "garbage-mask", ".png" if info.is_image else ".mp4"
            )
    else:
        out = _out_path(a, info, "garbage-matte", alpha=True)
    json_path = _json_path(a, out)
    mask_out = Path(a.mask_out) if a.mask_out else None
    writer: Any = None
    mwriter: _GrayWriter | None = None
    if not info.is_image:
        if a.mask_only:
            writer = _GrayWriter(out, info.fps)
        else:
            writer = core.VideoWriter(
                out,
                info.fps,
                alpha=True,
                audio_from=a.src,
                audio_start=a.start or 0.0,
                audio_duration=a.duration,
            )
        if mask_out is not None:
            mwriter = _GrayWriter(mask_out, info.fps)
    total = _total(info, a)
    pick = {0, total // 2, max(0, total - 1)}
    frames = core.read_frames(a.src, **_rng(a), max_side=a.max_side, alpha=has_alpha)
    t0 = time.time()
    stats, picks = [], []
    last = None
    for i, frame in enumerate(frames):
        t = (a.start or 0.0) + i / info.fps
        m = gm(t)
        rgb, al = _alpha_of(frame, has_alpha)
        new_a = al * m
        result = _to_u8(m) if a.mask_only else np.dstack([rgb, _to_u8(new_a)])
        if writer is not None:
            writer.write(result)
        else:
            core.save_image(out, result)
        if mask_out is not None:
            if mwriter is not None:
                mwriter.write(m)
            else:
                core.save_image(mask_out, _to_u8(m))
        stats.append(
            {
                "t": round(t, 4),
                "kept": round(float(m.mean()), 5),
                "visible": round(float(new_a.mean()), 5),
            }
        )
        rec = (i, t, rgb.copy(), al, m, new_a)
        if i in pick:
            picks.append(rec)
        last = rec
        _progress(i, total, t0)
    if writer is not None:
        writer.close()
    if mwriter is not None:
        mwriter.close()
    if last is not None and all(p[0] != last[0] for p in picks):
        picks.append(last)
    images, labels = [], []
    for i, t, rgb, al, m, new_a in picks[-3:]:
        h, w = m.shape
        board = _checker(h, w, max(8, min(h, w) // 40))
        when = "" if info.is_image else f" {t:.2f}秒"
        images += [
            gm.draw(_over(rgb, al, board), t),
            _alpha_img(m),
            _over(rgb, new_a, board),
        ]
        labels += [
            f"元{when}（緑 = 残す・赤 = 外す）",
            "マスク（白 = 残す）",
            "結果（市松模様 = 透明）",
        ]
    check = _sibling(out, "-check.jpg")
    core.save_image(check, core.contact_sheet(images, labels, cols=3, cell_width=320))
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "output": str(out),
            "mask_out": str(mask_out) if mask_out else None,
            "params": {
                "shapes": [
                    {
                        "kind": kind,
                        "keys": [{"t": t, "points": p.round(2)} for t, p in keys],
                    }
                    for kind, keys in shapes
                ],
                "feather": a.feather,
                "invert": a.invert,
            },
            "frames": len(stats),
            "per_frame": stats,
        },
    )
    moving = sum(len(k) > 1 for _, k in shapes)
    core.log(
        f"範囲 {len(shapes)} 個（残す {sum(k == 'keep' for k, _ in shapes)}・外す "
        f"{sum(k == 'drop' for k, _ in shapes)}・動く {moving}）、{len(stats)} コマ: 残した面積の平均 "
        f"{np.mean([s['kept'] for s in stats]) * 100:.1f}%"
    )
    core.log(f"保存: {out}")
    if mask_out is not None:
        core.log(f"保存: {mask_out}")
    core.log(f"保存: {check}")
    core.log(f"保存: {json_path}")


# ── フェザー ──────────────────────────────────────────────────────────────


@command(
    66,
    "feather",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--blur",
        type=float,
        default=0.0,
        help="縁をぼかす幅（px。アルファが 10%% から 90%% に変わる幅。既定 0）",
    ),
    arg("--choke", type=float, default=0.0, help="縁を内側へ縮める（px。端数も可）"),
    arg("--spread", type=float, default=0.0, help="縁を外側へ広げる（px。端数も可）"),
    arg(
        "--where",
        choices=["center", "inside", "outside"],
        default="center",
        help="ぼかしの置き場所: center = 元の縁がぼかしの真ん中（既定）/ inside = ぼかしを元の縁の内側に"
        "収める（外の背景がはみ出さない）/ outside = 元の縁の外に出す",
    ),
    arg(
        "--bg",
        help="背景（画像・動画）に重ねた MP4・画像を書き出す（透過つきの入力だけ）",
    ),
)
def feather(a: argparse.Namespace) -> None:
    """切り抜きの縁を縮める・広げる・ぼかす。透過つき WebM・PNG はアルファを直し、白黒のマスクはマスクのまま直す。"""
    core.require(
        a.blur > 0 or a.choke != 0 or a.spread != 0,
        "--blur・--choke・--spread のどれかを指定してください",
    )
    core.require(min(a.blur, a.choke, a.spread) >= 0, "値は 0 以上にしてください")
    info = core.probe(a.src)
    size = core.scaled_size(info.width, info.height, max_side=a.max_side)
    has_alpha = info.has_alpha or (info.is_image and _image_has_alpha(a.src))
    params = {"blur": a.blur, "choke": a.choke, "spread": a.spread, "where": a.where}
    frames = core.read_frames(a.src, **_rng(a), max_side=a.max_side, alpha=has_alpha)
    if has_alpha:
        out = _KeyOutput(a, info, "feather", size)
        for i, frame in enumerate(frames):
            rgb, al = _alpha_of(frame, True)
            new_a = _feather(
                al, choke=a.choke, spread=a.spread, blur=a.blur, where=a.where
            )
            fg = _extend_colors(rgb.astype(np.float32) / 255.0, al)
            h, w = al.shape
            view = _over(rgb, al, _checker(h, w, max(8, min(h, w) // 40)))
            out.write(i, view, new_a, fg)
        out.close(params)
        return
    # 透過の無い入力は、白黒のマスク（白 = 残す）として扱う
    core.require(not a.bg, "--bg は透過つきの入力（WebM・PNG）にだけ使えます")
    core.log("透過の無い入力なので、白黒のマスク（白 = 残す）として直します")
    if a.out:
        out_path = Path(a.out)
        ok = core.IMAGE_EXTS if info.is_image else _VIDEO_EXTS
        core.require(
            out_path.suffix.lower() in ok, f"出力の拡張子が入力と合いません: {out_path}"
        )
    else:
        out_path = core.default_out(
            a.src, "feather", ".png" if info.is_image else ".mp4"
        )
    json_path = _json_path(a, out_path)
    writer = None if info.is_image else _GrayWriter(out_path, info.fps)
    total = _total(info, a)
    pick = {0, total // 2, max(0, total - 1)}
    t0 = time.time()
    stats, images, labels = [], [], []
    for i, frame in enumerate(frames):
        m = core.to_gray(frame).astype(np.float32) / 255.0
        new_m = _feather(m, choke=a.choke, spread=a.spread, blur=a.blur, where=a.where)
        if writer is not None:
            writer.write(new_m)
        else:
            core.save_image(out_path, _to_u8(new_m))
        stats.append(
            {
                "t": core.t_of(i, info.fps, a.start or 0.0),
                "before": round(float(m.mean()), 5),
                "after": round(float(new_m.mean()), 5),
            }
        )
        if i in pick and len(images) < 6:
            images += [_alpha_img(m), _alpha_img(new_m)]
            when = (
                ""
                if info.is_image
                else f" {core.t_of(i, info.fps, a.start or 0.0):.2f}秒"
            )
            labels += [f"元のマスク{when}", "直したマスク"]
        _progress(i, total, t0)
    if writer is not None:
        writer.close()
    check = _sibling(out_path, "-check.jpg")
    core.save_image(check, core.contact_sheet(images, labels, cols=2, cell_width=320))
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "output": str(out_path),
            "params": params,
            "per_frame": stats,
        },
    )
    core.log(
        f"{len(stats)} コマ: 白の面積 {np.mean([s['before'] for s in stats]) * 100:.1f}% → "
        f"{np.mean([s['after'] for s in stats]) * 100:.1f}%"
    )
    core.log(f"保存: {out_path}")
    core.log(f"保存: {check}")
    core.log(f"保存: {json_path}")


# ── ルミナンスキー ─────────────────────────────────────────────────────────


def _luma_key(
    f: np.ndarray,
    *,
    low: float,
    high: float,
    channel: str,
    invert: bool,
    keep_color: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """明るさでアルファを決める。f は BGR の 0〜1。(アルファ, 前景の色) を返す。

    黒い背景（既定）: アルファ = (値 − low) ÷ (high − low)。値は max（3 色の最大。色の濃い火花も明るく数える）・
    luma（Rec.709 の輝度）・min。白い背景（invert）: 値 = 1 − min（3 色の最小）。
    前景の色は、背景の色（黒・白）を差し引いて割り戻す（暗い縁の黒ずみ・明るい縁の白っぽさが出ない）。keep_color なら元の色のまま。
    """
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    ch = channel if channel != "auto" else ("min" if invert else "max")
    if ch == "max":
        v = np.maximum(np.maximum(r, g), b)
    elif ch == "min":
        v = np.minimum(np.minimum(r, g), b)
    else:
        v = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if invert:
        v = 1.0 - v
    alpha = np.clip((v - low) / max(high - low, 1e-4), 0.0, 1.0).astype(np.float32)
    if keep_color:
        return alpha, f
    a3 = np.maximum(alpha[..., None], 1e-3)
    key = 1.0 if invert else 0.0
    fg = (f - (1.0 - alpha[..., None]) * key) / a3
    fg = np.where(alpha[..., None] > 0.004, fg, f)
    return alpha, np.clip(fg, 0.0, 1.0).astype(np.float32)


@command(
    67,
    "luma-key",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--low",
        type=float,
        default=0.04,
        help="これより暗い所を完全に透明にする（0〜1。黒の浮き・ノイズを消す。既定 0.04）",
    ),
    arg(
        "--high",
        type=float,
        default=1.0,
        help="これより明るい所を不透明にする（0〜1。既定 1.0 = 明るさに比例して透ける。煙を濃くしたい時は下げる）",
    ),
    arg(
        "--invert",
        action="store_true",
        help="白い背景を抜く（暗い所を残す。墨・黒い煙・白地の素材）",
    ),
    arg(
        "--channel",
        choices=["auto", "max", "luma", "min"],
        default="auto",
        help="明るさの測り方: auto（黒い背景は max = 3 色の最大、白い背景は min）/ luma = 輝度（ffmpeg の lumakey と同じ考え方）",
    ),
    arg(
        "--keep-color",
        action="store_true",
        help="色を元のままにする（既定は背景の黒・白を差し引いて割り戻し、縁の黒ずみを出さない）",
    ),
    arg("--bg", help="重ねる背景（画像か動画）。指定すると重ねた MP4・画像を書く"),
    arg("--mask", help="ガベージマット（白 = 残してよい所。画像・動画）"),
    arg("--alpha-out", help="アルファだけの白黒の映像（画像）も書き出す場所"),
)
def luma_key(a: argparse.Namespace) -> None:
    """明るさで抜く（黒い背景の火花・煙・光など。--invert で白い背景）。透過つき WebM か --bg に重ねた映像にする。"""
    core.require(
        0 <= a.low < a.high <= 1,
        "--low は --high より小さく、0〜1 にしてください",
    )
    info = core.probe(a.src)
    size = core.scaled_size(info.width, info.height, max_side=a.max_side)
    out = _KeyOutput(a, info, "luma-key", size)
    params = {
        "low": a.low,
        "high": a.high,
        "invert": a.invert,
        "channel": a.channel,
        "keep_color": a.keep_color,
    }
    for i, frame in enumerate(core.read_frames(a.src, **_rng(a), max_side=a.max_side)):
        f = frame[:, :, :3].astype(np.float32) / 255.0
        alpha, fg = _luma_key(
            f,
            low=a.low,
            high=a.high,
            channel=a.channel,
            invert=a.invert,
            keep_color=a.keep_color,
        )
        out.write(i, frame, alpha, fg)
    out.close(params)


# ── クリーンプレート・差分キー ──────────────────────────────────────────────


def _sample(
    src: str, n: int, a: argparse.Namespace, info: core.MediaInfo
) -> tuple[list[np.ndarray], list[float]]:
    """範囲から等間隔に n コマ取る（1 回の読み込みで。時刻は元の動画の秒）。"""
    start = a.start or 0.0
    span = a.duration or max(info.duration - start, 0.0)
    if a.max_frames:
        span = min(span, a.max_frames / info.fps)
    core.require(span > 0, "範囲が空です（--start・--duration を確かめてください）")
    n = max(3, min(n, round(span * info.fps)))
    frames = list(
        core.read_frames(
            src, start=start, duration=span, fps=n / span, max_side=a.max_side
        )
    )
    times = [
        round(start + (k + 0.5) * span / len(frames), 3) for k in range(len(frames))
    ]
    return frames, times


def _masked_median(
    stack: np.ndarray, exclude: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """画素ごとの中央値（exclude が True のコマは除く）。(プレート uint8, 使えたコマの数) を返す。

    メモリを抑えるため、横長の帯に分けて計算する。どのコマも除かれた画素は、除かずに中央値を取る。
    """
    n, h, w, _ = stack.shape
    plate = np.zeros((h, w, 3), np.uint8)
    count = np.full((h, w), n, np.int32)
    rows = max(8, int(2e7 // max(1, n * w * 3)))
    for y0 in range(0, h, rows):
        part = stack[:, y0 : y0 + rows]
        plain = np.median(part, axis=0)
        if exclude is None:
            plate[y0 : y0 + rows] = np.clip(plain + 0.5, 0, 255).astype(np.uint8)
            continue
        ex = exclude[:, y0 : y0 + rows]
        f = part.astype(np.float32)
        f[ex] = np.nan
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(f, axis=0)
        valid = (~ex).sum(0)
        med = np.where((valid > 0)[..., None], med, plain)
        plate[y0 : y0 + rows] = np.clip(med + 0.5, 0, 255).astype(np.uint8)
        count[y0 : y0 + rows] = valid
    return plate, count


def _people_boxes(
    frames: list[np.ndarray], detector: str, score: float
) -> list[np.ndarray]:
    """人の検出（共有の detectors）の枠を、少し広げて塗った地図（True = 人）の並び。"""
    from .detectors import ObjectDetector

    det = ObjectDetector(model=detector, score=score, classes=["person"])
    out = []
    for f in frames:
        h, w = f.shape[:2]
        m = np.zeros((h, w), bool)
        for d in det(f):
            x1, y1, x2, y2 = d.box
            bw, bh = x2 - x1, y2 - y1
            # 枠を左右 8%・上 5%・下 12%（足元の影）広げる
            xa, xb = int(max(0, x1 - 0.08 * bw)), int(min(w, x2 + 0.08 * bw))
            ya, yb = int(max(0, y1 - 0.05 * bh)), int(min(h, y2 + 0.12 * bh))
            m[ya:yb, xa:xb] = True
        out.append(m)
    return out


def _to_lab(img: np.ndarray) -> np.ndarray:
    """BGR（uint8 か 0〜255 の float）→ Lab（L 0〜100、a・b はおよそ ±127）。"""
    f = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
    return cv2.cvtColor(f, cv2.COLOR_BGR2LAB)


class _DiffKeyer:
    """クリーンプレート（人のいない背景）との色の差で前景を抜く。

    1. コマとプレートを少しぼかし（ノイズ・圧縮の乱れを抑える）、明るさの全体のずれ（露出のゆらぎ）を、差の小さい
       画素の色の比で合わせる。2. Lab の色の差（ΔE76）を求める。影（暗くなっただけで色味がほぼ同じ所）は差を 0 に
       する。3. しきい値の前後 softness の幅で 0〜1 にし、0.5 で切った形を開く・閉じる処理と、小さな塊・小さな穴の
       除去で整える。4. 縁の ±edge px だけ、色の差のなだらかな値（しきい値の前後 softness の幅）を使う
       （合成の試験で、ガイデッドフィルターで縁を作るより正解に近かった）。
    """

    def __init__(
        self,
        plate: np.ndarray,
        *,
        threshold: float,
        softness: float,
        shadows: str,
        clean: int | None,
        min_area: float,
        edge: float,
    ) -> None:
        h, w = plate.shape[:2]
        short = min(h, w)
        self.sigma = max(0.8, short / 540)
        self.plate = cv2.GaussianBlur(
            plate[:, :, :3].astype(np.float32), (0, 0), self.sigma
        )
        self.threshold = threshold
        self.softness = max(softness, 1e-3)
        self.shadows = shadows
        self.clean = clean if clean is not None else max(1, round(short / 270))
        self.min_area = min_area
        self.edge = edge if edge is not None else max(1.0, round(short / 270))

    def delta(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(色の差 ΔE, 明るさを合わせたプレート)。"""
        f = cv2.GaussianBlur(frame[:, :, :3].astype(np.float32), (0, 0), self.sigma)
        p = self.plate
        # 露出のゆらぎ: 差の小さい半分の画素で、色ごとの比の中央値を求めて合わせる
        d0 = np.abs(f - p).sum(-1)
        sel = d0 <= np.percentile(d0, 50)
        gain = np.median(f[sel] + 1.0, axis=0) / np.median(p[sel] + 1.0, axis=0)
        p = np.clip(p * gain[None, None], 0, 255)
        lf, lp = _to_lab(f), _to_lab(p)
        d = lf - lp
        de = np.sqrt((d * d).sum(-1))
        if self.shadows == "remove":
            de = np.where(_shadow(f, p), 0.0, de)
        return de.astype(np.float32), p

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        de, _ = self.delta(frame)
        lo = self.threshold - self.softness / 2
        soft = np.clip((de - lo) / self.softness, 0.0, 1.0)
        hard = (soft > 0.5).astype(np.uint8)
        r = self.clean
        if r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, k)
            hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, k)
        hard = _clean_blobs(hard.astype(bool), self.min_area).astype(np.uint8)
        e = round(self.edge)
        if e <= 0:
            return hard.astype(np.float32)
        # 縁の ±edge px だけ、色の差のなだらかな値（しきい値の前後 softness の幅）を使う。内側は 1、外は 0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))
        near = cv2.dilate(hard, k) > 0
        inner = cv2.erode(hard, k) > 0
        alpha = np.where(inner, 1.0, np.where(near, soft, 0.0))
        return alpha.astype(np.float32)


def _shadow(f: np.ndarray, p: np.ndarray) -> np.ndarray:
    """影らしい所（Horprasert ら 1999 の考え方）: 背景の色を k 倍しただけ（k = 0.35〜0.95）で、色のずれ
    |f − k×p| ÷ |p| が 0.06 未満。f・p は BGR（0〜255）。暗い無彩色の服を着た人も影と見分けられないので注意。"""
    pp = (p * p).sum(-1) + 1e-3
    k = (f * p).sum(-1) / pp
    cd = np.sqrt(((f - k[..., None] * p) ** 2).sum(-1)) / np.sqrt(pp)
    return (k > 0.35) & (k < 0.95) & (cd < 0.06) & (pp > 900)


def _clean_blobs(mask: np.ndarray, min_area: float) -> np.ndarray:
    """小さな塊（画面の min_area 未満）を消し、塊の中の小さな穴（同じ大きさ未満）を埋める。"""
    h, w = mask.shape
    lim = min_area * h * w
    m = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= lim
    out = keep[lab]
    inv = (~out).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for k in range(1, n):
        x, y, bw, bh, area = st[k]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and area < lim:
            out[lab == k] = True
    return out


def _plate_from(
    a: argparse.Namespace, info: core.MediaInfo, samples: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """--plate の画像（または動画の最初のコマ）か、映像から中央値で作ったプレート。"""
    size = core.scaled_size(info.width, info.height, max_side=a.max_side)
    if getattr(a, "plate", None):
        img = (
            core.frame_at(a.plate, 0.0)
            if not core.is_image(a.plate)
            else core.load_image(a.plate)
        )
        return _cover(img[:, :, :3], size), {"plate": str(a.plate)}
    core.require(
        not info.is_image, "画像 1 枚には --plate（人のいない背景の画像）が要ります"
    )
    frames, _ = _sample(a.src, samples, a, info)
    exclude = None
    if getattr(a, "exclude_people", False):
        exclude = np.stack(_people_boxes(frames, "fast", 0.3))
    plate, _ = _masked_median(np.stack(frames), exclude)
    how = "人を除いた" if exclude is not None else ""
    return plate, {"plate": f"映像の {len(frames)} コマの{how}中央値"}


@command(
    75,
    "clean-plate",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--samples",
        type=int,
        default=61,
        help="中央値を取るコマ数（範囲から等間隔に。既定 61）",
    ),
    arg(
        "--exclude-people",
        action="store_true",
        help="人の検出（MediaPipe の EfficientDet-Lite2）で人がいる所を中央値から除く（長く立ち止まる人を消す）",
    ),
    arg(
        "--detector",
        choices=["fast", "accurate"],
        default="fast",
        help="--exclude-people の検出器（fast: EfficientDet-Lite2。既定 / accurate: Faster R-CNN。CPU で 1 コマ数秒以上。"
        "試験の合成映像では両方同じ結果）",
    ),
    arg(
        "--exclude-mask",
        help="中央値から除く所の白黒マスク（白 = 除く。動画・画像。人以外の動く物に）",
    ),
    arg(
        "--video",
        help="動く人・物の所をプレートで埋めた映像（MP4）も書き出す場所（人を消した映像）",
    ),
    arg(
        "--threshold",
        type=float,
        default=12.0,
        help="--video で「動く物」とみなす色の差（Lab の ΔE。既定 12）",
    ),
)
def clean_plate(a: argparse.Namespace) -> None:
    """固定カメラの映像から、時間方向の中央値で人のいない背景（PNG）を作る。--video で人を消した映像も。"""
    info = core.probe(a.src)
    core.require(
        not info.is_image, "動画を指定してください（何コマもの中央値で作るため）"
    )
    out = Path(a.out) if a.out else core.default_out(a.src, "clean-plate", ".png")
    core.require(
        out.suffix.lower() in core.IMAGE_EXTS,
        f"出力は画像（.png）にしてください: {out}",
    )
    json_path = _json_path(a, out)
    t0 = time.time()
    frames, times = _sample(a.src, a.samples, a, info)
    core.log(f"{len(frames)} コマを取りました（{times[0]:.2f}〜{times[-1]:.2f} 秒）")
    stack = np.stack([f[:, :, :3] for f in frames])
    h, w = stack.shape[1:3]
    exclude = None
    if a.exclude_people:
        boxes = _people_boxes(frames, a.detector, 0.3)
        exclude = np.stack(boxes)
        core.log(
            f"人の検出: 人のいたコマ {sum(b.any() for b in boxes)}/{len(boxes)}、"
            f"除いた面積の平均 {np.mean([b.mean() for b in boxes]) * 100:.1f}%"
        )
    if a.exclude_mask:
        layer = _Layer(a.exclude_mask, (w, h), "mask")
        ex = []
        lm = layer.info
        for t in times:
            img = (
                core.load_image(a.exclude_mask, alpha=True)
                if lm.is_image
                else core.frame_at(a.exclude_mask, t, alpha=lm.has_alpha)
            )
            ex.append(layer._conv(img) > 0.5)
        exm = np.stack(ex)
        exclude = exm if exclude is None else (exclude | exm)
    plate, count = _masked_median(stack, exclude)
    core.save_image(out, plate)
    # 確かめにくい所: 半分以上のコマで、プレートと大きく違う（ずっと何かが居る・動き続ける）画素
    diff = np.abs(stack.astype(np.int16) - plate[None].astype(np.int16)).max(-1)
    busy = (diff > 25).mean(0)
    unsure = (busy > 0.5) | (count < 3)
    # 確認画像: 取ったコマ 3 枚・プレート・確かめにくい所
    picks = [0, len(frames) // 2, len(frames) - 1]
    heat = core.heatmap(busy, lo=0.0, hi=1.0)
    heat[unsure] = (0, 0, 255)
    images = [frames[k] for k in picks] + [plate, heat]
    labels = [f"元 {times[k]:.2f}秒" for k in picks] + [
        "クリーンプレート",
        "動きの多さ（赤 = 半分以上のコマで違う＝確かめて）",
    ]
    check = _sibling(out, "-check.jpg")
    core.save_image(check, core.contact_sheet(images, labels, cols=3, cell_width=360))
    data: dict[str, Any] = {
        "source": str(a.src),
        "output": str(out),
        "samples": len(frames),
        "times": times,
        "exclude_people": a.exclude_people,
        "unsure_ratio": round(float(unsure.mean()), 5),
        "busy_mean": round(float(busy.mean()), 5),
    }
    core.log(
        f"プレート {w}×{h}: 確かめにくい所 {unsure.mean() * 100:.2f}%"
        f"（半分以上のコマで違う画素。人が長く居た所など）"
    )
    if a.video:
        vout = Path(a.video)
        core.require(
            vout.suffix.lower() in _VIDEO_EXTS,
            f"--video は動画（.mp4）にしてください: {vout}",
        )
        keyer = _DiffKeyer(
            plate,
            threshold=a.threshold,
            softness=6.0,
            shadows="keep",
            clean=None,
            min_area=0.0005,
            edge=0.0,
        )
        grow = max(2, round(min(h, w) / 120))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        total = _total(info, a)
        filled = []
        with core.VideoWriter(
            vout,
            info.fps,
            audio_from=a.src,
            audio_start=a.start or 0.0,
            audio_duration=a.duration,
        ) as wr:
            t1 = time.time()
            for i, frame in enumerate(
                core.read_frames(a.src, **_rng(a), max_side=a.max_side)
            ):
                m = keyer(frame)
                m = cv2.dilate((m > 0.5).astype(np.uint8), k).astype(np.float32)
                m = _feather(m, blur=grow)
                _, padj = keyer.delta(frame)
                # 明るさを合わせたプレート（ぼかす前）で埋める
                gain = padj.reshape(-1, 3).mean(0) / (
                    keyer.plate.reshape(-1, 3).mean(0) + 1e-6
                )
                fill = np.clip(plate.astype(np.float32) * gain, 0, 255)
                res = fill * m[..., None] + frame[:, :, :3].astype(np.float32) * (
                    1 - m[..., None]
                )
                wr.write(np.clip(res + 0.5, 0, 255).astype(np.uint8))
                filled.append(float(m.mean()))
                _progress(i, total, t1)
        data["video"] = str(vout)
        data["video_filled_mean"] = round(float(np.mean(filled)), 5)
        core.log(f"埋めた面積の平均 {np.mean(filled) * 100:.1f}%")
        core.log(f"保存: {vout}")
    core.write_json(json_path, data)
    core.log(f"（中央値まで {time.time() - t0:.1f} 秒）")
    core.log(f"保存: {out}")
    core.log(f"保存: {check}")
    core.log(f"保存: {json_path}")


@command(
    74,
    "diff-key",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--plate",
        help="人のいない背景（画像。動画なら最初のコマ）。省略時は 75 と同じ方法で映像から作る（固定カメラ）",
    ),
    arg(
        "--threshold",
        type=float,
        default=12.0,
        help="前景とみなす色の差（Lab の ΔE。既定 12。背景に似た服が抜ける時は下げ、ノイズが残る時は上げる）",
    ),
    arg(
        "--softness",
        type=float,
        default=8.0,
        help="しきい値の前後で半透明にする幅（ΔE。既定 8）",
    ),
    arg(
        "--shadows",
        choices=["keep", "remove"],
        default="keep",
        help="影の扱い: keep = 前景に含める（既定）/ remove = 背景の色を暗くしただけの所を背景にする"
        "（灰色など無彩色の暗い服の人も一部消えることがある）",
    ),
    arg(
        "--clean",
        type=int,
        help="小さな点・すき間を消す大きさ（px。開く・閉じる処理の半径。既定は短い辺の 1/270）",
    ),
    arg(
        "--min-area",
        type=float,
        default=0.0005,
        help="これより小さい塊・穴は消す（画面に対する割合。既定 0.0005）",
    ),
    arg(
        "--edge",
        type=float,
        help="縁の半透明の幅（px。縁の内外この幅だけ、色の差のなだらかな値を使う。0 で固い縁。既定は短い辺の 1/270）",
    ),
    arg(
        "--samples",
        type=int,
        default=61,
        help="--plate を省いた時に中央値を取るコマ数（既定 61）",
    ),
    arg(
        "--exclude-people",
        action="store_true",
        help="--plate を省いた時、人の検出で人がいる所を中央値から除く（長く立ち止まる人がいる映像に。75 と同じ）",
    ),
    arg("--bg", help="差し替える背景（画像か動画）。指定すると重ねた MP4・画像を書く"),
    arg("--mask", help="ガベージマット（白 = 残してよい所。画像・動画）"),
    arg("--alpha-out", help="アルファだけの白黒の映像（画像）も書き出す場所"),
)
def diff_key(a: argparse.Namespace) -> None:
    """無人の背景（クリーンプレート）との色の差で、人や物を抜く（固定カメラ用）。"""
    info = core.probe(a.src)
    size = core.scaled_size(info.width, info.height, max_side=a.max_side)
    plate, pinfo = _plate_from(a, info, a.samples)
    core.log(f"プレート: {pinfo['plate']}")
    out = _KeyOutput(a, info, "diff-key", size)
    if not a.plate:
        pp = _sibling(out.out, "-plate.png")
        core.save_image(pp, plate)
        core.log(f"保存: {pp}")
    keyer = _DiffKeyer(
        plate,
        threshold=a.threshold,
        softness=a.softness,
        shadows=a.shadows,
        clean=a.clean,
        min_area=a.min_area,
        edge=a.edge,
    )
    params = {
        **pinfo,
        "threshold": a.threshold,
        "softness": a.softness,
        "shadows": a.shadows,
        "clean": keyer.clean,
        "min_area": a.min_area,
        "edge": keyer.edge,
    }
    for i, frame in enumerate(core.read_frames(a.src, **_rng(a), max_side=a.max_side)):
        alpha = keyer(frame)
        out.write(i, frame, alpha, frame[:, :, :3])
    out.close(params)
