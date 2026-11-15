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
import itertools
import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import core
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/scenes/（make_*.py で正解つきの試験映像を作り、test_*.py で照らす）
NOTES: dict[int, dict[str, str]] = {
    76: {
        "status": "◎",
        "how": "PySceneDetect 0.7.3（BSD-3-Clause）の AdaptiveDetector（既定。--method で content・histogram・hash も選べる）に、"
        "幅 256px に縮めたコマを渡す。後処理を 3 つ足した: (1) adaptive の比を、差がほぼ 0 の前後のコマ（重複したコマ）を除いて"
        "求め直す（低いコマ数の映像で小さな動きがカットに見えるのを防ぐ）、(2) 0.25 秒以内に直前の画へ戻る変化はフラッシュと"
        "してカットにしない。カットの直後 0.25 秒の変化（白を通したカットの戻り）はまとめる、(3) 0.3 秒より短い場面はまとめる。"
        "変化の大きさの下限（min_content）は PySceneDetect の既定 15 から 10 に下げた（同じ画角のジャンプカットを拾うため）。"
        "出力: 時刻の JSON・印を付けた確認用 MP4（カットから約 0.12 秒、赤い枠と「カット n（秒）」の帯。全コマに場面の番号。"
        "音声は残す）・各場面の先頭コマの一覧 JPG",
        "measured": "story540: カット 4/4 をコマ単位で一致（73・139・213・297 コマ = 3.042・5.792・8.875・12.375 秒。"
        "「3.0 秒」の境目は 72 と 73 コマの間）。v4: 11/11 をコマ単位で一致（白いフラッシュで始まる 2.5・6.0・16.0 秒を含む）。"
        "難しい合成映像（カット 6・同じ画角のジャンプカット 1・1 コマ 40px の速いパン・同じ場面の白 2 コマ・白 2 コマを通る"
        "カット・0.5 秒のディゾルブ）: カット 7/7 一致、パンと同じ場面の白は数えず、白を通るカットは 1 回だけ。"
        "カットの無い映像（vtest 79 秒・a-eye・b-lashlift・c-brow・手を振る tree.avi）: 誤検出 0。"
        "後処理 (1) の前は tree.avi（449 コマ中 381 コマが重複）で 4 件（min_content 10 なら 9 件）の誤検出だった。"
        "min_content 15 ではジャンプカット（変化 10.3）を見逃した。content・histogram も 21/21・誤検出 0 だがジャンプカットを"
        "見逃し、hash は tree.avi で 2 件誤検出。速さ: v4（540×960・600 コマ）で JSON と一覧だけなら 4.2 秒、確認用 MP4 も"
        "作ると 15.5 秒（他の作業と CPU を分け合った状態の壁時計）",
        "limits": "溶けるつなぎ目（フェード・ディゾルブ）はカットとして数えない（92 を使う）。0.25 秒以内に元の画へ戻る"
        "差し込みはフラッシュ扱い（--flash 0 で無効）。0.3 秒より短い場面はまとめる（--min-scene）。変化が 10 未満の"
        "ジャンプカットは見逃す（--min-content を下げると拾えるが誤検出が増え得る）。白いフラッシュで始まる場面は一覧の"
        "先頭コマが白くなる（代表コマは 83 を使う）。TransNetV2（MIT、ONNX）も比べたが v4 の白いフラッシュのカット 2 つを"
        "見逃したので使っていない",
        "tested": "story540・v4（作品の場面の始まりを正解に）、正解つきの合成映像 scenes-hard（make_scenes_hard.py）、"
        "カットの無い vtest・a-eye・b-lashlift・c-brow・tree.avi。test_scenes_sweep.py で方法と設定を比べ、"
        "review_scenes.py で確認用 MP4 のカット前後のコマを目で確認",
    },
    83: {
        "status": "◎",
        "how": "76 と同じ方法で場面に分け、各場面の両端（場面の 1 割・最大 0.1 秒）を除いたコマから選ぶ。鮮明さ"
        "（ラプラシアンの分散を約 6×10 の区画ごとに求め、上位 10% の区画の値の平方根。長辺 960px で測る）が上位 4 割のコマの"
        "うち、場面の平均（幅 48px の Lab の小さな画）にいちばん近いコマ。--per-scene で場面を等分して複数選ぶ。"
        "出力: 元の大きさの PNG（フォルダ）・一覧 JPG・JSON",
        "measured": "story540 の各場面で 4 割のコマ（計 131 コマ）をぼかし（σ=2.5）、2 コマずつ（計 10 コマ）を別の場面のコマに"
        "差し替えた試験: 5 場面すべてで、正しい場面の、ぼかしていない・差し替えていないコマを選んだ（5/5）。選んだコマの"
        "鮮明さは、その場面のコマの上位 18〜32%。v4: 12 場面から 12 枚。どれも文字の動きが落ち着いた後の画で、白いフラッシュの"
        "コマは選ばなかった（目で確認）。速さ: v4 で 7.7 秒、story540 で 4.6 秒",
        "limits": "「平均に近い」なので、場面の途中で大きく変わる（後から文字が出る等）と中間の状態を選ぶことがある"
        "（--per-scene 2〜3 で前後を拾う）。動きのぶれもぼけとして避ける。とても短い場面（5 コマ未満）は端のコマも候補に入る。"
        "小さなコマをすべて覚えるので、縦長 30fps で 1 分あたり約 110MB（横長は約 35MB）のメモリを使う",
        "tested": "make_keyframes_test.py の合成映像（ぼかし・差し替えの正解つき。test_keyframes.py）、v4、story540（--start・"
        "--duration・--per-scene 2）",
    },
    89: {
        "status": "◎",
        "how": "長辺 480px の灰色のコマで、前のコマとの差を 8×8 の区画ごとに平均し、どの区画も 5 以下かつ画面全体の平均の差が"
        " 0.3 以下（0〜255）なら同じ画とする（圧縮の乱れは広く薄く、ゆっくりした動きは平均の差が大きい）。続いた重複を"
        "まとめ、0.5 秒以上なら「停止」。前後の動きが大きい所の重複を「カクつきが目立つ」とする。重複のまとまりの間隔が"
        "そろっていれば（8 割以上が同じ間隔±1）フレームレート変換の跡として、元のコマ数 ≒ fps ×（1 − 1 回の重複の数 ÷ 間隔）を出す。--ffmpeg で ffmpeg の"
        " mpdecimate（落とすコマ）と freezedetect（止まった区間）の結果も JSON に入れて比べる。出力: JSON と確認画像"
        "（差のグラフと、前・元・重複・次のコマ）",
        "measured": "story540（もともと重複なし）の決まったコマを直前のコマで置き換えた試験（1 回の重複 5・2 回続き 1・"
        "18 コマの停止 1 = 正解 25 コマ）を H.264 の crf 16・23・30 で書き出し: 3 つとも 25/25 一致・誤検出 0。最初の"
        "「区画の最大 3 以下」だけの判定は crf 30 で 3 コマ見逃した（実測: crf 30 の重複は区画の最大 4.34 以下・平均 0.17 以下、"
        "ゆっくりした実写の動きは区画の最大 5.17 以上・平均 0.4 以上）。24fps → 30fps（ffmpeg の fps=30。正解は劣化のない"
        "中間ファイルで作る）: 30/30 一致・誤検出 0、「5 コマに 1 回」・元は 24.0fps と推定（重複の数 ÷ 間隔から）。"
        "ffmpeg mpdecimate も合成映像は全部一致したが、ゆっくり動く a-eye で 4 コマ・c-brow で 2 コマを重複と誤判定"
        "（こちらは 0）。重複の無い story540・a-eye・b-lashlift・c-brow・vtest: 0 コマ。tree.avi: 449 コマ中 381 コマが重複、"
        "約 6.6 コマに 1 回・1 回に 5.6 コマ → 元は約 2.3fps（mpdecimate も 381 コマ）。v4: 18 コマ（文字が止まった瞬間や、24fps の映像を"
        " 30fps の作品に入れた 0.8 秒など）。速さ: v4 で 3.9 秒、story540 で 1.3 秒",
        "limits": "ごくゆっくりした動き（区画の差 5 未満・平均 0.3 未満）は重複に見える。強く圧縮した映像（crf 30 超）は"
        " --hi を 6〜8 に上げる必要があるかもしれない（未確認）。粒子・ノイズが毎コマ変わる映像は重複を見つけにくい。"
        "止まった画の作品（文字だけの場面など）は正しく「同じ画」と出るが、不具合とは限らない（「目立つ」の印を見る）",
        "tested": "make_dups_test.py の合成映像（crf 16/23/30 と 24→30fps。test_duplicates.py）、story540・a-eye・b-lashlift・"
        "c-brow・vtest・tree.avi・v4",
    },
    92: {
        "status": "◎",
        "how": "76 のカットで区切り、幅 48px の小さなコマを画素のベクトルとして見る。溶けるつなぎ目の間は前の画と後の画の"
        "混ぜ物なので、両端の差（弦）が大きく（既定 30 以上）、1 コマずつの差の合計（道のり）に近い区間を探し、"
        "「弦の二乗 ÷ 道のり」が最大の区間を選んで、両端の画への近さ α（3% の幅）で始めと終わりを決める。分類: 端が一色"
        "（明るさのばらつき 8 以下。平均 40 以下は黒、215 以上は白）ならフェードイン・アウト（まっすぐさ 0.7 以上）、"
        "両端とも画なら、模様が似ておらず（相関 0.85 未満）途中のばらつきが両端を結んだ値の 0.9 倍以下にくぼむ物を"
        "ディゾルブ（人や物の動きではくぼまない）。白・黒への変化が 0.4 秒以下でカットと同時に起きる物は「フラッシュ」として"
        "フェードと分ける（76 の同じ場面のフラッシュも載せる）。出力: JSON と確認画像（明るさのグラフと、始め・中・終わりのコマ）",
        "measured": "ffmpeg の fade と xfade で既知の時刻に作った 15.7 秒の試験映像（黒からのフェードイン 1 秒・ディゾルブ 1 秒・"
        "白へのフェードアウト 0.5 秒・白からのフェードイン 0.5 秒・ディゾルブ 0.5 秒・黒へのフェードアウト 1 秒・"
        "黒からのフェードイン 0.5 秒・カットと同時に白から 0.25 秒で戻る）: 8/8 で種類と色が一致、始め・終わりの誤差は"
        "最大 1 コマ・平均 0.28 コマ（30fps）、余計な検出 0。追加の試験: 両方の画が大きく動く 2 秒のディゾルブは"
        " 1.567〜3.367 秒（正解 1.5〜3.5。始め +2・終わり −4 コマ）、ワイプ（wipeleft）と押し出し（slideleft）は誤検出なし、"
        "xfade の fadeblack は黒へ 6.0〜6.167 秒＋黒から 6.333〜7.0 秒（実際の明るさの変化と一致。fadeblack は 0.17 秒で黒に"
        "なり、少し止まってから戻る）。溶けるつなぎ目の無い story540・a-eye・b-lashlift・c-brow・vtest・tree.avi・Megamind: 0 件。"
        "v4: 白いフラッシュ 3 つ（2.5・6.0・16.0 秒。カットと同時に 90% の白が 0.22〜0.28 秒で引く）を「フラッシュ（白）」と"
        "分類し、フェード・ディゾルブは 0 件（押し出しの転換や文字の動きを誤認しない）。速さ: v4 で 4.4 秒、story540 で 2.5 秒",
        "limits": "両方の画が大きく動く長いディゾルブは境目が数コマずれる（上の例で終わり −4 コマ）。よく似た画どうしの"
        "ディゾルブ（差 30 未満）は見逃す（--min-change）。ワイプ・押し出し・スライドは溶けるつなぎ目ではないので出さない。"
        "一色まで行かない半端な暗転（50% の灰色まで等）は出さない。4 秒より長い物は --max-len を上げる。小さなコマを"
        "すべて覚えるので、縦長 30fps で 1 分あたり約 110MB（横長は約 35MB）のメモリを使う（長い動画は --start・--duration で区切る）。"
        "TransNetV2（MIT、ONNX）も比べたが、2 秒のディゾルブを見つけられず、v4 の白いフラッシュのカット 2 つを見逃したので"
        "使っていない",
        "tested": "make_fades_test.py（fade・xfade の正解つき）と make_fades_test2.py（2 秒のディゾルブ・wipeleft・fadeblack・"
        "slideleft）、scenes-hard（0.5 秒のディゾルブ 12.233〜12.767 秒を一致）、溶けるつなぎ目の無い映像 7 本、v4"
        "（test_fades.py）",
    },
    93: {
        "status": "◎",
        "how": "コマごとの鮮明さ = ラプラシアン（輪郭の強さ）の分散を約 6×10 の区画ごとに求め、上位 10% の区画の値の"
        "平方根（長辺 960px。背景がぼけた写真でも主役で測れる）。76 の方法で場面に分け、同じ場面の前後 1.5 秒のコマの"
        "上位 25%（75 パーセンタイル）を基準に、その 0.6 倍より低いコマをぼけとする。--remove でぼけたコマを抜いた動画も"
        "書く（音声なし）。画像 1 枚なら、鮮明さの数値・区画ごとの鮮明さの地図・ffmpeg blurdetect（輪郭の幅）の値を出す",
        "measured": "story540 の決まったコマをぼかした試験（ガウス σ 0.8・1.2・1.6・2.0・3.0 を各場面 1 コマずつ、σ2.0 の"
        " 3 コマ続き、横の動きぶれ 9px・15px）: 既定（0.6 倍）で σ1.2 以上 22/22・動きぶれ 4/4・σ0.8 は 2/5。ほかに 1 コマ"
        "（3.0 秒、元の映像でビューラーが動いてぶれているコマ。目で確認。手を加えていない story540 でもこの 1 コマだけを"
        "検出）。0.5 倍では追加の検出 0 だが σ1.2 が 4/5・9px の動きぶれが 1/2、0.7 倍では σ0.8 も 5/5 だが追加 3 コマ。"
        "--remove: 361 → 332 コマ、書き出した動画を測り直すとぼけ 0。画像: 自作のくっきりした文字 172.6 → 焦点ずれ"
        "（円形のぼかし）半径 1・2・3・4・6・8px で 94.8・60.8・37.1・23.8・11.2・8.7 と単調に下がる。OpenCV の"
        " text_defocus.jpg は 11.05（半径 6px 相当）。ffmpeg blurdetect も 4.25 → 12.48 と単調、text_defocus は 10.2。"
        "くっきりした写真（building・fruits・home）は 67〜184。速さ: v4 で 6.7 秒、story540 で 4.4 秒",
        "limits": "同じ場面の中で比べるので、場面全体がぼけている時は見つけない（画像の数値で絶対的に見る）。"
        "動きのぶれもぼけとして出る。鮮明さの値は写っている物（模様の多さ）と大きさで変わるので、別の画像どうしの比較は"
        "同じ大きさ・似た内容で。σ0.8 ほどの軽いぼけは半分ほど見逃す（--ratio 0.7 で拾えるが誤検出が増える）",
        "tested": "make_blur_test.py の合成映像（ぼかしの位置と強さの正解つき。test_blur.py）、story540、text_defocus.jpg と"
        "自作の文字画像（焦点ずれ 7 段階）、building.jpg・fruits.jpg・home.jpg",
    },
    99: {
        "status": "○",
        "how": "WCAG 2.3.1 の考え方: コマを長辺 320px で読み、sRGB を光の強さに戻して（IEC 61966-2-1 の式）相対輝度を求め、"
        "長辺 64 個の区画で平均する。区画ごとに山と谷を追い、相対輝度 0.1 以上の上下（暗い側が 0.8 未満）を 1 回の変化、"
        "反対向きの 2 回で 1 回の点滅とする。赤は WCAG 2.0/2.1 の注 3 の式（どちらかの状態で R/(R+G+B) ≥ 0.8、(R−G−B)×320 の"
        "変化が 20 より大きい）。直前 1 秒の点滅の回数を区画ごとに数え、画面の 25%（--area）以上の面積が達した回数を"
        "その時の点滅の回数とし、3 回（--max-flashes）を超える区間を出す（面積を画面全体に対する割合で見るのは ITU-R BT.1702 の"
        "考え方）。出力: JSON（コマごとの回数・区間・判定）とグラフ画像",
        "measured": "合成映像（30fps・32 秒）の 8 区間: 全画面の黒⇔白 2・3・4・8Hz → 2.0・3.0・4.0・8.0 回/秒（2・3Hz は"
        "範囲内、4・8Hz は超過）、画面の 10% だけ 8Hz → 0（範囲内）、30% だけ 8Hz → 8.0（超過）、暗い赤（sRGB 100,0,0）⇔黒"
        " 4Hz → 赤 4.0（超過。明るさは 0.5 で範囲内）、相対輝度 0.85 の灰⇔白 8Hz → 範囲内（暗い側が 0.8 以上）: 8/8 が期待"
        "どおり。超過区間は 4.13〜7.03・8.07〜11.03・16.07〜19.03 秒、赤 20.13〜23.03 秒（正解 4〜7・8〜11・16〜19・"
        "20〜23 秒）。v4（白いフラッシュ・明滅する光）: 最大 2.0 回/秒（1.43 秒、光る大きな文字「卒業」が飛ぶ所）、赤 0、"
        "既定の 25% では範囲内。--area 0.05 では 0.53〜2.57 秒が最大 4 回/秒、0.02 では 5 区間（報告のみ。広告は直していない）。"
        "速さ: v4 で 3.2 秒、story540 で 2.6 秒",
        "limits": "簡易な目安で、Harding FPA や PEAT のような認証用の判定ではない。面積は画面全体に対する割合で見る"
        "（WCAG の「視野 10 度のどこかの 25%」とは違う。1024×768 で見る机の画面なら、それは画面の約 2.8% の面積に当たる"
        "ので、厳しく見る時は --area 0.03）。表示の明るさ（cd/m²）は考えない（相対輝度 0.1・0.8 は、表示の最大 200cd/m² なら BT.1702 の"
        " 20cd/m²・160cd/m² に当たる）。縞・格子などの模様、5 秒を超えて続く点滅の扱いは見ていない。WCAG 2.2 の注（w3c/wcag の"
        "原文で確認）は赤の点滅を「R/(R+G+B) ≥ 0.8 の状態を含み、CIE 1976 UCS の色度の差が 0.2 を超える上下」に改めているが、"
        "この道具は 2.0/2.1 の注の式のまま（黒の色度の扱いなど ISO 9241-391 を確かめられないため、2.2 の式は入れていない）",
        "tested": "make_flash_test.py の合成映像（周波数・面積・赤・明るい灰の 8 区間。test_flashes.py）、v4（--area 0.25・"
        "0.1・0.05・0.02）",
    },
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


# ── 共通の部品 ────────────────────────────────────────────────────────────

_RED = (40, 40, 230)
_GREEN = (60, 160, 40)
_BLUE = (200, 110, 20)
_ORANGE = (20, 140, 240)
_GRAY = (150, 150, 150)
_PURPLE = (170, 60, 160)
_SHEET_MAX = 60  # 一覧画像に並べる数の上限


def _video(src: str) -> core.MediaInfo:
    info = core.probe(src)
    core.require(
        not info.is_image,
        "動画を指定してください（時間の流れを調べるコマンドなので、画像 1 枚では使えません）",
    )
    return info


def _rng(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": a.start or 0.0,
        "duration": a.duration,
        "max_frames": a.max_frames,
    }


def _width_size(info: core.MediaInfo, width: int) -> tuple[int, int]:
    """幅を width px にそろえた大きさ（元の方が小さければ元のまま。偶数）。"""
    if info.width <= width:
        return max(2, info.width - info.width % 2), max(
            2, info.height - info.height % 2
        )
    return width, max(2, round(info.height * width / info.width / 2) * 2)


def _abs_frame(i: int, fps: float, start: float) -> int:
    """調べた範囲の中のコマ番号 → 元の動画の中のコマ番号。"""
    return round(start * fps) + int(i)


def _luma(img: np.ndarray) -> np.ndarray:
    """BGR（0〜255）→ 明るさ（BT.601 の重み。0〜255 の float32）。"""
    f = img[..., :3].astype(np.float32)
    return f[..., 0] * 0.114 + f[..., 1] * 0.587 + f[..., 2] * 0.299


_LABELS: dict[tuple[Any, ...], np.ndarray] = {}


def _label(
    text: str,
    size: int,
    color: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """文字の札（小さな画像）。同じ札は作り直さない（動画の全コマに描く時に速くするため）。"""
    key = (text, size, color, bg)
    if key not in _LABELS:
        canvas = np.empty(
            (int(size * 1.8) + 8, int(len(text) * size * 1.1) + 16, 3), np.uint8
        )
        canvas[:] = bg
        core.draw_text(canvas, text, (6, 4), size=size, color=color, bg=bg, pad=4)
        ys, xs = np.nonzero(np.any(canvas != np.array(bg, np.uint8), axis=2))
        patch = canvas[: ys.max() + 6, : xs.max() + 7] if len(xs) else canvas
        _LABELS[key] = patch.copy()
    return _LABELS[key]


def _paste(img: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """札を画像に貼る（はみ出す所は切る）。"""
    h, w = img.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + pw), min(h, y + ph)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1, :3] = patch[y0 - y : y1 - y, x0 - x : x1 - x]


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def _chart(
    t: np.ndarray,
    series: list[tuple[np.ndarray, tuple[int, int, int], str]],
    *,
    title: str,
    width: int = 1400,
    height: int = 300,
    ylim: tuple[float, float] | None = None,
    log: bool = False,
    hlines: list[tuple[float, tuple[int, int, int], str]] | None = None,
    spans: list[tuple[float, float, tuple[int, int, int]]] | None = None,
    marks: list[tuple[float, tuple[int, int, int]]] | None = None,
    dots: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    """時間の流れのグラフ（横軸は秒）。series は (値, 色, 名前)。白地の BGR 画像を返す。"""
    hlines, spans, marks, dots = hlines or [], spans or [], marks or [], dots or []
    img = np.full((height, width, 3), 255, np.uint8)
    left, right, top, bottom = 70, 20, 36, 30
    pw, ph = width - left - right, height - top - bottom
    t = np.asarray(t, np.float64)
    t0 = float(t[0]) if len(t) else 0.0
    t1 = float(t[-1]) if len(t) > 1 else t0 + 1.0
    t1 = max(t1, t0 + 1e-3)

    def tr(v: Any) -> np.ndarray:
        v = np.asarray(v, np.float64)
        return np.log10(np.clip(v, 1e-3, None)) if log else v

    if ylim is not None:
        lo, hi = float(tr(ylim[0])), float(tr(ylim[1]))
    else:
        vals = [tr(s[0]) for s in series] + [tr([h[0] for h in hlines])]
        allv = np.concatenate([v[np.isfinite(v)] for v in vals if v.size] or [[0.0]])
        lo, hi = float(allv.min()), float(allv.max())
        pad = (hi - lo) * 0.06 or 1.0
        lo, hi = lo - pad, hi + pad
    if hi - lo < 1e-9:
        hi = lo + 1.0

    def px(tt: Any) -> np.ndarray:
        return np.round(left + (np.asarray(tt, np.float64) - t0) / (t1 - t0) * pw)

    def py(v: Any) -> np.ndarray:
        y = top + ph - (tr(v) - lo) / (hi - lo) * ph
        return np.round(np.clip(np.nan_to_num(y, nan=top + ph), top, top + ph))

    if spans:
        ov = img.copy()
        for a0, a1, color in spans:
            cv2.rectangle(ov, (int(px(a0)), top), (int(px(a1)), top + ph), color, -1)
        img = cv2.addWeighted(ov, 0.22, img, 0.78, 0)
    step = next(
        (s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800)
         if (t1 - t0) / s <= 16),
        3600,
    )  # fmt: skip
    for k in range(math.ceil(t0 / step), int(t1 / step) + 1):
        x = int(px(k * step))
        cv2.line(img, (x, top), (x, top + ph), (228, 228, 228), 1)
        core.draw_text(
            img, f"{k * step:g}", (x, top + ph + 4), size=13, color=(90, 90, 90),
            bg=None, anchor="mt",
        )  # fmt: skip
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = round(top + ph - k / 4 * ph)
        cv2.line(img, (left, y), (left + pw, y), (236, 236, 236), 1)
        core.draw_text(
            img, _fmt(10**v if log else v), (left - 6, y), size=13, color=(90, 90, 90),
            bg=None, anchor="rm",
        )  # fmt: skip
    for tt, color in marks:
        x = int(px(tt))
        cv2.line(img, (x, top), (x, top + ph), color, 1)
    for v, color, name in hlines:
        y = int(py(v))
        for x in range(left, left + pw, 14):
            cv2.line(img, (x, y), (min(x + 8, left + pw), y), color, 2)
        if name:
            core.draw_text(
                img, name, (left + pw - 6, y - 3), size=13, color=color, bg=None,
                anchor="rb",
            )  # fmt: skip
    for vals, color, _name in series:
        vals = np.asarray(vals, np.float64)
        xs, ys = px(t[: len(vals)]), py(vals)
        ok = np.isfinite(tr(vals))
        seg: list[list[float]] = []
        for x, y, good in zip(xs, ys, ok, strict=True):
            if good:
                seg.append([x, y])
            elif seg:
                cv2.polylines(
                    img, [np.array(seg, np.int32)], False, color, 2, cv2.LINE_AA
                )
                seg = []
        if seg:
            cv2.polylines(img, [np.array(seg, np.int32)], False, color, 2, cv2.LINE_AA)
    for tt, vv, color in dots:
        for x, y in zip(px(tt), py(vv), strict=True):
            cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (left, top), (left + pw, top + ph), (120, 120, 120), 1)
    core.draw_text(img, title, (left, 8), size=16, color=(20, 20, 20), bg=None)
    x = left + pw
    for _vals, color, name in reversed(series):
        if not name:
            continue
        patch = _label(name, 14, (40, 40, 40), (255, 255, 255))
        x -= patch.shape[1]
        _paste(img, patch, x, 6)
        yy = 6 + patch.shape[0] // 2
        cv2.line(img, (x - 26, yy), (x - 6, yy), color, 3)
        x -= 40
    core.draw_text(
        img, "秒", (left + pw, top + ph + 4), size=13, color=(90, 90, 90), bg=None,
        anchor="rt",
    )  # fmt: skip
    return img


def _stack(images: list[np.ndarray], width: int = 1400) -> np.ndarray:
    """画像を縦に並べる（幅をそろえる）。"""
    rows = []
    for im in images:
        im = im[:, :, :3]
        if im.shape[1] != width:
            h = max(1, round(im.shape[0] * width / im.shape[1]))
            im = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
        rows += [im, np.full((6, width, 3), 24, np.uint8)]
    return np.vstack(rows[:-1])


def _pick_frames(
    src: str, rng: dict[str, Any], wanted: set[int], max_side: int | None = 360
) -> dict[int, np.ndarray]:
    """調べた範囲の中のコマ番号を指定して、そのコマを読む（頭から順に読むのでコマ単位で正確）。"""
    got: dict[int, np.ndarray] = {}
    if not wanted:
        return got
    last = max(wanted)
    for i, f in enumerate(core.read_frames(src, max_side=max_side, **rng)):
        if i in wanted:
            got[i] = f
        if i >= last:
            break
    return got


def _gray_grid(h: int, w: int, short: int = 6) -> tuple[int, int]:
    """画面をほぼ正方形の区画に分ける時の (行, 列)。短い辺を short 個に分ける。"""
    if h >= w:
        return max(1, round(short * h / w)), short
    return short, max(1, round(short * w / h))


# ── 76 シーン検出 ─────────────────────────────────────────────────────────

# 方法ごとの既定のしきい値（PySceneDetect の既定と同じ）
_METHODS = {"adaptive": 3.0, "content": 27.0, "histogram": 0.2, "hash": 0.35}
_SMALL_W = 48  # 似ているかの判定・フェードの解析に使う小さなコマの幅（px）
_SAME = 12.0  # 「同じ画に戻った」とみなす差（小さなコマの HSV の平均の差）


@dataclass
class CutResult:
    """カット検出の結果。コマ番号は調べた範囲の中での番号（0 から）。"""

    fps: float
    start: float
    n: int
    cuts: list[int]
    score: np.ndarray
    ratio: np.ndarray | None
    flashes: list[tuple[int, int]]
    small: list[np.ndarray]
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def t(self, i: float) -> float:
        return round(self.start + i / self.fps, 4)

    def frame(self, i: int) -> int:
        return _abs_frame(i, self.fps, self.start)

    def scenes(self) -> list[tuple[int, int]]:
        """場面の (最初のコマ, 最後のコマ)。"""
        b = [0, *self.cuts, self.n]
        return [(b[k], b[k + 1] - 1) for k in range(len(b) - 1)]


def _hsv_diff(a: np.ndarray, b: np.ndarray) -> float:
    """2 つの小さなコマの違い（HSV の各成分の差の平均。PySceneDetect の content_val と同じ考え方）。"""
    ha = cv2.cvtColor(a, cv2.COLOR_BGR2HSV).astype(np.int16)
    hb = cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype(np.int16)
    return float(np.abs(ha - hb).mean())


def _robust_ratio(score: np.ndarray, c: int, width: int = 2, look: int = 12) -> float:
    """adaptive の比（このコマの変化 ÷ 前後 2 コマずつの変化の平均）を、重複したコマ（差がほぼ 0）を除いて求める。"""
    near: list[float] = []
    for d in range(1, look + 1):
        for j in (c - d, c + d):
            if 1 <= j < len(score) and score[j] >= 1.0 and len(near) < 2 * width:
                near.append(float(score[j]))
        if len(near) >= 2 * width:
            break
    if not near:
        return math.inf
    return float(score[c]) / (sum(near) / len(near))


def detect_cuts(
    src: str,
    *,
    method: str = "adaptive",
    threshold: float | None = None,
    min_content: float = 10.0,
    min_scene: float = 0.3,
    flash: float = 0.25,
    start: float = 0.0,
    duration: float | None = None,
    max_frames: int | None = None,
) -> CutResult:
    """カット（新しい場面の先頭のコマ）を見つける。83・92・93 からも使う。

    1. PySceneDetect の検出器（幅 256px に縮めたコマ）で、変化の大きいコマを拾う（場面の最短の長さは後で見る）。
    2. 拾ったコマの直前の画に、flash 秒以内に戻るなら、カットではなくフラッシュとする。
    3. カットの直後 flash 秒以内の変化（白を通したカットの戻りなど）と、min_scene 秒より短い場面はまとめる。
    """
    from scenedetect import FrameTimecode, StatsManager
    from scenedetect.detectors import (
        AdaptiveDetector,
        ContentDetector,
        HashDetector,
        HistogramDetector,
    )

    info = _video(src)
    fps = info.fps
    core.require(
        method in _METHODS, f"知らない方法です: {method}（{' / '.join(_METHODS)}）"
    )
    thr = _METHODS[method] if threshold is None else threshold
    if method == "adaptive":
        det: Any = AdaptiveDetector(
            adaptive_threshold=thr, min_scene_len=0, min_content_val=min_content
        )
    elif method == "content":
        det = ContentDetector(threshold=thr, min_scene_len=0)
    elif method == "histogram":
        det = HistogramDetector(threshold=thr, min_scene_len=0)
    else:
        det = HashDetector(threshold=thr, min_scene_len=0)
    stats = StatsManager()
    det.stats_manager = stats
    size = _width_size(info, 256)
    small_size = _width_size(info, _SMALL_W)
    raw: list[int] = []
    small: list[np.ndarray] = []
    frames = core.read_frames(
        src, start=start, duration=duration, max_frames=max_frames, size=size
    )
    for i, f in enumerate(frames):
        raw += [c.frame_num for c in det.process_frame(FrameTimecode(i, fps=fps), f)]
        small.append(cv2.resize(f, small_size, interpolation=cv2.INTER_AREA))
    n = len(small)
    raw += [
        c.frame_num for c in det.post_process(FrameTimecode(max(n - 1, 0), fps=fps))
    ]
    keys = det.get_metrics()
    score = np.zeros(n)
    ratio = np.zeros(n) if method == "adaptive" else None
    for i in range(n):
        tc = FrameTimecode(i, fps=fps)
        v = stats.get_metrics(tc, [keys[0]])[0]
        score[i] = float(v) if v is not None else 0.0
        if ratio is not None:
            r = stats.get_metrics(tc, [keys[-1]])[0]
            ratio[i] = float(r) if r is not None else 0.0
    if method == "adaptive":
        # 前後に同じ画（重複したコマ）が続くと比の分母が 0 になり、小さな動きもカットに見える
        # （OpenCV の tree.avi で実測）。差がほぼ 0 のコマを除いて比を求め直す
        raw = [c for c in raw if 0 < c < n and _robust_ratio(score, c) >= thr]
    # フラッシュ・短すぎる場面の整理
    win = max(1, round(flash * fps)) if flash > 0 else 0
    min_len = max(1, round(min_scene * fps))
    cuts: list[int] = []
    flashes: list[tuple[int, int]] = []
    skip = -1
    for c in sorted({c for c in raw if 0 < c < n}):
        if c <= skip:
            continue
        if win:
            before = small[c - 1]
            jump = _hsv_diff(before, small[c])
            back = next(
                (
                    j
                    for j in range(c + 1, min(c + win, n - 1) + 1)
                    if _hsv_diff(before, small[j]) <= min(_SAME, 0.5 * jump)
                ),
                None,
            )
            if back is not None:
                flashes.append((c, back - 1))
                skip = back
                continue
        if cuts and c - cuts[-1] < min_len:
            continue
        cuts.append(c)
        skip = c + win
    params = {
        "threshold": thr,
        "min_scene": min_scene,
        "flash": flash,
        "analysis_size": list(size),
    }
    if method == "adaptive":
        params["min_content"] = min_content
    return CutResult(fps, start, n, cuts, score, ratio, flashes, small, method, params)


_CUT_ARGS = [
    arg(
        "--method",
        choices=list(_METHODS),
        default="adaptive",
        help="カットの見つけ方（PySceneDetect）。adaptive＝前後のコマの変化と比べる（既定。速いカメラの動きに強い）、"
        "content＝色と明るさの差、histogram＝明るさの分布の差、hash＝画の指紋の差",
    ),
    arg(
        "--threshold",
        type=float,
        help="カットとみなす値（既定は方法ごと: adaptive 3.0・content 27・histogram 0.2・hash 0.35。小さいほど多く見つける）",
    ),
    arg(
        "--min-content",
        type=float,
        default=10.0,
        help="adaptive の時、カットとみなす変化の大きさ（content_val）の下限（既定 10。PySceneDetect の既定は 15）",
    ),
    arg(
        "--min-scene",
        type=float,
        default=0.3,
        help="場面の最短の長さ（秒。既定 0.3）",
    ),
    arg(
        "--flash",
        type=float,
        default=0.25,
        help="この秒数以内に元の画へ戻る変化はフラッシュとみなし、カットにしない（秒。既定 0.25。0 で無効）",
    ),
]


def _cuts_from_args(a: argparse.Namespace) -> CutResult:
    return detect_cuts(
        a.src,
        method=a.method,
        threshold=a.threshold,
        min_content=a.min_content,
        min_scene=a.min_scene,
        flash=a.flash,
        **_rng(a),
    )


@command(
    76,
    "scenes",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    *_CUT_ARGS,
    arg(
        "--no-video",
        action="store_true",
        help="印を付けた確認用の動画を作らない（速い。JSON と一覧画像だけ）",
    ),
)
def scenes(a: argparse.Namespace) -> None:
    """カットの変わり目を見つけ、時刻を JSON に、印を付けた確認用の動画を MP4 に、各場面の先頭コマの一覧を JPG に書く。"""
    res = _cuts_from_args(a)
    rng = _rng(a)
    out = Path(a.out) if a.out else core.default_out(a.src, "scenes", ".mp4")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    sheet_path = out.with_suffix(".jpg")
    scs = res.scenes()
    cut_info = []
    for c in res.cuts:
        item: dict[str, Any] = {
            "frame": res.frame(c),
            "time": res.t(c),
            "score": round(float(res.score[c]), 3),
        }
        if res.ratio is not None:
            item["adaptive_ratio"] = round(float(res.ratio[c]), 3)
        cut_info.append(item)
    data = {
        "source": str(a.src),
        "fps": res.fps,
        "start": res.start,
        "frames": res.n,
        "method": res.method,
        "params": res.params,
        "cuts": cut_info,
        "scenes": [
            {
                "no": k + 1,
                "start": res.t(s),
                "end": res.t(e + 1),
                "duration": round((e + 1 - s) / res.fps, 4),
                "first_frame": res.frame(s),
                "last_frame": res.frame(e),
            }
            for k, (s, e) in enumerate(scs)
        ],
        "flashes": [
            {
                "start": res.t(s),
                "end": res.t(e + 1),
                "frames": [res.frame(s), res.frame(e)],
                "note": "直前の画に戻ったので、カットにしていない",
            }
            for s, e in res.flashes
        ],
    }
    core.write_json(json_path, data)
    core.log(f"カット: {len(res.cuts)} 件（場面 {len(scs)} 個、{res.method} 方式）")
    for k, (s, e) in enumerate(scs):
        core.log(
            f"  場面 {k + 1:2d}: {res.t(s):7.3f}〜{res.t(e + 1):7.3f} 秒（{(e + 1 - s) / res.fps:.2f} 秒）"
        )
    if res.flashes:
        core.log(f"フラッシュ（カットにしない）: {len(res.flashes)} 件")
        for s, e in res.flashes:
            core.log(f"  {res.t(s):.3f}〜{res.t(e + 1):.3f} 秒")
    # 一覧画像: 各場面の先頭のコマ
    firsts = [s for s, _ in scs[:_SHEET_MAX]]
    got = _pick_frames(a.src, rng, set(firsts))
    imgs = [got[s] for s in firsts if s in got]
    labels = [
        f"{k + 1}: {res.t(s):.2f}〜{res.t(e + 1):.2f}秒"
        for k, (s, e) in enumerate(scs[:_SHEET_MAX])
        if s in got
    ]
    if len(scs) > _SHEET_MAX:
        core.log(f"一覧画像には最初の {_SHEET_MAX} 場面だけを並べます")
    core.save_image(
        sheet_path,
        core.contact_sheet(imgs, labels, cols=min(6, len(imgs)), cell_width=240),
    )
    # 確認用の動画: カットのコマ（から約 0.12 秒）に赤い枠と帯、全コマに場面の番号
    if not a.no_video:
        scene_of = np.zeros(res.n, np.int32)
        for k, (s, e) in enumerate(scs):
            scene_of[s : e + 1] = k
        flash_frames = {i for s, e in res.flashes for i in range(s, e + 1)}
        hold = max(1, round(0.12 * res.fps))

        def mark(frame: np.ndarray, i: int) -> np.ndarray:
            if i >= res.n:
                return frame
            h, w = frame.shape[:2]
            size = max(14, w // 22)
            k = int(scene_of[i])
            s0 = scs[k][0]
            if k > 0 and i - s0 < hold:
                th = max(6, w // 45)
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), th)
                frame[: int(size * 2.2)] = (0, 0, 255)
                patch = _label(
                    f"カット {k}（{res.t(s0):.3f} 秒）", size, bg=(0, 0, 255)
                )
                _paste(frame, patch, th + 4, th // 2 + 2)
            if i in flash_frames:
                patch = _label(
                    "フラッシュ（カットにしない）", size, (0, 0, 0), (0, 220, 255)
                )
                _paste(frame, patch, 8, h // 2)
            patch = _label(f"場面 {k + 1}/{len(scs)}", size)
            _paste(frame, patch, 8, h - patch.shape[0] - 8)
            return frame

        core.map_video(a.src, out, mark, max_side=a.max_side, progress=False, **rng)
        core.log(f"保存: {out}")
    core.log(f"保存: {sheet_path}")
    core.log(f"保存: {json_path}")


# ── 83 キーフレーム抽出 ───────────────────────────────────────────────────


def _sharpness(gray: np.ndarray) -> tuple[float, np.ndarray]:
    """鮮明さ: ラプラシアン（輪郭の強さ）の分散を区画ごとに求め、上位 10% の区画の値の平方根。

    いちばん鮮明な所で測るので、背景がぼけた写真（主役はくっきり）を、ぼけたコマと間違えにくい。
    """
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    h, w = lap.shape
    gy, gx = _gray_grid(h, w)
    th, tw = h // gy, w // gx
    tiles = lap[: th * gy, : tw * gx].reshape(gy, th, gx, tw)
    var = tiles.var(axis=(1, 3))
    return float(np.sqrt(np.percentile(var, 90))), var


def _sharpness_series(
    src: str, rng: dict[str, Any], max_side: int | None
) -> np.ndarray:
    vals = [
        _sharpness(core.to_gray(f))[0]
        for f in core.read_frames(src, max_side=max_side or 960, **rng)
    ]
    return np.array(vals, np.float64)


@command(
    83,
    "keyframes",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    *_CUT_ARGS,
    arg(
        "--per-scene",
        type=int,
        default=1,
        help="1 場面から選ぶ数（場面を等分して、それぞれから 1 枚。既定 1）",
    ),
    arg(
        "--top",
        type=float,
        default=0.4,
        help="鮮明さが上位この割合のコマの中から、場面の平均にいちばん近いコマを選ぶ（0〜1。既定 0.4）",
    ),
)
def keyframes(a: argparse.Namespace) -> None:
    """場面ごとに代表のコマ（ぶれ・ぼけが少なく、場面の平均に近いコマ）を選び、PNG・一覧画像・JSON に書く。"""
    core.require(a.per_scene >= 1, "--per-scene は 1 以上にしてください")
    core.require(0 < a.top <= 1, "--top は 0 より大きく 1 以下にしてください")
    rng = _rng(a)
    res = _cuts_from_args(a)
    sharp = _sharpness_series(a.src, rng, a.max_side)
    n = min(res.n, len(sharp))
    # 場面の平均に近いかは、小さなコマ（Lab の色）で比べる
    feats = np.stack(
        [
            cv2.cvtColor(s, cv2.COLOR_BGR2LAB).astype(np.float32).ravel()
            for s in res.small[:n]
        ]
    )
    picks = []
    for k, (s, e) in enumerate(res.scenes()):
        e = min(e, n - 1)
        if e < s:
            continue
        length = e - s + 1
        margin = min(
            round(0.1 * res.fps), length // 5
        )  # 場面の両端（変わり目）を避ける
        idx_all = np.arange(s + margin, e - margin + 1)
        for part in np.array_split(idx_all, min(a.per_scene, len(idx_all))):
            dist = np.linalg.norm(feats[part] - feats[part].mean(0), axis=1)
            sh = sharp[part]
            n_top = max(1, math.ceil(len(part) * a.top))
            cand = np.argsort(-sh, kind="stable")[:n_top]
            best = int(cand[np.argmin(dist[cand])])
            i = int(part[best])
            picks.append(
                {
                    "scene": k + 1,
                    "i": i,
                    "frame": res.frame(i),
                    "time": res.t(i),
                    "sharpness": round(float(sharp[i]), 3),
                    "sharpness_vs_best": round(float(sharp[i] / (sh.max() or 1.0)), 3),
                    "distance_rank": int((dist < dist[best]).sum()) + 1,
                    "candidates": len(part),
                }
            )
    folder = Path(a.out) if a.out else core.default_out(a.src, "keyframes", "")
    folder.mkdir(parents=True, exist_ok=True)
    json_path = Path(a.json) if a.json else folder.with_suffix(".json")
    sheet_path = folder.with_suffix(".jpg")
    want = {p["i"]: p for p in picks}
    thumbs: dict[int, np.ndarray] = {}
    last = max(want) if want else -1
    for i, f in enumerate(core.read_frames(a.src, **rng)):
        if i in want:
            p = want[i]
            path = folder / f"scene{p['scene']:02d}-{p['time']:.3f}s.png"
            core.save_image(path, f)
            p["path"] = str(path)
            scale = 360 / max(f.shape[:2])
            thumbs[i] = cv2.resize(
                f, (round(f.shape[1] * scale), round(f.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )  # fmt: skip
        if i >= last:
            break
    for p in picks:
        p.pop("i")
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": res.fps,
            "scenes": len(res.scenes()),
            "method": res.method,
            "params": {**res.params, "per_scene": a.per_scene, "top": a.top},
            "keyframes": picks,
        },
    )
    imgs = [thumbs[i] for i in sorted(thumbs)][:_SHEET_MAX]
    labels = [
        f"場面{want[i]['scene']} {want[i]['time']:.2f}秒" for i in sorted(thumbs)
    ][:_SHEET_MAX]
    core.save_image(
        sheet_path,
        core.contact_sheet(imgs, labels, cols=min(6, len(imgs)), cell_width=240),
    )
    core.log(f"場面 {len(res.scenes())} 個から {len(picks)} 枚を選びました")
    for p in picks:
        core.log(
            f"  場面 {p['scene']:2d}: {p['time']:7.3f} 秒（鮮明さ {p['sharpness']:.1f}、"
            f"場面で最も鮮明なコマの {p['sharpness_vs_best'] * 100:.0f}%）"
        )
    core.log(f"保存: {folder}/")
    core.log(f"保存: {sheet_path}")
    core.log(f"保存: {json_path}")


# ── 89 重複フレーム検出 ───────────────────────────────────────────────────


def _block_diff(
    prev: np.ndarray, gray: np.ndarray, block: int = 8
) -> tuple[float, float]:
    """2 つの灰色のコマの差: (8×8 区画ごとの平均の差の最大, 全体の平均の差)。0〜255。"""
    d = cv2.absdiff(prev, gray)
    h, w = d.shape
    hb, wb = h // block, w // block
    blocks = cv2.resize(
        d[: hb * block, : wb * block].astype(np.float32), (wb, hb),
        interpolation=cv2.INTER_AREA,
    )  # fmt: skip
    return float(blocks.max()), float(d.mean())


_SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*-?\d+\s+pts_time:([-\d.e]+)")
_FREEZE = re.compile(r"freeze_(start|end): ([\d.]+)")


def _ffmpeg_dups(src: str, rng: dict[str, Any], fps: float, n: int) -> dict[str, Any]:
    """ffmpeg の mpdecimate（落とすコマ）と freezedetect（止まった区間）の結果。比べる用。"""
    pre: list[str] = []
    if rng["start"]:
        pre += ["-ss", f"{rng['start']:.4f}"]
    post: list[str] = ["-t", f"{rng['duration']:.4f}"] if rng["duration"] else []
    text = core.ffmpeg_log(
        [*pre, "-i", src, *post, "-vf", "mpdecimate,showinfo", "-an", "-f", "null", "-"]
    )
    kept = {round(float(t) * fps) for _n, t in _SHOWINFO.findall(text)}
    dropped = sorted(i for i in range(n) if i not in kept)
    text = core.ffmpeg_log(
        [
            *pre,
            "-i",
            src,
            *post,
            "-vf",
            "freezedetect=n=-60dB:d=0.5",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    marks = _FREEZE.findall(text)
    freezes = []
    for kind, t in marks:
        if kind == "start":
            freezes.append({"start": round(rng["start"] + float(t), 4)})
        elif freezes and "end" not in freezes[-1]:
            freezes[-1]["end"] = round(rng["start"] + float(t), 4)
    return {"mpdecimate_dropped": dropped, "freezedetect": freezes}


@command(
    89,
    "duplicates",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--hi",
        type=float,
        default=5.0,
        help="重複とみなす差の上限（8×8 の区画ごとの平均の差、0〜255。どの区画もこれ以下なら同じ画。既定 5）",
    ),
    arg(
        "--mean-max",
        type=float,
        default=0.3,
        help="重複とみなす、画面全体の平均の差の上限（0〜255。既定 0.3）",
    ),
    arg(
        "--min-freeze",
        type=float,
        default=0.5,
        help="この秒数以上同じ画が続いたら「停止」とする（既定 0.5）",
    ),
    arg(
        "--motion",
        type=float,
        default=1.0,
        help="前後のコマの平均の差がこれ以上なら「動いている所」とみなす（重複が目立つ。既定 1.0）",
    ),
    arg(
        "--ffmpeg",
        action="store_true",
        help="ffmpeg の mpdecimate と freezedetect の結果も JSON に入れる（比べる用）",
    ),
)
def duplicates(a: argparse.Namespace) -> None:
    """同じ画の繰り返し（重複したコマ）と止まった区間を見つけ、カクつく所を JSON と確認画像に書く。"""
    info = _video(a.src)
    rng = _rng(a)
    fps = info.fps
    start = rng["start"]
    blk, mad = [np.nan], [np.nan]
    prev = None
    for f in core.read_frames(a.src, max_side=a.max_side or 480, **rng):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            b, m = _block_diff(prev, g)
            blk.append(b)
            mad.append(m)
        prev = g
    n = len(blk)
    blk_a, mad_a = np.array(blk), np.array(mad)
    # 同じ画: どの区画の差も hi 以下で、しかも全体の平均の差が mean_max 以下。圧縮の乱れは広く薄く、
    # ゆっくりした動きは平均の差が大きい（実測: 重複 0.01〜0.17、ゆっくりした動き 0.4 以上）
    dup = np.zeros(n, bool)
    dup[1:] = (blk_a[1:] <= a.hi) & (mad_a[1:] <= a.mean_max)
    # 続いている重複をまとめる（i〜j のコマが、i-1 のコマと同じ画）
    runs = []
    i = 1
    while i < n:
        if dup[i]:
            j = i
            while j + 1 < n and dup[j + 1]:
                j += 1
            near = [
                k
                for k in range(max(1, i - 6), min(n, j + 7))
                if not dup[k] and k != i - 1
            ]
            motion = float(np.median(mad_a[near])) if near else 0.0
            frames_held = j - i + 2  # 同じ画が続くコマ数（元のコマを含む）
            runs.append(
                {
                    "first_frame": _abs_frame(i - 1, fps, start),
                    "last_frame": _abs_frame(j, fps, start),
                    "start": core.t_of(i - 1, fps, start),
                    "end": core.t_of(j + 1, fps, start),
                    "repeats": j - i + 1,
                    "kind": "停止" if frames_held / fps >= a.min_freeze else "重複",
                    "motion_around": round(motion, 3),
                    "noticeable": bool(motion >= a.motion),
                }
            )
            i = j + 1
        else:
            i += 1
    dup_idx = np.nonzero(dup)[0]
    total_dups = int(dup.sum())
    # 一定の間隔で重複しているか（フレームレートの変換の跡）。重複のまとまりの始まりの間隔で見る
    periodic = None
    starts = np.array([r["first_frame"] for r in runs])
    if len(starts) >= 6:
        gaps = np.diff(starts)
        vals, counts = np.unique(gaps, return_counts=True)
        mode = int(vals[np.argmax(counts)])
        share = float(np.isin(gaps, [mode - 1, mode, mode + 1]).mean())
        if mode >= 2 and share >= 0.8:
            # 元のコマ数 ≒ fps ×（1 − 1 回あたりの重複の数 ÷ 間隔）。24→30fps なら 30 ×（1 − 1/5）= 24
            reps = float(np.mean([r["repeats"] for r in runs]))
            periodic = {
                "every_frames": round(float(np.mean(gaps)), 3),
                "most_common_gap": mode,
                "share_of_gaps": round(share, 3),
                "repeats_per_run": round(reps, 3),
                "estimated_source_fps": round(
                    fps * (1 - reps / float(np.mean(gaps))), 3
                ),
            }
    noticeable = [r for r in runs if r["noticeable"]]
    data: dict[str, Any] = {
        "source": str(a.src),
        "fps": fps,
        "frames": n,
        "params": {
            "hi": a.hi,
            "mean_max": a.mean_max,
            "min_freeze": a.min_freeze,
            "motion": a.motion,
        },
        "duplicates": total_dups,
        "unique_frames": n - total_dups,
        "effective_fps": round(fps * (n - total_dups) / max(n, 1), 3),
        "runs": runs,
        "noticeable_runs": len(noticeable),
        "periodic": periodic,
        "duplicate_frames": [_abs_frame(int(k), fps, start) for k in dup_idx],
    }
    if a.ffmpeg:
        ff = _ffmpeg_dups(a.src, rng, fps, n)
        ours = {int(k) for k in dup_idx}
        theirs = set(ff["mpdecimate_dropped"])
        ff["agree_with_mpdecimate"] = len(ours & theirs)
        ff["only_ours"] = sorted(_abs_frame(k, fps, start) for k in ours - theirs)
        ff["only_mpdecimate"] = sorted(_abs_frame(k, fps, start) for k in theirs - ours)
        ff["mpdecimate_dropped"] = [_abs_frame(k, fps, start) for k in sorted(theirs)]
        data["ffmpeg"] = ff
    out = Path(a.out) if a.out else core.default_out(a.src, "duplicates", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    core.write_json(json_path, data)
    core.log(
        f"重複したコマ: {total_dups} / {n} コマ（実質 {data['effective_fps']} fps）、"
        f"まとまり {len(runs)} 件（うち動きの中で目立つもの {len(noticeable)} 件）"
    )
    for r in runs[:20]:
        core.log(
            f"  {r['start']:7.3f} 秒〜: {r['kind']}（同じ画が {r['repeats'] + 1} コマ）"
            + ("  ← カクつきが目立つ" if r["noticeable"] else "")
        )
    if len(runs) > 20:
        core.log(f"  …ほか {len(runs) - 20} 件（JSON を参照）")
    if periodic:
        core.log(
            f"一定の間隔で重複: 約 {periodic['every_frames']} コマに 1 回 → "
            f"元は約 {periodic['estimated_source_fps']} fps の映像を {fps:g} fps にした跡の可能性"
        )
    if a.ffmpeg:
        ff = data["ffmpeg"]
        core.log(
            f"ffmpeg mpdecimate が落とすコマ: {len(ff['mpdecimate_dropped'])}（一致 {ff['agree_with_mpdecimate']}）、"
            f"freezedetect の停止区間: {len(ff['freezedetect'])} 件"
        )
    # 確認画像: 差のグラフ（対数）と、重複の例（前・重複・後）
    t = np.array([core.t_of(k, fps, start) for k in range(n)])
    spans = [(r["start"], r["end"], _ORANGE) for r in runs if r["kind"] == "停止"]
    chart = _chart(
        t,
        [(blk_a, _BLUE, "区画ごとの差の最大"), (mad_a, _GRAY, "平均の差")],
        title=f"コマどうしの差（対数）。赤い点 = 重複（{total_dups} コマ。両方の線が上限以下）、橙 = 停止",
        log=True,
        hlines=[
            (a.hi, _RED, f"区画の差の上限 {a.hi:g}"),
            (a.mean_max, _GRAY, f"平均の差の上限 {a.mean_max:g}"),
        ],
        spans=spans,
        dots=[(t[dup_idx], np.maximum(blk_a[dup_idx], 1e-3), _RED)]
        if total_dups
        else [],
    )
    parts = [chart]
    examples = (noticeable or runs)[:6]
    if examples:
        want = set()
        for r in examples:
            i0 = r["first_frame"] - round(start * fps)
            want |= {max(0, i0 - 1), i0, i0 + 1, min(n - 1, i0 + r["repeats"] + 1)}
        got = _pick_frames(a.src, rng, want, max_side=240)
        imgs, labels = [], []
        for r in examples:
            i0 = r["first_frame"] - round(start * fps)
            for k, name in (
                (i0 - 1, "前"),
                (i0, "元"),
                (i0 + 1, "重複"),
                (i0 + r["repeats"] + 1, "次"),
            ):
                if k in got and 0 <= k < n:
                    imgs.append(got[k])
                    labels.append(f"{name} {core.t_of(k, fps, start):.3f}秒")
        if imgs:
            parts.append(core.contact_sheet(imgs, labels, cols=8, cell_width=175))
    core.save_image(out, _stack(parts))
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 92 フェード検出 ───────────────────────────────────────────────────────


def _solid_kind(img: np.ndarray, max_std: float) -> str | None:
    """ほぼ一色のコマなら色の名前（黒・白・単色）、そうでなければ None。"""
    y = _luma(img)
    if float(y.std()) > max_std:
        return None
    m = float(y.mean())
    return "黒" if m <= 40 else "白" if m >= 215 else "単色"


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """2 つのコマの模様の似ている度合い（正規化相互相関。1 に近いほど同じ画）。"""
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 1e-6 else 0.0


def _gradual(
    x: np.ndarray,
    bounds: list[int],
    *,
    min_change: float,
    min_straight: float,
    min_len: int,
    max_len: int,
    delta: float,
    cut_share: float = 0.5,
) -> list[dict[str, Any]]:
    """直線的に混ざる変化（フェード・ディゾルブ）を探す。x は (コマ数, 画素) の小さなコマ。

    溶けるつなぎ目の間は、どのコマも「前の画」と「後の画」を混ぜた物になる。画素の空間では、
    前の画から後の画へまっすぐ進むので、区間の両端の差（弦）と、1 コマずつの差の合計（道のり）が
    近くなる（まっすぐさ = 弦 ÷ 道のり）。区間は「弦の二乗 ÷ 道のり」がいちばん大きい物を選ぶ
    （つなぎ目の外の動きを含めると道のりだけが伸び、つなぎ目の途中で切ると弦が縮むため、ちょうどの区間で最大）。
    区間はカット（bounds）をまたがない。見つけた区間は、両端の画への近さ α（0〜1）で境目を決め直す。
    """
    n = len(x)
    step = np.zeros(n, np.float32)
    if n > 1:
        step[1:] = np.sqrt(((x[1:] - x[:-1]) ** 2).mean(1))
    path = np.cumsum(step)
    cands = []
    for s, e in itertools.pairwise(bounds):
        for a0 in range(s, e - min_len):
            bmax = min(e - 1, a0 + max_len)
            if bmax - a0 < min_len:
                continue
            bs = np.arange(a0 + min_len, bmax + 1)
            chord = np.sqrt(((x[a0 + min_len : bmax + 1] - x[a0]) ** 2).mean(1))
            route = path[bs] - path[a0]
            mstep = np.maximum.accumulate(step[a0 + 1 : bmax + 1])[min_len - 1 :]
            ok = (chord >= min_change) & (chord >= min_straight * route)
            ok &= mstep <= cut_share * chord
            if ok.any():
                value = np.where(ok, chord**2 / np.maximum(route, 1e-6), -1.0)
                j = int(np.argmax(value))
                cands.append((float(value[j]), a0, int(bs[j]), s, e))
    cands.sort(key=lambda c: (-c[0], c[2] - c[1]))
    found: list[dict[str, Any]] = []
    for _value, a0, b0, s, e in cands:
        if any(not (b0 < f["st"] or a0 > f["en"]) for f in found):
            continue
        ext = max(2, min_len)
        lo, hi = max(s, a0 - ext), min(e - 1, b0 + ext)
        v = x[b0] - x[a0]
        vv = float(v @ v) or 1.0
        al = ((x[lo : hi + 1] - x[a0]) @ v) / vv
        # 真ん中（α≒0.5）は選んだ区間の中で探す（外には次の変わり目があるかもしれない）
        mid = a0 + int(np.argmin(np.abs(al[a0 - lo : b0 - lo + 1] - 0.5)))
        st = mid
        while st > lo and al[st - lo] > delta:
            st -= 1
        en = mid
        while en < hi and al[en - lo] < 1 - delta:
            en += 1
        if en - st < 2 or any(not (en < f["st"] or st > f["en"]) for f in found):
            continue
        seg_route = float(path[en] - path[st])
        seg_chord = float(np.sqrt(((x[en] - x[st]) ** 2).mean()))
        found.append(
            {
                "st": st,
                "en": en,
                "change": round(seg_chord, 2),
                "straightness": round(seg_chord / seg_route, 3) if seg_route else 1.0,
                "alpha": [round(float(v), 3) for v in al[st - lo : en - lo + 1]],
            }
        )
    found.sort(key=lambda f: f["st"])
    return found


@command(
    92,
    "fades",
    SRC,
    OUT,
    JSON,
    RANGE,
    *_CUT_ARGS[:2],
    arg(
        "--min-change",
        type=float,
        default=30.0,
        help="変わり目とみなす変化の大きさの下限（小さなコマの画素の差の二乗平均平方根、0〜255。既定 30）",
    ),
    arg(
        "--straight",
        type=float,
        default=0.25,
        help="候補にするまっすぐさの下限（両端の差 ÷ 1 コマずつの差の合計。既定 0.25。動く画どうしの長いディゾルブは"
        " 0.3 前後になる）。候補は、フェードならまっすぐさ 0.7 以上、ディゾルブなら途中のばらつきのくぼみで確かめる",
    ),
    arg(
        "--min-len",
        type=float,
        default=0.1,
        help="変わり目の最短の長さ（秒。これより短い物はカット扱い。既定 0.1）",
    ),
    arg(
        "--max-len",
        type=float,
        default=4.0,
        help="変わり目の最長の長さ（秒。既定 4）",
    ),
    arg(
        "--flash-max",
        type=float,
        default=0.4,
        help="白・黒への変化がこの秒数より短く、カットか反対向きの変化と続く物は、フェードでなく「フラッシュ」とする（既定 0.4）",
    ),
    arg(
        "--solid-std",
        type=float,
        default=8.0,
        help="一色の画面（黒・白など）とみなす明るさのばらつきの上限（0〜255。既定 8）",
    ),
)
def fades(a: argparse.Namespace) -> None:
    """フェードイン・アウト（黒・白へ）とディゾルブ（重ねて溶かすつなぎ目）を見つけ、時刻を JSON と確認画像に書く。"""
    rng = _rng(a)
    res = detect_cuts(a.src, method=a.method, threshold=a.threshold, **rng)
    fps, n = res.fps, res.n
    x = np.stack([s.astype(np.float32).ravel() for s in res.small])
    bounds = [0, *res.cuts, n]
    found = _gradual(
        x,
        bounds,
        min_change=a.min_change,
        min_straight=a.straight,
        min_len=max(2, round(a.min_len * fps)),
        max_len=max(3, round(a.max_len * fps)),
        delta=0.03,
    )
    cut_set = set(res.cuts)
    sd = np.array([float(_luma(s).std()) for s in res.small])
    items: list[dict[str, Any]] = []
    dropped = 0
    for f in found:
        st, en = f["st"], f["en"]
        ka = _solid_kind(res.small[st], a.solid_std)
        kb = _solid_kind(res.small[en], a.solid_std)
        if ka or kb:
            # 一色の画面が片方の端: フェード。動きが混ざっていても、ほぼまっすぐのはず
            if (ka and kb and ka == kb) or f["straightness"] < 0.7:
                dropped += 1
                continue
            kind, color = (
                ("fade", f"{ka}→{kb}")
                if ka and kb
                else ("fade_out", kb)
                if kb
                else ("fade_in", ka)
            )
        else:
            # 両端とも画: 別の画どうしで（模様が似ていない）、途中のコマのばらつきがくぼむ物だけを
            # ディゾルブとする。2 つの画を混ぜると、ばらつきは両端を直線で結んだ値より小さくなる。
            # 人や物の動きでは、ばらつきはほとんど変わらない（実測: ディゾルブ 0.69〜0.76、動き 1.0）
            al = np.clip(np.array(f["alpha"]), 0, 1)
            lin = (1 - al) * sd[st] + al * sd[en]
            dip = (
                float((sd[st : en + 1][1:-1] / np.maximum(lin[1:-1], 1e-6)).min())
                if en - st > 1
                else 1.0
            )
            f["dip"] = round(dip, 3)
            if _ncc(x[st], x[en]) >= 0.85 or dip > 0.9:
                dropped += 1
                continue
            kind, color = "dissolve", ""
        items.append(
            {
                "type": kind,
                "color": color,
                "start": res.t(st),
                "end": res.t(en),
                "duration": round((en - st) / fps, 4),
                "frames": [res.frame(st), res.frame(en)],
                "change": f["change"],
                "straightness": f["straightness"],
                **({"variance_dip": f["dip"]} if "dip" in f else {}),
                "st": st,
                "en": en,
            }
        )
    # フラッシュ（カットと同時の白い光が素早く引く、短い白・黒への出入り）はフェードと分ける
    flash_max = a.flash_max
    for k, it in enumerate(items):
        short = it["duration"] <= flash_max and it["color"] in ("白", "黒")
        if not short:
            continue
        if it["type"] == "fade_in" and (it["st"] in cut_set or it["st"] == 0):
            it["type"] = "flash"
            it["note"] = f"カットと同時に{it['color']}から素早く戻る"
        elif it["type"] == "fade_out" and (it["en"] + 1 in cut_set):
            it["type"] = "flash"
            it["note"] = f"{it['color']}へ素早く変わってカット"
        elif it["type"] == "fade_out" and k + 1 < len(items):
            nx = items[k + 1]
            if (
                nx["type"] == "fade_in"
                and nx["color"] == it["color"]
                and (nx["en"] - it["st"]) / fps <= flash_max * 2
            ):
                it["type"] = nx["type"] = "flash"
                it["note"] = nx["note"] = f"一瞬だけ{it['color']}くなる"
    for s, e in res.flashes:
        # 同じ場面に戻る一瞬の変化（76 の検出で「フラッシュ」とした所）。明るくなる物だけを光とする
        kind = _solid_kind(res.small[s], a.solid_std)
        brighter = float(
            _luma(res.small[s]).mean() - _luma(res.small[max(s - 1, 0)]).mean()
        )
        if not kind and brighter >= 40:
            kind = "明るい光"
        items.append(
            {
                "type": "flash" if kind else "insert",
                "color": kind or "",
                "start": res.t(s),
                "end": res.t(e + 1),
                "duration": round((e + 1 - s) / fps, 4),
                "frames": [res.frame(s), res.frame(e)],
                "note": "同じ場面の一瞬の光（前後は同じ画）"
                if kind
                else "一瞬だけ別の画が入る（前後は同じ画）",
                "st": s,
                "en": e,
            }
        )
    items.sort(key=lambda it: it["st"])
    names = {
        "fade_in": "フェードイン",
        "fade_out": "フェードアウト",
        "fade": "フェード（単色どうし）",
        "dissolve": "ディゾルブ",
        "flash": "フラッシュ",
        "insert": "一瞬の別の画",
    }
    for it in items:
        it["name"] = names[it["type"]] + (f"（{it['color']}）" if it["color"] else "")
    counts = {k: sum(1 for it in items if it["type"] == k) for k in names}
    out = Path(a.out) if a.out else core.default_out(a.src, "fades", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": fps,
            "frames": n,
            "params": {
                "min_change": a.min_change,
                "straight": a.straight,
                "min_len": a.min_len,
                "max_len": a.max_len,
                "flash_max": flash_max,
                "solid_std": a.solid_std,
                **res.params,
            },
            "counts": counts,
            "rejected_candidates": dropped,
            "cuts": [res.t(c) for c in res.cuts],
            "transitions": [
                {k: v for k, v in it.items() if k not in ("st", "en")} for it in items
            ],
        },
    )
    core.log(
        "見つけた変わり目: "
        + "、".join(f"{names[k]} {v}" for k, v in counts.items() if v)
        + f"（カット {len(res.cuts)} は別）"
        if items
        else f"溶けるつなぎ目は見つかりませんでした（カット {len(res.cuts)}）"
    )
    for it in items:
        core.log(
            f"  {it['start']:7.3f}〜{it['end']:7.3f} 秒（{it['duration']:.2f} 秒）: {it['name']}"
        )
    # 確認画像: 明るさのグラフと、変わり目ごとに「始め・中・終わり」
    t = np.array([res.t(k) for k in range(n)])
    mean_y = np.array([float(_luma(s).mean()) for s in res.small])
    step = np.zeros(n)
    step[1:] = np.sqrt(((x[1:] - x[:-1]) ** 2).mean(1))
    colors = {
        "fade_in": _GREEN,
        "fade_out": _BLUE,
        "fade": _BLUE,
        "dissolve": _PURPLE,
        "flash": _ORANGE,
        "insert": _GRAY,
    }
    chart = _chart(
        t,
        [(mean_y, (40, 40, 40), "画面の明るさ"), (step, _RED, "1 コマの変化")],
        title="明るさと変化（緑 = フェードイン、青 = フェードアウト、紫 = ディゾルブ、橙 = フラッシュ、灰の縦線 = カット）",
        ylim=(0, 255),
        spans=[
            (it["start"], it["end"] + 0.5 / fps, colors[it["type"]]) for it in items
        ],
        marks=[(res.t(c), _GRAY) for c in res.cuts],
    )
    parts = [chart]
    shown = items[:10]
    if shown:
        want = set()
        for it in shown:
            want |= {it["st"], (it["st"] + it["en"]) // 2, it["en"]}
        got = _pick_frames(a.src, rng, want, max_side=240)
        imgs, labels = [], []
        for it in shown:
            for k, name in (
                (it["st"], "始め"),
                ((it["st"] + it["en"]) // 2, "中"),
                (it["en"], "終わり"),
            ):
                if k in got:
                    imgs.append(got[k])
                    labels.append(f"{name} {res.t(k):.2f}秒")
        parts.append(core.contact_sheet(imgs, labels, cols=9, cell_width=155))
        names_list = [
            f"{k + 1}) {it['name']} {it['start']:.2f}〜{it['end']:.2f}秒"
            for k, it in enumerate(shown)
        ]
        lines = ["  ".join(names_list[k : k + 4]) for k in range(0, len(names_list), 4)]
        bar = np.full((14 + 26 * len(lines), 1400, 3), 255, np.uint8)
        for k, line in enumerate(lines):
            core.draw_text(
                bar, line, (10, 8 + 26 * k), size=15, color=(20, 20, 20), bg=None
            )
        parts.insert(1, bar)
    core.save_image(out, _stack(parts))
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 93 ピンぼけ検出 ───────────────────────────────────────────────────────

_BLURDETECT = re.compile(r"lavfi\.blur=([\d.]+)")


def _ffmpeg_blur(src: str) -> float | None:
    """ffmpeg の blurdetect（輪郭の幅で測るぼけの量。大きいほどぼけている）。画像 1 枚用。"""
    text = core.ffmpeg_log(
        ["-i", str(src), "-vf", "blurdetect,metadata=print", "-f", "null", "-"]
    )
    vals = [float(v) for v in _BLURDETECT.findall(text)]
    return round(float(np.mean(vals)), 3) if vals else None


@command(
    93,
    "blur-detect",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    *_CUT_ARGS[:2],
    arg(
        "--ratio",
        type=float,
        default=0.6,
        help="同じ場面の前後のコマ（鮮明な方）と比べて、鮮明さがこの割合より低いコマをぼけとする（既定 0.6）",
    ),
    arg(
        "--window",
        type=float,
        default=1.5,
        help="比べる前後の範囲（秒。既定 1.5）",
    ),
    arg(
        "--remove",
        metavar="MP4",
        help="ぼけたコマを抜いた動画の書き出し先（音声は付けない。コマを抜くと音とずれるため）",
    ),
)
def blur_detect(a: argparse.Namespace) -> None:
    """コマごとの鮮明さ（ラプラシアンの分散）を測り、同じ場面の中で相対的にぼけたコマを見つける。画像なら鮮明さの地図を作る。"""
    info = core.probe(a.src)
    if info.is_image:
        _blur_image(a)
        return
    rng = _rng(a)
    res = detect_cuts(a.src, method=a.method, threshold=a.threshold, **rng)
    sharp = _sharpness_series(a.src, rng, a.max_side)
    fps = res.fps
    n = min(len(sharp), res.n)
    sharp = sharp[:n]
    half = max(1, round(a.window * fps))
    ref = np.zeros(n)
    for s, e in res.scenes():
        e = min(e, n - 1)
        for i in range(s, e + 1):
            lo, hi = max(s, i - half), min(e, i + half)
            others = np.concatenate([sharp[lo:i], sharp[i + 1 : hi + 1]])
            # 前後のコマのうち鮮明な方（上位 25%）を基準にする。数コマぼけていても基準は下がらない
            ref[i] = np.percentile(others, 75) if len(others) else sharp[i]
    rel = sharp / np.maximum(ref, 1e-6)
    blurry = rel < a.ratio
    idx = np.nonzero(blurry)[0]
    ranges = []
    for i in idx:
        if ranges and i == ranges[-1][1] + 1:
            ranges[-1][1] = int(i)
        else:
            ranges.append([int(i), int(i)])
    out = Path(a.out) if a.out else core.default_out(a.src, "blur-detect", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": fps,
            "frames": n,
            "params": {
                "ratio": a.ratio,
                "window": a.window,
                "analysis_max_side": a.max_side or 960,
                **res.params,
            },
            "blurry_frames": len(idx),
            "ranges": [
                {
                    "start": res.t(s),
                    "end": res.t(e + 1),
                    "frames": [res.frame(s), res.frame(e)],
                    "min_relative": round(float(rel[s : e + 1].min()), 3),
                }
                for s, e in ranges
            ],
            "per_frame": [
                {
                    "frame": res.frame(i),
                    "time": res.t(i),
                    "sharpness": round(float(sharp[i]), 3),
                    "relative": round(float(rel[i]), 3),
                }
                for i in range(n)
            ],
        },
    )
    core.log(f"ぼけたコマ: {len(idx)} / {n} コマ（{len(ranges)} か所）")
    for s, e in ranges[:20]:
        core.log(
            f"  {res.t(s):7.3f}〜{res.t(e + 1):7.3f} 秒（鮮明さ 前後の {rel[s : e + 1].min() * 100:.0f}%）"
        )
    if len(ranges) > 20:
        core.log(f"  …ほか {len(ranges) - 20} か所（JSON を参照）")
    if a.remove:
        drop = {int(i) for i in idx}
        with core.VideoWriter(a.remove, fps) as w:
            for i, f in enumerate(core.read_frames(a.src, **rng)):
                if i not in drop:
                    w.write(f)
        core.log(f"ぼけたコマを抜いた動画（{n - len(drop)} コマ。音声なし）")
        core.log(f"保存: {a.remove}")
    # 確認画像: 鮮明さのグラフと、ぼけたコマ（と同じ場面の基準になる鮮明なコマ）
    t = np.array([res.t(k) for k in range(n)])
    chart = _chart(
        t,
        [(sharp, _BLUE, "鮮明さ"), (ref * a.ratio, _RED, f"基準 × {a.ratio:g}")],
        title=f"コマごとの鮮明さ（赤い点 = ぼけ {len(idx)} コマ、灰の縦線 = カット）",
        marks=[(res.t(c), _GRAY) for c in res.cuts],
        dots=[(t[idx], sharp[idx], _RED)] if len(idx) else [],
    )
    parts = [chart]
    shown = [s for s, _ in ranges[:8]]
    if shown:
        pairs = []
        for s in shown:
            scene = next((sc for sc in res.scenes() if sc[0] <= s <= sc[1]), (0, n - 1))
            lo, hi = max(scene[0], s - half), min(scene[1], s + half)
            best = lo + int(np.argmax(sharp[lo : hi + 1]))
            pairs.append((s, best))
        got = _pick_frames(a.src, rng, {k for p in pairs for k in p}, max_side=320)
        imgs, labels = [], []
        for s, best in pairs:
            if s in got and best in got:
                imgs += [got[s], got[best]]
                labels += [
                    f"ぼけ {res.t(s):.2f}秒 {rel[s] * 100:.0f}%",
                    f"比べたコマ {res.t(best):.2f}秒",
                ]
        if imgs:
            parts.append(core.contact_sheet(imgs, labels, cols=8, cell_width=175))
    core.save_image(out, _stack(parts))
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


def _blur_image(a: argparse.Namespace) -> None:
    """画像 1 枚: 鮮明さの数値と、区画ごとの鮮明さの地図（赤いほど鮮明）。"""
    img = next(core.read_frames(a.src, max_side=a.max_side or 960))
    gray = core.to_gray(img)
    score, var = _sharpness(gray)
    width = _ffmpeg_blur(a.src)
    tiles = np.sqrt(var)
    # 色の目盛りは画像ごとに変えない（ぼけた画像全体が青く見えるように。80 はくっきりした写真の目安）
    heat = core.heatmap(
        cv2.resize(tiles.astype(np.float32), (img.shape[1], img.shape[0]),
                   interpolation=cv2.INTER_NEAREST),
        lo=0.0,
        hi=max(80.0, float(tiles.max())),
    )  # fmt: skip
    over = core.overlay(img, heat, 0.45)
    out = Path(a.out) if a.out else core.default_out(a.src, "blur-detect", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "analysis_size": [img.shape[1], img.shape[0]],
            "sharpness": round(score, 3),
            "sharpness_global": round(
                float(np.sqrt(cv2.Laplacian(gray, cv2.CV_32F).var())), 3
            ),
            "ffmpeg_blurdetect": width,
            "tiles": np.round(tiles, 2),
            "note": "鮮明さは写っている物（模様の多さ）でも変わるので、画像どうしの比較は同じ場面・同じ大きさで行う",
        },
    )
    core.draw_text(
        over, f"鮮明さ {score:.1f}" + (f" / blurdetect {width:.2f}" if width else ""),
        (8, img.shape[0] - 8), size=max(16, img.shape[1] // 28), anchor="lb",
    )  # fmt: skip
    core.save_image(
        out,
        core.side_by_side(
            img, over, ("元", "鮮明さの地図（赤いほど鮮明・青いほどぼけ）")
        ),
    )
    core.log(
        f"鮮明さ: {score:.2f}（区画の上位 10%）"
        + (f"、ffmpeg blurdetect: {width:.3f}（大きいほどぼけ）" if width else "")
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 99 フラッシュ検出 ─────────────────────────────────────────────────────

# sRGB（0〜255）→ 光の強さに比例する値（0〜1）。WCAG の相対輝度の定義（IEC 61966-2-1 の式）
_LINEAR = np.array(
    [
        c / 255 / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4
        for c in range(256)
    ],
    np.float32,
)


class _Transitions:
    """区画ごとに「反対向きの変化」を数える（値の山と谷を追い、しきい値以上の上下を 1 回とする）。

    dark が数の時: 明るさ用。暗い側が dark 未満の変化だけ数える（WCAG: 相対輝度 0.8 未満）。
    dark が None の時: 赤用。変化の前後のどちらかが「飽和した赤」（sat が真）の時だけ数える。
    """

    def __init__(self, thr: float, dark: float | None) -> None:
        self.thr = thr
        self.dark = dark
        self.dir: np.ndarray | None = None

    def push(self, v: np.ndarray, sat: np.ndarray | None = None) -> np.ndarray:
        if self.dir is None:
            self.dir = np.zeros(v.shape, np.int8)
            self.ext, self.lo, self.hi = v.copy(), v.copy(), v.copy()
            s = sat if sat is not None else np.zeros(v.shape, bool)
            self.ext_sat, self.lo_sat, self.hi_sat = s.copy(), s.copy(), s.copy()
            return np.zeros(v.shape, bool)
        thr = self.thr
        sat = sat if sat is not None else np.zeros(v.shape, bool)
        d0, rising, falling = self.dir == 0, self.dir == 1, self.dir == -1
        # まだ向きが決まっていない区画: 最初からの最小・最大との差で見る
        new_lo, new_hi = d0 & (v < self.lo), d0 & (v > self.hi)
        self.lo = np.where(new_lo, v, self.lo)
        self.lo_sat = np.where(new_lo, sat, self.lo_sat)
        self.hi = np.where(new_hi, v, self.hi)
        self.hi_sat = np.where(new_hi, sat, self.hi_sat)
        up0 = d0 & (v - self.lo >= thr)
        dn0 = d0 & ~up0 & (self.hi - v >= thr)
        # 上がっている区画: 山を更新し、山から thr 以上下がったら折り返し
        grow = (rising & (v > self.ext)) | (falling & (v < self.ext))
        self.ext = np.where(grow, v, self.ext)
        self.ext_sat = np.where(grow, sat, self.ext_sat)
        rev_dn = rising & (self.ext - v >= thr)
        rev_up = falling & (v - self.ext >= thr)
        if self.dark is not None:
            ok_up0 = self.lo < self.dark
            ok_dn = v < self.dark
            ok_rup = self.ext < self.dark
        else:
            ok_up0 = self.lo_sat | sat
            ok_dn = (np.where(d0, self.hi_sat, self.ext_sat)) | sat
            ok_rup = self.ext_sat | sat
        trans = (up0 & ok_up0) | (dn0 & ok_dn) | (rev_dn & ok_dn) | (rev_up & ok_rup)
        go_up, go_dn = up0 | rev_up, dn0 | rev_dn
        self.dir[go_up] = 1
        self.dir[go_dn] = -1
        moved = go_up | go_dn
        self.ext = np.where(moved, v, self.ext)
        self.ext_sat = np.where(moved, sat, self.ext_sat)
        return trans


@command(
    99,
    "flashes",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--area",
        type=float,
        default=0.25,
        help="点滅とみなす面積の下限（画面に対する割合。既定 0.25 = 25%%）",
    ),
    arg(
        "--max-flashes",
        type=float,
        default=3.0,
        help="1 秒間に許す点滅の回数（これを超えると要注意。既定 3）",
    ),
    arg(
        "--cells",
        type=int,
        default=64,
        help="画面を区切る数（長い辺の方向。既定 64）",
    ),
)
def flashes(a: argparse.Namespace) -> None:
    """光過敏性発作の目安（WCAG 2.3.1 の一般の点滅と赤の点滅）で、強い点滅が続く区間を見つけ、JSON とグラフに書く。"""
    info = _video(a.src)
    rng = _rng(a)
    fps = info.fps
    start = rng["start"]
    win = max(2, round(fps))  # 1 秒のコマ数
    gen = _Transitions(0.1, 0.8)  # 相対輝度の 10% 以上、暗い側が 0.8 未満
    red = _Transitions(
        20.0, None
    )  # (R-G-B)×320 の変化が 20 より大きく、どちらかが飽和した赤
    hist_g: deque[np.ndarray] = deque()
    hist_r: deque[np.ndarray] = deque()
    cnt_g = cnt_r = None
    rate_g, rate_r, area_g, area_r, mean_l = [], [], [], [], []
    flip_g, flip_r = [], []  # そのコマで変化した区画の割合
    grid = None
    for f in core.read_frames(a.src, max_side=a.max_side or 320, **rng):
        lin = _LINEAR[f]
        if grid is None:
            h, w = lin.shape[:2]
            if h >= w:
                grid = (max(2, round(a.cells * w / h)), a.cells)
            else:
                grid = (a.cells, max(2, round(a.cells * h / w)))
        cells = cv2.resize(
            lin, grid, interpolation=cv2.INTER_AREA
        )  # 光の強さを平均する
        b, g, r = cells[..., 0], cells[..., 1], cells[..., 2]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        total = r + g + b
        sat = (total > 1e-6) & (r >= 0.8 * total)
        q = np.maximum(r - g - b, 0) * 320
        tg = gen.push(lum.ravel())
        tr = red.push(q.ravel(), sat.ravel())
        if cnt_g is None:
            cnt_g = np.zeros(tg.shape, np.int32)
            cnt_r = np.zeros(tr.shape, np.int32)
        hist_g.append(tg)
        hist_r.append(tr)
        cnt_g += tg
        cnt_r += tr
        if len(hist_g) > win:
            cnt_g -= hist_g.popleft()
            cnt_r -= hist_r.popleft()
        k = max(1, math.ceil(a.area * cnt_g.size))
        # 「画面の area 以上の面積が、1 秒間に何回点滅したか」= 区画の回数の上位 k 番目
        top_g = float(np.partition(cnt_g, cnt_g.size - k)[cnt_g.size - k])
        top_r = float(np.partition(cnt_r, cnt_r.size - k)[cnt_r.size - k])
        rate_g.append(top_g / 2)  # 反対向きの変化 2 回で 1 回の点滅
        rate_r.append(top_r / 2)
        limit = 2 * a.max_flashes
        area_g.append(float((cnt_g > limit).mean()))
        area_r.append(float((cnt_r > limit).mean()))
        flip_g.append(float(tg.mean()))
        flip_r.append(float(tr.mean()))
        mean_l.append(float(lum.mean()))
    n = len(rate_g)
    rg, rr = np.array(rate_g), np.array(rate_r)

    def intervals(rate: np.ndarray, flip: list[float]) -> list[dict[str, Any]]:
        # 超えた 1 秒の窓をつなぎ、実際に点滅している所（広い面積が変化したコマ）まで縮める
        bad = np.nonzero(rate > a.max_flashes)[0]
        spans: list[list[int]] = []
        for i in bad:
            s = max(0, i - win + 1)
            if spans and s <= spans[-1][1] + 1:
                spans[-1][1] = int(i)
            else:
                spans.append([int(s), int(i)])
        found = []
        for s, e in spans:
            busy = [k for k in range(s, e + 1) if flip[k] >= 0.5 * a.area]
            s2, e2 = (busy[0], busy[-1]) if busy else (s, e)
            found.append(
                {
                    "start": core.t_of(s2, fps, start),
                    "end": core.t_of(e2 + 1, fps, start),
                    "max_flashes_per_sec": float(rate[s : e + 1].max()),
                }
            )
        return found

    bad_g, bad_r = intervals(rg, flip_g), intervals(rr, flip_r)
    out = Path(a.out) if a.out else core.default_out(a.src, "flashes", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    t = np.array([core.t_of(k, fps, start) for k in range(n)])
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": fps,
            "frames": n,
            "params": {
                "area": a.area,
                "max_flashes": a.max_flashes,
                "cells": list(grid or (0, 0)),
                "general": "相対輝度の 10% 以上の上下を 1 回の変化とし、暗い側が 0.8 未満。2 回の変化で 1 回の点滅",
                "red": "飽和した赤（R/(R+G+B) ≥ 0.8）を含み、(R-G-B)×320 が 20 より大きく変わる上下（WCAG 2.0/2.1 の注の式）",
            },
            "pass": not bad_g and not bad_r,
            "general": {
                "max_flashes_per_sec": float(rg.max()) if n else 0.0,
                "at": float(t[int(np.argmax(rg))]) if n else 0.0,
                "intervals": bad_g,
            },
            "red": {
                "max_flashes_per_sec": float(rr.max()) if n else 0.0,
                "at": float(t[int(np.argmax(rr))]) if n else 0.0,
                "intervals": bad_r,
            },
            "per_frame": {
                "time": t,
                "general_flashes_per_sec": rg,
                "red_flashes_per_sec": rr,
                "mean_luminance": np.round(np.array(mean_l), 4),
            },
        },
    )
    verdict = "要注意" if bad_g or bad_r else "目安の範囲内"
    core.log(
        f"判定: {verdict}（画面の {a.area * 100:.0f}% 以上の面積で、1 秒に {a.max_flashes:g} 回を超える点滅）"
    )
    core.log(
        f"  明るさの点滅: 最大 {rg.max():.1f} 回/秒（{t[int(np.argmax(rg))]:.2f} 秒）、"
        f"超えた区間 {len(bad_g)} 件"
    )
    for it in bad_g:
        core.log(
            f"    {it['start']:.2f}〜{it['end']:.2f} 秒（最大 {it['max_flashes_per_sec']:.1f} 回/秒）"
        )
    core.log(
        f"  赤の点滅: 最大 {rr.max():.1f} 回/秒（{t[int(np.argmax(rr))]:.2f} 秒）、超えた区間 {len(bad_r)} 件"
    )
    for it in bad_r:
        core.log(
            f"    {it['start']:.2f}〜{it['end']:.2f} 秒（最大 {it['max_flashes_per_sec']:.1f} 回/秒）"
        )
    spans = [(it["start"], it["end"], _RED) for it in bad_g] + [
        (it["start"], it["end"], _PURPLE) for it in bad_r
    ]
    c1 = _chart(
        t,
        [(np.array(mean_l), (40, 40, 40), "画面の平均の相対輝度")],
        title="画面の明るさ（相対輝度 0〜1）",
        ylim=(0, 1),
        height=220,
    )
    c2 = _chart(
        t,
        [(rg, _RED, "明るさの点滅"), (rr, _PURPLE, "赤の点滅")],
        title=f"直前 1 秒の点滅の回数（画面の {a.area * 100:.0f}% 以上の面積で）。塗った所 = 超えた区間",
        hlines=[(a.max_flashes, _ORANGE, f"目安 {a.max_flashes:g} 回")],
        spans=spans,
        ylim=(0, max(a.max_flashes * 2, float(max(rg.max(), rr.max())) * 1.1, 1.0)),
    )
    c3 = _chart(
        t,
        [(np.array(area_g), _RED, "明るさ"), (np.array(area_r), _PURPLE, "赤")],
        title=f"1 秒に {a.max_flashes:g} 回を超えて点滅している面積の割合",
        hlines=[(a.area, _ORANGE, f"{a.area * 100:.0f}%")],
        ylim=(0, 1),
        height=220,
    )
    parts = [c1, c2, c3]
    # 並べるコマ: 超えた区間があればその始まりの 4 コマずつ、無ければ広い面積が変化したコマの上位 8 つ
    flip = np.maximum(np.array(flip_g), np.array(flip_r))
    picks: set[int] = set()
    for it in (bad_g + bad_r)[:2]:
        s = round((it["start"] - start) * fps)
        picks |= {min(n - 1, s + k) for k in range(4)}
    if not picks:
        for k in np.argsort(-flip, kind="stable"):
            if flip[k] <= 0 or len(picks) >= 8:
                break
            if all(abs(int(k) - p) >= 3 for p in picks):
                picks.add(int(k))
    got = _pick_frames(a.src, rng, picks, max_side=240)
    if got:
        ks = sorted(got)
        parts.append(
            core.contact_sheet(
                [got[k] for k in ks],
                [
                    f"{core.t_of(k, fps, start):.3f}秒 変化{flip[k] * 100:.0f}%"
                    for k in ks
                ],
                cols=8,
                cell_width=175,
            )
        )
    core.save_image(out, _stack(parts))
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")
