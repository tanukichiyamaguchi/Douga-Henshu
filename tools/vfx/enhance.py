"""画質を上げる（解像度・コマ・鮮明さ・粒子）。

担当するテクニック:
   26 超解像（upscale）
   27 フレーム補間（interpolate）
   29 ブレ除去（deblur）
   39 アンシャープマスク（sharpen）
   50 フィルムグレイン（grain）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
どのコマンドも、処理した映像（動画は MP4、画像は画像）と、数値の JSON、前後を並べた確認画像
（<出力>-compare.jpg）を書く。
"""

from __future__ import annotations

import argparse
import itertools
import math
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np

from . import core, models
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/enhance/（test_*.py が正解つきの試験、calib_*.py が設定の較正）。
# 速さは他の作業と CPU を分け合った状態の壁時計（測った時の負荷を書いた）。PSNR は明るさ（Y）。
NOTES: dict[int, dict[str, str]] = {
    26: {
        "status": "◎",
        "how": "auto（既定）: 3 倍以上は AI、それ未満は lanczos（試験の結果で決めた）。AI は Real-ESRGAN の realesr-general-x4v3 と"
        "ノイズを残す版 realesr-general-wdn-x4v3（どちらも BSD-3-Clause、GitHub のリリース）の重みを --denoise の割合で混ぜ"
        "（公式の inference_realesrgan.py の dni と同じ。既定 0 = 残す版だけ）、spandrel 0.4.2 で読む（RealESRGAN Compact と"
        "認識）。256px の区画（すべて同じ大きさ。重なりは 12px を直線の重みでなじませる）で 4 倍にし、仕上がりの大きさへ面積平均"
        "（INTER_AREA）で縮める（4 倍を超える分は lanczos）。CPU が AMX / AVX-512 BF16 に対応していれば bf16 で計算する"
        "（fp32 との差 54 dB・最大 5 段）。--width・--height・--scale で大きさ。透過つき（WebM・PNG）は透過を lanczos で拡大して"
        "残す。動画は色の行列と範囲を元に合わせて正確に変換し（core の既定の変換は平均 0.5 段暗くなる）、音声と色の印を残す",
        "measured": "正解つき（story540 の 5 コマ・building・fruits を INTER_AREA で縮めて戻す、元と比べる）: 4 倍・きれい: lanczos "
        "PSNR 35.14 dB・LPIPS 0.220（小さいほど見た目が近い）・勾配の比 0.73、AI 33.28 dB・LPIPS 0.181・0.90（AI はくっきりして"
        "見た目が近いが、画素は少しずれる）。4 倍・JPEG 品質 75: lanczos 0.375 → AI 0.249。H.264 crf 28 で圧縮した物: 2 倍 lanczos "
        "0.266・AI 0.253、3 倍 0.377・0.302、4 倍 0.456・0.347。きれいな 2 倍は lanczos 42.86 dB・LPIPS 0.043、AI 37.66 dB・0.098"
        "（AI が肌のきめをならし、建物の線を濃くして元から離れる）、きれいな 3 倍は lanczos 0.126・AI 0.135 → auto の境目を 3 倍に"
        "した。--denoise は 0・0.5・1 のうち、どの倍率・劣化でも 0 が最も元に近かった（4 倍 JPEG だけ 0.5 と同等）。"
        "動画: 固定カメラの vtest を 1/2 に縮めて戻すと、動かない背景のコマごとのちらつきが元 0.233・lanczos 0.189・AI 0.338"
        "（AI は 1.45 倍）。透過つきの切り抜き（WebM）を 2 倍: 透明な画素 49.56% → 49.55%。速さ（AI の 4 倍の部分、bf16、"
        "区画 256、負荷 4〜5）: 入力 270×480 で 0.15 秒（分けない）・540×960 で 1.0 秒・720×1280 で 1.5 秒/コマ。fp32 は 3〜4 倍"
        "遅い。コマンド全体（270×480 → 1080×1920・24 コマ、別の重い試験と同時）で 2.3 秒/コマ",
        "limits": "AI は細部を描き足すので、元に無い模様（髪の筋・肌のなめらかさ）になる。きれいな素材の 2〜3 倍では lanczos の方が"
        "元に近い（auto は lanczos を選ぶ。圧縮の傷が目立つ素材なら --method ai で 2 倍でも少し良くなった）。動画では、動かない"
        "背景が少しちらつく（上の 1.45 倍。気になる時は 31 時間方向のノイズ除去を後にかける）。顔は 34（顔の復元）ではないので、"
        "小さな顔の目鼻は作り直さない。4 倍を超える分は lanczos。CPU なので遅い（1080×1920 の入力は AI の部分だけで "
        "3.7 秒/コマ。短く切って試す）。元に色の印が無い SD の映像を HD に拡大すると、再生側が色の行列を推測で変えることがある（ffmpeg で"
        "拡大した時と同じ）",
        "tested": "test_upscale.py（images: 縮めて戻す正解つき 7 枚 × 2・4 倍 × きれい・JPEG、LPIPS は uv run --with lpips で一時的に"
        "足して測った。crossover: 2・3・4 倍 × きれい・H.264 crf 28。video: vtest のちらつき）、speed_upscale.py（区画と精度）、"
        "確認画像 26-upscale-truth.jpg（正解・lanczos・AI 3 種の拡大）・26-upscale-video-x4.jpg（270×480 の動画を 4 倍）。"
        "透過つきの cut-bath-540.webm を AI で 2 倍にして透過を確かめた",
    },
    27: {
        "status": "○",
        "how": "ffmpeg の minterpolate（mi_mode=mci・mc_mode=aobmc・me_mode=bidir・vsbmc=1）。最後のコマを tpad で 2 コマ分のばして"
        "から補間し、コマ数で切る（CLAUDE.md のスローモーションの作り方。末尾のコマ欠けを防ぐ）。--slow は setpts で長さを"
        "のばしてから補間する（音声は付けない）。つなぎ目の扱い: 前後のコマの差（幅 96px の灰色）が 10 より大きく、まわり ±12 "
        "コマの中央値の 5 倍より大きい所（カット・白いフラッシュ・急な変化）で区切り、区切りごとに補間して -c copy でつなぐ"
        "（区切りでは中間のコマを作らず、前のコマをのばす）。minterpolate 自身の場面転換の判定（scd）も既定の 10 から 5 に"
        "下げた。blend・dup も minterpolate の同名の方法（コマの時刻の扱いをそろえるため）",
        "measured": "正解つき（24fps の映像の 1 コマおきを抜いて 12fps にし、24fps に戻して抜いたコマと比べる。vtest は 10→5→10fps）: "
        "作ったコマの PSNR（mci / blend / dup）: まつげの施術 40.29 / 39.20 / 36.36、目が開く 41.59 / 36.72 / 34.67、眉の施術"
        "（速い手）34.14 / 33.15 / 30.93、カフェで笑う 34.78 / 29.41 / 26.36、人が歩く vtest 31.94 / 29.43 / 27.00 dB。元のコマは"
        "そのまま出る（46.6〜50.3 dB。H.264 の分だけ）。つなぎ目: 直す前（minterpolate だけ）はカットで 2 つの場面が混ざった"
        "コマができた（story の 3.04 秒）。区切った後は story540 全体の 2 倍でカット 4 か所の混ざり 0（361 → 722 コマ）、v4 の "
        "2 倍で、別の規則で見つけたつなぎ目 23 か所（カット・白いフラッシュ）のうち混ざり 1（同じ場面で速く動くロゴの文字）"
        "（600 → 1200 コマ、区切り 22）。区切りの見つけ方: カットの無い 10 本（元の fps と半分の fps）で余分な区切り 1（眉の"
        "施術の速い手）。スローモーション 2.5 倍: 48 → 120 コマ・5.000 秒（ちょうど）、元のコマが 2.5 コマごとに 50.05 dB で出る。"
        "速さ: 540×960・24 コマを 2 倍で 8.4〜9.4 秒（作るコマ 1 枚 0.18 秒ほど）、story540 全体（361 コマ）の 2 倍で 92 秒",
        "limits": "簡易な方法（ブロック単位の動き補償）。動きの推定はブロック単位なので、速く動く物（手・指）や、同じ場面で速く動く"
        "文字・ロゴは形が崩れることがある（AI の補間はこの崩れが少ないとされるが、ここでは比べていない）。"
        "文字を入れた完成品ではなく、文字を入れる前の素材にかける。カットでは前のコマをのばすので、作ったコマ 1 枚分だけ"
        "切り替わりが遅れる。区切りの見つけ方は速い動きも区切ることがあり、そこは補間しない（少しカクつく）。--no-breaks で"
        "区切らない。AI の補間（RIFE など。Hugging Face に MIT の重みもある）は spandrel で読めず、モデルの構造を書き写す必要が"
        "あるので入れていない。1080p は 540p の約 4 倍の時間がかかる",
        "tested": "test_interpolate.py（drop: 6 本の正解つき、slow: 長さとコマの位置、cuts: story540 全体と v4 全体のつなぎ目）、"
        "test_interp_scd.py（scd のしきい値 10・7・5・3 の比較）、calib_breaks.py（区切りの見つけ方の較正）、確認画像 "
        "27-interpolate-truth.jpg（各映像の最も動くコマの拡大）・27-interpolate-cut.jpg（カットで混ざったコマと直した後）・"
        "27-interpolate-v4-cuts.jpg・27-interpolate-slow.jpg",
    },
    29: {
        "status": "○",
        "how": "auto（既定）: ぶれの形を推定し、ピンぼけと、動画で画面全体の一様な動きのぶれは rl、それ以外（画像の手ぶれ・"
        "見分けられない）は ai。ai: NAFNet（GoPro の動きのぶれで学習、width64）。megvii-research の公式の重み（MIT）を"
        " Hugging Face の tog/nafnet-models（license: mit）から版を固定して取り、sha256 で照合する。spandrel 0.4.2 で読み、学習と同じ"
        " 256px の区画（重なり 32px をなじませる）で fp32 で推論（bf16 は速くならず、出力が崩れた）。wiener: 定数 K の古典的な"
        "ウィーナー、rl: リチャードソン・ルーシー（scikit-image 0.26）。どちらも明るさだけ（色はそのまま）、周りを折り返して広げ"
        "外側をなじませてから処理し、K = 0.01×σ^0.83・回数 = 45×σ^−0.75（σ は推定したノイズの強さ、最低 1。較正の最良値に"
        "合わせた式）。ぶれの形は --kernel と長さ・角度・半径・σ で指定するか、ケプストラム（256px の区画のハン窓の対数パワー"
        "スペクトルの平均を逆変換）から推定する: 動きのぶれは負の山の位置（長さ・角度。1 画素より細かく）、ピンぼけは半径"
        "ごとの平均のくぼみ。種類は較正した規則で見分ける（くぼみ ≥ 12 → ピンぼけ、山 ≥ 5・向きの違う 2 番目の山との比 ≥ 1.4"
        "・長さ ≥ 6px → 動き、それ以外は「なし」）。動画は 5 コマで推定し、多数決と中央値",
        "measured": "推定（8 枚に既知のぶれ、σ1 のノイズ、JPEG 品質 85 あり・なし）: ぶれ無し 16/16 を「なし」、動き（長さ 7〜31px・"
        "5 方向）79/80 を「動き」と判定し、長さの誤差の中央値 0.34px（90% 点 0.9px）・角度の誤差の中央値 0.3°（90% 点 2.1°）、"
        "ピンぼけ 半径 5・8・12 は 48/48（推定 4.68・7.75・11.46）、半径 3 は 5/16・2 は 0/16（「なし」）。OpenCV の実写の見本:"
        " licenseplate を動き 22.4px・134.9°（見本の値 22px・135°）、text_defocus をピンぼけ 半径 8.9px（見本の値 直径 19px）、"
        "text_motion は「なし」と判定（候補は 32.1px・86.8°。見本の値 31px・86°）。戻す（7 枚、σ1 のノイズ、推定した PSF、"
        "元との PSNR）: 直線の動き 15px で ぼけ 28.87 → wiener 33.33・rl 33.96・AI 32.20 dB、25px で 27.56 → 31.92・32.11・"
        "AI 33.28、ピンぼけ 半径 6 で 29.23 → 32.64・33.10（AI は 4 枚で 29.13 → 29.35）。推定した PSF は正しい PSF とほぼ同じ"
        "成績（15px: 33.33 対 33.32）。曲がった軌跡の合成の手ぶれ: 1 例は推定が「なし」で変えず（35.37）、もう 1 例は直線と"
        "推定した形でかえって悪化（30.15 → rl 29.00。正しい PSF なら 37.72）、AI も 4 枚で 34.94 → 30.35・30.01 → 27.41"
        "（位置を合わせても入力より悪い）。AI をぶれていない 4 枚にかけると元との差 34.45 dB（輪郭となめらかな所を作り変える）。"
        "H.264 の影響: 1 コマに 15px のぶれ（0・30・90°）を足し、圧縮なしなら AI 32.59 → 34.53・33.10 → 36.20・34.81 → 35.02、"
        "H.264 crf12 にすると 32.15 → 29.39・32.59 → 29.65・34.08 → 28.89（色の間引きだけなら 35.75 で、圧縮の傷が原因）。"
        "動画（story540 の 12 コマに横 15px のぶれを足して H.264 crf12）: 自動の推定 15.3px・180°、ぼけ 31.72 → wiener 33.78・"
        "rl 35.50、AI は 31.81 → 31.44（4 コマ）→ auto は動画の一様なぶれに rl を使う。実写の見本（目で確認）: 本物の手ぶれの"
        " text_motion・licenseplate は AI が最もきれいに戻し（数式・ナンバー 127-RFS が読める）、wiener・rl も読めるが縞が出た。"
        "ピンぼけの text_defocus は wiener・rl で読めるようになり、AI は戻せなかった。速さ: wiener 0.05〜0.09 秒/コマ・rl 1.0 秒/"
        "コマ（540×960）、AI 540×960 で 6.1 秒/コマ・600×482 の画像で 3.8 秒（負荷 5〜7）",
        "limits": "簡易な方法: wiener・rl は画面全体で同じぶれ（直線の手ぶれ・ピント外れ）を前提にしていて、被写体だけが動いたぶれ・"
        "場所で違うぶれ・曲がった軌跡の手ぶれには合わず、かえって悪くなることがある（上の合成の手ぶれ）。AI（NAFNet）は GoPro の"
        "動きのぶれで学習していて、ピンぼけは戻せず、ぶれていない・ぶれの弱い映像では輪郭やなめらかな所を作り変える（ぶれが"
        "見つからない時は注意を出す）。H.264 で圧縮した映像では、AI が圧縮の傷をぶれと取り違えて悪くなることがある（一様でない"
        "ぶれは AI しか無いので、確認画像で見る）。AI の前に軽くぼかすと合成の H.264 では良くなったが、実写の見本では効かなく"
        "なったので入れていない。wiener は動きの向きに沿った縞（波紋）が出やすい（rl の方が少ないが約 10 倍遅い）。半径 3px 以下の"
        "ピンぼけは見分けにくい。推定が「なし」の時は --kernel motion・defocus で候補の値を使える（text_motion はこれで戻った）。"
        "圧縮の強い素材はノイズの推定が 0 になるので、σ は最低 1 として扱う（結果がざらつく時は --strength を 0.5 などに下げる）。AI は CPU で遅い",
        "tested": "test_deblur.py（estimate: 推定の正解つき 192 例と実写の見本 3 枚、calib: 正しい PSF での較正、compare: 7 枚 × 直線 2・"
        "手ぶれ 2・ピンぼけの推定した PSF の wiener・rl と AI（AI の手ぶれ・ピンぼけは 4 枚）、identity: ぶれていない 4 枚に AI、"
        "real: 実写の見本 3 枚をコマンドで）、calib_wiener.py（定数 K の較正）、proto_blur_features.py（見分け方の較正）、"
        "test_deblur_shift.py（AI の位置を合わせて測り直す）、test_deblur_preblur.py（AI の前処理）、story540 に横 15px のぶれを足した"
        " H.264 の動画（wiener・rl・ai をコマンドで）、確認画像 29-deblur-real.jpg（実写の見本 × 3 方法）・29-deblur-truth.jpg・"
        "29-deblur-video.jpg・29-deblur-preblur.jpg",
    },
    39: {
        "status": "◎",
        "how": "usm（既定）: 明るさ（動画は Y の面、画像は YCrCb の Y）だけにアンシャープマスク（OpenCV。ガウスの半径 σ 1.0・強さ 1.2）。"
        "差（元 − ぼかし）のうち下限（推定したノイズの強さの 2 倍、最低 1 段）より小さい物を滑らかに弱め（ノイズを強めない）、"
        "結果はまわり（半径に合わせた正方形）の最小〜最大の範囲から出さない（--halo 0。縁の白い輪・黒い輪を抑える）。色の面は"
        "そのまま（動画は色の面を変えずに書き戻す）。cas（ffmpeg の AMD Contrast Adaptive Sharpening）・unsharp（ffmpeg の素の"
        "アンシャープマスク）も選べる",
        "measured": "斜めの縁（ISO 12233 の方法の簡易版）を σ1.0 でぼかした入力（MTF50 0.181 周期/px。ぼかす前 0.612）: 既定 → "
        "MTF50 0.254（×1.40）・白い輪 0.7%・平らな所のノイズ ×1.05。同じ強さで上限なし・下限なし → ×1.36・白い輪 3.5%・ノイズ ×1.92、"
        "ffmpeg unsharp 1.2 → ×1.37・3.4%・×2.10、cas 0.6 → ×1.09・ノイズ ×1.78、cas 1.0 → 平らな所が壊れた（ノイズ ×57）。"
        "実写 7 枚（σ1.0 でぼかし σ1 のノイズ）のぼかす前との PSNR: 入力 38.07 → 既定 39.64 dB（強さ 1.6 で 39.70）、上限なし "
        "0.8: 39.11・1.6: 38.53、unsharp 0.8: 39.22、cas 0.3・0.6: 37.88・37.73（入力より悪い）。story540（圧縮された動画。ノイズの"
        "推定は 0）: 輪郭の強さ ×1.32・平らな所のざらつき 1.18 → 1.17（下限 1 なし（0）だと ×1.48・1.32 に増えた）。"
        "速さ: 540×960 で 0.03 秒/コマ、画像 1 枚 0.04 秒",
        "limits": "ぼけを戻すのではなく輪郭の明暗差を強めるだけ（強いぼけ・動きのぶれは 29）。上限 0 は縁の明るさをまわりの範囲に"
        "とどめるので、強さを上げても輪郭は太く・硬くはならないが、細かい模様の中の明暗差は強まる。--halo -1 と --threshold 0 で"
        "素のアンシャープマスク。cas は試験では効きが弱く、強さを 0.9 までにした。透過は残さない（動画は MP4）",
        "tested": "test_sharpen.py（chart: 斜めの縁の合成画像で MTF50・白い輪・ノイズ、photo: 実写 7 枚の PSNR）、story540 2 秒・v4 1 秒"
        "（音声と色の印 bt709 が残る）・building.jpg をコマンドで処理、確認画像 39-sharpen-chart.jpg（縁と平らな所の拡大・縁の断面"
        "のグラフ）・39-sharpen-photo.jpg・39-sharpen-video.jpg",
    },
    50: {
        "status": "◎",
        "how": "明るさ（動画は Y の面、画像は YCrCb の Y）に、平均 0 のガウスの白色ノイズをガウスでぼかした粒を足す。粒の大きさは"
        "自己相関の半値全幅（px）で指定し、離散の自己相関で合うようにぼかしの σ を求める（--size 1 以下は 1 画素ごと）。"
        "強さ（--strength。標準偏差、0〜255 の段）は中間の明るさで指定の値、黒・白に向かって 0.35 倍まで弱める"
        "（0.35 + 0.65×4y(1−y)。--flat で一定）。乱数の種と、コマの番号から粒を作るので、コマごとに変わり（--static で同じ）、"
        "同じ種なら同じ結果。色の粒は --chroma（既定 0）。動画は H.264 に -tune grain を付けて書く（色の面は変えない）。"
        "HyperFrames の作品の中で粒子を足す時は、決まりどおり npx hyperframes media-treatment の details.grain"
        "（/media-use スキル）を使う。この道具は、素材のファイルそのものに前もって粒子を焼き込む用",
        "measured": "平らな灰色の動画（540×960・24 コマ、可逆）に強さ 4: 足した粒の標準偏差 3.94・3.94・3.92（大きさ 1・1.5・3。"
        "H.264 crf16 の後）、画像（PNG）4.01〜4.02、平均のずれ 0.02 以下、自己相関の半値全幅 1.00・1.51・3.04px（画像 1.00・1.50・"
        "2.98）、前後のコマの粒の相関 0.000（--static は 0.993）、色の面の変化 0。明るさごとの強さは式どおり（Y 118: 3.90、式 3.99）、"
        "真っ黒・真っ白に近い所は、はみ出す分を切るのでさらに弱い（Y 16: 0.49）。H.264 での残り方（実写に足した粒と理想の粒の"
        "相関）: crf16 0.828、-tune grain つき 0.883（ファイルは 1.27 倍）。縞: 暗いなめらかなグラデーションを 8bit に丸めた物"
        "（1 段の縞が約 53px ごと）で、縞が見える原因の平らな台地（5×5 が同じ値）が 92.5% → 強さ 1: 71.9%・2: 4.6%・4: 0.1% に"
        "減った。目のぼかし（σ4px）を通した階段のずれは 0.216 → 0.218〜0.225 で変わらない（粒は段を消すのではなく隠す）。"
        "速さ: 540×960 で 0.04〜0.05 秒/コマ",
        "limits": "縞（バンディング）を目立たなくするだけで、段そのものは消さない（消すのは 45。両方なら 45 の後にかける）。1 段の縞を"
        "隠すには強さ 2 以上が要った。粒は H.264 で少し形が変わる（上の相関）。強くすると圧縮後のファイルが大きくなる。"
        "本物のフィルムの粒の形（銀塩の粒子の集まり）や色ごとの粒の違いまでは真似ていない。透過は残さない（動画は MP4）",
        "tested": "test_grain.py（stats: 平らな灰色と階段の合成動画・PNG で標準偏差・平均・半値全幅・コマ間の相関・色の面・明るさごと、"
        "H.264 の設定ごとの残り方、band: 8bit の縞の台地の割合と目のぼかし後の誤差、video: story540 2 秒）、test_grain_encode.py"
        "（実写で -tune grain の有無）、確認画像 50-grain-banding.jpg（明暗 8 倍）・50-grain-frames.jpg（3 コマの粒）・50-grain-video.jpg",
    },
}


# ── 共通の部品 ────────────────────────────────────────────────────────────

# 色の変換は正確に丸める。core.read_frames の既定の変換は ffmpeg の切り捨てで平均 0.5 段暗く読み（灰色の階段で実測）、
# core.VideoWriter の既定の書き出しは手を加えないコマの往復でも 40.1 dB だった（この方法は 46.5 dB。どちらも H.264 crf 16）
_SWS = "accurate_rnd+full_chroma_int+full_chroma_inp"
_MATRIX = {
    "bt709": "bt709",
    "smpte170m": "bt601",
    "bt470bg": "bt601",
    "bt2020nc": "bt2020",
    "bt2020c": "bt2020",
}
_UNSET = {"", "unknown", "unspecified", "reserved"}


@dataclass(frozen=True)
class _Color:
    """元の映像の色の決まり。YUV と RGB の変換はこの行列で行い、書き出しには元と同じ印を付ける。"""

    # swscale の行列の名前（bt709 / bt601 / bt2020）。印が無ければ ffmpeg の既定の bt601
    matrix: str
    # フルレンジ（0〜255）で記録されているか（書き出しは tv の範囲にそろえる）
    full: bool
    # 書き出しに付ける色の印（元に印が無ければ付けない。ffmpeg で処理した時と同じ）
    tags: tuple[str, ...]


def _color(info: core.MediaInfo) -> _Color:
    v = next(
        (s for s in info.extra.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    space = str(v.get("color_space", ""))
    full = v.get("color_range") == "pc" or str(v.get("pix_fmt", "")).startswith("yuvj")
    tags: list[str] = []
    for key, opt in (
        ("color_space", "-colorspace"),
        ("color_primaries", "-color_primaries"),
        ("color_transfer", "-color_trc"),
    ):
        val = str(v.get(key, ""))
        if val not in _UNSET:
            tags += [opt, val]
    if tags or v.get("color_range") in ("tv", "pc"):
        tags += ["-color_range", "tv"]
    return _Color(_MATRIX.get(space, "bt601"), full, tuple(tags))


def _span(a: argparse.Namespace, info: core.MediaInfo) -> float:
    """処理する長さ（秒）。"""
    dur = max(info.duration - (a.start or 0.0), 0.0)
    if getattr(a, "duration", None):
        dur = min(dur, a.duration)
    if getattr(a, "max_frames", None):
        dur = min(dur, a.max_frames / info.fps)
    return dur


def _read(
    a: argparse.Namespace,
    info: core.MediaInfo,
    mode: str = "bgr",
    size: tuple[int, int] | None = None,
) -> Iterator[np.ndarray]:
    """処理する範囲（--start・--duration・--max-frames）の動画のコマを順に読む。

    mode="bgr": BGR ／ "bgra": 透過つき BGRA ／ "yuv": Y・U・V の 3 面を重ねた H×W×3（色の面は最近傍で広げ、
    範囲は tv にそろえる。明るさだけを変える処理用で、_Writer(mode="yuv") と組にすると、色の面は元とまったく同じに戻る）。
    """
    col = _color(info)
    w, h = size or (info.width, info.height)
    rng = "pc" if col.full else "tv"
    chain = [f"scale={w}:{h}:flags=area"] if (w, h) != (info.width, info.height) else []
    if mode == "yuv":
        chain.append(
            f"scale=flags=neighbor+{_SWS}:in_range={rng}:out_range=tv,format=yuv444p"
        )
        pix, ch = "yuv444p", 3
    else:
        pix, ch = ("bgra", 4) if mode == "bgra" else ("bgr24", 3)
        chain.append(
            f"scale=flags=bicubic+{_SWS}:in_color_matrix={col.matrix}:in_range={rng},format={pix}"
        )
    # 範囲は入力側で切り、コマ数は数えて止める。出力側の -t で切ると、生の画素で書き出す時のコマの時刻の丸めで
    # 最後のコマが落ちることがあった（-ss 1.85 -t 0.1 の 10fps で 0 コマ。2026-11-15 実測）
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
    if a.start:
        cmd += ["-ss", f"{a.start:.4f}"]
    if getattr(a, "duration", None):
        cmd += ["-t", f"{a.duration:.4f}"]
    if info.has_alpha and info.codec == "vp9":
        cmd += ["-c:v", "libvpx-vp9"]  # 標準の VP9 デコーダーは透過を捨てるため
    cmd += ["-i", str(info.path), "-an", "-vf", ",".join(chain)]
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix, "-"]
    size_bytes = w * h * ch
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    count = 0
    try:
        while a.max_frames is None or count < a.max_frames:
            buf = proc.stdout.read(size_bytes)
            if len(buf) < size_bytes:
                break
            arr = np.frombuffer(buf, np.uint8)
            if mode == "yuv":
                yield np.ascontiguousarray(arr.reshape(3, h, w).transpose(1, 2, 0))
            else:
                yield arr.reshape(h, w, ch).copy()
            count += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if count == 0:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise core.VfxError(f"コマを読めませんでした: {info.path}\n{err[-800:]}")


class _Writer:
    """コマを MP4（H.264。設定は core.MP4_ARGS と同じ）か、透過つきなら WebM（VP9）に書く。

    mode は _read と同じ（"bgr" / "bgra" / "yuv"）。audio=(元, 開始秒, 長さ) で元の音声の同じ範囲を入れる。
    色は元と同じ行列で YUV に戻し、元と同じ色の印を付ける。x264 には H.264 に足す設定（-tune grain など）。
    """

    def __init__(
        self,
        out: Path,
        fps: float,
        color: _Color,
        *,
        mode: str = "bgr",
        audio: tuple[Path, float, float | None] | None = None,
        x264: tuple[str, ...] = (),
    ) -> None:
        self.out, self.fps, self.color, self.mode = out, fps, color, mode
        self.audio, self.x264 = audio, x264
        self.proc: subprocess.Popen[bytes] | None = None
        self.count = 0

    def _open(self, w: int, h: int) -> None:
        core.require(
            w % 2 == 0 and h % 2 == 0, f"動画の大きさは偶数にしてください（{w}×{h}）"
        )
        webm = self.out.suffix.lower() == ".webm"
        pix = {"bgr": "bgr24", "bgra": "bgra", "yuv": "yuv444p"}[self.mode]
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error"]
        cmd += ["-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}"]
        cmd += ["-r", f"{self.fps:.6f}", "-i", "-"]
        if self.audio:
            src, start, dur = self.audio
            if start:
                cmd += ["-ss", f"{start:.4f}"]
            if dur:
                cmd += ["-t", f"{dur:.4f}"]
            cmd += ["-i", str(src), "-map", "0:v", "-map", "1:a", "-shortest"]
        if self.mode == "yuv":
            vf = f"scale=flags=area+{_SWS},format=yuv420p"
        else:
            fmt = "yuva420p" if webm and self.mode == "bgra" else "yuv420p"
            vf = (
                f"scale=flags=area+{_SWS}:out_color_matrix={self.color.matrix}"
                f":out_range=tv,format={fmt}"
            )
        cmd += ["-vf", vf]
        if webm:
            cmd += ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "24", "-row-mt", "1"]
            cmd += ["-auto-alt-ref", "0"] + (["-c:a", "libopus"] if self.audio else [])
        else:
            cmd += [*core.MP4_ARGS, *self.x264]
            cmd += ["-c:a", "aac", "-b:a", "192k"] if self.audio else []
        cmd += [*self.color.tags, str(self.out)]
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.mode == "bgr":
            frame = frame[:, :, :3]
        h, w = frame.shape[:2]
        if self.proc is None:
            self._open(w, h)
        assert self.proc is not None and self.proc.stdin is not None
        if self.mode == "yuv":
            frame = frame.transpose(2, 0, 1)
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.count += 1

    def close(self) -> Path:
        if self.proc is None:
            raise core.VfxError(f"書き出すコマがありません: {self.out}")
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = (
            self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        )
        if self.proc.wait() != 0:
            raise core.VfxError(f"書き出しに失敗しました: {self.out}\n{err[-800:]}")
        return self.out

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None and exc[0] is None:
            self.close()
        elif self.proc is not None:
            self.proc.kill()


def _audio(
    a: argparse.Namespace, info: core.MediaInfo
) -> tuple[Path, float, float | None] | None:
    if not info.has_audio:
        return None
    limited = bool(a.duration or a.max_frames)
    return info.path, a.start or 0.0, _span(a, info) if limited else None


def _paths(
    a: argparse.Namespace, info: core.MediaInfo, tag: str, *, alpha: bool = False
) -> tuple[Path, Path, Path]:
    """(出力, JSON, 確認画像) の場所。画像は画像（既定 PNG）、動画は MP4（透過を残す時は WebM）。"""
    if a.out:
        out = Path(a.out)
    else:
        ext = ".png" if info.is_image else (".webm" if alpha else ".mp4")
        out = core.default_out(a.src, tag, ext)
    if info.is_image:
        core.require(
            out.suffix.lower() in core.IMAGE_EXTS,
            f"画像の出力先は画像の拡張子（.png など）にしてください: {out}",
        )
    else:
        core.require(
            out.suffix.lower() in (".mp4", ".webm"),
            f"動画の出力先は .mp4 か .webm にしてください: {out}",
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    return out, json_path, out.with_name(out.stem + "-compare.jpg")


def _proc_size(info: core.MediaInfo, max_side: int | None) -> tuple[int, int]:
    """処理する大きさ。動画は偶数にそろえる（H.264 の都合）。画像は元のまま（縮める時だけ丸める）。"""
    if info.is_image:
        f = min(1.0, max_side / max(info.width, info.height)) if max_side else 1.0
        return max(1, round(info.width * f)), max(1, round(info.height * f))
    return core.scaled_size(info.width, info.height, max_side=max_side)


def _load(a: argparse.Namespace, info: core.MediaInfo) -> np.ndarray:
    """画像を BGRA で読む（--max-side があれば縮める）。"""
    img = core.load_image(a.src, alpha=True)
    w, h = _proc_size(info, getattr(a, "max_side", None))
    if (w, h) != (img.shape[1], img.shape[0]):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def _save_image(out: Path, img: np.ndarray) -> None:
    """透過が意味を持つ時だけ透過を残して保存する（JPEG は透過なし）。"""
    if img.shape[2] == 4 and (
        out.suffix.lower() in {".jpg", ".jpeg"} or (img[:, :, 3] == 255).all()
    ):
        img = img[:, :, :3]
    core.save_image(out, img)


def _finish(
    out: Path,
    json_path: Path,
    sheet_path: Path,
    sheet: np.ndarray | None,
    data: dict[str, Any],
) -> None:
    core.write_json(json_path, data)
    core.log(f"保存: {out}")
    if sheet is not None:
        core.save_image(sheet_path, sheet)
        core.log(f"保存: {sheet_path}")
    core.log(f"保存: {json_path}")


def _box_at(score: np.ndarray, s: int) -> tuple[int, int, int]:
    """score の合計が最大の、一辺 s の正方形（左上 x, y と一辺）。"""
    h, w = score.shape[:2]
    s = max(8, min(s, h, w))
    blur = cv2.blur(score.astype(np.float32), (s, s))
    half = s // 2
    inner = blur[half : h - s + half + 1, half : w - s + half + 1]
    y, x = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return int(x), int(y), s


def _compare(
    images: list[np.ndarray],
    labels: list[str],
    box: tuple[int, int, int] | None = None,
    cell_width: int = 360,
    enhance: float = 1.0,
) -> np.ndarray:
    """画像を横に並べ（大きさは最後の画像に合わせる）、下の段に同じ所の拡大を並べた確認画像を作る。

    box は最後の画像の大きさでの (x, y, 一辺)。省略時は最初と最後の差が最も大きい所。enhance > 1 なら、拡大の
    明暗の差を強調する（最初の画像の拡大の平均を中心に enhance 倍。細かい粒や段差を見やすくする）。
    """
    ref = images[-1]
    h, w = ref.shape[:2]
    ims = []
    for im in images:
        im = im[:, :, :3]
        if im.shape[:2] != (h, w):
            up = im.shape[1] < w
            im = cv2.resize(
                im, (w, h), interpolation=cv2.INTER_NEAREST if up else cv2.INTER_AREA
            )
        ims.append(im)
    if box is None:
        diff = cv2.absdiff(ims[0], ims[-1]).max(axis=2)
        box = _box_at(diff, min(h, w) // 4)
    x, y, s = box
    full = []
    for im in ims:
        im = im.copy()
        cv2.rectangle(
            im, (x, y), (x + s - 1, y + s - 1), (0, 220, 255), max(2, w // 240)
        )
        full.append(im)
    top = core.contact_sheet(full, labels, cols=len(ims), cell_width=cell_width)
    zoom = cell_width / s
    center = ims[0][y : y + s, x : x + s].reshape(-1, 3).mean(axis=0)
    crops = []
    for im in ims:
        c = im[y : y + s, x : x + s].astype(np.float32)
        if enhance != 1.0:
            c = (c - center) * enhance + center
        c = np.clip(c, 0, 255).astype(np.uint8)
        crops.append(
            cv2.resize(
                c,
                (cell_width, cell_width),
                interpolation=cv2.INTER_NEAREST if zoom >= 2 else cv2.INTER_AREA,
            )
        )
        note = f"拡大 ×{zoom:.1f}" + (f"・明暗 ×{enhance:g}" if enhance != 1.0 else "")
    bottom = core.contact_sheet(
        crops, [note] * len(crops), cols=len(crops), cell_width=cell_width
    )
    return np.vstack([top, bottom])


def _luma_f(img: np.ndarray) -> np.ndarray:
    """BGR → 明るさ（BT.601 の重み。0〜255 の float32）。"""
    f = img[:, :, :3].astype(np.float32)
    return f[..., 0] * 0.114 + f[..., 1] * 0.587 + f[..., 2] * 0.299


def _noise_sigma(y: np.ndarray) -> float:
    """明るさのノイズの強さ（標準偏差、0〜255）を 1 枚から推定する。

    2×2 の Haar ウェーブレットの対角成分の |値| の中央値 ÷ 0.6745 を 16×16 の区画ごとに求め、白飛び・黒つぶれの
    区画を除いた下位 25% の値（模様の多い区画を避ける。restore.py の 28 と同じ考え方）。
    """
    h, w = y.shape
    y = y[: h - h % 2, : w - w % 2].astype(np.float32)
    a, b, c, d = y[0::2, 0::2], y[0::2, 1::2], y[1::2, 0::2], y[1::2, 1::2]
    hh = np.abs(a - b - c + d) / 2
    blk = 16
    hb, wb = hh.shape[0] // blk, hh.shape[1] // blk
    if hb < 2 or wb < 2:
        return float(np.median(hh) / 0.6745)
    t = hh[: hb * blk, : wb * blk].reshape(hb, blk, wb, blk).transpose(0, 2, 1, 3)
    est = np.median(t.reshape(hb * wb, -1), axis=1) / 0.6745
    m = ((a + b + c + d) / 4)[: hb * blk, : wb * blk]
    m = m.reshape(hb, blk, wb, blk).mean(axis=(1, 3)).ravel()
    ok = (m > 20) & (m < 235)
    if ok.sum() < 8:
        ok[:] = True
    return float(np.percentile(est[ok], 25))


def _sample_frames(
    a: argparse.Namespace, info: core.MediaInfo, n: int, mode: str = "bgr"
) -> list[np.ndarray]:
    """処理する範囲から等間隔に n コマ読む（調べる用）。"""
    start, dur = a.start or 0.0, _span(a, info)
    frames = max(1, int(dur * info.fps))
    idx = sorted({int((k + 0.5) * frames / n) for k in range(n)})
    out = []
    for i in idx:
        one = argparse.Namespace(
            start=start + i / info.fps, duration=None, max_frames=1
        )
        size = _proc_size(info, getattr(a, "max_side", None))
        out.append(next(_read(one, info, mode, size)))
    return out


# ── PyTorch のモデル（区画に分けて推論） ──────────────────────────────────


def _starts(n: int, size: int, overlap: int) -> list[int]:
    """長さ n を、長さ size・重なり overlap 以上の区画で覆う開始位置（区画の大きさはすべて同じ）。"""
    if n <= size:
        return [0]
    k = math.ceil((n - size) / max(1, size - overlap)) + 1
    return [round(i * (n - size) / (k - 1)) for i in range(k)]


def _ramp(n: int, o: int, first: bool, last: bool) -> np.ndarray:
    """区画の重みの 1 次元の形（重なる側だけ、端の o px で 0 から 1 へ上がる）。"""
    r = np.ones(n, np.float32)
    o = min(o, n // 2)
    if o > 0:
        up = (np.arange(o, dtype=np.float32) + 0.5) / o
        if not first:
            r[:o] = np.minimum(r[:o], up)
        if not last:
            r[n - o :] = np.minimum(r[n - o :], up[::-1])
    return r


def _tiled(
    fn: Callable[[np.ndarray], np.ndarray],
    img: np.ndarray,
    tile: int,
    overlap: int,
    scale: int = 1,
) -> np.ndarray:
    """img（H×W×C の float32）を tile×tile の区画に分けて fn にかけ、重なりをなめらかに混ぜてつなぐ。

    区画はすべて同じ大きさにする（PyTorch の CPU 推論は、初めての大きさで準備に数秒かかるため）。
    """
    h, w = img.shape[:2]
    # 小さな画像は分けない方が速い（270×480 で 0.15 秒、区画に分けると 0.31 秒。2026-11-15 実測）
    if tile <= 0 or h * w <= 2.5 * tile * tile:
        return fn(img)
    th, tw = min(tile, h), min(tile, w)
    ys, xs = _starts(h, th, overlap), _starts(w, tw, overlap)
    out = np.zeros((h * scale, w * scale, img.shape[2]), np.float32)
    acc = np.zeros((h * scale, w * scale, 1), np.float32)
    o = overlap * scale
    for y0 in ys:
        ry = _ramp(th * scale, o, y0 == 0, y0 + th >= h)
        for x0 in xs:
            rx = _ramp(tw * scale, o, x0 == 0, x0 + tw >= w)
            res = fn(img[y0 : y0 + th, x0 : x0 + tw])
            wgt = (ry[:, None] * rx[None, :])[..., None]
            sy, sx = y0 * scale, x0 * scale
            out[sy : sy + th * scale, sx : sx + tw * scale] += res * wgt
            acc[sy : sy + th * scale, sx : sx + tw * scale] += wgt
    return out / np.maximum(acc, 1e-6)


def _state_dict(path: Path) -> dict[str, Any]:
    torch = models.torch_setup()
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            return sd[key]
    return sd


def _bf16_fast() -> bool:
    """この CPU で bfloat16 の行列計算が速いか（AMX か AVX-512 BF16 があるか）。"""
    torch = models.torch_setup()
    for name in ("_is_amx_tile_supported", "_is_avx512_bf16_supported"):
        f = getattr(torch.cpu, name, None)
        if f is not None and f():
            return True
    return False


class _TorchImage:
    """spandrel で読んだ画像のモデルを、RGB（0〜1）の numpy で呼ぶ。"""

    def __init__(self, desc: Any, *, bf16: bool) -> None:
        self.torch = models.torch_setup()
        self.net = desc.model.eval().to(memory_format=self.torch.channels_last)
        self.scale = int(desc.scale)
        self.bf16 = bf16

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]
        t = t.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.bf16):
            y = self.net(t)
        return y.float()[0].clamp(0, 1).permute(1, 2, 0).numpy()


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgr[:, :, 2::-1], dtype=np.float32) / 255


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb[:, :, ::-1] * 255 + 0.5, 0, 255).astype(np.uint8)


# ── 26 超解像 ─────────────────────────────────────────────────────────────

_UPSCALE = {
    "auto": "3 倍以上なら ai、それ未満なら lanczos（既定。試験で、きれいな素材の 2〜3 倍は lanczos の方が元に近かった）",
    "ai": "Real-ESRGAN（realesr-general-x4v3。BSD-3-Clause）で 4 倍にしてから仕上がりの大きさに合わせる。"
    "細部を描き足してくっきりさせる。圧縮の傷が目立つ素材にも向く",
    "lanczos": "lanczos（OpenCV の INTER_LANCZOS4）で拡大するだけ。速く、元に無い細部は作らない",
}
_AI_FROM = 3.0  # auto で AI を使う倍率（きれいな素材は 3 倍で lanczos が僅かに良く、4 倍で AI が良かった）
# Real-ESRGAN のノイズを残す版（公式の inference_realesrgan.py の --denoise_strength と同じく、重みを混ぜて使う）
_ESRGAN_WDN = models.ModelFile(
    models.GH
    + "xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "BSD-3-Clause",
    "Real-ESRGAN（ノイズを残す版）",
)


class _Upscaler:
    """Real-ESRGAN（4 倍）。denoise（0〜1）で 2 つの重みを混ぜる（1 = 強く消す版だけ、0 = 残す版だけ）。"""

    def __init__(self, denoise: float, tile: int, precision: str) -> None:
        import spandrel

        sd = _state_dict(models.fetch("realesrgan"))
        if denoise < 1:
            sd2 = _state_dict(models.fetch_spec(_ESRGAN_WDN))
            sd = {k: denoise * v + (1 - denoise) * sd2[k] for k, v in sd.items()}
        desc = spandrel.ModelLoader().load_from_state_dict(sd)
        bf16 = precision == "bf16" or (precision == "auto" and _bf16_fast())
        self.net = _TorchImage(desc, bf16=bf16)
        self.precision = "bf16" if bf16 else "fp32"
        self.tile = tile

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """BGR（uint8）→ 4 倍の BGR（uint8）。"""
        out = _tiled(self.net, _to_rgb(bgr), self.tile, 12, self.net.scale)
        return _to_bgr(out)


def _target_size(info: core.MediaInfo, a: argparse.Namespace) -> tuple[int, int, float]:
    """仕上がりの (幅, 高さ, 倍率)。動画は偶数にそろえる。"""
    w0, h0 = info.width, info.height
    if a.width and a.height:
        w, h = a.width, a.height
        if abs(w / h - w0 / h0) > 0.01 * w0 / h0:
            core.log("注意: --width と --height の縦横比が元と違うので、引き伸ばします")
    elif a.width:
        w, h = a.width, a.width * h0 / w0
    elif a.height:
        w, h = a.height * w0 / h0, a.height
    else:
        core.require(a.scale > 0, "--scale は 0 より大きくしてください")
        w, h = w0 * a.scale, h0 * a.scale
    if info.is_image:
        w, h = max(1, round(w)), max(1, round(h))
    else:
        w, h = max(2, round(w / 2) * 2), max(2, round(h / 2) * 2)
    return int(w), int(h), w / w0


@command(
    26,
    "upscale",
    SRC,
    OUT,
    JSON,
    RANGE,
    arg(
        "--scale",
        type=float,
        default=2.0,
        help="倍率（既定 2。--width か --height を指定するとそちらを使う）",
    ),
    arg("--width", type=int, help="仕上がりの幅（px。高さは縦横比を保って決める）"),
    arg("--height", type=int, help="仕上がりの高さ（px。幅は縦横比を保って決める）"),
    arg(
        "--method",
        choices=list(_UPSCALE),
        default="auto",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _UPSCALE.items()),
    ),
    arg(
        "--denoise",
        type=float,
        default=0.0,
        help="AI のノイズの消し方（0〜1。0 は粒子や細かい模様を残す、1 は強く消してなめらか。既定 0。"
        "試験ではどの倍率・劣化でも 0 が最も元に近かった。ザラつきの強い素材は 0.5〜1）",
    ),
    arg(
        "--tile",
        type=int,
        default=256,
        help="AI に一度に渡す区画の一辺（元の px。既定 256。0 で分けない）",
    ),
    arg(
        "--precision",
        choices=["auto", "fp32", "bf16"],
        default="auto",
        help="AI の計算の精度（auto: CPU が対応していれば bf16 で速く。fp32 との差は 54 dB で見分けられない）",
    ),
)
def upscale(a: argparse.Namespace) -> None:
    """低い解像度の映像・画像を拡大する（3 倍以上は Real-ESRGAN で細部を描き足す）。透過つき（WebM・PNG）は透過も拡大する。"""
    info = core.probe(a.src)
    core.require(0 <= a.denoise <= 1, "--denoise は 0〜1 にしてください")
    tw, th, factor = _target_size(info, a)
    core.require(factor >= 1, "縮小には使えません（倍率 1 以上にしてください）")
    if a.method == "auto":
        a.method = "ai" if factor >= _AI_FROM else "lanczos"
        core.log(
            f"方法: {a.method}（auto。{_AI_FROM:g} 倍以上は ai。"
            + (
                "圧縮の傷が目立つ素材なら --method ai も試す）"
                if a.method == "lanczos"
                else "元に忠実さを優先するなら --method lanczos）"
            )
        )
    if factor > 4 and a.method == "ai":
        core.log(
            f"注意: AI は 4 倍まで。残りの {factor / 4:.2f} 倍は lanczos で拡大します"
        )
    keep_alpha = info.has_alpha or bool(
        info.is_image and core.load_image(a.src, alpha=True)[:, :, 3].min() < 255
    )
    out, json_path, sheet_path = _paths(a, info, "upscale", alpha=keep_alpha)
    keep_alpha = keep_alpha and out.suffix.lower() in (".webm", ".png", ".webp")
    up = _Upscaler(a.denoise, a.tile, a.precision) if a.method == "ai" else None
    core.log(
        f"{info.width}×{info.height} → {tw}×{th}（{factor:.2f} 倍、{a.method}"
        + (f"、ノイズの消し方 {a.denoise:g}、{up.precision}" if up else "")
        + "）"
    )

    def process(frame: np.ndarray) -> np.ndarray:
        bgr = frame[:, :, :3]
        if up is not None:
            big = up(bgr)
            interp = cv2.INTER_AREA if tw <= big.shape[1] else cv2.INTER_LANCZOS4
            res = cv2.resize(big, (tw, th), interpolation=interp)
        else:
            res = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        if keep_alpha and frame.shape[2] == 4:
            alpha = cv2.resize(
                frame[:, :, 3], (tw, th), interpolation=cv2.INTER_LANCZOS4
            )
            res = np.dstack([res, alpha])
        return res

    t0 = time.time()
    if info.is_image:
        src = core.load_image(a.src, alpha=True)
        res = process(src)
        _save_image(out, res)
        n, mid = 1, (src, res)
    else:
        mode = "bgra" if keep_alpha else "bgr"
        total = max(1, round(_span(a, info) * info.fps))
        mid_i = total // 2
        mid = None
        writer = _Writer(out, info.fps, _color(info), mode=mode, audio=_audio(a, info))
        n = 0
        with writer:
            for i, frame in enumerate(_read(a, info, mode)):
                res = process(frame)
                writer.write(res)
                if i == mid_i or mid is None:
                    mid = (frame, res)
                n += 1
                if (i + 1) % max(1, total // 10) == 0:
                    core.log(
                        f"  {i + 1}/{total} コマ（{(time.time() - t0) / (i + 1):.2f} 秒/コマ）"
                    )
        assert mid is not None
    dt = time.time() - t0
    core.log(f"{n} コマ、{dt / n:.2f} 秒/コマ")
    src_f, res_f = mid
    lanczos = cv2.resize(src_f[:, :, :3], (tw, th), interpolation=cv2.INTER_LANCZOS4)
    nearest = cv2.resize(src_f[:, :, :3], (tw, th), interpolation=cv2.INTER_NEAREST)
    # 細部の多い所（AI と lanczos の差が大きい所。lanczos なら輪郭の強い所）を拡大して見せる
    score = (
        cv2.absdiff(lanczos, res_f[:, :, :3]).max(axis=2)
        if up is not None
        else np.abs(cv2.Laplacian(_luma_f(lanczos), cv2.CV_32F))
    )
    box = _box_at(score, max(32, min(tw, th) // 8))
    if up is not None:
        sheet = _compare(
            [nearest, lanczos, res_f],
            ["元（最近傍で拡大）", "lanczos", f"超解像（AI、ノイズ {a.denoise:g}）"],
            box,
        )
    else:
        sheet = _compare([nearest, res_f], ["元（最近傍で拡大）", "lanczos"], box)
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "model": "realesr-general-x4v3 + wdn (BSD-3-Clause)" if up else None,
            "denoise": a.denoise if up else None,
            "precision": up.precision if up else None,
            "size_in": [info.width, info.height],
            "size_out": [tw, th],
            "scale": round(factor, 4),
            "alpha": keep_alpha,
            "frames": n,
            "sec_per_frame": round(dt / n, 3),
        },
    )


# ── 27 フレーム補間 ───────────────────────────────────────────────────────

_INTERP = {
    "mci": "動きを推定して中間のコマを作る（ffmpeg の minterpolate の mci・aobmc・bidir・vsbmc。"
    "場面の変わり目はつながない。既定）",
    "blend": "前後のコマを重ねて混ぜる（minterpolate の blend。崩れないが、動く物は二重に見える）",
    "dup": "同じコマを繰り返すだけ（minterpolate の dup。なめらかにはならない。比べる用）",
}


def _interp_filter(method: str, fps: float) -> str:
    """どの方法も minterpolate で行う（コマの時刻の扱いをそろえるため。ffmpeg の framerate フィルターは 5→10fps で
    出力が 1 コマずれた。2026-11-15 実測）。"""
    if method == "mci":
        # 場面転換の判定（scd）は既定の 10 では見逃しがあったので 5 にする（試験で、カットの無い映像は結果が同じだった）
        return f"minterpolate=fps={fps:.6g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=5"
    if method in ("blend", "dup"):
        return f"minterpolate=fps={fps:.6g}:mi_mode={method}"
    raise core.VfxError(f"知らない方法です: {method}")


def _frame_diffs(a: argparse.Namespace, info: core.MediaInfo) -> np.ndarray:
    """処理する範囲の、前後のコマの差（幅 96px の灰色で、0〜255 の平均の差）。d[j] はコマ j と j+1 の差。"""
    w = 96
    h = max(2, round(info.height * w / info.width / 2) * 2)
    small = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for f in _read(a, info, "bgr", (w, h))
    ]
    return np.array(
        [np.mean(np.abs(x - y)) for x, y in itertools.pairwise(small)], np.float32
    )


def _breaks(
    d: np.ndarray, lo: float = 10.0, k: float = 5.0, win: int = 12
) -> list[int]:
    """補間でつないではいけない所（カット・フラッシュ・急な変化）。返すのは新しい側のコマの番号。

    差が lo より大きく、まわり ±win コマの差の中央値の k 倍より大きい所（較正: vfx-tests/enhance/calib_breaks.py。
    story540 のカット 4/4、カットの無い 10 本で余分 1）。差がほぼ 0 のコマ（重複）は中央値から除く。
    """
    out = []
    for j, v in enumerate(d):
        if v <= lo:
            continue
        around = np.r_[d[max(0, j - win) : j], d[j + 1 : j + win + 1]]
        around = around[around > 0.3]
        med = float(np.median(around)) if len(around) else 0.0
        if v > k * med:
            out.append(j + 1)
    return out


def _interp_part(
    a: argparse.Namespace,
    info: core.MediaInfo,
    j0: int,
    s: int,
    e: int,
    n_out: int,
    fps: float,
    size: tuple[int, int],
    out: Path,
    audio: bool,
) -> list[str]:
    """入力のコマ s〜e−1（範囲の先頭が元の j0 コマ目）を補間して n_out コマにし、out に書く。使ったフィルターを返す。"""
    f = info.fps
    chain = [f"trim=end_frame={e - s}"]
    if size != (info.width, info.height):
        chain.append(f"scale={size[0]}:{size[1]}:flags=area")
    # 最後のコマの先にも補間の相手が要るので、最後のコマを 2 コマ分のばしてから補間し、コマ数で切る（CLAUDE.md）
    chain.append(f"tpad=stop_mode=clone:stop_duration={2 / f:.6f}")
    if a.slow != 1:
        chain.append(f"setpts={a.slow:.6g}*PTS")
    chain += [
        _interp_filter(a.method, fps),
        f"trim=end_frame={n_out}",
        "setpts=PTS-STARTPTS",
    ]
    args: list[str] = []
    t = (
        j0 + s - 0.5
    ) / f  # このコマの半コマ前から読む（-ss はその時刻以降の最初のコマから）
    if t > 0:
        args += ["-ss", f"{t:.6f}"]
    args += ["-t", f"{(e - s + 2) / f:.6f}", "-i", str(a.src)]
    args += ["-map", "0:v:0"] + (["-map", "0:a:0?"] if audio else ["-an"])
    args += ["-vf", ",".join(chain), "-fps_mode", "passthrough", *core.MP4_ARGS]
    if audio:
        args += ["-c:a", "aac", "-b:a", "192k", "-t", f"{n_out / fps:.6f}"]
    core.ffmpeg([*args, *_color(info).tags, str(out)])
    return chain


@command(
    27,
    "interpolate",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--fps",
        type=float,
        help="仕上がりのコマ数／秒（省略時: 元の 2 倍。--slow の時は元と同じ）",
    ),
    arg(
        "--slow",
        type=float,
        default=1.0,
        help="スローモーションの倍率（2 = 長さ 2 倍・半分の速さ。既定 1 = 速さはそのまま）",
    ),
    arg(
        "--method",
        choices=list(_INTERP),
        default="mci",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _INTERP.items()),
    ),
    arg(
        "--no-breaks",
        action="store_true",
        help="カット・フラッシュを探して区切ることをしない（既定は区切り、そこでは中間のコマを作らず前後のコマを置く）",
    ),
)
def interpolate(a: argparse.Namespace) -> None:
    """前後のコマから中間のコマを作り、コマ数を増やしてなめらかにする（スローモーションも）。カットはつながない。"""
    info = core.probe(a.src)
    core.require(
        not info.is_image,
        "動画を指定してください（前後のコマを使う処理なので、画像 1 枚では使えません）",
    )
    core.require(a.slow > 0, "--slow は 0 より大きくしてください")
    fps = a.fps or (info.fps if a.slow != 1 else info.fps * 2)
    core.require(fps > 0, "--fps は 0 より大きくしてください")
    out, json_path, sheet_path = _paths(a, info, "interpolate")
    core.require(out.suffix.lower() == ".mp4", "出力先は .mp4 にしてください")
    f = info.fps
    j0 = math.ceil((a.start or 0.0) * f - 1e-6)  # 範囲の最初のコマ（元の何コマ目か）
    d = _frame_diffs(a, info)
    n_in = len(d) + 1
    breaks = [] if a.no_breaks else _breaks(d)
    keep_audio = info.has_audio and a.slow == 1
    if info.has_audio and not keep_audio:
        core.log("スローモーションでは音声を付けません（速さが合わないため）")
    size = _proc_size(info, a.max_side)
    # 区切りごとに補間する。各区切りの出力のコマ数は、通しの時刻から決める（区切りが多くても長さがずれない）
    bounds = [0, *breaks, n_in]
    parts = []
    for s, e in itertools.pairwise(bounds):
        k0, k1 = round(s / f * a.slow * fps), round(e / f * a.slow * fps)
        if k1 > k0:
            parts.append((s, e, k1 - k0))
    n_out = sum(p[2] for p in parts)
    core.log(
        f"{f:g} コマ/秒 → {fps:g} コマ/秒"
        + (f"、{a.slow:g} 倍の長さ（スローモーション）" if a.slow != 1 else "")
        + f"、{a.method}。区切り {len(breaks)} か所"
        + (
            "（"
            + "・".join(f"{(j0 + b) / f:.2f}" for b in breaks[:12])
            + " 秒"
            + ("…" if len(breaks) > 12 else "")
            + "）"
            if breaks
            else ""
        )
    )
    t0 = time.time()
    if len(parts) == 1:
        s, e, n = parts[0]
        chain = _interp_part(a, info, j0, s, e, n, fps, size, out, keep_audio)
    else:
        with tempfile.TemporaryDirectory(dir=out.parent, prefix=".interp-") as tmp:
            files = []
            for i, (s, e, n) in enumerate(parts):
                part = Path(tmp) / f"part{i:04d}.mp4"
                chain = _interp_part(a, info, j0, s, e, n, fps, size, part, False)
                files.append(part)
            lst = Path(tmp) / "list.txt"
            # concat の一覧では、パスの中の ' を '\'' と書く（ffmpeg の決まり）
            lines = ["file '" + str(p).replace("'", "'\\''") + "'\n" for p in files]
            lst.write_text("".join(lines), encoding="utf-8")
            args = ["-f", "concat", "-safe", "0", "-i", str(lst)]
            if keep_audio:
                if j0:
                    args += ["-ss", f"{j0 / f:.6f}"]
                args += ["-t", f"{n_out / fps:.6f}", "-i", str(a.src)]
                args += [
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                ]
            args += ["-c:v", "copy", "-movflags", "+faststart", str(out)]
            core.ffmpeg(args)
    dt = time.time() - t0
    oinfo = core.probe(out)
    n_real = oinfo.frames or round(oinfo.duration * oinfo.fps)
    core.log(f"{n_in} コマ → {n_real} コマ（{oinfo.duration:.3f} 秒）、{dt:.1f} 秒")
    # 確認画像: 区切りを除いて動きの最も大きい前後のコマと、その間に作ったコマ
    sheet, check = None, {}
    if n_in > 1:
        d_ok = d.copy()
        for b in breaks:
            d_ok[b - 1] = -1
        j = int(np.argmax(d_ok))
        k = max(0, min(n_real - 1, round((j + 0.5) / f * a.slow * fps)))
        one = argparse.Namespace(start=(j0 + j - 0.5) / f, duration=None, max_frames=2)
        prev_f, next_f = list(_read(one, info, "bgr", size))[:2]
        oone = argparse.Namespace(
            start=max(0.0, (k - 0.5) / fps), duration=None, max_frames=1
        )
        mid_f = next(_read(oone, oinfo, "bgr"))
        box = _box_at(cv2.absdiff(prev_f, next_f).max(axis=2), min(size) // 3)
        sheet = _compare(
            [prev_f, mid_f, next_f],
            [
                f"前のコマ（{(j0 + j) / f:.2f} 秒）",
                f"作ったコマ（出力 {k / fps:.2f} 秒）",
                f"次のコマ（{(j0 + j + 1) / f:.2f} 秒）",
            ],
            box,
        )
        check = {
            "prev_time": round((j0 + j) / f, 4),
            "output_frame": k,
            "motion": round(float(d[j]), 3),
        }
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "filter": ",".join(chain),
            "fps_in": f,
            "fps_out": fps,
            "slow": a.slow,
            "frames_in": n_in,
            "frames_out": n_real,
            "duration_out": oinfo.duration,
            "breaks": [
                {"frame": j0 + b, "time": round((j0 + b) / f, 4)} for b in breaks
            ],
            "parts": len(parts),
            "audio": keep_audio,
            "check": check,
            "seconds": round(dt, 2),
        },
    )


# ── 29 ブレ除去 ───────────────────────────────────────────────────────────

_DEBLUR = {
    "auto": "ぶれの形を推定し、ピンぼけなら rl、動画で画面全体の動きのぶれが見つかれば rl、それ以外（画像の手ぶれ・"
    "見分けられない）は ai（既定）",
    "ai": "NAFNet（GoPro で学習した動きのぶれ用、width64。MIT）。ぶれの形を指定しなくてよい。実写の手ぶれの見本で最も"
    "きれいだった。ピンぼけは戻せない",
    "wiener": "ウィーナーの逆畳み込み（速い。0.1 秒/コマ）。ぶれの形（動き・ピンぼけ・ガウス）を指定するか推定する。"
    "動きの向きに沿った縞が出やすい",
    "rl": "リチャードソン・ルーシー（scikit-image）の逆畳み込み。ぶれの形を指定するか推定する。試験ではウィーナーより"
    "良く縞も少なかったが、約 10 倍遅い（1 秒/コマ）",
}
_NAFNET = models.ModelFile(
    # 公式（megvii-research/NAFNet、MIT）の重みを Hugging Face に置いた物（license: mit）。版を固定する
    "https://huggingface.co/tog/nafnet-models/resolve/a0cb4f040b03d82f955482e1b5dd97116eece1c9/NAFNet-GoPro-width64.pth",
    "MIT",
    "NAFNet（動きのぶれの除去。GoPro で学習）",
    sha256="329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f",
)


def _psf_motion(length: float, angle: float) -> np.ndarray:
    """動きのぶれ（長さ length px の線。angle 度: 0 = 横、90 = 縦、45 = 右下がり（＼）、135 = 右上がり（／））。"""
    length = max(1.0, float(length))
    size = math.ceil(length) // 2 * 2 + 5
    ss = 8
    big = np.zeros((size * ss, size * ss), np.float32)
    c = (size * ss - 1) / 2
    th = math.radians(angle)
    n = int(length * ss * 2) + 1
    # 長さ L の線分（−L/2〜L/2）。スペクトルの零点の間隔が 1/L になり、推定（ケプストラム）の長さと一致する
    t = np.linspace(-length / 2, length / 2, n) * ss
    xs, ys = c + t * math.cos(th), c + t * math.sin(th)
    for x, y in zip(xs, ys, strict=True):
        x0, y0 = math.floor(x), math.floor(y)
        fx, fy = x - x0, y - y0
        big[y0, x0] += (1 - fx) * (1 - fy)
        big[y0, x0 + 1] += fx * (1 - fy)
        big[y0 + 1, x0] += (1 - fx) * fy
        big[y0 + 1, x0 + 1] += fx * fy
    big = cv2.blur(big, (ss, ss))  # 線に 1px の幅を持たせる
    psf = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    return psf / psf.sum()


def _psf_disk(radius: float) -> np.ndarray:
    """ピンぼけ（半径 radius px の円盤。縁は面積で滑らかに）。"""
    radius = max(0.5, float(radius))
    size = math.ceil(radius) * 2 + 3
    ss = 8
    yy, xx = np.mgrid[0 : size * ss, 0 : size * ss].astype(np.float32)
    c = (size * ss - 1) / 2
    big = ((xx - c) ** 2 + (yy - c) ** 2 <= (radius * ss) ** 2).astype(np.float32)
    psf = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    return psf / psf.sum()


def _psf_gauss(sigma: float) -> np.ndarray:
    size = math.ceil(sigma * 3) * 2 + 1
    k = cv2.getGaussianKernel(size, sigma)
    psf = (k @ k.T).astype(np.float32)
    return psf / psf.sum()


def _cepstrum(gray: np.ndarray, tile: int = 256) -> np.ndarray:
    """平均の対数パワースペクトル（ハン窓・半分ずつ重ねた区画）のケプストラム（中心が 0、tile×tile）。"""
    g = gray.astype(np.float32)
    h, w = g.shape
    tile = int(min(tile, 2 ** int(math.log2(min(h, w)))))
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    acc = np.zeros((tile, tile), np.float64)
    n = 0
    step = tile // 2
    for y in range(0, h - tile + 1, step):
        for x in range(0, w - tile + 1, step):
            t = g[y : y + tile, x : x + tile]
            t = (t - t.mean()) * win
            acc += np.abs(np.fft.fft2(t)) ** 2
            n += 1
    logp = np.log(acc / max(n, 1) + 1e-6)
    c = np.real(np.fft.ifft2(logp - logp.mean()))
    return np.fft.fftshift(c).astype(np.float32)


def _estimate_blur(gray: np.ndarray) -> dict[str, Any]:
    """ケプストラム（対数スペクトルの逆変換）から、ぶれの形を推定する。

    動きのぶれ（長さ L の線）はスペクトルに間隔 1/L の縞の零点を作り、ケプストラムの (L cosθ, L sinθ) に負の山が出る。
    ピンぼけ（直径 D の円盤）は輪の零点を作り、半径 ≒ D の輪の平均がくぼむ。
    見分け方（較正: vfx-tests/enhance/proto_blur_features.py。8 枚×なし・動き 5・ピンぼけ 5・ガウス、JPEG あり・なし）:
    輪のくぼみ D ≥ 12 ならピンぼけ、負の山 M ≥ 5 で、向きの違う 2 番目の山との比 ≥ 1.4、長さ ≥ 6px なら動き、
    それ以外は「なし」（ぶれが無いか、小さいか、見分けられない）。
    """
    c = _cepstrum(gray)
    t = c.shape[0]
    cy = cx = t // 2
    yy, xx = np.mgrid[0:t, 0:t]
    r = np.hypot(yy - cy, xx - cx)
    ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 180
    rmin, rmax = 3, min(t // 2 - 2, 100)
    ring = (r >= rmin) & (r <= rmax)
    sd = float(c[ring].std()) or 1e-6
    # 動き: 輪の中の最小値の位置（放物線の当てはめで画素より細かく）
    idx = np.argmin(np.where(ring, c, np.inf))
    py, px = np.unravel_index(int(idx), c.shape)
    dy = dx = 0.0
    if 0 < py < t - 1 and 0 < px < t - 1:
        a1, a0, a2 = c[py, px - 1], c[py, px], c[py, px + 1]
        den = a1 - 2 * a0 + a2
        dx = float(np.clip(0.5 * (a1 - a2) / den, -0.5, 0.5)) if den else 0.0
        a1, a2 = c[py - 1, px], c[py + 1, px]
        den = a1 - 2 * a0 + a2
        dy = float(np.clip(0.5 * (a1 - a2) / den, -0.5, 0.5)) if den else 0.0
    vx, vy = px + dx - cx, py + dy - cy
    length = float(math.hypot(vx, vy))
    angle = float(math.degrees(math.atan2(vy, vx)) % 180)
    m1 = float(-c[py, px] / sd)
    far = np.minimum(np.abs(ang - angle), 180 - np.abs(ang - angle)) > 20
    m2 = float(-np.min(np.where(ring & far, c, np.inf)) / sd)
    ratio = m1 / m2 if m2 > 0 else math.inf
    # ピンぼけ: 半径ごとの平均 p(r) の、まわり（±6）の中央値からのくぼみ ÷ その平均の誤差
    rb = np.round(r).astype(int).ravel()
    cnt = np.bincount(rb)
    mean_r = np.bincount(rb, c.ravel()) / np.maximum(cnt, 1)
    var_r = np.bincount(rb, (c * c).ravel()) / np.maximum(cnt, 1) - mean_r**2
    err = np.sqrt(np.maximum(var_r, 1e-12) / np.maximum(cnt, 1))
    dip, rr = 0.0, rmin
    for k in range(rmin + 2, rmax - 2):
        nb = np.r_[
            mean_r[max(rmin, k - 6) : k - 1], mean_r[k + 2 : min(rmax, k + 6) + 1]
        ]
        v = float((np.median(nb) - mean_r[k]) / err[k])
        if v > dip:
            dip, rr = v, k
    off = 0.0
    if rmin < rr < len(mean_r) - 1:
        a1, a0, a2 = mean_r[rr - 1], mean_r[rr], mean_r[rr + 1]
        den = a1 - 2 * a0 + a2
        off = float(np.clip(0.5 * (a1 - a2) / den, -0.5, 0.5)) if den else 0.0
    radius = (rr + off) / 2
    if dip >= 12:
        kind = "defocus"
    elif m1 >= 5 and ratio >= 1.4 and length >= 6:
        kind = "motion"
    else:
        kind = "none"
    return {
        "kind": kind,
        "length": round(length, 2),
        "angle": round(angle, 1),
        "radius": round(radius, 2),
        "motion_score": round(m1, 2),
        "motion_ratio": round(ratio, 2) if math.isfinite(ratio) else None,
        "disk_score": round(dip, 2),
    }


def _psf_from(
    a: argparse.Namespace, est: dict[str, Any] | None
) -> tuple[np.ndarray, dict[str, Any]]:
    """指定（--kernel と長さ・角度・半径・σ）か推定から、ぶれの形（PSF）を作る。"""
    kind = a.kernel
    if kind == "auto":
        core.require(
            est is not None and est["kind"] != "none",
            "ぶれの形を見分けられませんでした（ぶれが無いか小さいか、模様が少ない）。"
            + (
                f"弱い候補: 動き 長さ {est['length']:.1f}px・向き {est['angle']:.0f} 度、ピンぼけ 半径 "
                f"{est['radius']:.1f}px。--kernel motion か --kernel defocus を指定すると、その候補の値を使います"
                f"（--length・--angle・--radius で指定もできる）"
                if est
                else "--kernel と --length・--angle（動き）か --radius（ピンぼけ）を指定してください"
            ),
        )
        assert est is not None
        kind = est["kind"]
    if kind == "motion":
        length = a.length or (est or {}).get("length")
        angle = a.angle if a.angle is not None else (est or {}).get("angle")
        core.require(
            length and angle is not None,
            "動きのぶれは --length と --angle を指定してください",
        )
        return _psf_motion(length, angle), {
            "kind": "motion",
            "length": length,
            "angle": angle,
        }
    if kind == "defocus":
        radius = a.radius or (est or {}).get("radius")
        core.require(radius, "ピンぼけは --radius を指定してください")
        return _psf_disk(radius), {"kind": "defocus", "radius": radius}
    if kind == "gauss":
        core.require(a.sigma, "ガウスのぼけは --sigma を指定してください")
        return _psf_gauss(a.sigma), {"kind": "gauss", "sigma": a.sigma}
    raise core.VfxError(f"知らないぶれの形です: {kind}")


def _otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """PSF を画像の大きさに広げ、中心を原点に移してフーリエ変換する。"""
    pad = np.zeros(shape, np.float32)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.rfft2(pad)


def _padded(y: np.ndarray, m: int) -> np.ndarray:
    """周りを m px 折り返して広げ、外側を平均へなじませる（周期の境目で波紋が出ないように）。"""
    p = cv2.copyMakeBorder(y, m, m, m, m, cv2.BORDER_REFLECT)
    h, w = p.shape
    d = np.minimum.outer(
        np.minimum(np.arange(h), np.arange(h)[::-1]),
        np.minimum(np.arange(w), np.arange(w)[::-1]),
    ).astype(np.float32)
    wgt = np.clip(d / max(1, m // 2), 0, 1)
    blur = cv2.GaussianBlur(p, (0, 0), max(1.0, m / 4))
    return p * wgt + blur * (1 - wgt)


def _wiener(y: np.ndarray, psf: np.ndarray, k: float) -> np.ndarray:
    """ウィーナーの逆畳み込み（定数 K で整える古典的な形）。y は 0〜1 の明るさ。

    ラプラシアンで整える形（scikit-image の wiener と同じ）も試したが、動きのぶれの零点が多い低い周波数で整え方が弱く、
    正しい PSF でも 2〜3 dB 悪かった（vfx-tests/enhance/test_deblur.py calib と calib_wiener.py）。
    """
    m = max(psf.shape) * 2
    p = _padded(y.astype(np.float32), m)
    H = _otf(psf, p.shape)
    X = np.conj(H) * np.fft.rfft2(p) / (np.abs(H) ** 2 + k)
    x = np.fft.irfft2(X, s=p.shape)
    return x[m:-m, m:-m].astype(np.float32)


def _richardson_lucy(y: np.ndarray, psf: np.ndarray, iters: int) -> np.ndarray:
    from skimage.restoration import richardson_lucy

    m = max(psf.shape) * 2
    p = np.clip(_padded(y.astype(np.float32), m), 1e-3, 1)
    x = richardson_lucy(p, psf, num_iter=iters, clip=False)
    return np.asarray(x[m:-m, m:-m], np.float32)


class _Nafnet:
    """NAFNet（GoPro・width64）で動きのぶれを取る。256px の区画に分けて推論する（学習の区画と同じ大きさ）。"""

    def __init__(self, tile: int = 256) -> None:
        import spandrel

        desc = spandrel.ModelLoader().load_from_state_dict(
            _state_dict(models.fetch_spec(_NAFNET))
        )
        # bf16 は速くならず、出力が大きく崩れた（2026-11-14 実測）ので使わない
        self.net = _TorchImage(desc, bf16=False)
        self.tile = tile

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        return _to_bgr(_tiled(self.net, _to_rgb(bgr), self.tile, 32))


def _ycc(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2YCrCb)


def _from_ycc(ycc: np.ndarray, like: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    if like.ndim == 3 and like.shape[2] == 4:
        bgr = np.dstack([bgr, like[:, :, 3]])
    return bgr


def _y_range(video: bool) -> tuple[float, float]:
    """明るさの面の値の範囲（動画の YUV は tv の 16〜235、画像の YCrCb は 0〜255）。"""
    return (16.0, 235.0) if video else (0.0, 255.0)


@command(
    29,
    "deblur",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_DEBLUR),
        default="auto",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _DEBLUR.items()),
    ),
    arg(
        "--kernel",
        choices=["auto", "motion", "defocus", "gauss"],
        default="auto",
        help="wiener・rl で使うぶれの形（auto: 推定する／motion: 動き／defocus: ピンぼけ／gauss: ガウス）",
    ),
    arg("--length", type=float, help="動きのぶれの長さ（px）"),
    arg(
        "--angle",
        type=float,
        help="動きのぶれの向き（度。0 = 横、90 = 縦、45 = 右下がり＼、135 = 右上がり／）",
    ),
    arg("--radius", type=float, help="ピンぼけの円の半径（px）"),
    arg("--sigma", type=float, help="ガウスのぼけの σ（px）"),
    arg(
        "--strength",
        type=float,
        default=1.0,
        help="wiener・rl の効かせ方の倍率（既定 1。大きいとくっきりするが波紋とノイズが増える）",
    ),
)
def deblur(a: argparse.Namespace) -> None:
    """動きのぶれ・ピンぼけを戻す。手ぶれは AI（NAFNet）、ピンぼけはぶれの形を推定して逆畳み込み（リチャードソン・ルーシー）。"""
    info = core.probe(a.src)
    core.require(a.strength > 0, "--strength は 0 より大きくしてください")
    out, json_path, sheet_path = _paths(a, info, "deblur")
    video = not info.is_image
    # ぶれの形の推定（AI では参考に表示するだけ）。動画は 5 コマで推定し、中央値を使う
    if video:
        samples = [f[:, :, 0] for f in _sample_frames(a, info, 5, "yuv")]
    else:
        samples = [_ycc(_load(a, info))[:, :, 0]]
    ests = [_estimate_blur(s.astype(np.float32) / 255) for s in samples]
    est = _merge_estimates(ests)
    core.log(_describe_blur(est))
    if a.method == "auto":
        # 試験（NOTES の measured）: ピンぼけは AI では戻らず rl が良い。H.264 の動画では AI が圧縮の傷をぶれと取り違えて
        # かえって悪くなり、画面全体の一様な動きのぶれなら rl が良かった。実写の手ぶれの画像（JPEG）は AI が最もきれい
        uniform = est["kind"] == "defocus" or (video and est["kind"] == "motion")
        a.method = "rl" if uniform else "ai"
        core.log(
            f"方法: {a.method}（auto。ピンぼけ・動画の一様な動きのぶれは rl、それ以外は ai）"
        )
    if a.method == "ai" and est["kind"] == "none":
        core.log(
            "注意: はっきりしたぶれは見つかりませんでした。ぶれていない・ぶれの弱い映像に AI をかけると、輪郭やなめらかな"
            "所が作り変えられて元から離れることがあります（試験: ぶれていない画像で元との差 34.5 dB）。確認画像で見比べてください"
        )
    psf, psf_info = None, None
    params: dict[str, Any] = {}
    if a.method != "ai":
        psf, psf_info = _psf_from(a, est)
        core.log(f"使うぶれの形: {psf_info}")
        sigma = float(np.median([_noise_sigma(s.astype(np.float32)) for s in samples]))
        if video:
            sigma *= (
                255 / 219
            )  # 動画の明るさの面は 16〜235。較正した 0〜255 の尺度にそろえる
        if a.method == "wiener":
            params = {
                "K": _wiener_k(sigma) / a.strength,
                "noise_sigma": round(sigma, 3),
            }
        else:
            iters = max(1, round(_rl_iters(sigma) * a.strength))
            params = {"iterations": iters, "noise_sigma": round(sigma, 3)}
        core.log(f"設定: {params}")
    net = _Nafnet() if a.method == "ai" else None
    lo, hi = _y_range(video)

    def fix_luma(y: np.ndarray) -> np.ndarray:
        assert psf is not None
        f = (y.astype(np.float32) - lo) / (hi - lo)
        if a.method == "wiener":
            r = _wiener(f, psf, params["K"])
        else:
            r = _richardson_lucy(f, psf, params["iterations"])
        return np.clip(np.round(r * (hi - lo) + lo), lo, hi).astype(np.uint8)

    t0 = time.time()
    if not video:
        src = _load(a, info)
        if net is not None:
            res = net(src[:, :, :3])
            if src.shape[2] == 4:
                res = np.dstack([res, src[:, :, 3]])
        else:
            ycc = _ycc(src)
            ycc[:, :, 0] = fix_luma(ycc[:, :, 0])
            res = _from_ycc(ycc, src)
        _save_image(out, res)
        before, after, n = src[:, :, :3], res[:, :, :3], 1
    else:
        size = _proc_size(info, a.max_side)
        mode = "bgr" if net is not None else "yuv"
        total = max(1, round(_span(a, info) * info.fps))
        mid_i, mid = total // 2, None
        n = 0
        with _Writer(
            out, info.fps, _color(info), mode=mode, audio=_audio(a, info)
        ) as wr:
            for i, frame in enumerate(_read(a, info, mode, size)):
                if net is not None:
                    res = net(frame)
                else:
                    res = frame.copy()
                    res[:, :, 0] = fix_luma(frame[:, :, 0])
                wr.write(res)
                if i == mid_i or mid is None:
                    mid = (frame, res)
                n += 1
                if (i + 1) % max(1, total // 10) == 0:
                    core.log(
                        f"  {i + 1}/{total} コマ（{(time.time() - t0) / (i + 1):.2f} 秒/コマ）"
                    )
        assert mid is not None
        before, after = mid
        if mode == "yuv":
            before, after = _yuv_to_bgr(before), _yuv_to_bgr(after)
    dt = time.time() - t0
    core.log(f"{n} コマ、{dt / n:.2f} 秒/コマ")
    sheet = _compare([before, after], ["元", f"ブレ除去（{a.method}）"])
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "model": "NAFNet-GoPro-width64 (MIT)" if net is not None else None,
            "estimate": est,
            "estimates_per_frame": ests if video else None,
            "psf": psf_info,
            "params": params,
            "frames": n,
            "sec_per_frame": round(dt / n, 3),
        },
    )


def _merge_estimates(ests: list[dict[str, Any]]) -> dict[str, Any]:
    """複数のコマの推定をまとめる（多数決の種類と、その種類の中央値）。"""
    kinds = [e["kind"] for e in ests]
    kind = max(set(kinds), key=kinds.count)
    same = [e for e in ests if e["kind"] == kind] or ests
    out = dict(same[len(same) // 2])
    out["kind"] = kind
    for key in ("length", "radius", "motion_score", "disk_score"):
        out[key] = round(float(np.median([e[key] for e in same])), 2)
    # 角度は 180 度で一周するので、2 倍の角度で平均する
    th = np.radians([e["angle"] * 2 for e in same])
    out["angle"] = round(
        float(np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) / 2 % 180), 1
    )
    out["frames"] = len(ests)
    return out


def _describe_blur(est: dict[str, Any]) -> str:
    if est["kind"] == "motion":
        return f"推定したぶれ: 動き 長さ {est['length']:.1f}px・向き {est['angle']:.0f} 度（強さ {est['motion_score']:.1f}）"
    if est["kind"] == "defocus":
        return f"推定したぶれ: ピンぼけ 半径 {est['radius']:.1f}px（強さ {est['disk_score']:.1f}）"
    return "推定したぶれ: はっきりしたぶれは見つかりませんでした"


def _wiener_k(sigma: float) -> float:
    """ノイズの強さ σ（0〜255）→ ウィーナーの K。較正（正しい PSF・7 枚 × ぶれ 4 種）で PSNR が最良だった
    K（σ0.5: 0.0056、1: 0.010、2: 0.018、4: 0.032）に合わせた式。"""
    # 圧縮された実写はノイズの推定が 0 になることがあり、小さい K では JPEG・H.264 の傷が波紋とざらつきになった
    # （text_defocus・licenseplate・text_motion を目で比べた）ので、σ は最低 1 として扱う（較正の σ0.5 では −0.13 dB）
    return float(0.01 * max(sigma, 1.0) ** 0.83)


def _rl_iters(sigma: float) -> int:
    """ノイズの強さ σ → リチャードソン・ルーシーの回数。較正で最良だった回数（σ0.5: 70、1: 45、2: 30、4: 15）に合わせた式。"""
    return int(
        np.clip(round(45 * max(sigma, 1.0) ** -0.75), 8, 80)
    )  # σ の下限は _wiener_k と同じ


def _yuv_to_bgr(yuv: np.ndarray) -> np.ndarray:
    """_read の yuv（tv の範囲、BT.601 として）→ BGR（確認画像用）。"""
    y = (yuv[:, :, 0].astype(np.float32) - 16) * 255 / 219
    u = (yuv[:, :, 1].astype(np.float32) - 128) * 255 / 224
    v = (yuv[:, :, 2].astype(np.float32) - 128) * 255 / 224
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    return np.clip(np.dstack([b, g, r]) + 0.5, 0, 255).astype(np.uint8)


# ── 39 アンシャープマスク ─────────────────────────────────────────────────

_SHARPEN = {
    "usm": "アンシャープマスク（OpenCV。明るさだけ。小さな差＝ノイズを強めない下限と、縁の白い輪を抑える上限つき。既定）",
    "cas": "ffmpeg の cas（AMD の Contrast Adaptive Sharpening。明るさだけ。試験では効きが弱く、強さ 1.0 で平らな所が壊れたので 0.9 まで）",
    "unsharp": "ffmpeg の unsharp（素のアンシャープマスク。明るさだけ。下限・上限なし）",
}


def _usm(
    y: np.ndarray, radius: float, amount: float, threshold: float, halo: float | None
) -> np.ndarray:
    """明るさ（float32）にアンシャープマスクをかける。

    差（元 − ぼかし）のうち threshold より小さい物は滑らかに弱め（ノイズを強めない）、結果はまわりの最小〜最大から
    halo 段より外に出さない（縁の白い輪・黒い輪を抑える）。halo=None なら抑えない。
    """
    blur = cv2.GaussianBlur(y, (0, 0), radius)
    d = y - blur
    if threshold > 0:
        d = d * (1 - np.exp(-((d / threshold) ** 2)))
    out = y + amount * d
    if halo is not None:
        k = 2 * math.ceil(radius) + 1
        ker = np.ones((k, k), np.uint8)
        lo, hi = cv2.erode(y, ker), cv2.dilate(y, ker)
        out = np.clip(out, lo - halo, hi + halo)
    return out


def _sharpen_filter(method: str, radius: float, amount: float) -> str:
    """ffmpeg の方法の設定（明るさの面だけ）。"""
    if method == "cas":
        return f"cas=strength={min(0.9, max(0.0, amount)):.3f}:planes=1"
    size = int(np.clip(2 * round(2 * radius) + 1, 3, 23))
    return f"unsharp=luma_msize_x={size}:luma_msize_y={size}:luma_amount={amount:.3f}:chroma_amount=0"


def _ffmpeg_gray(y: np.ndarray, vf: str) -> np.ndarray:
    """明るさの面（uint8）1 枚に ffmpeg のフィルターをかける。"""
    h, w = y.shape
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-f", "rawvideo"]
    cmd += ["-pix_fmt", "gray", "-s", f"{w}x{h}", "-i", "-", "-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
    r = subprocess.run(
        cmd, input=np.ascontiguousarray(y).tobytes(), capture_output=True, check=False
    )
    if r.returncode != 0 or len(r.stdout) < w * h:
        raise core.VfxError(
            "ffmpeg が失敗しました: " + vf + "\n" + r.stderr.decode()[-800:]
        )
    return np.frombuffer(r.stdout, np.uint8)[: w * h].reshape(h, w).copy()


@command(
    39,
    "sharpen",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=list(_SHARPEN),
        default="usm",
        help="方法。" + " ／ ".join(f"{k}: {v}" for k, v in _SHARPEN.items()),
    ),
    arg(
        "--amount",
        type=float,
        default=1.2,
        help="強さ（既定 1.2。少しだけなら 0.5〜0.8。cas は 0〜0.9）",
    ),
    arg(
        "--radius",
        type=float,
        default=1.0,
        help="ぼかしの半径 σ（px。既定 1.0。大きいほど太い輪郭に効く）",
    ),
    arg(
        "--threshold",
        type=float,
        help="これより小さい差は強めない（0〜255 の段。省略時はノイズの強さの 2 倍、最低 1）",
    ),
    arg(
        "--halo",
        type=float,
        default=0.0,
        help="縁の白い輪・黒い輪の上限（まわり（半径 1 なら 3×3）の明るさの範囲から外へ出してよい段。既定 0 = 出さない。"
        "-1 で抑えない）",
    ),
)
def sharpen(a: argparse.Namespace) -> None:
    """輪郭をくっきりさせる（明るさだけ。ノイズを強めない下限と、縁の白い輪を抑える上限つき）。"""
    info = core.probe(a.src)
    core.require(
        a.amount > 0 and a.radius > 0,
        "--amount と --radius は 0 より大きくしてください",
    )
    out, json_path, sheet_path = _paths(a, info, "sharpen")
    video = not info.is_image
    if video:
        samples = [f[:, :, 0] for f in _sample_frames(a, info, 5, "yuv")]
    else:
        samples = [_ycc(_load(a, info))[:, :, 0]]
    sigma = float(np.median([_noise_sigma(s.astype(np.float32)) for s in samples]))
    # 下限はノイズの強さの 2 倍。圧縮された素材はノイズを測れない（0 になる）ことがあるので、最低 1 段にする
    threshold = a.threshold if a.threshold is not None else max(2.0 * sigma, 1.0)
    halo = None if a.halo < 0 else a.halo
    lo, hi = _y_range(video)
    vf = _sharpen_filter(a.method, a.radius, a.amount) if a.method != "usm" else None
    core.log(
        f"明るさのノイズ σ {sigma:.2f}、方法 {a.method}"
        + (
            f"（強さ {a.amount:g}・半径 {a.radius:g}・下限 {threshold:.2f}・上限 {halo}）"
            if vf is None
            else f"（{vf}）"
        )
    )

    def fix(y: np.ndarray) -> np.ndarray:
        if vf is not None:
            return _ffmpeg_gray(y, vf)
        r = _usm(y.astype(np.float32), a.radius, a.amount, threshold, halo)
        return np.clip(np.round(r), lo, hi).astype(np.uint8)

    t0 = time.time()
    if not video:
        src = _load(a, info)
        ycc = _ycc(src)
        ycc[:, :, 0] = fix(ycc[:, :, 0])
        res = _from_ycc(ycc, src)
        _save_image(out, res)
        before, after, n = src[:, :, :3], res[:, :, :3], 1
    else:
        size = _proc_size(info, a.max_side)
        total = max(1, round(_span(a, info) * info.fps))
        mid_i, mid, n = total // 2, None, 0
        with _Writer(
            out, info.fps, _color(info), mode="yuv", audio=_audio(a, info)
        ) as wr:
            for i, frame in enumerate(_read(a, info, "yuv", size)):
                res = frame.copy()
                res[:, :, 0] = fix(frame[:, :, 0])
                wr.write(res)
                if i == mid_i or mid is None:
                    mid = (frame, res)
                n += 1
        assert mid is not None
        before, after = _yuv_to_bgr(mid[0]), _yuv_to_bgr(mid[1])
    dt = time.time() - t0
    yb, ya = _luma_f(before), _luma_f(after)
    edge = np.abs(cv2.Laplacian(yb, cv2.CV_32F))
    flat = edge < np.percentile(edge, 30)
    stats = {
        "edge_gain": round(
            float(
                np.mean(np.abs(cv2.Laplacian(ya, cv2.CV_32F))[~flat])
                / max(1e-6, np.mean(edge[~flat]))
            ),
            3,
        ),
        "flat_noise_before": round(
            float(np.std((yb - cv2.GaussianBlur(yb, (0, 0), 2))[flat])), 3
        ),
        "flat_noise_after": round(
            float(np.std((ya - cv2.GaussianBlur(ya, (0, 0), 2))[flat])), 3
        ),
    }
    core.log(
        f"{n} コマ、{dt / n:.3f} 秒/コマ。確認したコマ: 輪郭の強さ ×{stats['edge_gain']}、平らな所のざらつき "
        f"{stats['flat_noise_before']} → {stats['flat_noise_after']}"
    )
    score = np.abs(cv2.Laplacian(yb, cv2.CV_32F))
    box = _box_at(score, min(before.shape[:2]) // 5)
    sheet = _compare([before, after], ["元", f"シャープ（{a.method}）"], box)
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "method": a.method,
            "amount": a.amount,
            "radius": a.radius,
            "threshold": round(threshold, 3),
            "halo": halo,
            "filter": vf,
            "noise_sigma": round(sigma, 3),
            "check": stats,
            "frames": n,
            "sec_per_frame": round(dt / n, 4),
        },
    )


# ── 50 フィルムグレイン ───────────────────────────────────────────────────


def _acf_fwhm(sigma: float) -> float:
    """白色ノイズを σ のガウス（離散）でぼかした時の、自己相関の半値全幅（px。整数のずれの値を直線でつないで測る）。"""
    k = 2 * math.ceil(4 * sigma) + 1
    g = cv2.getGaussianKernel(k, sigma).ravel()
    r = np.correlate(g, g, "full")[k - 1 :] / float(g @ g)
    j = int(np.argmax(r < 0.5))
    return float(2 * (j - 1 + (r[j - 1] - 0.5) / (r[j - 1] - r[j])))


@cache
def _grain_sigma(size: float) -> float:
    """粒の大きさ（自己相関の半値全幅、px）→ 白色ノイズをぼかすガウスの σ。

    連続の式（半値全幅 ≒ 3.33σ）は 2px 未満で合わない（1.5 を頼むと 1.20px になった）ので、離散の自己相関で合わせる。
    """
    lo, hi = 0.05, max(0.1, size)
    for _ in range(40):
        mid = (lo + hi) / 2
        if _acf_fwhm(mid) < size:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _grain_field(
    shape: tuple[int, int], size: float, rng: np.random.Generator
) -> np.ndarray:
    """平均 0・標準偏差 1 の粒（size は粒の大きさ＝自己相関の半値全幅、px。1 以下は 1 画素ごとの白色ノイズ）。"""
    n = rng.standard_normal(shape, dtype=np.float32)
    if size <= 1.0:
        return n
    s = _grain_sigma(round(size, 3))
    k = 2 * math.ceil(4 * s) + 1
    g = cv2.getGaussianKernel(k, s).astype(np.float32)
    n = cv2.sepFilter2D(n, -1, g, g, borderType=cv2.BORDER_REFLECT)
    return n / float(
        (g**2).sum()
    )  # ぼかした後の標準偏差（Σg² の 2 乗の平方根 = Σg²）で割って 1 に戻す


def _grain_weight(yn: np.ndarray, flat: bool) -> np.ndarray:
    """明るさ（0〜1）ごとの粒の強さの倍率。中間で 1、真っ黒・真っ白で 0.35（フィルムの粒は中間調で目立つ）。"""
    if flat:
        return np.ones_like(yn)
    return 0.35 + 0.65 * 4 * yn * (1 - yn)


@command(
    50,
    "grain",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--strength",
        type=float,
        default=4.0,
        help="粒の強さ（中間の明るさでの標準偏差。0〜255 の段。既定 4）",
    ),
    arg(
        "--size",
        type=float,
        default=1.5,
        help="粒の大きさ（px。1 で 1 画素ずつ、2〜3 で粗い。既定 1.5）",
    ),
    arg(
        "--chroma",
        type=float,
        default=0.0,
        help="色の粒の強さ（明るさの粒に対する割合 0〜1。既定 0 = 明るさだけ）",
    ),
    arg(
        "--flat",
        action="store_true",
        help="明るさによらず同じ強さにする（既定は中間調で強く、黒・白で弱く）",
    ),
    arg(
        "--static",
        action="store_true",
        help="毎コマ同じ粒にする（既定はコマごとに変わる）",
    ),
    arg("--seed", type=int, default=0, help="乱数の種（同じ種なら同じ粒）"),
)
def grain(a: argparse.Namespace) -> None:
    """フィルムのような粒子を足す（明るさだけ・粒の大きさ・コマごとに変わる）。素材のファイルに焼き込む用（HyperFrames の作品の中では media-treatment の grain を使う）。"""
    info = core.probe(a.src)
    core.require(
        a.strength >= 0 and a.size > 0,
        "--strength は 0 以上、--size は 0 より大きくしてください",
    )
    core.require(0 <= a.chroma <= 1, "--chroma は 0〜1 にしてください")
    out, json_path, sheet_path = _paths(a, info, "grain")
    video = not info.is_image
    lo, hi = _y_range(video)

    def add(frame: np.ndarray, i: int) -> np.ndarray:
        """Y・Cr・Cb（動画は Y・U・V）の 3 面に粒を足す。"""
        rng = np.random.default_rng([a.seed, 0 if a.static else i])
        h, w = frame.shape[:2]
        y = frame[:, :, 0].astype(np.float32)
        yn = np.clip((y - lo) / (hi - lo), 0, 1)
        amp = a.strength * _grain_weight(yn, a.flat)
        res = frame.copy()
        res[:, :, 0] = np.clip(
            np.round(y + amp * _grain_field((h, w), a.size, rng)), lo, hi
        ).astype(np.uint8)
        if a.chroma > 0:
            for c in (1, 2):
                v = frame[:, :, c].astype(np.float32) + a.chroma * amp * _grain_field(
                    (h, w), a.size, rng
                )
                res[:, :, c] = np.clip(
                    np.round(v), lo, 240.0 if video else 255.0
                ).astype(np.uint8)
        return res

    t0 = time.time()
    if not video:
        src = _load(a, info)
        ycc = _ycc(src)
        # YCrCb の並びは Y・Cr・Cb なので、色の面の扱いは動画の Y・U・V と同じでよい
        res = _from_ycc(add(ycc, 0), src)
        _save_image(out, res)
        before, after, n = src[:, :, :3], res[:, :, :3], 1
    else:
        size = _proc_size(info, a.max_side)
        total = max(1, round(_span(a, info) * info.fps))
        mid_i, mid, n = total // 2, None, 0
        # 粒は H.264 でつぶれやすいので -tune grain を付ける（粒の残り方は NOTES の measured）
        with _Writer(
            out,
            info.fps,
            _color(info),
            mode="yuv",
            audio=_audio(a, info),
            x264=("-tune", "grain"),
        ) as wr:
            for i, frame in enumerate(_read(a, info, "yuv", size)):
                res = add(frame, i)
                wr.write(res)
                if i == mid_i or mid is None:
                    mid = (frame, res)
                n += 1
        assert mid is not None
        before, after = _yuv_to_bgr(mid[0]), _yuv_to_bgr(mid[1])
    dt = time.time() - t0
    d = _luma_f(after) - _luma_f(before)
    core.log(
        f"{n} コマ、{dt / n:.3f} 秒/コマ。確認したコマの足した粒: 標準偏差 {d.std():.2f}・平均 {d.mean():+.2f}"
    )
    yb = _luma_f(before)
    # 粒が見やすい所（平らで中間の明るさ）を拡大する
    flatness = -cv2.blur(np.abs(cv2.Laplacian(yb, cv2.CV_32F)), (9, 9)) - 0.2 * np.abs(
        yb - 128
    )
    score = flatness - flatness.min()
    hh, ww = score.shape
    edge = np.zeros_like(score, bool)
    edge[: hh // 8], edge[-hh // 8 :], edge[:, : ww // 8], edge[:, -ww // 8 :] = (
        True,
    ) * 4
    score[edge] = 0  # 端（見出しの下など）は選ばない
    box = _box_at(score, min(before.shape[:2]) // 6)
    sheet = _compare(
        [before, after],
        ["元", f"粒子（強さ {a.strength:g}・大きさ {a.size:g}）"],
        box,
        enhance=3.0,
    )
    _finish(
        out,
        json_path,
        sheet_path,
        sheet,
        {
            "source": str(a.src),
            "output": str(out),
            "strength": a.strength,
            "size": a.size,
            "chroma": a.chroma,
            "flat": a.flat,
            "static": a.static,
            "seed": a.seed,
            "check": {
                "added_std": round(float(d.std()), 3),
                "added_mean": round(float(d.mean()), 3),
            },
            "frames": n,
            "sec_per_frame": round(dt / n, 4),
        },
    )
