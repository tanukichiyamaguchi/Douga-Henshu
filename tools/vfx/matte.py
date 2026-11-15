"""切り抜き・合成（切り分け）。

担当するテクニック:
   51 セグメンテーション（segment）
   53 ロトスコープ（rotoscope）
   56 マッティング（matte）
   60 マスクトラッキング（mask-track）
   61 インスタンス分割（instances）
   73 パノプティック分割（panoptic）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

import argparse
import colorsys
import os
import time
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import core, models
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/matte/（make_synth.py で正解つきの合成素材を作り、test_*.py で照らす。
# review_*.py で確認画像を作る）。合成素材の人物はすべて AI 生成の広告映像（plate-*・cut-*）から取った。
# 正解つきの合成素材（<スクラッチパッド>/vfx-test/synth/matte/）:
#   comp-move-clean / comp-move-crowd: 女性（plate-stand と cut-stand のアルファ）が vtest の上を横に動く 48 コマ。
#     crowd は人が歩く元のコマの上で、手前の柱（幅 36px）の後ろを通る。正解 = 貼ったアルファ
#   comp-multi: 人のいない vtest（60 コマの中央値）に 3 人を一部重ねて貼った画。正解 = 1 人ずつの地図
#   comp-hair-sky / -street: 女性の頭の周りに、太さ 0.5〜1.5px のはねた毛 70 本（4 倍で描いて縮めた正確なアルファ）
NOTES: dict[int, dict[str, str]] = {
    51: {
        "status": "◎",
        "how": "3 つのモデルから選ぶ。scene（既定）: UperNet ConvNeXt small の ADE20K 150 種類"
        "（openmmlab/upernet-convnext-small、MIT。空・壁・床・木・道路・人・車など。名前は日本語に）。縦横比を保って"
        "短い辺 512px で計算し（32 の倍数に詰める）、1/4 の大きさの確率から、出てきた種類だけを拡大して塗り分ける。"
        "person: MediaPipe の selfie_segmenter（Apache-2.0。人物だけ・速い）。parts: MediaPipe の selfie_multiclass"
        "（Apache-2.0。背景・髪・体の肌・顔の肌・服・小物）。動画は、前のコマの確率を DIS オプティカルフローで動かして"
        "今のコマと混ぜ（--smooth 0.5。動きの推定が外れた所は混ぜない）、ちらつきを抑える。出力: 色分けの重ね表示"
        "（MP4／JPG。凡例は日本語名、画像は面積の % も）、--cutout（選んだ種類だけを残した透過 WebM／PNG。確率の和を"
        "ガイデッドフィルターで元の画の輪郭に合わせる）、--masks（種類ごとの白黒 PNG／MP4）、JSON（コマごとの面積と平均）",
        "measured": "3 人を貼った合成（正解: 人の地図）の人の IoU: scene 0.980（--cutout。種類の地図では 0.962）・"
        "person 0.970・parts 0.950（背景以外すべて）。--cutout のガイデッドフィルターを半径 = 短い辺の 1/70・eps 0.01 に"
        "すると、境目の IoU（r=3px）が scene 0.689 → 0.742・person 0.642 → 0.687（3 人の合成と動く合成 4 コマ）。"
        "動く合成（正解アルファ）: scene の人のマスク動画 12 コマで IoU 平均 0.988（最小 0.985）、person の切り抜き"
        " 48 コマで 0.976（最小 0.914。髪がなびくコマで下がる）。"
        "ちらつき（固定カメラの vtest 30 コマで、画がほとんど変わらない画素の種類が隣のコマで変わった割合）: --smooth 0 で"
        " 1.14%、0.5 で 0.75%、0.8 で 0.48%。動く合成の人の IoU は 0.9866・0.9865・0.9858 で、なめらかさによる遅れは"
        "見られなかった。home.jpg: 空 52.6%・建物 47.4%（73 の DETR と空の IoU 0.96・建物 0.979）。salon.webp（解析のみ）:"
        " カーテン 69%（店の糸のカーテン。正しい）・壁 13%・床 13%・天井 3%・ひじ掛け椅子 2%。story のカフェ: 人 51%・木 37%"
        "・台座 10%（テーブルを「台座」と誤る）・椅子 1.4%。速さ（540×960）: scene 1 コマ 1.8〜2.2 秒（他の作業が重い時"
        " 5.8〜6.4 秒）・person 0.015 秒・parts 0.1 秒。メモリ（scene）約 1.5GB",
        "limits": "scene の種類は ADE20K の 150 種類で、似た物を取り違えることがある（カフェのテーブル → 台座）。人の輪郭は"
        "どのモデルも髪の毛までは追わない（3 人の合成の境目の IoU〔r=3px〕で --cutout が scene 0.71・person 0.66・parts 0.69）。"
        "髪の毛まで要る時は 56。scene は"
        " 1 コマ約 2 秒なので、15 秒・24fps の動画で約 12 分。動画の --masks は、--classes を省くと最初のコマに出た種類"
        "（最大 10）だけを書く（後から出てくる種類は書かれない）。parts はカメラに近い人向け（自撮り用のモデル）",
        "tested": "make_synth.py の comp-multi・comp-move-clean（正解つき）、vtest 30 コマ（ちらつき）、home.jpg・salon.webp・"
        "story のカフェ（test_segment.py。目で確認: vfx-review/51-segment.jpg）",
    },
    53: {
        "status": "◎",
        "how": "毎コマ人物の地図を求め（selfie: MediaPipe の動画モード〔既定〕／ vitmatte: 56 と同じ髪まで細かい切り抜き"
        "／ birefnet: BiRefNet／ --mask: 60 などの白黒マスク）、前のコマの地図を DIS オプティカルフローで動かして混ぜ"
        "（--smooth 0.6）、ガイデッドフィルター（半径 = 短い辺の 1/70、eps 0.01）で元の画の境目に合わせてから 0.5 で切る。"
        "小さな塊と小さな穴を消し、縁を 1px ぼかす。出力: 透過 WebM（VP9）、確認 MP4（外を暗くし、輪郭をマゼンタの線で"
        "描く）、JSON（コマごとの輪郭の多角形。approxPolyDP で間引き、元の動画の座標、穴は hole=true）。画像は PNG・JPG・JSON",
        "measured": "動く合成 48 コマ（正解アルファ）: 既定（selfie）IoU 平均 0.976（最小 0.915）・境目の IoU（r=3px）0.70・"
        "時間方向のふるえ 6.8×10⁻³。手順を比べると、地図を 0.5 で切っただけは IoU 0.972・境目 0.648・ふるえ 8.8×10⁻³、"
        "ガイデッド（半径 8・eps 0.01）を足すと境目 0.70・ふるえ 7.2×10⁻³（半径 4・eps 0.001 では 0.639 で悪くなる）、"
        "なめらか 0.6 を足すと 6.8×10⁻³（0.8 で 6.6×10⁻³）。--model vitmatte: IoU 0.993・境目 0.833・ふるえ 3.1×10⁻³"
        "（1 コマ 1.3 秒）。birefnet（20〜31 コマの 12 コマ）: IoU 0.985・境目 0.774（1 コマ 10.4 秒。同じコマで selfie は"
        " 0.946・0.662、vitmatte は 0.987・0.806）。JSON の多角形を塗り直すと WebM の 2 値の地図と IoU 0.998（最小 0.997。"
        "1 コマ平均 46 点）。HyperFrames の切り抜き（cut-*.webm）との一致: stand 50 コマ 0.973（最小 0.941）、laugh 45 コマ"
        " 0.928（最小 0.908。HF 側は机と画面の端の人まで含む）。60 のマスク動画を --mask で渡すと（--model vitmatte・8 コマ）"
        " IoU 0.992。速さ: selfie は 540×960・48 コマで約 5〜6 秒（書き出しを含む）",
        "limits": "既定の selfie は、風になびく髪の束を切り落とす（合成で IoU が 0.915 まで下がったコマ。--model vitmatte"
        " なら残る）。人以外は --mask（60 の出力）か --model birefnet（1 コマ約 10 秒・メモリ約 7GB）。出力は 1px ぼかした"
        "くっきりした輪郭で、半透明の毛先は出さない（半透明が要る時は 56）。多角形はコマごとに独立で、コマ間の点の"
        "対応（After Effects のロトのようなキーフレームの曲線）は作らない",
        "tested": "make_synth.py の comp-move-clean（正解つき。test_rotoscope.py・test_rotoscope_models.py）、plate-stand・"
        "plate-laugh と cut-*-540.webm、comp-move-crowd と 60 のマスク（test_chain.py）、画像 1 枚（comp-hair-street.png）。"
        "目で確認: vfx-review/53-rotoscope.jpg",
    },
    56: {
        "status": "◎",
        "how": "vitmatte（既定）: MediaPipe の人物と髪の地図から 3 値の地図（trimap）を作り（境目の内外 = 短い辺の 1.2%、"
        "地図が 0.1〜0.9 の自信のない所、髪の周り 10 倍の幅を「分からない」に）、ViTMatte small（hustvl/vitmatte-small-"
        "composition-1k、Apache-2.0）でアルファを求める。長い辺が 1280px を超える画は縮めて計算し、ガイデッドフィルターで"
        "戻す。続けて縁の色を推定し直す（Forte・Pitié 2021 の Blur-Fusion、2 段）。birefnet: BiRefNet lite の ONNX"
        "（rembg の配布、MIT。1024×1024 固定・ImageNet の正規化・onnxruntime の作業メモリの使い回しを切る）。"
        "birefnet-portrait: 同じく人物用（MIT）。guided: 人物の地図をガイデッドフィルターで整えるだけ。--mask で人以外も"
        "（60・53 の出力を渡せる）。出力: 透過 PNG（画像）／透過 WebM（動画）、--alpha-out（アルファだけ）、確認画像"
        "（元・アルファ・市松模様と緑に重ねた画・半透明が一番多い所の拡大）、JSON",
        "measured": "はねた毛の合成（正解アルファ。頭の周りで測る。数字は 空 / 街の背景）: vitmatte SAD 0.99 / 1.94・境目の"
        "帯の MSE 0.0018 / 0.0082・勾配の差 0.24 / 1.19・境目の IoU 0.84 / 0.80・毛のアルファの回収 85% / 78%"
        "（余計 9% / 22%）。髪の周りを広げる前（帯 6px だけ）は SAD 3.13 / 3.38・毛 19% / 22% で、MediaPipe の髪の地図は"
        "毛の画素の 2% 未満しか含まなかった。birefnet: SAD 1.96 / 2.74・境目 0.70 / 0.63・毛 79% / 50%。birefnet-"
        "portrait: SAD 1.83 / 2.91・毛 70% / 38%。guided: SAD 4.02 / 5.16（元の地図 4.10 / 5.03 と変わらない）。縁の色の"
        "推定: 別の背景に重ね直した時の色の差（0〜255、境目の帯）が、正解のアルファで 5.76→2.27 / 4.70→2.17、vitmatte の"
        "アルファで 6.68→3.32 / 6.67→4.37（推定は 1 コマ約 0.05 秒）。実写（AI 生成）の風になびく髪: vitmatte と birefnet は"
        "毛先まで残り、guided はにじむ（目で確認）。60 のマスクを --mask で渡した動画: IoU 0.993（8 コマ）。速さとメモリ:"
        " vitmatte 540×960 で 1 コマ 1.0〜1.9 秒・約 1.4GB、1080×1920 で 3.9 秒・約 2.0GB（540 の結果と IoU 0.992）。"
        "birefnet 10〜11 秒・最大 6.7GB（作業メモリの使い回しを入れたままだと 2 回目で 12GB）、birefnet-portrait"
        " 17〜18 秒・最大 8.0GB（入れたままでは途中でメモリ不足で止められた）、guided 0.03 秒",
        "limits": "vitmatte の粗い地図は MediaPipe の人物なので、人以外は --mask が要る（人物の地図が椅子なども含むと、"
        "それも残る）。髪の周りを広く「分からない」にするので、背景に細い線（街灯・電線）が多いと少し拾う（街の背景で"
        "余計 22%。--hair-band を小さくすると減る）。動画はコマごとの計算（ちらつきは --smooth で抑える）。birefnet は"
        " 7〜8GB 使うので他の重い作業と同時に動かさない。合成の毛は黒い細い線で、実際の細い毛とは見え方が違う",
        "tested": "make_synth.py の comp-hair-sky・comp-hair-street（正解のアルファと前景の色つき。test_matte.py・"
        "review_matte.py）、plate-stand（1.0・1.9 秒）・plate-cafe（1.2 秒）の実写（review_matte_real.py）、1080×1920 の画、"
        "動画（plate-laugh 3 コマ、comp-move-crowd と 60 のマスク）。目で確認: vfx-review/56-matte-*.jpg",
    },
    60: {
        "status": "◎",
        "how": "SAM 2.1 tiny（facebook/sam2.1-hiera-tiny、Apache-2.0）の動画版を transformers 5.2 の Sam2VideoModel で使う。"
        "1 コマずつ渡す逐次入力にし、覚える物を直近 16 コマと最初のコマに絞る（メモリが増え続けない）。最初のコマの指定:"
        " --box（枠）／ --point と --neg（点）／ --mask（白黒か透過つきの画像）／ 省略時は MediaPipe で一番大きい人物。"
        "出力のマスクはガイデッドフィルターで輪郭を合わせ、1px ぼかす。出力: 透過 WebM、確認 MP4（色・輪郭・枠・"
        "「見失った」）、--mask-out（白黒のマスク動画。53・56 の --mask に渡せる）、JSON（コマごとの有無・確からしさ・"
        "面積・枠）",
        "measured": "人が歩く vtest の上で女性が動き、柱の後ろを通る合成 48 コマ（正解アルファ、最初のコマの正解の枠で指定）:"
        " IoU 平均 0.990（最小 0.974）、見失い 0、柱の所に残したマスク 0 画素。人のいない背景の合成（指定なし = 自動）:"
        " 0.992（最小 0.979）。vtest の実写で歩く 1 人を 60 コマ（6 秒）: 見失い 0、5 コマごとの Faster R-CNN の人の検出との"
        "枠の IoU 0.876〜0.951（平均 0.913）。story の手（点で指定・24 コマ、腕を上げて髪を結ぶ速い動き）: 見失わず、"
        "手と前腕を頭の上まで追った（目で確認）。透過つき PNG を --mask で渡した時: HyperFrames の切り抜きと IoU"
        " 0.987〜0.999（4 コマ）。速さ: 1 コマ 2.2〜5.4 秒（768×576・540×960。他の作業の重さで変わる）、メモリ約 1.5GB",
        "limits": "遅い（10 秒・24fps の動画で 8〜20 分）。点の指定は SAM が「物」と考える範囲になる（手を指すと前腕まで"
        "入った。--neg か --box で絞る）。1 回に追えるのは 1 つ。画面から出て戻ってきた物を拾い直せるかは確かめていない。"
        "最初のコマ（--start の位置）に物が写っている必要がある",
        "tested": "make_synth.py の comp-move-crowd・comp-move-clean（正解つき）、vtest（Faster R-CNN の検出と照合）、"
        "story540 の 4.0 秒からの手（test_masktrack.py）、plate-stand と透過 PNG の指定。目で確認: vfx-review/60-mask-track.jpg",
    },
    61: {
        "status": "◎",
        "how": "torchvision の Mask R-CNN v2（maskrcnn_resnet50_fpn_v2、COCO の 80 種類、BSD-3-Clause。重みは"
        " download.pytorch.org）。短い辺を 512〜800px にして計算し、確からしさ 0.6 以上を採用。同じ種類で、小さい方の"
        "マスクの 70% 以上が大きい方に含まれる物は外す（隠れた人を上半身と全身の 2 人に数えるのを防ぐ）。重ね表示は"
        "番号ごとの色と「人 #3 0.98」の札。動画は前のコマの枠との重なり（0.2 以上・同じ種類）で同じ番号を付け続ける。"
        "出力: 重ね表示（MP4／JPG）、JSON（番号・種類・日本語名・確からしさ・元の座標の枠・面積）、--masks（画像: 1 つずつの"
        "白黒マスクと透過 PNG）、--split N（動画: 番号ごとの透過 WebM）",
        "measured": "3 人を一部重ねて貼った合成（正解: 1 人ずつの地図）: 3/3 人、1 人ずつの IoU 0.956・0.938・0.864"
        "（3 人目の正解には元の切り抜きの机が入っている）。ほかに誤り 5 件（机 → 本、街灯 → フリスビー、バン → トラック、"
        "ベンチ、車）。mp-cats_and_dogs.jpg: 猫 2・犬 2（正しい）。vtest 10 秒: 人 7/7（目で数えた 7 人）。含まれる物を"
        "外す前は、看板の後ろの人を 2 人に数えて 8 だった。story のカフェ: 人 1 は正しく、左端の暗い背景を人と誤った"
        "（0.75）。動画（vtest 30 コマ・7 人）: 番号 11 個、5 人は全コマ（29〜30）同じ番号、隠れた人は番号が変わった。"
        "速さ: 768×576 で 1 コマ 1.3 秒（他の作業が重い時 7.9 秒）、1200×600 の画像 2.2 秒、メモリ約 0.9GB",
        "limits": "COCO の 80 種類だけ。遠くの小さな物は見逃す。物でない物を物と誤ることがある（上の例）。--score を上げると"
        "減る。動画の番号付けは前のコマとの重なりだけの簡易な方法で、隠れたりすれ違ったりすると番号が変わる（しっかり"
        "追うなら 15 multi-track）。マスクの輪郭は粗い（髪などは 56 の --mask に渡して整える）",
        "tested": "make_synth.py の comp-multi（正解つき）、mp-cats_and_dogs.jpg、vtest（静止画と 30 コマの動画。--split 3）、"
        "story540 の 13 秒、salon.webp（解析のみ）（test_instances_panoptic.py。目で確認: vfx-review/61-instances.jpg）",
    },
    73: {
        "status": "○",
        "how": "DETR（ResNet-50）のパノプティック分割（facebook/detr-resnet-50-panoptic、Apache-2.0）。COCO panoptic の"
        " 133 種類（物 80・背景 53。背景の番号と名前は detectron2 の一覧で確認し、日本語名を付けた）。同じ種類の背景は"
        "まとめる。transformers 5.2 は背骨の ResNet-50 を timm で作るが timm が無いので、同じ形の torchvision の ResNet-50 に"
        "同じ重みを入れて差し替えた（timm を一時的に入れた公式の読み込みと比べ、マスクの値の差は最大 0.004、結果の塊は"
        "同じ）。さらに採用する候補にだけマスクを計算する（100 個すべてだと 800px の画で 12.4GB 使った。結果は普通の計算と"
        "画素単位で一致）。出力: 凡例つきの重ね表示（背景は面積の %、物は数）、JSON（塊ごとの種類・物か背景か・確からしさ・"
        "面積・枠）、--masks",
        "measured": "3 人の合成: 3/3 人、IoU 0.978・0.972・0.856。速くした計算と普通の計算の一致: home.jpg・合成とも"
        "塊の数が同じで画素の一致率 1.000。51（UperNet）との背景の一致: home.jpg の空 IoU 0.960・建物 0.979、合成の草 0.83・"
        "道路 0.68・建物 0.70。home.jpg: 空 52%・建物 47%・鳥（空を飛ぶ本物の鳥）。vtest 10 秒: 人 6/7（看板の後ろの人を"
        "見逃し）、--threshold 0.7 では 7/7 ＋誤り 1。mp-cats_and_dogs.jpg: 犬 2 匹を見逃した（0.7 でも同じ。0.5 では犬を"
        "猫と誤った）。salon.webp（解析のみ）: 糸のカーテンを床（82%）と誤った（51 はカーテンと正しく判定）。story の"
        "カフェ: 人・テーブル・椅子・木 41%。速さ: 540×960 で 1 コマ 0.56 秒、768×576 で 0.75 秒（他の作業が重い時"
        " 3.4 秒）、メモリ約 1GB",
        "limits": "DETR（2020 年）は今の手法より精度が低い（犬を見逃す・糸のカーテンを床と誤る・草と道路の境目が粗い）。"
        "より新しい Mask2Former の公開の重みはライセンスが other で使えず、OneFormer（MIT）の大きいモデルは CPU では"
        "重すぎるので入れていない。背景の種類を細かく知りたい時は 51（150 種類）の方が当たる。物の番号はコマをまたいで"
        "追わない。既定の --threshold 0.85 は誤りを減らす側（0.7 で隠れた人も拾うが誤りが増える）",
        "tested": "make_synth.py の comp-multi（正解つき）、home.jpg、vtest 10 秒、mp-cats_and_dogs.jpg、story540 のカフェ"
        "（静止画と 6 コマの動画）、salon.webp（解析のみ）、timm を使う公式の読み込みとの比較（uv run --with timm で一時的に）"
        "（test_instances_panoptic.py。目で確認: vfx-review/73-panoptic.jpg）",
    },
}

# このモジュールだけで使う Hugging Face のモデル（リポジトリ, ライセンス）。ライセンスは HF の license タグで確認
_VITMATTE = ("hustvl/vitmatte-small-composition-1k", "Apache-2.0")

# ── 種類の名前 ────────────────────────────────────────────────────────────

# ADE20K の 150 種類の日本語名（UperNet の出力の番号順。英語名は model.config.id2label）
ADE_JA = [
    "壁", "建物", "空", "床", "木", "天井", "道路", "ベッド", "窓", "草",
    "戸棚", "歩道", "人", "地面", "ドア", "テーブル", "山", "植物", "カーテン", "椅子",
    "車", "水", "絵", "ソファ", "棚", "家", "海", "鏡", "じゅうたん", "野原",
    "ひじ掛け椅子", "座席", "柵", "机", "岩", "洋服だんす", "電気スタンド", "浴槽", "手すり", "クッション",
    "台座", "箱", "柱", "看板", "たんす", "カウンター", "砂", "流し台", "高層ビル", "暖炉",
    "冷蔵庫", "観客席", "小道", "階段", "滑走路", "ケース", "ビリヤード台", "枕", "網戸", "階段の通路",
    "川", "橋", "本棚", "ブラインド", "ローテーブル", "トイレ", "花", "本", "丘", "ベンチ",
    "調理台", "コンロ", "ヤシの木", "アイランドキッチン", "コンピューター", "回転椅子", "船", "バー", "ゲーム機", "小屋",
    "バス", "タオル", "照明", "トラック", "塔", "シャンデリア", "日よけ", "街灯", "ブース", "テレビ",
    "飛行機", "土の道", "衣服", "ポール", "土地", "階段の手すり", "エスカレーター", "オットマン", "ボトル", "食器棚",
    "ポスター", "舞台", "バン", "大型船", "噴水", "ベルトコンベヤー", "天蓋", "洗濯機", "おもちゃ", "プール",
    "スツール", "たる", "かご", "滝", "テント", "かばん", "ミニバイク", "ゆりかご", "オーブン", "ボール",
    "食べ物", "段", "タンク", "商標", "電子レンジ", "鉢", "動物", "自転車", "湖", "食器洗い機",
    "スクリーン", "毛布", "彫刻", "換気フード", "壁の照明", "花瓶", "信号機", "トレイ", "ごみ箱", "扇風機",
    "桟橋", "ブラウン管", "皿", "モニター", "掲示板", "シャワー", "暖房器", "グラス", "時計", "旗",
]  # fmt: skip

# MediaPipe の selfie_multiclass の 6 種類
PARTS_JA = ["背景", "髪", "体の肌", "顔の肌", "服", "小物"]

# COCO panoptic の背景（stuff）53 種類: 番号（DETR の出力の番号 = COCO のカテゴリ番号）→ (英語, 日本語)。
# 番号と名前は detectron2 の builtin_meta.py（COCO_CATEGORIES）で確認した
PANO_STUFF = {
    92: ("banner", "垂れ幕"), 93: ("blanket", "毛布"), 95: ("bridge", "橋"), 100: ("cardboard", "段ボール"),
    107: ("counter", "カウンター"), 109: ("curtain", "カーテン"), 112: ("door-stuff", "ドア"),
    118: ("floor-wood", "木の床"), 119: ("flower", "花"), 122: ("fruit", "果物"), 125: ("gravel", "砂利"),
    128: ("house", "家"), 130: ("light", "照明"), 133: ("mirror-stuff", "鏡"), 138: ("net", "ネット"),
    141: ("pillow", "枕"), 144: ("platform", "台・ホーム"), 145: ("playingfield", "競技場"),
    147: ("railroad", "線路"), 148: ("river", "川"), 149: ("road", "道路"), 151: ("roof", "屋根"),
    154: ("sand", "砂"), 155: ("sea", "海"), 156: ("shelf", "棚"), 159: ("snow", "雪"), 161: ("stairs", "階段"),
    166: ("tent", "テント"), 168: ("towel", "タオル"), 171: ("wall-brick", "れんがの壁"),
    175: ("wall-stone", "石の壁"), 176: ("wall-tile", "タイルの壁"), 177: ("wall-wood", "木の壁"),
    178: ("water-other", "水"), 180: ("window-blind", "ブラインド"), 181: ("window-other", "窓"),
    184: ("tree-merged", "木"), 185: ("fence-merged", "柵"), 186: ("ceiling-merged", "天井"),
    187: ("sky-other-merged", "空"), 188: ("cabinet-merged", "戸棚"), 189: ("table-merged", "台"),
    190: ("floor-other-merged", "床"), 191: ("pavement-merged", "舗道"), 192: ("mountain-merged", "山"),
    193: ("grass-merged", "草"), 194: ("dirt-merged", "土"), 195: ("paper-merged", "紙"),
    196: ("food-other-merged", "食べ物"), 197: ("building-other-merged", "建物"), 198: ("rock-merged", "岩"),
    199: ("wall-other-merged", "壁"), 200: ("rug-merged", "じゅうたん"),
}  # fmt: skip


def _names_ja(text: str | None) -> list[str]:
    """'人,空' や 'person, sky' を名前の並びにする。"""
    if not text:
        return []
    return [t.strip() for t in text.replace("、", ",").split(",") if t.strip()]


# ── 共通の部品 ────────────────────────────────────────────────────────────


def _rng(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": a.start or 0.0,
        "duration": a.duration,
        "max_frames": a.max_frames,
    }


def _total(info: core.MediaInfo, a: argparse.Namespace) -> int:
    """処理するコマ数の見込み（進み具合の表示用）。"""
    if info.is_image:
        return 1
    span = a.duration or max(info.duration - (a.start or 0.0), 0.0)
    n = max(1, round(span * info.fps))
    return min(n, a.max_frames) if a.max_frames else n


def _out_path(
    a: argparse.Namespace,
    info: core.MediaInfo,
    tag: str,
    video_ext: str,
    image_ext: str,
) -> Path:
    if a.out:
        out = Path(a.out)
        if info.is_image:
            core.require(
                out.suffix.lower() in core.IMAGE_EXTS,
                f"画像の入力には画像の出力先（.png・.jpg など）を指定してください: {out}",
            )
        else:
            core.require(
                out.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"},
                f"動画の入力には動画の出力先（{video_ext} など）を指定してください: {out}",
            )
        return out
    return core.default_out(a.src, tag, image_ext if info.is_image else video_ext)


def _sibling(path: Path, suffix: str) -> Path:
    """出力の隣に置く別の出力の名前（例: x.webm → x-check.mp4）。"""
    return path.with_name(path.stem + suffix)


def _writer(
    path: Path, info: core.MediaInfo, a: argparse.Namespace, *, audio: bool = True
) -> core.VideoWriter:
    return core.VideoWriter(
        path,
        info.fps,
        audio_from=a.src if audio else None,
        audio_start=a.start or 0.0,
        audio_duration=a.duration,
    )


def _progress(i: int, total: int, t0: float) -> None:
    if total > 1 and ((i + 1) % max(1, total // 10) == 0 or i + 1 == total):
        core.log(f"  {i + 1}/{total} コマ（{time.time() - t0:.0f} 秒）")


def _src_scale(info: core.MediaInfo, frame: np.ndarray) -> tuple[float, float]:
    """処理したコマの座標 → 元の動画の座標 への倍率。"""
    return info.width / frame.shape[1], info.height / frame.shape[0]


def _color(i: int) -> tuple[int, int, int]:
    """番号ごとに決まった、見分けやすい色（BGR）。"""
    h = (i * 0.618033988749895 + 0.11) % 1.0
    s = 0.55 + 0.35 * ((i * 7) % 3) / 2
    v = 0.98 - 0.22 * ((i * 5) % 2)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)


@cache
def _font(size: int) -> Any:
    from PIL import ImageFont

    for f in core.JP_FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _legend(
    img: np.ndarray,
    items: list[tuple[str, tuple[int, int, int]]],
    *,
    size: int | None = None,
) -> np.ndarray:
    """左上に色見本と名前の凡例を描く（img を書き換えて返す。色は BGR）。"""
    from PIL import Image, ImageDraw

    if not items:
        return img
    h, w = img.shape[:2]
    size = size or max(13, min(w, h) // 34)
    font = _font(size)
    pad = max(3, size // 3)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_w = max(probe.textbbox((0, 0), t, font=font)[2] for t, _ in items)
    pw = min(w, size + pad * 3 + text_w)
    rows = min(len(items), max(1, (h - pad) // (size + pad)))
    ph = min(h, pad + rows * (size + pad))
    panel = (img[:ph, :pw, :3].astype(np.float32) * 0.3).astype(np.uint8)
    pil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for k, (text, color) in enumerate(items[:rows]):
        y = pad + k * (size + pad)
        d.rectangle((pad, y, pad + size, y + size), fill=color[::-1])
        d.text((pad * 2 + size, y), text, font=font, fill=(255, 255, 255))
    img[:ph, :pw, :3] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def _paint(
    frame: np.ndarray,
    labels: np.ndarray,
    colors: dict[int, tuple[int, int, int]],
    *,
    opacity: float = 0.5,
    edges: bool = True,
) -> np.ndarray:
    """番号の地図（-1 は塗らない）を色で重ね、境目に白い線を引く。"""
    out = frame[:, :, :3].copy()
    lut = np.zeros((max(colors, default=0) + 2, 3), np.float32)
    for k, c in colors.items():
        lut[k] = c
    valid = labels >= 0
    idx = np.where(valid, labels, lut.shape[0] - 1)
    painted = lut[idx]
    mix = out.astype(np.float32) * (1 - opacity) + painted * opacity
    out[valid] = np.clip(mix[valid], 0, 255).astype(np.uint8)
    if edges:
        lab = labels.astype(np.int32)
        edge = np.zeros(lab.shape, bool)
        edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
        edge[1:, :] |= lab[1:, :] != lab[:-1, :]
        out[edge] = (255, 255, 255)
    return out


def _guided(frame: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """ガイデッドフィルター（He ら 2010）: 地図 p の縁を、元の画の輪郭に沿わせる。"""
    guide = frame[:, :, :3].astype(np.float32) / 255.0
    out = cv2.ximgproc.guidedFilter(guide, p.astype(np.float32), int(radius), eps)
    return np.clip(out, 0.0, 1.0)


def _remap(img: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    """多チャンネルの地図も動かせる remap（OpenCV は 4 チャンネルまでなので分けて行う）。"""
    if img.ndim == 2 or img.shape[2] <= 4:
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    parts = [
        cv2.remap(
            np.ascontiguousarray(img[:, :, c : c + 4]),
            mx,
            my,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        for c in range(0, img.shape[2], 4)
    ]
    parts = [p[:, :, None] if p.ndim == 2 else p for p in parts]
    return np.concatenate(parts, axis=2)


class _Smoother:
    """時間方向のなめらかさ: 前のコマの結果を動き（DIS オプティカルフロー）に合わせて動かし、今のコマと混ぜる。

    動きを追ってから混ぜるので、動く所でも遅れ（残像）が出にくい。動きの推定が外れた所（動かした前のコマの画と
    今のコマの画の差が大きい所）は混ぜない。strength=0 なら何もしない。
    """

    def __init__(self, strength: float, work: int = 360) -> None:
        self.k = float(np.clip(strength, 0.0, 0.95))
        self.work = work
        self.prev_gray: np.ndarray | None = None
        self.prev: np.ndarray | None = None
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def __call__(self, frame: np.ndarray, prob: np.ndarray) -> np.ndarray:
        if self.k <= 0:
            return prob
        h, w = frame.shape[:2]
        f = min(1.0, self.work / max(h, w))
        gw, gh = max(16, round(w * f)), max(16, round(h * f))
        gray = cv2.resize(core.to_gray(frame), (gw, gh), interpolation=cv2.INTER_AREA)
        if self.prev is None or self.prev_gray is None or self.prev.shape != prob.shape:
            self.prev_gray, self.prev = gray, prob.astype(np.float32)
            return prob
        # 今のコマの各画素が、前のコマのどこにあったか（今 → 前 の動き）
        flow = self.dis.calc(gray, self.prev_gray, None)
        ph, pw = prob.shape[:2]
        fx = cv2.resize(flow[..., 0], (pw, ph)) * (pw / gw)
        fy = cv2.resize(flow[..., 1], (pw, ph)) * (ph / gh)
        gx, gy = np.meshgrid(
            np.arange(pw, dtype=np.float32), np.arange(ph, dtype=np.float32)
        )
        warped = _remap(self.prev, gx + fx, gy + fy)
        # 動きの推定が外れた所は混ぜない（動かした前のコマの画と今のコマの差で判断）
        gfx, gfy = np.meshgrid(
            np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32)
        )
        back = cv2.remap(
            self.prev_gray,
            gfx + flow[..., 0],
            gfy + flow[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        err = np.abs(gray.astype(np.float32) - back.astype(np.float32))
        conf = np.clip(1.0 - err / 24.0, 0.0, 1.0)
        wgt = cv2.resize(conf, (pw, ph)) * self.k
        if prob.ndim == 3:
            wgt = wgt[..., None]
        out = wgt * warped + (1.0 - wgt) * prob
        self.prev_gray, self.prev = gray, out.astype(np.float32)
        return out


def _keep_components(
    mask: np.ndarray, mode: str, min_frac: float = 0.002
) -> np.ndarray:
    """2 値の地図から、小さな塊を消す（mode='largest' なら一番大きい塊だけ残す）。"""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if mode == "largest":
        keep = [int(np.argmax(areas)) + 1]
    else:
        keep = [i + 1 for i, s in enumerate(areas) if s >= min_frac * m.size]
        if not keep:
            keep = [int(np.argmax(areas)) + 1]
    return np.isin(lab, keep)


def _fill_small_holes(mask: np.ndarray, max_frac: float = 0.001) -> np.ndarray:
    """小さな穴（地図の内側の、外とつながらない小さな背景）を埋める。"""
    inv = (~mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = mask.copy()
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and area <= max_frac * mask.size:
            out[lab == i] = True
    return out


def _rgba(
    frame: np.ndarray, alpha: np.ndarray, color: np.ndarray | None = None
) -> np.ndarray:
    """色（省略時は元の画）と 0〜1 のアルファから、透過つきの画（BGRA）を作る。"""
    bgr = frame[:, :, :3] if color is None else color
    a = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.dstack([bgr, a])


def _estimate_fg(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """半透明の縁の「前景だけの色」を推定する（Forte・Pitié 2021 の Blur-Fusion。2 段）。

    切り抜いた髪の縁には元の背景の色が混ざっている（観測 = α×前景 + (1−α)×背景）。そのまま別の背景に重ねると
    縁に元の背景の色の輪が出るので、周りの前景・背景の平均色から前景の色を求め直す。
    """
    f = img[:, :, :3].astype(np.float32) / 255.0
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)[..., None]
    short = min(img.shape[:2])
    r1 = max(9, round(90 * short / 1080))
    r2 = max(3, round(6 * short / 1080))

    def step(fg: np.ndarray, bg: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
        ba = cv2.blur(a, (r, r))[..., None]
        bf = cv2.blur(fg * a, (r, r)) / (ba + 1e-5)
        bb = cv2.blur(bg * (1 - a), (r, r)) / ((1 - ba) + 1e-5)
        fg = np.clip(bf + a * (f - a * bf - (1 - a) * bb), 0.0, 1.0)
        return fg, bb

    fg, bb = step(f, f, r1)
    fg, _ = step(fg, bb, r2)
    return np.clip(fg * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _mask_frames(path: str, size: tuple[int, int], **rng: Any) -> Iterator[np.ndarray]:
    """マスク（画像 1 枚か、白黒の動画・透過つき動画）を 0〜1 の地図として順に返す。画像なら同じ物を返し続ける。"""
    info = core.probe(path)
    if info.is_image:
        m = core.load_image(path, alpha=True)
        a = m[:, :, 3] if (m[:, :, 3] < 255).any() else core.to_gray(m[:, :, :3])
        a = (
            cv2.resize(a, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            / 255.0
        )
        while True:
            yield a
    for f in core.read_frames(path, size=size, alpha=info.has_alpha, **rng):
        if info.has_alpha:
            yield f[:, :, 3].astype(np.float32) / 255.0
        else:
            yield core.to_gray(f).astype(np.float32) / 255.0


# ── 人物の地図（MediaPipe）────────────────────────────────────────────────


_MP_KINDS = {"person": "selfie", "parts": "multiclass", "hair": "hair"}


class _Person:
    """MediaPipe の切り抜きの確率の地図（H×W×種類）。

    kind: person（人物。selfie。[背景, 人]）／ parts（部位。multiclass。PARTS_JA の 6 種類）／ hair（髪。[背景, 髪]）。
    結果の画素は、結果の物が生きているうちに写し取る（MediaPipe の結果の画素は呼び出しの後で解放される。
    共有の ai.PersonSegmenter は写さずに返すので、取っておいて後で読むと壊れた値か異常終了になった。2026-11-14 実測）。
    """

    def __init__(self, kind: str, video: bool) -> None:
        self.kind = kind
        self.video = video
        self.task = models.mp_task(
            _MP_KINDS[kind],
            video=video,
            output_confidence_masks=True,
            output_category_mask=False,
        )

    def __call__(self, frame: np.ndarray, t_ms: int | None) -> np.ndarray:
        res = models.mp_detect(
            self.task, "segment", frame, t_ms if self.video else None
        )
        masks = [
            np.array(np.squeeze(m.numpy_view()), dtype=np.float32, copy=True)
            for m in res.confidence_masks
        ]
        if self.kind in ("person", "hair"):
            p = masks[-1] if len(masks) > 1 else masks[0]
            return np.dstack([1.0 - p, p])
        return np.dstack(masks)

    def fg(self, frame: np.ndarray, t_ms: int | None) -> np.ndarray:
        """人物（または髪）の確率だけ（H×W）。"""
        return np.ascontiguousarray(self(frame, t_ms)[:, :, 1])


# ── 51 セグメンテーション ───────────────────────────────────────────────────


class _Scene:
    """UperNet（ConvNeXt small）の ADE20K 150 種類の意味の塗り分け（openmmlab/upernet-convnext-small、MIT）。"""

    def __init__(self, short: int = 512) -> None:
        self.torch = models.torch_setup()
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

        repo = models.hf_repo("ade")
        proc = AutoImageProcessor.from_pretrained(repo)
        self.mean = np.array(proc.image_mean, np.float32)
        self.std = np.array(proc.image_std, np.float32)
        self.net = UperNetForSemanticSegmentation.from_pretrained(repo).eval()
        self.en = [self.net.config.id2label[i].strip() for i in range(len(ADE_JA))]
        self.short = short

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """種類ごとの確率（入力の 1/4 の大きさ × 150）。"""
        torch = self.torch
        h, w = frame.shape[:2]
        f = self.short / min(h, w)
        nh, nw = max(32, round(h * f)), max(32, round(w * f))
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        x = cv2.resize(
            rgb, (nw, nh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
        )
        x = (x.astype(np.float32) / 255.0 - self.mean) / self.std
        ph, pw = -(-nh // 32) * 32, -(-nw // 32) * 32
        x = np.pad(x, ((0, ph - nh), (0, pw - nw), (0, 0)), mode="reflect")
        t = torch.from_numpy(x.transpose(2, 0, 1)[None].copy())
        feats = self.net.backbone.forward_with_filtered_kwargs(t).feature_maps
        logits = self.net.decode_head(feats)  # 入力の 1/4 の大きさ
        lh, lw = logits.shape[-2:]
        logits = logits[:, :, : round(lh * nh / ph), : round(lw * nw / pw)]
        prob = torch.softmax(logits, dim=1)[0].permute(1, 2, 0).numpy()
        return np.ascontiguousarray(prob, dtype=np.float32)


def _labels_from_prob(prob: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """小さな確率の地図から、出力の大きさの番号の地図を作る（出てくる種類だけを拡大して比べる）。"""
    w, h = size
    small = prob.argmax(axis=2)
    present = np.unique(small)
    if prob.shape[:2] == (h, w):
        return small.astype(np.int32)
    up = [
        cv2.resize(prob[:, :, c], (w, h), interpolation=cv2.INTER_LINEAR)
        for c in present
    ]
    return present[np.argmax(np.stack(up, axis=0), axis=0)].astype(np.int32)


_SEG_MODELS = {
    "scene": "openmmlab/upernet-convnext-small（MIT）",
    "person": "MediaPipe selfie_segmenter（Apache-2.0）",
    "parts": "MediaPipe selfie_multiclass_256x256（Apache-2.0）",
}


def _soft_classes(
    frame: np.ndarray, prob: np.ndarray, classes: list[int]
) -> np.ndarray:
    """選んだ種類の地図（0〜1）: 確率を足して出力の大きさにし、元の画の輪郭に沿わせる。"""
    h, w = frame.shape[:2]
    p = prob[:, :, classes].sum(axis=2)
    if p.shape != (h, w):
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
    # 半径 = 短い辺の 1/70・eps 0.01（正解つきの合成で境目の IoU が scene 0.689 → 0.742、person 0.642 → 0.687）
    return _guided(frame, p, max(2, round(min(h, w) / 70)), 1e-2)


def _class_index(names: list[str], ja: list[str], en: list[str]) -> list[int]:
    out = []
    for n in names:
        if n in ja:
            out.append(ja.index(n))
        elif n.lower() in [e.lower() for e in en]:
            out.append([e.lower() for e in en].index(n.lower()))
        else:
            raise core.VfxError(
                f"知らない種類です: {n}（使える名前の例: {', '.join(ja[:12])} …）"
            )
    return out


@command(
    51,
    "segment",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--model",
        choices=["scene", "person", "parts"],
        default="scene",
        help="scene: 空・壁・道・木・人など 150 種類（UperNet。既定）／ person: 人物だけ（MediaPipe。速い）"
        "／ parts: 髪・体の肌・顔の肌・服・小物（MediaPipe）",
    ),
    arg(
        "--classes",
        help="切り抜き・マスクにする種類（日本語名か英語名をカンマ区切り。例: 人,空）。--cutout の既定は人",
    ),
    arg(
        "--cutout",
        help="選んだ種類だけを残した透過つきの出力先（動画は .webm、画像は .png）",
    ),
    arg(
        "--masks",
        help="種類ごとの白黒マスクを書き出すフォルダ（画像は PNG、動画は MP4）",
    ),
    arg(
        "--smooth",
        type=float,
        default=0.5,
        help="動画のちらつきを抑える強さ（0〜0.9。動きに合わせて前のコマと混ぜる。0 で無効。既定 0.5）",
    ),
    arg(
        "--min-area",
        type=float,
        default=0.005,
        help="凡例・JSON に載せる種類の、画面に占める割合の下限（既定 0.005）",
    ),
    arg("--opacity", type=float, default=0.5, help="色を重ねる濃さ（0〜1。既定 0.5）"),
)
def segment(a: argparse.Namespace) -> None:
    """画素ごとに種類（人・空・壁・髪・服など）を塗り分け、色分けの重ね表示・種類ごとのマスク・透過の切り抜き・面積の JSON を書く。"""
    info = core.probe(a.src)
    out = _out_path(a, info, "segment", ".mp4", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    if a.model == "scene":
        core.log("UperNet（ADE20K 150 種類）を読み込みます")
        scene = _Scene()
        ja, en = ADE_JA, scene.en
        person_idx = 12
    else:
        scene = None
        ja = ["背景", "人"] if a.model == "person" else PARTS_JA
        en = ["background", "person"] if a.model == "person" else [
            "background", "hair", "body-skin", "face-skin", "clothes", "others"
        ]  # fmt: skip
        person_idx = 1
    chosen = _class_index(_names_ja(a.classes), ja, en)
    cut_classes = chosen or (
        [person_idx] if a.model != "parts" else list(range(1, len(ja)))
    )
    mp_seg = None if scene else _Person(a.model, video=not info.is_image)
    smooth = _Smoother(0.0 if info.is_image else a.smooth)
    colors = {i: _color(i) for i in range(len(ja))}
    total = _total(info, a)
    writer = None if info.is_image else _writer(out, info, a)
    cut_writer = None
    if a.cutout:
        cut_path = Path(a.cutout)
        if not info.is_image:
            core.require(
                cut_path.suffix.lower() == ".webm",
                "--cutout は .webm を指定してください（透過つき）",
            )
            cut_writer = core.VideoWriter(cut_path, info.fps, alpha=True)
    mask_dir = Path(a.masks) if a.masks else None
    mask_writers: dict[int, core.VideoWriter] = {}
    frames_json: list[dict[str, Any]] = []
    sums: dict[int, float] = {}
    seen: dict[int, int] = {}
    t0 = time.time()
    model_time = 0.0
    n = 0
    start = a.start or 0.0
    for i, frame in enumerate(core.read_frames(a.src, **_rng(a), max_side=a.max_side)):
        h, w = frame.shape[:2]
        tm = time.time()
        if scene is not None:
            prob = scene(frame)
        else:
            assert mp_seg is not None
            prob = mp_seg(frame, round(core.t_of(i, info.fps, start) * 1000))
        model_time += time.time() - tm
        prob = smooth(frame, prob)
        labels = _labels_from_prob(prob, (w, h))
        counts = np.bincount(labels.ravel(), minlength=len(ja)) / labels.size
        present = [int(c) for c in np.argsort(-counts) if counts[c] >= a.min_area]
        for c in present:
            sums[c] = sums.get(c, 0.0) + float(counts[c])
            seen[c] = seen.get(c, 0) + 1
        rec = {
            "t": core.t_of(i, info.fps, start),
            "areas": {ja[c]: round(float(counts[c]), 5) for c in present},
        }
        frames_json.append(rec)
        # 色分けの重ね表示と凡例
        view = _paint(frame, labels, colors, opacity=a.opacity)
        items = [
            (f"{ja[c]} {counts[c] * 100:.0f}%" if info.is_image else ja[c], colors[c])
            for c in present[:12]
        ]
        view = _legend(view, items)
        if a.cutout:
            alpha = _soft_classes(frame, prob, cut_classes)
            if info.is_image:
                core.save_image(a.cutout, _rgba(frame, alpha))
            else:
                assert cut_writer is not None
                cut_writer.write(_rgba(frame, alpha))
        if mask_dir is not None:
            targets = chosen or present[:10]
            if info.is_image:
                for c in targets:
                    m = (labels == c).astype(np.uint8) * 255
                    core.save_image(mask_dir / f"{ja[c]}.png", m)
            else:
                if i == 0:
                    for c in targets:
                        mask_writers[c] = core.VideoWriter(
                            mask_dir / f"{ja[c]}.mp4", info.fps
                        )
                for c, mw in mask_writers.items():
                    mw.write((labels == c).astype(np.uint8) * 255)
        if writer is not None:
            writer.write(view)
        else:
            core.save_image(out, view)
        n += 1
        _progress(i, total, t0)
    if writer is not None:
        writer.close()
    if cut_writer is not None:
        cut_writer.close()
    for mw in mask_writers.values():
        mw.close()
    summary = {
        ja[c]: {"en": en[c], "mean_area": round(sums[c] / n, 5), "frames": seen[c]}
        for c in sorted(sums, key=lambda c: -sums[c])
    }
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "model": a.model,
            "model_name": _SEG_MODELS[a.model],
            "size": [int(frame.shape[1]), int(frame.shape[0])],
            "fps": info.fps,
            "frames_processed": n,
            "seconds_per_frame": round(model_time / max(n, 1), 3),
            "classes": summary,
            "frames": frames_json,
        },
    )
    core.log(f"{n} コマ。見つかった種類（平均の面積）:")
    for name, s in list(summary.items())[:12]:
        core.log(f"  {name}（{s['en']}）{s['mean_area'] * 100:.1f}%")
    core.log(f"1 コマあたり {model_time / max(n, 1):.2f} 秒（モデルの計算だけ）")
    if a.cutout:
        core.log(f"保存: {a.cutout}")
    if mask_dir is not None:
        core.log(f"保存: {mask_dir}/")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 文字の書き込み（一度にまとめて描く）───────────────────────────────────


def _draw_labels(
    img: np.ndarray,
    items: list[tuple[str, tuple[int, int], tuple[int, int, int]]],
    *,
    size: int | None = None,
) -> np.ndarray:
    """(文字, 左下の位置, 背景色 BGR) の並びを、1 回の変換でまとめて描く（img を書き換えて返す）。"""
    from PIL import Image, ImageDraw

    if not items:
        return img
    h, w = img.shape[:2]
    size = size or max(12, min(w, h) // 40)
    font = _font(size)
    pil = Image.fromarray(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    pad = max(2, size // 5)
    for text, (x, y), color in items:
        box = d.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        x = int(np.clip(x, 0, max(0, w - tw - 2 * pad)))
        y = int(np.clip(y, th + 2 * pad, h))
        d.rectangle((x, y - th - 2 * pad, x + tw + 2 * pad, y), fill=color[::-1])
        d.text((x + pad, y - th - pad - box[1]), text, font=font, fill=(255, 255, 255))
    img[:, :, :3] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def _outline(
    img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thick: int = 2
) -> None:
    """マスクの輪郭を描く（img を書き換える）。"""
    cs, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, cs, -1, color, thick, cv2.LINE_AA)


def _xywh_src(box: Any, sx: float, sy: float) -> list[float]:
    """(x1, y1, x2, y2) の枠を、元の動画の座標の [x, y, 幅, 高さ] にする。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    return [
        round(x1 * sx, 1),
        round(y1 * sy, 1),
        round((x2 - x1) * sx, 1),
        round((y2 - y1) * sy, 1),
    ]


def _box_iou(a: Any, b: Any) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class _IdKeeper:
    """動画で、同じ物に同じ番号を付け続ける（前のコマの枠との重なりが大きい物を、同じ種類どうしで順に結ぶ）。"""

    def __init__(self, iou: float = 0.2, max_gap: int = 12) -> None:
        self.iou = iou
        self.max_gap = max_gap
        self.tracks: dict[
            int, tuple[str, Any, int]
        ] = {}  # 番号 → (種類, 枠, 最後に見えたコマ)
        self.next_id = 1

    def __call__(self, labels: list[str], boxes: list[Any], frame: int) -> list[int]:
        pairs = []
        for di, (lab, box) in enumerate(zip(labels, boxes, strict=True)):
            for tid, (tlab, tbox, last) in self.tracks.items():
                if tlab == lab and frame - last <= self.max_gap:
                    v = _box_iou(box, tbox)
                    if v >= self.iou:
                        pairs.append((v, di, tid))
        ids = [0] * len(boxes)
        used: set[int] = set()
        for _v, di, tid in sorted(pairs, reverse=True):
            if ids[di] == 0 and tid not in used:
                ids[di] = tid
                used.add(tid)
        for di in range(len(boxes)):
            if ids[di] == 0:
                ids[di] = self.next_id
                self.next_id += 1
            self.tracks[ids[di]] = (labels[di], boxes[di], frame)
        return ids


# ── 61 インスタンス分割 ───────────────────────────────────────────────────


class _MaskRCNN:
    """torchvision の Mask R-CNN v2（maskrcnn_resnet50_fpn_v2。BSD-3-Clause。COCO の 80 種類）。"""

    def __init__(self, score: float, short: int) -> None:
        self.torch = models.torch_setup()
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.names = weights.meta["categories"]
        self.net = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            box_score_thresh=min(score, 0.3),
            min_size=short,
            max_size=max(1333, short * 2),
        ).eval()
        self.score = score

    def __call__(self, frame: np.ndarray) -> list[dict[str, Any]]:
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        x = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        out = self.net([x])[0]
        res = []
        for box, s, lab, m in zip(
            out["boxes"].numpy(),
            out["scores"].numpy(),
            out["labels"].numpy(),
            out["masks"].numpy(),
            strict=True,
        ):
            if s >= self.score:
                res.append(
                    {
                        "box": box,
                        "score": float(s),
                        "label": self.names[int(lab)],
                        "mask": m[0],
                    }
                )
        return res


def _drop_contained(
    dets: list[dict[str, Any]], ratio: float = 0.7
) -> list[dict[str, Any]]:
    """同じ種類で、マスクの ratio 以上が別のマスクに含まれる小さい方を外す（隠れた人を「上半身」と「全身」の
    2 つに数える二重検出を消す。vtest の看板の後ろの人で確認）。"""
    masks = [d["mask"] >= 0.5 for d in dets]
    areas = [int(m.sum()) for m in masks]
    drop: set[int] = set()
    for i in range(len(dets)):
        for j in range(len(dets)):
            if i == j or dets[i]["label"] != dets[j]["label"] or i in drop or j in drop:
                continue
            small, big = (i, j) if areas[i] <= areas[j] else (j, i)
            if (
                areas[small]
                and (masks[small] & masks[big]).sum() / areas[small] >= ratio
            ):
                drop.add(small)
    return [d for k, d in enumerate(dets) if k not in drop]


def _coco_classes(text: str | None) -> set[str] | None:
    """'人,車' や 'person,car' を COCO の英語名の集まりにする。"""
    from .detectors import COCO_JA

    names = _names_ja(text)
    if not names:
        return None
    rev = {v: k for k, v in COCO_JA.items()}
    out = set()
    for n in names:
        if n in COCO_JA:
            out.add(n)
        elif n in rev:
            out.add(rev[n])
        else:
            raise core.VfxError(
                f"知らない種類です: {n}（COCO の 80 種類。例: 人, 車, 犬, 猫, カップ）"
            )
    return out


@command(
    61,
    "instances",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--score",
        type=float,
        default=0.6,
        help="採用する確からしさの下限（0〜1。既定 0.6）",
    ),
    arg(
        "--classes",
        help="残す種類（日本語名か英語名をカンマ区切り。例: 人 / person,car。省略時は 80 種類すべて）",
    ),
    arg(
        "--masks",
        help="1 つずつの白黒マスクと透過つき PNG を書き出すフォルダ（画像の時）",
    ),
    arg(
        "--split",
        type=int,
        default=0,
        help="動画で、番号ごとに別々の透過つき WebM を書き出す数（先に現れた順。既定 0 = 書かない）",
    ),
    arg("--opacity", type=float, default=0.5, help="色を重ねる濃さ（0〜1。既定 0.5）"),
)
def instances(a: argparse.Namespace) -> None:
    """人や物を 1 つずつ別の色・別のマスクに分ける（Mask R-CNN）。重ね表示・JSON・1 つずつのマスクを書く。"""
    from .detectors import COCO_JA

    info = core.probe(a.src)
    out = _out_path(a, info, "instances", ".mp4", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    keep = _coco_classes(a.classes)
    pw, ph = core.scaled_size(info.width, info.height, None, a.max_side)
    short = int(np.clip(min(pw, ph), 512, 800))
    core.log(f"Mask R-CNN v2 を読み込みます（短い辺 {short}px で計算）")
    net = _MaskRCNN(a.score, short)
    keeper = _IdKeeper()
    writer = None if info.is_image else _writer(out, info, a)
    split_dir = _sibling(out, "-split") if a.split and not info.is_image else None
    split_writers: dict[int, core.VideoWriter] = {}
    total = _total(info, a)
    t0 = time.time()
    model_time = 0.0
    start = a.start or 0.0
    frames_json: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    n = 0
    for i, frame in enumerate(core.read_frames(a.src, **_rng(a), max_side=a.max_side)):
        h, w = frame.shape[:2]
        sx, sy = _src_scale(info, frame)
        tm = time.time()
        dets = [d for d in net(frame) if keep is None or d["label"] in keep]
        dets = _drop_contained(dets)
        model_time += time.time() - tm
        ids = keeper([d["label"] for d in dets], [d["box"] for d in dets], i)
        # 重なった所は確からしさの高い方に塗る
        lab_map = np.full((h, w), -1, np.int32)
        for k in np.argsort([d["score"] for d in dets]):
            lab_map[dets[k]["mask"] >= 0.5] = ids[k]
        colors = {tid: _color(tid) for tid in ids}
        view = _paint(frame, lab_map, colors, opacity=a.opacity, edges=False)
        texts = []
        recs = []
        for d, tid in zip(dets, ids, strict=True):
            m = d["mask"] >= 0.5
            _outline(view, m, colors[tid])
            ja = COCO_JA.get(d["label"], d["label"])
            x1, y1 = int(d["box"][0]), int(d["box"][1])
            texts.append((f"{ja} #{tid} {d['score']:.2f}", (x1, y1), colors[tid]))
            recs.append(
                {
                    "id": tid,
                    "class": d["label"],
                    "ja": ja,
                    "score": round(d["score"], 4),
                    "box": _xywh_src(d["box"], sx, sy),
                    "area": round(float(m.mean()), 5),
                }
            )
            counts[ja] = counts.get(ja, 0) + 1
        view = _draw_labels(view, texts)
        frames_json.append({"t": core.t_of(i, info.fps, start), "instances": recs})
        if info.is_image:
            core.save_image(out, view)
            if a.masks:
                mdir = Path(a.masks)
                for d, tid in zip(dets, ids, strict=True):
                    ja = COCO_JA.get(d["label"], d["label"])
                    m = (d["mask"] >= 0.5).astype(np.uint8) * 255
                    core.save_image(mdir / f"{tid:02d}-{ja}.png", m)
                    core.save_image(
                        mdir / f"{tid:02d}-{ja}-cut.png",
                        _rgba(frame, d["mask"].clip(0, 1)),
                    )
        else:
            assert writer is not None
            writer.write(view)
            if split_dir is not None:
                for d, tid in zip(dets, ids, strict=True):
                    if tid not in split_writers and len(split_writers) < a.split:
                        ja = COCO_JA.get(d["label"], d["label"])
                        sw = core.VideoWriter(
                            split_dir / f"{tid:02d}-{ja}.webm", info.fps, alpha=True
                        )
                        blank = np.zeros((h, w, 4), np.uint8)
                        for _ in range(i):  # 現れる前のコマは透明にする
                            sw.write(blank)
                        split_writers[tid] = sw
                by_id = {tid: d for d, tid in zip(dets, ids, strict=True)}
                for tid, sw in split_writers.items():
                    d = by_id.get(tid)
                    alpha = (
                        d["mask"].clip(0, 1)
                        if d is not None
                        else np.zeros((h, w), np.float32)
                    )
                    sw.write(_rgba(frame, alpha))
        n += 1
        _progress(i, total, t0)
    if writer is not None:
        writer.close()
    for sw in split_writers.values():
        sw.close()
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "model": "torchvision maskrcnn_resnet50_fpn_v2（COCO。BSD-3-Clause）",
            "score": a.score,
            "size": [int(w), int(h)],
            "fps": info.fps,
            "frames_processed": n,
            "seconds_per_frame": round(model_time / max(n, 1), 3),
            "ids": keeper.next_id - 1,
            "frames": frames_json,
        },
    )
    per = ", ".join(
        f"{k} {v / n:.1f}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    core.log(
        f"{n} コマ。1 コマあたりの数: {per or 'なし'}。番号の数: {keeper.next_id - 1}"
    )
    core.log(f"1 コマあたり {model_time / max(n, 1):.2f} 秒（モデルの計算だけ）")
    if split_writers:
        core.log(f"保存: {split_dir}/（{len(split_writers)} 本）")
    if a.masks and info.is_image:
        core.log(f"保存: {a.masks}/")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 73 パノプティック分割 ───────────────────────────────────────────────────


def _tv_resnet50() -> Any:
    """torchvision の ResNet-50 を、timm の resnet50 と同じ部品名・同じ出力（4 段の特徴）で作る。"""
    import torch
    import torchvision

    class TVResNet50(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            r = torchvision.models.resnet50(weights=None)
            self.conv1, self.bn1, self.act1, self.maxpool = (
                r.conv1,
                r.bn1,
                r.relu,
                r.maxpool,
            )
            self.layer1, self.layer2, self.layer3, self.layer4 = (
                r.layer1,
                r.layer2,
                r.layer3,
                r.layer4,
            )

        def forward(self, x: Any) -> list[Any]:
            x = self.maxpool(self.act1(self.bn1(self.conv1(x))))
            c1 = self.layer1(x)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)
            return [c1, c2, c3, self.layer4(c3)]

    return TVResNet50().eval()


class _Panoptic:
    """DETR（ResNet-50）のパノプティック分割（facebook/detr-resnet-50-panoptic。Apache-2.0。COCO panoptic 133 種類）。

    transformers 5.2 はこのモデルの背骨（ResNet-50）を timm で作るが、timm は入っていない。そこで同じ形の torchvision の
    ResNet-50 に同じ重みを読み込んで差し替える（timm を使う公式の読み込みと、マスクの値の差は最大 0.004 で一致を確認）。
    また、物の候補（100 個）のうち採用する物だけにマスクの計算をする（全部に計算すると 800px の画像で 12GB 使った）。
    """

    def __init__(self) -> None:
        self.torch = models.torch_setup()
        from huggingface_hub import hf_hub_download
        from transformers import (
            DetrConfig,
            DetrForSegmentation,
            DetrImageProcessor,
            ResNetConfig,
        )
        from transformers.utils import logging as hf_logging

        repo = models.hf_repo("panoptic")
        self.proc = DetrImageProcessor.from_pretrained(repo)
        cfg = DetrConfig.from_pretrained(repo)
        cfg.backbone_config = ResNetConfig(
            out_features=["stage1", "stage2", "stage3", "stage4"]
        )
        level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # 背骨の重みが「合わない」という長い表を出さない（下で読み込む）
        try:
            net = DetrForSegmentation.from_pretrained(repo, config=cfg).eval()
        finally:
            hf_logging.set_verbosity(level)
        tv = _tv_resnet50()
        sd = self.torch.load(
            hf_hub_download(repo, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        pre = "detr.model.backbone.conv_encoder.model."
        res = tv.load_state_dict(
            {k[len(pre) :]: v for k, v in sd.items() if k.startswith(pre)}, strict=False
        )
        bad = [k for k in res.missing_keys if not k.endswith("num_batches_tracked")]
        if bad or res.unexpected_keys:
            raise core.VfxError(
                f"DETR の背骨の重みを読み込めません: {bad[:3]} {res.unexpected_keys[:3]}"
            )
        net.detr.model.backbone.model = tv
        self.net = net
        self.names = {int(k): v for k, v in cfg.id2label.items()}
        self.num_labels = len(cfg.id2label)

    def _forward(self, pixel_values: Any, pixel_mask: Any, threshold: float) -> Any:
        """DetrForSegmentation.forward と同じ計算。ただしマスクは採用する候補だけに計算する。"""
        from types import SimpleNamespace

        torch = self.torch
        det = self.net.detr
        feats = det.model.backbone(pixel_values, pixel_mask)
        feature_map, mask = feats[-1]
        projected = det.model.input_projection(feature_map)
        flat = projected.flatten(2).permute(0, 2, 1)
        pos = det.model.position_embedding(
            shape=feature_map.shape,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
            mask=mask,
        )
        flat_mask = mask.flatten(1)
        enc = det.model.encoder(
            inputs_embeds=flat,
            attention_mask=flat_mask,
            spatial_position_embeddings=pos,
        )
        qpos = det.model.query_position_embeddings.weight.unsqueeze(0)
        dec = det.model.decoder(
            inputs_embeds=torch.zeros_like(qpos),
            attention_mask=None,
            spatial_position_embeddings=pos,
            object_queries_position_embeddings=qpos,
            encoder_hidden_states=enc.last_hidden_state,
            encoder_attention_mask=flat_mask,
        )
        seq = dec[0]
        logits = det.class_labels_classifier(seq)
        scores, labels = logits.softmax(-1).max(-1)
        keep = ((labels != logits.shape[-1] - 1) & (scores > threshold))[0]
        if int(keep.sum()) == 0:
            return None
        seq = seq[:, keep]
        hh, ww = feature_map.shape[-2:]
        memory = enc.last_hidden_state.permute(0, 2, 1).view(
            1, self.net.config.d_model, hh, ww
        )
        att = flat_mask.view(1, hh, ww)
        att = torch.where(
            att.unsqueeze(1).unsqueeze(1),
            torch.tensor(0.0, dtype=memory.dtype),
            torch.finfo(memory.dtype).min,
        )
        bbox_mask = self.net.bbox_attention(seq, memory, attention_mask=att)
        seg = self.net.mask_head(
            features=projected,
            attention_masks=bbox_mask,
            fpn_features=[feats[2][0], feats[1][0], feats[0][0]],
        )
        masks = seg.view(1, int(keep.sum()), seg.shape[-2], seg.shape[-1])
        return SimpleNamespace(logits=logits[:, keep], pred_masks=masks)

    def __call__(
        self, frame: np.ndarray, threshold: float, short: int
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        inputs = self.proc(
            images=rgb,
            size={"shortest_edge": short, "longest_edge": 1333},
            return_tensors="pt",
        )
        out = self._forward(inputs["pixel_values"], inputs["pixel_mask"], threshold)
        if out is None:
            return np.full((h, w), -1, np.int32), []
        res = self.proc.post_process_panoptic_segmentation(
            out,
            threshold=threshold,
            target_sizes=[(h, w)],
            label_ids_to_fuse=set(PANO_STUFF),
        )[0]
        seg = res["segmentation"].numpy().astype(np.int32)
        return seg, res["segments_info"]

    def name(self, label: int) -> tuple[str, str, bool]:
        """番号 → (英語, 日本語, 物か)。"""
        from .detectors import COCO_JA

        if label in PANO_STUFF:
            en, ja = PANO_STUFF[label]
            return en, ja, False
        en = self.names.get(label, str(label))
        return en, COCO_JA.get(en, en), True


@command(
    73,
    "panoptic",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--threshold",
        type=float,
        default=0.85,
        help="採用する確からしさの下限（0〜1。既定 0.85）",
    ),
    arg(
        "--min-area",
        type=float,
        default=0.0,
        help="これより小さい塊（画面に占める割合）は表と JSON から外す（既定 0 = 16 画素未満だけ外す）",
    ),
    arg("--masks", help="塊ごとの白黒マスクを書き出すフォルダ（画像の時）"),
    arg(
        "--opacity", type=float, default=0.55, help="色を重ねる濃さ（0〜1。既定 0.55）"
    ),
)
def panoptic(a: argparse.Namespace) -> None:
    """物（人・車など 1 つずつ）と背景（空・道・草・壁など）をまとめて塗り分け、凡例つきの画と JSON を書く（DETR）。"""
    info = core.probe(a.src)
    out = _out_path(a, info, "panoptic", ".mp4", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    core.log("DETR panoptic を読み込みます")
    net = _Panoptic()
    writer = None if info.is_image else _writer(out, info, a)
    total = _total(info, a)
    t0 = time.time()
    model_time = 0.0
    start = a.start or 0.0
    frames_json: list[dict[str, Any]] = []
    n = 0
    for i, frame in enumerate(core.read_frames(a.src, **_rng(a), max_side=a.max_side)):
        h, w = frame.shape[:2]
        sx, sy = _src_scale(info, frame)
        short = int(np.clip(min(h, w), 480, 800))
        tm = time.time()
        seg, infos = net(frame, a.threshold, short)
        model_time += time.time() - tm
        recs = []
        colors: dict[int, tuple[int, int, int]] = {}
        per_class: dict[int, int] = {}
        lab_map = np.full((h, w), -1, np.int32)
        for s in sorted(infos, key=lambda s: -(seg == s["id"]).sum()):
            m = seg == s["id"]
            area = float(m.mean())
            if area < a.min_area or m.sum() < 16:
                continue
            en, ja, thing = net.name(int(s["label_id"]))
            k = per_class.get(s["label_id"], 0)
            per_class[s["label_id"]] = k + 1
            base = _color(int(s["label_id"]))
            # 物は 1 つずつ明るさを変えて見分ける
            col = (
                base
                if not thing
                else tuple(
                    int(np.clip(c * (1.0 - 0.18 * (k % 4)), 0, 255)) for c in base
                )
            )
            sid = len(recs)
            colors[sid] = col  # type: ignore[assignment]
            lab_map[m] = sid
            ys, xs = np.nonzero(m)
            box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            recs.append(
                {
                    "id": sid + 1,
                    "class": en,
                    "ja": ja,
                    "thing": thing,
                    "score": round(float(s["score"]), 4),
                    "area": round(area, 5),
                    "box": _xywh_src(box, sx, sy),
                }
            )
        view = _paint(frame, lab_map, colors, opacity=a.opacity)
        for r, sid in zip(recs, range(len(recs)), strict=True):
            if r["thing"]:
                _outline(view, lab_map == sid, (255, 255, 255), 1)
        # 凡例: 背景は名前と面積、物は種類ごとの数
        items: list[tuple[str, tuple[int, int, int]]] = []
        done: set[str] = set()
        for r, sid in zip(recs, range(len(recs)), strict=True):
            if r["thing"]:
                if r["ja"] in done:
                    continue
                done.add(r["ja"])
                cnt = sum(1 for q in recs if q["ja"] == r["ja"] and q["thing"])
                items.append((f"{r['ja']} ×{cnt}", colors[sid]))
            else:
                items.append((f"{r['ja']} {r['area'] * 100:.0f}%", colors[sid]))
        view = _legend(view, items[:14])
        frames_json.append({"t": core.t_of(i, info.fps, start), "segments": recs})
        if info.is_image:
            core.save_image(out, view)
            if a.masks:
                for r, sid in zip(recs, range(len(recs)), strict=True):
                    core.save_image(
                        Path(a.masks) / f"{r['id']:02d}-{r['ja']}.png",
                        (lab_map == sid).astype(np.uint8) * 255,
                    )
        else:
            assert writer is not None
            writer.write(view)
        n += 1
        _progress(i, total, t0)
    if writer is not None:
        writer.close()
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "model": "facebook/detr-resnet-50-panoptic（Apache-2.0）",
            "threshold": a.threshold,
            "size": [int(w), int(h)],
            "fps": info.fps,
            "frames_processed": n,
            "seconds_per_frame": round(model_time / max(n, 1), 3),
            "frames": frames_json,
        },
    )
    last = frames_json[-1]["segments"] if frames_json else []
    things = sum(1 for r in last if r["thing"])
    stuff = ", ".join(
        f"{r['ja']} {r['area'] * 100:.0f}%" for r in last if not r["thing"]
    )
    core.log(f"{n} コマ。最後のコマ: 物 {things} 個、背景: {stuff or 'なし'}")
    core.log(f"1 コマあたり {model_time / max(n, 1):.2f} 秒（モデルの計算だけ）")
    if a.masks and info.is_image:
        core.log(f"保存: {a.masks}/")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 切り抜きの地図（アルファ）を求める部品（53・56 で使う）──────────────────

_IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class _BiRefNet:
    """BiRefNet の ONNX（rembg の配布。MIT）。入力は 1024×1024 固定、ImageNet の正規化、出力はアルファの元（ロジット）。"""

    def __init__(self, key: str) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 4
        # 作業用メモリの使い回し（arena）を切る。入れたままだと 2 回目の実行で 12GB まで増えた（lite で実測）
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        self.s = ort.InferenceSession(
            str(models.fetch(key)), so, providers=["CPUExecutionProvider"]
        )
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        interp = cv2.INTER_AREA if max(h, w) > 1024 else cv2.INTER_LINEAR
        x = (
            cv2.resize(rgb, (1024, 1024), interpolation=interp).astype(np.float32)
            / 255.0
        )
        x = (x - _IMNET_MEAN) / _IMNET_STD
        y = self.s.run(None, {self.inp: x.transpose(2, 0, 1)[None]})[0][0, 0]
        p = 1.0 / (1.0 + np.exp(-y.astype(np.float32)))
        return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


class _VitMatte:
    """ViTMatte（small。hustvl/vitmatte-small-composition-1k。Apache-2.0）。

    3 値の地図（trimap: 白 = 必ず残す・黒 = 必ず消す・灰 = 分からない）の灰の帯の中で、髪の毛などの細かいアルファを求める。
    長い辺が work px より大きい画は縮めて計算し、元の大きさに戻してからガイデッドフィルターで輪郭を合わせる。
    """

    def __init__(self, work: int = 1280) -> None:
        self.torch = models.torch_setup()
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        self.proc = VitMatteImageProcessor.from_pretrained(_VITMATTE[0])
        self.net = VitMatteForImageMatting.from_pretrained(_VITMATTE[0]).eval()
        self.work = work

    def __call__(self, frame: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        f = min(1.0, self.work / max(h, w))
        img, tri = frame[:, :, :3], trimap
        if f < 1.0:
            size = (max(32, round(w * f)), max(32, round(h * f)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            tri = cv2.resize(tri, size, interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = self.proc(images=rgb, trimaps=tri, return_tensors="pt")
        alpha = self.net(**inputs).alphas[0, 0].numpy()[: img.shape[0], : img.shape[1]]
        alpha = alpha.astype(np.float32)
        if f < 1.0:
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = _guided(frame, alpha, 2, 1e-4)
        alpha[trimap == 255] = 1.0
        alpha[trimap == 0] = 0.0
        return np.clip(alpha, 0.0, 1.0)


def _ellipse(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _trimap(
    prob: np.ndarray,
    band: int,
    hair: np.ndarray | None = None,
    hair_band: int | None = None,
) -> np.ndarray:
    """粗い地図（0〜1）から 3 値の地図を作る。

    髪も人物に含め（人物と髪の地図の大きい方）、境目の内外 band px と、地図が 0.1〜0.9 の自信のない所を
    「分からない」にする（風になびく髪は人物の地図が中途半端になりやすく、0.5 で切ると毛先が切れた）。
    髪（髪の地図が 0.5 以上）の周りは、外へ hair_band px（既定は band の 10 倍）まで「分からない」を広げる。
    はねた毛は髪の地図にも人物の地図にもほとんど入らないため（合成の試験で、毛だけの画素のうち髪の地図が
    0.5 以上だったのは 2% 未満）、帯を広くしないと拾えない。
    """
    q = np.maximum(prob, hair) if hair is not None else prob
    hard = (q >= 0.5).astype(np.uint8)
    tri = np.full(prob.shape, 128, np.uint8)
    tri[(cv2.erode(hard, _ellipse(band)) > 0) & (q >= 0.9)] = 255
    tri[(cv2.dilate(hard, _ellipse(band)) == 0) & (q <= 0.1)] = 0
    if hair is not None and (hair >= 0.5).any():
        hb = hair_band if hair_band is not None else 10 * band
        hair_m = (hair >= 0.5).astype(np.uint8)
        around = cv2.dilate(hair_m, _ellipse(max(1, hb))) > 0
        tri[around & (tri == 0)] = 128  # はねた毛を拾うため、髪の外側の帯を広げる
        near = cv2.dilate(hair_m, _ellipse(2 * band)) > 0
        inner = cv2.erode(hard, _ellipse(2 * band)) > 0
        tri[near & (tri == 255) & ~inner] = (
            128  # 毛先のすき間を拾うため、内側も少し広げる
        )
    return tri


class _Matter:
    """1 コマの切り抜きの地図（0〜1）を求める。

    model: vitmatte（粗い地図 → 3 値の地図 → ViTMatte）／ birefnet・birefnet-portrait（BiRefNet だけ）／
    guided（粗い地図 → ガイデッドフィルター）／ selfie（粗い地図のまま）。粗い地図は MediaPipe の人物の切り抜き
    （mask を渡した時はそれ）。
    """

    def __init__(
        self,
        model: str,
        *,
        video: bool,
        band: int | None = None,
        hair: bool = True,
        hair_band: int | None = None,
    ) -> None:
        self.model = model
        self.band = band
        self.hair_band = hair_band
        self.person = (
            _Person("person", video=video)
            if model in ("vitmatte", "guided", "selfie")
            else None
        )
        self.hair = (
            _Person("hair", video=video) if model == "vitmatte" and hair else None
        )
        self.vm = _VitMatte() if model == "vitmatte" else None
        self.bn = None
        if model.startswith("birefnet"):
            self.bn = _BiRefNet(
                "birefnet_lite" if model == "birefnet" else "birefnet_portrait"
            )

    def band_for(self, frame: np.ndarray) -> int:
        return self.band or max(3, round(min(frame.shape[:2]) * 0.012))

    def __call__(
        self, frame: np.ndarray, t_ms: int | None, mask: np.ndarray | None = None
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.bn is not None:
            p = self.bn(frame)
            if mask is not None:  # 渡されたマスクの近くだけ残す
                near = (
                    cv2.dilate(
                        (mask >= 0.5).astype(np.uint8),
                        _ellipse(self.band_for(frame) * 2),
                    )
                    > 0
                )
                p = np.where(near, p, 0.0).astype(np.float32)
            return p
        if mask is not None:
            p = mask.astype(np.float32)
        else:
            assert self.person is not None
            p = self.person.fg(frame, t_ms)
        if self.model == "selfie":
            return p
        if self.model == "guided":
            # eps を小さくすると画の模様まで写るので 0.01（正解つきの合成で境目の IoU が最もよかった）
            return _guided(frame, p, max(2, round(min(h, w) / 135)), 1e-2)
        hair = None
        band = self.band_for(frame)
        if self.hair is not None:
            hair = self.hair.fg(frame, t_ms)
            if (
                mask is not None
            ):  # 渡されたマスクの近くの髪だけ使う（別の人の髪を拾わない）
                near = cv2.dilate((mask >= 0.5).astype(np.uint8), _ellipse(4 * band))
                hair = hair * (near > 0)
        assert self.vm is not None
        tri = _trimap(p, band, hair, self.hair_band)
        return self.vm(frame, tri)


def _soft_area(alpha: np.ndarray) -> float:
    return float(((alpha > 0.02) & (alpha < 0.98)).mean())


def _checker(h: int, w: int, s: int = 16) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    c = ((y // s + x // s) % 2).astype(bool)
    img = np.where(c[..., None], 205, 150).astype(np.uint8)
    return np.repeat(img, 3, axis=2)


def _over(
    color: np.ndarray, alpha: np.ndarray, bg: np.ndarray | tuple[int, int, int]
) -> np.ndarray:
    """透過の画を背景に重ねる。"""
    if isinstance(bg, tuple):
        bg = np.full(color.shape, bg, np.uint8)
    return core.overlay(bg, color, alpha.astype(np.float32))


def _soft_crop(alpha: np.ndarray, size: int) -> tuple[int, int]:
    """半透明の画素（髪の縁など）がいちばん多い size×size の場所の左上。"""
    soft = ((alpha > 0.02) & (alpha < 0.98)).astype(np.float32)
    h, w = alpha.shape
    size = min(size, h, w)
    dens = cv2.boxFilter(
        soft, -1, (size, size), normalize=False, borderType=cv2.BORDER_CONSTANT
    )
    half = size // 2
    dens[:half, :] = -1
    dens[:, :half] = -1
    dens[h - (size - half) + 1 :, :] = -1
    dens[:, w - (size - half) + 1 :] = -1
    y, x = np.unravel_index(int(np.argmax(dens)), dens.shape)
    return int(np.clip(x - half, 0, w - size)), int(np.clip(y - half, 0, h - size))


def _matte_sheet(
    frame: np.ndarray, alpha: np.ndarray, color: np.ndarray, label: str
) -> np.ndarray:
    """確認用の 1 行: 元・アルファ・市松模様・緑に重ねた画・髪の縁の拡大（元と緑）。"""
    h, w = frame.shape[:2]
    a8 = cv2.cvtColor(np.clip(alpha * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    ck = _over(color, alpha, _checker(h, w))
    gr = _over(color, alpha, (60, 190, 60))
    size = max(48, min(h, w) // 3)
    x, y = _soft_crop(alpha, size)
    z = max(1, 360 // size)

    def zoom(img: np.ndarray) -> np.ndarray:
        return cv2.resize(
            img[y : y + size, x : x + size],
            None,
            fx=z,
            fy=z,
            interpolation=cv2.INTER_NEAREST,
        )

    marked = frame[:, :, :3].copy()
    cv2.rectangle(
        marked, (x, y), (x + size, y + size), (0, 220, 255), max(1, min(h, w) // 200)
    )
    ims = [marked, a8, ck, gr, zoom(frame[:, :, :3]), zoom(a8), zoom(gr)]
    labels = [
        label,
        "アルファ",
        "市松模様に重ねた",
        "緑に重ねた",
        "拡大（元）",
        "拡大（アルファ）",
        "拡大（緑）",
    ]
    return core.contact_sheet(ims, labels, cols=7, cell_width=260)


# ── 53 ロトスコープ ───────────────────────────────────────────────────────


def _polygons(
    mask: np.ndarray, eps: float, sx: float, sy: float
) -> list[dict[str, Any]]:
    """2 値の地図の輪郭を、間引いた多角形（元の動画の座標）にする。穴は hole=true。"""
    cs, hier = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    out = []
    for c, hrow in zip(cs, hier[0] if hier is not None else [], strict=False):
        if cv2.contourArea(c) < 4:
            continue
        p = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        p[:, 0] *= sx
        p[:, 1] *= sy
        out.append({"hole": bool(hrow[3] >= 0), "points": np.round(p, 1).tolist()})
    return out


def _rotoscope_view(frame: np.ndarray, hard: np.ndarray, t: float | None) -> np.ndarray:
    """確認用: 切り抜いた外を暗くし、輪郭線（マゼンタ）を描く。"""
    view = frame[:, :, :3].copy()
    view[~hard] = (view[~hard].astype(np.float32) * 0.35).astype(np.uint8)
    th = max(2, min(frame.shape[:2]) // 220)
    _outline(view, hard, (255, 0, 255), th)
    if t is not None:
        view = _draw_labels(
            view, [(f"{t:.2f} 秒", (6, frame.shape[0] - 6), (40, 40, 40))]
        )
    return view


@command(
    53,
    "rotoscope",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--model",
        choices=["selfie", "vitmatte", "birefnet"],
        default="selfie",
        help="selfie: MediaPipe の人物の切り抜き（速い。既定）／ vitmatte: 56 と同じ髪まで細かい切り抜き"
        "（なびく髪の束も残る。540×960 で 1 コマ約 1.5 秒）／ birefnet: BiRefNet（人以外も。1 コマ約 10 秒・メモリ約 7GB）",
    ),
    arg(
        "--mask",
        help="人物の代わりに使う粗いマスク（白＝残す。白黒の動画・透過つき動画・画像。60 の出力など）",
    ),
    arg(
        "--check",
        help="輪郭線を描いた確認用の出力先（省略時は <出力>-check.mp4。画像なら .jpg）",
    ),
    arg(
        "--smooth",
        type=float,
        default=0.6,
        help="時間方向のなめらかさ（0〜0.9。動きに合わせて前のコマと混ぜ、輪郭のふるえを抑える。0 で無効。既定 0.6）",
    ),
    arg(
        "--edge",
        choices=["guided", "none"],
        default="guided",
        help="輪郭を元の画の境目に合わせる（既定 guided）",
    ),
    arg(
        "--feather",
        type=float,
        default=1.0,
        help="輪郭のぼかし（px。既定 1.0。0 でくっきり）",
    ),
    arg(
        "--keep",
        choices=["all", "largest"],
        default="all",
        help="all: 小さな塊だけ消す（既定）／ largest: 一番大きい塊だけ",
    ),
    arg(
        "--simplify",
        type=float,
        default=1.5,
        help="JSON の多角形の点を間引く許容差（px。既定 1.5）",
    ),
)
def rotoscope(a: argparse.Namespace) -> None:
    """毎コマ人物を切り抜き（輪郭を整えて時間方向にもなめらかに）、透過つき WebM・輪郭線の確認 MP4・輪郭の多角形の JSON を書く。"""
    info = core.probe(a.src)
    out = _out_path(a, info, "rotoscope", ".webm", ".png")
    if not info.is_image:
        core.require(
            out.suffix.lower() == ".webm", "出力は透過つきの .webm にしてください"
        )
    check = (
        Path(a.check)
        if a.check
        else _sibling(out, "-check" + (".jpg" if info.is_image else ".mp4"))
    )
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    core.log(
        f"切り抜きのモデル: {a.model}" + ("（--mask の地図を使う）" if a.mask else "")
    )
    matter = _Matter(a.model, video=not info.is_image)
    smooth = _Smoother(0.0 if info.is_image else a.smooth)
    pw, ph = core.scaled_size(info.width, info.height, None, a.max_side)
    masks = _mask_frames(a.mask, (pw, ph), **_rng(a)) if a.mask else None
    writer = None if info.is_image else core.VideoWriter(out, info.fps, alpha=True)
    cwriter = None if info.is_image else _writer(check, info, a)
    total = _total(info, a)
    t0 = time.time()
    model_time = 0.0
    start = a.start or 0.0
    frames_json: list[dict[str, Any]] = []
    n = 0
    for i, frame in enumerate(core.read_frames(a.src, **_rng(a), max_side=a.max_side)):
        h, w = frame.shape[:2]
        sx, sy = _src_scale(info, frame)
        t = core.t_of(i, info.fps, start)
        m = next(masks, None) if masks is not None else None
        tm = time.time()
        p = matter(frame, None if info.is_image else round(t * 1000), m)
        model_time += time.time() - tm
        p = smooth(frame, p)
        if a.edge == "guided":
            # 半径は短い辺の 1/70（540px で 8px）、eps 0.01: 正解つきの合成で境目の IoU が 0.649 → 0.70 に上がった
            p = _guided(frame, p, max(2, round(min(h, w) / 70)), 1e-2)
        hard = _fill_small_holes(_keep_components(p >= 0.5, a.keep))
        alpha = hard.astype(np.float32)
        if a.feather > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), a.feather)
        rgba = _rgba(frame, alpha)
        view = _rotoscope_view(frame, hard, None if info.is_image else t)
        polys = _polygons(hard, a.simplify, sx, sy)
        frames_json.append(
            {
                "t": t,
                "area": round(float(hard.mean()), 5),
                "points": sum(len(q["points"]) for q in polys),
                "polygons": polys,
            }
        )
        if info.is_image:
            core.save_image(out, rgba)
            core.save_image(check, view)
        else:
            assert writer is not None and cwriter is not None
            writer.write(rgba)
            cwriter.write(view)
        n += 1
        _progress(i, total, t0)
    if writer is not None and cwriter is not None:
        writer.close()
        cwriter.close()
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "model": a.model,
            "size": [int(w), int(h)],
            "source_size": [info.width, info.height],
            "fps": info.fps,
            "frames_processed": n,
            "seconds_per_frame": round(model_time / max(n, 1), 3),
            "note": "polygons は元の動画の座標（px）。hole=true は穴",
            "frames": frames_json,
        },
    )
    areas = [f["area"] for f in frames_json]
    core.log(
        f"{n} コマ。人物の面積 平均 {np.mean(areas) * 100:.1f}%（最小 {min(areas) * 100:.1f}%・最大 {max(areas) * 100:.1f}%）"
    )
    core.log(f"1 コマあたり {model_time / max(n, 1):.2f} 秒（モデルの計算だけ）")
    core.log(f"保存: {out}")
    core.log(f"保存: {check}")
    core.log(f"保存: {json_path}")


# ── 56 マッティング ───────────────────────────────────────────────────────


@command(
    56,
    "matte",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--model",
        choices=["vitmatte", "birefnet", "birefnet-portrait", "guided"],
        default="vitmatte",
        help="vitmatte: 人物の粗い切り抜き → 髪の周りを広げた 3 値の地図 → ViTMatte（既定。540×960 で 1 コマ約 2 秒）"
        "／ birefnet: BiRefNet lite（人以外も。約 10 秒・メモリ約 7GB）／ birefnet-portrait: BiRefNet の人物用"
        "（約 17〜27 秒・メモリ約 8GB）／ guided: ガイデッドフィルターだけ（速い）",
    ),
    arg(
        "--mask",
        help="粗いマスク（白＝残す。画像か白黒・透過つきの動画）。人以外を切り抜く時や、60・53 の出力を整える時",
    ),
    arg(
        "--band",
        type=int,
        help="vitmatte の「分からない帯」の幅（px。既定は短い辺の 1.2%%）",
    ),
    arg(
        "--hair-band",
        type=int,
        help="vitmatte で、髪の周りの「分からない帯」を外へ広げる幅（px。既定は --band の 10 倍）。はねた毛を拾う。"
        "背景に細い線が多い時は小さくする",
    ),
    arg("--no-hair", action="store_true", help="髪の周りの帯を広げない（vitmatte）"),
    arg(
        "--no-fg",
        action="store_true",
        help="縁の色を推定し直さない（元の画の色のまま。既定は推定する）",
    ),
    arg(
        "--alpha-out", help="アルファだけの白黒の出力先（画像なら .png、動画なら .mp4）"
    ),
    arg(
        "--smooth",
        type=float,
        default=0.0,
        help="動画のちらつきを抑える強さ（0〜0.9。既定 0）",
    ),
    arg(
        "--check",
        help="確認用の画像（元・アルファ・重ねた画・髪の縁の拡大）の出力先（省略時は <出力>-check.jpg）",
    ),
)
def matte(a: argparse.Namespace) -> None:
    """髪の毛まで細かく切り抜き、縁の色も直した透過つきの PNG（画像）か WebM（動画）を書く。"""
    info = core.probe(a.src)
    out = _out_path(a, info, "matte", ".webm", ".png")
    if not info.is_image:
        core.require(
            out.suffix.lower() == ".webm", "出力は透過つきの .webm にしてください"
        )
    else:
        core.require(
            out.suffix.lower() in {".png", ".webp", ".tif", ".tiff"},
            "出力は透過を保てる .png にしてください",
        )
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    check = Path(a.check) if a.check else _sibling(out, "-check.jpg")
    core.log(
        f"切り抜きのモデル: {a.model}" + ("（--mask の地図を使う）" if a.mask else "")
    )
    matter = _Matter(
        a.model,
        video=not info.is_image,
        band=a.band,
        hair=not a.no_hair,
        hair_band=a.hair_band,
    )
    smooth = _Smoother(0.0 if info.is_image else a.smooth)
    pw, ph = core.scaled_size(info.width, info.height, None, a.max_side)
    masks = _mask_frames(a.mask, (pw, ph), **_rng(a)) if a.mask else None
    writer = None if info.is_image else core.VideoWriter(out, info.fps, alpha=True)
    awriter = None
    if a.alpha_out and not info.is_image:
        awriter = core.VideoWriter(a.alpha_out, info.fps)
    total = _total(info, a)
    picks = {0, total // 2, total - 1}
    rows: list[np.ndarray] = []
    t0 = time.time()
    model_time = 0.0
    start = a.start or 0.0
    frames_json: list[dict[str, Any]] = []
    last: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    n = 0
    for i, frame in enumerate(core.read_frames(a.src, **_rng(a), max_side=a.max_side)):
        t = core.t_of(i, info.fps, start)
        m = next(masks, None) if masks is not None else None
        tm = time.time()
        alpha = matter(frame, None if info.is_image else round(t * 1000), m)
        alpha = smooth(frame, alpha)
        color = frame[:, :, :3] if a.no_fg else _estimate_fg(frame, alpha)
        model_time += time.time() - tm
        frames_json.append(
            {
                "t": t,
                "area": round(float(alpha.mean()), 5),
                "soft": round(_soft_area(alpha), 5),
            }
        )
        if info.is_image:
            core.save_image(out, _rgba(frame, alpha, color))
            if a.alpha_out:
                core.save_image(
                    a.alpha_out, np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
                )
        else:
            assert writer is not None
            writer.write(_rgba(frame, alpha, color))
            if awriter is not None:
                awriter.write(np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8))
        if i in picks:
            rows.append(
                _matte_sheet(
                    frame, alpha, color, f"元 {t:.2f}秒" if not info.is_image else "元"
                )
            )
        last = (frame, alpha, color, t)
        n += 1
        _progress(i, total, t0)
    if writer is not None:
        writer.close()
    if awriter is not None:
        awriter.close()
    if last is not None and not info.is_image and (n - 1) not in picks:
        # コマ数の見込みが実際より多かった時は、最後のコマも確認用に並べる
        rows.append(_matte_sheet(last[0], last[1], last[2], f"元 {last[3]:.2f}秒"))
    width = max(r.shape[1] for r in rows)
    core.save_image(
        check,
        np.vstack(
            [
                r if r.shape[1] == width else cv2.resize(r, (width, r.shape[0]))
                for r in rows
            ]
        ),
    )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "model": a.model,
            "fg_estimate": not a.no_fg,
            "frames_processed": n,
            "seconds_per_frame": round(model_time / max(n, 1), 3),
            "frames": frames_json,
        },
    )
    soft = np.mean([f["soft"] for f in frames_json])
    core.log(f"{n} コマ。半透明（髪の縁など）の画素 平均 {soft * 100:.2f}%")
    core.log(f"1 コマあたり {model_time / max(n, 1):.2f} 秒（縁の色の推定を含む）")
    if a.alpha_out:
        core.log(f"保存: {a.alpha_out}")
    core.log(f"保存: {out}")
    core.log(f"保存: {check}")
    core.log(f"保存: {json_path}")


# ── 60 マスクトラッキング ─────────────────────────────────────────────────


class _Sam2Track:
    """SAM 2.1 tiny（facebook/sam2.1-hiera-tiny。Apache-2.0）の動画版を、1 コマずつ渡して追う（transformers 5.2 の
    Sam2VideoModel の逐次入力）。覚えておくのは直近 16 コマと最初のコマだけにして、メモリが増え続けないようにする。"""

    KEEP = 16  # 物のポインタは直近 16 コマ、記憶は直近 6 コマまでしか使わない（モデルの設定）

    def __init__(self) -> None:
        self.torch = models.torch_setup()
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        repo = models.hf_repo("sam2")
        self.proc = Sam2VideoProcessor.from_pretrained(repo)
        self.net = Sam2VideoModel.from_pretrained(repo).eval()
        self.sess = self.proc.init_video_session(
            inference_device="cpu", dtype=self.torch.float32
        )

    def __call__(
        self, frame: np.ndarray, i: int, prompt: tuple[str, Any] | None = None
    ) -> tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        inputs = self.proc(images=rgb, device="cpu", return_tensors="pt")
        size = inputs.original_sizes[0]
        if prompt is not None:
            kind, value = prompt
            if kind == "mask":
                self.proc.add_inputs_to_inference_session(
                    self.sess, frame_idx=i, obj_ids=1, input_masks=value
                )
            elif kind == "box":
                self.proc.add_inputs_to_inference_session(
                    self.sess,
                    frame_idx=i,
                    obj_ids=1,
                    input_boxes=[[value]],
                    original_size=size,
                )
            else:
                pts, labs = value
                self.proc.add_inputs_to_inference_session(
                    self.sess,
                    frame_idx=i,
                    obj_ids=1,
                    input_points=[[pts]],
                    input_labels=[[labs]],
                    original_size=size,
                )
        out = self.net(
            inference_session=self.sess, frame=inputs.pixel_values[0], frame_idx=i
        )
        m = self.proc.post_process_masks(
            [out.pred_masks], original_sizes=inputs.original_sizes, binarize=False
        )[0]
        logits = m.reshape(-1, *m.shape[-2:])[0].numpy().astype(np.float32)
        score = float(out.object_score_logits.flatten()[0])
        self.sess.processed_frames.pop(i - 1, None)
        for od in self.sess.output_dict_per_obj.values():
            od["non_cond_frame_outputs"].pop(i - self.KEEP - 1, None)
        return logits, score


def _auto_person(frame: np.ndarray) -> np.ndarray:
    """最初のコマで一番大きい人物の地図（MediaPipe）。"""
    p = _Person("person", video=False).fg(frame, None)
    hard = _keep_components(p >= 0.5, "largest")
    core.require(
        hard.sum() > 0.002 * hard.size,
        "最初のコマに人物が見つかりません（--box か --point で範囲を指定してください）",
    )
    return hard


@command(
    60,
    "mask-track",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg("--box", help="最初のコマで追う物の枠 x,y,幅,高さ（元の動画の px）"),
    arg("--point", help="最初のコマで追う物の上の点 'x,y x,y ...'（元の動画の px）"),
    arg("--neg", help="追う物ではない所の点 'x,y ...'（--point と一緒に使う）"),
    arg(
        "--mask",
        help="最初のコマの範囲を白で塗った画像（元の動画と同じ大きさが望ましい）",
    ),
    arg(
        "--overlay", help="色を重ねた確認用 MP4 の出力先（省略時は <出力>-overlay.mp4）"
    ),
    arg(
        "--mask-out",
        help="白黒のマスク動画（MP4）の出力先（53・56 の --mask に渡せる）",
    ),
    arg(
        "--feather", type=float, default=1.0, help="透過の輪郭のぼかし（px。既定 1.0）"
    ),
)
def mask_track(a: argparse.Namespace) -> None:
    """最初のコマで決めた範囲（枠・点・マスク。省略時は一番大きい人物）を SAM 2.1 で最後まで追い、透過つき WebM と確認用 MP4 を書く。"""
    info = core.probe(a.src)
    core.require(
        not info.is_image, "動画を指定してください（画像 1 枚なら 51 か 56 を使う）"
    )
    out = _out_path(a, info, "mask-track", ".webm", ".png")
    core.require(out.suffix.lower() == ".webm", "出力は透過つきの .webm にしてください")
    overlay_path = Path(a.overlay) if a.overlay else _sibling(out, "-overlay.mp4")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    core.log("SAM 2.1 tiny（動画）を読み込みます")
    tracker = _Sam2Track()
    writer = core.VideoWriter(out, info.fps, alpha=True)
    owriter = _writer(overlay_path, info, a)
    mwriter = core.VideoWriter(a.mask_out, info.fps) if a.mask_out else None
    total = _total(info, a)
    t0 = time.time()
    model_time = 0.0
    start = a.start or 0.0
    frames_json: list[dict[str, Any]] = []
    prompt_desc = ""
    color = (255, 120, 0)
    n = 0
    for i, frame in enumerate(core.read_frames(a.src, **_rng(a), max_side=a.max_side)):
        h, w = frame.shape[:2]
        sx, sy = _src_scale(info, frame)
        prompt = None
        if i == 0:
            if a.mask:
                m = (
                    next(_mask_frames(a.mask, (w, h))) >= 0.5
                )  # 白黒の画像か、透過つき PNG
                prompt, prompt_desc = ("mask", m), f"マスク {a.mask}"
            elif a.box:
                x, y, bw, bh = core.parse_box(a.box)
                prompt = ("box", [x / sx, y / sy, (x + bw) / sx, (y + bh) / sy])
                prompt_desc = f"枠 {a.box}"
            elif a.point:
                pos = core.parse_points(a.point)
                neg = (
                    core.parse_points(a.neg) if a.neg else np.zeros((0, 2), np.float32)
                )
                pts = [
                    [float(px / sx), float(py / sy)] for px, py in np.vstack([pos, neg])
                ]
                labs = [1] * len(pos) + [0] * len(neg)
                prompt, prompt_desc = (
                    ("points", (pts, labs)),
                    f"点 {a.point}" + (f"（外 {a.neg}）" if a.neg else ""),
                )
            else:
                prompt, prompt_desc = (
                    ("mask", _auto_person(frame)),
                    "自動（一番大きい人物）",
                )
            core.log(f"最初のコマの指定: {prompt_desc}")
        tm = time.time()
        logits, score = tracker(frame, i, prompt)
        model_time += time.time() - tm
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        p = _guided(frame, p, max(2, round(min(h, w) / 135)), 1e-3)
        hard = p >= 0.5
        present = score > 0 and hard.any()
        if not present:
            hard[:] = False
        alpha = hard.astype(np.float32)
        if a.feather > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), a.feather)
        writer.write(_rgba(frame, alpha))
        if mwriter is not None:
            mwriter.write(hard.astype(np.uint8) * 255)
        view = frame[:, :, :3].copy()
        rec: dict[str, Any] = {
            "t": core.t_of(i, info.fps, start),
            "present": bool(present),
            "score": round(score, 3),
        }
        if present:
            view = _paint(
                view, np.where(hard, 0, -1), {0: color}, opacity=0.45, edges=False
            )
            _outline(view, hard, (255, 255, 255), max(1, min(h, w) // 300))
            ys, xs = np.nonzero(hard)
            box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            cv2.rectangle(
                view, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1
            )
            rec.update(
                {"area": round(float(hard.mean()), 5), "box": _xywh_src(box, sx, sy)}
            )
            label = f"{rec['t']:.2f} 秒"
        else:
            label = f"{rec['t']:.2f} 秒 見失った"
        view = _draw_labels(view, [(label, (6, h - 6), (40, 40, 40))])
        owriter.write(view)
        frames_json.append(rec)
        n += 1
        _progress(i, total, t0)
    writer.close()
    owriter.close()
    if mwriter is not None:
        mwriter.close()
    lost = sum(1 for f in frames_json if not f["present"])
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "model": "facebook/sam2.1-hiera-tiny（Apache-2.0。transformers の Sam2VideoModel）",
            "prompt": prompt_desc,
            "size": [int(w), int(h)],
            "fps": info.fps,
            "frames_processed": n,
            "lost_frames": lost,
            "seconds_per_frame": round(model_time / max(n, 1), 3),
            "frames": frames_json,
        },
    )
    core.log(f"{n} コマ追った（見失ったコマ {lost}）")
    core.log(f"1 コマあたり {model_time / max(n, 1):.2f} 秒（モデルの計算だけ）")
    if a.mask_out:
        core.log(f"保存: {a.mask_out}")
    core.log(f"保存: {out}")
    core.log(f"保存: {overlay_path}")
    core.log(f"保存: {json_path}")
