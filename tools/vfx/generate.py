"""切り抜き・合成（奥行き・穴埋め・描き足し）。

担当するテクニック:
   54 インペインティング（inpaint）
   55 深度推定（depth）
   58 背景ぼかし（bg-blur）
   59 アウトペインティング（outpaint）
   68 2.5Dパララックス（parallax）
   69 空の置き換え（sky-replace）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import core, models
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/generate/（bl_scene.py で Blender の正解つきの場面を作り、test_*.py で照らす）
#   bl_scene.py: 石畳・猿の頭・玉・輪・箱・柱・建物の列・物理的な空の場面（Cycles、35mm）。光線を飛ばして正しい奥行き・物の番号・
#   空の割合（3×3 の超標本化）を出す。被写界深度つき・物を消した場面・カメラを動かした場面・連番も描ける。
#   速さはどれも、他の作業と CPU 4 コアを分け合った状態の壁時計（同じ処理でも数倍ぶれた）。
NOTES: dict[int, dict[str, str]] = {
    54: {
        "status": "◎",
        "how": "LaMa（big-lama。advimman/lama の重み、Apache-2.0。Sanster/models の TorchScript 版。MD5 は lama-cleaner の値と一致）を"
        "torch.jit で CPU 実行。穴のまとまりの周りだけ切り出し（余白 = 穴の大きさの 0.75 倍・最低 64px、長辺 1024px まで）、8 の倍数に"
        "詰めて計算し、穴の所だけ 1px なじませて貼る（LaMa は穴の外を変えない: 差の最大 0 を確認）。--method telea・ns は OpenCV。"
        "マスク: --mask（画像・マスク動画）・--box・--person（MediaPipe の人物の切り抜き）・--object（COCO 80 種類の検出の枠。fast /"
        " accurate）、--dilate で広げ、自動のマスクは前後 2 コマを足し合わせる（--mask-time）。動画は 3 通り: plate（固定カメラ。"
        "最大 80 コマの「穴でない時」の中央値で背景を作り、どのコマでも隠れていた所だけ LaMa で 1 回埋め、各コマは穴の周りの明るさの比"
        "（なめらかなゲインの地図）を合わせて貼る）・track（動くカメラ。隣どうしのコマの射影変換を、穴でない所の特徴点で求めてつなぎ、"
        "穴の画素ごとに「見えているいちばん近い時刻のコマ」から 1 回だけ読む。明るさは重なる所の画素ごとの比の中央値で合わせる。"
        "どのコマでも見えない所は前の結果を同じ変換で運び、最初だけ LaMa。端の数 px の細い隙間は Telea）・frame（コマごと）。"
        "auto はカメラの動き（5 コマの全体の動きの最大）が 3px 未満なら plate、それ以上なら track",
        "measured": "Blender の場面から猿の頭（画面の 9.3%）を消す（正解 = 猿の無い場面を描いた画）: 穴の PSNR lama 24.6dB・telea 15.7・"
        "ns 16.5、SSIM 0.86・0.67・0.67（960×540 で 5.7 秒）。実写・生成の 6 枚に既知の穴: 四角 10% で lama 19.4dB・telea 15.7・ns 15.4、"
        "筆の線 10% で 25.9・21.5・21.7（lama 1 枚 約 2.2 秒）。vtest（固定カメラ）に AI 生成の人物の切り抜きを貼って消す 60 コマ: "
        "plate 32.6dB（最低 27.0）、穴の中のコマ間の変化 0.30（正解 0.36）／ frame（LaMa）23.7dB・変化 7.23（ちらつく）／ telea 22.5dB。"
        "plate は 768×576・60 コマで 6 秒。Blender のカメラが横に動き玉が転がる 24 コマ（玉は画面の 0.6〜3.8%。正解 = 玉の無い場面）: "
        "track 25.3dB（最低 19.4）・誤差のちらつき 9.1 ／ frame 20.7dB・15.6 ／ plate 19.1dB。vtest の歩く人を --object 人 で 6 秒消す:"
        " 人の形が残ったコマ 60 コマ中 2（旧設定 score 0.4・前後の足し合わせ無しでは 13、--detector accurate では 0 だが 401 秒）。"
        "home.jpg の空の鳥 31 羽（空の切り抜きの中の暗い点をマスクに）: 鳥の所の暗さ（ブラックハット）の平均 14.3 → 5.1、マスクの外の"
        "変化 0 画素",
        "limits": "LaMa は顔・文字・規則的でない大きな構造は作り直せない（顔の半分を隠した試験では背景で埋めた）。広い穴は細部がぼける。"
        "CPU で 960×540 の画 1 枚 2〜10 秒（混み具合でぶれる）、frame はコマ数 × それだけかかる。影・映り込みはマスクに入れない限り残る"
        "（--dilate で少し広げる）。--person は近くに写った人向き（遠くの小さな人は --object 人）。検出は柱の陰の人などを取りこぼし、"
        "その分は消え残る。plate は背景が止まっていて、穴の所が他のコマで一度は見えている時に向く（明るさの変化はゲインで合わせる）。"
        "track は遠い背景・地面のような平面に合う射影変換なので、穴の後ろが手前の立体（視差が大きい物）だと縁がずれる。"
        "オプティカルフローで穴の中の動きを補ってたどる方式も試したが、正解つきの試験で劣った（22〜23.7dB）。track は全コマを覚える",
        "tested": "test_inpaint.py（still: Blender の猿／holes: home・building・fruits・story の 2 場面・vtest に四角と筆の線の穴／"
        "plate: vtest に cut-stand を貼る／track: Blender の anim 24 コマ／birds: home.jpg／real: vtest の人）、test_inpaint_people.py"
        "（検出の設定の比較）、exp_track.py（track の方式の比較）。目で確認: vfx-review/54-inpaint-*.jpg",
    },
    55: {
        "status": "◎",
        "how": "Depth Anything V2 Small（depth-anything/Depth-Anything-V2-Small-hf、Apache-2.0。ai.DepthEstimator と同じモデル）の"
        "逆深度（手前ほど大きい）を、1〜99% 点で 0〜1 にして出す（手前が白。--far-white で逆）。--detail でモデルに入れる短い辺"
        "（既定 518px）。動画は時間方向に揃える: 前のコマの結果を DIS オプティカルフローで今のコマへ動かし、往復の動きが合う画素で"
        "今のコマの推定のスケールとずれを 1 次式で前の結果に合わせ（外れ値に強い 2 回の最小二乗）、その画素だけ前の結果と混ぜる"
        "（--smooth 0.5）。動きの合う画素が 3 割未満か相関 0.5 未満なら場面が変わったとして前の結果を捨てる。白〜黒の範囲は場面の頭で"
        "決め、その後は 1 コマ 5% ずつ追う。出力: 画像は灰色 PNG（8/16bit）・色分け PNG・確認 JPG・JSON（3×3 の区画の平均など）、"
        "動画は灰色・色分け・左右に並べた MP4（音声は残す）と JSON",
        "measured": "Blender の場面（正解の Z 深度。逆深度でスケールとずれを合わせて比べる。空は除く）: AbsRel 0.147・δ1（25% 以内）"
        "0.854・順位相関 0.994（比べる相手の「画面の下ほど手前」は 0.893・0.167・0.861）。--detail 364 で 0.144・0.913、700 で 0.132・"
        "0.938。空は予測でも最も遠い（空の平均 0.0002、空以外の 5% 点 0.045）。静止画に雑音と明るさの揺れを足した 48 コマ（正解は"
        "全コマ同じ）のコマ間の変化: コマごとに正規化 0.0057 → smooth 0 0.0042・0.5 0.0021・0.8 0.0012。カメラが動く Blender の 24 コマ"
        "（全コマ 1 組の合わせで比べる = スケールが揃っているか）: コマごとに正規化 AbsRel 0.165（最悪のコマ 0.306）・δ1 0.754・ちらつき"
        " 0.023 → smooth 0.5 で 0.157（最悪 0.180）・0.839・0.012（正解自体の変化 0.010）。smooth 0.8 は遅れて δ1 0.797 に下がる。"
        "story540 の 5.5〜9.5 秒: 場面の頭を 5.792・8.875 秒（本当のカット 5.79・8.88 秒）で検出。home.jpg: 空 0.0005・建物 0.835。"
        "story の人物と背景: 7 秒 0.72 / 0.08、10.5 秒 0.74 / 0.03、14 秒 0.24 / 0.01（手前の机がもっと近い）。速さ: 540×960 で"
        "1 コマ約 0.93 秒（96 コマ 89.7 秒）",
        "limits": "相対的な奥行き（距離の単位は無く、比も 1 次式の不定がある）。遠い所の差はつぶれる（Blender でも誤差は奥の建物に集中）。"
        "鏡・ガラス・空の映り込みは誤る（鏡の前の場面では人物 0.39・背景 0.34 と差が小さかった）。物の縁はなだらか（数 px）。"
        "動画のなめらかさを上げすぎると、速く動く物が遅れる（0.8 で精度が下がった）。表示の範囲がゆっくり追うので、場面の中で"
        "カメラが大きく寄ると明るさが徐々に変わる",
        "tested": "test_depth.py（still: Blender の場面／noise: 雑音と明るさの揺れの 48 コマ／seq: Blender のカメラが動く 24 コマ／"
        "real: home.jpg・story540 の 5 場面と人物の切り抜き・mp-cat_and_dog・building）、CLI で story540 の 5.5〜9.5 秒（--style side）。"
        "目で確認: vfx-review/55-depth-*.jpg",
    },
    58: {
        "status": "◎",
        "how": "55 の奥行き（ガイデッドフィルターで輪郭に沿わせる）と、ピントの奥行き（--focus auto: 人物がいれば人物の中央値、いなければ"
        "画面中央の 75% 点／ x,y の点／ 0〜1 の数）の差に比例して、ぼけの半径を決める（薄いレンズの CoC は逆深度の差に比例するので、"
        "相対的な逆深度でも形は正しい。--strength で差 1 の時の半径、既定は短い辺の 3%。--dof の幅はくっきり、--near で手前側の割合）。"
        "人物（MediaPipe の切り抜き）は必ずくっきり残す（--protect）。描き方: 半径を 1.5px ごとの層に分け（三角形の重みで隣と分け合い"
        "段差を出さない）、奥の層から円盤の形でぼかした色と覆う割合を重ね、最後に覆う割合で割る（ピントの合った物の色が背景に"
        "にじまず、手前のぼけは奥へ広がる）。光の足し合わせなので線形の色で計算。明るい点は少し持ち上げて丸い玉ボケに（--highlights）。"
        "動画は奥行きを時間方向に揃え、ピントの奥行きもゆっくり追う（場面の頭で取り直す）",
        "measured": "Blender の正解（Cycles の光学的な被写界深度。35mm・f/0.3・ピントは猿の頭 3.02m。背景のぼけの半径は最大 18.2px）: "
        "strength を正解の式（半径 = 55.1px·m × 逆深度の差）と推定の奥行きの傾きから決めると、全体の PSNR 33.3dB・SSIM 0.978"
        "（正解の奥行きを使うと 33.0・0.982）、猿以外を一様にぼかすと 31.3・0.967、ぼかさないと 24.9・0.776。奥行きで差が出る 3.5〜8m の"
        "所: 34.6dB（正解の奥行き 35.6、一様 30.1、ぼかさない 26.0）。ピントの所（猿）40.0dB。CLI も同じ値。実写（AI 生成）の人物の"
        "鮮明さ（ラプラシアンの分散）の前後の比 0.985〜1.005、背景は 0.027〜0.47（窓の場面 0.027・浴室 0.14・木の葉 0.31）。"
        "plate-laugh 45 コマの動画: 人物 1.005・背景 0.47、背景のコマ間の変化は元より増えない（−0.03）。髪の縁にハローは見えない"
        "（拡大で確認）。速さ: 960×540 の画像 8.7 秒、540×960 の動画 1 コマ約 3.3〜3.8 秒",
        "limits": "奥行きの推定が誤る所（ガラス・鏡・細い物）はぼけ方も誤る。人物の保護は MediaPipe の人物の切り抜きに頼るので、"
        "人が小さい・後ろ向きなど切り抜きが外れると縁がぼける。手前の物のぼけの後ろ（隠れていた背景）は作らない（色が伸びる）。"
        "画面の端でぼけの形が欠ける現象（口径食）・色収差は作らない。本物の玉ボケは明るさの記録の範囲を超えた光で出るので、白飛びした所は"
        "--highlights で近似するだけ。OpenCV 4.14 のガイデッドフィルターは 3 チャンネルの案内画で eps ≤ 2e-3 だと縁を保たない"
        "（段差の試験で確認。1e-2 を使う）",
        "tested": "test_bgblur.py（blender: Blender の光学ぼけと比べる／real: story540 の 7・14 秒、plate-bath、plate-laugh の動画）。"
        "目で確認: vfx-review/58-bg-blur-*.jpg",
    },
    59: {
        "status": "○",
        "how": "仕上がりの形（--aspect・--size・--pad、--align）に広げた画の、新しい所を埋める。lama: 新しい所をマスクにして LaMa で"
        "描く（計算は長辺 512px。--limit。--steps で外へ少しずつ）／ blur: ぼかして暗くした拡大で埋める（縦長を横長に置く定番）／"
        " mirror: 折り返して写し、離れるほどぼかす。動画の lama は 54 と同じ仕組み: plate（固定カメラ。背景の中央値の画を 1 回だけ"
        "描き足して全コマで使い、明るさを合わせる）・track（動くカメラ。他のコマに写っている外側の景色を射影変換で運び、無い所は"
        "前の結果を運ぶ。最初だけ LaMa）・frame。auto はカメラの動きで選ぶ。有料の生成 API は使わない（使う場合は別の道具で）",
        "measured": "6 枚（home・building・vtest・Blender は中央の 9:16 を、story の 2 場面は中央の正方形を切り出し、元の大きさまで描き足して"
        "元の画と比べる。新しい所は画面の 44〜58%）の平均: lama（512）PSNR 16.0dB（境目から 16px 以内 21.8・それより外 15.9）、"
        "lama（1024）15.1（境目 23.5・外 14.8）、3 回に分けると 12.4、blur 12.8（境目 13.2）、mirror 14.0（境目 18.6）。1 枚 約 4 秒。"
        "目で見ると、境目の近くは続きが自然、遠くはぼけた繰り返しになる（建物の塔はつぶれ、壁・木の葉・セーターは自然）。1024 では"
        "遠くが灰色の平らな面になりやすく、512 の方が模様が続いた。横に流した動画（Blender の場面を 9:16 の窓でパン、48 コマ → 16:9。"
        "正解 = 窓の外の本当の画）: track 33.9dB（最低 32.7。新しい所の 74% は他のコマの本当の画から）、plate 17.1、blur 11.2。"
        "story540 のカフェ 2.5 秒（手持ち）を 9:16 → 1:1: 60 コマ 34 秒、描き足した所のコマ間の変化 2.7（元の映像 6.6）",
        "limits": "広い所を LaMa で描くと、構造（建物・人）は作れず、ぼけた繰り返しになる（大きく描き足すほど目立つ）。人物の体が"
        "画面の端で切れていると、その先は描けない。動画の plate・track は描き足した所が「止まった背景」なので、端で動く物（手・"
        "隣の人の肩）は続かず、境目に段差が見えることがある。frame はちらつく。本格的な描き足し（拡散モデルの生成）より質は"
        "低い（この環境の CPU では動かさない）",
        "tested": "test_outpaint.py（images: 6 枚の切り出しと元の画の比較／pan: Blender の場面のパン／story: story540 のカフェ）。"
        "目で確認: vfx-review/59-outpaint-*.jpg",
    },
    68: {
        "status": "○",
        "how": "55 の奥行き（輪郭に沿わせる）を逆深度 q（いちばん奥 = 1/--depth-range、手前 = 1）にし、多値の大津の方法で層（既定 3）に"
        "分ける（層の番号はメディアンで整え、縁の画素が中間の層の細い帯にならないようにする）。人物が写っていれば（人物の検出 ＋"
        " MediaPipe の切り抜き）まとめて手前の層に。奥の層は、手前の層の後ろに隠れていた所を LaMa で 1 回だけ埋め（いちばん奥は全部）、"
        "手前の層は色の縁に沿う柔らかい切り抜きで少し外まで含める。カメラの動き（dolly・dolly-out・truck・pedestal・circle・dolly-zoom）"
        "はピンホールの式 p' = c + zoom·((p − c) − S·q)/(1 − tz·q) で各コマを描き直す: 出力の画素ごとに元の位置を反復で求め"
        "（手前寄り・奥寄りの 2 通りから始め、写し直して 0.7px 以内に戻る解の手前の方。戻らない所はその層では透明）、奥の層から重ねる。"
        "画面の外が見えないよう、全コマの端の動きから必要なだけ拡大する",
        "measured": "Blender の正解（カメラを本当に横 0.3m・0.6m、前へ 0.8m・1.6m 動かして描いた画。推定の奥行きは正解の逆深度へ"
        " 1 次式で合わせて本当の量で動かす）: 横 0.3m で PSNR 25.0dB・SSIM 0.853（1 層 24.6、正解の奥行き 30.9、画面全体の 2D の"
        "最良の拡大・移動 18.8、動かさない 16.9）、横 0.6m 22.0（2D 16.8）、前 0.8m 25.2（2D 18.7）、前 1.6m 21.4（2D 16.3）。"
        "反復の解を確かめる前は、層の中の段差（玉とその後ろの地面）で物が二重に写り、正解の奥行きでも 25.8dB だった（横 0.3m）。"
        "home.jpg・mp-cat_and_dog・story540 の人物で 3 秒（90 コマ）: 13〜38 秒、描き直しは 1 コマ約 0.1 秒、端を隠す拡大は"
        " 1.01〜1.12 倍。目で見ると、標準の大きさでは自然。dolly-zoom の最後で頭の上に細い筋、猫の耳の先に小さな点が残る",
        "limits": "写真 1 枚の奥行きなので、隠れていた所は LaMa の想像（大きく動かすほど見える）。層は奥行きのしきい値で分けるので、"
        "細い物（髪の毛先・耳の先・枝）は層の境目で欠けたり点が残ったりする（人物は切り抜きで補う）。奥行きの縁のずれで、動きが大きいと"
        "縁が裂ける（--amount を 1 以下に）。本格的な 3D 写真の手法（層ごとに色と奥行きを学習で埋める LDI）より簡易: 隠れた所の奥行きは"
        "近くの値を伸ばすだけ。--depth-range は写真に合わせて選ぶ（距離の比は推定できない）",
        "tested": "test_parallax.py（blender: Blender の truck・dolly の正解の画／real: home.jpg の truck・dolly、mp-cat_and_dog の"
        " dolly・circle、story540 の 7 秒の dolly-zoom・truck）。目で確認: vfx-review/68-parallax-*.jpg",
    },
    69: {
        "status": "◎",
        "how": "UperNet ConvNeXt small（openmmlab/upernet-convnext-small、MIT。ADE20K の「sky」）で空の確率を求め（短い辺 512px、"
        "1/4 の大きさの確率を広げる）、0.5 を中心に 6 倍に急にしてからガイデッドフィルター（半径 = 短い辺の 1/70、eps 1e-2）で輪郭に"
        "沿わせる。縁の画素は古い空の色を抜く: (I − a·B)/(1 − a)（B = 近くの確かな空の色の平均）を、a が大きいほど近くの前景の色の"
        "平均へ寄せて混ぜる。新しい空: --sky の画像・動画（覆う大きさに拡大・上寄せ）か、--preset（sunset・dusk・blue・night の"
        "グラデーション・横に長い雲・太陽の光・星を作る）。前景の色を新しい空へ寄せる（--harmonize: 線形の光で空の色の比を 0.6〜1.6 に"
        "抑えて k 乗、明るさは 0.3k 乗）。動画は前のコマの空の確率をオプティカルフローで運んで混ぜ（--smooth）、空でない所の全体の"
        "動き（相似変換）を積み重ねて、1.35 倍に作った空をずらす（--motion auto）",
        "measured": "Blender の場面（空の割合の正解は 3×3 の光線）: IoU 0.981・平均の誤差 0.012（確率そのまま 0.977・0.017、"
        "ガイデッドだけ 0.981・0.022）。AI 生成の人物の切り抜きを本物の空（home.jpg の空）に重ねた画（正解 = 1 − 人物の透過）: IoU 0.988・"
        "誤差 0.012、髪の縁の帯の誤差 0.089（確率そのまま 0.125）。新しい夕焼けへの置き換えを正解の割合で重ねた画と比べると、髪の縁の帯の"
        " PSNR 23.1dB（古い空の色を抜かないと 21.5、最初に試した「境目で空と前景の色の間に置く」＋そのまま抜く方式は 20.1）。"
        "home.jpg を 2 倍にして 512×384 の窓で右へ 1 コマ 4px パンした 30 コマ: 推定した動き −3.99px/コマ（誤差 0.01px）、"
        "新しい空は前のコマを 4px ずらした物と一致（差 0.066。動かさない場合の差 0.268。--motion none では逆に 0.055 と 0.416）、"
        "空の切り抜きはずれを戻して IoU 0.9997（最低 0.998）。home.jpg の空 52.9%・building.jpg 9.8%（木の枝の間も抜けた）。"
        "速さ: 960×540 で切り抜き 1.9 秒、画像は読み込みを含め 6〜7 秒、512×384 の動画 1 コマ約 2 秒",
        "limits": "空と判定されない物（遠い山・霧・水面の映り込み）は置き換えない。鳥・電線は空に含まれて消えることがある（home.jpg の"
        "鳥は一部が残り一部が消えた）。前景の色寄せは全体に一様（夕日の向き・影の方向は変えない）。night では建物が昼の明るさの"
        "まま（--harmonize 1 でも暗さは 0.3 乗しか追わない）。動画の空の動きは相似変換（パン・ズーム）だけで、空と景色の視差・"
        "雲の流れは作らない。UperNet は CPU で 1 コマ約 2 秒",
        "tested": "test_sky.py（blender: Blender の空の割合／comp: cut-laugh を home.jpg の空に重ねた正解つきの画／pan: home.jpg の"
        "パン／real: home.jpg・building.jpg を 4 種類の空で）、exp_sky.py・exp_sky2.py（縁の作り方と色の抜き方の比較）。"
        "目で確認: vfx-review/69-sky-*.jpg",
    },
}


# ── 共通の部品 ────────────────────────────────────────────────────────────


def _rng(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": getattr(a, "start", 0.0) or 0.0,
        "duration": getattr(a, "duration", None),
        "max_frames": getattr(a, "max_frames", None),
    }


def _out(a: argparse.Namespace, info: core.MediaInfo, tag: str, video_ext: str = ".mp4",
         image_ext: str = ".png") -> Path:  # fmt: skip
    if a.out:
        return Path(a.out)
    return core.default_out(a.src, tag, image_ext if info.is_image else video_ext)


def _json_path(a: argparse.Namespace, out: Path) -> Path:
    return Path(a.json) if getattr(a, "json", None) else out.with_suffix(".json")


def _even(img: np.ndarray) -> np.ndarray:
    return core.ensure_even(img)


def _frames(src: str | Path, a: argparse.Namespace, **kw: Any) -> Iterator[np.ndarray]:
    return core.read_frames(src, max_side=getattr(a, "max_side", None), **_rng(a), **kw)


def _total(info: core.MediaInfo, a: argparse.Namespace) -> int:
    if info.is_image:
        return 1
    n = round((a.duration or max(info.duration - (a.start or 0.0), 0.0)) * info.fps)
    if a.max_frames:
        n = min(n, a.max_frames)
    return max(n, 1)


def _progress(i: int, total: int, t0: float) -> None:
    step = max(1, total // 10)
    if total > 1 and ((i + 1) % step == 0 or i + 1 == total):
        el = time.time() - t0
        core.log(f"  {i + 1}/{total} コマ（{el:.0f} 秒・1 コマ {el / (i + 1):.2f} 秒）")


def _guided(
    guide_bgr: np.ndarray, p: np.ndarray, radius: int, eps: float
) -> np.ndarray:
    """ガイデッドフィルター（He ら 2010）: 地図 p の縁を元の画の輪郭に沿わせる。

    OpenCV 4.14 の 3 チャンネル（0〜1 の float）の案内画では、eps が 2e-3 以下だと縁を保たず、ただの平均（ぼかし）に
    なる（段差の試験で確認）。縁を保つには eps = 1e-2 前後を使う（1 チャンネルの案内画とは振る舞いが逆）。"""
    g = guide_bgr[:, :, :3].astype(np.float32) / 255.0
    return cv2.ximgproc.guidedFilter(g, p.astype(np.float32), max(1, int(radius)), eps)


def _feather(mask: np.ndarray, px: float) -> np.ndarray:
    """0/1 の地図の縁を px ぼかした 0〜1 の地図にする。"""
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    if px <= 0:
        return m
    k = int(max(3, round(px * 3)) | 1)
    return cv2.GaussianBlur(m, (k, k), px)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if px <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m, k)


def _srgb_to_lin(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    y = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ── 奥行き（Depth Anything V2 Small） ─────────────────────────────────────


class _Depth:
    """Depth Anything V2 Small（Apache-2.0。ai.DepthEstimator と同じモデル）の、正規化する前の逆深度。

    値が大きいほど手前。スケールとずれ（a·d+b）はコマごとに不定（相対的な奥行き）。
    """

    def __init__(self, detail: int = 518) -> None:
        from .ai import DepthEstimator

        core.require(detail >= 140, "--detail は 140 以上にしてください")
        self.est = DepthEstimator()
        self.torch = self.est.torch
        self.detail = int(round(detail / 14) * 14)

    def raw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        kw: dict[str, Any] = {}
        if self.detail != 518:
            kw["size"] = {"height": self.detail, "width": self.detail}
        inputs = self.est.proc(images=rgb, return_tensors="pt", **kw)
        pred = self.est.net(**inputs).predicted_depth
        pred = self.torch.nn.functional.interpolate(
            pred[:, None], size=(h, w), mode="bicubic", align_corners=False
        )
        return pred[0, 0].numpy().astype(np.float32)


def _norm01(
    d: np.ndarray, lo: float | None = None, hi: float | None = None
) -> np.ndarray:
    lo = float(np.percentile(d, 1)) if lo is None else lo
    hi = float(np.percentile(d, 99)) if hi is None else hi
    return np.clip((d - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """y ≈ a·x + b（外れ値に強い 2 回の最小二乗）。(a, b, 相関) を返す。"""
    if len(x) > 40000:
        idx = np.random.default_rng(0).choice(len(x), 40000, replace=False)
        x, y = x[idx], y[idx]
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    a, b = np.polyfit(x, y, 1)
    r = np.abs(a * x + b - y)
    keep = r <= np.percentile(r, 80)
    if keep.sum() > 50:
        a, b = np.polyfit(x[keep], y[keep], 1)
    c = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else 0.0
    return float(a), float(b), c


class _Flow:
    """前後のコマの動き（DIS オプティカルフロー）を、縮めた灰色の画で求めて元の大きさに戻す。"""

    def __init__(self, side: int = 480) -> None:
        self.side = side
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def small(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        f = min(1.0, self.side / max(h, w))
        g = core.to_gray(frame)
        if f < 1.0:
            g = cv2.resize(
                g,
                (max(8, round(w * f)), max(8, round(h * f))),
                interpolation=cv2.INTER_AREA,
            )
        return g

    def warp_prev(self, gray: np.ndarray, prev_gray: np.ndarray, size: tuple[int, int],
                  thr: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
        """今のコマの各画素が前のコマのどこにあったか（remap 用の地図 x, y）と、往復の動きが合う所。"""
        w, h = size
        fcp = self.dis.calc(gray, prev_gray, None)  # 今 → 前
        fpc = self.dis.calc(prev_gray, gray, None)  # 前 → 今
        sh, sw = gray.shape
        gx, gy = np.meshgrid(
            np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32)
        )
        mx, my = gx + fcp[..., 0], gy + fcp[..., 1]
        back = cv2.remap(
            fpc,
            mx,
            my,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(1e3, 1e3),
        )
        err = np.hypot(fcp[..., 0] + back[..., 0], fcp[..., 1] + back[..., 1])
        inside = (mx >= 0) & (my >= 0) & (mx <= sw - 1) & (my <= sh - 1)
        ok = ((err < thr) & inside).astype(np.uint8)
        fx, fy = w / sw, h / sh
        flow = cv2.resize(fcp, (w, h), interpolation=cv2.INTER_LINEAR)
        X, Y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        mapx = X + flow[..., 0] * fx
        mapy = Y + flow[..., 1] * fy
        valid = cv2.resize(ok, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return mapx, mapy, valid


class _DepthVideo:
    """動画の奥行きを時間方向になめらかにする。

    前のコマの結果をオプティカルフローで今のコマへ動かし、(1) 今のコマの推定のスケールとずれを前の結果に合わせ
    （コマごとに明るさの範囲が跳ねるのを防ぐ）、(2) 動きが確かな画素だけ前の結果と混ぜる（smooth の割合）。
    場面が切り替わった（動きの合う画素が少ない・相関が低い）時は前の結果を捨てる。表示の範囲（白〜黒）は
    場面の頭で決め、その後はゆっくり（1 コマ 5%）追う。
    """

    def __init__(
        self, model: _Depth, smooth: float = 0.5, flow_side: int = 480
    ) -> None:
        self.model = model
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self.flow = _Flow(flow_side)
        self.prev: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self.lo = 0.0
        self.hi = 1.0
        self.log: list[dict[str, Any]] = []

    def raw_smoothed(
        self, frame: np.ndarray, raw: np.ndarray | None = None
    ) -> np.ndarray:
        d = self.model.raw(frame) if raw is None else raw
        h, w = d.shape
        gray = self.flow.small(frame)
        rec: dict[str, Any] = {"reset": False}
        if self.prev is None or self.prev_gray is None or self.prev.shape != d.shape:
            s = d
            rec["reset"] = True
        else:
            mapx, mapy, valid = self.flow.warp_prev(gray, self.prev_gray, (w, h))
            frac = float(valid.mean())
            rec["valid"] = round(frac, 3)
            prev_w = cv2.remap(
                self.prev, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            if frac < 0.3:
                s = d
                rec["reset"] = True
            else:
                a, b, c = _fit_affine(d[valid], prev_w[valid])
                rec.update({"a": round(a, 4), "b": round(b, 4), "corr": round(c, 3)})
                if c < 0.5 or a <= 0:
                    s = d
                    rec["reset"] = True
                else:
                    s = a * d + b
                    if self.smooth > 0:
                        k = self.smooth
                        s = np.where(valid, (1 - k) * s + k * prev_w, s).astype(
                            np.float32
                        )
        lo_t, hi_t = float(np.percentile(s, 1)), float(np.percentile(s, 99))
        if rec["reset"]:
            self.lo, self.hi = lo_t, hi_t
        else:
            self.lo += 0.05 * (lo_t - self.lo)
            self.hi += 0.05 * (hi_t - self.hi)
        self.prev = s.astype(np.float32)
        self.prev_gray = gray
        self.log.append(rec)
        return self.prev

    def __call__(self, frame: np.ndarray, raw: np.ndarray | None = None) -> np.ndarray:
        s = self.raw_smoothed(frame, raw)
        return _norm01(s, self.lo, self.hi)


def _depth_view(d01: np.ndarray, style: str, far_white: bool = False) -> np.ndarray:
    v = 1.0 - d01 if far_white else d01
    if style == "color":
        return core.heatmap(d01, lo=0.0, hi=1.0)
    g = np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _grid_means(d01: np.ndarray, n: int = 3) -> list[list[float]]:
    h, w = d01.shape
    return [
        [
            round(
                float(
                    d01[
                        i * h // n : (i + 1) * h // n, j * w // n : (j + 1) * w // n
                    ].mean()
                ),
                3,
            )
            for j in range(n)
        ]
        for i in range(n)
    ]


# ── 55 深度推定 ───────────────────────────────────────────────────────────


@command(
    55,
    "depth",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--style",
        choices=["gray", "color", "side"],
        default="gray",
        help="動画の出力: gray（手前が白の灰色。マスクに使える。既定）／ color（色分け）／ side（元の映像と左右に並べる）",
    ),
    arg(
        "--far-white",
        action="store_true",
        help="灰色の地図を、遠い所が白・手前が黒にする",
    ),
    arg(
        "--smooth",
        type=float,
        default=0.5,
        help="動画の時間方向のなめらかさ（0〜0.95。前のコマの結果を混ぜる割合。0 でコマごとに独立。既定 0.5）",
    ),
    arg(
        "--detail",
        type=int,
        default=518,
        help="モデルに入れる短い辺（px。14 の倍数に丸める。大きいほど細かいが遅い。既定 518）",
    ),
    arg(
        "--bits",
        type=int,
        choices=[8, 16],
        default=8,
        help="画像の灰色の PNG の階調（既定 8。16 で段差が出にくい）",
    ),
)
def depth(a: argparse.Namespace) -> None:
    """画面の奥行き（相対的な距離。手前が白）を推定し、灰色・色分けの地図と JSON を書き出す。"""
    info = core.probe(a.src)
    out = _out(a, info, "depth")
    json_path = _json_path(a, out)
    t0 = time.time()
    core.log("Depth Anything V2 Small（Apache-2.0）を読み込みます")
    model = _Depth(a.detail)
    if info.is_image:
        img = next(_frames(a.src, a))
        raw = model.raw(img)
        d01 = _norm01(raw)
        g = 1.0 - d01 if a.far_white else d01
        if a.bits == 16:
            gray = np.clip(g * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        else:
            gray = np.clip(g * 255.0 + 0.5, 0, 255).astype(np.uint8)
        core.save_image(out, gray)
        color_path = out.with_name(out.stem + "-color.png")
        core.save_image(color_path, core.heatmap(d01, lo=0.0, hi=1.0))
        prev_path = out.with_name(out.stem + "-preview.jpg")
        core.save_image(
            prev_path,
            core.contact_sheet(
                [img, _depth_view(d01, "gray", a.far_white), _depth_view(d01, "color")],
                [
                    "元の画",
                    "奥行き（手前が白）" if not a.far_white else "奥行き（遠いほど白）",
                    "色分け（赤が手前）",
                ],
                cols=3,
                cell_width=min(480, img.shape[1]),
            ),
        )
        core.write_json(
            json_path,
            {
                "source": str(a.src),
                "size": [img.shape[1], img.shape[0]],
                "model": "depth-anything/Depth-Anything-V2-Small-hf（Apache-2.0）",
                "detail": model.detail,
                "meaning": "0〜1 の相対的な奥行き（1 が手前）。距離の単位はない（並び順と比の目安）",
                "near_white": not a.far_white,
                "raw_percentiles": {
                    "p1": float(np.percentile(raw, 1)),
                    "p99": float(np.percentile(raw, 99)),
                },
                "mean": float(d01.mean()),
                "grid3x3_mean": _grid_means(d01),
                "seconds": round(time.time() - t0, 2),
            },
        )
        core.log(
            f"奥行きの平均 {d01.mean():.2f}（0 = 遠い・1 = 手前）。3×3 の区画の平均: {_grid_means(d01)}"
        )
        core.log(f"保存: {color_path}")
        core.log(f"保存: {prev_path}")
        core.log(f"保存: {json_path}")
        core.log(f"保存: {out}")
        return
    vid = _DepthVideo(model, a.smooth)
    total = _total(info, a)
    writer = core.VideoWriter(
        out,
        info.fps,
        audio_from=a.src,
        audio_start=a.start or 0.0,
        audio_duration=a.duration,
    )
    keep_t = set(np.linspace(0, total - 1, 6).round().astype(int).tolist())
    sheet, labels, per = [], [], []
    with writer:
        for i, frame in enumerate(_frames(a.src, a)):
            frame = _even(frame)
            d01 = vid(frame)
            if a.style == "side":
                view = np.hstack(
                    [frame[:, :, :3], _depth_view(d01, "gray", a.far_white)]
                )
            else:
                view = _depth_view(d01, a.style, a.far_white)
            writer.write(view)
            rec = vid.log[-1]
            per.append(
                {
                    "t": core.t_of(i, info.fps, a.start or 0.0),
                    "mean": round(float(d01.mean()), 4),
                    **rec,
                }
            )
            if i in keep_t:
                sheet += [frame, _depth_view(d01, "color")]
                labels += [
                    f"{per[-1]['t']:.2f} 秒",
                    "奥行き" + ("（場面の頭）" if rec["reset"] else ""),
                ]
            _progress(i, total, t0)
    resets = [p["t"] for p in per if p["reset"]]
    prev_path = out.with_name(out.stem + "-preview.jpg")
    core.save_image(
        prev_path, core.contact_sheet(sheet, labels, cols=4, cell_width=300)
    )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "model": "depth-anything/Depth-Anything-V2-Small-hf（Apache-2.0）",
            "detail": model.detail,
            "smooth": vid.smooth,
            "style": a.style,
            "near_white": not a.far_white,
            "frames": len(per),
            "fps": info.fps,
            "scene_starts": resets,
            "per_frame": per,
            "seconds": round(time.time() - t0, 2),
        },
    )
    core.log(
        f"{len(per)} コマ。場面の頭（前の結果を捨てた所）: {len(resets)} か所 {resets[:12]}"
    )
    core.log(f"保存: {prev_path}")
    core.log(f"保存: {json_path}")
    core.log(f"保存: {out}")


# ── 穴埋め（LaMa・OpenCV） ───────────────────────────────────────────────


class _Lama:
    """LaMa（big-lama。advimman/lama の重み、Apache-2.0。Sanster/models の TorchScript 版）で穴を埋める。"""

    def __init__(self) -> None:
        self.torch = models.torch_setup()
        path = models.fetch("lama")
        self.net = self.torch.jit.load(str(path), map_location="cpu").eval()

    def run(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """img（BGR）と mask（>0 が穴）→ 同じ大きさの BGR。大きさは内部で 8 の倍数に詰める。"""
        torch = self.torch
        h, w = img.shape[:2]
        ph, pw = (-h) % 8, (-w) % 8
        im = img[:, :, :3]
        mk = (mask > 0).astype(np.uint8)
        if ph or pw:
            im = cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            mk = cv2.copyMakeBorder(mk, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        x = (
            torch.from_numpy(
                cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].copy()
            ).float()
            / 255.0
        )
        m = torch.from_numpy(mk.astype(np.float32)[None, None].copy())
        with torch.inference_mode():
            y = self.net(x, m)[0].permute(1, 2, 0).numpy()
        y = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)[:h, :w]
        return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


@cache
def _lama() -> _Lama:
    core.log("LaMa（big-lama、Apache-2.0）を読み込みます")
    return _Lama()


def _boxes_for(
    mask: np.ndarray, margin_ratio: float, min_margin: int
) -> list[list[int]]:
    """穴のまとまりごとに、周りを含めた枠（x0, y0, x1, y1）。重なる枠はまとめる。"""
    h, w = mask.shape
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        m = max(min_margin, round(margin_ratio * max(bw, bh)))
        boxes.append(
            [max(0, x - m), max(0, y - m), min(w, x + bw + m), min(h, y + bh + m)]
        )
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        out: list[list[int]] = []
        for b in boxes:
            for o in out:
                if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    merged = True
                    break
            else:
                out.append(list(b))
        boxes = out
    return boxes


def _paste(
    dst: np.ndarray, src: np.ndarray, hole: np.ndarray, feather: float = 1.0
) -> np.ndarray:
    """穴の所だけ src を貼る（縁は feather px なじませる。穴の中は必ず src）。"""
    hole = hole > 0
    if feather > 0:
        al = _feather(_dilate(hole, max(1, round(feather))), feather)
        al[hole] = 1.0
    else:
        al = hole.astype(np.float32)
    return core.overlay(dst, src, al)


def _fill(img: np.ndarray, hole: np.ndarray, method: str = "lama", *, limit: int = 1024,
          radius: int = 5) -> np.ndarray:  # fmt: skip
    """穴（>0）を埋めた画を返す。lama は穴の周りだけ切り出して計算する（大きな画を速く）。"""
    hole = (hole > 0).astype(np.uint8)
    if not hole.any():
        return img[:, :, :3].copy()
    base = img[:, :, :3]
    if method in ("telea", "ns"):
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        res = cv2.inpaint(np.ascontiguousarray(base), hole, radius, flag)
        return _paste(base, res, hole, 0.0)
    core.require(method == "lama", f"知らない方法です: {method}（lama / telea / ns）")
    h, w = hole.shape
    out = base.copy()
    boxes = _boxes_for(hole, 0.75, 64)
    if sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) > 0.6 * h * w:
        boxes = [[0, 0, w, h]]
    lama = _lama()
    for x0, y0, x1, y1 in boxes:
        crop = base[y0:y1, x0:x1]
        cm = hole[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        f = min(1.0, limit / max(ch, cw))
        if f < 1.0:
            sw, sh = max(8, round(cw * f)), max(8, round(ch * f))
            cs = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
            ms = cv2.resize(cm, (sw, sh), interpolation=cv2.INTER_AREA) > 0
            res = lama.run(cs, ms.astype(np.uint8))
            res = cv2.resize(res, (cw, ch), interpolation=cv2.INTER_CUBIC)
        else:
            res = lama.run(crop, cm)
        out[y0:y1, x0:x1] = _paste(out[y0:y1, x0:x1], res, cm, 1.0)
    return out


# ── マスクの作り方 ────────────────────────────────────────────────────────


def _parse_names(text: str) -> list[str]:
    from .detectors import COCO_JA

    inv = {v: k for k, v in COCO_JA.items()}
    out = []
    for n in [s.strip() for s in text.replace("、", ",").split(",") if s.strip()]:
        if n in COCO_JA:
            out.append(n)
        elif n in inv:
            out.append(inv[n])
        else:
            raise core.VfxError(
                f"知らない種類です: {n}（COCO の 80 種類。例: 人, 車, 鳥, 犬, person, bird）"
            )
    return out


class _Masker:
    """--mask（画像・動画）・--box・--person・--object から、コマごとの穴（0/1）を作る。"""

    def __init__(
        self, a: argparse.Namespace, info: core.MediaInfo, size: tuple[int, int]
    ) -> None:
        self.size = size  # (w, h)
        w, h = size
        self.static = np.zeros((h, w), np.uint8)
        self.video_mask: Iterator[np.ndarray] | None = None
        self.person = None
        self.detector = None
        self.info = info
        auto = False
        if getattr(a, "mask", None):
            mi = core.probe(a.mask)
            if mi.is_image:
                m = core.load_image(a.mask, alpha=True)
                g = (
                    m[:, :, 3]
                    if (m[:, :, 3] < 250).any() and not (m[:, :, :3] > 127).any()
                    else core.to_gray(m)
                )
                self.static |= cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA) > 127
            else:
                self.video_mask = core.read_frames(
                    a.mask, start=a.start or 0.0, size=(w, h)
                )
        for b in getattr(a, "box", None) or []:
            x, y, bw, bh = core.parse_box(b)
            sx, sy = w / info.width, h / info.height
            x0, y0 = max(0, round(x * sx)), max(0, round(y * sy))
            self.static[y0 : round((y + bh) * sy), x0 : round((x + bw) * sx)] = 1
        if getattr(a, "person", False):
            from .ai import PersonSegmenter

            self.person = PersonSegmenter("selfie", video=not info.is_image)
            auto = True
        if getattr(a, "object", None):
            from .detectors import ObjectDetector

            names = _parse_names(a.object)
            self.detector = ObjectDetector(
                getattr(a, "detector", "fast"), getattr(a, "score", 0.4), names
            )
            auto = True
        core.require(
            self.static.any() or self.video_mask is not None or auto,
            "消す所を指定してください（--mask・--box・--person・--object のどれか）",
        )
        short = min(w, h)
        d = getattr(a, "dilate", None)
        self.dilate = (
            int(d) if d is not None else (round(0.012 * short) + 2 if auto else 2)
        )
        mt = getattr(a, "mask_time", None)
        self.time_pad = (
            int(mt) if mt is not None else (2 if auto and not info.is_image else 0)
        )

    def __call__(self, frame: np.ndarray, i: int) -> np.ndarray:
        h, w = frame.shape[:2]
        m = self.static.copy()
        if self.video_mask is not None:
            mf = next(self.video_mask, None)
            if mf is not None:
                m |= core.to_gray(mf) > 127
        if self.person is not None:
            t_ms = round(i * 1000.0 / self.info.fps) if not self.info.is_image else None
            p = self.person(frame, t_ms)
            if p.shape != (h, w):
                p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
            pm = (p > 0.5).astype(np.uint8)
            pm = cv2.morphologyEx(
                pm,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            )
            m |= pm
        if self.detector is not None:
            for d in self.detector(frame):
                x1, y1, x2, y2 = (round(v) for v in d.box)
                m[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = 1
        return _dilate(m, self.dilate)


def _mask_stream(
    masker: _Masker, frames: Iterator[np.ndarray]
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """(番号, コマ, 穴) を順に返す。穴は前後 time_pad コマの穴を足し合わせる（検出の取りこぼしを補う）。"""
    from collections import deque

    k = masker.time_pad
    buf: deque[tuple[int, np.ndarray, np.ndarray]] = deque()

    def emit(i: int) -> tuple[int, np.ndarray, np.ndarray]:
        frame = next(f for j, f, _m in buf if j == i)
        m = np.zeros(frame.shape[:2], np.uint8)
        for j, _f, r in buf:
            if abs(j - i) <= k:
                m |= r
        return i, frame, m

    last = -1
    for j, f in enumerate(frames):
        f = _even(f)[:, :, :3]
        buf.append((j, f, masker(f, j)))
        last = j
        if j - k >= 0:
            yield emit(j - k)
        while buf and buf[0][0] < j - 2 * k + 1:
            buf.popleft()
    for i in range(max(0, last - k + 1), last + 1):
        yield emit(i)


# ── 動画の穴埋め: 固定カメラ（クリーンプレート）・動くカメラ（前後のコマを重ねる） ───────────


def _gain_field(
    frame: np.ndarray, ref: np.ndarray, known: np.ndarray, sigma_ratio: float = 0.06
) -> np.ndarray:
    """frame ≈ gain × ref となる、なめらかな明るさ・色の比（チャンネルごと）。known が 1 の所だけで測る。

    known から遠い所（描き足した広い所など）は、known 全体の比へなめらかに移る。
    """
    h, w = frame.shape[:2]
    f = 0.125
    sw, sh = max(8, round(w * f)), max(8, round(h * f))
    fs = (
        cv2.resize(frame[:, :, :3], (sw, sh), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        + 1.0
    )
    rs = (
        cv2.resize(ref[:, :, :3], (sw, sh), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        + 1.0
    )
    ks = cv2.resize(known.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    ks = (ks > 0.99).astype(np.float32)
    if ks.sum() < 4:
        return np.ones((h, w, 3), np.float32)
    sig = max(1.0, sigma_ratio * max(sw, sh))
    wb = cv2.GaussianBlur(ks, (0, 0), sig)
    s = np.clip(wb / 0.15, 0.0, 1.0)
    g = np.empty_like(fs)
    for c in range(3):
        num = cv2.GaussianBlur(fs[:, :, c] * ks, (0, 0), sig)
        den = cv2.GaussianBlur(rs[:, :, c] * ks, (0, 0), sig)
        g_all = float((fs[:, :, c] * ks).sum() / max(1e-3, (rs[:, :, c] * ks).sum()))
        g[:, :, c] = s * (num / np.maximum(den, 1e-3)) + (1.0 - s) * g_all
    g = np.clip(g, 0.5, 2.0)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)


def _masked_median(
    stack: np.ndarray, valid: np.ndarray, rows: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    """(N,H,W,3) の中で、valid（N,H,W）の画素だけの中央値。見えた回数も返す。"""
    _n, h, w, _ = stack.shape
    out = np.zeros((h, w, 3), np.uint8)
    cnt = valid.sum(axis=0).astype(np.int32)
    for y0 in range(0, h, rows):
        s = stack[:, y0 : y0 + rows].astype(np.float32)
        v = valid[:, y0 : y0 + rows]
        s[~v] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(s, axis=0)
        out[y0 : y0 + rows] = np.nan_to_num(np.clip(med + 0.5, 0, 255), nan=0).astype(
            np.uint8
        )
    return out, cnt


def _sample_idx(total: int, k: int) -> list[int]:
    return sorted(
        set(
            np.linspace(0, max(0, total - 1), min(k, total))
            .round()
            .astype(int)
            .tolist()
        )
    )


def _homography(
    prev_gray: np.ndarray, gray: np.ndarray, known: np.ndarray | None
) -> np.ndarray:
    """前のコマ → 今のコマの射影変換（穴でない所の特徴点を追う）。求まらなければ単位行列。"""
    from .ai import track_points

    mk = None
    if known is not None:
        mk = cv2.erode(known.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    pts = cv2.goodFeaturesToTrack(prev_gray, 800, 0.01, 7, mask=mk, blockSize=7)
    if pts is None or len(pts) < 12:
        return np.eye(3)
    p0 = pts.reshape(-1, 2)
    p1, ok = track_points(prev_gray, gray, p0)
    if ok.sum() < 12:
        return np.eye(3)
    H, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 1.5)
    if H is None or inl is None or inl.sum() < 10:
        return np.eye(3)
    return H


def _camera_motion(frames: list[np.ndarray]) -> float:
    """最初のコマと他のコマの間の、画面の動きの最大値（px。動かないカメラなら 1 px ほど）。"""
    from .ai import global_motion

    g0 = core.to_gray(frames[0])
    worst = 0.0
    h, w = g0.shape
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
    for f in frames[1:]:
        m, _n = global_motion(g0, core.to_gray(f))
        if m is None:
            return 1e3
        moved = corners @ m[:, :2].T + m[:, 2]
        worst = max(worst, float(np.abs(moved - corners).max()))
    return worst


@command(
    54,
    "inpaint",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--mask",
        help="消す所のマスク（白 = 消す）。PNG など、または動画と同じ長さのマスク動画",
    ),
    arg(
        "--box",
        action="append",
        help="消す所の枠 x,y,幅,高さ（元の大きさの px。何度でも指定できる）",
    ),
    arg(
        "--person",
        action="store_true",
        help="人物を自動で見つけて消す（MediaPipe の人物の切り抜き。近くに写った人向き）",
    ),
    arg(
        "--object",
        help="消す物の種類（COCO の 80 種類。日本語か英語をカンマ区切り。例: 人,車）。見つけた枠を消す",
    ),
    arg(
        "--detector",
        choices=["fast", "accurate"],
        default="fast",
        help="--object の検出器（fast: EfficientDet-Lite2・速い／accurate: Faster R-CNN v2・小さな物にも強いが 1 コマ数秒）",
    ),
    arg(
        "--score",
        type=float,
        default=0.3,
        help="--object の確からしさの下限（0〜1。既定 0.3。plate では余分に消しても背景になるだけなので低めがよい）",
    ),
    arg(
        "--mask-time",
        type=int,
        help="前後このコマ数の穴を足し合わせる（検出の取りこぼし対策。既定: --person・--object は 2、それ以外は 0）",
    ),
    arg(
        "--dilate",
        type=int,
        help="マスクを広げる px（既定: --person・--object は短い辺の 1.2%%＋2、それ以外は 2）",
    ),
    arg(
        "--method",
        choices=["lama", "telea", "ns"],
        default="lama",
        help="埋め方: lama（AI。広い穴も自然。既定）／ telea・ns（OpenCV。速いが広い穴はぼやける）",
    ),
    arg(
        "--video-mode",
        choices=["auto", "frame", "plate", "track"],
        default="auto",
        help="動画の埋め方: plate（固定カメラ。他のコマで見えている背景を使う）／ track（動くカメラ。前後のコマを"
        "射影変換で重ね、見えているコマから運ぶ）／ frame（コマごとに埋める。ちらつく）／ auto（カメラの動きで plate か"
        " track を選ぶ。既定）",
    ),
    arg("--window", type=int, default=30, help="track で前後に探すコマ数（既定 30）"),
    arg(
        "--save-mask",
        action="store_true",
        help="使ったマスクも書き出す（<出力>-mask.png / .mp4）",
    ),
)
def inpaint(a: argparse.Namespace) -> None:
    """消したい所（マスク・枠・人物・物）を、周りや他のコマの背景で自然に埋める。"""
    info = core.probe(a.src)
    out = _out(a, info, "inpaint")
    json_path = _json_path(a, out)
    t0 = time.time()
    if info.is_image:
        img = _even(next(_frames(a.src, a)))
        masker = _Masker(a, info, (img.shape[1], img.shape[0]))
        hole = masker(img, 0)
        res = _fill(img, hole, a.method)
        core.save_image(out, res)
        view = img[:, :, :3].copy()
        view[hole > 0] = (
            0.45 * view[hole > 0] + 0.55 * np.array([60, 60, 255])
        ).astype(np.uint8)
        prev_path = out.with_name(out.stem + "-preview.jpg")
        core.save_image(
            prev_path,
            core.side_by_side(view, res, ("消す所（赤）", f"埋めた結果（{a.method}）")),
        )
        if a.save_mask:
            core.save_image(out.with_name(out.stem + "-mask.png"), hole * 255)
        core.write_json(
            json_path,
            {
                "source": str(a.src),
                "method": a.method,
                "hole_percent": round(float(hole.mean() * 100), 3),
                "dilate": masker.dilate,
                "seconds": round(time.time() - t0, 2),
            },
        )
        core.log(f"穴: 画面の {hole.mean() * 100:.2f}%（{a.method}）")
        core.log(f"保存: {prev_path}")
        core.log(f"保存: {json_path}")
        core.log(f"保存: {out}")
        return
    total = _total(info, a)
    first = _even(next(_frames(a.src, a)))
    size = (first.shape[1], first.shape[0])
    mode, motion = _video_mode(
        a.video_mode, a.method, lambda: _probe_frames(a, info, total, size)
    )
    job = _VJob(
        src=a.src,
        fps=info.fps,
        start=a.start or 0.0,
        duration=a.duration,
        total=total,
        out=out,
        frames=lambda: _frames(a.src, a),
        masker=lambda: _Masker(a, info, size),
        method=a.method,
        window=a.window,
        save_mask=a.save_mask,
    )
    stats = _run_video(job, mode)
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "method": a.method,
            "camera_motion_px": motion,
            **stats,
            "seconds": round(time.time() - t0, 2),
        },
    )
    core.log(f"保存: {json_path}")
    core.log(f"保存: {out}")


@dataclass
class _VJob:
    """動画の穴埋め（inpaint）と描き足し（outpaint）で共通の、動画の読み書きと穴の作り方。"""

    src: str
    fps: float
    start: float
    duration: float | None
    total: int
    out: Path
    frames: Callable[
        [], Iterator[np.ndarray]
    ]  # 処理するコマ（描き足しは広げた画）を順に
    masker: Callable[
        [], Any
    ]  # 呼ぶたびに新しい「穴を作る物」（(コマ, 番号) → 穴、time_pad を持つ）
    method: str = "lama"
    window: int = 30
    save_mask: bool = False
    hole_label: str = "消す所"
    limit: int = 1024  # LaMa で計算する長辺の上限

    def writer(self) -> core.VideoWriter:
        return core.VideoWriter(
            self.out,
            self.fps,
            audio_from=self.src,
            audio_start=self.start,
            audio_duration=self.duration,
        )

    def mask_writer(self) -> core.VideoWriter | None:
        return (
            core.VideoWriter(self.out.with_name(self.out.stem + "-mask.mp4"), self.fps)
            if self.save_mask
            else None
        )


class _StaticHole:
    """どのコマも同じ穴（描き足す所）。"""

    time_pad = 0

    def __init__(self, hole: np.ndarray) -> None:
        self.hole = hole

    def __call__(self, frame: np.ndarray, i: int) -> np.ndarray:
        return self.hole


def _hole_view(frame: np.ndarray, hole: np.ndarray) -> np.ndarray:
    v = frame[:, :, :3].copy()
    v[hole > 0] = (0.45 * v[hole > 0] + 0.55 * np.array([60, 60, 255])).astype(np.uint8)
    return v


def _save_sheet(job: _VJob, sheet: list[np.ndarray]) -> None:
    if not sheet:
        return
    labels = []
    for k in range(len(sheet) // 2):
        labels += [f"{job.hole_label} {k + 1}", "結果"]
    p = job.out.with_name(job.out.stem + "-preview.jpg")
    core.save_image(p, core.contact_sheet(sheet, labels, cols=4, cell_width=320))
    core.log(f"保存: {p}")


def _vf_frames(job: _VJob, stats: dict[str, Any]) -> None:
    """コマごとに埋める（ちらつくが、どんな映像でも使える）。"""
    mw = job.mask_writer()
    t0 = time.time()
    areas = []
    sheet: list[np.ndarray] = []
    keep = set(_sample_idx(job.total, 4))
    with job.writer() as wr:
        for i, frame, hole in _mask_stream(job.masker(), job.frames()):
            res = _fill(frame, hole, job.method, limit=job.limit)
            wr.write(res)
            if mw is not None:
                mw.write(hole * 255)
            areas.append(float(hole.mean()))
            if i in keep:
                sheet += [_hole_view(frame, hole), res]
            _progress(i, job.total, t0)
    if mw is not None:
        mw.close()
    stats.update(
        {
            "frames": len(areas),
            "hole_percent_mean": round(100 * float(np.mean(areas)), 3),
        }
    )
    _save_sheet(job, sheet)


def _vf_plate(job: _VJob, stats: dict[str, Any]) -> None:
    """固定カメラ: 穴でない時のコマの中央値で背景（クリーンプレート）を作り、穴をそれで埋める。

    どのコマでも見えない所は、背景の画の上で 1 回だけ LaMa で埋める（時間方向に揃う）。各コマでは明るさ・色の
    ゆっくりした変化を、穴の周りの比（ゲインの地図）で合わせる。
    """
    t0 = time.time()
    masker = job.masker()
    idx = set(_sample_idx(job.total, 80))
    frames, valid = [], []
    for i, frame in enumerate(job.frames()):
        if i in idx:
            frame = _even(frame)[:, :, :3]
            frames.append(frame)
            valid.append(masker(frame, i) == 0)
    core.log(f"背景を作ります（{len(frames)} コマの、穴でない所の中央値）")
    plate, cnt = _masked_median(np.stack(frames), np.stack(valid))
    never = (cnt == 0).astype(np.uint8)
    stats["plate_frames"] = len(frames)
    stats["never_visible_percent"] = round(100 * float(never.mean()), 3)
    if never.any():
        core.log(
            f"どのコマでも隠れていた所（画面の {never.mean() * 100:.2f}%）を {job.method} で埋めます"
        )
        plate = _fill(plate, _dilate(never, 2), job.method, limit=job.limit)
    del frames, valid
    mw = job.mask_writer()
    areas = []
    sheet: list[np.ndarray] = []
    keep = set(_sample_idx(job.total, 4))
    with job.writer() as wr:
        for i, frame, hole in _mask_stream(job.masker(), job.frames()):
            if hole.any():
                gain = _gain_field(frame, plate, _dilate(hole, 6) == 0)
                fill = np.clip(plate.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                res = _paste(frame, fill, hole, 1.5)
            else:
                res = frame
            wr.write(res)
            if mw is not None:
                mw.write(hole * 255)
            areas.append(float(hole.mean()))
            if i in keep:
                sheet += [_hole_view(frame, hole), res]
            _progress(i, job.total, t0)
    if mw is not None:
        mw.close()
    p = job.out.with_name(job.out.stem + "-plate.png")
    core.save_image(p, plate)
    core.log(f"保存: {p}")
    stats.update(
        {
            "frames": len(areas),
            "hole_percent_mean": round(100 * float(np.mean(areas)), 3),
        }
    )
    _save_sheet(job, sheet)
    core.log(
        f"{len(areas)} コマ。穴は平均 {100 * np.mean(areas):.2f}%・どのコマでも見えなかった所 {stats['never_visible_percent']}%"
    )


def _vf_track(job: _VJob, stats: dict[str, Any]) -> None:
    """動くカメラ: 前後のコマを射影変換で今のコマに重ね、穴の画素が見えている「いちばん近い時刻」のコマから埋める。

    射影変換は隣どうしのコマで、穴でない所の特徴点を追って求め、つないで使う（遠い背景・地面などの平面に合う）。
    どのコマでも見えない所は、前のコマの結果を同じ変換で運んで使い（時間方向に揃う）、最初の 1 回は LaMa で埋める。
    （オプティカルフローで穴の中の動きを補ってたどる方式も試したが、正解つきの試験で劣った: NOTES 54）
    """
    t0 = time.time()
    frames, holes = [], []
    for _i, frame, hole in _mask_stream(job.masker(), job.frames()):
        frames.append(frame)
        holes.append(hole)
        core.require(
            len(frames) * frame.size < 2.5e9,
            "track は全コマを覚えるため、長すぎます。--duration か --max-side を小さくしてください",
        )
    n = len(frames)
    h, w = frames[0].shape[:2]
    grays = [core.to_gray(f) for f in frames]
    C = [np.eye(3)]  # C[i]: コマ 0 → コマ i
    for i in range(1, n):
        C.append(
            _homography(grays[i - 1], grays[i], (holes[i - 1] == 0) & (holes[i] == 0))
            @ C[-1]
        )
    core.log(f"{n} コマの動きを求めました（{time.time() - t0:.0f} 秒）")
    mw = job.mask_writer()
    win = max(1, job.window)
    offsets = [d for k in range(1, win + 1) for d in (-k, k)]
    ones = np.ones((h, w), np.uint8)
    # 明るさ合わせ用の 1/4 の画と、見えている所の枠（候補のコマを素早く絞るため）
    sf = 0.25
    S = np.diag([sf, sf, 1.0])
    ssz = (max(8, round(w * sf)), max(8, round(h * sf)))
    smalls = [
        cv2.resize(f, ssz, interpolation=cv2.INTER_AREA).astype(np.float32) + 1.0
        for f in frames
    ]
    known_s = [
        cv2.resize((hm == 0).astype(np.uint8), ssz, interpolation=cv2.INTER_NEAREST)
        for hm in holes
    ]
    kbox = []
    for hm in holes:
        yy, xx = np.nonzero(hm == 0)
        kbox.append((xx.min(), yy.min(), xx.max(), yy.max()) if len(xx) else None)

    def gain(t: int, s: int, Hst: np.ndarray) -> np.ndarray:
        """コマ s を t に重ねた時の、チャンネルごとの明るさの比（重なる所の画素ごとの比の中央値。ずれに強い）。"""
        Hs = S @ Hst @ np.linalg.inv(S)
        ws = cv2.warpPerspective(smalls[s], Hs, ssz, flags=cv2.INTER_LINEAR)
        ok = (cv2.warpPerspective(known_s[s], Hs, ssz, flags=cv2.INTER_NEAREST) > 0) & (
            known_s[t] > 0
        )
        if ok.sum() < 200:
            return np.ones(3, np.float32)
        g = np.median(smalls[t][ok] / ws[ok], axis=0)
        return np.clip(g, 0.8, 1.25).astype(np.float32)

    prev_out = None
    lama_px = reused_px = hole_total = 0
    sheet: list[np.ndarray] = []
    keep = set(_sample_idx(n, 4))
    with job.writer() as wr:
        for t in range(n):
            frame, hole = frames[t], holes[t] > 0
            ys, xs = np.nonzero(hole)
            hole_total += len(ys)
            if len(ys) == 0:
                res = frame
            else:
                fill = frame.astype(np.float32).copy()
                got = np.zeros(len(ys), bool)
                rem = np.arange(len(ys))
                pts = np.stack([xs, ys], 1).astype(np.float32)
                for d in offsets:
                    s = t + d
                    if s < 0 or s >= n or kbox[s] is None:
                        continue
                    Hst = C[t] @ np.linalg.inv(C[s])  # コマ s → コマ t
                    # コマ s の見えている所の枠を t に写し、その中の、まだ埋まっていない穴の画素だけを調べる
                    bx0, by0, bx1, by1 = kbox[s]
                    cor = cv2.perspectiveTransform(
                        np.float32([[[bx0, by0], [bx1, by0], [bx0, by1], [bx1, by1]]]),
                        Hst,
                    )[0]
                    cx0, cy0 = cor.min(0) - 2
                    cx1, cy1 = cor.max(0) + 2
                    cand = rem[
                        (xs[rem] >= cx0)
                        & (xs[rem] <= cx1)
                        & (ys[rem] >= cy0)
                        & (ys[rem] <= cy1)
                    ]
                    if cand.size == 0:
                        continue
                    src = cv2.perspectiveTransform(pts[cand][None], np.linalg.inv(Hst))[
                        0
                    ]
                    sx, sy = src[:, 0], src[:, 1]
                    ok = (sx >= 0) & (sy >= 0) & (sx <= w - 1) & (sy <= h - 1)
                    xi = np.clip(np.round(sx).astype(np.int32), 0, w - 1)
                    yi = np.clip(np.round(sy).astype(np.int32), 0, h - 1)
                    ok &= holes[s][yi, xi] == 0
                    if not ok.any():
                        continue
                    sel = cand[ok]
                    vals = cv2.remap(
                        frames[s], sx[ok][None], sy[ok][None], cv2.INTER_LINEAR
                    )[0].astype(np.float32)
                    fill[ys[sel], xs[sel]] = vals * gain(t, s, Hst)
                    got[sel] = True
                    rem = rem[~got[rem]]
                    if rem.size == 0:
                        break
                need = np.zeros((h, w), bool)
                need[ys[~got], xs[~got]] = True
                if need.any() and prev_out is not None:
                    Hp = C[t] @ np.linalg.inv(C[t - 1])
                    wp = cv2.warpPerspective(
                        prev_out, Hp, (w, h), flags=cv2.INTER_LINEAR
                    )
                    use = need & (cv2.warpPerspective(ones, Hp, (w, h)) > 0)
                    fill[use] = wp[use]
                    reused_px += int(use.sum())
                    need &= ~use
                fill8 = np.clip(fill + 0.5, 0, 255).astype(np.uint8)
                if need.any():
                    lama_px += int(need.sum())
                    nm = need.astype(np.uint8)
                    # カメラが動いて端に出る数 px の細い隙間は、速い Telea で十分（LaMa は太い所だけ）
                    thin = cv2.distanceTransform(nm, cv2.DIST_L2, 3).max() <= 4.0
                    fill8 = _fill(
                        fill8,
                        nm,
                        "telea" if thin and prev_out is not None else job.method,
                        limit=job.limit,
                    )
                res = _paste(frame, fill8, hole, 1.0)
            prev_out = res
            wr.write(res)
            if mw is not None:
                mw.write(holes[t] * 255)
            if t in keep:
                sheet += [_hole_view(frame, holes[t]), res]
            _progress(t, n, t0)
    if mw is not None:
        mw.close()
    hp = max(1, hole_total)
    stats.update(
        {
            "frames": n,
            "hole_percent_mean": round(100 * hole_total / (n * w * h), 3),
            "filled_from_other_frames_percent": round(
                100 * (1 - (lama_px + reused_px) / hp), 2
            ),
            "filled_from_previous_result_percent": round(100 * reused_px / hp, 2),
            "filled_by_ai_percent": round(100 * lama_px / hp, 2),
        }
    )
    _save_sheet(job, sheet)
    core.log(
        f"{n} コマ。穴のうち、他のコマから {stats['filled_from_other_frames_percent']}%・前の結果から "
        f"{stats['filled_from_previous_result_percent']}%・{job.method} で {stats['filled_by_ai_percent']}%"
    )


def _probe_frames(
    a: argparse.Namespace, info: core.MediaInfo, total: int, size: tuple[int, int]
) -> list[np.ndarray]:
    """範囲の中から 5 コマ（カメラの動きを調べる用）。"""
    return [
        _even(core.frame_at(a.src, (a.start or 0.0) + i / info.fps, size=size))
        for i in _sample_idx(total, 5)
    ]


def _video_mode(
    mode: str, method: str, frames: Callable[[], list[np.ndarray]]
) -> tuple[str, float | None]:
    """auto のとき、カメラが動かなければ plate、動けば track（LaMa 以外は frame）。"""
    if mode != "auto":
        return mode, None
    if method != "lama":
        return "frame", None
    motion = _camera_motion(frames())
    mode = "plate" if motion < 3.0 else "track"
    core.log(f"カメラの動き: 最大 {motion:.1f} px → {mode}")
    return mode, motion


def _run_video(job: _VJob, mode: str) -> dict[str, Any]:
    stats: dict[str, Any] = {"mode": mode}
    {"frame": _vf_frames, "plate": _vf_plate, "track": _vf_track}[mode](job, stats)
    return stats


# ── 58 背景ぼかし（奥行きによる被写界深度） ─────────────────────────────────


@cache
def _disk(r: float) -> np.ndarray:
    """半径 r px の円盤のぼけ（合計 1。縁は 1px なめらか）。レンズの丸い絞りのぼけの形。"""
    R = math.ceil(r)
    y, x = np.mgrid[-R : R + 1, -R : R + 1].astype(np.float32)
    k = np.clip(r + 0.5 - np.sqrt(x * x + y * y), 0.0, 1.0)
    return (k / k.sum()).astype(np.float32)


def _blur_disk(img: np.ndarray, r: float) -> np.ndarray:
    """円盤のぼけ。半径 6px 以上は半分の大きさで計算して戻す（速さのため）。"""
    if r < 0.5:
        return img
    if r >= 6.0:
        h, w = img.shape[:2]
        small = cv2.resize(
            img, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA
        )
        b = cv2.filter2D(
            small, -1, _disk(round(r / 2, 1)), borderType=cv2.BORDER_REFLECT
        )
        return cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.filter2D(img, -1, _disk(round(r, 1)), borderType=cv2.BORDER_REFLECT)


def _dof_render(img: np.ndarray, rho: np.ndarray, *, step: float = 1.5, max_layers: int = 18,
                highlights: float = 0.0) -> np.ndarray:  # fmt: skip
    """ぼけの半径の地図 rho（px。+ がピントより手前、- が奥、0 がピント）に従って、層に分けてぼかす。

    rho を step px ごとの層に分け（隣の層と三角形の重みで分け合うので段差が出ない）、奥の層から順に、
    円盤でぼかした色と覆う割合を重ねる（手前のぼけは奥に広がり、奥のぼけはピントの合った物の上に乗らない）。
    最後に覆う割合で割り、ピントの合った物の色が背景ににじまないようにする。光の足し合わせなので sRGB を線形にして計算する。
    """
    lin = _srgb_to_lin(img[:, :, :3])
    h, w = rho.shape
    rmin, rmax = float(rho.min()), float(rho.max())
    step = max(step, (rmax - rmin) / max(1, max_layers))
    C = np.zeros((h, w, 3), np.float32)
    A = np.zeros((h, w), np.float32)
    boost = None
    if highlights > 0:
        lum = lin @ np.array([0.0722, 0.7152, 0.2126], np.float32)
        t = np.clip((lum - 0.7) / 0.3, 0.0, 1.0)
        boost = lin * (1.0 + highlights * 4.0 * (t * t * (3 - 2 * t)))[..., None]
    for k in range(math.floor(rmin / step), math.ceil(rmax / step) + 1):
        c = k * step
        wk = np.clip(1.0 - np.abs(rho - c) / step, 0.0, 1.0)
        if wk.max() < 1e-3:
            continue
        r = abs(c)
        src = boost if (boost is not None and r > 1.5) else lin
        pre = np.dstack([src * wk[..., None], wk]).astype(np.float32)
        b = _blur_disk(pre, r)
        ak = np.clip(b[:, :, 3], 0.0, 1.0)
        C = b[:, :, :3] + C * (1.0 - ak[..., None])
        A = ak + A * (1.0 - ak)
    out = C / np.maximum(A, 1e-4)[..., None]
    return _lin_to_srgb(out)


def _refine_depth(frame: np.ndarray, d01: np.ndarray) -> np.ndarray:
    h, w = d01.shape
    return np.clip(_guided(frame, d01, max(2, round(min(h, w) / 100)), 1e-2), 0.0, 1.0)


def _parse_focus(text: str) -> tuple[str, Any]:
    t = (text or "auto").strip()
    if t == "auto":
        return "auto", None
    if "," in t:
        x, y = (float(v) for v in t.split(","))
        return "point", (x, y)
    try:
        v = float(t)
    except ValueError as e:
        raise core.VfxError(
            f"--focus は auto・x,y・0〜1 の数のどれかです: {text}"
        ) from e
    core.require(0.0 <= v <= 1.0, "--focus の数は 0（遠い）〜1（手前）です")
    return "depth", v


def _focus_depth(
    d01: np.ndarray,
    person: np.ndarray | None,
    mode: str,
    val: Any,
    scale: tuple[float, float],
) -> float:
    h, w = d01.shape
    if mode == "depth":
        return float(val)
    if mode == "point":
        x = int(np.clip(round(val[0] * scale[0]), 0, w - 1))
        y = int(np.clip(round(val[1] * scale[1]), 0, h - 1))
        r = max(2, min(h, w) // 100)
        return float(
            np.median(d01[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1])
        )
    if person is not None and (person > 0.5).mean() > 0.01:
        return float(np.median(d01[person > 0.5]))
    c = d01[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return float(np.percentile(c, 75))


@command(
    58,
    "bg-blur",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--focus",
        default="auto",
        help="ピントを合わせる所: auto（人物。いなければ画面中央の手前寄り。既定）／ x,y（元の大きさの px の点）／ 0〜1（奥行きの値）",
    ),
    arg(
        "--strength",
        type=float,
        help="ぼけの大きさ: 奥行きの差 1 のときの半径（px。既定は短い辺の 3%%）",
    ),
    arg(
        "--near",
        type=float,
        default=1.0,
        help="ピントより手前のぼけの割合（0 で手前はぼかさない。既定 1 = 光学と同じ）",
    ),
    arg(
        "--dof",
        type=float,
        default=0.03,
        help="ピントの前後でくっきり残す奥行きの幅（0〜1。既定 0.03）",
    ),
    arg(
        "--protect",
        choices=["person", "none"],
        default="person",
        help="person: 人物（MediaPipe の切り抜き）は必ずくっきり残す（既定）／ none: 奥行きだけで決める",
    ),
    arg(
        "--highlights",
        type=float,
        default=0.5,
        help="明るい点を丸い玉ボケにする強さ（0〜2。既定 0.5）",
    ),
    arg(
        "--smooth",
        type=float,
        default=0.5,
        help="動画の奥行きの時間方向のなめらかさ（0〜0.95。既定 0.5）",
    ),
    arg(
        "--save-map",
        action="store_true",
        help="ぼけの半径の地図も書き出す（<出力>-map.png / .mp4。青 = 奥・赤 = 手前）",
    ),
)
def bg_blur(a: argparse.Namespace) -> None:
    """奥行き（Depth Anything V2）と人物の切り抜きで、ピントの合った所はくっきり、遠い所ほど強く丸くぼかす。"""
    info = core.probe(a.src)
    out = _out(a, info, "bg-blur")
    json_path = _json_path(a, out)
    t0 = time.time()
    mode, val = _parse_focus(a.focus)
    core.log("Depth Anything V2 Small（Apache-2.0）を読み込みます")
    model = _Depth(518)
    seg = None
    if a.protect == "person":
        from .ai import PersonSegmenter

        seg = PersonSegmenter("selfie", video=not info.is_image)

    def person_map(frame: np.ndarray, i: int) -> np.ndarray | None:
        if seg is None:
            return None
        p = seg(frame, None if info.is_image else round(i * 1000.0 / info.fps))
        h, w = frame.shape[:2]
        if p.shape != (h, w):
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        if (p > 0.5).mean() < 0.01:
            return None
        return np.clip(
            _guided(frame, p, max(2, round(min(h, w) / 135)), 1e-2), 0.0, 1.0
        )

    def render(
        frame: np.ndarray, d01: np.ndarray, pm: np.ndarray | None, fd: float
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = d01.shape
        scale = min(h, w) / min(info.width, info.height)
        strength = a.strength * scale if a.strength else 0.03 * min(h, w)
        delta = d01 - fd
        mag = np.maximum(np.abs(delta) - a.dof, 0.0) / max(1e-3, 1.0 - a.dof)
        rho = np.where(delta >= 0, a.near * strength * mag, -strength * mag).astype(
            np.float32
        )
        if pm is not None:
            rho *= 1.0 - pm
        return _dof_render(frame, rho, highlights=a.highlights), rho

    def map_view(rho: np.ndarray) -> np.ndarray:
        m = max(1.0, float(np.abs(rho).max()))
        v = np.zeros((*rho.shape, 3), np.uint8)
        v[..., 0] = np.clip(-rho / m * 255, 0, 255).astype(np.uint8)
        v[..., 2] = np.clip(rho / m * 255, 0, 255).astype(np.uint8)
        return v

    if info.is_image:
        img = _even(next(_frames(a.src, a)))[:, :, :3]
        d01 = _refine_depth(img, _norm01(model.raw(img)))
        pm = person_map(img, 0)
        sx, sy = img.shape[1] / info.width, img.shape[0] / info.height
        fd = _focus_depth(d01, pm, mode, val, (sx, sy))
        res, rho = render(img, d01, pm, fd)
        core.save_image(out, res)
        prev_path = out.with_name(out.stem + "-preview.jpg")
        core.save_image(
            prev_path,
            core.contact_sheet(
                [img, res, map_view(rho)],
                [
                    "元の画",
                    "背景ぼかし",
                    "ぼけの半径（青 = 奥・赤 = 手前・黒 = くっきり）",
                ],
                cols=3,
                cell_width=min(480, img.shape[1]),
            ),
        )
        if a.save_map:
            core.save_image(out.with_name(out.stem + "-map.png"), map_view(rho))
        info_json = {
            "focus_depth": round(fd, 4),
            "person_protected": pm is not None,
            "max_radius_px": round(float(np.abs(rho).max()), 2),
        }
        core.write_json(
            json_path,
            {"source": str(a.src), **info_json, "seconds": round(time.time() - t0, 2)},
        )
        core.log(
            f"ピントの奥行き {fd:.3f}・最大のぼけの半径 {np.abs(rho).max():.1f} px・人物 {'あり' if pm is not None else 'なし'}"
        )
        core.log(f"保存: {prev_path}")
        core.log(f"保存: {json_path}")
        core.log(f"保存: {out}")
        return
    vid = _DepthVideo(model, a.smooth)
    total = _total(info, a)
    keep = set(_sample_idx(total, 4))
    sheet, labels, per = [], [], []
    fd_s = None
    mapw = (
        core.VideoWriter(out.with_name(out.stem + "-map.mp4"), info.fps)
        if a.save_map
        else None
    )
    with core.VideoWriter(
        out,
        info.fps,
        audio_from=a.src,
        audio_start=a.start or 0.0,
        audio_duration=a.duration,
    ) as wr:
        for i, frame in enumerate(_frames(a.src, a)):
            frame = _even(frame)[:, :, :3]
            d01 = _refine_depth(frame, vid(frame))
            reset = vid.log[-1]["reset"]
            pm = person_map(frame, i)
            sx, sy = frame.shape[1] / info.width, frame.shape[0] / info.height
            fd = _focus_depth(d01, pm, mode, val, (sx, sy))
            fd_s = fd if (fd_s is None or reset) else 0.85 * fd_s + 0.15 * fd
            res, rho = render(frame, d01, pm, fd_s)
            wr.write(res)
            if mapw is not None:
                mapw.write(map_view(rho))
            per.append(
                {
                    "t": core.t_of(i, info.fps, a.start or 0.0),
                    "focus": round(fd_s, 4),
                    "person": pm is not None,
                    "scene_start": reset,
                }
            )
            if i in keep:
                sheet += [frame, res]
                labels += [f"{per[-1]['t']:.2f} 秒", "ぼかした結果"]
            _progress(i, total, t0)
    if mapw is not None:
        mapw.close()
    prev_path = out.with_name(out.stem + "-preview.jpg")
    core.save_image(
        prev_path, core.contact_sheet(sheet, labels, cols=4, cell_width=300)
    )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "frames": len(per),
            "per_frame": per,
            "seconds": round(time.time() - t0, 2),
        },
    )
    core.log(f"{len(per)} コマ。人物を守ったコマ {sum(p['person'] for p in per)}")
    core.log(f"保存: {prev_path}")
    core.log(f"保存: {json_path}")
    core.log(f"保存: {out}")


# ── 59 アウトペインティング（画面の外側を描き足す） ───────────────────────────


def _parse_ratio(text: str) -> float:
    try:
        x, y = (float(v) for v in text.replace("：", ":").replace("/", ":").split(":"))
    except ValueError as e:
        raise core.VfxError(
            f"縦横比は 幅:高さ で指定してください（例 1:1、16:9）: {text}"
        ) from e
    core.require(x > 0 and y > 0, f"縦横比が正しくありません: {text}")
    return x / y


def _canvas_plan(
    w: int, h: int, a: argparse.Namespace, scale: float
) -> tuple[int, int, int, int, int, int]:
    """(画の幅, 高さ, 広げた後の幅, 高さ, 画を置く x, y)。画は切らずに、足りない側を描き足す。"""
    if a.pad:
        parts = [round(float(v) * scale) for v in a.pad.replace(" ", "").split(",")]
        core.require(
            len(parts) == 4 and min(parts) >= 0,
            "--pad は 上,右,下,左 の 4 つの px で指定してください",
        )
        t, r, b, l = parts
        W, H = w + l + r, h + t + b
        return w, h, W + W % 2, H + H % 2, l, t
    if a.size:
        try:
            tw, th = (int(v) for v in a.size.lower().replace("×", "x").split("x"))
        except ValueError as e:
            raise core.VfxError(
                f"--size は 幅x高さ で指定してください（例 1920x1080）: {a.size}"
            ) from e
        tw, th = round(tw * scale), round(th * scale)
        f = min(tw / w, th / h)
        w, h = max(2, round(w * f / 2) * 2), max(2, round(h * f / 2) * 2)
        W, H = tw + tw % 2, th + th % 2
    else:
        core.require(
            bool(a.aspect),
            "--aspect・--size・--pad のどれかで、仕上がりの形を指定してください",
        )
        R = _parse_ratio(a.aspect)
        if w / h < R:
            W, H = round(h * R), h
        else:
            W, H = w, round(w / R)
        W, H = W + W % 2, H + H % 2
    al = a.align
    x0 = {"left": 0, "right": W - w}.get(al, (W - w) // 2)
    y0 = {"top": 0, "bottom": H - h}.get(al, (H - h) // 2)
    return w, h, W, H, x0, y0


def _place(
    frame: np.ndarray, w: int, h: int, W: int, H: int, x0: int, y0: int
) -> np.ndarray:
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(
            frame,
            (w, h),
            interpolation=cv2.INTER_AREA if frame.shape[1] > w else cv2.INTER_CUBIC,
        )
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[y0 : y0 + h, x0 : x0 + w] = frame[:, :, :3]
    return canvas


def _outpaint_blur(canvas: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """ぼかして暗くした拡大で埋める（縦長の動画を横長に置く時によくある形）。"""
    x0, y0, w, h = box
    H, W = canvas.shape[:2]
    src = canvas[y0 : y0 + h, x0 : x0 + w]
    s = max(W / w, H / h)
    bw, bh = math.ceil(w * s), math.ceil(h * s)
    small_f = 0.25
    sm = cv2.resize(
        src,
        (max(2, round(bw * small_f)), max(2, round(bh * small_f))),
        interpolation=cv2.INTER_AREA,
    )
    sm = cv2.GaussianBlur(sm, (0, 0), max(1.0, 0.03 * max(W, H) * small_f))
    big = cv2.resize(sm, (bw, bh), interpolation=cv2.INTER_LINEAR)
    oy, ox = (bh - H) // 2, (bw - W) // 2
    out = (big[oy : oy + H, ox : ox + W].astype(np.float32) * 0.8).astype(np.uint8)
    out[y0 : y0 + h, x0 : x0 + w] = src
    return out


def _outpaint_mirror(canvas: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """折り返して写し、元の画から離れるほどぼかす。"""
    x0, y0, w, h = box
    H, W = canvas.shape[:2]
    src = canvas[y0 : y0 + h, x0 : x0 + w]
    pad = cv2.copyMakeBorder(
        src, y0, H - y0 - h, x0, W - x0 - w, cv2.BORDER_REFLECT_101
    )
    hole = np.ones((H, W), np.uint8)
    hole[y0 : y0 + h, x0 : x0 + w] = 0
    dist = cv2.distanceTransform(hole, cv2.DIST_L2, 5)
    band = max(8.0, 0.25 * float(dist.max()))
    k = np.clip(dist / band, 0.0, 1.0)[..., None]
    blurred = cv2.GaussianBlur(pad, (0, 0), max(1.0, 0.02 * max(W, H)))
    out = (pad.astype(np.float32) * (1 - k) + blurred.astype(np.float32) * k).astype(
        np.uint8
    )
    out[y0 : y0 + h, x0 : x0 + w] = src
    return out


def _outpaint_lama(
    canvas: np.ndarray, box: tuple[int, int, int, int], steps: int, limit: int
) -> np.ndarray:
    """LaMa で描き足す。steps > 1 なら、外へ少しずつ（毎回、前に描いた所も手がかりにして）広げる。"""
    x0, y0, w, h = box
    H, W = canvas.shape[:2]
    out = canvas.copy()
    known = np.zeros((H, W), bool)
    known[y0 : y0 + h, x0 : x0 + w] = True
    steps = max(1, steps)
    for k in range(1, steps + 1):
        f = k / steps
        cx0 = round(x0 - f * x0)
        cy0 = round(y0 - f * y0)
        cx1 = round(x0 + w + f * (W - x0 - w))
        cy1 = round(y0 + h + f * (H - y0 - h))
        crop = out[cy0:cy1, cx0:cx1]
        hole = (~known[cy0:cy1, cx0:cx1]).astype(np.uint8)
        if hole.any():
            res = _fill(crop, hole, "lama", limit=limit)
            out[cy0:cy1, cx0:cx1] = res
            known[cy0:cy1, cx0:cx1] = True
    return out


@command(
    59,
    "outpaint",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--aspect",
        help="仕上がりの縦横比（例 1:1、16:9、4:5）。元の画は切らずに、足りない側を描き足す",
    ),
    arg(
        "--size",
        help="仕上がりの大きさ 幅x高さ（px。元の画をこの中に収まる大きさにしてから描き足す）",
    ),
    arg("--pad", help="上,右,下,左 に描き足す px（元の大きさの px）"),
    arg(
        "--align",
        choices=["center", "left", "right", "top", "bottom"],
        default="center",
        help="元の画を置く位置（既定 中央）",
    ),
    arg(
        "--method",
        choices=["lama", "blur", "mirror"],
        default="lama",
        help="lama（AI で続きを描く。既定）／ blur（ぼかした拡大で埋める。速く安定）／ mirror（折り返して、離れるほどぼかす）",
    ),
    arg(
        "--steps", type=int, default=1, help="lama で何回に分けて外へ広げるか（既定 1）"
    ),
    arg(
        "--limit",
        type=int,
        default=512,
        help="lama で計算する長辺の上限（px。既定 512。広い所を描き足す時は 1024 より 512 の方が続きの模様が自然だった）",
    ),
    arg(
        "--video-mode",
        choices=["auto", "frame", "plate", "track"],
        default="auto",
        help="動画で lama の時: plate（固定カメラ。描き足した所は動かない）／ track（動くカメラ。他のコマで写っている外側を"
        "運んで使い、無い所は前の結果を運ぶ）／ frame（コマごと。ちらつく）／ auto（カメラの動きで選ぶ。既定）",
    ),
    arg("--window", type=int, default=30, help="track で前後に探すコマ数（既定 30）"),
)
def outpaint(a: argparse.Namespace) -> None:
    """画面の外側を描き足して、縦横比や大きさを変える（例: 9:16 の映像を 1:1・16:9 に）。有料の生成 API は使わない。"""
    info = core.probe(a.src)
    out = _out(a, info, "outpaint")
    json_path = _json_path(a, out)
    t0 = time.time()
    first = _even(next(_frames(a.src, a)))[:, :, :3]
    scale = first.shape[1] / info.width
    w, h, W, H, x0, y0 = _canvas_plan(first.shape[1], first.shape[0], a, scale)
    core.require(
        W >= w and H >= h,
        "仕上がりが元の画より小さくなります（描き足す所がありません）",
    )
    hole = np.ones((H, W), np.uint8)
    hole[y0 : y0 + h, x0 : x0 + w] = 0
    box = (x0, y0, w, h)
    core.log(
        f"{first.shape[1]}×{first.shape[0]} → {W}×{H}（元の画は {w}×{h}・位置 {x0},{y0}。描き足す所 {hole.mean() * 100:.0f}%）"
    )

    def one(canvas: np.ndarray) -> np.ndarray:
        if a.method == "blur":
            return _outpaint_blur(canvas, box)
        if a.method == "mirror":
            return _outpaint_mirror(canvas, box)
        return _outpaint_lama(canvas, box, a.steps, a.limit)

    def preview(src: np.ndarray, res: np.ndarray) -> np.ndarray:
        v = res.copy()
        cv2.rectangle(
            v,
            (x0, y0),
            (x0 + w - 1, y0 + h - 1),
            (255, 255, 255),
            max(1, W // 400),
            cv2.LINE_AA,
        )
        return v

    meta = {"source": str(a.src), "method": a.method, "canvas": [W, H], "placed": [x0, y0, w, h],
            "new_area_percent": round(float(hole.mean() * 100), 2)}  # fmt: skip
    if info.is_image:
        canvas = _place(first, w, h, W, H, x0, y0)
        res = one(canvas)
        core.save_image(out, res)
        p = out.with_name(out.stem + "-preview.jpg")
        core.save_image(p, core.contact_sheet([first, preview(first, res)], ["元の画", f"描き足した結果（{a.method}。白枠が元）"],
                                              cols=2, cell_width=min(640, W)))  # fmt: skip
        core.write_json(json_path, {**meta, "seconds": round(time.time() - t0, 2)})
        core.log(f"保存: {p}")
        core.log(f"保存: {json_path}")
        core.log(f"保存: {out}")
        return
    total = _total(info, a)
    if a.method in ("blur", "mirror"):
        keep = set(_sample_idx(total, 4))
        sheet: list[np.ndarray] = []
        with core.VideoWriter(
            out,
            info.fps,
            audio_from=a.src,
            audio_start=a.start or 0.0,
            audio_duration=a.duration,
        ) as wr:
            for i, frame in enumerate(_frames(a.src, a)):
                res = one(_place(_even(frame), w, h, W, H, x0, y0))
                wr.write(res)
                if i in keep:
                    sheet += [frame, preview(frame, res)]
                _progress(i, total, t0)
        p = out.with_name(out.stem + "-preview.jpg")
        core.save_image(
            p,
            core.contact_sheet(
                sheet, ["元", "結果"] * (len(sheet) // 2), cols=4, cell_width=320
            ),
        )
        core.write_json(
            json_path, {**meta, "frames": total, "seconds": round(time.time() - t0, 2)}
        )
        core.log(f"保存: {p}")
        core.log(f"保存: {json_path}")
        core.log(f"保存: {out}")
        return
    size = (first.shape[1], first.shape[0])
    mode, motion = _video_mode(
        a.video_mode, "lama", lambda: _probe_frames(a, info, total, size)
    )
    job = _VJob(
        src=a.src,
        fps=info.fps,
        start=a.start or 0.0,
        duration=a.duration,
        total=total,
        out=out,
        frames=lambda: (
            _place(_even(f), w, h, W, H, x0, y0) for f in _frames(a.src, a)
        ),
        masker=lambda: _StaticHole(hole),
        method="lama",
        window=a.window,
        hole_label="描き足す所",
        limit=a.limit,
    )
    stats = _run_video(job, mode)
    core.write_json(
        json_path,
        {
            **meta,
            "camera_motion_px": motion,
            **stats,
            "seconds": round(time.time() - t0, 2),
        },
    )
    core.log(f"保存: {json_path}")
    core.log(f"保存: {out}")


# ── 68 2.5D パララックス（写真 1 枚から奥行きのある動き） ─────────────────────


def _ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


def _nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """valid でない所を、いちばん近い valid の画素の値で埋める（なめらかにして返す）。"""
    if valid.all():
        return values.astype(np.float32)
    if not valid.any():
        return np.full(values.shape, float(values.mean()), np.float32)
    inv = (~valid).astype(np.uint8)
    _d, lab = cv2.distanceTransformWithLabels(
        inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    ys, xs = np.nonzero(valid)
    lut = np.zeros(lab.max() + 1, np.float32)
    # ラベルは 0 の画素（valid）ごとに 1 から振られる（行の順）
    lut[lab[ys, xs]] = values[ys, xs]
    out = lut[lab]
    out[valid] = values[valid]
    return cv2.GaussianBlur(out, (0, 0), 2.0)


class _Parallax:
    """写真を奥行きで層に分け、層ごとに隠れた所を埋めてから、カメラの動きで描き直す。

    画素 p（中心 c からの位置）、逆深度 q（手前ほど大きい）、カメラの横・縦の移動 S（px。q=1 の時のずれ）と前後の移動 tz、
    画角の倍率 zoom のとき、写る位置は p' = c + zoom·((p − c) − S·q) / (1 − tz·q)（ピンホールカメラの式）。
    描く時は、出力の画素ごとに元の位置を逆に求める（層の中の逆深度はなめらかなので、数回の反復で収まる）。
    """

    def __init__(self, img: np.ndarray, q: np.ndarray, layers: int, fill: str, limit: int, band: float,
                 person: np.ndarray | None = None) -> None:  # fmt: skip
        h, w = q.shape
        if person is not None:
            # 人物の髪の縁は奥行きが中間になり、層からはがれやすい。人物の中は奥行きの下限を上げて、顔と一緒に動かす
            pm = person > 0.5
            q = q.copy()
            q[pm] = np.maximum(q[pm], float(np.percentile(q[pm], 30)))
        self.size = (w, h)
        self.c = ((w - 1) / 2.0, (h - 1) / 2.0)
        self.layers: list[
            tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = []  # (色 BGR float, 覆う割合, 逆深度)
        qn = (q - q.min()) / max(1e-6, float(q.max() - q.min()))
        th: list[float] = []
        if layers > 1:
            from skimage.filters import threshold_multiotsu

            try:
                th = [
                    float(v) for v in threshold_multiotsu(qn, classes=layers, nbins=128)
                ]
            except ValueError:
                th = []
        self.thresholds = th
        lab = np.digitize(qn, th) if th else np.zeros_like(qn, dtype=np.int64)
        # 奥行きの縁はなだらかなので、縁の画素が中間の層の細い帯になり、別の速さで動いてはがれる。
        # 層の番号をメディアンで整えて細い帯を消す（周りの多い方の層に入れる）
        ks = max(5, round(min(h, w) / 100) | 1)
        lab = cv2.medianBlur(lab.astype(np.uint8), ks).astype(np.int64)
        if person is not None and layers > 1:
            lab[person > 0.5] = (
                int(lab.max()) if lab.max() > 0 else 1
            )  # 人物はまとめていちばん手前の層に
        n = int(lab.max()) + 1
        self.coverage = [round(float((lab == k).mean()), 4) for k in range(n)]
        bandpx = max(4, round(band))
        # 奥行きの縁は物の輪郭より少し内側に出やすい（耳の先・髪がはみ出して奥の層に入る）。手前の層は、元の画の色の
        # 縁に沿う柔らかい切り抜き（ガイデッドフィルター）で r px まで外へ広げ、奥の層はその分も含めて隠れた所を埋める。
        r = max(3, round(0.012 * min(h, w)))
        soft: list[np.ndarray] = []
        for k in range(n):
            own = lab == k
            if k == 0:
                soft.append(np.ones((h, w), np.float32))
                continue
            g = np.clip(_guided(img, own.astype(np.float32), r, 1e-2), 0.0, 1.0)
            if person is not None and k == n - 1:
                g = np.maximum(
                    g, np.clip(person, 0.0, 1.0)
                )  # 人物の切り抜きの柔らかい縁（髪の毛先）も使う
            g[~(_dilate(own, r) > 0)] = 0.0
            g[own] = np.maximum(g[own], 0.999)
            g[lab > k] = 0.0  # さらに手前の層の所には広げない
            soft.append(g)
        for k in range(n):
            own = lab == k
            nearer_soft = np.zeros((h, w), np.float32)
            for j in range(k + 1, n):
                nearer_soft = np.maximum(nearer_soft, soft[j])
            nearer = nearer_soft > 0.02
            if k == 0:
                ext = nearer  # いちばん奥は、手前の層の後ろ全部を埋めて、すき間の無い背景にする
            else:
                ext = nearer & (_dilate(own, bandpx) > 0)
            col = img[:, :, :3].copy()
            if ext.any():
                # 隠れていた所と、その縁の 2px（手前の物の色が混じった画素）を埋める
                col = _fill(col, _dilate(ext, 2), fill, limit=limit)
            mine = soft[k] > 0.02
            # 層の縁の画素は、少し内側（芯）の奥行きで動かす（縁だけ遅れてはがれないように）
            core_k = cv2.erode(own.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if core_k.sum() < 50:
                core_k = own
            qk = np.where(core_k, q, _nearest_fill(q, core_k)).astype(np.float32)
            if k == 0:
                alpha = np.ones((h, w), np.float32)
            else:
                alpha = np.maximum(
                    soft[k], _feather(ext.astype(np.float32), 0.8) * (~mine)
                )
                alpha[own] = 1.0
            self.layers.append((col.astype(np.float32), alpha.astype(np.float32), qk))

    def render(self, S: tuple[float, float], tz: float, zoom: float) -> np.ndarray:
        w, h = self.size
        cx, cy = self.c
        X, Y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        out = np.zeros((h, w, 3), np.float32)
        for li, (col, alpha, q) in enumerate(self.layers):
            # 出力の画素ごとに、元の位置 p を p = c + (p' − c)(1 − tz·q(p))/zoom + S·q(p) の反復で求める。
            # 層の中にも奥行きの段差（物とその後ろの地面）があり、反復が段差の両側を行き来して物が二重に写るので、
            # 手前寄り・奥寄りの 2 通りから始め、写し直して元の位置に戻る（0.7px 以内）解のうち手前の方を使う。
            # どちらも戻らない所は、この層では見えない所（後ろの層が見える）とする。
            best_x = best_y = best_q = None
            ok_any = np.zeros((h, w), bool)
            for q0 in (float(q.max()), float(q.min())):
                px = cx + (X - cx) * (1.0 - tz * q0) / zoom + S[0] * q0
                py = cy + (Y - cy) * (1.0 - tz * q0) / zoom + S[1] * q0
                for _ in range(6):
                    qq = cv2.remap(
                        q, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                    )
                    k = (1.0 - tz * qq) / zoom
                    px = cx + (X - cx) * k + S[0] * qq
                    py = cy + (Y - cy) * k + S[1] * qq
                qq = cv2.remap(
                    q, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                den = np.maximum(1.0 - tz * qq, 1e-3)
                fx = cx + zoom * ((px - cx) - S[0] * qq) / den
                fy = cy + zoom * ((py - cy) - S[1] * qq) / den
                ok = np.hypot(fx - X, fy - Y) < 0.7
                if best_x is None:
                    best_x, best_y, best_q = px, py, np.where(ok, qq, -1.0)
                else:
                    take = ok & (qq > best_q)
                    best_x = np.where(take, px, best_x)
                    best_y = np.where(take, py, best_y)
                    best_q = np.where(take, qq, best_q)
                ok_any |= ok
            c = cv2.remap(
                col, best_x, best_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            a = cv2.remap(
                alpha, best_x, best_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            if li > 0:
                a = a * ok_any
            a = a[..., None]
            out = c * a + out * (1.0 - a)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _person_matte(img: np.ndarray) -> np.ndarray | None:
    """人物の柔らかい切り抜き（0〜1）。人物の検出（EfficientDet-Lite2、確からしさ 0.5 以上）の枠の中だけ使う。

    MediaPipe の人物の切り抜きだけでは、人のいない写真（黄色い建物など）でも人物と出ることがあったため。
    """
    from .ai import PersonSegmenter
    from .detectors import ObjectDetector

    h, w = img.shape[:2]
    boxes = [d.box for d in ObjectDetector("fast", 0.5, ["person"])(img)]
    if not boxes:
        return None
    pmap = PersonSegmenter("selfie")(img)
    if pmap.shape != (h, w):
        pmap = cv2.resize(pmap, (w, h), interpolation=cv2.INTER_LINEAR)
    inside = np.zeros((h, w), np.float32)
    for x1, y1, x2, y2 in boxes:
        mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        inside[
            max(0, round(y1 - my)) : round(y2 + my),
            max(0, round(x1 - mx)) : round(x2 + mx),
        ] = 1.0
    pmap = pmap * inside
    if (pmap > 0.5).mean() < 0.02:
        return None
    return np.clip(_guided(img, pmap, max(2, round(min(h, w) / 135)), 1e-2), 0.0, 1.0)


def _move_params(
    move: str, amount: float, t: float, w: int, h: int, q_subject: float
) -> tuple[tuple[float, float], float, float]:
    """動き方と進み（0〜1）から、(S, tz, zoom) を求める。S・tz は手前（q=1）の物を基準にした量。"""
    e = _ease(t)
    A = amount
    if move == "truck":
        return ((e - 0.5) * 0.07 * w * A, 0.0), 0.0, 1.0
    if move == "pedestal":
        return (0.0, (e - 0.5) * 0.07 * h * A), 0.0, 1.0
    if move == "circle":
        th = 2 * math.pi * t
        return (0.03 * w * A * math.cos(th), 0.03 * w * A * math.sin(th)), 0.0, 1.0
    if move == "dolly":
        return (0.0, 0.0), 0.16 * A * e, 1.0
    if move == "dolly-out":
        return (0.0, 0.0), 0.16 * A * (1.0 - e), 1.0
    if move == "dolly-zoom":
        tz = (
            -0.6 * A * e
        )  # 後ろへ下がりながら、被写体の大きさが変わらないように画角を狭める
        return (0.0, 0.0), tz, 1.0 - tz * q_subject
    raise core.VfxError(f"知らない動きです: {move}")


@command(
    68,
    "parallax",
    SRC,
    OUT,
    JSON,
    SIZE,
    arg(
        "--move",
        choices=["dolly", "dolly-out", "truck", "pedestal", "circle", "dolly-zoom"],
        default="dolly",
        help="カメラの動き: dolly（前へ寄る。既定）／ dolly-out（下がる）／ truck（横に動く）／ pedestal（縦に動く）"
        "／ circle（小さく円を描く）／ dolly-zoom（下がりながら画角を狭める。めまいのような効果）",
    ),
    arg(
        "--amount",
        type=float,
        default=1.0,
        help="動きの大きさ（1 が標準。0.5 で控えめ、2 で大きく）",
    ),
    arg("--seconds", type=float, default=4.0, help="動画の長さ（秒。既定 4）"),
    arg("--fps", type=float, default=30.0, help="コマの速さ（既定 30）"),
    arg("--layers", type=int, default=3, help="奥行きで分ける層の数（1〜5。既定 3）"),
    arg(
        "--depth-range",
        type=float,
        default=4.0,
        help="いちばん奥といちばん手前の距離の比（既定 4。大きいほど奥が動かない）",
    ),
    arg(
        "--fill",
        choices=["lama", "telea"],
        default="lama",
        help="層の後ろに隠れていた所の埋め方（既定 lama）",
    ),
    arg(
        "--limit",
        type=int,
        default=768,
        help="lama で計算する長辺の上限（px。既定 768）",
    ),
    arg(
        "--person",
        choices=["auto", "off"],
        default="auto",
        help="auto: 人物が写っていれば（人物の検出と MediaPipe の切り抜き）、髪の毛先まで含めてまとめて手前の層にする（既定）"
        "／ off: 奥行きだけで分ける",
    ),
)
def parallax(a: argparse.Namespace) -> None:
    """写真 1 枚を奥行きで層に分け、隠れていた所を埋めて、奥行きのあるカメラの動き（2.5D）の動画にする。"""
    info = core.probe(a.src)
    core.require(
        info.is_image,
        "写真（画像 1 枚）を指定してください（動画は 1 コマを書き出してから使ってください）",
    )
    core.require(1 <= a.layers <= 5, "--layers は 1〜5 です")
    core.require(a.depth_range >= 1.0, "--depth-range は 1 以上です")
    out = (
        Path(a.out) if a.out else core.default_out(a.src, f"parallax-{a.move}", ".mp4")
    )
    json_path = _json_path(a, out)
    t0 = time.time()
    img = _even(core.load_image(a.src))
    if a.max_side:
        img = _even(
            cv2.resize(
                img,
                core.scaled_size(img.shape[1], img.shape[0], max_side=a.max_side),
                interpolation=cv2.INTER_AREA,
            )
        )
    h, w = img.shape[:2]
    core.log("Depth Anything V2 Small（Apache-2.0）で奥行きを求めます")
    model = _Depth(518)
    d01 = _refine_depth(img, _norm01(model.raw(img)))
    q_far = 1.0 / a.depth_range
    q = (q_far + d01 * (1.0 - q_far)).astype(np.float32)
    person = None
    if a.person == "auto":
        person = _person_matte(img)
        if person is not None:
            core.log(
                f"人物を見つけました（画面の {(person > 0.5).mean() * 100:.0f}%）。人物はまとめて手前の層にします"
            )
    # 被写体（dolly-zoom で大きさを保つ所）: 画面中央の手前寄り
    q_subject = float(np.percentile(q[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4], 75))
    n = max(2, round(a.seconds * a.fps))
    params = [
        _move_params(a.move, a.amount, i / (n - 1), w, h, q_subject) for i in range(n)
    ]
    # 層の後ろを埋める幅 = 層どうしのずれの最大
    max_shift = 0.0
    for S, tz, zoom in params:
        for qq in (q_far, 1.0):
            sc = zoom / max(1e-3, 1.0 - tz * qq)
            max_shift = max(max_shift, abs(S[0]) * qq * sc, abs(S[1]) * qq * sc)
        if tz:
            max_shift = max(
                max_shift,
                0.5 * max(w, h) * abs(1.0 / (1 - tz) - 1.0 / (1 - tz * q_far)),
            )
    core.log(
        f"{a.layers} 層に分け、隠れていた所を {a.fill} で埋めます（ずれの最大 {max_shift:.0f} px）"
    )
    par = _Parallax(img, q, a.layers, a.fill, a.limit, max_shift * 1.2 + 6, person)
    # 画面の外が見えないように、少し拡大する（全コマで、端の画素がどれだけ内側へ動くか）
    edge = np.zeros((h, w), bool)
    edge[:2], edge[-2:], edge[:, :2], edge[:, -2:] = True, True, True, True
    ys, xs = np.nonzero(edge)
    qe = q[ys, xs]
    need = 1.0
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    for S, tz, zoom in params:
        px = cx + zoom * ((xs - cx) - S[0] * qe) / (1 - tz * qe)
        py = cy + zoom * ((ys - cy) - S[1] * qe) / (1 - tz * qe)
        # 端が内側に入り込んだ量（左右・上下それぞれ）→ その分を拡大で隠す
        gap = max(float(np.max(np.where(xs < 2, px, -1e9))) + 1, float(w - 1 - np.min(np.where(xs > w - 3, px, 1e9))) + 1,
                  float(np.max(np.where(ys < 2, py, -1e9))) + 1, float(h - 1 - np.min(np.where(ys > h - 3, py, 1e9))) + 1, 0.0)  # fmt: skip
        need = max(need, 1.0 + 2.0 * gap / min(w, h))
    keep = set(_sample_idx(n, 3))
    sheet, labels = [], []
    with core.VideoWriter(out, a.fps) as wr:
        for i, (S, tz, zoom) in enumerate(params):
            fr = par.render(S, tz, zoom * need)
            wr.write(fr)
            if i in keep:
                sheet.append(fr)
                labels.append(f"{i / a.fps:.1f} 秒")
            _progress(i, n, t0)
    lay = np.zeros((h, w), np.float32)
    for k, th in enumerate(par.thresholds):
        lay[((q - q.min()) / max(1e-6, float(q.max() - q.min()))) >= th] = k + 1
    sheet += [
        img,
        core.heatmap(d01, lo=0, hi=1),
        core.heatmap(lay, lo=0, hi=max(1, len(par.thresholds))),
    ]
    labels += ["元の写真", "奥行き（赤が手前）", f"層（{len(par.layers)} 層）"]
    p = out.with_name(out.stem + "-preview.jpg")
    core.save_image(
        p, core.contact_sheet(sheet, labels, cols=3, cell_width=min(420, w))
    )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "move": a.move,
            "amount": a.amount,
            "frames": n,
            "fps": a.fps,
            "layers": len(par.layers),
            "layer_coverage": par.coverage,
            "person_layer": person is not None,
            "zoom_to_hide_edges": round(need, 4),
            "max_layer_shift_px": round(max_shift, 1),
            "seconds": round(time.time() - t0, 2),
        },
    )
    core.log(
        f"{n} コマ（{a.seconds} 秒）。層 {len(par.layers)}（面積 {par.coverage}）・端を隠す拡大 {need:.3f} 倍"
    )
    core.log(f"保存: {p}")
    core.log(f"保存: {json_path}")
    core.log(f"保存: {out}")


# ── 69 空の置き換え ───────────────────────────────────────────────────────

_ADE_SKY = 2  # ADE20K の「sky」


class _SkySeg:
    """UperNet ConvNeXt small（openmmlab/upernet-convnext-small、MIT。ADE20K 150 種類）の「空」の確率（入力の大きさ）。"""

    def __init__(self, short: int = 512) -> None:
        self.torch = models.torch_setup()
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

        repo = models.hf_repo("ade")
        proc = AutoImageProcessor.from_pretrained(repo)
        self.mean = np.array(proc.image_mean, np.float32)
        self.std = np.array(proc.image_std, np.float32)
        self.net = UperNetForSemanticSegmentation.from_pretrained(repo).eval()
        self.short = short

    def __call__(self, frame: np.ndarray) -> np.ndarray:
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
        with torch.inference_mode():
            feats = self.net.backbone.forward_with_filtered_kwargs(t).feature_maps
            logits = self.net.decode_head(feats)
        lh, lw = logits.shape[-2:]
        logits = logits[:, :, : round(lh * nh / ph), : round(lw * nw / pw)]
        prob = torch.softmax(logits, dim=1)[0, _ADE_SKY].numpy()
        return cv2.resize(
            prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )


def _sky_alpha(
    frame: np.ndarray, prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """空の確率を、元の画の縁に合う柔らかい切り抜き（0〜1）にする。近くの空の色・前景の色の地図も返す。

    確率は 1/4 の大きさで求めた物を広げるので縁がぼやける。0.5 を中心に 6 倍に急にしてから、ガイデッドフィルター
    （半径 = 短い辺の 1/70）で元の画の輪郭に沿わせる。正解つきの 2 つの試験で、そのままの確率・ガイデッドだけ・
    色の分離（境目で空と前景の色の間に置く）より誤差が小さかった（NOTES 69）。
    """
    h, w = prob.shape
    r = max(2, round(min(h, w) / 70))
    steep = np.clip((prob - 0.5) * 6.0 + 0.5, 0.0, 1.0)
    a = np.clip(_guided(frame, steep, r, 1e-2), 0.0, 1.0).astype(np.float32)
    img = frame[:, :, :3].astype(np.float32)
    sig = max(2.0, min(h, w) / 40)

    def local_mean(m: np.ndarray) -> np.ndarray:
        wsum = cv2.GaussianBlur(m, (0, 0), sig) + 1e-4
        return np.dstack(
            [cv2.GaussianBlur(img[:, :, c] * m, (0, 0), sig) / wsum for c in range(3)]
        )

    B = local_mean((a > 0.97).astype(np.float32))
    F = local_mean((a < 0.03).astype(np.float32))
    return a, B, F


def _make_sky(w: int, h: int, preset: str, horizon: float) -> np.ndarray:
    """空の絵を作る（BGR）。horizon は空の下端（0〜1、画面の高さに対する位置）。"""
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None] / max(0.15, horizon)
    y = np.clip(y, 0.0, 1.2)
    stops = {
        "sunset": [
            (0.0, (70, 40, 40)),
            (0.45, (110, 80, 170)),
            (0.75, (80, 140, 245)),
            (1.0, (120, 200, 255)),
        ],
        "dusk": [
            (0.0, (60, 25, 20)),
            (0.5, (120, 60, 70)),
            (0.85, (130, 110, 210)),
            (1.0, (150, 170, 240)),
        ],
        "blue": [(0.0, (190, 120, 50)), (0.6, (230, 175, 110)), (1.0, (245, 225, 200))],
        "night": [(0.0, (30, 10, 5)), (0.7, (70, 30, 20)), (1.0, (110, 60, 50))],
    }[preset]
    sky = np.zeros((h, w, 3), np.float32)
    ys = np.array([s[0] for s in stops], np.float32)
    for c in range(3):
        cs = np.array([s[1][c] for s in stops], np.float32)
        sky[:, :, c] = np.interp(y, ys, cs)
    rng = np.random.default_rng(7)
    if preset in ("sunset", "dusk", "blue"):
        # 雲: 大きさの違う雑音を重ね、横に引き伸ばす。地平線に近いほど薄く細く
        n = np.zeros((h, w), np.float32)
        for k, amp in ((6, 0.55), (14, 0.3), (32, 0.15)):
            g = rng.random((max(3, k * 2), max(3, k // 2))).astype(
                np.float32
            )  # 行が多く列が少ない = 横に長い形
            n += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        n = cv2.GaussianBlur(n, (0, 0), max(1.0, w / 300))
        fade = np.clip(1.0 - y * 0.9, 0.15, 1.0)
        n = np.clip((n - 0.62) * 4.0, 0.0, 1.0) * fade
        n = n[..., None]
        cloud = np.array(
            {
                "sunset": (140, 160, 250),
                "dusk": (160, 130, 210),
                "blue": (250, 250, 250),
            }[preset],
            np.float32,
        )
        sky = sky * (1 - 0.5 * n) + cloud * 0.5 * n
    if preset in ("sunset", "dusk"):
        # 地平線近くの太陽の光
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = 0.62 * w, horizon * h * 0.92
        d = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        glow = np.exp(-((d / 0.18) ** 2))[..., None]
        sky = (
            sky * (1 - 0.6 * glow) + np.array((170, 230, 255), np.float32) * 0.6 * glow
        )
    if preset == "night":
        stars = (rng.random((h, w)) > 0.9985).astype(np.float32)
        sky += cv2.GaussianBlur(stars, (0, 0), 0.6)[..., None] * 600
    return np.clip(sky, 0, 255).astype(np.uint8)


def _cover(img: np.ndarray, w: int, h: int, scale: float = 1.0) -> np.ndarray:
    """img を w×h（× scale）を覆う大きさに拡大し、上下は上寄せ・左右は中央で切り出す。"""
    W, H = round(w * scale), round(h * scale)
    s = max(W / img.shape[1], H / img.shape[0])
    big = cv2.resize(
        img[:, :, :3],
        (math.ceil(img.shape[1] * s), math.ceil(img.shape[0] * s)),
        interpolation=cv2.INTER_CUBIC,
    )
    ox = (big.shape[1] - W) // 2
    return big[:H, ox : ox + W]


def _harmonize(
    fg: np.ndarray, sky_old: np.ndarray, sky_new: np.ndarray, m: np.ndarray, k: float
) -> np.ndarray:
    """前景の色を新しい空の色へ寄せる（線形の光で、空の色の比を 0.6〜1.6 に抑えて k 乗。明るさの比は弱めに 0.3k 乗）。"""
    if k <= 0 or m.sum() < 50:
        return fg
    lo = _srgb_to_lin(sky_old)[m].mean(0) + 1e-3
    ln = _srgb_to_lin(sky_new)[m].mean(0) + 1e-3
    wts = np.array([0.0722, 0.7152, 0.2126], np.float32)
    g = ln / lo
    lum = float((g * wts).sum())
    color = np.clip(g / lum, 0.6, 1.6) ** k
    bright = np.clip(lum, 0.5, 2.0) ** (0.3 * k)
    lin = _srgb_to_lin(fg) * (color * bright)[None, None, :]
    return _lin_to_srgb(lin)


def _sky_composite(frame: np.ndarray, alpha: np.ndarray, B_old: np.ndarray, F_near: np.ndarray, sky: np.ndarray,
                   harmonize: float) -> np.ndarray:  # fmt: skip
    """縁の画素から古い空の色を抜いて、新しい空と重ねる。

    抜いた前景の色 (I − a·B)/(1 − a) は a が 1 に近いほど誤差が大きくなるので、近くの前景の色の平均と (1 − a) : a で混ぜる
    （正解つきの合成で、抜かない・そのまま抜く より髪の縁の誤差が小さかった）。
    """
    img = frame[:, :, :3].astype(np.float32)
    a = alpha[..., None]
    exact = np.where(a < 0.95, (img - a * B_old) / np.maximum(1.0 - a, 0.05), img)
    F = np.clip((1.0 - a) * exact + a * F_near, 0, 255).astype(np.uint8)
    sure_sky = alpha > 0.9
    F = _harmonize(F, frame[:, :, :3], sky, sure_sky, harmonize)
    out = sky.astype(np.float32) * a + F.astype(np.float32) * (1 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


@command(
    69,
    "sky-replace",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg("--sky", help="新しい空（画像か動画）。省略時は --preset の空を作る"),
    arg(
        "--preset",
        choices=["sunset", "dusk", "blue", "night"],
        default="sunset",
        help="--sky が無い時に作る空: sunset（夕焼け。既定）／ dusk（夕暮れ）／ blue（青空と雲）／ night（夜空）",
    ),
    arg(
        "--harmonize",
        type=float,
        default=0.4,
        help="前景の色を新しい空の色に寄せる強さ（0〜1。0 で寄せない。既定 0.4）",
    ),
    arg(
        "--motion",
        choices=["auto", "none"],
        default="auto",
        help="動画で、新しい空をカメラの動き（空以外の所の動き）に合わせて動かす（auto。既定）／ none（動かさない）",
    ),
    arg(
        "--smooth",
        type=float,
        default=0.5,
        help="動画の空の切り抜きの時間方向のなめらかさ（0〜0.9。既定 0.5）",
    ),
    arg(
        "--save-mask",
        action="store_true",
        help="空の切り抜き（白 = 空）も書き出す（<出力>-mask.png / .mp4）",
    ),
)
def sky_replace(a: argparse.Namespace) -> None:
    """空の領域だけを見つけて（UperNet ADE20K の「空」）、新しい空（画像・動画・作った夕焼けなど）に置き換える。"""
    info = core.probe(a.src)
    out = _out(a, info, "sky-replace")
    json_path = _json_path(a, out)
    t0 = time.time()
    core.log("UperNet ConvNeXt small（ADE20K、MIT）を読み込みます")
    seg = _SkySeg()
    sky_info = core.probe(a.sky) if a.sky else None
    sky_frames = None
    if a.sky and sky_info is not None and not sky_info.is_image:
        sky_frames = core.read_frames(a.sky)
    sky_img = (
        core.load_image(a.sky)
        if (a.sky and sky_info is not None and sky_info.is_image)
        else None
    )

    def new_sky(w: int, h: int, horizon: float, margin: float) -> np.ndarray:
        if sky_frames is not None:
            f = next(sky_frames, None)
            if f is not None:
                new_sky.last = f  # type: ignore[attr-defined]
            src = getattr(new_sky, "last", None)
            core.require(src is not None, f"空の動画を読めません: {a.sky}")
            return _cover(src, w, h, margin)
        if sky_img is not None:
            return _cover(sky_img, w, h, margin)
        return _make_sky(
            round(w * margin), round(h * margin), a.preset, horizon / margin
        )

    def horizon_of(alpha: np.ndarray) -> float:
        rows = np.nonzero((alpha > 0.5).mean(axis=1) > 0.02)[0]
        return float((rows.max() + 1) / alpha.shape[0]) if len(rows) else 0.5

    if info.is_image:
        img = _even(next(_frames(a.src, a)))[:, :, :3]
        h, w = img.shape[:2]
        prob = seg(img)
        alpha, B, Fn = _sky_alpha(img, prob)
        sky = new_sky(w, h, horizon_of(alpha), 1.0)
        res = _sky_composite(img, alpha, B, Fn, sky, a.harmonize)
        core.save_image(out, res)
        if a.save_mask:
            core.save_image(
                out.with_name(out.stem + "-mask.png"),
                np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8),
            )
        p = out.with_name(out.stem + "-preview.jpg")
        core.save_image(p, core.contact_sheet([img, (alpha * 255).astype(np.uint8), res], ["元の画", "空の切り抜き", "置き換えた結果"],
                                              cols=3, cell_width=min(480, w)))  # fmt: skip
        core.write_json(json_path, {"source": str(a.src), "sky": a.sky or f"preset:{a.preset}", "sky_percent": round(float(alpha.mean() * 100), 2),
                                    "seconds": round(time.time() - t0, 2)})  # fmt: skip
        core.log(f"空: 画面の {alpha.mean() * 100:.1f}%")
        core.log(f"保存: {p}")
        core.log(f"保存: {json_path}")
        core.log(f"保存: {out}")
        return
    total = _total(info, a)
    flow = _Flow(480)
    prev_prob = prev_gray = None
    off = np.zeros(2)  # 空の絵をずらした量（px）
    zoom = 1.0
    margin = 1.35 if a.motion == "auto" else 1.0
    big_sky = None
    keep = set(_sample_idx(total, 4))
    sheet, per = [], []
    mw = (
        core.VideoWriter(out.with_name(out.stem + "-mask.mp4"), info.fps)
        if a.save_mask
        else None
    )
    with core.VideoWriter(
        out,
        info.fps,
        audio_from=a.src,
        audio_start=a.start or 0.0,
        audio_duration=a.duration,
    ) as wr:
        for i, frame in enumerate(_frames(a.src, a)):
            frame = _even(frame)[:, :, :3]
            h, w = frame.shape[:2]
            prob = seg(frame)
            gray = flow.small(frame)
            moved = (0.0, 0.0, 1.0)
            if prev_prob is not None and prev_gray is not None:
                mapx, mapy, valid = flow.warp_prev(gray, prev_gray, (w, h))
                if valid.mean() > 0.3 and a.smooth > 0:
                    pw = cv2.remap(
                        prev_prob,
                        mapx,
                        mapy,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    prob = np.where(
                        valid, (1 - a.smooth) * prob + a.smooth * pw, prob
                    ).astype(np.float32)
                if a.motion == "auto":
                    from .ai import decompose_similarity, global_motion

                    pg = cv2.resize(prev_gray, (w, h))
                    cg = cv2.resize(gray, (w, h))
                    fg = (
                        cv2.erode(
                            (prev_prob < 0.3).astype(np.uint8),
                            np.ones((9, 9), np.uint8),
                        )
                        > 0
                    ).astype(np.uint8) * 255
                    m, _n = global_motion(pg, cg, mask=fg)
                    if m is not None:
                        dx, dy, _ang, sc = decompose_similarity(m)
                        off += (dx, dy)
                        zoom *= sc
                        moved = (dx, dy, sc)
            alpha, B, Fn = _sky_alpha(frame, prob)
            if big_sky is None or sky_frames is not None:
                big_sky = new_sky(w, h, horizon_of(alpha), margin)
            # 大きく作った空の、中央から off だけずらした所を切り出す（カメラが右へ動くと景色は左へ動く）
            Hs, Ws = big_sky.shape[:2]
            M = np.float32(
                [
                    [zoom, 0, (w - Ws * zoom) / 2 + off[0]],
                    [0, zoom, (h - Hs * zoom) / 2 + off[1]],
                ]
            )
            sky = cv2.warpAffine(
                big_sky,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            res = _sky_composite(frame, alpha, B, Fn, sky, a.harmonize)
            wr.write(res)
            if mw is not None:
                mw.write(np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8))
            per.append({"t": core.t_of(i, info.fps, a.start or 0.0), "sky_percent": round(float(alpha.mean() * 100), 2),
                        "sky_shift": [round(float(off[0]), 2), round(float(off[1]), 2)], "step": [round(v, 3) for v in moved]})  # fmt: skip
            if i in keep:
                sheet += [frame, res]
            prev_prob, prev_gray = prob, gray
            _progress(i, total, t0)
    if mw is not None:
        mw.close()
    p = out.with_name(out.stem + "-preview.jpg")
    core.save_image(
        p,
        core.contact_sheet(
            sheet, ["元", "置き換えた結果"] * (len(sheet) // 2), cols=4, cell_width=320
        ),
    )
    core.write_json(json_path, {"source": str(a.src), "sky": a.sky or f"preset:{a.preset}", "frames": len(per), "per_frame": per,
                                "seconds": round(time.time() - t0, 2)})  # fmt: skip
    core.log(
        f"{len(per)} コマ。空の面積の平均 {np.mean([p_['sky_percent'] for p_ in per]):.1f}%"
    )
    core.log(f"保存: {p}")
    core.log(f"保存: {json_path}")
    core.log(f"保存: {out}")
