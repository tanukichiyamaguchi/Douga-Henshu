"""自動で見つける（画面の中の物・文字・色）。

担当するテクニック:
   78 物体検出（objects）
   79 文字認識（ocr）
   90 ロゴ検出（logos）
   94 顕著性マップ（saliency）
   96 テキスト検出（text-detect）
   97 代表色の抽出（palette）
   98 QRコード検出（qr）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
使う AI（どれも商用で使えるライセンス。初めて使う時に取得）:
  Faster R-CNN v2（torchvision）/ EfficientDet-Lite2（MediaPipe）… 78（detectors.py）
  Grounding DINO tiny（IDEA-Research/grounding-dino-tiny、Apache-2.0。英語の言葉だけ）… 78・90 の --query
  EasyOCR 1.7.2（Apache-2.0）の CRAFT（MIT）と japanese_g2（Apache-2.0）… 79・96 の --model craft
  PP-OCRv3 の文字検出（OpenCV Zoo、Apache-2.0）… 96・94 の文字の重み
  YuNet（OpenCV Zoo、MIT）… 94 の顔の重み、WeChat QRCode（Apache-2.0）… 98
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import core, models
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/find/（test_*.py が正解つきの試験素材を作って照らす。改良の比べ合いは work/）
# 速さは他の作業と CPU（4 コア）を分け合った状態の壁時計（負荷 4〜9 の時があり、ばらつく）
NOTES: dict[int, dict[str, str]] = {
    78: {
        "status": "◎",
        "how": "COCO の 80 種類: detectors.ObjectDetector（accurate = torchvision の Faster R-CNN v2、fast = MediaPipe の "
        "EfficientDet-Lite2）。種類ごとの重なり除去に加え、ほぼ同じ枠（IoU 0.7 以上）に別の名前が付いた物は確からしさの高い方"
        "だけ残す。名前は日本語。--classes で種類を絞る（日本語・英語）。--query で 80 種類にない物を英語の言葉で探す"
        "（Grounding DINO tiny。IDEA-Research/grounding-dino-tiny、Apache-2.0。transformers 5.2 で動作。入力の短い辺は 800px まで、"
        "下限 0.45）。動画は --every 秒ごと（既定 accurate 0.2・fast 0.1・--query 0.5）に調べ、間のコマは同じ名前で重なりの"
        "大きい枠どうしを直線で補って MP4 に描く（音声は残す）。出力: 枠と名前を描いた画像・MP4・一覧 JPG、JSON（コマごとの名前・"
        "確からしさ・枠〔元の画素〕と、名前ごとの映ったコマの数・同時に映った最大の数・最初と最後の時刻）",
        "measured": "貼り付け試験（mp-cat_and_dog.jpg の猫と犬〔枠は目で確認〕を、story540 などのコマ 7 枚に高さ 48〜320px・"
        "左右反転ありで貼った 40 枚。名前が合い IoU 0.5 以上を当たり）: accurate 38/40（高さ 48px の 2 枚を見逃し）・当たりの "
        "IoU 平均 0.959・違う名前 0・離れた所の余計 0・1 枚 2.4〜2.6 秒、fast 19/40（高さ 128px 以上でないと見つからない。48・80px は "
        '0/16）・0.08 秒、--query "cat, dog" 28/40（下限 0.45。0.35 なら 30/40）・猫と犬の取り違え 5〜6 件・2.2〜3.8 秒。'
        "mp-cats_and_dogs.jpg: 猫 2・犬 2（重なり除去の前は、コーギーに「犬 0.99」と「猫 0.54」が重なって出た）、"
        "mp-cat_and_dog.jpg: 猫 1・犬 1（＋ 画面全体を「ベンチ 0.50」と誤検出）。fruits.jpg: 80 種類ではかんきつ類も"
        'キウイもみな「オレンジ」、--query "orange, lemon, lime, kiwi" でキウイ・レモン・オレンジ 2・ライムを正しく分けた。'
        "vtest（目で数えて 6 人の 3 コマ）: accurate 6・6・6 人、fast 5・6・5 人（ほかに白いバンを「トラック」、余計な"
        '「フリスビー」「ベンチ」各 1）。story540 で --query "eyelash curler, table"（0.5 秒ごと 31 コマ。正解はビューラーの場面 '
        "0〜3.04 秒・カフェ 12.38〜15.04 秒）: ビューラー 7/7・テーブル 6/6、余計 0（下限 0.35 では別の場面の目元を "
        "eyelash curler とした余計が 11 件〔確からしさ 0.35〜0.41。正しい物は 0.53 以上〕あったので、既定を 0.45 にした）。story540 の 3〜9 秒（0.2 秒ごと 30 コマ、accurate）: 手に持ったスマートフォン（0.98〜0.99）・浴室の流し台を見つけ、余計な「はさみ」1 回。日差しの場面では同じ人に全身と上半身の 2 つの枠が出るコマがあった（重なり 0.7 未満なので残る）。枠と名前を描いた MP4（間のコマは直線で補う）まで 70.6 秒",
        "limits": "80 種類は COCO の物だけ（化粧品・ビューラーなどは --query）。--query は英語だけ（Claude が訳して渡す。日本語は "
        "80 種類の名前だけ自動で直す）で、似た言葉（eyelash と目元など）に引っぱられることがある。高さ 50px 未満の小さな物は"
        "見逃しやすい（fast は 130px 未満）。accurate は CPU で 1 コマ 2〜4 秒・--query は 2〜5 秒なので、長い動画は --every を"
        "大きく、--max-side を小さく。枠は箱（形の切り抜きは 51〜75 を使う）。同じ人に全身と上半身の 2 つの枠が出ることがある（人数は 95 で数える）。動画の間のコマの枠は直線で補った位置なので、"
        "速い動きではずれる",
        "tested": "test_objects.py（猫・犬の貼り付け 40 枚 × accurate・fast・--query、story540 の --query を時間の正解と照合）、story540 の 3〜9 秒の MP4、"
        "mp-cat_and_dog・mp-cats_and_dogs・fruits（目で確認）、vtest の 3 コマ（人を目で数えて照合）。確認画像 78-objects-*.jpg",
    },
    79: {
        "status": "○",
        "how": "EasyOCR 1.7.2（Apache-2.0）の日本語＋英語（CRAFT で文字の場所〔MIT〕、japanese_g2 で認識〔Apache-2.0〕）。重みは "
        "GitHub のリリースから取ってキャッシュに置く（EasyOCR 自身には取得させず、読み込む時に md5 を照合させる）。EasyOCR に "
        "numpy を渡すと CRAFT には BGR のまま渡り灰色も BGR として作られるので、CRAFT には RGB、認識には正しい灰色を自分で渡す。"
        "CRAFT の拡大率は長辺がおよそ 1440px になる倍率（1〜2）。認識は灰色と白黒反転の両方で行い、枠ごとに確からしさの高い方"
        "（白字に黒い縁のテロップ向け）。確からしさ 0.1 未満は捨てる。動画は --every 秒ごと（既定 0.5）に読み、同じ場所"
        "（IoU 0.3 以上）の同じ文字（NFKC・空白なしで、編集距離が長さの 1/4 以内）をまとめて映っている区間を出す（文字は確からしさの"
        "合計がいちばん多い読み）。出力: JSON（まとめた文字・区間・四隅・読みの候補と、コマごとの読み）、文字と枠を描いた画像"
        "（動画は文字のあるコマの一覧）、--srt で時刻つきの SRT",
        "measured": "採点: NFKC・空白なしで、正解の行ごとに最も近い読みの行を当てた文字の正解率（1 − 編集距離 ÷ 文字数）。v4 の 2 コマ"
        "（14.7 秒「初回限定 まつげパーマ＋眉WAX / モチ約2倍 ／ 黄金比の似合わせ眉 / 7,980円（税込）」、17.6 秒「KATE stage LASH / "
        "蒲田西口店 / 蒲田駅西口 徒歩2分 …」の 10 行・118 文字。明朝体の飾り文字）: 0.958（誤り 5: 「＋」を形の同じ漢字「十」と 2 回、"
        "大きな桃色の「円」を落とす・「p」と読む、「／」を落とす）。1 コマ 3.1 秒。設定の比べ合い（反転なし）: 拡大率 1.0 では 0.847、1.5 で 0.958、"
        "先に 2 倍に拡大しても 0.958 のまま（2〜7 倍遅い）、beam search も変わらない。合成（story540 の 3 コマに 5 行・78 文字を書く。画面に収まらない行は"
        "折り返す）: 淡い帯に黒字 14〜60px で 0.933〜0.969、白字に黒い縁 14・18・24・32・44・60px で 0.641・0.728・0.826・0.815・"
        "0.815・0.754（反転して読む工夫の前は 0.426・0.703・0.687・0.662・0.795・0.677）。遠近をつけた札「OPEN / 営業中」（幅 "
        "240px）: 奥行き方向 0・25・45 度（上下 0・20 度）で 6/6 とも 1.000、60 度で 0.29〜0.43。文字の無い story540: 読んだ物 0 件"
        "（10 コマ・16 コマの 2 回）。動画: v4 の 13〜20 秒を 0.5 秒ごと（14 コマ）で、まとめた文字 24 件。区間は作品の場面の始まり（data-start 13.5・16.0 秒）に合う（13.0〜13.5 秒は前の場面の札、14.0〜16.0 秒は値段の場面、16.5〜20.0 秒は店の案内。文字が出てくる動きの分だけ遅れる）。同じ値段でも「7,980」（1.00）と「7,980p」（0.30）のように読みが分かれると別の件になる。44.5 秒（モデルの読み込みを含む）",
        "limits": "EasyOCR の精度の限界: 白字に黒い縁のテロップは 6〜8 割の正解率（黒字・淡い帯の上の字は 9 割 3 分以上）。"
        "「＋」と「十」、「ー」と「一」、「:」と「;」など形の同じ字を取り違える。強い遠近（奥行き 60 度）・飾りの強い大きな字（光る"
        "桃色の「円」）は落とすことがある。縦書きは試していない。より強い OCR（PaddleOCR の PP-OCRv5 など）は入れていない（新しい"
        "依存関係が要る）。CPU で 540×960 の 1 コマ約 3 秒（他の作業と CPU を分け合うと 10〜25 秒）なので、長い動画は --every を"
        "大きく",
        "tested": "test_ocr.py（v4 の正解つき 2 コマ、合成 36 枚〔2 つの書き方 × 6 つの大きさ × 3 コマ〕、遠近の札 8 枚、文字の無い"
        " story540）、採点は ocr_eval.py。設定の比べ合いは ocr_eval.py の実行・work/ocr_polarity.py・ocr_mag_big.py・"
        "ocr_outline_nopolarity.py。確認画像 79-ocr.jpg",
    },
    90: {
        "status": "○",
        "how": "--ref のマーク画像を探す（本命）: SIFT を RootSIFT にして「画面 → マーク」の向きで照合（比 0.85。同じマークが"
        "複数映っても比の試験で消えない）、USAC MAGSAC でホモグラフィー、見つけた所の照合を除いて繰り返す（最大 --max-count 個）。"
        "長辺 1000px 以下の画面は 2 倍に拡大して特徴点をとる。見つけた所をマークの形に戻してマークとの正規化相互相関を測り、"
        "0.3 未満と、形のおかしい四角（凸でない・角が 20〜160 度の外など）は捨てる。特徴点が 30 個未満のマークは色の型の照合"
        "（大きさ 24 段階・BGR の TM_CCOEFF_NORMED・0.82 以上）。透過 PNG は見える所だけで照合する。--ref が無い時は Grounding "
        "DINO tiny（Apache-2.0）で言葉（既定 logo）の候補を出す。出力: JSON（四隅・枠・確からしさ・映っている区間）、枠を描いた"
        "画像・MP4・一覧 JPG",
        "measured": "自作のマーク（円の紋章。実在のロゴではない）を story540・v4・vtest の 11 コマに、幅 50〜260px・奥行き方向"
        " ±50 度・上下 ±30 度・回転 ±180 度・明るさ 0.7〜1.2 倍・ぼかし σ0〜1.2・JPEG 80 で 1〜2 個貼った 80 枚（100 個）: "
        "見つけた 84/100、四隅の誤差 中央 1.74px・平均 2.61px、余計 0、マークの無いコマ 33 枚の誤検出 0、1 枚 約 200ms。"
        "見逃し 16 個のうち 15 個は上辺 100px 未満。改良の経過（同じ 100 個）: 最初の版（SIFT・比 0.75・拡大なし）59 → "
        "RootSIFT・比 0.85・2 倍に拡大で 84（どれも誤検出 0。マーク側を傾けて特徴点を増やす ASIFT は単独で 57〜76、拡大と"
        "合わせても 65〜79 で、使っていない）。動く合成動画（マークが 0.5〜2.5 秒に動いて回る。0.2 秒ごとに調べる）: 区間 "
        "0.625〜2.50 秒（マークのある 9/9 コマ）、四隅の誤差 平均 1.20px・最大 1.92px、無いコマの誤検出 0。OpenCV の box.png を"
        " box_in_scene.png から: 1 個（照合 178 点・相関 0.79。目で確認）。模様の少ないマーク（特徴点 12 個 → 型の照合）: 幅 "
        "40〜240px・明るさを変えた 20 枚で 18/20（IoU 平均 0.81）、マークの無い 33 枚の誤検出 0（灰色の型の照合では誤検出 "
        "157 件。正解 0.88 以上と誤り 0.87 までが重なったので色にした。色では正解 0.85 以上・誤り 0.79 まで）。言葉「logo」: "
        "貼ったマークは 37/37（確からしさ 0.74〜0.87）だが、ロゴの無い story540 でもコマごとの最大が 0.37〜0.70 で、0.5 以上が"
        " 15 コマ中 5 コマ",
        "limits": "マークの画像（--ref）がある時が本命で、無い時の言葉の検索は候補探し（ロゴの無い映像でも 3 割ほどのコマに"
        "候補が出る。一覧画像を目で見て判断する。--min-score 0.75 で減る）。幅 100px 未満・強い遠近（奥行き 50 度近く）・ぼけた"
        "マークは見逃しやすい。型の照合（模様の少ないマーク）は回転・遠近に弱い。左右反転したマークは別物とみなす。実在の会社の"
        "ロゴでの試験はしていない",
        "tested": "test_logos.py（自作マークの貼り付け 80 枚・マークの無いコマ 33 枚・模様の少ないマーク 20 枚・box.png・動く合成"
        "動画・言葉「logo」）、改良の比べ合いは work/logos/variants*.py・template_scores.py・query_scores.py。確認画像 "
        "90-logos-*.jpg",
    },
    94: {
        "status": "○",
        "how": "学習した視線のモデルではなく、古典的な方法の組み合わせ（長辺 320px で計算）: OpenCV の spectral residual"
        "（Hou & Zhang 2007）・色の珍しさ（Cheng ほか 2011 の histogram contrast をもとに、少ない色も近い色に置き換えずに比べる）・"
        "向き（4 方向の Gabor の強さ）を、それぞれ Itti ほか 1998 の正規化 N()（山が 1 つだけ目立つ地図を強める）にかけて平均。"
        "そこに顔（YuNet。目の高さを中心に。重み 0.6）・文字（PP-OCRv3 の文字検出。顔を文字と見なすことがあるので、顔に重なる枠は除く。0.4）・動き（DIS の動きから画面全体の動きを"
        "引いた、周りと違う動き。動画のみ。0.4）を足し、画面の中央を重くする（0.3）。動画は前のコマと半々に混ぜる（カットでは"
        "混ぜない）。--method spectral / fine で OpenCV の 1 つだけも選べる。出力: 熱の色の重ね表示（山に番号。画像は元と並べた "
        "JPG、動画は MP4・一覧）、--map で白黒の地図、JSON（コマごとの山 3 つ・3×3 の区画ごとの平均と、全体の平均・いちばん"
        "目が行きにくい区画）",
        "measured": "探索画面（灰色の地に 7×7 個、1 つだけ違う物。各 30 枚）で、いちばんの山が違う 1 つに当たった数: 色 27/30"
        "（中央の重み 0.3 で 29/30）・向き 30/30・大きさ 30/30・明るさ 30/30。OpenCV の spectral residual だけ: 色 24・向き 2・"
        "大きさ 30・明るさ 30、fine grained だけ: 28・3・17・29。自然な背景（story540 の 5 コマと OpenCV のサンプル 4 枚）に"
        "小さな物（水色の丸・魚の写真・赤い札）を貼った 120 枚: 顔・文字の重みなしで 1 位 94〜96/120・上位 3 つ 109〜115/120"
        "（spectral residual＋fine grained の最初の版は 44/120・59/120、少ない色を置き換える元の histogram contrast では "
        "71/120）。既定（顔・文字あり）は 1 位 91/120・上位 3 つ 111/120（顔のある場面では顔が 1 位になることがある。文字の枠から顔を除く前は 71/120・107/120 で、PP-OCRv3 が顔を文字とした分だけ顔が強すぎた）。story540 の顔のある "
        "24 コマで 1 位が顔に来た数: 顔の重み 0 で 9、0.6（既定）で 23、1.0 で 24。v4 の文字のある 38 コマで 1 位が文字に来た数: "
        "文字の重み 0 で 34、0.4 で 36。速さ: 540×960 で 1 コマ約 60ms（顔・文字・動きあり）。story540（361 コマ）の MP4 まで 26.7 秒、v4 を 0.1 秒ごと（200 コマを調べ 600 コマを描く）20.1 秒",
        "limits": "人の視線の記録で学習したモデル（DeepGaze など）ではないので、実際の見られ方の目安。視線の記録のデータ"
        "（MIT1003・SALICON 等）はライセンスがはっきりしないので使わず、正解の分かる合成の試験で確かめた。顔・文字の重みは"
        "広告向けの目安で、強くすると山はほぼ必ず顔・文字に来る（--faces 0 --text 0 で見た目の目立ちやすさだけ）。模様の多い画面"
        "では山が散る。動きは画面全体の平行移動だけを引く（ズーム・回転は引かない）。3×3 の区画の値で、文字を置く場所の候補を"
        "選べる（値が小さい区画）",
        "tested": "test_saliency.py（探索画面 4 種 × 30 枚、自然な背景の小さな物 120 枚、顔・文字の重みの効き方）、組み合わせの"
        "比べ合いは work/sal/variants*.py、story540・v4 の MP4（94-saliency-*.jpg を目で確認）、fruits.jpg・home.jpg・"
        "HappyFish.jpg",
    },
    96: {
        "status": "◎",
        "how": "既定は EasyOCR の CRAFT（clovaai/CRAFT-pytorch の重み、MIT。79 と同じ。文字らしさ 0.7）で文字の場所を探す"
        "（CRAFT は RGB で学習しているので RGB で渡す）。--model ppocr で PP-OCRv3 の DB（OpenCV Zoo の "
        "text_detection_cn_ppocrv3、Apache-2.0。長辺 960px・枠の点数 0.5）も選べる（約 20 倍速いが、顔・目を文字と見なすので、"
        "YuNet で見つけた顔に半分以上重なる枠は除く）。動画は --every 秒ごと（既定 0.5）に調べ、文字の枠の面積の合計が画面の "
        "0.05%（--min-area）以上のコマを「文字あり」として区間にまとめる。出力: 枠を描いた画像・MP4・一覧 JPG、JSON（コマごとの"
        "枠・面積、文字がある区間）",
        "measured": "合成（story540 の 5 コマに、分かっている文を 12〜60px・白字に黒い縁／淡い帯に黒字で 4 行ずつ書いた 48 枚・"
        "192 行。正解の枠の 5 割以上を覆えば見つけた）: craft 192/192（どの大きさも 32/32）・検出の面積のうち正解の近くにある割合"
        " 0.997・1 枚 1.1〜2.3 秒、ppocr は顔を除く前 192/192・1.000・46ms、除いた後 138/192（顔の上に書いた行も除かれる）。"
        "文字の無い story540（0.5 秒ごと 31 コマ）で「文字あり」としたコマ: craft 0、ppocr 25（顔を除いた後 9）。ppocr は顔"
        "そのもの（枠の点数 0.96〜0.99）・目・ビューラーの金具を文字にした（96-text-detect-ppocr-false.jpg）。v4（0.5 秒ごと "
        "40 コマ）: craft 35・ppocr 35（除いた後 29）で、2 つの判定が分かれた 8 コマを目で確かめると craft が 8/8 正しく（白い"
        "フラッシュの中の薄い題字 3 コマを文字あり、顔の寄り・部屋など 5 コマを文字なし）、ppocr は 2/8。craft の枠は v4 の"
        "小さな筆記体の英字・桃色の値段・薄い題字まで囲んだ（目で確認）。速さ: v4（20 秒）を 0.5 秒ごと: craft 60〜133 秒、"
        "ppocr 3.5〜5.4 秒。v4 の 14〜20 秒の MP4（craft・0.5 秒ごと）: 12/12 コマで文字あり、区間 14.00〜20.00 秒（18.1 秒）",
        "limits": "craft は CPU で 1 コマ 1〜2 秒（長い動画は --every を大きく、急ぐ時は --model ppocr〔顔・目の誤検出あり〕）。"
        "縦書き・強い遠近の字は合成の試験に入れていない（79 の札は 45 度まで読めた）。文字の形をした模様（ロゴの記号など）を"
        "文字に数えることはあり得る（試験の素材には無かった）。読むのは 79",
        "tested": "test_text_detect.py（合成 48 枚・192 行、story540・v4 を CLI で）、work/textdet_variants.py（ppocr の言語・顔の"
        "除外の比べ合い）、判定の分かれた v4 の 8 コマを目で確認（work/textdet/v4-disagree.jpg）。確認画像 "
        "96-text-detect-*.jpg",
    },
    97: {
        "status": "◎",
        "how": "CIELAB の k 平均（OpenCV の kmeans。k-means++・8 回試して最良・乱数の種は固定）。色の中心は「色の変化が小さい"
        "半分の画素」（Lab の 3 成分の Sobel の和が中央値以下）で求め、割合はすべての画素を最も近い中心に振り分けて数える"
        "（輪郭の中間色が 1 色として選ばれるのを防ぐ）。中心どうしの色差 CIEDE2000（Sharma ほか 2005 の式を実装）が 3 未満の色は"
        "まとめる。長辺 240px に縮めて数える。動画は 76 のカット検出で場面に分け（--every で一定間隔にも）、各区間の両端 1 割を"
        "除いた最大 12 コマを使う。動画全体の代表色は区間の長さで重みをつける。出力: 色見本の帯の PNG（区間ごとに代表のコマと"
        "割合どおりの帯、#RRGGBB と %）、JSON（#RRGGBB・RGB・Lab・割合）",
        "measured": "CIEDE2000 の実装: Sharma ほかの試験データ 12 組で最大誤差 0.00004。合成画像 40 枚（3〜6 色・割合 4% 以上を "
        "20px の格子に混ぜた 960×540）: 劣化なしで色の数 40/40 一致・色差 ΔE00 平均 0.04・最大 0.49、割合の誤差 平均 1.1 点・"
        "最大 2.9 点。ノイズ σ6＋JPEG 85: 40/40、ΔE00 平均 0.18・最大 0.57、割合 平均 1.2 点・最大 3.2 点（1 枚 60〜90ms）。"
        "すべての画素で k 平均した最初の版は、輪郭の中間色が 1 色に選ばれて ΔE00 最大 40.6（劣化なし）だった。5 色の画像に 8 色を"
        "頼む: まとめると正解の 5 色、まとめない（--merge 0）と同じ色が 2 つずつに割れて 8 色。3 場面（各 2 秒・別の 4 色）の"
        "合成動画（H.264）: 場面 3/3（境目 2.0・4.0 秒）、ΔE00 最大 2.39・平均 1.07〜1.37、割合の誤差 最大 2.4 点（2.0 秒）。"
        "story540: 76 と同じ 5 場面、肌・茶・ベージュ・浴室の灰・植物の緑など見た目どおり（目で確認。3.3 秒）。starry_night.jpg: "
        "青 3 色・薄い黄緑・額の黒（0.2 秒）",
        "limits": "面積の多い順なので、小さな差し色（星月夜の黄色い月など）は 5 色に入らない（-n を増やす）。色の名前は出さない"
        "（#RRGGBB から言う）。場面の途中で色が大きく変わる（暗転・フラッシュ）と混ざった色になる（--every で細かく区切る）。"
        "H.264 の色の間引き（4:2:0）で細かい模様の色は少しずれる（上の動画で ΔE00 最大 2.4）。透過は見ない",
        "tested": "test_palette.py（Sharma の試験データ、合成画像 40 枚 × 2 条件、5 色の画像に 8 色、3 場面の合成動画）、story540・"
        "starry_night.jpg（目で確認）。確認画像 97-palette-*.png・97-palette-truth.jpg",
    },
    98: {
        "status": "◎",
        "how": "WeChat QRCode（OpenCV contrib。CNN の検出と超解像。Apache-2.0。モデルは WeChatCV/opencv_3rdparty から取得）と "
        "OpenCV の QRCodeDetectorAruco を両方使い、同じ中身・同じ場所の物は 1 つにする。1 つも読めないコマは 2 倍に拡大（長辺 "
        "1280px 以下の時）・鮮鋭化して読み直す。動画は --every 秒ごと（既定 0.1）に読み、同じ中身が続く所を区間にまとめる"
        "（0.5 秒を超える隙間で切る）。出力: JSON（中身・区間・時刻ごとの四隅）、枠と中身を描いた画像・MP4・一覧 JPG",
        "measured": "cv2.QRCodeEncoder で作った QR（中身 4 種類。日本語を含む）を story540 のコマに 1 マス 1.5〜4px・奥行き方向 "
        "0〜55 度・ぼかし σ0〜2 で貼った 144 枚（各条件 4 枚。JPEG 80）: 既定 61/144（読み直しなし 53、WeChat だけ 48、aruco "
        "だけ 24、OpenCV の標準 15）。1 マス 4px は 27/36、1 マス 3〜4px・傾き 35 度以下・σ1 以下は 32/32。読めた物の四隅の"
        "誤差 平均 0.67px・最大 1.46px、違う中身を読んだ数 0。合成動画（4 秒・24fps。QR-A が動きながら 0.5〜2.0 秒、QR-B が"
        "傾いて 2.5〜3.5 秒）: 区間 0.50〜2.00・2.50〜3.50 秒で一致、余計 0（3.7 秒）。QR の無い v4（20 秒）・story540（15 秒）を 0.1 秒ごと: 読んだ物 0（7.4 秒・5.6 秒）。速さ: 540×960 で 1 コマ 25〜55ms",
        "limits": "1 マス 2px 未満（QR 全体で 60px ほど）、奥行き方向 55 度の傾き、σ2 のぼかしはほぼ読めない（上の表）。"
        "読めない QR の場所は出さない（読めた物だけ）。QR 以外（バーコード・マイクロ QR など）は読まない",
        "tested": "test_qr.py（144 枚の条件別の表と方法の比べ合い、2 つの QR の合成動画を CLI で）、読み直しの比べ合いは "
        "work/qr_fallback.py。確認画像 98-qr-synth.jpg・98-qr-video.jpg",
    },
}


# ── 共通の部品 ────────────────────────────────────────────────────────────

_GREEN = (60, 180, 40)
_YELLOW = (0, 210, 255)
_SHEET_MAX = 24  # 確認用の一覧画像に並べるコマの数の上限


def _rng(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": getattr(a, "start", 0.0) or 0.0,
        "duration": getattr(a, "duration", None),
        "max_frames": getattr(a, "max_frames", None),
    }


def _size(info: core.MediaInfo, a: argparse.Namespace) -> tuple[int, int]:
    """処理する大きさ（--max-side を守り、偶数にする）。"""
    return core.scaled_size(info.width, info.height, None, getattr(a, "max_side", None))


def _paths(
    a: argparse.Namespace, tag: str, info: core.MediaInfo, video_ext: str = ".mp4"
) -> tuple[Path, Path, Path]:
    """(出力, JSON, 確認用の一覧 JPG)。画像なら出力そのものが注釈つきの画像。"""
    ext = ".jpg" if info.is_image else video_ext
    out = Path(a.out) if a.out else core.default_out(a.src, tag, ext)
    json_path = Path(a.json) if getattr(a, "json", None) else out.with_suffix(".json")
    sheet = out if out.suffix.lower() in core.IMAGE_EXTS else out.with_suffix(".jpg")
    return out, json_path, sheet


@dataclass
class _Sample:
    i: int  # 調べた範囲の中のコマ番号（0 から）
    t: float  # 元の動画の中の時刻（秒）
    frame: np.ndarray


def _samples(
    src: str,
    info: core.MediaInfo,
    a: argparse.Namespace,
    every: float,
    size: tuple[int, int],
) -> Iterator[_Sample]:
    """--every 秒ごとのコマ（画像なら 1 枚）。時刻はコマ単位で正確（頭から順に読んで間引く）。"""
    rng = _rng(a)
    step = max(1, round(every * info.fps)) if every > 0 and not info.is_image else 1
    for i, f in enumerate(core.read_frames(src, size=size, **rng)):
        if i % step == 0:
            yield _Sample(
                i, 0.0 if info.is_image else core.t_of(i, info.fps, rng["start"]), f
            )


def _step(info: core.MediaInfo, every: float) -> int:
    return max(1, round(every * info.fps)) if every > 0 else 1


def _fmt_t(t: float) -> str:
    return f"{t:.2f}秒"


_TAGS: dict[tuple[Any, ...], np.ndarray] = {}


def _tag_patch(text: str, size: int, bg: tuple[int, int, int]) -> np.ndarray:
    """色つきの札（文字は背景に合わせて白か黒）。同じ札は作り直さない（動画の全コマに描くため）。"""
    key = (text, size, bg)
    if key not in _TAGS:
        luma = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
        fg = (0, 0, 0) if luma > 150 else (255, 255, 255)
        width = int(len(text) * size * 1.08) + 14
        canvas = np.empty((int(size * 1.45) + 8, width, 3), np.uint8)
        canvas[:] = bg
        core.draw_text(canvas, text, (5, 3), size=size, color=fg, bg=None)
        cols = np.nonzero(np.any(canvas != np.array(bg, np.uint8), axis=(0, 2)))[0]
        right = min(width, int(cols.max()) + 6) if len(cols) else width
        _TAGS[key] = canvas[:, :right].copy()
    return _TAGS[key]


def _put_tag(
    img: np.ndarray,
    text: str,
    x: float,
    y: float,
    bg: tuple[int, int, int],
    *,
    size: int = 16,
    above: bool = True,
) -> None:
    """(x, y) を左下（above=False なら左上）にして札を貼る。画面からはみ出さないよう寄せる。"""
    patch = _tag_patch(text, size, bg)
    ph, pw = patch.shape[:2]
    h, w = img.shape[:2]
    x0 = int(min(max(0, x), max(0, w - pw)))
    y0 = int(y - ph) if above else int(y)
    if y0 < 0:
        y0 = int(y) if above else 0
    y0 = min(max(0, y0), max(0, h - ph))
    x1, y1 = min(w, x0 + pw), min(h, y0 + ph)
    img[y0:y1, x0:x1, :3] = patch[: y1 - y0, : x1 - x0]


def _thick(img: np.ndarray) -> int:
    return max(2, round(min(img.shape[:2]) / 300))


def _draw_box(
    img: np.ndarray,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    text: str = "",
    *,
    size: int | None = None,
) -> None:
    """枠（x1, y1, x2, y2）と、左上に名前の札を描く。"""
    x1, y1, x2, y2 = (round(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, _thick(img), cv2.LINE_AA)
    if text:
        s = size or max(13, min(img.shape[:2]) // 34)
        _put_tag(img, text, x1, y1, color, size=s)


def _draw_poly(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    text: str = "",
    *,
    size: int | None = None,
) -> None:
    """四角形（4×2 の点）と札を描く。"""
    p = np.round(np.asarray(pts, np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], True, color, _thick(img), cv2.LINE_AA)
    if text:
        s = size or max(13, min(img.shape[:2]) // 34)
        top = p.reshape(-1, 2)
        _put_tag(img, text, top[:, 0].min(), top[:, 1].min(), color, size=s)


def _box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _poly_box(pts: np.ndarray) -> tuple[float, float, float, float]:
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    return (
        float(p[:, 0].min()),
        float(p[:, 1].min()),
        float(p[:, 0].max()),
        float(p[:, 1].max()),
    )


@dataclass
class _Item:
    """注釈つきの動画に描く物 1 つ。geom は枠（x1, y1, x2, y2）か四角形（4×2）。処理した大きさの座標。"""

    geom: np.ndarray
    label: str
    color: tuple[int, int, int]
    key: str = ""  # 前後のコマで同じ物を結ぶ時の名前（同じ名前で、重なりが大きい物どうしを結ぶ）

    @property
    def box(self) -> tuple[float, float, float, float]:
        g = np.asarray(self.geom, np.float64)
        if g.shape == (4,):
            return float(g[0]), float(g[1]), float(g[2]), float(g[3])
        return _poly_box(g)


def _match(a: list[_Item], b: list[_Item], min_iou: float = 0.2) -> dict[int, int]:
    """前のコマの物 → 次のコマの物（同じ key で、重なりが大きい順に 1 対 1 で結ぶ）。"""
    pairs = []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x.key == y.key and np.asarray(x.geom).shape == np.asarray(y.geom).shape:
                v = _box_iou(x.box, y.box)
                if v >= min_iou:
                    pairs.append((v, i, j))
    used_a: set[int] = set()
    used_b: set[int] = set()
    out: dict[int, int] = {}
    for _v, i, j in sorted(pairs, reverse=True):
        if i not in used_a and j not in used_b:
            out[i] = j
            used_a.add(i)
            used_b.add(j)
    return out


def _render_video(
    src: str,
    info: core.MediaInfo,
    a: argparse.Namespace,
    size: tuple[int, int],
    keyed: dict[int, list[_Item]],
    out: Path,
    *,
    hold: int,
) -> Path:
    """間引いたコマで見つけた物を、全部のコマに描いて MP4 に書く（次に見つけたコマまで位置を直線で補う）。

    hold: 最後に見つけたコマから何コマ先まで描くか（次のコマで見つからなかった物は、そこで消す）。
    """
    rng = _rng(a)
    keys = sorted(keyed)
    matches = {k0: _match(keyed[k0], keyed[k1]) for k0, k1 in itertools.pairwise(keys)}
    writer = core.VideoWriter(
        out,
        info.fps,
        audio_from=a.src,
        audio_start=rng["start"],
        audio_duration=rng["duration"],
    )
    pos = 0
    with writer:
        for i, frame in enumerate(core.read_frames(src, size=size, **rng)):
            while pos + 1 < len(keys) and keys[pos + 1] <= i:
                pos += 1
            items: list[_Item] = []
            if keys and keys[pos] <= i:
                k0 = keys[pos]
                k1 = keys[pos + 1] if pos + 1 < len(keys) else None
                m = matches.get(k0, {})
                w = (i - k0) / (k1 - k0) if k1 is not None else 0.0
                for n, it in enumerate(keyed[k0]):
                    if k1 is not None and n in m:
                        g0 = np.asarray(it.geom, np.float64)
                        g1 = np.asarray(keyed[k1][m[n]].geom, np.float64)
                        items.append(
                            _Item(g0 * (1 - w) + g1 * w, it.label, it.color, it.key)
                        )
                    elif i - k0 <= hold:
                        items.append(it)
            for it in items:
                g = np.asarray(it.geom)
                if g.shape == (4,):
                    _draw_box(frame, it.box, it.color, it.label)
                else:
                    _draw_poly(frame, g, it.color, it.label)
            writer.write(frame)
    return out


def _sheet(images: list[np.ndarray], labels: list[str], cols: int = 4) -> np.ndarray:
    """確認用の一覧（多ければ等間隔に間引く）。"""
    if len(images) > _SHEET_MAX:
        idx = np.linspace(0, len(images) - 1, _SHEET_MAX).round().astype(int)
        images = [images[i] for i in idx]
        labels = [labels[i] for i in idx]
    cell = 360 if images[0].shape[0] >= images[0].shape[1] else 480
    return core.contact_sheet(
        images, labels, cols=min(cols, len(images)), cell_width=cell
    )


class _SheetBuffer:
    """確認用の一覧に載せるコマを溜める。多くなったら 1 つおきに減らし、全体から均等に残す（長い動画でもメモリを
    使いすぎないよう、最大 2 × _SHEET_MAX 枚）。"""

    def __init__(self) -> None:
        self.items: list[tuple[np.ndarray, str]] = []
        self.stride = 1
        self.n = 0

    def add(self, img: np.ndarray, label: str) -> None:
        if self.n % self.stride == 0:
            self.items.append((img, label))
            if len(self.items) > 2 * _SHEET_MAX:
                self.items = self.items[::2]
                self.stride *= 2
        self.n += 1

    def __bool__(self) -> bool:
        return bool(self.items)

    def first(self) -> np.ndarray:
        return self.items[0][0]

    def sheet(self, cols: int = 4) -> np.ndarray:
        return _sheet([i for i, _ in self.items], [t for _, t in self.items], cols)


# ── 97 代表色の抽出 ──────────────────────────────────────────────────────

_PAL_SIDE = 240  # 色を数える時のコマの長辺（px）
_PAL_PIXELS = 60000  # 1 区間で k 平均に使う画素の数の上限
_PAL_FRAMES = 12  # 1 区間から使うコマの数の上限


def _to_lab(bgr: np.ndarray) -> np.ndarray:
    """BGR（uint8）→ CIELAB（D65。L 0〜100）の float32（N×3）。"""
    f = bgr.reshape(-1, 1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(f, cv2.COLOR_BGR2Lab).reshape(-1, 3)


def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    f = cv2.cvtColor(
        np.asarray(lab, np.float32).reshape(-1, 1, 3), cv2.COLOR_Lab2BGR
    ).reshape(-1, 3)
    return np.clip(np.round(f * 255), 0, 255).astype(np.uint8)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 の色差（Sharma ほか 2005 の式）。lab1・lab2 は (..., 3)。"""
    lab1 = np.asarray(lab1, np.float64)
    lab2 = np.asarray(lab2, np.float64)
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    cm7 = ((c1 + c2) / 2) ** 7
    g = 0.5 * (1 - np.sqrt(cm7 / (cm7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    zero = c1p * c2p == 0
    dlp, dcp = l2 - l1, c2p - c1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(zero, 0.0, dhp)
    dhh = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp / 2))
    lpm, cpm = (l1 + l2) / 2, (c1p + c2p) / 2
    hs = h1p + h2p
    hpm = np.where(
        zero,
        hs,
        np.where(
            np.abs(h1p - h2p) <= 180,
            hs / 2,
            np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2),
        ),
    )
    t = (
        1
        - 0.17 * np.cos(np.radians(hpm - 30))
        + 0.24 * np.cos(np.radians(2 * hpm))
        + 0.32 * np.cos(np.radians(3 * hpm + 6))
        - 0.20 * np.cos(np.radians(4 * hpm - 63))
    )
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    cpm7 = cpm**7
    rc = 2 * np.sqrt(cpm7 / (cpm7 + 25.0**7))
    sl = 1 + 0.015 * (lpm - 50) ** 2 / np.sqrt(20 + (lpm - 50) ** 2)
    sc = 1 + 0.045 * cpm
    sh = 1 + 0.015 * cpm * t
    rt = -np.sin(np.radians(2 * dtheta)) * rc
    return np.sqrt(
        (dlp / sl) ** 2
        + (dcp / sc) ** 2
        + (dhh / sh) ** 2
        + rt * (dcp / sc) * (dhh / sh)
    )


def _hex(bgr: np.ndarray) -> str:
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02X}{g:02X}{b:02X}"


def _lab_and_edges(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """画像（BGR）→ (Lab の画素 N×3, 画素ごとの色の変化の大きさ N)。変化は Lab の 3 成分の Sobel の和。"""
    lab = cv2.cvtColor(img[:, :, :3].astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    grad = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        grad += np.abs(cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3))
        grad += np.abs(cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3))
    return lab.reshape(-1, 3), grad.ravel()


def palette_of(
    images: list[np.ndarray],
    k: int = 5,
    merge: float = 3.0,
    *,
    weights: list[float] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """画像（BGR。複数のコマでもよい）の代表色を、CIELAB の k 平均で k 色まで求め、多い順に返す。

    - 色の中心は「色の変化が小さい半分の画素」（輪郭でない所）で求める。輪郭の画素は 2 つの色が混ざった中間の色で、
      数が多いと中間の色が 1 色として選ばれてしまうため（合成画像の試験で確かめた）。
    - 割合は、すべての画素を最も近い中心に振り分けて数える。
    - 中心どうしの色差（CIEDE2000）が merge 未満の色は 1 つにまとめる（見分けられない色を 2 つ出さないため）。
    - weights: コマごとの重み（場面の長さなど）。重いコマから多くの画素を使う。
    """
    core.require(bool(images), "色を数える画像がありません")
    rs = np.random.default_rng(seed)
    w = np.asarray(weights if weights is not None else [1.0] * len(images), np.float64)
    w = w / w.sum()
    labs, grads = [], []
    for img, wi in zip(images, w, strict=True):
        lab, grad = _lab_and_edges(img)
        quota = min(len(lab), max(1, round(_PAL_PIXELS * 2 * wi)))
        if quota < len(lab):
            idx = rs.choice(len(lab), quota, replace=False)
            lab, grad = lab[idx], grad[idx]
        labs.append(lab)
        grads.append(grad)
    lab = np.concatenate(labs)
    grad = np.concatenate(grads)
    fit = lab[grad <= np.percentile(grad, 50)] if len(lab) >= 64 else lab
    if len(fit) > _PAL_PIXELS:
        fit = fit[rs.choice(len(fit), _PAL_PIXELS, replace=False)]
    distinct = len(np.unique(np.round(fit, 1), axis=0))
    k = max(1, min(k, distinct))
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, labels, centers = cv2.kmeans(
        np.ascontiguousarray(fit, np.float32), k, None, crit, 8, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel()
    groups = [[i] for i in range(k)]
    cents = [centers[i].astype(np.float64) for i in range(k)]
    # 見分けられないほど近い色をまとめる（近い組から順に）
    while len(cents) > 1:
        best = None
        for i in range(len(cents)):
            for j in range(i + 1, len(cents)):
                d = float(delta_e2000(cents[i], cents[j]))
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None or best[0] >= merge:
            break
        _, i, j = best
        groups[i] += groups.pop(j)
        mask = np.isin(labels, groups[i])
        cents[i] = fit[mask].mean(axis=0).astype(np.float64) if mask.any() else cents[i]
        del cents[j]
    cmat = np.array(cents, np.float32)
    # すべての画素（輪郭も含む）を最も近い中心に振り分けて割合を数える
    near = np.argmin(((lab[:, None, :] - cmat[None, :, :]) ** 2).sum(axis=2), axis=1)
    counts = np.bincount(near, minlength=len(cents)).astype(np.float64)
    total = counts.sum()
    out = []
    for n in np.argsort(-counts):
        if counts[n] <= 0:
            continue
        c = cents[n]
        bgr = _lab_to_bgr(c)[0]
        out.append(
            {
                "hex": _hex(bgr),
                "rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])],
                "lab": [round(float(v), 2) for v in c],
                "share": round(float(counts[n] / total), 4),
            }
        )
    return out


def _swatch_bar(colors: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
    """割合に合わせた色の帯（色ごとに #RRGGBB と % を書く）。"""
    bar = np.full((height, width, 3), 255, np.uint8)
    x = 0
    for n, c in enumerate(colors):
        w = round(c["share"] * width) if n < len(colors) - 1 else width - x
        if w <= 0:
            continue
        r, g, b = c["rgb"]
        bar[:, x : x + w] = (b, g, r)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        fg = (20, 20, 20) if luma > 140 else (245, 245, 245)
        # 帯が細い時は字を小さくする（#RRGGBB の 7 文字が入る大きさ）
        size = min(max(11, min(20, height // 5)), int((w - 6) / 4.4))
        if size >= 9:
            core.draw_text(bar, c["hex"], (x + 3, 6), size=size, color=fg, bg=None)
            core.draw_text(
                bar,
                f"{c['share'] * 100:.0f}%",
                (x + 3, 10 + size),
                size=size,
                color=fg,
                bg=None,
            )
        x += w
    return bar


def _palette_row(
    thumb: np.ndarray, colors: list[dict[str, Any]], title: str, width: int = 1400
) -> np.ndarray:
    """左に代表のコマ、右に色の帯と見出し。"""
    h = 200
    tw = max(1, round(thumb.shape[1] * h / thumb.shape[0]))
    tw = min(tw, width // 2)
    th = cv2.resize(thumb[:, :, :3], (tw, h), interpolation=cv2.INTER_AREA)
    row = np.full((h, width, 3), 250, np.uint8)
    row[:, :tw] = th
    core.draw_text(row, title, (tw + 12, 8), size=18, color=(30, 30, 30), bg=None)
    bar = _swatch_bar(colors, width - tw - 24, h - 48)
    row[40 : 40 + bar.shape[0], tw + 12 : tw + 12 + bar.shape[1]] = bar
    return row


@command(
    97,
    "palette",
    SRC,
    OUT,
    JSON,
    RANGE,
    arg("-n", "--colors", type=int, default=5, help="取り出す色の数（既定 5）"),
    arg(
        "--every",
        type=float,
        default=0.0,
        help="動画を何秒ごとに区切るか（既定 0 = カットで区切って場面ごと）",
    ),
    arg(
        "--merge",
        type=float,
        default=3.0,
        help="この色差（CIEDE2000）未満の色は 1 つにまとめる（既定 3。0 でまとめない）",
    ),
)
def palette(a: argparse.Namespace) -> None:
    """画像・場面の代表色を CIELAB の k 平均で取り出し、色見本の帯（PNG）と JSON に書く。"""
    info = core.probe(a.src)
    core.require(1 <= a.colors <= 16, "--colors は 1〜16 で指定してください")
    ext = ".png"
    out = Path(a.out) if a.out else core.default_out(a.src, "palette", ext)
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    size = core.scaled_size(info.width, info.height, None, _PAL_SIDE)
    if info.is_image:
        img = core.load_image(a.src)
        small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        colors = palette_of([small], a.colors, a.merge)
        thumb_h = min(img.shape[0], 720)
        thumb = cv2.resize(
            img,
            (max(1, round(img.shape[1] * thumb_h / img.shape[0])), thumb_h),
            interpolation=cv2.INTER_AREA,
        )
        bar = _swatch_bar(colors, thumb.shape[1], max(80, thumb_h // 6))
        core.save_image(
            out, np.vstack([thumb, np.full((6, thumb.shape[1], 3), 255, np.uint8), bar])
        )
        core.write_json(
            json_path,
            {
                "source": str(a.src),
                "params": {"colors": a.colors, "merge": a.merge},
                "colors": colors,
            },
        )
        core.log(
            f"代表色 {len(colors)} 色: "
            + "  ".join(f"{c['hex']} {c['share'] * 100:.0f}%" for c in colors)
        )
        core.log(f"保存: {out}")
        core.log(f"保存: {json_path}")
        return
    rng = _rng(a)
    fps = info.fps
    # 区間（調べた範囲の中のコマ番号で [始め, 終わり]）
    if a.every and a.every > 0:
        step = max(1, round(a.every * fps))
        n = rng["max_frames"] or round(
            (rng["duration"] or max(info.duration - rng["start"], 0)) * fps
        )
        segs = [(s, min(s + step, n) - 1) for s in range(0, max(n, 1), step)]
        how = f"{a.every:g} 秒ごと"
    else:
        from .scenes import detect_cuts

        cuts = detect_cuts(a.src, **rng)
        segs = cuts.scenes()
        how = "場面ごと（76 のカット検出で区切る）"
    # 区間ごとに等間隔に最大 12 コマを選ぶ（両端の 1 割は避ける。溶けるつなぎ目の色が混ざらないよう）
    want: dict[int, int] = {}
    for si, (s, e) in enumerate(segs):
        length = e - s + 1
        margin = int(length * 0.1) if length >= 10 else 0
        lo, hi = s + margin, e - margin
        for f in np.unique(
            np.linspace(lo, hi, min(_PAL_FRAMES, hi - lo + 1)).round().astype(int)
        ):
            want[int(f)] = si
    picked: dict[int, list[np.ndarray]] = {si: [] for si in range(len(segs))}
    last = max(want) if want else -1
    for i, f in enumerate(core.read_frames(a.src, size=size, **rng)):
        if i in want:
            picked[want[i]].append(f)
        if i >= last:
            break
    rows = []
    results = []
    all_frames: list[np.ndarray] = []
    all_weights: list[float] = []
    for si, (s, e) in enumerate(segs):
        frames = picked[si]
        if not frames:
            continue
        colors = palette_of(frames, a.colors, a.merge)
        t0, t1 = core.t_of(s, fps, rng["start"]), core.t_of(e + 1, fps, rng["start"])
        results.append(
            {"start": t0, "end": t1, "frames_used": len(frames), "colors": colors}
        )
        all_frames += frames
        all_weights += [(e - s + 1) / len(frames)] * len(frames)
        title = f"{t0:.2f}〜{t1:.2f} 秒"
        rows.append(_palette_row(frames[len(frames) // 2], colors, title))
        core.log(
            f"  {title}: "
            + "  ".join(f"{c['hex']} {c['share'] * 100:.0f}%" for c in colors)
        )
    core.require(bool(results), "色を数えるコマがありません")
    # 動画全体（区間の長さで重みをつけて画素を選ぶ）
    overall = palette_of(all_frames, a.colors, a.merge, weights=all_weights)
    mid = picked[len(segs) // 2] or next(v for v in picked.values() if v)
    rows.insert(0, _palette_row(mid[len(mid) // 2], overall, "全体"))
    sheet = np.vstack(
        [np.vstack([r, np.full((6, r.shape[1], 3), 255, np.uint8)]) for r in rows]
    )
    core.save_image(out, sheet)
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "params": {
                "colors": a.colors,
                "merge": a.merge,
                "every": a.every,
                "segments": how,
            },
            "overall": overall,
            "segments": results,
        },
    )
    core.log(
        f"{how}: {len(results)} 区間。全体: "
        + "  ".join(f"{c['hex']} {c['share'] * 100:.0f}%" for c in overall)
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 98 QR コード検出 ─────────────────────────────────────────────────────

_WECHAT_URL = (
    "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/"
)
_WECHAT = {
    n: models.ModelFile(_WECHAT_URL + n, "Apache-2.0", "WeChat QRCode")
    for n in ("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")
}


class QrReader:
    """QR コードを読む。method: auto（WeChat と OpenCV の 2 つを合わせる）/ wechat / aruco / opencv。

    retry=True なら、1 つも読めなかったコマを 2 倍に拡大（長辺 1280px 以下の時）・鮮鋭化して読み直す
    （小さい・少しぼけたコードが読める。合成画像の試験で 144 枚中 53 → 61 枚）。
    """

    def __init__(self, method: str = "auto", retry: bool = True) -> None:
        core.require(
            method in {"auto", "wechat", "aruco", "opencv"},
            f"知らない方法です: {method}（auto / wechat / aruco / opencv）",
        )
        self.method = method
        self.retry = retry
        self.wechat = self.aruco = self.std = None
        if method in {"auto", "wechat"}:
            p = {n: str(models.fetch_spec(m)) for n, m in _WECHAT.items()}
            self.wechat = cv2.wechat_qrcode.WeChatQRCode(
                p["detect.prototxt"],
                p["detect.caffemodel"],
                p["sr.prototxt"],
                p["sr.caffemodel"],
            )
        if method in {"auto", "aruco"}:
            self.aruco = cv2.QRCodeDetectorAruco()
        if method == "opencv":
            self.std = cv2.QRCodeDetector()

    def __call__(self, frame: np.ndarray) -> list[dict[str, Any]]:
        img = np.ascontiguousarray(frame[:, :, :3])
        found = self._read(img)
        if not found and self.retry:
            scale = 2.0 if max(img.shape[:2]) <= 1280 else 1.0
            big = (
                cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                if scale != 1.0
                else img
            )
            soft = cv2.GaussianBlur(big, (0, 0), scale)
            sharp = cv2.addWeighted(big, 2.0, soft, -1.0, 0)
            found = self._read(sharp)
            for f in found:
                f["points"] = f["points"] / scale
                f["by"] += "（拡大・鮮鋭化して読み直し）"
        return found

    def _read(self, img: np.ndarray) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if self.wechat is not None:
            texts, points = self.wechat.detectAndDecode(img)
            for t, p in zip(texts, points, strict=False):
                if t:
                    found.append(
                        {
                            "text": t,
                            "points": np.asarray(p, np.float64).reshape(4, 2),
                            "by": "wechat",
                        }
                    )
        for det, name in ((self.aruco, "aruco"), (self.std, "opencv")):
            if det is None:
                continue
            try:
                ok, texts, points, _ = det.detectAndDecodeMulti(img)
            except cv2.error:
                ok = False
            if ok and points is not None:
                for t, p in zip(texts, points, strict=False):
                    if t:
                        found.append(
                            {
                                "text": t,
                                "points": np.asarray(p, np.float64).reshape(4, 2),
                                "by": name,
                            }
                        )
        # 同じ物を 2 つの方法で読んだ時は 1 つにする（先に読んだ方を残す）
        out: list[dict[str, Any]] = []
        for f in found:
            c = f["points"].mean(axis=0)
            dup = False
            for g in out:
                side = max(np.ptp(g["points"][:, 0]), np.ptp(g["points"][:, 1]), 1.0)
                if (
                    g["text"] == f["text"]
                    and np.linalg.norm(g["points"].mean(axis=0) - c) < side / 2
                ):
                    dup = True
                    break
            if not dup:
                out.append(f)
        return out


@command(
    98,
    "qr",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=["auto", "wechat", "aruco", "opencv"],
        default="auto",
        help="読み方（既定 auto = WeChat QRCode の CNN と OpenCV の QRCodeDetectorAruco を両方使う）",
    ),
    arg(
        "--every",
        type=float,
        default=0.1,
        help="動画を何秒ごとに調べるか（既定 0.1。0 で全部のコマ）",
    ),
    arg(
        "--no-video",
        action="store_true",
        help="動画の時、枠を描いた MP4 を作らない（JSON と一覧だけ）",
    ),
)
def qr(a: argparse.Namespace) -> None:
    """QR コードを見つけて読み取り、中身・四隅・映っている時刻を JSON に、枠を描いた画像・動画を書く。"""
    info = core.probe(a.src)
    out, json_path, sheet_path = _paths(a, "qr", info)
    size = _size(info, a)
    sx, sy = info.width / size[0], info.height / size[1]
    reader = QrReader(a.method)
    keyed: dict[int, list[_Item]] = {}
    hits = []
    shots = _SheetBuffer()
    for s in _samples(a.src, info, a, a.every, size):
        codes = reader(s.frame)
        items = []
        for c in codes:
            pts = c["points"]
            hits.append(
                {
                    "t": s.t,
                    "text": c["text"],
                    "points": [[round(x * sx, 1), round(y * sy, 1)] for x, y in pts],
                    "by": c["by"],
                }
            )
            items.append(_Item(pts, c["text"][:40], _GREEN, c["text"]))
        keyed[s.i] = items
        if items and not info.is_image:
            vis = s.frame.copy()
            for it in items:
                _draw_poly(vis, it.geom, it.color, it.label)
            shots.add(vis, _fmt_t(s.t))
    # 同じ中身が続く区間にまとめる
    gap = max(0.5, 2.5 * a.every)
    segs: list[dict[str, Any]] = []
    open_by_text: dict[str, dict[str, Any]] = {}
    for h in hits:
        cur = open_by_text.get(h["text"])
        if cur is not None and h["t"] - cur["end"] <= gap:
            cur["end"] = h["t"]
            cur["hits"] += 1
            cur["track"].append({"t": h["t"], "points": h["points"]})
        else:
            cur = {
                "text": h["text"],
                "start": h["t"],
                "end": h["t"],
                "hits": 1,
                "track": [{"t": h["t"], "points": h["points"]}],
            }
            segs.append(cur)
            open_by_text[h["text"]] = cur
    step = _step(info, a.every)
    for sg in segs:
        sg["end"] = round(sg["end"] + (0 if info.is_image else step / info.fps), 4)
    data = {
        "source": str(a.src),
        "params": {"method": a.method, "every": a.every, "size": list(size)},
        "codes": [{k: v for k, v in sg.items()} for sg in segs],
    }
    core.write_json(json_path, data)
    texts = sorted({sg["text"] for sg in segs})
    core.log(f"読めた QR コード: {len(texts)} 種類")
    for sg in segs:
        span = (
            ""
            if info.is_image
            else f"  {sg['start']:.2f}〜{sg['end']:.2f} 秒（{sg['hits']} 回）"
        )
        core.log(f"  「{sg['text']}」{span}")
    if info.is_image:
        vis = next(core.read_frames(a.src, size=size))
        for it in keyed.get(0, []):
            _draw_poly(vis, it.geom, it.color, it.label)
        core.save_image(out, vis)
    else:
        if shots:
            core.save_image(sheet_path, shots.sheet())
            core.log(f"保存: {sheet_path}")
        if not a.no_video:
            _render_video(a.src, info, a, size, keyed, out, hold=step)
    if info.is_image or not a.no_video:
        core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 96 テキスト検出の部品（94 でも使う） ─────────────────────────────────

_PPOCR = {
    "cn": models.ModelFile(
        models.ZOO + "text_detection_ppocr/text_detection_cn_ppocrv3_2023may.onnx",
        "Apache-2.0",
        "PP-OCRv3 の文字検出（中国語の学習。日本語・英語の文字も見つかる）",
    ),
    "en": models.ModelFile(
        models.ZOO + "text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
        "Apache-2.0",
        "PP-OCRv3 の文字検出（英語）",
    ),
}
# PP-OCRv3（PaddleOCR）の前処理の平均（OpenCV Zoo の demo と同じ値）
_PPOCR_MEAN = (122.67891434, 116.66876762, 104.00698793)


def drop_face_boxes(
    boxes: list[tuple[np.ndarray, float]], faces: list[Any]
) -> list[tuple[np.ndarray, float]]:
    """顔（YuNet の枠を上下左右に 2 割広げた範囲）に半分以上が入る文字の枠を除く。PP-OCRv3 は顔や目を文字と
    見なすことがあるため（文字の無い story540 の 0.5 秒ごと 30 コマのうち 20 コマ → 除くと 3 コマ）。"""
    out = []
    for q, s in boxes:
        b = _poly_box(q)
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
        cover = 0.0
        for f in faces:
            x, y, w, h = f.box
            fx1, fy1, fx2, fy2 = x - 0.2 * w, y - 0.2 * h, x + 1.2 * w, y + 1.2 * h
            iw = max(0.0, min(b[2], fx2) - max(b[0], fx1))
            ih = max(0.0, min(b[3], fy2) - max(b[1], fy1))
            cover = max(cover, iw * ih / area)
        if cover < 0.5:
            out.append((q, s))
    return out


class TextDetector:
    """文字のある場所（四角形）を見つける。

    model='craft': EasyOCR の CRAFT（79 と同じ検出。正確だが遅い。96 の既定）。
    model='ppocr': PP-OCRv3 の DB（OpenCV の dnn。速い）。lang='cn'（漢字・かな・英字）か 'en'（英字）。
      顔・目を文字と見なすことがあるので、faces=True なら YuNet で見つけた顔に重なる枠を除く。
    返り値: [(4×2 の四隅, 確からしさ), ...]（入力の画像の座標）
    """

    def __init__(
        self,
        model: str = "craft",
        *,
        lang: str = "cn",
        threshold: float = 0.5,
        side: int = 960,
        faces: bool = False,
    ) -> None:
        core.require(
            model in {"ppocr", "craft"}, f"知らない方法です: {model}（ppocr / craft）"
        )
        self.model = model
        self.threshold = threshold
        self.side = side
        if model == "ppocr":
            core.require(lang in _PPOCR, f"知らない言語です: {lang}（cn / en）")
            net = cv2.dnn.readNet(str(models.fetch_spec(_PPOCR[lang])))
            self.net = cv2.dnn.TextDetectionModel_DB(net)
            self.net.setBinaryThreshold(0.3)
            self.net.setPolygonThreshold(threshold)
            self.net.setUnclipRatio(2.0)
            self.net.setMaxCandidates(400)
            self.size: tuple[int, int] | None = None
        else:
            self.reader = easyocr_reader()
        self.face_det = None
        if faces and model == "ppocr":
            from .detectors import FaceDetector

            self.face_det = FaceDetector(score=0.6)

    def __call__(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        img = np.ascontiguousarray(frame[:, :, :3])
        if self.model == "craft":
            return _craft_boxes(self.reader, img, self.threshold)
        out = self._ppocr(img)
        if self.face_det is not None and out:
            out = drop_face_boxes(out, self.face_det(img))
        return out

    def _ppocr(self, img: np.ndarray) -> list[tuple[np.ndarray, float]]:
        h, w = img.shape[:2]
        # 長辺を side px にして、縦横を 32 の倍数にそろえる（DB の網の決まり）
        f = self.side / max(h, w)
        iw = max(32, round(w * f / 32) * 32)
        ih = max(32, round(h * f / 32) * 32)
        if self.size != (iw, ih):
            self.net.setInputParams(1.0 / 255.0, (iw, ih), _PPOCR_MEAN)
            self.size = (iw, ih)
        resized = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
        polys, scores = self.net.detect(resized)
        out = []
        for p, s in zip(polys, scores, strict=False):
            pts = np.asarray(p, np.float64).reshape(4, 2) * [w / iw, h / ih]
            out.append((pts, float(s)))
        return out


_EASYOCR = {
    "craft_mlt_25k.pth": models.ModelFile(
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "MIT（CRAFT。clovaai/CRAFT-pytorch）",
        "EasyOCR の文字検出",
    ),
    "japanese_g2.pth": models.ModelFile(
        "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/japanese_g2.zip",
        "Apache-2.0（EasyOCR）",
        "EasyOCR の文字認識（日本語・英語）",
    ),
}
_READER: list[Any] = []


def easyocr_reader() -> Any:
    """EasyOCR 1.7.2（日本語＋英語）の Reader。重みは GitHub のリリースから取ってキャッシュに置き、EasyOCR 自身には
    取得させない（md5 は EasyOCR が読み込む時に照合する）。79・96 で同じ物を使う。"""
    if _READER:
        return _READER[0]
    import zipfile

    models.torch_setup()
    import easyocr

    d = models.CACHE / "easyocr"
    d.mkdir(parents=True, exist_ok=True)
    for name, spec in _EASYOCR.items():
        if not (d / name).exists():
            with zipfile.ZipFile(models.fetch_spec(spec)) as zf:
                zf.extract(name, d)
    try:
        reader = easyocr.Reader(
            ["ja", "en"],
            gpu=False,
            model_storage_directory=str(d),
            user_network_directory=str(d / "user_network"),
            download_enabled=False,
            verbose=False,
        )
    except FileNotFoundError as e:
        raise core.VfxError(
            f"EasyOCR の重みが壊れているか足りません（{d} を消して取り直す）: {e}"
        ) from e
    _READER.append(reader)
    return reader


def easyocr_detect(
    reader: Any,
    bgr: np.ndarray,
    *,
    mag: float = 1.0,
    text_threshold: float = 0.7,
    low_text: float = 0.4,
) -> tuple[list[Any], list[Any]]:
    """CRAFT で文字の場所を探す（EasyOCR の detect。CRAFT は RGB で学習しているので RGB で渡す）。"""
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr[:, :, :3]), cv2.COLOR_BGR2RGB)
    hl, fl = reader.detect(
        rgb, mag_ratio=mag, text_threshold=text_threshold, low_text=low_text
    )
    return hl[0], fl[0]


def _craft_boxes(
    reader: Any, img: np.ndarray, threshold: float
) -> list[tuple[np.ndarray, float]]:
    """CRAFT の枠を、TextDetector と同じ形（4×2 の四隅, 確からしさ）にする（CRAFT は枠ごとの確からしさを返さないので 1.0）。"""
    hl, fl = easyocr_detect(reader, img, text_threshold=max(0.3, threshold + 0.2))
    out = []
    for x1, x2, y1, y2 in hl:
        out.append(
            (np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64), 1.0)
        )
    for pts in fl:
        out.append((np.asarray(pts, np.float64).reshape(4, 2), 1.0))
    return out


def easyocr_read(
    reader: Any,
    bgr: np.ndarray,
    *,
    mag: float = 1.0,
    decoder: str = "greedy",
    text_threshold: float = 0.7,
    low_text: float = 0.4,
    both: bool = True,
) -> list[tuple[np.ndarray, str, float]]:
    """文字を読む: CRAFT（RGB）で場所を探し、認識は正しい灰色で行う（EasyOCR に numpy を渡すと、灰色を BGR として
    作る一方で CRAFT にはそのまま渡すため、自分で分けて呼ぶ）。返り値: [(4×2 の四隅, 文字, 確からしさ), ...]

    both=True なら白黒を反転した灰色でも読み、枠ごとに確からしさの高い方を使う（黒い縁の白字のテロップで、
    合成の試験の正解率が 14〜44px で 0.54〜0.92 → 0.86〜0.95。黒字は変わらない）。"""
    hl, fl = easyocr_detect(
        reader, bgr, mag=mag, text_threshold=text_threshold, low_text=low_text
    )
    if not hl and not fl:
        return []
    gray = cv2.cvtColor(np.ascontiguousarray(bgr[:, :, :3]), cv2.COLOR_BGR2GRAY)
    res = reader.recognize(gray, hl, fl, decoder=decoder)
    if both:
        inv = reader.recognize(255 - gray, hl, fl, decoder=decoder)
        merged = []
        for ra in res:
            best = ra
            for rb in inv:
                same = np.allclose(np.asarray(ra[0], float), np.asarray(rb[0], float))
                if same and rb[2] > best[2]:
                    best = rb
            merged.append(best)
        res = merged
    return [
        (np.asarray(b, np.float64).reshape(4, 2), str(t), float(c)) for b, t, c in res
    ]


# ── 94 顕著性マップ ──────────────────────────────────────────────────────

_SAL_SIDE = 320  # 顕著性を求める時の長辺（px）


def _norm01(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(m)


def _itti_n(m: np.ndarray) -> np.ndarray:
    """Itti の正規化 N(): 0〜1 にそろえ、(1 − ほかの山の平均)² を掛ける（山が 1 つだけ目立つ地図を強め、
    同じ高さの山が並ぶ地図を弱める。Itti, Koch & Niebur 1998）。"""
    m = _norm01(m)
    k = max(3, int(min(m.shape) * 0.08) | 1)
    dil = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    pk = np.sort(m[(m >= dil - 1e-6) & (m > 0.05)])[::-1]
    mbar = float(pk[1:].mean()) if len(pk) > 1 else 0.0
    return m * (1 - mbar) ** 2


def _color_rarity(small: np.ndarray) -> np.ndarray:
    """色の珍しさ（Cheng ほか 2011 の histogram contrast をもとにした）: 色を 12 段階×3 に分け、画面の多くを占める色
    （画素の 95%）との色差（Lab）を、その色の画素の多さで重みづけて足す。少なくて、ほかと違う色ほど高い。

    元の方法は少ない色を近い色に置き換えて数を減らすが、それでは小さな物の珍しい色が消える（試験で確かめた）ので、
    すべての色をそのまま比べる。"""
    q = (small[:, :, :3].astype(np.int32) * 12) // 256
    idx = q[..., 0] * 144 + q[..., 1] * 12 + q[..., 2]
    hist = np.bincount(idx.ravel(), minlength=12**3).astype(np.float64)
    used = np.nonzero(hist)[0]
    order = used[np.argsort(-hist[used])]
    cum = np.cumsum(hist[order]) / idx.size
    keep = order[: int(np.searchsorted(cum, 0.95)) + 1]

    def centers(bins: np.ndarray) -> np.ndarray:
        c = np.stack([bins // 144, (bins // 12) % 12, bins % 12], 1)
        return ((c + 0.5) * 256 / 12).clip(0, 255).astype(np.uint8)

    lab_keep = _to_lab(centers(keep)).astype(np.float64)
    freq = hist[keep] / hist[keep].sum()
    lab_used = _to_lab(centers(used)).astype(np.float64)
    dist = np.linalg.norm(lab_used[:, None] - lab_keep[None], axis=2)
    lut = np.zeros(12**3, np.float32)
    lut[used] = (dist * freq[None]).sum(axis=1)
    return lut[idx]


_GABOR: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _orientation(gray: np.ndarray) -> np.ndarray:
    """向きの目立ちやすさ: 4 方向の Gabor の強さ（対になる 2 つのフィルターの二乗和の平方根）を、それぞれ N() で
    正規化して足す（周りと違う向きの線が目立つ。Itti ほか 1998 の向きの通り道）。"""
    total = np.zeros_like(gray)
    for th in (0, 45, 90, 135):
        if th not in _GABOR:
            _GABOR[th] = (
                cv2.getGaborKernel((17, 17), 3.0, np.radians(th), 8.0, 0.5, 0),
                cv2.getGaborKernel((17, 17), 3.0, np.radians(th), 8.0, 0.5, np.pi / 2),
            )
        ke, ko = _GABOR[th]
        e = np.sqrt(cv2.filter2D(gray, -1, ke) ** 2 + cv2.filter2D(gray, -1, ko) ** 2)
        total += _itti_n(cv2.GaussianBlur(e, (0, 0), 3))
    return total


class SaliencyModel:
    """目が行く場所の推定（0〜1 の地図。長辺 320px）。学習した視線のモデルではなく、次の和:

    - 低い階層の目立ちやすさ（method='full'）: OpenCV の spectral residual（Hou & Zhang 2007）・色の珍しさ
      （Cheng ほか 2011）・向き（Gabor）の 3 つを、Itti の N() で正規化して平均する。'spectral'・'fine'
      （OpenCV の fine grained。Montabone & Soto 2010）はそれぞれ 1 つだけ
    - 顔（YuNet。目の高さを中心に重みを足す）・文字（PP-OCRv3 の文字検出）・動き（動画のみ。DIS の動きから画面全体の
      動きを引いた、周りと違う動き）を、それぞれの重みで足す
    - 画面の中央を重くする（center）。動画は前のコマと半々に混ぜてちらつきを抑える（カットでは混ぜない）
    """

    def __init__(
        self,
        method: str = "full",
        *,
        faces: float = 0.6,
        text: float = 0.4,
        motion: float = 0.4,
        center: float = 0.3,
    ) -> None:
        core.require(
            method in {"full", "spectral", "fine"},
            f"知らない方法です: {method}（full / spectral / fine）",
        )
        self.method = method
        self.w_faces, self.w_text, self.w_motion, self.center = (
            faces,
            text,
            motion,
            center,
        )
        self.sr = cv2.saliency.StaticSaliencySpectralResidual_create()
        self.fg = cv2.saliency.StaticSaliencyFineGrained_create()
        self.face_det = None
        if (
            faces > 0 or text > 0
        ):  # 文字の重みでも、顔に重なる文字の枠を除くのに顔を使う
            from .detectors import FaceDetector

            self.face_det = FaceDetector(score=0.6)
        self.text_det = TextDetector("ppocr") if text > 0 else None
        self.reset()

    def reset(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.ema: np.ndarray | None = None
        self.text_cache: list[tuple[np.ndarray, float]] | None = None
        self.text_age = 0
        self.parts: dict[str, np.ndarray] = {}
        self.faces: list[Any] = []

    def __call__(self, frame: np.ndarray, *, video: bool = False) -> np.ndarray:
        h, w = frame.shape[:2]
        f = _SAL_SIDE / max(h, w)
        sw, sh = max(1, round(w * f)), max(1, round(h * f))
        small = cv2.resize(frame[:, :, :3], (sw, sh), interpolation=cv2.INTER_AREA)
        sigma = 0.025 * max(sw, sh)
        if self.method == "full":
            _, sr = self.sr.computeSaliency(small)
            sr = cv2.GaussianBlur(sr.astype(np.float32), (0, 0), sigma)
            hc = cv2.GaussianBlur(_color_rarity(small), (0, 0), sigma / 2)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            ori = cv2.GaussianBlur(_orientation(gray), (0, 0), sigma / 2)
            low = _norm01(np.mean([_itti_n(sr), _itti_n(hc), _itti_n(ori)], axis=0))
        else:
            det = self.sr if self.method == "spectral" else self.fg
            _, m = det.computeSaliency(small)
            low = _norm01(cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma))
        total = low.copy()
        self.parts = {"low": low}
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        if self.face_det is not None:
            self.faces = self.face_det(frame)
        if self.w_faces > 0:
            fm = np.zeros((sh, sw), np.float32)
            for face in self.faces:
                x, y, fw, fh = face.box
                cx, cy = (x + fw / 2) * f, (y + 0.42 * fh) * f
                sx, sy = max(2.0, 0.55 * fw * f), max(2.0, 0.6 * fh * f)
                g = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
                fm = np.maximum(fm, min(1.0, face.score) * g)
            total += self.w_faces * fm
            self.parts["faces"] = fm
        if self.text_det is not None:
            if self.text_cache is None or not video or self.text_age >= 4:
                # PP-OCRv3 は顔・目も文字と見なすことがあるので、見つけた顔に重なる枠は除く（顔は顔の重みで足す）
                self.text_cache = drop_face_boxes(self.text_det(frame), self.faces)
                self.text_age = 0
            self.text_age += 1
            tm = np.zeros((sh, sw), np.float32)
            for pts, _s in self.text_cache:
                cv2.fillPoly(
                    tm, [np.round(pts * f).astype(np.int32).reshape(-1, 1, 2)], 1.0
                )
            tm = cv2.GaussianBlur(tm, (0, 0), sigma / 2) if tm.any() else tm
            total += self.w_text * tm
            self.parts["text"] = tm
        cut = False
        if video and self.w_motion > 0:
            from .ai import optical_flow

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mm = np.zeros((sh, sw), np.float32)
            if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
                cut = (
                    float(np.abs(gray.astype(np.float32) - self.prev_gray).mean()) > 40
                )
                if not cut:
                    flow = optical_flow(self.prev_gray, gray, "dis")
                    med = np.median(flow.reshape(-1, 2), axis=0)
                    mag = np.linalg.norm(flow - med, axis=2)
                    mm = np.clip(mag / max(float(np.percentile(mag, 99)), 1.5), 0, 1)
                    mm = cv2.GaussianBlur(mm.astype(np.float32), (0, 0), sigma)
            self.prev_gray = gray
            total += self.w_motion * mm
            self.parts["motion"] = mm
        if self.center > 0:
            g = np.exp(
                -(
                    ((xx - sw / 2) / (0.35 * sw)) ** 2
                    + ((yy - sh / 2) / (0.35 * sh)) ** 2
                )
                / 2
            )
            total *= (1 - self.center) + self.center * g
        total = _norm01(total)
        if video:
            if self.ema is None or cut or self.ema.shape != total.shape:
                self.ema = total
            else:
                self.ema = 0.5 * self.ema + 0.5 * total
            total = _norm01(self.ema)
        return total


def saliency_peaks(
    sal: np.ndarray, n: int = 3, min_value: float = 0.3
) -> list[tuple[int, int, float]]:
    """顕著性の地図の山（局所的な最大）を高い順に n 個。(x, y, 値)。山どうしは短い辺の 12% 以上離す。"""
    k = max(3, int(min(sal.shape) * 0.12) | 1)
    dil = cv2.dilate(sal, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ys, xs = np.nonzero((sal >= dil - 1e-6) & (sal >= min_value))
    order = np.argsort(-sal[ys, xs])
    peaks: list[tuple[int, int, float]] = []
    for i in order:
        x, y, v = int(xs[i]), int(ys[i]), float(sal[ys[i], xs[i]])
        if all((x - px) ** 2 + (y - py) ** 2 >= (k / 2) ** 2 for px, py, _ in peaks):
            peaks.append((x, y, v))
        if len(peaks) >= n:
            break
    return peaks


def _grid3(sal: np.ndarray) -> list[list[float]]:
    h, w = sal.shape
    return [
        [
            round(
                float(
                    sal[
                        r * h // 3 : (r + 1) * h // 3, c * w // 3 : (c + 1) * w // 3
                    ].mean()
                ),
                3,
            )
            for c in range(3)
        ]
        for r in range(3)
    ]


def _sal_overlay(
    frame: np.ndarray, sal: np.ndarray, peaks: list[tuple[int, int, float]]
) -> np.ndarray:
    """熱の色（TURBO）を、目立つ所ほど濃く重ね（0.15 未満は重ねない）、山に番号を付ける。"""
    h, w = frame.shape[:2]
    big = np.clip(cv2.resize(sal, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)
    heat = core.heatmap(big, lo=0.0, hi=1.0)
    out = core.overlay(frame, heat, np.clip((big - 0.15) / 0.85, 0, 1) ** 0.9 * 0.65)
    f = w / sal.shape[1]
    r = max(6, round(min(h, w) * 0.025))
    for n, (x, y, _v) in enumerate(peaks, 1):
        c = (round(x * f), round(y * f))
        cv2.circle(out, c, r, (255, 255, 255), max(2, r // 4), cv2.LINE_AA)
        _put_tag(out, str(n), c[0] + r, c[1] - r, (30, 30, 30), size=max(13, r * 2))
    return out


@command(
    94,
    "saliency",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--method",
        choices=["full", "spectral", "fine"],
        default="full",
        help="低い階層の目立ちやすさの求め方（既定 full = spectral residual・色の珍しさ・向きを Itti の方法で合わせる。"
        "spectral・fine は OpenCV の 1 つだけ）",
    ),
    arg(
        "--faces",
        type=float,
        default=0.6,
        help="顔の重み（既定 0.6。0 で顔を使わない）",
    ),
    arg(
        "--text",
        type=float,
        default=0.4,
        help="文字の重み（既定 0.4。0 で文字を使わない）",
    ),
    arg(
        "--motion",
        type=float,
        default=0.4,
        help="動きの重み（動画のみ。既定 0.4。0 で使わない）",
    ),
    arg(
        "--center",
        type=float,
        default=0.3,
        help="画面の中央を重くする度合い（0〜1。既定 0.3）",
    ),
    arg(
        "--every",
        type=float,
        default=0.0,
        help="動画を何秒ごとに調べるか（既定 0 = 全部のコマ）",
    ),
    arg(
        "--map",
        action="store_true",
        help="重ね表示でなく、白黒の地図（白 = 目が行く）を書き出す（合成のマスクなどに使う）",
    ),
)
def saliency(a: argparse.Namespace) -> None:
    """目が行く場所を推定し、熱の色の重ね表示（画像・MP4）と、山の位置・3×3 の区画ごとの値を JSON に書く。"""
    info = core.probe(a.src)
    out, json_path, sheet_path = _paths(a, "saliency", info)
    size = _size(info, a)
    model = SaliencyModel(
        a.method, faces=a.faces, text=a.text, motion=a.motion, center=a.center
    )
    sx = info.width / size[0]
    entries = []
    sheet_imgs, sheet_labels = [], []
    acc = None
    writer = None
    if not info.is_image:
        rng = _rng(a)
        writer = core.VideoWriter(
            out,
            info.fps,
            audio_from=a.src,
            audio_start=rng["start"],
            audio_duration=rng["duration"],
        )
    step = _step(info, a.every)
    total_frames = (a.max_frames or 0) or round(
        (a.duration or max(info.duration - (a.start or 0.0), 0)) * info.fps
    )
    # 一覧に載せるコマ（調べるコマの中から等間隔に 12 枚）
    sheet_at = {
        (int(x) // step) * step
        for x in np.linspace(0, max(total_frames - 1, 0), 12).round()
    }
    sal: np.ndarray | None = None
    peaks: list[tuple[int, int, float]] = []
    for s in _samples(a.src, info, a, 0.0, size):
        fresh = s.i % step == 0 or sal is None
        if fresh:
            sal = model(s.frame, video=not info.is_image)
            to_src = sx * s.frame.shape[1] / sal.shape[1]  # 地図の座標 → 元の画素
            peaks = saliency_peaks(sal)
            acc = sal.copy() if acc is None else acc + sal
            entries.append(
                {
                    "t": s.t,
                    "peaks": [
                        [round(x * to_src), round(y * to_src), round(v, 3)]
                        for x, y, v in peaks
                    ],
                    "grid3x3": _grid3(sal),
                    "faces": len(model.faces),
                    "text_boxes": len(model.text_cache or []),
                }
            )
        assert sal is not None
        # 調べないコマには、直前に調べた地図を今のコマに重ねる
        if a.map:
            vis = cv2.cvtColor(
                (cv2.resize(sal, (s.frame.shape[1], s.frame.shape[0])) * 255).astype(
                    np.uint8
                ),
                cv2.COLOR_GRAY2BGR,
            )
        else:
            vis = _sal_overlay(s.frame, sal, peaks)
        if writer is not None:
            writer.write(vis)
            if fresh and s.i in sheet_at:
                sheet_imgs.append(vis)
                sheet_labels.append(_fmt_t(s.t))
        else:
            core.save_image(
                out, core.side_by_side(s.frame, vis, ("元", "顕著性（目が行く所）"))
            )
    if writer is not None:
        writer.close()
    core.require(bool(entries), "調べるコマがありません")
    mean_map = _norm01(acc / len(entries))
    to_src = info.width / mean_map.shape[1]
    summary = {
        "peaks": [
            [round(x * to_src), round(y * to_src), round(v, 3)]
            for x, y, v in saliency_peaks(mean_map)
        ],
        "grid3x3": _grid3(mean_map),
    }
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "params": {
                "method": a.method,
                "faces": a.faces,
                "text": a.text,
                "motion": a.motion,
                "center": a.center,
                "every": a.every,
                "size": list(size),
            },
            "note": "peaks は目が行きやすい所の [x, y, 値]（元の画素の座標。値は 0〜1）。grid3x3 は画面を 3×3 に分けた"
            "区画ごとの平均（上の行から）。値が小さい区画は、文字を置いても主役を隠しにくい",
            "summary": summary,
            "frames": entries,
        },
    )
    if not info.is_image and sheet_imgs:
        core.save_image(sheet_path, _sheet(sheet_imgs, sheet_labels, cols=6))
        core.log(f"保存: {sheet_path}")
    p = summary["peaks"]
    core.log(
        f"調べたコマ: {len(entries)}。いちばん目が行く所（平均）: "
        + ("、".join(f"({x}, {y})" for x, y, _ in p) if p else "なし")
    )
    g = summary["grid3x3"]
    quiet = min(
        ((r, c) for r in range(3) for c in range(3)), key=lambda rc: g[rc[0]][rc[1]]
    )
    names = (("上", "中", "下"), ("左", "中", "右"))
    core.log(
        f"いちばん目が行きにくい区画: {names[0][quiet[0]]}・{names[1][quiet[1]]}（{g[quiet[0]][quiet[1]]:.2f}）"
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 言葉で探す（Grounding DINO。78・90 の --query） ────────────────────────


def _phrases(query: str) -> list[str]:
    """「.」「,」「、」で区切った言葉の並び（英語）。日本語は COCO の 80 種類の名前だけ英語に直す。"""
    from .detectors import COCO_JA

    ja_to_en = {v: k for k, v in COCO_JA.items()}
    out = []
    for p in query.replace("、", ",").replace(".", ",").split(","):
        p = p.strip().lower()
        if not p:
            continue
        if not p.isascii():
            core.require(
                p in ja_to_en,
                f"--query は英語で指定してください（「{p}」を英語に直して渡す。例: ビューラー → eyelash curler）。"
                "Grounding DINO の言葉の部分は英語だけで学習しています",
            )
            p = ja_to_en[p]
        for art in ("a ", "an ", "the "):
            p = p.removeprefix(art)
        out.append(p)
    core.require(bool(out), "--query に探す物の名前を入れてください")
    return out


class GroundingDino:
    """言葉で物を探す（Grounding DINO tiny。IDEA-Research/grounding-dino-tiny、Apache-2.0。英語の言葉だけ）。"""

    def __init__(
        self, box_threshold: float = 0.35, text_threshold: float = 0.25
    ) -> None:
        self.torch = models.torch_setup()
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        repo = models.hf_repo("gdino")
        self.proc = AutoProcessor.from_pretrained(repo)
        self.net = AutoModelForZeroShotObjectDetection.from_pretrained(repo).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def __call__(self, frame: np.ndarray, phrases: list[str]) -> list[dict[str, Any]]:
        from PIL import Image

        h, w = frame.shape[:2]
        # 短い辺は 800px（モデルの既定）まで。小さな入力は拡大しない（速さのため。結果はほぼ同じ）
        short = min(800, min(h, w))
        self.proc.image_processor.size = {
            "shortest_edge": short,
            "longest_edge": round(short * 1333 / 800),
        }
        img = Image.fromarray(cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB))
        text = ". ".join(phrases) + "."
        inputs = self.proc(images=img, text=text, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.net(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]
        dets = []
        # 何も見つからない時、transformers 5.2 は text_labels に [''] を返す（枠は 0 個）ので、長さをそろえない
        for b, s, lab in zip(
            res["boxes"].tolist(),
            res["scores"].tolist(),
            res["text_labels"],
            strict=False,
        ):
            label = _closest_phrase(lab, phrases)
            if label:
                x1, y1, x2, y2 = (float(v) for v in b)
                dets.append(
                    {
                        "box": (max(0.0, x1), max(0.0, y1), min(w, x2), min(h, y2)),
                        "score": float(s),
                        "label": label,
                    }
                )
        return _nms(dets, 0.5, same_label=True) if dets else dets


def _closest_phrase(label: str, phrases: list[str]) -> str:
    """モデルが返す名前（言葉の一部や、2 つの言葉が混ざった物）を、渡した言葉のどれかに戻す。"""
    words = set(label.lower().split())
    if not words:
        return ""
    best = max(phrases, key=lambda p: len(words & set(p.split())) / len(set(p.split())))
    return best if words & set(best.split()) else ""


def _nms(
    dets: list[dict[str, Any]], iou: float, same_label: bool
) -> list[dict[str, Any]]:
    """重なった枠を、確からしさの高い方だけ残す（same_label=True なら同じ名前どうしだけ比べる）。"""
    keep: list[dict[str, Any]] = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(
            (same_label and k["label"] != d["label"])
            or _box_iou(k["box"], d["box"]) < iou
            for k in keep
        ):
            keep.append(d)
    return keep


def _summarize(entries: list[dict[str, Any]], key: str = "objects") -> dict[str, Any]:
    """名前ごとに、見えたコマの数・同時に映った最大の数・最初と最後の時刻をまとめる。"""
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        counts: dict[str, int] = {}
        for o in e[key]:
            counts[o["label"]] = counts.get(o["label"], 0) + 1
        for lab, n in counts.items():
            s = out.setdefault(
                lab, {"frames": 0, "max_count": 0, "first": e["t"], "last": e["t"]}
            )
            s["frames"] += 1
            s["max_count"] = max(s["max_count"], n)
            s["last"] = e["t"]
    return out


# ── 78 物体検出 ──────────────────────────────────────────────────────────


@command(
    78,
    "objects",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--model",
        choices=["accurate", "fast"],
        default="accurate",
        help="accurate = Faster R-CNN v2（精度重視。既定）、fast = EfficientDet-Lite2（速さ重視）。どちらも COCO の 80 種類",
    ),
    arg(
        "--score",
        type=float,
        help="残す確からしさの下限（0〜1。既定: 80 種類 0.5、--query 0.45。小さな物を拾う時は下げる）",
    ),
    arg(
        "--classes",
        help="種類で絞る（英語か日本語の名前を「,」区切り。例: person,cup や 人,カップ）",
    ),
    arg(
        "-q",
        "--query",
        help="80 種類にない物を言葉で探す（英語。「,」区切りで複数。例: eyelash curler, mirror）。"
        "Grounding DINO tiny を使う（--model・--classes は使わない）",
    ),
    arg(
        "--every",
        type=float,
        help="動画を何秒ごとに調べるか（既定: accurate 0.2・fast 0.1・--query 0.5。間のコマは位置を直線で補う）",
    ),
    arg(
        "--no-video",
        action="store_true",
        help="動画の時、枠を描いた MP4 を作らない（JSON と一覧だけ）",
    ),
)
def objects(a: argparse.Namespace) -> None:
    """映っている物を見つけ、枠と日本語の名前を描いた画像・MP4 と、位置の JSON を書く。"""
    from .detectors import COCO_JA, ObjectDetector, color_for

    info = core.probe(a.src)
    out, json_path, sheet_path = _paths(a, "objects", info)
    size = _size(info, a)
    sx, sy = info.width / size[0], info.height / size[1]
    if a.query:
        phrases = _phrases(a.query)
        # 0.45: story540 で「eyelash curler」が別の場面の目元に 0.35〜0.41 で出た（正しい物は 0.53 以上）
        score = a.score if a.score is not None else 0.45
        gd = GroundingDino(box_threshold=score)
        every = a.every if a.every is not None else 0.5

        def detect(frame: np.ndarray) -> list[dict[str, Any]]:
            return gd(frame, phrases)

        palette = {p: color_for(i) for i, p in enumerate(phrases)}
        what = "言葉で探す（Grounding DINO tiny）: " + ", ".join(phrases)
    else:
        classes = None
        if a.classes:
            ja_to_en = {v: k for k, v in COCO_JA.items()}
            classes = []
            for c in a.classes.replace("、", ",").split(","):
                c = c.strip()
                if not c:
                    continue
                en = ja_to_en.get(c, c.lower())
                core.require(
                    en in COCO_JA,
                    f"COCO の 80 種類にない名前です: {c}（80 種類にない物は --query で探す）",
                )
                classes.append(en)
        score = a.score if a.score is not None else 0.5
        det = ObjectDetector(a.model, score, classes)
        every = (
            a.every if a.every is not None else (0.2 if a.model == "accurate" else 0.1)
        )

        def detect(frame: np.ndarray) -> list[dict[str, Any]]:
            dets = [
                {"box": d.box, "score": d.score, "label": d.label, "ja": d.ja}
                for d in det(frame)
            ]
            # 検出器は種類ごとにしか重なりを除かないので、ほぼ同じ枠に別の名前が付いた物は、高い方だけ残す
            # （試験: コーギーに「犬 0.99」と「猫 0.54」が重なって出た）
            return _nms(dets, 0.7, same_label=False)

        names = list(COCO_JA)
        palette = {n: color_for(i) for i, n in enumerate(names)}
        what = f"COCO の 80 種類（{a.model}）" + (
            f"・絞り込み {classes}" if classes else ""
        )
    core.log(f"{what}。{'画像' if info.is_image else f'{every:g} 秒ごとに調べる'}")
    entries = []
    keyed: dict[int, list[_Item]] = {}
    shots = _SheetBuffer()
    for s in _samples(a.src, info, a, every, size):
        dets = detect(s.frame)
        items = []
        objs = []
        for d in dets:
            name = d.get("ja", d["label"])
            x1, y1, x2, y2 = d["box"]
            objs.append(
                {
                    "label": d["label"],
                    "ja": name,
                    "score": round(d["score"], 3),
                    "box": [
                        round(x1 * sx),
                        round(y1 * sy),
                        round((x2 - x1) * sx),
                        round((y2 - y1) * sy),
                    ],
                }
            )
            color = palette.get(d["label"], _YELLOW)
            items.append(
                _Item(
                    np.array(d["box"], np.float64),
                    f"{name} {d['score']:.2f}",
                    color,
                    d["label"],
                )
            )
        entries.append({"t": s.t, "objects": objs})
        keyed[s.i] = items
        if not info.is_image:
            vis = s.frame.copy()
            for it in items:
                _draw_box(vis, it.box, it.color, it.label)
            shots.add(vis, f"{_fmt_t(s.t)} {len(items)} 個")
    summary = _summarize(entries)
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "params": {
                "model": "grounding-dino-tiny" if a.query else a.model,
                "query": a.query,
                "classes": a.classes,
                "score": score,
                "every": every,
                "size": list(size),
            },
            "note": "box は [x, y, 幅, 高さ]（元の画素の座標）。t は秒",
            "summary": summary,
            "frames": entries,
        },
    )
    core.log(
        f"見つけた物（名前: 映ったコマの数・同時に映った最大の数）: {len(summary)} 種類"
    )
    for lab, s in sorted(summary.items(), key=lambda kv: -kv[1]["frames"]):
        ja = (
            lab if a.query else COCO_JA.get(lab, lab)
        )  # 言葉で探した時は渡した言葉のまま
        span = "" if info.is_image else f"、{s['first']:.2f}〜{s['last']:.2f} 秒"
        core.log(f"  {ja}: {s['frames']} コマ・最大 {s['max_count']} 個{span}")
    if info.is_image:
        vis = next(core.read_frames(a.src, size=size))
        for it in keyed.get(0, []):
            _draw_box(vis, it.box, it.color, it.label)
        core.save_image(out, vis)
    else:
        if shots:
            core.save_image(sheet_path, shots.sheet())
            core.log(f"保存: {sheet_path}")
        if not a.no_video:
            _render_video(a.src, info, a, size, keyed, out, hold=_step(info, every))
    if info.is_image or not a.no_video:
        core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 90 ロゴ検出 ──────────────────────────────────────────────────────────


def _quad_ok(quad: np.ndarray, shape: tuple[int, ...]) -> bool:
    """ホモグラフィーで写した四隅が、ありえる形か（凸・小さすぎない・角が 20〜160 度・向かい合う辺の比が 5 倍以内）。"""
    q = np.asarray(quad, np.float64).reshape(4, 2)
    if not np.all(np.isfinite(q)):
        return False
    q32 = q.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(q32):
        return False
    area = abs(cv2.contourArea(q32))
    if area < 144 or area > 4 * shape[0] * shape[1]:
        return False
    for i in range(4):
        v1, v2 = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        c = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        if not 20 <= np.degrees(np.arccos(np.clip(c, -1, 1))) <= 160:
            return False
    sides = [float(np.linalg.norm(q[i] - q[(i + 1) % 4])) for i in range(4)]
    for a, b in ((sides[0], sides[2]), (sides[1], sides[3])):
        if max(a, b) > 5 * max(min(a, b), 1e-6):
            return False
    return True


def _rootsift(des: np.ndarray) -> np.ndarray:
    """RootSIFT（Arandjelović & Zisserman 2012）: SIFT の記述子を L1 で正規化して平方根をとる（照合が安定する）。"""
    des = des / (np.abs(des).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(des).astype(np.float32)


class LogoFinder:
    """--ref のマーク（画像）を画面の中から探す。

    method='features': SIFT（RootSIFT）の特徴点を「画面 → マーク」の向きで照合し（同じマークが 2 つ映っていても比の
    試験で消えないように。比は 0.85）、USAC MAGSAC でホモグラフィーを求める。見つけた所の照合を除いて繰り返し、
    最大 max_count 個。長辺 1000px 以下の画面は 2 倍に拡大してから特徴点をとる（小さなマークのため）。見つけた所を
    マークの形に戻してマークとの相関（明るさの正規化相互相関）を測り、min_score（既定 0.3）未満は捨てる。
    method='template': 大きさを 24 段階に変えた色の型の照合（BGR の TM_CCOEFF_NORMED。既定 0.82。回転・遠近には
    弱い）。模様が少なく特徴点が取れないマーク向け。auto は特徴点が 30 個以上なら features、少なければ template。
    """

    def __init__(
        self,
        ref_path: str,
        *,
        method: str = "auto",
        min_score: float | None = None,
        max_count: int = 5,
        min_inliers: int = 10,
    ) -> None:
        img = core.load_image(ref_path, alpha=True)
        h, w = img.shape[:2]
        f = (
            800 / max(h, w)
            if max(h, w) > 800
            else (240 / max(h, w) if max(h, w) < 240 else 1.0)
        )
        if f != 1.0:
            interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=interp)
        self.bgr = np.ascontiguousarray(img[:, :, :3])
        self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        self.mask = np.where(img[:, :, 3] > 127, 255, 0).astype(np.uint8)
        self.has_alpha = bool((img[:, :, 3] < 250).any())
        self.size = (img.shape[1], img.shape[0])
        self.sift = cv2.SIFT_create(nfeatures=4000)
        kmask = (
            cv2.erode(self.mask, np.ones((5, 5), np.uint8)) if self.has_alpha else None
        )
        self.kp, des = self.sift.detectAndCompute(self.gray, kmask)
        n = 0 if des is None else len(self.kp)
        self.des = _rootsift(des) if n else des
        self.method = (
            method if method != "auto" else ("features" if n >= 30 else "template")
        )
        if self.method == "features":
            core.require(
                n >= 8,
                f"マークの画像から特徴点が {n} 個しか取れません（模様の少ないマーク）。--method template を使ってください",
            )
            self.matcher = cv2.FlannBasedMatcher(
                {"algorithm": 1, "trees": 5}, {"checks": 64}
            )
        self.min_score = (
            min_score
            if min_score is not None
            else (0.3 if self.method == "features" else 0.82)
        )
        self.max_count = max_count
        self.min_inliers = min_inliers
        self.n_keypoints = n

    def __call__(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.method == "features":
            return self._features(core.to_gray(frame))
        return self._template(np.ascontiguousarray(frame[:, :, :3]))

    def _verify(self, gray: np.ndarray, hm: np.ndarray, quad: np.ndarray) -> float:
        """見つけた所をマークの形に戻し、マークとの正規化相互相関（マスクの中）を、小さい方の解像度で測る。"""
        w, h = self.size
        back = cv2.warpPerspective(
            gray, hm, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )
        ref, mask = self.gray, self.mask
        s = math.sqrt(
            abs(cv2.contourArea(quad.astype(np.float32).reshape(-1, 1, 2))) / (w * h)
        )
        if s < 1:
            sw, sh = max(8, round(w * s)), max(8, round(h * s))
            back = cv2.resize(back, (sw, sh), interpolation=cv2.INTER_AREA)
            ref = cv2.resize(ref, (sw, sh), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        m = mask > 0
        a = back[m].astype(np.float64)
        b = ref[m].astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        den = math.sqrt(float((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / den) if den > 0 else 0.0

    def _features(self, gray: np.ndarray) -> list[dict[str, Any]]:
        # 小さな画面は 2 倍にしてから特徴点をとる（試験: 見つけた数 100 個中 59 → 84。誤検出は 0 のまま）
        up = 2.0 if max(gray.shape[:2]) <= 1000 else 1.0
        big = (
            cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            if up != 1.0
            else gray
        )
        kp, des = self.sift.detectAndCompute(big, None)
        if des is None or len(kp) < self.min_inliers:
            return []
        pairs = self.matcher.knnMatch(_rootsift(des), self.des, k=2)
        good = [
            p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.85 * p[1].distance
        ]
        if len(good) < self.min_inliers:
            return []
        src = np.float32([self.kp[m.trainIdx].pt for m in good])
        dst = np.float32([kp[m.queryIdx].pt for m in good]) / up
        w, h = self.size
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        alive = np.ones(len(good), bool)
        found: list[dict[str, Any]] = []
        for _ in range(self.max_count * 3):
            idx = np.nonzero(alive)[0]
            if len(found) >= self.max_count or len(idx) < self.min_inliers:
                break
            hm, inl = cv2.findHomography(src[idx], dst[idx], cv2.USAC_MAGSAC, 5.0)
            if hm is None or inl is None:
                break
            inl = inl.ravel().astype(bool)
            if inl.sum() < self.min_inliers:
                break
            quad = cv2.perspectiveTransform(corners, hm).reshape(4, 2)
            if not _quad_ok(quad, gray.shape):
                alive[idx[inl]] = False
                continue
            q32 = quad.astype(np.float32).reshape(-1, 1, 2)
            inside = np.array(
                [
                    cv2.pointPolygonTest(q32, (float(x), float(y)), True) >= -3
                    for x, y in dst[idx]
                ]
            )
            alive[idx[inside | inl]] = False
            score = self._verify(gray, hm, quad)
            if score >= self.min_score:
                found.append(
                    {
                        "quad": quad,
                        "score": score,
                        "inliers": int(inl.sum()),
                        "method": "features",
                    }
                )
        return found

    def _template(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        """色の型の照合（灰色だと、模様の少ないマークでは別の物と見分けられなかった: 正解 0.88 以上・誤り 0.87 まで。
        色なら正解 0.85 以上・誤り 0.79 まで）。"""
        gh, gw = bgr.shape[:2]
        w0, h0 = self.size
        lo = max(12 / min(w0, h0), 0.02)
        hi = min(gw / w0, gh / h0, 3.0)
        cands = []
        for s in np.geomspace(lo, hi, 24) if hi > lo else []:
            tw, th = round(w0 * s), round(h0 * s)
            if tw < 12 or th < 12 or tw > gw or th > gh:
                continue
            tpl = cv2.resize(self.bgr, (tw, th), interpolation=cv2.INTER_AREA)
            m = None
            if self.has_alpha:
                m1 = cv2.resize(self.mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                m = cv2.merge([m1, m1, m1])
            r = cv2.matchTemplate(bgr, tpl, cv2.TM_CCOEFF_NORMED, mask=m)
            r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
            k = max(3, int(min(tw, th) * 0.5) | 1)
            peak = (r >= cv2.dilate(r, np.ones((k, k), np.uint8)) - 1e-6) & (
                r >= self.min_score
            )
            for y, x in zip(*np.nonzero(peak), strict=True):
                cands.append(
                    {
                        "box": (float(x), float(y), float(x + tw), float(y + th)),
                        "score": float(r[y, x]),
                        "label": "mark",
                    }
                )
        out = []
        for d in _nms(cands, 0.3, same_label=False)[: self.max_count]:
            x1, y1, x2, y2 = d["box"]
            quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)
            out.append(
                {"quad": quad, "score": d["score"], "inliers": 0, "method": "template"}
            )
        return out


@command(
    90,
    "logos",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--ref",
        help="探すマークの画像（PNG・JPG。透過 PNG なら見える所だけで照合）。省略時は --query の言葉で探す",
    ),
    arg(
        "-q",
        "--query",
        default="logo",
        help="--ref が無い時に言葉で探す（英語。既定 logo。Grounding DINO tiny）。マークの画像が無い時の"
        "「候補」探しで、ロゴの無い映像でも 3 割ほどのコマに候補が出る（一覧画像で目で確かめる）",
    ),
    arg(
        "--method",
        choices=["auto", "features", "template"],
        default="auto",
        help="--ref の探し方（既定 auto = 特徴点が取れれば features〔大きさ・回転・遠近に強い〕、取れなければ template）",
    ),
    arg(
        "--min-score",
        type=float,
        help="残す確からしさ（既定: features 0.3〔マークとの相関〕・template 0.82・言葉 0.5。"
        "言葉で誤検出を減らすなら 0.75）",
    ),
    arg("--max-count", type=int, default=5, help="1 コマで探す数の上限（既定 5）"),
    arg(
        "--every",
        type=float,
        help="動画を何秒ごとに調べるか（既定: --ref 0.2・言葉 0.5。間のコマは位置を直線で補う）",
    ),
    arg(
        "--no-video",
        action="store_true",
        help="動画の時、枠を描いた MP4 を作らない（JSON と一覧だけ）",
    ),
)
def logos(a: argparse.Namespace) -> None:
    """映り込んだマーク（--ref の画像、または言葉）を探し、四隅・時刻を JSON に、枠を描いた画像・MP4 を書く。"""
    info = core.probe(a.src)
    out, json_path, sheet_path = _paths(a, "logos", info)
    size = _size(info, a)
    sx, sy = info.width / size[0], info.height / size[1]
    if a.ref:
        finder = LogoFinder(
            a.ref, method=a.method, min_score=a.min_score, max_count=a.max_count
        )
        every = a.every if a.every is not None else 0.2
        what = (
            f"マーク {Path(a.ref).name} を探す（{finder.method}。特徴点 {finder.n_keypoints} 個、"
            f"残す確からしさ {finder.min_score}）"
        )

        def detect(frame: np.ndarray) -> list[dict[str, Any]]:
            return finder(frame)
    else:
        phrases = _phrases(a.query)
        # 0.5: 貼ったマークは 0.74〜0.87、ロゴの無い story540 でもコマの最大は 0.37〜0.70 だった（候補として出す）
        score = a.min_score if a.min_score is not None else 0.5
        gd = GroundingDino(box_threshold=score)
        every = a.every if a.every is not None else 0.5
        what = f"言葉で探す（Grounding DINO tiny）: {', '.join(phrases)}（残す確からしさ {score}）"

        def detect(frame: np.ndarray) -> list[dict[str, Any]]:
            out = []
            for d in gd(frame, phrases)[: a.max_count]:
                x1, y1, x2, y2 = d["box"]
                quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)
                out.append(
                    {
                        "quad": quad,
                        "score": d["score"],
                        "inliers": 0,
                        "method": "query",
                        "label": d["label"],
                    }
                )
            return out

    core.log(what)
    entries = []
    keyed: dict[int, list[_Item]] = {}
    shots = _SheetBuffer()
    for s in _samples(a.src, info, a, every, size):
        found = detect(s.frame)
        items, recs = [], []
        for d in found:
            q = d["quad"]
            qq = [[round(x * sx, 1), round(y * sy, 1)] for x, y in q]
            x1, y1, x2, y2 = _poly_box(np.array(qq))
            recs.append(
                {
                    "label": d.get("label", "mark"),
                    "score": round(d["score"], 3),
                    "quad": qq,
                    "box": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)],
                    "inliers": d["inliers"],
                    "method": d["method"],
                }
            )
            items.append(
                _Item(
                    q, f"{d.get('label', 'マーク')} {d['score']:.2f}", _YELLOW, "mark"
                )
            )
        entries.append({"t": s.t, "logos": recs})
        keyed[s.i] = items
        if items and not info.is_image:
            vis = s.frame.copy()
            for it in items:
                _draw_poly(vis, it.geom, it.color, it.label)
            shots.add(vis, _fmt_t(s.t))
    hits = [e for e in entries if e["logos"]]
    step = _step(info, every)
    segs: list[dict[str, Any]] = []
    for e in hits:
        if segs and e["t"] - segs[-1]["end"] <= max(0.5, 2.5 * every):
            segs[-1]["end"] = e["t"]
        else:
            segs.append({"start": e["t"], "end": e["t"]})
    for sg in segs:
        sg["end"] = round(sg["end"] + (0 if info.is_image else step / info.fps), 4)
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "params": {
                "ref": a.ref,
                "query": None if a.ref else a.query,
                "method": finder.method if a.ref else "grounding-dino-tiny",
                "every": every,
                "size": list(size),
            },
            "note": "quad は四隅 [[x, y] × 4]（マークの左上・右上・右下・左下の順。元の画素の座標）。box は [x, y, 幅, 高さ]",
            "segments": segs,
            "frames": entries,
        },
    )
    n = sum(len(e["logos"]) for e in entries)
    core.log(f"見つけた数: {n}（{len(hits)}/{len(entries)} コマ）")
    for sg in segs if not info.is_image else []:
        core.log(f"  映っている区間: {sg['start']:.2f}〜{sg['end']:.2f} 秒")
    if info.is_image:
        vis = next(core.read_frames(a.src, size=size))
        for it in keyed.get(0, []):
            _draw_poly(vis, it.geom, it.color, it.label)
        core.save_image(out, vis)
    else:
        if shots:
            core.save_image(sheet_path, shots.sheet())
            core.log(f"保存: {sheet_path}")
        if not a.no_video:
            _render_video(a.src, info, a, size, keyed, out, hold=step)
    if info.is_image or not a.no_video:
        core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 79 文字認識 ──────────────────────────────────────────────────────────


def _norm_text(t: str) -> str:
    """比べる時の形: NFKC（全角の英数字・記号を半角に）と空白を除く。"""
    import unicodedata

    return "".join(unicodedata.normalize("NFKC", t).split())


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _auto_mag(shape: tuple[int, ...]) -> float:
    """CRAFT に渡す拡大率: 長辺がおよそ 1440px になる倍率（1〜2）。v4（540×960）では 1.5 で正解率 0.847 → 0.958。"""
    return float(np.clip(1440 / max(shape[:2]), 1.0, 2.0))


def _srt_time(t: float) -> str:
    ms = max(0, round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@command(
    79,
    "ocr",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--every",
        type=float,
        default=0.5,
        help="動画を何秒ごとに読むか（既定 0.5。0 で全部のコマ。1 コマ数秒かかる）",
    ),
    arg(
        "--min-conf",
        type=float,
        default=0.1,
        help="残す確からしさの下限（0〜1。既定 0.1。飾りの多い大きな字は 0.3 前後でも読めていることがある）",
    ),
    arg(
        "--mag",
        type=float,
        help="文字を探す時の拡大率（既定: 長辺がおよそ 1440px になる倍率。1〜2）",
    ),
    arg(
        "--srt",
        action="store_true",
        help="動画の時、読んだ文字を映っている時刻つきの SRT（字幕の形式）にも書く",
    ),
)
def ocr(a: argparse.Namespace) -> None:
    """画面の文字を読み（EasyOCR。日本語・英語）、文字・枠・確からしさ・時刻を JSON に、注釈つきの画像に書く。

    動画は --every 秒ごとに読み、同じ場所の同じ文字（編集距離が長さの 1/4 以内）を 1 つにまとめて、映っている区間を出す。
    """
    info = core.probe(a.src)
    out = Path(a.out) if a.out else core.default_out(a.src, "ocr", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    size = _size(info, a)
    sx, sy = info.width / size[0], info.height / size[1]
    reader = easyocr_reader()
    mag = a.mag or _auto_mag((size[1], size[0]))
    every = max(0.0, a.every)
    step = _step(info, every)
    gap = max(1.0, 2.5 * step / info.fps)
    frames = []
    tracks: list[dict[str, Any]] = []
    shots = _SheetBuffer()
    for s in _samples(a.src, info, a, every, size):
        items = [
            (p, t.strip(), c)
            for p, t, c in easyocr_read(reader, s.frame, mag=mag)
            if c >= a.min_conf and t.strip()
        ]
        recs = []
        for p, t, c in items:
            quad = [[round(x * sx, 1), round(y * sy, 1)] for x, y in p]
            box = _poly_box(np.array(quad))
            recs.append({"text": t, "conf": round(c, 3), "quad": quad})
            key = _norm_text(t)
            match = None
            for tr in tracks:
                if s.t - tr["end"] > gap or tr["_seen"] == s.t:
                    continue
                d = _edit(key, tr["_key"])
                if d <= max(1, len(max(key, tr["_key"], key=len)) // 4) and (
                    _box_iou(box, tr["_box"]) >= 0.3
                ):
                    match = tr
                    break
            if match is None:
                match = {
                    "_key": key,
                    "_box": box,
                    "start": s.t,
                    "end": s.t,
                    "_votes": {},
                    "_seen": None,
                }
                tracks.append(match)
            match["end"] = s.t
            match["_seen"] = s.t
            match["_box"] = box
            match["_votes"][t] = match["_votes"].get(t, 0.0) + c
            match.setdefault("_best", (c, quad))
            if c > match["_best"][0]:
                match["_best"] = (c, quad)
        frames.append({"t": s.t, "texts": recs})
        if items:
            vis = s.frame.copy()
            for p, t, c in items:
                _draw_poly(vis, p, _GREEN)
                _put_tag(
                    vis,
                    t,
                    p[:, 0].min(),
                    p[:, 1].min(),
                    _GREEN,
                    size=max(13, min(vis.shape[:2]) // 40),
                )
            shots.add(vis, "" if info.is_image else _fmt_t(s.t))
    texts = []
    for tr in tracks:
        text = max(tr["_votes"].items(), key=lambda kv: kv[1])[0]
        end = tr["end"] + (0 if info.is_image else step / info.fps)
        texts.append(
            {
                "text": text,
                "start": tr["start"],
                "end": round(end, 4),
                "conf": round(tr["_best"][0], 3),
                "quad": tr["_best"][1],
                "variants": sorted(tr["_votes"], key=lambda v: -tr["_votes"][v]),
            }
        )
    texts.sort(key=lambda d: (d["start"], d["quad"][0][1], d["quad"][0][0]))
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "params": {
                "every": every,
                "min_conf": a.min_conf,
                "mag": mag,
                "size": list(size),
            },
            "note": "texts は同じ場所の同じ文字をまとめた物（start〜end 秒に映っている）。quad は四隅（元の画素の座標）。"
            "frames は調べたコマごとの読み取り",
            "texts": texts,
            "frames": frames,
        },
    )
    core.log(f"読めた文字: {len(texts)} 件（{len(frames)} コマを読んだ）")
    for d in texts[:200]:
        span = "" if info.is_image else f"{d['start']:.2f}〜{d['end']:.2f} 秒  "
        core.log(f"  {span}「{d['text']}」（{d['conf']:.2f}）")
    if a.srt and not info.is_image:
        srt = out.with_suffix(".srt")
        lines = []
        for n, d in enumerate(texts, 1):
            lines += [
                str(n),
                f"{_srt_time(d['start'])} --> {_srt_time(d['end'])}",
                d["text"],
                "",
            ]
        srt.write_text("\n".join(lines), encoding="utf-8")
        core.log(f"保存: {srt}")
    if info.is_image:
        vis = shots.first() if shots else next(core.read_frames(a.src, size=size))
        core.save_image(out, vis)
        core.log(f"保存: {out}")
    elif shots:
        core.save_image(out, shots.sheet())
        core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 96 テキスト検出 ──────────────────────────────────────────────────────


@command(
    96,
    "text-detect",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--model",
        choices=["craft", "ppocr"],
        default="craft",
        help="craft = EasyOCR の CRAFT（79 と同じ。正確。既定）、ppocr = PP-OCRv3 の文字検出（OpenCV。約 20 倍速いが、"
        "顔・目を文字と見なすことがあるので顔に重なる枠は除く。それでも文字の無い映像の 1 割ほどのコマに出る）",
    ),
    arg(
        "--threshold",
        type=float,
        default=0.5,
        help="文字とみなす確からしさ（0〜1。既定 0.5。craft では文字らしさの下限 = この値 + 0.2）",
    ),
    arg(
        "--min-area",
        type=float,
        default=0.0005,
        help="「文字がある」とみなす、文字の枠の面積の合計の下限（画面に対する割合。既定 0.0005）",
    ),
    arg(
        "--every",
        type=float,
        default=0.5,
        help="動画を何秒ごとに調べるか（既定 0.5。craft は 1 コマ約 1 秒）",
    ),
    arg(
        "--no-video",
        action="store_true",
        help="動画の時、枠を描いた MP4 を作らない（JSON と一覧だけ）",
    ),
)
def text_detect(a: argparse.Namespace) -> None:
    """文字のある場所を見つけ、枠を描いた画像・MP4 と、枠・「文字がある区間」の JSON を書く（読むのは 79）。

    生成した映像に、頼んでいない文字（看板・ロゴの崩れた文字など）が入っていないかの確認にも使える。
    """
    info = core.probe(a.src)
    out, json_path, sheet_path = _paths(a, "text-detect", info)
    size = _size(info, a)
    sx, sy = info.width / size[0], info.height / size[1]
    det = TextDetector(a.model, threshold=a.threshold, faces=a.model == "ppocr")
    area_total = float(size[0] * size[1])
    entries = []
    keyed: dict[int, list[_Item]] = {}
    shots = _SheetBuffer()
    for s in _samples(a.src, info, a, a.every, size):
        boxes = det(s.frame)
        recs, items = [], []
        area = 0.0
        for pts, sc in boxes:
            area += abs(cv2.contourArea(pts.astype(np.float32).reshape(-1, 1, 2)))
            recs.append(
                {
                    "quad": [[round(x * sx, 1), round(y * sy, 1)] for x, y in pts],
                    "score": round(sc, 3),
                }
            )
            items.append(_Item(pts, "", _GREEN, "text"))
        frac = area / area_total
        has = bool(recs) and frac >= a.min_area
        entries.append(
            {"t": s.t, "has_text": has, "area": round(frac, 5), "boxes": recs}
        )
        keyed[s.i] = items
        if has and not info.is_image:
            vis = s.frame.copy()
            for it in items:
                _draw_poly(vis, it.geom, it.color)
            shots.add(vis, f"{_fmt_t(s.t)} {len(items)} 個")
    step = _step(info, a.every)
    segs: list[dict[str, Any]] = []
    for e in entries:
        if not e["has_text"]:
            continue
        if segs and e["t"] - segs[-1]["end"] <= max(0.5, 1.5 * a.every):
            segs[-1]["end"] = e["t"]
            segs[-1]["max_area"] = max(segs[-1]["max_area"], e["area"])
        else:
            segs.append({"start": e["t"], "end": e["t"], "max_area": e["area"]})
    for sg in segs:
        sg["end"] = round(sg["end"] + (0 if info.is_image else step / info.fps), 4)
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "params": {
                "model": a.model,
                "threshold": a.threshold,
                "min_area": a.min_area,
                "every": a.every,
                "size": list(size),
            },
            "note": "segments は文字が映っている区間。area は文字の枠の面積の合計（画面に対する割合）。quad は四隅（元の画素の座標）",
            "segments": segs,
            "frames": entries,
        },
    )
    with_text = sum(e["has_text"] for e in entries)
    core.log(
        f"文字のあるコマ: {with_text}/{len(entries)}、枠の数: {sum(len(e['boxes']) for e in entries)}"
    )
    for sg in segs if not info.is_image else []:
        core.log(
            f"  文字がある区間: {sg['start']:.2f}〜{sg['end']:.2f} 秒（面積 最大 {sg['max_area'] * 100:.2f}%）"
        )
    if info.is_image:
        vis = next(core.read_frames(a.src, size=size))
        for it in keyed.get(0, []):
            _draw_poly(vis, it.geom, it.color)
        core.save_image(out, vis)
    else:
        if shots:
            core.save_image(sheet_path, shots.sheet())
            core.log(f"保存: {sheet_path}")
        if not a.no_video:
            _render_video(a.src, info, a, size, keyed, out, hold=step)
    if info.is_image or not a.no_video:
        core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")
