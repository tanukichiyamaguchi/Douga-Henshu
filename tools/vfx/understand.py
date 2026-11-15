"""自動で見つける（映像の内容の理解）。

担当するテクニック:
   18 動作認識（action）
   80 ハイライト検出（highlights）
   84 画像キャプション（caption）
   86 自然言語で場面検索（search）
   87 異常検知（anomalies）
   91 シーン分類（scene-classify）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
使う AI（どれも商用で使えるライセンス。初めて使う時に Hugging Face から取得）:
  SigLIP 2（google/siglip2-base-patch16-224、Apache-2.0）… 86・91・80 の --query・87 の「内容」
  X-CLIP（microsoft/xclip-base-patch32、MIT）… 18
  Florence-2 base（florence-community/Florence-2-base、MIT）… 84
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import core
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/understand/（make_*.py で正解つきの試験映像を作り、test_*.py で照らす）
NOTES: dict[int, dict[str, str]] = {
    86: {
        "status": "◎",
        "how": "SigLIP 2（google/siglip2-base-patch16-224、Apache-2.0。多言語で日本語の言葉も使える）で、--step 秒ごと"
        "（既定 0.5）のコマと言葉の特徴量を比べ、当てはまる確率（シグモイド）を出す。「|」で並べた言い方は言葉の特徴量を"
        "平均して 1 つの言葉にする。確率の山を高い順に選び、山から対数オッズ 2.0（--spread）以内で、基準の約 1/4.5 以上の"
        "コマを 1 つの区間にまとめる（1 コマの落ち込みはつなぐ）。2 つ目からの山は 1 位との差 2.5（--relative）以内だけ。"
        "山が --threshold（既定 0.005）未満なら「見つからない」。出力: JSON（区間・コマごとの確率）と確認画像（確率のグラフと、"
        "区間の始め・山・終わりのコマ）。画像 1 枚なら当てはまるかどうか",
        "measured": "正解つきの合成映像 haystack（16 種類の素材を 2.4〜3 秒ずつつないだ 41.5 秒）で、いちばんの区間が正しい"
        "部分に来た数: 英語の言葉 15/16（区間の重なり IoU 平均 0.81）、日本語だけ 10/16（0.54）、日本語|英語の平均 14/16"
        "（0.75）。素材に無い 9 つの言葉（スキー・サッカー・料理・水中の魚・ギター・赤ちゃん・飛行機・雨の傘・パソコン）を"
        "「見つからない」とした数: 英語 9/9・両方 9/9・日本語 8/9（日本語の「ギターを弾く人」が白黒写真の男性に 0.0085）。"
        "当たりの山の確率は 0.0067〜0.97、無い物の最大は英語 0.0013・日本語 0.0085。しきい値を変えると英語は 0.002〜0.005 で"
        " 15/16・9/9（0.001 で負 8/9、0.01 で正 13/16）。story540: 「ビューラーでまつげを上げている」0.00〜2.75 秒"
        "（目で見た正解 0〜3.0）、「カフェで笑う女性|a woman laughing at a cafe」12.25〜14.75（正解 12.38〜15.04）、"
        "「鏡の前の女性」2.75〜6.25（正解 3.0〜5.79）、「海辺を歩く人」「夜の街」は見つからない（最大 0.0007・0.0002）。"
        "画像 fruits.jpg: 果物 0.094・a lemon cut in half 0.018 → 当てはまる、a dog・夜の街 0.0000 → 当てはまらない。"
        "速さ: story540（15 秒・30 コマ）9.7 秒、haystack（41.5 秒・83 コマ）16.5 秒（どちらも SigLIP の読み込みを含む）",
        "limits": "日本語だけの言葉は英語より当たりにくい（上の 10/16 と 15/16）。Claude が英語に訳して渡すか、"
        "「日本語|英語」で両方渡す。動作（髪を結ぶ等）は 1 コマの見た目で見るので苦手（story540 の「髪を結んでいる女性」は"
        "外れた。動作は 18 を使う）。暗い CG アニメの場面は「アニメのキャラクター」「animated cartoon characters」で"
        "見つからなかった（最大 0.0004）。似た場面が続くと 1 つの区間にまとまったり、山が隣の場面に出たりする（日本語の"
        "「眉毛の手入れ」がまつげの施術の場面を取った）。区間の端は --step 単位（既定 0.5 秒）。確率の大きさは言葉によって"
        " 1 桁以上違うので、言葉どうしの比べ合いには使わない",
        "tested": "make_haystack.py（16 部分の正解つき）と test_search.py（日本語・英語・両方 × 正 16・負 9、しきい値の"
        "比較）、story540（目で確かめた区間）、fruits.jpg、--max-frames の範囲。確認画像 86-search-haystack.jpg・"
        "86-search-story.jpg",
    },
    91: {
        "status": "◎",
        "how": "場面に分け（既定は 76 と同じカット検出、--interval で一定の間隔）、場面の中の 3 コマ（--frames。両端を避けて"
        "等間隔）を SigLIP 2（Apache-2.0）で候補の言葉と照らし、対数オッズを 3 コマで平均して、組（--labels。既定は"
        "「屋内,屋外」と「昼,夜」）の中で softmax。組の中でいちばんの候補と確率を出し、--min-conf（0.6）未満は"
        "「？（はっきりしない）」。言い方は、素の言葉（屋内・屋外・昼・夜）が「〜で撮った写真」「〜の場面」・英語の文より"
        "良かった（言い方を選んだ 20 枚で 屋内／屋外 16/16・昼／夜 20/20）。出力: JSON と一覧画像（各場面の真ん中のコマに答え）",
        "measured": "言い方を選ぶのに使っていない素材で: UCF101 の試験用 75 本のうち屋内／屋外を目で決められた 68 本、"
        "実写の夜の写真 10 枚（花火 4・夜の道路など 6）、空撮 1 枚 → 屋内／屋外 79/79（「？」0 件）、昼／夜（屋外の 33 件。"
        "夜 12・昼 21）33/33。haystack（16 部分）: 屋内／屋外 15/15、屋外の昼／夜 10/10。story540: ビューラーの寄り 屋内"
        " 0.68、洗面所 屋内 0.94、戸口の日差し 屋外 0.75、空を見上げる顔 屋外 0.95、カフェ 屋外 0.99、すべて昼（目で見た"
        "答えと同じ）。自由な候補: 撮り方を英語（a close-up of a face / a medium shot of the upper body / a full-body shot）"
        "で 5/5、日本語（顔のアップ・上半身・全身）で 3/5（外れた 2 つは「？」0.50・0.55）。速さ: story540（5 場面）"
        "11.1 秒、haystack（16 場面）18.2 秒",
        "limits": "屋内の場面の昼／夜は意味が薄い（体育館・室内でも 夜 0.53〜0.71 と出た）。夜は実写 12 件で確かめた程度。"
        "自由な候補は英語の方が当たりやすい。場面の区切りは 76 の方法なので、似た場面どうしはつながることがある（haystack で"
        "まつげと眉の施術が 1 場面）。1 場面 3 コマの平均なので、場面の途中で屋内から屋外へ出る等は --interval を短くして分ける",
        "tested": "test_scene_classify.py（UCF101 試験用 75 本・夜の写真 10 枚・aero3。正解は目で付けた）、proto_siglip.py"
        "（言い方の比較）、haystack、story540（既定・撮り方の候補）、vtest（--interval 20）、v4（--start 5 --duration 6）。"
        "確認画像 91-scene-classify-story.jpg・91-scene-classify-haystack.jpg",
    },
    84: {
        "status": "○",
        "how": "Florence-2 base（florence-community/Florence-2-base、MIT。transformers 5.2 の Florence2ForConditionalGeneration"
        " で動くことを確認）の <CAPTION>（--detail 1 で <DETAILED_CAPTION>、2 で <MORE_DETAILED_CAPTION>）。動画は 76 と"
        "同じ方法で場面に分け、場面の中の 3 コマ（--candidates）でそれぞれ文を作り、SigLIP 2 で場面の 3 コマ全部との近さ"
        "（対数オッズの平均）がいちばん高い文を選ぶ（1 コマだけの作り話を落とす）。画像はビームサーチの上位 3 つから同じように"
        "選ぶ。--large で Florence-2 large（MIT、1.55GB）。文は英語。日本語の説明は、一覧画像（コマと英語の文）を Claude が"
        "見て書く",
        "measured": "中身の分かっている画像 17 枚（OpenCV・MediaPipe の見本、夜の写真）で、正解の言葉（例: 猫と犬の写真に"
        " cat と dog）が入った数: 既定（detail 0）17/17、detail 1 も 17/17。story540 の 5 場面（detail 0）: 4/5（ビューラーを"
        "知らず「metal mask」「whisk」）。目で読んだ誤り（detail 0、画像 17＋場面 5）: 選び直しなしで明らかな誤り 2"
        "（ビューラー＝metal mask、カフェの場面に無い laptop computer）、選び直しありで 1（whisk）と一部の誤り 1（手に持った"
        "チェッカー板を computer screen）。選び直しで laptop の文は SigLIP の点 −5.80（選ばれた文は 1.88）で落ちた。"
        "detail 1 は細部の作り話が増えた（猫 4 匹と犬 1 匹（実際は 2 と 2）、箱の文字「Pistachio Cookies」（実際は Crisp"
        " Almond Cookies）、白いドレス（実際はセーター））。large は 1 枚 5〜9 秒（base 1.6〜1.8 秒）で、laptop は出ず"
        "ビューラーを「mascara を塗るブラシ」と少し近いが、数・文字の誤りは残った。速さ（base）: 画像 1 枚 11.0 秒"
        "（選び直しなし 7.9 秒。どちらも読み込みを含む）、story540（5 場面 × 3 候補）41.3 秒",
        "limits": "英語の文だけ（日本語は Claude が書く）。道具・商品の名前（ビューラー等）は知らないことが多い。数（何匹か）"
        "や書かれた文字は誤りやすい。詳しくするほど作り話が増えるので、既定の detail 0 を勧める。大きな視覚言語モデルの"
        "説明よりは粗い。場面の代表は 3 コマなので、場面の途中で内容が変わる時は --interval で短く区切る",
        "tested": "test_caption.py（画像 17 枚＋story540 に正解の言葉）、review_caption.py（8 枚の文を目で確認）、"
        "proto_caption_consensus.py（選び直しの効果）、proto_florence_large.py（base と large の比較）、vtest"
        "（--interval 30 --candidates 1 --beams 1）。確認画像 84-caption-story.jpg・84-caption-images.jpg",
    },
    18: {
        "status": "○",
        "how": "X-CLIP（microsoft/xclip-base-patch32、MIT。Kinetics-400 で学習した 8 コマの動画と言葉の照合）のゼロショット。"
        "--window 秒（既定 2）から等間隔に 8 コマを取り、--step（0.5 秒）ずつずらした窓ごとに、候補の動作（英語。既定は日常の"
        "動作 30 種類、--labels で置き換え、--find で足す）との当てはまりを softmax で確率にする。窓ごとのいちばんの動作を"
        "つないだ流れと、--find の動作の確率が --threshold（0.3）以上の区間を出す。日本語の頼み（例: 手を振る）は Claude が"
        "英語（waving hand）に訳して --find に渡す。transformers 5.2.0 では XCLIPProcessor(videos=...) が空を返すので、中の"
        "画像処理（VideoMAE）と言葉の処理を直接使う。画像 1 枚は同じ画を 8 枚並べて推定",
        "measured": "Hugging Face の見本 eating_spaghetti.mp4（候補 3 つ）: eating spaghetti 0.998（公式の例と同じ）。"
        "UCF101 の試験用 75 本（10 種類、窓の確率を平均して 1 本の答え）: 候補 10 個で crop 65/75（86.7%）・pad 67/75"
        "（89.3%）・squash 63/75（84.0%）、日常の動作 30 個を足した 40 個（crop）で 63/75（84.0%）。間違いの多くは"
        " basketball と dunk の取り違え（設定により 7〜11 本）、ほかに口紅を brushing teeth・eating・applying eye makeup と"
        "した 0〜2 本。story540（縦長、目で確かめた 5 つの区間の 16 窓）: いちばんが正しい窓 9/16（crop・pad）・7/16"
        "（squash）。ビューラー 4/4・髪を結ぶ 2/2・笑う 3/3 は正しく、スマートフォンを見る 0/2（taking a selfie）・空を"
        "見上げる 0/5（顔の寄りを curling eyelashes とした）。--find の区間の重なり（IoU）: ビューラー 0.52、髪を結ぶ 0.76、"
        "笑う 0.95。画像 mp-pose.jpg: doing yoga 0.91。vtest（広い画角で小さな人が歩く）: walking がいちばんの窓 0/57"
        "（すべて waving hand。窓 4・8 秒、人の近くの切り出し、言い方 people walking でも 0）。速さ: story540（27 窓 ×"
        " 候補 33）14.3 秒、spaghetti（17 窓 × 3）8.1 秒",
        "limits": "人が小さく映る広い画（監視カメラ風）は見分けられず、waving hand を選びがち（vtest で 57/57）。そのため"
        "「手を振る」を探す時は特に結果のコマを目で確かめる（手を振る正解つきの映像では確かめていない）。似た動作"
        "（バスケットボールとダンク）は取り違える。顔の寄りの小さな動き（見上げる・目を開ける）は苦手。確率は候補の中での"
        "割合なので、候補に無い動作の場面でもどれかが選ばれる（story540 の日差しの場面は tying hair とされた）。縦長の映像は"
        " --fit crop（既定）と pad で差が小さかった（UCF で 2 本差）",
        "tested": "test_action.py（A: UCF101 試験用 75 本 × 4 設定、B: story540 の区間、C: vtest）、explore_action_vtest.py"
        "（vtest の失敗の調べ物）、proto_xclip_modes.py、eating_spaghetti.mp4、mp-pose.jpg。確認画像 18-action-story.jpg・"
        "18-action-ucf.jpg",
    },
    80: {
        "status": "○",
        "how": "盛り上がりの点数を --bin（0.25 秒）ごとに出す。材料: 音の大きさ（ffmpeg の ebur128 の瞬時ラウドネス M）、"
        "動きの量（長辺 160px の明るさの前のコマとの差。区間ごとの中央値でカットの跳ねを抑える）、カットの密度（76 の方法で"
        "見つけたカットの、区間の長さの窓の中の数）、任意で言葉との近さ（--query。SigLIP 2）。各材料を動画の中の順位（0〜1）に"
        "して重みつき平均（--weights）。変化が小さい材料（音 3dB 未満など）と音声の無い動画の音は使わない。区間の長さ"
        "（--length 3 秒）の平均が高い所から重ならないように --top 個選び、端から 0.5 秒以内のカットに端を合わせる。--reel で"
        "区間をつないだ MP4（音は 0.03 秒で溶かす）。出力: JSON・グラフと区間のコマの確認画像",
        "measured": "正解つきの合成映像（24 秒・音つき。静かな所に、速いカット 6 つ＋大きな音の 6〜9 秒と、笑う人＋大きな音の"
        " 15〜17 秒）: 上位 2 区間が 6.0〜9.0 秒（IoU 1.00）と 15.0〜17.875 秒（IoU 0.70。正解は 2 秒・区間は約 3 秒なので、"
        "正解を全部含めば 0.67〜0.7）。画だけ（audio=0）でも 6.0〜9.0・15.0〜18.125（IoU 1.00・0.64）、音だけ"
        "（motion=0,cuts=0）でも 6.0〜9.0・15.0〜17.875。v4（完成した広告 20 秒）: 上位 3 区間 1.0〜4.1（冒頭の「卒業」）・"
        "6.0〜9.1（曲の山＝「上がる。」）・10.0〜12.6（「好き」）。価格の強調 13.5〜16.6 は 4 位（0.52。3 位は 0.53）。"
        "--query 'a big price tag text|大きな価格の文字' では 13.5〜16.0 が選ばれ（言葉の順位 0.91）、代わりに 6〜9 が外れた。"
        "story540（音なし）: 音を使わずに 4.4〜7.4（髪を結ぶ〜日差しの場面へのカット）・7.4〜10.4・12.4〜14.9（笑う）。"
        "--reel: 8.900 秒（区間の合計 8.875 秒）・映像と音声あり。速さ: 合成 24 秒で 4.3〜4.7 秒、v4 で 7.0 秒（--reel を"
        "含む）、story540 で 3.2 秒",
        "limits": "音・動き・カットの量で見る簡易な方法で、意味（笑い・決めぜりふ・価格の強調など）は見ない（--query で"
        "言葉を足すと拾えるが、他の区間が押し出されることがある）。編集済みの広告のように最初から最後までカットが多く"
        "音量が一定の作品では、点数の差が小さい（v4 の 3 秒平均は 1〜15 秒の間 0.50〜0.66）。音の大きさは BGM と声を"
        "区別しない。区間の長さは固定（端だけカットに合わせる）",
        "tested": "make_highlights.py（正解つきの合成映像）と test_highlights.py（既定・画だけ・音だけ、v4・story540、"
        "--reel の長さと音声を ffprobe で確認）、v4 の --top 5 と --query、--max-frames。確認画像 80-highlights-synth.jpg・"
        "80-highlights-v4.jpg・80-highlights-v4-query.jpg・80-highlights-story.jpg",
    },
    87: {
        "status": "◎",
        "how": "コマごとに 5 つの特徴を出す: 速い動き（DIS オプティカルフローの速さの上位 5%）・画面全体の動き（フローの"
        "中央値）・明るさ・色と形（16×16 の Lab の小さな画）・内容（SigLIP 2 の特徴量、--look-step 0.25 秒ごと。--no-ai で"
        "使わない）。76 の方法で場面に分け、同じ場面の前と後ろ（各 --window 2 秒）の中央値と、その 2 つの真ん中の 3 つと"
        "比べた差のいちばん小さい物を「ずれ」とする（カットのように変わって続く物や、なだらかな変化は小さく、前後のどちらとも"
        "違う一瞬だけが大きい）。場面の端は隣の 2 コマとも比べる。ずれ ÷（1.4826 × 動画全体のずれの中央値）＝頑健な z 値が"
        " --z（8）以上のコマをつなげて区間にする。動きは log(1+速さ/0.5px) で比べる（速く動く場面の中の揺らぎより、静かな所の"
        "急な動きを重く見る）。色と形の差は色の成分ごとの画素の差の中央値（数人が歩く程度の一部の変化に鈍い）。さらに、"
        "1 秒以下の短い場面で前後の場面どうしが似ている物を「差し込み」（違いが色の全体のずれなら「明るさ」）とする",
        "measured": "vtest（固定カメラ・10fps）の 39 秒に 6 種類の異常を入れた正解つき映像: 別の映像 5 コマの差し込み"
        "（8.0〜8.5 秒）・1 コマの乱れ（15.0）・3 コマの光（+90。20.0〜20.3）・揺れ（±12px、25〜26）・2 秒の停止（30〜32）・"
        "4 倍の早送り（35〜36）→ 既定（z 8）で 6/6 を見つけ、余計な検出 0。z は 差し込み 90・乱れ 96・光 85.5・揺れ 28.3・"
        "停止 13.6（再開の 32.0 秒で検出）・早送り 18.7。手を加えない同じ映像: 0 件（普通の揺らぎの z は最大 6.8）。"
        "z 6 では余計 3 件（同じ映像の普通の所で 5 件）、z 12 では 6/6・余計 0。--no-ai でも 6/6（5.9 秒。SigLIP ありは"
        " 26.7 秒）、--fps 5 --max-frames 120 でも範囲内の 3/3。v4（完成した広告）: 白いフラッシュ 3 つ（2.5・6.0・16.0 秒）"
        "をすべて明るさの異常として検出（z 85〜91）、ほかに文字の動き・押し出しの転換など 11 件（計 14）。AI の映像（手を加え"
        "ない）: a-eye 0 件、b-lashlift 1 件（手袋の手が入る 0.25〜0.58 秒）、c-brow 2 件（手袋の手）、story540 2 件（3.0 秒の"
        "カット直前のぶれたビューラー、日差しの場面の明るさ）。速さ: 合成 39 秒（390 コマ）26.7 秒、v4（600 コマ）23〜32 秒、"
        "story540 18.4 秒",
        "limits": "「前後と違う一瞬」を探すので、2 秒（--window）より長く続く変化（2 秒の停止そのもの等）は異常にしない"
        "（停止は再開の飛びで見つかった）。編集済みの作品では、演出（フラッシュ・押し出しの転換・文字の動き）も異常として出る"
        "（v4 で 14 件）。人の手が画面に入る・速く動き出すなど、ふつうの出来事も「いつもと違う動き」として出る。1 秒以下の"
        "場面で前後がつながる物は「差し込み」とするので、意図した短いカットバックも出る。基準は z 8（6 では普通の映像でも"
        "出始め、12 では弱い異常を落とすおそれ）",
        "tested": "make_anomalies.py（6 種類の異常の正解つき）と test_anomalies.py（z 6・8・12、手を加えない vtest・a-eye・"
        "b-lashlift・c-brow・story540・v4）、--fps 5・--no-ai の確認。確認画像 87-anomalies-synth.jpg・"
        "87-anomalies-story.jpg・87-anomalies-v4.jpg",
    },
}

# ── 共通の部品 ────────────────────────────────────────────────────────────

_BLUE = (200, 110, 20)
_RED = (40, 40, 230)
_GREEN = (60, 160, 40)
_ORANGE = (20, 140, 240)
_PURPLE = (170, 60, 160)
_TEAL = (150, 140, 20)
_COLORS = [_BLUE, _RED, _GREEN, _ORANGE, _PURPLE, _TEAL, (60, 60, 60), (40, 180, 200)]
_SHEET_MAX = 40  # 一覧画像に並べる数の上限


def _rng(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": a.start or 0.0,
        "duration": a.duration,
        "max_frames": a.max_frames,
    }


def _end(info: core.MediaInfo, rng: dict[str, Any], fps: float | None = None) -> float:
    """調べる範囲の終わり（元の動画の中の秒）。--max-frames は fps（既定は元の fps）のコマの数として数える。"""
    t1 = info.duration
    if rng["duration"]:
        t1 = min(t1, rng["start"] + rng["duration"])
    if rng["max_frames"]:
        t1 = min(t1, rng["start"] + rng["max_frames"] / (fps or info.fps))
    return t1


def _span_rng(rng: dict[str, Any], end: float) -> dict[str, Any]:
    """同じ範囲を、コマ数でなく秒で表した物（別の間隔でコマを読み直す時に使う）。"""
    return {"start": rng["start"], "duration": end - rng["start"], "max_frames": None}


def _need_video(info: core.MediaInfo, what: str) -> None:
    core.require(
        not info.is_image,
        f"動画を指定してください（{what}は時間の流れを見るので、画像 1 枚では使えません）",
    )


def _sample(
    src: str, info: core.MediaInfo, step: float, rng: dict[str, Any], max_side: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """step 秒ごとのコマと、その時刻（元の動画の中の秒）。画像なら 1 枚（時刻 0）。

    --max-frames は、ここでは「調べるコマ（step 秒ごと）の数」の上限として使う。
    """
    if info.is_image:
        return list(core.read_frames(src, max_side=max_side)), np.zeros(1)
    frames = core.read_all(src, fps=1.0 / step, max_side=max_side, **rng)
    return frames, rng["start"] + np.arange(len(frames)) * step


def _frame(
    src: str, info: core.MediaInfo, t: float, max_side: int | None = None
) -> np.ndarray:
    """t 秒のコマ（範囲の外は端に寄せる）。画像ならその画像。"""
    if info.is_image:
        return next(core.read_frames(src, max_side=max_side))
    t = min(max(t, 0.0), max(info.duration - 1.5 / info.fps, 0.0))
    return core.frame_at(src, t, max_side=max_side)


def _split_scenes(
    src: str, info: core.MediaInfo, rng: dict[str, Any], interval: float
) -> list[tuple[float, float]]:
    """場面の区切り（始め・終わりの秒）。interval > 0 なら一定の間隔、0 ならカット（76 と同じ方法）。"""
    t0, t1 = rng["start"], _end(info, rng)
    if interval > 0:
        edges = [float(v) for v in np.arange(t0, t1, interval)] + [t1]
        if len(edges) > 2 and edges[-1] - edges[-2] < interval * 0.3:
            edges.pop(-2)  # 最後の短すぎる区間は 1 つ前に混ぜる
        return [(round(s, 4), round(e, 4)) for s, e in itertools.pairwise(edges)]
    from .scenes import detect_cuts

    res = detect_cuts(str(src), **rng)
    return [(res.t(s), res.t(e + 1)) for s, e in res.scenes()]


def _inner_times(s: float, e: float, k: int) -> list[float]:
    """場面の中の k 個の時刻（両端の変わり目を避けて等間隔）。"""
    margin = min(0.1 * (e - s), 0.25)
    lo, hi = s + margin, e - margin
    if hi <= lo:
        return [(s + e) / 2]
    return [lo + (hi - lo) * (j + 0.5) / k for j in range(k)]


def _runs(flags: np.ndarray, gap: int = 0) -> list[tuple[int, int]]:
    """True が続く所を (最初, 最後) の番号の組にする。gap 個までの False の切れ目はつなぐ。"""
    idx = np.flatnonzero(flags)
    if not len(idx):
        return []
    out = []
    lo = hi = int(idx[0])
    for i in idx[1:]:
        if i - hi - 1 <= gap:
            hi = int(i)
        else:
            out.append((lo, hi))
            lo = hi = int(i)
    out.append((lo, hi))
    return out


def _sigmoid(x: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def _softmax(x: Any, axis: int = -1) -> np.ndarray:
    z = np.asarray(x, np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _split_list(text: str) -> list[str]:
    """「,」（全角の「、」「，」も）で区切った候補の一覧。"""
    return [s.strip() for s in re.split(r"[,，、]", text) if s.strip()]


# ── SigLIP 2（画像と言葉の照合） ─────────────────────────────────────────

# 読み込んだ AI（同じプロセスで何度も使う時に読み直さない）
_MODELS: dict[str, Any] = {}


def _cached(key: str, factory: Any) -> Any:
    if key not in _MODELS:
        _MODELS[key] = factory()
    return _MODELS[key]


def _siglip() -> Any:
    from .ai import ImageText

    return _cached("siglip", ImageText)


def _text_emb(it: Any, texts: list[str]) -> np.ndarray:
    """言葉の特徴量。「|」で区切った言い方は、特徴量を平均して 1 つの言葉として使う（言い方の揺れをならす）。"""
    out = []
    for q in texts:
        parts = [p.strip() for p in q.split("|") if p.strip()]
        core.require(bool(parts), f"空の言葉があります: {q!r}")
        e = it.embed_texts(parts).mean(axis=0)
        out.append(e / (np.linalg.norm(e) + 1e-12))
    return np.stack(out).astype(np.float64)


def _logits(it: Any, img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    """画像×言葉の対数オッズ（SigLIP の logit。当てはまる確率は 1/(1+e^-x)）。"""
    scale = float(it.net.logit_scale.exp())
    bias = float(it.net.logit_bias)
    return img_emb.astype(np.float64) @ txt_emb.T.astype(np.float64) * scale + bias


# ── 確認画像の部品 ────────────────────────────────────────────────────────


def _text_w(text: str, size: int) -> int:
    """文字の幅のおおよそ（全角は size、半角は 0.58×size）。"""
    return int(sum(size if ord(c) > 0x2000 else size * 0.58 for c in text))


def _wrap(text: str, size: int, width: int) -> list[str]:
    """文を幅 width px に収まるように折り返す（英語は単語ごと、日本語は文字ごと）。"""
    tokens = re.findall(r"\S+\s*", text) if " " in text else list(text)
    lines: list[str] = []
    cur = ""
    for tok in tokens:
        if cur and _text_w(cur + tok.rstrip(), size) > width:
            lines.append(cur.rstrip())
            cur = tok.lstrip()
        else:
            cur += tok
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def _header(
    text: str,
    width: int,
    size: int = 18,
    bg: tuple[int, int, int] = (40, 40, 40),
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    lines = [ln for raw in text.split("\n") for ln in _wrap(raw, size, width - 20)]
    lh = int(size * 1.5)
    img = np.full((lh * max(1, len(lines)) + 12, width, 3), bg, np.uint8)
    for k, line in enumerate(lines):
        core.draw_text(img, line, (10, 6 + k * lh), size=size, color=color, bg=None)
    return img


def _stack(images: list[np.ndarray], width: int = 1200) -> np.ndarray:
    """画像を縦に並べる（広い物は幅に縮め、狭い物は拡大せず右を空ける）。"""
    rows = []
    for im in images:
        im = im[:, :, :3]
        if im.shape[1] > width:
            h = max(1, round(im.shape[0] * width / im.shape[1]))
            im = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
        elif im.shape[1] < width:
            pad = np.full((im.shape[0], width - im.shape[1], 3), 24, np.uint8)
            im = np.hstack([im, pad])
        rows += [im, np.full((6, width, 3), 24, np.uint8)]
    return np.vstack(rows[:-1])


def _card(
    img: np.ndarray,
    lines: list[str],
    width: int = 300,
    size: int = 14,
    color: tuple[int, int, int] = (20, 20, 20),
    max_h: int | None = None,
) -> np.ndarray:
    """コマの下に文字の欄を付けた札。"""
    h = max(1, round(img.shape[0] * width / img.shape[1]))
    top = cv2.resize(img[:, :, :3], (width, h), interpolation=cv2.INTER_AREA)
    if max_h and h > max_h:  # 縦長の画は上下を詰める
        y = (h - max_h) // 2
        top = top[y : y + max_h]
    wrapped = [w for ln in lines for w in (_wrap(ln, size, width - 10) or [""])]
    lh = int(size * 1.5)
    panel = np.full((lh * max(1, len(wrapped)) + 8, width, 3), 255, np.uint8)
    for k, line in enumerate(wrapped):
        core.draw_text(panel, line, (5, 4 + k * lh), size=size, color=color, bg=None)
    return np.vstack([top, panel])


def _grid(
    cells: list[np.ndarray], cols: int, bg: tuple[int, int, int] = (24, 24, 24)
) -> np.ndarray:
    cw = max(c.shape[1] for c in cells)
    ch = max(c.shape[0] for c in cells)
    rows = math.ceil(len(cells) / cols)
    sheet = np.full(
        (rows * (ch + 4) - 4, min(cols, len(cells)) * (cw + 4) - 4, 3), bg, np.uint8
    )
    for i, c in enumerate(cells):
        y, x = (i // cols) * (ch + 4), (i % cols) * (cw + 4)
        sheet[y : y + c.shape[0], x : x + c.shape[1]] = c
    return sheet


def _chart(
    t: Any,
    series: list[tuple[Any, tuple[int, int, int], str]],
    *,
    title: str,
    width: int = 1200,
    height: int = 230,
    ylim: tuple[float, float] | None = None,
    log: bool = False,
    spans: list[tuple[float, float, tuple[int, int, int]]] | None = None,
    hlines: list[tuple[float, tuple[int, int, int], str]] | None = None,
    marks: list[tuple[float, tuple[int, int, int]]] | None = None,
    xlim: tuple[float, float] | None = None,
) -> np.ndarray:
    """時間の流れのグラフ（横軸は秒）。series は (値, 色, 名前)。白地の BGR 画像を返す。"""
    spans, hlines, marks = spans or [], hlines or [], marks or []
    left, right, top, bottom = 64, 14, 34, 26
    # 凡例が見出しに重なる時は、凡例を見出しの下の段に置く
    names = [name for _v, _c, name in series if name]
    legend_w = sum(_text_w(n, 13) + 36 for n in names)
    two_rows = legend_w > width - left - right - _text_w(title, 15) - 20
    if two_rows:
        top += 22
        height += 22
    img = np.full((height, width, 3), 255, np.uint8)
    pw, ph = width - left - right, height - top - bottom
    t = np.asarray(t, np.float64)
    if xlim is not None:
        t0, t1 = float(xlim[0]), float(xlim[1])
    else:
        t0 = float(t[0]) if len(t) else 0.0
        t1 = float(t[-1]) if len(t) > 1 else t0 + 1.0
    t1 = max(t1, t0 + 1e-3)

    def tr(v: Any) -> np.ndarray:
        v = np.asarray(v, np.float64)
        return np.log10(np.clip(v, 1e-6, None)) if log else v

    if ylim is not None:
        lo, hi = float(tr(ylim[0])), float(tr(ylim[1]))
    else:
        vals = [tr(s[0]).ravel() for s in series] + [tr([h[0] for h in hlines])]
        finite = [v[np.isfinite(v)] for v in vals if v.size]
        allv = np.concatenate(finite) if finite else np.zeros(1)
        allv = allv if allv.size else np.zeros(1)
        lo, hi = float(allv.min()), float(allv.max())
        pad = (hi - lo) * 0.06 or 1.0
        lo, hi = lo - pad, hi + pad
    if hi - lo < 1e-9:
        hi = lo + 1.0

    def px(tt: Any) -> np.ndarray:
        return np.round(left + (np.asarray(tt, np.float64) - t0) / (t1 - t0) * pw)

    def py(v: Any) -> np.ndarray:
        y = top + ph - (tr(v) - lo) / (hi - lo) * ph
        y = np.nan_to_num(y, nan=top + ph, posinf=top, neginf=top + ph)
        return np.round(np.clip(y, top, top + ph))

    if spans:
        ov = img.copy()
        for a0, a1, color in spans:
            x0, x1 = int(px(a0)), int(px(a1))
            cv2.rectangle(ov, (x0, top), (max(x1, x0 + 2), top + ph), color, -1)
        img = cv2.addWeighted(ov, 0.25, img, 0.75, 0)
    step = next(
        (s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
         if (t1 - t0) / s <= 16),
        1800,
    )  # fmt: skip
    for k in range(math.ceil(t0 / step), int(t1 / step) + 1):
        x = int(px(k * step))
        cv2.line(img, (x, top), (x, top + ph), (228, 228, 228), 1)
        core.draw_text(
            img, f"{k * step:g}", (x, top + ph + 4), size=12, color=(90, 90, 90),
            bg=None, anchor="mt",
        )  # fmt: skip
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = round(top + ph - k / 4 * ph)
        cv2.line(img, (left, y), (left + pw, y), (236, 236, 236), 1)
        core.draw_text(
            img, f"{10**v:.2g}" if log else f"{v:.3g}", (left - 5, y), size=12,
            color=(90, 90, 90), bg=None, anchor="rm",
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
                img, name, (left + pw - 6, y - 3), size=12, color=color, bg=None,
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
                cv2.polylines(img, [np.array(seg, np.int32)], False, color, 2)
                seg = []
        if len(seg) == 1:
            cv2.circle(img, (int(seg[0][0]), int(seg[0][1])), 3, color, -1)
        elif seg:
            cv2.polylines(img, [np.array(seg, np.int32)], False, color, 2)
    cv2.rectangle(img, (left, top), (left + pw, top + ph), (120, 120, 120), 1)
    core.draw_text(img, title, (left, 8), size=15, color=(20, 20, 20), bg=None)
    x, ly = left + pw, 32 if two_rows else 10
    for _vals, color, name in reversed(series):
        if not name:
            continue
        x -= _text_w(name, 13)
        core.draw_text(img, name, (x, ly), size=13, color=(40, 40, 40), bg=None)
        cv2.line(img, (x - 24, ly + 8), (x - 5, ly + 8), color, 3)
        x -= 36
    core.draw_text(
        img, "秒", (left + pw, top + ph + 4), size=12, color=(90, 90, 90), bg=None,
        anchor="rt",
    )  # fmt: skip
    return img


# ── 86 自然言語で場面検索 ─────────────────────────────────────────────────


def _peak_intervals(
    score: np.ndarray,
    *,
    top: int,
    spread: float,
    floor: float,
    relative: float = math.inf,
    extend: float = 1.5,
) -> list[tuple[int, int, int]]:
    """点数の山を高い順に選び、山から spread 以内（かつ floor − extend 以上）のコマを 1 つの区間にする。

    山そのものは floor 以上が要る（見つかったかの判定）。区間の広がりは floor より extend だけ低い所まで
    認める（弱いが正しい場面で区間が細切れになるのを防ぐ）。2 つ目からの山は、いちばんの山から
    relative 以内の物だけ。1 コマだけの落ち込みはつなぐ。(最初, 山, 最後) の番号の組を返す。
    """
    n = len(score)
    used = np.zeros(n, bool)
    out: list[tuple[int, int, int]] = []
    for _ in range(top):
        cand = np.where(used, -np.inf, score)
        k = int(np.argmax(cand))
        if not np.isfinite(cand[k]) or cand[k] < floor:
            break
        if out and cand[k] < score[out[0][1]] - relative:
            break
        ok = (score >= max(cand[k] - spread, floor - extend)) & ~used
        i0 = i1 = k
        while i0 >= 1 and (ok[i0 - 1] or (i0 >= 2 and ok[i0 - 2] and not used[i0 - 1])):
            i0 -= 1 if ok[i0 - 1] else 2
        while i1 + 1 < n and (
            ok[i1 + 1] or (i1 + 2 < n and ok[i1 + 2] and not used[i1 + 1])
        ):
            i1 += 1 if ok[i1 + 1] else 2
        used[max(0, i0 - 1) : i1 + 2] = (
            True  # 隣に別の区間がくっつかないよう 1 コマ空ける
        )
        out.append((i0, k, i1))
    return out


@command(
    86,
    "search",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "-q",
        "--query",
        action="append",
        required=True,
        help="探したい場面を言葉で（日本語でも英語でも。何度でも指定できる）。"
        "「|」で同じ意味の言い方を並べると、平均して 1 つの言葉として探す"
        "（例: 'カフェで笑う女性|a woman laughing at a cafe'。英語を足すと当たりやすい）",
    ),
    arg("--step", type=float, default=0.5, help="調べるコマの間隔（秒。既定 0.5）"),
    arg(
        "--threshold",
        type=float,
        default=0.005,
        help="見つかったとみなす確率の下限（0〜1。既定 0.005。SigLIP の確率は小さめに出る）",
    ),
    arg("--top", type=int, default=3, help="言葉ごとに出す区間の数の上限（既定 3）"),
    arg(
        "--spread",
        type=float,
        default=2.0,
        help="区間の広がり: 山の対数オッズからこの差までのコマを同じ区間に入れる（既定 2.0。大きいほど長い区間）",
    ),
    arg(
        "--relative",
        type=float,
        default=2.5,
        help="2 つ目からの区間は、いちばんの山との差（対数オッズ）がこの値以内の山だけ（既定 2.5 ≒ 確率の比 12 倍まで）",
    ),
)
def search(a: argparse.Namespace) -> None:
    """言葉で表した場面を探し、当てはまる区間を JSON に、確率のグラフと区間のコマを並べた確認画像を JPG に書く。"""
    core.require(a.step > 0, "--step は 0 より大きくしてください")
    core.require(0 < a.threshold < 1, "--threshold は 0 と 1 の間にしてください")
    info = core.probe(a.src)
    rng = _rng(a)
    out = Path(a.out) if a.out else core.default_out(a.src, "search", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    frames, t = _sample(a.src, info, a.step, rng, a.max_side or 384)
    core.log(f"{len(frames)} コマを SigLIP 2 で調べます")
    it = _siglip()
    lg = _logits(it, it.embed_images(frames), _text_emb(it, a.query))
    t_lo = float(t[0])
    t_hi = (
        t_lo
        if info.is_image
        else min(_end(info, rng, 1.0 / a.step), float(t[-1]) + a.step)
    )
    floor = _logit(a.threshold)
    results: list[dict[str, Any]] = []
    picks: list[list[tuple[int, int, int]]] = []
    for k, q in enumerate(a.query):
        col = lg[:, k]
        best = int(np.argmax(col))
        found = _peak_intervals(
            col, top=a.top, spread=a.spread, floor=floor, relative=a.relative
        )
        picks.append(found)
        results.append(
            {
                "query": q,
                "found": bool(found),
                "best_time": round(float(t[best]), 3),
                "best_prob": round(float(_sigmoid(col[best])), 6),
                "intervals": [
                    {
                        "start": round(max(t_lo, float(t[i0]) - a.step / 2), 3),
                        "end": round(min(t_hi, float(t[i1]) + a.step / 2), 3),
                        "peak_time": round(float(t[ip]), 3),
                        "peak_prob": round(float(_sigmoid(col[ip])), 6),
                    }
                    for i0, ip, i1 in found
                ],
            }
        )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "is_image": info.is_image,
            "step": a.step,
            "threshold": a.threshold,
            "spread": a.spread,
            "model": "google/siglip2-base-patch16-224",
            "results": results,
            "times": [round(float(v), 3) for v in t],
            "probs": {
                q: [round(float(v), 6) for v in _sigmoid(lg[:, k])]
                for k, q in enumerate(a.query)
            },
        },
    )
    for r in results:
        if info.is_image:
            mark = "当てはまる" if r["found"] else "当てはまらない"
            core.log(f"「{r['query']}」: 確率 {r['best_prob']:.4f} → {mark}")
            continue
        if not r["found"]:
            core.log(
                f"「{r['query']}」: 見つかりませんでした（最も近いのは {r['best_time']:.1f} 秒、"
                f"確率 {r['best_prob']:.4f} < 基準 {a.threshold}）"
            )
            continue
        core.log(f"「{r['query']}」: {len(r['intervals'])} 区間")
        for iv in r["intervals"]:
            core.log(
                f"  {iv['start']:6.2f}〜{iv['end']:6.2f} 秒（山 {iv['peak_time']:.1f} 秒、"
                f"確率 {iv['peak_prob']:.4f}）"
            )
    # 確認画像
    if info.is_image:
        lines = [
            f"「{r['query']}」 {r['best_prob']:.4f} "
            + ("当てはまる" if r["found"] else "当てはまらない")
            for r in results
        ]
        sheet = _card(frames[0], lines, width=min(720, frames[0].shape[1]), size=16)
    else:
        rows = []
        for k, r in enumerate(results):
            spans = [(iv["start"], iv["end"], _GREEN) for iv in r["intervals"]]
            summary = (
                f"「{r['query']}」 {len(r['intervals'])} 区間"
                if r["found"]
                else f"「{r['query']}」 見つからない（最大 {r['best_prob']:.4f}、{r['best_time']:.1f} 秒）"
            )
            rows.append(_header(summary, 1200, size=17))
            rows.append(
                _chart(
                    t,
                    [(_sigmoid(lg[:, k]), _BLUE, "当てはまる確率")],
                    title="確率（対数目盛り）。緑＝見つけた区間、赤の点線＝基準",
                    log=True,
                    ylim=(1e-5, 1.0),
                    spans=spans,
                    hlines=[(a.threshold, _RED, f"基準 {a.threshold}")],
                )
            )
            cells, labels = [], []
            for n, (i0, ip, i1) in enumerate(picks[k]):
                for name, i in (("始め", i0), ("山", ip), ("終わり", i1)):
                    cells.append(frames[i])
                    labels.append(
                        f"{n + 1}{name} {t[i]:.1f}秒 {_sigmoid(lg[i, k]):.3f}"
                    )
            if not cells:
                best = int(np.argmax(lg[:, k]))
                cells, labels = (
                    [frames[best]],
                    [f"最も近い（基準未満） {t[best]:.1f}秒"],
                )
            rows.append(
                core.contact_sheet(
                    cells, labels, cols=min(9, len(cells)), cell_width=200
                )
            )
        sheet = _stack(rows)
    core.save_image(out, sheet)
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 91 シーン分類 ─────────────────────────────────────────────────────────

_DEFAULT_GROUPS = ["屋内,屋外", "昼,夜"]


@command(
    91,
    "scene-classify",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--labels",
        action="append",
        help="分類の候補を「,」区切りで（1 回の指定が 1 つの組で、組の中から最も当てはまる 1 つを選ぶ。"
        "何度でも指定できる。日本語でも英語でも。既定は「屋内,屋外」と「昼,夜」の 2 組）",
    ),
    arg(
        "--interval",
        type=float,
        default=0.0,
        help="一定の間隔（秒）で区切る（既定 0 = カットで区切る。76 と同じ方法）",
    ),
    arg("--frames", type=int, default=3, help="場面ごとに見るコマの数（既定 3）"),
    arg(
        "--min-conf",
        type=float,
        default=0.6,
        help="組の中の確率がこれに届かない時は「はっきりしない」とする（0〜1。既定 0.6）",
    ),
)
def scene_classify(a: argparse.Namespace) -> None:
    """場面ごとに、屋内／屋外・昼／夜などの候補から当てはまる物を選び、JSON と一覧画像（JPG）に書く。"""
    groups = [_split_list(g) for g in (a.labels or _DEFAULT_GROUPS)]
    for g in groups:
        core.require(len(g) >= 2, f"組には候補を 2 つ以上入れてください: {g}")
    core.require(a.frames >= 1, "--frames は 1 以上にしてください")
    info = core.probe(a.src)
    rng = _rng(a)
    out = Path(a.out) if a.out else core.default_out(a.src, "scene-classify", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    scenes = (
        [(0.0, 0.0)] if info.is_image else _split_scenes(a.src, info, rng, a.interval)
    )
    core.log(f"場面 {len(scenes)} 個を SigLIP 2 で分類します")
    it = _siglip()
    flat = [lab for g in groups for lab in g]
    temb = _text_emb(it, flat)
    items: list[dict[str, Any]] = []
    cards: list[np.ndarray] = []
    for k, (s, e) in enumerate(scenes):
        times = [0.0] if info.is_image else _inner_times(s, e, a.frames)
        frames = [_frame(a.src, info, tt, a.max_side or 384) for tt in times]
        lg = _logits(it, it.embed_images(frames), temb).mean(axis=0)
        res = []
        off = 0
        for g in groups:
            p = _softmax(lg[off : off + len(g)])
            j = int(np.argmax(p))
            res.append(
                {
                    "labels": {
                        lab: round(float(v), 4) for lab, v in zip(g, p, strict=True)
                    },
                    "best": g[j],
                    "prob": round(float(p[j]), 4),
                    "clear": bool(p[j] >= a.min_conf),
                }
            )
            off += len(g)
        items.append(
            {
                "no": k + 1,
                "start": round(s, 3),
                "end": round(e, 3),
                "times": [round(tt, 3) for tt in times],
                "groups": res,
                "sigmoid": {
                    lab: round(float(v), 6)
                    for lab, v in zip(flat, _sigmoid(lg), strict=True)
                },
            }
        )
        verdict = "・".join(
            f"{r['best']}{'' if r['clear'] else '？'} {r['prob']:.2f}" for r in res
        )
        where = "画像" if info.is_image else f"{s:6.2f}〜{e:6.2f} 秒"
        core.log(f"  場面 {k + 1:2d}: {where} → {verdict}")
        if k < _SHEET_MAX:
            head = "画像" if info.is_image else f"{k + 1}: {s:.2f}〜{e:.2f}秒"
            cards.append(
                _card(
                    frames[len(frames) // 2],
                    [head, verdict],
                    width=240,
                    size=14,
                    max_h=300,
                )
            )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "model": "google/siglip2-base-patch16-224",
            "split": "image"
            if info.is_image
            else (f"interval {a.interval}" if a.interval > 0 else "cuts"),
            "groups": groups,
            "min_conf": a.min_conf,
            "scenes": items,
        },
    )
    sheet = _grid(cards, cols=min(6, len(cards)))
    note = "？＝はっきりしない（組の中の確率が --min-conf 未満）"
    core.save_image(
        out, _stack([_header(note, sheet.shape[1], size=15), sheet], sheet.shape[1])
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 84 画像キャプション ───────────────────────────────────────────────────

_TASKS = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
# --large で使う（MIT。1.55GB。2026-11-14 に https://huggingface.co/api/models で確認）
_FLORENCE_LARGE = "florence-community/Florence-2-large"


class _Florence:
    """Florence-2（florence-community/Florence-2-base か -large、MIT）で、画像の内容を英語の文にする。"""

    def __init__(self, large: bool = False) -> None:
        from .models import hf_repo, torch_setup

        torch = torch_setup()
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.repo = _FLORENCE_LARGE if large else hf_repo("florence")
        self.proc = AutoProcessor.from_pretrained(self.repo)
        self.net = Florence2ForConditionalGeneration.from_pretrained(
            self.repo, dtype=torch.float32
        ).eval()

    def __call__(
        self, frame: np.ndarray, task: str, beams: int = 3, n: int = 1
    ) -> list[str]:
        """文の候補を n 個（ビームサーチの上位から）返す。"""
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        inputs = self.proc(text=task, images=rgb, return_tensors="pt")
        ids = self.net.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=max(beams, n),
            num_return_sequences=n,
            do_sample=False,
            early_stopping=max(beams, n) > 1,  # ビーム 1 の時の無害な警告を出さない
        )
        out = []
        for text in self.proc.batch_decode(ids, skip_special_tokens=False):
            # 候補を複数返すと、短い文の後ろに詰め物（<pad>）が残る
            text = text.replace("<pad>", "")
            parsed = self.proc.post_process_generation(
                text, task=task, image_size=(rgb.shape[1], rgb.shape[0])
            )
            out.append(str(parsed.get(task, "")).strip())
        return out


@command(
    84,
    "caption",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--detail",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="文の詳しさ: 0＝短い 1 文（既定）、1＝2〜3 文、2＝さらに詳しく（どれも見えない物を書く作り話が混ざる。詳しいほど多い）",
    ),
    arg(
        "--interval",
        type=float,
        default=0.0,
        help="一定の間隔（秒）で区切る（既定 0 = カットで区切る。76 と同じ方法）",
    ),
    arg(
        "--candidates",
        type=int,
        default=3,
        help="場面ごとに文を作るコマの数（画像ならビームサーチの上位の数）。2 以上なら、SigLIP 2 で場面のコマ全部に"
        "いちばん合う文を選ぶ（1 コマだけの作り話を減らす。既定 3。1 で速い）",
    ),
    arg(
        "--beams",
        type=int,
        default=3,
        help="文を作る時に比べる候補の数（ビームサーチ。既定 3。1 で速くなる）",
    ),
    arg(
        "--large",
        action="store_true",
        help="Florence-2 large（MIT、1.55GB を初回に取得）を使う。base より少し正確だが 3〜5 倍遅い",
    ),
)
def caption(a: argparse.Namespace) -> None:
    """場面ごとの代表コマ（画像なら 1 枚）に映っている内容を英語の文にし、JSON と一覧画像（JPG）に書く。

    文は英語。日本語の説明が要る時は、一覧画像（コマと英語の文を並べた物）を Claude が見て日本語で書く。
    """
    core.require(
        a.beams >= 1 and a.candidates >= 1,
        "--beams と --candidates は 1 以上にしてください",
    )
    task = _TASKS[a.detail]
    info = core.probe(a.src)
    rng = _rng(a)
    out = Path(a.out) if a.out else core.default_out(a.src, "caption", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    side = a.max_side or 768
    scenes = (
        [(0.0, 0.0)] if info.is_image else _split_scenes(a.src, info, rng, a.interval)
    )
    core.log(
        f"{len(scenes)} 場面の文を Florence-2 で作ります（{task}、候補 {a.candidates}）"
    )
    fl = _cached(
        "florence-large" if a.large else "florence", lambda: _Florence(a.large)
    )
    it = _siglip() if a.candidates > 1 else None
    results, cards = [], []
    for k, (s, e) in enumerate(scenes):
        if info.is_image:
            times = [0.0]
            frames = [_frame(a.src, info, 0.0, side)]
            cands = [(c, 0) for c in fl(frames[0], task, beams=a.beams, n=a.candidates)]
        else:
            times = _inner_times(s, e, a.candidates)
            frames = [_frame(a.src, info, tt, side) for tt in times]
            cands = [(fl(f, task, beams=a.beams)[0], j) for j, f in enumerate(frames)]
        texts = list(dict.fromkeys(c for c, _ in cands if c)) or [""]
        if it is not None and len(texts) > 1:
            # 場面のコマ全部との近さ（SigLIP の対数オッズの平均）が最大の文を選ぶ
            score = _logits(it, it.embed_images(frames), it.embed_texts(texts)).mean(0)
        else:
            score = np.zeros(len(texts))
        best = int(np.argmax(score))
        j = next((j for c, j in cands if c == texts[best]), 0)
        results.append(
            {
                "no": k + 1,
                "start": round(s, 3),
                "end": round(e, 3),
                "time": round(times[j], 3),
                "caption": texts[best],
                "candidates": [
                    {"text": tx, "siglip": round(float(sc), 3)}
                    for tx, sc in zip(texts, score, strict=True)
                ],
            }
        )
        head = (
            "画像"
            if info.is_image
            else f"場面 {k + 1}: {s:.2f}〜{e:.2f}秒（{times[j]:.2f}秒のコマ）"
        )
        core.log(f"  {head}: {texts[best]}")
        if k < _SHEET_MAX:
            cards.append(
                _card(frames[j], [head, texts[best]], width=300, size=14, max_h=400)
            )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "model": fl.repo,
            "task": task,
            "candidates": a.candidates,
            "rerank": "google/siglip2-base-patch16-224" if it is not None else None,
            "language": "en",
            "note": "文は英語。日本語の説明は、一覧画像を見て書く",
            "items": results,
        },
    )
    sheet = _grid(cards, cols=min(4, len(cards)))
    core.save_image(out, sheet)
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 18 動作認識 ───────────────────────────────────────────────────────────

# 既定の候補（日常の動作。英語。X-CLIP は英語で学習している）
_ACTIONS = [
    "walking", "running", "standing", "sitting", "talking", "laughing", "smiling",
    "waving hand", "clapping", "dancing", "jumping", "eating", "drinking", "cooking",
    "reading a book", "writing", "typing on a keyboard", "looking at a smartphone",
    "talking on the phone", "taking a selfie", "applying makeup", "brushing hair",
    "brushing teeth", "washing face", "hugging", "shaking hands", "riding a bicycle",
    "driving a car", "exercising", "playing guitar",
]  # fmt: skip


class _XClip:
    """X-CLIP（microsoft/xclip-base-patch32、MIT）で、8 コマの動画と動作の言葉を照らす。"""

    def __init__(self) -> None:
        from .models import hf_repo, torch_setup

        self.torch = torch_setup()
        from transformers import XCLIPModel, XCLIPProcessor

        repo = hf_repo("xclip")
        # transformers 5.2.0 の XCLIPProcessor(videos=...) は空を返す（2026-11-14 実測）ので、中の部品を直接使う
        proc = XCLIPProcessor.from_pretrained(repo)
        self.tok = proc.tokenizer
        self.img_proc = proc.image_processor
        self.net = XCLIPModel.from_pretrained(repo).eval()

    def logits(
        self, clips: list[list[np.ndarray]], labels: list[str], batch: int = 4
    ) -> np.ndarray:
        """clips（各 8 コマ、BGR）× labels の当てはまり（logit。softmax で確率）。"""
        tok = self.tok(labels, padding=True, return_tensors="pt")
        out = []
        for i in range(0, len(clips), batch):
            videos = [
                [cv2.cvtColor(f[:, :, :3], cv2.COLOR_BGR2RGB) for f in clip]
                for clip in clips[i : i + batch]
            ]
            px = self.img_proc(videos, return_tensors="pt")["pixel_values"]
            o = self.net(
                pixel_values=px,
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
            )
            out.append(o.logits_per_video.numpy().astype(np.float64))
        return np.concatenate(out)


def _fit_square(frame: np.ndarray, mode: str) -> np.ndarray:
    """縦長・横長のコマを X-CLIP に渡す形にする。crop＝そのまま（X-CLIP が中央を正方形に切る）、
    pad＝黒い余白で正方形に、squash＝縦横の比を変えて正方形に。"""
    if mode == "crop":
        return frame
    h, w = frame.shape[:2]
    s = max(h, w)
    if mode == "squash":
        return cv2.resize(frame, (s, s), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((s, s, 3), np.uint8)
    y, x = (s - h) // 2, (s - w) // 2
    canvas[y : y + h, x : x + w] = frame[:, :, :3]
    return canvas


@command(
    18,
    "action",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--labels",
        help="動作の候補（英語。「,」区切り）。省略時は日常の動作 30 種類。"
        "日本語の頼み（例: 手を振る）は Claude が英語（waving hand）に訳して渡す",
    ),
    arg(
        "--find",
        action="append",
        help="区間を探したい動作（英語。何度でも指定できる。候補に無ければ足す）",
    ),
    arg(
        "--window",
        type=float,
        default=2.0,
        help="1 回に見る長さ（秒。この中から等間隔に 8 コマ。既定 2.0）",
    ),
    arg("--step", type=float, default=0.5, help="窓をずらす間隔（秒。既定 0.5）"),
    arg(
        "--threshold",
        type=float,
        default=0.3,
        help="--find の動作とみなす確率（候補の中での割合）の下限（0〜1。既定 0.3）",
    ),
    arg(
        "--fit",
        choices=["crop", "pad", "squash"],
        default="crop",
        help="縦長・横長の映像の渡し方: crop＝中央を正方形に切る（既定）、pad＝黒い余白を足す、squash＝縦横の比を変える",
    ),
)
def action(a: argparse.Namespace) -> None:
    """短い窓（既定 2 秒・8 コマ）ごとに、候補の動作のどれに当たるかを X-CLIP で見分け、JSON・グラフ・一覧画像に書く。"""
    core.require(
        a.window > 0 and a.step > 0, "--window と --step は 0 より大きくしてください"
    )
    labels = _split_list(a.labels) if a.labels else list(_ACTIONS)
    find = [f.strip() for f in (a.find or []) if f.strip()]
    labels += [f for f in find if f not in labels]
    core.require(len(labels) >= 2, "候補は 2 つ以上にしてください")
    info = core.probe(a.src)
    rng = _rng(a)
    out = Path(a.out) if a.out else core.default_out(a.src, "action", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    nf = 8
    side = a.max_side or 320
    if info.is_image:
        img = _frame(a.src, info, 0.0, side)
        frames, fps_s, starts = [img] * nf, 1.0, [0]
        centers = np.zeros(1)
    else:
        fps_s = nf / a.window
        frames = core.read_all(a.src, fps=fps_s, max_side=side, **rng)
        n = len(frames)
        hop = max(1, round(a.step * fps_s))
        starts = list(range(0, max(n - nf, 0) + 1, hop))
        if n > nf and starts[-1] != n - nf:
            starts.append(n - nf)  # 最後の端も見る
        centers = rng["start"] + (np.array(starts) + min(n, nf) / 2) / fps_s
    clips = []
    for s in starts:
        if len(frames) >= s + nf:
            idx = list(range(s, s + nf))
        else:
            idx = np.linspace(0, len(frames) - 1, nf).round().astype(int).tolist()
        clips.append([_fit_square(frames[i], a.fit) for i in idx])
    core.log(f"{len(clips)} 個の窓 × 候補 {len(labels)} 個を X-CLIP で照らします")
    xc = _cached("xclip", _XClip)
    probs = _softmax(xc.logits(clips, labels), axis=1)
    top1 = probs.argmax(axis=1)
    half = a.step / 2
    t_lo = rng["start"]
    t_hi = 0.0 if info.is_image else _end(info, rng, fps_s)

    def span(i0: int, i1: int) -> tuple[float, float]:
        if info.is_image:
            return 0.0, 0.0
        lo = t_lo if i0 == 0 else max(t_lo, float(centers[i0]) - half)
        hi = t_hi if i1 == len(centers) - 1 else min(t_hi, float(centers[i1]) + half)
        return round(lo, 3), round(hi, 3)

    segments = []
    for i0, i1 in _runs_same(top1):
        lo, hi = span(i0, i1)
        lab = labels[int(top1[i0])]
        segments.append(
            {
                "label": lab,
                "start": lo,
                "end": hi,
                "mean_prob": round(float(probs[i0 : i1 + 1, int(top1[i0])].mean()), 4),
            }
        )
    found = []
    for f in find:
        j = labels.index(f)
        col = probs[:, j]
        ivs = []
        for i0, i1 in _runs(col >= a.threshold, gap=1):
            lo, hi = span(i0, i1)
            k = i0 + int(np.argmax(col[i0 : i1 + 1]))
            ivs.append(
                {
                    "start": lo,
                    "end": hi,
                    "peak_time": round(float(centers[k]), 3),
                    "peak_prob": round(float(col[k]), 4),
                }
            )
        best = int(np.argmax(col))
        found.append(
            {
                "label": f,
                "found": bool(ivs),
                "best_time": round(float(centers[best]), 3),
                "best_prob": round(float(col[best]), 4),
                "intervals": ivs,
            }
        )
    windows = []
    for i, s in enumerate(starts):
        order = np.argsort(-probs[i])[:5]
        windows.append(
            {
                "center": round(float(centers[i]), 3),
                "start": round(float(centers[i]) - a.window / 2, 3)
                if not info.is_image
                else 0.0,
                "end": round(float(centers[i]) + a.window / 2, 3)
                if not info.is_image
                else 0.0,
                "top": [[labels[j], round(float(probs[i, j]), 4)] for j in order],
            }
        )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "model": "microsoft/xclip-base-patch32",
            "window": a.window,
            "step": a.step,
            "fit": a.fit,
            "threshold": a.threshold,
            "labels": labels,
            "find": found,
            "segments": segments,
            "windows": windows,
            "probs": [[round(float(v), 4) for v in row] for row in probs],
        },
    )
    if info.is_image:
        order = np.argsort(-probs[0])[:5]
        lines = [f"{labels[j]}  {probs[0, j]:.3f}" for j in order]
        core.log("画像（動きは見ていない。様子からの推定）: " + "、".join(lines))
        sheet = _card(
            frames[0], ["画像（動きは見ない推定）", *lines], width=480, size=16
        )
        core.save_image(out, sheet)
        core.log(f"保存: {out}")
        core.log(f"保存: {json_path}")
        return
    core.log(f"いちばん当てはまる動作の流れ（{len(segments)} 区間）:")
    for sg in segments:
        core.log(
            f"  {sg['start']:6.2f}〜{sg['end']:6.2f} 秒: {sg['label']}（平均 {sg['mean_prob']:.2f}）"
        )
    for fr in found:
        if fr["found"]:
            core.log(f"「{fr['label']}」: {len(fr['intervals'])} 区間")
            for iv in fr["intervals"]:
                core.log(
                    f"  {iv['start']:6.2f}〜{iv['end']:6.2f} 秒（最大 {iv['peak_prob']:.2f}、{iv['peak_time']:.1f} 秒）"
                )
        else:
            core.log(
                f"「{fr['label']}」: 見つかりませんでした（最大 {fr['best_prob']:.2f}、{fr['best_time']:.1f} 秒 < 基準 {a.threshold}）"
            )
    # 確認画像: グラフ（探した動作、無ければ平均の高い 4 つ）・いちばん当てはまる動作の帯・区間のコマ
    show = [labels.index(f) for f in find] or list(np.argsort(-probs.mean(axis=0))[:4])
    series = [
        (probs[:, j], _COLORS[n % len(_COLORS)], labels[j]) for n, j in enumerate(show)
    ]
    spans = [
        (iv["start"], iv["end"], _COLORS[n % len(_COLORS)])
        for n, fr in enumerate(found)
        for iv in fr["intervals"]
    ]
    chart = _chart(
        centers,
        series,
        title="候補の中での確率（窓の中心の時刻）。帯＝探した動作の区間（線と同じ色）",
        ylim=(0, 1),
        spans=spans,
        hlines=[(a.threshold, _RED, f"基準 {a.threshold}")] if find else [],
        xlim=(t_lo, t_hi),
    )
    strip = _label_strip(segments, labels, t_lo, t_hi)
    cells, cell_labels = [], []
    for sg in segments[:_SHEET_MAX]:
        mid = (sg["start"] + sg["end"]) / 2
        i = int(np.clip(round((mid - rng["start"]) * fps_s), 0, len(frames) - 1))
        cells.append(frames[i])
        cell_labels.append(f"{sg['start']:.1f}〜{sg['end']:.1f}秒 {sg['label']}")
    sheet = _stack(
        [
            chart,
            strip,
            core.contact_sheet(
                cells, cell_labels, cols=min(6, len(cells)), cell_width=200
            ),
        ]
    )
    core.save_image(out, sheet)
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


def _runs_same(values: np.ndarray) -> list[tuple[int, int]]:
    """同じ値が続く所を (最初, 最後) の組にする。"""
    out = []
    lo = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[lo]:
            out.append((lo, i - 1))
            lo = i
    return out


def _label_strip(
    segments: list[dict[str, Any]], labels: list[str], t0: float, t1: float
) -> np.ndarray:
    """いちばん当てはまる動作を、時間の帯に色分けして描く。"""
    width, left, right = 1200, 64, 14
    pw = width - left - right
    names = list(dict.fromkeys(sg["label"] for sg in segments))
    color = {n: _COLORS[k % len(_COLORS)] for k, n in enumerate(names)}
    legend_rows = math.ceil(len(names) / 4)
    img = np.full((54 + 22 * legend_rows, width, 3), 255, np.uint8)
    core.draw_text(
        img, "いちばん当てはまる動作", (left, 4), size=14, color=(20, 20, 20), bg=None
    )
    span = max(t1 - t0, 1e-3)
    for sg in segments:
        x0 = left + round((sg["start"] - t0) / span * pw)
        x1 = left + round((sg["end"] - t0) / span * pw)
        cv2.rectangle(img, (x0, 26), (max(x1, x0 + 1), 46), color[sg["label"]], -1)
    for k, n in enumerate(names):
        x = left + (k % 4) * 280
        y = 54 + (k // 4) * 22
        cv2.rectangle(img, (x, y + 3), (x + 16, y + 15), color[n], -1)
        core.draw_text(img, n, (x + 22, y), size=13, color=(40, 40, 40), bg=None)
    return img


# ── 80 ハイライト検出 ─────────────────────────────────────────────────────


def _loudness(
    src: str, start: float, duration: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """ebur128 の瞬時ラウドネス（M。0.4 秒の窓、0.1 秒ごと。LUFS）と、その窓の中心の時刻。"""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.4f}"]
    cmd += ["-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration:.4f}"]
    cmd += [
        "-vn",
        "-af",
        "ebur128=metadata=1,ametadata=mode=print:key=lavfi.r128.M:file=-",
        "-f",
        "null",
        "-",
    ]
    text = core.run(cmd).stdout
    ts, vs = [], []
    t: float | None = None
    for line in text.splitlines():
        if "pts_time:" in line:
            t = float(line.rsplit("pts_time:", 1)[1].split()[0])
        elif line.startswith("lavfi.r128.M=") and t is not None:
            v = float(line.split("=", 1)[1])
            ts.append(start + t - 0.2)
            vs.append(max(v, -70.0) if math.isfinite(v) else -70.0)
    return np.array(ts), np.array(vs)


def _rank01(x: np.ndarray) -> np.ndarray:
    """値の順位を 0〜1 にする（同じ値は順位の平均）。"""
    n = len(x)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return ranks / max(n - 1, 1)


def _bin_stat(
    t: np.ndarray, v: np.ndarray, edges: np.ndarray, how: str = "mean"
) -> np.ndarray:
    """時刻 t の値 v を、区切り edges の区間ごとにまとめる（値が無い区間は nan）。"""
    out = np.full(len(edges) - 1, np.nan)
    k = np.searchsorted(edges, t, side="right") - 1
    for b in range(len(out)):
        sel = v[(k == b) & np.isfinite(v)]
        if len(sel):
            out[b] = float(np.median(sel) if how == "median" else np.mean(sel))
    return out


def _fill_nan(v: np.ndarray) -> np.ndarray:
    """nan を前後の値でうめる。"""
    v = v.copy()
    ok = np.isfinite(v)
    if not ok.any():
        return np.zeros_like(v)
    idx = np.arange(len(v))
    v[~ok] = np.interp(idx[~ok], idx[ok], v[ok])
    return v


_HL_PARTS = {
    "audio": ("音の大きさ", 3.0),  # 名前, 「変化が小さい」とみなす幅（p95−p5。dB）
    "motion": ("動きの量", 0.5),  # 前のコマとの明るさの差（0〜255）
    "cuts": ("カットの密度", 0.5),  # 窓の中のカットの数
    "query": ("言葉との近さ", 1.0),  # SigLIP の対数オッズ
}


def _weights(text: str) -> dict[str, float]:
    w = {k: 1.0 for k in _HL_PARTS}
    for item in _split_list(text):
        core.require("=" in item, f"--weights は 名前=数 の形で: {item}")
        k, v = item.split("=", 1)
        core.require(k.strip() in w, f"知らない名前です: {k}（{' / '.join(w)}）")
        w[k.strip()] = float(v)
    return w


def _make_reel(
    src: str, segs: list[tuple[float, float]], out: Path, audio: bool
) -> Path:
    """区間をつないだ MP4 を作る（音のつなぎ目は 0.03 秒で溶かして、ぷつっと鳴らないようにする）。"""
    fc = []
    for k, (s, e) in enumerate(segs):
        fc.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{k}]")
        if audio:
            d = max(e - s, 0.07)
            fc.append(
                f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:d=0.03,afade=t=out:st={d - 0.03:.3f}:d=0.03[a{k}]"
            )
    ins = "".join(f"[v{k}]" + (f"[a{k}]" if audio else "") for k in range(len(segs)))
    fc.append(
        f"{ins}concat=n={len(segs)}:v=1:a={int(audio)}[v]" + ("[a]" if audio else "")
    )
    cmd = ["-i", str(src), "-filter_complex", ";".join(fc), "-map", "[v]"]
    cmd += ["-map", "[a]"] if audio else []
    cmd += core.MP4_ARGS + (["-c:a", "aac", "-b:a", "192k"] if audio else [])
    out.parent.mkdir(parents=True, exist_ok=True)
    core.ffmpeg([*cmd, str(out)])
    return out


@command(
    80,
    "highlights",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg("--top", type=int, default=3, help="選ぶ区間の数（既定 3）"),
    arg("--length", type=float, default=3.0, help="1 区間の長さ（秒。既定 3）"),
    arg("--bin", type=float, default=0.25, help="点数を出す間隔（秒。既定 0.25）"),
    arg(
        "--weights",
        default="audio=1,motion=1,cuts=1,query=1",
        help="点数の重み（audio＝音の大きさ、motion＝動きの量、cuts＝カットの密度、query＝言葉との近さ。"
        "既定はすべて 1。0 で使わない）",
    ),
    arg(
        "-q",
        "--query",
        help="見せ場を言葉で（例: 笑顔、商品のアップ）。SigLIP 2 で近さを点数に足す（省略時は使わない）",
    ),
    arg(
        "--snap",
        type=float,
        default=0.5,
        help="区間の端からこの秒数以内にカットがあれば、端をカットに合わせる（0 で合わせない。既定 0.5）",
    ),
    arg(
        "--reel", action="store_true", help="選んだ区間を時間の順につないだ MP4 も作る"
    ),
)
def highlights(a: argparse.Namespace) -> None:
    """音の大きさ・動きの量・カットの密度（と言葉との近さ）から盛り上がりの点数を出し、上位の区間を JSON・グラフに書く。"""
    core.require(
        a.top >= 1 and a.length > 0 and a.bin > 0,
        "--top・--length・--bin を確かめてください",
    )
    info = core.probe(a.src)
    _need_video(info, "ハイライト検出")
    rng = _rng(a)
    weights = _weights(a.weights)
    out = Path(a.out) if a.out else core.default_out(a.src, "highlights", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    t0, t1 = rng["start"], _end(info, rng)
    edges = np.arange(t0, t1 + 1e-9, a.bin)
    if edges[-1] < t1 - 1e-6:
        edges = np.append(edges, t1)
    centers = (edges[:-1] + edges[1:]) / 2
    nb = len(centers)
    raw: dict[str, np.ndarray] = {}
    # 音の大きさ
    if info.has_audio and weights["audio"] > 0:
        lt, lv = _loudness(a.src, t0, t1 - t0)
        if len(lt):
            raw["audio"] = _fill_nan(_bin_stat(lt, lv, edges))
    # 動きの量（前のコマとの差。カットの 1 コマの跳ねは区間の中央値で抑える）
    if weights["motion"] > 0:
        diffs, times, prev = [], [], None
        for i, f in enumerate(core.read_frames(a.src, max_side=160, **rng)):
            g = core.to_gray(f).astype(np.float32)
            if prev is not None:
                diffs.append(float(np.abs(g - prev).mean()))
                times.append(t0 + i / info.fps)
            prev = g
        if diffs:
            raw["motion"] = _fill_nan(
                _bin_stat(np.array(times), np.array(diffs), edges, how="median")
            )
    # カットの密度（76 と同じ方法でカットを見つけ、区間の長さの窓の中の数を数える）
    from .scenes import detect_cuts

    res = detect_cuts(str(a.src), **rng)
    cut_times = np.array([res.t(c) for c in res.cuts])
    if weights["cuts"] > 0:
        raw["cuts"] = np.array(
            [
                float(np.sum(np.abs(cut_times - c) <= a.length / 2))
                if len(cut_times)
                else 0.0
                for c in centers
            ]
        )
    # 言葉との近さ
    if a.query and weights["query"] > 0:
        step = max(a.bin, 0.5)
        frames, ft = _sample(a.src, info, step, _span_rng(rng, t1), a.max_side or 384)
        it = _siglip()
        lg = _logits(it, it.embed_images(frames), _text_emb(it, [a.query]))[:, 0]
        raw["query"] = np.interp(centers, ft + step / 2, lg)
    used, dropped = {}, {}
    for k, v in raw.items():
        spread = float(np.percentile(v, 95) - np.percentile(v, 5))
        if spread < _HL_PARTS[k][1]:
            dropped[k] = f"変化が小さい（幅 {spread:.2f}）"
            continue
        used[k] = _rank01(v)
    if info.has_audio is False:
        dropped["audio"] = "音声が無い"
    core.require(
        bool(used), "点数に使える材料がありません（音も動きもカットも変化が小さい）"
    )
    wsum = sum(weights[k] for k in used)
    score = sum(weights[k] * used[k] for k in used) / wsum
    # 区間の点数: 区間（--length）の中の点数の平均
    nw = max(1, round(a.length / a.bin))
    kernel = np.ones(nw) / nw
    seg_score = np.convolve(score, kernel, mode="same")
    half = a.length / 2
    valid = (centers - half >= t0 - 1e-6) & (centers + half <= t1 + 1e-6)
    if not valid.any():
        valid[:] = True
    picks: list[tuple[float, float, float]] = []
    taken = np.zeros(nb, bool)
    for _ in range(a.top):
        cand = np.where(valid & ~taken, seg_score, -np.inf)
        k = int(np.argmax(cand))
        if not np.isfinite(cand[k]):
            break
        s, e = max(t0, centers[k] - half), min(t1, centers[k] + half)
        if a.snap > 0 and len(cut_times):
            near = cut_times[np.abs(cut_times - s) <= a.snap]
            if len(near):
                s = float(near[np.argmin(np.abs(near - s))])
            near = cut_times[np.abs(cut_times - e) <= a.snap]
            if len(near):
                e = float(near[np.argmin(np.abs(near - e))])
        picks.append((round(s, 3), round(e, 3), float(seg_score[k])))
        taken |= np.abs(centers - centers[k]) < a.length  # 重ならないように
    picks.sort()
    segs = []
    for s, e, sc in picks:
        sel = (centers >= s) & (centers < e)
        parts = {k: round(float(used[k][sel].mean()), 3) for k in used if sel.any()}
        raw_means = {k: round(float(raw[k][sel].mean()), 3) for k in raw if sel.any()}
        segs.append(
            {
                "start": s,
                "end": e,
                "score": round(sc, 4),
                "parts": parts,
                "raw": raw_means,
            }
        )
    reel_path = None
    if a.reel and segs:
        reel_path = out.with_suffix(".mp4")
        _make_reel(
            a.src, [(sg["start"], sg["end"]) for sg in segs], reel_path, info.has_audio
        )
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "bin": a.bin,
            "length": a.length,
            "weights": {k: weights[k] for k in used},
            "used": {k: _HL_PARTS[k][0] for k in used},
            "dropped": {k: f"{_HL_PARTS[k][0]}: {v}" for k, v in dropped.items()},
            "query": a.query,
            "cuts": [round(float(c), 3) for c in cut_times],
            "segments": segs,
            "reel": str(reel_path) if reel_path else None,
            "times": [round(float(c), 3) for c in centers],
            "score": [round(float(v), 4) for v in score],
            "parts": {k: [round(float(x), 4) for x in v] for k, v in used.items()},
            "raw": {k: [round(float(x), 4) for x in v] for k, v in raw.items()},
        },
    )
    core.log("使った材料: " + "、".join(_HL_PARTS[k][0] for k in used))
    for k, v in dropped.items():
        core.log(f"使わなかった材料: {_HL_PARTS[k][0]}（{v}）")
    core.log(f"盛り上がりの上位 {len(segs)} 区間（時間の順）:")
    for sg in segs:
        detail = "、".join(f"{_HL_PARTS[k][0]} {v:.2f}" for k, v in sg["parts"].items())
        core.log(
            f"  {sg['start']:6.2f}〜{sg['end']:6.2f} 秒  点数 {sg['score']:.2f}（{detail}）"
        )
    # 確認画像: 材料ごとの順位・合計の点数・選んだ区間と、区間のコマ
    colors = {"audio": _BLUE, "motion": _ORANGE, "cuts": _PURPLE, "query": _TEAL}
    spans = [(sg["start"], sg["end"], _GREEN) for sg in segs]
    chart1 = _chart(
        centers,
        [(used[k], colors[k], _HL_PARTS[k][0]) for k in used],
        title="材料ごとの順位（0〜1。動画の中で大きいほど 1）",
        ylim=(0, 1),
        spans=spans,
        marks=[(c, (190, 190, 190)) for c in cut_times],
    )
    chart2 = _chart(
        centers,
        [(score, (60, 60, 60), "合計"), (seg_score, _RED, f"{a.length:g} 秒の平均")],
        title="盛り上がりの点数。緑＝選んだ区間、灰色の縦線＝カット",
        ylim=(0, 1),
        spans=spans,
        marks=[(c, (190, 190, 190)) for c in cut_times],
    )
    cells, labels = [], []
    for n, sg in enumerate(segs):
        for name, tt in (
            ("始め", sg["start"] + 0.1),
            ("中", (sg["start"] + sg["end"]) / 2),
            ("終わり", sg["end"] - 0.1),
        ):
            cells.append(_frame(a.src, info, tt, 360))
            labels.append(f"{n + 1}{name} {tt:.1f}秒")
    rows = [chart1, chart2]
    if cells:
        rows.append(
            core.contact_sheet(cells, labels, cols=min(9, len(cells)), cell_width=200)
        )
    core.save_image(out, _stack(rows))
    if reel_path:
        core.log(f"保存: {reel_path}")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 87 異常検知 ───────────────────────────────────────────────────────────

# 特徴 → (名前, ばらつきの下限)。ばらつきが下限より小さい時は下限で割る（静かな映像の小さな揺れで騒がない）
_AN_FEATS = {
    # 動きは log(1 + 速さ/0.5px) で比べる（速く動く場面の中の揺らぎより、静かな所の急な動きを重く見る）
    "motion": ("速い動き", 0.1),  # 動きの速さの上位 5% の値（px/コマ、幅 256px で）
    "shake": ("画面全体の動き", 0.1),  # 動きの中央値（px/コマ）
    "brightness": ("明るさ", 0.75),  # 明るさの平均（0〜255）
    "look": ("色と形", 1.0),  # 16×16 の Lab の小さな画の差
    "content": ("内容", 0.01),  # SigLIP 2 の特徴量の差（1 − cos）
}
_AN_KIND = {
    "motion": "動きが急に速い・遅い",
    "shake": "画面全体が急に動く（揺れ・飛び）",
    "brightness": "明るさが急に変わる（光る・暗くなる）",
    "look": "見た目が前後と違う（別の画が混ざる・乱れる）",
    "content": "映っている物が前後と違う",
    "insert": "前後とつながらない短い差し込み",
}


def _two_sided(
    x: np.ndarray, scene: np.ndarray, w: int, kind: str = "scalar"
) -> np.ndarray:
    """各点と、同じ場面の前と後ろ（各 w 点）の中央値との差を求め、いちばん小さい物を返す。

    比べる相手は 前の中央値・後ろの中央値・その 2 つの真ん中（なだらかに変わる所では差が小さい）。
    カット（変わってそのまま続く）は片側と、なだらかな変化は真ん中と似ているので小さく、差し込み・
    一瞬の光・揺れのように前後のどちらとも違う点だけが大きくなる。場面の端で片側しか無い時は、
    すぐ隣の 2 点とも比べる（場面の始めの数コマだけが違う段差を数えない）。前後とも 3 点未満なら nan。
    kind: scalar＝差の絶対値、look＝_look_dist（小さな画の違い）、cosine＝1 − cos。
    """

    def dist(a: np.ndarray, b: np.ndarray) -> float:
        if kind == "scalar":
            return abs(float(a - b))
        if kind == "look":
            return _look_dist(a, b)
        return 1.0 - float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        )

    n = len(x)
    good = np.all(np.isfinite(x.reshape(n, -1)), axis=1)
    dev = np.full(n, np.nan)
    for i in range(n):
        if not good[i]:
            continue
        refs = []
        for lo, hi in ((max(i - w, 0), i), (i + 1, min(i + w + 1, n))):
            idx = [j for j in range(lo, hi) if good[j] and scene[j] == scene[i]]
            refs.append(np.median(x[idx], axis=0) if len(idx) >= 3 else None)
        cands = [dist(x[i], r) for r in refs if r is not None]
        if not cands:
            continue
        if refs[0] is not None and refs[1] is not None:
            cands.append(dist(x[i], (refs[0] + refs[1]) / 2))
        else:
            side = range(i + 1, min(i + 3, n)) if refs[0] is None else range(i - 2, i)
            near = [j for j in side if 0 <= j and good[j] and scene[j] == scene[i]]
            if near:
                cands.append(dist(x[i], np.median(x[near], axis=0)))
        dev[i] = min(cands)
    return dev


def _look_dist(a: np.ndarray, b: np.ndarray) -> float:
    """小さな画（16×16 の Lab）どうしの違い: 色の成分ごとに画素の差の中央値をとり、その最大。

    数人が歩くような一部の変化には鈍く、画面全体が変わる（別の画が混ざる・光る）と大きくなる。
    """
    d = np.abs(a.reshape(-1, 3) - b.reshape(-1, 3))
    return float(np.median(d, axis=0).max())


def _robust_z(dev: np.ndarray, floor: float) -> tuple[np.ndarray, float]:
    """ずれ ÷ ばらつき（1.4826 × ずれの中央値。下限 floor）。ばらつきも返す。"""
    med = float(np.nanmedian(dev)) if np.isfinite(dev).any() else 0.0
    sigma = max(1.4826 * med, floor)
    return dev / sigma, sigma


@command(
    87,
    "anomalies",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--z",
        type=float,
        default=8.0,
        help="異常とみなす外れ具合（頑健な z 値 = 前後とのずれ ÷ 動画全体のふだんのずれ。既定 8）",
    ),
    arg(
        "--window",
        type=float,
        default=2.0,
        help="比べる前後の長さ（秒。既定 2。これより長い変化は「異常」でなく流れとみなす）",
    ),
    arg(
        "--fps",
        type=float,
        help="調べるコマの数（毎秒。省略時は元のまま。長い動画は 10 などに下げると速い）",
    ),
    arg(
        "--look-step",
        type=float,
        default=0.25,
        help="SigLIP 2 で内容を見る間隔（秒。既定 0.25）",
    ),
    arg(
        "--no-ai",
        action="store_true",
        help="SigLIP 2 を使わない（速い。内容の違いは色と形の小さな画だけで見る）",
    ),
    arg(
        "--max-insert",
        type=float,
        default=1.0,
        help="この長さ以下の場面で、前後の場面どうしがつながっている物を「差し込み」とする（秒。既定 1.0。0 で見ない）",
    ),
)
def anomalies(a: argparse.Namespace) -> None:
    """コマごとの動き・明るさ・見た目・内容が前後と比べて急に外れる所を探し、区間を JSON に、グラフとコマを JPG に書く。"""
    core.require(
        a.z > 0 and a.window > 0, "--z と --window は 0 より大きくしてください"
    )
    info = core.probe(a.src)
    _need_video(info, "異常検知")
    from .ai import optical_flow

    rng = _rng(a)
    out = Path(a.out) if a.out else core.default_out(a.src, "anomalies", ".jpg")
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    fps = a.fps or info.fps
    aw = 256
    ah = max(2, round(info.height * aw / info.width / 2) * 2)
    feats: dict[str, list[Any]] = {
        k: [] for k in ("motion", "shake", "brightness", "look")
    }
    prev = None
    frames = core.read_frames(a.src, fps=a.fps, size=(aw, ah), **rng)
    for f in frames:
        g = core.to_gray(f)
        feats["brightness"].append(float(g.mean()))
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        feats["look"].append(
            cv2.resize(lab, (16, 16), interpolation=cv2.INTER_AREA)
            .astype(np.float32)
            .ravel()
        )
        if prev is None:
            feats["motion"].append(np.nan)
            feats["shake"].append(np.nan)
        else:
            flow = optical_flow(prev, g)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            feats["motion"].append(float(np.percentile(mag, 95)))
            feats["shake"].append(
                float(np.hypot(np.median(flow[..., 0]), np.median(flow[..., 1])))
            )
        prev = g
    n = len(feats["brightness"])
    core.require(n >= 8, "コマが少なすぎます（8 コマ以上の動画にしてください）")
    t = rng["start"] + np.arange(n) / fps
    t_end = min(_end(info, rng, fps), float(t[-1]) + 1 / fps)
    # 場面（76 と同じ方法）。カットの直後のコマの動きは、前の場面との差なので数えない
    from .scenes import detect_cuts

    res = detect_cuts(str(a.src), **rng)
    bounds = [res.t(s) for s, _ in res.scenes()]
    # 区切りの秒は 4 桁に丸めてあるので、半コマの余裕を持たせて場面に振り分ける
    half = 0.5 / info.fps
    scene = np.searchsorted(np.array(bounds), t + half, side="right") - 1
    first = np.r_[True, scene[1:] != scene[:-1]]
    arrays = {
        "motion": np.array(feats["motion"], np.float64),
        "shake": np.array(feats["shake"], np.float64),
        "brightness": np.array(feats["brightness"], np.float64),
        "look": np.stack(feats["look"]).astype(np.float64),
    }
    for k in ("motion", "shake"):
        arrays[k][first] = np.nan
        arrays[k] = np.log1p(arrays[k] / 0.5)
    w = max(3, round(a.window * fps))
    z: dict[str, np.ndarray] = {}
    sigma: dict[str, float] = {}
    for k, x in arrays.items():
        dev = _two_sided(x, scene, w, "look" if k == "look" else "scalar")
        z[k], sigma[k] = _robust_z(dev, _AN_FEATS[k][1])
    content_t = None
    if not a.no_ai:
        cf, content_t = _sample(
            a.src, info, a.look_step, _span_rng(rng, t_end), a.max_side or 384
        )
        cf = [f for f, tt in zip(cf, content_t, strict=True) if tt < t_end]
        content_t = content_t[: len(cf)]
        core.log(f"内容の変化を SigLIP 2 で見ます（{len(cf)} コマ）")
        emb = _siglip().embed_images(cf).astype(np.float64)
        cscene = np.searchsorted(np.array(bounds), content_t + half, side="right") - 1
        cw = max(3, round(a.window / a.look_step))
        cdev = _two_sided(emb, cscene, cw, "cosine")
        cz, sigma["content"] = _robust_z(cdev, _AN_FEATS["content"][1])
        # コマの時刻へ（いちばん近い SigLIP のコマの値）
        j = np.clip(
            np.round((t - content_t[0]) / a.look_step).astype(int), 0, len(cz) - 1
        )
        z["content"] = cz[j]
    names = list(z)
    zmat = np.stack([z[k] for k in names], axis=1)
    zmax = np.nanmax(np.where(np.isfinite(zmat), zmat, -np.inf), axis=1)
    zmax[~np.isfinite(zmax)] = 0.0
    flags = zmax >= a.z
    gap = max(1, round(0.15 * fps))
    events: list[dict[str, Any]] = []
    for i0, i1 in _runs(flags, gap=gap):
        k = i0 + int(np.argmax(zmax[i0 : i1 + 1]))
        per = {
            nm: round(float(np.nanmax(z[nm][i0 : i1 + 1])), 2)
            for nm in names
            if np.isfinite(z[nm][i0 : i1 + 1]).any()
        }
        main = max(per, key=lambda nm: per[nm])
        events.append(
            {
                "start": round(float(t[i0]), 3),
                "end": round(min(float(t[i1]) + 1 / fps, t_end), 3),
                "peak_time": round(float(t[k]), 3),
                "peak_z": round(float(zmax[k]), 2),
                "main": main,
                "kind": _AN_KIND[main],
                "z": per,
            }
        )
    # 短い差し込み: 前後の場面どうしは似ていて、間の短い場面だけが違う
    look = arrays["look"]
    scs = res.scenes()
    if a.max_insert > 0:
        for m in range(1, len(scs) - 1):
            s_t, e_t = res.t(scs[m][0]), res.t(scs[m][1] + 1)
            if e_t - s_t > a.max_insert + 1e-6:
                continue
            idx_p = np.flatnonzero(scene == m - 1)[-3:]
            idx_m = np.flatnonzero(scene == m)
            idx_n = np.flatnonzero(scene == m + 1)[:3]
            if not (len(idx_p) and len(idx_m) and len(idx_n)):
                continue
            vp, vm, vn = (np.median(look[ix], axis=0) for ix in (idx_p, idx_m, idx_n))
            d_pn = _look_dist(vp, vn)
            d_in = min(_look_dist(vp, vm), _look_dist(vm, vn))
            if d_pn < 0.5 * d_in:
                zi = d_in / sigma["look"]
                if zi < a.z:
                    continue
                # 違いのほとんどが色の全体のずれ（明るさ）なら、光った・暗くなった物とする
                diff = (vm - vp).reshape(-1, 3)
                shape = float(
                    np.median(np.abs(diff - np.median(diff, axis=0)), axis=0).max()
                )
                main = "brightness" if shape < 0.35 * d_in else "insert"
                kind = _AN_KIND[main]
                if main == "brightness":
                    kind += "（前後とつながらない短い場面）"
                events.append(
                    {
                        "start": round(s_t, 3),
                        "end": round(e_t, 3),
                        "peak_time": round((s_t + e_t) / 2, 3),
                        "peak_z": round(zi, 2),
                        "main": main,
                        "kind": kind,
                        "z": {"look": round(zi, 2)},
                        "note": f"前後の場面どうしの差 {d_pn:.1f}、この場面との差 {d_in:.1f}"
                        f"（形の差 {shape:.1f}）",
                    }
                )
    events = _merge_events(events)
    core.write_json(
        json_path,
        {
            "source": str(a.src),
            "fps": fps,
            "z_threshold": a.z,
            "window": a.window,
            "features": {nm: _AN_FEATS[nm][0] for nm in names},
            "sigma": {nm: round(v, 5) for nm, v in sigma.items()},
            "scenes": len(scs),
            "events": events,
            "times": [round(float(v), 3) for v in t],
            "z": {
                nm: [None if not np.isfinite(v) else round(float(v), 2) for v in z[nm]]
                for nm in names
            },
        },
    )
    core.log(
        f"異常: {len(events)} 件（基準 z ≥ {a.z:g}、場面 {len(scs)} 個、{n} コマ）"
    )
    for ev in events:
        core.log(
            f"  {ev['start']:7.2f}〜{ev['end']:7.2f} 秒  z={ev['peak_z']:.1f}  {ev['kind']}"
        )
    # 確認画像: 特徴ごとの z のグラフと、各区間の前・山・後のコマ
    fcolor = {
        "motion": _ORANGE,
        "shake": _PURPLE,
        "brightness": _TEAL,
        "look": _BLUE,
        "content": _RED,
    }
    spans = [
        (ev["start"], max(ev["end"], ev["start"] + 1 / fps), _RED) for ev in events
    ]
    chart = _chart(
        t,
        [
            (
                np.clip(np.nan_to_num(z[nm], nan=0.0), 0.05, None),
                fcolor[nm],
                _AN_FEATS[nm][0],
            )
            for nm in names
        ],
        title="前後とのずれ（頑健な z 値、対数目盛り）。赤い帯＝異常、灰色の縦線＝カット",
        log=True,
        ylim=(0.1, max(100.0, float(np.nanmax(zmax)) * 1.5)),
        spans=spans,
        hlines=[(a.z, _RED, f"基準 {a.z:g}")],
        marks=[(b, (190, 190, 190)) for b in bounds[1:]],
    )
    cells, labels = [], []
    for n_ev, ev in enumerate(events[:8]):
        for name, tt in (
            ("前", ev["start"] - 0.4),
            ("山", ev["peak_time"]),
            ("後", ev["end"] + 0.3),
        ):
            cells.append(_frame(a.src, info, tt, 320))
            labels.append(f"{n_ev + 1}{name} {tt:.2f}秒")
    rows = [chart]
    if cells:
        rows.append(core.contact_sheet(cells, labels, cols=9, cell_width=200))
    core.save_image(out, _stack(rows))
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


def _merge_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """重なる区間をまとめる。時間の順に並べる。

    種類は z の大きい方を残すが、短い場面の決まり（差し込み・光る。note つき）で見つけた物があれば、
    そちらの種類を優先する（場面の境目の動きの乱れより、何が起きたかを正しく表すため）。
    """
    events = sorted(events, key=lambda ev: ev["start"])
    out: list[dict[str, Any]] = []
    for ev in events:
        if (
            out and ev["start"] <= out[-1]["end"] + 0.15
        ):  # 0.15 秒以内に続く物もまとめる
            last = out[-1]
            last["end"] = max(last["end"], ev["end"])
            for k, v in ev["z"].items():
                last["z"][k] = max(last["z"].get(k, 0.0), v)
            rule_ev, rule_last = "note" in ev, "note" in last
            if ev["peak_z"] > last["peak_z"]:
                last["peak_time"], last["peak_z"] = ev["peak_time"], ev["peak_z"]
                if rule_ev or not rule_last:
                    last["main"], last["kind"] = ev["main"], ev["kind"]
            elif rule_ev and not rule_last:
                last["main"], last["kind"] = ev["main"], ev["kind"]
            if rule_ev:
                last["note"] = ev["note"]
            continue
        out.append(dict(ev, z=dict(ev["z"])))
    return out
