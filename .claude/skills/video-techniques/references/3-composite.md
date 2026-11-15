# 3/4 切り抜き・合成（人や物を切り抜いて、消して、重ねる）

`uv run python -m vfx <コマンド> --help` で引数の詳細が出る（tools/ で実行）。

## 51. セグメンテーション（`segment`） ◎

- 何をする: 画素ごとに、人・物・背景などの種類に分ける
- 頼み方の例: 「画面の中の人の部分を塗り分けて」
- 中身: 3 つのモデルから選ぶ。scene（既定）: UperNet ConvNeXt small の ADE20K 150 種類（openmmlab/upernet-convnext-small、MIT。空・壁・床・木・道路・人・車など。名前は日本語に）。縦横比を保って短い辺 512px で計算し（32 の倍数に詰める）、1/4 の大きさの確率から、出てきた種類だけを拡大して塗り分ける。person: MediaPipe の selfie_segmenter（Apache-2.0。人物だけ・速い）。parts: MediaPipe の selfie_multiclass（Apache-2.0。背景・髪・体の肌・顔の肌・服・小物）。動画は、前のコマの確率を DIS オプティカルフローで動かして今のコマと混ぜ（--smooth 0.5。動きの推定が外れた所は混ぜない）、ちらつきを抑える。出力: 色分けの重ね表示（MP4／JPG。凡例は日本語名、画像は面積の % も）、--cutout（選んだ種類だけを残した透過 WebM／PNG。確率の和をガイデッドフィルターで元の画の輪郭に合わせる）、--masks（種類ごとの白黒 PNG／MP4）、JSON（コマごとの面積と平均）
- 実測: 3 人を貼った合成（正解: 人の地図）の人の IoU: scene 0.980（--cutout。種類の地図では 0.962）・person 0.970・parts 0.950（背景以外すべて）。--cutout のガイデッドフィルターを半径 = 短い辺の 1/70・eps 0.01 にすると、境目の IoU（r=3px）が scene 0.689 → 0.742・person 0.642 → 0.687（3 人の合成と動く合成 4 コマ）。動く合成（正解アルファ）: scene の人のマスク動画 12 コマで IoU 平均 0.988（最小 0.985）、person の切り抜き 48 コマで 0.976（最小 0.914。髪がなびくコマで下がる）。ちらつき（固定カメラの vtest 30 コマで、画がほとんど変わらない画素の種類が隣のコマで変わった割合）: --smooth 0 で 1.14%、0.5 で 0.75%、0.8 で 0.48%。動く合成の人の IoU は 0.9866・0.9865・0.9858 で、なめらかさによる遅れは見られなかった。home.jpg: 空 52.6%・建物 47.4%（73 の DETR と空の IoU 0.96・建物 0.979）。salon.webp（解析のみ）: カーテン 69%（店の糸のカーテン。正しい）・壁 13%・床 13%・天井 3%・ひじ掛け椅子 2%。story のカフェ: 人 51%・木 37%・台座 10%（テーブルを「台座」と誤る）・椅子 1.4%。速さ（540×960）: scene 1 コマ 1.8〜2.2 秒（他の作業が重い時 5.8〜6.4 秒）・person 0.015 秒・parts 0.1 秒。メモリ（scene）約 1.5GB
- 限界・注意: scene の種類は ADE20K の 150 種類で、似た物を取り違えることがある（カフェのテーブル → 台座）。人の輪郭はどのモデルも髪の毛までは追わない（3 人の合成の境目の IoU〔r=3px〕で --cutout が scene 0.71・person 0.66・parts 0.69）。髪の毛まで要る時は 56。scene は 1 コマ約 2 秒なので、15 秒・24fps の動画で約 12 分。動画の --masks は、--classes を省くと最初のコマに出た種類（最大 10）だけを書く（後から出てくる種類は書かれない）。parts はカメラに近い人向け（自撮り用のモデル）
- 確かめた素材と方法: make_synth.py の comp-multi・comp-move-clean（正解つき）、vtest 30 コマ（ちらつき）、home.jpg・salon.webp・story のカフェ（test_segment.py。目で確認: vfx-review/51-segment.jpg）

```bash
cd tools && uv run python -m vfx segment [-h] [-o OUT] [--json JSON] [--start START]
                             [--duration DURATION] [--max-frames MAX_FRAMES]
                             [--max-side MAX_SIDE]
                             [--model {scene,person,parts}]
                             [--classes CLASSES] [--cutout CUTOUT]
                             [--masks MASKS] [--smooth SMOOTH]
                             [--min-area MIN_AREA] [--opacity OPACITY]
                             src
```

## 52. クロマキー（`chroma-key`） ◎

- 何をする: 緑や青の単色の背景を抜き、別の背景にする
- 頼み方の例: 「グリーンバックで撮った人の背景を別の景色にして」
- 中身: 色差キー（Vlahos の差分マット。緑なら G − max(R, B)）を、場所ごとの背景の色で割る（Nuke の IBK と同じ考え方）。背景らしい画素（緑の強さと、その G に対する比が代表の色に近い）を種にして、人物の後ろの背景の色を push-pull（Gortler ら 1996）で埋め、大まかな背景で一度アルファを求めてほぼ透明な所を種にし直す 2 段で作る。アルファ = 1 − 緑の強さ ÷ その場所の背景の緑の強さ（黒 0.08・白 0.9 で切り詰め）。境目の帯だけ、すぐ内側の不透明な所の色と背景の色を結ぶ線への射影で求め直す（色差の式は肌・クリーム色の縁で濃く出るため）。前景の色は背景の色を差し引いて割り戻し（縁の緑が消える）、続けてスピル除去（57 の average）、半透明の縁は「緑を赤・青の小さい方より下げない色」と内側の色を半々に混ぜる（引きすぎの赤紫止め）。代表の色は画面の端 8% の帯から自動（緑か青か）か --color。--method chromakey / hsvkey で ffmpeg のフィルター（hsvkey は明るさを無視させる）。出力: 透過 WebM（VP9）・PNG、--bg で背景（画像・動画）に重ねた MP4・画像、--alpha-out（白黒）、--mask（62 のマスクで外す）、確認画像、JSON（コマごとの透明・半透明の割合）。外部のモデルは使わない
- 実測: 正解のアルファと前景の色がある合成グリーンバック（AI 生成の人物、むらのある緑〔明るさ 0.6〜1.15 倍・しわ・場所による色味〕、縁の強いスピル、ノイズ σ2、H.264 の 4:2:0。540×960・40 コマのうち 8 コマ）: IoU 0.9962（最小 0.9943）、アルファの平均誤差 0.79/255、境目の帯 29.1、story のカフェに重ねた色の差 1.63（帯 7.68）。射影なしは 0.9958・1.15・帯 32.0、場所ごとの背景を使わず代表の色 1 つで割ると 0.9886・6.37（背景を前景と誤る 1806px）。ブルーバック 0.9963・0.73。縁に元の撮影の明るさが残る版（green-raw）0.9949・0.94。ffmpeg chromakey は最良の 0.05/0.03 で 0.9923・1.83・帯 51.0（暗い隅を前景と誤る 390px）、0.08/0.05 で人物の 3415px が抜け、0.12 で人物がほぼ消える（IoU 0.07）。hsvkey は明るさを使うと最良 0.889、無視させて 0.03/0.02 で 0.9930・1.48。ブルーバックの 1 コマで chromakey は暗い左側の 13197px を前景と誤った（diff は誤差 0.53/255）。縁の色: 帯の緑の出っ張りはスピル除去なし 3.23 → 1.47（正解 1.39）。半透明の縁の赤紫（引きすぎ）は +0.34 → −0.04、帯の色の差 8.12 → 7.68。速さ: 540×960・40 コマで 8.6 秒（WebM の書き出しを含む）
- 限界・注意: スピルの強い縁は、透けているのか緑を浴びたのか色だけでは区別できず、アルファが薄く出る（強いスピルを足した合成で境目の帯の平均 −0.06〜−0.1。正解の前景の色を与えた射影なら誤差 0.7〜3/255 なので、限界は前景の色の推定）。薄い時は --clip-white を下げる。背景に近い緑・青の服や小物は抜ける（キーの原理）。1 本の髪の毛の細かさは、背景の色の推定（2px のぼかし）と 4:2:0 の色の解像度で丸まる。画面の端の 5% 未満しか緑・青でないと止まる（--color で指定）。照明スタンドなど緑でない物は前景に残る（62 で外す）。ffmpeg の方法は前景の色を直さず（スピル除去だけ掛ける）、明るさのむらに弱く、しきい値の幅も狭い。実写のグリーンバック素材は手元に無く、確かめたのは合成した素材だけ
- 確かめた素材と方法: make_green.py の green-clean・green-raw・blue-clean（正解つき）、test_chroma.py（方法と設定の比較）、diag_band*.py（境目の誤差の出どころ）、test_fringe.py（縁の色）、test_hsvkey.py、review_chroma.py（目で確認: vfx-review/52-chroma-key.jpg・52-chroma-key-blue.jpg）、CLI で WebM・--bg（story540 の動画）・--alpha-out・画像 1 枚

```bash
cd tools && uv run python -m vfx chroma-key [-h] [-o OUT] [--json JSON] [--start START]
                                [--duration DURATION]
                                [--max-frames MAX_FRAMES]
                                [--max-side MAX_SIDE] [--color COLOR]
                                [--method {diff,chromakey,hsvkey}]
                                [--clip-black CLIP_BLACK]
                                [--clip-white CLIP_WHITE]
                                [--despill {average,max,min,none}]
                                [--similarity SIMILARITY] [--blend BLEND]
                                [--bg BG] [--mask MASK]
                                [--alpha-out ALPHA_OUT]
                                src
```

## 53. ロトスコープ（`rotoscope`） ◎

- 何をする: コマごとに輪郭を取り、人や物を切り抜く
- 頼み方の例: 「人物の輪郭を全部のコマで切り抜いて」
- 中身: 毎コマ人物の地図を求め（selfie: MediaPipe の動画モード〔既定〕／ vitmatte: 56 と同じ髪まで細かい切り抜き／ birefnet: BiRefNet／ --mask: 60 などの白黒マスク）、前のコマの地図を DIS オプティカルフローで動かして混ぜ（--smooth 0.6）、ガイデッドフィルター（半径 = 短い辺の 1/70、eps 0.01）で元の画の境目に合わせてから 0.5 で切る。小さな塊と小さな穴を消し、縁を 1px ぼかす。出力: 透過 WebM（VP9）、確認 MP4（外を暗くし、輪郭をマゼンタの線で描く）、JSON（コマごとの輪郭の多角形。approxPolyDP で間引き、元の動画の座標、穴は hole=true）。画像は PNG・JPG・JSON
- 実測: 動く合成 48 コマ（正解アルファ）: 既定（selfie）IoU 平均 0.976（最小 0.915）・境目の IoU（r=3px）0.70・時間方向のふるえ 6.8×10⁻³。手順を比べると、地図を 0.5 で切っただけは IoU 0.972・境目 0.648・ふるえ 8.8×10⁻³、ガイデッド（半径 8・eps 0.01）を足すと境目 0.70・ふるえ 7.2×10⁻³（半径 4・eps 0.001 では 0.639 で悪くなる）、なめらか 0.6 を足すと 6.8×10⁻³（0.8 で 6.6×10⁻³）。--model vitmatte: IoU 0.993・境目 0.833・ふるえ 3.1×10⁻³（1 コマ 1.3 秒）。birefnet（20〜31 コマの 12 コマ）: IoU 0.985・境目 0.774（1 コマ 10.4 秒。同じコマで selfie は 0.946・0.662、vitmatte は 0.987・0.806）。JSON の多角形を塗り直すと WebM の 2 値の地図と IoU 0.998（最小 0.997。1 コマ平均 46 点）。HyperFrames の切り抜き（cut-*.webm）との一致: stand 50 コマ 0.973（最小 0.941）、laugh 45 コマ 0.928（最小 0.908。HF 側は机と画面の端の人まで含む）。60 のマスク動画を --mask で渡すと（--model vitmatte・8 コマ） IoU 0.992。速さ: selfie は 540×960・48 コマで約 5〜6 秒（書き出しを含む）
- 限界・注意: 既定の selfie は、風になびく髪の束を切り落とす（合成で IoU が 0.915 まで下がったコマ。--model vitmatte なら残る）。人以外は --mask（60 の出力）か --model birefnet（1 コマ約 10 秒・メモリ約 7GB）。出力は 1px ぼかしたくっきりした輪郭で、半透明の毛先は出さない（半透明が要る時は 56）。多角形はコマごとに独立で、コマ間の点の対応（After Effects のロトのようなキーフレームの曲線）は作らない
- 確かめた素材と方法: make_synth.py の comp-move-clean（正解つき。test_rotoscope.py・test_rotoscope_models.py）、plate-stand・plate-laugh と cut-*-540.webm、comp-move-crowd と 60 のマスク（test_chain.py）、画像 1 枚（comp-hair-street.png）。目で確認: vfx-review/53-rotoscope.jpg

```bash
cd tools && uv run python -m vfx rotoscope [-h] [-o OUT] [--json JSON] [--start START]
                               [--duration DURATION] [--max-frames MAX_FRAMES]
                               [--max-side MAX_SIDE]
                               [--model {selfie,vitmatte,birefnet}]
                               [--mask MASK] [--check CHECK] [--smooth SMOOTH]
                               [--edge {guided,none}] [--feather FEATHER]
                               [--keep {all,largest}] [--simplify SIMPLIFY]
                               src
```

## 54. インペインティング（`inpaint`） ◎

- 何をする: 消したい物の跡を、周りに合わせて埋める
- 頼み方の例: 「後ろを歩く人を消して、そこを背景で埋めて」
- 中身: LaMa（big-lama。advimman/lama の重み、Apache-2.0。Sanster/models の TorchScript 版。MD5 は lama-cleaner の値と一致）をtorch.jit で CPU 実行。穴のまとまりの周りだけ切り出し（余白 = 穴の大きさの 0.75 倍・最低 64px、長辺 1024px まで）、8 の倍数に詰めて計算し、穴の所だけ 1px なじませて貼る（LaMa は穴の外を変えない: 差の最大 0 を確認）。--method telea・ns は OpenCV。マスク: --mask（画像・マスク動画）・--box・--person（MediaPipe の人物の切り抜き）・--object（COCO 80 種類の検出の枠。fast / accurate）、--dilate で広げ、自動のマスクは前後 2 コマを足し合わせる（--mask-time）。動画は 3 通り: plate（固定カメラ。最大 80 コマの「穴でない時」の中央値で背景を作り、どのコマでも隠れていた所だけ LaMa で 1 回埋め、各コマは穴の周りの明るさの比（なめらかなゲインの地図）を合わせて貼る）・track（動くカメラ。隣どうしのコマの射影変換を、穴でない所の特徴点で求めてつなぎ、穴の画素ごとに「見えているいちばん近い時刻のコマ」から 1 回だけ読む。明るさは重なる所の画素ごとの比の中央値で合わせる。どのコマでも見えない所は前の結果を同じ変換で運び、最初だけ LaMa。端の数 px の細い隙間は Telea）・frame（コマごと）。auto はカメラの動き（5 コマの全体の動きの最大）が 3px 未満なら plate、それ以上なら track
- 実測: Blender の場面から猿の頭（画面の 9.3%）を消す（正解 = 猿の無い場面を描いた画）: 穴の PSNR lama 24.6dB・telea 15.7・ns 16.5、SSIM 0.86・0.67・0.67（960×540 で 5.7 秒）。実写・生成の 6 枚に既知の穴: 四角 10% で lama 19.4dB・telea 15.7・ns 15.4、筆の線 10% で 25.9・21.5・21.7（lama 1 枚 約 2.2 秒）。vtest（固定カメラ）に AI 生成の人物の切り抜きを貼って消す 60 コマ: plate 32.6dB（最低 27.0）、穴の中のコマ間の変化 0.30（正解 0.36）／ frame（LaMa）23.7dB・変化 7.23（ちらつく）／ telea 22.5dB。plate は 768×576・60 コマで 6 秒。Blender のカメラが横に動き玉が転がる 24 コマ（玉は画面の 0.6〜3.8%。正解 = 玉の無い場面）: track 25.3dB（最低 19.4）・誤差のちらつき 9.1 ／ frame 20.7dB・15.6 ／ plate 19.1dB。vtest の歩く人を --object 人 で 6 秒消す: 人の形が残ったコマ 60 コマ中 2（旧設定 score 0.4・前後の足し合わせ無しでは 13、--detector accurate では 0 だが 401 秒）。home.jpg の空の鳥 31 羽（空の切り抜きの中の暗い点をマスクに）: 鳥の所の暗さ（ブラックハット）の平均 14.3 → 5.1、マスクの外の変化 0 画素
- 限界・注意: LaMa は顔・文字・規則的でない大きな構造は作り直せない（顔の半分を隠した試験では背景で埋めた）。広い穴は細部がぼける。CPU で 960×540 の画 1 枚 2〜10 秒（混み具合でぶれる）、frame はコマ数 × それだけかかる。影・映り込みはマスクに入れない限り残る（--dilate で少し広げる）。--person は近くに写った人向き（遠くの小さな人は --object 人）。検出は柱の陰の人などを取りこぼし、その分は消え残る。plate は背景が止まっていて、穴の所が他のコマで一度は見えている時に向く（明るさの変化はゲインで合わせる）。track は遠い背景・地面のような平面に合う射影変換なので、穴の後ろが手前の立体（視差が大きい物）だと縁がずれる。オプティカルフローで穴の中の動きを補ってたどる方式も試したが、正解つきの試験で劣った（22〜23.7dB）。track は全コマを覚える
- 確かめた素材と方法: test_inpaint.py（still: Blender の猿／holes: home・building・fruits・story の 2 場面・vtest に四角と筆の線の穴／plate: vtest に cut-stand を貼る／track: Blender の anim 24 コマ／birds: home.jpg／real: vtest の人）、test_inpaint_people.py（検出の設定の比較）、exp_track.py（track の方式の比較）。目で確認: vfx-review/54-inpaint-*.jpg

```bash
cd tools && uv run python -m vfx inpaint [-h] [-o OUT] [--json JSON] [--start START]
                             [--duration DURATION] [--max-frames MAX_FRAMES]
                             [--max-side MAX_SIDE] [--mask MASK] [--box BOX]
                             [--person] [--object OBJECT]
                             [--detector {fast,accurate}] [--score SCORE]
                             [--mask-time MASK_TIME] [--dilate DILATE]
                             [--method {lama,telea,ns}]
                             [--video-mode {auto,frame,plate,track}]
                             [--window WINDOW] [--save-mask]
                             src
```

## 55. 深度推定（`depth`） ◎

- 何をする: 画面の各部分が手前か奥かを推定する
- 頼み方の例: 「この写真の手前と奥を白黒の濃淡で表して」
- 中身: Depth Anything V2 Small（depth-anything/Depth-Anything-V2-Small-hf、Apache-2.0。ai.DepthEstimator と同じモデル）の逆深度（手前ほど大きい）を、1〜99% 点で 0〜1 にして出す（手前が白。--far-white で逆）。--detail でモデルに入れる短い辺（既定 518px）。動画は時間方向に揃える: 前のコマの結果を DIS オプティカルフローで今のコマへ動かし、往復の動きが合う画素で今のコマの推定のスケールとずれを 1 次式で前の結果に合わせ（外れ値に強い 2 回の最小二乗）、その画素だけ前の結果と混ぜる（--smooth 0.5）。動きの合う画素が 3 割未満か相関 0.5 未満なら場面が変わったとして前の結果を捨てる。白〜黒の範囲は場面の頭で決め、その後は 1 コマ 5% ずつ追う。出力: 画像は灰色 PNG（8/16bit）・色分け PNG・確認 JPG・JSON（3×3 の区画の平均など）、動画は灰色・色分け・左右に並べた MP4（音声は残す）と JSON
- 実測: Blender の場面（正解の Z 深度。逆深度でスケールとずれを合わせて比べる。空は除く）: AbsRel 0.147・δ1（25% 以内）0.854・順位相関 0.994（比べる相手の「画面の下ほど手前」は 0.893・0.167・0.861）。--detail 364 で 0.144・0.913、700 で 0.132・0.938。空は予測でも最も遠い（空の平均 0.0002、空以外の 5% 点 0.045）。静止画に雑音と明るさの揺れを足した 48 コマ（正解は全コマ同じ）のコマ間の変化: コマごとに正規化 0.0057 → smooth 0 0.0042・0.5 0.0021・0.8 0.0012。カメラが動く Blender の 24 コマ（全コマ 1 組の合わせで比べる = スケールが揃っているか）: コマごとに正規化 AbsRel 0.165（最悪のコマ 0.306）・δ1 0.754・ちらつき 0.023 → smooth 0.5 で 0.157（最悪 0.180）・0.839・0.012（正解自体の変化 0.010）。smooth 0.8 は遅れて δ1 0.797 に下がる。story540 の 5.5〜9.5 秒: 場面の頭を 5.792・8.875 秒（本当のカット 5.79・8.88 秒）で検出。home.jpg: 空 0.0005・建物 0.835。story の人物と背景: 7 秒 0.72 / 0.08、10.5 秒 0.74 / 0.03、14 秒 0.24 / 0.01（手前の机がもっと近い）。速さ: 540×960 で1 コマ約 0.93 秒（96 コマ 89.7 秒）
- 限界・注意: 相対的な奥行き（距離の単位は無く、比も 1 次式の不定がある）。遠い所の差はつぶれる（Blender でも誤差は奥の建物に集中）。鏡・ガラス・空の映り込みは誤る（鏡の前の場面では人物 0.39・背景 0.34 と差が小さかった）。物の縁はなだらか（数 px）。動画のなめらかさを上げすぎると、速く動く物が遅れる（0.8 で精度が下がった）。表示の範囲がゆっくり追うので、場面の中でカメラが大きく寄ると明るさが徐々に変わる
- 確かめた素材と方法: test_depth.py（still: Blender の場面／noise: 雑音と明るさの揺れの 48 コマ／seq: Blender のカメラが動く 24 コマ／real: home.jpg・story540 の 5 場面と人物の切り抜き・mp-cat_and_dog・building）、CLI で story540 の 5.5〜9.5 秒（--style side）。目で確認: vfx-review/55-depth-*.jpg

```bash
cd tools && uv run python -m vfx depth [-h] [-o OUT] [--json JSON] [--start START]
                           [--duration DURATION] [--max-frames MAX_FRAMES]
                           [--max-side MAX_SIDE] [--style {gray,color,side}]
                           [--far-white] [--smooth SMOOTH] [--detail DETAIL]
                           [--bits {8,16}]
                           src
```

## 56. マッティング（`matte`） ◎

- 何をする: 髪の毛先のような細かい境目まで、透明度つきで切り抜く
- 頼み方の例: 「人物を髪の毛の先まで透過で切り抜いて」
- 中身: vitmatte（既定）: MediaPipe の人物と髪の地図から 3 値の地図（trimap）を作り（境目の内外 = 短い辺の 1.2%、地図が 0.1〜0.9 の自信のない所、髪の周り 10 倍の幅を「分からない」に）、ViTMatte small（hustvl/vitmatte-small-composition-1k、Apache-2.0）でアルファを求める。長い辺が 1280px を超える画は縮めて計算し、線形の拡大で元の大きさに戻す。続けて縁の色を推定し直す（Forte・Pitié 2021 の Blur-Fusion、2 段）。birefnet: BiRefNet lite の ONNX（rembg の配布、MIT。1024×1024 固定・ImageNet の正規化・onnxruntime の作業メモリの使い回しを切る）。birefnet-portrait: 同じく人物用（MIT）。guided: 人物の地図をガイデッドフィルターで整えるだけ。--mask で人以外も（60・53 の出力を渡せる）。出力: 透過 PNG（画像）／透過 WebM（動画）、--alpha-out（アルファだけ）、確認画像（元・アルファ・市松模様と緑に重ねた画・半透明が一番多い所の拡大）、JSON
- 実測: はねた毛の合成（正解アルファ。頭の周りで測る。数字は 空 / 街の背景）: vitmatte SAD 0.99 / 1.94・境目の帯の MSE 0.0018 / 0.0082・勾配の差 0.24 / 1.19・境目の IoU 0.84 / 0.80・毛のアルファの回収 85% / 78%（余計 9% / 22%）。髪の周りを広げる前（帯 6px だけ）は SAD 3.13 / 3.38・毛 19% / 22% で、MediaPipe の髪の地図は毛の画素の 2% 未満しか含まなかった。birefnet: SAD 1.96 / 2.74・境目 0.70 / 0.63・毛 79% / 50%。birefnet-portrait: SAD 1.83 / 2.91・毛 70% / 38%。guided: SAD 4.02 / 5.16（元の地図 4.10 / 5.03 と変わらない）。縁の色の推定: 別の背景に重ね直した時の色の差（0〜255、境目の帯）が、正解のアルファで 5.76→2.27 / 4.70→2.17、vitmatte のアルファで 6.68→3.32 / 6.67→4.37（推定は 1 コマ約 0.05 秒）。実写（AI 生成）の風になびく髪: vitmatte と birefnet は毛先まで残り、guided はにじむ（目で確認）。60 のマスクを --mask で渡した動画: IoU 0.993（8 コマ）。速さとメモリ: vitmatte 540×960 で 1 コマ 1.0〜1.9 秒・約 1.4GB、1080×1920 で 3.9 秒・約 2.0GB（540 の結果と IoU 0.992）。birefnet 10〜11 秒・最大 6.7GB（作業メモリの使い回しを入れたままだと 2 回目で 12GB）、birefnet-portrait 17〜18 秒・最大 8.0GB（入れたままでは途中でメモリ不足で止められた）、guided 0.03 秒。1080×1920（はねた毛の合成の前景・アルファ・背景を 2 倍にして重ねた物。正解も 2 倍）で、縮めて計算した後の戻し方を比べた境目の帯の平均誤差（0〜255。空 / 街の背景）: 線形の拡大だけ 12.2 / 16.6、ガイデッドフィルター eps 0.01 で 14.0 / 18.8、以前の eps 0.0001（OpenCV 4.14 ではただの平均になる）で 19.3 / 22.3。2026-11-15 に線形の拡大だけに変えた
- 限界・注意: vitmatte の粗い地図は MediaPipe の人物なので、人以外は --mask が要る（人物の地図が椅子なども含むと、それも残る）。髪の周りを広く「分からない」にするので、背景に細い線（街灯・電線）が多いと少し拾う（街の背景で余計 22%。--hair-band を小さくすると減る）。動画はコマごとの計算（ちらつきは --smooth で抑える）。birefnet は 7〜8GB 使うので他の重い作業と同時に動かさない。合成の毛は黒い細い線で、実際の細い毛とは見え方が違う
- 確かめた素材と方法: make_synth.py の comp-hair-sky・comp-hair-street（正解のアルファと前景の色つき。test_matte.py・review_matte.py）、plate-stand（1.0・1.9 秒）・plate-cafe（1.2 秒）の実写（review_matte_real.py）、1080×1920 の画、動画（plate-laugh 3 コマ、comp-move-crowd と 60 のマスク）。目で確認: vfx-review/56-matte-*.jpg

```bash
cd tools && uv run python -m vfx matte [-h] [-o OUT] [--json JSON] [--start START]
                           [--duration DURATION] [--max-frames MAX_FRAMES]
                           [--max-side MAX_SIDE]
                           [--model {vitmatte,birefnet,birefnet-portrait,guided}]
                           [--mask MASK] [--band BAND] [--hair-band HAIR_BAND]
                           [--no-hair] [--no-fg] [--alpha-out ALPHA_OUT]
                           [--smooth SMOOTH] [--check CHECK]
                           src
```

## 57. スピル除去（`despill`） ◎

- 何をする: 背景の色が人の縁や肌に移ったのを取る
- 頼み方の例: 「グリーンバックの緑が髪や肌に映ったのを取って」
- 中身: 緑なら G を上限 L で切る: average L = (R+B)/2（既定。ffmpeg の despill の既定〔mix 0.5・expand 0〕と同じ式で、同じ画像で最大 1/255 の差）/ max L = max(R, B)（弱い。黄・水色を守る）/ min L = min(R, B)（強い）。青は R と G で同じ。--strength で効き具合、--keep-luma で消した分の明るさ（Rec.709 の係数）を 3 色に足し戻す、--mask で場所を限る。透過つきの入力（WebM・PNG）はアルファを保つ。出力: 入力と同じ形式、確認画像（元・結果・消した量の地図）、JSON
- 実測: 52 の合成（スピルの重みが分かっている）をスピル除去なしでキーした前景に掛け、正解の色（スピル前）と Lab の ΔE で比べた（不透明な所）: スピルの強い所（重み 0.15 以上）で なし 9.75 → average 4.15・max 6.80・min 7.46・average＋--keep-luma 6.04、ほぼ無い所で 3.06 → 2.60・2.89・14.76（min は肌を赤くする）・2.67。ブルーバックの強い所: 7.49 → 5.42・5.69・4.98・5.50。かぶりの無い色見本の変化（average / max / min）: 黄 48.7 / 0 / 91.9、水色 45.4 / 0 / 107.6、肌 0 / 0 / 14〜21、葉の緑 66.5 / 58.6 / 75.0、クリーム 4.7 / 0 / 16.4、白・灰・赤・紫は 0。速さ: 540×960 の透過 WebM 12 コマで 2.1 秒
- 限界・注意: 「緑が赤と青より多い」所を一律に直すので、本当に緑の物（葉・緑の服）は灰色になる（--mask で場所を限る）。average は黄色をオレンジに、水色を青に寄せる（黄・水色が大事なら max）。緑の光で一緒に足された赤・青の分は消えないので、かぶった所は少し明るいまま。--keep-luma はこの試験では正解から遠くなった（明るさを足すため）。実写のかぶりは試していない（合成した素材と色見本だけ）
- 確かめた素材と方法: test_despill.py（make_green.py の合成・色見本・ffmpeg despill との比較）、review_despill.py（目で確認: vfx-review/57-despill.jpg・57-despill-chart.jpg）、CLI で透過 WebM・普通の MP4（--method max）・画像

```bash
cd tools && uv run python -m vfx despill [-h] [-o OUT] [--json JSON] [--start START]
                             [--duration DURATION] [--max-frames MAX_FRAMES]
                             [--max-side MAX_SIDE] [--color {auto,green,blue}]
                             [--method {average,max,min}]
                             [--strength STRENGTH] [--keep-luma] [--mask MASK]
                             src
```

## 58. 背景ぼかし（`bg-blur`） ◎

- 何をする: 奥行きを使って、背景だけをぼかす
- 頼み方の例: 「人物はくっきりのまま、後ろだけぼかして」
- 中身: 55 の奥行き（ガイデッドフィルターで輪郭に沿わせる）と、ピントの奥行き（--focus auto: 人物がいれば人物の中央値、いなければ画面中央の 75% 点／ x,y の点／ 0〜1 の数）の差に比例して、ぼけの半径を決める（薄いレンズの CoC は逆深度の差に比例するので、相対的な逆深度でも形は正しい。--strength で差 1 の時の半径、既定は短い辺の 3%。--dof の幅はくっきり、--near で手前側の割合）。人物（MediaPipe の切り抜き）は必ずくっきり残す（--protect）。描き方: 半径を 1.5px ごとの層に分け（三角形の重みで隣と分け合い段差を出さない）、奥の層から円盤の形でぼかした色と覆う割合を重ね、最後に覆う割合で割る（ピントの合った物の色が背景ににじまず、手前のぼけは奥へ広がる）。光の足し合わせなので線形の色で計算。明るい点は少し持ち上げて丸い玉ボケに（--highlights）。動画は奥行きを時間方向に揃え、ピントの奥行きもゆっくり追う（場面の頭で取り直す）
- 実測: Blender の正解（Cycles の光学的な被写界深度。35mm・f/0.3・ピントは猿の頭 3.02m。背景のぼけの半径は最大 18.2px）: strength を正解の式（半径 = 55.1px·m × 逆深度の差）と推定の奥行きの傾きから決めると、全体の PSNR 33.3dB・SSIM 0.978（正解の奥行きを使うと 33.0・0.982）、猿以外を一様にぼかすと 31.3・0.967、ぼかさないと 24.9・0.776。奥行きで差が出る 3.5〜8m の所: 34.6dB（正解の奥行き 35.6、一様 30.1、ぼかさない 26.0）。ピントの所（猿）40.0dB。CLI も同じ値。実写（AI 生成）の人物の鮮明さ（ラプラシアンの分散）の前後の比 0.985〜1.005、背景は 0.027〜0.47（窓の場面 0.027・浴室 0.14・木の葉 0.31）。plate-laugh 45 コマの動画: 人物 1.005・背景 0.47、背景のコマ間の変化は元より増えない（−0.03）。髪の縁にハローは見えない（拡大で確認）。速さ: 960×540 の画像 8.7 秒、540×960 の動画 1 コマ約 3.3〜3.8 秒
- 限界・注意: 奥行きの推定が誤る所（ガラス・鏡・細い物）はぼけ方も誤る。人物の保護は MediaPipe の人物の切り抜きに頼るので、人が小さい・後ろ向きなど切り抜きが外れると縁がぼける。手前の物のぼけの後ろ（隠れていた背景）は作らない（色が伸びる）。画面の端でぼけの形が欠ける現象（口径食）・色収差は作らない。本物の玉ボケは明るさの記録の範囲を超えた光で出るので、白飛びした所は--highlights で近似するだけ。OpenCV 4.14 のガイデッドフィルターは 3 チャンネルの案内画で eps ≤ 2e-3 だと縁を保たない（段差の試験で確認。1e-2 を使う）
- 確かめた素材と方法: test_bgblur.py（blender: Blender の光学ぼけと比べる／real: story540 の 7・14 秒、plate-bath、plate-laugh の動画）。目で確認: vfx-review/58-bg-blur-*.jpg

```bash
cd tools && uv run python -m vfx bg-blur [-h] [-o OUT] [--json JSON] [--start START]
                             [--duration DURATION] [--max-frames MAX_FRAMES]
                             [--max-side MAX_SIDE] [--focus FOCUS]
                             [--strength STRENGTH] [--near NEAR] [--dof DOF]
                             [--protect {person,none}]
                             [--highlights HIGHLIGHTS] [--smooth SMOOTH]
                             [--save-map]
                             src
```

## 59. アウトペインティング（`outpaint`） ○

- 何をする: 画面の外側を、続きの絵として描き足す
- 頼み方の例: 「縦長の映像の左右を描き足して横長にして」
- 中身: 仕上がりの形（--aspect・--size・--pad、--align）に広げた画の、新しい所を埋める。lama: 新しい所をマスクにして LaMa で描く（計算は長辺 512px。--limit。--steps で外へ少しずつ）／ blur: ぼかして暗くした拡大で埋める（縦長を横長に置く定番）／ mirror: 折り返して写し、離れるほどぼかす。動画の lama は 54 と同じ仕組み: plate（固定カメラ。背景の中央値の画を 1 回だけ描き足して全コマで使い、明るさを合わせる）・track（動くカメラ。他のコマに写っている外側の景色を射影変換で運び、無い所は前の結果を運ぶ。最初だけ LaMa）・frame。auto はカメラの動きで選ぶ。有料の生成 API は使わない（使う場合は別の道具で）
- 実測: 6 枚（home・building・vtest・Blender は中央の 9:16 を、story の 2 場面は中央の正方形を切り出し、元の大きさまで描き足して元の画と比べる。新しい所は画面の 44〜58%）の平均: lama（512）PSNR 16.0dB（境目から 16px 以内 21.8・それより外 15.9）、lama（1024）15.1（境目 23.5・外 14.8）、3 回に分けると 12.4、blur 12.8（境目 13.2）、mirror 14.0（境目 18.6）。1 枚 約 4 秒。目で見ると、境目の近くは続きが自然、遠くはぼけた繰り返しになる（建物の塔はつぶれ、壁・木の葉・セーターは自然）。1024 では遠くが灰色の平らな面になりやすく、512 の方が模様が続いた。横に流した動画（Blender の場面を 9:16 の窓でパン、48 コマ → 16:9。正解 = 窓の外の本当の画）: track 33.9dB（最低 32.7。新しい所の 74% は他のコマの本当の画から）、plate 17.1、blur 11.2。story540 のカフェ 2.5 秒（手持ち）を 9:16 → 1:1: 60 コマ 34 秒、描き足した所のコマ間の変化 2.7（元の映像 6.6）
- 限界・注意: 広い所を LaMa で描くと、構造（建物・人）は作れず、ぼけた繰り返しになる（大きく描き足すほど目立つ）。人物の体が画面の端で切れていると、その先は描けない。動画の plate・track は描き足した所が「止まった背景」なので、端で動く物（手・隣の人の肩）は続かず、境目に段差が見えることがある。frame はちらつく。本格的な描き足し（拡散モデルの生成）より質は低い（この環境の CPU では動かさない）
- 確かめた素材と方法: test_outpaint.py（images: 6 枚の切り出しと元の画の比較／pan: Blender の場面のパン／story: story540 のカフェ）。目で確認: vfx-review/59-outpaint-*.jpg

```bash
cd tools && uv run python -m vfx outpaint [-h] [-o OUT] [--json JSON] [--start START]
                              [--duration DURATION] [--max-frames MAX_FRAMES]
                              [--max-side MAX_SIDE] [--aspect ASPECT]
                              [--size SIZE] [--pad PAD]
                              [--align {center,left,right,top,bottom}]
                              [--method {lama,blur,mirror}] [--steps STEPS]
                              [--limit LIMIT]
                              [--video-mode {auto,frame,plate,track}]
                              [--window WINDOW]
                              src
```

## 60. マスクトラッキング（`mask-track`） ◎

- 何をする: 最初に決めた切り抜きの範囲を、動きに合わせて追う
- 頼み方の例: 「最初のコマで囲んだ物を、最後まで切り抜き続けて」
- 中身: SAM 2.1 tiny（facebook/sam2.1-hiera-tiny、Apache-2.0）の動画版を transformers 5.2 の Sam2VideoModel で使う。1 コマずつ渡す逐次入力にし、覚える物を直近 16 コマと最初のコマに絞る（メモリが増え続けない）。最初のコマの指定: --box（枠）／ --point と --neg（点）／ --mask（白黒か透過つきの画像）／ 省略時は MediaPipe で一番大きい人物。出力のマスクはガイデッドフィルターで輪郭を合わせ、1px ぼかす。出力: 透過 WebM、確認 MP4（色・輪郭・枠・「見失った」）、--mask-out（白黒のマスク動画。53・56 の --mask に渡せる）、JSON（コマごとの有無・確からしさ・面積・枠）
- 実測: 人が歩く vtest の上で女性が動き、柱の後ろを通る合成 48 コマ（正解アルファ、最初のコマの正解の枠で指定）: IoU 平均 0.990（最小 0.974）、見失い 0、柱の所に残したマスク 0 画素。人のいない背景の合成（指定なし = 自動）: 0.992（最小 0.979）。vtest の実写で歩く 1 人を 60 コマ（6 秒）: 見失い 0、5 コマごとの Faster R-CNN の人の検出との枠の IoU 0.876〜0.951（平均 0.913）。story の手（点で指定・24 コマ、腕を上げて髪を結ぶ速い動き）: 見失わず、手と前腕を頭の上まで追った（目で確認）。透過つき PNG を --mask で渡した時: HyperFrames の切り抜きと IoU 0.987〜0.999（4 コマ）。速さ: 1 コマ 2.2〜5.4 秒（768×576・540×960。他の作業の重さで変わる）、メモリ約 1.5GB。ここまでの数値は 2026-11-15 の前の版。2026-11-15 に、0.5 で切る前のガイデッドフィルターの eps を 1e-3（OpenCV 4.14 の 3 チャンネルの案内画ではただの平均になる）から 1e-2 に変えた: 柱の合成 48 コマで IoU 0.9893 → 0.9917・境目の IoU（3px）0.801 → 0.842、自動の合成 24 コマで 0.9905 → 0.9919・0.716 → 0.753（ガイデッドフィルターなしは 0.9904・0.822 / 0.9911・0.735）
- 限界・注意: 遅い（10 秒・24fps の動画で 8〜20 分）。点の指定は SAM が「物」と考える範囲になる（手を指すと前腕まで入った。--neg か --box で絞る）。1 回に追えるのは 1 つ。画面から出て戻ってきた物を拾い直せるかは確かめていない。最初のコマ（--start の位置）に物が写っている必要がある
- 確かめた素材と方法: make_synth.py の comp-move-crowd・comp-move-clean（正解つき）、vtest（Faster R-CNN の検出と照合）、story540 の 4.0 秒からの手（test_masktrack.py）、plate-stand と透過 PNG の指定。目で確認: vfx-review/60-mask-track.jpg

```bash
cd tools && uv run python -m vfx mask-track [-h] [-o OUT] [--json JSON] [--start START]
                                [--duration DURATION]
                                [--max-frames MAX_FRAMES]
                                [--max-side MAX_SIDE] [--box BOX]
                                [--point POINT] [--neg NEG] [--mask MASK]
                                [--overlay OVERLAY] [--mask-out MASK_OUT]
                                [--feather FEATHER]
                                src
```

## 61. インスタンス分割（`instances`） ◎

- 何をする: 同じ種類の物でも、1 つずつ別に切り分ける
- 頼み方の例: 「写っている人を一人ずつ別々に切り抜いて」
- 中身: torchvision の Mask R-CNN v2（maskrcnn_resnet50_fpn_v2、COCO の 80 種類、BSD-3-Clause。重みは download.pytorch.org）。短い辺を 512〜800px にして計算し、確からしさ 0.6 以上を採用。同じ種類で、小さい方のマスクの 70% 以上が大きい方に含まれる物は外す（隠れた人を上半身と全身の 2 人に数えるのを防ぐ）。重ね表示は番号ごとの色と「人 #3 0.98」の札。動画は前のコマの枠との重なり（0.2 以上・同じ種類）で同じ番号を付け続ける。出力: 重ね表示（MP4／JPG）、JSON（番号・種類・日本語名・確からしさ・元の座標の枠・面積）、--masks（画像: 1 つずつの白黒マスクと透過 PNG）、--split N（動画: 番号ごとの透過 WebM）
- 実測: 3 人を一部重ねて貼った合成（正解: 1 人ずつの地図）: 3/3 人、1 人ずつの IoU 0.956・0.938・0.864（3 人目の正解には元の切り抜きの机が入っている）。ほかに誤り 5 件（机 → 本、街灯 → フリスビー、バン → トラック、ベンチ、車）。mp-cats_and_dogs.jpg: 猫 2・犬 2（正しい）。vtest 10 秒: 人 7/7（目で数えた 7 人）。含まれる物を外す前は、看板の後ろの人を 2 人に数えて 8 だった。story のカフェ: 人 1 は正しく、左端の暗い背景を人と誤った（0.75）。動画（vtest 30 コマ・7 人）: 番号 11 個、5 人は全コマ（29〜30）同じ番号、隠れた人は番号が変わった。速さ: 768×576 で 1 コマ 1.3 秒（他の作業が重い時 7.9 秒）、1200×600 の画像 2.2 秒、メモリ約 0.9GB
- 限界・注意: COCO の 80 種類だけ。遠くの小さな物は見逃す。物でない物を物と誤ることがある（上の例）。--score を上げると減る。動画の番号付けは前のコマとの重なりだけの簡易な方法で、隠れたりすれ違ったりすると番号が変わる（しっかり追うなら 15 multi-track）。マスクの輪郭は粗い（髪などは 56 の --mask に渡して整える）
- 確かめた素材と方法: make_synth.py の comp-multi（正解つき）、mp-cats_and_dogs.jpg、vtest（静止画と 30 コマの動画。--split 3）、story540 の 13 秒、salon.webp（解析のみ）（test_instances_panoptic.py。目で確認: vfx-review/61-instances.jpg）

```bash
cd tools && uv run python -m vfx instances [-h] [-o OUT] [--json JSON] [--start START]
                               [--duration DURATION] [--max-frames MAX_FRAMES]
                               [--max-side MAX_SIDE] [--score SCORE]
                               [--classes CLASSES] [--masks MASKS]
                               [--split SPLIT] [--opacity OPACITY]
                               src
```

## 62. ガベージマット（`garbage-matte`） ◎

- 何をする: 大まかな形で、要らない範囲を外す
- 頼み方の例: 「画面の端に写った機材を大まかに外して」
- 中身: 多角形・四角（元の映像の px か %）を残す（--keep）・外す（--drop）範囲にし、4×4 の細かさで塗って縮めた 0〜1 のマスクにする（縁の画素は半分の値。多角形は 1 つずつ塗って重ねる）。'@秒 ...; @秒 ...' で時間とともに直線で動かせる（前後は端の形のまま）。--spec の JSON でも指定できる。--feather で縁をぼかす（66 と同じ）、--invert。透過つきの入力はアルファに掛け、普通の映像は外した所を透明にした WebM・PNG。--mask-only でマスク（白 = 残す）だけ、--mask-out でマスクも書く（52・67・74 の --mask に渡せる。灰色のまま書くので白が 255 のまま）。確認画像は範囲の線（緑 = 残す・赤 = 外す）・マスク・結果
- 実測: 合成グリーンバックに不要物（左上の黒い旗と支柱〔動かない〕、右から左へ一定の速さで動くブームマイク）を置いた 40 コマ: 52 のキーだけでは不要物が前景に残り、人物との IoU 0.9328（不要物の残り 1 コマ平均 22314px）。動かない多角形 1 つと、2 つの時刻で動かす四角 1 つで外すと IoU 0.9964・アルファの平均誤差 0.83/255・不要物の残り 0px（全コマ）。マスクを先に作って 52 の --mask に渡しても 0.9964・0.84。塗りの正確さ（面積の差）: 端数の位置の 100×50 の四角 +0.013%、三角 +0.002%、30 度回した正方形 +0.042%、3.5×2.5px の小さな四角 +0.07px²。動く四角の左端（4 つの時刻）は期待より +0.13px。試験の途中で直した不具合: 多角形をまとめて塗ると重なった所が穴になり不要物の 9% が残った、マスクの MP4 の白が 250 になり人物が 2% 透けた。速さ: 540×960・40 コマの WebM で 3.3 秒
- 限界・注意: 形は手で指定する（自動では追わない。動く物は 2 つ以上の時刻で形を決め、間は直線で動く。曲がった動きは時刻を増やす）。どの時刻も同じ数の点にする。外す範囲に人物が入ると人物も消える（52 の確認画像を見て範囲を決める）。物に沿って自動で追うなら 60（mask-track）
- 確かめた素材と方法: make_green.py の green-stand（不要物の地図つき）、test_garbage.py（不要物の残り・人物の IoU・塗りの面積・動く形）、review_garbage.py（目で確認: vfx-review/62-garbage-matte.jpg）、画像 1 枚（--drop-box の %）

```bash
cd tools && uv run python -m vfx garbage-matte [-h] [-o OUT] [--json JSON] [--start START]
                                   [--duration DURATION]
                                   [--max-frames MAX_FRAMES]
                                   [--max-side MAX_SIDE] [--keep KEEP]
                                   [--drop DROP] [--keep-box KEEP_BOX]
                                   [--drop-box DROP_BOX] [--spec SPEC]
                                   [--feather FEATHER] [--invert]
                                   [--mask-only] [--mask-out MASK_OUT]
                                   src
```

## 63. リライティング（`relight`） 未

- 何をする: 合成する人物の明るさや影を、背景の光に合わせる
- 頼み方の例: 「合成した人物に当たる光の向きを、背景に合わせて」

## 64. 色合わせ（`color-match`） 未

- 何をする: 合成した物の色合いを背景にそろえる
- 頼み方の例: 「あとから合成した人の色を、背景の色合いにそろえて」

## 65. 接地影（`contact-shadow`） 未

- 何をする: 合成した物の足元に影を付け、地面に置く
- 頼み方の例: 「合成した商品の下に影を付けて、浮かないようにして」

## 66. フェザー（`feather`） ◎

- 何をする: 切り抜きの縁をぼかして、背景になじませる
- 頼み方の例: 「切り抜いた縁が硬いので、少しぼかしてなじませて」
- 中身: アルファの縁を明るさのモルフォロジー（楕円の最大・最小。端数は前後の 2 つの大きさを混ぜる）で縮める（--choke）・広げる（--spread）、ガウスでぼかす（--blur は 10〜90% の幅 W。σ = W ÷ 2.563）。--where で置き場所: center = 元の縁が 50%、inside = 元の縁で 10%（ぼかしを内側に収め、外の背景がはみ出さない）、outside = 元の縁で 90%。透過つき WebM・PNG はアルファを直し（広げた所の色は内側の色を埋め広げる）、透過の無い入力は白黒のマスクとして直す。--bg で背景に重ねた映像も
- 実測: 2 値の円（半径 150）・長方形で、元の縁からの距離ごとのアルファの平均から求めた位置（円 / 長方形）: blur 4・10・20 の 10〜90% の幅 3.70・9.52・19.53 / 4.14・10.05・20.03、50% のずれ 0.13px 以内。inside（blur 10）: 10% が +0.02 / +0.15px、50% が −4.87 / −4.88（期待 0・−5）。outside: 90% が +0.12 / +0.10、50% が +4.82 / +5.12。choke 5 → 50% が −4.62 / −4.88、choke 2.5 → −2.38 / −2.38、spread 7 → +6.88 / +7.12、choke 3＋blur 6 → −2.66 / −2.88・幅 5.97 / 6.09。半透明のある実際のアルファ（AI 生成の人物の切り抜き）: choke 2・spread 3・choke 1.5 で 0.5 の輪郭が平均 −1.87・+2.87・−1.32px 動いた。速さ: 1080×1920 で choke 10 が 1 コマ 21ms、blur 20 が 12ms
- 限界・注意: 楕円の形のモルフォロジーなので、曲がった縁の縮み・広がりは 0.1〜0.4px 小さく出る（円で choke 5 → 4.62）。端数（2.5 など）は前後の大きさを混ぜるだけなので、2 値の縁には 1px の段がつく。縮めると細い所（毛先など）から先に消える（モルフォロジーの性質）。広げた所の色は内側の色を埋め広げた物で、本当の色ではない
- 確かめた素材と方法: test_feather.py（2 値の円・長方形の正解の縁で 10・50・90% の位置、cut-stand の正解のアルファ、CLI の PNG・WebM・--bg、速さ）、review_feather.py（目で確認: vfx-review/66-feather.jpg）

```bash
cd tools && uv run python -m vfx feather [-h] [-o OUT] [--json JSON] [--start START]
                             [--duration DURATION] [--max-frames MAX_FRAMES]
                             [--max-side MAX_SIDE] [--blur BLUR]
                             [--choke CHOKE] [--spread SPREAD]
                             [--where {center,inside,outside}] [--bg BG]
                             src
```

## 67. ルミナンスキー（`luma-key`） ◎

- 何をする: 明るさで、透明にする所を決めて切り抜く
- 頼み方の例: 「黒い背景で撮った物を、明るさで抜いて」
- 中身: 黒い背景（既定）: 値 = 3 色の最大（--channel max。色の濃い火花も明るく数える）、アルファ = (値 − low) ÷ (high − low)（既定 0.04〜1.0。明るさに比例して透ける）。白い背景（--invert）: 値 = 1 − 3 色の最小。前景の色は背景の黒（白）を差し引いて割り戻す（乗算済みの色として扱う。縁の黒ずみ・白っぽさが出ない）。--keep-color で元の色のまま、--channel luma で輝度（ffmpeg の lumakey と同じ考え方）。出力は 52 と同じ（透過 WebM・PNG、--bg、--alpha-out、--mask、確認画像、JSON）
- 実測: 黒地に手続き的に作った火花（橙。3 色の最大が 1）と煙（明るい灰。最大 0.82）（正解のアルファと乗算済みの色。ノイズ σ1・H.264。540×960・48 コマのうち 8 コマ）を story のカフェに重ね、正解の合成と比べた色の差（0〜255）: 既定 火花 8.34・煙 4.85、low 0 で 7.96・3.92、high 0.82（煙の本当の明るさ）で煙 2.65・火花 9.51、luma で火花 15.84（橙を暗く数える）、--keep-color で 17.65・30.16（煙が灰色に濁り、火花のにじみが 21 暗い）、ffmpeg lumakey（0/0.04/0.96）で 21.77・32.92（にじみが 22 暗い）。黒が 0.03 浮いてノイズ σ4 の素材では、素材の無い所の背景の濁りが low 0 で 3.84、0.04 で 0.86、0.06 で 0.14。白地の墨（濃さ 0.15 の灰）: 既定 5.28、low 0・high 0.85（素材に合わせた値）で 1.11、--keep-color で 30.13。速さ: 540×960・48 コマ（WebM）で 6.8 秒
- 限界・注意: 明るさから透け具合を決めるので、素材の本当の濃さは分からない（灰色の煙は最大 0.82 の明るさなので、既定では正解より薄く重なる。濃くしたい時は --high を素材のいちばん明るい所の値に下げる）。黒い煙・影など暗い部分は黒い背景からは抜けない。low を上げると薄い煙の端が欠ける。白地の素材の明るい部分（白い服など）は透ける。実写の火花・煙の素材は試していない（手続き的に作った素材だけ）
- 確かめた素材と方法: make_luma.py の fx-black・ink-white（正解つき）、test_luma.py（方法と設定・ffmpeg lumakey・黒の浮き）、review_luma.py（目で確認: vfx-review/67-luma-key.jpg）、CLI で WebM・--bg（MP4）・--invert・画像 1 枚

```bash
cd tools && uv run python -m vfx luma-key [-h] [-o OUT] [--json JSON] [--start START]
                              [--duration DURATION] [--max-frames MAX_FRAMES]
                              [--max-side MAX_SIDE] [--low LOW] [--high HIGH]
                              [--invert] [--channel {auto,max,luma,min}]
                              [--keep-color] [--bg BG] [--mask MASK]
                              [--alpha-out ALPHA_OUT]
                              src
```

## 68. 2.5Dパララックス（`parallax`） ○

- 何をする: 写真を手前と奥の層に分け、カメラが動いたように見せる
- 頼み方の例: 「1 枚の写真を、奥行きのある動く映像にして」
- 中身: 55 の奥行き（輪郭に沿わせる）を逆深度 q（いちばん奥 = 1/--depth-range、手前 = 1）にし、多値の大津の方法で層（既定 3）に分ける（層の番号はメディアンで整え、縁の画素が中間の層の細い帯にならないようにする）。人物が写っていれば（人物の検出 ＋ MediaPipe の切り抜き）まとめて手前の層に。奥の層は、手前の層の後ろに隠れていた所を LaMa で 1 回だけ埋め（いちばん奥は全部）、手前の層は色の縁に沿う柔らかい切り抜きで少し外まで含める。カメラの動き（dolly・dolly-out・truck・pedestal・circle・dolly-zoom）はピンホールの式 p' = c + zoom·((p − c) − S·q)/(1 − tz·q) で各コマを描き直す: 出力の画素ごとに元の位置を反復で求め（手前寄り・奥寄りの 2 通りから始め、写し直して 0.7px 以内に戻る解の手前の方。戻らない所はその層では透明）、奥の層から重ねる。画面の外が見えないよう、全コマの端の動きから必要なだけ拡大する
- 実測: Blender の正解（カメラを本当に横 0.3m・0.6m、前へ 0.8m・1.6m 動かして描いた画。推定の奥行きは正解の逆深度へ 1 次式で合わせて本当の量で動かす）: 横 0.3m で PSNR 25.0dB・SSIM 0.853（1 層 24.6、正解の奥行き 30.9、画面全体の 2D の最良の拡大・移動 18.8、動かさない 16.9）、横 0.6m 22.0（2D 16.8）、前 0.8m 25.2（2D 18.7）、前 1.6m 21.4（2D 16.3）。反復の解を確かめる前は、層の中の段差（玉とその後ろの地面）で物が二重に写り、正解の奥行きでも 25.8dB だった（横 0.3m）。home.jpg・mp-cat_and_dog・story540 の人物で 3 秒（90 コマ）: 13〜38 秒、描き直しは 1 コマ約 0.1 秒、端を隠す拡大は 1.01〜1.12 倍。目で見ると、標準の大きさでは自然。dolly-zoom の最後で頭の上に細い筋、猫の耳の先に小さな点が残る
- 限界・注意: 写真 1 枚の奥行きなので、隠れていた所は LaMa の想像（大きく動かすほど見える）。層は奥行きのしきい値で分けるので、細い物（髪の毛先・耳の先・枝）は層の境目で欠けたり点が残ったりする（人物は切り抜きで補う）。奥行きの縁のずれで、動きが大きいと縁が裂ける（--amount を 1 以下に）。本格的な 3D 写真の手法（層ごとに色と奥行きを学習で埋める LDI）より簡易: 隠れた所の奥行きは近くの値を伸ばすだけ。--depth-range は写真に合わせて選ぶ（距離の比は推定できない）
- 確かめた素材と方法: test_parallax.py（blender: Blender の truck・dolly の正解の画／real: home.jpg の truck・dolly、mp-cat_and_dog の dolly・circle、story540 の 7 秒の dolly-zoom・truck）。目で確認: vfx-review/68-parallax-*.jpg

```bash
cd tools && uv run python -m vfx parallax [-h] [-o OUT] [--json JSON]
                              [--max-side MAX_SIDE]
                              [--move {dolly,dolly-out,truck,pedestal,circle,dolly-zoom}]
                              [--amount AMOUNT] [--seconds SECONDS]
                              [--fps FPS] [--layers LAYERS]
                              [--depth-range DEPTH_RANGE]
                              [--fill {lama,telea}] [--limit LIMIT]
                              [--person {auto,off}]
                              src
```

## 69. 空の置き換え（`sky-replace`） ◎

- 何をする: 空の部分だけを見つけて、別の空にする
- 頼み方の例: 「曇り空を青空に差し替えて」
- 中身: UperNet ConvNeXt small（openmmlab/upernet-convnext-small、MIT。ADE20K の「sky」）で空の確率を求め（短い辺 512px、1/4 の大きさの確率を広げる）、0.5 を中心に 6 倍に急にしてからガイデッドフィルター（半径 = 短い辺の 1/70、eps 1e-2）で輪郭に沿わせる。縁の画素は古い空の色を抜く: (I − a·B)/(1 − a)（B = 近くの確かな空の色の平均）を、a が大きいほど近くの前景の色の平均へ寄せて混ぜる。新しい空: --sky の画像・動画（覆う大きさに拡大・上寄せ）か、--preset（sunset・dusk・blue・night のグラデーション・横に長い雲・太陽の光・星を作る）。前景の色を新しい空へ寄せる（--harmonize: 線形の光で空の色の比を 0.6〜1.6 に抑えて k 乗、明るさは 0.3k 乗）。動画は前のコマの空の確率をオプティカルフローで運んで混ぜ（--smooth）、空でない所の全体の動き（相似変換）を積み重ねて、1.35 倍に作った空をずらす（--motion auto）
- 実測: Blender の場面（空の割合の正解は 3×3 の光線）: IoU 0.981・平均の誤差 0.012（確率そのまま 0.977・0.017、ガイデッドだけ 0.981・0.022）。AI 生成の人物の切り抜きを本物の空（home.jpg の空）に重ねた画（正解 = 1 − 人物の透過）: IoU 0.988・誤差 0.012、髪の縁の帯の誤差 0.089（確率そのまま 0.125）。新しい夕焼けへの置き換えを正解の割合で重ねた画と比べると、髪の縁の帯の PSNR 23.1dB（古い空の色を抜かないと 21.5、最初に試した「境目で空と前景の色の間に置く」＋そのまま抜く方式は 20.1）。home.jpg を 2 倍にして 512×384 の窓で右へ 1 コマ 4px パンした 30 コマ: 推定した動き −3.99px/コマ（誤差 0.01px）、新しい空は前のコマを 4px ずらした物と一致（差 0.066。動かさない場合の差 0.268。--motion none では逆に 0.055 と 0.416）、空の切り抜きはずれを戻して IoU 0.9997（最低 0.998）。home.jpg の空 52.9%・building.jpg 9.8%（木の枝の間も抜けた）。速さ: 960×540 で切り抜き 1.9 秒、画像は読み込みを含め 6〜7 秒、512×384 の動画 1 コマ約 2 秒
- 限界・注意: 空と判定されない物（遠い山・霧・水面の映り込み）は置き換えない。鳥・電線は空に含まれて消えることがある（home.jpg の鳥は一部が残り一部が消えた）。前景の色寄せは全体に一様（夕日の向き・影の方向は変えない）。night では建物が昼の明るさのまま（--harmonize 1 でも暗さは 0.3 乗しか追わない）。動画の空の動きは相似変換（パン・ズーム）だけで、空と景色の視差・雲の流れは作らない。UperNet は CPU で 1 コマ約 2 秒
- 確かめた素材と方法: test_sky.py（blender: Blender の空の割合／comp: cut-laugh を home.jpg の空に重ねた正解つきの画／pan: home.jpg のパン／real: home.jpg・building.jpg を 4 種類の空で）、exp_sky.py・exp_sky2.py（縁の作り方と色の抜き方の比較）。目で確認: vfx-review/69-sky-*.jpg

```bash
cd tools && uv run python -m vfx sky-replace [-h] [-o OUT] [--json JSON] [--start START]
                                 [--duration DURATION]
                                 [--max-frames MAX_FRAMES]
                                 [--max-side MAX_SIDE] [--sky SKY]
                                 [--preset {sunset,dusk,blue,night}]
                                 [--harmonize HARMONIZE]
                                 [--motion {auto,none}] [--smooth SMOOTH]
                                 [--save-mask]
                                 src
```

## 70. スクリーン合成（`screen-blend`） 未

- 何をする: 黒い背景の素材の、光の部分だけを重ねる
- 頼み方の例: 「黒い背景の光の粒の素材を、映像に重ねて」

## 71. ライトラップ（`light-wrap`） 未

- 何をする: 背景の光を人物の縁に少しにじませて、なじませる
- 頼み方の例: 「合成した人物の縁に、背景の明るさを少し回り込ませて」

## 72. モーションブラー付加（`motion-blur`） 未

- 何をする: 合成した物の動きに合ったぶれを付ける
- 頼み方の例: 「動かした文字や物に、速さに合ったぶれを付けて」

## 73. パノプティック分割（`panoptic`） ○

- 何をする: 物の 1 つずつと、空・道などの背景を、まとめて塗り分ける
- 頼み方の例: 「画面全体を、人・車・空・道などに塗り分けて」
- 中身: DETR（ResNet-50）のパノプティック分割（facebook/detr-resnet-50-panoptic、Apache-2.0）。COCO panoptic の 133 種類（物 80・背景 53。背景の番号と名前は detectron2 の一覧で確認し、日本語名を付けた）。同じ種類の背景はまとめる。transformers 5.2 は背骨の ResNet-50 を timm で作るが timm が無いので、同じ形の torchvision の ResNet-50 に同じ重みを入れて差し替えた（timm を一時的に入れた公式の読み込みと比べ、マスクの値の差は最大 0.004、結果の塊は同じ）。さらに採用する候補にだけマスクを計算する（100 個すべてだと 800px の画で 12.4GB 使った。結果は普通の計算と画素単位で一致）。出力: 凡例つきの重ね表示（背景は面積の %、物は数）、JSON（塊ごとの種類・物か背景か・確からしさ・面積・枠）、--masks
- 実測: 3 人の合成: 3/3 人、IoU 0.978・0.972・0.856。速くした計算と普通の計算の一致: home.jpg・合成とも塊の数が同じで画素の一致率 1.000。51（UperNet）との背景の一致: home.jpg の空 IoU 0.960・建物 0.979、合成の草 0.83・道路 0.68・建物 0.70。home.jpg: 空 52%・建物 47%・鳥（空を飛ぶ本物の鳥）。vtest 10 秒: 人 6/7（看板の後ろの人を見逃し）、--threshold 0.7 では 7/7 ＋誤り 1。mp-cats_and_dogs.jpg: 犬 2 匹を見逃した（0.7 でも同じ。0.5 では犬を猫と誤った）。salon.webp（解析のみ）: 糸のカーテンを床（82%）と誤った（51 はカーテンと正しく判定）。story のカフェ: 人・テーブル・椅子・木 41%。速さ: 540×960 で 1 コマ 0.56 秒、768×576 で 0.75 秒（他の作業が重い時 3.4 秒）、メモリ約 1GB
- 限界・注意: DETR（2020 年）は今の手法より精度が低い（犬を見逃す・糸のカーテンを床と誤る・草と道路の境目が粗い）。より新しい Mask2Former の公開の重みはライセンスが other で使えず、OneFormer（MIT）の大きいモデルは CPU では重すぎるので入れていない。背景の種類を細かく知りたい時は 51（150 種類）の方が当たる。物の番号はコマをまたいで追わない。既定の --threshold 0.85 は誤りを減らす側（0.7 で隠れた人も拾うが誤りが増える）
- 確かめた素材と方法: make_synth.py の comp-multi（正解つき）、home.jpg、vtest 10 秒、mp-cats_and_dogs.jpg、story540 のカフェ（静止画と 6 コマの動画）、salon.webp（解析のみ）、timm を使う公式の読み込みとの比較（uv run --with timm で一時的に）（test_instances_panoptic.py。目で確認: vfx-review/73-panoptic.jpg）

```bash
cd tools && uv run python -m vfx panoptic [-h] [-o OUT] [--json JSON] [--start START]
                              [--duration DURATION] [--max-frames MAX_FRAMES]
                              [--max-side MAX_SIDE] [--threshold THRESHOLD]
                              [--min-area MIN_AREA] [--masks MASKS]
                              [--opacity OPACITY]
                              src
```

## 74. 差分キー（`diff-key`） ◎

- 何をする: 人のいない背景の画と比べて、違う所を切り抜く
- 頼み方の例: 「誰もいない時の画と比べて、人だけを切り抜いて」
- 中身: プレート（--plate の画像。省略時は 75 と同じ中央値。--exclude-people で人を除いた中央値）とコマを少しぼかし、差の小さい半分の画素の色の比で露出のゆらぎを合わせ、Lab の色の差（ΔE76）を求める。しきい値 12（前後 8 の幅で 0〜1）で切り、開く・閉じる処理（短い辺の 1/270）と小さな塊・穴（画面の 0.05% 未満）の除去で整え、縁の ±2px（短い辺の 1/270）だけ色の差のなだらかな値を使う。影は既定で前景に含め、--shadows remove で Horprasert ら（1999）の考え方（背景を 0.35〜0.95 倍しただけで、色のずれ 6% 未満）で背景にする。出力は 52 と同じ（透過 WebM・PNG、--bg、--alpha-out、--mask、確認画像、JSON）。プレートを作った時は <出力>-plate.png も
- 実測: 固定カメラの広場の合成（vtest の中央値〔正解の背景〕に AI 生成の人物 4 人。1 人は 75% のコマで同じ所に立ち止まる。影 1 つ・露出のゆらぎ ±2.5%・ノイズ σ2・H.264。768×576・100 コマのうち 13 コマ。影の所を除き、全コマまとめた IoU）: 正解の背景をプレートにして IoU 0.963・アルファの平均誤差 0.72/255、75 の人を除いた中央値で 0.968・0.63、単純な中央値では立ち止まった人がプレートに入り 0.731（その人の 46% しか抜けない）。縁の作り方: 色の差のなだらかな値 0.87・帯 24.8（しきい値 14 で比較）、ガイデッドフィルター 1.91・帯 55.9、2 値 0.90。しきい値 10・12・14 で IoU 0.962・0.963・0.955（10 は背景の誤り、14 は抜けが増える）。--shadows remove: 影の 99.3% を背景にできたが、灰緑の服の立ち止まった人の 53% も影として消え IoU 0.809（このため既定は keep）。速さ: 768×576 で 1 コマ約 50ms（キーだけ）、vtest 100 コマ（--plate・--bg・--alpha-out）で 9.2 秒、広場の合成 100 コマ（--exclude-people でプレートから作る）で 23.7 秒
- 限界・注意: 固定カメラ専用（カメラが動くとプレートと合わない。先に手ぶれ補正で止める）。背景に近い色の服・物は抜ける（--threshold を下げる）。影は既定で前景になる（remove は無彩色の暗い服の人も消す）。照明が大きく変わる所・背景が動く所（木の揺れ・水面・画面）は前景に出る。半透明（毛先）は色の差の幅でしか表せない
- 確かめた素材と方法: make_plaza.py（正解の背景・人のアルファ・影の地図）、test_diffkey.py・scan_diffkey.py・test_diffkey_final.py（縁の作り方・しきい値・影・プレートの比較）、exp_shadow.py（影の判定）、vtest の実写（目で確認: vfx-review/74-diff-key.jpg）、画像 1 枚＋--plate

```bash
cd tools && uv run python -m vfx diff-key [-h] [-o OUT] [--json JSON] [--start START]
                              [--duration DURATION] [--max-frames MAX_FRAMES]
                              [--max-side MAX_SIDE] [--plate PLATE]
                              [--threshold THRESHOLD] [--softness SOFTNESS]
                              [--shadows {keep,remove}] [--clean CLEAN]
                              [--min-area MIN_AREA] [--edge EDGE]
                              [--samples SAMPLES] [--exclude-people] [--bg BG]
                              [--mask MASK] [--alpha-out ALPHA_OUT]
                              src
```

## 75. クリーンプレート（`clean-plate`） ◎

- 何をする: 通る人や物を除いた、背景だけの画を作る
- 頼み方の例: 「人が行き来する映像から、人のいない背景の画を作って」
- 中身: 固定カメラの映像の範囲から等間隔に --samples コマ（既定 61）を 1 回の読み込みで取り、画素ごとの中央値を求める（横長の帯に分けて計算し、メモリを抑える）。--exclude-people: 共有の detectors で人の枠を求め（既定 fast = MediaPipe の EfficientDet-Lite2〔Apache-2.0〕、accurate = torchvision の Faster R-CNN v2〔BSD-3-Clause〕）、枠を左右 8%・上 5%・下 12%（足元の影）広げた所を、その画素の中央値から除く（全部のコマで除かれた画素は除かずに取る）。--exclude-mask で人以外の物も除ける。出力: PNG、確認画像（取ったコマ・プレート・動きの多さの地図。半分以上のコマでプレートと 25 以上違う画素を赤）、JSON（取った時刻・確かめにくい所の割合）。--video: 74 と同じ色の差（しきい値 12・影も含める）で動く物を求め、短い辺の 1/120 広げてぼかした所を、明るさを合わせたプレートで埋めた MP4（人を消した映像）
- 実測: 広場の合成（正解の背景つき。1 人が 75% のコマで同じ所に立ち止まる。768×576・100 コマ）: 単純な中央値は正解の背景との差 平均 4.48・10 を超える画素 2.17%・立ち止まった人の所 114.1（人が残る）。--exclude-people（fast）は 2.19・0.00%・4.4（人が消えた）、accurate も 2.20・0.00%・4.4。正解の背景を書き出し（core.VideoWriter の MP4。2026-11-15 に色の変換を正確にする前の版）に通しただけで差が 2.20 あるので、書き出しを通した正解と比べると 1.24（単純な中央値は 3.54）。vtest（実写の固定カメラ・79 秒）: 元のコマに Faster R-CNN で 6・7・4・6 人（5・20・40・60 秒）→ プレートは 0 人、確かめにくい所 0.04%。--video（広場の合成）: 人のいた所の、書き出しを通した正解の背景との差 81.1 → 3.14（人のいない所は 3.09）。速さ: vtest で 3.1 秒、広場の合成で単純な中央値 1.8 秒・fast 11.3 秒（モデルの読み込みを含む）・accurate 697 秒（他の作業と CPU を分け合った状態で 1 コマ 10 秒以上）
- 限界・注意: 固定カメラ専用（揺れる映像は先に手ぶれ補正で止める）。範囲の半分以上の時間そこに居続けた物は単純な中央値に残る（人なら --exclude-people、人以外は --exclude-mask）。--exclude-people の枠の中は使えるコマが減るので少しノイズが多い（差の地図で薄い四角に見える。正解との差は 10 未満）。--video は差の大きい所を全部埋めるので、影も埋まる。照明が変わる映像・動く背景（木・水面）は試していない（中央値は平均的な明るさ・ぼけた模様になるはず）
- 確かめた素材と方法: make_plaza.py（正解の背景つき）、test_cleanplate.py（単純な中央値・--exclude-people の fast と accurate・vtest の人数・--video）、review_plate_diff.py（目で確認: vfx-review/75-clean-plate.jpg）、画像を渡した時のエラー

```bash
cd tools && uv run python -m vfx clean-plate [-h] [-o OUT] [--json JSON] [--start START]
                                 [--duration DURATION]
                                 [--max-frames MAX_FRAMES]
                                 [--max-side MAX_SIDE] [--samples SAMPLES]
                                 [--exclude-people]
                                 [--detector {fast,accurate}]
                                 [--exclude-mask EXCLUDE_MASK] [--video VIDEO]
                                 [--threshold THRESHOLD]
                                 src
```

