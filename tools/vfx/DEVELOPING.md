# tools/vfx の作り方（開発する人向け）

映像処理テクニック 100 選を、この環境（GPU なし・CPU 4 コア・メモリ 15GB・許可された通信先のみ）で
実行できるようにする道具。使う人向けの説明は `.claude/skills/video-techniques/SKILL.md`。

## 状況（2026-11-15）

- **100 項目のうち 74 項目を実装した**。13 のモジュールを担当ごとに分けて作り、各担当が正解つきの試験をして状態
  （◎ 54・○ 19・△ 1）を付けた。**別の AI による検証はしていない**。
- 残りの 26 項目（`face.py`・`body.py`・`camera.py`・`compose.py`・`optics.py` の担当分）は作っていない。ユーザーの判断で、
  74 項目で完了とした。これらのファイルには、担当の番号を書いた冒頭の説明と空の `NOTES` だけがある。
- 26 項目に使える共通の部品は既にある: `models.FILES` の MediaPipe（`mp_pose_lite`・`mp_pose_full`・`mp_pose_heavy`・
  `mp_hand_landmarker`・`mp_gesture`・`mp_face_landmarker`・`mp_face_detector`）、`detectors.FaceDetector`（YuNet）と
  `face_recognizer`（SFace）。`models.FILES` の `gfpgan` は、まだどのコマンドも使っていない（使う前にライセンスを確かめる）。
- 2026-11-15 の仕上げで、74 のコマンドすべてを短い素材で 1 回ずつ動かし、全部が書き出しまで終わることを確かめた
  （出力の確認画像の一部を目で見て、色や形が崩れていないことも確かめた）。

### 2026-11-15 に共通の部品で直したこと

- **色の変換**: `core.read_frames`・`core.VideoWriter` は、ffmpeg の既定の変換（丸めが粗い）をやめ、正確な変換
  （`core.SWS_FLAGS`）を使う。書き出しには BT.709 の色の印（`core.BT709_TAGS`）を付ける。前は書いて読み戻すと白 255 が
  251・灰色 128 が 124 になった。今は 255・128 のまま、色見本の差は 1 以内。HyperFrames に置いて書き出しても、MP4 は
  灰色の段と色見本が元の値から ±2 以内（透過つき WebM は写真の部分が平均 1 段暗く描かれる。HyperFrames 側の読み込み）。
  既存の切り抜き WebM の透明度は、前と同じ値で読める。
- **時刻**: `core.t_of` は、`--start` がコマの境目でない時、最初のコマの本当の時刻から数える（`read_frames` は start
  以降の最初のコマから読む。24fps・start 5.8 → 5.8333 秒のコマ。8 通りの start で画素が一致することを確かめた）。
  `read_frames(fps=...)` で間引いて読んだ時は `snap=False`。
- **人物の切り抜き**: `ai.PersonSegmenter` は結果を写して返す（MediaPipe の結果の画素は次の呼び出しで解放される）。
- **RAFT**: `ai.RaftFlow(large=True)` の重みを、torchvision の既定（KITTI でも学習した C_T_SKHT_V2。KITTI のデータは
  CC BY-NC-SA 3.0）から C_T_V2（FlyingChairs・FlyingThings3D だけ）に変えた。small の既定は前から C_T_V2。
- **56（matte）**: 縮めて計算したアルファを元の大きさに戻した後のガイデッドフィルター（eps 1e-4）をやめた（下の
  「分かっている癖」の guidedFilter。1080×1920 の正解つきの試験で、境目の帯の平均誤差 19.3 → 12.2。eps 1e-2 では 14.0）。
- **60（mask-track）**: 0.5 で切る前のガイデッドフィルターの eps を 1e-3 から 1e-2 にした（同じ癖。正解つきの合成で
  境目の IoU 0.801 → 0.842・0.716 → 0.753。かけない時より良かった）。

## 構成

| ファイル | 中身 | 変更してよい人 |
|---|---|---|
| `core.py` | 動画・画像の読み書き（ffmpeg）、JSON、日本語の文字入れ、一覧画像、エラー | 共有（担当者は変えない） |
| `models.py` | モデルの取得とキャッシュ（`fetch` / `fetch_spec`）、MediaPipe の作業、PyTorch の設定 | 共有 |
| `ai.py` | 奥行き・人物の切り抜き・オプティカルフロー・全体の動き・画像と言葉の照合（SigLIP 2） | 共有 |
| `detectors.py` | 物体検出（COCO 80 種類・日本語名）、顔検出（YuNet）、顔の特徴量（SFace） | 共有 |
| `registry.py` / `cli.py` / `catalog.py` | コマンドの登録・入口・100 項目の一覧 | 共有 |
| `flow.py` ほか 18 ファイル | テクニックの実装（担当ごとに 1 ファイル。冒頭に担当の番号） | 担当者 |

共有ファイルを変えたい時（新しい共通部品、依存関係の追加など）は、自分では変えず、結果の報告に「必要なこと」として書く。
（この「担当者」の決まりは、2026-11-14〜15 に複数の AI で並行して作った時の物。1 つのセッションで続きを作る時は、
共有ファイルも直してよい（Claude の読み取り。ユーザーの決定ではない）。コミット・プッシュは、ユーザーの指示とこの環境の
決まりに従う。下の「してはいけないこと」の 1 つ目も同じ。）
自分のモジュールの中に小さな部品を書くのは自由。

## コマンドの書き方

`scenes.py` の 88（黒コマ検出）が完成見本。要点:

```python
from . import core
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command


@command(
    16,
    "frame-diff",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg("--threshold", type=float, default=25, help="変化とみなす差（0〜255）"),
)
def frame_diff(a):
    """1 行目が --help に出る説明。"""
    out = Path(a.out) if a.out else core.default_out(a.src, "frame-diff", ".mp4")
    ...
    core.log(f"保存: {out}")
```

- **入力**: 動画か画像（`core.probe` / `core.read_frames` / `core.frame_at` / `core.load_image`）。透過つき WebM も読める
  （`alpha=True` で BGRA）。画像 1 枚でも意味のある処理は、画像でも動くようにする。
- **出力**: `-o` で場所。省略時は `core.default_out(src, コマンド名, 拡張子)`（`out/vfx/` の下）。
  - 処理した映像: MP4（`core.VideoWriter` か `core.map_video`。H.264・キーフレーム細かめで HyperFrames にそのまま入る）。
    透過が要る物は `.webm`（VP9 の透過）。元に音声があれば残す（`map_video` は自動）。
  - 数値の結果: JSON（`core.write_json`。時刻は秒、元の動画の中の時刻）。
  - 確かめる用の画像: `core.contact_sheet` / `core.side_by_side` で 1 枚にまとめた JPG。
- **範囲と大きさ**: `RANGE`（`--start` `--duration` `--max-frames`）、`SIZE`（`--max-side`）を付けると、試験が速くなる。
- **表示**: 日本語。最後に `保存: <パス>` を出す。数字の要約（何件見つけたか等）も出す。
- **失敗**: 使う人に伝える失敗は `core.VfxError("日本語の説明")`（cli がまとめて表示）。素の例外を握りつぶさない。
- **重い import**（torch・transformers・mediapipe）は関数の中で行う（`python -m vfx list` を速く保つため）。
- **時刻**は `core.t_of(コマ番号, fps, start)` で求める（`--start` がコマの境目でない時の分も合わせてくれる）。
  `read_frames(fps=...)` で間引いて読んだ時は `snap=False` を付ける。
- **自前で ffmpeg に BGR のコマを渡して書き出す時**は、`-vf core.to_yuv_filter()` と `core.BT709_TAGS` を組で付ける
  （`-pix_fmt yuv420p` だけだと、ffmpeg の既定の変換で白 255 が 251 になる）。
- **argparse の help の文字に `%` を書く時は `%%`** にする（そのままだと `--help` が ValueError で落ちる。3 つの担当で起きた）。
- **MediaPipe の結果の画素**（`numpy_view()`）は次の呼び出しで解放される。取っておくなら写す。
- **モデル**: 単体ファイルは `models.fetch_spec(ModelFile(url, license))`（curl で取得、`~/.cache/douga-vfx/`）。
  Hugging Face は transformers で読む（`~/.cache/huggingface/`）。PyTorch を使う時は `models.torch_setup()`。
  **商用で使えるライセンス（Apache-2.0・MIT・BSD など）だけ**。HF の `license:` を API で確かめ、`other` や
  非営利（CC BY-NC、S-Lab、NVIDIA Source Code License など）は使わない。ライセンスは NOTES の how に書く。
  取得できる通信先: huggingface.co（と CDN）、download.pytorch.org、github.com のリリース、raw.githubusercontent.com、
  media.githubusercontent.com、storage.googleapis.com（MediaPipe）。それ以外（dl.fbaipublicfiles.com、Google Drive 等）は遮断。
- **書式**: `ruff check vfx/<自分のファイル>` と `ruff format vfx/<自分のファイル>` を通す（tools/ で実行）。

## 確かめ方（必須）

1. 各テクニックを試験素材で実際に動かす。できるだけ**正解が分かる試験**にする（既知の劣化・変形・合成を加えてから
   元に戻せるか、差を数値で測る。例: 既知の角度だけ傾けて水平補正 → 戻った角度の誤差）。
2. 結果を `<スクラッチパッド>/vfx-review/<番号>-<コマンド名>*.jpg` に 1 枚の画像（前後・注釈・並べたコマ）として
   書き出し、**自分の目で見て**判断する（Read で画像を開く）。動画の結果は、要所のコマを並べた画像で見る。
3. 結果をモジュールの `NOTES[番号]` に書く:
   - `status`: ◎ 実装し、試験で期待どおりと確かめた ／ ○ 簡易な方法（近似）で実装し確かめた（本格的な手法との差を
     limits に書く）／ △ 条件付き（必要な素材・情報がそろえば使える。条件を limits に書く）／ × この環境では不可（理由）。
     **確かめていないことを ◎ にしない。** 迷ったら低い方にする。
   - `how`: 中身（使う手法・道具・モデル名とライセンス）。
   - `measured`: 実測の数字（正解との差、件数、速さ「540×960・48 コマで 6 秒」など）。**実際に出た値だけ**。
   - `limits`: 限界・失敗した例・調整のこつ。
   - `tested`: 使った素材と確かめ方。
4. `uv run python -m vfx <コマンド> --help` と `uv run python -m vfx info <番号>` が正しく出ることを確かめる。

## 試験素材

作った時（2026-11-14〜15）のセッションの一時的な場所にあった物。**新しいセッションには残っていない**（試験の台本・
確認画像も同じ。リポジトリには入れていない）。続きを作る時は、同じような素材を用意し直す（広告の映像は `work/` の
生成素材から作った。`work/` も Git には入っていない）。

`<スクラッチパッド>` = 作った時のセッションのスクラッチパッド（`/tmp/claude-0/.../scratchpad`）

- `<スクラッチパッド>/vfx-test/media/`（広告の生成映像を 540×960 に縮めた物。人物は AI 生成。ただし `v4-540.mp4` の 11.5〜13.5 秒には
  実在のスタッフの写真と名前の札が写るので、その区間は顔を加工する試験に使わない）
  - `story540.mp4`（15 秒・24fps。カット 5 つ: 0〜3.0 ビューラーの寄り／3.0〜5.79 洗面所の鏡の前／5.79〜8.88 日差しで
    見上げる顔／8.88〜12.38 顔の寄り／12.38〜15.04 カフェで笑う）
  - `a-eye-540.mp4`（目のアップ・目が開く）、`b-lashlift-540.mp4`（まつげパーマの施術）、`c-brow-540.mp4`（眉の施術）
  - `plate-{bath,stand,laugh,cafe}-540.mp4`（短い場面）と `cut-*-540.webm`（同じ場面の人物の切り抜き。透過つき）
  - `v4-540.mp4`（完成した 20 秒の広告。音声つき。日本語の文字・白いフラッシュ・明滅する光の演出あり）
  - `vtest.mp4`（OpenCV のサンプル。固定カメラで人が歩く 768×576・10fps・79 秒）
  - `salon.webp`（実在の店内写真。解析は可。加工した結果を外に出さない）
- `<スクラッチパッド>/vfx-test/ext/`（公開サンプル。確かめるだけに使い、リポジトリに入れない）
  - OpenCV: `basketball1/2.png`（オプティカルフロー）、`box.png`・`box_in_scene.png`（平面の照合）、`left01〜14.jpg`
    （チェッカーボード。レンズの校正）、`graf1/3.png`（ホモグラフィー）、`home.jpg`・`building.jpg`（空と建物）、
    `aero1/3.jpg`（空撮・かすみ）、`fruits.jpg`、`starry_night.jpg`、`text_defocus.jpg`（ぼけた文字）、`tree.avi`、
    `Megamind.avi`、`vtest.avi`、`pic1.png`、`HappyFish.jpg`
  - MediaPipe: `mp-victory.jpg`・`mp-assets-victory.jpg`（ピースサイン）、`mp-thumbs_up/down.jpg`、`mp-pointing_up.jpg`、
    `mp-woman_hands.jpg`、`mp-pose.jpg`（全身のヨガ）、`mp-cat_and_dog.jpg`、`mp-cats_and_dogs.jpg`、`mp-portrait.jpg`
    （**実在の人物**。検出の確認だけに使い、顔を加工する試験には使わない）
- `<スクラッチパッド>/vfx-test/synth/`（自分で作った試験映像の置き場。ffmpeg の `testsrc2` などで作ってよい）

## 分かっている癖（各担当の報告と、仕上げの時の実測）

- **OpenCV 4.14 の `cv2.ximgproc.guidedFilter` は、3 チャンネル・0〜1 の float の案内画で eps が 1e-3 以下だと、縁を
  保たず、ただの平均（箱型のぼかし）と同じ値になる**（灰色・色つきの段差で確かめた。eps 2e-3 から縁を保ち始め、1e-2 で
  保つ）。1 チャンネルの案内画では、小さい eps で縁を保つ（逆）。3 チャンネルの案内画には eps 1e-2 前後を使う。
- OpenCV 4.14 の dnn で `models.FILES['youtu_reid']`（YouTu ReID）を使う時、1 回に入れる枚数を途中で変えると、その網の
  以後の出力がすべて壊れる（最後は NaN）。1 枚ずつか、常に同じ枚数で通す（`mot._BodyEmbed` は 1 枚ずつ）。
- BiRefNet（onnxruntime）は、作業メモリの使い回し（`enable_cpu_mem_arena`・`enable_mem_pattern`）を切らないと、2 回目の
  実行で 12GB まで増える（切ると lite 6.7GB・portrait 8.0GB）。
- transformers 5.2.0: DETR panoptic は背骨の ResNet-50 を作るのに timm が要る（入れていない）。`matte.py` は torchvision の
  ResNet-50 に差し替えて動かしている（公式と出力が一致）。`XCLIPProcessor(videos=...)` は空の辞書を返すので、tokenizer と
  image_processor を直接使う（`understand._XClip`）。SigLIP 2 は `AutoProcessor`・`AutoTokenizer` が失敗するので、
  `GemmaTokenizerFast` と `AutoImageProcessor` を別々に読む（`ai.ImageText`）。
- `detectors.ObjectDetector` には種類をまたいだ重複の除去が無い（同じ犬の枠に「犬 0.99」と「猫 0.54」が出た）。
  `find.py` は `_nms(dets, 0.7, same_label=False)` で除いている。Faster R-CNN v2 は同じ人に全身と上半身の 2 つの枠を出す
  ことがある（`mot.py` の `--nested` で入れ子の枠を除ける）。
- `ai.DepthEstimator` はコマごとに正規化するので、動画ではちらつく。`generate.py` の `_DepthVideo`（フローで時間方向に
  そろえる）で、ちらつきが 0.023 → 0.012 になった。
- `ai.optical_flow` の farneback は、模様の少ない顔と大きな動きで崩れる（誤差 2.4〜17.8px。DIS は 0.06〜0.2px）。DIS を使う。
- ffmpeg の `tmedian` は最初と最後のコマを落として 1 コマずれる（`restore.py` は `tpad` で直した）。
- `core.draw_text` は呼ぶたびに画像全体を PIL に変えるので、全コマに文字を描くと遅い（`scenes.py` は札を一度だけ作って
  貼る `_label`・`_paste` を持つ）。
- `core.IMAGE_EXTS` に `.avif`・`.heic`・`.jxl` が無く、AVIF（HDR の静止画）は動画として扱われる（`tone.py` は自前で読む）。
- `flow.py` の block-match の `--search` は PyAV（av 18.1.0）の filter.Graph と side_data を使う。av の版を上げる時は確かめる。
- 試験の出力でディスクが一杯になったことがある（作っている時に空きが 8KB）。大きな中間ファイル（.npz など）は読んだら消す。
- 速さの数値の多くは、他の担当と CPU を分け合った状態（負荷 4〜10）で測った物。

## 直していない課題（2026-11-15 時点）

- 共通にまとめると重複が減る物（任意）: UperNet の読み込み（`matte._Scene`・`generate._SkySeg`）、時間のグラフと縦並べ
  （`scenes.py`・`understand.py` の `_chart`・`_stack`）、モジュールの中に持っているモデル（ViTMatte・Florence-2 large・
  NAFNet・Real-ESRGAN wdn・CRAFT・PP-OCRv3・WeChat QR・EasyOCR の japanese_g2）、`track.py` の HyperFrames 用の `.js` の
  書き出しとカットの見分け（`_CutWatch`）。
- `models.FILES['lama']` に sha256 を入れる（担当が取得したファイルで
  `344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9`）。
- `detectors.ObjectDetector`（accurate）は短い辺を 800px に拡大して検出するので遅い（540×960 で 1 コマ 2〜2.5 秒。
  混んでいると 15 秒）。大きさを選べる引数があるとよい。`detectors.color_for(1)` と `color_for(5)` は色が近い。
- 日本語の OCR を強くするには、PaddleOCR（PP-OCRv5）などの新しい依存関係が要る（白い文字に黒い縁のテロップは EasyOCR で
  正解率 6〜8 割）。フレーム補間を AI（RIFE。MIT）にするには、モデルの構造のコードを入れる判断が要る。

## してはいけないこと

- コミット・プッシュ（まとめて行う）。担当以外のファイル・共有ファイルの変更。他の人の出力を消すこと。
  （並行して作る担当者向け。1 つのセッションで続きを作る時は、上の「構成」の下の注のとおり）
- 有料 API（OpenAI・BytePlus・ElevenLabs）を呼ぶこと。
- 証明書の検証を無効にすること。
- 実在の人物の顔を加工する試験（`work/` の作品にある実在のスタッフの写真は使わない。`mp-portrait.jpg` は検出だけ）。
  実在の人（スタッフ・お客様）の名前は、ファイル名も含めてリポジトリに書かない。
- 非商用・ライセンス不明のモデルを使うこと。
- 一度に重い処理を並べて走らせること（CPU は 4 コアを他の作業と分け合っている）。試験は数秒の映像・小さめの画像で。
