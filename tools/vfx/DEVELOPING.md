# tools/vfx の作り方（開発する人向け）

映像処理テクニック 100 選を、この環境（GPU なし・CPU 4 コア・メモリ 15GB・許可された通信先のみ）で
実行できるようにする道具。使う人向けの説明は `.claude/skills/video-techniques/SKILL.md`。

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

`<スクラッチパッド>` = `/tmp/claude-0/-home-user-Douga-Henshu/ab02e0dc-2f09-56b1-a1d6-ed872e314e04/scratchpad`

- `<スクラッチパッド>/vfx-test/media/`（広告の生成映像を 540×960 に縮めた物。人物はすべて AI 生成）
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

## してはいけないこと

- コミット・プッシュ（まとめて行う）。担当以外のファイル・共有ファイルの変更。他の人の出力を消すこと。
- 有料 API（OpenAI・BytePlus・ElevenLabs）を呼ぶこと。
- 証明書の検証を無効にすること。
- 実在の人物の顔を加工する試験（`work/` の `yuu.png` は使わない。`mp-portrait.jpg` は検出だけ）。
- 非商用・ライセンス不明のモデルを使うこと。
- 一度に重い処理を並べて走らせること（CPU は 4 コアを他の作業と分け合っている）。試験は数秒の映像・小さめの画像で。
