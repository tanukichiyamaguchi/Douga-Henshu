# Douga-Henshu（動画編集）— Claude 向けガイド

コードで動画を作るためのリポジトリ。開発はすべて Claude Code のクラウド環境
「Douga-Henshu」（GitHub 連携）で行い、ローカル環境は使わない。

## 構成

| 場所 | 中身 |
|---|---|
| `remotion/` | Remotion（React で動画）。`src/Root.tsx` に動画を登録する。サンプル: `SampleTitle`（日本語タイトル）、`SampleThree`（Three.js の 3D） |
| `hyperframes/` | HyperFrames（HTML + GSAP で動画）。`index.html` が本体。専用の `CLAUDE.md` あり |
| `tools/` | Python ツール（uv 管理）: `narrate.py`（Chatterbox ナレーション）、`transcribe.py`（faster-whisper 字幕）、`blender_render.py`（Blender 3D） |
| `setup/cloud-setup.sh` | セッション開始前に依存関係・描画用ブラウザ・スキルを入れるスクリプト |
| `.mcp.json` | Chrome DevTools MCP（書き出し前の画面確認用） |
| `out/` など | 書き出し物。Git には入れない（`.gitignore` 済み） |

## この環境の前提（必ず守る）

- **GPU なし・CPU 4 コア・メモリ 15GB**。3D やAI処理は CPU で動く分だけ遅い。
- **外部通信はプロキシ経由で、許可された通信先のみ**。許可済み: npm / PyPI / GitHub /
  Ubuntu apt、`huggingface.co`・`us.aws.cdn.hf.co`・`cas-server.xethub.hf.co`（AIモデル）、
  `download.pytorch.org`・`download-r2.pytorch.org`（CPU 版 PyTorch）、`remotion.media`
  （Remotion 同梱ブラウザ）、`cdn.jsdelivr.net`（GSAP）、Google Fonts。
  これ以外（YouTube など）は遮断される。
- **Node.js から外部通信する時は `NODE_USE_ENV_PROXY=1` が必要**（`.claude/settings.json`
  で設定済み。npm スクリプトにも付けてある）。無いと `ENOTFOUND` で失敗する。
- **証明書の検証は絶対に無効化しない**（`--ignore-certificate-errors` 等は使わない）。
- このリポジトリは**公開リポジトリ**。秘密情報・他人の素材・ライセンス不明のファイルを
  コミットしない。

## よく使うコマンド

```bash
# Remotion（remotion/ で実行）
npm run dev                                   # Studio（プレビュー）
npm run render -- SampleTitle out/title.mp4   # 動画を書き出す
npm run still -- SampleTitle out/title.png --frame=60   # 静止画 1 枚
npm run lint                                  # ESLint + 型チェック

# HyperFrames（hyperframes/ で実行）
npm run check                                 # 書き出し前の総合チェック（必ず通す）
npx hyperframes snapshot --at 0,1,3           # 指定秒のフレームを PNG に
npm run render -- -o renders/out.mp4          # 動画を書き出す

# Python ツール（tools/ で実行）
uv run python narrate.py --text "読み上げる文章" --out ../out/narration.wav
uv run python transcribe.py ../out/narration.wav --out-dir ../out
uv run python blender_render.py --text "動画編集" --out ../out/blender.mp4
```

## 実測で分かった注意点

- **Remotion の描画用ブラウザは Remotion 同梱のものを使う**（公式スキルの推奨）。
  `--browser-executable` は指定しない。
- **Remotion はブラウザを「プロキシなし」で起動する**ため、`remotion.config.ts` で
  `HTTPS_PROXY` がある時だけプロキシを渡している。消すと Web フォント等が証明書エラーになる。
- **Remotion の `--license-key=free-license` は CLI 4.0.543 では拒否される**。
  無料ライセンス対象（個人・従業員 3 人以下の営利企業・非営利団体）なのでキーは付けない。
- **日本語フォント**: Remotion は `@remotion/google-fonts/NotoSansJP`、HyperFrames は
  `font-family: "Noto Sans JP"`（書き出し時に自動取得）。システムには IPA ゴシックもある。
- **Chatterbox は短すぎる文（「こんにちは。」だけ等）で意味のない音声を作ることがある**。
  `narrate.py` は 20 文字未満の文を次の文とつなげて読ませる。読み上げ後は
  `transcribe.py` で文字起こしして内容を確認すると確実。
- **Chatterbox の PyPI 版（0.1.7）は多言語モデル V2**。README にある V3 は GitHub の
  最新コードのみ。生成音声には電子透かし（PerTh）が必ず入る。
- 依存関係の固定理由: `setuptools<82`（透かしライブラリが `pkg_resources` を使う）、
  `av<19`（faster-whisper 1.2.1 が PyAV 19 で動かない）。どちらも実測で確認済み。
- **速さの目安（CPU 4 コアで実測）**: Remotion 1080p・3 秒 = 7〜13 秒 /
  HyperFrames 1080p・4 秒 = 約 7 秒 / Chatterbox = 音声の約 4 倍の時間 /
  faster-whisper（large-v3-turbo）= 音声の約 1/3 の時間 /
  Blender Cycles 960×540・16 サンプル = 1 フレーム約 3 秒。
- AI モデル（Chatterbox 約 3GB、Whisper 0.5〜1.6GB）は**初回使用時に**取得される
  （セッションごとに 1 回）。

## 生成 AI（有料 API）

リアルな映像・画像・音楽は外部の有料 API で作る（`tools/` で実行）。

| 用途 | スクリプト | サービス | 通信先 |
|---|---|---|---|
| 映像 | `gen_video.py` | BytePlus ModelArk の Seedance | `ark.ap-southeast.bytepluses.com`（生成した映像の取得は `ark-acg-ap-southeast-1.tos-ap-southeast-1.volces.com`） |
| 画像（標準） | `gen_image_openai.py` | OpenAI の GPT Image（既定 `gpt-image-2.5-flare`） | `api.openai.com` |
| 画像（予備） | `gen_image.py` | BytePlus ModelArk の Seedream | `ark.ap-southeast.bytepluses.com` |
| 音楽（BGM） | `gen_music.py` | ElevenLabs Music | `api.elevenlabs.io` |

```bash
uv run python gen_video.py --prompt "..." --ratio 9:16 --resolution 720p --duration 5 --no-audio --out ../out/shot.mp4
uv run python gen_video.py --task-id <タスクID> --out ../out/shot.mp4   # 生成済みの結果だけ取り直す（無料）
uv run python gen_image_openai.py --prompt "..." --size 1536x1024 --out ../out/image.png
uv run python gen_image.py --prompt "..." --out ../out/image.png   # Seedream（予備）
uv run python gen_music.py --prompt "明るいアコースティック、120BPM" --seconds 30 --out ../out/bgm.mp3
```

- **API キーはスクリプトにもリポジトリにも書かない。** クラウド環境のネットワークシークレットとして
  保存し、プロキシが通信時に自動で付ける（スクリプトはダミー値を送る）。
- **状況（2026-11-14 時点）:** 公式 SDK（`byteplussdk-arkruntime` 3.0.66 / `elevenlabs` 2.78.0）で
  作成し、模擬サーバーで送信内容を確認済み。
  - BytePlus: キー登録済み。**Seedance 2.5 と Seedream 5.5 で実際の生成を確認済み**
    （映像 720p・5 秒は約 135 秒、画像は約 17 秒）。既定モデルは `dreamina-seedance-2-5-260628`（映像）と
    `dola-seedream-5-5-261028`（画像）。映像の取得に失敗しても、`--task-id` で料金をかけずに取り直せる。
  - OpenAI: キー登録済み。**`gpt-image-2.5-flare` で実際の生成を確認済み**（1536×1024・約 19 秒・PNG 約 2.3MB）。
    このアカウントでは `gpt-image-2.5-sunburst`・`gpt-image-2.6-flare`・`gpt-image-2.6-sunburst` も使える
    （`GET https://api.openai.com/v1/models` で確認）。OpenAI のキーは `sk-` で始まる。`key_` で始まるのは
    キーの管理番号（ID）で、キーの代わりにはならない。
  - ElevenLabs: キー未登録。
  - 他に使えるモデル（例）: 映像 `dreamina-seedance-2-0-fast-260128`、画像 `dola-seedream-5-0-pro-260628`。
    一覧は `curl -H "Authorization: Bearer proxy-injected" https://ark.ap-southeast.bytepluses.com/api/v3/models`。
- Seedance の参考画像（`--image`）は公開 URL か BytePlus の素材 ID のみ。役割（role）は公式ドキュメントで
  確認できた `reference_image` だけを使う。
- どれも従量課金。試作は短い秒数・低い解像度から始め、作り直しは原因を特定してから行う。
- Suno は公式の公開 API がない（2026 年 7 月に招待制プログラムを発表したのみ）ため使わない。
  非公式の API は規約違反・法的リスクがあるので使わない。

## Google Drive 連携

claude.ai の Google Drive コネクタ（`mcp__Google_Drive__*`）を使う。API キー等は不要。

- 使ってよいのは **マイドライブ直下の `Douga-Henshu` フォルダの中だけ**。他のフォルダ
  （書類・個人情報を含む）は検索・閲覧・変更しない。フォルダ ID は公開リポジトリに書かず、
  `search_files` で `title = 'Douga-Henshu'` のフォルダを探してから中を見る。
  - `01_素材`: ユーザーが置く素材（画像・ロゴ・台本・音声など）。ここから取り込む。
  - `02_完成動画`: 書き出した完成品を保存する。
- **保存（出力）**: `create_file` の `base64Content` にファイル参照
  `sha256:<sha256>:<絶対パス>` を渡し、`contentMimeType: video/mp4`・
  `disableConversionToGoogleType: true`・`parentId` に `02_完成動画` の ID を指定する。
  保存のたびにユーザーの承認が必要。131KB の MP4 と 2.25MB の PNG で動作確認済み。
  **コネクタは 1 回 4.2MB まで（ファイル本体は約 3.1MB まで）**。5.3MB の画像は「上限超過」で送れなかった
  （2026-11-14 実測）。それより大きい完成品は、コネクタでは保存できない。分割・圧縮などで勝手に
  回避せず、ユーザーに保存方法を確認する（このアプリへの直接送付は可能）。
- **取り込み（入力）**: `download_file_content` は中身を base64 で会話に流すため、画像・台本など
  小さいファイル向け。数十 MB 以上の動画素材は現実的でない（上限は未確認）。
- 保存後は `viewUrl`（Drive で開くリンク）をユーザーに伝える。

### 大きなファイル（3.1MB 超）は Drive API で（`tools/drive.py`）

```bash
uv run python drive.py whoami                         # つながるアカウントとフォルダを確認
uv run python drive.py upload ../out/final.mp4        # 02_完成動画 に保存（大きくても可）
uv run python drive.py list                           # 01_素材 の一覧
uv run python drive.py download 台本.txt --out ../out/input/   # 01_素材 から取り込む
```

- 認証情報は環境変数 `GOOGLE_OAUTH_CLIENT_ID`・`GOOGLE_OAUTH_CLIENT_SECRET`・`GOOGLE_OAUTH_REFRESH_TOKEN`。
  **ユーザーが環境の設定画面で登録する**（値をリポジトリや会話に書かない）。新しいセッションから有効。
- 通信先 `oauth2.googleapis.com`・`www.googleapis.com` は許可済み。
- 分割アップロード（resumable、8MB 単位）。模擬サーバーで 20MB の保存・取り戻しが完全一致することを確認済み。
  **本物の Drive での確認は未実施**（認証情報の登録待ち）。
- コネクタ（3.1MB 以下）と Drive API のどちらでも、使うのは `Douga-Henshu` フォルダの中だけ。

## スキルと MCP

セッション開始時に `~/.claude/skills/` へ入る（`setup/cloud-setup.sh`）:
Remotion Skills（`remotion-*`）、HyperFrames（`hyperframes` ほか。`reverse-any-video` と
`style-sampling` は有料の HyperFrames Pro 機能）、Impeccable（デザインの講評・仕上げ）。
Chrome DevTools MCP は `.mcp.json` で起動する（Google への利用統計送信はオフ）。

## 入れていないもの（理由）

- **ComfyUI**: GPU がなく、画像・動画生成 AI を実用的な速度で動かせない。
- **Blender MCP**: 手元の PC で開いた Blender を操作する仕組みのため。代わりに
  `tools/blender_render.py` のように Python から Blender を直接動かす。
- **yt-dlp**: YouTube への通信が遮断されている。規約・著作権の問題もある。
