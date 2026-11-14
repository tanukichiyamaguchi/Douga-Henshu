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
