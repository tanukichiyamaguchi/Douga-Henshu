# 環境・道具・外部サービスの詳しい記録

環境の前提・コマンド・生成 AI の API・Google Drive・書体・映像処理の道具の細部。CLAUDE.md の要点の元。

> この資料は、2026-11-18 まで `CLAUDE.md` に書いていた文を、そのまま移した物です（作り替えの理由は
> `docs/notes/2026-11-18-token-investigation.md`）。いま有効な決まりは `CLAUDE.md` を見てください。
> 文中の「上の」「下の」「この文書」は、元の `CLAUDE.md` の中での位置を指します（節の名前で探してください）。
> まるごと読まず、`grep -n` で要る節を探して、その節だけを読んでください。

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
- **Web ページを直接開く取得は、許可されていない通信先では失敗する**（2026-11-14、ニュースサイトのページの取得が、名前解決の
  失敗〔許可された通信先の外〕で止まった。店や Meta の情報は、ページを開かずに検索結果の要約で確かめた）。調べた結果を使う
  時は、そのことを報告に書く。
- このリポジトリは**公開リポジトリ**。秘密情報・他人の素材・ライセンス不明のファイルを
  コミットしない。実在の人（スタッフ・お客様）の名前は、リポジトリのファイル（中身とファイル名）にも、コミットの文・PR の
  本文にも書かない（作品の画面に載せるかは、広告ごとにユーザーに確かめる）。push した物は、ファイルから消しても履歴と PR に
  残る。入ってしまったブランチは 1 つのコミットにまとめて（squash で）main に入れ、履歴まで消すかはユーザーに確かめる。
  **他人の図表や文章を、まとまった量そのまま写さない。自分の言葉で書く**（2026-11-15、映像処理テクニック 100 選の図表に
  ついて、そのまま載せてよいかを尋ね、ユーザーが「自分の言葉に書き換える」を選んだ。番号と項目名は図表のまま。ほかの図表・
  文章にも当てはめるのは Claude の読み取り）。店の表記や規約の名前など、言い換えると意味が変わる短い語句は、出典つきで引いてよい。

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
uv run python tighten_pauses.py ../out/narration.mp3 --max-pause 0.12 --out ../out/narration-tight.wav   # 文の間を 0.12 秒に詰める（話す速さを上げる時は --tempo 1.25 を足す。「毎回守ること」のテンポの行）
uv run python tempo_check.py ../work/<作品>/out.mp4 --lines ../work/<作品>/lines.json   # テンポを測る（絵の動き・カットの長さ・話す速さと文の間。目安と比べて ○× を出す）
uv run python voice_check.py --before 受け取った声.mp3 --after 加工した声.wav             # 声が加工で変わっていないか（高い音・大きさの幅）。候補の声を比べる時は、ファイルを並べる
uv run python blender_render.py --text "動画編集" --out ../out/blender.mp4
uv run python fetch_fonts.py ../work/<作品> --font "<書体名>:<太さ>" --font "<書体名>:<太さ>"   # 使う文字だけの書体を assets/fonts/ へ（書体は作品ごとに選ぶ。上の「トンマナの決め方」）
uv run python -m vfx list                      # 映像処理テクニック 100 選の一覧と状態（74 項目が使える）
uv run python -m vfx info 56                   # 1 項目の中身・実測・限界と引数
uv run python -m vfx matte ../work/<作品>/assets/in.mp4 -o ../work/<作品>/assets/cut.webm   # 例: 人物を髪まで切り抜く（透過 WebM）
```

## 実測で分かった注意点

- **Remotion の描画用ブラウザは Remotion 同梱のものを使う**（公式スキルの推奨）。
  `--browser-executable` は指定しない。
- **Remotion はブラウザを「プロキシなし」で起動する**ため、`remotion.config.ts` で
  `HTTPS_PROXY` がある時だけプロキシを渡している。消すと Web フォント等が証明書エラーになる。
- **Remotion の `--license-key=free-license` は CLI 4.0.543 では拒否される**。
  無料ライセンス対象（個人・従業員 3 人以下の営利企業・非営利団体）なのでキーは付けない。
- **日本語フォントの読み込み**: Remotion は `@remotion/google-fonts/NotoSansJP`、HyperFrames は
  `font-family: "Noto Sans JP"`（書き出し時に自動取得）。システムには IPA ゴシックもある。これは読み込み方で、
  既定の書体ではない（Noto Sans JP は `font-family` の 2 番目に書く予備。書体は上の既定値・トンマナのとおり選ぶ）。
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

### 書体（フォント）の入れ方（2026-11-14）

- **HyperFrames が名前だけで使えるのは同梱の 18 書体**（Noto Sans JP・Playfair Display・EB Garamond・
  Montserrat など）。それ以外は作品の `assets/fonts/` にファイルを置き、その書体を使う場面の `<style>` に
  `@font-face` を書く（無いと lint の `font_family_without_font_face` で止まる）。
- **`fetch_fonts.py` で、作品の文字だけを切り出したファイルを Google Fonts から取る**（CSS API の `text=`。
  1 ファイル 20〜50KB）。最後に貼り付け用の `@font-face` を表示する。**文字を変えたら取り直す**（ファイルに
  無い文字は、`font-family` で次に書いた書体で描かれる）。`font-family` の 2 番目には同梱の書体（Noto Sans JP）を書く。
- 選び方: 候補を実際の場面の静止画に並べて比べる（Playwright で HTML を撮影）。どの書体にするかはトンマナの
  一部なので、作品ごとに決める（上の「トンマナの決め方」）。明朝は細い線が背景に沈みやすいので、小さい字は
  600 以上・影付き。
- Shippori Mincho B1 は 800 まで、Zen Kaku Gothic New は 900 まで。CSS の `font-weight` は、ファイルを取った太さと
  同じ値にする。
- 斜体は太さに `i` を付けて取る（`--font "Cormorant Garamond:600,600i"`。`@font-face` に `font-style: italic` が付く）。
- **Cormorant Garamond と Playfair Display の数字は、そのままだと高さがふぞろい**（オールドスタイル数字。3・4・7・9 が
  下へはみ出し、0・1・2 が小さい）。`font-feature-settings: "lnum" 1` で高さのそろう字形になる（`fetch_fonts.py` で
  取ったファイルでも効く）。価格・日数・比率の数字には必ず付ける。Bodoni Moda・Noto Serif Display・しっぽり明朝は
  そのままでそろっている。

### 映像処理テクニック 100 選（tools/vfx）で分かったこと（2026-11-15）

- ユーザーから画像で渡された「AI動画の映像処理テクニック100選」を記録し、74 項目をコマンドにした。**ユーザーの判断で
  74 項目で完了**（「現在の74項目で完了で大丈夫です」）。残りの 26 項目は、頼まれたらまだ無いことを伝え、作るかを確認する。
  状態の記号は作った AI が試験して付けた物で、別の AI による検証はしていない。
- **ffmpeg に BGR のコマを渡して H.264 にする時、`-pix_fmt yuv420p` だけだと色の変換の丸めが粗く、白 255 が 251・
  灰色 128 が 124 に下がる**（読み戻す側の既定の変換でも切り捨てで暗くなる）。
  `-vf "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int+full_chroma_inp+bitexact,format=yuv420p"`
  と `-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv` で、元の値のまま残る（tools/vfx の
  `core.VideoWriter` はこの形）。HyperFrames は動画の色の印に従って読み（印が無い物は BT.601 とみなす）、自分の書き出しには
  BT.709 の印を付ける。
- MediaPipe は libEGL・libGLESv2 が無いと読み込めない（`setup/cloud-setup.sh` で入れる）。
- `uv run --with <パッケージ>` で作った一時的な環境は `~/.cache/uv` に残る。CUDA 版の torch などが入り、2 つで約 10GB に
  なってディスクの枠を圧迫した。使い終わったら消す。

## 生成 AI（有料 API）

リアルな映像・画像・音楽は外部の有料 API で作る（`tools/` で実行）。

| 用途 | スクリプト | サービス | 通信先 |
|---|---|---|---|
| 映像 | `gen_video.py` | BytePlus ModelArk の Seedance | `ark.ap-southeast.bytepluses.com`（生成した映像の取得は `ark-acg-ap-southeast-1.tos-ap-southeast-1.volces.com`） |
| 画像（標準） | `gen_image_openai.py` | OpenAI の GPT Image（2026-11-17 から `gpt-image-2.6-flare`・品質 high・1024×1536。上の「毎回守ること」） | `api.openai.com` |
| 画像（予備） | `gen_image.py` | BytePlus ModelArk の Seedream | `ark.ap-southeast.bytepluses.com` |
| 音楽（BGM） | `gen_music.py` | ElevenLabs Music | `api.elevenlabs.io` |
| 効果音 | `gen_sfx.py` | ElevenLabs の効果音生成 | `api.elevenlabs.io` |
| ナレーション | `narrate_elevenlabs.py` | ElevenLabs の音声合成（標準: 声 Harune・モデル `eleven_v4`） | `api.elevenlabs.io` |

```bash
uv run python gen_video.py --prompt "..." --ratio 9:16 --resolution 720p --duration 5 --no-audio --out ../out/shot.mp4
uv run python gen_video.py --task-id <タスクID> --out ../out/shot.mp4   # 生成済みの結果だけ取り直す（無料）
uv run python gen_video.py --prompt "画像1の店内で..." --image ../work/<作品>/assets/interior.jpg --ratio 9:16 --resolution 720p --duration 5 --no-audio --out ../out/shot.mp4   # 手元の画像を参考にする
uv run python gen_image_openai.py --model gpt-image-2.6-flare --quality high --size 1024x1536 --prompt "..." --out ../out/image.png   # 縦長（横長は 1536x1024。透明の背景は --background transparent）
uv run python gen_image.py --prompt "..." --out ../out/image.png   # Seedream（予備）
uv run python gen_music.py --prompt "明るいアコースティック、120BPM" --seconds 30 --out ../out/bgm.mp3
uv run python gen_sfx.py --prompt "short airy whoosh for a fast camera whip pan" --seconds 0.6 --takes 2 --out ../out/whoosh.mp3
```

- **API キーはスクリプトにもリポジトリにも書かない。** クラウド環境のネットワークシークレットとして
  保存し、プロキシが通信時に自動で付ける（スクリプトはダミー値を送る）。
- **状況（2026-11-14 時点）:** 公式 SDK（`byteplussdk-arkruntime` 3.0.66 / `elevenlabs` 2.78.0）で
  作成し、模擬サーバーで送信内容を確認済み。
  - BytePlus: キー登録済み。**Seedance 2.5 と Seedream 5.5 で実際の生成を確認済み**
    （映像 720p・5 秒は約 135 秒、画像は約 17 秒）。既定モデルは `dreamina-seedance-2-5-260628`（映像）と
    `dola-seedream-5-5-261028`（画像）。映像の取得に失敗しても、`--task-id` で料金をかけずに取り直せる。
  - OpenAI: キー登録済み。**`gpt-image-2.5-flare` で実際の生成を確認済み**（1536×1024・約 19 秒・PNG 約 2.3MB）。
    **`gpt-image-2.6-sunburst`・品質 max でも、1440×2560 と 2160×3840 の生成を確認した**（2026-11-17。約 2 分・PNG 4.4〜9.5MB）。
    **`gpt-image-2.6-flare`・品質 high・1024×1536 も確認した**（2026-11-17。透明の背景あり・なしとも 1 枚 9 秒）。
    **`gpt-image-2.6-flare` は 2560×1440・1440×2560・1440×1440 も作れた**（2026-11-20。品質 high で 1 枚 11〜18 秒）。API の
    `chatgpt-image-latest` は 1024×1024・1024×1536・1536×1024 だけで、ほかの大きさは 400 が返る（2026-11-20）。
    このアカウントでは `gpt-image-2.5-sunburst`・`gpt-image-2.6-flare`・`gpt-image-2.6-sunburst` も使える
    （`GET https://api.openai.com/v1/models` で確認）。OpenAI のキーは `sk-` で始まる。`key_` で始まるのは
    キーの管理番号（ID）で、キーの代わりにはならない。
  - ElevenLabs: キー登録済み（Creator プラン、月 131,000 クレジット）。**音声合成と音楽の生成を確認済み**。
    日本語対応モデルは `eleven_v4`・`eleven_v4_turbo`・`eleven_v3`・`eleven_multilingual_v2` など（`--list-models`）。
    声は共有ボイスの Harune（`c2XJrw7TvNGtOc6r0ijG`）をユーザーが試聴して選んだ。約 50 文字で約 8 秒、
    文字起こしで台本どおりと確認。BGM は 10 秒が約 5 秒で生成（MP3・ステレオ・44.1kHz）。
    日本語の共有ボイスの探し方: `GET https://api.elevenlabs.io/v1/shared-voices?language=ja`。
    効果音（`gen_sfx.py`）も生成を確認済み（0.5〜0.6 秒・MP3。音量は −14〜−17 LUFS と大きく、ピークが 0 dB を
    超えるものもあるので、動画では `data-volume` 0.3〜0.4 程度に下げる）。
    このキーには `user_read` 権限が無く、`GET /v1/user/subscription`（プランの確認）は 401 になる。
  - 他に使えるモデル（例）: 映像 `dreamina-seedance-2-0-fast-260128`、画像 `dola-seedream-5-0-pro-260628`。
    一覧は `curl -H "Authorization: Bearer proxy-injected" https://ark.ap-southeast.bytepluses.com/api/v3/models`。
- Seedance の参考画像（`--image`）は、公開 URL・BytePlus の素材 ID・手元の画像ファイルのどれか（手元のファイルは
  データ URI にして送る）。役割（role）は `reference_image`（既定）と `first_frame`（`--image-role first_frame`）を
  実際の生成で確認した（2026-11-15）。人物の顔が写った画像は断られることがある（上の 2026-11-15 の節）。
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

### 完成品の受け渡し（2026-11-14 ユーザー決定）

- 完成品は **このアプリに直接送る**（SendUserFile。大きさの制限なし）。
  - 動画は `display: "attach"`（ダウンロード用のカード）で送る（2026-11-19。`render` や指定なしで送った約 22MB の MP4 を、
    ユーザーが保存できなかった。同じファイルを `attach` で送り直すと保存できた。経過は `production-log.md`）。
- **3.1MB 以下なら、Drive の `02_完成動画` にも保存する**（コネクタ。保存のたびに承認が必要）。
- 3.1MB を超えるものの Drive 保存（下記 `drive.py`）は **ユーザーの判断で後日に見送り**。
  認証情報が未登録のまま使おうとしない。

### 大きなファイル（3.1MB 超）は Drive API で（`tools/drive.py`、現在は未設定）

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
このリポジトリの `.claude/skills/` にあるスキル（`video-techniques`）は、リポジトリと一緒に読み込まれる。
Chrome DevTools MCP は `.mcp.json` で起動する（Google への利用統計送信はオフ）。

## 入れていないもの（理由）

- **ComfyUI**: GPU がなく、画像・動画生成 AI を実用的な速度で動かせない。
- **Blender MCP**: 手元の PC で開いた Blender を操作する仕組みのため。代わりに
  `tools/blender_render.py` のように Python から Blender を直接動かす。
- **yt-dlp**: YouTube への通信が遮断されている。規約・著作権の問題もある。
