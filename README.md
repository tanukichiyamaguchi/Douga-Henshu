# Douga-Henshu（動画編集）

Claude Code（クラウド環境）に「〇〇の動画を作って」と一言頼むだけで、台本・映像・画像・
ナレーション・BGM・字幕を用意して、完成した動画まで作ってもらうためのリポジトリです。

## 使い方

1. Claude Code でセッションを始める時に、クラウド環境「**Douga-Henshu**」を選びます。
   動画ツール・スキル・描画用ブラウザは、セッション開始時に自動で準備されます（約 2 分）。
2. 作りたい動画を、ふつうの文章で伝えます。**「/」で始まる指示は不要**です。

```text
新メニューの抹茶ラテを紹介する動画を作って。店名は〇〇。
```

```text
年末年始の営業時間のお知らせ動画を作って。12/30〜1/3は休み。横型で。
```

店名・商品名・日付など、内容に欠かせない情報が足りない時だけ、Claude から一つずつ質問します。
形・長さ・演出は下の「標準の設定」で進みます。違う形にしたい時は、依頼の文章に書き添えてください
（例:「横型で」「男性の声で」「BGM なしで」）。

### Claude が行う手順

1. 依頼を整理し、台本と絵コンテ（場面ごとの秒数・文字・ナレーション・素材）を作る
2. 費用を見積もる（1 本 10 ドルを超えそうな時だけ、生成の前に確認）
3. 映像・画像・BGM・ナレーションを生成する
4. ナレーションを文字起こしして、台本どおりに読めているか確認する
5. HyperFrames で組み立て、自動チェックと場面ごとの画像確認をしてから書き出す
6. 完成品をお届けし、使った素材・費用の見積もり・確認できなかった点を報告する

## 標準の設定

| 項目 | 標準 |
|---|---|
| 形 | 縦型 9:16（1080×1920）・15〜30 秒 |
| 編集 | HyperFrames（HTML + GSAP） |
| 映像の素材 | Seedance 2.5（720p・5 秒単位） |
| 画像の素材 | GPT Image 2.5 flare（予備に Seedream 5.5） |
| ナレーション | ElevenLabs v4・声「Harune」（女性・落ち着いたナレーション） |
| BGM | ElevenLabs Music（歌なし） |
| 字幕 | あり（ナレーションに合わせて表示） |
| 書体 | Noto Sans JP |
| 費用 | 生成 AI の合計が 1 本 10 ドルまでは確認なし |
| お届け | このアプリに直接送る。3MB 程度までの小さいファイルは Google Drive にも保存 |

## 使っている道具

| 役割 | ツール |
|---|---|
| 編集（標準） | [HyperFrames](https://github.com/heygen-com/hyperframes) 0.9.43 + 公式スキル、[GSAP](https://github.com/greensock/GSAP) |
| 編集（React） | [Remotion](https://github.com/remotion-dev/remotion) 4.0.543 + [Remotion Skills](https://github.com/remotion-dev/skills)、[Three.js](https://github.com/mrdoob/three.js) |
| 映像の生成 | BytePlus ModelArk の Seedance 2.5（`tools/gen_video.py`） |
| 画像の生成 | OpenAI の GPT Image 2.5（`tools/gen_image_openai.py`）、BytePlus の Seedream 5.5（`tools/gen_image.py`） |
| ナレーション | ElevenLabs（`tools/narrate_elevenlabs.py`）、無料の代替に [Chatterbox](https://github.com/resemble-ai/chatterbox)（`tools/narrate.py`） |
| BGM | ElevenLabs Music（`tools/gen_music.py`） |
| 字幕 | [faster-whisper](https://github.com/SYSTRAN/faster-whisper)（`tools/transcribe.py`） |
| 3D | [Blender](https://github.com/blender/blender) 5.3.0（`tools/blender_render.py`） |
| 書き出し・確認 | [FFmpeg](https://github.com/FFmpeg/FFmpeg)、[Chrome DevTools MCP](https://github.com/ChromeDevTools/chrome-devtools-mcp)、[Impeccable](https://github.com/pbakaus/impeccable)（デザインの講評） |
| Google Drive | Google Drive 連携（コネクタ）、大きなファイル用の `tools/drive.py`（未設定） |

入れていないもの: ComfyUI（GPU がないため）、Blender MCP（手元の PC 向けのため）、
yt-dlp（YouTube への通信が遮断されており、規約・著作権の問題もあるため）、
Suno（誰でも使える公式の API がないため）。

## 必要な契約と API キー

| サービス | 用途 | 状態 |
|---|---|---|
| BytePlus ModelArk | 映像（Seedance）・画像（Seedream） | 登録済み（従量課金・事前入金） |
| OpenAI API | 画像（GPT Image） | 登録済み（ChatGPT の月額プランとは別契約） |
| ElevenLabs | ナレーション・BGM | 登録済み（Creator プラン） |

API キーは、クラウド環境「Douga-Henshu」の**ネットワークシークレット**として保存されています。
通信の時に自動で付けられるため、スクリプトにもこのリポジトリにも書かれていません。
キーを変える時は、環境の設定画面から入れ直すか、Claude に「キーを入れ直したい」と伝えてください。

## 費用の目安（2026 年 11 月時点）

| 項目 | 目安 |
|---|---|
| 映像（Seedance 2.5・720p） | 約 0.23 ドル／秒（5 秒で約 1.2 ドル） |
| 画像（GPT Image） | 生成量に応じた課金。1 枚の公式な金額は非公表 |
| ナレーション（ElevenLabs v4） | 1,000 文字あたり 0.08 ドル（30 秒で 1〜2 セント） |
| BGM（ElevenLabs Music） | 契約プランのクレジットから引かれる |

正確な料金は、各サービスの管理画面で確認してください。

## 処理時間の目安（実測）

| 処理 | 目安 |
|---|---|
| 映像の生成（Seedance・720p・5 秒） | 約 2〜3 分 |
| 画像の生成（GPT Image / Seedream） | 約 20 秒 |
| ナレーションの生成（ElevenLabs） | 数秒 |
| BGM の生成（10 秒） | 約 5 秒 |
| 字幕づくり（faster-whisper） | 音声の長さの約 1/3 |
| HyperFrames の書き出し（1080p・4 秒） | 約 7 秒 |

## 完成品の受け取り

- 完成した動画は、このアプリに直接送られます。
- 3MB 程度までの小さいファイルは、Google Drive の `Douga-Henshu/02_完成動画` にも保存します
  （保存のたびに承認画面が出ます）。素材は `Douga-Henshu/01_素材` に置けば取り込めます。
- それより大きいファイルを Drive に自動保存する仕組み（`tools/drive.py`）は用意済みですが、
  Google の認証情報が未登録のため、まだ使っていません。

## 困った時

自動の準備が失敗した時は、リポジトリのルートで次を実行すると入れ直せます。

```bash
bash setup/cloud-setup.sh
```

## ライセンスと注意

- **Remotion**: 個人・従業員 3 人以下の営利企業・非営利団体は無料。それ以外は
  [会社ライセンス](https://www.remotion.dev/docs/license)が必要です。
- **HyperFrames**: Apache-2.0。`reverse-any-video` と `style-sampling` のスキルは有料の Pro 機能なので使いません。
- **GSAP**: 商用利用も無料（独自の「no charge」ライセンス）。条文は https://gsap.com/standard-license で確認してください。
- **生成 AI**: 実在の人物や他社のロゴ・キャラクターに似せた生成はしません。BGM・音声の商用利用の範囲は、
  ElevenLabs の契約プランの規約で確認してください。
- **Chatterbox**: 生成した音声には電子透かしが入ります。他人の声を真似る場合は本人の許可が必要です。
- **Blender / FFmpeg**: GPL / LGPL。このリポジトリには本体を含めず、実行時に導入します。
- このリポジトリは公開されています。秘密情報や、権利を持たない素材はコミットしないでください。
  作品ごとの作業フォルダ（`work/`）と書き出し物（`out/`）はコミットされない設定です。

Claude 向けの詳しい手順と技術的な注意点は [CLAUDE.md](CLAUDE.md) にまとめています。
