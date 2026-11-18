# Douga-Henshu（動画編集）

Claude Code（クラウド環境）に、コードで動画を作らせるためのリポジトリです。
タイトルアニメーション、3D、ナレーション、字幕までを、GPU のないクラウド環境で作れます。

## 使えるもの

| 役割 | ツール | 状態 |
|---|---|---|
| 動画の土台（React） | [Remotion](https://github.com/remotion-dev/remotion) 4.0.543 + [Remotion Skills](https://github.com/remotion-dev/skills) | 書き出し確認済み |
| 動画の土台（HTML） | [HyperFrames](https://github.com/heygen-com/hyperframes) 0.9.43 + 公式スキル | 書き出し確認済み |
| 動き・文字アニメーション | [GSAP](https://github.com/greensock/GSAP)（HyperFrames 経由） | 書き出し確認済み |
| ブラウザの 3D | [Three.js](https://github.com/mrdoob/three.js) 0.187.0（Remotion 経由） | 書き出し確認済み |
| 本格 3D | [Blender](https://github.com/blender/blender) 5.3.0（Python から直接操作） | 書き出し確認済み |
| エンコード・確認 | [FFmpeg](https://github.com/FFmpeg/FFmpeg) 6.1.1（環境に最初から入っている） | 確認済み |
| ナレーション | [Chatterbox](https://github.com/resemble-ai/chatterbox) 0.1.7（日本語対応） | 生成確認済み |
| 字幕 | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.1 | 文字起こし確認済み |
| デザインの講評・仕上げ | [Impeccable](https://github.com/pbakaus/impeccable) 4.9.1 | 起動確認済み |
| 書き出し前の画面確認 | [Chrome DevTools MCP](https://github.com/ChromeDevTools/chrome-devtools-mcp) 1.13.0 | 動作確認済み |

入れていないもの: ComfyUI（GPU がないため）、Blender MCP（手元の PC 向けのため）、
yt-dlp（YouTube への通信が遮断されており、規約・著作権の問題もあるため）。

## 使い方

クラウド環境「**Douga-Henshu**」でセッションを始め、Claude に作りたい動画を伝えます。
依存関係・描画用ブラウザ・スキルは、セッション開始時に自動で準備されます（約 2 分）。

依頼の例:

```text
/hyperframes で、新商品紹介の 15 秒の縦型動画を作って。タイトルは「〇〇」。
```

```text
remotion で、この台本のナレーションと字幕つきの 30 秒の解説動画を作って。
```

自動の準備が失敗した時は、リポジトリのルートで次を実行すると入れ直せます。

```bash
bash setup/cloud-setup.sh
```

## 処理時間の目安（CPU 4 コアで実測）

| 処理 | 目安 |
|---|---|
| Remotion 1920×1080・3 秒の書き出し | 7〜13 秒 |
| HyperFrames 1920×1080・4 秒の書き出し | 約 7 秒 |
| Chatterbox ナレーション生成 | 音声の長さの約 4 倍 |
| faster-whisper 字幕（large-v3-turbo） | 音声の長さの約 1/3 |
| Blender（960×540・16 サンプル） | 1 フレーム約 3 秒 |

## ライセンスと注意

- **Remotion**: 個人・従業員 3 人以下の営利企業・非営利団体は無料。それ以外は
  [会社ライセンス](https://www.remotion.dev/docs/license)が必要です。
- **HyperFrames**: Apache-2.0。`reverse-any-video` と `style-sampling` のスキルは有料の Pro 機能です。
- **GSAP**: 商用利用も無料（独自の「no charge」ライセンス）。条文は https://gsap.com/standard-license で確認してください。
- **Chatterbox**: 生成した音声には電子透かしが入ります。他人の声を真似る場合は本人の許可が必要です。
- **Blender / FFmpeg**: GPL / LGPL。このリポジトリには本体を含めず、実行時に導入します。
- このリポジトリは公開されています。秘密情報や、権利を持たない素材はコミットしないでください。

いま有効な決まりと、よく当たる落とし穴は [CLAUDE.md](CLAUDE.md) に、決まりの経緯・測った記録・作品ごとの記録は [docs/notes/](docs/notes/) にまとめています。
