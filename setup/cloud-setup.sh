#!/usr/bin/env bash
# Douga-Henshu を Claude Code のクラウド環境で使える状態にする。
# クラウド環境「Douga-Henshu」のセットアップスクリプトから、セッション開始前に毎回呼ばれる。
# 手動で入れ直したい時も、リポジトリのルートで `bash setup/cloud-setup.sh` を実行すればよい。
#
# 各ツールの版はここと各 lock ファイルで固定している。上げる時はまとめて動作確認すること。
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

# このクラウド環境では外部通信はプロキシ経由のみ。Node.js にもプロキシを使わせる。
export NODE_USE_ENV_PROXY=1
export HYPERFRAMES_NO_TELEMETRY=1 HYPERFRAMES_NO_UPDATE_CHECK=1
export CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS=1

HYPERFRAMES_VERSION=0.9.43
CHROME_DEVTOOLS_MCP_VERSION=1.13.0
REMOTION_SKILLS_COMMIT=7e541aa4d7361dfaca0780a565a47307eb7bdfbb   # remotion 4.0.543 と対応
IMPECCABLE_COMMIT=c5dd5fc267b486d991229c4d4647e67ba955f39d       # impeccable 4.9.1
SKILLS_DIR="$HOME/.claude/skills"

log() { echo "[douga-setup] $*"; }

log "1/6 Remotion の依存関係と描画用ブラウザ"
(cd remotion && npm ci --no-audit --no-fund --loglevel=error && npx remotion browser ensure)

log "2/6 HyperFrames の CLI と描画用ブラウザ"
(cd hyperframes && npx --yes "hyperframes@${HYPERFRAMES_VERSION}" browser ensure)

log "3/6 Python ツール（Blender / faster-whisper / Chatterbox / 映像処理 tools/vfx）"
# MediaPipe（tools/vfx）の本体は libEGL.so.1 と libGLESv2.so.2 を必要とする（無いと読み込みで失敗する。2026-11-14 実測）
if ! ldconfig -p | grep -q "libEGL.so.1" || ! ldconfig -p | grep -q "libGLESv2.so.2"; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends libegl1 libgles2
fi
(cd tools && uv sync --frozen)

log "4/6 Chrome DevTools MCP を取得（.mcp.json から起動される）"
npx --yes "chrome-devtools-mcp@${CHROME_DEVTOOLS_MCP_VERSION}" --version >/dev/null

# 指定コミットのリポジトリから、必要なフォルダだけを取り出す。
fetch_dir() { # $1=リポジトリURL $2=コミット $3=取り出すパス $4=作業先
  git init -q "$4"
  git -C "$4" remote add origin "$1"
  git -C "$4" fetch -q --depth 1 --filter=blob:none origin "$2"
  git -C "$4" checkout -q FETCH_HEAD -- "$3"
}

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "$SKILLS_DIR"

log "5/6 スキル: Remotion Skills と Impeccable"
fetch_dir https://github.com/remotion-dev/skills "$REMOTION_SKILLS_COMMIT" skills "$TMP_DIR/remotion"
for dir in "$TMP_DIR"/remotion/skills/*/; do
  name=$(basename "$dir")
  rm -rf "${SKILLS_DIR:?}/$name"
  cp -r "$dir" "$SKILLS_DIR/$name"
done
fetch_dir https://github.com/pbakaus/impeccable "$IMPECCABLE_COMMIT" .claude/skills/impeccable "$TMP_DIR/impeccable"
rm -rf "${SKILLS_DIR:?}/impeccable"
cp -r "$TMP_DIR/impeccable/.claude/skills/impeccable" "$SKILLS_DIR/impeccable"

log "6/6 スキル: HyperFrames（CLI の版に合わせた中核スキル）"
(cd "$REPO_ROOT" && npx --yes "hyperframes@${HYPERFRAMES_VERSION}" skills update >/dev/null)

log "完了。入っているスキル:"
ls -1 "$SKILLS_DIR"
