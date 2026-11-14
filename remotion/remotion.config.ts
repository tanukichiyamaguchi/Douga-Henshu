/**
 * Note: When using the Node.JS APIs, the config file
 * doesn't apply. Instead, pass options directly to the APIs.
 *
 * All configuration options: https://remotion.dev/docs/config
 */

import { Config } from "@remotion/cli/config";

Config.setRspack(true);
Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);
Config.setDefaultPremountInSeconds(2);

// Claude Code のクラウド環境では、外部への通信は HTTPS_PROXY のプロキシを
// 通したものだけが許可される。Remotion は既定でブラウザを「プロキシなし」で
// 起動するため、そのままだと Web フォントや外部画像の取得が証明書エラーで失敗する。
// プロキシが設定されている時だけ、ブラウザにもそのプロキシを使わせる。
// （証明書の検証は無効化しない。ローカルの配信サーバーへの通信は直接つなぐ。）
const proxy = process.env.HTTPS_PROXY ?? process.env.https_proxy;
if (proxy) {
  Config.setChromiumFlags({
    "no-proxy-server": false,
    "proxy-server": proxy,
    "proxy-bypass-list": "localhost;127.0.0.1;[::1]",
  });
}
