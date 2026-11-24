"""広告でよく使われるフリーの BGM と効果音を、決まった 2 つのサイトから探して取る。

2026-11-23 にユーザーが決めたこと（「今後、有名な、よく広告で使われるフリーのBGMや効果音あるじゃないですか？
それを使用してほしい」）:
  - 効果音は「効果音ラボ」、BGM は「OpenTracks（旧 DOVA-SYNDROME）」から先に探す。
  - 合う物が無い時だけ、ElevenLabs で作る（gen_sfx.py・gen_music.py）。どの音をどこから取ったかを、納品の報告に書く。
規約の要点・確かめた日・気を付けることは docs/notes/free-audio.md。

使い方（tools/ で実行）:
  uv run python free_audio.py se キラキラ 決定                     # 効果音ラボで探す（語ごとに探して、番号を付けて並べる）
  uv run python free_audio.py se-get キラキラ 決定 --pick 1,4 --out ../work/<作品>/assets/se
                                                                  # 上と同じ語で探し直し、その番号の音を取る
  uv run python free_audio.py bgm-tags                             # BGM の検索タグの一覧
  uv run python free_audio.py bgm --tag 明るい --tag ポップ --detail
                                                                  # ダウンロードの多い順に探す（--detail で歌の有無と作曲者の条件も）
  uv run python free_audio.py bgm-info 2445                        # トラックごとの長さとループ、タグ、作曲者の条件
  uv run python free_audio.py bgm-get 2445 --track 2 --out ../work/<作品>/assets/bgm
  uv run python free_audio.py measure ../work/<作品>/assets/se/*.mp3   # 長さ・大きさ・帯域・鳴り始めと鳴り終わり
  uv run python free_audio.py sources ../work/<作品>/assets         # 納品の報告に書く「音の出どころ」の表

- 取った音は、作品のフォルダ（work/。Git に入れない）に置く。公開リポジトリに入れると再配布になるので、
  リポジトリの中の Git に入る場所には書き出さない（このスクリプトが止める）。
- 取った音の横に「<ファイル名>.source.json」（どのサイトの、誰の、何という音を、いつ、どの条件で取ったか）を書く。
- Claude は音を聞けない。名前・説明・タグ・ダウンロード数と、measure の数字で選ぶ。ユーザーが曲名や URL で指定してもよい。
- 効果音ラボは、音のファイルへの直リンクを禁止している。ファイルの URL は作品のフォルダの記録にだけ書き、リポジトリの
  資料・PR・公開するページには書かない（書くのは、サイトのページの URL）。
- 通信は curl で行う（このクラウド環境のプロキシと証明書の設定をそのまま使うため）。サイトの負担にならないよう、続けて
  読む時は 1 秒あける。取ったファイルは取り直さない（OpenTracks は、同じファイルを続けてダウンロードできない）。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
REPO = Path(__file__).resolve().parents[1]
JST = timezone(timedelta(hours=9))
WAIT = 1.0  # 続けて読む時にあける秒数
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}

SE_SITE = "https://soundeffect-lab.info"
SE_TERMS = [f"{SE_SITE}/agreement/", f"{SE_SITE}/faq/"]
# 音のファイルの置き場所（/sound/<分類>/mp3/...）と、サイトの分類の名前（2026-11-23 にサイトで確かめた）
SE_CATEGORIES = {
    "battle": "戦闘",
    "anime": "演出・アニメ",
    "voice": "声素材",
    "various": "生活",
    "machine": "機械・乗り物",
    "environment": "環境音",
    "button": "ボタン・システム音",
    "animal": "自然・動物",
}
SE_ITEM = re.compile(
    r"<li><span>(?P<name>(?:(?!</li>).)*?)</span>(?P<desc>(?:(?!</li>).)*?)"
    r'<a href="(?P<path>/sound/[^"]+?\.mp3)"',
    re.S,
)

OT_SITE = "https://opentracks.com"
OT_TERMS = [
    f"{OT_SITE}/help/articles/license/",
    f"{OT_SITE}/help/articles/license-usage/",
    f"{OT_SITE}/help/articles/license-prohibitions/",
    f"{OT_SITE}/help/articles/instagram/",
]
OT_WORKER = (
    "dova-worker.tracks-cid.workers.dev"  # OpenTracks が音のファイルを渡すサーバー
)
VOCAL_TAGS = {"ボーカル", "コーラス"}  # NFKC でそろえた後のタグの名前
# 曲をほかの所で配信している（準備している）ことにふれた言葉。2026-11-23 に、コメントに「TuneCore 社を介して
# 各種ストリーミングサイトでの配信準備中」と書いた曲があった。当てはまらない言葉にも反応するので、知らせるだけにする
ELSEWHERE = re.compile(
    r"TuneCore|チューンコア|ストリーミング|Spotify|Apple ?Music|iTunes|LINE ?MUSIC|Amazon ?Music"
    r"|サブスク|音楽配信|配信中|配信準備|配信開始|Content ?ID",
    re.I,
)

_last_request = 0.0


# ---------------------------------------------------------------- 共通


def norm(s: str) -> str:
    """半角カナ（ﾎﾟｯﾌﾟ）と全角（ポップ）をそろえる。"""
    return unicodedata.normalize("NFKC", html.unescape(s)).strip()


def clean(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    return re.sub(r"[ \t\r\f\v]+", " ", s).strip()


def one_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def short(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def now_jst() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def curl(args: list[str], url: str) -> bytes:
    global _last_request
    gap = time.monotonic() - _last_request
    if gap < WAIT:
        time.sleep(WAIT - gap)
    # プロキシが通信先を断ったかを見分けるため、プロキシの返事（http_connect）を標準エラーに書かせる。
    # --fail を付けると、プロキシに断られた時も「returned error: 403」としか出ず、サイトの 403 と区別できない。
    marker = "%{stderr}\n[http_connect=%{http_connect}]"
    args = list(args)
    if "-w" in args:
        args[args.index("-w") + 1] += marker
    else:
        args += ["-w", marker]
    result = subprocess.run(
        ["curl", "-sS", "--fail", "-A", UA, *args, url],
        capture_output=True,
        check=False,
    )
    _last_request = time.monotonic()
    err = result.stderr.decode("utf-8", "replace")
    connect = re.search(r"\[http_connect=(\d+)\]", err)
    err = re.sub(r"\s*\[http_connect=\d+\]", "", err).strip()
    if result.returncode != 0:
        host = urllib.parse.urlsplit(url).hostname
        if connect and connect.group(1) not in {"000", "200"}:
            sys.exit(
                f"通信先 {host} は、この環境のネットワークの設定で許可されていません（プロキシの返事 "
                f"{connect.group(1)}）。\nこの通信先 1 つを環境の設定に足す提案をして、承認を受けてから使う"
                "（CLAUDE.md「この環境の前提」）。"
            )
        sys.exit(f"取得に失敗しました: {url}\n{err}")
    return result.stdout


def get_text(url: str, extra: list[str] | None = None) -> str:
    return curl(extra or [], url).decode("utf-8", "replace")


def check_out_dir(out: Path) -> Path:
    """取った音を、Git に入る場所へ書き出さない。"""
    out = out.resolve()
    try:
        out.relative_to(REPO)
    except ValueError:
        return out  # リポジトリの外（スクラッチパッドなど）
    probe = out / "probe.mp3"
    ignored = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", str(probe)], check=False
    )
    if ignored.returncode != 0:
        sys.exit(
            f"{out} は Git に入る場所です。取った音はリポジトリに入れない（公開リポジトリなので、再配布になる）。"
            "work/<作品>/assets/ の下を指定する。"
        )
    return out


def probe_audio(path: Path) -> dict | None:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bit_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        data = json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") or []
    if not streams:
        return None
    s = streams[0]
    return {
        "codec": s.get("codec_name"),
        "sample_rate": int(s.get("sample_rate") or 0),
        "channels": s.get("channels"),
        "bit_rate": int(s["bit_rate"]) if s.get("bit_rate") else None,
        "seconds": round(float(data.get("format", {}).get("duration") or 0), 3),
    }


def download(url: str, dest: Path, extra: list[str]) -> None:
    part = dest.with_name(dest.name + ".part")
    curl([*extra, "-o", str(part)], url)
    if probe_audio(part) is None:
        part.unlink(missing_ok=True)
        sys.exit(
            f"音のファイルではない物が返ってきました（取り直さずに止めます）: {url}"
        )
    part.rename(dest)


def write_source(path: Path, info: dict) -> None:
    data = path.read_bytes()
    record = {
        "file": path.name,
        **info,
        "downloaded_at": now_jst(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "format": probe_audio(path),
        "measure": measure_file(path),
    }
    side = path.with_name(path.name + ".source.json")
    side.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  記録: {side.name}")


# ---------------------------------------------------------------- 効果音ラボ


def se_search(word: str, pages: int) -> tuple[list[dict], int]:
    """1 つの語で探す。名前と説明に、その語（ひと続きの文字）を含む物が出る。1 ページ 30 件。"""
    items: list[dict] = []
    last = 1
    page = 1
    while page <= min(pages, last):
        query = {"s": word} if page == 1 else {"s": word, "page": page}
        url = f"{SE_SITE}/sound/search.php?" + urllib.parse.urlencode(query)
        s = get_text(url)
        numbers = re.findall(r'search\.php\?s=[^"&]*&(?:amp;)?page=(\d+)', s)
        last = max([last, *(int(n) for n in numbers)])
        for m in SE_ITEM.finditer(s):
            path = m.group("path")
            folder = path.split("/")[2]
            items.append(
                {
                    "name": one_line(clean(m.group("name"))),
                    "desc": one_line(clean(m.group("desc"))),
                    "path": path,
                    "folder": folder,
                    "category": SE_CATEGORIES.get(folder, folder),
                    "found_by": url,
                }
            )
        page += 1
    return items, last


def se_collect(words: list[str], pages: int) -> list[dict]:
    """語ごとに探して、同じ音を 1 つにまとめる（2 語を一度に入れると、ひと続きの文字として探されて 0 件になる）。"""
    merged: dict[str, dict] = {}
    for word in words:
        items, last = se_search(word, pages)
        more = (
            f"（{last} ページのうち {pages} ページまで読んだ。続きは --pages を増やす）"
            if last > pages
            else ""
        )
        print(f"「{word}」: {len(items)} 件{more}")
        for it in items:
            if it["path"] in merged:
                merged[it["path"]]["words"].append(word)
            else:
                merged[it["path"]] = {**it, "words": [word]}
    return list(merged.values())


def cmd_se(args: argparse.Namespace) -> None:
    results = se_collect(args.words, args.pages)
    for i, it in enumerate(results, 1):
        print(
            f"{i:3}. {it['name']}｜{short(it['desc'], 40)}｜{it['category']}｜{it['path']}"
        )
    if not results:
        print(
            "見つかりませんでした（言い換えて探す。合う物が無ければ ElevenLabs で作る）"
        )
    else:
        print(
            "取る時: se-get に同じ語を同じ順で渡し、--pick に番号かファイルの場所（/sound/...）を渡す"
        )


def pick_items(picks: list[str], results: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    for token in [t.strip() for p in picks for t in p.split(",") if t.strip()]:
        if token.isdigit():
            n = int(token)
            if not 1 <= n <= len(results):
                sys.exit(f"番号 {n} はありません（1〜{len(results)}）")
            chosen.append(results[n - 1])
        else:
            path = urllib.parse.urlsplit(token).path or token
            hits = [it for it in results if it["path"] == path]
            if not hits:
                sys.exit(f"{token} は、この語の検索結果にありません")
            chosen.append(hits[0])
    return chosen


def cmd_se_get(args: argparse.Namespace) -> None:
    out = check_out_dir(args.out)
    results = se_collect(args.words, args.pages)
    chosen = pick_items(args.pick, results)
    out.mkdir(parents=True, exist_ok=True)
    for it in chosen:
        url = SE_SITE + it["path"]
        dest = out / f"selab-{it['folder']}-{Path(it['path']).name}"
        print(f"{it['name']}（{it['category']}）→ {dest.name}")
        if dest.exists():
            print("  取得済み（取り直さない）")
        else:
            download(url, dest, ["-e", it["found_by"]])
        notes = [
            "再配布（音そのものを配る・素材集・音が主役の動画）は不可。直リンク禁止（ファイルの URL を公開の場に書かない）",
        ]
        if it["folder"] == "voice":
            notes.append(
                "声素材: 声優の尊厳を傷つける使い方・声優が被害を受ける使い方は禁止。クレジットを書くなら「声素材提供：効果音ラボ」の形"
            )
            print(f"  → {notes[-1]}")
        side = dest.with_name(dest.name + ".source.json")
        if not side.exists():
            write_source(
                dest,
                {
                    "site": "効果音ラボ",
                    "name": it["name"],
                    "description": it["desc"],
                    "category": it["category"],
                    "creator": "効果音ラボ",
                    "page": f"{SE_SITE}/sound/{it['folder']}/",
                    "found_by": it["found_by"],
                    "file_url": url,
                    "terms": SE_TERMS,
                    "notes": notes,
                },
            )


# ---------------------------------------------------------------- OpenTracks


def ot_tag_table() -> dict[str, tuple[str, str, str]]:
    """タグの名前（NFKC でそろえた物）→（種類 tags_m など、番号 m01 など、サイトの表記）"""
    s = get_text(f"{OT_SITE}/bgm/search")
    table: dict[str, tuple[str, str, str]] = {}
    for kind in ("tags_m", "tags_r", "tags_i"):
        pattern = (
            r'<input type="checkbox" name="'
            + kind
            + r'" value="([a-z]\d+)"[^>]*>([^<]+)</label>'
        )
        for code, label in re.findall(pattern, s):
            table[norm(label)] = (kind, code, label.strip())
    if not table:
        sys.exit("タグの一覧を読めませんでした（サイトの作りが変わったかもしれない）")
    return table


def resolve_tags(names: list[str]) -> list[tuple[str, str, str]]:
    if not names:
        return []
    table = ot_tag_table()
    chosen = []
    for name in names:
        key = norm(name)
        if key in table:
            chosen.append(table[key])
            continue
        near = [k for k in table if key in k]
        if len(near) == 1:
            chosen.append(table[near[0]])
            continue
        hint = "、".join(near) if near else "bgm-tags で一覧を見る"
        sys.exit(f"タグ「{name}」がありません（候補: {hint}）")
    return chosen


def cmd_bgm_tags(args: argparse.Namespace) -> None:
    table = ot_tag_table()
    groups = {
        "tags_m": "雰囲気・感情・世界観",
        "tags_r": "リズム・ジャンル",
        "tags_i": "楽器",
    }
    for kind, title in groups.items():
        names = [k for k, v in table.items() if v[0] == kind]
        print(f"{title}（{len(names)}）: " + " ".join(names))


def parse_list(s: str) -> list[dict]:
    items = []
    for block in s.split('<div class="bgmlist" ')[1:]:
        bgm_id = re.search(r'data-private-bgm="(\d+)"', block)
        title = re.search(
            r'class="bgmlist-tracktitle"><a href="/bgm/detail/\d+">(.*?)</a>'
            r'<span>by<a href="/creator/detail/(\d+)">(.*?)</a>',
            block,
            re.S,
        )
        if not bgm_id or not title:
            continue

        def field(cls: str) -> str:
            m = re.search(r'class="' + cls + r'">(.*?)</div>', block, re.S)
            return one_line(clean(m.group(1))) if m else ""

        tags_html = re.search(r'class="bgmlist-icotags">(.*?)</div>', block, re.S)
        tags = re.findall(
            r"<span>([^<]*)</span>", tags_html.group(1) if tags_html else ""
        )
        loop = "bgmlist-icoloop_yes" in block
        items.append(
            {
                "id": int(bgm_id.group(1)),
                "title": one_line(clean(title.group(1))),
                "composer_id": title.group(2),
                "composer": one_line(clean(title.group(3))),
                "kind": field("bgmlist-icobgm"),
                "tracks": re.sub(r"\D", "", field("bgmlist-icolist")) or "1",
                "time": field("bgmlist-icotime").rstrip("~"),
                "loop": loop,
                "downloads": int(re.sub(r"\D", "", field("bgmlist-icodl")) or 0),
                "tags": [norm(t) for t in tags if t != "..."],
                "tags_cut": "..." in tags,
                "comment": field("bgmlist-trackdetail"),
            }
        )
    return items


def cmd_bgm(args: argparse.Namespace) -> None:
    tags = resolve_tags(args.tag)
    base = [
        ("sort", "2" if args.sort == "dl" else "1"),
        ("kind", {"any": "0", "bgm": "1", "jingle": "2"}[args.kind]),
        ("duration", str(args.duration)),
        ("loop", {"any": "0", "yes": "1", "no": "2"}[args.loop]),
        ("sheet", "0"),
        ("vocal", "0"),
        *[(kind, code) for kind, code, _ in tags],
    ]
    if args.keyword:
        base.append(("keywords", args.keyword))
    if args.exclude:
        base.append(("negative_keywords", args.exclude))
    order = "ダウンロードの多い順" if args.sort == "dl" else "新しい順"
    shown = 0
    for page in range(1, args.pages + 1):
        url = f"{OT_SITE}/bgm?" + urllib.parse.urlencode([*base, ("page", str(page))])
        s = get_text(url)
        if page == 1:
            head = re.search(r'<div class="home-headex">(.*?)</div>', s, re.S)
            print(
                f"OpenTracks（{order}）: {one_line(clean(head.group(1))) if head else '?'}"
            )
            print(f"  検索の URL: {url.replace('&page=1', '')}")
        items = parse_list(s)
        if not items:
            break
        for it in items:
            shown += 1
            loop = "ループ可" if it["loop"] else "ループなし"
            tags = " ".join(it["tags"]) + ("…" if it["tags_cut"] else "")
            print(
                f"{it['id']:>6}｜{it['title']}｜{it['composer']}｜{it['kind']}・{it['tracks']} トラック・"
                f"最短 {it['time']}・{loop}｜DL {it['downloads']:,}｜{tags}"
            )
            if args.detail:
                d = ot_detail(it["id"])
                vocal = (
                    "歌もの（タグにボーカル・コーラス）"
                    if d["vocal"]
                    else "歌のタグなし"
                )
                elsewhere = (
                    "｜コメントにほかの所での配信の言葉あり（bgm-info で読む）"
                    if ELSEWHERE.search(d["comment"])
                    else ""
                )
                print(
                    f"        → {vocal}｜作曲者の条件: {d['composer_license']}{elsewhere}"
                )
    if shown == 0:
        print(
            "見つかりませんでした（条件を減らす。合う物が無ければ ElevenLabs で作る）"
        )
    elif not args.detail:
        print(
            "歌の有無と作曲者の条件は --detail か bgm-info で確かめる（一覧のタグは途中で切れている）"
        )


def ot_detail(bgm_id: int, track: int = 1) -> dict:
    url = f"{OT_SITE}/bgm/detail/{bgm_id}" + (f"/track/{track}" if track > 1 else "")
    s = get_text(url)
    head = re.search(
        r'<h1 class="trialL-title">(.*?)<span>\s*written by <a href="/creator/detail/(\d+)">(.*?)</a>',
        s,
        re.S,
    )
    box = re.search(
        r'<div class="trialL-detail">(.*?)<div class="trialL-below">', s, re.S
    )
    if not head or not box:
        sys.exit(
            f"曲のページを読めませんでした（番号の間違いか、サイトの作りが変わった）: {url}"
        )
    fields = dict(
        re.findall(r"<div>([^<>]+)<span>：</span>([^<]*)</div>", box.group(1))
    )
    loop = re.search(r"trialL-detailloop_(yes|no)", box.group(1))
    comment = re.search(r'<p class="trialL-below_trackP">(.*?)</p>', s, re.S)
    tags_html = re.search(r'<div class="trialL-below_tags">(.*?)</div>', s, re.S)
    tags = re.findall(
        r'<a href="/bgm\?tags_[mri]=[a-z]\d+">([^<]*)</a>',
        tags_html.group(1) if tags_html else "",
    )
    license_ = re.search(
        r'<span class="trialL-subm_composerLicense">(.*?)</span>', s, re.S
    )
    date = re.search(r'trialL-above_date">公開日<span>：</span>([^<]*)</div>', s)
    count = fields.get("Tracks", "1/1").split("/")
    return {
        "id": bgm_id,
        "url": url,
        "title": one_line(clean(head.group(1))),
        "composer_id": head.group(2),
        "composer": one_line(clean(head.group(3))),
        "kind": fields.get("素材種別", "").strip(),
        "track": int(count[0]),
        "tracks": int(count[-1]),
        "length": fields.get("再生時間", "").strip(),
        "loop": bool(loop and loop.group(1) == "yes"),
        "downloads": int(re.sub(r"\D", "", fields.get("DL", "")) or 0),
        "published": date.group(1).strip() if date else "",
        "comment": clean(comment.group(1)) if comment else "",
        "tags": [norm(t) for t in tags],
        "vocal": any(norm(t) in VOCAL_TAGS for t in tags),
        "composer_license": one_line(clean(license_.group(1))) if license_ else "?",
    }


def ot_composer_terms(composer_id: str) -> str:
    s = get_text(f"{OT_SITE}/creator/detail/{composer_id}")
    m = re.search(r'<p class="creator-mainprof_addLicense">(.*?)</p>', s, re.S)
    return clean(m.group(1)) if m else ""


def ot_info(bgm_id: int) -> dict:
    info = ot_detail(bgm_id)
    tracks = [{"track": 1, "length": info["length"], "loop": info["loop"]}]
    for n in range(2, info["tracks"] + 1):
        t = ot_detail(bgm_id, n)
        tracks.append({"track": n, "length": t["length"], "loop": t["loop"]})
    info["track_list"] = tracks
    info["composer_terms"] = (
        ot_composer_terms(info["composer_id"])
        if info["composer_license"] != "サイト準拠"
        else ""
    )
    return info


def warnings_for(info: dict) -> list[str]:
    out = []
    if info["vocal"]:
        out.append(
            "歌もの（タグにボーカル・コーラス）。CLAUDE.md の BGM は歌なし。歌の無いトラックがあるかを、コメントで確かめる"
        )
    if info["composer_license"] != "サイト準拠":
        terms = info.get("composer_terms") or "（作曲者のページで読めなかった）"
        out.append(f"作曲者の条件あり（読んで、守れるかを確かめる）: {terms}")
    words = sorted({m.group(0) for m in ELSEWHERE.finditer(info["comment"])})
    if words:
        out.append(
            f"コメントに、ほかの所での配信にふれた言葉がある（{'・'.join(words)}）。曲が音楽配信に出ていると、"
            "配信の会社が音の指紋（Content ID など）を登録していて、広告に権利の申し立てが来るおそれがある"
            "（Claude の解釈）。コメントを読み、当てはまる時はほかの曲を選ぶ"
        )
    return out


def cmd_bgm_info(args: argparse.Namespace) -> None:
    info = ot_info(args.id)
    print(
        f"{info['id']}「{info['title']}」 作曲: {info['composer']}（作曲者の条件: {info['composer_license']}）"
    )
    print(
        f"  公開日 {info['published']}・ダウンロード {info['downloads']:,}・素材の種類 {info['kind']}"
    )
    for t in info["track_list"]:
        print(
            f"  トラック{t['track']}: {t['length']}・{'ループ可' if t['loop'] else 'ループなし'}"
        )
    print(f"  タグ: {' '.join(info['tags'])}")
    print("  コメント: " + one_line(info["comment"]))
    print(f"  ページ: {OT_SITE}/bgm/detail/{info['id']}")
    for line in warnings_for(info):
        print(f"  → {line}")


def cmd_bgm_get(args: argparse.Namespace) -> None:
    out = check_out_dir(args.out)
    dest = out / f"ot{args.id}-{args.track}.mp3"
    side = dest.with_name(dest.name + ".source.json")
    if dest.exists() and side.exists():
        print(f"取得済み（取り直さない）: {dest}")
        return
    info = ot_info(args.id)
    if not 1 <= args.track <= info["tracks"]:
        sys.exit(f"トラック {args.track} はありません（この曲は 1〜{info['tracks']}）")
    print(f"{info['id']}「{info['title']}」 作曲: {info['composer']} → {dest.name}")
    for line in warnings_for(info):
        print(f"  → {line}")
    label = ""
    if dest.exists():
        print("  音は取得済み（取り直さない）。記録だけ書く")
    else:
        out.mkdir(parents=True, exist_ok=True)
        label = ot_download(args.id, args.track, dest)
    track = info["track_list"][args.track - 1]
    write_source(
        dest,
        {
            "site": "OpenTracks（旧 DOVA-SYNDROME）",
            "name": info["title"],
            "track": args.track,
            "track_label": label,
            "track_length": track["length"],
            "track_loop": track["loop"],
            "creator": info["composer"],
            "creator_page": f"{OT_SITE}/creator/detail/{info['composer_id']}",
            "creator_license": info["composer_license"],
            "creator_terms": info["composer_terms"],
            "vocal": info["vocal"],
            "tags": info["tags"],
            "comment": info["comment"],
            "page": f"{OT_SITE}/bgm/detail/{info['id']}",
            "terms": OT_TERMS,
            "notes": [
                "クレジット不要。音を取り出せる形で配る・音楽が主役の作品・JASRAC や Content ID への登録は不可",
            ],
            "warnings": warnings_for(info),
        },
    )


def ot_download(bgm_id: int, track: int, dest: Path) -> str:
    """ダウンロードの画面で選び、返ってきた行き先（OpenTracks のファイルのサーバー）から取る。"""
    page = f"{OT_SITE}/bgm/detail/{bgm_id}/download"
    with tempfile.TemporaryDirectory() as tmp:
        jar = str(Path(tmp) / "cookies.txt")
        s = get_text(page, ["-c", jar, "-b", jar])
        token = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', s)
        select = re.search(r'<select name="track"[^>]*>(.*?)</select>', s, re.S)
        if not token or not select:
            sys.exit(
                f"ダウンロードの画面を読めませんでした（サイトの作りが変わったかもしれない）: {page}"
            )
        options = dict(
            re.findall(r'<option value="(\d+)"[^>]*>([^<]*)</option>', select.group(1))
        )
        if str(track) not in options:
            sys.exit(f"ダウンロードの画面に、トラック {track} がありません: {options}")
        label = one_line(html.unescape(options[str(track)]))
        print(f"  選ぶ: {label}")
        answer = curl(
            [
                "-c",
                jar,
                "-b",
                jar,
                "-e",
                page,
                "-o",
                os.devnull,
                "-w",
                "%{http_code} %{redirect_url}",
                "--data-urlencode",
                f"csrfmiddlewaretoken={token.group(1)}",
                "--data-urlencode",
                f"track={track}",
            ],
            page,
        ).decode("utf-8", "replace")
    code, _, location = answer.partition(" ")
    target = urllib.parse.urlsplit(location.strip())
    if (
        code not in {"301", "302", "303"}
        or target.scheme != "https"
        or target.hostname != OT_WORKER
    ):
        sys.exit(
            f"ダウンロードの行き先が、いつもと違います（{code} {target.scheme}://{target.hostname}）。"
            "サイトの作りが変わったかもしれないので、取らずに止めます"
        )
    download(location.strip(), dest, [])
    return label


# ---------------------------------------------------------------- 測る


def decode(path: Path, sr: int):
    import numpy as np

    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def loudness(path: Path) -> tuple[float | None, float | None]:
    """ffmpeg の ebur128 で、全体の大きさ（LUFS）と、山（dBTP）。0.4 秒より短い音は LUFS が出ない。"""
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostats",
            "-hide_banner",
            "-i",
            str(path),
            "-vn",
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    summary = result.stderr[result.stderr.rfind("Summary:") :]
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    p = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    lufs = float(i.group(1)) if i else None
    if lufs is not None and lufs <= -70:
        lufs = None
    return lufs, float(p.group(1)) if p else None


def measure_file(path: Path) -> dict:
    import numpy as np
    from scipy.signal import stft

    sr = 48000
    x = decode(path, sr)
    if len(x) < sr // 100:
        return {"seconds": round(len(x) / sr, 3), "note": "短すぎて測れない"}
    hop = sr // 100  # 10 ミリ秒ごとの大きさ
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-12)
    top = float(db.max())
    if top < -100:
        return {"seconds": round(len(x) / sr, 3), "note": "無音"}
    loud = np.where(db >= top - 30)[0]
    tail = np.where(db >= top - 40)[0]
    f, _, z = stft(x, fs=sr, nperseg=2048, noverlap=1536, window="hann")
    power = np.abs(z) ** 2
    energy = power.sum(axis=0)
    spectrum = power[:, energy >= energy.max() * 10 ** (-40 / 10)].sum(axis=1)
    band = (f >= 20) & (f < 20000)
    total = spectrum[band].sum() + 1e-20

    def share(lo: float, hi: float) -> int:
        return round(float(spectrum[(f >= lo) & (f < hi)].sum() / total * 100))

    lufs, peak = loudness(path)
    return {
        "seconds": round(len(x) / sr, 3),
        "lufs": lufs,
        "true_peak_db": peak,
        "loudest_10ms_db": round(top, 1),
        "low_under_300hz_pct": share(20, 300),
        "mid_300_3000hz_pct": share(300, 3000),
        "high_over_3000hz_pct": share(3000, 20000),
        "centroid_hz": round(float((f[band] * spectrum[band]).sum() / total)),
        "onset_s": round(float(loud[0]) * hop / sr, 2),
        "peak_s": round(float(db.argmax()) * hop / sr, 2),
        "end_s": round(float(tail[-1] + 1) * hop / sr, 2),
    }


def measure_notes(m: dict) -> list[str]:
    """仮の目安（Claude が決めた線。聞こえ方は、この数字では分からない）。"""
    out = []
    if "loudest_10ms_db" not in m:
        return [m.get("note", "")]
    if m["loudest_10ms_db"] < -45:
        out.append("ほぼ無音（10 ミリ秒ごとの大きさの山が −45dBFS 未満）")
    if m["low_under_300hz_pct"] >= 80:
        out.append("300Hz 未満が 8 割以上（スマホのスピーカーでは聞こえにくい）")
    if m["onset_s"] >= 0.3:
        out.append(
            f"鳴り始めまで {m['onset_s']} 秒ある（置く時は、この分を切るか、ずらす）"
        )
    return out


def cmd_measure(args: argparse.Namespace) -> None:
    results = []
    for path in args.files:
        fmt = probe_audio(path)
        if fmt is None:
            print(f"{path.name}: 音のファイルとして読めない")
            continue
        m = measure_file(path)
        results.append({"file": str(path), "format": fmt, **m})
        print(
            f"{path.name}（{fmt['codec']}・{fmt['sample_rate']}Hz・{fmt['channels']}ch・{m['seconds']} 秒）"
        )
        if "loudest_10ms_db" in m:
            lufs = (
                f"{m['lufs']} LUFS"
                if m["lufs"] is not None
                else "LUFS なし（0.4 秒未満など）"
            )
            peak = f"{m['true_peak_db']} dBTP" if m["true_peak_db"] is not None else "−"
            print(
                f"  大きさ {lufs}・山 {peak}・10ms の山 {m['loudest_10ms_db']} dBFS"
                f"｜帯域 300Hz 未満 {m['low_under_300hz_pct']}%・300〜3kHz {m['mid_300_3000hz_pct']}%・"
                f"3kHz 以上 {m['high_over_3000hz_pct']}%（重心 {m['centroid_hz']:,}Hz）"
            )
            print(
                f"  鳴り始め {m['onset_s']} 秒・山 {m['peak_s']} 秒・鳴り終わり {m['end_s']} 秒"
                "（山から −30dB で始まり、−40dB で終わりとした）"
            )
        for line in measure_notes(m):
            print(f"  → {line}")
    print("（目安は Claude が決めた仮の線。どう聞こえるかは、この数字では分からない）")
    if args.json:
        args.json.write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"保存: {args.json}")


# ---------------------------------------------------------------- 出どころ


def cmd_sources(args: argparse.Namespace) -> None:
    files = sorted(
        p for p in args.dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT
    )
    rows, missing = [], []
    for path in files:
        side = path.with_name(path.name + ".source.json")
        if not side.exists():
            missing.append(path)
            continue
        d = json.loads(side.read_text(encoding="utf-8"))
        if d.get("site", "").startswith("OpenTracks"):
            name = f"{d['name']}（トラック{d['track']}・{d.get('track_length', '')}）"
            terms = f"作曲者の条件: {d.get('creator_license', '?')}"
        else:
            name = f"{d['name']}（{d.get('category', '')}）"
            terms = "サイトの規約どおり"
        rows.append(
            (
                path.relative_to(args.dir),
                d.get("site", "?"),
                name,
                d.get("creator", ""),
                terms,
                d.get("page", ""),
            )
        )
    print("| ファイル | 出どころ | 音・曲の名前 | 作った人 | 条件 | ページ |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    if missing:
        print(
            "\n出どころの記録が無い音（ElevenLabs で作った物・ナレーションなど。報告には手で書く）:"
        )
        for path in missing:
            print(f"  - {path.relative_to(args.dir)}")


# ---------------------------------------------------------------- 入口


def main() -> None:
    parser = argparse.ArgumentParser(
        description="フリーの BGM（OpenTracks）と効果音（効果音ラボ）を探して取る"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("se", help="効果音ラボで探す")
    p.add_argument("words", nargs="+", help="探す語（語ごとに探して、まとめて並べる）")
    p.add_argument(
        "--pages",
        type=int,
        default=3,
        help="1 語あたり読むページ数（1 ページ 30 件。既定 3）",
    )
    p.set_defaults(func=cmd_se)

    p = sub.add_parser(
        "se-get", help="効果音ラボの音を取る（se と同じ語で探し直して、番号で選ぶ）"
    )
    p.add_argument("words", nargs="+", help="se に渡したのと同じ語（同じ順）")
    p.add_argument(
        "--pick",
        action="append",
        required=True,
        help="番号（1,4 の形で複数可）か、ファイルの場所 /sound/...",
    )
    p.add_argument("--pages", type=int, default=3, help="se と同じ値にする（既定 3）")
    p.add_argument(
        "--out", type=Path, required=True, help="置き場所（work/<作品>/assets/se など）"
    )
    p.set_defaults(func=cmd_se_get)

    p = sub.add_parser("bgm-tags", help="OpenTracks の検索タグの一覧")
    p.set_defaults(func=cmd_bgm_tags)

    p = sub.add_parser("bgm", help="OpenTracks で BGM を探す")
    p.add_argument(
        "--tag",
        action="append",
        default=[],
        help="検索タグ（複数はすべてを満たす物。ポップ・ﾎﾟｯﾌﾟ どちらの書き方でもよい）",
    )
    p.add_argument("--keyword", help="キーワード")
    p.add_argument("--exclude", help="除くキーワード")
    p.add_argument(
        "--sort",
        choices=["dl", "new"],
        default="dl",
        help="dl: ダウンロードの多い順（既定）、new: 新しい順",
    )
    p.add_argument(
        "--kind",
        choices=["bgm", "jingle", "any"],
        default="bgm",
        help="素材の種類（既定 bgm。jingle は短い曲）",
    )
    p.add_argument(
        "--duration",
        type=int,
        choices=range(8),
        default=0,
        help="長さ（いちばん短いトラック）。0: 指定なし、1: 〜0:29、2: 0:30〜0:59、3: 1:00〜1:29、4: 1:30〜1:59、5: 2:00〜2:29、6: 2:30〜2:59、7: 3:00〜",
    )
    p.add_argument(
        "--loop",
        choices=["any", "yes", "no"],
        default="any",
        help="ループできるトラック",
    )
    p.add_argument(
        "--pages", type=int, default=2, help="読むページ数（1 ページ 5 曲。既定 2）"
    )
    p.add_argument(
        "--detail",
        action="store_true",
        help="曲ごとのページも読み、歌の有無と作曲者の条件を出す",
    )
    p.set_defaults(func=cmd_bgm)

    p = sub.add_parser(
        "bgm-info", help="OpenTracks の曲の、トラック・タグ・作曲者の条件"
    )
    p.add_argument("id", type=int, help="曲の番号（/bgm/detail/<番号>）")
    p.set_defaults(func=cmd_bgm_info)

    p = sub.add_parser("bgm-get", help="OpenTracks の曲を取る")
    p.add_argument("id", type=int, help="曲の番号")
    p.add_argument(
        "--track", type=int, default=1, help="トラックの番号（bgm-info で見る。既定 1）"
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="置き場所（work/<作品>/assets/bgm など）",
    )
    p.set_defaults(func=cmd_bgm_get)

    p = sub.add_parser("measure", help="長さ・大きさ・帯域・鳴り始めと鳴り終わりを測る")
    p.add_argument("files", type=Path, nargs="+", help="音のファイル")
    p.add_argument("--json", type=Path, help="数字を JSON で保存する先")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser(
        "sources", help="フォルダの中の音の出どころを、報告に貼れる表にする"
    )
    p.add_argument("dir", type=Path, help="作品の assets フォルダなど")
    p.set_defaults(func=cmd_sources)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
