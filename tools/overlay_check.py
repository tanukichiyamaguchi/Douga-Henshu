"""書き出した MP4 を、文字なしの土台の動画と全コマ比べて、足した物（文字・札・飾り）の範囲を測る。

`npm run check` は DOM の範囲を見るので、白い縁取り（text-shadow・box-shadow）と、出てくる時の行き過ぎ
（back.out など）は入らない。この道具は、書き出した絵そのものを比べるので、それらも入る。

測る物:
- 安全域（既定は x 140〜940・y 300〜1240。1080×1920 の広告の決まり）からはみ出すコマ
- 顔（顔の点を結んだ面）と重なるコマ、手（見つかったコマだけ）と重なるコマ
- 場面ごとの、足した物の範囲（--scenes で場面の始まりのコマを渡した時）

使い方（tools/ で）:
  uv run python overlay_check.py <書き出した MP4> <土台の MP4> [--json <出力>] [--scenes 0,211,277]

土台の動画は、書き出しと同じ長さ・同じコマ数・同じ絵で、文字や札だけが無い物（映像を 1 本に組んだ物、または
文字の場面を外して書き出した物）。色の差が小さい物（土台とほぼ同じ色の文字）は見つからない。手は、ぶれたコマでは
見つからないことがある。顔・手は土台の側で探す。顔は、顔の点を結んだ面なので、髪は入らない（髪との近さは、
コマの一覧などで別に見る）。顔・手が見つからなかったコマは測れていないので、結果に、見つかったコマ数を出す。
"""
import argparse
import json
import subprocess

import cv2
import mediapipe as mp
import numpy as np

from vfx import models


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return int(st["width"]), int(st["height"]), float(num) / float(den)


def frames(path, w, h):
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}:in_color_matrix=bt709", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def hull_mask(pts, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.int32(pts)), 255)
    return m


def runs(idx):
    out, s, prev = [], None, None
    for i in idx:
        if s is None:
            s = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append((s, prev))
            s = prev = i
    if s is not None:
        out.append((s, prev))
    return out


def main():
    ap = argparse.ArgumentParser(description="書き出しと土台を全コマ比べ、足した物の範囲と、顔・手との重なりを測る")
    ap.add_argument("render", help="書き出した MP4")
    ap.add_argument("base", help="文字なしの土台の MP4（同じ長さ・同じコマ数）")
    ap.add_argument("--json", help="コマごとの結果を書き出す先")
    ap.add_argument("--safe", default="140,940,300,1240", help="安全域 x0,x1,y0,y1（既定は 1080×1920 の広告の決まり）")
    ap.add_argument("--th", type=int, default=40, help="足した物とみなす色の差（0〜255）")
    ap.add_argument("--tol", type=int, default=4, help="圧縮のにじみの許し幅（px）")
    ap.add_argument("--scenes", help="場面の始まりのコマ（例: 0,211,277）。場面ごとの範囲を出す")
    a = ap.parse_args()
    w, h, fps = probe(a.render)
    x0, x1, y0, y1 = (int(v) for v in a.safe.split(","))
    face = models.mp_task("face_landmarker", video=False, num_faces=3)
    try:
        hand = models.mp_task("hand_landmarker", video=False, num_hands=2)
    except Exception as e:          # モデルが取れない時は、手は測らない
        hand = None
        print("手のモデルは使えません:", str(e)[:80])
    k = np.ones((5, 5), np.uint8)
    rows = []
    for f, (r, b) in enumerate(zip(frames(a.render, w, h), frames(a.base, w, h))):
        d = cv2.absdiff(r, b).max(axis=2)
        m = cv2.morphologyEx((d > a.th).astype(np.uint8) * 255, cv2.MORPH_OPEN, k)
        row = {"f": f, "t": round(f / fps, 3), "px": int((m > 0).sum())}
        added = row["px"] > 200
        if added:
            ys, xs = np.where(m > 0)
            row["box"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
        small = cv2.cvtColor(cv2.resize(b, (w // 2, h // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
        row["faces"] = []
        fo = 0
        for lm in face.detect(img).face_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in lm])
            row["faces"].append([int(pts[:, 0].min()), int(pts[:, 0].max()), int(pts[:, 1].min()), int(pts[:, 1].max())])
            if added:
                fo += int(((m > 0) & (hull_mask(pts, w, h) > 0)).sum())
        row["face_overlap"] = fo
        ho = 0
        if hand is not None:
            hr = hand.detect(img)
            row["hands"] = len(hr.hand_landmarks)
            for lm in hr.hand_landmarks:
                pts = np.array([[p.x * w, p.y * h] for p in lm])
                if added:
                    ho += int(((m > 0) & (hull_mask(pts, w, h) > 0)).sum())
        row["hand_overlap"] = ho
        rows.append(row)
    face.close()
    if hand is not None:
        hand.close()
    if a.json:
        with open(a.json, "w") as fp:
            json.dump(rows, fp)
    print(f"コマ数: {len(rows)}（{w}×{h}・{fps:g}fps） 足した物があるコマ: {sum(1 for r in rows if 'box' in r)}")
    nz = [r["px"] for r in rows if "box" not in r]
    print("足した物が無いコマの、差のある画素数（雑音の目安）: 最大", max(nz) if nz else 0)
    bad = [r["f"] for r in rows if "box" in r and (
        r["box"][0] < x0 - a.tol or r["box"][1] > x1 + a.tol or r["box"][2] < y0 - a.tol or r["box"][3] > y1 + a.tol)]
    print(f"安全域（x {x0}〜{x1}・y {y0}〜{y1}）の外に出たコマ: {len(bad)}")
    for s, e in runs(bad):
        bx = np.array([rows[i]["box"] for i in range(s, e + 1)])
        print(f"  コマ {s}–{e}（{s / fps:.2f}–{e / fps:.2f} 秒） x {bx[:, 0].min()}–{bx[:, 1].max()} y {bx[:, 2].min()}–{bx[:, 3].max()}")
    for key, name in (("face_overlap", "顔"), ("hand_overlap", "手")):
        if key == "hand_overlap" and hand is None:
            print("手との重なり: 測っていません（手のモデルが使えなかった）")
            continue
        found = sum(1 for r in rows if (r["faces"] if key == "face_overlap" else r.get("hands", 0)))
        hit = [r["f"] for r in rows if r[key] > 30]
        print(f"{name}と重なったコマ: {len(hit)}（{name}が見つかったコマ: {found}。見つからないコマは測れていない）")
        for s, e in runs(hit):
            print(f"  コマ {s}–{e}（{s / fps:.2f}–{e / fps:.2f} 秒） 重なりの最大 {max(rows[i][key] for i in range(s, e + 1))} 画素")
    if a.scenes:
        starts = [int(v) for v in a.scenes.split(",")]
        for n, s in enumerate(starts):
            e = (starts[n + 1] if n + 1 < len(starts) else len(rows)) - 1
            bx = np.array([rows[i]["box"] for i in range(s, e + 1) if "box" in rows[i]])
            if len(bx):
                print(f"場面 {n + 1}（コマ {s}–{e}）: x {bx[:, 0].min()}–{bx[:, 1].max()} y {bx[:, 2].min()}–{bx[:, 3].max()}")


if __name__ == "__main__":
    main()
