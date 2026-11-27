"""決まった読み（ひらがな）を、音声に時刻合わせする。認識の結果は使わず、読みを当てるだけ。

使う所: 話す映像の口の動きに、別に作った声を合わせる時。声と、映像の素材の音の両方で、かなごとの時刻を出して
対応点にし、映像の側（使うコマ）を選ぶ（声は伸び縮みさせない）。唇を閉じる音（ま・ば・ぱ行）の時刻で、
合っているかを確かめるのにも使う。

使い方（tools/ で）:
  uv run python kana_align.py <音声> <読みのファイル> --out <JSON>
読みのファイル: 1 行に 1 句（ひらがな）。出力: [{"phrase","start","end","kana":[[字,開始,終了],...]}, ...]

初めて使う時に、時刻合わせのモデル（約 1.2GB）を Hugging Face から取る。
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
_cache = {}


def load_audio(path, sr=16000):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def model():
    if "m" not in _cache:
        _cache["p"] = Wav2Vec2Processor.from_pretrained(MODEL)
        _cache["m"] = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
    return _cache["p"], _cache["m"]


def align(path, phrases):
    """phrases: ひらがなの句のリスト。返り値: 句ごとの時刻と、字ごとの時刻。"""
    proc, net = model()
    vocab = proc.tokenizer.get_vocab()
    blank = proc.tokenizer.pad_token_id
    x = load_audio(path)
    with torch.no_grad():
        inp = proc(x, sampling_rate=16000, return_tensors="pt")
        logits = net(inp.input_values).logits
    logp = torch.log_softmax(logits, dim=-1)
    sec_per_frame = len(x) / 16000 / logp.shape[1]
    toks, owner, dropped = [], [], []
    for pi, ph in enumerate(phrases):
        for ch in ph:
            if ch in vocab:
                toks.append(vocab[ch]); owner.append((pi, ch))
            elif ch.strip():
                dropped.append(ch)
    targets = torch.tensor([toks], dtype=torch.int32)
    ali, _ = AF.forced_align(logp, targets, blank=blank)
    ali = ali[0].tolist()
    # 連続する同じ字をまとめ、字ごとの区間にする
    spans = []
    k = -1
    prev = blank
    for f, t in enumerate(ali):
        if t != blank and t != prev:
            k += 1
            spans.append([f, f + 1])
        elif t != blank and t == prev:
            spans[k][1] = f + 1
        prev = t
    assert len(spans) == len(toks), (len(spans), len(toks))
    out = [{"phrase": ph, "kana": []} for ph in phrases]
    for (pi, ch), (a, b) in zip(owner, spans):
        out[pi]["kana"].append([ch, round(a * sec_per_frame, 3), round(b * sec_per_frame, 3)])
    for o in out:
        if o["kana"]:
            o["start"], o["end"] = o["kana"][0][1], o["kana"][-1][2]
    return out, dropped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("phrases"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    phrases = [l.strip() for l in Path(a.phrases).read_text(encoding="utf-8").splitlines() if l.strip()]
    res, dropped = align(a.audio, phrases)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    for r in res:
        print(f'{r.get("start", -1):6.2f}-{r.get("end", -1):6.2f} {r["phrase"]}')
    if dropped:
        print("語彙に無く、飛ばした字:", "".join(sorted(set(dropped))))
