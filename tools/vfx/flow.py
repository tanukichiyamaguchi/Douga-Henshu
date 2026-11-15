"""動きの解析（オプティカルフロー系）。

担当するテクニック:
    1 オプティカルフロー（optical-flow）
   13 動き補償（motion-compensate）
   16 フレーム間差分（frame-diff）
   23 シーンフロー（scene-flow）
   24 ブロックマッチング（block-match）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。

共通の約束:
- 動画は「t のコマ → 次のコマ」の動きを t のコマに描く（色や矢印が、そのコマの中身の位置にそろう）。
  最後のコマは直前の結果をそのまま使う。フレーム間差分と動き補償の diff は「前のコマ → 今のコマ」の差を今のコマに描く。
- 画像は 2 枚（SRC と --next）の間を調べる。
- 数値（JSON）の px は元の映像の px（--max-side で縮めて解析しても、元の大きさに直す）。.npz は解析した大きさの px。
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import ai, core, models
from .registry import JSON, OUT, RANGE, SIZE, SRC, arg, command

# 確かめた結果（番号 → status / how / measured / limits / tested）
# 試験の台本: <スクラッチパッド>/vfx-tests/flow/（make_flow_tests.py で正解つきの素材を作り、test_*.py で照らす。
# run_all.sh でまとめて実行）
NOTES: dict[int, dict[str, str]] = {
    1: {
        "status": "◎",
        "how": "各コマから次のコマへの、画素ごとの動き（横・縦の px）を測る。既定は OpenCV の DIS（ai.optical_flow、"
        "プリセット MEDIUM）。--method で Farneback、torchvision 0.21 の RAFT small・RAFT large も選べる。RAFT の重みは"
        "どちらも FlyingChairs と FlyingThings3D だけで学習した C_T_V2（small は ai.RaftFlow の既定。large は torchvision の"
        "既定 C_T_SKHT_V2 が KITTI（CC BY-NC-SA）でも学習しているので、C_T_V2 を指定）。torchvision は BSD-3-Clause"
        "（torchvision の説明では、学習済みの重みには学習データ由来の条件が付くことがあるとされる。既定の dis はモデルを"
        "使わない）。出力: MP4（overlay＝元の映像に色を重ねて白い矢印／color＝色だけ（色相 = 向き、明るさ = 速さ）／"
        "arrows＝向きの色の矢印。右下に色の見本の輪）、コマごとの動きの量（平均・中央値・95% 点・動いている所の割合・"
        "画面全体の中央値の動き ≒ カメラの動き。元の px）の JSON とグラフ、--save-flow で生のフロー（float16 の .npz）。"
        "色の明るさの上限は、前もって 16 組を DIS で測った 99% 点に固定（コマごとに変えるとちらつくため）。"
        "画像は --next の 2 枚の間",
        "measured": "正解つきの合成の組（graf 640×480 と AI 生成の顔 540×960 × 平行移動 (7,−4)・小数の移動 (3.4,−2.7)・"
        "回転 2°＋拡大 3%・大きな移動 (−24,13)・2 つの動き）の EPE（終点の誤差の平均）: dis 0.056〜0.203px（2 つの動きの"
        "物の縁では 3.3〜3.4px）、raft 0.078〜0.183px（縁 2.7px）、raft-large 0.037〜0.081px（縁 0.59〜0.62px）、"
        "farneback 0.25〜17.8px（顔の肌と 24px の動きで崩れる）。vtest に既知のパン（最大 8.3px/コマ）を足した 72 コマ: "
        "画面全体の中央値の動きと正解の差 平均 0.034・最大 0.067px（dis）、0.043・0.090px（farneback）、0.068・0.155px"
        "（raft、23 組）。固定カメラの vtest: 中央値の動き 0.004px 以下。速さ（540×960 の 1 組。CPU が空いている時）: "
        "dis 0.02 秒、farneback 0.09 秒、raft 1.1 秒、raft-large 2.6 秒（他の作業と CPU を分け合うと raft は 3〜5 倍"
        "遅くなった）。vtest 4 秒（768×576・40 コマ）を確認用 MP4 とグラフ込みで 3.5 秒",
        "limits": "dis は物の縁（動きの境目）で 3px 前後ずれる（縁が大事な時は raft-large。ただし dis の約 100 倍遅い）。"
        "farneback は模様の少ない所（肌）と大きな動きに弱い。模様の無い壁などでは dis が数 px の見かけの動きを出すことが"
        "ある（basketball の壁。色に出る）。カットをまたぐコマの動きには意味が無い。矢印は 4px 未満を描かない（小さな"
        "動きは色で見る）。--save-flow は 540×960 で 1 コマ約 2MB をメモリに貯めてから書く（1.5GB を超えそうな時は止まる。"
        "--max-side・--duration で小さく）",
        "tested": "make_flow_tests.py の正解つきの画像の組 10 組 × 4 方法と、vtest に既知のパンを足した pan-vtest.mkv"
        "（test_optical_flow.py）。basketball1/2・vtest・story は目で確認",
    },
    13: {
        "status": "◎",
        "how": "画面全体の動き（--model: homography（既定）・affine・similarity・translation）を、前のコマの特徴点"
        "（goodFeaturesToTrack）を LK で追い（往復で確かめる。ai.track_points）、RANSAC で当てはめて求める。"
        "align: 変換をつないで最初のコマ（--start の位置）に写し、さらに最初のコマの特徴点と直接合わせ直す（つなぎの誤差が"
        "たまらない）。average: 前後 N コマ（既定 5）を今のコマに写し、はみ出した所を除いて平均（--stat median で中央値）。"
        "diff: 前のコマを今のコマに写してから差を取る（しきい値 25。はみ出した所は除く）。画像 2 枚（--next）: SIFT の"
        "対応＋RANSAC で 2 枚目を 1 枚目に写す。JSON にコマごとの変換（元の px の 3×3）・合った点の数・ずれ・カメラの"
        "動きの目安、diff では補償なしの変化の割合も",
        "measured": "vtest（固定カメラ）から既知の軌跡で切り出した 72 コマ（640×480。±50・±30px のパン、さらに ±2.5° の"
        "回転と ±5% の拡大）: align の 4 隅のずれ（正解の変換との差）は平均 0.03〜0.13・最大 0.39px（カメラの動きは 4 隅で"
        "最大 77px。translation 0.032、similarity 0.044〜0.045、affine 0.051〜0.052、homography 0.115〜0.126）。diff: "
        "動く人の正解（元の固定カメラの差）に対し 精度 100%・再現率 98.9%（回転・拡大つきも同じ）。補償なしの差（16）は"
        "精度 19.2%（背景の輪郭も出る）で、変化の割合は 12.5% → 1.9% に減った。average（σ12 のノイズ・5 コマ）: PSNR "
        "26.6 → 34.2dB（mean）・34.1dB（median）。合わせずに 5 コマ平均すると 23.7dB（ぼける）。graf1 を既知の"
        "ホモグラフィーで写した画像: 4 隅のずれ 平均 0.32・最大 0.54px（SIFT の対応 1619 点）。固定カメラの vtest を "
        "align: 4 隅の動き 平均 0.07px。速さ: 640×480・72 コマの align が約 3.7 秒",
        "limits": "画面全体を 1 つの平面の動きとみなす。奥行きの差が大きい場面（手前の柱と奥の部屋。story 6〜7.5 秒）"
        "では合わせきれず、diff の変化が減らない（9.9% → 9.9%）・align で画がゆがむ。動く物が画面の大半を占めると背景を"
        "取り違える。align は最初のコマから離れるほど端が黒くなる（切り抜かない）。カットをまたぐと合わない（動きなし"
        "として扱い、JSON の failed に数える）。動きの種類が分かっている時は、少ない自由度（translation・similarity）の方が"
        "正確（上の数値）。ECC（画素の明るさの相関）で詰める方法も試したが、合成のパンで 4 隅のずれが平均 0.87〜1.13px と"
        "特徴点だけ（0.12px 前後）より悪かったので入れていない（歩く人に引っ張られる）",
        "tested": "test_motion_compensate.py: vtest に既知のパン・回転・拡大を足した映像（正解の変換・正解の動く人・"
        "ノイズなしの映像と比べる）、graf1 を既知のホモグラフィーで写した画像。graf1/graf3・vtest・story は目で確認",
    },
    16: {
        "status": "◎",
        "how": "今のコマと前のコマ（--gap で N コマ前）の差を、明るさ（既定）か色（--color。B・G・R の差の最大）で測る。"
        "比べる前に σ1px のぼかし（--blur）。しきい値（既定 25）以上を「変化した所」とし、MP4（heat＝暗くした映像に熱の"
        "色で重ねる／mask＝白黒の型／diff＝差そのもの）、コマごとの変化の割合・差の平均・99% 点の JSON とグラフ、"
        "画面の半分以上（--big）が変わったコマの一覧を書く。画像は --next の 2 枚の差（2 枚目に描く）",
        "measured": "一様な四角が動く合成映像（640×360・48 コマ。正解が画素単位で分かる）: --blur 0 で IoU 1.000"
        "（明るさ・色とも。見逃し・誤検出 0）。明るさの差が 19 の青い四角は明るさでは拾わず（しきい値 25 未満なので正しい）、"
        "--color で拾う。既定のぼかし（σ1）は縁を約 1px 広げ IoU 0.75（色 0.67）、縁から 2px より離れた誤検出は 0。"
        "σ4 のノイズを足すと、ぼかし 0 では誤検出 平均 61・最大 461px/コマ（差 19 の所がノイズで 25 を超える）、"
        "ぼかし 1 で 0。v4（20 秒の広告）: 画面の半分以上が変わった 17 コマのうち 9 コマが 76 のカット（カット 11 個中"
        " 9 個）、残り 8 コマは白いフラッシュが引く間のコマ。拾わなかった 2 つのカットは似た画どうし（変化 37%・47%）。"
        "vtest（固定カメラ）: 変化 平均 1.2%（歩く人の輪郭）。速さ: v4（540×960・600 コマ）で確認用 MP4 とグラフ込み 15 秒",
        "limits": "カメラが動くと背景の輪郭も変化として出る（その時は 13 の --mode diff）。「画面の半分以上の変化」は"
        "カットとは限らない（白いフラッシュが引く間も入り、似た画どうしのカットは入らない。カットは 76）。ぼかしは"
        "変化した所の縁を約 1px 広げる（くっきりした CG は --blur 0）。明るさが同じで色だけ違う変化は --color で拾う。"
        "ゆっくりした動きは --gap を大きくする",
        "tested": "test_frame_diff.py: 一様な四角が動く合成映像（正解の型と画素単位で比べる。ノイズあり・なし）、"
        "v4（76 scenes のカットと比べる）。vtest・story は目で確認",
    },
    23: {
        "status": "○",
        "how": "2D のフロー（DIS。--flow で farneback・raft）から、物が 1 コマで何倍に大きく見えたか（局所の拡大率 s）を"
        "求め、奥行き方向の動き（s > 1 は近づく。距離は約 1/s 倍になる）とする（optical expansion の考え方）。細かい"
        "尺度（σ2px）でフローを微分して画素ごとに s = √det(I + J) を出し、行き帰りのフローが合わない所（隠れる所）と動きの"
        "境目（微分が大きい所）を除いてから σ（短い辺の 1/30）でならす（境目をまたいでならすと、近づく物のまわりに"
        "「遠ざかる」輪ができた）。--fov の画角で 3D の動き（距離に対する割合 ΔX/Z・ΔY/Z・ΔZ/Z）も .npz に出す。"
        "--method depth は Depth Anything V2 Small（Apache-2.0。ai.DepthEstimator のモデルの生の出力）の逆深度をフローに"
        "沿って前後のコマで比べる方法（比べる用）。出力: 近づく所を赤・遠ざかる所を青で塗り、白い矢印で横の動きを描いた"
        " MP4、コマごとの近づく・遠ざかる所の割合・画面全体の拡大（カメラが寄る・引く）・ぶつかるまでの秒数の目安の"
        " JSON とグラフ。画像は --next の 2 枚の間",
        "measured": "画面全体が 1 コマ 1.5% 大きく／小さくなる合成映像（長辺 480 で解析）: 画面全体 +1.450%／−1.408%"
        "（正解 +1.5／−1.48%。長辺 960 では +1.492%）、画素ごとの誤差の中央値 0.12〜0.17%、近づく・遠ざかるの見分け"
        " 99.6%／99.9%。止まった背景の上で板が 1 コマ 2% 大きく・小さくなる合成映像（640×480）: 板の値の誤差 平均 "
        "0.09〜0.12・最大 0.4%、見分けの正しさ 左 100%・右 100%・背景 99.1%。2 枚の画像（回転 2°＋拡大 3%）: +2.96%。"
        "depth 方式は画面全体の拡大を 0 とみなし（+0.007%）、板の見分けも 55〜59%（背景 18%）と不正確で、4〜36 倍遅い。"
        "実写 story（人がカメラへ歩いてくる 2.9 秒。AI 生成）: 顔の枠（YuNet）の大きさは ×1.32（+0.44%/コマ）、顔の枠の"
        "中の値を積み上げると ×1.42〜1.43（+0.55%/コマ）。近づく向きは合い、速さは約 1.25 倍に出た（コマごとの値は顔の"
        "枠の揺れが大きく比べられない）。固定カメラの vtest の止まった背景で近づく・遠ざかるとした画素 0.32%。"
        "速さ: 640×480・36 コマで約 7 秒（DIS を行き帰り 2 回）",
        "limits": "単眼の近似。「大きく見える割合」から求めるので、物の向きが変わる・形が変わる動き（腕・脚・口・髪・服）"
        "にも色が付き、実写では斑になる。遠くの小さな歩く人（vtest）は 1 コマ 0.5% 前後しか大きさが変わらず手足の動きに"
        "埋もれる（人ごとの向きの一致 46〜49% ＝ 当て推量と同じ）。3D の動きは距離に対する割合で、実際の速さ（m/秒）は"
        "分からない（--fov は仮の画角）。動きの境目・隠れる所には色を塗らない。depth 方式は Depth Anything の出力が"
        "コマごとに倍率が決まらないので画面全体の中央値でそろえる（そのため画面全体の近づきは測れない）",
        "tested": "test_scene_flow.py（zoom-in・zoom-out・loom の正解と比べる、vtest、story と顔の枠）、"
        "test_scene_flow_real.py（story の積み上げ、vtest の人ごと）、graf の回転＋拡大の 2 枚",
    },
    24: {
        "status": "◎",
        "how": "16×16（--block）の区画ごとに、前のコマの ±16px（--range）をすべて探し、差の絶対値の和（SAD）が最小の"
        "所を選ぶ（自前の全探索。ずれごとに画面全体の差を取り、区画ごとに平均。差が同じなら小さいずれ）。隣のずれの SAD "
        "から「等角直線」の当てはめで 1px より細かく求める（--no-subpixel で 1px 単位）。--search で ffmpeg の mestimate"
        "（PyAV のフィルター経由。esa＝全探索・tss・tdls・ntss・fss・ds・hexbs・epzs・umh。1px 単位）も選べる。模様の"
        "少ない区画（明るさの標準偏差 3 未満）は除く。出力: 区画ごとの矢印（--view color で区画を動きの色で塗る）の MP4、"
        "コマごとの要約（最も多い動き・平均・動いている区画の割合・一致の悪さ）の JSON、--save-vectors で区画ごとの"
        "動きの .npz。画像は --next の 2 枚で、JSON に全区画の動き",
        "measured": "正解つきの画像の組（graf 640×480・AI 生成の顔 540×960）: 平行移動 (7,−4) と 2 つの動き（板 (−9,5)・"
        "背景 (3,1)）は full・esa・umh で 100%、epzs で 99.9〜100% の区画が正しい（誤差 0.5px 未満）。tss 52〜64%・"
        "ds 61〜93%・hexbs 45〜62%（局所の谷に落ちる）。(−24,13) は ±16 では 0%、--range 32 で 100%。小数 (3.4,−2.7): "
        "誤差 0.5px 未満が 72.6〜80.6%（1px 単位だけだと 35〜55%）、成分ごとの誤差の中央値 0.11〜0.17px、1.5px を超える"
        "外れ 2.5〜4.8%。回転 2°＋拡大 3%（探す範囲に入る区画）: 83〜86%・平均誤差 0.34〜0.37px（1px 単位では 64%）、"
        "--range 32 で全区画 83〜87%。実写 vtest の 2 コマ: 自前の全探索（1px 単位）と ffmpeg esa の答えが模様のある"
        " 1591 区画で 100% 一致。速さ（540×960・16×16・±16）: full 0.37〜0.41 秒/コマ、esa 0.27〜0.28、epzs 0.007、"
        "ds 0.014〜0.015、hexbs 0.009〜0.010。story で測った平均の速さは full 9.3・epzs 8.7・ds 7.4・hexbs 5.8px/コマ"
        "（速い探し方は大きな動きを小さく見積もる）",
        "limits": "区画の中は 1 つの平行移動とみなすので、回転・拡大・物の縁をまたぐ区画は不正確。探す範囲より速い"
        "動きは測れない（--range を上げると時間は (2×範囲+1)² に比例）。小数のずれは、まっすぐな縁や模様の少ない区画で"
        " 1px 以上外れることがある（開口問題）。速い探し方（tss・ds・hexbs）は大きな動きで局所の谷に落ちる。細かい動きの"
        "地図が欲しい時は 01 optical-flow",
        "tested": "test_block_match.py: 正解つきの画像の組 10 組を 8 通りの探し方で、--range 32、vtest の実写で ffmpeg "
        "esa と答え合わせ、story で速さ。vtest・story は目で確認",
    },
}

NEXT = arg(
    "--next",
    help="2 枚目の画像（SRC が画像の時。SRC → NEXT の動きを調べる）",
)
NO_LEGEND = arg(
    "--no-legend",
    action="store_true",
    help="色の見本と文字を描かない（映像を合成に使う時）",
)

# ── 入力 ──────────────────────────────────────────────────────────────────


@dataclass
class _Input:
    """調べる物: 動画（範囲つき）か、画像 2 枚。コマは解析する大きさで返す。"""

    src: str
    info: core.MediaInfo
    size: tuple[int, int]  # 解析する大きさ（幅, 高さ）
    to_src: float  # 解析の px × to_src = 元の px
    rng: dict[str, Any]
    pair: list[np.ndarray] | None = None
    next_path: str | None = None

    @property
    def fps(self) -> float:
        return self.info.fps

    @property
    def start(self) -> float:
        return float(self.rng["start"])

    @property
    def is_pair(self) -> bool:
        return self.pair is not None

    def frames(self) -> Iterator[np.ndarray]:
        if self.pair is not None:
            yield from self.pair
            return
        yield from core.read_frames(self.src, size=self.size, **self.rng)

    def t(self, i: int) -> float:
        return core.t_of(i, self.fps, self.start)

    def frame_no(self, i: int) -> int:
        """調べた範囲の中のコマ番号 → 元の動画の中のコマ番号。"""
        return round(self.start * self.fps) + int(i)

    def count(self) -> int:
        """処理するコマ数の見込み（進み具合の表示と、前もって測る時の間引きに使う）。"""
        if self.pair is not None:
            return 2
        dur = self.rng["duration"] or max(self.info.duration - self.start, 0.0)
        n = max(1, round(dur * self.fps))
        if self.rng["max_frames"]:
            n = min(n, int(self.rng["max_frames"]))
        return n

    def writer(self, out: Path) -> core.VideoWriter:
        """元の音声を同じ範囲で残す書き出し。"""
        return core.VideoWriter(
            out,
            self.fps,
            audio_from=self.src,
            audio_start=self.start,
            audio_duration=self.rng["duration"],
        )

    def head(self) -> dict[str, Any]:
        """JSON の先頭に入れる共通の情報。"""
        d: dict[str, Any] = {"source": str(self.src)}
        if self.next_path:
            d["next"] = str(self.next_path)
        else:
            d.update(fps=self.fps, start=self.start)
        d.update(
            analysis_size=list(self.size),
            source_size=[self.info.width, self.info.height],
            to_source=round(self.to_src, 6),
        )
        return d


def _open(
    a: argparse.Namespace,
    *,
    default_max_side: int | None = None,
    same_size: bool = True,
) -> _Input:
    info = core.probe(a.src)
    nxt = getattr(a, "next", None)
    max_side = a.max_side or default_max_side
    if default_max_side and not a.max_side:
        core.log(
            f"解析は長辺 {default_max_side}px で行います（--max-side で変えられます）"
        )
    rng = {
        "start": a.start or 0.0,
        "duration": a.duration,
        "max_frames": a.max_frames,
    }
    if info.is_image:
        core.require(
            bool(nxt),
            "画像の時は --next で 2 枚目の画像を指定してください（2 枚の間の動きを調べます）",
        )
        first = core.load_image(a.src)
        second = core.load_image(nxt)
        h, w = first.shape[:2]
        if same_size and second.shape[:2] != (h, w):
            core.log(
                f"2 枚目の大きさ（{second.shape[1]}×{second.shape[0]}）を 1 枚目（{w}×{h}）に合わせます"
            )
            second = cv2.resize(second, (w, h), interpolation=cv2.INTER_AREA)
        k = 1.0
        if max_side and max(w, h) > max_side:
            k = max_side / max(w, h)
        imgs = []
        for im in (first, second):
            if k != 1.0:
                ih, iw = im.shape[:2]
                im = cv2.resize(
                    im,
                    (max(1, round(iw * k)), max(1, round(ih * k))),
                    interpolation=cv2.INTER_AREA,
                )
            imgs.append(im)
        size = (imgs[0].shape[1], imgs[0].shape[0])
        return _Input(a.src, info, size, w / size[0], rng, imgs, str(nxt))
    core.require(not nxt, "--next は SRC が画像の時だけ使えます")
    size = core.scaled_size(info.width, info.height, max_side=max_side)
    return _Input(a.src, info, size, info.width / size[0], rng)


def _out_paths(
    a: argparse.Namespace, inp: _Input, tag: str, ext: str = ".mp4"
) -> tuple[Path, Path]:
    out = (
        Path(a.out)
        if a.out
        else core.default_out(a.src, tag, ".png" if inp.is_pair else ext)
    )
    json_path = Path(a.json) if a.json else out.with_suffix(".json")
    return out, json_path


def _progress(i: int, n: int) -> None:
    if n >= 40 and (i + 1) % max(1, n // 10) == 0:
        core.log(f"  {i + 1}/{n} コマ")


# ── 動きの測り方 ───────────────────────────────────────────────────────────

_FLOWS = ["dis", "farneback", "raft", "raft-large"]
_FLOW_HELP = (
    "動きの測り方。dis＝OpenCV の DIS（速い。既定。540×960 で約 0.02 秒/コマ）、farneback＝OpenCV の Farneback"
    "（模様の少ない所と大きな動きに弱い）、raft＝RAFT small（約 1.1 秒/コマ。なめらかな動きの正確さは dis と同じくらい）、"
    "raft-large＝RAFT large（約 2.6 秒/コマ。最も正確で、物の縁の誤差は dis の約 1/5）。raft は長辺 960px で解析する"
)


class _RaftLargeCT(ai.RaftFlow):
    """RAFT large のうち、FlyingChairs と FlyingThings3D だけで学習した重み（C_T_V2）。

    torchvision の既定（C_T_SKHT_V2）は KITTI（CC BY-NC-SA）でも学習しているので使わない。
    """

    def __init__(self) -> None:
        self.torch = models.torch_setup()
        from torchvision.models import optical_flow as of

        self.weights = of.Raft_Large_Weights.C_T_V2
        self.net = of.raft_large(weights=self.weights).eval()


def _flow_fn(method: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """(前のコマ, 次のコマ) → 前のコマの各画素の動き（H×W×2、px）を返す関数。"""
    if method in ("dis", "farneback"):

        def fn(p: np.ndarray, c: np.ndarray) -> np.ndarray:
            return ai.optical_flow(core.to_gray(p), core.to_gray(c), method)

        return fn
    if method == "raft":
        return ai.RaftFlow()
    if method == "raft-large":
        return _RaftLargeCT()
    raise core.VfxError(f"知らない方法です: {method}（{' / '.join(_FLOWS)}）")


def _auto_max_mag(inp: _Input, pairs: int = 16) -> float:
    """色の明るさの上限（解析の px/コマ）。いくつかのコマの組で DIS の動きを測り、その 99% 点。"""
    n = inp.count()
    if n < 2:
        return 1.0
    want = {
        int(v) for v in np.linspace(1, n - 1, min(pairs, n - 1)).round().astype(int)
    }
    mags: list[np.ndarray] = []
    prev = None
    for i, f in enumerate(inp.frames()):
        g = core.to_gray(f)
        if i in want and prev is not None:
            fl = ai.optical_flow(prev, g, "dis")
            mags.append(np.hypot(fl[..., 0], fl[..., 1])[::4, ::4].ravel())
        prev = g
        if i >= max(want):
            break
    if not mags:
        return 1.0
    return max(1.0, float(np.percentile(np.concatenate(mags), 99)))


def _flow_stats(flow: np.ndarray, to_src: float, min_mag: float) -> dict[str, float]:
    """フローの要約（元の px/コマ）。dx・dy は中央値（画面の大部分の動き ≒ カメラの動き）。"""
    fx, fy = flow[..., 0] * to_src, flow[..., 1] * to_src
    mag = np.hypot(fx, fy)
    return {
        "mean": round(float(mag.mean()), 4),
        "median": round(float(np.median(mag)), 4),
        "p95": round(float(np.percentile(mag, 95)), 4),
        "p999": round(float(np.percentile(mag, 99.9)), 4),
        "dx": round(float(np.median(fx)), 4),
        "dy": round(float(np.median(fy)), 4),
        "moving": round(float((mag >= min_mag).mean()), 5),
    }


# ── 描く部品 ──────────────────────────────────────────────────────────────

# 向き → 色（ai.flow_to_color と同じ: 色相 = 向き。右が赤、下が黄緑、左が水色、上が紫）
_HUES = cv2.cvtColor(
    np.stack(
        [
            np.arange(180, dtype=np.uint8),
            np.full(180, 255, np.uint8),
            np.full(180, 255, np.uint8),
        ],
        -1,
    )[None],
    cv2.COLOR_HSV2BGR,
)[0]


def _dir_color(u: float, v: float) -> tuple[int, int, int]:
    ang = math.atan2(v, u) % (2 * math.pi)
    b, g, r = _HUES[int(ang * 180 / math.pi / 2) % 180]
    return int(b), int(g), int(r)


def _arrow(
    img: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: tuple[int, int, int],
    thick: int = 1,
) -> None:
    """縁取りつきの矢印（暗い所でも明るい所でも見える）。4px より短い物は、矢じりがつぶれて点の模様に
    見えるので描かない（小さな動きは色で見る）。"""
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if length < 4.0:
        return
    a = (round(p0[0]), round(p0[1]))
    b = (round(p1[0]), round(p1[1]))
    tip = min(0.45, max(0.18, 5.0 / length))
    cv2.arrowedLine(img, a, b, (0, 0, 0), thick + 2, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(img, a, b, color, thick, cv2.LINE_AA, tipLength=tip)


def _flow_arrows(
    img: np.ndarray,
    flow: np.ndarray,
    step: int,
    scale: float,
    min_mag: float,
    color: tuple[int, int, int] | None = None,
) -> None:
    """格子の点ごとに、周り（step 四方）の平均の動きを矢印で描く。color が None なら向きの色。"""
    h, w = flow.shape[:2]
    sm = cv2.blur(flow, (step, step))
    thick = 2 if max(h, w) >= 900 else 1
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            u, v = float(sm[y, x, 0]), float(sm[y, x, 1])
            if math.hypot(u, v) < min_mag:
                continue
            _arrow(
                img,
                (x, y),
                (x + u * scale, y + v * scale),
                color or _dir_color(u, v),
                thick,
            )


def _font_size(img: np.ndarray) -> int:
    return max(13, img.shape[1] // 34)


def _tag(img: np.ndarray, text: str) -> None:
    """左上の見出し。細い映像では、はみ出さないよう文字を小さくする（下限 10px）。"""
    # 幅の目安: 全角の字は 1 字、半角は 0.6 字
    units = sum(1.0 if ord(ch) > 0x2E80 else 0.6 for ch in text)
    size = _font_size(img)
    size = max(10, min(size, int((img.shape[1] - 20) / max(units, 1.0))))
    core.draw_text(img, text, (8, 8), size=size, bg=(0, 0, 0), pad=4)


def _wheel_legend(img: np.ndarray, text: str) -> None:
    """色の見本（色相 = 向き、明るさ = 速さ）を右下に描く。"""
    h, w = img.shape[:2]
    r = max(18, round(min(w, h) * 0.06))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    mag = np.hypot(xx, yy) / r
    ang = np.arctan2(yy, xx) % (2 * np.pi)
    hsv = np.zeros((2 * r + 1, 2 * r + 1, 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(mag, 0, 1) * 255).astype(np.uint8)
    disk = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    inside = mag <= 1.0
    size = _font_size(img)
    m = 8
    x0, y0 = w - 2 * r - 1 - m, h - 2 * r - 1 - m - int(size * 1.6)
    if x0 < 0 or y0 < 0:
        return
    box = img[y0 - 4 : h - m + 4, x0 - 4 : w - m + 4]
    box[:] = (box.astype(np.float32) * 0.35).astype(np.uint8)
    roi = img[y0 : y0 + 2 * r + 1, x0 : x0 + 2 * r + 1]
    roi[inside] = disk[inside]
    core.draw_text(
        img,
        text,
        (w - m, h - m),
        size=size,
        color=(255, 255, 255),
        bg=None,
        anchor="rb",
    )


def _bar_legend(
    img: np.ndarray,
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
    texts: tuple[str, str, str],
) -> None:
    """右下に、左の色 → 右の色の帯を描く。texts は (左端, 帯の上の中央, 右端)。

    細い映像でも重ならないよう、中央の説明は帯の上、両端の説明は帯の下に置く。
    """
    h, w = img.shape[:2]
    size = _font_size(img)
    line = int(size * 1.45)
    bw, bh, m = min(int(w * 0.55), 320), max(8, size // 2), 8
    x0 = w - bw - m
    y_bar = h - m - line - bh
    y_top = y_bar - line
    if x0 < 6 or y_top < 6:
        return
    box = img[y_top - 4 : h - m + 4, x0 - 6 : w - m + 4]
    box[:] = (box.astype(np.float32) * 0.35).astype(np.uint8)
    ramp = np.linspace(-1, 1, bw, dtype=np.float32)
    left, right = np.float32(colors[0]), np.float32(colors[1])
    a = np.abs(ramp)[:, None]
    row = np.where(ramp[:, None] < 0, left * a, right * a)
    img[y_bar : y_bar + bh, x0 : x0 + bw] = row[None].astype(np.uint8)
    white = (255, 255, 255)
    core.draw_text(img, texts[1], (x0 + bw // 2, y_bar - 3), size=size, color=white, bg=None, anchor="mb")  # fmt: skip
    core.draw_text(img, texts[0], (x0, h - m), size=size, color=white, bg=None, anchor="lb")  # fmt: skip
    core.draw_text(img, texts[2], (x0 + bw, h - m), size=size, color=white, bg=None, anchor="rb")  # fmt: skip


def _heat(
    frame: np.ndarray, d: np.ndarray, thr: float, valid: np.ndarray | None = None
) -> np.ndarray:
    """差の大きさを熱の色で重ねる（元の映像は暗くする）。しきい値の半分から浮かび上がり、しきい値で不透明になる。"""
    base = frame[:, :, :3].astype(np.float32) * 0.45
    norm = np.clip(d / (3 * thr), 0, 1)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = np.clip((d - 0.5 * thr) / (0.5 * thr), 0, 1)
    if valid is not None:
        alpha = alpha * valid
    alpha = alpha[..., None]
    out = base * (1 - alpha) + col.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _plt() -> Any:
    """日本語の書体を設定した matplotlib（画面なしで描く）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for f in core.JP_FONTS:
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=f
            ).get_name()
            break
    return plt


def _chart(
    path: Path,
    t: list[float],
    series: list[tuple[list[float], str, str]],
    *,
    title: str,
    ylabel: str,
    hlines: list[tuple[float, str, str]] | None = None,
    marks: list[float] | None = None,
) -> Path:
    """時間の流れのグラフ（横軸は秒）を JPG に書く。series は (値, 名前, 色)。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=100)
    for y, label, color in series:
        ax.plot(t[: len(y)], y, label=label, color=color, lw=1.4)
    for y, label, color in hlines or []:
        ax.axhline(y, ls="--", color=color, lw=1, label=label)
    for x in marks or []:
        ax.axvline(x, color="#d33", lw=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("秒")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if len(series) + len(hlines or []) > 1:
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=100)
    plt.close(fig)
    return path


def _save_npz(path: Path, **arrays: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def _check_npz_size(inp: _Input, per_frame: int, what: str) -> None:
    """保存する数値が大きすぎないか（メモリに貯めてから書くため）。per_frame はコマあたりのバイト数。"""
    total = inp.count() * per_frame
    core.require(
        total < 1.5e9,
        f"{what}が大きすぎます（約 {total / 1e9:.1f}GB）。--max-side・--duration・--max-frames で小さくしてください",
    )


# ── 01 オプティカルフロー ──────────────────────────────────────────────────


def _render_flow(
    frame: np.ndarray,
    flow: np.ndarray,
    *,
    view: str,
    max_mag: float,
    step: int,
    scale: float,
    min_mag: float,
    legend: bool,
    label: str,
    unit: float,
) -> np.ndarray:
    if view == "color":
        img = ai.flow_to_color(flow, max_mag)
    elif view == "arrows":
        img = (frame[:, :, :3].astype(np.float32) * 0.75).astype(np.uint8)
        _flow_arrows(img, flow, step, scale, min_mag)
    else:
        col = ai.flow_to_color(flow, max_mag)
        img = cv2.addWeighted(frame[:, :, :3], 0.45, col, 0.55, 0)
        _flow_arrows(img, flow, step, scale, min_mag, color=(255, 255, 255))
    if legend:
        _tag(img, label)
        if view == "arrows":
            core.draw_text(
                img,
                f"矢印の長さ ×{scale:.2g}",
                (img.shape[1] - 8, img.shape[0] - 8),
                size=_font_size(img),
                bg=(0, 0, 0),
                anchor="rb",
            )
        else:
            _wheel_legend(img, f"最大 {max_mag * unit:.3g} px/コマ")
    return img


@command(
    1,
    "optical-flow",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    NEXT,
    arg("--method", choices=_FLOWS, default="dis", help=_FLOW_HELP),
    arg(
        "--view",
        choices=["overlay", "color", "arrows"],
        default="overlay",
        help="描き方。overlay＝元の映像に色を半分重ね、白い矢印（既定）、color＝色だけ（色相 = 向き、明るさ = 速さ）、"
        "arrows＝元の映像に向きの色の矢印",
    ),
    arg(
        "--max-mag",
        type=float,
        help="色が最も明るくなる速さ（元の映像の px/コマ。省略時は自動: いくつかのコマで測った動きの 99%% 点）",
    ),
    arg(
        "--step",
        type=int,
        help="矢印の間隔（解析する大きさの px。省略時は長辺の 1/40）",
    ),
    arg(
        "--arrow-scale",
        type=float,
        help="矢印の長さの倍率（省略時は自動: 色が最も明るくなる速さで、矢印の間隔の 1.2 倍の長さ）",
    ),
    arg(
        "--min-mag",
        type=float,
        default=0.5,
        help="これより遅い動き（元の映像の px/コマ）は止まっているとみなし、矢印を描かない（既定 0.5）",
    ),
    arg(
        "--save-flow",
        action="store_true",
        help="生のフロー（コマごとの横・縦の動き。解析する大きさの px、float16）を .npz に保存する",
    ),
    NO_LEGEND,
)
def optical_flow(a: argparse.Namespace) -> None:
    """画素ごとの動きの向きと速さを測り、色と矢印の動画（MP4）、コマごとの動きの量（JSON）、生のフロー（.npz）を書く。"""
    inp = _open(a, default_max_side=960 if a.method.startswith("raft") else None)
    out, json_path = _out_paths(a, inp, "optical-flow")
    npz_path = out.with_suffix(".npz")
    fn = _flow_fn(a.method)
    k = inp.to_src
    step = max(4, a.step or round(max(inp.size) / 40))
    min_mag = a.min_mag / k
    w, h = inp.size
    if a.save_flow and not inp.is_pair:
        _check_npz_size(inp, w * h * 4, "保存するフロー")
    label_base = f"オプティカルフロー（{a.method}）"
    t0 = time.time()

    if inp.pair is not None:
        flow = fn(inp.pair[0], inp.pair[1])
        mag = np.hypot(flow[..., 0], flow[..., 1])
        max_mag = (
            a.max_mag / k if a.max_mag else max(1.0, float(np.percentile(mag, 99)))
        )
        scale = a.arrow_scale or 1.2 * step / max_mag
        stats = _flow_stats(flow, k, a.min_mag)
        img = _render_flow(
            inp.pair[0], flow, view=a.view, max_mag=max_mag, step=step, scale=scale,
            min_mag=min_mag, legend=not a.no_legend,
            label=f"{label_base} 平均 {stats['mean']:.2f}px", unit=k,
        )  # fmt: skip
        core.save_image(out, img)
        data = {
            **inp.head(),
            "method": a.method,
            "max_mag": round(max_mag * k, 4),
            "seconds": round(time.time() - t0, 3),
            "flow": stats,
        }
        if a.save_flow:
            _save_npz(
                npz_path,
                flow=flow.astype(np.float16)[None],
                to_source=k,
                method=a.method,
            )
            data["npz"] = str(npz_path)
            core.log(f"保存: {npz_path}")
        core.write_json(json_path, data)
        core.log(
            f"平均の速さ {stats['mean']:.2f} px、中央値の動き（横 {stats['dx']:+.2f}・縦 {stats['dy']:+.2f} px）、"
            f"動いている所 {stats['moving'] * 100:.1f}%"
        )
        core.log(f"保存: {out}")
        core.log(f"保存: {json_path}")
        return

    if a.max_mag:
        max_mag = a.max_mag / k
    else:
        max_mag = _auto_max_mag(inp)
        core.log(f"色が最も明るくなる速さ: {max_mag * k:.2f} px/コマ（自動）")
    scale = a.arrow_scale or 1.2 * step / max_mag
    n_est = inp.count()
    rows: list[dict[str, Any]] = []
    flows: list[np.ndarray] = []
    prev = None
    flow = None
    count = 0
    t_flow = 0.0
    with inp.writer(out) as writer:

        def emit(frame: np.ndarray, fl: np.ndarray, i: int, stats: dict | None) -> None:
            label = label_base
            if stats is not None:
                label += f" 平均 {stats['mean']:.2f}px/コマ"
            writer.write(
                _render_flow(
                    frame,
                    fl,
                    view=a.view,
                    max_mag=max_mag,
                    step=step,
                    scale=scale,
                    min_mag=min_mag,
                    legend=not a.no_legend,
                    label=label,
                    unit=k,
                )
            )

        for i, f in enumerate(inp.frames()):
            if prev is not None:
                t1 = time.time()
                flow = fn(prev, f)
                t_flow += time.time() - t1
                stats = _flow_stats(flow, k, a.min_mag)
                rows.append({"frame": inp.frame_no(i - 1), "t": inp.t(i - 1), **stats})
                if a.save_flow:
                    flows.append(flow.astype(np.float16))
                emit(prev, flow, i - 1, stats)
            prev = f
            count = i + 1
            _progress(i, n_est)
        core.require(count >= 2, "動画のコマが 2 つ以上必要です")
        assert prev is not None and flow is not None
        emit(prev, flow, count - 1, None)  # 最後のコマは直前の動きをそのまま使う
    secs = time.time() - t0
    mean = [r["mean"] for r in rows]
    data = {
        **inp.head(),
        "method": a.method,
        "frames": count,
        "max_mag": round(max_mag * k, 4),
        "min_mag": a.min_mag,
        "seconds": round(secs, 2),
        "seconds_per_frame_flow": round(t_flow / max(1, len(rows)), 4),
        "note": "各コマから次のコマへの動き（元の映像の px/コマ）。mean・median・p95・p999 は速さの平均・中央値・"
        "95% 点・99.9% 点、dx・dy は画面全体の中央値（≒カメラの動き）、moving は --min-mag 以上動いた画素の割合",
        "summary": {
            "mean": round(float(np.mean(mean)), 4),
            "max_mean": round(float(np.max(mean)), 4),
            "max_mean_t": rows[int(np.argmax(mean))]["t"],
        },
        "per_frame": rows,
    }
    if a.save_flow:
        _save_npz(
            npz_path,
            flow=np.stack(flows),
            t=np.array([r["t"] for r in rows]),
            frame=np.array([r["frame"] for r in rows]),
            to_source=k,
            method=a.method,
        )
        data["npz"] = str(npz_path)
        core.log(f"保存: {npz_path}")
    core.write_json(json_path, data)
    chart = _chart(
        out.with_suffix(".jpg"),
        [r["t"] for r in rows],
        [
            (mean, "平均の速さ", "#1f6fd1"),
            ([r["dx"] for r in rows], "横の動き（中央値）", "#e07b00"),
            ([r["dy"] for r in rows], "縦の動き（中央値）", "#2a9d3a"),
        ],
        title=f"オプティカルフロー（{a.method}）",
        ylabel="px/コマ",
    )
    core.log(
        f"{count} コマ: 平均の速さ {data['summary']['mean']:.2f} px/コマ、最も速い {data['summary']['max_mean']:.2f}"
        f"（{data['summary']['max_mean_t']:.2f} 秒）。動きの計算 {data['seconds_per_frame_flow']:.3f} 秒/コマ"
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {chart}")
    core.log(f"保存: {json_path}")


# ── 16 フレーム間差分 ──────────────────────────────────────────────────────


def _diff_prep(frame: np.ndarray, color: bool, blur: float) -> np.ndarray:
    x = frame[:, :, :3] if color else core.to_gray(frame)
    x = x.astype(np.float32)
    if blur > 0:
        x = cv2.GaussianBlur(x, (0, 0), blur)
    return x


def _diff(p: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = cv2.absdiff(p, c)
    return d.max(axis=2) if d.ndim == 3 else d


def _diff_view(
    frame: np.ndarray,
    d: np.ndarray | None,
    *,
    view: str,
    thr: float,
    gain: float,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    if d is None:
        if view == "heat":
            return (frame[:, :, :3].astype(np.float32) * 0.45).astype(np.uint8)
        return np.zeros_like(frame[:, :, :3])
    if view == "mask":
        m = d >= thr
        if valid is not None:
            m &= valid > 0
        return cv2.cvtColor(m.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    if view == "diff":
        g = np.clip(d * gain, 0, 255).astype(np.uint8)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return _heat(frame, d, thr, valid)


def _diff_stats(
    d: np.ndarray, thr: float, valid: np.ndarray | None = None
) -> dict[str, float]:
    v = d if valid is None else d[valid > 0]
    if v.size == 0:
        return {"changed": 0.0, "mean": 0.0, "p99": 0.0}
    return {
        "changed": round(float((v >= thr).mean()), 5),
        "mean": round(float(v.mean()), 3),
        "p99": round(float(np.percentile(v, 99)), 2),
    }


@command(
    16,
    "frame-diff",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    NEXT,
    arg(
        "--gap",
        type=int,
        default=1,
        help="何コマ前と比べるか（既定 1 = 隣のコマ。ゆっくりした動きは 2〜5 で見やすくなる）",
    ),
    arg(
        "--threshold",
        type=float,
        default=25,
        help="変化とみなす差（0〜255。既定 25）",
    ),
    arg(
        "--blur",
        type=float,
        default=1.0,
        help="比べる前のぼかし（ガウスの σ、px。圧縮やノイズのちらつきを抑える。0 で無し。既定 1）",
    ),
    arg(
        "--color",
        action="store_true",
        help="色の差で測る（B・G・R の差の最大。明るさが同じで色だけ違う物も拾う。既定は明るさの差）",
    ),
    arg(
        "--view",
        choices=["heat", "mask", "diff"],
        default="heat",
        help="描き方。heat＝暗くした元の映像に差を熱の色で重ねる（既定）、mask＝変化した所が白の白黒（合成の型に使える）、"
        "diff＝差の大きさそのもの（灰色。--gain 倍）",
    ),
    arg(
        "--gain",
        type=float,
        default=4.0,
        help="diff の描き方で、差を何倍に明るくするか（既定 4）",
    ),
    arg(
        "--big",
        type=float,
        default=0.5,
        help="画面のこの割合以上が変わったコマを「大きな変化」（カット・フラッシュの候補）として挙げる（0〜1。既定 0.5）",
    ),
    NO_LEGEND,
)
def frame_diff(a: argparse.Namespace) -> None:
    """前後のコマの差（変化した所）を測り、色で重ねた動画（MP4）、コマごとの変化の割合（JSON とグラフ）を書く。"""
    core.require(a.gap >= 1, "--gap は 1 以上にしてください")
    core.require(a.threshold > 0, "--threshold は 0 より大きくしてください")
    inp = _open(a)
    out, json_path = _out_paths(a, inp, "frame-diff")
    thr = a.threshold
    t0 = time.time()
    what = "色" if a.color else "明るさ"

    def label(st: dict[str, float] | None) -> str:
        if st is None:
            return f"フレーム間差分（{a.gap} コマ前と比べるコマがまだ無い）"
        return f"フレーム間差分（{a.gap} コマ前・{what}の差 {thr:g} 以上）変化 {st['changed'] * 100:.1f}%"

    if inp.pair is not None:
        p, c = (_diff_prep(f, a.color, a.blur) for f in inp.pair)
        d = _diff(p, c)
        st = _diff_stats(d, thr)
        img = _diff_view(inp.pair[1], d, view=a.view, thr=thr, gain=a.gain)
        if not a.no_legend:
            _tag(img, f"差分（{what}の差 {thr:g} 以上）変化 {st['changed'] * 100:.1f}%")
        core.save_image(out, img)
        core.write_json(
            json_path,
            {
                **inp.head(),
                "params": {"threshold": thr, "blur": a.blur, "color": a.color},
                "diff": st,
                "note": "2 枚目の画像に描いた。changed は差がしきい値以上の画素の割合",
            },
        )
        core.log(f"変化した所: {st['changed'] * 100:.2f}%（差の平均 {st['mean']:.2f}）")
        core.log(f"保存: {out}")
        core.log(f"保存: {json_path}")
        return

    rows: list[dict[str, Any]] = []
    buf: deque[np.ndarray] = deque(maxlen=a.gap)
    n_est = inp.count()
    count = 0
    with inp.writer(out) as writer:
        for i, f in enumerate(inp.frames()):
            x = _diff_prep(f, a.color, a.blur)
            st = None
            d = None
            if len(buf) == a.gap:
                d = _diff(buf[0], x)
                st = _diff_stats(d, thr)
                rows.append({"frame": inp.frame_no(i), "t": inp.t(i), **st})
            img = _diff_view(f, d, view=a.view, thr=thr, gain=a.gain)
            if not a.no_legend:
                _tag(img, label(st))
            writer.write(img)
            buf.append(x)
            count = i + 1
            _progress(i, n_est)
    core.require(bool(rows), f"コマが {a.gap + 1} 個以上必要です")
    big = [r for r in rows if r["changed"] >= a.big]
    changed = [r["changed"] for r in rows]
    data = {
        **inp.head(),
        "frames": count,
        "params": {
            "gap": a.gap,
            "threshold": thr,
            "blur": a.blur,
            "color": a.color,
            "big": a.big,
        },
        "note": "各コマと gap コマ前の差。changed は差がしきい値以上の画素の割合、mean は差の平均（0〜255）",
        "summary": {
            "mean_changed": round(float(np.mean(changed)), 5),
            "max_changed": round(float(np.max(changed)), 5),
            "max_changed_t": rows[int(np.argmax(changed))]["t"],
            "big_changes": len(big),
        },
        "big_changes": [
            {"frame": r["frame"], "t": r["t"], "changed": r["changed"]} for r in big
        ],
        "per_frame": rows,
        "seconds": round(time.time() - t0, 2),
    }
    core.write_json(json_path, data)
    chart = _chart(
        out.with_suffix(".jpg"),
        [r["t"] for r in rows],
        [([c * 100 for c in changed], "変化した画素の割合", "#d1495b")],
        title=f"フレーム間差分（{a.gap} コマ前と比べる・差 {thr:g} 以上）",
        ylabel="%",
        hlines=[(a.big * 100, f"大きな変化（{a.big * 100:.0f}%）", "#888888")],
        marks=[r["t"] for r in big],
    )
    s = data["summary"]
    core.log(
        f"{count} コマ: 変化した所の平均 {s['mean_changed'] * 100:.2f}%、最大 {s['max_changed'] * 100:.1f}%"
        f"（{s['max_changed_t']:.2f} 秒）。画面の {a.big * 100:.0f}% 以上が変わったコマ: {len(big)} 個"
    )
    for r in big[:20]:
        core.log(f"  {r['t']:.3f} 秒（コマ {r['frame']}）: {r['changed'] * 100:.1f}%")
    core.log(f"保存: {out}")
    core.log(f"保存: {chart}")
    core.log(f"保存: {json_path}")


# ── 24 ブロックマッチング ─────────────────────────────────────────────────

_FF_SEARCH = ["esa", "tss", "tdls", "ntss", "fss", "ds", "hexbs", "epzs", "umh"]


class _FullSearch:
    """全探索のブロックマッチング（SAD = 差の絶対値の和）。区画ごとに、前のコマで最も似た所を探す。"""

    def __init__(self, block: int, search: int, subpixel: bool) -> None:
        self.b, self.r, self.subpixel = block, search, subpixel
        r = search
        # 小さいずれから順に並べる（差が同じなら、小さいずれを選ぶ）
        self.disp = sorted(
            ((dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)),
            key=lambda d: (abs(d[0]) + abs(d[1]), d[1], d[0]),
        )
        self.index = np.zeros((2 * r + 1, 2 * r + 1), np.int32)
        for k, (dx, dy) in enumerate(self.disp):
            self.index[dy + r, dx + r] = k

    def __call__(
        self, prev: np.ndarray, cur: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(動き hb×wb×2 = 前のコマ → 今のコマ, 一致の悪さ hb×wb = 1 画素あたりの差の平均)。"""
        b, r = self.b, self.r
        h, w = cur.shape
        hb, wb = h // b, w // b
        c = cur[: hb * b, : wb * b]
        pad = cv2.copyMakeBorder(prev, r, r, r, r, cv2.BORDER_REPLICATE)
        cost = np.empty((len(self.disp), hb, wb), np.float32)
        for k, (dx, dy) in enumerate(self.disp):
            ref = pad[r + dy : r + dy + hb * b, r + dx : r + dx + wb * b]
            d = cv2.absdiff(c, ref).astype(np.float32)
            cost[k] = cv2.resize(d, (wb, hb), interpolation=cv2.INTER_AREA)
        best = cost.argmin(0)
        sad = np.take_along_axis(cost, best[None], 0)[0]
        # 今の区画は、前のコマの (x+dx, y+dy) から来た
        d = np.array(self.disp, np.float32)[best]
        if self.subpixel:
            d = d + self._subpixel(cost, d.astype(np.int32), sad)
        return -d, sad

    def _subpixel(self, cost: np.ndarray, d: np.ndarray, sad: np.ndarray) -> np.ndarray:
        """隣のずれの差から、1 px より細かいずれを求める（SAD に合う「等角直線」の当てはめ）。"""
        r = self.r
        off = np.zeros(d.shape, np.float32)
        yy, xx = np.mgrid[0 : d.shape[0], 0 : d.shape[1]]
        for axis in (0, 1):
            step = np.zeros(2, np.int32)
            step[axis] = 1
            lo, hi = d - step, d + step
            ok = (np.abs(lo).max(-1) <= r) & (np.abs(hi).max(-1) <= r)
            klo = self.index[
                np.clip(lo[..., 1], -r, r) + r, np.clip(lo[..., 0], -r, r) + r
            ]
            khi = self.index[
                np.clip(hi[..., 1], -r, r) + r, np.clip(hi[..., 0], -r, r) + r
            ]
            s_lo, s_hi = cost[klo, yy, xx], cost[khi, yy, xx]
            denom = 2 * (np.maximum(s_lo, s_hi) - sad)
            o = np.where(denom > 1e-6, (s_lo - s_hi) / np.maximum(denom, 1e-6), 0.0)
            off[..., axis] = np.where(ok, np.clip(o, -0.5, 0.5), 0.0)
        return off


class _MEstimate:
    """ffmpeg の mestimate（PyAV のフィルター経由）。1 px 単位。出力は 1 コマ遅れて出てくる。"""

    def __init__(
        self, method: str, block: int, search: int, size: tuple[int, int]
    ) -> None:
        from fractions import Fraction

        import av

        self.av = av
        self.b = block
        self.size = size
        self.graph = av.filter.Graph()
        self.src = self.graph.add_buffer(
            width=size[0], height=size[1], format="gray", time_base=Fraction(1, 1000)
        )
        me = self.graph.add(
            "mestimate", f"method={method}:mb_size={block}:search_param={search}"
        )
        self.sink = self.graph.add("buffersink")
        self.src.link_to(me)
        me.link_to(self.sink)
        self.graph.configure()
        self.n = 0

    def push(self, gray: np.ndarray) -> list[tuple[int, np.ndarray]]:
        fr = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(gray), format="gray")
        fr.pts = self.n
        self.n += 1
        self.src.push(fr)
        out = []
        while True:
            try:
                f = self.sink.pull()
            except (self.av.error.BlockingIOError, self.av.error.EOFError, BlockingIOError, EOFError):  # fmt: skip
                break
            out.append((int(f.pts), self._grid(f)))
        return out

    def _grid(self, f: Any) -> np.ndarray:
        w, h = self.size
        hb, wb = h // self.b, w // self.b
        vec = np.zeros((hb, wb, 2), np.float32)
        for sd in f.side_data:
            if "MOTION" not in str(sd.type).upper():
                continue
            mv = sd.to_ndarray()
            mv = mv[mv["source"] < 0]
            bx = (mv["dst_x"].astype(np.int32) - self.b // 2) // self.b
            by = (mv["dst_y"].astype(np.int32) - self.b // 2) // self.b
            ok = (bx >= 0) & (bx < wb) & (by >= 0) & (by < hb)
            # この区画は、前のコマの src から今の dst へ動いた
            vec[by[ok], bx[ok], 0] = (mv["dst_x"] - mv["src_x"])[ok]
            vec[by[ok], bx[ok], 1] = (mv["dst_y"] - mv["src_y"])[ok]
        return vec


def _block_sad(
    prev: np.ndarray, cur: np.ndarray, vec: np.ndarray, b: int
) -> np.ndarray:
    """区画ごとの、選んだ動きでの一致の悪さ（1 画素あたりの差の平均）。ffmpeg の結果の評価に使う。"""
    hb, wb = vec.shape[:2]
    h, w = prev.shape
    out = np.zeros((hb, wb), np.float32)
    for by in range(hb):
        for bx in range(wb):
            dx, dy = vec[by, bx]
            x0, y0 = bx * b - round(float(dx)), by * b - round(float(dy))
            x0, y0 = min(max(x0, 0), w - b), min(max(y0, 0), h - b)
            blk = cur[by * b : by * b + b, bx * b : bx * b + b].astype(np.float32)
            out[by, bx] = np.abs(blk - prev[y0 : y0 + b, x0 : x0 + b]).mean()
    return out


def _flat_blocks(gray: np.ndarray, b: int, thr: float) -> np.ndarray:
    """模様の少ない区画（明るさの標準偏差が thr 未満）。動きが決まらない。"""
    h, w = gray.shape
    hb, wb = h // b, w // b
    g = gray[: hb * b, : wb * b].astype(np.float32)
    m = cv2.resize(g, (wb, hb), interpolation=cv2.INTER_AREA)
    m2 = cv2.resize(g * g, (wb, hb), interpolation=cv2.INTER_AREA)
    return np.sqrt(np.maximum(m2 - m * m, 0)) < thr


def _render_blocks(
    frame: np.ndarray,
    vec: np.ndarray,
    flat: np.ndarray,
    b: int,
    *,
    view: str,
    scale: float,
    max_mag: float,
    legend: bool,
    label: str,
) -> np.ndarray:
    img = (frame[:, :, :3].astype(np.float32) * 0.7).astype(np.uint8)
    hb, wb = vec.shape[:2]
    if view == "color":
        f = vec.copy()
        f[flat] = 0
        col = ai.flow_to_color(f, max_mag)
        col = cv2.resize(col, (wb * b, hb * b), interpolation=cv2.INTER_NEAREST)
        roi = img[: hb * b, : wb * b]
        roi[:] = cv2.addWeighted(roi, 0.4, col, 0.6, 0)
        for x in range(0, wb * b + 1, b):
            cv2.line(img, (x, 0), (x, hb * b), (40, 40, 40), 1)
        for y in range(0, hb * b + 1, b):
            cv2.line(img, (0, y), (wb * b, y), (40, 40, 40), 1)
    else:
        thick = 2 if max(img.shape[:2]) >= 900 else 1
        for by in range(hb):
            for bx in range(wb):
                cx, cy = bx * b + b / 2, by * b + b / 2
                if flat[by, bx]:
                    cv2.circle(img, (round(cx), round(cy)), 1, (110, 110, 110), -1)
                    continue
                u, v = float(vec[by, bx, 0]), float(vec[by, bx, 1])
                if math.hypot(u, v) < 0.5:  # 止まっている区画は描かない
                    continue
                _arrow(
                    img,
                    (cx, cy),
                    (cx + u * scale, cy + v * scale),
                    _dir_color(u, v),
                    thick,
                )
    if legend:
        _tag(img, label)
        if view == "color":
            _wheel_legend(img, f"最大 {max_mag:.3g} px/コマ")
        else:
            core.draw_text(
                img,
                f"矢印の長さ ×{scale:.2g}　灰色の点＝模様が無い区画",
                (img.shape[1] - 8, img.shape[0] - 8),
                size=_font_size(img),
                bg=(0, 0, 0),
                anchor="rb",
            )
    return img


def _block_stats(
    vec: np.ndarray, sad: np.ndarray, flat: np.ndarray, to_src: float
) -> dict[str, Any]:
    ok = ~flat
    if not ok.any():
        return {"reliable": 0.0}
    v = vec[ok] * to_src
    mag = np.hypot(v[:, 0], v[:, 1])
    q = np.round(v).astype(np.int32)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    dom = keys[int(np.argmax(counts))]
    return {
        "reliable": round(float(ok.mean()), 4),
        "mean_dx": round(float(v[:, 0].mean()), 4),
        "mean_dy": round(float(v[:, 1].mean()), 4),
        "dominant": [int(dom[0]), int(dom[1])],
        "dominant_share": round(float(counts.max() / len(q)), 4),
        "moving": round(float((mag >= 1.0).mean()), 4),
        "mean_speed": round(float(mag.mean()), 4),
        "match_error": round(float(np.median(sad[ok])), 3),
    }


@command(
    24,
    "block-match",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    NEXT,
    arg("--block", type=int, default=16, help="区画の大きさ（px。既定 16）"),
    arg(
        "--range",
        dest="search_range",
        metavar="PX",
        type=int,
        default=16,
        help="探す範囲（±px。既定 16。これより速い動きは測れない）",
    ),
    arg(
        "--search",
        choices=["full", *_FF_SEARCH],
        default="full",
        help="探し方。full＝全探索（自前。0.1 px 単位まで求める。既定）。ほかは ffmpeg の mestimate（1 px 単位）: "
        "esa＝全探索、tss＝3 段階探索、tdls＝2 次元対数探索、ntss＝新 3 段階探索、fss＝4 段階探索、ds＝ダイヤモンド探索、"
        "hexbs＝六角形探索、epzs＝予測つきの区域探索、umh＝不均一な多重六角形探索",
    ),
    arg(
        "--no-subpixel",
        action="store_true",
        help="1 px より細かいずれを求めない（full の時）",
    ),
    arg(
        "--flat",
        type=float,
        default=3.0,
        help="明るさの標準偏差がこれ未満の区画は、模様が無く動きが決まらないので除く（0〜255。既定 3）",
    ),
    arg(
        "--view",
        choices=["arrows", "color"],
        default="arrows",
        help="描き方。arrows＝区画ごとの矢印（既定）、color＝区画を動きの色（色相 = 向き、明るさ = 速さ）で塗る",
    ),
    arg(
        "--arrow-scale",
        type=float,
        help="矢印の長さの倍率（省略時は自動: 速い動き（99%% 点）が区画の大きさの 0.9 倍の長さ。1 で実際の長さ）",
    ),
    arg(
        "--save-vectors",
        action="store_true",
        help="区画ごとの動き（解析する大きさの px）と一致の悪さを .npz に保存する",
    ),
    NO_LEGEND,
)
def block_match(a: argparse.Namespace) -> None:
    """小さな区画ごとに、前のコマで最も似た所を探して動きを求め、矢印の動画（MP4）と動きの数値（JSON・.npz）を書く。"""
    b, r = a.block, a.search_range
    core.require(4 <= b <= 128, "--block は 4〜128 にしてください")
    core.require(1 <= r <= 64, "--range は 1〜64 にしてください")
    if a.search != "full":
        core.require(b >= 8, "ffmpeg の探し方では --block は 8 以上にしてください")
        core.require(r >= 4, "ffmpeg の探し方では --range は 4 以上にしてください")
    inp = _open(a)
    out, json_path = _out_paths(a, inp, "block-match")
    npz_path = out.with_suffix(".npz")
    k = inp.to_src
    w, h = inp.size
    hb, wb = h // b, w // b
    core.require(
        hb >= 1 and wb >= 1, "画面が区画より小さいです（--block を小さくしてください）"
    )
    full = _FullSearch(b, r, not a.no_subpixel) if a.search == "full" else None
    label = f"ブロックマッチング（{b}×{b}・±{r}px・{a.search}）"
    t0 = time.time()

    def make_ff() -> _MEstimate:
        return _MEstimate(a.search, b, r, (w, h))

    if inp.pair is not None:
        g0, g1 = (core.to_gray(f) for f in inp.pair)
        if full is not None:
            vec, sad = full(g0, g1)
        else:
            me = make_ff()
            res = dict(me.push(g0) + me.push(g1) + me.push(g1))
            vec = res[1]
            sad = _block_sad(g0, g1, vec, b)
        flat = _flat_blocks(g1, b, a.flat)
        mag = np.hypot(vec[..., 0], vec[..., 1])[~flat]
        p99 = max(1.0, float(np.percentile(mag, 99))) if mag.size else 1.0
        scale = a.arrow_scale or 0.9 * b / p99
        img = _render_blocks(
            inp.pair[1], vec, flat, b, view=a.view, scale=scale, max_mag=p99,
            legend=not a.no_legend, label=label,
        )  # fmt: skip
        core.save_image(out, img)
        stats = _block_stats(vec, sad, flat, k)
        data = {
            **inp.head(),
            "params": {"block": b, "range": r, "search": a.search, "subpixel": full is not None and not a.no_subpixel},
            "note": "区画（解析する大きさで block×block）ごとの、1 枚目 → 2 枚目の動き [横, 縦]（元の px）。null は模様が無い区画",
            "grid": [hb, wb],
            "stats": stats,
            "vectors": [
                [None if flat[y, x] else [round(float(vec[y, x, 0] * k), 2), round(float(vec[y, x, 1] * k), 2)] for x in range(wb)]
                for y in range(hb)
            ],
            "seconds": round(time.time() - t0, 3),
        }  # fmt: skip
        if a.save_vectors:
            _save_npz(
                npz_path,
                vectors=vec[None],
                sad=sad[None],
                flat=flat[None],
                block=b,
                to_source=k,
            )
            data["npz"] = str(npz_path)
            core.log(f"保存: {npz_path}")
        core.write_json(json_path, data)
        core.log(
            f"区画 {hb}×{wb}（模様のある区画 {stats.get('reliable', 0) * 100:.0f}%）: 最も多い動き "
            f"{stats.get('dominant')}（{stats.get('dominant_share', 0) * 100:.0f}%）、平均 "
            f"({stats.get('mean_dx', 0):+.2f}, {stats.get('mean_dy', 0):+.2f}) px"
        )
        core.log(f"保存: {out}")
        core.log(f"保存: {json_path}")
        return

    # 動画: 矢印の倍率は、いくつかのコマで DIS の動きを測って決める（探す範囲より速い動きは測れないので、範囲で頭打ち）
    if a.arrow_scale:
        scale = a.arrow_scale
    else:
        p99 = min(_auto_max_mag(inp), float(r))
        scale = 0.9 * b / p99
    if a.save_vectors:
        _check_npz_size(inp, hb * wb * 13, "保存する動き")
    n_est = inp.count()
    rows: list[dict[str, Any]] = []
    vecs, sads, flats = [], [], []
    me = make_ff() if full is None else None
    pending: dict[
        int, np.ndarray
    ] = {}  # ffmpeg: コマ番号 → 色つきのコマ（結果が出るまで待つ）
    grays: dict[int, np.ndarray] = {}
    t_search = 0.0
    count = 0
    with inp.writer(out) as writer:

        def emit(
            i: int,
            frame: np.ndarray,
            vec: np.ndarray | None,
            sad: np.ndarray | None,
            gray: np.ndarray,
        ) -> None:
            flat = _flat_blocks(gray, b, a.flat)
            if vec is None:
                vec = np.zeros((hb, wb, 2), np.float32)
                sad = np.zeros((hb, wb), np.float32)
                stats = None
            else:
                assert sad is not None
                stats = _block_stats(vec, sad, flat, k)
                rows.append({"frame": inp.frame_no(i), "t": inp.t(i), **stats})
                if a.save_vectors:
                    vecs.append(vec.astype(np.float32))
                    sads.append(sad.astype(np.float16))
                    flats.append(flat)
            text = (
                label
                if stats is None
                else f"{label} 平均 {stats.get('mean_speed', 0):.2f}px"
            )
            writer.write(
                _render_blocks(
                    frame,
                    vec,
                    flat,
                    b,
                    view=a.view,
                    scale=scale,
                    max_mag=0.9 * b / scale,
                    legend=not a.no_legend,
                    label=text,
                )
            )

        prev_g = None
        for i, f in enumerate(inp.frames()):
            g = core.to_gray(f)
            t1 = time.time()
            if full is not None:
                if prev_g is None:
                    emit(i, f, None, None, g)
                else:
                    vec, sad = full(prev_g, g)
                    t_search += time.time() - t1
                    emit(i, f, vec, sad, g)
            else:
                assert me is not None
                pending[i] = f
                grays[i] = g
                results = me.push(g)
                t_search += time.time() - t1
                for j, vec in results:
                    if j in pending:
                        fj, gj = pending.pop(j), grays[j]
                        if j == 0:
                            emit(j, fj, None, None, gj)
                        else:
                            emit(j, fj, vec, _block_sad(grays[j - 1], gj, vec, b), gj)
                    grays = {kk: vv for kk, vv in grays.items() if kk >= j}
            prev_g = g
            count = i + 1
            _progress(i, n_est)
        if me is not None and pending:
            # 最後のコマの結果は、同じコマをもう 1 度入れると出てくる
            last = max(pending)
            t1 = time.time()
            results = me.push(grays[last])
            t_search += time.time() - t1
            for j, vec in results:
                if j in pending:
                    fj, gj = pending.pop(j), grays[j]
                    if j == 0:
                        emit(j, fj, None, None, gj)
                    else:
                        emit(j, fj, vec, _block_sad(grays[j - 1], gj, vec, b), gj)
    core.require(bool(rows), "動画のコマが 2 つ以上必要です")
    speed = [r_["mean_speed"] for r_ in rows if "mean_speed" in r_]
    data = {
        **inp.head(),
        "frames": count,
        "params": {"block": b, "range": r, "search": a.search, "subpixel": full is not None and not a.no_subpixel, "flat": a.flat},
        "grid": [hb, wb],
        "note": "各コマの区画が、前のコマからどれだけ動いたか（元の px）。dominant は最も多い動き（1 px 単位）、"
        "match_error は一致の悪さ（1 画素あたりの差の平均、0〜255。大きいと見つからなかった区画が多い）",
        "summary": {
            "mean_speed": round(float(np.mean(speed)), 4) if speed else 0.0,
            "seconds_per_frame_search": round(t_search / max(1, len(rows)), 4),
        },
        "per_frame": rows,
        "seconds": round(time.time() - t0, 2),
    }  # fmt: skip
    if a.save_vectors:
        _save_npz(
            npz_path, vectors=np.stack(vecs), sad=np.stack(sads), flat=np.stack(flats),
            t=np.array([r_["t"] for r_ in rows]), frame=np.array([r_["frame"] for r_ in rows]),
            block=b, to_source=k,
        )  # fmt: skip
        data["npz"] = str(npz_path)
        core.log(f"保存: {npz_path}")
    core.write_json(json_path, data)
    core.log(
        f"{count} コマ・区画 {hb}×{wb}: 平均の速さ {data['summary']['mean_speed']:.2f} px/コマ、"
        f"探す時間 {data['summary']['seconds_per_frame_search']:.3f} 秒/コマ"
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 13 動き補償 ───────────────────────────────────────────────────────────

_MODELS = ["homography", "affine", "similarity", "translation"]


def _m3(m: np.ndarray) -> np.ndarray:
    return (
        np.vstack([m, [0.0, 0.0, 1.0]]) if m.shape == (2, 3) else m.astype(np.float64)
    )


def _fit(
    src: np.ndarray, dst: np.ndarray, model: str, thr: float = 2.0
) -> tuple[np.ndarray | None, np.ndarray]:
    """点の組（src → dst）に合う変換（3×3）と、合った点の印。"""
    n = len(src)
    none = np.zeros(n, bool)
    if n < 4 and model != "translation" or n < 2:
        return None, none
    if model == "homography":
        m, inl = cv2.findHomography(
            src, dst, cv2.RANSAC, thr, maxIters=3000, confidence=0.995
        )
    elif model == "affine":
        m, inl = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=3000
        )
    elif model == "similarity":
        m, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=3000
        )
    else:
        d = dst - src
        med = np.median(d, 0)
        inl = np.linalg.norm(d - med, axis=1) < thr
        if inl.sum() < 2:
            return None, none
        dx, dy = d[inl].mean(0)
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64), inl
    if m is None or inl is None:
        return None, none
    return _m3(m), inl.ravel().astype(bool)


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return p[:, :2] / p[:, 2:3]


def _features(gray: np.ndarray, n: int = 1200) -> np.ndarray | None:
    pts = cv2.goodFeaturesToTrack(gray, n, 0.005, 7, blockSize=7)
    return None if pts is None else pts.reshape(-1, 2).astype(np.float32)


@dataclass
class _Est:
    h: np.ndarray  # 今のコマの px → 基準のコマの px
    ok: bool
    inliers: int
    residual: float  # 合った点の、当てはめとのずれ（px、中央値）


def _track_fit(
    ref_gray: np.ndarray, gray: np.ndarray, model: str, pts: np.ndarray | None = None
) -> _Est:
    """基準のコマの特徴点を今のコマへ追い、「今 → 基準」の変換を求める。"""
    if pts is None:
        pts = _features(ref_gray)
    if pts is None or len(pts) < 8:
        return _Est(np.eye(3), False, 0, math.nan)
    p1, ok = ai.track_points(ref_gray, gray, pts)
    if ok.sum() < 8:
        return _Est(np.eye(3), False, int(ok.sum()), math.nan)
    h, inl = _fit(p1[ok], pts[ok], model)
    if h is None or inl.sum() < 8:
        return _Est(np.eye(3), False, int(inl.sum()), math.nan)
    res = np.linalg.norm(_apply(h, p1[ok][inl]) - pts[ok][inl], axis=1)
    return _Est(h, True, int(inl.sum()), float(np.median(res)))


def _sift_pair(a_gray: np.ndarray, b_gray: np.ndarray, model: str) -> _Est:
    """離れた 2 枚の画像: SIFT の特徴量で対応を取り、「2 枚目 → 1 枚目」の変換を求める。"""
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(a_gray, None)
    kb, db = sift.detectAndCompute(b_gray, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return _Est(np.eye(3), False, 0, math.nan)
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [
        m
        for m, n in (p for p in matches if len(p) == 2)
        if m.distance < 0.75 * n.distance
    ]
    if len(good) < 8:
        return _Est(np.eye(3), False, len(good), math.nan)
    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([ka[m.trainIdx].pt for m in good])
    thr = max(2.0, max(a_gray.shape) / 300)
    h, inl = _fit(src, dst, model, thr)
    if h is None or inl.sum() < 8:
        return _Est(np.eye(3), False, int(inl.sum()), math.nan)
    res = np.linalg.norm(_apply(h, src[inl]) - dst[inl], axis=1)
    return _Est(h, True, int(inl.sum()), float(np.median(res)))


def _warp(
    img: np.ndarray, h: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """「img の px → 出力の px」の h で img を写す。(写した画, 中身のある所 0〜1)"""
    out = cv2.warpPerspective(
        img, h, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    ones = np.full(img.shape[:2], 255, np.uint8)
    valid = cv2.warpPerspective(
        ones, h, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return out, valid.astype(np.float32) / 255.0


def _params(h: np.ndarray) -> dict[str, float]:
    """3×3 を「移動・回転・拡大」の目安に分ける（画面の中心で見た値ではなく、原点で見た値）。"""
    hn = h / h[2, 2]
    return {
        "dx": round(float(hn[0, 2]), 3),
        "dy": round(float(hn[1, 2]), 3),
        "angle": round(float(np.degrees(np.arctan2(hn[1, 0], hn[0, 0]))), 4),
        "scale": round(float(np.sqrt(abs(np.linalg.det(hn[:2, :2])))), 5),
    }


def _to_src_h(h: np.ndarray, k: float) -> list[list[float]]:
    """解析の px の変換 → 元の px の変換（JSON 用）。"""
    s = np.diag([1 / k, 1 / k, 1.0])
    m = np.linalg.inv(s) @ h @ s
    m = m / m[2, 2]
    return [[round(float(v), 6) for v in row] for row in m]


@command(
    13,
    "motion-compensate",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    arg(
        "--next",
        help="2 枚目の画像（SRC が画像の時。NEXT を SRC に合わせる。大きさや向きが違ってもよい）",
    ),
    arg(
        "--mode",
        choices=["align", "average", "diff"],
        default="align",
        help="align＝すべてのコマを最初のコマ（--start の位置）に合わせる（カメラの動きが消える。既定）、"
        "average＝前後のコマを今のコマに合わせて重ねる（ノイズが減る。--stat median で動く人や物が消える）、"
        "diff＝前のコマを今のコマに合わせてから差を取る（カメラが動いても、動く物だけが出る）",
    ),
    arg(
        "--model",
        choices=_MODELS,
        default="homography",
        help="合わせる動きの種類。homography＝遠近の変形まで（既定）、affine＝ゆがみまで、similarity＝移動・回転・拡大、"
        "translation＝移動だけ",
    ),
    arg(
        "--frames",
        type=int,
        default=5,
        help="average で重ねるコマ数（今のコマを中心に前後。奇数。既定 5）",
    ),
    arg(
        "--stat",
        choices=["mean", "median"],
        default="mean",
        help="average の重ね方。mean＝平均（ノイズが減る・動く物は流れる）、median＝中央値（動く人や物が消え、背景が残る）",
    ),
    arg("--gap", type=int, default=1, help="diff で何コマ前と比べるか（既定 1）"),
    arg(
        "--threshold",
        type=float,
        default=25,
        help="diff で変化とみなす差（0〜255。既定 25）",
    ),
    arg(
        "--blur",
        type=float,
        default=1.0,
        help="diff で比べる前のぼかし（ガウスの σ、px。既定 1）",
    ),
    arg(
        "--view",
        choices=["heat", "mask"],
        default="heat",
        help="diff の描き方。heat＝熱の色で重ねる（既定）、mask＝変化した所が白の白黒",
    ),
    NO_LEGEND,
)
def motion_compensate(a: argparse.Namespace) -> None:
    """カメラの動き（画面全体の動き）を求めてコマどうしを合わせ、揃えた映像・重ねた映像・動く物だけの差を書く。"""
    core.require(a.frames >= 2, "--frames は 2 以上にしてください")
    core.require(a.gap >= 1, "--gap は 1 以上にしてください")
    inp = _open(a, same_size=False)
    out, json_path = _out_paths(a, inp, f"motion-compensate-{a.mode}")
    k = inp.to_src
    t0 = time.time()
    if inp.pair is not None:
        _mc_pair(a, inp, out, json_path)
        return
    w, h = inp.size
    n_est = inp.count()
    rows: list[dict[str, Any]] = []
    fails = 0
    model = a.model
    title ={"align": "動き補償（最初のコマに合わせる）", "average": f"動き補償で {a.frames} コマを重ねる（{a.stat}）", "diff": f"動き補償してから差分（{a.gap} コマ前）"}[a.mode]  # fmt: skip

    with inp.writer(out) as writer:
        if a.mode == "align":
            ref_gray = None
            ref_pts = None
            prev_gray = None
            c_h = np.eye(3)  # 今のコマ → 最初のコマ
            for i, f in enumerate(inp.frames()):
                g = core.to_gray(f)
                row: dict[str, Any] = {"frame": inp.frame_no(i), "t": inp.t(i)}
                if ref_gray is None:
                    ref_gray, ref_pts = g, _features(g, 1500)
                    row.update(ok=True, inliers=0, residual_px=0.0)
                else:
                    assert prev_gray is not None
                    step = _track_fit(prev_gray, g, model)
                    if not step.ok:
                        fails += 1
                    c_h = c_h @ step.h
                    # 最初のコマと直接合わせて、つなぎの誤差がたまらないようにする
                    warped, valid = _warp(g, c_h, (w, h))
                    pts = ref_pts
                    if pts is not None:
                        pts = pts[
                            valid[
                                pts[:, 1].astype(int).clip(0, h - 1),
                                pts[:, 0].astype(int).clip(0, w - 1),
                            ]
                            > 0.99
                        ]
                    fix = _track_fit(ref_gray, warped, model, pts)
                    if fix.ok:
                        c_h = fix.h @ c_h
                    row.update(ok=bool(step.ok), direct=bool(fix.ok), inliers=fix.inliers if fix.ok else step.inliers, residual_px=round(fix.residual * k, 3) if fix.ok else None, step=_params(step.h))  # fmt: skip
                c_h = c_h / c_h[2, 2]
                row["to_first"] = _to_src_h(c_h, k)
                row["camera"] = _params(np.linalg.inv(c_h))
                img, _ = _warp(f, c_h, (w, h))
                if not a.no_legend:
                    _tag(img, f"{title} {inp.t(i):.2f}秒")
                writer.write(img)
                rows.append(row)
                prev_gray = g
                _progress(i, n_est)
        else:
            half = a.frames // 2
            look = half if a.mode == "average" else 0
            frames_buf: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = deque()
            # i のコマ → 最初のコマ（つないだ物。近いコマどうしの変換に使う）
            c_h = np.eye(3)
            prev_gray = None
            keep = max(a.frames, a.gap + 1) + 1

            def emit(ci: int) -> None:
                items = {i: (f, g, c) for i, f, g, c in frames_buf}
                f, g, cc = items[ci]
                row: dict[str, Any] = {"frame": inp.frame_no(ci), "t": inp.t(ci)}
                inv_c = np.linalg.inv(cc)
                if a.mode == "average":
                    lo, hi = ci - half, ci + (a.frames - 1 - half)
                    acc = []
                    for j in range(lo, hi + 1):
                        if j not in items:
                            continue
                        fj, _, cj = items[j]
                        hj = inv_c @ cj  # j のコマ → 今のコマ
                        wj, vj = _warp(fj, hj, (w, h))
                        acc.append((wj.astype(np.float32), vj))
                    if a.stat == "mean":
                        num = sum(x * v[..., None] for x, v in acc)
                        den = sum(v for _, v in acc)[..., None]
                        img = np.clip(num / np.maximum(den, 1e-3), 0, 255).astype(
                            np.uint8
                        )
                    else:
                        base = f.astype(np.float32)
                        stack = np.stack(
                            [np.where(v[..., None] > 0.99, x, base) for x, v in acc]
                        )
                        img = np.median(stack, axis=0).astype(np.uint8)
                    row["used"] = len(acc)
                    text = f"{title} {len(acc)} コマ"
                else:
                    j = ci - a.gap
                    if j not in items:
                        img = _diff_view(f, None, view=a.view, thr=a.threshold, gain=1)
                        text = f"{title}（比べるコマがまだ無い）"
                    else:
                        fj, gj, cj = items[j]
                        hj = inv_c @ cj
                        wj, vj = _warp(gj, hj, (w, h))
                        vj = cv2.erode(
                            (vj > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8)
                        )
                        x0 = _diff_prep(g, False, a.blur)
                        x1 = _diff_prep(wj, False, a.blur)
                        d = _diff(x1, x0)
                        st = _diff_stats(d, a.threshold, vj)
                        plain = _diff_stats(
                            _diff(_diff_prep(gj, False, a.blur), x0), a.threshold
                        )
                        row.update(changed=st["changed"], mean=st["mean"], changed_plain=plain["changed"], mean_plain=plain["mean"])  # fmt: skip
                        img = _diff_view(
                            f,
                            d,
                            view=a.view,
                            thr=a.threshold,
                            gain=1,
                            valid=vj.astype(np.float32),
                        )
                        text = f"{title} 変化 {st['changed'] * 100:.1f}%（補償なし {plain['changed'] * 100:.1f}%）"
                if not a.no_legend:
                    _tag(img, text)
                writer.write(img)
                rows.append(row)

            next_emit = 0
            for i, f in enumerate(inp.frames()):
                g = core.to_gray(f)
                if prev_gray is not None:
                    step = _track_fit(prev_gray, g, model)
                    if not step.ok:
                        fails += 1
                    c_h = c_h @ step.h
                    c_h = c_h / c_h[2, 2]
                frames_buf.append((i, f, g, c_h.copy()))
                while len(frames_buf) > keep:
                    frames_buf.popleft()
                while next_emit + look <= i:
                    emit(next_emit)
                    next_emit += 1
                prev_gray = g
                _progress(i, n_est)
            last = frames_buf[-1][0]
            while next_emit <= last:
                emit(next_emit)
                next_emit += 1
    data: dict[str, Any] = {
        **inp.head(),
        "frames": len(rows),
        "mode": a.mode,
        "model": model,
        "failed": fails,
        "seconds": round(time.time() - t0, 2),
    }
    if a.mode == "align":
        data["note"] = (
            "to_first は「このコマの px → 最初のコマの px」の 3×3（元の px）。camera はその逆（カメラの動きの目安）"
        )
    elif a.mode == "average":
        data["note"] = (
            f"今のコマを中心に {a.frames} コマを合わせて重ねた（{a.stat}）。used は重ねたコマ数（端では少ない）"
        )
    else:
        data["note"] = (
            "changed は合わせた後の差がしきい値以上の画素の割合（画面の外にはみ出た所を除く）、changed_plain は合わせない時"
        )
        ch = [r_["changed"] for r_ in rows if "changed" in r_]
        pl = [r_["changed_plain"] for r_ in rows if "changed_plain" in r_]
        if ch:
            data["summary"] = {
                "mean_changed": round(float(np.mean(ch)), 5),
                "mean_changed_plain": round(float(np.mean(pl)), 5),
            }
    data["per_frame"] = rows
    core.write_json(json_path, data)
    if fails:
        core.log(
            f"動きを求められなかったコマ: {fails} 個（カット・模様の無い画面など。動きなしとして扱った）"
        )
    if a.mode == "align":
        res = [
            r_["residual_px"] for r_ in rows[1:] if r_.get("residual_px") is not None
        ]
        if res:
            core.log(
                f"{len(rows)} コマを最初のコマに合わせました（合わせた後のずれの中央値 {np.median(res):.2f} px）"
            )
    elif a.mode == "diff" and "summary" in data:
        s = data["summary"]
        core.log(
            f"{len(rows)} コマ: 変化した所の平均 {s['mean_changed'] * 100:.2f}%（合わせない時 {s['mean_changed_plain'] * 100:.2f}%）"
        )
    else:
        core.log(f"{len(rows)} コマを、前後のコマと合わせて重ねました")
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


def _mc_pair(a: argparse.Namespace, inp: _Input, out: Path, json_path: Path) -> None:
    """画像 2 枚: 2 枚目を 1 枚目に合わせる。"""
    assert inp.pair is not None
    first, second = inp.pair
    ga, gb = core.to_gray(first), core.to_gray(second)
    e = _sift_pair(ga, gb, a.model)
    core.require(
        e.ok,
        "2 枚の画像の対応が見つかりませんでした（同じ物が写っているか確かめてください）",
    )
    hm = e.h
    w, h = first.shape[1], first.shape[0]
    warped, valid = _warp(second, hm, (w, h))
    data: dict[str, Any] = {
        **inp.head(),
        "mode": a.mode,
        "model": a.model,
        "inliers": e.inliers,
        "residual_px": round(e.residual * inp.to_src, 3),
        "next_to_src": _to_src_h(hm, inp.to_src),
        "note": "next_to_src は「2 枚目の px → 1 枚目の px」の 3×3（元の px）",
    }
    if a.mode == "align":
        img = warped
    elif a.mode == "average":
        v = valid[..., None]
        img = np.clip(
            (first.astype(np.float32) + warped.astype(np.float32) * v) / (1 + v), 0, 255
        ).astype(np.uint8)
    else:
        vv = cv2.erode(
            (valid > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(np.float32)
        d = _diff(_diff_prep(warped, False, a.blur), _diff_prep(first, False, a.blur))
        st = _diff_stats(d, a.threshold, vv)
        data["diff"] = st
        img = _diff_view(first, d, view=a.view, thr=a.threshold, gain=1, valid=vv)
        if not a.no_legend:
            _tag(img, f"合わせた後の差 変化 {st['changed'] * 100:.1f}%")
    core.save_image(out, img)
    core.write_json(json_path, data)
    core.log(
        f"対応点 {e.inliers} 個で合わせました（ずれの中央値 {e.residual * inp.to_src:.2f} px）"
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {json_path}")


# ── 23 シーンフロー ───────────────────────────────────────────────────────


def _fb_valid(fwd: np.ndarray, bwd: np.ndarray) -> np.ndarray:
    """行きと帰りのフローが打ち消し合う画素（隠れる所・見失った所を除く）。"""
    h, w = fwd.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + fwd[..., 0], gy + fwd[..., 1]
    back = cv2.remap(bwd, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.hypot(fwd[..., 0] + back[..., 0], fwd[..., 1] + back[..., 1])
    mag = np.hypot(fwd[..., 0], fwd[..., 1])
    inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return inside & (err < 0.5 + 0.05 * mag)


def _jacobian(flow: np.ndarray, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """細かい尺度でのフローの微分。(det(I + J), |J|)。"""
    f = cv2.GaussianBlur(flow, (0, 0), sigma)
    u, v = np.ascontiguousarray(f[..., 0]), np.ascontiguousarray(f[..., 1])
    ux = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=1) * 0.5
    uy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=1) * 0.5
    vx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=1) * 0.5
    vy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=1) * 0.5
    det = (1 + ux) * (1 + vy) - uy * vx
    return det, np.sqrt(ux * ux + uy * uy + vx * vx + vy * vy)


def _boundary_free(flow: np.ndarray, valid: np.ndarray, gap: int) -> np.ndarray:
    """拡大率を求めてよい画素: 行き帰りが合い、動きの境目（フローが急に変わる所）から離れた所。"""
    _, jn = _jacobian(flow)
    edge = (jn > 0.12 * gap).astype(np.uint8)
    edge = cv2.dilate(edge, np.ones((7, 7), np.uint8))
    return valid & (edge == 0)


def _expansion(
    flow: np.ndarray, sigma: float, ok: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """局所の拡大率 s（中身が s 倍に大きく見えた。s > 1 は近づいた）と、その確かさ（0〜1）。

    細かい尺度でフローを微分し（局所のアフィン変換の行列 J）、log s = ½ log det(I + J) を、
    動きの境目と隠れる所（ok が偽）を除いて σ でならす。境目をまたいでならすと、近づく物のまわりに
    「遠ざかる」輪ができる（合成映像で確認）ため、先に微分してから境目を除いてならす。
    """
    det, _ = _jacobian(flow)
    logs = 0.5 * np.log(np.clip(det, 0.25, 4.0))
    wgt = ok.astype(np.float32)
    num = cv2.GaussianBlur(logs * wgt, (0, 0), sigma)
    den = cv2.GaussianBlur(wgt, (0, 0), sigma)
    return np.exp(num / np.maximum(den, 1e-3)), den


class _RawDepth:
    """Depth Anything V2 Small の生の出力（相対的な逆深度。大きいほど手前。0〜1 にそろえる前の値）。"""

    def __init__(self) -> None:
        self.est = ai.DepthEstimator()

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        e = self.est
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        inputs = e.proc(images=rgb, return_tensors="pt")
        pred = e.net(**inputs).predicted_depth
        pred = e.torch.nn.functional.interpolate(
            pred[:, None], size=(h, w), mode="bicubic", align_corners=False
        )
        return np.maximum(pred[0, 0].numpy().astype(np.float32), 1e-3)


def _depth_expansion(
    d0: np.ndarray, d1: np.ndarray, flow: np.ndarray, ok: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """奥行きから求めた拡大率: フローに沿って「次のコマの逆深度 ÷ 今のコマの逆深度」。

    逆深度 ∝ 1/距離 なので、この比は「距離が何分の 1 になったか」= 拡大率に当たる。Depth Anything の出力は
    コマごとに倍率が決まらないので、画面全体の中央値で割ってそろえる（そのため画面全体の近づきは消える）。
    """
    h, w = d0.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    d1w = cv2.remap(
        d1,
        gx + flow[..., 0],
        gy + flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logr = np.log(np.clip(d1w / d0, 0.2, 5.0))
    med = float(np.median(logr[ok])) if ok.any() else 0.0
    wgt = ok.astype(np.float32)
    num = cv2.GaussianBlur((logr - med) * wgt, (0, 0), sigma)
    den = cv2.GaussianBlur(wgt, (0, 0), sigma)
    return np.exp(num / np.maximum(den, 1e-3)), den


_NEAR = (40, 70, 255)  # 近づく（赤）BGR
_FAR = (255, 150, 30)  # 遠ざかる（青）
_SCENE_NPZ_NOTE = (
    "flow: 2D の動き（px/コマ、解析する大きさ）、rate: 1 コマあたりの拡大率 − 1（+ 近づく）、valid: 使えた画素、"
    "sf: 3D の動き（ΔX/Z, ΔY/Z, ΔZ/Z。X 右・Y 下・Z 奥。そのコマの距離 Z に対する割合/コマ。fov の画角で計算）、"
    "depth: Depth Anything V2 の生の出力（相対的な逆深度。--method depth の時だけ）"
)


def _scene_arrays(res: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """.npz に入れる配列（大きさを抑えるため float16 と bool）。"""
    out = {
        "flow": res["flow"].astype(np.float16),
        "rate": res["rate"].astype(np.float16),
        "valid": res["valid"] > 0,
        "sf": res["sf"].astype(np.float16),
    }
    if "depth" in res:
        out["depth"] = res["depth"].astype(np.float16)
    return out


def _render_scene(
    frame: np.ndarray,
    rate: np.ndarray,
    valid: np.ndarray,
    flow: np.ndarray,
    *,
    max_rate: float,
    scale: float | None,
    step: int,
    min_mag: float,
    legend: bool,
    label: str,
) -> np.ndarray:
    """近づく所を赤、遠ざかる所を青で塗る（濃さ = 速さ）。scale があれば横・縦の動きの白い矢印も描く。"""
    base = frame[:, :, :3].astype(np.float32) * 0.5
    r = np.clip(rate / max_rate, -1, 1)
    near, far = np.float32(_NEAR), np.float32(_FAR)
    col = np.where(r[..., None] > 0, near, far)
    alpha = np.clip((np.abs(r) - 0.1) / 0.6, 0, 1) * valid
    img = base * (1 - alpha[..., None]) + col * alpha[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    if scale:
        _flow_arrows(img, flow, step, scale, min_mag, color=(255, 255, 255))
    if legend:
        _tag(img, label)
        _bar_legend(
            img, (_FAR, _NEAR), ("遠ざかる", f"±{max_rate * 100:.2g}%/コマ", "近づく")
        )
    return img


@command(
    23,
    "scene-flow",
    SRC,
    OUT,
    JSON,
    RANGE,
    SIZE,
    NEXT,
    arg(
        "--method",
        choices=["expansion", "depth"],
        default="expansion",
        help="奥行き方向の動きの求め方。expansion＝フローから、物が大きく・小さく見える割合（局所の拡大率）を求める（既定）、"
        "depth＝Depth Anything V2 の奥行きを前後のコマで比べる（比べる用。遅く、合成の試験では見分けが 5〜6 割と不正確。"
        "画面全体の近づき・遠ざかりは測れない）",
    ),
    arg(
        "--flow",
        choices=["dis", "farneback", "raft"],
        default="dis",
        help="2D の動きの測り方（既定 dis）",
    ),
    arg(
        "--gap",
        type=int,
        default=1,
        help="何コマ先と比べるか（既定 1。ゆっくりした近づきは 2〜4 にすると安定する。結果は 1 コマあたりに直す）",
    ),
    arg(
        "--smooth",
        type=float,
        help="拡大率を求める広さ（ガウスの σ、解析する大きさの px。省略時は短い辺の 1/30）",
    ),
    arg(
        "--max-rate",
        type=float,
        help="色が最も濃くなる近づく・遠ざかる速さ（%%/コマ。省略時は自動: いくつかのコマで測った値の 98%% 点）",
    ),
    arg(
        "--threshold",
        type=float,
        default=0.3,
        help="近づく・遠ざかるとみなす速さ（%%/コマ。既定 0.3）",
    ),
    arg(
        "--fov",
        type=float,
        default=60.0,
        help="横の画角（度。既定 60）。.npz の 3D の向き（距離に対する割合）を求めるのに使う",
    ),
    arg("--no-arrows", action="store_true", help="横・縦の動きの矢印を描かない"),
    arg(
        "--save-flow",
        action="store_true",
        help="2D の動き・拡大率・3D の動き（距離に対する割合）を .npz に保存する",
    ),
    NO_LEGEND,
)
def scene_flow(a: argparse.Namespace) -> None:
    """2D の動きに奥行き方向の動き（近づく・遠ざかる）を加えて測り、色で塗った動画（MP4）と数値（JSON・.npz）を書く。"""
    core.require(a.gap >= 1, "--gap は 1 以上にしてください")
    inp = _open(a, default_max_side=960 if a.flow == "raft" else None)
    out, json_path = _out_paths(a, inp, "scene-flow")
    npz_path = out.with_suffix(".npz")
    k = inp.to_src
    w, h = inp.size
    sigma = a.smooth or max(4.0, min(w, h) / 30)
    fn = _flow_fn(a.flow)
    depth = _RawDepth() if a.method == "depth" else None
    step = max(8, round(max(w, h) / 30))
    thr = a.threshold / 100
    fx = (w / 2) / math.tan(math.radians(a.fov) / 2)
    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32) - w / 2, np.arange(h, dtype=np.float32) - h / 2
    )
    if a.save_flow and not inp.is_pair:
        _check_npz_size(inp, w * h * (13 if depth is None else 15), "保存する数値")
    t0 = time.time()

    def analyze(
        f0: np.ndarray,
        f1: np.ndarray,
        d0: np.ndarray | None = None,
        d1: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        fwd = fn(f0, f1)
        bwd = fn(f1, f0)
        valid = _fb_valid(fwd, bwd)
        ok = _boundary_free(fwd, valid, a.gap)
        if depth is not None:
            d0 = depth(f0) if d0 is None else d0
            d1 = depth(f1) if d1 is None else d1
            s, conf = _depth_expansion(d0, d1, fwd, ok, sigma)
        else:
            s, conf = _expansion(fwd, sigma, ok)
        s1 = s ** (1.0 / a.gap)  # 1 コマあたり
        rate = s1 - 1.0
        # 3D の動き（距離 Z に対する割合）: ΔX/Z = (x'/s − x)/f、ΔZ/Z = 1/s − 1
        x1, y1 = gx + fwd[..., 0], gy + fwd[..., 1]
        sf = np.stack([(x1 / s - gx) / fx, (y1 / s - gy) / fx, 1 / s - 1], -1) / a.gap
        # 使える画素: 行き帰りが合い、まわりに拡大率を求められた画素が十分ある所
        vd = (valid & (conf > 0.3)).astype(np.float32)
        out_d = {"flow": fwd / a.gap, "s": s1, "rate": rate, "valid": vd, "sf": sf}
        if depth is not None:
            out_d["depth"] = d0
        return out_d

    def stats(res: dict[str, np.ndarray]) -> dict[str, Any]:
        v = res["valid"] > 0
        rate = res["rate"][v]
        if rate.size == 0:
            return {"valid": 0.0}
        near = rate > thr
        far = rate < -thr
        out_s: dict[str, Any] = {
            "valid": round(float(v.mean()), 4),
            "camera_rate": round(float(np.median(rate)) * 100, 4),
            "approaching": round(float(near.mean()), 4),
            "receding": round(float(far.mean()), 4),
            "max_approach": round(float(np.percentile(rate, 99.5)) * 100, 4),
            "max_recede": round(float(-np.percentile(rate, 0.5)) * 100, 4),
        }
        if near.any():
            ys, xs = np.nonzero(v)
            wgt = rate[near]
            out_s["approach_center"] = [
                round(float(np.average(xs[near], weights=wgt)) * k, 1),
                round(float(np.average(ys[near], weights=wgt)) * k, 1),
            ]
            top = float(np.percentile(rate[near], 90))
            out_s["ttc_s"] = (
                round((1 + top) / top / inp.fps, 3) if inp.fps and top > 0 else None
            )
        return out_s

    def label(st: dict[str, Any] | None) -> str:
        base = f"シーンフロー（{a.method}・{a.flow}）"
        if not st or "camera_rate" not in st:
            return base
        return f"{base} 近づく {st['approaching'] * 100:.0f}%・遠ざかる {st['receding'] * 100:.0f}%"

    if inp.pair is not None:
        res = analyze(inp.pair[0], inp.pair[1])
        st = stats(res)
        good = res["valid"] > 0
        max_rate = (
            a.max_rate / 100
            if a.max_rate
            else max(0.003, float(np.percentile(np.abs(res["rate"][good]), 98)))
            if good.any()
            else 0.01
        )
        p99 = float(
            np.percentile(np.hypot(res["flow"][..., 0], res["flow"][..., 1]), 99)
        )
        scale = None if a.no_arrows else 1.2 * step / max(1.0, p99)
        img = _render_scene(
            inp.pair[0], res["rate"], res["valid"], res["flow"], max_rate=max_rate, scale=scale,
            step=step, min_mag=0.5 / k, legend=not a.no_legend, label=label(st),
        )  # fmt: skip
        core.save_image(out, img)
        data = {
            **inp.head(),
            "method": a.method,
            "flow": a.flow,
            "gap": a.gap,
            "smooth": sigma,
            "max_rate": max_rate * 100,
            "stats": st,
        }
        if a.save_flow:
            _save_npz(
                npz_path,
                **{kk: vv[None] for kk, vv in _scene_arrays(res).items()},
                to_source=k,
                fov=a.fov,
                note=_SCENE_NPZ_NOTE,
            )
            data["npz"] = str(npz_path)
            core.log(f"保存: {npz_path}")
        core.write_json(json_path, data)
        core.log(
            f"近づく所 {st.get('approaching', 0) * 100:.1f}%・遠ざかる所 {st.get('receding', 0) * 100:.1f}%、画面全体 {st.get('camera_rate', 0):+.2f}%/コマ"
        )
        core.log(f"保存: {out}")
        core.log(f"保存: {json_path}")
        return

    # 色の濃さと矢印の長さの目安: いくつかのコマの組で測る（DIS・拡大率のみ。速い）
    auto_rate, p99 = _auto_rate(inp, sigma, a.gap)
    if a.max_rate:
        max_rate = a.max_rate / 100
    else:
        max_rate = auto_rate
        core.log(f"色が最も濃くなる速さ: {max_rate * 100:.2f}%/コマ（自動）")
    scale = None if a.no_arrows else 1.2 * step / max(1.0, p99)
    n_est = inp.count()
    rows: list[dict[str, Any]] = []
    saved: dict[str, list[np.ndarray]] = {}
    buf: deque[tuple[int, np.ndarray, np.ndarray | None]] = deque()
    last_res = None
    count = 0
    with inp.writer(out) as writer:

        def emit(
            i: int,
            frame: np.ndarray,
            res: dict[str, np.ndarray],
            st: dict[str, Any] | None,
        ) -> None:
            writer.write(
                _render_scene(
                    frame,
                    res["rate"],
                    res["valid"],
                    res["flow"],
                    max_rate=max_rate,
                    scale=scale,
                    step=step,
                    min_mag=0.5 / k,
                    legend=not a.no_legend,
                    label=label(st),
                )
            )

        for i, f in enumerate(inp.frames()):
            dcur = depth(f) if depth is not None else None
            buf.append((i, f, dcur))
            if len(buf) > a.gap:
                j, fj, dj = buf.popleft()
                res = analyze(fj, f, dj, dcur)
                st = stats(res)
                rows.append({"frame": inp.frame_no(j), "t": inp.t(j), **st})
                if a.save_flow:
                    for kk, vv in _scene_arrays(res).items():
                        saved.setdefault(kk, []).append(vv)
                emit(j, fj, res, st)
                last_res = res
            count = i + 1
            _progress(i, n_est)
        core.require(last_res is not None, f"動画のコマが {a.gap + 1} 個以上必要です")
        assert last_res is not None
        for _j, fj, _dj in buf:  # 最後の gap コマは、直前の結果をそのまま使う
            emit(_j, fj, last_res, None)
    near = [r_.get("approaching", 0) for r_ in rows]
    far = [r_.get("receding", 0) for r_ in rows]
    cam = [r_.get("camera_rate", 0) for r_ in rows]
    data = {
        **inp.head(),
        "frames": count,
        "method": a.method,
        "flow": a.flow,
        "gap": a.gap,
        "smooth": round(sigma, 2),
        "max_rate": round(max_rate * 100, 4),
        "threshold": a.threshold,
        "note": "rate・camera_rate は 1 コマあたりの「大きく見える割合」（%/コマ。+ は近づく、− は遠ざかる。距離が 1 コマで "
        "約その割合だけ縮む・伸びる）。camera_rate は画面全体の中央値（カメラが寄る・引く）、approaching・receding は"
        "しきい値を超えた画素の割合、ttc_s は近づく所がそのまま来た時にぶつかるまでの秒数の目安",
        "summary": {
            "mean_camera_rate": round(float(np.mean(cam)), 4),
            "mean_approaching": round(float(np.mean(near)), 4),
            "mean_receding": round(float(np.mean(far)), 4),
        },
        "per_frame": rows,
        "seconds": round(time.time() - t0, 2),
    }
    if a.save_flow:
        _save_npz(
            npz_path, **{kk: np.stack(vv) for kk, vv in saved.items()}, t=np.array([r_["t"] for r_ in rows]),
            to_source=k, fov=a.fov, note=_SCENE_NPZ_NOTE,
        )  # fmt: skip
        data["npz"] = str(npz_path)
        core.log(f"保存: {npz_path}")
    core.write_json(json_path, data)
    chart = _chart(
        out.with_suffix(".jpg"),
        [r_["t"] for r_ in rows],
        [
            ([v * 100 for v in near], "近づく所（%）", "#d1495b"),
            ([v * 100 for v in far], "遠ざかる所（%）", "#1f6fd1"),
            ([v * 100 for v in cam], "画面全体の拡大（×100 %/コマ）", "#555555"),
        ],
        title=f"シーンフロー（{a.method}・{a.flow}）",
        ylabel="%",
    )
    s = data["summary"]
    core.log(
        f"{count} コマ: 近づく所の平均 {s['mean_approaching'] * 100:.1f}%・遠ざかる所 {s['mean_receding'] * 100:.1f}%、"
        f"画面全体の拡大 {s['mean_camera_rate']:+.3f}%/コマ"
    )
    core.log(f"保存: {out}")
    core.log(f"保存: {chart}")
    core.log(f"保存: {json_path}")


def _auto_rate(
    inp: _Input, sigma: float, gap: int, pairs: int = 12
) -> tuple[float, float]:
    """いくつかのコマの組を DIS で測った目安: (拡大率の 98% 点（1 コマあたり）, 動きの 99% 点（px/コマ）)。"""
    n = inp.count()
    if n <= gap:
        return 0.01, 1.0
    want = {
        int(v) for v in np.linspace(gap, n - 1, min(pairs, n - gap)).round().astype(int)
    }
    hist: deque[np.ndarray] = deque(maxlen=gap + 1)
    vals, mags = [], []
    for i, f in enumerate(inp.frames()):
        hist.append(core.to_gray(f))
        if i in want and len(hist) == gap + 1:
            g0, g1 = hist[0], hist[-1]
            fwd = ai.optical_flow(g0, g1, "dis")
            bwd = ai.optical_flow(g1, g0, "dis")
            ok = _boundary_free(fwd, _fb_valid(fwd, bwd), gap)
            s, conf = _expansion(fwd, sigma, ok)
            good = conf > 0.3
            vals.append(np.abs(s ** (1.0 / gap) - 1)[good][::7])
            mags.append(np.hypot(fwd[..., 0], fwd[..., 1])[::4, ::4].ravel() / gap)
        if i >= max(want):
            break
    v = np.concatenate(vals) if vals else np.zeros(0)
    rate = float(max(0.003, np.percentile(v, 98))) if v.size else 0.01
    p99 = float(np.percentile(np.concatenate(mags), 99)) if mags else 1.0
    return rate, p99
