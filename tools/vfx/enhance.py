"""画質を上げる（解像度・コマ・鮮明さ・粒子）。

担当するテクニック:
   26 超解像（upscale）
   27 フレーム補間（interpolate）
   29 ブレ除去（deblur）
   39 アンシャープマスク（sharpen）
   50 フィルムグレイン（grain）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
