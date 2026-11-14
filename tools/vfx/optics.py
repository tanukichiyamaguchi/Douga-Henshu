"""画質を上げる（レンズと光学・色付け）。

担当するテクニック:
   37 モアレ除去（demoire）
   38 カラー化（colorize）
   42 色収差補正（fix-chroma）
   43 レンズ歪み補正（undistort）
   44 かすみ除去（dehaze）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
