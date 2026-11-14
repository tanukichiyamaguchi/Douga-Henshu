"""画質を上げる（明るさと色）。

担当するテクニック:
   32 逆トーンマッピング（inverse-tonemap）
   33 トーンマッピング（tonemap）
   35 ホワイトバランス（white-balance）
   40 低照度補正（low-light）
   47 周辺減光の補正（devignette）
   48 ガンマ補正（gamma）
   49 ヒストグラム平坦化（equalize）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
