"""動きの解析（追跡・はめ込み）。

担当するテクニック:
    2 モーショントラッキング（track）
    6 平面トラッキング（planar-track）
    7 特徴点追跡（feature-track）
   25 メッシュワープ（mesh-warp）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
