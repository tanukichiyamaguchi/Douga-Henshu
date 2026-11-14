"""動きの解析（揺れの補正）。

担当するテクニック:
    3 手ぶれ補正（stabilize）
   11 ローリングシャッター補正（rolling-shutter）
   22 カメラパスの平滑化（smooth-path）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
