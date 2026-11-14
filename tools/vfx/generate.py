"""切り抜き・合成（奥行き・穴埋め・描き足し）。

担当するテクニック:
   54 インペインティング（inpaint）
   55 深度推定（depth）
   58 背景ぼかし（bg-blur）
   59 アウトペインティング（outpaint）
   68 2.5Dパララックス（parallax）
   69 空の置き換え（sky-replace）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
