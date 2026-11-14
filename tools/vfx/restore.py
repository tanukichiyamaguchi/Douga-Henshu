"""画質を上げる（ノイズ・ちらつき・縞・ゴミ）。

担当するテクニック:
   28 ノイズ除去（denoise）
   30 圧縮ノイズ除去（deblock）
   31 時間方向のノイズ除去（temporal-denoise）
   36 フリッカー除去（deflicker）
   41 デインターレース（deinterlace）
   45 バンディング除去（deband）
   46 ゴミ・傷の除去（dust-removal）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
