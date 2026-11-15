"""切り抜き・合成（なじませる合成）。

担当するテクニック:
   63 リライティング（relight）
   64 色合わせ（color-match）
   65 接地影（contact-shadow）
   70 スクリーン合成（screen-blend）
   71 ライトラップ（light-wrap）
   72 モーションブラー付加（motion-blur）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
