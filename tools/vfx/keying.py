"""切り抜き・合成（キーとマット）。

担当するテクニック:
   52 クロマキー（chroma-key）
   57 スピル除去（despill）
   62 ガベージマット（garbage-matte）
   66 フェザー（feather）
   67 ルミナンスキー（luma-key）
   74 差分キー（diff-key）
   75 クリーンプレート（clean-plate）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
