"""切り抜き・合成（切り分け）。

担当するテクニック:
   51 セグメンテーション（segment）
   53 ロトスコープ（rotoscope）
   56 マッティング（matte）
   60 マスクトラッキング（mask-track）
   61 インスタンス分割（instances）
   73 パノプティック分割（panoptic）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
