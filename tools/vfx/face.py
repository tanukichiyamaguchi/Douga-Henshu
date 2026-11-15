"""顔（ランドマーク・視線・復元・検出・表情・モザイク）。

担当するテクニック:
    8 顔ランドマーク（face-landmarks）
   17 視線推定（gaze）
   34 顔の復元（face-restore）
   77 顔検出（faces）
   81 表情認識（expressions）
   82 自動モザイク（auto-mosaic）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
