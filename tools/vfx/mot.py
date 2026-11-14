"""動きの解析と自動検出（複数の人の検出・追跡）。

担当するテクニック:
   12 背景差分（bg-subtract）
   15 複数物体追跡（multi-track）
   85 人物の再識別（reid）
   95 人数カウント（count-people）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
