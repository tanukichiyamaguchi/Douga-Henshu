"""動きの解析と自動検出（体・手）。

担当するテクニック:
    4 姿勢推定（pose）
    9 ハンドトラッキング（hand-track）
   10 モーションキャプチャ（mocap）
   19 3D姿勢推定（pose3d）
  100 ジェスチャー認識（gestures）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
