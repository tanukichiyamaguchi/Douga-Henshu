"""動きの解析（カメラの動きと立体）。

担当するテクニック:
    5 カメラトラッキング（camera-track）
   14 SfM（sfm）
   20 ジャイロ補正（gyro-check）
   21 水平補正（horizon）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
