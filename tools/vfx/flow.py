"""動きの解析（オプティカルフロー系）。

担当するテクニック:
    1 オプティカルフロー（optical-flow）
   13 動き補償（motion-compensate）
   16 フレーム間差分（frame-diff）
   23 シーンフロー（scene-flow）
   24 ブロックマッチング（block-match）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
