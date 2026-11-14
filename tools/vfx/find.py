"""自動で見つける（画面の中の物・文字・色）。

担当するテクニック:
   78 物体検出（objects）
   79 文字認識（ocr）
   90 ロゴ検出（logos）
   94 顕著性マップ（saliency）
   96 テキスト検出（text-detect）
   97 代表色の抽出（palette）
   98 QRコード検出（qr）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
