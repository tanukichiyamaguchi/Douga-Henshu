"""自動で見つける（映像の内容の理解）。

担当するテクニック:
   18 動作認識（action）
   80 ハイライト検出（highlights）
   84 画像キャプション（caption）
   86 自然言語で場面検索（search）
   87 異常検知（anomalies）
   91 シーン分類（scene-classify）

各テクニックは registry.command で登録し、確かめた結果を NOTES に書く（tools/vfx/DEVELOPING.md）。
"""

from __future__ import annotations

# 確かめた結果（番号 → status / how / measured / limits / tested）
NOTES: dict[int, dict[str, str]] = {}
