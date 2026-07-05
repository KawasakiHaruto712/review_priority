"""
変更データストレージモジュール

変更(Change)データの保存機能を提供します。
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from src.collectors.storage.base_storage import BaseStorage

logger = logging.getLogger(__name__)


class ChangeStorage(BaseStorage):
    """変更データストレージクラス"""
    
    def save_changes(self, component: str, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """個別の change_<num>.json を書き出す（サマリーは書かない）。

        途中保存（チャンクごと）に使う。サマリー用の flat 行リストを返すので、
        呼び出し側で蓄積して最後に write_summary へ渡す（全チャンク分をまとめて集計するため）。
        """
        component_dir = self.output_dir / component / "changes"
        component_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        for change in data:
            change_number = change.get("change_number")
            if change_number:
                self.save_json(component_dir / f"change_{change_number}.json", change)
            rows.append(self._flat_row(change))
        return rows

    def write_summary(self, component: str, rows: List[Dict[str, Any]]):
        """蓄積した flat 行から summary.json / changes.csv を書き出す。"""
        component_dir = self.output_dir / component / "changes"
        component_dir.mkdir(parents=True, exist_ok=True)
        self.save_json(component_dir / "summary.json", {"total_changes": len(rows)})
        if rows:
            self.save_csv(component_dir / "changes.csv", rows)

    def save_component_data(self, component: str, data: List[Dict[str, Any]]):
        """一括保存（個別ファイル＋サマリー）。従来互換のショートカット。"""
        rows = self.save_changes(component, data)
        self.write_summary(component, rows)
        logger.info(f"{component}: {len(data)}件の変更データを保存しました")

    def _flat_row(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """CSV サマリー1行分の平坦化辞書（軽量。全チャンク蓄積してもメモリを圧迫しない）。"""
        return {
            "change_id": change.get("change_id", ""),
            "change_number": change.get("change_number", change.get("_number", "")),
            "project": change.get("project", ""),
            "branch": change.get("branch", ""),
            "subject": change.get("subject", ""),
            "created": change.get("created", ""),
            "updated": change.get("updated", ""),
            "merged": change.get("submitted", ""),  # submittedフィールドがマージ日時
            "status": change.get("status", ""),
            "owner_name": change.get("owner", {}).get("name", ""),
            "owner_email": change.get("owner", {}).get("email", ""),
            "insertions": change.get("insertions", 0),
            "deletions": change.get("deletions", 0),
            "comments_count": sum(
                len(comments) for comments in change.get("comments", {}).values()
            ) if change.get("comments") else 0,
            "reviewers_count": len(
                change.get("reviewers", {}).get("REVIEWER", [])
            ) if change.get("reviewers") else 0,
        }
