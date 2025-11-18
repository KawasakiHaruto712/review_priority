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
    
    def save_component_data(self, component: str, data: List[Dict[str, Any]]):
        """
        コンポーネントの変更データを保存
        
        Args:
            component: コンポーネント名
            data: 変更データのリスト
        """
        # コンポーネントディレクトリの作成
        component_dir = self.output_dir / component / "changes"
        component_dir.mkdir(parents=True, exist_ok=True)
        
        # 個別の変更データを保存
        for change in data:
            change_number = change.get("change_number")
            if change_number:
                change_file = component_dir / f"change_{change_number}.json"
                self.save_json(change_file, change)
        
        # サマリーデータを保存
        self._save_summary(component, data)
        
        logger.info(f"{component}: {len(data)}件の変更データを保存しました")
    
    def _save_summary(self, component: str, changes: List[Dict[str, Any]]):
        """
        サマリーデータを保存
        
        Args:
            component: コンポーネント名
            changes: 変更データのリスト
        """
        component_dir = self.output_dir / component / "changes"
        
        # JSONサマリー
        summary = {
            "total_changes": len(changes),
        }
        summary_file = component_dir / "summary.json"
        self.save_json(summary_file, summary)
        
        # CSVサマリー
        if changes:
            flat_data = []
            for change in changes:
                flat_change = {
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
                        len(comments) 
                        for comments in change.get("comments", {}).values()
                    ) if change.get("comments") else 0,
                    "reviewers_count": len(
                        change.get("reviewers", {}).get("REVIEWER", [])
                    ) if change.get("reviewers") else 0,
                }
                flat_data.append(flat_change)
            
            csv_file = component_dir / "changes.csv"
            self.save_csv(csv_file, flat_data)
