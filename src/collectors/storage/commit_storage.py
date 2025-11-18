"""
コミットデータストレージモジュール

コミットデータの保存機能を提供します。
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from src.collectors.storage.base_storage import BaseStorage

logger = logging.getLogger(__name__)


class CommitStorage(BaseStorage):
    """コミットデータストレージクラス"""
    
    def save_component_data(self, component: str, data: List[Dict[str, Any]]):
        """
        コンポーネントのコミットデータを保存
        
        Args:
            component: コンポーネント名
            data: コミットデータのリスト
        """
        # コンポーネントディレクトリの作成
        component_dir = self.output_dir / component / "commits"
        component_dir.mkdir(parents=True, exist_ok=True)
        
        # 個別のコミットデータを保存
        for commit in data:
            change_number = commit.get("change_number")
            revision_id = commit.get("revision_id", "")
            if change_number:
                short_revision_id = revision_id[:8] if len(revision_id) > 8 else revision_id
                commit_file = component_dir / f"commit_{change_number}_{short_revision_id}.json"
                self.save_json(commit_file, commit)
        
        # サマリーデータを保存
        self._save_summary(component, data)
        
        logger.info(f"{component}: {len(data)}件のコミットデータを保存しました")
    
    def _save_summary(self, component: str, commits: List[Dict[str, Any]]):
        """
        サマリーデータを保存
        
        Args:
            component: コンポーネント名
            commits: コミットデータのリスト
        """
        component_dir = self.output_dir / component / "commits"
        
        # JSONサマリー
        summary = {
            "total_commits": len(commits),
        }
        summary_file = component_dir / "summary.json"
        self.save_json(summary_file, summary)
        
        # CSVサマリー
        if commits:
            flat_commits = []
            for commit in commits:
                flat_commit = {
                    "change_id": commit.get("change_id", ""),
                    "change_number": commit.get("change_number", ""),
                    "revision_id": commit.get("revision_id", ""),
                    "commit_message": commit.get("commit", {}).get("message", ""),
                    "author_name": commit.get("commit", {}).get("author", {}).get("name", ""),
                    "author_email": commit.get("commit", {}).get("author", {}).get("email", ""),
                    "commit_time": commit.get("commit", {}).get("committer", {}).get("date", ""),
                    "files_changed": len(commit.get("files", [])),
                    "lines_inserted": sum(
                        f.get("lines_inserted", 0) 
                        for f in commit.get("file_changes", {}).values()
                    ),
                    "lines_deleted": sum(
                        f.get("lines_deleted", 0) 
                        for f in commit.get("file_changes", {}).values()
                    ),
                    "parent_count": len(commit.get("commit", {}).get("parents", [])),
                }
                flat_commits.append(flat_commit)
            
            csv_file = component_dir / "commits.csv"
            self.save_csv(csv_file, flat_commits)
