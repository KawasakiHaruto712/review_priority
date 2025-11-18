"""
CommitStorageのテスト
"""
import pytest
import json
import pandas as pd
from pathlib import Path
from src.collectors.storage.commit_storage import CommitStorage


class TestCommitStorage:
    """CommitStorageクラスのテスト"""
    
    def test_initialization(self, tmp_path):
        """初期化のテスト"""
        storage = CommitStorage(base_dir=str(tmp_path))
        
        assert storage.base_dir == tmp_path
        assert tmp_path.exists()
    
    def test_save_json(self, tmp_path):
        """JSON形式での保存のテスト"""
        storage = CommitStorage(base_dir=str(tmp_path))
        
        data = [
            {
                "commit": "abc123",
                "author": "John Doe",
                "subject": "Fix bug",
                "date": "2024-01-01"
            },
            {
                "commit": "def456",
                "author": "Jane Smith",
                "subject": "Add feature",
                "date": "2024-01-02"
            }
        ]
        
        storage.save_json(data, "test_commits")
        
        json_file = tmp_path / "test_commits.json"
        assert json_file.exists()
        
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data) == 2
        assert loaded_data[0]["commit"] == "abc123"
    
    def test_save_csv(self, tmp_path):
        """CSV形式での保存のテスト"""
        storage = CommitStorage(base_dir=str(tmp_path))
        
        data = [
            {
                "commit": "abc123",
                "author": "John Doe",
                "subject": "Fix bug"
            },
            {
                "commit": "def456",
                "author": "Jane Smith",
                "subject": "Add feature"
            }
        ]
        
        storage.save_csv(data, "test_commits")
        
        csv_file = tmp_path / "test_commits.csv"
        assert csv_file.exists()
        
        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert df.iloc[0]["commit"] == "abc123"
    
    def test_save_summary(self, tmp_path):
        """サマリー保存のテスト"""
        storage = CommitStorage(base_dir=str(tmp_path))
        
        summary = {
            "total_commits": 50,
            "unique_authors": 10,
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-12-31"
            }
        }
        
        storage.save_summary(summary, "commit_summary")
        
        summary_file = tmp_path / "commit_summary.json"
        assert summary_file.exists()
        
        with open(summary_file, 'r', encoding='utf-8') as f:
            loaded_summary = json.load(f)
        
        assert loaded_summary["total_commits"] == 50
        assert loaded_summary["unique_authors"] == 10
    
    def test_save_commit_with_parents(self, tmp_path):
        """親コミット情報を含むデータの保存のテスト"""
        storage = CommitStorage(base_dir=str(tmp_path))
        
        data = [
            {
                "commit": "abc123",
                "subject": "Merge commit",
                "parents": [
                    {"commit": "parent1"},
                    {"commit": "parent2"}
                ]
            }
        ]
        
        storage.save_json(data, "commits_with_parents")
        
        json_file = tmp_path / "commits_with_parents.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data[0]["parents"]) == 2
        assert loaded_data[0]["parents"][0]["commit"] == "parent1"


class TestCommitStorageIntegration:
    """CommitStorageの統合テスト"""
    
    def test_complete_save_workflow(self, tmp_path):
        """完全な保存ワークフローのテスト"""
        storage = CommitStorage(base_dir=str(tmp_path))
        
        # コミットデータ
        commits = [
            {
                "commit": f"commit{i}",
                "author": f"Author {i % 3}",
                "subject": f"Commit {i}",
                "date": "2024-01-01"
            }
            for i in range(10)
        ]
        
        # JSON保存
        storage.save_json(commits, "all_commits")
        
        # CSV保存
        storage.save_csv(commits, "all_commits")
        
        # サマリー保存
        summary = {
            "total": len(commits),
            "unique_authors": len(set(c["author"] for c in commits))
        }
        storage.save_summary(summary, "commits_summary")
        
        # 全てのファイルが作成されたことを確認
        assert (tmp_path / "all_commits.json").exists()
        assert (tmp_path / "all_commits.csv").exists()
        assert (tmp_path / "commits_summary.json").exists()
