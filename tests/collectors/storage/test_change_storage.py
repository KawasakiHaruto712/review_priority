"""
ChangeStorageのテスト
"""
import pytest
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.collectors.storage.change_storage import ChangeStorage


class TestChangeStorage:
    """ChangeStorageクラスのテスト"""
    
    def test_initialization(self, tmp_path):
        """初期化のテスト"""
        storage = ChangeStorage(base_dir=str(tmp_path))
        
        assert storage.base_dir == tmp_path
        assert tmp_path.exists()
    
    def test_save_json(self, tmp_path):
        """JSON形式での保存のテスト"""
        storage = ChangeStorage(base_dir=str(tmp_path))
        
        data = [
            {
                "id": "change1",
                "project": "openstack/nova",
                "subject": "Fix bug",
                "status": "MERGED"
            },
            {
                "id": "change2",
                "project": "openstack/nova",
                "subject": "Add feature",
                "status": "NEW"
            }
        ]
        
        storage.save_json(data, "test_changes")
        
        # ファイルが作成されたか確認
        json_file = tmp_path / "test_changes.json"
        assert json_file.exists()
        
        # 内容を確認
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data) == 2
        assert loaded_data[0]["id"] == "change1"
        assert loaded_data[1]["id"] == "change2"
    
    def test_save_csv(self, tmp_path):
        """CSV形式での保存のテスト"""
        storage = ChangeStorage(base_dir=str(tmp_path))
        
        data = [
            {
                "id": "change1",
                "project": "openstack/nova",
                "subject": "Fix bug",
                "status": "MERGED"
            },
            {
                "id": "change2",
                "project": "openstack/nova",
                "subject": "Add feature",
                "status": "NEW"
            }
        ]
        
        storage.save_csv(data, "test_changes")
        
        # ファイルが作成されたか確認
        csv_file = tmp_path / "test_changes.csv"
        assert csv_file.exists()
        
        # 内容を確認
        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert df.iloc[0]["id"] == "change1"
        assert df.iloc[1]["id"] == "change2"
    
    def test_save_summary(self, tmp_path):
        """サマリー保存のテスト"""
        storage = ChangeStorage(base_dir=str(tmp_path))
        
        summary = {
            "total_changes": 100,
            "merged": 60,
            "new": 25,
            "abandoned": 15,
            "collection_date": "2024-01-15"
        }
        
        storage.save_summary(summary, "test_summary")
        
        # ファイルが作成されたか確認
        summary_file = tmp_path / "test_summary.json"
        assert summary_file.exists()
        
        # 内容を確認
        with open(summary_file, 'r', encoding='utf-8') as f:
            loaded_summary = json.load(f)
        
        assert loaded_summary["total_changes"] == 100
        assert loaded_summary["merged"] == 60
    
    def test_save_empty_data(self, tmp_path):
        """空データの保存のテスト"""
        storage = ChangeStorage(base_dir=str(tmp_path))
        
        # 空のリスト
        storage.save_json([], "empty_changes")
        
        json_file = tmp_path / "empty_changes.json"
        assert json_file.exists()
        
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == []
    
    def test_save_nested_data(self, tmp_path):
        """ネストされたデータの保存のテスト"""
        storage = ChangeStorage(base_dir=str(tmp_path))
        
        data = [
            {
                "id": "change1",
                "project": "openstack/nova",
                "owner": {
                    "name": "John Doe",
                    "email": "john@example.com"
                },
                "reviewers": [
                    {"name": "Alice", "score": 2},
                    {"name": "Bob", "score": -1}
                ]
            }
        ]
        
        storage.save_json(data, "nested_changes")
        
        json_file = tmp_path / "nested_changes.json"
        assert json_file.exists()
        
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data[0]["owner"]["name"] == "John Doe"
        assert len(loaded_data[0]["reviewers"]) == 2
    
    def test_save_with_unicode(self, tmp_path):
        """Unicode文字を含むデータの保存のテスト"""
        storage = ChangeStorage(base_dir=str(tmp_path))
        
        data = [
            {
                "id": "change1",
                "subject": "バグ修正: 日本語のテスト",
                "description": "これは日本語の説明です 🚀"
            }
        ]
        
        storage.save_json(data, "unicode_changes")
        
        json_file = tmp_path / "unicode_changes.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data[0]["subject"] == "バグ修正: 日本語のテスト"
        assert loaded_data[0]["description"] == "これは日本語の説明です 🚀"
    
    def test_save_overwrites_existing_file(self, tmp_path):
        """既存ファイルの上書きのテスト"""
        storage = ChangeStorage(base_dir=str(tmp_path))
        
        # 最初の保存
        data1 = [{"id": "change1"}]
        storage.save_json(data1, "test_overwrite")
        
        # 上書き保存
        data2 = [{"id": "change2"}, {"id": "change3"}]
        storage.save_json(data2, "test_overwrite")
        
        # 上書きされたことを確認
        json_file = tmp_path / "test_overwrite.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data) == 2
        assert loaded_data[0]["id"] == "change2"


class TestChangeStorageIntegration:
    """ChangeStorageの統合テスト"""
    
    def test_complete_save_workflow(self, tmp_path):
        """完全な保存ワークフローのテスト"""
        storage = ChangeStorage(base_dir=str(tmp_path))
        
        # 変更データ
        changes = [
            {
                "id": f"change{i}",
                "project": "openstack/nova",
                "subject": f"Change {i}",
                "status": "MERGED" if i % 2 == 0 else "NEW"
            }
            for i in range(10)
        ]
        
        # JSON保存
        storage.save_json(changes, "all_changes")
        
        # CSV保存
        storage.save_csv(changes, "all_changes")
        
        # サマリー保存
        summary = {
            "total": len(changes),
            "merged": len([c for c in changes if c["status"] == "MERGED"]),
            "new": len([c for c in changes if c["status"] == "NEW"])
        }
        storage.save_summary(summary, "changes_summary")
        
        # 全てのファイルが作成されたことを確認
        assert (tmp_path / "all_changes.json").exists()
        assert (tmp_path / "all_changes.csv").exists()
        assert (tmp_path / "changes_summary.json").exists()
