"""
ストレージクラスの簡略化されたテスト
"""
import pytest
import json
from pathlib import Path
from src.collectors.storage.base_storage import BaseStorage
from src.collectors.storage.change_storage import ChangeStorage
from src.collectors.storage.commit_storage import CommitStorage


class TestBaseStorage:
    """BaseStorageのテスト"""
    
    def test_cannot_instantiate_abstract_class(self):
        """抽象クラスを直接インスタンス化できないことのテスト"""
        with pytest.raises(TypeError):
            BaseStorage(output_dir="/tmp/test")


class TestChangeStorage:
    """ChangeStorageのテスト"""
    
    def test_initialization(self, tmp_path):
        """初期化のテスト"""
        storage = ChangeStorage(output_dir=str(tmp_path))
        assert storage.output_dir == tmp_path
    
    def test_save_component_data(self, tmp_path):
        """コンポーネントデータ保存のテスト"""
        storage = ChangeStorage(output_dir=str(tmp_path))
        
        data = [
            {"change_number": "123", "id": "change1", "project": "openstack/nova"},
            {"change_number": "456", "id": "change2", "project": "openstack/nova"}
        ]
        
        storage.save_component_data(
            component="nova",
            data=data
        )
        
        # ファイルが作成されたことを確認
        component_dir = tmp_path / "nova" / "changes"
        assert component_dir.exists()
    
    def test_save_empty_data(self, tmp_path):
        """空データの保存のテスト"""
        storage = ChangeStorage(output_dir=str(tmp_path))
        
        # 空データでもエラーにならない
        storage.save_component_data(
            component="nova",
            data=[]
        )


class TestCommitStorage:
    """CommitStorageのテスト"""
    
    def test_initialization(self, tmp_path):
        """初期化のテスト"""
        storage = CommitStorage(output_dir=str(tmp_path))
        assert storage.output_dir == tmp_path
    
    def test_save_component_data(self, tmp_path):
        """コンポーネントデータ保存のテスト"""
        storage = CommitStorage(output_dir=str(tmp_path))
        
        data = [
            {
                "change_number": "123",
                "revision_id": "abc123",
                "commit": {
                    "message": "Test commit",
                    "author": {"name": "John", "email": "john@example.com"},
                    "committer": {"date": "2024-01-01"}
                }
            }
        ]
        
        # エラーなく保存できることを確認
        storage.save_component_data(
            component="nova",
            data=data
        )
