"""
BaseStorageの抽象基底クラスのテスト
"""
import pytest
from pathlib import Path
from src.collectors.storage.base_storage import BaseStorage


class ConcreteStorage(BaseStorage):
    """テスト用の具象クラス"""
    
    def __init__(self, base_dir: str):
        super().__init__(base_dir)
    
    def save(self, data: dict, filename: str) -> None:
        """テスト用の保存メソッド"""
        pass


class TestBaseStorage:
    """BaseStorageクラスのテスト"""
    
    def test_initialization(self, tmp_path):
        """初期化のテスト"""
        storage = ConcreteStorage(base_dir=str(tmp_path))
        
        assert storage.base_dir == tmp_path
        assert tmp_path.exists()
    
    def test_ensure_directory_exists(self, tmp_path):
        """ディレクトリ作成のテスト"""
        storage = ConcreteStorage(base_dir=str(tmp_path))
        
        sub_dir = tmp_path / "subdir" / "nested"
        storage.ensure_directory_exists(sub_dir)
        
        assert sub_dir.exists()
        assert sub_dir.is_dir()
    
    def test_cannot_instantiate_base_class(self):
        """BaseStorageを直接インスタンス化できないことのテスト"""
        with pytest.raises(TypeError):
            BaseStorage(base_dir="/tmp/test")
    
    def test_base_dir_creation(self, tmp_path):
        """ベースディレクトリの自動作成のテスト"""
        new_dir = tmp_path / "new_storage"
        assert not new_dir.exists()
        
        storage = ConcreteStorage(base_dir=str(new_dir))
        
        assert new_dir.exists()
        assert storage.base_dir == new_dir
