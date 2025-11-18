"""
CollectorConfigの簡略化されたテスト
"""
import pytest
from src.collectors.config.collector_config import CollectorConfig


class TestCollectorConfig:
    """CollectorConfigのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        config = CollectorConfig()
        assert config.config is not None
        assert isinstance(config.config, dict)
    
    def test_get_endpoint_config(self):
        """エンドポイント設定取得のテスト"""
        config = CollectorConfig()
        
        changes_config = config.get_endpoint_config("changes")
        assert changes_config is not None
        assert isinstance(changes_config, dict)
    
    def test_get_collection_config(self):
        """収集設定取得のテスト"""
        config = CollectorConfig()
        
        collection_config = config.get_collection_config()
        assert collection_config is not None
        assert "batch_size" in collection_config
        assert isinstance(collection_config["batch_size"], int)
    
    def test_get_retry_config(self):
        """リトライ設定取得のテスト"""
        config = CollectorConfig()
        
        retry_config = config.get_retry_config()
        assert retry_config is not None
        assert "max_retries" in retry_config
        assert "base_delay" in retry_config
        assert "max_delay" in retry_config
    
    def test_get_storage_config(self):
        """ストレージ設定取得のテスト"""
        config = CollectorConfig()
        
        storage_config = config.get_storage_config()
        assert storage_config is not None
        assert "output_dir" in storage_config
        assert "formats" in storage_config
