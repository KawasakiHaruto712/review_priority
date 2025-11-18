"""
CollectorConfigのテスト
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import yaml
from src.collectors.config.collector_config import CollectorConfig


class TestCollectorConfig:
    """CollectorConfigクラスのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        config = CollectorConfig()
        
        assert config.config_data is not None
        assert isinstance(config.config_data, dict)
    
    def test_get_endpoint_config(self):
        """エンドポイント設定取得のテスト"""
        config = CollectorConfig()
        
        changes_config = config.get_endpoint_config("changes")
        
        assert changes_config is not None
        assert "path" in changes_config
        assert "method" in changes_config
        assert changes_config["method"] == "GET"
    
    def test_get_endpoint_config_all_endpoints(self):
        """全エンドポイントの設定取得のテスト"""
        config = CollectorConfig()
        
        endpoint_names = [
            "changes",
            "change_detail",
            "comments",
            "reviewers",
            "file_content",
            "file_diff",
            "commit",
            "commit_parents"
        ]
        
        for name in endpoint_names:
            endpoint_config = config.get_endpoint_config(name)
            assert endpoint_config is not None
            assert "path" in endpoint_config
            assert "method" in endpoint_config
    
    def test_get_endpoint_config_not_found(self):
        """存在しないエンドポイント設定のテスト"""
        config = CollectorConfig()
        
        with pytest.raises(KeyError):
            config.get_endpoint_config("non_existent_endpoint")
    
    def test_get_collection_config(self):
        """収集設定取得のテスト"""
        config = CollectorConfig()
        
        collection_config = config.get_collection_config()
        
        assert collection_config is not None
        assert "default_limit" in collection_config
        assert "batch_size" in collection_config
        assert isinstance(collection_config["default_limit"], int)
    
    def test_get_retry_config(self):
        """リトライ設定取得のテスト"""
        config = CollectorConfig()
        
        retry_config = config.get_retry_config()
        
        assert retry_config is not None
        assert "max_retries" in retry_config
        assert "initial_wait" in retry_config
        assert "max_wait" in retry_config
        assert "backoff_factor" in retry_config
        assert "jitter" in retry_config
    
    def test_get_storage_config(self):
        """ストレージ設定取得のテスト"""
        config = CollectorConfig()
        
        storage_config = config.get_storage_config()
        
        assert storage_config is not None
        assert "base_dir" in storage_config
        assert "formats" in storage_config
        assert isinstance(storage_config["formats"], list)
    
    def test_retry_config_values(self):
        """リトライ設定の値のテスト"""
        config = CollectorConfig()
        
        retry_config = config.get_retry_config()
        
        # デフォルト値の確認
        assert retry_config["max_retries"] == 3
        assert retry_config["initial_wait"] == 30
        assert retry_config["max_wait"] == 3840
        assert retry_config["backoff_factor"] == 2.0
        assert retry_config["jitter"] is True
    
    def test_storage_config_values(self):
        """ストレージ設定の値のテスト"""
        config = CollectorConfig()
        
        storage_config = config.get_storage_config()
        
        # デフォルト値の確認
        assert storage_config["base_dir"] == "data/openstack_collected"
        assert "json" in storage_config["formats"]
        assert "csv" in storage_config["formats"]
    
    def test_collection_config_values(self):
        """収集設定の値のテスト"""
        config = CollectorConfig()
        
        collection_config = config.get_collection_config()
        
        # デフォルト値の確認
        assert collection_config["default_limit"] == 500
        assert collection_config["batch_size"] == 100
    
    @patch("builtins.open", new_callable=mock_open, read_data="""
endpoints:
  test_endpoint:
    path: "/test"
    method: "GET"
    description: "Test endpoint"

collection:
  default_limit: 1000
  batch_size: 200

retry:
  max_retries: 5
  initial_wait: 10
  max_wait: 1000
  backoff_factor: 1.5
  jitter: false

storage:
  base_dir: "test/data"
  formats:
    - json
""")
    def test_custom_config_loading(self, mock_file):
        """カスタム設定の読み込みのテスト"""
        config = CollectorConfig()
        
        # エンドポイント設定
        test_endpoint = config.get_endpoint_config("test_endpoint")
        assert test_endpoint["path"] == "/test"
        
        # 収集設定
        collection = config.get_collection_config()
        assert collection["default_limit"] == 1000
        assert collection["batch_size"] == 200
        
        # リトライ設定
        retry = config.get_retry_config()
        assert retry["max_retries"] == 5
        assert retry["initial_wait"] == 10
        
        # ストレージ設定
        storage = config.get_storage_config()
        assert storage["base_dir"] == "test/data"


class TestCollectorConfigIntegration:
    """CollectorConfigの統合テスト"""
    
    def test_complete_configuration_flow(self):
        """完全な設定フローのテスト"""
        config = CollectorConfig()
        
        # 全ての設定が取得できることを確認
        endpoints = [
            "changes", "change_detail", "comments", "reviewers",
            "file_content", "file_diff", "commit", "commit_parents"
        ]
        
        for endpoint_name in endpoints:
            endpoint_config = config.get_endpoint_config(endpoint_name)
            assert endpoint_config is not None
            assert "path" in endpoint_config
        
        collection_config = config.get_collection_config()
        assert collection_config is not None
        
        retry_config = config.get_retry_config()
        assert retry_config is not None
        
        storage_config = config.get_storage_config()
        assert storage_config is not None
    
    def test_config_consistency(self):
        """設定の整合性のテスト"""
        config = CollectorConfig()
        
        # リトライ設定の整合性
        retry = config.get_retry_config()
        assert retry["initial_wait"] <= retry["max_wait"]
        assert retry["max_retries"] >= 0
        assert retry["backoff_factor"] > 0
        
        # 収集設定の整合性
        collection = config.get_collection_config()
        assert collection["default_limit"] > 0
        assert collection["batch_size"] > 0
        assert collection["batch_size"] <= collection["default_limit"]
