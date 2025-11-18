"""
ChangeCollectorの簡略化されたテスト
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.collectors.change_collector import ChangeCollector


class TestChangeCollector:
    """ChangeCollectorのテスト"""
    
    @patch.dict('os.environ', {
        'GERRIT_USERNAME': 'test_user',
        'GERRIT_PASSWORD': 'test_pass'
    })
    def test_initialization_from_env(self):
        """環境変数からの初期化のテスト"""
        collector = ChangeCollector()
        
        assert collector.username == 'test_user'
        assert collector.password == 'test_pass'
    
    def test_initialization_with_params(self):
        """パラメータ指定での初期化のテスト"""
        collector = ChangeCollector(
            username="direct_user",
            password="direct_pass"
        )
        
        assert collector.username == "direct_user"
        assert collector.password == "direct_pass"
    
    def test_session_creation(self):
        """セッション作成のテスト"""
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        assert collector.session is not None
        assert "Authorization" in collector.session.headers
        assert "Basic" in collector.session.headers["Authorization"]

