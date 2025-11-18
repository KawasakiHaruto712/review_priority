"""
RetryHandlerの簡略化されたテスト
"""
import pytest
from unittest.mock import Mock, patch
from requests.exceptions import ConnectionError, Timeout
from src.collectors.base.retry_handler import RetryConfig, retry_with_backoff


class TestRetryConfig:
    """RetryConfigクラスのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        config = RetryConfig()
        
        assert config.max_retries == 10
        assert config.base_delay == 30.0
        assert config.max_delay == 3840.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True
    
    def test_custom_values(self):
        """カスタム値での初期化"""
        config = RetryConfig(
            max_retries=5,
            base_delay=10.0,
            max_delay=100.0,
            backoff_factor=1.5,
            jitter=False
        )
        
        assert config.max_retries == 5
        assert config.base_delay == 10.0
        assert config.max_delay == 100.0
        assert config.backoff_factor == 1.5
        assert config.jitter is False
    
    def test_get_delay_without_jitter(self):
        """ジッターなしの遅延計算"""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter=False
        )
        
        # 1回目: 1.0
        assert config.get_delay(0) == 1.0
        # 2回目: 2.0
        assert config.get_delay(1) == 2.0
        # 3回目: 4.0
        assert config.get_delay(2) == 4.0
        # 4回目: 8.0
        assert config.get_delay(3) == 8.0
        # 5回目: 10.0 (上限)
        assert config.get_delay(4) == 10.0


class TestRetryWithBackoff:
    """retry_with_backoffデコレータのテスト"""
    
    @patch('time.sleep')
    def test_success_on_first_try(self, mock_sleep):
        """初回で成功するケース"""
        @retry_with_backoff()
        def sample_func():
            return "success"
        
        result = sample_func()
        
        assert result == "success"
        mock_sleep.assert_not_called()
    
    @patch('time.sleep')
    def test_success_after_retries(self, mock_sleep):
        """リトライ後に成功するケース"""
        mock_func = Mock(side_effect=[
            ConnectionError("Error"),
            ConnectionError("Error"),
            "success"
        ])
        
        @retry_with_backoff()
        def sample_func():
            return mock_func()
        
        result = sample_func()
        
        assert result == "success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2
    
    @patch('time.sleep')
    def test_timeout_exception(self, mock_sleep):
        """Timeout例外のリトライ"""
        mock_func = Mock(side_effect=[
            Timeout("Timeout"),
            "success"
        ])
        
        @retry_with_backoff()
        def sample_func():
            return mock_func()
        
        result = sample_func()
        
        assert result == "success"
        assert mock_func.call_count == 2
        assert mock_sleep.call_count == 1
    
    def test_non_retryable_exception(self):
        """リトライ対象外の例外"""
        @retry_with_backoff()
        def sample_func():
            raise ValueError("Not retryable")
        
        with pytest.raises(ValueError, match="Not retryable"):
            sample_func()
