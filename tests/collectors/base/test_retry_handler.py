"""
RetryConfigとリトライハンドラーのテスト
"""
import pytest
import time
from unittest.mock import Mock, patch
from src.collectors.base.retry_handler import RetryConfig, retry_with_backoff


class TestRetryConfig:
    """RetryConfigクラスのテスト"""
    
    def test_default_config(self):
        """デフォルト設定のテスト"""
        config = RetryConfig()
        assert config.max_retries == 10
        assert config.base_delay == 30.0
        assert config.max_delay == 3840.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True
    
    def test_custom_config(self):
        """カスタム設定のテスト"""
        config = RetryConfig(
            max_retries=5,
            base_delay=10.0,
            max_delay=1000.0,
            backoff_factor=1.5,
            jitter=False
        )
        assert config.max_retries == 5
        assert config.base_delay == 10.0
        assert config.max_delay == 1000.0
        assert config.backoff_factor == 1.5
        assert config.jitter is False
    
    def test_get_delay(self):
        """待機時間計算のテスト"""
        config = RetryConfig(base_delay=10.0, backoff_factor=2.0, jitter=False)
        
        # 1回目のリトライ
        wait_time = config.get_delay(0)
        assert wait_time == 10.0
        
        # 2回目のリトライ
        wait_time = config.get_delay(1)
        assert wait_time == 20.0
        
        # 3回目のリトライ
        wait_time = config.get_delay(2)
        assert wait_time == 40.0
    
    def test_get_delay_with_max(self):
        """最大待機時間制限のテスト"""
        config = RetryConfig(
            base_delay=100.0,
            max_delay=200.0,
            backoff_factor=2.0,
            jitter=False
        )
        
        # 最大値を超える場合
        wait_time = config.get_delay(5)  # 100 * 2^5 = 3200 > 200
        assert wait_time == 200.0
    
    def test_get_delay_with_jitter(self):
        """ジッターありの待機時間計算のテスト"""
        config = RetryConfig(base_delay=10.0, backoff_factor=2.0, jitter=True)
        
        # ジッターがあるため、範囲内かどうかを確認
        wait_time = config.get_delay(0)
        assert 0 <= wait_time <= 10.0
        
        wait_time = config.get_delay(1)
        assert 0 <= wait_time <= 20.0


class TestRetryWithBackoff:
    """retry_with_backoffデコレータのテスト"""
    
    def test_success_on_first_try(self):
        """1回目で成功するケース"""
        mock_func = Mock(return_value="success")
        config = RetryConfig(max_retries=3)
        
        decorated_func = retry_with_backoff(config)(mock_func)
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_success_after_retries(self):
        """リトライ後に成功するケース"""
        mock_func = Mock(side_effect=[
            Exception("Error 1"),
            Exception("Error 2"),
            "success"
        ])
        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        
        decorated_func = retry_with_backoff(config)(mock_func)
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_fail_after_max_retries(self):
        """最大リトライ回数後に失敗するケース"""
        mock_func = Mock(side_effect=Exception("Persistent error"))
        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        
        decorated_func = retry_with_backoff(config)(mock_func)
        
        with pytest.raises(Exception, match="Persistent error"):
            decorated_func()
        
        assert mock_func.call_count == 4  # 初回 + 3回のリトライ
    
    def test_retry_with_specific_exceptions(self):
        """特定の例外のみリトライするケース"""
        mock_func = Mock(side_effect=[
            ConnectionError("Network error"),
            TimeoutError("Timeout"),
            "success"
        ])
        config = RetryConfig(max_retries=5, base_delay=0.01, jitter=False)
        
        decorated_func = retry_with_backoff(
            config,
            retry_exceptions=(ConnectionError, TimeoutError)
        )(mock_func)
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_no_retry_for_non_specified_exception(self):
        """指定されていない例外はリトライしないケース"""
        mock_func = Mock(side_effect=ValueError("Invalid value"))
        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        
        decorated_func = retry_with_backoff(
            config,
            retry_exceptions=(ConnectionError,)
        )(mock_func)
        
        with pytest.raises(ValueError, match="Invalid value"):
            decorated_func()
        
        assert mock_func.call_count == 1  # リトライされない
    
    @patch('time.sleep')
    def test_wait_time_between_retries(self, mock_sleep):
        """リトライ間の待機時間のテスト"""
        mock_func = Mock(side_effect=[
            Exception("Error 1"),
            Exception("Error 2"),
            "success"
        ])
        config = RetryConfig(
            max_retries=3,
            base_delay=10.0,
            backoff_factor=2.0,
            jitter=False
        )
        
        decorated_func = retry_with_backoff(config)(mock_func)
        result = decorated_func()
        
        assert result == "success"
        # 1回目のリトライ前: 10秒, 2回目のリトライ前: 20秒
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 10.0
        assert mock_sleep.call_args_list[1][0][0] == 20.0
    
    def test_with_function_arguments(self):
        """関数の引数が正しく渡されるかのテスト"""
        def test_func(a, b, c=3):
            if a < 0:
                raise ValueError("Negative value")
            return a + b + c
        
        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        decorated_func = retry_with_backoff(config)(test_func)
        
        result = decorated_func(1, 2, c=4)
        assert result == 7
    
    def test_preserve_function_metadata(self):
        """デコレータが関数のメタデータを保持するかのテスト"""
        def test_func():
            """Test function docstring"""
            pass
        
        config = RetryConfig()
        decorated_func = retry_with_backoff(config)(test_func)
        
        assert decorated_func.__name__ == "test_func"
        assert decorated_func.__doc__ == "Test function docstring"


class TestRetryIntegration:
    """リトライ機能の統合テスト"""
    
    @patch('time.sleep')
    def test_realistic_retry_scenario(self, mock_sleep):
        """実際のAPIコール失敗シナリオ"""
        call_count = 0
        
        def unstable_api_call():
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:
                raise ConnectionError(f"Connection failed (attempt {call_count})")
            return {"status": "success", "data": "result"}
        
        config = RetryConfig(
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0,
            backoff_factor=2.0,
            jitter=False
        )
        
        decorated_func = retry_with_backoff(
            config,
            retry_exceptions=(ConnectionError,)
        )(unstable_api_call)
        
        result = decorated_func()
        
        assert result == {"status": "success", "data": "result"}
        assert call_count == 3
        assert mock_sleep.call_count == 2  # 2回のリトライ
    
    def test_exponential_backoff_sequence(self):
        """指数バックオフの正しいシーケンス"""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=100.0,
            backoff_factor=2.0,
            jitter=False
        )
        
        # 待機時間のシーケンスを確認
        expected_sequence = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0, 100.0]  # 100が最大値
        
        for attempt, expected_wait in enumerate(expected_sequence):
            actual_wait = config.get_delay(attempt)
            assert actual_wait == expected_wait
