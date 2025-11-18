"""
リトライハンドラーモジュール

指数バックオフを用いたリトライ処理を提供します。
"""

import time
import random
import logging
import requests
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class RetryConfig:
    """リトライ設定クラス"""
    
    def __init__(self, max_retries: int = 10, base_delay: float = 30.0,
                 max_delay: float = 3840.0, backoff_factor: float = 2.0,
                 jitter: bool = True):
        """
        Args:
            max_retries: 最大リトライ回数
            base_delay: 初期待機時間（秒）
            max_delay: 最大待機時間（秒）
            backoff_factor: バックオフ係数（指数的増加の倍率）
            jitter: ジッターを有効にするか（ランダムな揺らぎを追加）
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, retry_count: int) -> float:
        """
        リトライ回数に基づいて待機時間を計算
        
        Args:
            retry_count: 現在のリトライ回数（0から開始）
            
        Returns:
            待機時間（秒）
        """
        # 指数バックオフによる待機時間の計算
        delay = self.base_delay * (self.backoff_factor ** retry_count)
        
        # 最大待機時間を超えないように制限
        delay = min(delay, self.max_delay)
        
        # ジッターを追加（ランダムな揺らぎで同時リクエストの衝突を避ける）
        if self.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


def retry_with_backoff(retry_config: Optional[RetryConfig] = None) -> Callable:
    """
    指数バックオフを用いたリトライデコレータ
    
    Args:
        retry_config: リトライ設定
        
    Returns:
        デコレートされた関数
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for retry_count in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    
                    if retry_count >= retry_config.max_retries:
                        logger.error(f"最大リトライ回数到達: {e}")
                        raise
                    
                    delay = retry_config.get_delay(retry_count)
                    
                    # エラーの種類によってログレベルを調整
                    if isinstance(e, requests.exceptions.Timeout):
                        log_level = logging.WARNING
                        error_type = "タイムアウト"
                    elif isinstance(e, requests.exceptions.ConnectionError):
                        log_level = logging.WARNING
                        error_type = "接続エラー"
                    elif hasattr(e, 'response') and e.response is not None:
                        if e.response.status_code == 429:
                            log_level = logging.WARNING
                            error_type = "レート制限"
                            delay = max(delay, 30)
                        elif 500 <= e.response.status_code < 600:
                            log_level = logging.WARNING
                            error_type = f"サーバーエラー({e.response.status_code})"
                        else:
                            log_level = logging.ERROR
                            error_type = f"HTTPエラー({e.response.status_code})"
                    else:
                        log_level = logging.ERROR
                        error_type = "ネットワークエラー"
                    
                    logger.log(
                        log_level,
                        f"{error_type}が発生しました。{delay:.1f}秒後にリトライします "
                        f"(試行回数: {retry_count + 1}/{retry_config.max_retries + 1}): {e}"
                    )
                    
                    time.sleep(delay)
                except Exception as e:
                    logger.error(f"予期しないエラー: {e}")
                    raise
            
            raise last_exception
        
        return wrapper
    return decorator
