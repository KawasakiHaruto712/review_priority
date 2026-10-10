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
    
    def __init__(self, max_retries: int = 5, base_delay: float = 5.0,
                 max_delay: float = 60.0, backoff_factor: float = 2.0,
                 jitter: bool = True):
        """
        Args:
            max_retries: 最大リトライ回数
            base_delay: 初期待機時間（秒）
            max_delay: 最大待機時間（秒）
            backoff_factor: バックオフ係数（指数的増加の倍率）
            jitter: ジッターを有効にするか（ランダムな揺らぎを追加）

        **この既定値は endpoint_config.yaml の retry と揃えること。**
        BaseAPIClient / FileContentEndpoint は `@retry_with_backoff()` を
        引数なしで付けており、設定ファイルではなく**この既定値**を使う
        （ChangeCollector が作る self.retry_config は実際のリクエストに渡っていない）。
        以前の既定（10 回・30〜3840 秒）では 1 つの失敗に最大 4 時間 15 分を費やし、
        実際に Qt の収集で 4 時間 17 分を溶かした。現在は合計 135 秒（約 2 分）。
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


# サーバ側の実行時間制限は**ここでは投げ直さず、即座に呼び出し側へ返す**。
# 原因が 2 通りあり、待つべき場面が限られるため。
#   (a) 一時的な混雑   … 同じ要求が普段 2.5 秒で返るのに、混んだ瞬間だけ 5 秒超になる
#   (b) 要求が重すぎる … 巨大な Change を含むページ。何度投げ直しても通らない
# (a) は「要求件数 n を半分にして出し直す」ことでも回復するので、呼び出し側は
# まず n を下げて即座に再試行し、n=1 まで下げてもなお駄目なときだけ待って投げ直す
# （`change_collector._collect_range`）。ここで一律に待つと、(b) のときに
# n を下げる段階ごとに待ち時間が積み上がる（実測で 1 ページあたり 3.5 分の空費）。


class ServerDeadlineExceeded(requests.exceptions.HTTPError):
    """サーバ側の実行時間制限に達した（例: Gerrit の RestApi.timeout=5000ms）。

    DEADLINE_RETRY_DELAYS ぶん投げ直しても通らなかったときだけ送出される。
    呼び出し側は「要求を軽くする」（件数を減らす・期間を割る）必要がある。
    """


def _is_server_deadline_error(e: Exception) -> bool:
    """サーバ側の実行時間制限によるエラーか（本文で判定する）。"""
    resp = getattr(e, "response", None)
    if resp is None or resp.status_code < 500:
        return False
    body = (resp.text or "")[:200]
    return "Deadline Exceeded" in body or "deadline exceeded" in body


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

                    # 実行時間制限は待ち方を呼び出し側に委ねる（上のコメント参照）
                    if _is_server_deadline_error(e):
                        raise ServerDeadlineExceeded(str(e), response=e.response) from e

                    # 読み込みのタイムアウトも投げ直さず、呼び出し側に返す。
                    # 該当件数の多いクエリはサーバ側の処理が重く、同じものを投げ直しても
                    # 速くならない。投げ直すと 1 回 120 秒 × 6 回で約 14 分を無駄にし、
                    # **相手のサーバにも重い処理を何度もさせる**。呼び出し側は区間を割って
                    # 軽くしてから取り直す（ChangeCollector._collect_range_adaptive）。
                    # 2026-10-02、chromium/src の 1 年ぶん（約 21 万件）で発生した。
                    if isinstance(e, requests.exceptions.ReadTimeout):
                        raise

                    # HTTP 400（要求が不正）も投げ直さない。何度投げても同じ結果になる
                    # （project_selection/design.md §6.6「HTTP 400 は投げ直さない」）。
                    # 2026-10-06、chromium/src の深いページ（1 クエリ 100 ページの上限）で
                    # 400 を 6 回ずつ投げ直し、1 回の失敗に約 2 分を空費していた。
                    # 対象は 400 だけ。403 は Wikimedia がアクセス頻度の制限に使うので投げ直しを続ける
                    resp = getattr(e, "response", None)
                    if resp is not None and resp.status_code == 400:
                        logger.error(f"HTTPエラー(400)のため投げ直しません: {e}")
                        raise

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
