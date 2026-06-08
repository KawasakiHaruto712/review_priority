"""
特徴量評価器。

trend_models.features.extractor.FeatureExtractor を内部で利用し、
(change_number, measurement_time) 単位でメモ化する。

FeatureExtractor は period_start を渡すと analysis_time を
max(change.created, period_start) にする仕様であり、
本ロジックではスナップショットの母集合チケットは必ず created <= T を満たすため、
period_start=T を渡せば analysis_time = T が得られる。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.analysis.shared_ranking.constants import FEATURE_NAMES

logger = logging.getLogger(__name__)


class FeatureEvaluator:
    """
    (change_number, measurement_time) -> 16 特徴量 dict のメモ化評価器。

    内部で trend_models の FeatureExtractor を 1 インスタンスだけ保持する。
    """

    def __init__(
        self,
        all_prs_df: pd.DataFrame,
        releases_df: pd.DataFrame,
        project_name: str,
    ) -> None:
        # trend_models -> shared_ranking -> trend_models の循環インポートを避けるため、
        # FeatureExtractor は遅延 import する。
        from src.analysis.trend_models.features.extractor import FeatureExtractor

        self._extractor = FeatureExtractor(
            all_prs_df=all_prs_df,
            releases_df=releases_df,
            project_name=project_name,
        )
        self._cache: Dict[Tuple[int, datetime], Dict[str, Any]] = {}

    def evaluate(
        self,
        change: Dict[str, Any],
        change_number: int,
        measurement_time: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        (change_number, measurement_time) における 16 特徴量を返す。

        Returns:
            16 特徴量のみを含む dict。失敗時は None。
        """
        key = (int(change_number), measurement_time)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        try:
            raw = self._extractor.extract(change, period_start=measurement_time)
        except Exception as exc:
            logger.warning(
                "特徴量計算失敗: change=%s, T=%s, err=%s",
                change_number,
                measurement_time,
                exc,
            )
            return None

        features = {name: raw.get(name) for name in FEATURE_NAMES}
        # None が混じる場合は埋めておく（後段で扱いやすいように 0 / 0.0）
        for name in FEATURE_NAMES:
            if features[name] is None:
                features[name] = 0.0 if name in {
                    "elapsed_time",
                    "merge_rate",
                    "recent_merge_rate",
                    "days_to_major_release",
                } else 0

        self._cache[key] = features
        return features

    def cache_size(self) -> int:
        return len(self._cache)
