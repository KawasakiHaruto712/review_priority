"""
サンプル抽出モジュール（薄いラッパ）

shared_ranking 経由で生成されたイベント駆動ランキングデータから
analysis_date に該当する行を抽出する。

旧仕様（日次スナップショット）のロジックは廃止し、shared_ranking が
生成する event_ranking_<hash>.pkl を一次キャッシュとして利用する。
詳細は src/analysis/shared_ranking/design.md を参照。
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.daily_regression.utils.constants import (
    EXCLUSION_WINDOW_SECONDS,
    MAX_CENSORING_SECONDS,
)
from src.analysis.shared_ranking import load_or_build_event_ranking_dataset

logger = logging.getLogger(__name__)


def _to_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.date()


def extract_daily_samples(
    analysis_date: date,
    all_changes: List[Dict],
    bot_names: List[str],
    *,
    project_name: str = "nova",
    release_version: Optional[str] = None,
    period_start: Optional[datetime] = None,
    period_end: Optional[datetime] = None,
    releases_df: Optional[pd.DataFrame] = None,
    all_prs_df: Optional[pd.DataFrame] = None,
    max_censoring_seconds: int = MAX_CENSORING_SECONDS,
    exclusion_window_seconds: int = EXCLUSION_WINDOW_SECONDS,  # noqa: ARG001 (互換維持のため受領)
) -> pd.DataFrame:
    """
    指定日の Open Change を shared_ranking 経由で抽出する。

    旧 IF（analysis_date, all_changes, bot_names）は維持しつつ、shared_ranking
    のキャッシュ参照に必要な追加情報を kwargs として受け取る。

    Returns:
        pd.DataFrame: 当日分のスナップショット行。
        共有スキーマ（shared_ranking design.md §4.2）+ 旧互換のため
        'review_priority_rank', 'time_to_review_seconds', 'analysis_time' を含む。
    """
    if release_version is None or releases_df is None or all_prs_df is None:
        raise ValueError(
            "extract_daily_samples は shared_ranking 用の release_version / "
            "releases_df / all_prs_df が必須です"
        )
    if period_start is None or period_end is None:
        raise ValueError(
            "extract_daily_samples には release 全期間の period_start / period_end "
            "を渡してください"
        )

    base_df = load_or_build_event_ranking_dataset(
        project_name=project_name,
        release_version=release_version,
        period_start=period_start,
        period_end=period_end,
        all_changes=all_changes,
        all_prs_df=all_prs_df,
        releases_df=releases_df,
        bot_names=bot_names,
        max_censoring_seconds=max_censoring_seconds,
    )

    if base_df.empty:
        return base_df

    target_date = _to_date(analysis_date)
    if target_date is None:
        return pd.DataFrame()

    if "analysis_date" in base_df.columns:
        # analysis_date は date 型を期待。dtype が異なる場合に備えて変換する。
        date_series = base_df["analysis_date"].apply(_to_date)
        mask = date_series == target_date
    else:
        date_series = pd.to_datetime(base_df["measurement_time"]).dt.date
        mask = date_series == target_date

    df = base_df.loc[mask].copy()
    if df.empty:
        return df

    logger.debug("%s: %d サンプルを抽出", target_date, len(df))
    return df
