"""
Trend Models Analysis - イベント駆動ランキング データセット (薄いラッパ)

本モジュールは shared_ranking モジュールへの薄いラッパであり、
trend_models 既存呼び出し箇所のシグネチャを維持したまま新しい
イベント駆動ランキング（design.md §5）を返す。

旧仕様の「日次スナップショット」は廃止され、shared_ranking の
event_ranking_<hash>.pkl を一次キャッシュとして利用する。
詳細は src/analysis/shared_ranking/design.md を参照。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.shared_ranking import (
    MAX_CENSORING_SECONDS,
    MIN_QUERY_SIZE,
    load_or_build_event_ranking_dataset,
)
from src.analysis.trend_models.utils.constants import (
    RANKING_MAX_CENSORING_SECONDS,
    RANKING_MIN_QUERY_SIZE,
)

logger = logging.getLogger(__name__)


def _resolve_release_period(
    releases_df: pd.DataFrame,
    project_name: str,
    release_version: str,
) -> Optional[tuple]:
    """releases_df から (release_date, next_release_date) を求める。"""
    if releases_df is None or releases_df.empty:
        return None

    project_col = "project" if "project" in releases_df.columns else "component"
    if project_col not in releases_df.columns:
        return None

    project_df = releases_df[releases_df[project_col] == project_name].copy()
    if project_df.empty:
        return None
    project_df = project_df.sort_values("release_date").reset_index(drop=True)

    matched = project_df[project_df["version"] == release_version]
    if matched.empty:
        return None
    idx = matched.index[0]

    current = pd.to_datetime(project_df.loc[idx, "release_date"])
    if idx + 1 < len(project_df):
        next_dt = pd.to_datetime(project_df.loc[idx + 1, "release_date"])
    else:
        return None

    return (
        current.to_pydatetime() if hasattr(current, "to_pydatetime") else current,
        next_dt.to_pydatetime() if hasattr(next_dt, "to_pydatetime") else next_dt,
    )


def build_daily_ranking_dataset(
    project_name: str,
    release_version: str,
    period_type: str,
    period_start: datetime,
    period_end: datetime,
    all_changes: List[Dict[str, Any]],
    all_prs_df: pd.DataFrame,
    releases_df: pd.DataFrame,
    bot_names: List[str],
    min_query_size: int = RANKING_MIN_QUERY_SIZE,
    max_censoring_seconds: int = RANKING_MAX_CENSORING_SECONDS,
) -> pd.DataFrame:
    """
    旧 IF を維持した薄いラッパ。

    内部では shared_ranking の load_or_build_event_ranking_dataset を呼び、
    リリース全期間（release_date 〜 next_release_date）の中間データを 1 度だけ
    生成・キャッシュし、period_type に対応する [period_start, period_end) で
    フィルタして返す。
    """
    if period_start >= period_end:
        return pd.DataFrame()

    # キャッシュ共有のため、ビルドは「リリース全期間」で行う
    full_period = _resolve_release_period(releases_df, project_name, release_version)
    if full_period is None:
        # フォールバック: 呼び出された範囲そのものを使う（キャッシュは早期/後期で別物になる）
        full_start, full_end = period_start, period_end
    else:
        full_start, full_end = full_period

    # 引数の period_start/end が full 範囲を超える場合は、ビルド範囲を広げる
    full_start = min(full_start, period_start)
    full_end = max(full_end, period_end)

    # shared_ranking の既定値（MIN_QUERY_SIZE, MAX_CENSORING_SECONDS）と同値であれば
    # キャッシュキーは daily_regression と一致する
    base_df = load_or_build_event_ranking_dataset(
        project_name=project_name,
        release_version=release_version,
        period_start=full_start,
        period_end=full_end,
        all_changes=all_changes,
        all_prs_df=all_prs_df,
        releases_df=releases_df,
        bot_names=bot_names,
        min_query_size=min_query_size if min_query_size is not None else MIN_QUERY_SIZE,
        max_censoring_seconds=(
            max_censoring_seconds
            if max_censoring_seconds is not None
            else MAX_CENSORING_SECONDS
        ),
    )

    if base_df.empty:
        return base_df

    mask = (base_df["measurement_time"] >= pd.Timestamp(period_start)) & (
        base_df["measurement_time"] < pd.Timestamp(period_end)
    )
    df = base_df.loc[mask].copy()

    if df.empty:
        return df

    df["period_type"] = period_type

    logger.info(
        "trend_models ランキングデータ抽出: project=%s, release=%s, period=%s, rows=%d, queries=%d",
        project_name,
        release_version,
        period_type,
        len(df),
        df["query_id"].nunique() if "query_id" in df.columns else 0,
    )
    return df
