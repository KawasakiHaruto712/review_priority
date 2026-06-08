"""
shared_ranking のオーケストレータ。

公開 API:
  - build_event_ranking_dataset(...)        : 全件を計算して DataFrame を返す
  - load_or_build_event_ranking_dataset(...) : キャッシュ読み出し → 失敗時に build
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.shared_ranking.cache import (
    compute_param_hash,
    get_cache_paths,
    load_cache,
    save_cache,
)
from src.analysis.shared_ranking.constants import (
    FEATURE_NAMES,
    LOGIC_VERSION,
    MAX_CENSORING_SECONDS,
    MIN_QUERY_SIZE,
)
from src.analysis.shared_ranking.event_extractor import (
    MeasurementEvent,
    extract_events_in_period,
)
from src.analysis.shared_ranking.feature_evaluator import FeatureEvaluator
from src.analysis.shared_ranking.snapshot_builder import (
    assign_dense_rank,
    build_change_meta_index,
    collect_population,
)

logger = logging.getLogger(__name__)


# プロセス内 LRU 風キャッシュ（ファイル I/O を抑止）
_RUNTIME_CACHE: Dict[str, pd.DataFrame] = {}


def _build_query_id(event: MeasurementEvent, project: str, release: str) -> str:
    return (
        f"{project}_{release}_{event.measurement_time:%Y%m%dT%H%M%S}"
        f"_{event.trigger_change_number}_{event.trigger_revision_number}"
    )


def build_event_ranking_dataset(
    project_name: str,
    release_version: str,
    period_start: datetime,
    period_end: datetime,
    all_changes: List[Dict[str, Any]],
    all_prs_df: pd.DataFrame,
    releases_df: pd.DataFrame,
    bot_names: List[str],
    min_query_size: int = MIN_QUERY_SIZE,
    max_censoring_seconds: int = MAX_CENSORING_SECONDS,
) -> pd.DataFrame:
    """
    1 つの (project, release) に対するイベント駆動ランキングデータを生成する。

    出力スキーマは design.md §4.2 を参照。
    """
    if period_start >= period_end:
        return pd.DataFrame()

    # 1. 計測イベント抽出
    events = extract_events_in_period(all_changes, period_start, period_end)
    if not events:
        logger.warning(
            "計測イベントなし: project=%s, release=%s, period=[%s, %s)",
            project_name,
            release_version,
            period_start,
            period_end,
        )
        return pd.DataFrame()

    # 2. change index 構築
    change_dict_index: Dict[int, Dict[str, Any]] = {}
    for change in all_changes:
        cn = change.get("change_number")
        if cn is None:
            cn = change.get("_number")
        if cn is None:
            continue
        try:
            change_dict_index[int(cn)] = change
        except (TypeError, ValueError):
            continue

    change_meta_index = build_change_meta_index(all_changes)

    # 3. 特徴量評価器
    evaluator = FeatureEvaluator(
        all_prs_df=all_prs_df,
        releases_df=releases_df,
        project_name=project_name,
    )

    # 4. 各計測イベントごとに 1 行（トリガー PR のみ）を生成する。
    #    母集合は rank 計算のためだけに使い、行としては保存しない。
    rows: List[Dict[str, Any]] = []
    skipped_small = 0
    skipped_no_population = 0
    skipped_trigger_already_reviewed = 0
    skipped_feature_failed = 0

    total_events = len(events)
    for i, event in enumerate(events):
        if (i + 1) % 500 == 0:
            logger.info(
                "snapshot 構築進捗: %d / %d (rows=%d, feature_cache=%d)",
                i + 1,
                total_events,
                len(rows),
                evaluator.cache_size(),
            )

        T = event.measurement_time
        population = collect_population(
            measurement_time=T,
            change_meta_index=change_meta_index,
            change_dict_index=change_dict_index,
            bot_names=bot_names,
            max_censoring_seconds=max_censoring_seconds,
        )

        if not population:
            skipped_no_population += 1
            continue
        if len(population) < min_query_size:
            skipped_small += 1
            continue

        # 母集合の中からトリガー PR を探す。
        # revision_update イベントで、トリガー PR が既に過去レビュー済の場合は
        # collect_population が未検証フィルタで除外しているため見つからない。
        trigger_cn = int(event.trigger_change_number)
        trigger_idx: Optional[int] = None
        for idx, item in enumerate(population):
            if item.change_number == trigger_cn:
                trigger_idx = idx
                break

        if trigger_idx is None:
            skipped_trigger_already_reviewed += 1
            continue

        rank_values = assign_dense_rank([item.time_to_review_seconds for item in population])
        trigger_item = population[trigger_idx]
        trigger_rank = rank_values[trigger_idx]
        denom = max(len(population) - 1, 1)

        features = evaluator.evaluate(trigger_item.change, trigger_item.change_number, T)
        if features is None:
            skipped_feature_failed += 1
            continue

        query_id = _build_query_id(event, project_name, release_version)
        analysis_date = T.date()

        row: Dict[str, Any] = {
            "project": project_name,
            "release": release_version,
            "event_type": event.event_type,
            "trigger_change_number": event.trigger_change_number,
            "trigger_revision_number": event.trigger_revision_number,
            "measurement_time": T,
            "query_id": query_id,
            "query_size": len(population),
            "change_number": trigger_item.change_number,
            "time_to_review_seconds": trigger_item.time_to_review_seconds,
            "reviewed": trigger_item.reviewed,
            "censored": trigger_item.censored,
            "review_priority_rank": trigger_rank,
            "review_priority_rank_pct": (trigger_rank - 1.0) / float(denom),
            "analysis_date": analysis_date,
            "analysis_time": T,
        }
        row.update(features)
        rows.append(row)

    logger.info(
        "shared_ranking 構築完了: project=%s, release=%s, events=%d, "
        "rows=%d, skipped(small)=%d, skipped(empty)=%d, "
        "skipped(trigger_reviewed)=%d, skipped(feature_failed)=%d, feature_cache=%d",
        project_name,
        release_version,
        total_events,
        len(rows),
        skipped_small,
        skipped_no_population,
        skipped_trigger_already_reviewed,
        skipped_feature_failed,
        evaluator.cache_size(),
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # カラム順を整える
    leading_cols = [
        "project",
        "release",
        "event_type",
        "trigger_change_number",
        "trigger_revision_number",
        "measurement_time",
        "query_id",
        "query_size",
        "change_number",
        "time_to_review_seconds",
        "reviewed",
        "censored",
        "review_priority_rank",
        "review_priority_rank_pct",
        "analysis_date",
        "analysis_time",
    ]
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    other_cols = [c for c in df.columns if c not in leading_cols and c not in feature_cols]
    df = df[leading_cols + feature_cols + other_cols]

    # dtype を整える
    df["measurement_time"] = pd.to_datetime(df["measurement_time"])
    df["analysis_time"] = pd.to_datetime(df["analysis_time"])

    return df


def load_or_build_event_ranking_dataset(
    project_name: str,
    release_version: str,
    period_start: datetime,
    period_end: datetime,
    all_changes: List[Dict[str, Any]],
    all_prs_df: pd.DataFrame,
    releases_df: pd.DataFrame,
    bot_names: List[str],
    *,
    min_query_size: int = MIN_QUERY_SIZE,
    max_censoring_seconds: int = MAX_CENSORING_SECONDS,
    use_cache: bool = True,
    rebuild: bool = False,
) -> pd.DataFrame:
    """
    キャッシュ → ランタイム → ファイル → ビルド の優先順で取得する。
    """
    param_hash = compute_param_hash(
        project_name=project_name,
        release_version=release_version,
        period_start=period_start,
        period_end=period_end,
        min_query_size=min_query_size,
        max_censoring_seconds=max_censoring_seconds,
        feature_names=FEATURE_NAMES,
        logic_version=LOGIC_VERSION,
    )
    runtime_key = f"{project_name}/{release_version}/{param_hash}"
    pkl_path, meta_path = get_cache_paths(project_name, release_version, param_hash)

    if not rebuild:
        cached_df = _RUNTIME_CACHE.get(runtime_key)
        if cached_df is not None:
            return cached_df

        if use_cache:
            cached_df = load_cache(pkl_path)
            if cached_df is not None:
                _RUNTIME_CACHE[runtime_key] = cached_df
                return cached_df

    df = build_event_ranking_dataset(
        project_name=project_name,
        release_version=release_version,
        period_start=period_start,
        period_end=period_end,
        all_changes=all_changes,
        all_prs_df=all_prs_df,
        releases_df=releases_df,
        bot_names=bot_names,
        min_query_size=min_query_size,
        max_censoring_seconds=max_censoring_seconds,
    )

    _RUNTIME_CACHE[runtime_key] = df

    if use_cache and not df.empty:
        save_cache(
            df,
            pkl_path=pkl_path,
            meta_path=meta_path,
            metadata={
                "project": project_name,
                "release": release_version,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "min_query_size": int(min_query_size),
                "max_censoring_seconds": int(max_censoring_seconds),
                "feature_names": list(FEATURE_NAMES),
                "param_hash": param_hash,
            },
        )

    return df


def clear_runtime_cache() -> None:
    """テスト用にプロセス内ランタイムキャッシュをクリアする。"""
    _RUNTIME_CACHE.clear()
