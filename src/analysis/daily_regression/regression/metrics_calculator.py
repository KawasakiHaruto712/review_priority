"""
メトリクス計算モジュール
サンプルに対して16種類のメトリクスを計算する
trend_metrics の metrics_calculator と同様のロジック
"""

import logging
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

from src.features.change_metrics import (
    calculate_lines_added,
    calculate_lines_deleted,
    calculate_files_changed,
    calculate_elapsed_time,
    calculate_revision_count,
    check_test_code_presence,
)
from src.features.bug_metrics import calculate_bug_fix_confidence
from src.features.refactoring_metrics import calculate_refactoring_confidence
from src.features.developer_metrics import (
    calculate_past_report_count,
    calculate_recent_report_count,
    calculate_merge_rate,
    calculate_recent_merge_rate,
    get_owner_email,
)
from src.features.project_metrics import (
    calculate_days_to_major_release,
    calculate_predictive_target_ticket_count,
    calculate_reviewed_lines_in_period,
)
from src.features.review_metrics import ReviewStatusAnalyzer
from src.config.path import DEFAULT_DATA_DIR, DEFAULT_CONFIG
from src.analysis.daily_regression.utils.constants import (
    METRIC_COLUMNS,
    DEFAULT_TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

# ReviewStatusAnalyzerのインスタンスを保持（再利用のため）
_review_analyzer = None


def get_review_analyzer():
    """ReviewStatusAnalyzerのシングルトンインスタンスを取得"""
    global _review_analyzer
    if _review_analyzer is None:
        keywords_path = DEFAULT_DATA_DIR / "processed" / "review_keywords.json"
        label_path = DEFAULT_DATA_DIR / "processed" / "review_label.json"
        config_path = DEFAULT_CONFIG / "gerrymanderconfig.ini"

        if keywords_path.exists() and label_path.exists():
            _review_analyzer = ReviewStatusAnalyzer(keywords_path, config_path, label_path)
        else:
            logger.warning("Review metrics configuration files not found.")
            _review_analyzer = None
    return _review_analyzer


def _calculate_single_change_metrics(
    change: Dict[str, Any],
    analysis_time,
    all_changes_df: pd.DataFrame,
    releases_df: pd.DataFrame,
    project_name: str,
) -> Optional[Dict[str, Any]]:
    """
    1つのChangeに対して全メトリクスを計算する

    Args:
        change: Changeデータ
        analysis_time: 分析時点（datetime）
        all_changes_df: 全Change履歴のDataFrame
        releases_df: リリース情報のDataFrame
        project_name: プロジェクト名

    Returns:
        Optional[Dict]: メトリクス辞書。計算不能時はNone
    """
    metrics = {}

    # Change Metrics
    metrics['lines_added'] = calculate_lines_added(change, analysis_time)
    metrics['lines_deleted'] = calculate_lines_deleted(change, analysis_time)
    metrics['files_changed'] = calculate_files_changed(change, analysis_time)
    metrics['elapsed_time'] = calculate_elapsed_time(change, analysis_time)
    metrics['revision_count'] = calculate_revision_count(change, analysis_time)
    metrics['test_code_presence'] = check_test_code_presence(change)

    # Bug Metrics
    metrics['bug_fix_confidence'] = calculate_bug_fix_confidence(change)

    # Refactoring Metrics
    metrics['refactoring_confidence'] = calculate_refactoring_confidence(change)

    # Developer Metrics
    owner_email = get_owner_email(change)
    if owner_email:
        metrics['past_report_count'] = calculate_past_report_count(
            owner_email, all_changes_df, analysis_time
        )
        metrics['recent_report_count'] = calculate_recent_report_count(
            owner_email, all_changes_df, analysis_time
        )
        metrics['merge_rate'] = calculate_merge_rate(
            owner_email, all_changes_df, analysis_time
        )
        metrics['recent_merge_rate'] = calculate_recent_merge_rate(
            owner_email, all_changes_df, analysis_time
        )
    else:
        metrics['past_report_count'] = 0
        metrics['recent_report_count'] = 0
        metrics['merge_rate'] = 0.0
        metrics['recent_merge_rate'] = 0.0

    # Project Metrics
    metrics['days_to_major_release'] = calculate_days_to_major_release(
        analysis_time, project_name, releases_df
    )
    metrics['open_ticket_count'] = calculate_predictive_target_ticket_count(
        all_changes_df, analysis_time
    )
    metrics['reviewed_lines_in_period'] = calculate_reviewed_lines_in_period(
        all_changes_df, analysis_time
    )

    # Review Metrics
    analyzer = get_review_analyzer()
    if analyzer:
        change_copy = change.copy()
        metrics['uncompleted_requests'] = analyzer.analyze_pr_status(
            change_copy, analysis_time
        )
    else:
        metrics['uncompleted_requests'] = 0

    return metrics


def calculate_daily_metrics(
    samples_df: pd.DataFrame,
    all_changes: List[Dict],
    all_changes_df: pd.DataFrame,
    releases_df: pd.DataFrame,
    project_name: str,
    target_col: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """
    サンプルに対して16メトリクスを計算する

    Args:
        samples_df: extract_daily_samples() の出力DataFrame
        all_changes: 全Changeデータのリスト（辞書形式）
        all_changes_df: 全Changeデータの DataFrame
        releases_df: リリース情報DataFrame
        project_name: プロジェクト名
        target_col: 目的変数のカラム名

    Returns:
        pd.DataFrame: 目的変数 + 16メトリクスのカラムを持つDataFrame
    """
    if samples_df.empty:
        cols = ['change_number', target_col] + METRIC_COLUMNS
        return pd.DataFrame(columns=cols)

    if target_col not in samples_df.columns:
        raise ValueError(f"samples_dfに目的変数カラムがありません: {target_col}")

    # change_number → change辞書のマッピングを構築
    change_map = {}
    for change in all_changes:
        cn = change.get('_number') or change.get('change_number', 0)
        change_map[cn] = change

    result_records = []

    for _, row in samples_df.iterrows():
        change_number = row['change_number']
        analysis_time = row['analysis_time']
        target_value = row[target_col]

        change = change_map.get(change_number)
        if change is None:
            continue

        metrics = _calculate_single_change_metrics(
            change, analysis_time, all_changes_df, releases_df, project_name
        )
        if metrics is None:
            continue

        # 欠損値チェック: いずれかのメトリクスがNoneなら除外
        has_none = False
        for col in METRIC_COLUMNS:
            if metrics.get(col) is None:
                has_none = True
                break
        if has_none:
            continue

        # elapsed_time が -1.0 の場合は除外
        if metrics.get('elapsed_time', -1.0) == -1.0:
            continue

        record = {
            'change_number': change_number,
            target_col: target_value,
        }
        record.update(metrics)
        result_records.append(record)

    if not result_records:
        cols = ['change_number', target_col] + METRIC_COLUMNS
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(result_records)
