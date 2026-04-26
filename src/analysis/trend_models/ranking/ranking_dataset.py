"""
Trend Models Analysis - Ranking dataset helpers
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.analysis.trend_models.utils.constants import RANKING_LABEL_COLUMN_BY_MODE


def get_ranking_target_column(label_mode: str) -> str:
    """ラベルモードから目的変数カラム名を取得する。"""
    if label_mode not in RANKING_LABEL_COLUMN_BY_MODE:
        raise ValueError(
            f"無効な ranking label mode: {label_mode}. "
            f"choices={list(RANKING_LABEL_COLUMN_BY_MODE.keys())}"
        )
    return RANKING_LABEL_COLUMN_BY_MODE[label_mode]


def build_ranking_matrix(
    df: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
    query_col: str = 'query_id',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ランキング学習用の行列(X, y, group_sizes)を返す。"""
    if target_col not in df.columns:
        raise KeyError(f"target column not found: {target_col}")

    feature_cols = [col for col in feature_names if col in df.columns]
    if not feature_cols:
        raise ValueError("利用可能な特徴量カラムがありません")

    work = df.copy()
    if query_col in work.columns:
        work = work.sort_values([query_col, 'analysis_date', 'change_number'], kind='stable')

    X = work[feature_cols].values
    y = work[target_col].values

    if query_col in work.columns:
        group_sizes = (
            work.groupby(query_col)
            .size()
            .astype(int)
            .values
        )
    else:
        group_sizes = np.array([len(work)], dtype=int)

    return X, y, group_sizes
