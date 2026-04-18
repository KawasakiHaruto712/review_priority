"""
OLS実行モジュール
statsmodelsを用いた最小二乗法による重回帰分析
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.daily_regression.utils.constants import METRIC_COLUMNS, MIN_SAMPLES

logger = logging.getLogger(__name__)


def execute_ols(
    df: pd.DataFrame,
    target_col: str = 'time_to_review_seconds',
    feature_cols: Optional[List[str]] = None,
    min_samples: int = MIN_SAMPLES,
    standardize: bool = False,
) -> Optional[Dict]:
    """
    OLS重回帰分析を実行する

    Args:
        df: 目的変数 + 説明変数のカラムを持つDataFrame
        target_col: 目的変数のカラム名
        feature_cols: 説明変数のカラム名リスト（省略時はMETRIC_COLUMNS）
        min_samples: 最小サンプル数（不足時はNoneを返す）
        standardize: Trueの場合、説明変数と目的変数をz-score標準化してから
            OLSを実行する。係数は標準化回帰係数（β）となり、
            特徴量間で直接比較可能になる。

    Returns:
        Optional[Dict]: 回帰結果の辞書。スキップ時はNone
        {
            'coefficients': {
                'const': {'coef': float, 'std_err': float, 't_value': float, 'p_value': float},
                'metric_name': {...},
                ...
            },
            'r_squared': float,
            'adj_r_squared': float,
            'f_statistic': float,
            'f_pvalue': float,
            'n_samples': int,
            'standardized': bool,
        }
    """
    if feature_cols is None:
        feature_cols = METRIC_COLUMNS

    if len(df) < min_samples:
        logger.debug(f"サンプル数不足: {len(df)} < {min_samples}")
        return None

    # 説明変数の数 + 1 (定数項) よりサンプル数が少ない場合もスキップ
    n_features = len(feature_cols)
    if len(df) <= n_features + 1:
        logger.debug(f"サンプル数が説明変数数+1以下: {len(df)} <= {n_features + 1}")
        return None

    y = df[target_col].values.astype(float)
    X = df[feature_cols].values.astype(float)

    # 分散0の説明変数を除外（sm.add_constant が定数列と誤認するのを防ぐ）
    variances = np.var(X, axis=0)
    valid_mask = variances > 0
    active_feature_cols = [f for f, v in zip(feature_cols, valid_mask) if v]

    if not active_feature_cols:
        logger.debug("有効な説明変数がありません（全て分散0）")
        return None

    X = X[:, valid_mask]

    # サンプル数の再チェック（有効な特徴量数で）
    if len(df) <= len(active_feature_cols) + 1:
        logger.debug(
            f"サンプル数が有効説明変数数+1以下: "
            f"{len(df)} <= {len(active_feature_cols) + 1}"
        )
        return None

    # z-score標準化（説明変数・目的変数の両方）
    if standardize:
        y_std = np.std(y, ddof=0)
        if y_std == 0:
            logger.debug("目的変数の分散が0のためスキップ")
            return None
        y = (y - np.mean(y)) / y_std

        X_means = np.mean(X, axis=0)
        X_stds = np.std(X, axis=0, ddof=0)
        # 分散0チェックは既に valid_mask で除外済みだが念のため
        X_stds[X_stds == 0] = 1.0
        X = (X - X_means) / X_stds

    # 定数項を追加
    X = sm.add_constant(X, has_constant='add')

    try:
        model = sm.OLS(y, X)
        results = model.fit()
    except Exception as e:
        logger.warning(f"OLS実行エラー: {e}")
        return None

    # 係数情報の構築
    param_names = ['const'] + list(active_feature_cols)
    coefficients = {}
    for i, name in enumerate(param_names):
        coefficients[name] = {
            'coef': float(results.params[i]),
            'std_err': float(results.bse[i]),
            't_value': float(results.tvalues[i]),
            'p_value': float(results.pvalues[i]),
        }

    # 除外されたメトリクスはNaNで埋める
    for col in feature_cols:
        if col not in coefficients:
            coefficients[col] = {
                'coef': np.nan,
                'std_err': np.nan,
                't_value': np.nan,
                'p_value': np.nan,
            }

    # F統計量の取得
    try:
        f_statistic = float(results.fvalue)
        f_pvalue = float(results.f_pvalue)
    except Exception:
        f_statistic = np.nan
        f_pvalue = np.nan

    return {
        'coefficients': coefficients,
        'r_squared': float(results.rsquared),
        'adj_r_squared': float(results.rsquared_adj),
        'f_statistic': f_statistic,
        'f_pvalue': f_pvalue,
        'n_samples': len(df),
        'standardized': standardize,
    }
