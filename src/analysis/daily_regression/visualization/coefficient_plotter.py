"""
回帰係数の時系列プロットモジュール
各メトリクスの回帰係数を日付ごとに折れ線グラフ（線のみ、マーカーなし）で描画する
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from src.analysis.daily_regression.utils.constants import (
    METRIC_COLUMNS,
    METRIC_DISPLAY_NAMES,
    VISUALIZATION_CONFIG,
)

logger = logging.getLogger(__name__)


def plot_coefficient_timeseries(
    daily_results: pd.DataFrame,
    metric_name: str,
    version: str,
    output_path: Path,
    significance_level: float = 0.05,
) -> None:
    """
    1つのメトリクスの回帰係数の時系列プロットを生成する

    Args:
        daily_results: 日ごとの回帰結果DataFrame
            必須カラム: 'date', 'coef_{metric_name}', 'pvalue_{metric_name}'
        metric_name: メトリクス名
        version: バージョン名（タイトル用）
        output_path: 出力先パス
        significance_level: 有意水準（デフォルト: 0.05）
    """
    coef_col = f'coef_{metric_name}'
    pvalue_col = f'pvalue_{metric_name}'

    if coef_col not in daily_results.columns:
        logger.warning(f"カラム {coef_col} が存在しません")
        return

    # スキップされていない行のみ抽出
    plot_df = daily_results.dropna(subset=[coef_col]).copy()
    if plot_df.empty:
        logger.warning(f"{metric_name}: プロットデータが空です")
        return

    plot_df['date'] = pd.to_datetime(plot_df['date'])
    plot_df = plot_df.sort_values('date')

    fig_size = VISUALIZATION_CONFIG.get('figure_size', (16, 8))
    dpi = VISUALIZATION_CONFIG.get('dpi', 300)

    fig, ax = plt.subplots(figsize=fig_size)

    # 折れ線グラフ（線のみ、マーカーなし）
    ax.plot(
        plot_df['date'],
        plot_df[coef_col],
        color=VISUALIZATION_CONFIG.get('line_color', '#1f77b4'),
        alpha=VISUALIZATION_CONFIG.get('line_alpha', 0.8),
        linewidth=VISUALIZATION_CONFIG.get('line_width', 1.0),
        marker='',  # マーカーなし
    )

    # y=0 の水平線
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    ax.set_title(f'{display_name} - Standardized Regression Coefficient (version {version})', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Standardized Coefficient (β)', fontsize=12)

    # X軸の日付フォーマット
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logger.debug(f"プロットを保存: {output_path}")


def plot_all_coefficients(
    daily_results: pd.DataFrame,
    version: str,
    output_dir: Path,
    metric_columns: Optional[List[str]] = None,
    significance_level: float = 0.05,
) -> None:
    """
    全メトリクスの回帰係数プロットを一括生成する

    Args:
        daily_results: 日ごとの回帰結果DataFrame
        version: バージョン名
        output_dir: 出力ディレクトリ
        metric_columns: メトリクス名リスト（省略時はMETRIC_COLUMNS）
        significance_level: 有意水準
    """
    if metric_columns is None:
        metric_columns = METRIC_COLUMNS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_format = VISUALIZATION_CONFIG.get('save_format', 'png')

    for metric_name in metric_columns:
        output_path = output_dir / f'coef_{metric_name}.{save_format}'
        plot_coefficient_timeseries(
            daily_results=daily_results,
            metric_name=metric_name,
            version=version,
            output_path=output_path,
            significance_level=significance_level,
        )

    logger.info(f"全メトリクスのプロットを保存しました: {output_dir}")


def plot_r_squared_timeseries(
    daily_results: pd.DataFrame,
    version: str,
    output_path: Path,
) -> None:
    """
    決定係数（R²）と自由度調整済み決定係数（adj R²）の時系列プロットを生成する

    Args:
        daily_results: 日ごとの回帰結果DataFrame
            必須カラム: 'date', 'r_squared', 'adj_r_squared'
        version: バージョン名（タイトル用）
        output_path: 出力先パス
    """
    required_cols = ['date', 'r_squared', 'adj_r_squared']
    for col in required_cols:
        if col not in daily_results.columns:
            logger.warning(f"カラム {col} が存在しません")
            return

    plot_df = daily_results.dropna(subset=['r_squared']).copy()
    if plot_df.empty:
        logger.warning("R²プロットデータが空です")
        return

    plot_df['date'] = pd.to_datetime(plot_df['date'])
    plot_df = plot_df.sort_values('date')

    fig_size = VISUALIZATION_CONFIG.get('figure_size', (16, 8))
    dpi = VISUALIZATION_CONFIG.get('dpi', 300)

    fig, ax = plt.subplots(figsize=fig_size)

    # R² の折れ線
    ax.plot(
        plot_df['date'],
        plot_df['r_squared'],
        color='#1f77b4',
        alpha=0.8,
        linewidth=1.2,
        marker='',
        label='R²',
    )

    # Adjusted R² の折れ線
    ax.plot(
        plot_df['date'],
        plot_df['adj_r_squared'],
        color='#ff7f0e',
        alpha=0.8,
        linewidth=1.2,
        marker='',
        label='Adjusted R²',
    )

    ax.set_title(f'R² and Adjusted R² (version {version})', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)

    # X軸の日付フォーマット
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logger.debug(f"R²プロットを保存: {output_path}")
