"""
トレンドプロッターモジュール
メトリクスの分布や変化を可視化するグラフを生成する
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.analysis.trend_metrics.utils.constants import METRIC_COLUMNS

logger = logging.getLogger(__name__)

# グラフのスタイル設定
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True


def plot_boxplots_8groups(
    groups: Dict[str, List[Dict]],
    output_dir: Path,
    metrics: List[str] = None
) -> List[Path]:
    """
    8グループのメトリクス分布をボックスプロットで可視化
    
    Args:
        groups: グループ別のChangeデータ
        output_dir: 出力ディレクトリ
        metrics: 可視化対象メトリクスのリスト（省略時はMETRIC_COLUMNS使用）
    
    Returns:
        List[Path]: 生成された画像ファイルのパスリスト
    """
    if metrics is None:
        metrics = METRIC_COLUMNS
    
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # データフレームの作成
    data_records = []
    for group_name, changes in groups.items():
        # グループ名から属性を抽出
        period = 'Early' if group_name.startswith('early_') else 'Late'
        
        if '_non_core_reviewed' in group_name:
            reviewer_type = 'Non-Core Reviewed'
        elif '_non_core_not_reviewed' in group_name:
            reviewer_type = 'Non-Core Not Reviewed'
        elif '_core_reviewed' in group_name:
            reviewer_type = 'Core Reviewed'
        elif '_core_not_reviewed' in group_name:
            reviewer_type = 'Core Not Reviewed'
        else:
            reviewer_type = 'Unknown'
            
        for change in changes:
            record = {
                'Period': period,
                'Reviewer Type': reviewer_type,
                'Group': group_name
            }
            
            # メトリクス値を追加
            for metric in metrics:
                val = change.get(metric)
                record[metric] = val if val is not None else np.nan
                
            data_records.append(record)
            
    df = pd.DataFrame(data_records)
    
    if df.empty:
        logger.warning("可視化対象のデータがありません")
        return []
    
    # プロット設定: (showfliers, filename_suffix)
    plot_configs = [
        (True, ""),
        (False, "_no_outliers")
    ]
    
    for show_outliers, suffix in plot_configs:
        # グリッドプロット（全メトリクスをまとめて出力）
        # 4x4のグリッドなどで出力（メトリクス数に応じて調整）
        n_metrics = len(metrics)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        
        # 表示順序の定義
        hue_order = ['Early', 'Late']
        order = ['Core Reviewed', 'Core Not Reviewed', 'Non-Core Reviewed', 'Non-Core Not Reviewed']
        
        # 凡例用のハンドルとラベル
        handles, labels = None, None

        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                
                # データが存在するか確認
                if df[metric].notna().sum() > 0:
                    sns.boxplot(
                        data=df,
                        x='Reviewer Type',
                        y=metric,
                        hue='Period',
                        order=order,
                        hue_order=hue_order,
                        ax=ax,
                        showfliers=show_outliers,  # 外れ値の表示設定
                        palette="Set2"
                    )
                    
                    # 凡例情報を取得（最初の有効なプロットから）
                    if handles is None and ax.get_legend() is not None:
                        handles, labels = ax.get_legend_handles_labels()
                    
                    # 個別の凡例は削除
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                    
                    ax.set_title(metric, fontsize=14, fontweight='bold')
                    ax.set_xlabel('')
                    ax.set_ylabel('Value')
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    ax.set_title(metric)
        
        # 余ったサブプロットを非表示
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
            
        # 共通の凡例を図の上部に追加
        if handles and labels:
            fig.legend(
                handles, 
                labels, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 1.02), 
                ncol=2, 
                title='Period', 
                fontsize=12, 
                title_fontsize=12
            )
            
        plt.tight_layout()
        
        output_path = output_dir / f'boxplots_8groups_grid{suffix}.pdf'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        generated_files.append(output_path)
        logger.info(f"ボックスプロットを保存しました: {output_path}")
    
    return generated_files
    
    output_path = output_dir / 'boxplots_8groups_grid.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    generated_files.append(output_path)
    logger.info(f"8グループ比較ボックスプロットを生成しました: {output_path}")
    
    return generated_files


def plot_trend_lines(
    comparison_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path
) -> Path:
    """
    レビューアタイプ別のトレンドライン（前期→後期の変化）を可視化
    
    Args:
        comparison_results: compare_early_vs_late_by_reviewer_type()の結果
        output_dir: 出力ディレクトリ
    
    Returns:
        Path: 生成された画像ファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データの整形
    plot_data = []
    
    for reviewer_type, metrics_data in comparison_results.items():
        # 表示用の名前
        display_name = reviewer_type.replace('_', ' ').title()
        
        for metric, info in metrics_data.items():
            early_mean = info.get('early_mean')
            late_mean = info.get('late_mean')
            
            if early_mean is not None and late_mean is not None:
                # 前期データ
                plot_data.append({
                    'Reviewer Type': display_name,
                    'Metric': metric,
                    'Period': 'Early',
                    'Value': early_mean
                })
                # 後期データ
                plot_data.append({
                    'Reviewer Type': display_name,
                    'Metric': metric,
                    'Period': 'Late',
                    'Value': late_mean
                })
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        logger.warning("トレンドライン用のデータがありません")
        return None
        
    metrics = df['Metric'].unique()
    n_metrics = len(metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            metric_df = df[df['Metric'] == metric]
            
            sns.pointplot(
                data=metric_df,
                x='Period',
                y='Value',
                hue='Reviewer Type',
                ax=ax,
                markers='o',
                linestyles='-',
                palette="tab10"
            )
            
            ax.set_title(metric, fontsize=14, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Mean Value')
            
            # 凡例は最初のプロットのみ、または別枠で表示したいが、
            # ここでは簡易的に各プロットに表示（ただし被らないように調整が必要）
            if i == 0:
                ax.legend(loc='best')
            else:
                ax.get_legend().remove()
                
    # 余ったサブプロットを非表示
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    
    output_path = output_dir / 'trend_lines_by_reviewer_type.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"トレンドラインプロットを生成しました: {output_path}")
    
    return output_path


def plot_metric_changes(
    comparison_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path
) -> Path:
    """
    メトリクスの変化率をバーチャートで可視化
    
    Args:
        comparison_results: compare_early_vs_late_by_reviewer_type()の結果
        output_dir: 出力ディレクトリ
    
    Returns:
        Path: 生成された画像ファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データの整形
    plot_data = []
    
    for reviewer_type, metrics_data in comparison_results.items():
        display_name = reviewer_type.replace('_', ' ').title()
        
        for metric, info in metrics_data.items():
            change_pct = info.get('change_percentage')
            
            if change_pct is not None:
                plot_data.append({
                    'Reviewer Type': display_name,
                    'Metric': metric,
                    'Change (%)': change_pct
                })
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        logger.warning("変化率プロット用のデータがありません")
        return None
        
    # メトリクスごとにプロット
    plt.figure(figsize=(15, 10))
    
    # ヒートマップ形式またはバーチャートで表現
    # ここではバーチャートを使用（メトリクスをX軸、変化率をY軸、レビューアタイプで色分け）
    
    sns.barplot(
        data=df,
        x='Metric',
        y='Change (%)',
        hue='Reviewer Type',
        palette="tab10"
    )
    
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(rotation=45, ha='right')
    plt.title('Percentage Change from Early to Late Period', fontsize=16)
    plt.tight_layout()
    
    output_path = output_dir / 'metric_change_summary.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"メトリクス変化サマリーを生成しました: {output_path}")
    
    return output_path
