"""
可視化モジュール
ボックスプロットとヒートマップの生成を行う
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path

from src.config.release_constants import LOG_SCALE_THRESHOLD

logger = logging.getLogger(__name__)

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class MetricsVisualizer:
    """
    メトリクスの可視化を行うクラス
    """
    
    def __init__(self, figsize=(20, 20), dpi=300):
        """
        Args:
            figsize (tuple): 図のサイズ（インチ）
            dpi (int): 解像度
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Seabornのスタイル設定
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
    
    def create_boxplots(
        self,
        data: pd.DataFrame,
        group_column: str,
        metric_columns: List[str],
        output_path: Path,
        log_scale_metrics: Optional[List[str]] = None,
        auto_log_scale: bool = True
    ):
        """
        4×4グリッドのボックスプロットを作成
        
        Args:
            data (pd.DataFrame): データフレーム
            group_column (str): グループ化するカラム名
            metric_columns (List[str]): プロットするメトリクスのカラム名リスト
            output_path (Path): 出力先パス
            log_scale_metrics (Optional[List[str]]): 対数軸を使用するメトリクスのリスト
            auto_log_scale (bool): 自動的に対数軸を適用するかどうか
        """
        if log_scale_metrics is None:
            log_scale_metrics = []
        
        # 自動対数軸判定
        if auto_log_scale:
            log_scale_metrics = self._identify_log_scale_metrics(data, metric_columns)
        
        # グリッドのサイズを計算
        n_metrics = len(metric_columns)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols  # 切り上げ
        
        # 図を作成
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        # Always flatten to handle consistently
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # 各メトリクスに対してボックスプロットを作成
        for idx, metric in enumerate(metric_columns):
            ax = axes[idx]
            
            if metric not in data.columns:
                logger.warning(f"メトリクス '{metric}' がデータに存在しません")
                ax.text(0.5, 0.5, f'{metric}\n(No data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            try:
                # ボックスプロットを描画
                sns.boxplot(
                    data=data,
                    x=group_column,
                    y=metric,
                    ax=ax,
                    hue=group_column,
                    palette='Set2',
                    legend=False
                )
                
                # タイトルとラベルの設定
                ax.set_title(metric, fontsize=12, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Value', fontsize=10)
                
                # x軸ラベルの回転
                ax.tick_params(axis='x', rotation=45)
                
                # 対数軸の適用
                if metric in log_scale_metrics:
                    ax.set_yscale('log')
                    ax.set_ylabel('Value (log scale)', fontsize=10)
                
                # グリッド線の追加
                ax.grid(True, alpha=0.3, linestyle='--')
                
            except Exception as e:
                logger.error(f"メトリクス '{metric}' のプロットでエラー: {e}")
                ax.text(0.5, 0.5, f'{metric}\n(Error)', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # 使用しないサブプロットを非表示
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        # レイアウトの調整
        plt.tight_layout()
        
        # 保存
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ボックスプロットを保存しました: {output_path}")
        except Exception as e:
            logger.error(f"ボックスプロットの保存に失敗しました: {e}")
            raise
        finally:
            plt.close()
    
    def create_heatmap(
        self,
        test_results: Dict[str, Dict[str, Dict]],
        output_path: Path,
        metric_columns: List[str]
    ):
        """
        統計検定結果のp値ヒートマップを作成
        
        Args:
            test_results (Dict): 検定結果の辞書
            output_path (Path): 出力先パス
            metric_columns (List[str]): メトリクスのカラム名リスト
        """
        # p値の行列を作成
        comparison_pairs = list(test_results.keys())
        p_value_matrix = []
        
        for pair in comparison_pairs:
            p_values = []
            for metric in metric_columns:
                if metric in test_results[pair]:
                    p_val = test_results[pair][metric].get('p_value')
                    p_values.append(p_val if p_val is not None else np.nan)
                else:
                    p_values.append(np.nan)
            p_value_matrix.append(p_values)
        
        # DataFrameに変換
        p_value_df = pd.DataFrame(
            p_value_matrix,
            index=[pair.replace('_vs_', '\nvs\n') for pair in comparison_pairs],
            columns=metric_columns
        )
        
        # 図を作成
        fig, ax = plt.subplots(figsize=(max(12, len(metric_columns)), 
                                        max(6, len(comparison_pairs))))
        
        # ヒートマップを描画
        sns.heatmap(
            p_value_df,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=0.1,
            cbar_kws={'label': 'p-value'},
            ax=ax,
            linewidths=0.5,
            linecolor='gray'
        )
        
        # タイトルとラベルの設定
        ax.set_title('Statistical Test Results (p-values)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Comparisons', fontsize=12, fontweight='bold')
        
        # x軸ラベルの回転
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # レイアウトの調整
        plt.tight_layout()
        
        # 保存
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ヒートマップを保存しました: {output_path}")
        except Exception as e:
            logger.error(f"ヒートマップの保存に失敗しました: {e}")
            raise
        finally:
            plt.close()
    
    def _identify_log_scale_metrics(
        self,
        data: pd.DataFrame,
        metric_columns: List[str],
        threshold: float = LOG_SCALE_THRESHOLD
    ) -> List[str]:
        """
        対数軸を適用すべきメトリクスを自動判定
        
        Args:
            data (pd.DataFrame): データフレーム
            metric_columns (List[str]): メトリクスのカラム名リスト
            threshold (float): 最大値/最小値の比率の閾値
            
        Returns:
            List[str]: 対数軸を適用するメトリクスのリスト
        """
        log_scale_metrics = []
        
        for metric in metric_columns:
            if metric not in data.columns:
                continue
            
            metric_data = data[metric].dropna()
            
            if len(metric_data) == 0:
                continue
            
            # 負の値や0を含む場合は対数軸を適用しない
            if (metric_data <= 0).any():
                continue
            
            min_val = metric_data.min()
            max_val = metric_data.max()
            
            # 最小値が0に近い場合はスキップ
            if min_val < 1e-10:
                continue
            
            # 範囲比率を計算
            range_ratio = max_val / min_val
            
            if range_ratio > threshold:
                log_scale_metrics.append(metric)
                logger.info(
                    f"メトリクス '{metric}' に対数軸を適用 "
                    f"(範囲比: {range_ratio:.1f})"
                )
        
        return log_scale_metrics
    
    def create_summary_plot(
        self,
        statistics: Dict[str, Dict[str, Dict[str, float]]],
        output_path: Path,
        metric_columns: List[str]
    ):
        """
        メトリクスの平均値比較プロットを作成
        
        Args:
            statistics (Dict): 統計量の辞書
            output_path (Path): 出力先パス
            metric_columns (List[str]): メトリクスのカラム名リスト
        """
        groups = list(statistics.keys())
        
        # データの準備
        mean_values = {group: [] for group in groups}
        std_values = {group: [] for group in groups}
        
        for metric in metric_columns:
            for group in groups:
                if metric in statistics[group]:
                    mean_values[group].append(statistics[group][metric]['mean'])
                    std_values[group].append(statistics[group][metric]['std'])
                else:
                    mean_values[group].append(0)
                    std_values[group].append(0)
        
        # 図を作成
        x = np.arange(len(metric_columns))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 各グループのバーを描画
        colors = sns.color_palette('Set2', len(groups))
        for i, group in enumerate(groups):
            offset = width * (i - len(groups) / 2 + 0.5)
            ax.bar(x + offset, mean_values[group], width,
                  label=group, color=colors[i], 
                  yerr=std_values[group], capsize=3, alpha=0.8)
        
        # ラベルとタイトルの設定
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
        ax.set_title('Mean Comparison across Groups', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_columns, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # レイアウトの調整
        plt.tight_layout()
        
        # 保存
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"サマリープロットを保存しました: {output_path}")
        except Exception as e:
            logger.error(f"サマリープロットの保存に失敗しました: {e}")
            raise
        finally:
            plt.close()
