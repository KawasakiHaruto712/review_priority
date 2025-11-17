"""
統計分析モジュール
Mann-Whitney U検定と記述統計量の計算を行う
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy import stats
from pathlib import Path
import json

from src.config.release_constants import SIGNIFICANCE_LEVEL

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    統計分析を行うクラス
    記述統計量の計算とMann-Whitney U検定を実行
    """
    
    def __init__(self, significance_level: float = SIGNIFICANCE_LEVEL):
        """
        Args:
            significance_level (float): 統計検定の有意水準（デフォルト: 0.05）
        """
        self.significance_level = significance_level
    
    def calculate_descriptive_statistics(
        self, 
        data: pd.DataFrame, 
        group_column: str,
        metric_columns: List[str]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        グループごとに記述統計量を計算
        
        Args:
            data (pd.DataFrame): データフレーム
            group_column (str): グループ化するカラム名
            metric_columns (List[str]): 統計量を計算するメトリクスのカラム名リスト
            
        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: グループ -> メトリクス -> 統計量の辞書
        """
        statistics = {}
        
        for group_name in data[group_column].unique():
            group_data = data[data[group_column] == group_name]
            statistics[group_name] = {}
            
            for metric in metric_columns:
                if metric not in group_data.columns:
                    logger.warning(f"メトリクス '{metric}' がデータに存在しません")
                    continue
                
                metric_data = group_data[metric].dropna()
                
                if len(metric_data) == 0:
                    logger.warning(f"グループ '{group_name}' のメトリクス '{metric}' にデータがありません")
                    continue
                
                statistics[group_name][metric] = {
                    'count': int(len(metric_data)),
                    'mean': float(metric_data.mean()),
                    'std': float(metric_data.std()),
                    'min': float(metric_data.min()),
                    '25%': float(metric_data.quantile(0.25)),
                    '50%': float(metric_data.quantile(0.50)),  # median
                    '75%': float(metric_data.quantile(0.75)),
                    'max': float(metric_data.max())
                }
        
        return statistics
    
    def perform_mann_whitney_test(
        self,
        data: pd.DataFrame,
        group_column: str,
        group1_name: str,
        group2_name: str,
        metric_columns: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        2つのグループ間でMann-Whitney U検定を実行
        
        Args:
            data (pd.DataFrame): データフレーム
            group_column (str): グループ化するカラム名
            group1_name (str): グループ1の名前
            group2_name (str): グループ2の名前
            metric_columns (List[str]): 検定するメトリクスのカラム名リスト
            
        Returns:
            Dict[str, Dict[str, Any]]: メトリクス -> 検定結果の辞書
        """
        test_results = {}
        
        group1_data = data[data[group_column] == group1_name]
        group2_data = data[data[group_column] == group2_name]
        
        for metric in metric_columns:
            if metric not in data.columns:
                logger.warning(f"メトリクス '{metric}' がデータに存在しません")
                continue
            
            metric1 = group1_data[metric].dropna()
            metric2 = group2_data[metric].dropna()
            
            if len(metric1) < 3 or len(metric2) < 3:
                logger.warning(
                    f"メトリクス '{metric}' のサンプル数が不足しています "
                    f"(group1: {len(metric1)}, group2: {len(metric2)})"
                )
                test_results[metric] = {
                    'statistic': None,
                    'p_value': None,
                    'significant': False,
                    'effect_size': None,
                    'error': 'Insufficient sample size'
                }
                continue
            
            try:
                # Mann-Whitney U検定を実行
                statistic, p_value = stats.mannwhitneyu(
                    metric1, metric2, alternative='two-sided'
                )
                
                # 効果量(rank-biserial correlation)を計算
                effect_size = self._calculate_rank_biserial_correlation(
                    statistic, len(metric1), len(metric2)
                )
                
                test_results[metric] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': bool(p_value < self.significance_level),  # Convert to Python bool
                    'effect_size': float(effect_size),
                    'n1': int(len(metric1)),
                    'n2': int(len(metric2))
                }
                
            except Exception as e:
                logger.error(f"メトリクス '{metric}' の検定でエラー: {e}")
                test_results[metric] = {
                    'statistic': None,
                    'p_value': None,
                    'significant': False,
                    'effect_size': None,
                    'error': str(e)
                }
        
        return test_results
    
    def _calculate_rank_biserial_correlation(
        self, 
        u_statistic: float, 
        n1: int, 
        n2: int
    ) -> float:
        """
        Rank-biserial correlation（効果量）を計算
        
        Args:
            u_statistic (float): Mann-Whitney UのU統計量
            n1 (int): グループ1のサンプル数
            n2 (int): グループ2のサンプル数
            
        Returns:
            float: Rank-biserial correlation（-1 ~ 1の範囲）
        """
        # rank-biserial correlation = 1 - (2U)/(n1*n2)
        r = 1 - (2 * u_statistic) / (n1 * n2)
        return r
    
    def perform_multiple_comparisons(
        self,
        data: pd.DataFrame,
        group_column: str,
        metric_columns: List[str],
        comparison_pairs: List[tuple]
    ) -> Dict[str, Dict[str, Any]]:
        """
        複数のグループペア間で統計検定を実行
        
        Args:
            data (pd.DataFrame): データフレーム
            group_column (str): グループ化するカラム名
            metric_columns (List[str]): 検定するメトリクスのカラム名リスト
            comparison_pairs (List[tuple]): 比較するグループペアのリスト
                例: [('early_reviewed', 'late_reviewed'), ...]
            
        Returns:
            Dict[str, Dict[str, Any]]: 比較ペア -> メトリクス -> 検定結果の辞書
        """
        all_test_results = {}
        
        for group1, group2 in comparison_pairs:
            comparison_key = f"{group1}_vs_{group2}"
            logger.info(f"比較を実行中: {comparison_key}")
            
            test_results = self.perform_mann_whitney_test(
                data, group_column, group1, group2, metric_columns
            )
            
            all_test_results[comparison_key] = test_results
        
        return all_test_results
    
    def save_statistics(
        self,
        statistics: Dict[str, Dict[str, Dict[str, float]]],
        output_path: Path
    ):
        """
        記述統計量をJSONファイルに保存
        
        Args:
            statistics (Dict): 統計量の辞書
            output_path (Path): 出力先パス
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False)
            logger.info(f"記述統計量を保存しました: {output_path}")
        except Exception as e:
            logger.error(f"記述統計量の保存に失敗しました: {e}")
            raise
    
    def save_test_results(
        self,
        test_results: Dict[str, Dict[str, Any]],
        output_path: Path
    ):
        """
        検定結果をJSONファイルに保存
        
        Args:
            test_results (Dict): 検定結果の辞書
            output_path (Path): 出力先パス
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            logger.info(f"検定結果を保存しました: {output_path}")
        except Exception as e:
            logger.error(f"検定結果の保存に失敗しました: {e}")
            raise
