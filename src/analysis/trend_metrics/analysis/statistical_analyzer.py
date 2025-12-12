"""
統計分析モジュール
8グループのメトリクスデータに対して記述統計量を計算し、統計検定を実行する
"""

import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.trend_metrics.utils.constants import (
    METRIC_COLUMNS,
    STATISTICAL_TEST_CONFIG
)

logger = logging.getLogger(__name__)


def calculate_summary_statistics(
    groups: Dict[str, List[Dict]],
    metrics: List[str] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    各グループのメトリクスについて記述統計量を計算
    
    Args:
        groups: グループ別のChangeデータ
                例: {'early_core_reviewed': [...], 'late_core_reviewed': [...]}
        metrics: 計算対象メトリクスのリスト（省略時はMETRIC_COLUMNS使用）
    
    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: グループ別・メトリクス別の統計量
        例: {
            'early_core_reviewed': {
                'lines_added': {
                    'count': 105,
                    'mean': 145.2,
                    'std': 52.1,
                    'min': 10,
                    '25%': 80,
                    '50%': 130,
                    '75%': 190,
                    'max': 500
                },
                ...
            },
            ...
        }
    """
    if metrics is None:
        metrics = METRIC_COLUMNS
    
    summary = {}
    
    for group_name, changes in groups.items():
        if not changes:
            logger.warning(f"グループ {group_name} にデータがありません")
            summary[group_name] = {}
            continue
        
        group_stats = {}
        
        for metric in metrics:
            # メトリクス値を抽出（Noneや欠損値を除外）
            values = [
                change.get(metric)
                for change in changes
                if change.get(metric) is not None
            ]
            
            if not values:
                logger.warning(f"グループ {group_name} のメトリクス {metric} にデータがありません")
                group_stats[metric] = {
                    'count': 0,
                    'mean': None,
                    'std': None,
                    'min': None,
                    '25%': None,
                    '50%': None,
                    '75%': None,
                    'max': None
                }
                continue
            
            # 記述統計量を計算
            values_array = np.array(values)
            group_stats[metric] = {
                'count': len(values),
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array, ddof=1)) if len(values) > 1 else 0.0,
                'min': float(np.min(values_array)),
                '25%': float(np.percentile(values_array, 25, method='midpoint')),
                '50%': float(np.percentile(values_array, 50, method='midpoint')),
                '75%': float(np.percentile(values_array, 75, method='midpoint')),
                'max': float(np.max(values_array))
            }
        
        summary[group_name] = group_stats
        logger.info(f"グループ {group_name} の統計量を計算しました（{len(changes)} 件）")
    
    return summary


def mann_whitney_u_test(
    group1_values: List[float],
    group2_values: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Mann-Whitney U検定を実行
    
    Args:
        group1_values: グループ1の値リスト
        group2_values: グループ2の値リスト
        alpha: 有意水準（デフォルト: 0.05）
    
    Returns:
        Dict[str, Any]: 検定結果
        {
            'statistic': U統計量,
            'p_value': p値,
            'significant': 有意かどうか (p < alpha),
            'test_method': 'mann_whitney_u'
        }
    """
    if len(group1_values) < 2 or len(group2_values) < 2:
        return {
            'statistic': None,
            'p_value': None,
            'significant': False,
            'test_method': 'mann_whitney_u',
            'error': 'サンプル数が不足しています'
        }
    
    try:
        statistic, p_value = stats.mannwhitneyu(
            group1_values,
            group2_values,
            alternative='two-sided'
        )
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'test_method': 'mann_whitney_u'
        }
    except Exception as e:
        logger.error(f"Mann-Whitney U検定でエラー: {e}")
        return {
            'statistic': None,
            'p_value': None,
            'significant': False,
            'test_method': 'mann_whitney_u',
            'error': str(e)
        }


def calculate_effect_size_cohens_d(
    group1_values: List[float],
    group2_values: List[float]
) -> float:
    """
    Cohen's d（効果量）を計算
    
    Args:
        group1_values: グループ1の値リスト
        group2_values: グループ2の値リスト
    
    Returns:
        float: Cohen's d（効果量）
    """
    if len(group1_values) < 2 or len(group2_values) < 2:
        return None
    
    try:
        group1_array = np.array(group1_values)
        group2_array = np.array(group2_values)
        
        mean1 = np.mean(group1_array)
        mean2 = np.mean(group2_array)
        
        std1 = np.std(group1_array, ddof=1)
        std2 = np.std(group2_array, ddof=1)
        
        n1 = len(group1_values)
        n2 = len(group2_values)
        
        # プールされた標準偏差
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean1 - mean2) / pooled_std
        
        return float(cohens_d)
    except Exception as e:
        logger.error(f"Cohen's d計算でエラー: {e}")
        return None


def _parse_group_name(group_name: str) -> Dict[str, str]:
    """
    グループ名を要素に分解する
    
    Args:
        group_name: グループ名
        
    Returns:
        Dict[str, str]: 要素の辞書 {'period': ..., 'reviewer_type': ..., 'review_status': ...}
        解析できない場合はNone
    """
    parts = {}
    
    # 1. Period (early/late)
    if group_name.startswith('early_'):
        parts['period'] = 'early'
        remainder = group_name[6:] # len('early_')
    elif group_name.startswith('late_'):
        parts['period'] = 'late'
        remainder = group_name[5:] # len('late_')
    else:
        return None

    # 2. Reviewer Type (core/non_core/all)
    if remainder.startswith('non_core_'):
        parts['reviewer_type'] = 'non_core'
        remainder = remainder[9:] # len('non_core_')
    elif remainder.startswith('core_'):
        parts['reviewer_type'] = 'core'
        remainder = remainder[5:] # len('core_')
    else:
        # SEPARATE_CORE_REVIEWERS=False の場合
        parts['reviewer_type'] = 'all'
        
    # 3. Review Status (reviewed/not_reviewed)
    if remainder == 'reviewed':
        parts['review_status'] = 'reviewed'
    elif remainder == 'not_reviewed':
        parts['review_status'] = 'not_reviewed'
    else:
        return None
        
    return parts


def _are_comparable_groups(group1: str, group2: str) -> bool:
    """
    2つのグループが比較可能か（要素が1つだけ異なるか）を判定する
    
    Args:
        group1: グループ名1
        group2: グループ名2
        
    Returns:
        bool: 比較可能な場合True
    """
    p1 = _parse_group_name(group1)
    p2 = _parse_group_name(group2)
    
    if not p1 or not p2:
        return False
        
    diff_count = 0
    if p1['period'] != p2['period']:
        diff_count += 1
    if p1['reviewer_type'] != p2['reviewer_type']:
        diff_count += 1
    if p1['review_status'] != p2['review_status']:
        diff_count += 1
    
    # 1つだけ異なる場合のみ比較対象とする
    return diff_count == 1


def perform_pairwise_tests(
    groups: Dict[str, List[Dict]],
    metric: str,
    test_config: Dict = None
) -> Dict[str, Dict[str, Any]]:
    """
    全グループ間でペアワイズ検定を実行
    ただし、要素が1つだけ異なるペアのみを比較対象とする
    
    Args:
        groups: グループ別のChangeデータ
        metric: 検定対象メトリクス
        test_config: 検定設定（省略時はSTATISTICAL_TEST_CONFIG使用）
    
    Returns:
        Dict[str, Dict[str, Any]]: ペア別の検定結果
    """
    if test_config is None:
        test_config = STATISTICAL_TEST_CONFIG
    
    alpha = test_config.get('alpha', 0.05)
    use_bonferroni = test_config.get('use_bonferroni', True)
    
    group_names = list(groups.keys())
    results = {}
    
    # 比較対象となるペアを抽出
    comparable_pairs = []
    for i, group1_name in enumerate(group_names):
        for group2_name in group_names[i+1:]:
            if _are_comparable_groups(group1_name, group2_name):
                comparable_pairs.append((group1_name, group2_name))
    
    # ペア数を計算（Bonferroni補正用）
    # 実際に比較を行うペア数で補正する
    num_comparisons = len(comparable_pairs)
    if num_comparisons == 0:
        logger.warning(f"メトリクス {metric} について比較可能なペアが見つかりませんでした")
        return {}
        
    adjusted_alpha = alpha / num_comparisons if use_bonferroni else alpha
    
    # 抽出したペアで検定
    for group1_name, group2_name in comparable_pairs:
        # メトリクス値を抽出
        group1_values = [
            change.get(metric)
            for change in groups[group1_name]
            if change.get(metric) is not None
        ]
        group2_values = [
            change.get(metric)
            for change in groups[group2_name]
            if change.get(metric) is not None
        ]
        
        # 検定を実行
        test_result = mann_whitney_u_test(group1_values, group2_values, adjusted_alpha)
        
        # 効果量を計算
        effect_size = calculate_effect_size_cohens_d(group1_values, group2_values)
        
        # 結果を保存
        pair_key = f"{group1_name}_vs_{group2_name}"
        results[pair_key] = {
            **test_result,
            'effect_size': effect_size,
            'alpha': adjusted_alpha,
            'bonferroni_corrected': use_bonferroni,
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_n': len(group1_values),
            'group2_n': len(group2_values)
        }
    
    logger.info(
        f"メトリクス {metric} について {num_comparisons} ペアの検定を完了しました "
        f"(adjusted α={adjusted_alpha:.4f})"
    )
    
    return results


def perform_all_statistical_tests(
    groups: Dict[str, List[Dict]],
    metrics: List[str] = None,
    test_config: Dict = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    全メトリクスについて統計検定を実行
    
    Args:
        groups: グループ別のChangeデータ
        metrics: 検定対象メトリクスのリスト（省略時はMETRIC_COLUMNS使用）
        test_config: 検定設定（省略時はSTATISTICAL_TEST_CONFIG使用）
    
    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: メトリクス別・ペア別の検定結果
        例: {
            'lines_added': {
                'early_core_reviewed_vs_late_core_reviewed': {...},
                ...
            },
            ...
        }
    """
    if metrics is None:
        metrics = METRIC_COLUMNS
    
    if test_config is None:
        test_config = STATISTICAL_TEST_CONFIG
    
    all_results = {}
    
    for metric in metrics:
        logger.info(f"メトリクス {metric} の検定を開始...")
        metric_results = perform_pairwise_tests(groups, metric, test_config)
        all_results[metric] = metric_results
    
    logger.info(f"全メトリクス ({len(metrics)} 種類) の検定が完了しました")
    
    return all_results


def summarize_significant_results(
    test_results: Dict[str, Dict[str, Dict[str, Any]]],
    alpha: float = 0.05
) -> Dict[str, List[Dict[str, Any]]]:
    """
    有意な結果をサマリー
    
    Args:
        test_results: perform_all_statistical_tests()の結果
        alpha: 有意水準
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: メトリクス別の有意な結果リスト
    """
    summary = {}
    
    for metric, pairs in test_results.items():
        significant_pairs = []
        
        for pair_name, result in pairs.items():
            if result.get('significant', False):
                significant_pairs.append({
                    'pair': pair_name,
                    'p_value': result.get('p_value'),
                    'effect_size': result.get('effect_size'),
                    'group1_name': result.get('group1_name'),
                    'group2_name': result.get('group2_name')
                })
        
        if significant_pairs:
            # p値でソート
            significant_pairs.sort(key=lambda x: x['p_value'])
            summary[metric] = significant_pairs
    
    logger.info(f"{len(summary)} メトリクスで有意な差が検出されました")
    
    return summary


def interpret_effect_size(cohens_d: float) -> str:
    """
    Cohen's dの効果量を解釈
    
    Args:
        cohens_d: Cohen's d値
    
    Returns:
        str: 効果量の解釈 ('small', 'medium', 'large', 'very large')
    """
    if cohens_d is None:
        return 'unknown'
    
    abs_d = abs(cohens_d)
    
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    elif abs_d < 1.2:
        return 'large'
    else:
        return 'very large'
