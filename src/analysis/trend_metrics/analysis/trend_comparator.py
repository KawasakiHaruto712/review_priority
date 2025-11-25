"""
トレンド比較モジュール
前期/後期の変化を分析し、レビューアタイプ別の傾向を比較する
"""

import logging
from typing import Dict, List, Any, Tuple
import numpy as np

from src.analysis.trend_metrics.utils.constants import METRIC_COLUMNS

logger = logging.getLogger(__name__)


def extract_period_from_group_name(group_name: str) -> str:
    """
    グループ名から期間（early/late）を抽出
    
    Args:
        group_name: グループ名（例: 'early_core_reviewed'）
    
    Returns:
        str: 期間 ('early' or 'late')
    """
    if group_name.startswith('early_'):
        return 'early'
    elif group_name.startswith('late_'):
        return 'late'
    else:
        return 'unknown'


def extract_reviewer_type_from_group_name(group_name: str) -> str:
    """
    グループ名からレビューアタイプを抽出
    
    Args:
        group_name: グループ名（例: 'early_core_reviewed'）
    
    Returns:
        str: レビューアタイプ（例: 'core_reviewed', 'core_not_reviewed'など）
    """
    if group_name.startswith('early_'):
        return group_name[6:]
    elif group_name.startswith('late_'):
        return group_name[5:]
    else:
        return 'unknown'


def calculate_metric_change(
    early_values: List[float],
    late_values: List[float]
) -> Dict[str, Any]:
    """
    前期→後期のメトリクス変化を計算
    
    Args:
        early_values: 前期の値リスト
        late_values: 後期の値リスト
    
    Returns:
        Dict[str, Any]: 変化の統計情報
        {
            'early_mean': 前期の平均,
            'late_mean': 後期の平均,
            'change_absolute': 絶対変化量,
            'change_percentage': 変化率（%）,
            'direction': 変化の方向 ('increase', 'decrease', 'no_change'),
            'early_n': 前期のサンプル数,
            'late_n': 後期のサンプル数
        }
    """
    if not early_values or not late_values:
        return {
            'early_mean': None,
            'late_mean': None,
            'change_absolute': None,
            'change_percentage': None,
            'direction': 'unknown',
            'early_n': len(early_values) if early_values else 0,
            'late_n': len(late_values) if late_values else 0
        }
    
    early_mean = np.mean(early_values)
    late_mean = np.mean(late_values)
    
    change_absolute = late_mean - early_mean
    
    # 変化率の計算（ゼロ除算を回避）
    if early_mean != 0:
        change_percentage = (change_absolute / early_mean) * 100
    else:
        change_percentage = None
    
    # 変化の方向を判定
    if abs(change_absolute) < 1e-6:  # ほぼゼロ
        direction = 'no_change'
    elif change_absolute > 0:
        direction = 'increase'
    else:
        direction = 'decrease'
    
    return {
        'early_mean': float(early_mean),
        'late_mean': float(late_mean),
        'change_absolute': float(change_absolute),
        'change_percentage': float(change_percentage) if change_percentage is not None else None,
        'direction': direction,
        'early_n': len(early_values),
        'late_n': len(late_values)
    }


def compare_early_vs_late_by_reviewer_type(
    groups: Dict[str, List[Dict]],
    metrics: List[str] = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    レビューアタイプ別に前期/後期を比較
    
    Args:
        groups: グループ別のChangeデータ
        metrics: 比較対象メトリクスのリスト（省略時はMETRIC_COLUMNS使用）
    
    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: レビューアタイプ別・メトリクス別の比較結果
        例: {
            'core_reviewed': {
                'lines_added': {
                    'early_mean': 145.2,
                    'late_mean': 132.5,
                    'change_percentage': -8.7,
                    'direction': 'decrease',
                    ...
                },
                ...
            },
            'non_core_reviewed': {...},
            'core_not_reviewed': {...},
            'non_core_not_reviewed': {...}
        }
    """
    if metrics is None:
        metrics = METRIC_COLUMNS
    
    # レビューアタイプごとにグループをまとめる
    reviewer_types = {}
    
    for group_name, changes in groups.items():
        reviewer_type = extract_reviewer_type_from_group_name(group_name)
        period = extract_period_from_group_name(group_name)
        
        if reviewer_type not in reviewer_types:
            reviewer_types[reviewer_type] = {'early': None, 'late': None}
        
        reviewer_types[reviewer_type][period] = changes
    
    # レビューアタイプ別に比較
    comparison_results = {}
    
    for reviewer_type, periods in reviewer_types.items():
        early_changes = periods.get('early', [])
        late_changes = periods.get('late', [])
        
        if not early_changes or not late_changes:
            logger.warning(
                f"レビューアタイプ {reviewer_type} の前期または後期データがありません"
            )
            continue
        
        type_results = {}
        
        for metric in metrics:
            # メトリクス値を抽出
            early_values = [
                change.get(metric)
                for change in early_changes
                if change.get(metric) is not None
            ]
            late_values = [
                change.get(metric)
                for change in late_changes
                if change.get(metric) is not None
            ]
            
            # 変化を計算
            change_info = calculate_metric_change(early_values, late_values)
            type_results[metric] = change_info
        
        comparison_results[reviewer_type] = type_results
        logger.info(
            f"レビューアタイプ {reviewer_type} の前期/後期比較を完了 "
            f"(前期: {len(early_changes)} 件, 後期: {len(late_changes)} 件)"
        )
    
    return comparison_results


def identify_diverging_trends(
    comparison_results: Dict[str, Dict[str, Dict[str, Any]]],
    threshold_percentage: float = 10.0
) -> Dict[str, Dict[str, Any]]:
    """
    レビューアタイプ間で傾向が異なるメトリクスを特定
    
    Args:
        comparison_results: compare_early_vs_late_by_reviewer_type()の結果
        threshold_percentage: 変化率の閾値（%）
    
    Returns:
        Dict[str, Dict[str, Any]]: 乖離しているメトリクスの情報
        例: {
            'lines_added': {
                'core_reviewed_change': -8.7,
                'non_core_reviewed_change': 17.0,
                'divergence': 25.7,
                'interpretation': 'core_reviewedは減少、non_core_reviewedは増加'
            },
            ...
        }
    """
    diverging = {}
    
    # メトリクス一覧を取得（最初のレビューアタイプから）
    first_type = list(comparison_results.keys())[0]
    metrics = list(comparison_results[first_type].keys())
    
    for metric in metrics:
        changes_by_type = {}
        
        # 各レビューアタイプの変化率を収集
        for reviewer_type, type_results in comparison_results.items():
            metric_info = type_results.get(metric, {})
            change_pct = metric_info.get('change_percentage')
            
            if change_pct is not None:
                changes_by_type[reviewer_type] = {
                    'change_percentage': change_pct,
                    'direction': metric_info.get('direction')
                }
        
        if len(changes_by_type) < 2:
            continue
        
        # 変化率の範囲（最大 - 最小）を計算
        change_values = [info['change_percentage'] for info in changes_by_type.values()]
        divergence = max(change_values) - min(change_values)
        
        # 閾値を超える乖離があれば記録
        if divergence > threshold_percentage:
            # 解釈を生成
            interpretations = []
            for reviewer_type, info in changes_by_type.items():
                direction = info['direction']
                change_pct = info['change_percentage']
                interpretations.append(
                    f"{reviewer_type}は{direction}（{change_pct:+.1f}%）"
                )
            
            diverging[metric] = {
                **changes_by_type,
                'divergence': divergence,
                'interpretation': '、'.join(interpretations)
            }
    
    logger.info(f"{len(diverging)} メトリクスで乖離する傾向が検出されました")
    
    return diverging


def calculate_group_size_changes(
    groups: Dict[str, List[Dict]]
) -> Dict[str, Dict[str, Any]]:
    """
    グループサイズの変化を計算（前期→後期）
    
    Args:
        groups: グループ別のChangeデータ
    
    Returns:
        Dict[str, Dict[str, Any]]: レビューアタイプ別のサイズ変化
        例: {
            'core_reviewed': {
                'early_count': 105,
                'late_count': 92,
                'change_absolute': -13,
                'change_percentage': -12.4
            },
            ...
        }
    """
    # レビューアタイプごとにサイズを集計
    type_counts = {}
    
    for group_name, changes in groups.items():
        reviewer_type = extract_reviewer_type_from_group_name(group_name)
        period = extract_period_from_group_name(group_name)
        
        if reviewer_type not in type_counts:
            type_counts[reviewer_type] = {'early': 0, 'late': 0}
        
        type_counts[reviewer_type][period] = len(changes)
    
    # 変化を計算
    size_changes = {}
    
    for reviewer_type, counts in type_counts.items():
        early_count = counts.get('early', 0)
        late_count = counts.get('late', 0)
        
        change_absolute = late_count - early_count
        
        if early_count != 0:
            change_percentage = (change_absolute / early_count) * 100
        else:
            change_percentage = None
        
        size_changes[reviewer_type] = {
            'early_count': early_count,
            'late_count': late_count,
            'change_absolute': change_absolute,
            'change_percentage': change_percentage
        }
    
    logger.info("グループサイズの変化を計算しました")
    
    return size_changes


def generate_trend_summary(
    comparison_results: Dict[str, Dict[str, Dict[str, Any]]],
    test_results: Dict[str, Dict[str, Dict[str, Any]]] = None,
    size_changes: Dict[str, Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    トレンド分析の総合サマリーを生成
    
    Args:
        comparison_results: compare_early_vs_late_by_reviewer_type()の結果
        test_results: perform_all_statistical_tests()の結果（オプション）
        size_changes: calculate_group_size_changes()の結果（オプション）
    
    Returns:
        Dict[str, Any]: 総合サマリー
    """
    summary = {
        'reviewer_type_comparison': comparison_results,
        'group_size_changes': size_changes,
        'key_findings': []
    }
    
    # 主要な発見をまとめる
    findings = []
    
    # 1. グループサイズの変化
    if size_changes:
        for reviewer_type, change_info in size_changes.items():
            change_pct = change_info.get('change_percentage')
            if change_pct is not None and abs(change_pct) > 10:
                findings.append({
                    'type': 'group_size_change',
                    'reviewer_type': reviewer_type,
                    'change_percentage': change_pct,
                    'description': f"{reviewer_type}のグループサイズが{change_pct:+.1f}%変化"
                })
    
    # 2. 有意な変化のあるメトリクス
    if test_results:
        for metric, pairs in test_results.items():
            significant_pairs = [
                pair for pair, result in pairs.items()
                if result.get('significant', False)
            ]
            if significant_pairs:
                findings.append({
                    'type': 'significant_change',
                    'metric': metric,
                    'significant_pairs': len(significant_pairs),
                    'description': f"{metric}で{len(significant_pairs)}ペアに有意な差"
                })
    
    summary['key_findings'] = findings
    
    logger.info(f"トレンド分析サマリーを生成しました（{len(findings)} 件の主要な発見）")
    
    return summary
