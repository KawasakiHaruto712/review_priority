"""
ヒートマップジェネレーターモジュール
統計検定の結果（p値、効果量）をヒートマップで可視化する
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# グラフのスタイル設定
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'


def generate_heatmap(
    test_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path,
    metric_filter: List[str] = None,
    alpha: float = 0.05
) -> Path:
    """
    統計検定結果（p値）のヒートマップを生成
    
    Args:
        test_results: perform_all_statistical_tests()の結果
        output_dir: 出力ディレクトリ
        metric_filter: 可視化対象メトリクスのリスト（省略時は全て）
        alpha: 有意水準（有意なセルを強調するため）
    
    Returns:
        Path: 生成された画像ファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データの整形
    # 行: メトリクス, 列: 比較ペア
    data_records = []
    
    metrics = list(test_results.keys())
    if metric_filter:
        metrics = [m for m in metrics if m in metric_filter]
        
    if not metrics:
        logger.warning("ヒートマップ用のメトリクスがありません")
        return None
        
    # 比較ペアのリストを取得（最初のメトリクスから）
    first_metric = metrics[0]
    pairs = list(test_results[first_metric].keys())
    
    # ペア名を短縮・整形
    pair_labels = {}
    for pair in pairs:
        # 例: early_core_reviewed_vs_late_core_reviewed -> Core: Early vs Late
        # 簡易的な置換ロジック
        label = pair.replace('_reviewed', '').replace('_', ' ')
        pair_labels[pair] = label
    
    # p値のマトリックスを作成
    p_values = []
    effect_sizes = []
    
    for metric in metrics:
        p_row = []
        es_row = []
        for pair in pairs:
            result = test_results.get(metric, {}).get(pair, {})
            
            p_val = result.get('p_value')
            if p_val is None:
                p_val = np.nan
                
            es = result.get('effect_size')
            if es is None:
                es = np.nan
            
            p_row.append(p_val)
            es_row.append(es)
            
        p_values.append(p_row)
        effect_sizes.append(es_row)
    
    # DataFrame化
    df_p = pd.DataFrame(p_values, index=metrics, columns=[pair_labels[p] for p in pairs]).astype(float)
    df_es = pd.DataFrame(effect_sizes, index=metrics, columns=[pair_labels[p] for p in pairs]).astype(float)
    
    if df_p.empty:
        logger.warning("ヒートマップ用のデータが空です")
        return None
        
    # プロット作成
    plt.figure(figsize=(15, len(metrics) * 0.8 + 2))
    
    # p値のヒートマップ
    # 色分け: 有意水準未満なら濃い色、そうでなければ薄い色
    # カスタムカラーマップ: 緑（有意）〜白（非有意）
    
    # p値を対数変換して色の差を出しやすくする（オプション）
    # ここでは単純にp < 0.05, p < 0.01, p < 0.001 で色分けするアプローチも可
    
    # p値のアノテーション用テキスト作成
    annot_text = df_p.map(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
    
    # 有意なセルを強調するためのマスク
    mask_sig = df_p < alpha
    
    # ヒートマップ描画
    # p値が小さいほど濃い赤にする（逆転）
    sns.heatmap(
        df_p,
        annot=annot_text,
        fmt="",
        cmap="Reds_r",  # 小さい値（有意）ほど濃い赤
        vmin=0,
        vmax=0.1,       # 0.1以上はすべて薄い色
        cbar_kws={'label': 'p-value'},
        linewidths=.5,
        linecolor='gray'
    )
    
    # 有意なセルに枠線を付けるなどの強調も可能だが、ここではシンプルに
    
    plt.title(f'Statistical Significance (p-values, alpha={alpha})', fontsize=16)
    plt.tight_layout()
    
    output_path = output_dir / 'heatmap_pvalues.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"p値ヒートマップを生成しました: {output_path}")
    
    # 効果量のヒートマップも生成（オプション）
    plt.figure(figsize=(15, len(metrics) * 0.8 + 2))
    
    annot_es = df_es.map(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
    
    sns.heatmap(
        df_es.abs(),  # 絶対値で大きさを見る
        annot=annot_es,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1.0,     # 効果量1.0以上は濃い青
        cbar_kws={'label': "Effect Size (Cohen's d)"},
        linewidths=.5,
        linecolor='gray'
    )
    
    plt.title("Effect Sizes (Cohen's d)", fontsize=16)
    plt.tight_layout()
    
    output_path_es = output_dir / 'heatmap_effect_sizes.pdf'
    plt.savefig(output_path_es, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"効果量ヒートマップを生成しました: {output_path_es}")
    
    return output_path


def generate_aggregated_heatmap(
    overall_results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
    metric_filter: List[str] = None,
    alpha: float = 0.05
) -> Path:
    """
    全リリースの統計検定結果（p値、効果量）をまとめたヒートマップテーブルを生成
    Early vs Late の比較に特化して表示する
    
    複数のテーブルを生成:
    1. 統合版（Reviewed + Not Reviewed）
    2. Reviewed のみ
    3. Not Reviewed のみ
    
    Args:
        overall_results: {release: {metric: {pair: {'p_value': ..., 'effect_size': ...}}}}
        output_dir: 出力ディレクトリ
        metric_filter: 可視化対象メトリクスのリスト
        alpha: 有意水準
        
    Returns:
        Path: 生成された画像ファイルのパス（統合版）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    releases = list(overall_results.keys())
    
    # 全てのメトリクスを収集（順序を保持）
    all_metrics = []
    metrics_seen = set()
    for release in releases:
        res_release = overall_results[release]
        for metric in res_release.keys():
            if metric_filter and metric not in metric_filter:
                continue
            if metric not in metrics_seen:
                all_metrics.append(metric)
                metrics_seen.add(metric)
    sorted_metrics = all_metrics

    # 1. 統合版テーブルを生成
    target_pairs_combined = {
        'early_reviewed_vs_late_reviewed': 'Reviewed',
        'early_not_reviewed_vs_late_not_reviewed': 'Not Reviewed'
    }
    output_path_combined = _generate_single_heatmap(
        overall_results, releases, sorted_metrics, target_pairs_combined,
        output_dir, 'aggregated_heatmap_table.pdf', 
        "Early vs Late Comparison (p-values & Effect Sizes)", alpha
    )
    
    # 2. Reviewed のみのテーブルを生成
    target_pairs_reviewed = {
        'early_reviewed_vs_late_reviewed': 'Reviewed'
    }
    _generate_single_heatmap(
        overall_results, releases, sorted_metrics, target_pairs_reviewed,
        output_dir, 'aggregated_heatmap_table_reviewed.pdf',
        "Early vs Late Comparison - Reviewed Only", alpha
    )
    
    # 3. Not Reviewed のみのテーブルを生成
    target_pairs_not_reviewed = {
        'early_not_reviewed_vs_late_not_reviewed': 'Not Reviewed'
    }
    _generate_single_heatmap(
        overall_results, releases, sorted_metrics, target_pairs_not_reviewed,
        output_dir, 'aggregated_heatmap_table_not_reviewed.pdf',
        "Early vs Late Comparison - Not Reviewed Only", alpha
    )
    
    return output_path_combined


def _generate_single_heatmap(
    overall_results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    releases: List[str],
    sorted_metrics: List[str],
    target_pairs: Dict[str, str],
    output_dir: Path,
    filename: str,
    title: str,
    alpha: float = 0.05
) -> Path:
    """
    単一のヒートマップテーブルを生成するヘルパー関数
    
    Args:
        overall_results: 全リリースの統計検定結果
        releases: リリースのリスト
        sorted_metrics: ソート済みメトリクスのリスト
        target_pairs: 表示対象のペア定義 {pair_key: label}
        output_dir: 出力ディレクトリ
        filename: 出力ファイル名
        title: グラフタイトル
        alpha: 有意水準
        
    Returns:
        Path: 生成された画像ファイルのパス
    """
    # 列の定義 (Metric -> Category -> Stat)
    columns = []
    for metric in sorted_metrics:
        for pair_key, pair_label in target_pairs.items():
            columns.append((metric, pair_label, 'p-value'))
            columns.append((metric, pair_label, 'effect_size'))
            
    # データ埋め込み
    data = []
    for release in releases:
        row = []
        for metric in sorted_metrics:
            for pair_key, pair_label in target_pairs.items():
                # 値の取得
                val_dict = overall_results.get(release, {}).get(metric, {}).get(pair_key, {})
                p_val = val_dict.get('p_value', np.nan)
                es = val_dict.get('effect_size', np.nan)
                
                row.append(p_val)
                row.append(es)
        data.append(row)
        
    if not data:
        logger.warning("集約ヒートマップ用のデータがありません")
        return None

    # DataFrame作成
    df = pd.DataFrame(data, index=releases, columns=pd.MultiIndex.from_tuples(columns, names=['Metric', 'Category', 'Stat']))
    
    # プロット描画
    cell_width = 1.2
    cell_height = 0.6
    
    n_rows = len(releases)
    n_cols = len(columns)
    
    width = max(n_cols * cell_width + 2, 10)
    height = max(n_rows * cell_height + 4, 6)
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # 軸の設定
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    
    # カラーマップ
    cmap_p = plt.cm.Reds
    cmap_es = plt.cm.Blues
    
    # 描画ループ
    for i, release in enumerate(releases):
        # 行ラベル (リリース名)
        ax.text(-0.2, i + 0.5, release, ha='right', va='center', fontweight='bold', fontsize=11)
        
        for j, col_tuple in enumerate(columns):
            metric, category, stat_type = col_tuple
            val = df.iloc[i, j]
            
            if pd.isna(val):
                color = 'white'
                text_val = "-"
            else:
                if stat_type == 'p-value':
                    if val > alpha:
                        color = 'white'
                    else:
                        norm_val = 1.0 - (val / alpha)
                        norm_val = 0.2 + 0.8 * norm_val
                        color = cmap_p(norm_val)
                    
                    text_val = f"{val:.3f}"
                    if val < 0.001:
                        text_val = "<.001"
                        
                else: # effect_size
                    norm_val = min(abs(val), 1.0)
                    color = cmap_es(norm_val)
                    text_val = f"{val:.2f}"
            
            # セルの描画
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
            
            # テキスト描画
            text_color = 'black'
            if stat_type == 'effect_size' and abs(val) > 0.5:
                text_color = 'white'
            if stat_type == 'p-value' and val < 0.01:
                text_color = 'white'
                
            ax.text(j + 0.5, i + 0.5, text_val, ha='center', va='center', color=text_color, fontsize=9)

    # ヘッダー描画
    # Metric (最上段)
    current_metric = None
    start_col = 0
    header_y_metric = -1.8
    header_y_category = -1.0
    header_y_stat = -0.3
    
    for j, col_tuple in enumerate(columns):
        metric = col_tuple[0]
        if metric != current_metric:
            if current_metric is not None:
                center = (start_col + j) / 2
                ax.text(center, header_y_metric, current_metric, ha='center', va='center', fontweight='bold', fontsize=11)
                # メトリクス区切り線
                ax.plot([j, j], [header_y_metric - 0.5, n_rows], color='black', linewidth=1.5)
            
            current_metric = metric
            start_col = j
            
    # 最後のメトリクス
    center = (start_col + n_cols) / 2
    ax.text(center, header_y_metric, current_metric, ha='center', va='center', fontweight='bold', fontsize=11)

    # Category (2段目: Reviewed / Not Reviewed)
    current_category = None
    cat_start_col = 0
    for j, col_tuple in enumerate(columns):
        category = col_tuple[1]
        # カテゴリが変わるか、メトリクスが変わったら描画
        if category != current_category or (j > 0 and columns[j][0] != columns[j-1][0]):
            if current_category is not None and (j > cat_start_col):
                center = (cat_start_col + j) / 2
                ax.text(center, header_y_category, current_category, ha='center', va='center', fontsize=10)
                # カテゴリ区切り線
                ax.plot([j, j], [header_y_category - 0.3, n_rows], color='gray', linewidth=1.0)
            
            current_category = category
            cat_start_col = j
            
    # 最後のカテゴリ
    center = (cat_start_col + n_cols) / 2
    ax.text(center, header_y_category, current_category, ha='center', va='center', fontsize=10)

    # Stat (3段目: p-value / effect size)
    for j, col_tuple in enumerate(columns):
        stat_type = col_tuple[2]
        label = "p-value" if stat_type == 'p-value' else "effect size"
        ax.text(j + 0.5, header_y_stat, label, ha='center', va='center', fontsize=9)
        # Stat区切り線
        if j > 0:
             ax.plot([j, j], [0, n_rows], color='lightgray', linewidth=0.5)

    # 外枠
    ax.plot([0, n_cols], [0, 0], color='black', linewidth=1.5)
    ax.plot([0, n_cols], [n_rows, n_rows], color='black', linewidth=1.5)
    ax.plot([0, 0], [header_y_metric - 0.5, n_rows], color='black', linewidth=1.5)
    ax.plot([n_cols, n_cols], [header_y_metric - 0.5, n_rows], color='black', linewidth=1.5)
    
    # ヘッダーの横線
    ax.plot([0, n_cols], [header_y_category - 0.4, header_y_category - 0.4], color='black', linewidth=1) # Metricの下
    ax.plot([0, n_cols], [header_y_stat - 0.4, header_y_stat - 0.4], color='gray', linewidth=1) # Categoryの下

    plt.title(title, y=1.2, fontsize=14)
    
    # 余白調整
    plt.subplots_adjust(left=0.15, right=0.98, top=0.8, bottom=0.05)
    
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"集約ヒートマップテーブルを生成しました: {output_path}")
    return output_path
