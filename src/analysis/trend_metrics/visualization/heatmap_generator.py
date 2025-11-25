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
