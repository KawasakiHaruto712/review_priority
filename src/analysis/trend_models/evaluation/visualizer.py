"""
Trend Models Analysis - 可視化

本モジュールでは、評価結果の可視化やレポート生成機能を提供します。
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非インタラクティブバックエンド
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from src.config.path import DEFAULT_DATA_DIR
from src.analysis.trend_models.evaluation.evaluator import (
    CVResult,
    RankingCVResult,
    summarize_cv_results,
    summarize_ranking_cv_results,
)
from src.analysis.trend_models.utils.constants import OUTPUT_DIR_NAME

logger = logging.getLogger(__name__)

# ロギング設定
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Visualizer:
    """
    評価結果の可視化を行うクラス
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        project_name: str = 'default',
        filename_suffix: str = None
    ):
        """
        Args:
            output_dir: 出力ディレクトリのパス
            project_name: プロジェクト名
            filename_suffix: ファイル名に追加するサフィックス（モデル名等）
        """
        if output_dir is None:
            output_dir = DEFAULT_DATA_DIR / 'analysis' / OUTPUT_DIR_NAME / project_name
        
        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self.filename_suffix = filename_suffix
        
        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
    
    def _generate_filename(self, prefix: str, extension: str) -> str:
        """ファイル名を生成する"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.filename_suffix:
            return f'{prefix}_{self.project_name}_{self.filename_suffix}_{timestamp}.{extension}'
        else:
            return f'{prefix}_{self.project_name}_{timestamp}.{extension}'
    
    def save_cv_detail(
        self,
        results: List[CVResult],
        filename: str = None
    ) -> Path:
        """
        交差検証の詳細結果をCSVとして保存する
        
        Args:
            results: 交差検証結果のリスト
            filename: 出力ファイル名（Noneの場合は自動生成）
            
        Returns:
            Path: 出力ファイルパス
        """
        if filename is None:
            filename = self._generate_filename('cv_detail', 'csv')
        
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # feature_importancesカラムはJSON文字列に変換
        if 'feature_importances' in df.columns:
            df['feature_importances'] = df['feature_importances'].apply(
                lambda x: json.dumps(x) if x else ''
            )
        
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"詳細結果を保存: {output_path}")
        return output_path
    
    def save_cv_summary(
        self,
        results: List[CVResult],
        filename: str = None
    ) -> Path:
        """
        交差検証のサマリー結果をCSVとして保存する
        
        Args:
            results: 交差検証結果のリスト
            filename: 出力ファイル名（Noneの場合は自動生成）
            
        Returns:
            Path: 出力ファイルパス
        """
        if filename is None:
            filename = self._generate_filename('cv_summary', 'csv')
        
        summary_df = summarize_cv_results(results)
        
        output_path = self.output_dir / filename
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"サマリー結果を保存: {output_path}")
        return output_path
    
    def save_results_json(
        self,
        results: List[CVResult],
        metadata: Dict[str, Any] = None,
        filename: str = None
    ) -> Path:
        """
        全体の結果をJSONとして保存する
        
        Args:
            results: 交差検証結果のリスト
            metadata: メタデータ
            filename: 出力ファイル名（Noneの場合は自動生成）
            
        Returns:
            Path: 出力ファイルパス
        """
        if filename is None:
            filename = self._generate_filename('results', 'json')
        
        # サマリーを計算
        summary_df = summarize_cv_results(results)
        
        output_data = {
            'cross_validation': {
                'n_results': len(results),
                'summary': summary_df.to_dict('records'),
                'detail': [r.to_dict() for r in results]
            },
            'metadata': metadata or {
                'project': self.project_name,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"全体結果を保存: {output_path}")
        return output_path
    
    def plot_cross_period_heatmap(
        self,
        results: List[CVResult],
        metric: str = 'f1',
        filename: str = None
    ) -> Optional[Path]:
        """
        期間交差評価のヒートマップを作成する
        
        Args:
            results: 交差検証結果のリスト
            metric: 表示するメトリクス ('precision', 'recall', 'f1')
            filename: 出力ファイル名（Noneの場合は自動生成）
            
        Returns:
            Path: 出力ファイルパス、またはNone（描画ライブラリがない場合）
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            logger.warning("matplotlib または seaborn がインストールされていません")
            return None
        
        if filename is None:
            filename = self._generate_filename(f'heatmap_{metric}', 'png')
        
        # 結果をDataFrameに変換
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # developer_type='all' のデータのみ使用
        df = df[df['developer_type'] == 'all']
        
        # ピボットテーブルを作成
        pivot = df.pivot_table(
            index='eval_period',
            columns='train_period',
            values=metric,
            aggfunc='mean'
        )
        
        # ヒートマップを作成
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            ax=ax
        )
        ax.set_title(f'{metric.upper()} - Cross Period Evaluation ({self.project_name})')
        ax.set_xlabel('Train Period')
        ax.set_ylabel('Eval Period')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'figures' / filename
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"ヒートマップを保存: {output_path}")
        return output_path
    
    def plot_cv_results(
        self,
        results: List[CVResult],
        filename: str = None
    ) -> Optional[Path]:
        """
        交差検証結果の棒グラフを作成する
        
        Args:
            results: 交差検証結果のリスト
            filename: 出力ファイル名（Noneの場合は自動生成）
            
        Returns:
            Path: 出力ファイルパス、またはNone（描画ライブラリがない場合）
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib がインストールされていません")
            return None
        
        if filename is None:
            filename = self._generate_filename('cv_results', 'png')
        
        # 結果をDataFrameに変換
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # developer_type='all' のデータのみ使用
        df = df[df['developer_type'] == 'all']
        
        # 期間の組み合わせでグループ化
        df['period_combo'] = df['train_period'] + ' → ' + df['eval_period']
        
        # 集計
        grouped = df.groupby('period_combo').agg({
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean'
        }).reset_index()
        
        # 棒グラフを作成
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(grouped))
        width = 0.25
        
        ax.bar(x - width, grouped['precision'], width, label='Precision')
        ax.bar(x, grouped['recall'], width, label='Recall')
        ax.bar(x + width, grouped['f1'], width, label='F1')
        
        ax.set_xlabel('Period Combination (Train → Eval)')
        ax.set_ylabel('Score')
        ax.set_title(f'Cross Validation Results ({self.project_name})')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped['period_combo'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'figures' / filename
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"棒グラフを保存: {output_path}")
        return output_path
    
    def plot_feature_importances(
        self,
        feature_importances: Dict[str, float],
        filename: str = None,
        top_n: int = 10
    ) -> Optional[Path]:
        """
        特徴量重要度の棒グラフを作成する
        
        Args:
            feature_importances: 特徴量名と重要度の辞書
            filename: 出力ファイル名（Noneの場合は自動生成）
            top_n: 表示する上位特徴量の数
            
        Returns:
            Path: 出力ファイルパス、またはNone（描画ライブラリがない場合）
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib がインストールされていません")
            return None
        
        if filename is None:
            filename = self._generate_filename('feature_importance', 'png')
        
        # 上位N件を抽出
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        # 棒グラフを作成
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y = np.arange(len(features))
        ax.barh(y, importances)
        ax.set_yticks(y)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances ({self.project_name})')
        ax.invert_yaxis()  # 上位を上に
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'figures' / filename
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"特徴量重要度グラフを保存: {output_path}")
        return output_path

    def save_rank_detail(
        self,
        results: List[RankingCVResult],
        filename: str = None,
    ) -> Path:
        """ランキング評価の詳細結果を CSV 保存する。"""
        if filename is None:
            filename = self._generate_filename('rank_detail', 'csv')

        df = pd.DataFrame([result.to_dict() for result in results])
        if 'feature_importances' in df.columns:
            df['feature_importances'] = df['feature_importances'].apply(
                lambda value: json.dumps(value) if value else ''
            )

        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"ランキング詳細結果を保存: {output_path}")
        return output_path

    def save_rank_summary(
        self,
        results: List[RankingCVResult],
        filename: str = None,
    ) -> Path:
        """ランキング評価サマリーを CSV 保存する。"""
        if filename is None:
            filename = self._generate_filename('rank_summary', 'csv')

        summary_df = summarize_ranking_cv_results(results)
        output_path = self.output_dir / filename
        summary_df.to_csv(output_path, index=False)
        logger.info(f"ランキングサマリーを保存: {output_path}")
        return output_path

    def save_ranking_results_json(
        self,
        results: List[RankingCVResult],
        metadata: Dict[str, Any] = None,
        filename: str = None,
    ) -> Path:
        """ランキング評価結果を JSON 保存する。"""
        if filename is None:
            filename = self._generate_filename('ranking_results', 'json')

        summary_df = summarize_ranking_cv_results(results)
        output_data = {
            'ranking_validation': {
                'n_results': len(results),
                'summary': summary_df.to_dict('records'),
                'detail': [result.to_dict() for result in results],
            },
            'metadata': metadata or {
                'project': self.project_name,
                'timestamp': datetime.now().isoformat(),
            },
        }

        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"ランキング結果 JSON を保存: {output_path}")
        return output_path

    def plot_ranking_heatmap(
        self,
        results: List[RankingCVResult],
        metric: str = 'ndcg_at_10',
        filename: str = None,
    ) -> Optional[Path]:
        """ランキング評価のヒートマップを作成する。"""
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            logger.warning("matplotlib または seaborn がインストールされていません")
            return None

        if filename is None:
            filename = self._generate_filename(f'rank_heatmap_{metric}', 'png')

        df = pd.DataFrame([result.to_dict() for result in results])
        if df.empty:
            return None

        if 'developer_type' in df.columns:
            df = df[df['developer_type'] == 'all']

        pivot = df.pivot_table(
            index='eval_period',
            columns='train_period',
            values=metric,
            aggfunc='mean',
        )
        if pivot.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title(f'{metric} - Ranking Evaluation ({self.project_name})')
        ax.set_xlabel('Train Period')
        ax.set_ylabel('Eval Period')

        plt.tight_layout()
        output_path = self.output_dir / 'figures' / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"ランキングヒートマップを保存: {output_path}")
        return output_path

    def plot_ranking_ndcg(
        self,
        results: List[RankingCVResult],
        filename: str = None,
    ) -> Optional[Path]:
        """NDCG@K の比較棒グラフを作成する。"""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib がインストールされていません")
            return None

        if filename is None:
            filename = self._generate_filename('rank_ndcg', 'png')

        df = pd.DataFrame([result.to_dict() for result in results])
        if df.empty:
            return None

        if 'developer_type' in df.columns:
            df = df[df['developer_type'] == 'all']

        if df.empty:
            return None

        df['period_combo'] = df['train_period'] + ' → ' + df['eval_period']
        grouped = df.groupby('period_combo').agg({
            'ndcg_at_5': 'mean',
            'ndcg_at_10': 'mean',
            'ndcg_at_20': 'mean',
        }).reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(grouped))
        width = 0.25

        ax.bar(x - width, grouped['ndcg_at_5'], width, label='NDCG@5')
        ax.bar(x, grouped['ndcg_at_10'], width, label='NDCG@10')
        ax.bar(x + width, grouped['ndcg_at_20'], width, label='NDCG@20')

        ax.set_xlabel('Period Combination (Train → Eval)')
        ax.set_ylabel('Score')
        ax.set_title(f'Ranking NDCG Results ({self.project_name})')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped['period_combo'], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        output_path = self.output_dir / 'figures' / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"ランキング NDCG グラフを保存: {output_path}")
        return output_path


def generate_evaluation_report(
    results: List[CVResult],
    project_name: str,
    metadata: Dict[str, Any] = None,
    output_dir: Path = None,
    filename_suffix: str = None
) -> Dict[str, Path]:
    """
    評価レポートを生成する（CSV, JSON, グラフを出力）
    
    Args:
        results: 交差検証結果のリスト
        project_name: プロジェクト名
        metadata: メタデータ
        output_dir: 出力ディレクトリ
        filename_suffix: ファイル名に追加するサフィックス（モデル名等）
        
    Returns:
        Dict[str, Path]: 出力ファイル名とパスの辞書
    """
    visualizer = Visualizer(
        output_dir=output_dir, 
        project_name=project_name,
        filename_suffix=filename_suffix
    )
    
    outputs = {}
    
    # 詳細結果を保存
    outputs['cv_detail'] = visualizer.save_cv_detail(results)
    
    # サマリーを保存
    outputs['cv_summary'] = visualizer.save_cv_summary(results)
    
    # JSON結果を保存
    outputs['results_json'] = visualizer.save_results_json(results, metadata)
    
    # ヒートマップを生成
    heatmap_path = visualizer.plot_cross_period_heatmap(results, metric='f1')
    if heatmap_path:
        outputs['heatmap_f1'] = heatmap_path
    
    # 棒グラフを生成
    barplot_path = visualizer.plot_cv_results(results)
    if barplot_path:
        outputs['cv_barplot'] = barplot_path
    
    # 特徴量重要度を集計して表示
    all_importances = {}
    for r in results:
        if r.feature_importances:
            for feature, importance in r.feature_importances.items():
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(importance)
    
    if all_importances:
        avg_importances = {f: np.mean(v) for f, v in all_importances.items()}
        importance_path = visualizer.plot_feature_importances(avg_importances)
        if importance_path:
            outputs['feature_importance'] = importance_path
    
    logger.info(f"評価レポート生成完了: {len(outputs)}ファイル")
    return outputs


def generate_ranking_report(
    results: List[RankingCVResult],
    project_name: str,
    metadata: Dict[str, Any] = None,
    output_dir: Path = None,
    filename_suffix: str = None,
) -> Dict[str, Path]:
    """ランキング評価レポートを生成する（CSV, JSON, グラフ）。"""
    visualizer = Visualizer(
        output_dir=output_dir,
        project_name=project_name,
        filename_suffix=filename_suffix,
    )

    outputs: Dict[str, Path] = {}
    outputs['rank_detail'] = visualizer.save_rank_detail(results)
    outputs['rank_summary'] = visualizer.save_rank_summary(results)
    outputs['ranking_results_json'] = visualizer.save_ranking_results_json(results, metadata)

    heatmap_path = visualizer.plot_ranking_heatmap(results, metric='ndcg_at_10')
    if heatmap_path:
        outputs['rank_heatmap'] = heatmap_path

    ndcg_plot_path = visualizer.plot_ranking_ndcg(results)
    if ndcg_plot_path:
        outputs['rank_ndcg'] = ndcg_plot_path

    all_importances = {}
    for result in results:
        if result.feature_importances:
            for feature, importance in result.feature_importances.items():
                all_importances.setdefault(feature, []).append(importance)

    if all_importances:
        avg_importances = {feature: np.mean(values) for feature, values in all_importances.items()}
        importance_path = visualizer.plot_feature_importances(avg_importances)
        if importance_path:
            outputs['feature_importance'] = importance_path

    logger.info(f"ランキング評価レポート生成完了: {len(outputs)}ファイル")
    return outputs
