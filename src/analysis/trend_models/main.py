"""
Trend Models Analysis - メインエントリーポイント

本モジュールでは、トレンドモデル分析のメインクラスとエントリーポイントを提供します。
OpenStackプロジェクトにおいて、レビュー優先度予測モデルを構築・評価します。
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from src.config.path import DEFAULT_DATA_DIR
from src.analysis.trend_models.utils.constants import (
    TREND_MODEL_CONFIG,
    ANALYSIS_PERIODS,
    FEATURE_NAMES,
    MODEL_TYPES,
    DEFAULT_MODEL_TYPE,
    OUTPUT_DIR_NAME,
)
from src.analysis.trend_models.utils.data_loader import (
    load_major_releases_summary,
    load_all_changes,
    load_core_developers,
    load_bot_names_from_config,
    filter_changes_by_period,
    get_period_dates,
    get_release_pairs,
    changes_to_dataframe,
)
from src.analysis.trend_models.features.extractor import (
    FeatureExtractor,
    extract_features_from_changes,
)
from src.analysis.trend_models.features.preprocessor import (
    Preprocessor,
    preprocess_data,
)
from src.analysis.trend_models.models.trend_predictor import TrendPredictor
from src.analysis.trend_models.evaluation.evaluator import (
    Evaluator,
    leave_one_out_cv,
    CVResult,
    summarize_cv_results,
)
from src.analysis.trend_models.evaluation.visualizer import (
    Visualizer,
    generate_evaluation_report,
)

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path = None, level: int = logging.INFO):
    """ロギングの設定"""
    # 既存のハンドラをクリア
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # フォーマッタを作成
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # ファイルハンドラ
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'trend_models_{timestamp}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    root_logger.setLevel(level)
    
    # src.features.review_metricsのログはWARNINGレベル以上のみ出力
    # （毎回のキーワード/ラベルロード情報をノイズとして抑制）
    logging.getLogger('src.features.review_metrics').setLevel(logging.WARNING)


class TrendModelsAnalyzer:
    """
    トレンドモデル分析のメインクラス
    
    分析全体のオーケストレーション、データの読み込み、交差検証の実行、
    結果の保存とレポート生成を行います。
    """
    
    def __init__(
        self,
        project_name: str = 'nova',
        releases: List[str] = None,
        model_type: str = None,
        split_by_developer_type: bool = False,
        data_dir: Path = None,
        output_dir: Path = None
    ):
        """
        Args:
            project_name: 分析対象プロジェクト名
            releases: 分析対象リリースのリスト（Noneの場合はデフォルト）
            model_type: 使用するモデルタイプ
            split_by_developer_type: 開発者タイプで分割するかどうか
            data_dir: データディレクトリのパス
            output_dir: 出力ディレクトリのパス
        """
        self.project_name = project_name
        self.releases = releases or TREND_MODEL_CONFIG['project'].get(project_name, [])
        self.model_type = model_type or DEFAULT_MODEL_TYPE
        self.split_by_developer_type = split_by_developer_type
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.data_dir / 'analysis' / OUTPUT_DIR_NAME / project_name
        
        # データの初期化
        self.releases_df = None
        self.all_changes = []
        self.all_prs_df = None
        self.core_developers = {}
        self.bot_names = []
        
        # 結果
        self.cv_results: List[CVResult] = []
        self.release_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        logger.info(f"TrendModelsAnalyzer初期化: project={project_name}, "
                   f"releases={len(self.releases)}件, model={self.model_type}")
    
    def load_data(self) -> bool:
        """
        分析に必要なデータを読み込む
        
        Returns:
            bool: 読み込み成功時True
        """
        logger.info("データ読み込み開始...")
        
        # リリース情報を読み込み
        self.releases_df = load_major_releases_summary(self.data_dir)
        if self.releases_df.empty:
            logger.error("リリース情報の読み込みに失敗しました")
            return False
        
        # Changeデータを読み込み（openstack_collectedを使用）
        self.all_changes = load_all_changes(
            self.project_name, self.data_dir, use_collected=True
        )
        if not self.all_changes:
            logger.error("Changeデータの読み込みに失敗しました")
            return False
        
        # DataFrameに変換
        self.all_prs_df = changes_to_dataframe(self.all_changes)
        
        # コア開発者情報を読み込み
        self.core_developers = load_core_developers(self.project_name, self.data_dir)
        
        # ボット名を読み込み
        self.bot_names = load_bot_names_from_config()
        
        logger.info(f"データ読み込み完了: changes={len(self.all_changes)}件")
        return True
    
    def prepare_release_data(self) -> bool:
        """
        各リリース・期間ごとのデータを準備する
        
        Returns:
            bool: 準備成功時True
        """
        logger.info("リリースデータの準備開始...")
        
        # リリースペアを取得
        release_pairs = get_release_pairs(
            self.releases_df, self.project_name, self.releases
        )
        
        if not release_pairs:
            logger.error("有効なリリースペアがありません")
            return False
        
        # Preprocessorを作成
        preprocessor = Preprocessor(
            feature_names=FEATURE_NAMES,
            bot_names=self.bot_names,
            core_developers=self.core_developers
        )
        
        # 各リリースに対してデータを準備
        for current, next_rel in release_pairs:
            release_version = current['version']
            current_date = current['release_date']
            next_date = next_rel['release_date']
            
            if pd.isna(current_date) or pd.isna(next_date):
                continue
            
            # datetime型に変換
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            if isinstance(next_date, str):
                next_date = pd.to_datetime(next_date)
            
            # Python datetime に変換
            current_date = current_date.to_pydatetime() if hasattr(current_date, 'to_pydatetime') else current_date
            next_date = next_date.to_pydatetime() if hasattr(next_date, 'to_pydatetime') else next_date
            
            logger.info(f"リリース {release_version} のデータ準備中...")
            
            self.release_data[release_version] = {}
            
            # 各期間タイプに対してデータを準備
            for period_type in ['early', 'late', 'all']:
                try:
                    period_start, period_end = get_period_dates(
                        current_date, next_date, period_type
                    )
                    
                    # 期間でフィルタリング
                    filtered_changes = filter_changes_by_period(
                        self.all_changes,
                        period_start,
                        period_end,
                        next_date
                    )
                    
                    if not filtered_changes:
                        logger.warning(f"リリース {release_version}/{period_type}: "
                                      f"対象Changeなし")
                        continue
                    
                    # 特徴量を抽出
                    features_df = extract_features_from_changes(
                        filtered_changes,
                        self.all_prs_df,
                        self.releases_df,
                        self.project_name,
                        period_start
                    )
                    
                    # ラベル付け
                    features_df = preprocessor.add_labels(
                        features_df, filtered_changes, period_start, period_end
                    )
                    
                    # 開発者タイプ付与（期間中にCore/Non-Coreからレビューされたか）
                    if self.split_by_developer_type:
                        features_df = preprocessor.add_developer_type(
                            features_df, filtered_changes, self.project_name,
                            period_start, period_end
                        )
                    
                    # 欠損値処理
                    features_df = preprocessor.handle_missing_values(features_df)
                    
                    self.release_data[release_version][period_type] = features_df
                    
                    logger.info(f"リリース {release_version}/{period_type}: "
                               f"{len(features_df)}件 (レビュー済み: {features_df['reviewed'].sum()}件)")
                    
                except Exception as e:
                    logger.error(f"リリース {release_version}/{period_type} の"
                                f"データ準備に失敗: {e}")
        
        logger.info(f"リリースデータの準備完了: {len(self.release_data)}リリース")
        return len(self.release_data) > 0
    
    def run_cross_validation(self) -> List[CVResult]:
        """
        Leave-One-Out交差検証を実行する
        
        Returns:
            List[CVResult]: 交差検証結果のリスト
        """
        logger.info("交差検証開始...")
        
        # 正規化のためにPreprocessorを作成
        preprocessor = Preprocessor(feature_names=FEATURE_NAMES)
        
        # 全データを結合して正規化パラメータを計算
        all_dfs = []
        for release_data in self.release_data.values():
            for period_df in release_data.values():
                all_dfs.append(period_df)
        
        if not all_dfs:
            logger.error("交差検証に使用できるデータがありません")
            return []
        
        all_data = pd.concat(all_dfs, ignore_index=True)
        preprocessor.normalize(all_data, fit=True)  # Scalerをフィット
        
        # 各リリースのデータを正規化
        normalized_release_data = {}
        for release, period_data in self.release_data.items():
            normalized_release_data[release] = {}
            for period, df in period_data.items():
                normalized_release_data[release][period] = preprocessor.normalize(df, fit=False)
        
        # Leave-One-Out交差検証を実行
        self.cv_results = leave_one_out_cv(
            normalized_release_data,
            feature_names=FEATURE_NAMES,
            model_type=self.model_type,
            split_by_developer_type=self.split_by_developer_type
        )
        
        logger.info(f"交差検証完了: {len(self.cv_results)}件の結果")
        return self.cv_results
    
    def generate_report(self) -> Dict[str, Path]:
        """
        結果のレポートを生成する
        
        Returns:
            Dict[str, Path]: 出力ファイルのパス
        """
        logger.info("レポート生成開始...")
        
        metadata = {
            'project': self.project_name,
            'releases': self.releases,
            'model_type': self.model_type,
            'split_by_developer_type': self.split_by_developer_type,
            'n_changes': len(self.all_changes),
            'timestamp': datetime.now().isoformat()
        }
        
        # モデル名を含めた出力ディレクトリ
        model_output_dir = self.output_dir / self.model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = generate_evaluation_report(
            self.cv_results,
            self.project_name,
            metadata,
            model_output_dir,
            filename_suffix=self.model_type
        )
        
        logger.info(f"レポート生成完了: {len(outputs)}ファイル")
        return outputs
    
    def run(self) -> Dict[str, Any]:
        """
        全処理を実行する
        
        Returns:
            Dict[str, Any]: 実行結果（結果サマリー、出力ファイルパスなど）
        """
        logger.info(f"=== トレンドモデル分析開始: {self.project_name} ===")
        
        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ読み込み（既に読み込まれていればスキップ）
        if not self.all_changes:
            if not self.load_data():
                return {'success': False, 'error': 'データ読み込み失敗'}
        
        # データ準備（既に準備されていればスキップ）
        if not self.release_data:
            if not self.prepare_release_data():
                return {'success': False, 'error': 'データ準備失敗'}
        
        # 交差検証
        self.run_cross_validation()
        
        if not self.cv_results:
            return {'success': False, 'error': '交差検証結果なし'}
        
        # レポート生成
        outputs = self.generate_report()
        
        # サマリーを計算
        summary_df = summarize_cv_results(self.cv_results)
        
        result = {
            'success': True,
            'project': self.project_name,
            'n_releases': len(self.release_data),
            'n_cv_results': len(self.cv_results),
            'summary': summary_df.to_dict('records'),
            'outputs': {k: str(v) for k, v in outputs.items()}
        }
        
        logger.info(f"=== トレンドモデル分析完了: {self.project_name} ===")
        return result
    
    def run_with_model(self, model_type: str) -> Dict[str, Any]:
        """
        指定されたモデルタイプで交差検証とレポート生成を実行する
        
        データは既に準備されている前提（load_data, prepare_release_dataを先に呼ぶ）
        
        Args:
            model_type: 使用するモデルタイプ
            
        Returns:
            Dict[str, Any]: 実行結果
        """
        if not self.release_data:
            logger.error("リリースデータが準備されていません。prepare_release_data()を先に呼んでください。")
            return {'success': False, 'error': 'リリースデータ未準備'}
        
        # モデルタイプを設定
        self.model_type = model_type
        
        logger.info(f"=== モデル評価開始: {model_type} ===")
        
        # 交差検証
        self.run_cross_validation()
        
        if not self.cv_results:
            return {'success': False, 'error': '交差検証結果なし'}
        
        # レポート生成
        outputs = self.generate_report()
        
        # サマリーを計算
        summary_df = summarize_cv_results(self.cv_results)
        
        result = {
            'success': True,
            'project': self.project_name,
            'model_type': model_type,
            'n_releases': len(self.release_data),
            'n_cv_results': len(self.cv_results),
            'summary': summary_df.to_dict('records'),
            'outputs': {k: str(v) for k, v in outputs.items()}
        }
        
        logger.info(f"=== モデル評価完了: {model_type} ===")
        return result


def main():
    """コマンドラインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description='Trend Models Analysis - レビュー優先度予測モデルの分析'
    )
    parser.add_argument(
        '--project', '-p',
        type=str,
        default='nova',
        help='分析対象プロジェクト名 (default: nova)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        choices=['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'tabnet', 'ft_transformer'],
        help='使用するモデルタイプ。指定しない場合はconstants.pyのMODEL_TYPESを使用'
    )
    parser.add_argument(
        '--split-by-developer', '-d',
        action='store_true',
        help='開発者タイプ（Core/Non-Core）で結果を分割する'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='データディレクトリのパス'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='出力ディレクトリのパス'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細ログを出力する'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリを決定
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
        output_dir = data_dir / 'analysis' / OUTPUT_DIR_NAME / args.project
    
    # ログディレクトリ
    log_dir = output_dir / 'logs'
    
    # ロギング設定（ファイル出力を有効化）
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir=log_dir, level=log_level)
    
    # 使用するモデルタイプを決定
    if args.model:
        # コマンドラインで指定された場合はそのモデルのみ使用
        model_types = [args.model]
    elif MODEL_TYPES:
        # constants.pyのMODEL_TYPESを使用
        model_types = MODEL_TYPES
    else:
        # デフォルトモデルを使用
        model_types = [DEFAULT_MODEL_TYPE]
    
    print(f"\n分析対象モデル: {', '.join(model_types)}")
    
    # アナライザを作成（最初のモデルタイプで初期化）
    analyzer = TrendModelsAnalyzer(
        project_name=args.project,
        model_type=model_types[0],
        split_by_developer_type=args.split_by_developer,
        data_dir=args.data_dir,
        output_dir=output_dir
    )
    
    # データ読み込みと準備（1回のみ）
    print(f"\n{'='*60}")
    print("データ読み込み・準備")
    print(f"{'='*60}")
    
    if not analyzer.load_data():
        print("エラー: データ読み込みに失敗しました")
        sys.exit(1)
    
    if not analyzer.prepare_release_data():
        print("エラー: データ準備に失敗しました")
        sys.exit(1)
    
    print(f"データ準備完了: {len(analyzer.release_data)}リリース")
    
    all_results = {}
    
    # 各モデルで評価を実行
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"モデル: {model_type}")
        print(f"{'='*60}")
        
        # モデル評価を実行（データは再利用）
        result = analyzer.run_with_model(model_type)
        all_results[model_type] = result
        
        if result['success']:
            print(f"\n分析完了: {result['project']} ({model_type})")
            print(f"  リリース数: {result['n_releases']}")
            print(f"  CV結果数: {result['n_cv_results']}")
            print(f"  出力ファイル:")
            for name, path in result['outputs'].items():
                print(f"    - {name}: {path}")
        else:
            print(f"\n分析失敗 ({model_type}): {result.get('error', '不明なエラー')}")
    
    # 全体の結果サマリー
    print(f"\n{'='*60}")
    print("全モデルの分析完了")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in all_results.values() if r['success'])
    print(f"成功: {success_count}/{len(model_types)} モデル")
    
    if success_count < len(model_types):
        sys.exit(1)


if __name__ == '__main__':
    main()
