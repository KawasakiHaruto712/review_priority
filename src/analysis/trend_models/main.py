"""
Trend Models Analysis - メインエントリーポイント

本モジュールでは、トレンドモデル分析のメインクラスとエントリーポイントを提供します。
OpenStackプロジェクトにおいて、レビュー優先度予測モデルを構築・評価します。
"""

import argparse
import hashlib
import json
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
    TASK_MODES,
    DEFAULT_TASK_MODE,
    RANKING_MODEL_TYPES,
    DEFAULT_RANKING_MODEL_TYPE,
    RANKING_LABEL_COLUMN_BY_MODE,
    DEFAULT_RANKING_LABEL_MODE,
    RANKING_K_VALUES,
    RANKING_MIN_QUERY_SIZE,
    RANKING_MAX_CENSORING_SECONDS,
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
from src.analysis.trend_models.classification_models.trend_predictor import TrendPredictor
from src.analysis.trend_models.ranking.daily_rank_builder import build_daily_ranking_dataset
from src.analysis.trend_models.ranking.ranking_dataset import get_ranking_target_column
from src.analysis.trend_models.evaluation.evaluator import (
    Evaluator,
    leave_one_out_cv,
    CVResult,
    RankingCVResult,
    summarize_cv_results,
    leave_one_out_ranking_cv,
    summarize_ranking_cv_results,
)
from src.analysis.trend_models.evaluation.visualizer import (
    Visualizer,
    generate_evaluation_report,
    generate_ranking_report,
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

    CACHE_VERSION = 'v1'
    
    def __init__(
        self,
        project_name: str = 'nova',
        releases: List[str] = None,
        model_type: str = None,
        task_mode: str = DEFAULT_TASK_MODE,
        ranking_model_type: str = DEFAULT_RANKING_MODEL_TYPE,
        ranking_label_mode: str = DEFAULT_RANKING_LABEL_MODE,
        ranking_k_values: List[int] = None,
        split_by_developer_type: bool = False,
        use_cache: bool = True,
        data_dir: Path = None,
        output_dir: Path = None
    ):
        """
        Args:
            project_name: 分析対象プロジェクト名
            releases: 分析対象リリースのリスト（Noneの場合はデフォルト）
            model_type: 使用するモデルタイプ
            task_mode: 実行モード（classification / ranking / both）
            ranking_model_type: ranking で使用するモデルタイプ
            ranking_label_mode: ranking 目的変数モード（rank / rank_pct / time）
            ranking_k_values: ranking 評価で利用する K 値の配列
            split_by_developer_type: 開発者タイプで分割するかどうか
            use_cache: ranking の中間結果キャッシュを利用するかどうか
            data_dir: データディレクトリのパス
            output_dir: 出力ディレクトリのパス
        """
        self.project_name = project_name
        self.releases = releases or TREND_MODEL_CONFIG['project'].get(project_name, [])
        self.model_type = model_type or DEFAULT_MODEL_TYPE
        self.task_mode = task_mode
        if self.task_mode not in TASK_MODES:
            raise ValueError(f"無効な task_mode: {self.task_mode}. choices={TASK_MODES}")

        self.ranking_model_type = ranking_model_type or DEFAULT_RANKING_MODEL_TYPE
        self.ranking_label_mode = ranking_label_mode or DEFAULT_RANKING_LABEL_MODE
        if self.ranking_label_mode not in RANKING_LABEL_COLUMN_BY_MODE:
            raise ValueError(
                f"無効な ranking_label_mode: {self.ranking_label_mode}. "
                f"choices={list(RANKING_LABEL_COLUMN_BY_MODE.keys())}"
            )
        self.ranking_target_col = get_ranking_target_column(self.ranking_label_mode)

        if ranking_k_values:
            self.ranking_k_values = sorted(set(int(k) for k in ranking_k_values if int(k) > 0))
        else:
            self.ranking_k_values = RANKING_K_VALUES

        self.split_by_developer_type = split_by_developer_type
        self.use_cache = use_cache
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.data_dir / 'analysis' / OUTPUT_DIR_NAME / project_name
        self.cache_dir = self.output_dir / 'cache'
        
        # データの初期化
        self.releases_df = None
        self.all_changes = []
        self.all_prs_df = None
        self.core_developers = {}
        self.bot_names = []
        
        # 結果
        self.cv_results: List[CVResult] = []
        self.ranking_cv_results: List[RankingCVResult] = []
        self.release_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.ranking_release_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        logger.info(f"TrendModelsAnalyzer初期化: project={project_name}, "
               f"releases={len(self.releases)}件, model={self.model_type}, "
               f"task_mode={self.task_mode}, ranking_model={self.ranking_model_type}, "
             f"ranking_target={self.ranking_target_col}, cache={'on' if self.use_cache else 'off'}")
    
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

    def _build_cache_key(
        self,
        stage: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """実行設定に対応するキャッシュキーを生成する。"""
        payload: Dict[str, Any] = {
            'cache_version': self.CACHE_VERSION,
            'stage': stage,
            'project_name': self.project_name,
            'releases': self.releases,
            'split_by_developer_type': self.split_by_developer_type,
            'feature_names': FEATURE_NAMES,
            'data_dir': str(self.data_dir),
        }
        if extra:
            payload.update(extra)

        payload_text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload_text.encode('utf-8')).hexdigest()[:16]

    def _get_cache_path(
        self,
        cache_name: str,
        stage: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """キャッシュファイルパスを生成する。"""
        cache_key = self._build_cache_key(stage=stage, extra=extra)
        return self.cache_dir / f'{cache_name}_{cache_key}.pkl'
    
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

    def prepare_ranking_release_data(self) -> bool:
        """
        各リリース・期間ごとの ranking 学習データを準備する

        Returns:
            bool: 準備成功時True
        """
        logger.info("ランキング用リリースデータの準備開始...")

        ranking_data_cache_path = None
        if self.use_cache:
            ranking_data_cache_path = self._get_cache_path(
                cache_name='ranking_release_data',
                stage='ranking_release_data',
                extra={
                    'ranking_min_query_size': RANKING_MIN_QUERY_SIZE,
                    'ranking_max_censoring_seconds': RANKING_MAX_CENSORING_SECONDS,
                },
            )
            if ranking_data_cache_path.exists():
                try:
                    cached_release_data = pd.read_pickle(ranking_data_cache_path)
                    if isinstance(cached_release_data, dict):
                        self.ranking_release_data = cached_release_data
                        logger.info(
                            "ランキング用リリースデータをキャッシュから復元: %s",
                            ranking_data_cache_path,
                        )
                        return len(self.ranking_release_data) > 0
                except Exception as e:
                    logger.warning(
                        "ランキング用リリースデータのキャッシュ読み込み失敗。再計算します: %s",
                        e,
                    )

        release_pairs = get_release_pairs(
            self.releases_df, self.project_name, self.releases
        )
        if not release_pairs:
            logger.error("有効なリリースペアがありません")
            return False

        self.ranking_release_data = {}

        preprocessor = Preprocessor(
            feature_names=FEATURE_NAMES,
            bot_names=self.bot_names,
            core_developers=self.core_developers,
        )

        for current, next_rel in release_pairs:
            release_version = current['version']
            current_date = current['release_date']
            next_date = next_rel['release_date']

            if pd.isna(current_date) or pd.isna(next_date):
                continue

            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            if isinstance(next_date, str):
                next_date = pd.to_datetime(next_date)

            current_date = current_date.to_pydatetime() if hasattr(current_date, 'to_pydatetime') else current_date
            next_date = next_date.to_pydatetime() if hasattr(next_date, 'to_pydatetime') else next_date

            logger.info(f"ランキングデータ準備中: release={release_version}")
            self.ranking_release_data[release_version] = {}

            for period_type in ['early', 'late', 'all']:
                try:
                    period_start, period_end = get_period_dates(
                        current_date, next_date, period_type
                    )

                    ranking_df = build_daily_ranking_dataset(
                        project_name=self.project_name,
                        release_version=release_version,
                        period_type=period_type,
                        period_start=period_start,
                        period_end=period_end,
                        all_changes=self.all_changes,
                        all_prs_df=self.all_prs_df,
                        releases_df=self.releases_df,
                        bot_names=self.bot_names,
                        min_query_size=RANKING_MIN_QUERY_SIZE,
                        max_censoring_seconds=RANKING_MAX_CENSORING_SECONDS,
                    )

                    if ranking_df.empty:
                        logger.warning(
                            f"ランキングデータなし: release={release_version}, period={period_type}"
                        )
                        continue

                    # 欠損値処理
                    ranking_df = preprocessor.handle_missing_values(ranking_df)

                    if self.split_by_developer_type:
                        ranking_df = preprocessor.add_developer_type(
                            ranking_df,
                            self.all_changes,
                            self.project_name,
                            period_start,
                            period_end,
                        )

                    self.ranking_release_data[release_version][period_type] = ranking_df

                    logger.info(
                        "ランキングデータ準備完了: release=%s/%s, rows=%d, queries=%d",
                        release_version,
                        period_type,
                        len(ranking_df),
                        ranking_df['query_id'].nunique() if 'query_id' in ranking_df.columns else 0,
                    )

                except Exception as e:
                    logger.error(
                        f"ランキングデータ準備失敗: release={release_version}/{period_type}, error={e}"
                    )

        logger.info(
            f"ランキング用リリースデータ準備完了: {len(self.ranking_release_data)}リリース"
        )

        if self.use_cache and ranking_data_cache_path is not None and self.ranking_release_data:
            try:
                ranking_data_cache_path.parent.mkdir(parents=True, exist_ok=True)
                pd.to_pickle(self.ranking_release_data, ranking_data_cache_path)
                logger.info(
                    "ランキング用リリースデータをキャッシュ保存: %s",
                    ranking_data_cache_path,
                )
            except Exception as e:
                logger.warning(
                    "ランキング用リリースデータのキャッシュ保存失敗: %s",
                    e,
                )

        return len(self.ranking_release_data) > 0
    
    def _normalize_release_data(
        self,
        release_data: Dict[str, Dict[str, pd.DataFrame]],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """release/period 単位データを共通スケーラで正規化する。"""
        preprocessor = Preprocessor(feature_names=FEATURE_NAMES)

        all_dfs = []
        for period_data in release_data.values():
            for period_df in period_data.values():
                all_dfs.append(period_df)

        if not all_dfs:
            return {}

        all_data = pd.concat(all_dfs, ignore_index=True)
        preprocessor.normalize(all_data, fit=True)

        normalized_release_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        for release, period_data in release_data.items():
            normalized_release_data[release] = {}
            for period, df in period_data.items():
                normalized_release_data[release][period] = preprocessor.normalize(df, fit=False)

        return normalized_release_data

    def run_cross_validation(self) -> List[CVResult]:
        """
        Leave-One-Out交差検証を実行する
        
        Returns:
            List[CVResult]: 交差検証結果のリスト
        """
        logger.info("交差検証開始...")

        normalized_release_data = self._normalize_release_data(self.release_data)
        if not normalized_release_data:
            logger.error("交差検証に使用できるデータがありません")
            return []
        
        # Leave-One-Out交差検証を実行
        self.cv_results = leave_one_out_cv(
            normalized_release_data,
            feature_names=FEATURE_NAMES,
            model_type=self.model_type,
            split_by_developer_type=self.split_by_developer_type
        )
        
        logger.info(f"交差検証完了: {len(self.cv_results)}件の結果")
        return self.cv_results

    def run_ranking_cross_validation(self) -> List[RankingCVResult]:
        """
        ランキング用 Leave-One-Out 交差検証を実行する

        Returns:
            List[RankingCVResult]: ランキング交差検証結果
        """
        logger.info("ランキング交差検証開始...")

        ranking_cv_cache_path = None
        if self.use_cache:
            ranking_cv_cache_path = self._get_cache_path(
                cache_name='ranking_cv_results',
                stage='ranking_cv_results',
                extra={
                    'ranking_model_type': self.ranking_model_type,
                    'ranking_label_mode': self.ranking_label_mode,
                    'ranking_target_col': self.ranking_target_col,
                    'ranking_k_values': self.ranking_k_values,
                },
            )
            if ranking_cv_cache_path.exists():
                try:
                    cached_cv_results = pd.read_pickle(ranking_cv_cache_path)
                    if isinstance(cached_cv_results, list) and all(
                        isinstance(item, RankingCVResult) for item in cached_cv_results
                    ):
                        self.ranking_cv_results = cached_cv_results
                        logger.info(
                            "ランキング交差検証結果をキャッシュから復元: %s",
                            ranking_cv_cache_path,
                        )
                        return self.ranking_cv_results

                    if isinstance(cached_cv_results, list) and all(
                        isinstance(item, dict) for item in cached_cv_results
                    ):
                        self.ranking_cv_results = [
                            RankingCVResult(**item) for item in cached_cv_results
                        ]
                        logger.info(
                            "ランキング交差検証結果をキャッシュから復元: %s",
                            ranking_cv_cache_path,
                        )
                        return self.ranking_cv_results
                except Exception as e:
                    logger.warning(
                        "ランキング交差検証結果のキャッシュ読み込み失敗。再計算します: %s",
                        e,
                    )

        normalized_release_data = self._normalize_release_data(self.ranking_release_data)
        if not normalized_release_data:
            logger.error("ランキング交差検証に使用できるデータがありません")
            return []

        self.ranking_cv_results = leave_one_out_ranking_cv(
            normalized_release_data,
            feature_names=FEATURE_NAMES,
            ranking_model_type=self.ranking_model_type,
            label_mode=self.ranking_label_mode,
            k_values=self.ranking_k_values,
            split_by_developer_type=self.split_by_developer_type,
        )

        if self.use_cache and ranking_cv_cache_path is not None and self.ranking_cv_results:
            try:
                ranking_cv_cache_path.parent.mkdir(parents=True, exist_ok=True)
                pd.to_pickle(self.ranking_cv_results, ranking_cv_cache_path)
                logger.info(
                    "ランキング交差検証結果をキャッシュ保存: %s",
                    ranking_cv_cache_path,
                )
            except Exception as e:
                logger.warning(
                    "ランキング交差検証結果のキャッシュ保存失敗: %s",
                    e,
                )

        logger.info(f"ランキング交差検証完了: {len(self.ranking_cv_results)}件の結果")
        return self.ranking_cv_results
    
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

    def generate_ranking_report(self) -> Dict[str, Path]:
        """
        ランキング評価レポートを生成する

        Returns:
            Dict[str, Path]: 出力ファイルのパス
        """
        logger.info("ランキングレポート生成開始...")

        metadata = {
            'project': self.project_name,
            'releases': self.releases,
            'task_mode': self.task_mode,
            'ranking_model_type': self.ranking_model_type,
            'ranking_label_mode': self.ranking_label_mode,
            'ranking_target_col': self.ranking_target_col,
            'ranking_k_values': self.ranking_k_values,
            'split_by_developer_type': self.split_by_developer_type,
            'n_changes': len(self.all_changes),
            'timestamp': datetime.now().isoformat(),
        }

        model_output_dir = self.output_dir / self.ranking_model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)

        outputs = generate_ranking_report(
            self.ranking_cv_results,
            self.project_name,
            metadata,
            model_output_dir,
            filename_suffix=self.ranking_model_type,
        )

        logger.info(f"ランキングレポート生成完了: {len(outputs)}ファイル")
        return outputs
    
    def run(self) -> Dict[str, Any]:
        """
        全処理を実行する
        
        Returns:
            Dict[str, Any]: 実行結果（結果サマリー、出力ファイルパスなど）
        """
        logger.info(
            f"=== トレンドモデル分析開始: {self.project_name} (mode={self.task_mode}) ==="
        )
        
        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ読み込み（既に読み込まれていればスキップ）
        if not self.all_changes:
            if not self.load_data():
                return {'success': False, 'error': 'データ読み込み失敗'}

        result: Dict[str, Any] = {
            'success': True,
            'project': self.project_name,
            'task_mode': self.task_mode,
            'outputs': {},
        }

        classification_success = True
        ranking_success = True

        if self.task_mode in ['classification', 'both']:
            if not self.release_data:
                if not self.prepare_release_data():
                    if self.task_mode == 'classification':
                        return {'success': False, 'error': '分類データ準備失敗'}
                    classification_success = False

            if classification_success:
                self.run_cross_validation()
                if not self.cv_results:
                    if self.task_mode == 'classification':
                        return {'success': False, 'error': '分類交差検証結果なし'}
                    classification_success = False

            if classification_success:
                cls_outputs = self.generate_report()
                cls_summary_df = summarize_cv_results(self.cv_results)
                result['classification'] = {
                    'success': True,
                    'model_type': self.model_type,
                    'n_releases': len(self.release_data),
                    'n_cv_results': len(self.cv_results),
                    'summary': cls_summary_df.to_dict('records'),
                    'outputs': {name: str(path) for name, path in cls_outputs.items()},
                }
            else:
                result['classification'] = {
                    'success': False,
                    'error': '分類評価に失敗しました',
                }

        if self.task_mode in ['ranking', 'both']:
            if not self.ranking_release_data:
                if not self.prepare_ranking_release_data():
                    if self.task_mode == 'ranking':
                        return {'success': False, 'error': 'ランキングデータ準備失敗'}
                    ranking_success = False

            if ranking_success:
                self.run_ranking_cross_validation()
                if not self.ranking_cv_results:
                    if self.task_mode == 'ranking':
                        return {'success': False, 'error': 'ランキング交差検証結果なし'}
                    ranking_success = False

            if ranking_success:
                rank_outputs = self.generate_ranking_report()
                rank_summary_df = summarize_ranking_cv_results(self.ranking_cv_results)
                result['ranking'] = {
                    'success': True,
                    'model_type': self.ranking_model_type,
                    'target_col': self.ranking_target_col,
                    'k_values': self.ranking_k_values,
                    'n_releases': len(self.ranking_release_data),
                    'n_cv_results': len(self.ranking_cv_results),
                    'summary': rank_summary_df.to_dict('records'),
                    'outputs': {name: str(path) for name, path in rank_outputs.items()},
                }
            else:
                result['ranking'] = {
                    'success': False,
                    'error': 'ランキング評価に失敗しました',
                }

        if self.task_mode == 'classification':
            result['success'] = classification_success
            result['outputs'] = result.get('classification', {}).get('outputs', {})
        elif self.task_mode == 'ranking':
            result['success'] = ranking_success
            result['outputs'] = result.get('ranking', {}).get('outputs', {})
        else:
            result['success'] = classification_success and ranking_success
            merged_outputs = {}
            if result.get('classification', {}).get('outputs'):
                merged_outputs['classification'] = result['classification']['outputs']
            if result.get('ranking', {}).get('outputs'):
                merged_outputs['ranking'] = result['ranking']['outputs']
            result['outputs'] = merged_outputs

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

    def run_with_ranking_model(self, ranking_model_type: str) -> Dict[str, Any]:
        """
        指定されたランキングモデルタイプで交差検証とレポート生成を実行する

        データは既に準備されている前提（load_data, prepare_ranking_release_dataを先に呼ぶ）

        Args:
            ranking_model_type: ranking で使用するモデルタイプ

        Returns:
            Dict[str, Any]: 実行結果
        """
        if not self.ranking_release_data:
            logger.error(
                "ランキング用リリースデータが準備されていません。"
                "prepare_ranking_release_data()を先に呼んでください。"
            )
            return {'success': False, 'error': 'ランキング用リリースデータ未準備'}

        self.ranking_model_type = ranking_model_type

        logger.info(f"=== ランキングモデル評価開始: {ranking_model_type} ===")

        self.run_ranking_cross_validation()
        if not self.ranking_cv_results:
            return {'success': False, 'error': 'ランキング交差検証結果なし'}

        outputs = self.generate_ranking_report()
        summary_df = summarize_ranking_cv_results(self.ranking_cv_results)

        result = {
            'success': True,
            'project': self.project_name,
            'model_type': ranking_model_type,
            'target_col': self.ranking_target_col,
            'k_values': self.ranking_k_values,
            'n_releases': len(self.ranking_release_data),
            'n_cv_results': len(self.ranking_cv_results),
            'summary': summary_df.to_dict('records'),
            'outputs': {name: str(path) for name, path in outputs.items()},
        }

        logger.info(f"=== ランキングモデル評価完了: {ranking_model_type} ===")
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
        '--task-mode',
        type=str,
        default=DEFAULT_TASK_MODE,
        choices=TASK_MODES,
        help='実行モード: classification / ranking / both'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        choices=['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'tabnet', 'ft_transformer'],
        help='分類モデルタイプ。classification時は未指定でconstants.pyのMODEL_TYPESを使用'
    )
    parser.add_argument(
        '--ranking-model',
        type=str,
        default=None,
        choices=RANKING_MODEL_TYPES,
        help='ranking モデルタイプ。未指定時はconstants.pyのRANKING_MODEL_TYPESを順次実行'
    )
    parser.add_argument(
        '--ranking-label',
        type=str,
        default=DEFAULT_RANKING_LABEL_MODE,
        choices=list(RANKING_LABEL_COLUMN_BY_MODE.keys()),
        help='ranking 目的変数モード (rank / rank_pct / time)'
    )
    parser.add_argument(
        '--k-values',
        type=str,
        default=','.join(str(k) for k in RANKING_K_VALUES),
        help='ranking 評価で使用する K 値（カンマ区切り、例: 5,10,20）'
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
    parser.add_argument(
        '--disable-cache',
        action='store_true',
        help='ランキング中間データとCV結果のキャッシュ再利用を無効化する'
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

    try:
        ranking_k_values = sorted({
            int(token.strip())
            for token in args.k_values.split(',')
            if token.strip()
        })
        ranking_k_values = [k for k in ranking_k_values if k > 0]
    except ValueError:
        print("エラー: --k-values は整数のカンマ区切りで指定してください")
        sys.exit(1)

    if not ranking_k_values:
        ranking_k_values = RANKING_K_VALUES

    if args.ranking_model:
        ranking_model_types = [args.ranking_model]
    elif RANKING_MODEL_TYPES:
        ranking_model_types = RANKING_MODEL_TYPES
    else:
        ranking_model_types = [DEFAULT_RANKING_MODEL_TYPE]
    
    if args.task_mode == 'classification':
        # 既存互換: classification では複数モデルを順次実行
        if args.model:
            model_types = [args.model]
        elif MODEL_TYPES:
            model_types = MODEL_TYPES
        else:
            model_types = [DEFAULT_MODEL_TYPE]

        print(f"\n分析モード: classification")
        print(f"分類モデル: {', '.join(model_types)}")

        analyzer = TrendModelsAnalyzer(
            project_name=args.project,
            model_type=model_types[0],
            task_mode='classification',
            split_by_developer_type=args.split_by_developer,
            use_cache=not args.disable_cache,
            data_dir=args.data_dir,
            output_dir=output_dir,
        )

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
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"モデル: {model_type}")
            print(f"{'='*60}")

            result = analyzer.run_with_model(model_type)
            all_results[model_type] = result

            if result['success']:
                print(f"\n分析完了: {result['project']} ({model_type})")
                print(f"  リリース数: {result['n_releases']}")
                print(f"  CV結果数: {result['n_cv_results']}")
                print("  出力ファイル:")
                for name, path in result['outputs'].items():
                    print(f"    - {name}: {path}")
            else:
                print(f"\n分析失敗 ({model_type}): {result.get('error', '不明なエラー')}")

        print(f"\n{'='*60}")
        print("全モデルの分析完了")
        print(f"{'='*60}")

        success_count = sum(1 for r in all_results.values() if r['success'])
        print(f"成功: {success_count}/{len(model_types)} モデル")

        if success_count < len(model_types):
            sys.exit(1)
        return

    if args.task_mode == 'ranking':
        print(f"\n分析モード: ranking")
        print(f"ランキングモデル: {', '.join(ranking_model_types)}")
        print(f"ランキング目的変数: {args.ranking_label}")
        print(f"ランキング K 値: {ranking_k_values}")
        print(f"キャッシュ利用: {'無効' if args.disable_cache else '有効'}")

        analyzer = TrendModelsAnalyzer(
            project_name=args.project,
            model_type=args.model or DEFAULT_MODEL_TYPE,
            task_mode='ranking',
            ranking_model_type=ranking_model_types[0],
            ranking_label_mode=args.ranking_label,
            ranking_k_values=ranking_k_values,
            split_by_developer_type=args.split_by_developer,
            use_cache=not args.disable_cache,
            data_dir=args.data_dir,
            output_dir=output_dir,
        )

        print(f"\n{'='*60}")
        print("ランキングデータ読み込み・準備")
        print(f"{'='*60}")

        if not analyzer.load_data():
            print("エラー: データ読み込みに失敗しました")
            sys.exit(1)

        if not analyzer.prepare_ranking_release_data():
            print("エラー: ランキングデータ準備に失敗しました")
            sys.exit(1)

        print(f"ランキングデータ準備完了: {len(analyzer.ranking_release_data)}リリース")

        all_ranking_results = {}
        for ranking_model_type in ranking_model_types:
            print(f"\n{'='*60}")
            print(f"ランキングモデル: {ranking_model_type}")
            print(f"{'='*60}")

            result = analyzer.run_with_ranking_model(ranking_model_type)
            all_ranking_results[ranking_model_type] = result

            if result['success']:
                print(f"\n分析完了: {result['project']} ({ranking_model_type})")
                print(f"  リリース数: {result['n_releases']}")
                print(f"  CV結果数: {result['n_cv_results']}")
                print("  出力ファイル:")
                for name, path in result['outputs'].items():
                    print(f"    - {name}: {path}")
            else:
                print(f"\n分析失敗 ({ranking_model_type}): {result.get('error', '不明なエラー')}")

        print(f"\n{'='*60}")
        print("全ランキングモデルの分析完了")
        print(f"{'='*60}")

        ranking_success_count = sum(1 for r in all_ranking_results.values() if r['success'])
        print(f"成功: {ranking_success_count}/{len(ranking_model_types)} モデル")

        if ranking_success_count < len(ranking_model_types):
            sys.exit(1)
        return

    # both: 分類1モデル + rankingモデル群
    classification_model = args.model or DEFAULT_MODEL_TYPE
    print(f"\n分析モード: both")
    print(f"分類モデル: {classification_model}")
    print(f"ランキングモデル: {', '.join(ranking_model_types)}")
    print(f"ランキング目的変数: {args.ranking_label}")
    print(f"ランキング K 値: {ranking_k_values}")
    print(f"キャッシュ利用: {'無効' if args.disable_cache else '有効'}")

    analyzer = TrendModelsAnalyzer(
        project_name=args.project,
        model_type=classification_model,
        task_mode='both',
        ranking_model_type=ranking_model_types[0],
        ranking_label_mode=args.ranking_label,
        ranking_k_values=ranking_k_values,
        split_by_developer_type=args.split_by_developer,
        use_cache=not args.disable_cache,
        data_dir=args.data_dir,
        output_dir=output_dir,
    )

    print(f"\n{'='*60}")
    print("データ読み込み・準備")
    print(f"{'='*60}")

    if not analyzer.load_data():
        print("エラー: データ読み込みに失敗しました")
        sys.exit(1)

    if not analyzer.prepare_release_data():
        print("エラー: 分類データ準備に失敗しました")
        sys.exit(1)

    if not analyzer.prepare_ranking_release_data():
        print("エラー: ランキングデータ準備に失敗しました")
        sys.exit(1)

    print(f"分類データ準備完了: {len(analyzer.release_data)}リリース")
    print(f"ランキングデータ準備完了: {len(analyzer.ranking_release_data)}リリース")

    print(f"\n{'='*60}")
    print(f"分類モデル: {classification_model}")
    print(f"{'='*60}")
    classification_result = analyzer.run_with_model(classification_model)

    if classification_result['success']:
        print(f"\n分析完了: {classification_result['project']} ({classification_model})")
        print(f"  リリース数: {classification_result['n_releases']}")
        print(f"  CV結果数: {classification_result['n_cv_results']}")
        print("  出力ファイル:")
        for name, path in classification_result['outputs'].items():
            print(f"    - {name}: {path}")
    else:
        print(f"\n分析失敗 ({classification_model}): {classification_result.get('error', '不明なエラー')}")

    ranking_results = {}
    for ranking_model_type in ranking_model_types:
        print(f"\n{'='*60}")
        print(f"ランキングモデル: {ranking_model_type}")
        print(f"{'='*60}")

        result = analyzer.run_with_ranking_model(ranking_model_type)
        ranking_results[ranking_model_type] = result

        if result['success']:
            print(f"\n分析完了: {result['project']} ({ranking_model_type})")
            print(f"  リリース数: {result['n_releases']}")
            print(f"  CV結果数: {result['n_cv_results']}")
            print("  出力ファイル:")
            for name, path in result['outputs'].items():
                print(f"    - {name}: {path}")
        else:
            print(f"\n分析失敗 ({ranking_model_type}): {result.get('error', '不明なエラー')}")

    print(f"\n{'='*60}")
    print("both モード分析完了")
    print(f"{'='*60}")

    ranking_success_count = sum(1 for r in ranking_results.values() if r['success'])
    print(f"分類成功: {classification_result['success']}")
    print(f"ランキング成功: {ranking_success_count}/{len(ranking_model_types)} モデル")

    if (not classification_result['success']) or (ranking_success_count < len(ranking_model_types)):
        sys.exit(1)


if __name__ == '__main__':
    main()
