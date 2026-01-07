"""
Trend Models Analysis - 特徴量抽出

本モジュールでは、src/featuresモジュールを使用して16種類のメトリクスを計算し、
特徴量としてDataFrame化する機能を提供します。
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from src.config.path import DEFAULT_DATA_DIR, DEFAULT_CONFIG
from src.analysis.trend_models.utils.constants import FEATURE_NAMES

# src/features からのインポート
from src.features.bug_metrics import calculate_bug_fix_confidence
from src.features.change_metrics import (
    calculate_lines_added,
    calculate_lines_deleted,
    calculate_files_changed,
    calculate_elapsed_time,
    calculate_revision_count,
    check_test_code_presence,
)
from src.features.developer_metrics import (
    get_owner_email,
    calculate_past_report_count,
    calculate_recent_report_count,
    calculate_merge_rate,
    calculate_recent_merge_rate,
)
from src.features.project_metrics import (
    calculate_days_to_major_release,
    calculate_predictive_target_ticket_count,
    calculate_reviewed_lines_in_period,
)
from src.features.refactoring_metrics import calculate_refactoring_confidence
from src.features.review_metrics import ReviewStatusAnalyzer

logger = logging.getLogger(__name__)

# ロギング設定
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FeatureExtractor:
    """
    Changeデータから特徴量を抽出するクラス
    
    src/featuresモジュールの関数を使用して16種類のメトリクスを計算します。
    """
    
    def __init__(
        self,
        all_prs_df: pd.DataFrame,
        releases_df: pd.DataFrame,
        project_name: str,
        review_analyzer: ReviewStatusAnalyzer = None,
        data_dir: Path = None
    ):
        """
        Args:
            all_prs_df: 全Changeのデータフレーム（開発者メトリクス計算用）
            releases_df: リリース情報のデータフレーム
            project_name: プロジェクト名
            review_analyzer: レビュー分析クラスのインスタンス（オプション）
            data_dir: データディレクトリのパス
        """
        self.all_prs_df = all_prs_df
        self.releases_df = releases_df
        self.project_name = project_name
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        
        # ReviewStatusAnalyzerの初期化
        if review_analyzer:
            self.review_analyzer = review_analyzer
        else:
            try:
                processed_dir = self.data_dir / 'processed'
                config_dir = DEFAULT_CONFIG
                self.review_analyzer = ReviewStatusAnalyzer(
                    extraction_keywords_path=processed_dir / 'review_keywords.json',
                    gerrymander_config_path=config_dir / 'gerrymanderconfig.ini',
                    review_label_path=processed_dir / 'review_label.json'
                )
            except Exception as e:
                logger.warning(f"ReviewStatusAnalyzerの初期化に失敗: {e}")
                self.review_analyzer = None
    
    def _determine_analysis_time(
        self,
        change_data: Dict[str, Any],
        period_start: datetime = None
    ) -> datetime:
        """
        分析時点を決定する
        
        Args:
            change_data: Changeデータ
            period_start: 期間開始日時
            
        Returns:
            datetime: 分析時点
        """
        created = change_data.get('created')
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace('.000000000', '').replace(' ', 'T'))
        elif not isinstance(created, datetime):
            # 現在時刻をフォールバックとして使用
            return datetime.now()
        
        # 期間開始時点で既にオープンだった場合は期間開始日時を使用
        if period_start and created < period_start:
            return period_start
        
        return created
    
    def extract(
        self,
        change_data: Dict[str, Any],
        period_start: datetime = None
    ) -> Dict[str, Any]:
        """
        単一のChangeから全ての特徴量を抽出する
        
        Args:
            change_data: Changeデータ
            period_start: 期間開始日時（分析時点の決定に使用）
            
        Returns:
            Dict: 特徴量名をキー、特徴量値を値とする辞書
        """
        analysis_time = self._determine_analysis_time(change_data, period_start)
        developer_email = get_owner_email(change_data)
        
        features = {
            'change_number': change_data.get('change_number'),
            'analysis_time': analysis_time,
        }
        
        # Bug Metrics
        try:
            features['bug_fix_confidence'] = calculate_bug_fix_confidence(change_data)
        except Exception as e:
            logger.debug(f"bug_fix_confidence計算エラー: {e}")
            features['bug_fix_confidence'] = 0
        
        # Change Metrics
        try:
            features['lines_added'] = calculate_lines_added(change_data, analysis_time)
        except Exception as e:
            logger.debug(f"lines_added計算エラー: {e}")
            features['lines_added'] = 0
        
        try:
            features['lines_deleted'] = calculate_lines_deleted(change_data, analysis_time)
        except Exception as e:
            logger.debug(f"lines_deleted計算エラー: {e}")
            features['lines_deleted'] = 0
        
        try:
            features['files_changed'] = calculate_files_changed(change_data, analysis_time)
        except Exception as e:
            logger.debug(f"files_changed計算エラー: {e}")
            features['files_changed'] = 0
        
        try:
            features['elapsed_time'] = calculate_elapsed_time(change_data, analysis_time)
        except Exception as e:
            logger.debug(f"elapsed_time計算エラー: {e}")
            features['elapsed_time'] = 0.0
        
        try:
            features['revision_count'] = calculate_revision_count(change_data, analysis_time)
        except Exception as e:
            logger.debug(f"revision_count計算エラー: {e}")
            features['revision_count'] = 1
        
        try:
            features['test_code_presence'] = check_test_code_presence(change_data)
        except Exception as e:
            logger.debug(f"test_code_presence計算エラー: {e}")
            features['test_code_presence'] = 0
        
        # Developer Metrics
        try:
            features['past_report_count'] = calculate_past_report_count(
                developer_email, self.all_prs_df, analysis_time
            )
        except Exception as e:
            logger.debug(f"past_report_count計算エラー: {e}")
            features['past_report_count'] = 0
        
        try:
            features['recent_report_count'] = calculate_recent_report_count(
                developer_email, self.all_prs_df, analysis_time
            )
        except Exception as e:
            logger.debug(f"recent_report_count計算エラー: {e}")
            features['recent_report_count'] = 0
        
        try:
            features['merge_rate'] = calculate_merge_rate(
                developer_email, self.all_prs_df, analysis_time
            )
        except Exception as e:
            logger.debug(f"merge_rate計算エラー: {e}")
            features['merge_rate'] = 0.0
        
        try:
            features['recent_merge_rate'] = calculate_recent_merge_rate(
                developer_email, self.all_prs_df, analysis_time
            )
        except Exception as e:
            logger.debug(f"recent_merge_rate計算エラー: {e}")
            features['recent_merge_rate'] = 0.0
        
        # Project Metrics
        try:
            features['days_to_major_release'] = calculate_days_to_major_release(
                analysis_time, self.project_name, self.releases_df
            )
        except Exception as e:
            logger.debug(f"days_to_major_release計算エラー: {e}")
            features['days_to_major_release'] = -1.0
        
        try:
            features['open_ticket_count'] = calculate_predictive_target_ticket_count(
                self.all_prs_df, analysis_time
            )
        except Exception as e:
            logger.debug(f"open_ticket_count計算エラー: {e}")
            features['open_ticket_count'] = 0
        
        try:
            features['reviewed_lines_in_period'] = calculate_reviewed_lines_in_period(
                self.all_prs_df, analysis_time
            )
        except Exception as e:
            logger.debug(f"reviewed_lines_in_period計算エラー: {e}")
            features['reviewed_lines_in_period'] = 0
        
        # Refactoring Metrics
        try:
            features['refactoring_confidence'] = calculate_refactoring_confidence(change_data)
        except Exception as e:
            logger.debug(f"refactoring_confidence計算エラー: {e}")
            features['refactoring_confidence'] = 0
        
        # Review Metrics
        try:
            if self.review_analyzer:
                uncompleted = self.review_analyzer.analyze_pr_status(
                    change_data, analysis_time
                )
                features['uncompleted_requests'] = uncompleted
            else:
                features['uncompleted_requests'] = 0
        except Exception as e:
            logger.debug(f"uncompleted_requests計算エラー: {e}")
            features['uncompleted_requests'] = 0
        
        return features


def extract_features_from_changes(
    changes: List[Dict[str, Any]],
    all_prs_df: pd.DataFrame,
    releases_df: pd.DataFrame,
    project_name: str,
    period_start: datetime = None,
    review_analyzer: ReviewStatusAnalyzer = None,
    data_dir: Path = None
) -> pd.DataFrame:
    """
    複数のChangeから特徴量を抽出してDataFrameを作成する
    
    Args:
        changes: Changeデータのリスト
        all_prs_df: 全Changeのデータフレーム
        releases_df: リリース情報のデータフレーム
        project_name: プロジェクト名
        period_start: 期間開始日時
        review_analyzer: レビュー分析クラスのインスタンス
        data_dir: データディレクトリのパス
        
    Returns:
        pd.DataFrame: 特徴量のDataFrame
    """
    extractor = FeatureExtractor(
        all_prs_df=all_prs_df,
        releases_df=releases_df,
        project_name=project_name,
        review_analyzer=review_analyzer,
        data_dir=data_dir
    )
    
    records = []
    total = len(changes)
    
    for i, change in enumerate(changes):
        if (i + 1) % 100 == 0:
            logger.debug(f"特徴量抽出中: {i + 1}/{total}")
        
        try:
            features = extractor.extract(change, period_start)
            records.append(features)
        except Exception as e:
            logger.warning(f"Change {change.get('change_number', 'N/A')} の特徴量抽出に失敗: {e}")
    
    df = pd.DataFrame(records)
    
    logger.debug(f"特徴量抽出完了: {len(df)}件")
    return df
