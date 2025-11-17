"""
Release Impact Analysis のメイン分析ロジック
各リリース期間ごとにメトリクスを抽出し、統計分析・可視化を行う
"""

import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from src.config.path import DEFAULT_DATA_DIR
from src.config.release_constants import (
    RELEASE_IMPACT_ANALYSIS, 
    RELEASE_ANALYSIS_PERIODS,
    REVIEW_COUNT_THRESHOLD
)
from src.learning.irl_models import ReviewPriorityDataProcessor
from src.release_impact.metrics_analysis import StatisticalAnalyzer, MetricsVisualizer

logger = logging.getLogger(__name__)


class ReleaseMetricsComparator:
    """
    リリース期間ごとのメトリクス比較を行うクラス
    """
    
    def __init__(self, project_name: str):
        """
        Args:
            project_name (str): プロジェクト名（例: 'nova', 'neutron'）
        """
        self.project_name = project_name
        self.data_processor = ReviewPriorityDataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = MetricsVisualizer()
        
        # プロジェクトのリリース情報を取得
        if project_name not in RELEASE_IMPACT_ANALYSIS:
            raise ValueError(f"プロジェクト '{project_name}' は定義されていません")
        
        self.target_releases = RELEASE_IMPACT_ANALYSIS[project_name]['target_release']
        
        # メトリクスのカラム名リスト
        self.metric_columns = [
            'bug_fix_confidence',
            'lines_added',
            'lines_deleted',
            'files_changed',
            'elapsed_time',
            'revision_count',
            'test_code_presence',
            'past_report_count',
            'recent_report_count',
            'merge_rate',
            'recent_merge_rate',
            'days_to_major_release',
            'open_ticket_count',
            'reviewed_lines_in_period',
            'refactoring_confidence',
            'uncompleted_requests'
        ]
        
        logger.info(f"ReleaseMetricsComparator initialized for project: {project_name}")
    
    def load_release_dates(self) -> pd.DataFrame:
        """
        リリース日付データを読み込む
        
        Returns:
            pd.DataFrame: リリース情報のDataFrame
        """
        releases_csv_path = DEFAULT_DATA_DIR / "openstack" / "releases_summary.csv"
        
        if not releases_csv_path.exists():
            raise FileNotFoundError(f"リリースファイルが見つかりません: {releases_csv_path}")
        
        releases_df = pd.read_csv(releases_csv_path)
        releases_df['release_date'] = pd.to_datetime(releases_df['release_date'])
        
        # 対象プロジェクトのリリースのみをフィルタ
        project_releases = releases_df[
            (releases_df['component'] == self.project_name) &
            (releases_df['version'].isin(self.target_releases))
        ].copy()
        
        logger.info(f"Loaded {len(project_releases)} releases for {self.project_name}")
        
        return project_releases
    
    def get_release_date(self, releases_df: pd.DataFrame, version: str) -> Optional[datetime]:
        """
        指定バージョンのリリース日を取得
        
        Args:
            releases_df (pd.DataFrame): リリース情報のDataFrame
            version (str): バージョン番号
            
        Returns:
            Optional[datetime]: リリース日（見つからない場合はNone）
        """
        release_row = releases_df[releases_df['version'] == version]
        
        if release_row.empty:
            logger.warning(f"バージョン {version} のリリース日が見つかりません")
            return None
        
        return release_row.iloc[0]['release_date']
    
    def extract_period_changes(
        self,
        all_changes_df: pd.DataFrame,
        base_date: datetime,
        offset_start: int,
        offset_end: int,
        review_status: str
    ) -> pd.DataFrame:
        """
        指定期間内のChangeを抽出
        
        Args:
            all_changes_df (pd.DataFrame): 全Changeのデータ
            base_date (datetime): 基準日
            offset_start (int): 開始日のオフセット（日数）
            offset_end (int): 終了日のオフセット（日数）
            review_status (str): レビューステータス（'reviewed' or 'not_reviewed'）
            
        Returns:
            pd.DataFrame: 抽出されたChangeのDataFrame
        """
        start_date = base_date + timedelta(days=offset_start)
        end_date = base_date + timedelta(days=offset_end)
        
        logger.info(
            f"期間抽出: {start_date.date()} ~ {end_date.date()} "
            f"(review_status: {review_status})"
        )
        
        # 期間内にオープンだったChangeを抽出
        # 条件1: 期間終了前に作成されている
        # 条件2: 期間開始時点でまだクローズされていない（マージも放棄もされていない）
        period_changes = all_changes_df[
            (all_changes_df['created'] <= end_date) &
            (
                (all_changes_df['merged'].isna()) | (all_changes_df['merged'] >= start_date)
            ) &
            (
                (all_changes_df['abandoned'].isna()) | (all_changes_df['abandoned'] >= start_date)
            )
        ].copy()
        
        # review_countを計算（messagesの数から算出）
        def count_reviews(row):
            messages = row.get('messages', [])
            if not isinstance(messages, list):
                return 0
            # ボット以外のメッセージ数をカウント
            return len([m for m in messages if m.get('author', {}).get('name', '')])
        
        period_changes['review_count'] = period_changes.apply(count_reviews, axis=1)
        
        # レビューステータスでフィルタ
        if review_status == 'reviewed':
            filtered_changes = period_changes[
                period_changes['review_count'] > REVIEW_COUNT_THRESHOLD
            ]
        else:  # 'not_reviewed'
            filtered_changes = period_changes[
                period_changes['review_count'] <= REVIEW_COUNT_THRESHOLD
            ]
        
        logger.info(f"抽出されたChange数: {len(filtered_changes)}")
        
        return filtered_changes
    
    def analyze_release_pair(
        self,
        current_release: str,
        next_release: str,
        releases_df: pd.DataFrame,
        all_changes_df: pd.DataFrame,
        all_releases_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        1つのリリース期間を分析
        current_releaseの初期・終期（次のリリースまで）の4期間を比較
        
        Args:
            current_release (str): 分析対象のリリースバージョン
            next_release (str): 次のリリースバージョン（終期の計算に使用）
            releases_df (pd.DataFrame): プロジェクトのリリース情報
            all_changes_df (pd.DataFrame): 全Changeのデータ
            all_releases_df (pd.DataFrame): 全プロジェクトのリリース情報
            
        Returns:
            Tuple[pd.DataFrame, Dict, Dict]: (メトリクスデータ, 統計量, 検定結果)
        """
        logger.info(f"=== リリース期間分析開始: {current_release} (終期基準: {next_release}) ===")
        
        # リリース日を取得
        current_date = self.get_release_date(releases_df, current_release)
        next_date = self.get_release_date(releases_df, next_release)
        
        if current_date is None or next_date is None:
            raise ValueError(f"リリース日が取得できません: {current_release}, {next_release}")
        
        logger.info(f"分析対象リリース日: {current_date.date()}")
        logger.info(f"次リリース日（終期計算用）: {next_date.date()}")
        
        # 4つの期間グループを抽出
        all_metrics = []
        
        for period_name, period_config in RELEASE_ANALYSIS_PERIODS.items():
            logger.info(f"--- 期間グループ処理中: {period_name} ---")
            
            # 基準日を決定
            if period_config['base_date'] == 'current_release':
                base_date = current_date
            else:  # 'next_release'
                base_date = next_date
            
            # 期間内のChangeを抽出
            period_changes = self.extract_period_changes(
                all_changes_df,
                base_date,
                period_config['offset_start'],
                period_config['offset_end'],
                period_config['review_status']
            )
            
            if len(period_changes) == 0:
                logger.warning(f"期間 '{period_name}' にデータがありません")
                continue
            
            # 分析時点を期間の終了日に設定
            analysis_time = base_date + timedelta(days=period_config['offset_end'])
            
            # メトリクスを抽出
            try:
                features_df = self.data_processor.extract_features(
                    period_changes,
                    analysis_time,
                    self.project_name,
                    all_releases_df
                )
                
                # 期間グループ名を追加
                features_df['period_group'] = period_name
                
                all_metrics.append(features_df)
                
                logger.info(f"期間 '{period_name}' のメトリクス抽出完了: {len(features_df)} 件")
                
            except Exception as e:
                logger.error(f"期間 '{period_name}' のメトリクス抽出でエラー: {e}")
                continue
        
        # 全期間のメトリクスを結合
        if not all_metrics:
            raise ValueError("すべての期間でメトリクスの抽出に失敗しました")
        
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        logger.info(f"全期間のメトリクス結合完了: {len(metrics_df)} 件")
        
        # 記述統計量を計算
        statistics = self.statistical_analyzer.calculate_descriptive_statistics(
            metrics_df,
            'period_group',
            self.metric_columns
        )
        
        # 統計検定を実行
        comparison_pairs = [
            ('early_reviewed', 'late_reviewed'),
            ('early_not_reviewed', 'late_not_reviewed'),
            ('early_reviewed', 'early_not_reviewed'),
            ('late_reviewed', 'late_not_reviewed')
        ]
        
        test_results = self.statistical_analyzer.perform_multiple_comparisons(
            metrics_df,
            'period_group',
            self.metric_columns,
            comparison_pairs
        )
        
        return metrics_df, statistics, test_results
    
    def save_results(
        self,
        release_period_name: str,
        metrics_df: pd.DataFrame,
        statistics: Dict,
        test_results: Dict
    ):
        """
        分析結果を保存
        
        Args:
            release_period_name (str): リリース期間の名前
            metrics_df (pd.DataFrame): メトリクスデータ
            statistics (Dict): 統計量
            test_results (Dict): 検定結果
        """
        # 出力ディレクトリを作成
        output_dir = DEFAULT_DATA_DIR / "release_impact" / f"{self.project_name}_{release_period_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"結果を保存中: {output_dir}")
        
        # メトリクスデータをCSVに保存
        csv_path = output_dir / "metrics_data.csv"
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"メトリクスデータを保存: {csv_path}")
        
        # 統計量を保存
        stats_path = output_dir / "summary_statistics.json"
        self.statistical_analyzer.save_statistics(statistics, stats_path)
        
        # 検定結果を保存
        test_path = output_dir / "test_results.json"
        self.statistical_analyzer.save_test_results(test_results, test_path)
        
        # ボックスプロットを作成
        boxplot_path = output_dir / "boxplots_4x4.pdf"
        self.visualizer.create_boxplots(
            metrics_df,
            'period_group',
            self.metric_columns,
            boxplot_path
        )
        
        # ヒートマップを作成
        heatmap_path = output_dir / "heatmap.pdf"
        self.visualizer.create_heatmap(
            test_results,
            heatmap_path,
            self.metric_columns
        )
        
        # サマリープロットを作成
        summary_path = output_dir / "summary_plot.pdf"
        self.visualizer.create_summary_plot(
            statistics,
            summary_path,
            self.metric_columns
        )
        
        logger.info(f"すべての結果を保存完了: {output_dir}")
    
    def run_analysis_with_preloaded_data(
        self,
        all_changes_df: pd.DataFrame,
        all_releases_df: pd.DataFrame
    ):
        """
        事前にロードされたデータを使用して全リリース期間の分析を実行
        
        Args:
            all_changes_df (pd.DataFrame): 全プロジェクトの全Changeデータ
            all_releases_df (pd.DataFrame): 全プロジェクトのリリース情報
        """
        logger.info(f"===== {self.project_name} の分析を開始 =====")
        
        # 対象プロジェクトのChangeのみをフィルタ
        project_changes = all_changes_df[
            all_changes_df['component'] == self.project_name
        ].copy()
        
        logger.info(f"プロジェクト '{self.project_name}' のChange数: {len(project_changes)}")
        
        # リリース日付データを読み込み
        releases_df = self.load_release_dates()
        
        # 各リリース期間を分析（次のリリースまでの期間）
        for i in range(len(self.target_releases) - 1):
            current_release = self.target_releases[i]
            next_release = self.target_releases[i + 1]
            release_period_name = f"{current_release}_period"
            
            try:
                # 分析を実行
                metrics_df, statistics, test_results = self.analyze_release_pair(
                    current_release,
                    next_release,
                    releases_df,
                    project_changes,
                    all_releases_df
                )
                
                # 結果を保存
                self.save_results(
                    release_period_name,
                    metrics_df,
                    statistics,
                    test_results
                )
                
                logger.info(f"リリース期間 {current_release} の分析完了\n")
                
            except Exception as e:
                logger.error(f"リリース期間 {current_release} の分析でエラー: {e}")
                continue
        
        logger.info(f"===== {self.project_name} の分析完了 =====")
    
    def run_analysis(self):
        """
        全リリース期間の分析を実行（データを内部でロード）
        単一プロジェクトを分析する場合に使用
        """
        logger.info(f"===== {self.project_name} の分析を開始 =====")
        
        # データを読み込み
        logger.info("データを読み込み中...")
        all_changes_df, all_releases_df = self.data_processor.load_openstack_data()
        
        # 対象プロジェクトのChangeのみをフィルタ
        project_changes = all_changes_df[
            all_changes_df['component'] == self.project_name
        ].copy()
        
        logger.info(f"プロジェクト '{self.project_name}' のChange数: {len(project_changes)}")
        
        # リリース日付データを読み込み
        releases_df = self.load_release_dates()
        
        # 各リリース期間を分析（次のリリースまでの期間）
        for i in range(len(self.target_releases) - 1):
            current_release = self.target_releases[i]
            next_release = self.target_releases[i + 1]
            release_period_name = f"{current_release}_period"
            
            try:
                # 分析を実行
                metrics_df, statistics, test_results = self.analyze_release_pair(
                    current_release,
                    next_release,
                    releases_df,
                    project_changes,
                    all_releases_df
                )
                
                # 結果を保存
                self.save_results(
                    release_period_name,
                    metrics_df,
                    statistics,
                    test_results
                )
                
                logger.info(f"リリース期間 {current_release} の分析完了\n")
                
            except Exception as e:
                logger.error(f"リリース期間 {current_release} の分析でエラー: {e}")
                continue
        
        logger.info(f"===== {self.project_name} の分析完了 =====")


def main():
    """
    メイン関数
    """
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # データプロセッサを1回だけ初期化（キーワード・ラベル・ボットデータを一度だけロード）
    logger.info("共通データを読み込み中...")
    data_processor = ReviewPriorityDataProcessor()
    
    # 全プロジェクトで共有するデータを一度だけロード
    logger.info("OpenStackデータを読み込み中...")
    all_changes_df, all_releases_df = data_processor.load_openstack_data()
    logger.info(f"全Change数: {len(all_changes_df)}, 全リリース数: {len(all_releases_df)}")
    
    # 分析対象のプロジェクト
    projects = ['nova', 'neutron', 'cinder', 'glance', 'keystone', 'swift']
    
    for project in projects:
        try:
            comparator = ReleaseMetricsComparator(project)
            # 既にロードされたdata_processorと共有データを使用
            comparator.data_processor = data_processor
            comparator.run_analysis_with_preloaded_data(all_changes_df, all_releases_df)
        except Exception as e:
            logger.error(f"プロジェクト {project} の分析でエラー: {e}")
            continue


if __name__ == '__main__':
    main()
