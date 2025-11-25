"""
Trend Metrics Analysis のメインエントリーポイント
前期/後期でメトリクスの値がどのように変化するかを分析する
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.trend_metrics.utils.constants import (
    TREND_ANALYSIS_CONFIG,
    ANALYSIS_PERIODS,
    ANALYSIS_GROUPS,
    METRIC_COLUMNS,
    OUTPUT_DIR_BASE,
    METRIC_DATA_SCOPE,
    RECENT_DATA_PERIOD_DAYS
)
from src.analysis.trend_metrics.utils.data_loader import (
    load_major_releases_summary,
    get_release_date,
    load_core_developers,
    load_all_changes,
    load_bot_names_from_config
)
from src.analysis.trend_metrics.metrics_extraction import (
    calculate_periods,
    extract_changes_in_period,
    add_reviewer_info_to_changes,
    classify_changes_into_groups
)
from src.analysis.trend_metrics.metrics_extraction.metrics_calculator import (
    calculate_metrics,
    enrich_changes_with_line_metrics,
    enrich_changes_with_owner_email
)
from src.analysis.trend_metrics.analysis.statistical_analyzer import (
    calculate_summary_statistics,
    perform_all_statistical_tests,
    summarize_significant_results
)
from src.analysis.trend_metrics.analysis.trend_comparator import (
    compare_early_vs_late_by_reviewer_type
)
from src.analysis.trend_metrics.visualization import (
    plot_boxplots_8groups,
    plot_trend_lines,
    plot_metric_changes,
    generate_heatmap
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrendMetricsAnalyzer:
    """
    トレンドメトリクス分析のメインクラス
    前期/後期でコアレビューア/非コアレビューアのレビュー行動とメトリクスの変化を分析
    """
    
    def __init__(
        self,
        project_name: Optional[str] = None,
        target_releases: Optional[List[str]] = None
    ):
        """
        Args:
            project_name (str, optional): プロジェクト名（例: 'nova', 'neutron'）
                指定しない場合はconstants.pyの設定を使用
            target_releases (List[str], optional): 分析対象リリース [current, next]
                指定しない場合はconstants.pyの設定を使用
        """
        self.project_name = project_name or TREND_ANALYSIS_CONFIG['project']
        self.target_releases = target_releases or TREND_ANALYSIS_CONFIG['target_releases']
        
        if len(self.target_releases) != 2:
            raise ValueError("target_releasesは[current_release, next_release]の2要素が必要です")
        
        self.current_release = self.target_releases[0]
        self.next_release = self.target_releases[1]
        
        # 出力ディレクトリの設定
        self.output_dir = Path(OUTPUT_DIR_BASE) / (
            f"{self.project_name}_{self.current_release}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TrendMetricsAnalyzer initialized")
        logger.info(f"  Project: {self.project_name}")
        logger.info(f"  Releases: {self.current_release} -> {self.next_release}")
        logger.info(f"  Output: {self.output_dir}")
    
    def run_analysis(self) -> Dict:
        """
        メイン分析フローを実行
        
        Returns:
            Dict: 分析結果のサマリー
        """
        logger.info("=" * 80)
        logger.info("Trend Metrics Analysis 開始")
        logger.info("=" * 80)
        
        try:
            # 1. データ準備フェーズ
            logger.info("\n[Phase 1] データ準備")
            data_context = self._prepare_data()
            
            # 2. データ抽出フェーズ
            logger.info("\n[Phase 2] 期間別データ抽出")
            period_data = self._extract_period_data(data_context)
            
            # 3. レビューア分類フェーズ
            logger.info("\n[Phase 3] レビューアタイプ分類")
            classified_data = self._classify_by_reviewer(period_data, data_context)
            
            # 4. 統計分析フェーズ
            logger.info("\n[Phase 4] 統計分析")
            analysis_results = self._perform_statistical_analysis(classified_data)
            
            # 5. 可視化フェーズ
            logger.info("\n[Phase 5] 可視化")
            self._generate_visualizations(classified_data, analysis_results)
            
            # 6. データの保存
            logger.info("\n[Phase 6] データの保存")
            summary = self._save_classified_data(classified_data, analysis_results, data_context)
            
            logger.info("\n" + "=" * 80)
            logger.info("分析完了")
            logger.info(f"結果は {self.output_dir} に保存されました")
            logger.info("=" * 80)
            
            return summary
            
        except Exception as e:
            logger.error(f"分析中にエラーが発生しました: {e}", exc_info=True)
            raise
    
    def _prepare_data(self) -> Dict:
        """
        データ準備フェーズ
        - リリース日付の取得
        - 期間の計算
        - Changeデータの読み込み
        - コアレビューア情報の読み込み
        - bot名の読み込み
        
        Returns:
            Dict: データコンテキスト
        """
        logger.info("  メジャーリリース情報を読み込んでいます...")
        releases_df = load_major_releases_summary()
        
        logger.info("  リリース日を取得しています...")
        current_release_date = get_release_date(releases_df, self.project_name, self.current_release)
        next_release_date = get_release_date(releases_df, self.project_name, self.next_release)
        
        logger.info("  期間を計算しています...")
        periods = calculate_periods(current_release_date, next_release_date, ANALYSIS_PERIODS)
        early_period, late_period = periods
        
        logger.info("  Changeデータを読み込んでいます...")
        all_changes = load_all_changes(self.project_name)
        
        logger.info("  コアレビューア情報を読み込んでいます...")
        core_reviewers = load_core_developers()
        
        logger.info("  bot名リストを読み込んでいます...")
        bot_names = load_bot_names_from_config()
        
        data_context = {
            'releases_df': releases_df,
            'current_release_date': current_release_date,
            'next_release_date': next_release_date,
            'early_period': early_period,
            'late_period': late_period,
            'all_changes': all_changes,
            'core_reviewers': core_reviewers,
            'bot_names': bot_names
        }
        
        logger.info(f"  データ準備完了: {len(all_changes)} 件のChange")
        
        return data_context
    
    def _extract_period_data(self, data_context: Dict) -> Dict:
        """
        期間別データ抽出フェーズ
        - 前期/後期のChangeを抽出
        - レビューア情報を抽出
        - メトリクスを計算して付与
        
        Args:
            data_context (Dict): データコンテキスト
            
        Returns:
            Dict: 期間別データ
        """
        all_changes = data_context['all_changes']
        early_period = data_context['early_period']
        late_period = data_context['late_period']
        bot_names = data_context['bot_names']
        next_release_date = data_context['next_release_date']
        releases_df = data_context['releases_df']
        
        # DataFrameの準備（開発者メトリクス計算用）
        # 事前にlines_addedとlines_deletedを計算して追加
        # これにより、project_metrics.pyのcalculate_reviewed_lines_in_periodでの警告を回避
        enrich_changes_with_line_metrics(all_changes)
        
        # owner_emailも事前に計算して追加（get_owner_emailのロジックを使用するため）
        enrich_changes_with_owner_email(all_changes)

        all_changes_df = pd.DataFrame(all_changes)

        # 必要なカラムが存在しない場合は追加
        for col in ['created', 'updated', 'submitted', 'merged']:
            if col not in all_changes_df.columns:
                all_changes_df[col] = pd.NaT
        
        # owner_emailカラムはenrich_changes_with_owner_emailで追加済みなので、ここでは何もしない
        # ただし、万が一欠損している場合のためにfillnaしておく
        if 'owner_email' not in all_changes_df.columns:
             all_changes_df['owner_email'] = ''
        all_changes_df['owner_email'] = all_changes_df['owner_email'].fillna('')
        
        # 日時カラムを変換
        for col in ['created', 'updated', 'submitted', 'merged']:
            if col in all_changes_df.columns:
                all_changes_df[col] = all_changes_df[col].astype(str).str.replace('.000000000', '', regex=False)
                all_changes_df[col] = pd.to_datetime(all_changes_df[col], errors='coerce')
        
        # mergedカラムがNaTの場合、submittedカラムの値で埋める
        # Gerritのデータではsubmittedがマージ日時を表すことが多いため
        if 'submitted' in all_changes_df.columns and 'merged' in all_changes_df.columns:
            all_changes_df['merged'] = all_changes_df['merged'].fillna(all_changes_df['submitted'])
        
        logger.info("  前期のChangeを抽出しています...")
        early_changes = extract_changes_in_period(all_changes, early_period, next_release_date)
        
        logger.info("  後期のChangeを抽出しています...")
        late_changes = extract_changes_in_period(all_changes, late_period, next_release_date)
        
        logger.info("  レビューア情報を抽出しています...")
        early_changes = add_reviewer_info_to_changes(early_changes, bot_names, early_period)
        late_changes = add_reviewer_info_to_changes(late_changes, bot_names, late_period)
        
        logger.info("  メトリクスを計算しています...")
        for change in early_changes:
            metrics = calculate_metrics(change, all_changes_df, releases_df, self.project_name, early_period[0])
            change.update(metrics)
            
        for change in late_changes:
            metrics = calculate_metrics(change, all_changes_df, releases_df, self.project_name, late_period[0])
            change.update(metrics)
        
        period_data = {
            'early_changes': early_changes,
            'late_changes': late_changes
        }
        
        return period_data
    
    def _classify_by_reviewer(self, period_data: Dict, data_context: Dict) -> Dict:
        """
        レビューア分類フェーズ
        - 各Changeをレビューアタイプで分類
        - 8グループに分割
        
        Args:
            period_data (Dict): 期間別データ
            data_context (Dict): データコンテキスト
            
        Returns:
            Dict: 分類済みデータ（8グループ）
        """
        early_changes = period_data['early_changes']
        late_changes = period_data['late_changes']
        core_reviewers = data_context['core_reviewers']
        
        logger.info("  Changeをレビューアタイプで分類しています...")
        classified_data = classify_changes_into_groups(
            early_changes,
            late_changes,
            core_reviewers,
            self.project_name
        )
        
        return classified_data
    
    def _perform_statistical_analysis(self, classified_data: Dict) -> Dict:
        """
        統計分析フェーズ
        - 記述統計量の計算
        - 統計検定の実行
        - 前期/後期の比較
        
        Args:
            classified_data (Dict): 分類済みデータ
            
        Returns:
            Dict: 分析結果
        """
        logger.info("  記述統計量を計算しています...")
        summary_stats = calculate_summary_statistics(classified_data)
        
        logger.info("  統計検定を実行しています...")
        test_results = perform_all_statistical_tests(classified_data)
        
        logger.info("  有意な結果を抽出しています...")
        significant_results = summarize_significant_results(test_results)
        
        logger.info("  前期/後期の比較を行っています...")
        comparison_results = compare_early_vs_late_by_reviewer_type(classified_data)
        
        return {
            'summary_stats': summary_stats,
            'test_results': test_results,
            'significant_results': significant_results,
            'comparison_results': comparison_results
        }
    
    def _generate_visualizations(self, classified_data: Dict, analysis_results: Dict):
        """
        可視化フェーズ
        - 箱ひげ図の生成
        - トレンドラインの生成
        - 変化率グラフの生成
        - ヒートマップの生成
        
        Args:
            classified_data (Dict): 分類済みデータ
            analysis_results (Dict): 分析結果
        """
        logger.info("  グラフを生成しています...")
        
        # 1. 分布の可視化
        plot_boxplots_8groups(classified_data, self.output_dir)
        
        # 2. トレンドの可視化
        plot_trend_lines(analysis_results['comparison_results'], self.output_dir)
        
        # 3. 変化率の可視化
        plot_metric_changes(analysis_results['comparison_results'], self.output_dir)
        
        # 4. 統計検定結果のヒートマップ
        generate_heatmap(analysis_results['test_results'], self.output_dir)
    
    def _save_classified_data(
        self,
        classified_data: Dict,
        analysis_results: Dict,
        data_context: Dict
    ) -> Dict:
        """
        分類済みデータの保存
        - 基本情報をCSVに保存
        - グループ統計をJSONに保存
        - 統計分析結果をJSONに保存
        
        Args:
            classified_data (Dict): 分類済みデータ
            analysis_results (Dict): 分析結果
            data_context (Dict): データコンテキスト
            
        Returns:
            Dict: サマリー情報
        """
        logger.info("  分類結果を保存しています...")
        
        # DataFrameに変換
        records = []
        for group_name, changes in classified_data.items():
            period = 'early' if group_name.startswith('early_') else 'late'
            reviewer_type = group_name.replace('early_', '').replace('late_', '')
            
            for change in changes:
                record = {
                    'change_number': change.get('change_number', change.get('_number')),
                    'project': self.project_name,
                    'period': period,
                    'reviewer_type': reviewer_type,
                    'created': change.get('created'),
                    'status': change.get('status'),
                    'subject': change.get('subject'),
                    'reviewers': ','.join(change.get('reviewers', []))
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        
        # CSVに保存
        csv_path = self.output_dir / 'classified_changes.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"  分類データを保存: {csv_path}")
        
        # グループ統計を保存
        group_stats = {
            group: len(changes) 
            for group, changes in classified_data.items()
        }
        
        stats_path = self.output_dir / 'group_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(group_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"  グループ統計を保存: {stats_path}")
        
        # 統計分析結果を保存
        analysis_path = self.output_dir / 'statistical_analysis.json'
        # numpy型などをシリアライズ可能にするためのカスタムエンコーダが必要かもしれないが
        # ここでは簡易的にdefault=strを使用
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"  統計分析結果を保存: {analysis_path}")
        
        # サマリー作成
        summary = {
            'project': self.project_name,
            'releases': self.target_releases,
            'output_dir': str(self.output_dir),
            'groups_count': group_stats,
            'total_changes': len(df),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        summary_path = self.output_dir / 'analysis_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"  サマリーを保存: {summary_path}")
        
        return summary


def main():
    """
    メイン関数
    """
    # デフォルト設定で分析を実行
    analyzer = TrendMetricsAnalyzer()
    
    try:
        summary = analyzer.run_analysis()
        
        print("\n" + "=" * 80)
        print("分析サマリー")
        print("=" * 80)
        print(f"プロジェクト: {summary['project']}")
        print(f"リリース: {summary['releases'][0]} -> {summary['releases'][1]}")
        print(f"出力先: {summary['output_dir']}")
        print(f"総Change数: {summary['total_changes']}")
        print(f"\nグループ別件数:")
        for group, count in summary['groups_count'].items():
            print(f"  {group}: {count}")
        print(f"\n完了時刻: {summary['timestamp']}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"分析に失敗しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
