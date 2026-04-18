"""
Daily Regression Analysis メインモジュール
日付ごとにOLS重回帰分析を実行し、回帰係数の時系列的遷移を分析する
"""

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.analysis.daily_regression.utils.constants import (
    DAILY_REGRESSION_CONFIG,
    METRIC_COLUMNS,
    OUTPUT_DIR_BASE,
    MAX_CENSORING_SECONDS,
    MIN_SAMPLES,
    EXCLUSION_WINDOW_SECONDS,
    VISUALIZATION_CONFIG,
    TARGET_COLUMN_BY_MODE,
    DEFAULT_TARGET_MODE,
)
from src.analysis.daily_regression.utils.data_loader import (
    load_major_releases_summary,
    get_release_date,
    load_all_changes,
    load_bot_names_from_config,
)
from src.analysis.daily_regression.regression.sample_extractor import (
    extract_daily_samples,
)
from src.analysis.daily_regression.regression.metrics_calculator import (
    calculate_daily_metrics,
    get_review_analyzer,
)
from src.analysis.daily_regression.regression.ols_executor import execute_ols
from src.analysis.daily_regression.visualization.coefficient_plotter import (
    plot_all_coefficients,
    plot_r_squared_timeseries,
)
from src.analysis.trend_metrics.metrics_extraction.metrics_calculator import (
    enrich_changes_with_line_metrics,
    enrich_changes_with_owner_email,
)

logger = logging.getLogger(__name__)


class DailyRegressionAnalyzer:
    """日次重回帰分析を実行するメインクラス"""

    def __init__(
        self,
        project_name: str = "nova",
        versions: Optional[List[str]] = None,
        target_mode: str = DEFAULT_TARGET_MODE,
    ):
        """
        Args:
            project_name: プロジェクト名（デフォルト: nova）
            versions: 分析対象バージョンリスト（省略時はconstants.pyの定義を使用）
            target_mode: 目的変数モード（'rank' または 'time'）
        """
        self.project_name = project_name
        if versions is not None:
            self.versions = versions
        else:
            project_config = DAILY_REGRESSION_CONFIG.get('project', {})
            self.versions = project_config.get(project_name, [])

        if target_mode not in TARGET_COLUMN_BY_MODE:
            allowed = ', '.join(TARGET_COLUMN_BY_MODE.keys())
            raise ValueError(f"target_modeは次のいずれかを指定してください: {allowed}")

        self.target_mode = target_mode
        self.target_col = TARGET_COLUMN_BY_MODE[target_mode]

        logger.info(
            f"DailyRegressionAnalyzer初期化: project={project_name}, "
            f"versions={len(self.versions)}個, target={self.target_mode} ({self.target_col})"
        )

    def run_analysis(self) -> Dict:
        """
        全バージョンの日次回帰分析を実行する

        Returns:
            Dict: 全体サマリー
        """
        logger.info("=== Daily Regression Analysis 開始 ===")

        # 1. データ準備
        data_context = self._prepare_data()

        all_version_results = {}
        summary_dir = Path(OUTPUT_DIR_BASE) / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # 2. バージョンごとのループ
        for version in self.versions:
            logger.info(f"--- バージョン {version} の分析開始 ---")
            try:
                version_result = self._analyze_version(version, data_context)
                all_version_results[version] = version_result
            except Exception as e:
                logger.error(f"バージョン {version} の分析でエラー: {e}")
                all_version_results[version] = {'error': str(e)}

        # 3. 全体サマリーの保存
        summary = self._build_summary(all_version_results)
        summary_path = summary_dir / "analysis_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"全体サマリーを保存しました: {summary_path}")
        logger.info("=== Daily Regression Analysis 完了 ===")

        return summary

    def _prepare_data(self) -> Dict:
        """
        データ読み込み（Change, リリース, Bot）

        Returns:
            Dict: data_context
        """
        logger.info("データ準備中...")

        releases_df = load_major_releases_summary()
        all_changes = load_all_changes(project=self.project_name)
        bot_names = load_bot_names_from_config()

        # 行数メトリクスとowner_emailを事前計算
        all_changes = enrich_changes_with_line_metrics(all_changes)
        all_changes = enrich_changes_with_owner_email(all_changes)

        # DataFrame化
        all_changes_df = self._build_changes_dataframe(all_changes)

        # ReviewStatusAnalyzerの事前初期化
        get_review_analyzer()

        logger.info(
            f"データ準備完了: changes={len(all_changes)}, "
            f"releases={len(releases_df)}, bots={len(bot_names)}"
        )

        return {
            'releases_df': releases_df,
            'all_changes': all_changes,
            'all_changes_df': all_changes_df,
            'bot_names': bot_names,
        }

    def _build_changes_dataframe(self, all_changes: List[Dict]) -> pd.DataFrame:
        """Changeリストからメトリクス計算用のDataFrameを構築する"""
        records = []
        for change in all_changes:
            created_str = change.get('created')
            if not created_str:
                continue

            try:
                created_dt = pd.to_datetime(created_str)
            except Exception:
                continue

            # merged / submitted の取得
            submitted_str = change.get('submitted')
            updated_str = change.get('updated')
            status = change.get('status', '')

            merged_dt = None
            if status == 'MERGED':
                if submitted_str:
                    try:
                        merged_dt = pd.to_datetime(submitted_str)
                    except Exception:
                        pass
                if merged_dt is None and updated_str:
                    try:
                        merged_dt = pd.to_datetime(updated_str)
                    except Exception:
                        pass

            owner_email = change.get('owner_email', '')
            if not owner_email:
                owner = change.get('owner', {})
                owner_email = owner.get('email', '')

            records.append({
                'created': created_dt,
                'merged': merged_dt,
                'status': status,
                'owner_email': owner_email,
                'lines_added': change.get('lines_added', 0),
                'lines_deleted': change.get('lines_deleted', 0),
                'updated': pd.to_datetime(updated_str) if updated_str else pd.NaT,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df['owner_email'] = df['owner_email'].fillna('')
        return df

    def _get_analysis_period(
        self, version: str, data_context: Dict
    ) -> Tuple[date, date]:
        """
        バージョンの分析期間（リリース日～次リリース前日）を取得する

        Args:
            version: 対象バージョン
            data_context: データコンテキスト

        Returns:
            Tuple[date, date]: (開始日, 終了日)
        """
        releases_df = data_context['releases_df']

        release_date = get_release_date(
            releases_df, self.project_name, version
        )
        start_date = release_date.date()

        # 次のリリースを探す
        version_idx = self.versions.index(version)
        if version_idx < len(self.versions) - 1:
            next_version = self.versions[version_idx + 1]
            next_release_date = get_release_date(
                releases_df, self.project_name, next_version
            )
            end_date = next_release_date.date() - timedelta(days=1)
        else:
            # 最終バージョン: 次のメジャーリリースを検索
            project_col = 'project' if 'project' in releases_df.columns else 'component'
            project_releases = releases_df[
                releases_df[project_col] == self.project_name
            ].sort_values('release_date')

            future_releases = project_releases[
                project_releases['release_date'] > release_date
            ]
            if not future_releases.empty:
                next_release_date = future_releases.iloc[0]['release_date']
                end_date = next_release_date.date() - timedelta(days=1)
            else:
                # フォールバック: 6ヶ月後
                end_date = start_date + timedelta(days=180)

        logger.info(f"バージョン {version} 分析期間: {start_date} ～ {end_date}")
        return start_date, end_date

    def _analyze_version(self, version: str, data_context: Dict) -> Dict:
        """
        1バージョンの日次回帰分析を実行する

        Args:
            version: バージョン
            data_context: データコンテキスト

        Returns:
            Dict: バージョンの分析結果
        """
        start_date, end_date = self._get_analysis_period(version, data_context)

        daily_results_list = []
        skipped_dates = []
        total_dates = 0

        # 日付リストを生成してtqdmで進捗表示
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)

        for current_date in tqdm(
            date_list,
            desc=f"v{version}",
            unit="day",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} days [{elapsed}<{remaining}]",
        ):
            total_dates += 1
            result = self._analyze_single_day(current_date, data_context)

            if result is None:
                skipped_dates.append(str(current_date))
            else:
                daily_results_list.append(result)

        # 結果をDataFrameに変換
        if daily_results_list:
            daily_df = pd.DataFrame(daily_results_list)
        else:
            daily_df = pd.DataFrame()

        # 出力ディレクトリ
        output_dir = Path(OUTPUT_DIR_BASE) / f"{self.project_name}_{version}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 結果保存
        self._save_results(version, daily_df, daily_results_list,
                           skipped_dates, start_date, end_date,
                           total_dates, output_dir)

        # 可視化
        if not daily_df.empty:
            plots_dir = output_dir / "plots"
            self._generate_visualizations(version, daily_df, plots_dir)

        return {
            'version': version,
            'start_date': str(start_date),
            'end_date': str(end_date),
            'total_dates': total_dates,
            'analyzed_dates': len(daily_results_list),
            'skipped_dates_count': len(skipped_dates),
            'target_mode': self.target_mode,
            'target_column': self.target_col,
        }

    def _analyze_single_day(
        self, analysis_date: date, data_context: Dict
    ) -> Optional[Dict]:
        """
        1日分の回帰分析を実行する

        Args:
            analysis_date: 分析対象日
            data_context: データコンテキスト

        Returns:
            Optional[Dict]: 回帰結果の行データ。スキップ時はNone
        """
        # 1. サンプル抽出
        samples_df = extract_daily_samples(
            analysis_date=analysis_date,
            all_changes=data_context['all_changes'],
            bot_names=data_context['bot_names'],
        )

        if len(samples_df) < MIN_SAMPLES:
            return None

        # 2. メトリクス計算
        metrics_df = calculate_daily_metrics(
            samples_df=samples_df,
            all_changes=data_context['all_changes'],
            all_changes_df=data_context['all_changes_df'],
            releases_df=data_context['releases_df'],
            project_name=self.project_name,
            target_col=self.target_col,
        )

        if len(metrics_df) < MIN_SAMPLES:
            return None

        # 3. OLS実行（標準化回帰係数を算出）
        ols_result = execute_ols(
            metrics_df,
            target_col=self.target_col,
            standardize=True,
        )

        if ols_result is None:
            return None

        # 4. 行データの構築
        row = {
            'date': str(analysis_date),
            'n_samples': ols_result['n_samples'],
            'r_squared': ols_result['r_squared'],
            'adj_r_squared': ols_result['adj_r_squared'],
            'f_statistic': ols_result['f_statistic'],
            'f_pvalue': ols_result['f_pvalue'],
        }

        for metric_name in METRIC_COLUMNS:
            coef_info = ols_result['coefficients'].get(metric_name, {})
            row[f'coef_{metric_name}'] = coef_info.get('coef', np.nan)
            row[f'pvalue_{metric_name}'] = coef_info.get('p_value', np.nan)
            row[f'stderr_{metric_name}'] = coef_info.get('std_err', np.nan)
            row[f'tvalue_{metric_name}'] = coef_info.get('t_value', np.nan)

        # 定数項
        const_info = ols_result['coefficients'].get('const', {})
        row['coef_const'] = const_info.get('coef', np.nan)
        row['pvalue_const'] = const_info.get('p_value', np.nan)

        return row

    def _save_results(
        self,
        version: str,
        daily_df: pd.DataFrame,
        daily_results_list: List[Dict],
        skipped_dates: List[str],
        start_date: date,
        end_date: date,
        total_dates: int,
        output_dir: Path,
    ) -> None:
        """結果をCSV/JSONに保存する"""
        output_dir.mkdir(parents=True, exist_ok=True)

        if not daily_df.empty:
            # daily_coefficients.csv
            coef_cols = ['date', 'n_samples', 'r_squared', 'adj_r_squared']
            for m in METRIC_COLUMNS:
                coef_cols.extend([f'coef_{m}', f'pvalue_{m}'])
            existing_cols = [c for c in coef_cols if c in daily_df.columns]
            daily_df[existing_cols].to_csv(
                output_dir / 'daily_coefficients.csv', index=False
            )

            # daily_regression_stats.csv
            stats_cols = [
                'date', 'n_samples', 'r_squared', 'adj_r_squared',
                'f_statistic', 'f_pvalue'
            ]
            existing_stats_cols = [c for c in stats_cols if c in daily_df.columns]
            daily_df[existing_stats_cols].to_csv(
                output_dir / 'daily_regression_stats.csv', index=False
            )

        # daily_regression_detail.json
        detail = {
            'version': version,
            'project': self.project_name,
            'analysis_period': {
                'start': str(start_date),
                'end': str(end_date),
            },
            'daily_results': {},
            'skipped_dates': skipped_dates,
            'metadata': {
                'max_censoring_seconds': MAX_CENSORING_SECONDS,
                'min_samples': MIN_SAMPLES,
                'exclusion_window_seconds': EXCLUSION_WINDOW_SECONDS,
                'target_mode': self.target_mode,
                'target_column': self.target_col,
                'total_dates': total_dates,
                'analyzed_dates': len(daily_results_list),
                'skipped_dates_count': len(skipped_dates),
            }
        }

        for row_dict in daily_results_list:
            date_str = row_dict['date']
            coefficients = {}

            # 定数項
            if 'coef_const' in row_dict:
                coefficients['const'] = {
                    'coef': row_dict.get('coef_const'),
                    'p_value': row_dict.get('pvalue_const'),
                }

            for m in METRIC_COLUMNS:
                coefficients[m] = {
                    'coef': row_dict.get(f'coef_{m}'),
                    'std_err': row_dict.get(f'stderr_{m}'),
                    't_value': row_dict.get(f'tvalue_{m}'),
                    'p_value': row_dict.get(f'pvalue_{m}'),
                }

            detail['daily_results'][date_str] = {
                'n_samples': row_dict.get('n_samples'),
                'r_squared': row_dict.get('r_squared'),
                'adj_r_squared': row_dict.get('adj_r_squared'),
                'f_statistic': row_dict.get('f_statistic'),
                'f_pvalue': row_dict.get('f_pvalue'),
                'coefficients': coefficients,
            }

        with open(output_dir / 'daily_regression_detail.json', 'w',
                  encoding='utf-8') as f:
            json.dump(detail, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"バージョン {version} の結果を保存しました: {output_dir}")

    def _generate_visualizations(
        self, version: str, daily_df: pd.DataFrame, output_dir: Path
    ) -> None:
        """可視化を生成する"""
        plot_all_coefficients(
            daily_results=daily_df,
            version=version,
            output_dir=output_dir,
        )

        # R² / Adjusted R² の時系列プロット
        save_format = VISUALIZATION_CONFIG.get('save_format', 'png')
        r_squared_path = output_dir / f'r_squared.{save_format}'
        plot_r_squared_timeseries(
            daily_results=daily_df,
            version=version,
            output_path=r_squared_path,
        )

        logger.info(f"バージョン {version} の可視化を保存しました: {output_dir}")

    def _build_summary(self, all_version_results: Dict) -> Dict:
        """全体サマリーを構築する"""
        return {
            'project': self.project_name,
            'versions_analyzed': len(all_version_results),
            'version_results': all_version_results,
            'config': {
                'max_censoring_seconds': MAX_CENSORING_SECONDS,
                'min_samples': MIN_SAMPLES,
                'exclusion_window_seconds': EXCLUSION_WINDOW_SECONDS,
                'metric_columns': METRIC_COLUMNS,
                'target_mode': self.target_mode,
                'target_column': self.target_col,
            }
        }


def main():
    """エントリーポイント"""
    parser = argparse.ArgumentParser(
        description='日次回帰分析（目的変数: 順位 または レビュー待ち時間）'
    )
    parser.add_argument(
        '--target-mode',
        choices=list(TARGET_COLUMN_BY_MODE.keys()),
        default=DEFAULT_TARGET_MODE,
        help='目的変数モード: rank=順位, time=レビュー待ち時間(秒)',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    analyzer = DailyRegressionAnalyzer(target_mode=args.target_mode)
    summary = analyzer.run_analysis()

    print(f"\n分析完了: {summary.get('versions_analyzed', 0)} バージョン")
    for version, result in summary.get('version_results', {}).items():
        if 'error' in result:
            print(f"  {version}: エラー - {result['error']}")
        else:
            print(
                f"  {version}: {result.get('analyzed_dates', 0)}/{result.get('total_dates', 0)} 日分析"
            )


if __name__ == '__main__':
    main()
