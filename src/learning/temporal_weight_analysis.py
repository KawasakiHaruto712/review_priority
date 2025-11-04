"""
時系列重み分析モジュール
"""
"""
このファイルは、逆強化学習（IRL）モデルの重み（weights）の時系列変動を分析するためのものです。
スライディングウィンドウ方式を用いて、プロジェクト毎に時系列で重みの変動を追跡し、
CSV、JSON、PDF形式で結果を出力します。
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import warnings
import configparser
import sys
import os
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

# irl_models.pyから関数をインポート
try:
    from src.learning.irl_models import (
        extract_learning_events,
        get_open_changes_at_time,
        calculate_review_priorities,
        MaxEntIRLModel,
        ReviewPriorityDataProcessor
    )
except ImportError as e:
    # ロガーを設定してからエラーメッセージ出力
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"必要なモジュールのインポートに失敗: {e}")
    raise

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 可視化ライブラリの条件付きインポート
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    
    # 日本語フォント設定
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("matplotlib/seabornが利用できません。PDF出力は無効になります。")

# 設定とデータパスのインポート
try:
    from src.config.path import DEFAULT_DATA_DIR, DEFAULT_CONFIG
    from src.utils.constants import OPENSTACK_CORE_COMPONENTS, START_DATE, END_DATE, SLIDING_WINDOW_DAYS, SLIDING_WINDOW_STEP_DAYS
except ImportError as e:
    logger.warning(f"設定ファイルのインポートに失敗: {e}")
    # デフォルト値を設定
    from pathlib import Path
    DEFAULT_DATA_DIR = Path("data")
    DEFAULT_CONFIG = Path("src/config")
    START_DATE = "2022-01-01"
    END_DATE = "2022-12-31"
    OPENSTACK_CORE_COMPONENTS = ['nova', 'neutron', 'keystone', 'glance', 'cinder', 'swift']


class TemporalWeightAnalyzer:
    """
    時系列重み分析クラス
    スライディングウィンドウでIRLモデルの重み変動を分析
    """

    def __init__(self, window_size: int = SLIDING_WINDOW_DAYS, sliding_step: int = SLIDING_WINDOW_STEP_DAYS):
        """
        Args:
            window_size (int): ウィンドウサイズ（日数）デフォルト14日（2週間）
            sliding_step (int): ウィンドウをずらす間隔（日数）デフォルト1日
        """
        self.window_size = window_size
        self.sliding_step = sliding_step
        self.data_processor = ReviewPriorityDataProcessor()
        
        # 特徴量カラム（16種類のメトリクス）
        self.feature_columns = [
            'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
            'elapsed_time', 'revision_count', 'test_code_presence',
            'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
            'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
            'refactoring_confidence', 'uncompleted_requests'
        ]
        
        # 結果格納用
        self.temporal_results = {}  # project -> results
        self.weight_history = {}    # project -> weight history
        
    def generate_time_windows(self, start_date: str, end_date: str) -> List[Tuple[datetime, datetime]]:
        """
        分析期間をウィンドウに分割
        
        Args:
            start_date (str): 開始日 'YYYY-MM-DD'
            end_date (str): 終了日 'YYYY-MM-DD'
            
        Returns:
            List[Tuple[datetime, datetime]]: ウィンドウ期間のリスト
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        windows = []
        current_start = start_dt
        
        while current_start + timedelta(days=self.window_size) <= end_dt:
            window_end = current_start + timedelta(days=self.window_size)
            windows.append((current_start, window_end))
            current_start += timedelta(days=self.sliding_step)
        
        logger.info(f"生成されたウィンドウ数: {len(windows)} (期間: {start_date} ～ {end_date})")
        return windows
    
    def load_bot_names(self) -> List[str]:
        """
        ボット名リストを設定ファイルから読み込み
        
        Returns:
            List[str]: ボット名のリスト
        """
        bot_names = []
        try:
            config_path = DEFAULT_CONFIG / "gerrymanderconfig.ini"
            config = configparser.ConfigParser()
            config.read(config_path)
            if config.has_option('organization', 'bots'):
                bot_names = [name.strip() for name in config.get('organization', 'bots').split(',')]
        except Exception as e:
            logger.error(f"ボット名の読み込みに失敗: {e}")
            # デフォルトのボット名を設定
            bot_names = ['jenkins', 'elasticrecheck', 'zuul']
        
        logger.info(f"ボット名リスト: {bot_names}")
        return bot_names
    
    def analyze_window(self, project_changes: pd.DataFrame, window_start: datetime, 
                      window_end: datetime, bot_names: List[str], project: str, 
                      releases_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        単一ウィンドウの分析を実行
        
        Args:
            project_changes: プロジェクトのChangeデータ
            window_start: ウィンドウ開始時刻
            window_end: ウィンドウ終了時刻
            bot_names: ボット名のリスト
            project: プロジェクト名
            releases_df: リリースデータ
            
        Returns:
            Optional[Dict[str, Any]]: 分析結果（失敗時はNone）
        """
        try:
            # 学習イベントを抽出（プロジェクト全体から）
            learning_events = extract_learning_events(project_changes, bot_names)
            
            # ウィンドウ期間内のイベントのみにフィルタ
            learning_events = [
                event for event in learning_events
                if window_start <= event['timestamp'] < window_end
            ]
            
            if not learning_events:
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'training_samples': 0,
                    'learning_events': 0,
                    'training_stats': None,
                    'weights': {feature: 0.0 for feature in self.feature_columns},
                    'feature_names': self.feature_columns,
                    'irl_status': 'failure',
                    'error_message': 'No learning events found'
                }
            
            # 学習データを作成
            learning_data_list = []
            
            for event in learning_events:
                event_time = event['timestamp']
                
                # その時刻にオープンだったChangeを取得（プロジェクト全体から）
                open_changes = get_open_changes_at_time(project_changes, event_time)
                
                if len(open_changes) < 2:
                    continue
                
                # 優先度重みを計算
                priority_weights = calculate_review_priorities(open_changes, event_time, bot_names)
                
                if not priority_weights:
                    continue
                
                # 特徴量を抽出
                features_df = self.data_processor.extract_features(
                    open_changes, event_time, project, releases_df
                )
                
                if features_df.empty:
                    continue
                
                # 優先度重みを設定
                features_df['priority_weight'] = features_df['change_number'].apply(
                    lambda x: priority_weights.get(str(x), 0.01)
                )
                
                learning_data_list.append(features_df)
            
            if not learning_data_list:
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'training_samples': 0,
                    'learning_events': 0,
                    'training_stats': None,
                    'weights': {feature: 0.0 for feature in self.feature_columns},
                    'feature_names': self.feature_columns,
                    'irl_status': 'failure',
                    'error_message': 'No valid learning data created'
                }
            
            # 学習データを統合
            combined_features = pd.concat(learning_data_list, ignore_index=True)
            
            # 特徴量とラベルを準備
            X = combined_features[self.feature_columns].fillna(0).values
            y = combined_features['priority_weight'].values
            
            if len(X) < 5:  # 最小学習データ数チェック
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'training_samples': len(X),
                    'learning_events': len(learning_data_list),
                    'training_stats': None,
                    'weights': {feature: 0.0 for feature in self.feature_columns},
                    'feature_names': self.feature_columns,
                    'irl_status': 'failure',
                    'error_message': f'Insufficient training data: {len(X)} samples (minimum 5 required)'
                }
            
            # IRLモデルの学習
            model = MaxEntIRLModel(learning_rate=0.01, max_iterations=1000)
            model.feature_names = self.feature_columns
            
            training_stats = model.fit(X, y)
            
            # 結果を作成
            result = {
                'window_start': window_start,
                'window_end': window_end,
                'training_samples': len(X),
                'learning_events': len(learning_data_list),
                'training_stats': training_stats,
                'weights': dict(zip(self.feature_columns, model.weights.tolist())),
                'feature_names': self.feature_columns,
                'irl_status': 'success'  # 成功した場合
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"ウィンドウ分析エラー ({window_start} - {window_end}): {e}")
            # 失敗した場合でも基本情報は返す
            return {
                'window_start': window_start,
                'window_end': window_end,
                'training_samples': 0,
                'learning_events': 0,
                'training_stats': None,
                'weights': {feature: 0.0 for feature in self.feature_columns},
                'feature_names': self.feature_columns,
                'irl_status': 'failure',
                'error_message': str(e)
            }
    
    def run_temporal_analysis(self, projects: List[str] = None, 
                            start_date: str = START_DATE, 
                            end_date: str = END_DATE) -> Dict[str, Any]:
        """
        時系列重み分析のメイン実行関数
        
        Args:
            projects: 分析対象プロジェクト（Noneの場合は全プロジェクト）
            start_date: 分析開始日
            end_date: 分析終了日
            
        Returns:
            Dict[str, Any]: 分析結果
        """
        # プロジェクト設定
        target_projects = projects if projects else OPENSTACK_CORE_COMPONENTS
        
        # データ読み込み
        logger.info("データ読み込み開始")
        changes_df, releases_df = self.data_processor.load_openstack_data()
        if changes_df.empty:
            return {"error": "データが見つかりません"}
        
        # 日付範囲でフィルタリング
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        
        mask = (changes_df['created'] >= start_datetime) & (changes_df['created'] <= end_datetime)
        changes_df = changes_df[mask]
        logger.info(f"期間フィルタ後のデータ数: {len(changes_df)} ({start_date} ～ {end_date})")
        
        # ボット名を読み込み
        bot_names = self.load_bot_names()
        
        # 時間ウィンドウを生成
        windows = self.generate_time_windows(start_date, end_date)
        
        # プロジェクト毎に分析
        for project in target_projects:
            logger.info(f"\n=== プロジェクト '{project}' の時系列重み分析開始 ===")
            
            # プロジェクトのデータをフィルタ
            project_changes = changes_df[changes_df['component'] == project]
            if project_changes.empty:
                logger.warning(f"プロジェクト '{project}' のデータが見つかりません")
                continue
            
            logger.info(f"プロジェクト '{project}': {len(project_changes)}件のChange")
            
            # プロジェクトの結果を初期化
            self.temporal_results[project] = []
            self.weight_history[project] = {feature: [] for feature in self.feature_columns}
            
            # 各ウィンドウで分析
            successful_windows = 0
            failed_windows = 0
            for i, (window_start, window_end) in enumerate(windows):
                if i % 10 == 0:
                    logger.info(f"プロジェクト '{project}': ウィンドウ処理中 {i+1}/{len(windows)}")
                
                result = self.analyze_window(
                    project_changes, window_start, window_end, 
                    bot_names, project, releases_df
                )
                
                # 成功/失敗に関係なく結果を記録
                if result:
                    self.temporal_results[project].append(result)
                    
                    # 重み履歴を記録（成功した場合のみ有効な重み）
                    for feature in self.feature_columns:
                        self.weight_history[project][feature].append({
                            'date': window_start.strftime('%Y-%m-%d'),
                            'weight': result['weights'][feature] if result['irl_status'] == 'success' else 0.0,
                            'window_start': window_start.strftime('%Y-%m-%d'),
                            'window_end': window_end.strftime('%Y-%m-%d'),
                            'status': result['irl_status']
                        })
                    
                    if result['irl_status'] == 'success':
                        successful_windows += 1
                    else:
                        failed_windows += 1
            
            logger.info(f"プロジェクト '{project}': 成功 {successful_windows}, 失敗 {failed_windows}, 合計 {len(windows)} ウィンドウ")
        
        # 全体結果をまとめ
        analysis_summary = {
            'analysis_config': {
                'window_size': self.window_size,
                'sliding_step': self.sliding_step,
                'start_date': start_date,
                'end_date': end_date,
                'target_projects': target_projects
            },
            'projects_analyzed': len(self.temporal_results),
            'total_windows': len(windows),
            'feature_dimensions': len(self.feature_columns)
        }
        
        logger.info("時系列重み分析完了")
        return analysis_summary
    
    def save_results(self, output_dir: Path = None) -> Dict[str, List[str]]:
        """
        分析結果を各種形式で保存
        
        Args:
            output_dir: 出力ディレクトリ（Noneの場合はデフォルト）
            
        Returns:
            Dict[str, List[str]]: 保存されたファイルのパス
        """
        if output_dir is None:
            output_dir = DEFAULT_DATA_DIR / "temporal_analysis"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {
            'csv': [],
            'json': [],
            'pdf': []
        }
        
        for project in self.temporal_results.keys():
            if not self.temporal_results[project]:
                continue
                
            # CSV形式で保存（行：日程、列：特徴量）
            csv_data = []
            for result in self.temporal_results[project]:
                row = {
                    'window_start': result['window_start'].strftime('%Y-%m-%d'),
                    'window_end': result['window_end'].strftime('%Y-%m-%d'),
                    'irl_status': result['irl_status'],
                    'training_samples': result['training_samples'],
                    'learning_events': result['learning_events']
                }
                # エラーメッセージがある場合は追加
                if 'error_message' in result:
                    row['error_message'] = result['error_message']
                else:
                    row['error_message'] = ''
                    
                row.update(result['weights'])
                csv_data.append(row)
            
            if csv_data:
                csv_df = pd.DataFrame(csv_data)
                csv_path = output_dir / f"temporal_weights_{project}.csv"
                csv_df.to_csv(csv_path, index=False, encoding='utf-8')
                saved_files['csv'].append(str(csv_path))
                logger.info(f"CSV保存完了: {csv_path}")
            
            # JSON形式で保存
            json_data = {
                'project': project,
                'analysis_config': {
                    'window_size': self.window_size,
                    'sliding_step': self.sliding_step
                },
                'results': [
                    {
                        **result,
                        'window_start': result['window_start'].isoformat(),
                        'window_end': result['window_end'].isoformat()
                    }
                    for result in self.temporal_results[project]
                ],
                'weight_history': self.weight_history[project]
            }
            
            json_path = output_dir / f"temporal_weights_{project}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
            saved_files['json'].append(str(json_path))
            logger.info(f"JSON保存完了: {json_path}")
            
            # PDFグラフを生成
            pdf_path = self.create_weight_visualization(project, output_dir)
            if pdf_path:
                saved_files['pdf'].append(pdf_path)
        
        return saved_files
    
    def create_weight_visualization(self, project: str, output_dir: Path) -> Optional[str]:
        """
        重みの時系列変動をPDFグラフとして出力
        
        Args:
            project: プロジェクト名
            output_dir: 出力ディレクトリ
            
        Returns:
            Optional[str]: 保存されたPDFファイルのパス
        """
        try:
            if not VISUALIZATION_AVAILABLE:
                logger.warning(f"Project {project}: matplotlib/seabornが利用できないため、PDF作成をスキップします")
                return None
                
            if not self.temporal_results[project]:
                logger.warning(f"Project {project}: 分析結果が空のため、PDF作成をスキップします")
                return None
            
            # データを準備
            dates = [result['window_start'] for result in self.temporal_results[project]]
            weights_data = {
                feature: [result['weights'][feature] for result in self.temporal_results[project]]
                for feature in self.feature_columns
            }
            
            # 図を作成（4x4のサブプロット）
            fig, axes = plt.subplots(4, 4, figsize=(20, 16))
            fig.suptitle(f'Temporal Weight Analysis - {project}', fontsize=16, fontweight='bold')
            
            # 各特徴量の重み変動をプロット
            for i, feature in enumerate(self.feature_columns):
                row = i // 4
                col = i % 4
                ax = axes[row, col]
                
                ax.plot(dates, weights_data[feature], linewidth=0.8)
                ax.set_title(feature, fontsize=10, fontweight='bold')
                ax.set_xlabel('Date', fontsize=8)
                ax.set_ylabel('Weight', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
            
            # レイアウトを調整
            plt.tight_layout()
            
            # PDFとして保存
            pdf_path = output_dir / f"temporal_weights_{project}.pdf"
            plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"PDF保存完了: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"PDF生成エラー ({project}): {e}")
            return None


def run_temporal_weight_analysis(projects: List[str] = None) -> Dict[str, Any]:
    """
    時系列重み分析のメイン実行関数
    
    Args:
        projects: 分析対象プロジェクト
        
    Returns:
        Dict[str, Any]: 分析結果
    """
    try:
        # 分析器を初期化
        analyzer = TemporalWeightAnalyzer()
        
        # 分析を実行
        results = analyzer.run_temporal_analysis(projects)
        
        if "error" not in results:
            # 結果を保存
            saved_files = analyzer.save_results()
            results['saved_files'] = saved_files
        
        return results
        
    except Exception as e:
        logger.error(f"時系列重み分析エラー: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    """
    メイン実行部分
    時系列重み分析を実行
    """
    import os
    import sys
    
    # パスを追加
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, project_root)
    
    print(f"時系列重み分析開始")
    print(f"ウィンドウサイズ: {SLIDING_WINDOW_DAYS}日")
    print(f"スライディングステップ: {SLIDING_WINDOW_STEP_DAYS}日")
    print(f"分析期間: {START_DATE} ～ {END_DATE}")
    
    try:
        results = run_temporal_weight_analysis()
        
        if "error" not in results:
            print(f"\n=== 時系列重み分析完了 ===")
            print(f"分析プロジェクト数: {results['projects_analyzed']}")
            print(f"総ウィンドウ数: {results['total_windows']}")
            print(f"特徴量次元数: {results['feature_dimensions']}")
            
            if 'saved_files' in results:
                print(f"\n保存ファイル:")
                for file_type, paths in results['saved_files'].items():
                    print(f"  {file_type.upper()}: {len(paths)}ファイル")
                    for path in paths:
                        print(f"    - {path}")
        else:
            print(f"エラー: {results['error']}")
            
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        raise
