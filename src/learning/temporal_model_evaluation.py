"""
時系列的なレビュー優先順位付けモデルの評価モジュール
Balanced Random Forestを用いてウィンドウごとの正負例で学習・評価を行う
"""
import os
import sys
import logging
import warnings
import json
import pickle
import configparser
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from imblearn.ensemble import BalancedRandomForestClassifier

warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

# 必要なモジュールをインポート
try:
    from src.learning.irl_models import (
        extract_learning_events,
        get_open_changes_at_time,
        ReviewPriorityDataProcessor
    )
except ImportError as e:
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
    from src.utils.constants import (
        OPENSTACK_CORE_COMPONENTS, 
        START_DATE, 
        END_DATE, 
        SLIDING_WINDOW_DAYS, 
        SLIDING_WINDOW_STEP_DAYS
    )
except ImportError as e:
    logger.warning(f"設定ファイルのインポートに失敗: {e}")
    # デフォルト値を設定
    from pathlib import Path
    DEFAULT_DATA_DIR = Path("data")
    DEFAULT_CONFIG = Path("src/config")
    START_DATE = "2022-01-01"
    END_DATE = "2022-12-31"
    SLIDING_WINDOW_DAYS = 14
    SLIDING_WINDOW_STEP_DAYS = 1
    OPENSTACK_CORE_COMPONENTS = ['nova', 'neutron', 'keystone', 'glance', 'cinder', 'swift']


class TemporalModelEvaluator:
    """
    時系列的なレビュー優先順位付けモデルの評価クラス
    Balanced Random Forestを用いてウィンドウごとに学習・評価を行う
    """

    def __init__(self, window_size: int = SLIDING_WINDOW_DAYS, 
                 sliding_step: int = SLIDING_WINDOW_STEP_DAYS,
                 random_state: int = 42):
        """
        Args:
            window_size (int): ウィンドウサイズ（日数）
            sliding_step (int): ウィンドウをずらす間隔（日数）
            random_state (int): 乱数シード（再現性のため）
        """
        self.window_size = window_size
        self.sliding_step = sliding_step
        self.random_state = random_state
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
        self.evaluation_results = {}  # project -> results
        
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
    
    def extract_window_labels(self, project_changes: pd.DataFrame, 
                             window_start: datetime, window_end: datetime,
                             bot_names: List[str]) -> Dict[str, int]:
        """
        ウィンドウ期間中の正負例ラベルを抽出
        
        Args:
            project_changes: プロジェクトのChangeデータ
            window_start: ウィンドウ開始時刻
            window_end: ウィンドウ終了時刻
            bot_names: ボット名のリスト
            
        Returns:
            Dict[str, int]: change_numberとラベル(0/1)の辞書
        """
        labels = {}
        
        for _, change in project_changes.iterrows():
            change_number = str(change.get('change_number', change.get('id', '')))
            
            # ウィンドウ期間中にオープンかチェック
            try:
                created_time = pd.to_datetime(change.get('created'))
                
                # ウィンドウ開始前に作成されていて、かつウィンドウ期間中にクローズしていない
                # または、ウィンドウ期間中に作成された
                if created_time > window_end:
                    continue
                
                # マージまたはクローズ時刻をチェック
                is_open_in_window = False
                merged_time = change.get('merged')
                if pd.notna(merged_time):
                    merged_time = pd.to_datetime(merged_time)
                    if merged_time >= window_start:
                        is_open_in_window = True
                else:
                    # マージされていない場合は期間中オープン
                    is_open_in_window = True
                
                if not is_open_in_window:
                    continue
                    
            except (ValueError, AttributeError):
                continue
            
            # ウィンドウ期間中にレビューされたかチェック
            reviewed = False
            if hasattr(change, 'messages') and change.get('messages'):
                messages = change.get('messages', [])
                for message in messages:
                    try:
                        msg_date = pd.to_datetime(message.get('date'))
                        author_name = message.get('author', {}).get('name', '')
                        
                        # ボットでない著者によるレビュー
                        is_bot = any(bot in author_name.lower() for bot in bot_names)
                        
                        if not is_bot and window_start <= msg_date < window_end:
                            reviewed = True
                            break
                    except (ValueError, AttributeError):
                        continue
            
            # 正例(1): ウィンドウ期間中にレビューされた
            # 負例(0): ウィンドウ期間中にレビューされなかった
            labels[change_number] = 1 if reviewed else 0
        
        return labels
    
    def get_changes_open_in_window(self, project_changes: pd.DataFrame,
                                   window_start: datetime, window_end: datetime) -> pd.DataFrame:
        """
        ウィンドウ期間中に一度でもオープンだった期間があるChangeを取得
        
        Args:
            project_changes: プロジェクトのChangeデータ
            window_start: ウィンドウ開始時刻
            window_end: ウィンドウ終了時刻
            
        Returns:
            pd.DataFrame: ウィンドウ期間中に一度でもオープンだったChange
        """
        changes_in_window = []
        
        for _, change in project_changes.iterrows():
            try:
                created_time = pd.to_datetime(change.get('created'))
                
                # ウィンドウ終了後に作成されたChangeは除外
                if created_time >= window_end:
                    continue
                
                # マージまたはクローズ時刻を取得
                merged_time = change.get('merged')
                if pd.notna(merged_time):
                    merged_time = pd.to_datetime(merged_time)
                    # ウィンドウ開始前にマージされた場合は除外
                    if merged_time <= window_start:
                        continue
                
                # ここまで来たら、ウィンドウ期間中に一度でもオープンだった
                changes_in_window.append(change)
                
            except (ValueError, AttributeError):
                continue
        
        if changes_in_window:
            return pd.DataFrame(changes_in_window)
        else:
            return pd.DataFrame()
    
    def evaluate_window(self, project_changes: pd.DataFrame, 
                       window_start: datetime, window_end: datetime,
                       bot_names: List[str], project: str,
                       releases_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        単一ウィンドウでの学習・評価を実行
        
        学習データ: window_start の 14日前 ～ window_start
        評価データ: window_start ～ window_end
        
        Args:
            project_changes: プロジェクトのChangeデータ
            window_start: ウィンドウ開始時刻
            window_end: ウィンドウ終了時刻
            bot_names: ボット名のリスト
            project: プロジェクト名
            releases_df: リリースデータ
            
        Returns:
            Optional[Dict[str, Any]]: 評価結果（失敗時はNone）
        """
        try:
            # === 学習データの準備 ===
            # window_start の 14日前から window_start までの期間
            train_window_start = window_start - timedelta(days=self.window_size)
            train_window_end = window_start
            
            # 学習用：期間中に一度でもオープンだったChangeを取得
            train_changes = self.get_changes_open_in_window(
                project_changes, train_window_start, train_window_end
            )
            
            if train_changes.empty:
                logger.warning(f"ウィンドウ {window_start.date()}: 学習データなし")
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'train_window_start': train_window_start,
                    'train_window_end': train_window_end,
                    'total_samples': 0,
                    'train_samples': 0,
                    'test_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'evaluation_status': 'failure',
                    'error_message': 'No training data in the training window'
                }
            
            # 学習用ラベルを抽出
            train_labels = self.extract_window_labels(
                train_changes, train_window_start, train_window_end, bot_names
            )
            
            if not train_labels:
                logger.warning(f"ウィンドウ {window_start.date()}: 学習ラベルなし")
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'train_window_start': train_window_start,
                    'train_window_end': train_window_end,
                    'total_samples': 0,
                    'train_samples': 0,
                    'test_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'evaluation_status': 'failure',
                    'error_message': 'No training labels extracted'
                }
            
            # 学習用特徴量を抽出（train_window_start時点の情報を使用）
            train_features_df = self.data_processor.extract_features(
                train_changes, train_window_start, project, releases_df
            )
            
            if train_features_df.empty:
                logger.warning(f"ウィンドウ {window_start.date()}: 学習特徴量抽出失敗")
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'train_window_start': train_window_start,
                    'train_window_end': train_window_end,
                    'total_samples': 0,
                    'train_samples': 0,
                    'test_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'evaluation_status': 'failure',
                    'error_message': 'Failed to extract training features'
                }
            
            # 学習データにラベルをマージ
            train_features_df['label'] = train_features_df['change_number'].astype(str).map(train_labels)
            train_features_df = train_features_df.dropna(subset=['label'])
            
            if len(train_features_df) < 10:  # 最低限のデータ数
                logger.warning(f"ウィンドウ {window_start.date()}: 学習データ不足 ({len(train_features_df)}件)")
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'train_window_start': train_window_start,
                    'train_window_end': train_window_end,
                    'total_samples': 0,
                    'train_samples': len(train_features_df),
                    'test_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'evaluation_status': 'failure',
                    'error_message': f'Insufficient training data: {len(train_features_df)} samples (minimum 10 required)'
                }
            
            # 学習データにクラスが2つ以上あるか確認
            if train_features_df['label'].nunique() < 2:
                logger.warning(f"ウィンドウ {window_start.date()}: 学習データのクラスが不足")
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'train_window_start': train_window_start,
                    'train_window_end': train_window_end,
                    'total_samples': 0,
                    'train_samples': len(train_features_df),
                    'test_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'evaluation_status': 'failure',
                    'error_message': 'Insufficient class diversity in training data (need both positive and negative samples)'
                }
            
            # === 評価データの準備 ===
            # window_start から window_end までの期間
            # 評価用：期間中に一度でもオープンだったChangeを取得
            test_changes = self.get_changes_open_in_window(
                project_changes, window_start, window_end
            )
            
            if test_changes.empty:
                logger.warning(f"ウィンドウ {window_start.date()}: 評価データなし")
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'train_window_start': train_window_start,
                    'train_window_end': train_window_end,
                    'total_samples': 0,
                    'train_samples': len(train_features_df),
                    'test_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'evaluation_status': 'failure',
                    'error_message': 'No evaluation data in the evaluation window'
                }
            
            # 評価用ラベルを抽出
            test_labels = self.extract_window_labels(
                test_changes, window_start, window_end, bot_names
            )
            
            if not test_labels:
                logger.warning(f"ウィンドウ {window_start.date()}: 評価ラベルなし")
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'train_window_start': train_window_start,
                    'train_window_end': train_window_end,
                    'total_samples': 0,
                    'train_samples': len(train_features_df),
                    'test_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'evaluation_status': 'failure',
                    'error_message': 'No evaluation labels extracted'
                }
            
            # 評価用特徴量を抽出（window_start時点の情報を使用）
            test_features_df = self.data_processor.extract_features(
                test_changes, window_start, project, releases_df
            )
            
            if test_features_df.empty:
                logger.warning(f"ウィンドウ {window_start.date()}: 評価特徴量抽出失敗")
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'train_window_start': train_window_start,
                    'train_window_end': train_window_end,
                    'total_samples': 0,
                    'train_samples': len(train_features_df),
                    'test_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'evaluation_status': 'failure',
                    'error_message': 'Failed to extract evaluation features'
                }
            
            # 評価データにラベルをマージ
            test_features_df['label'] = test_features_df['change_number'].astype(str).map(test_labels)
            test_features_df = test_features_df.dropna(subset=['label'])
            
            if len(test_features_df) < 5:  # 評価データは最低5件
                logger.warning(f"ウィンドウ {window_start.date()}: 評価データ不足 ({len(test_features_df)}件)")
                return {
                    'window_start': window_start,
                    'window_end': window_end,
                    'train_window_start': train_window_start,
                    'train_window_end': train_window_end,
                    'total_samples': len(test_features_df),
                    'train_samples': len(train_features_df),
                    'test_samples': len(test_features_df),
                    'positive_samples': 0,
                    'negative_samples': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'evaluation_status': 'failure',
                    'error_message': f'Insufficient evaluation data: {len(test_features_df)} samples (minimum 5 required)'
                }
            
            # === モデル学習と評価 ===
            # 学習データの準備
            X_train = train_features_df[self.feature_columns].fillna(0).values
            y_train = train_features_df['label'].values
            
            # 評価データの準備
            X_test = test_features_df[self.feature_columns].fillna(0).values
            y_test = test_features_df['label'].values
            
            # Balanced Random Forestで学習
            model = BalancedRandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # 予測
            y_pred = model.predict(X_test)
            
            # 評価指標を計算
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # 正負例の統計（評価データ）
            positive_count = int((test_features_df['label'] == 1).sum())
            negative_count = int((test_features_df['label'] == 0).sum())
            
            # 結果を作成
            result = {
                'window_start': window_start,
                'window_end': window_end,
                'train_window_start': train_window_start,
                'train_window_end': train_window_end,
                'total_samples': len(test_features_df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'positive_samples': positive_count,
                'negative_samples': negative_count,
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'evaluation_status': 'success'
            }
            
            logger.info(f"ウィンドウ {window_start.date()}-{window_end.date()}: "
                       f"Train={len(X_train)}, Test={len(X_test)}, "
                       f"F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}")
            
            return result
            
        except Exception as e:
            logger.warning(f"ウィンドウ評価エラー ({window_start.date()} - {window_end.date()}): {e}")
            return {
                'window_start': window_start,
                'window_end': window_end,
                'train_window_start': window_start - timedelta(days=self.window_size),
                'train_window_end': window_start,
                'total_samples': 0,
                'train_samples': 0,
                'test_samples': 0,
                'positive_samples': 0,
                'negative_samples': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'evaluation_status': 'failure',
                'error_message': str(e)
            }
    
    def run_temporal_evaluation(self, projects: List[str] = None,
                               start_date: str = START_DATE,
                               end_date: str = END_DATE) -> Dict[str, Any]:
        """
        時系列モデル評価のメイン実行関数
        
        Args:
            projects: 分析対象プロジェクト（Noneの場合は全プロジェクト）
            start_date: 分析開始日
            end_date: 分析終了日
            
        Returns:
            Dict[str, Any]: 評価結果
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
        
        # プロジェクト毎に評価
        for project in target_projects:
            logger.info(f"\n=== プロジェクト '{project}' の時系列評価開始 ===")
            
            # プロジェクトのデータをフィルタ
            project_changes = changes_df[changes_df['component'] == project]
            if project_changes.empty:
                logger.warning(f"プロジェクト '{project}': データなし")
                continue
            
            logger.info(f"プロジェクト '{project}': {len(project_changes)}件のChange")
            
            # プロジェクトの結果を初期化
            self.evaluation_results[project] = []
            
            # 各ウィンドウで評価
            successful_windows = 0
            failed_windows = 0
            
            for i, (window_start, window_end) in enumerate(windows):
                logger.info(f"ウィンドウ {i+1}/{len(windows)}: {window_start.date()} - {window_end.date()}")
                
                result = self.evaluate_window(
                    project_changes, window_start, window_end,
                    bot_names, project, releases_df
                )
                
                if result:
                    self.evaluation_results[project].append(result)
                    if result.get('evaluation_status') == 'success':
                        successful_windows += 1
                    else:
                        failed_windows += 1
                else:
                    failed_windows += 1
            
            logger.info(f"プロジェクト '{project}': 成功 {successful_windows}, 失敗 {failed_windows}, 合計 {len(windows)} ウィンドウ")
        
        # 全体結果をまとめ
        evaluation_summary = {
            'evaluation_config': {
                'window_size': self.window_size,
                'sliding_step': self.sliding_step,
                'start_date': start_date,
                'end_date': end_date,
                'random_state': self.random_state,
                'target_projects': target_projects
            },
            'projects_evaluated': len(self.evaluation_results),
            'total_windows': len(windows),
            'feature_dimensions': len(self.feature_columns)
        }
        
        logger.info("時系列モデル評価完了")
        return evaluation_summary
    
    def save_results(self, output_dir: Path = None) -> Dict[str, List[str]]:
        """
        評価結果を各種形式で保存
        
        Args:
            output_dir: 出力ディレクトリ（Noneの場合はデフォルト）
            
        Returns:
            Dict[str, List[str]]: 保存されたファイルのパス
        """
        if output_dir is None:
            output_dir = DEFAULT_DATA_DIR / "temporal_evaluation"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {
            'csv': [],
            'json': [],
            'pdf': []
        }
        
        for project in self.evaluation_results.keys():
            if not self.evaluation_results[project]:
                continue
                
            # CSV形式で保存
            csv_data = []
            for result in self.evaluation_results[project]:
                row = {
                    'window_start': result['window_start'].strftime('%Y-%m-%d'),
                    'window_end': result['window_end'].strftime('%Y-%m-%d'),
                    'train_window_start': result.get('train_window_start', result['window_start'] - timedelta(days=self.window_size)).strftime('%Y-%m-%d'),
                    'train_window_end': result.get('train_window_end', result['window_start']).strftime('%Y-%m-%d'),
                    'total_samples': result['total_samples'],
                    'train_samples': result['train_samples'],
                    'test_samples': result['test_samples'],
                    'positive_samples': result['positive_samples'],
                    'negative_samples': result['negative_samples'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                    'status': result['evaluation_status'],
                    'error_message': result.get('error_message', '')
                }
                csv_data.append(row)
            
            if csv_data:
                csv_df = pd.DataFrame(csv_data)
                csv_path = output_dir / f"temporal_evaluation_{project}.csv"
                csv_df.to_csv(csv_path, index=False, encoding='utf-8')
                saved_files['csv'].append(str(csv_path))
                logger.info(f"CSV保存完了: {csv_path}")
            
            # JSON形式で保存（サマリー統計を含む）
            successful_results = [r for r in self.evaluation_results[project] if r.get('evaluation_status') == 'success']
            
            if successful_results:
                f1_scores = [r['f1_score'] for r in successful_results]
                precisions = [r['precision'] for r in successful_results]
                recalls = [r['recall'] for r in successful_results]
                
                summary_stats = {
                    'mean_f1': float(np.mean(f1_scores)),
                    'std_f1': float(np.std(f1_scores)),
                    'mean_precision': float(np.mean(precisions)),
                    'std_precision': float(np.std(precisions)),
                    'mean_recall': float(np.mean(recalls)),
                    'std_recall': float(np.std(recalls)),
                    'successful_windows': len(successful_results),
                    'total_windows': len(self.evaluation_results[project])
                }
            else:
                summary_stats = {
                    'mean_f1': 0.0,
                    'std_f1': 0.0,
                    'mean_precision': 0.0,
                    'std_precision': 0.0,
                    'mean_recall': 0.0,
                    'std_recall': 0.0,
                    'successful_windows': 0,
                    'total_windows': len(self.evaluation_results[project])
                }
            
            json_data = {
                'project': project,
                'evaluation_config': {
                    'window_size': self.window_size,
                    'sliding_step': self.sliding_step,
                    'random_state': self.random_state
                },
                'summary_statistics': summary_stats,
                'results': [
                    {
                        **result,
                        'window_start': result['window_start'].isoformat(),
                        'window_end': result['window_end'].isoformat()
                    }
                    for result in self.evaluation_results[project]
                ]
            }
            
            json_path = output_dir / f"temporal_evaluation_{project}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            saved_files['json'].append(str(json_path))
            logger.info(f"JSON保存完了: {json_path}")
            
            # PDFグラフを生成
            pdf_path = self.create_evaluation_visualization(project, output_dir)
            if pdf_path:
                saved_files['pdf'].append(pdf_path)
        
        return saved_files
    
    def create_evaluation_visualization(self, project: str, output_dir: Path) -> Optional[str]:
        """
        評価結果の可視化レポートをPDF形式で出力
        temporal_weight_analysis.pyと同様の折れ線グラフスタイル
        
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
                
            results = self.evaluation_results[project]
            successful_results = [r for r in results if r.get('evaluation_status') == 'success']
            
            if not successful_results:
                logger.warning(f"Project {project}: 可視化する成功結果なし")
                return None
            
            # データを準備
            dates = [result['window_start'] for result in successful_results]
            
            # 評価指標データ
            metrics_data = {
                'F1 Score': [result['f1_score'] for result in successful_results],
                'Precision': [result['precision'] for result in successful_results],
                'Recall': [result['recall'] for result in successful_results]
            }
            
            # サンプル数データ
            samples_data = {
                'Total Samples': [result['total_samples'] for result in successful_results],
                'Positive Samples': [result['positive_samples'] for result in successful_results],
                'Negative Samples': [result['negative_samples'] for result in successful_results],
                'Train Samples': [result['train_samples'] for result in successful_results],
                'Test Samples': [result['test_samples'] for result in successful_results]
            }
            
            # 比率データ
            positive_ratios = [result['positive_samples'] / result['total_samples'] 
                             if result['total_samples'] > 0 else 0 
                             for result in successful_results]
            
            # 図を作成（4x2のサブプロット - temporal_weight_analysis.pyと同様のスタイル）
            fig, axes = plt.subplots(4, 2, figsize=(20, 16))
            fig.suptitle(f'Temporal Model Evaluation - {project}', fontsize=16, fontweight='bold')
            
            # 1. F1スコアの時系列変化
            ax1 = axes[0, 0]
            ax1.plot(dates, metrics_data['F1 Score'], linewidth=0.8)
            ax1.set_title('F1 Score Over Time', fontsize=10, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=8)
            ax1.set_ylabel('F1 Score', fontsize=8)
            ax1.set_ylim([0, 1])
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45, labelsize=8)
            ax1.tick_params(axis='y', labelsize=8)
            
            # 2. Precisionの時系列変化
            ax2 = axes[0, 1]
            ax2.plot(dates, metrics_data['Precision'], linewidth=0.8, color='green')
            ax2.set_title('Precision Over Time', fontsize=10, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=8)
            ax2.set_ylabel('Precision', fontsize=8)
            ax2.set_ylim([0, 1])
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45, labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)
            
            # 3. Recallの時系列変化
            ax3 = axes[1, 0]
            ax3.plot(dates, metrics_data['Recall'], linewidth=0.8, color='orange')
            ax3.set_title('Recall Over Time', fontsize=10, fontweight='bold')
            ax3.set_xlabel('Date', fontsize=8)
            ax3.set_ylabel('Recall', fontsize=8)
            ax3.set_ylim([0, 1])
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45, labelsize=8)
            ax3.tick_params(axis='y', labelsize=8)
            
            # 4. 全評価指標の比較
            ax4 = axes[1, 1]
            ax4.plot(dates, metrics_data['F1 Score'], label='F1 Score', linewidth=0.8)
            ax4.plot(dates, metrics_data['Precision'], label='Precision', linewidth=0.8)
            ax4.plot(dates, metrics_data['Recall'], label='Recall', linewidth=0.8)
            ax4.set_title('All Metrics Comparison', fontsize=10, fontweight='bold')
            ax4.set_xlabel('Date', fontsize=8)
            ax4.set_ylabel('Score', fontsize=8)
            ax4.set_ylim([0, 1])
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45, labelsize=8)
            ax4.tick_params(axis='y', labelsize=8)
            
            # 5. 総サンプル数の時系列変化
            ax5 = axes[2, 0]
            ax5.plot(dates, samples_data['Total Samples'], linewidth=0.8, color='blue')
            ax5.set_title('Total Samples Over Time', fontsize=10, fontweight='bold')
            ax5.set_xlabel('Date', fontsize=8)
            ax5.set_ylabel('Total Samples', fontsize=8)
            ax5.set_ylim(bottom=0)
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis='x', rotation=45, labelsize=8)
            ax5.tick_params(axis='y', labelsize=8)
            
            # 6. 正負例サンプル数の時系列変化
            ax6 = axes[2, 1]
            ax6.plot(dates, samples_data['Positive Samples'], label='Positive', linewidth=0.8, color='green')
            ax6.plot(dates, samples_data['Negative Samples'], label='Negative', linewidth=0.8, color='red')
            ax6.set_title('Positive/Negative Samples Over Time', fontsize=10, fontweight='bold')
            ax6.set_xlabel('Date', fontsize=8)
            ax6.set_ylabel('Sample Count', fontsize=8)
            ax6.set_ylim(bottom=0)
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
            ax6.tick_params(axis='x', rotation=45, labelsize=8)
            ax6.tick_params(axis='y', labelsize=8)
            
            # 7. 学習/評価データ数の時系列変化
            ax7 = axes[3, 0]
            ax7.plot(dates, samples_data['Train Samples'], label='Train', linewidth=0.8, color='blue')
            ax7.plot(dates, samples_data['Test Samples'], label='Test', linewidth=0.8, color='purple')
            ax7.set_title('Train/Test Samples Over Time', fontsize=10, fontweight='bold')
            ax7.set_xlabel('Date', fontsize=8)
            ax7.set_ylabel('Sample Count', fontsize=8)
            ax7.set_ylim(bottom=0)
            ax7.legend(fontsize=8)
            ax7.grid(True, alpha=0.3)
            ax7.tick_params(axis='x', rotation=45, labelsize=8)
            ax7.tick_params(axis='y', labelsize=8)
            
            # 8. 正例比率の時系列変化
            ax8 = axes[3, 1]
            ax8.plot(dates, positive_ratios, linewidth=0.8, color='green')
            ax8.set_title('Positive Sample Ratio Over Time', fontsize=10, fontweight='bold')
            ax8.set_xlabel('Date', fontsize=8)
            ax8.set_ylabel('Positive Ratio', fontsize=8)
            ax8.set_ylim([0, 1])
            ax8.grid(True, alpha=0.3)
            ax8.tick_params(axis='x', rotation=45, labelsize=8)
            ax8.tick_params(axis='y', labelsize=8)
            
            # レイアウトを調整
            plt.tight_layout()
            
            # PDFとして保存
            pdf_path = output_dir / f"temporal_evaluation_{project}.pdf"
            plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"PDF保存完了: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"PDF生成エラー ({project}): {e}")
            return None


def run_temporal_model_evaluation(projects: List[str] = None) -> Dict[str, Any]:
    """
    時系列モデル評価のメイン実行関数
    
    Args:
        projects: 分析対象プロジェクト
        
    Returns:
        Dict[str, Any]: 評価結果
    """
    try:
        # 評価器を初期化
        evaluator = TemporalModelEvaluator()
        
        # 評価を実行
        results = evaluator.run_temporal_evaluation(projects)
        
        if "error" not in results:
            # 結果を保存
            saved_files = evaluator.save_results()
            logger.info(f"結果保存完了: {saved_files}")
        
        return results
        
    except Exception as e:
        logger.error(f"時系列モデル評価エラー: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    """
    メイン実行部分
    時系列モデル評価を実行
    """
    import argparse
    
    # パスを追加
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, project_root)
    
    print(f"時系列モデル評価開始")
    print(f"ウィンドウサイズ: {SLIDING_WINDOW_DAYS}日")
    print(f"スライディングステップ: {SLIDING_WINDOW_STEP_DAYS}日")
    print(f"分析期間: {START_DATE} ～ {END_DATE}")
    
    try:
        results = run_temporal_model_evaluation()
        
        if "error" not in results:
            print("\n=== 評価完了 ===")
            print(f"評価プロジェクト数: {results['projects_evaluated']}")
            print(f"総ウィンドウ数: {results['total_windows']}")
            print(f"特徴量次元: {results['feature_dimensions']}")
        else:
            print(f"エラー: {results['error']}")
            
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        raise