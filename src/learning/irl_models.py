"""
逆強化学習（Inverse Reinforcement Learning, IRL）を用いたコードレビュー優先順位付けモデル
最大エントロピーIRLアルゴリズムを実装し、レビューの優先順位を学習する
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import scipy.optimize as optimize
from sklearn.preprocessing import StandardScaler
import pickle

# 既存のモジュールをインポート
from src.config.path import DEFAULT_DATA_DIR, DEFAULT_CONFIG
from src.features.bug_metrics import calculate_bug_fix_confidence
from src.features.change_metrics import (
    calculate_lines_added, calculate_lines_deleted, calculate_files_changed,
    calculate_elapsed_time, calculate_revision_count, check_test_code_presence
)
from src.features.developer_metrics import (
    calculate_past_report_count, calculate_recent_report_count,
    calculate_merge_rate, calculate_recent_merge_rate
)
from src.features.project_metrics import (
    calculate_days_to_major_release, calculate_predictive_target_ticket_count,
    calculate_reviewed_lines_in_period, add_lines_info_to_dataframe
)
from src.features.refactoring_metrics import calculate_refactoring_confidence
from src.features.review_metrics import ReviewStatusAnalyzer

# ロギング設定 - エラーのみ出力
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_bot_author(author_name: str, bot_names: List[str]) -> bool:
    """
    著者がボットかどうかを判定
    
    Args:
        author_name (str): 著者名
        bot_names (List[str]): ボット名のリスト
        
    Returns:
        bool: ボットの場合True
    """
    if not author_name or not bot_names:
        return False
    return any(bot in author_name.lower() for bot in bot_names)


def extract_learning_events(changes_df: pd.DataFrame, bot_names: List[str]) -> List[Dict]:
    """
    学習イベント（bot以外のauthorによるレビューコメント投稿）を抽出
    
    Args:
        changes_df (pd.DataFrame): チェンジデータ
        bot_names (List[str]): ボット名のリスト
        
    Returns:
        List[Dict]: 学習イベントのリスト
    """
    learning_events = []
    
    for _, change in changes_df.iterrows():
        if not hasattr(change, 'messages') or not change.get('messages'):
            continue
            
        # メッセージを時系列でソート
        messages = sorted(change.get('messages', []), key=lambda x: x.get('date', ''))
        
        for message in messages:
            author = message.get('author', {}).get('name', '')
            message_date = message.get('date', '')
            
            # bot以外の著者によるメッセージかチェック
            if not is_bot_author(author, bot_names) and author and message_date:
                try:
                    event_datetime = datetime.fromisoformat(message_date.replace('Z', '+00:00'))
                    learning_events.append({
                        'timestamp': event_datetime,
                        'change_id': change.get('change_number', change.get('id', '')),
                        'author': author,
                        'message_date': message_date
                    })
                except (ValueError, AttributeError) as e:
                    logger.warning(f"日付解析エラー: {message_date}, エラー: {e}")
                    continue
    
    # 時系列でソート
    learning_events.sort(key=lambda x: x['timestamp'])
    
    return learning_events


def get_open_changes_at_time(changes_df: pd.DataFrame, target_time: datetime) -> pd.DataFrame:
    """
    指定時刻にオープンだったChangeを取得
    
    Args:
        changes_df (pd.DataFrame): チェンジデータ
        target_time (datetime): 対象時刻
        
    Returns:
        pd.DataFrame: 指定時刻にオープンだったChange
    """
    open_changes = []
    
    for _, change in changes_df.iterrows():
        try:
            # Created時刻の解析
            if pd.isna(change['created']):
                continue
            
            created_time = change['created']
            if not isinstance(created_time, datetime):
                if isinstance(created_time, str):
                    created_time = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                else:
                    continue
            
            # Updated時刻の解析（クローズ時刻の代理）
            updated_time = None
            if hasattr(change, 'updated') and pd.notna(change['updated']):
                updated_time = change['updated']
                if not isinstance(updated_time, datetime):
                    if isinstance(updated_time, str):
                        updated_time = datetime.fromisoformat(updated_time.replace('Z', '+00:00'))
            
            # 指定時刻にオープンだったかチェック
            if created_time <= target_time:
                # まだクローズされていない、または指定時刻より後にクローズされた
                if not updated_time or updated_time > target_time:
                    open_changes.append(change)
                    
        except (ValueError, AttributeError) as e:
            logger.warning(f"Change {getattr(change, 'change_number', 'unknown')} の時刻解析エラー: {e}")
            continue
    
    return pd.DataFrame(open_changes) if open_changes else pd.DataFrame()


def calculate_review_priorities(open_changes: pd.DataFrame, current_time: datetime, 
                              bot_names: List[str]) -> Dict[str, float]:
    """
    オープンなChange全体に対する優先順位重みを計算
    全てのChangeを次のレビュー時刻順で順位付けし、(総数-順位)/総数で重み付け
    
    Args:
        open_changes (pd.DataFrame): オープンなChange
        current_time (datetime): 現在時刻（学習タイミング）
        bot_names (List[str]): ボット名のリスト
        
    Returns:
        Dict[str, float]: change_numberと優先度重みの辞書
    """
    if open_changes.empty:
        return {}
    
    # 各Changeの次のレビュー時刻を取得
    review_times = {}
    
    for _, change in open_changes.iterrows():
        change_number = str(change.get('change_number', change.get('id', '')))
        
        if not hasattr(change, 'messages') or not change.get('messages'):
            # レビュー予定がない場合は無限大の時刻を設定（最低優先度）
            review_times[change_number] = datetime.max.replace(tzinfo=timezone.utc)
            continue
            
        messages = sorted(change.get('messages', []), key=lambda x: x.get('date', ''))
        
        next_review_time = None
        for message in messages:
            author = message.get('author', {}).get('name', '')
            message_date = message.get('date', '')
            
            if not is_bot_author(author, bot_names) and author and message_date:
                try:
                    message_time = datetime.fromisoformat(message_date.replace('Z', '+00:00'))
                    
                    # タイムゾーンの統一（両方ともUTCに統一）
                    if current_time.tzinfo is None:
                        current_time_utc = current_time.replace(tzinfo=timezone.utc)
                    else:
                        current_time_utc = current_time
                    
                    if message_time.tzinfo is None:
                        message_time = message_time.replace(tzinfo=timezone.utc)
                    
                    # 現在時刻以降の最初のレビューを記録
                    if message_time >= current_time_utc:
                        next_review_time = message_time
                        break
                        
                except (ValueError, AttributeError):
                    continue
        
        # レビュー時刻が見つからない場合は最低優先度
        if next_review_time is None:
            review_times[change_number] = datetime.max.replace(tzinfo=timezone.utc)
        else:
            review_times[change_number] = next_review_time
    
    # レビュー時刻順でソート（早い順 = 高優先度）
    sorted_changes = sorted(review_times.items(), key=lambda x: x[1])
    
    # 重みを計算：(総数 - 順位) / 総数
    total_count = len(sorted_changes)
    priorities = {}
    
    for rank, (change_number, _) in enumerate(sorted_changes):
        # 順位は0から始まるので、重みは (total_count - rank) / total_count
        weight = (total_count - rank) / total_count
        priorities[change_number] = weight
    
    return priorities


class MaxEntIRLModel:
    """
    最大エントロピー逆強化学習モデル
    レビューアーの行動から優先順位付けの重みを学習する
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Args:
            learning_rate (float): 学習率
            max_iterations (int): 最大反復回数
            tolerance (float): 収束判定の閾値
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.feature_scaler = StandardScaler()
        self.feature_names = []
        
    def compute_partition_function(self, features: np.ndarray, weights: np.ndarray) -> float:
        """
        分配関数Z(θ)を計算
        
        Args:
            features (np.ndarray): 特徴量行列 (n_samples, n_features)
            weights (np.ndarray): 重みベクトル (n_features,)
            
        Returns:
            float: 分配関数の値
        """
        # 重み付きスコアを計算
        scores = np.dot(features, weights)
        # 数値安定性のためmax値を引く
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        return np.sum(exp_scores)
    
    def compute_expected_features(self, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        期待特徴量ベクトルを計算
        
        Args:
            features (np.ndarray): 特徴量行列
            weights (np.ndarray): 重みベクトル
            
        Returns:
            np.ndarray: 期待特徴量ベクトル
        """
        scores = np.dot(features, weights)
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # 期待特徴量を計算
        expected_features = np.sum(features * probabilities.reshape(-1, 1), axis=0)
        return expected_features
    
    def compute_gradient(self, features: np.ndarray, weights: np.ndarray, 
                        empirical_features: np.ndarray) -> np.ndarray:
        """
        勾配を計算
        
        Args:
            features (np.ndarray): 特徴量行列
            weights (np.ndarray): 重みベクトル
            empirical_features (np.ndarray): 経験的特徴量ベクトル
            
        Returns:
            np.ndarray: 勾配ベクトル
        """
        expected_features = self.compute_expected_features(features, weights)
        gradient = empirical_features - expected_features
        return gradient
    
    def objective_function(self, weights: np.ndarray, features: np.ndarray, 
                          empirical_features: np.ndarray) -> float:
        """
        目的関数（負の対数尤度）を計算
        
        Args:
            weights (np.ndarray): 重みベクトル
            features (np.ndarray): 特徴量行列
            empirical_features (np.ndarray): 経験的特徴量ベクトル
            
        Returns:
            float: 目的関数の値
        """
        log_partition = np.log(self.compute_partition_function(features, weights))
        empirical_score = np.dot(empirical_features, weights)
        return log_partition - empirical_score
    
    def fit(self, features: np.ndarray, priorities: np.ndarray) -> Dict[str, Any]:
        """
        最大エントロピーIRLでモデルを学習
        
        Args:
            features (np.ndarray): 特徴量行列 (n_samples, n_features)
            priorities (np.ndarray): 優先順位ベクトル (n_samples,)
            
        Returns:
            Dict[str, Any]: 学習結果の統計情報
        """
        # 特徴量の正規化
        features_normalized = self.feature_scaler.fit_transform(features)
        
        # 経験的特徴量を計算（優先順位による重み付き平均）
        # 優先順位が高いほど重要度が高いとする
        priority_weights = priorities / np.sum(priorities)
        empirical_features = np.sum(features_normalized * priority_weights.reshape(-1, 1), axis=0)
        
        # 重みを初期化
        n_features = features_normalized.shape[1]
        initial_weights = np.random.normal(0, 0.1, n_features)
        
        # 最適化
        result = optimize.minimize(
            fun=self.objective_function,
            x0=initial_weights,
            args=(features_normalized, empirical_features),
            method='L-BFGS-B',
            jac=lambda w, f, ef: -self.compute_gradient(f, w, ef),
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        self.weights = result.x
        
        # 学習結果の統計情報
        training_stats = {
            'converged': result.success,
            'final_objective': result.fun,
            'iterations': result.nit
        }
        
        return training_stats
    
    def predict_priority_scores(self, features: np.ndarray) -> np.ndarray:
        """
        新しいデータに対して優先順位スコアを予測
        
        Args:
            features (np.ndarray): 特徴量行列
            
        Returns:
            np.ndarray: 優先順位スコア
        """
        if self.weights is None:
            raise ValueError("モデルが学習されていません。先にfitメソッドを呼び出してください。")
        
        # 特徴量を正規化
        features_normalized = self.feature_scaler.transform(features)
        
        # 優先順位スコアを計算
        scores = np.dot(features_normalized, self.weights)
        return scores
    
    def save_model(self, model_path: Path):
        """
        学習済みモデルを保存
        
        Args:
            model_path (Path): 保存先パス
        """
        model_data = {
            'weights': self.weights,
            'feature_scaler': self.feature_scaler,
            'feature_names': self.feature_names,
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path: Path):
        """
        学習済みモデルを読み込み
        
        Args:
            model_path (Path): モデルファイルのパス
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.feature_scaler = model_data['feature_scaler']
        self.feature_names = model_data['feature_names']
        self.learning_rate = model_data['learning_rate']
        self.max_iterations = model_data['max_iterations']
        self.tolerance = model_data['tolerance']


class ReviewPriorityDataProcessor:
    """
    コードレビューの優先順位付けデータを処理するクラス
    """
    
    def __init__(self):
        self.review_analyzer = ReviewStatusAnalyzer(
            extraction_keywords_path=DEFAULT_DATA_DIR / "processed" / "review_keywords.json",
            gerrymander_config_path=DEFAULT_CONFIG / "gerrymanderconfig.ini",
            review_label_path=DEFAULT_DATA_DIR / "processed" / "review_label.json"
        )
        
    def load_openstack_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        OpenStackのPRデータを読み込む
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (changes_df, releases_df)
        """
        changes_data = []
        openstack_dir = DEFAULT_DATA_DIR / "openstack"
        
        # 各コンポーネントのchangesデータを読み込み
        for component_dir in openstack_dir.iterdir():
            if component_dir.is_dir():
                changes_dir = component_dir / "changes"
                if changes_dir.exists():
                    for change_file in changes_dir.glob("change_*.json"):
                        try:
                            with open(change_file, 'r', encoding='utf-8') as f:
                                change_data = json.load(f)
                                change_data['component'] = component_dir.name
                                changes_data.append(change_data)
                        except Exception as e:
                            logger.warning(f"ファイル読み込みエラー {change_file}: {e}")
        
        # DataFrameに変換
        changes_df = pd.DataFrame(changes_data)
        
        # owner_emailカラムを追加（開発者メトリクス計算のため）
        changes_df['owner_email'] = changes_df['owner'].apply(
            lambda x: x.get('email', '') if isinstance(x, dict) else ''
        )
        
        # 日時カラムを変換
        for col in ['created', 'updated', 'submitted', 'merged']:
            if col in changes_df.columns:
                # nanosecond precision (.000000000) を削除してから変換
                changes_df[col] = changes_df[col].astype(str).str.replace('.000000000', '', regex=False)
                changes_df[col] = pd.to_datetime(changes_df[col], errors='coerce')
        
        # リリースデータの読み込み
        releases_csv_path = DEFAULT_DATA_DIR / "openstack" / "releases_summary.csv"
        if releases_csv_path.exists():
            try:
                releases_df = pd.read_csv(releases_csv_path)
                # release_dateをdatetime型に変換
                releases_df['release_date'] = pd.to_datetime(releases_df['release_date'])
            except Exception as e:
                logger.error(f"リリースデータ読み込みエラー: {e}")
                releases_df = pd.DataFrame()
        else:
            logger.warning(f"リリースファイルが見つかりません: {releases_csv_path}")
            releases_df = pd.DataFrame()
        
        return changes_df, releases_df
    
    def extract_features(self, changes_df: pd.DataFrame, analysis_time: datetime, project_name: str = None, releases_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        各PRから16種類のメトリクスを抽出
        
        Args:
            changes_df (pd.DataFrame): PRデータ
            analysis_time (datetime): 分析時点
            project_name (str): プロジェクト名（行数情報の取得に使用）
            releases_df (pd.DataFrame): リリースデータ（日数計算に使用）
            
        Returns:
            pd.DataFrame: 特徴量データ
        """
        # プロジェクト名が指定されている場合、行数情報を追加
        if project_name and 'change_number' in changes_df.columns:
            changes_df = add_lines_info_to_dataframe(changes_df, project_name)
        
        features_list = []
        processed_count = 0
        error_count = 0
        
        for idx, change_row in changes_df.iterrows():
            try:
                # pandas Seriesを辞書に変換
                change_data = change_row.to_dict()
                
                # created フィールドを文字列に変換（特徴量計算関数が期待する形式）
                if pd.notna(change_data['created']):
                    change_data['created'] = change_data['created'].strftime('%Y-%m-%d %H:%M:%S')
                
                # 分析時点より前に作成されたPRのみを対象
                if pd.isna(change_row['created']) or change_row['created'] > analysis_time:
                    continue
                
                processed_count += 1
                if processed_count <= 5:  # 最初の5件をログ出力（デバッグ用）
                    logger.debug(f"処理中のPR: {change_data['change_number']}")

                features = {}
                features['change_number'] = change_data['change_number']
                features['component'] = change_data['component']
                features['created'] = change_row['created']  # datetimeオブジェクトを使用
                
                # 1. バグ修正確信度
                features['bug_fix_confidence'] = calculate_bug_fix_confidence(
                    change_data.get('subject', ''),
                    change_data.get('message', '')
                )
                
                # 2-4. 変更メトリクス
                # 行数情報がDataFrameに追加されている場合はそれを使用、そうでなければJSONから計算
                if 'lines_added' in change_data and pd.notna(change_data['lines_added']):
                    features['lines_added'] = int(change_data['lines_added'])
                    features['lines_deleted'] = int(change_data.get('lines_deleted', 0))
                    features['files_changed'] = int(change_data.get('files_changed', 0))
                else:
                    # JSONファイルから計算（プロジェクト名が必要）
                    if project_name:
                        change_file_path = DEFAULT_DATA_DIR / 'openstack' / project_name / 'changes' / f'change_{int(change_data["change_number"])}.json'
                        try:
                            if change_file_path.exists():
                                with open(change_file_path, 'r', encoding='utf-8') as f:
                                    change_json_data = json.load(f)
                                features['lines_added'] = calculate_lines_added(change_json_data)
                                features['lines_deleted'] = calculate_lines_deleted(change_json_data)
                                features['files_changed'] = calculate_files_changed(change_json_data)
                            else:
                                features['lines_added'] = 0
                                features['lines_deleted'] = 0
                                features['files_changed'] = 0
                        except Exception as e:
                            logger.warning(f"JSONファイル読み込みエラー {change_file_path}: {e}")
                            features['lines_added'] = 0
                            features['lines_deleted'] = 0
                            features['files_changed'] = 0
                    else:
                        # フォールバック: change_dataから直接計算を試行
                        features['lines_added'] = calculate_lines_added(change_data)
                        features['lines_deleted'] = calculate_lines_deleted(change_data)
                        features['files_changed'] = calculate_files_changed(change_data)
                
                # 5. 経過時間
                features['elapsed_time'] = calculate_elapsed_time(change_data, analysis_time)
                
                # 6. リビジョン数
                features['revision_count'] = calculate_revision_count(change_data, analysis_time)
                
                # 7. テストコード存在確認
                features['test_code_presence'] = check_test_code_presence(change_data)
                
                # 8-11. 開発者メトリクス
                owner_data = change_data.get('owner', {})
                owner_email = owner_data.get('email', '') if isinstance(owner_data, dict) else ''
                if owner_email:
                    features['past_report_count'] = calculate_past_report_count(
                        owner_email, changes_df, analysis_time
                    )
                    features['recent_report_count'] = calculate_recent_report_count(
                        owner_email, changes_df, analysis_time
                    )
                    features['merge_rate'] = calculate_merge_rate(
                        owner_email, changes_df, analysis_time
                    )
                    features['recent_merge_rate'] = calculate_recent_merge_rate(
                        owner_email, changes_df, analysis_time
                    )
                else:
                    features['past_report_count'] = 0
                    features['recent_report_count'] = 0
                    features['merge_rate'] = 0.0
                    features['recent_merge_rate'] = 0.0
                
                # 12-14. プロジェクトメトリクス
                # リリースデータが利用可能な場合は実際の日数を計算、そうでなければ-1.0
                if releases_df is not None and not releases_df.empty:
                    features['days_to_major_release'] = calculate_days_to_major_release(
                        analysis_time, change_data.get('component', ''), releases_df
                    )
                else:
                    features['days_to_major_release'] = -1.0  # リリースデータがないため仮値
                features['open_ticket_count'] = calculate_predictive_target_ticket_count(
                    changes_df, analysis_time
                )
                features['reviewed_lines_in_period'] = calculate_reviewed_lines_in_period(
                    changes_df, analysis_time
                )
                
                # 15. リファクタリング確信度
                features['refactoring_confidence'] = calculate_refactoring_confidence(
                    change_data.get('subject', ''),
                    change_data.get('message', '')
                )
                
                # 16. 未完了修正要求数
                features['uncompleted_requests'] = self.review_analyzer.analyze_pr_status(change_data)
                
                features_list.append(features)
                
            except Exception as e:
                error_count += 1
                change_id = change_data.get('change_number', 'unknown') if 'change_data' in locals() else 'unknown'
                if error_count <= 3:  # 最初の3件のエラーのみログ出力
                    logger.warning(f"特徴量抽出エラー {error_count} (change {change_id}): {e}")
                continue
        
        features_df = pd.DataFrame(features_list)
        
        return features_df
    
    def calculate_priorities_from_review_time(self, features_df: pd.DataFrame, analysis_time: datetime) -> np.ndarray:
        """
        analysis_timeから次のレビュータイミングまでの時間をもとに優先順位を算出
        短いレビュー待ち時間ほど高い優先順位とする
        
        Args:
            features_df (pd.DataFrame): 特徴量データ
            analysis_time (datetime): 分析時点
            
        Returns:
            np.ndarray: 優先順位スコア（高いほど優先度が高い）
        """
        priorities = []
        debug_count = 0
        
        for idx, row in features_df.iterrows():
            change_number = row['change_number']
            
            # レビューコメントデータから次のレビュー時刻を取得
            next_review_time = self._find_next_review_time(change_number, analysis_time)
            
            if next_review_time is not None:
                # analysis_timeから次のレビューまでの時間（分数）
                time_to_review = (next_review_time - analysis_time).total_seconds() / 60
                
                if time_to_review > 0:
                    # 短い時間ほど高い優先度（逆数を使用）
                    priority = 1.0 / (time_to_review + 1.0)  # +1.0で0除算回避
                else:
                    priority = 0
                    
                # デバッグ情報削除
                if debug_count < 3:
                    debug_count += 1
            else:
                # レビュー時刻が不明な場合はデフォルト優先度
                priority = 0.01
                if debug_count < 3:
                    debug_count += 1
            
            priorities.append(priority)
        
        priorities = np.array(priorities)
        
        # 正規化
        if np.sum(priorities) > 0:
            priorities = priorities / np.sum(priorities)
        else:
            priorities = np.ones(len(priorities)) / len(priorities)
        
        return priorities
    
    def _find_next_review_time(self, change_number: int, analysis_time: datetime) -> datetime:
        """
        指定されたPRの次のレビュー時刻を取得
        
        Args:
            change_number (int): PRの番号
            analysis_time (datetime): 分析時点
            
        Returns:
            datetime: 次のレビュー時刻（見つからない場合はNone）
        """
        try:
            # messagesデータから次のレビューコメントを探す
            for component_name in ['neutron', 'nova', 'cinder', 'glance', 'keystone', 'swift']:
                change_file_path = DEFAULT_DATA_DIR / "openstack" / component_name / "changes" / f"change_{change_number}.json"
                
                if change_file_path.exists():
                    with open(change_file_path, 'r', encoding='utf-8') as f:
                        change_data = json.load(f)
                    
                    messages = change_data.get('messages', [])
                    
                    # analysis_time以降のメッセージを時系列順にソート
                    future_messages = []
                    for msg in messages:
                        try:
                            msg_date_str = msg.get('date', '').replace('.000000000', '')
                            msg_date = datetime.fromisoformat(msg_date_str.replace(' ', 'T'))
                            
                            if msg_date > analysis_time:
                                future_messages.append((msg_date, msg))
                        except (ValueError, TypeError):
                            continue
                    
                    # 時系列順にソート
                    future_messages.sort(key=lambda x: x[0])
                    
                    # botではない投稿者からの最初のメッセージを探す
                    for msg_date, msg in future_messages:
                        author_name = msg.get('author', {}).get('name', '').lower()
                        author_username = msg.get('author', {}).get('username', '').lower()
                        author_email = msg.get('author', {}).get('email', '').lower()
                        
                        # botかどうかをチェック（名前、ユーザー名、メールアドレスで判定）
                        is_bot = any(
                            bot_name.lower() in author_name or 
                            bot_name.lower() in author_username or 
                            bot_name.lower() in author_email
                            for bot_name in self.review_analyzer.bot_names
                        )
                        
                        # botではない場合、そのメッセージをレビューコメントとして扱う
                        if not is_bot:
                            return msg_date
                    
                    # レビューメッセージが見つからない場合、最初の未来のメッセージを返す
                    if future_messages:
                        return future_messages[0][0]
                    
                    break  # ファイルが見つかったらループを抜ける
            
            return None
            
        except Exception as e:
            # デバッグログは削除して、次の処理を続行
            return None


def run_temporal_irl_analysis(projects: List[str] = None) -> Dict[str, Any]:
    """
    時系列逆強化学習分析を実行
    bot以外のauthorがレビューコメントを投稿したタイミングで学習を行う
    
    Args:
        projects (List[str]): 分析対象プロジェクト（Noneの場合は全プロジェクト）
        
    Returns:
        Dict[str, Any]: 分析結果
    """
    try:
        from ..utils.constants import START_DATE, END_DATE, OPENSTACK_CORE_COMPONENTS
    except ImportError:
        # 絶対インポートを試行
        try:
            from src.utils.constants import START_DATE, END_DATE, OPENSTACK_CORE_COMPONENTS
        except ImportError:
            # デフォルト値を使用
            START_DATE = "2015-01-01"
            END_DATE = "2024-12-31"
            OPENSTACK_CORE_COMPONENTS = ["nova", "neutron", "swift", "cinder", "keystone", "glance"]
    
    try:
        # プロジェクト設定
        target_projects = projects if projects else OPENSTACK_CORE_COMPONENTS
        
        # データ読み込み
        data_processor = ReviewPriorityDataProcessor()
        changes_df, releases_df = data_processor.load_openstack_data()
        if changes_df.empty:
            return {"error": "データが見つかりません"}
        
        # 日付範囲でフィルタリング
        start_datetime = datetime.strptime(START_DATE, "%Y-%m-%d")
        end_datetime = datetime.strptime(END_DATE, "%Y-%m-%d")
        
        # created日時による期間フィルタ
        mask = (changes_df['created'] >= start_datetime) & (changes_df['created'] <= end_datetime)
        changes_df = changes_df[mask]
        logger.info(f"期間フィルタ後のデータ数: {len(changes_df)} ({START_DATE} ～ {END_DATE})")
        
        # ボット名の読み込み
        bot_names = []
        try:
            config_path = DEFAULT_CONFIG / "gerrymanderconfig.ini"
            import configparser
            config = configparser.ConfigParser()
            config.read(config_path)
            if config.has_option('organization', 'bots'):
                bot_names = [name.strip() for name in config.get('organization', 'bots').split(',')]
        except Exception as e:
            logger.error(f"ボット名の読み込みに失敗: {e}")
            # デフォルトのボット名を設定
            bot_names = ['jenkins', 'elasticrecheck', 'zuul']
        
        # 時系列学習の実行
        all_learning_data = []
        total_events = 0
        successful_events = 0
        
        for project in target_projects:
            logger.info(f"\n=== プロジェクト '{project}' の時系列分析開始 ===")
            
            # プロジェクトのデータをフィルタ
            project_changes = changes_df[changes_df['component'] == project]
            if project_changes.empty:
                logger.warning(f"プロジェクト '{project}' のデータが見つかりません")
                continue
                
            logger.info(f"プロジェクト '{project}': {len(project_changes)}件のChange")
            
            # 学習イベントを抽出
            learning_events = extract_learning_events(project_changes, bot_names)
            
            # 期間内のイベントのみに限定
            filtered_events = []
            for event in learning_events:
                if start_datetime <= event['timestamp'] <= end_datetime:
                    filtered_events.append(event)
            
            learning_events = filtered_events
            total_events += len(learning_events)
            
            if not learning_events:
                logger.warning(f"プロジェクト '{project}': 学習イベントが見つかりません")
                continue
            
            logger.info(f"プロジェクト '{project}': {len(learning_events)}件の学習イベント")
            
            # 各学習イベントで学習データを作成
            for i, event in enumerate(learning_events):
                if i % 100 == 0:
                    logger.info(f"プロジェクト '{project}': 学習イベント処理中 {i+1}/{len(learning_events)}")
                
                event_time = event['timestamp']
                
                # その時刻にオープンだったChangeを取得
                open_changes = get_open_changes_at_time(project_changes, event_time)
                
                if len(open_changes) < 2:  # 選択肢が少なすぎる場合はスキップ
                    continue
                
                # 全オープンChangeに対する優先度重みを計算
                priority_weights = calculate_review_priorities(open_changes, event_time, bot_names)
                
                if not priority_weights:
                    continue
                
                # 特徴量を抽出
                try:
                    features_df = data_processor.extract_features(open_changes, event_time, project, releases_df)
                    
                    if features_df.empty:
                        continue
                    
                    # 各Changeに優先度重みを設定
                    features_df['priority_weight'] = features_df['change_number'].apply(
                        lambda x: priority_weights.get(str(x), 0.01)
                    )
                    
                    # 学習データに追加
                    learning_data = {
                        'project': project,
                        'event_time': event_time,
                        'features': features_df,
                        'priority_weights': priority_weights,
                        'open_changes_count': len(open_changes)
                    }
                    all_learning_data.append(learning_data)
                    successful_events += 1
                    
                except Exception as e:
                    logger.warning(f"学習イベント処理エラー: {e}")
                    continue
        
        logger.info(f"\n時系列学習データ作成完了:")
        logger.info(f"  総学習イベント数: {total_events}")
        logger.info(f"  成功した学習データ数: {successful_events}")
        
        if not all_learning_data:
            return {"error": "学習データが作成されませんでした"}
        
        # 特徴量カラム
        feature_columns = [
            'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
            'elapsed_time', 'revision_count', 'test_code_presence',
            'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
            'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
            'refactoring_confidence', 'uncompleted_requests'
        ]
        
        # プロジェクト毎にデータを分離して個別学習
        project_learning_data = {}
        for learning_data in all_learning_data:
            project = learning_data['project']
            if project not in project_learning_data:
                project_learning_data[project] = []
            project_learning_data[project].append(learning_data)
        
        # 結果の保存
        results_dir = DEFAULT_DATA_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        
        # 時系列範囲を含むファイル名
        start_date_str = START_DATE.replace('-', '')
        end_date_str = END_DATE.replace('-', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # プロジェクト毎の結果を格納
        project_results = {}
        overall_training_samples = 0
        
        for project, project_data in project_learning_data.items():
            logger.info(f"\n=== プロジェクト '{project}' のモデル学習開始 ===")
            
            # プロジェクトの学習データを統合
            project_features = []
            for learning_data in project_data:
                features = learning_data['features']
                if not features.empty:
                    project_features.append(features)
            
            if not project_features:
                logger.warning(f"プロジェクト '{project}': 有効な特徴量データがありません")
                project_results[project] = {"error": "有効な特徴量データがありません"}
                continue
            
            # プロジェクトの特徴量を統合
            combined_features = pd.concat(project_features, ignore_index=True)
            
            # 特徴量とラベルを抽出
            X = combined_features[feature_columns].fillna(0).values
            y = combined_features['priority_weight'].values
            
            if len(X) < 5:  # 最小学習データ数チェック
                logger.warning(f"プロジェクト '{project}': 学習データが不足しています (件数: {len(X)})")
                project_results[project] = {"error": f"学習データ不足 (件数: {len(X)})"}
                continue
            
            logger.info(f"プロジェクト '{project}' 学習データサイズ: {X.shape}")
            overall_training_samples += len(X)
            
            # IRLモデルの学習
            model = MaxEntIRLModel(learning_rate=0.01, max_iterations=1000)
            model.feature_names = feature_columns
            
            training_stats = model.fit(X, y)
            logger.info(f"プロジェクト '{project}' 学習完了: 収束={training_stats['converged']}, 反復回数={training_stats['iterations']}")
            
            # プロジェクト個別のモデル保存
            model_path = results_dir / f"irl_model_{project}_{start_date_str}_{end_date_str}.pkl"
            model.save_model(model_path)
            
            # プロジェクト結果
            project_results[project] = {
                "training_stats": training_stats,
                "model_path": str(model_path),
                "data_summary": {
                    "training_samples": len(X),
                    "learning_events": len(project_data)
                },
                "feature_weights": dict(zip(feature_columns, model.weights.tolist()))
            }
        
        # 全体結果をまとめ
        results = {
            "date_range": {
                "start": START_DATE,
                "end": END_DATE
            },
            "timestamp": timestamp,
            "data_summary": {
                "target_projects": target_projects,
                "total_learning_events": total_events,
                "successful_learning_data": successful_events,
                "total_training_samples": overall_training_samples,
                "feature_dimensions": len(feature_columns),
                "projects_analyzed": len([p for p in project_results.keys() if "error" not in project_results[p]]),
                "projects_failed": len([p for p in project_results.keys() if "error" in project_results[p]])
            },
            "project_results": project_results
        }
        
        # 結果保存
        results_path = results_dir / f"irl_analysis_{start_date_str}_{end_date_str}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"時系列分析完了: 結果は {results_path} に保存されました")
        
        return results
        
    except Exception as e:
        logger.error(f"時系列分析エラー: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    """
    メイン実行部分
    時系列IRLモデルによる分析を実行
    """
    import os
    
    # 相対インポートの問題を解決するためにパスを追加
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..')
    import sys
    sys.path.insert(0, project_root)
    
    try:
        from src.utils.constants import START_DATE, END_DATE
    except ImportError:
        # フォールバック: 相対インポートを試行
        try:
            from ..utils.constants import START_DATE, END_DATE
        except ImportError:
            # 最終フォールバック: デフォルト値を使用
            print("警告: constants.pyの読み込みに失敗しました。デフォルト値を使用します。")
            START_DATE = "2015-01-01"
            END_DATE = "2024-12-31"
    
    print(f"分析設定: {START_DATE} ～ {END_DATE}")
    
    try:
        results = run_temporal_irl_analysis()
        
        if "error" not in results:
            print(f"\n=== 時系列IRLモデル分析完了（プロジェクト毎） ===")
            print(f"期間: {results['date_range']['start']} ～ {results['date_range']['end']}")
            print(f"総学習データ: {results['data_summary']['total_training_samples']}件")
            print(f"分析成功プロジェクト: {results['data_summary']['projects_analyzed']}件")
            print(f"分析失敗プロジェクト: {results['data_summary']['projects_failed']}件")
            
            # プロジェクト毎の結果概要を表示
            print("\n各プロジェクトの結果:")
            for project, result in results['project_results'].items():
                if "error" not in result:
                    stats = result['training_stats']
                    print(f"  {project}: 学習データ{result['data_summary']['training_samples']}件, "
                          f"収束={stats['converged']}, 反復{stats['iterations']}回")
                else:
                    print(f"  {project}: エラー - {result['error']}")
        else:
            print(f"エラー: {results['error']}")
            
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        raise