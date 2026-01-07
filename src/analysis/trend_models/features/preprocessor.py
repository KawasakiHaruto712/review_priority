"""
Trend Models Analysis - データ前処理

本モジュールでは、レビュー済み/未レビューのラベル付け、欠損値処理、
特徴量の正規化などの前処理機能を提供します。
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.analysis.trend_models.utils.constants import FEATURE_NAMES
from src.analysis.trend_models.utils.data_loader import (
    get_reviewers_in_period,
    load_bot_names_from_config,
)

logger = logging.getLogger(__name__)

# ロギング設定
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Preprocessor:
    """
    特徴量データの前処理を行うクラス
    """
    
    def __init__(
        self,
        feature_names: List[str] = None,
        bot_names: List[str] = None,
        core_developers: Dict[str, List[Dict[str, str]]] = None
    ):
        """
        Args:
            feature_names: 使用する特徴量名のリスト
            bot_names: ボット名のリスト（レビューア判定時に除外）
            core_developers: コア開発者情報
        """
        self.feature_names = feature_names or FEATURE_NAMES
        self.bot_names = bot_names or load_bot_names_from_config()
        self.core_developers = core_developers or {}
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def add_labels(
        self,
        features_df: pd.DataFrame,
        changes: List[Dict[str, Any]],
        period_start: datetime,
        period_end: datetime
    ) -> pd.DataFrame:
        """
        各Changeにレビュー済み/未レビューのラベルを付与する
        
        Args:
            features_df: 特徴量のDataFrame
            changes: Changeデータのリスト（messagesを含む）
            period_start: 期間開始日時
            period_end: 期間終了日時
            
        Returns:
            pd.DataFrame: ラベルが追加されたDataFrame
        """
        df = features_df.copy()
        
        # change_numberをキーとしたChangeデータのマッピング
        change_map = {c.get('change_number'): c for c in changes}
        
        labels = []
        reviewer_counts = []
        
        for idx, row in df.iterrows():
            change_number = row.get('change_number')
            change = change_map.get(change_number, {})
            
            # 期間内のレビューアを取得
            reviewers = get_reviewers_in_period(
                change, period_start, period_end, self.bot_names
            )
            
            # レビューアが1人以上いれば reviewed=True (ラベル=1)
            reviewed = 1 if len(reviewers) > 0 else 0
            labels.append(reviewed)
            reviewer_counts.append(len(reviewers))
        
        df['reviewed'] = labels
        df['reviewer_count'] = reviewer_counts
        
        logger.info(f"ラベル付け完了: レビュー済み={sum(labels)}, 未レビュー={len(labels) - sum(labels)}")
        return df
    
    def add_developer_type(
        self,
        features_df: pd.DataFrame,
        changes: List[Dict[str, Any]],
        project_name: str,
        period_start: datetime = None,
        period_end: datetime = None
    ) -> pd.DataFrame:
        """
        レビューア開発者タイプ（Core/Non-Core）を追加する
        
        期間中にCore開発者からレビューされたか、Non-Core開発者からレビューされたかを判定
        
        Args:
            features_df: 特徴量のDataFrame
            changes: Changeデータのリスト
            project_name: プロジェクト名
            period_start: 期間開始日（レビューアの判定に使用）
            period_end: 期間終了日（レビューアの判定に使用）
            
        Returns:
            pd.DataFrame: 開発者タイプが追加されたDataFrame
        """
        df = features_df.copy()
        
        # コア開発者のメールアドレスを取得
        core_emails = set()
        project_core = self.core_developers.get(project_name, [])
        for member in project_core:
            if isinstance(member, dict):
                email = member.get('email', '')
                if email:
                    core_emails.add(email.lower())
            elif isinstance(member, str):
                core_emails.add(member.lower())
        
        # change_numberをキーとしたChangeデータのマッピング
        change_map = {c.get('change_number'): c for c in changes}
        
        developer_types = []
        for idx, row in df.iterrows():
            change_number = row.get('change_number')
            change = change_map.get(change_number, {})
            
            # 期間中のレビューアを取得
            reviewers = self._get_period_reviewers(
                change, period_start, period_end
            )
            
            # レビューアにCore開発者がいるかどうかで判定
            has_core_reviewer = False
            for reviewer_email in reviewers:
                if reviewer_email.lower() in core_emails:
                    has_core_reviewer = True
                    break
            
            if has_core_reviewer:
                developer_types.append('core')
            else:
                developer_types.append('non-core')
        
        df['developer_type'] = developer_types
        
        core_count = sum(1 for dt in developer_types if dt == 'core')
        logger.info(f"レビューア開発者タイプ付与: Core={core_count}, Non-Core={len(developer_types) - core_count}")
        return df
    
    def _get_period_reviewers(
        self,
        change: Dict[str, Any],
        period_start: datetime = None,
        period_end: datetime = None
    ) -> List[str]:
        """
        期間中にレビューしたレビューアのメールアドレスリストを取得
        
        Args:
            change: Changeデータ
            period_start: 期間開始日
            period_end: 期間終了日
            
        Returns:
            List[str]: レビューアのメールアドレスリスト
        """
        reviewers = []
        
        # Change作成者を取得（除外用）
        owner = change.get('owner', {})
        owner_email = owner.get('email', '').lower()
        
        messages = change.get('messages', [])
        
        for message in messages:
            # メッセージ日時をパース
            msg_date_str = message.get('date', '')
            msg_date = None
            if msg_date_str:
                try:
                    msg_date_str = str(msg_date_str).replace('.000000000', '').replace(' ', 'T')
                    msg_date = datetime.fromisoformat(msg_date_str)
                except (ValueError, TypeError):
                    continue
            
            if msg_date is None:
                continue
            
            # 期間内かチェック
            if period_start and msg_date < period_start:
                continue
            if period_end and msg_date > period_end:
                continue
            
            # 著者情報を取得
            author = message.get('author', {})
            author_email = author.get('email', '').lower()
            
            # ボットを除外
            author_username = author.get('username', '')
            is_bot = False
            for bot_name in self.bot_names:
                if bot_name.lower() in author_username.lower():
                    is_bot = True
                    break
                if bot_name.lower() in author_email.lower():
                    is_bot = True
                    break
            
            if is_bot:
                continue
            
            # SERVICE_USERタグがあれば除外
            if 'SERVICE_USER' in author.get('tags', []):
                continue
            
            # Change作成者を除外
            if author_email == owner_email:
                continue
            
            # 自動生成されたメッセージを除外
            tag = message.get('tag', '')
            if tag.startswith('autogenerated:'):
                continue
            
            # レビューアとして追加
            if author_email:
                reviewers.append(author_email)
        
        return reviewers
    
    def handle_missing_values(
        self,
        features_df: pd.DataFrame,
        strategy: str = 'zero'
    ) -> pd.DataFrame:
        """
        欠損値を処理する
        
        Args:
            features_df: 特徴量のDataFrame
            strategy: 欠損値処理方法 ('zero', 'mean', 'median', 'drop')
            
        Returns:
            pd.DataFrame: 欠損値が処理されたDataFrame
        """
        df = features_df.copy()
        
        # 特徴量カラムのみを処理
        feature_cols = [col for col in self.feature_names if col in df.columns]
        
        if strategy == 'zero':
            df[feature_cols] = df[feature_cols].fillna(0)
        elif strategy == 'mean':
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
        elif strategy == 'median':
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        elif strategy == 'drop':
            df = df.dropna(subset=feature_cols)
        
        # -1.0（計算不可を表す値）も0に置き換え
        for col in feature_cols:
            df.loc[df[col] < 0, col] = 0
        
        logger.info(f"欠損値処理完了 (strategy={strategy}): {len(df)}件")
        return df
    
    def normalize(
        self,
        features_df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        特徴量を正規化（標準化）する
        
        Args:
            features_df: 特徴量のDataFrame
            fit: Scalerをフィットするかどうか（学習時はTrue、評価時はFalse）
            
        Returns:
            pd.DataFrame: 正規化されたDataFrame
        """
        df = features_df.copy()
        
        # 特徴量カラムのみを正規化
        feature_cols = [col for col in self.feature_names if col in df.columns]
        
        if len(feature_cols) == 0:
            logger.warning("正規化対象の特徴量カラムがありません")
            return df
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            self._is_fitted = True
        else:
            if not self._is_fitted:
                logger.warning("Scalerがフィットされていません。先にfit=Trueで呼び出してください")
                return df
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        logger.debug(f"正規化完了: {len(feature_cols)}カラム")
        return df
    
    def get_feature_matrix(
        self,
        features_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        特徴量行列とラベル配列を取得する
        
        Args:
            features_df: 特徴量のDataFrame
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (特徴量行列X, ラベル配列y)
        """
        feature_cols = [col for col in self.feature_names if col in features_df.columns]
        
        X = features_df[feature_cols].values
        y = features_df['reviewed'].values if 'reviewed' in features_df.columns else None
        
        return X, y
    
    def split_by_developer_type(
        self,
        features_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        開発者タイプでデータを分割する
        
        Args:
            features_df: 特徴量のDataFrame
            
        Returns:
            Dict[str, pd.DataFrame]: 開発者タイプをキー、DataFrameを値とする辞書
        """
        if 'developer_type' not in features_df.columns:
            logger.warning("developer_typeカラムがありません。全データを返します")
            return {'all': features_df}
        
        result = {}
        for dtype in features_df['developer_type'].unique():
            result[dtype] = features_df[features_df['developer_type'] == dtype].copy()
        
        # 全データも含める
        result['all'] = features_df.copy()
        
        return result


def preprocess_data(
    features_df: pd.DataFrame,
    changes: List[Dict[str, Any]],
    period_start: datetime,
    period_end: datetime,
    project_name: str = None,
    bot_names: List[str] = None,
    core_developers: Dict[str, List[Dict[str, str]]] = None,
    normalize: bool = True,
    add_developer_type: bool = False,
    missing_value_strategy: str = 'zero'
) -> pd.DataFrame:
    """
    データ前処理のエントリーポイント
    
    Args:
        features_df: 特徴量のDataFrame
        changes: Changeデータのリスト
        period_start: 期間開始日時
        period_end: 期間終了日時
        project_name: プロジェクト名（開発者タイプ付与に使用）
        bot_names: ボット名のリスト
        core_developers: コア開発者情報
        normalize: 正規化を行うかどうか
        add_developer_type: 開発者タイプを追加するかどうか
        missing_value_strategy: 欠損値処理方法
        
    Returns:
        pd.DataFrame: 前処理されたDataFrame
    """
    preprocessor = Preprocessor(
        bot_names=bot_names,
        core_developers=core_developers
    )
    
    # ラベル付け
    df = preprocessor.add_labels(features_df, changes, period_start, period_end)
    
    # 開発者タイプ付与
    if add_developer_type and project_name:
        df = preprocessor.add_developer_type(df, changes, project_name)
    
    # 欠損値処理
    df = preprocessor.handle_missing_values(df, strategy=missing_value_strategy)
    
    # 正規化
    if normalize:
        df = preprocessor.normalize(df, fit=True)
    
    return df
