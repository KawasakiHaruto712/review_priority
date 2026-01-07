"""
Trend Models Analysis - モデル評価

本モジュールでは、予測モデルの評価機能を提供します。
Precision, Recall, F1スコアの計算や交差検証の実行を行います。
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from src.analysis.trend_models.models.trend_predictor import TrendPredictor
from src.analysis.trend_models.utils.constants import (
    FEATURE_NAMES,
    DEFAULT_MODEL_TYPE,
)

logger = logging.getLogger(__name__)

# ロギング設定
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class EvaluationResult:
    """評価結果を格納するデータクラス"""
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: np.ndarray
    n_train: int
    n_eval: int
    n_positive: int
    n_negative: int
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        d = asdict(self)
        d['confusion_matrix'] = self.confusion_matrix.tolist()
        return d


@dataclass
class CVResult:
    """交差検証結果を格納するデータクラス"""
    eval_release: str
    eval_period: str
    train_period: str
    developer_type: str
    precision: float
    recall: float
    f1: float
    n_train: int
    n_eval: int
    feature_importances: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


class Evaluator:
    """
    モデル評価を行うクラス
    """
    
    def __init__(
        self,
        model_type: str = None,
        feature_names: List[str] = None
    ):
        """
        Args:
            model_type: モデルタイプ
            feature_names: 特徴量名のリスト
        """
        self.model_type = model_type or DEFAULT_MODEL_TYPE
        self.feature_names = feature_names or FEATURE_NAMES
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_train: int = 0
    ) -> EvaluationResult:
        """
        予測結果を評価する
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            n_train: 学習データ数
            
        Returns:
            EvaluationResult: 評価結果
        """
        # ゼロ除算を避けるためzero_divisionパラメータを設定
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        n_positive = int(np.sum(y_true == 1))
        n_negative = int(np.sum(y_true == 0))
        
        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            confusion_matrix=cm,
            n_train=n_train,
            n_eval=len(y_true),
            n_positive=n_positive,
            n_negative=n_negative
        )
    
    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        model: TrendPredictor = None
    ) -> Tuple[EvaluationResult, TrendPredictor]:
        """
        モデルを学習して評価する
        
        Args:
            X_train: 学習用特徴量
            y_train: 学習用ラベル
            X_eval: 評価用特徴量
            y_eval: 評価用ラベル
            model: 使用するモデル（Noneの場合は新規作成）
            
        Returns:
            Tuple[EvaluationResult, TrendPredictor]: (評価結果, 学習済みモデル)
        """
        if model is None:
            model = TrendPredictor(
                model_type=self.model_type,
                feature_names=self.feature_names
            )
        
        # 学習
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_eval)
        
        # 評価
        result = self.evaluate(y_eval, y_pred, n_train=len(X_train))
        
        return result, model


def evaluate_model(
    model: TrendPredictor,
    X_eval: np.ndarray,
    y_eval: np.ndarray
) -> EvaluationResult:
    """
    学習済みモデルを評価する
    
    Args:
        model: 学習済みモデル
        X_eval: 評価用特徴量
        y_eval: 評価用ラベル
        
    Returns:
        EvaluationResult: 評価結果
    """
    evaluator = Evaluator()
    y_pred = model.predict(X_eval)
    return evaluator.evaluate(y_eval, y_pred)


def cross_period_evaluation(
    train_data: Dict[str, pd.DataFrame],
    eval_data: Dict[str, pd.DataFrame],
    feature_names: List[str] = None,
    model_type: str = None
) -> List[CVResult]:
    """
    期間交差評価を行う
    
    学習期間と評価期間の組み合わせで性能を比較します。
    
    Args:
        train_data: 期間タイプをキー、学習用DataFrameを値とする辞書
        eval_data: 期間タイプをキー、評価用DataFrameを値とする辞書
        feature_names: 特徴量名のリスト
        model_type: モデルタイプ
        
    Returns:
        List[CVResult]: 交差検証結果のリスト
    """
    feature_names = feature_names or FEATURE_NAMES
    evaluator = Evaluator(model_type=model_type, feature_names=feature_names)
    
    results = []
    
    train_periods = ['early', 'late', 'all']
    eval_periods = ['early', 'late']
    
    for train_period in train_periods:
        if train_period not in train_data:
            continue
        
        train_df = train_data[train_period]
        if len(train_df) == 0:
            continue
        
        # 特徴量とラベルを取得
        X_train = train_df[[col for col in feature_names if col in train_df.columns]].values
        y_train = train_df['reviewed'].values
        
        for eval_period in eval_periods:
            if eval_period not in eval_data:
                continue
            
            eval_df = eval_data[eval_period]
            if len(eval_df) == 0:
                continue
            
            X_eval = eval_df[[col for col in feature_names if col in eval_df.columns]].values
            y_eval = eval_df['reviewed'].values
            
            # モデルを学習して評価
            result, model = evaluator.train_and_evaluate(X_train, y_train, X_eval, y_eval)
            
            cv_result = CVResult(
                eval_release='',  # リリース情報は呼び出し元で設定
                eval_period=eval_period,
                train_period=train_period,
                developer_type='all',
                precision=result.precision,
                recall=result.recall,
                f1=result.f1,
                n_train=result.n_train,
                n_eval=result.n_eval,
                feature_importances=model.get_feature_importance_dict()
            )
            results.append(cv_result)
            
            logger.info(f"期間交差評価: 学習={train_period} → 評価={eval_period}, "
                       f"F1={result.f1:.4f}")
    
    return results


def leave_one_out_cv(
    release_data: Dict[str, Dict[str, pd.DataFrame]],
    feature_names: List[str] = None,
    model_type: str = None,
    split_by_developer_type: bool = False
) -> List[CVResult]:
    """
    Leave-One-Out交差検証を実行する
    
    各リリースを順番に評価データとして使用し、残りのリリースで学習します。
    
    Args:
        release_data: リリースをキー、期間タイプ別DataFrameの辞書を値とする辞書
        feature_names: 特徴量名のリスト
        model_type: モデルタイプ
        split_by_developer_type: 開発者タイプで分割するかどうか
        
    Returns:
        List[CVResult]: 交差検証結果のリスト
    """
    feature_names = feature_names or FEATURE_NAMES
    evaluator = Evaluator(model_type=model_type, feature_names=feature_names)
    
    releases = list(release_data.keys())
    results = []
    
    train_periods = ['early', 'late', 'all']
    eval_periods = ['early', 'late']
    
    for eval_release in releases:
        # 評価データ
        eval_data = release_data[eval_release]
        
        # 学習データ（評価リリース以外）
        train_releases = [r for r in releases if r != eval_release]
        
        for train_period in train_periods:
            # 学習データを結合
            train_dfs = []
            for release in train_releases:
                if train_period in release_data[release]:
                    train_dfs.append(release_data[release][train_period])
            
            if not train_dfs:
                continue
            
            train_df = pd.concat(train_dfs, ignore_index=True)
            if len(train_df) == 0:
                continue
            
            # 特徴量とラベルを取得
            feature_cols = [col for col in feature_names if col in train_df.columns]
            X_train = train_df[feature_cols].values
            y_train = train_df['reviewed'].values
            
            for eval_period in eval_periods:
                if eval_period not in eval_data:
                    continue
                
                eval_df = eval_data[eval_period]
                if len(eval_df) == 0:
                    continue
                
                # 開発者タイプで分割
                if split_by_developer_type and 'developer_type' in eval_df.columns:
                    developer_types = eval_df['developer_type'].unique().tolist() + ['all']
                else:
                    developer_types = ['all']
                
                for dev_type in developer_types:
                    if dev_type == 'all':
                        eval_df_filtered = eval_df
                    else:
                        eval_df_filtered = eval_df[eval_df['developer_type'] == dev_type]
                    
                    if len(eval_df_filtered) == 0:
                        continue
                    
                    X_eval = eval_df_filtered[feature_cols].values
                    y_eval = eval_df_filtered['reviewed'].values
                    
                    # モデルを学習して評価
                    result, model = evaluator.train_and_evaluate(X_train, y_train, X_eval, y_eval)
                    
                    cv_result = CVResult(
                        eval_release=eval_release,
                        eval_period=eval_period,
                        train_period=train_period,
                        developer_type=dev_type,
                        precision=result.precision,
                        recall=result.recall,
                        f1=result.f1,
                        n_train=result.n_train,
                        n_eval=result.n_eval,
                        feature_importances=model.get_feature_importance_dict()
                    )
                    results.append(cv_result)
                    
                    logger.debug(f"LOO-CV: 評価={eval_release}/{eval_period}, "
                                f"学習={train_period}, 開発者={dev_type}, "
                                f"F1={result.f1:.4f}")
    
    logger.info(f"Leave-One-Out交差検証完了: {len(results)}件の結果")
    return results


def summarize_cv_results(results: List[CVResult]) -> pd.DataFrame:
    """
    交差検証結果を集計する
    
    Args:
        results: 交差検証結果のリスト
        
    Returns:
        pd.DataFrame: 集計結果のDataFrame
    """
    # 結果をDataFrameに変換
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # 期間ごとに集計
    summary = df.groupby(['eval_period', 'train_period', 'developer_type']).agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'n_train': 'mean',
        'n_eval': 'mean',
    }).round(4)
    
    # カラム名をフラット化
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    return summary
