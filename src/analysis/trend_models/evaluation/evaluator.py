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
    ndcg_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.analysis.trend_models.classification_models.trend_predictor import TrendPredictor
from src.analysis.trend_models.ranking.ranking_predictor import RankingPredictor
from src.analysis.trend_models.ranking.ranking_dataset import (
    build_ranking_matrix,
    get_ranking_target_column,
)
from src.analysis.trend_models.utils.constants import (
    FEATURE_NAMES,
    DEFAULT_MODEL_TYPE,
    DEFAULT_RANKING_LABEL_MODE,
    DEFAULT_RANKING_MODEL_TYPE,
    RANKING_K_VALUES,
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


@dataclass
class RankingEvaluationResult:
    """ランキング評価結果を格納するデータクラス"""
    ndcg_at_5: float
    ndcg_at_10: float
    ndcg_at_20: float
    mrr: float
    spearman: float
    pairwise_accuracy: float
    precision: float
    recall: float
    f1: float
    mae: float
    rmse: float
    n_queries: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankingCVResult:
    """ランキング LOO-CV 結果を格納するデータクラス"""
    eval_release: str
    eval_period: str
    train_period: str
    developer_type: str
    target_col: str
    ndcg_at_5: float
    ndcg_at_10: float
    ndcg_at_20: float
    mrr: float
    spearman: float
    pairwise_accuracy: float
    precision: float
    recall: float
    f1: float
    mae: float
    rmse: float
    n_train: int
    n_eval: int
    n_queries: int
    feature_importances: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
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


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _to_relevance_from_target(y_target: np.ndarray, target_col: str) -> np.ndarray:
    """小さいほど優先の target を、NDCG 用の大きいほど relevant な値へ変換する。"""
    y = np.asarray(y_target, dtype=float)

    if target_col.endswith('rank_pct'):
        return 1.0 - np.clip(y, 0.0, 1.0)

    if target_col.endswith('rank'):
        return 1.0 / np.maximum(y, 1.0)

    if 'time_to_review_seconds' in target_col:
        return 1.0 / (1.0 + np.maximum(y, 0.0))

    span = float(y.max() - y.min()) if len(y) > 0 else 0.0
    if span <= 0:
        return np.ones_like(y)
    return (y.max() - y) / span


def _compute_mrr(y_binary: np.ndarray, pred_scores: np.ndarray) -> float:
    order = np.argsort(-pred_scores)
    for rank_idx, sample_idx in enumerate(order, start=1):
        if y_binary[sample_idx] > 0:
            return 1.0 / float(rank_idx)
    return 0.0


def _compute_pairwise_accuracy(y_target: np.ndarray, pred_scores: np.ndarray) -> float:
    n = len(y_target)
    if n < 2:
        return 0.0

    correct = 0.0
    total = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y_target[i] == y_target[j]:
                continue

            true_i_better = y_target[i] < y_target[j]
            pred_i_better = pred_scores[i] > pred_scores[j]

            total += 1
            if pred_scores[i] == pred_scores[j]:
                correct += 0.5
            elif true_i_better == pred_i_better:
                correct += 1.0

    if total == 0:
        return 0.0
    return float(correct / total)


def evaluate_ranking_by_query(
    eval_df: pd.DataFrame,
    target_col: str,
    score_col: str = 'pred_score',
    pred_target_col: str = 'pred_target',
    query_col: str = 'query_id',
    reviewed_col: str = 'reviewed',
    k_values: List[int] = None,
) -> RankingEvaluationResult:
    """query 単位でランキング指標を評価し、平均を返す。"""
    if eval_df.empty:
        return RankingEvaluationResult(
            ndcg_at_5=0.0,
            ndcg_at_10=0.0,
            ndcg_at_20=0.0,
            mrr=0.0,
            spearman=0.0,
            pairwise_accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            mae=0.0,
            rmse=0.0,
            n_queries=0,
        )

    if k_values is None:
        k_values = RANKING_K_VALUES

    ndcg_values = {k: [] for k in k_values}
    mrr_values: List[float] = []
    spearman_values: List[float] = []
    pairwise_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []
    mae_values: List[float] = []
    rmse_values: List[float] = []

    valid_queries = 0

    grouped = eval_df.groupby(query_col) if query_col in eval_df.columns else [('all', eval_df)]
    for _, group in grouped:
        if len(group) < 2:
            continue

        y_target = group[target_col].to_numpy(dtype=float)
        pred_scores = group[score_col].to_numpy(dtype=float)
        if pred_target_col in group.columns:
            pred_target = group[pred_target_col].to_numpy(dtype=float)
        else:
            pred_target = -pred_scores
        relevance = _to_relevance_from_target(y_target, target_col)

        valid_queries += 1

        for k in k_values:
            k_eff = max(1, min(k, len(group)))
            ndcg = ndcg_score([relevance], [pred_scores], k=k_eff)
            ndcg_values[k].append(float(ndcg))

        if reviewed_col in group.columns:
            y_binary = group[reviewed_col].to_numpy(dtype=int)
        else:
            best = y_target == np.min(y_target)
            y_binary = best.astype(int)

        mrr_values.append(_compute_mrr(y_binary, pred_scores))
        pairwise_values.append(_compute_pairwise_accuracy(y_target, pred_scores))

        spearman = pd.Series(pred_scores).corr(pd.Series(-y_target), method='spearman')
        if pd.notna(spearman):
            spearman_values.append(float(spearman))

        positive_count = int(np.sum(y_binary == 1))
        top_n = positive_count if positive_count > 0 else max(1, int(np.ceil(len(group) * 0.1)))
        top_n = min(top_n, len(group))

        order = np.argsort(-pred_scores)
        y_pred_binary = np.zeros(len(group), dtype=int)
        y_pred_binary[order[:top_n]] = 1

        precision_values.append(precision_score(y_binary, y_pred_binary, zero_division=0))
        recall_values.append(recall_score(y_binary, y_pred_binary, zero_division=0))
        f1_values.append(f1_score(y_binary, y_pred_binary, zero_division=0))
        mae_values.append(float(mean_absolute_error(y_target, pred_target)))
        rmse_values.append(float(np.sqrt(mean_squared_error(y_target, pred_target))))

    ndcg_mean = {k: _safe_mean(values) for k, values in ndcg_values.items()}
    return RankingEvaluationResult(
        ndcg_at_5=ndcg_mean.get(5, 0.0),
        ndcg_at_10=ndcg_mean.get(10, 0.0),
        ndcg_at_20=ndcg_mean.get(20, 0.0),
        mrr=_safe_mean(mrr_values),
        spearman=_safe_mean(spearman_values),
        pairwise_accuracy=_safe_mean(pairwise_values),
        precision=_safe_mean(precision_values),
        recall=_safe_mean(recall_values),
        f1=_safe_mean(f1_values),
        mae=_safe_mean(mae_values),
        rmse=_safe_mean(rmse_values),
        n_queries=valid_queries,
    )


def leave_one_out_ranking_cv(
    release_data: Dict[str, Dict[str, pd.DataFrame]],
    feature_names: List[str] = None,
    ranking_model_type: str = None,
    label_mode: str = DEFAULT_RANKING_LABEL_MODE,
    k_values: List[int] = None,
    split_by_developer_type: bool = False,
) -> List[RankingCVResult]:
    """ランキング用 Leave-One-Out 交差検証を実行する。"""
    feature_names = feature_names or FEATURE_NAMES
    ranking_model_type = ranking_model_type or DEFAULT_RANKING_MODEL_TYPE
    target_col = get_ranking_target_column(label_mode)
    k_values = k_values or RANKING_K_VALUES

    releases = list(release_data.keys())
    results: List[RankingCVResult] = []

    train_periods = ['early', 'late', 'all']
    eval_periods = ['early', 'late']

    for eval_release in releases:
        eval_data = release_data[eval_release]
        train_releases = [release for release in releases if release != eval_release]

        for train_period in train_periods:
            train_dfs = []
            for train_release in train_releases:
                period_df = release_data[train_release].get(train_period)
                if period_df is not None and len(period_df) > 0:
                    train_dfs.append(period_df)

            if not train_dfs:
                continue

            train_df = pd.concat(train_dfs, ignore_index=True)
            if target_col not in train_df.columns or 'query_id' not in train_df.columns:
                continue

            try:
                X_train, y_train, group_train = build_ranking_matrix(
                    train_df,
                    feature_names=feature_names,
                    target_col=target_col,
                    query_col='query_id',
                )
            except (KeyError, ValueError) as exc:
                logger.warning("ランキング学習データ生成失敗: %s", exc)
                continue

            model = RankingPredictor(
                model_type=ranking_model_type,
                feature_names=feature_names,
            )
            model.fit(X_train, y_train, group=group_train)

            for eval_period in eval_periods:
                if eval_period not in eval_data:
                    continue

                eval_df = eval_data[eval_period]
                if len(eval_df) == 0 or target_col not in eval_df.columns:
                    continue

                if split_by_developer_type and 'developer_type' in eval_df.columns:
                    developer_types = eval_df['developer_type'].unique().tolist() + ['all']
                else:
                    developer_types = ['all']

                for dev_type in developer_types:
                    if dev_type == 'all':
                        eval_df_filtered = eval_df.copy()
                    else:
                        eval_df_filtered = eval_df[eval_df['developer_type'] == dev_type].copy()

                    if len(eval_df_filtered) == 0:
                        continue

                    feature_cols = [col for col in feature_names if col in eval_df_filtered.columns]
                    if not feature_cols:
                        continue

                    X_eval = eval_df_filtered[feature_cols].values
                    pred_target = model.predict(X_eval)
                    pred_scores = model.predict_score(X_eval)
                    eval_df_filtered['pred_target'] = pred_target
                    eval_df_filtered['pred_score'] = pred_scores

                    evaluation = evaluate_ranking_by_query(
                        eval_df=eval_df_filtered,
                        target_col=target_col,
                        score_col='pred_score',
                        pred_target_col='pred_target',
                        query_col='query_id',
                        reviewed_col='reviewed',
                        k_values=k_values,
                    )

                    results.append(
                        RankingCVResult(
                            eval_release=eval_release,
                            eval_period=eval_period,
                            train_period=train_period,
                            developer_type=dev_type,
                            target_col=target_col,
                            ndcg_at_5=evaluation.ndcg_at_5,
                            ndcg_at_10=evaluation.ndcg_at_10,
                            ndcg_at_20=evaluation.ndcg_at_20,
                            mrr=evaluation.mrr,
                            spearman=evaluation.spearman,
                            pairwise_accuracy=evaluation.pairwise_accuracy,
                            precision=evaluation.precision,
                            recall=evaluation.recall,
                            f1=evaluation.f1,
                            mae=evaluation.mae,
                            rmse=evaluation.rmse,
                            n_train=len(train_df),
                            n_eval=len(eval_df_filtered),
                            n_queries=evaluation.n_queries,
                            feature_importances=model.get_feature_importance_dict(),
                        )
                    )

    logger.info("ランキング Leave-One-Out 交差検証完了: %d件", len(results))
    return results


def summarize_ranking_cv_results(results: List[RankingCVResult]) -> pd.DataFrame:
    """ランキング CV 結果を集計する。"""
    if not results:
        return pd.DataFrame(columns=[
            'eval_period',
            'train_period',
            'developer_type',
            'ndcg_at_5_mean',
            'ndcg_at_10_mean',
            'ndcg_at_20_mean',
            'mrr_mean',
            'spearman_mean',
            'pairwise_accuracy_mean',
            'precision_mean',
            'recall_mean',
            'f1_mean',
            'mae_mean',
            'rmse_mean',
            'n_train_mean',
            'n_eval_mean',
            'n_queries_mean',
        ])

    df = pd.DataFrame([result.to_dict() for result in results])

    summary = df.groupby(['eval_period', 'train_period', 'developer_type']).agg({
        'ndcg_at_5': ['mean', 'std'],
        'ndcg_at_10': ['mean', 'std'],
        'ndcg_at_20': ['mean', 'std'],
        'mrr': ['mean', 'std'],
        'spearman': ['mean', 'std'],
        'pairwise_accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'n_train': 'mean',
        'n_eval': 'mean',
        'n_queries': 'mean',
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    return summary
