"""順位ベースの評価指標（§2.8）。3 指標とも 0〜1 のサイズ非依存スケール。

- 予測時間から導出した順位 と 正解順位（実 time_to_next_review）のズレを測る。
- MAE/RMSE は順位を [0,1] に正規化（(rank-1)/(n-1)）してから算出（アクティブ集合サイズ非依存）。
- NDCG@n は定義上すでに 0〜1。relevance は正解順位の反転（早いほど高 relevance）。
- 各指標は「ある計測点 T のアクティブ集合（n 件）」の中で計算する。n<2 は呼ばない（順位が自明）。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _norm_ranks(values: np.ndarray) -> np.ndarray:
    """昇順順位（小さい値＝1位）を [0,1] に正規化。1位→0.0, 最下位→1.0。"""
    n = len(values)
    ranks = rankdata(values, method="average")  # 1..n（同値は平均順位）
    return (ranks - 1.0) / (n - 1.0)


def rank_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """正規化順位の平均絶対誤差（小さいほど良い）。"""
    return float(np.mean(np.abs(_norm_ranks(y_pred) - _norm_ranks(y_true))))


def rank_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """正規化順位の二乗平均平方根誤差（小さいほど良い）。"""
    diff = _norm_ranks(y_pred) - _norm_ranks(y_true)
    return float(np.sqrt(np.mean(diff ** 2)))


def ndcg_at(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
    """NDCG@n（大きいほど良い、0〜1）。relevance = 正解順位の反転（早い＝高 relevance）。"""
    size = len(y_true)
    # relevance: 実時間が短いほど高い。正解順位(昇順,1..size)を反転 → size - rank + 1
    true_rank = rankdata(y_true, method="ordinal")
    relevance = size - true_rank + 1  # 1..size（最速が最大）

    # 予測順（予測時間の昇順＝高優先順）に並べる
    order = np.argsort(y_pred, kind="stable")
    k = min(n, size)
    gains = relevance[order][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # 位置 1..k → log2(2..k+1)
    dcg = float(np.sum(gains * discounts))

    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else 0.0


_FUNCS = {"mae": rank_mae, "rmse": rank_rmse}


def compute(y_true: np.ndarray, y_pred: np.ndarray, metric_names: list[str], ndcg_n: int) -> dict:
    """指定指標をまとめて算出（同一 T のアクティブ集合に対して）。"""
    out = {}
    for m in metric_names:
        if m == "ndcg":
            out[m] = ndcg_at(y_true, y_pred, ndcg_n)
        else:
            out[m] = _FUNCS[m](y_true, y_pred)
    return out
