"""変化区間の判定（距離固定の位置比較＋並べ替え検定。§7）。

このファイルは pytest のテストではなく、変化区間の「検定＝統計的判定」を行う本体モジュール。

重要: 並べ替え検定は **計算済み行列（セル値）を並べ替えて統計量を測り直すだけ**で、モデルの再学習はしない。
`PERMUTATION_N` は「数値シャッフルの回数」であってモデル学習の回数ではない（§7.2）。

統計量の具体式は実装時に決める方針（§5.9）。本実装は第1版として「距離を固定した位置系列に対する
チェンジポイント統計量（最も大きい前後平均差）」を採用する。MAE/RMSE/NDCG で良い向きが違っても、
絶対差なので向きに依らず変化の大きさを測れる。後で差し替え可能。
"""
from __future__ import annotations

import numpy as np

from src.analysis.preliminary_analysis.concept_drift_existence.evaluation.drift_matrix import MatrixResult


def positions_at_distance(value: np.ndarray, d: int) -> list[tuple[int, float]]:
    """距離 d を固定し、位置 p ごとの (p, score) を取り出す（NaN は除外）。§7.1。"""
    row = value[d - 1]
    return [(p, float(row[p])) for p in range(len(row)) if not np.isnan(row[p])]


def _changepoint_stat(scores: np.ndarray) -> tuple[float, int | None]:
    """位置系列の最大「前後平均差」と、その分割位置インデックスを返す。"""
    n = len(scores)
    best, best_k = 0.0, None
    for k in range(1, n):
        gap = abs(float(np.mean(scores[:k])) - float(np.mean(scores[k:])))
        if gap > best:
            best, best_k = gap, k
    return best, best_k


def detect_drift(result: MatrixResult, n_perm: int, alpha: float, seed: int = 0) -> dict:
    """距離ごとに位置系列のチェンジポイント統計量を並べ替え検定し、変化区間の有無を返す。

    返り値: {drift_exists, min_p_value, alpha, by_distance: {d: {p_value, statistic, change_position, n_positions}}}
    """
    rng = np.random.default_rng(seed)
    value = result.value
    by_distance: dict[int, dict] = {}
    min_p = 1.0
    drift = False

    for d in range(1, result.bin_count + 1):
        pts = positions_at_distance(value, d)
        if len(pts) < 3:
            continue  # 位置が少なすぎて横断比較できない
        positions = [p for p, _ in pts]
        scores = np.array([s for _, s in pts], dtype=float)
        obs, k = _changepoint_stat(scores)
        # 並べ替え（位置ラベルのシャッフル＝scores の並べ替え）。再学習なし。
        cnt = sum(1 for _ in range(n_perm)
                  if _changepoint_stat(rng.permutation(scores))[0] >= obs)
        p_value = (1 + cnt) / (n_perm + 1)
        # 報告する位置は 1 始まり（図/CSV と一致。positions は内部 0 始まりの列 index なので +1）
        change_position = positions[k] + 1 if k is not None else None
        by_distance[d] = {"p_value": p_value, "statistic": obs,
                          "change_position": change_position, "n_positions": len(pts)}
        min_p = min(min_p, p_value)
        if p_value < alpha:
            drift = True

    return {"drift_exists": drift, "min_p_value": min_p, "alpha": alpha,
            "by_distance": by_distance}
