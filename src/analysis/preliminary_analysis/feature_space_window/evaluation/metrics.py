"""評価指標（lookback_window/design.md §7 の複製。feature_space_window/design.md §1・§4。各ディレクトリ自己完結のため重複を許す）。全部まとめて算出し、後から描画で選ぶ。

1 評価日（1 クエリ）分の (y_true, score) から次を返す：
- auc          : ROC-AUC（両クラス必要）
- map          : Average Precision（正例が1つでも可）
- precision/recall/f1 : しきい値 0.5
- precision@k / recall@k / f1@k : k=5,10,20 と その日の正例数（'@pos'）
- norm_rank    : 実際にレビューされた Change の正規化順位の平均（0=最上位, 小さいほど良い）
- mrr          : 最上位の正例の逆順位（1/rank, 1-indexed）
- precision_plugin/recall_plugin/f1_plugin（＋k_plugin・thr_plugin）: plug-in の判定（2026-10-22。pretrained_encoders/design.md §15.4）
計算不能は NaN。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

NAN = float("nan")


def _topk(y_sorted: np.ndarray, k: int, n_pos: int) -> tuple[float, float, float]:
    """スコア降順に並べた y_sorted の上位 k での precision/recall/f1。"""
    n = len(y_sorted)
    if k is None or k <= 0 or n_pos <= 0:
        return NAN, NAN, NAN
    kk = min(k, n)
    hits = float(y_sorted[:kk].sum())
    prec = hits / kk
    rec = hits / n_pos
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _plugin(y_sorted: np.ndarray, p_sorted: np.ndarray, n_pos: int) -> dict:
    """plug-in の判定（pretrained_encoders/design.md §15.4。2026-10-22。lookback_window と同じもの）。

    確率の大きい順に並べ、2 × 上位 k 件の確率の和 ÷ (k ＋ 全件の確率の和) が最大の k（k ≥ 1）を選び、上位 k 件を正例と判定する
    （F1 の期待値の近似。Lewis 1995 式 (17)、Dembczyński ら 2017 §4.1）。正解ラベルは k を選ぶのに使わない。
    """
    n = len(p_sorted)
    if n == 0:
        return {k: NAN for k in PLUGIN_COLUMNS}
    s = np.cumsum(p_sorted)
    g = 2.0 * s / (np.arange(1, n + 1) + s[-1])
    k = int(np.argmax(g)) + 1
    hits = float(y_sorted[:k].sum())
    return {"precision_plugin": hits / k,
            "recall_plugin": hits / n_pos if n_pos > 0 else NAN,
            "f1_plugin": 2.0 * hits / (k + n_pos) if (k + n_pos) > 0 else NAN,
            "k_plugin": float(k),
            "thr_plugin": float(p_sorted[k - 1])}


PLUGIN_COLUMNS = ["precision_plugin", "recall_plugin", "f1_plugin", "k_plugin", "thr_plugin"]


def compute_day_metrics(y_true, score, k_list, threshold: float = 0.5) -> dict:
    """1 評価日の全指標を {指標名: 値} で返す。"""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(score, dtype=float)
    n = len(yt)
    n_pos = int(yt.sum())
    n_neg = n - n_pos
    both = n_pos > 0 and n_neg > 0
    yhat = (yp >= threshold).astype(int)

    out: dict[str, float] = {}
    out["auc"] = float(roc_auc_score(yt, yp)) if both else NAN
    out["map"] = float(average_precision_score(yt, yp)) if n_pos > 0 else NAN
    out["precision"] = float(precision_score(yt, yhat, zero_division=0))
    out["recall"] = float(recall_score(yt, yhat, zero_division=0))
    out["f1"] = float(f1_score(yt, yhat, zero_division=0))

    # スコア降順（top-k・順位系で共有）
    order = np.argsort(-yp, kind="stable")
    y_sorted = yt[order]
    for k, label in [(k, str(k)) for k in k_list] + [(n_pos, "pos")]:
        prec, rec, f1 = _topk(y_sorted, k, n_pos)
        out[f"precision@{label}"] = prec
        out[f"recall@{label}"] = rec
        out[f"f1@{label}"] = f1
    out.update(_plugin(y_sorted, yp[order], n_pos))     # plug-in の F1（2026-10-22）

    # 正規化順位・MRR（正例が1つ以上・要素2つ以上）
    if n_pos > 0 and n > 1:
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)         # 0-indexed rank（0=最上位）
        pos_ranks = ranks[yt == 1]
        out["norm_rank"] = float(np.mean(pos_ranks / (n - 1)))
        out["mrr"] = float(np.max(1.0 / (pos_ranks + 1)))  # 最上位正例の逆順位
    else:
        out["norm_rank"] = NAN
        out["mrr"] = NAN
    return out


def metric_columns(k_list) -> list[str]:
    """metrics テーブルに載る指標カラム名の一覧（固定順）。"""
    cols = ["auc", "map", "precision", "recall", "f1"]
    for label in [str(k) for k in k_list] + ["pos"]:
        cols += [f"precision@{label}", f"recall@{label}", f"f1@{label}"]
    cols += ["norm_rank", "mrr"]
    cols += PLUGIN_COLUMNS                              # plug-in の F1（2026-10-22。後ろに足し、今までの列の並びは変えない）
    return cols
