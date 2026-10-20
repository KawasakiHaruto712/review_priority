"""評価日ごとの比較（design.md §3.3）：特徴量ごとにマン–ホイットニーの U 検定とクリフの δ。

δ ＝ P(評価側 ＞ 学習側) − P(評価側 ＜ 学習側) ＝ 2U ÷ (n₁ × n₂) − 1（U は評価側の U。同順位は半分ずつ数える）。
値が一定の特徴量（評価側と学習側を合わせて 1 種類の値）は検定できないので δ ＝ 0・p は NaN（design.md §2）。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def compare(eval_x: np.ndarray, train_x: np.ndarray) -> dict[str, np.ndarray]:
    """eval_x (n₁, F) と train_x (n₂, F) を列（特徴量）ごとに比べる。各値は (F,) の配列。

    返り値のキー：n_eval, n_train, u, p, delta, median_eval, median_train, constant
    どちらかが 0 件なら、件数以外は NaN。
    """
    eval_x = np.asarray(eval_x, dtype=np.float64)
    train_x = np.asarray(train_x, dtype=np.float64)
    f = eval_x.shape[1]
    n1, n2 = eval_x.shape[0], train_x.shape[0]
    out = {"n_eval": np.full(f, n1), "n_train": np.full(f, n2)}
    nan = np.full(f, np.nan)
    if n1 == 0 or n2 == 0:
        out.update(u=nan.copy(), p=nan.copy(), delta=nan.copy(), median_eval=nan.copy(), median_train=nan.copy(),
                   constant=np.zeros(f, dtype=bool))
        return out
    both = np.vstack([eval_x, train_x])
    constant = both.min(axis=0) == both.max(axis=0)
    u = np.full(f, n1 * n2 / 2.0)
    p = nan.copy()
    live = ~constant
    if live.any():
        res = mannwhitneyu(eval_x[:, live], train_x[:, live], alternative="two-sided", axis=0, method="auto")
        u[live] = res.statistic
        p[live] = res.pvalue
    delta = 2.0 * u / (n1 * n2) - 1.0
    delta[constant] = 0.0
    out.update(u=u, p=p, delta=delta, median_eval=np.median(eval_x, axis=0), median_train=np.median(train_x, axis=0),
               constant=constant)
    return out
