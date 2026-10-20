"""日の代表・標準化・距離・近い k 日を選ぶ（design.md §3）。

- 日の代表：その日の集合の埋め込み（凍結エンコーダの出力）の平均ベクトル。ラベルは使わない。
- 近さ：各次元を標準化したユークリッド距離。標準化の標準偏差は、評価日 X ごとに候補（X より前の全日）の日の代表から出し直す
  （累積の和で出す）。標準偏差が STD_EPS 未満の次元は 1 とみなす（事前学習の Scaler と同じ決まり）。
  平均は距離の差で打ち消し合うので、標準偏差だけが効く。
- 並び：距離の小さい順。同じ距離なら評価日に時間で近い方を先にする。
"""
from __future__ import annotations

from datetime import date

import numpy as np

from src.analysis.preliminary_analysis.feature_space_window.utils import constants


def day_means(embeds: list[np.ndarray]) -> np.ndarray:
    """集合ごとの埋め込み (L, d) → 日の代表 (日数, d)（float64）。"""
    return np.stack([np.asarray(e, dtype=np.float64).mean(axis=0) for e in embeds])


def rank_candidates(cand_means: np.ndarray, cand_days_before: np.ndarray, x_mean: np.ndarray,
                    std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """候補を近い順に並べた (候補の位置, 距離) を返す。

    cand_means       : (n, d) 候補の日の代表
    cand_days_before : (n,) 評価日の何日前か（同じ距離のときに小さい方を先にする）
    x_mean           : (d,) 評価日の代表
    std              : (d,) 標準化に使う標準偏差（STD_EPS 未満はすでに 1 にしたもの）
    """
    if len(cand_means) == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    dist = np.sqrt((((cand_means - x_mean) / std) ** 2).sum(axis=1))
    order = np.lexsort((cand_days_before, dist))     # 主キー：距離、次：評価日に近い順
    return order, dist[order]


class DayIndex:
    """全日の代表を持ち、評価日ごとに候補の標準偏差と近い日を出す（1 seed 分）。"""

    def __init__(self, dates: list[date], means: np.ndarray):
        if list(dates) != sorted(dates):
            raise ValueError("dates は昇順で渡してください")
        self.dates = list(dates)
        self.ord = np.array([d.toordinal() for d in self.dates])
        self.means = np.asarray(means, dtype=np.float64)
        zero = np.zeros((1, self.means.shape[1]))
        self._c1 = np.vstack([zero, np.cumsum(self.means, axis=0)])
        self._c2 = np.vstack([zero, np.cumsum(self.means ** 2, axis=0)])

    def n_before(self, x: date) -> int:
        """x より前の日の数（候補の数）。"""
        return int(np.searchsorted(self.ord, x.toordinal(), side="left"))

    def std_before(self, n: int) -> np.ndarray:
        """先頭 n 日（候補）の日の代表の標準偏差（母標準偏差。Scaler と同じ ddof=0）。STD_EPS 未満は 1。"""
        if n <= 0:
            return np.ones(self.means.shape[1])
        m = self._c1[n] / n
        var = np.maximum(self._c2[n] / n - m ** 2, 0.0)
        sd = np.sqrt(var)
        return np.where(sd < constants.STD_EPS, 1.0, sd)

    def nearest(self, x: date, x_mean: np.ndarray, max_k: int) -> tuple[list[date], np.ndarray, np.ndarray]:
        """x に近い順に最大 max_k 日：(日付, 距離, 評価日の何日前か)。候補は x より前の全日。"""
        n = self.n_before(x)
        days_before = x.toordinal() - self.ord[:n]
        order, dist = rank_candidates(self.means[:n], days_before, np.asarray(x_mean, dtype=np.float64),
                                      self.std_before(n))
        order, dist = order[:max_k], dist[:max_k]
        return [self.dates[i] for i in order], dist, days_before[order]
