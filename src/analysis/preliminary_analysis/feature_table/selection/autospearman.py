"""順位相関と AutoSpearman の 1 段目（design.md §3.2）。

AutoSpearman：Jiarpakdee・Tantithamthavorn・Treude 2018（ICSME。arXiv 1806.09791）の Algorithm 1 の Line 1〜10 と、
著者の R パッケージ Rnalytica の get.automated.spearman。2 段目の VIF は使わない（§3.1）。
6 件のリポジトリをまとめるため、相関の強い組の判定と「組の外との |ρ| の平均」を、リポジトリごとの値から作る（本人判断）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def spearman_matrix(x: np.ndarray) -> np.ndarray:
    """列ごとの Spearman の順位相関行列（各列を同順位は平均の順位にし、ピアソンの相関を取る）。

    値が一定の列（標準偏差 0）は、その列の行と列を NaN にする（ρ が出ない。§3.2 の 3）。
    """
    n, p = x.shape
    r = np.empty((n, p), dtype=np.float64)
    for j in range(p):
        r[:, j] = rankdata(x[:, j], method="average")
    r -= r.mean(axis=0)
    norm = np.sqrt(np.einsum("ij,ij->j", r, r))
    cov = r.T @ r
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = cov / np.outer(norm, norm)
    const = norm == 0
    rho[const, :] = np.nan
    rho[:, const] = np.nan
    np.fill_diagonal(rho, np.where(const, np.nan, 1.0))
    return np.clip(rho, -1.0, 1.0)


@dataclass
class Aggregate:
    """6 件の |ρ| をまとめたもの。abs_rho は (リポジトリ数, p, p)。"""
    names: list[str]
    abs_rho: np.ndarray
    threshold: float

    @property
    def count_ge(self) -> np.ndarray:
        """|ρ| ≥ 閾値のリポジトリの数（ρ が出ないリポジトリは「閾値未満」に数える）。"""
        with np.errstate(invalid="ignore"):
            return np.sum(np.nan_to_num(self.abs_rho, nan=0.0) >= self.threshold, axis=0)

    @property
    def count_defined(self) -> np.ndarray:
        """ρ が出るリポジトリの数（割合の決まりの分母。2026-10-20 追加）。"""
        return np.sum(~np.isnan(self.abs_rho), axis=0)

    @property
    def mean_abs(self) -> np.ndarray:
        """|ρ| の平均（ρ が出ないリポジトリを除く。全部出なければ NaN）。"""
        return _nanmean(self.abs_rho, axis=0)

    def other_mean(self, i: int, j: int) -> float:
        """特徴量 i の「組の外の特徴量との |ρ| の平均」。組 (i, j) 以外の全特徴量（すでに落としたものも含む）との
        |ρ| の平均をリポジトリごとに出し、それを 6 件で平均する（ρ が出ない値は除く）。"""
        mask = np.ones(len(self.names), dtype=bool)
        mask[[i, j]] = False
        per_repo = _nanmean(self.abs_rho[:, i, mask], axis=1)
        return float(_nanmean(per_repo, axis=0))

    def strong_pairs(self, rule: str, ratio: tuple[int, int] = (2, 3)) -> list[tuple[int, int]]:
        """相関が強い組（i < j）。rule は "majority<k>"（k 件以上）か "mean"（|ρ| の平均 ≥ 閾値）か
        "ratio"（|ρ| ≥ 閾値のリポジトリが、ρ が出るリポジトリの ratio 以上。ρ が 1 件も出なければ強くない。2026-10-20 追加）。
        並びは |ρ| の平均の大きい順、同じなら特徴量の並び順（§3.2 の 6）。"""
        if rule == "mean":
            with np.errstate(invalid="ignore"):
                hit = np.nan_to_num(self.mean_abs, nan=0.0) >= self.threshold
        elif rule == "ratio":
            num, den = ratio              # 小数の誤差を避けて整数で比べる：den × 件数 ≥ num × 出る数
            defined = self.count_defined
            hit = (defined > 0) & (den * self.count_ge >= num * defined)
        elif rule.startswith("majority"):
            hit = self.count_ge >= int(rule[len("majority"):])
        else:
            raise ValueError(f"知らない決まり: {rule}")
        mean = np.nan_to_num(self.mean_abs, nan=0.0)
        p = len(self.names)
        pairs = [(i, j) for i in range(p) for j in range(i + 1, p) if hit[i, j]]
        pairs.sort(key=lambda ij: (-mean[ij], ij[0], ij[1]))
        return pairs


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    """NaN を除いた平均。全部 NaN のところは NaN（警告を出さない）。"""
    valid = ~np.isnan(a)
    n = valid.sum(axis=axis)
    s = np.where(valid, a, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def aggregate(names: list[str], rhos: list[np.ndarray], threshold: float) -> Aggregate:
    return Aggregate(names, np.abs(np.stack(rhos)), threshold)


def auto_spearman(agg: Aggregate, rule: str, ratio: tuple[int, int] = (2, 3)) -> tuple[list[str], list[dict]]:
    """AutoSpearman の 1 段目。残す特徴量（元の並び）と、手順の記録を返す（§3.2 の 6・§3.3）。

    強い組から順に、組の外との |ρ| の平均が小さい方を残し、大きい方を落とす。同じなら組の先（特徴量の並びで前）を残す。
    落とした特徴量を含む組は、それ以降は見ない。
    """
    dropped: set[int] = set()
    steps: list[dict] = []
    mean = agg.mean_abs
    for i, j in agg.strong_pairs(rule, ratio):
        if i in dropped or j in dropped:
            continue
        mi, mj = agg.other_mean(i, j), agg.other_mean(j, i)
        mi_c = np.inf if np.isnan(mi) else mi
        mj_c = np.inf if np.isnan(mj) else mj
        keep, drop = (i, j) if mi_c <= mj_c else (j, i)
        dropped.add(drop)
        steps.append({"step": len(steps) + 1, "feature_a": agg.names[i], "feature_b": agg.names[j],
                      "mean_abs_rho": float(mean[i, j]), "repos_ge_threshold": int(agg.count_ge[i, j]),
                      "repos_defined": int(agg.count_defined[i, j]),
                      "other_mean_a": mi, "other_mean_b": mj,
                      "kept": agg.names[keep], "dropped": agg.names[drop]})
    kept = [n for k, n in enumerate(agg.names) if k not in dropped]
    return kept, steps
