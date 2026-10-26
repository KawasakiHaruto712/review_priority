"""ほかの分析の出力を読む（design.md §1・§2）。読むだけで、書き込みはしない。"""
from __future__ import annotations

import json

import pandas as pd

from src.analysis.preliminary_analysis.release_best_size.utils import constants


def versions(project: str) -> list[str]:
    """その project の評価する版（古い順）。"""
    v = json.loads(constants.versions_path(project).read_text(encoding="utf-8"))["versions"]
    return list(v)


def size_metrics(method: str, project: str, version: str) -> pd.DataFrame:
    """metrics.csv → 列 day・size・seed・missing・value（F1 値）。汎用ヘッド（general）の行は除く。"""
    col = constants.SIZE_COLUMN[method]
    d = pd.read_csv(constants.metrics_path(method, project, version),
                    usecols=["day", col, "seed", "missing", constants.METRIC])
    d = d.rename(columns={col: "size", constants.METRIC: "value"})
    return d[d["size"].isin(constants.SIZES)].reset_index(drop=True)


def day_positions(method: str, project: str, version: str) -> pd.Series:
    """精度表の出力に保存された day → position（保存物の向き）。1 日に 2 つの位置があれば止める。"""
    d = pd.read_csv(constants.daily_path(method, project, version), usecols=["day", "position"])
    n = d.groupby("day")["position"].nunique()
    if (n > 1).any():
        raise ValueError(f"{project} {version}：1 日に複数の評価時期があります（{list(n[n > 1].index[:3])}）")
    return d.drop_duplicates("day").set_index("day")["position"].astype(int).sort_index()


def selected_sizes(method: str) -> dict[str, str]:
    """{リポジトリ -> 4.1 で選んだ N のラベル}。"""
    sel = json.loads(constants.selected_path(method).read_text(encoding="utf-8"))
    return {p: v["label"] for p, v in sel.items()}


def summary_table(method: str, project: str) -> pd.DataFrame:
    """4.1 の要約表（index ＝ N のラベル）。"""
    return pd.read_csv(constants.summary_path(method, project), index_col=0)
