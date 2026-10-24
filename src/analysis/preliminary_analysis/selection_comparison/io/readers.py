"""ほかの分析の出力を読む（design.md §1）。読むだけで、書き込みはしない。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.selection_comparison.utils import constants


def versions(project: str) -> list[str]:
    """その project の評価する版（古い順）。"""
    meta = json.loads(constants.versions_path(project).read_text(encoding="utf-8"))
    v = meta["versions"]
    return list(v) if isinstance(v, dict) else list(v)


def daily_values(path) -> pd.DataFrame:
    """daily_metrics.csv.gz → 評価日 × 行 ごとの F1 値（5 seed の中央値。欠測・NaN を除く。精度表と同じ決まり。design.md §2）。
    列：day・position（保存物の向き）・distance（int）・value。汎用ヘッドの行は除く。"""
    d = pd.read_csv(path, low_memory=False)
    d = d[d["distance"].astype(str) != "general"]
    if "missing" in d.columns:
        d = d[d["missing"] == 0]
    d = d.assign(distance=d["distance"].astype(int))
    out = d.groupby(["day", "position", "distance"])[constants.METRIC].median().rename("value").reset_index()
    return out.dropna(subset=["value"])


def matrix_values(kind: str, project: str, version: str) -> tuple[np.ndarray, np.ndarray | None]:
    """リリースごとの精度表の値 [行 d, 列 p（保存物の向き）] と評価日数。"""
    m = json.loads(constants.matrix_path(kind, project, version).read_text(encoding="utf-8"))
    value = np.array(m["value"], dtype=float)
    n_days = np.array(m["n_days"], dtype=float) if m.get("n_days") is not None else None
    return value, n_days


def drift_cells(kind: str, project: str, version: str) -> pd.DataFrame:
    """特徴量別の分布の cells.csv（行・列・特徴量・群ごとの δ の平均など）。"""
    return pd.read_csv(constants.drift_dir(kind, project, version) / "cells.csv")


def drift_summary(kind: str, project: str, version: str) -> pd.DataFrame:
    """特徴量別の分布の cells_summary.csv（行・列・群ごとの |δ| の平均・|δ| ≥ 0.147 の数）。"""
    return pd.read_csv(constants.drift_dir(kind, project, version) / "cells_summary.csv")
