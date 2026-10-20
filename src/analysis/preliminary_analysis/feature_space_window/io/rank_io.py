"""順位の集計の保存（design.md §5.1）。lookback_window/io/rank_io.py の複製。

`summary/rank_<metric>.csv` を 1 ファイルで保存し、冒頭に対象日数のコメント行を付ける（Excel 用に BOM 付き）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.preliminary_analysis.feature_space_window.utils import constants


def save_ranking(df: pd.DataFrame, ndays: dict[str, int], project: str, metric: str) -> Path:
    out_dir = constants.OUTPUT_ROOT / project / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rank_{metric}.csv"
    meta = "# 対象日数: " + ", ".join(f"{k}={v}" for k, v in ndays.items())
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(meta + "\n")
        f.write(df.to_csv(index_label="size"))
    return path
