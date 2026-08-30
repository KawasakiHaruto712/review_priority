"""順位集計表の保存（design.md §11.3）。

`rank_<metric>.csv` を 1 ファイルで保存する。冒頭に対象日数のコメント行を付ける
（版ごとに一定の値で窓に依存しないため、表本体には入れずコメントにする）。
出力先: `<OUTPUT_ROOT>/<project>/summary/`（§11.1 の summary_<metric>.csv と並ぶ）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.preliminary_analysis.lookback_window.utils import constants


def save_ranking(df: pd.DataFrame, ndays: dict[str, int], project: str, metric: str) -> Path:
    """順位集計表を CSV で保存し、パスを返す。"""
    out_dir = constants.OUTPUT_ROOT / project / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rank_{metric}.csv"

    meta = "# 対象日数: " + ", ".join(f"{k}={v}" for k, v in ndays.items())
    csv_body = df.to_csv(index_label="window")
    # Excel が UTF-8 を誤認して文字化けしないよう BOM 付き（utf-8-sig）で保存
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(meta + "\n")
        f.write(csv_body)
    return path
