"""生予測・指標テーブルの保存/読込（design.md §8）。2 段保存で後から再計算なしに指標追加・描画選択。

- predictions.csv.gz : Change ごと (day, window, seed, change_id, y_true, score)
- metrics.csv        : (day, window, seed, カウント, missing, missing_reason, 各指標...)
出力先: <OUTPUT_ROOT>/<project>/<version>/
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.preliminary_analysis.lookback_window.utils import constants

_PRED_COLUMNS = ["day", "window", "seed", "change_id", "y_true", "score"]


def version_dir(project: str, version: str) -> Path:
    return constants.OUTPUT_ROOT / project / version


def save_predictions(rows: list[tuple], project: str, version: str) -> Path | None:
    """生予測を predictions.csv.gz に保存。"""
    d = version_dir(project, version)
    d.mkdir(parents=True, exist_ok=True)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=_PRED_COLUMNS)
    path = d / "predictions.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    with open(d / "predictions_meta.json", "w", encoding="utf-8") as f:
        json.dump({"project": project, "version": version, "columns": _PRED_COLUMNS,
                   "n_rows": len(df),
                   "note": "y_true=Δ以内レビューの2値, score=正例確率。window='general'は汎用ヘッド。"
                           "再学習なしで指標を計算し直せる。"},
                  f, ensure_ascii=False, indent=2)
    return path


def save_metrics(metric_rows: list[dict], project: str, version: str) -> Path:
    """指標テーブルを metrics.csv に保存。"""
    d = version_dir(project, version)
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metric_rows)
    path = d / "metrics.csv"
    df.to_csv(path, index=False)
    return path


def load_predictions(project: str, version: str) -> pd.DataFrame:
    return pd.read_csv(version_dir(project, version) / "predictions.csv.gz", compression="gzip")


def load_metrics(project: str, version: str) -> pd.DataFrame:
    return pd.read_csv(version_dir(project, version) / "metrics.csv")
