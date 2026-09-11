"""結果の csv / json 出力（§5.10, §8）。

検定結果（p値・有意判定・変化点）は drift_test.json のみに出す（png には描かない。§5.10）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.concept_drift_detection.evaluation.drift_matrix import MatrixResult


def _value_df(value: np.ndarray) -> pd.DataFrame:
    """四角行列を DataFrame に（行=距離 d、列=位置 p）。

    距離は **0 始まり（d0..）**＝ d0 が step1 の fresh 窓。位置は 1 始まり（p1..）で図と一致。
    """
    n = value.shape[0]
    return pd.DataFrame(value, index=[f"d{d}" for d in range(n)],
                        columns=[f"p{p}" for p in range(1, value.shape[1] + 1)])


def write_matrix(result: MatrixResult, meta: dict, out_dir: Path) -> None:
    """drift_matrix.csv / drift_matrix.json / position_by_distance.csv を書く。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 行列（セル値＝反復中央値）
    _value_df(result.value).to_csv(out_dir / "drift_matrix.csv")

    # json: 行列＋ばらつき＋各反復の生スコア＋メタ（位置キーは 1 始まり p1..）
    per_repeat = {f"d{d}_p{p + 1}": v for (d, p), v in result.per_repeat.items()}
    payload = {
        "meta": {**meta, "metric": result.metric, "bin_count": result.bin_count},
        "value": _value_df(result.value).where(pd.notna(result.value), None).values.tolist(),
        "iqr": _value_df(result.iqr).where(pd.notna(result.iqr), None).values.tolist(),
    }
    if meta.get("save_per_repeat", True):
        payload["per_repeat"] = per_repeat
    with open(out_dir / "drift_matrix.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 距離固定の位置別スコア（long 形式）。distance は 0 始まり、position は 1 始まり（図と一致）。
    rows = []
    for d in range(result.bin_count):
        for p in range(result.bin_count):
            v = result.value[d, p]
            rows.append({"distance": d, "position": p + 1,
                         "value": None if np.isnan(v) else float(v)})
    pd.DataFrame(rows).to_csv(out_dir / "position_by_distance.csv", index=False)


def _to_array(grid) -> np.ndarray:
    """None を NaN に直して 2 次元 float 配列にする。"""
    return np.array([[np.nan if v is None else v for v in row] for row in grid], dtype=float)


def load_matrix(json_path: Path) -> MatrixResult:
    """保存済み drift_matrix.json から MatrixResult を復元する（図の再描画用。再計算しない）。"""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    meta = payload["meta"]
    value = _to_array(payload["value"])
    iqr = _to_array(payload["iqr"]) if payload.get("iqr") else np.full_like(value, np.nan)
    return MatrixResult(meta["metric"], int(meta["bin_count"]), value, iqr)


def write_drift_test(drift_result: dict, out_dir: Path) -> None:
    """検定結果を drift_test.json に書く（png には出さない）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "drift_test.json", "w", encoding="utf-8") as f:
        json.dump(drift_result, f, ensure_ascii=False, indent=2)


def write_daily_metrics(rows: list[dict], meta: dict, out_dir: Path) -> Path | None:
    """(評価日, 距離, seed) ごとの指標を daily_metrics.csv.gz に保存する（§10）。

    rows: drift_matrix の daily_sink（`_daily_row` の dict 列）。1 版で約 4.7 万行。
    これがあれば **位置（列）のまとめ方を後から変えて再集計できる**（モデル再実行不要）。
    `distance="general"` の行は汎用ヘッド（距離に非依存）。

    注：**Change 1 件ごとの予測は保存しない**（26 距離 × 約 180 日 × 10 seed × 1 日約 260 件
    ≒ 1,200 万行/版となり非現実的。§10）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    path = out_dir / "daily_metrics.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    with open(out_dir / "daily_metrics_meta.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "columns": list(df.columns), "n_rows": len(df),
                   "note": "評価日ごと（Δ=1）の指標。distance は 0 始まり（d0=step1 の fresh 窓）、"
                           "distance='general' は汎用ヘッド。position は 1 始まり。"
                           "位置のまとめ方を変えて再集計できる。"},
                  f, ensure_ascii=False, indent=2)
    return path


def load_daily_metrics(path: Path) -> "pd.DataFrame":
    """保存済み daily_metrics.csv.gz を DataFrame で読み込む（位置の再集計用）。"""
    return pd.read_csv(path, compression="gzip")


def write_summary(per_version: list[dict], meta: dict, out_dir: Path) -> None:
    """リリース横断の本数集計（N 本中 k 本で変化区間あり。§7.3）を書く。

    per_version: [{"version":..., "drift_exists":bool, "min_p_value":float}, ...]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(per_version)
    k = sum(1 for r in per_version if r["drift_exists"])
    df = pd.DataFrame(per_version)
    df.to_csv(out_dir / "drift_count.csv", index=False)
    summary = {**meta, "n_releases": n, "n_drift": k,
               "drift_ratio": (k / n if n else None), "per_version": per_version}
    with open(out_dir / "drift_count.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
