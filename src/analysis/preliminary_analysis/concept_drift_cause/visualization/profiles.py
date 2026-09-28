"""作図に渡すデータの組み立て（design.md §9.4.1, §9.4.5, §9.4.7）。

`ig_daily.csv.gz` と距離×時期行列の `daily_metrics.csv.gz` の 2 つだけを読み、
セルごとの「15 次元の平均と標準偏差」と、タイトルに載せる精度・規模を返す。
**抽出（`--mode extract`）の出力には依存しない**（§9.4.9）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.concept_drift_cause.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.features.feature_builder import FEATURE_NAMES

logger = logging.getLogger(__name__)

# 汎用ヘッドの行は距離を持たない（`distance` にこの値が入る。§9.1）
GENERAL_TAG = "general"


def load_daily(project: str, version: str) -> pd.DataFrame | None:
    """`ig_daily.csv.gz` を読む。無ければ None。"""
    path = (constants.OUTPUT_ROOT / project / constants.MODEL_NAME / version
            / constants.IG_DAILY_NAME)
    if not path.exists():
        logger.warning(f"[{project} {version}] {path} がありません（--mode compute が未実行）")
        return None
    return pd.read_csv(path)


def load_metrics(project: str, version: str) -> pd.DataFrame | None:
    """距離×時期行列の `daily_metrics.csv.gz` を読む（精度・件数の出どころ）。"""
    path = (constants.DETECTION_ROOT / project / constants.MODEL_NAME / version
            / "daily_metrics.csv.gz")
    if not path.exists():
        logger.warning(f"[{project} {version}] {path} がありません"
                       f"（距離×時期行列の分析が未実行）")
        return None
    return pd.read_csv(path)


def build_series(daily: pd.DataFrame, series: str, group: str) -> pd.DataFrame | None:
    """1 系列ぶんの「評価日 × 距離」の 15 次元プロファイルを返す。

    series が "diff" のときは、**同じ `(評価日, seed)` の probe と汎用ヘッドを引いた**ものを返す
    （§9.4.1）。翻訳表（encoder とその日のデータで決まる部分）が相殺され、プローブ重みの
    違いだけが残る。
    """
    g = daily[daily["group"] == group]
    if series == "probe":
        out = g[g["head"] == "probe"].copy()
    elif series == "general":
        out = g[g["head"] == "general"].copy()
    elif series == "diff":
        p = g[g["head"] == "probe"].copy()
        gen = g[g["head"] == "general"][["day", "seed"] + FEATURE_NAMES]
        out = p.merge(gen, on=["day", "seed"], suffixes=("", "_g"), how="inner")
        missing = len(p) - len(out)
        if missing:
            logger.warning(f"同じ (評価日, seed) の汎用ヘッドが無い行が {missing} 件あり除外しました")
        for f in FEATURE_NAMES:
            out[f] = out[f] - out[f + "_g"]
    else:
        raise ValueError(f"未知の系列: {series}")
    if out.empty:
        return None
    return out


def cell_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """セル（距離 × 位置）ごとに 15 次元の平均と標準偏差を出す（§8 の集約と同じ）。

    返す列: distance, position, n_days, <特徴量>, <特徴量>_std
    """
    if "position" not in rows.columns:
        raise ValueError("position 列がありません")
    keys = ["distance", "position"]
    agg = {f: ["mean", "std"] for f in FEATURE_NAMES}
    agg["day"] = "nunique"
    t = rows.groupby(keys, dropna=False).agg(agg)
    t.columns = [f"{a}_std" if b == "std" else ("n_days" if a == "day" else a)
                 for a, b in t.columns]
    return t.reset_index()


def cell_info(metrics: pd.DataFrame, rows: pd.DataFrame, group: str) -> pd.DataFrame:
    """タイトルに載せる精度・規模をセルごとに出す（§9.4.7）。

    AUC は **probe の実測値**（差分ではない）。距離×時期行列のヒートマップと同じ値になるよう、
    IG を計算した `(評価日, 距離, seed)` に対応する行だけを拾って平均する。
    """
    key = ["day", "distance", "seed"]
    m = metrics.copy()
    m["distance"] = m["distance"].astype(str)
    r = rows[key + ["position"]].copy()
    r["distance"] = r["distance"].astype(str)
    j = r.merge(m[key + ["auc", "n_pos_eval", "n_neg_eval"]], on=key, how="left")
    n_col = "n_pos_eval" if group == "pos" else "n_neg_eval"
    out = (j.groupby(["distance", "position"], dropna=False)
             .agg(auc=("auc", "mean"), n_changes=(n_col, "mean"))
             .reset_index())
    return out


def axis_limit(stats_by_cell: list[pd.DataFrame]) -> float:
    """軸の端（§9.4.5）。**「|平均| ＋ 標準偏差」のパーセンタイル**で決める。

    平均だけで決めるとエラーバーが枠外に出る（実測で 2〜3 倍）。最大値に合わせると
    外れ値 1 つで全体が潰れる。箱ひげ図の Q3+1.5IQR は、値の大半がゼロ近辺に集中する
    この分布では極端に小さくなるため使わない。
    """
    vals = []
    for t in stats_by_cell:
        if t is None or t.empty:
            continue
        mean = t[FEATURE_NAMES].abs()
        std = t[[f + "_std" for f in FEATURE_NAMES]].fillna(0.0)
        std.columns = FEATURE_NAMES
        vals.append((mean + std).values.ravel())
    if not vals:
        return 1.0
    a = np.concatenate(vals)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    lim = float(np.percentile(a, constants.PLOT_AXIS_PERCENTILE))
    return lim if lim > 0 else 1.0
