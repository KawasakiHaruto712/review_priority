"""保存済みの IG 日次結果から表を組み立てる（design.md §8, §9.3）。

計算（重い）と抽出（軽い）を分けているので、ここはモデルを一切動かさない。
`ig_daily.csv.gz` と距離×時期行列の分析の `daily_metrics.csv.gz` を読むだけ。

集約（§8）:
    ① (評価日, 距離) ごとに 1 モデルのプロファイル ← ig_daily にそのまま入っている
    ② セル内の日で平均      → セルの値
       セル内の日で標準偏差  → std（**日間のばらつき**）
       有効日数              → 列に出す
精度（§7）は距離×時期行列の分析の daily_metrics から、**距離×時期行列の分析と同じ集約順**（日ごとに seed 中央値 → 日で平均）で
付ける。特徴量の寄与度分析が選ぶモデルは各日の中央モデルそのものなので、距離×時期行列の分析のセル値と一致する。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.concept_drift_cause.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.model.set_transformer import FEATURE_NAMES

logger = logging.getLogger(__name__)

ACC_COLS = ["auc"] + [f"recall@{k}" for k in constants.K_LIST]


def accuracy_by_cell(daily: pd.DataFrame, days_by_cell: dict) -> dict:
    """距離×時期行列の分析の daily_metrics から、セルごとの精度を返す（§7）。

    days_by_cell : {(distance, position) -> その セルで IG を計算した評価日の集合}
    **IG を計算した日に限定**して集計する（中断時に精度とプロファイルが別の日を見るのを防ぐ）。
    """
    d = daily[daily["missing"] == 0].copy()
    d["distance"] = d["distance"].astype(str)
    # ① 日ごとに seed 中央値（距離×時期行列の分析と同じ順序）
    med = d.groupby(["distance", "day"])[ACC_COLS].median()
    out = {}
    for (dist, pos), days in days_by_cell.items():
        key = [(str(dist), day) for day in days]
        sub = med.reindex(key).dropna(how="all")
        out[(dist, pos)] = {c: (float(sub[c].mean()) if len(sub) else np.nan) for c in ACC_COLS}
    return out


def _cell_frame(rows: pd.DataFrame) -> dict:
    """1 セルぶんの行（評価日ごと）→ 特徴量ごとの平均・std・寄与率 ＋ 日数。"""
    feats = rows[list(FEATURE_NAMES)]
    mean = feats.mean(axis=0)
    std = feats.std(axis=0, ddof=1) if len(feats) > 1 else pd.Series(np.nan, index=FEATURE_NAMES)
    total = float(mean.sum())
    # 寄与率は生の値から復元できるので保存はしないが、ここで出す（§9.1）
    share = mean / total if total != 0 else pd.Series(np.nan, index=FEATURE_NAMES)
    out = {"n_days": int(len(rows)), "n_changes": float(rows["n_changes"].mean()),
           "sum_ig": total}
    for f in FEATURE_NAMES:
        out[f] = float(mean[f])
        out[f"{f}_std"] = float(std[f])
        out[f"{f}_share"] = float(share[f])
    return out


def build_table(ig_daily: pd.DataFrame, daily_metrics: pd.DataFrame,
                head: str, group: str) -> pd.DataFrame:
    """1 つの (ヘッド, 群) についてセル×特徴量の表を作る。

    汎用ヘッドの行は `(評価日, seed)` で持っているので（§9.1）、probe 行の seed を見て
    同じ `(評価日, seed)` の行を引き当ててから距離ごとに並べ直す。
    """
    probe = ig_daily[(ig_daily["head"] == "probe") & (ig_daily["group"] == group)]
    if probe.empty:
        return pd.DataFrame()
    if head == "probe":
        rows = probe
    else:
        gen = ig_daily[(ig_daily["head"] == "general") & (ig_daily["group"] == group)]
        if gen.empty:
            return pd.DataFrame()
        # probe 側の (day, distance, seed, position) に、同じ (day, seed) の汎用ヘッドを付ける
        keys = probe[["day", "distance", "position", "seed"]]
        rows = keys.merge(gen.drop(columns=["distance", "position"]), on=["day", "seed"], how="inner")

    days_by_cell = {k: set(v["day"]) for k, v in rows.groupby(["distance", "position"])}
    acc = accuracy_by_cell(daily_metrics, days_by_cell)

    out = []
    for (dist, pos), sub in rows.groupby(["distance", "position"]):
        rec = {"distance": dist, "position": int(pos), **acc.get((dist, pos), {}),
               **_cell_frame(sub)}
        out.append(rec)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["_order"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df.sort_values(["_order", "position"], kind="stable").drop(columns="_order")
    cols = ["distance", "position", "n_days", "n_changes"] + ACC_COLS + ["sum_ig"]
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest]


def write_tables(ig_daily: pd.DataFrame, daily_metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    """probe_pos / probe_neg / general_pos / general_neg の 4 表を書く（§9.3）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for head in ("probe", "general"):
        for group in ("pos", "neg"):
            df = build_table(ig_daily, daily_metrics, head, group)
            if df.empty:
                logger.warning(f"該当データなしのため表をスキップ: {head}_{group}")
                continue
            path = out_dir / f"{head}_{group}.csv"
            # Excel が UTF-8 を誤認して文字化けしないよう BOM 付きで保存
            df.to_csv(path, index=False, encoding=constants.CSV_ENCODING)
            paths.append(path)
    return paths
