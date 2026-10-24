"""A：精度の比較（design.md §3。論文 4.b.i・5.b.i）。

時間的な距離の d ＝ 0 と特徴空間上の距離の d ＝ 0 を、同じ評価日で比べる。差 ＝ 空間 − 時間（正 ＝ 空間の方が高い）。
- 平均はそれぞれのモデルに値のある日すべて（精度表の値と一致させる）。
- 勝った日の割合と検定（ウィルコクソンの符号順位検定・両側）は、両方に値のある日。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.analysis.preliminary_analysis.selection_comparison.io import readers
from src.analysis.preliminary_analysis.selection_comparison.utils import constants

COLUMNS = ["time_mean", "space_mean", "diff", "n_time", "n_space", "n_paired", "space_win_rate", "time_win_rate", "p_value"]


def paired_test(diff: pd.Series) -> float:
    """評価日ごとの差のウィルコクソンの符号順位検定（両側）。差がすべて 0 か日がなければ NaN。"""
    d = diff.dropna()
    if d.empty or not (d != 0).any():
        return float("nan")
    return float(wilcoxon(d).pvalue)


def daily_pairs(project: str) -> pd.DataFrame:
    """その project の全版について、評価日ごとの時間（d ＝ 0）と空間（d ＝ 0）の F1 値。片方だけの日も残す（NaN）。
    列：project・version・day・position（保存物の向き）・time・space。"""
    frames = []
    for v in readers.versions(project):
        t = readers.daily_values(constants.time_daily_path(project, v))
        s = readers.daily_values(constants.space_daily_path(project, v))
        t = t[t["distance"] == 0].set_index(["day", "position"])["value"].rename("time")
        s = s[s["distance"] == 0].set_index(["day", "position"])["value"].rename("space")
        j = pd.concat([t, s], axis=1).reset_index()
        j.insert(0, "version", v)
        j.insert(0, "project", project)
        frames.append(j)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["project", "version", "day", "position", "time", "space"])


def summarize(rows: pd.DataFrame) -> dict:
    """1 つのまとまり（列）の比較。"""
    time, space = rows["time"].dropna(), rows["space"].dropna()
    paired = rows.dropna(subset=["time", "space"])
    d = paired["space"] - paired["time"]
    tm = float(time.mean()) if len(time) else np.nan
    sm = float(space.mean()) if len(space) else np.nan
    return {"time_mean": tm, "space_mean": sm, "diff": sm - tm,
            "n_time": int(len(time)), "n_space": int(len(space)), "n_paired": int(len(paired)),
            "space_win_rate": float((d > 0).mean()) if len(d) else np.nan,
            "time_win_rate": float((d < 0).mean()) if len(d) else np.nan,
            "p_value": paired_test(d)}


def _with_display(df: pd.DataFrame) -> pd.DataFrame:
    """表示の向きの列（p0 ＝ 初期）を足す。全体の行（position が "all"）はそのまま。"""
    disp = [constants.position_display(p) if p != "all" else "all" for p in df["position"]]
    df.insert(df.columns.get_loc("position") + 1, "position_display", disp)
    return df


def by_phase(pairs: pd.DataFrame) -> pd.DataFrame:
    """評価時期ごと（5 リリースまとめ）と全体。1 行 ＝ リポジトリ × 評価時期。"""
    out = []
    for project, g in pairs.groupby("project", sort=False):
        for pos, x in g.groupby("position"):
            out.append({"project": project, "position": int(pos), **summarize(x)})
        out.append({"project": project, "position": "all", **summarize(g)})
    return _with_display(pd.DataFrame(out, columns=["project", "position"] + COLUMNS))


def by_release(pairs: pd.DataFrame) -> pd.DataFrame:
    """リリースごと。1 行 ＝ リポジトリ × 版 × 評価時期。"""
    out = []
    for (project, version), g in pairs.groupby(["project", "version"], sort=False):
        for pos, x in g.groupby("position"):
            out.append({"project": project, "version": version, "position": int(pos), **summarize(x)})
    return _with_display(pd.DataFrame(out, columns=["project", "version", "position"] + COLUMNS))


def _mark(p: float) -> str:
    return "" if not np.isfinite(p) else ("**" if p < 0.05 / (3 * len(constants.PROJECTS)) else ("*" if p < 0.05 else ""))


def phase_markdown(table: pd.DataFrame) -> str:
    """評価時期ごとの表（表示の向き。差 ＝ 空間 − 時間）。"""
    heads = [constants.position_text(constants.N_COLS - 1 - q) for q in range(constants.N_COLS)]
    lines = ["| リポジトリ | " + " | ".join(heads) + " | 全体 |", "|---" * (constants.N_COLS + 2) + "|"]
    for project, g in table.groupby("project", sort=False):
        cells = []
        for q in list(range(constants.N_COLS)) + ["all"]:
            r = g[g["position_display"] == q]
            cells.append("—" if r.empty else f"{r['diff'].iloc[0]:+.3f}{_mark(r['p_value'].iloc[0])}")
        lines.append(f"| {project} | " + " | ".join(cells) + " |")
    lines += ["", "差 ＝ 空間（特徴空間上の距離の d ＝ 0）− 時間（時間的な距離の d ＝ 0）の F1 値。正 ＝ 空間の方が高い。",
              f"* p < 0.05、** 多重比較を補正しても有意（p < 0.05 ÷ {3 * len(constants.PROJECTS)}）。評価日ごとの差のウィルコクソンの符号順位検定。",
              "平均はそれぞれのモデルに値のある日すべて、勝った日の割合と検定は両方に値のある日。"]
    return "\n".join(lines) + "\n"


def run(out_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全リポジトリの比較を出して保存する。返り値は（評価時期ごと, リリースごと）。"""
    pairs = pd.concat([daily_pairs(p) for p in constants.PROJECTS], ignore_index=True)
    phase, release = by_phase(pairs), by_release(pairs)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase.to_csv(out_dir / "accuracy_by_phase.csv", index=False, encoding=constants.ENCODING)
    release.to_csv(out_dir / "accuracy_by_release.csv", index=False, encoding=constants.ENCODING)
    (out_dir / "accuracy_by_phase.md").write_text(phase_markdown(phase), encoding="utf-8")
    return phase, release
