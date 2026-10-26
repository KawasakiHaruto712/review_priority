"""本文用の集計（design.md §6）。best_size.csv の表から数を出す。"""
from __future__ import annotations

import pandas as pd

from src.analysis.preliminary_analysis.release_best_size.utils import constants


def per_phase(table: pd.DataFrame) -> pd.DataFrame:
    """選び方 × リポジトリ × 評価時期ごとに、5 リリースの最良の N の並び（古い順）と種類の数。"""
    rows = []
    for (method, project, q), g in table.groupby(["method", "project", "position_display"], sort=False):
        bests = list(g["best"])
        known = [b for b in bests if isinstance(b, str)]
        rows.append({"method": method, "project": project, "position_display": q,
                     "phase_name": constants.PHASE_NAMES[q], "bests": bests,
                     "n_kinds": len(set(known)), "all_same": len(known) == len(bests) and len(set(known)) == 1})
    return pd.DataFrame(rows)


def run(table: pd.DataFrame) -> dict:
    """選び方ごとに §6 の 1〜4 を出す。"""
    phases = per_phase(table)
    out = {}
    for method in constants.METHODS:
        t = table[table["method"] == method]
        p = phases[phases["method"] == method]
        diff = t["diff_best_minus_selected"].dropna()
        out[method] = {
            "n_phase_all_same": int(p["all_same"].sum()),                      # 1. 5 リリースの最良の N がすべて同じ
            "n_phase": int(len(p)),
            "n_kinds_distribution": {int(k): int(v) for k, v in p["n_kinds"].value_counts().sort_index().items()},   # 2.
            "n_cell_selected_best": int(t["is_selected_best"].sum()),          # 3. 最良が 4.1 の N と一致
            "n_cell": int(len(t)),
            "n_cell_no_best": int(t["best"].isna().sum()),
            "diff_median": float(diff.median()) if len(diff) else None,         # 4. 最良 − 4.1 の N
            "diff_max": float(diff.max()) if len(diff) else None,
            "n_days_any_min": int(t["n_days_any"].min()),
            "n_days_any_max": int(t["n_days_any"].max()),
        }
    return out


def markdown(summary: dict, table: pd.DataFrame) -> str:
    """人が読む集計（数だけ。評価の言葉は書かない）。"""
    phases = per_phase(table)
    lines = ["# リリースごとに適した日数 N：本文用の集計", ""]
    for method in constants.METHODS:
        s = summary[method]
        lines += [f"## {constants.METHOD_LABEL[method]}による選び方", "",
                  f"- 5 リリースの最良の N がすべて同じだった「リポジトリ × 評価時期」：{s['n_phase_all_same']} / {s['n_phase']}",
                  "- 5 リリースで最良になった N の種類の数：" + "・".join(f"{k} 種類 {v} 件" for k, v in s["n_kinds_distribution"].items()),
                  f"- 最良の N が 4.1 で選んだ N と一致したマス：{s['n_cell_selected_best']} / {s['n_cell']}"
                  + (f"（最良なし {s['n_cell_no_best']}）" if s["n_cell_no_best"] else ""),
                  f"- F1 値の差（最良 − 4.1 の N）：中央値 {s['diff_median']:.4f}・最大 {s['diff_max']:.4f}",
                  f"- マスあたりの評価日：{s['n_days_any_min']}〜{s['n_days_any_max']} 日", "",
                  "| リポジトリ | 評価時期 | 5 リリースの最良の N（古い順） | 種類 |", "|---|---|---|---|"]
        p = phases[phases["method"] == method]
        for project in constants.PAPER_ORDER:
            for _, r in p[p["project"] == project].sort_values("position_display").iterrows():
                bests = "・".join(constants.SIZE_TEXT.get(b, "なし") if isinstance(b, str) else "なし" for b in r["bests"])
                lines.append(f"| {constants.REPO_TEXT[project]} | p{r['position_display']}（{r['phase_name']}） | {bests} | {r['n_kinds']} |")
        lines.append("")
    return "\n".join(lines)
