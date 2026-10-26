"""出力（design.md §5）：best_size.csv・best_size.md・best_size_table.tex・summary.json・summary.md。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.preliminary_analysis.release_best_size.compute import summary as summary_mod
from src.analysis.preliminary_analysis.release_best_size.utils import constants


def _cell(table: pd.DataFrame, method: str, project: str, version: str, q: int) -> pd.Series:
    r = table[(table["method"] == method) & (table["project"] == project)
              & (table["version"] == version) & (table["position_display"] == q)]
    return r.iloc[0]


def _versions(table: pd.DataFrame, project: str) -> list[str]:
    """表の版の並び（古い順。表を作った順のまま）。"""
    return list(dict.fromkeys(table.loc[table["project"] == project, "version"]))


def markdown(table: pd.DataFrame) -> str:
    """選び方ごとに 30 行 × 3 列（表示の向き）。4.1 の N と違うマスは **太字**。"""
    lines = ["# リリースごとに最良となった日数 N", "",
             "太字は 4.1 で選んだ N と違うマス。評価時期は表示の向き（p0 ＝ 初期）。", "",
             "| リポジトリ | リリース | " + " | ".join(f"{constants.METHOD_LABEL[m]} p{q}" for m in constants.METHODS
                                                for q in range(constants.N_COLS)) + " |",
             "|---|---|" + "---|" * (2 * constants.N_COLS)]
    for project in constants.PAPER_ORDER:
        for version in _versions(table, project):
            cells = []
            for m in constants.METHODS:
                for q in range(constants.N_COLS):
                    r = _cell(table, m, project, version, q)
                    t = constants.SIZE_TEXT[r["best"]] if isinstance(r["best"], str) else "－"
                    cells.append(t if r["is_selected_best"] else f"**{t}**")
            lines.append(f"| {constants.REPO_TEXT[project]} | {version} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def latex(table: pd.DataFrame, summary: dict) -> str:
    """論文用の表（design.md §5.1）。30 行 × 時間 p0〜p2・特徴空間 p0〜p2。4.1 の N と違うマスは太字。"""
    lo = min(s["n_days_any_min"] for s in summary.values())
    hi = max(s["n_days_any_max"] for s in summary.values())
    n = constants.N_COLS
    lines = [r"\begin{table}[tb]",
             r"\caption{リリースと評価時期ごとに最も精度が高かったN（太字は\ref{subsec:method:size}節で選んだNと異なるもの）}",
             r"\label{tab:release}", r"\centering", r"\footnotesize",
             r"\begin{tabular}{ll" + "c" * (2 * n) + "}", r"\hline",
             " & & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{constants.METHOD_LABEL[m]}}}" for m in constants.METHODS) + r" \\",
             rf"\cline{{3-{2 + 2 * n}}}",
             "リポジトリ & リリース & " + " & ".join(f"p{q}" for _ in constants.METHODS for q in range(n)) + r" \\",
             r"\hline"]
    for project in constants.PAPER_ORDER:
        versions = _versions(table, project)
        for i, version in enumerate(versions):
            head = rf"\multirow{{{len(versions)}}}{{*}}{{{constants.REPO_TEXT[project]}}}" if i == 0 else ""
            cells = []
            for m in constants.METHODS:
                for q in range(n):
                    r = _cell(table, m, project, version, q)
                    t = constants.SIZE_TEXT[r["best"]] if isinstance(r["best"], str) else "－"
                    cells.append(t if r["is_selected_best"] else rf"\textbf{{{t}}}")
            lines.append(f"{head} & {version} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", "",
              rf"\vspace{{1mm}}{{\footnotesize 各マスの評価日は{lo}〜{hi}日．}}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_all(table: pd.DataFrame, summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "best_size.csv", index=False, encoding=constants.ENCODING)
    (out / "best_size.md").write_text(markdown(table), encoding="utf-8")
    (out / "best_size_table.tex").write_text(latex(table, summary), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(summary_mod.markdown(summary, table), encoding="utf-8")
