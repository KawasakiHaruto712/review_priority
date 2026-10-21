"""6 件をまとめた報告用の表（design.md §5.2。lookback_window/visualization/report.py の複製。2026-10-21 追加）。

各リポジトリの summary/summary_<metric>.csv（§5）と summary/rank_<metric>.csv（§5.1）を読み、
<OUTPUT_ROOT>/report/ に次を書く（CSV は数字そのまま、Markdown は報告にそのまま貼れる形）。ほかの出力には触らない。
- rank_<metric>_all.csv・.md：行＝大きさ、列＝リポジトリ、「全体」の 1 位率 / 最下位率（%）。
  Markdown は各列の 1 位率の最大と最下位率の最小を太字にし、対象日数と同着の注記を付ける。
- best_<metric>_all.csv・.md：リポジトリごとに全評価日の平均（Ave）が最良の大きさと、その値 ± 標準偏差。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_rank(path: Path, index_col: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """rank_<metric>.csv（冒頭に「# 対象日数: 版=日数, …」の行）を読む。"""
    with open(path, encoding="utf-8-sig") as f:
        head = f.readline()
        df = pd.read_csv(f, index_col=index_col)
    ndays = {k: int(v) for k, v in re.findall(r"([^,:\s]+)=(\d+)", head)}
    return df, ndays


def _bold(s: str) -> str:
    return f"**{s}**"


def build_report(projects: list[str], metric: str, root: Path, labels_desc: list[str], *,
                 index_col: str, row_name: str, lower_is_better: bool = False) -> Path | None:
    """報告用の表を書いて、出力先を返す。どのリポジトリにも結果がなければ None。

    lower_is_better（norm_rank など）なら、平均が最も小さいものを「最良」にする（design.md §5.1。2026-10-21）。
    """
    first, last, ndays, best = {}, {}, {}, {}
    for p in projects:
        s_path = root / p / "summary" / f"summary_{metric}.csv"
        r_path = root / p / "summary" / f"rank_{metric}.csv"
        if r_path.exists():
            rk, nd = read_rank(r_path, index_col)
            first[p] = rk["全体_1位率"].reindex(labels_desc)
            last[p] = rk["全体_最下位率"].reindex(labels_desc)
            ndays[p] = nd.get("全体", np.nan)
        else:
            logger.warning(f"[{p}] 順位の集計がありません（{r_path}。先に --mode rank）")
        if s_path.exists():
            sm = pd.read_csv(s_path, index_col=index_col).reindex(labels_desc)
            if sm["Ave"].notna().any():
                b = sm["Ave"].idxmin() if lower_is_better else sm["Ave"].idxmax()
                best[p] = (b, float(sm.loc[b, "Ave"]), float(sm.loc[b, "Ave_std"]))
        else:
            logger.warning(f"[{p}] 要約表がありません（{s_path}。先に --mode run）")
    if not first and not best:
        return None
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    cols = [p for p in projects if p in first]

    # ── 1 位率 / 最下位率 ──
    csv = pd.DataFrame(index=labels_desc)
    for p in cols:
        csv[f"{p}_1位率"] = first[p]
        csv[f"{p}_最下位率"] = last[p]
    csv.loc["対象日数"] = [ndays[p] if c.endswith("_1位率") else np.nan for p in cols for c in (f"{p}_1位率", f"{p}_最下位率")]
    csv.to_csv(out / f"rank_{metric}_all.csv", index_label=index_col, encoding="utf-8-sig")
    lines = [f"| {row_name} | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for lab in labels_desc:
        cells = []
        for p in cols:
            f_v, l_v = first[p][lab], last[p][lab]
            f_s = "-" if pd.isna(f_v) else f"{f_v:.1f}"
            l_s = "-" if pd.isna(l_v) else f"{l_v:.1f}"
            if pd.notna(f_v) and f_v == first[p].max():
                f_s = _bold(f_s)
            if pd.notna(l_v) and l_v == last[p].min():
                l_s = _bold(l_s)
            cells.append(f"{f_s} / {l_s}")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines += ["", "左＝1 位率、右＝最下位率（%）。太字は各列の最大（1 位率）／最小（最下位率）。",
              "※ 対象日数：" + ", ".join(f"{p} {ndays[p]}" for p in cols),
              "※ 同着は該当するすべてに加算するため、1 位率の合計は 100% を超えることがある。"]
    (out / f"rank_{metric}_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── 平均が最良 ──
    bcols = [p for p in projects if p in best]
    pd.DataFrame([{"project": p, "best": best[p][0], "mean": best[p][1], "std": best[p][2]} for p in bcols]) \
        .to_csv(out / f"best_{metric}_all.csv", index=False, encoding="utf-8-sig")
    blines = ["| | " + " | ".join(bcols) + " |", "|" + "---|" * (len(bcols) + 1),
              "| 最良 | " + " | ".join(best[p][0] for p in bcols) + " |",
              f"| 平均 {metric.upper()} | " + " | ".join(f"{best[p][1]:.3f} ± {best[p][2]:.3f}" for p in bcols) + " |",
              "", f"全評価日の平均（要約表の Ave）が最良の{row_name}。"]
    (out / f"best_{metric}_all.md").write_text("\n".join(blines) + "\n", encoding="utf-8")
    return out
