"""表を読んで相関で絞り、結果を書く（design.md §3.3）。"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.feature_table import constants as C
from src.analysis.preliminary_analysis.feature_table.build.table import load_table
from src.analysis.preliminary_analysis.feature_table.selection.autospearman import aggregate, auto_spearman, spearman_matrix

logger = logging.getLogger(__name__)


def _summary_rows(project: str, names: list[str], feats: np.ndarray, filled: np.ndarray,
                  filled_names: list[str]) -> list[dict]:
    """特徴量ごとの、代わりの値を入れた割合と 0 でない値の割合（feature_summary.csv。§3.3）。"""
    fpos = {n: k for k, n in enumerate(filled_names)}
    n = len(feats)
    rows = []
    for j, name in enumerate(names):
        col = feats[:, j]
        rows.append({"project": project, "feature": name, "rows": n,
                     "filled_rate": float(filled[:, fpos[name]].mean()) if name in fpos and n else 0.0,
                     "nonzero_rate": float(np.mean(col != 0)) if n else 0.0,
                     "constant": bool(n and np.all(col == col[0]))})
    return rows


def run(projects: list[str], table_root: Path = C.TABLE_ROOT, out_dir: Path = C.SELECTION_ROOT,
        threshold: float = C.SPEARMAN_THRESHOLD) -> dict[str, list[str]]:
    """6 件の表から相関を出し、5 通りの決まりで残す特徴量を決めて書く（採るのは C.ADOPTED_RULE）。決まりごとの残る特徴量を返す。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] | None = None
    rhos, summary, metas = [], [], {}
    for p in projects:
        tab = load_table(p, root=table_root)
        tnames = [str(n) for n in tab["feature_names"]]
        if names is None:
            names = tnames
        elif tnames != names:
            raise ValueError(f"[{p}] 表の特徴量の並びがほかと違います")
        feats = tab["features"]
        logger.info(f"[{p}] 順位相関：{feats.shape[0]} 行 × {feats.shape[1]} 列")
        rho = spearman_matrix(feats)
        rhos.append(rho)
        pd.DataFrame(rho, index=names, columns=names).to_csv(out_dir / f"corr_{p}.csv")
        summary += _summary_rows(p, names, feats, tab["filled"], [str(n) for n in tab["filled_names"]])
        meta_path = Path(table_root) / p / f"{C.PERIOD_PRETRAIN}_meta.json"
        metas[p] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        del tab, feats
    pd.DataFrame(summary).to_csv(out_dir / "feature_summary.csv", index=False)

    agg = aggregate(names, rhos, threshold)
    # pairs.csv：全部の組
    rows = []
    mean, cnt = agg.mean_abs, agg.count_ge
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            row = {"feature_a": names[i], "feature_b": names[j]}
            row.update({f"rho_{p}": float(r[i, j]) for p, r in zip(projects, rhos)})
            row.update({"repos_ge_threshold": int(cnt[i, j]), "repos_defined": int(agg.count_defined[i, j]),
                        "mean_abs_rho": float(mean[i, j]),
                        "other_mean_a": agg.other_mean(i, j), "other_mean_b": agg.other_mean(j, i)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "pairs.csv", index=False)

    kept_by_rule: dict[str, list[str]] = {}
    for rule in C.RULES:
        kept, steps = auto_spearman(agg, rule, C.RATIO)
        kept_by_rule[rule] = kept
        pd.DataFrame(steps, columns=["step", "feature_a", "feature_b", "mean_abs_rho", "repos_ge_threshold", "repos_defined",
                                     "other_mean_a", "other_mean_b", "kept", "dropped"]).to_csv(
            out_dir / f"steps_{rule}.csv", index=False)
        logger.info(f"決まり {rule}：{len(kept)} 個を残す（{len(names) - len(kept)} 個を落とす）")
    sel = pd.DataFrame({"feature": names})
    for rule in C.RULES:
        sel[rule] = [int(n in set(kept_by_rule[rule])) for n in names]
    sel.to_csv(out_dir / "selected.csv", index=False)

    with open(out_dir / "selection_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"created_at": datetime.now(timezone.utc).isoformat(), "projects": projects,
                   "threshold": threshold, "rules": C.RULES, "ratio": list(C.RATIO), "adopted_rule": C.ADOPTED_RULE,
                   "table_meta": metas}, fh, ensure_ascii=False, indent=2)
    return kept_by_rule
