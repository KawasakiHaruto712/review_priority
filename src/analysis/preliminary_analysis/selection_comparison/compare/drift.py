"""C：分布の比較（design.md §5。論文 4.a.iv・4.b.ii のデータ側）。

B で選んだ列について、2 つの学習の日の集まりと評価日の分布のずれを並べる。data_drift の保存済みの結果を読むだけ。
- 同じ選び方の中：その選び方の 行 d ＝ 0 と相手の行。
- 2 つの選び方の間：時間的な距離の 行 d ＝ 0 と、特徴空間上の距離の 行 d ＝ 0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.selection_comparison.io import readers
from src.analysis.preliminary_analysis.selection_comparison.utils import constants


def model_drift(kind: str, project: str, version: str, position: int, row: int) -> dict:
    """1 つの学習の日の集まり（kind の 行 row）の、評価日とのずれ。全体の群。"""
    s = readers.drift_summary(kind, project, version)
    s = s[(s["group"] == constants.DRIFT_GROUP) & (s["position"] == position) & (s["row"] == row)]
    c = readers.drift_cells(kind, project, version)
    c = c[(c["group"] == constants.DRIFT_GROUP) & (c["position"] == position) & (c["row"] == row)]
    c = c.assign(_abs=c["delta_mean"].abs()).sort_values("_abs", ascending=False, kind="stable")
    big = c[c["_abs"] >= constants.EFFECT_BOUND]
    return {"abs_delta_mean": float(s["abs_delta_mean"].iloc[0]) if len(s) else np.nan,
            "n_ge_bound": int(s["n_ge_bound"].iloc[0]) if len(s) else None,
            "features": big["feature"].tolist(), "deltas": [float(x) for x in big["delta_mean"]]}


def _pair_rows(base: dict, a: tuple, b: tuple) -> list[dict]:
    """2 つのモデル（(名前, kind, 行)）を並べた 2 行。一覧には、相手の一覧に出てこない特徴量に印（only_this）。"""
    p, v, pos = base["project"], base["version"], base["position"]
    da, db = model_drift(a[1], p, v, pos, a[2]), model_drift(b[1], p, v, pos, b[2])
    out = []
    for (name, kind, row), me, other in ((a, da, db), (b, db, da)):
        out.append({**base, "position_display": constants.position_display(pos), "model": name, "kind": kind, "row": row,
                    "abs_delta_mean": me["abs_delta_mean"], "n_ge_bound": me["n_ge_bound"],
                    "features": ";".join(me["features"]),
                    "only_this": ";".join(f for f in me["features"] if f not in set(other["features"]))})
    return out


def within_table(sel: dict) -> pd.DataFrame:
    rows = []
    for kind, sides in sel["within"].items():
        for side in ("won", "lost"):
            for c in sides[side]:
                base = {"comparison": kind, "side": side, "project": c["project"], "version": c["version"],
                        "position": c["position"], "diff": c["diff"]}
                rows += _pair_rows(base, ("d0", kind, 0), (f"d{c['opponent']}", kind, c["opponent"]))
    return pd.DataFrame(rows)


def cross_table(sel: dict) -> pd.DataFrame:
    rows = []
    for side in ("space_won", "time_won"):
        for c in sel["cross"][side]:
            base = {"comparison": "cross", "side": side, "project": c["project"], "version": c["version"],
                    "position": c["position"], "diff": c["diff"]}
            rows += _pair_rows(base, ("time_d0", "time_distance", 0), ("space_d0", "feature_distance", 0))
    return pd.DataFrame(rows)


def markdown(table: pd.DataFrame, title: str) -> str:
    """人が読む表（表示の向き）。一覧は |δ| の大きい順、相手に出てこない特徴量を太字。"""
    lines = [f"# {title}", "", "| 区分 | リポジトリ | 版 | 列 | モデル | 差（精度） | |δ| の平均 | |δ| ≥ 0.147 の数 | ずれが小さくない特徴量（太字 ＝ 相手に出てこない） |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        only = set(str(r["only_this"]).split(";")) if r["only_this"] else set()
        feats = "、".join(f"**{f}**" if f in only else f for f in (str(r["features"]).split(";") if r["features"] else []))
        lines.append(f"| {r['side']} | {r['project']} | {r['version']} | {constants.position_text(r['position'])} | "
                     f"{r['model']} | {r['diff']:+.3f} | {r['abs_delta_mean']:.3f} | {r['n_ge_bound']} | {feats} |")
    return "\n".join(lines) + "\n"


def run(sel: dict, out_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    within, cross = within_table(sel), cross_table(sel)
    out_dir.mkdir(parents=True, exist_ok=True)
    within.to_csv(out_dir / "drift_within.csv", index=False, encoding=constants.ENCODING)
    cross.to_csv(out_dir / "drift_cross.csv", index=False, encoding=constants.ENCODING)
    (out_dir / "drift_within.md").write_text(markdown(within, "分布の比較：同じ選び方の中（d0 と相手）"), encoding="utf-8")
    (out_dir / "drift_cross.md").write_text(markdown(cross, "分布の比較：2 つの選び方の間（時間の d0 と空間の d0）"), encoding="utf-8")
    return within, cross
