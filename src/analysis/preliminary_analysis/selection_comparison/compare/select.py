"""B：精度が大きく変わった列を選ぶ（design.md §4）。

- 同じ選び方の中：リリースごとの精度表で、列ごとに d ＝ 0 − 相手（d ＝ 1・2 の値の大きい方）。
- 2 つの選び方の間：A のリリースごとの表の 空間 − 時間。
差の大きさの順だけで選ぶ（有意かどうか・評価日の数は見ない。本人判断）。同じ差のときは リポジトリ（PROJECTS の順）・版（古い順）・列（保存物の番号）の順。
上回った側は差が正の列だけ、下回った側は差が負の列だけから選ぶ（足りなければ K 列より少ない）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.selection_comparison.io import readers
from src.analysis.preliminary_analysis.selection_comparison.utils import constants


def within_columns(kind: str) -> pd.DataFrame:
    """その選び方の、リリースごとの全列の d ＝ 0 − 相手。"""
    rows = []
    for project in constants.PROJECTS:
        for vi, version in enumerate(readers.versions(project)):
            value, n_days = readers.matrix_values(kind, project, version)
            for pos in range(value.shape[1]):
                others = {d: value[d, pos] for d in range(1, value.shape[0]) if np.isfinite(value[d, pos])}
                if not np.isfinite(value[0, pos]) or not others:
                    continue
                opp = max(others, key=lambda d: (others[d], -d))      # 値の大きい方。同じなら番号の小さい方
                rows.append({"project": project, "version": version, "_v": vi, "position": pos,
                             "d0": float(value[0, pos]), "opponent": int(opp), "opponent_value": float(others[opp]),
                             "diff": float(value[0, pos] - others[opp]),
                             "n_days": int(n_days[0, pos]) if n_days is not None and np.isfinite(n_days[0, pos]) else None})
    return pd.DataFrame(rows)


def _order(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_p=df["project"].map(constants.PROJECTS.index))


def pick(df: pd.DataFrame, k: int, positive: bool) -> list[dict]:
    """差が正（positive）または負の列から、差の絶対値の大きい順に k 列。同じ差は リポジトリ・版・列 の順。"""
    d = _order(df[df["diff"] > 0] if positive else df[df["diff"] < 0])
    d = d.assign(_key=-d["diff"] if positive else d["diff"])
    d = d.sort_values(["_key", "_p", "_v", "position"], kind="stable").head(k)
    out = []
    for r in d.to_dict("records"):
        item = {"project": r["project"], "version": r["version"], "position": int(r["position"]),
                "position_display": constants.position_display(r["position"]), "diff": r["diff"]}
        for key in ("d0", "opponent", "opponent_value", "time_mean", "space_mean"):
            if key in r and r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key])):
                item[key] = r[key]
        n = r.get("n_days", r.get("n_paired"))
        item["n_days"] = None if n is None or (isinstance(n, float) and np.isnan(n)) else int(n)
        out.append(item)
    return out


def cross_columns(release: pd.DataFrame) -> pd.DataFrame:
    """A のリリースごとの表から、2 つの選び方の間の差（空間 − 時間）。"""
    ver_index = {(p, v): i for p in constants.PROJECTS for i, v in enumerate(readers.versions(p))}
    r = release.dropna(subset=["diff"]).copy()
    r["_v"] = [ver_index[(p, v)] for p, v in zip(r["project"], r["version"])]
    r["position"] = r["position"].astype(int)
    return r


def select(release: pd.DataFrame, k: int = constants.TOP_K) -> dict:
    """選んだ列（selected_columns.json の中身）。"""
    within = {}
    for kind in constants.KINDS:
        cols = within_columns(kind)
        within[kind] = {"won": pick(cols, k, True), "lost": pick(cols, k, False)}
    cross = cross_columns(release)
    return {"top_k": k, "within": within,
            "cross": {"space_won": pick(cross, k, True), "time_won": pick(cross, k, False)}}


def markdown(sel: dict) -> str:
    """人が読む一覧（表示の向き）。"""
    lines = [f"# 選んだ列（上回った側・下回った側それぞれ上位 {sel['top_k']} 列。差の大きさの順だけで選んだ）", ""]
    for kind, sides in sel["within"].items():
        lines += [f"## 同じ選び方の中：{constants.KIND_LABEL[kind]}（差 ＝ d0 − 相手）", "",
                  "| 区分 | リポジトリ | 版 | 列（表示／保存物） | d0 | 相手 | 差 | 評価日 |", "|---|---|---|---|---|---|---|---|"]
        for side, label in (("won", "d0 が上回った"), ("lost", "d0 が下回った")):
            for i, c in enumerate(sides[side]):
                lines.append(f"| {label if i == 0 else ''} | {c['project']} | {c['version']} | "
                             f"{constants.position_text(c['position'])}／p{c['position']} | {c['d0']:.3f} | "
                             f"d{c['opponent']} {c['opponent_value']:.3f} | {c['diff']:+.3f} | {c['n_days']} |")
        lines.append("")
    lines += ["## 2 つの選び方の間（差 ＝ 空間 − 時間。d0 どうし）", "",
              "| 区分 | リポジトリ | 版 | 列（表示／保存物） | 時間 | 空間 | 差 | 評価日（両方） |", "|---|---|---|---|---|---|---|---|"]
    for side, label in (("space_won", "空間が上回った"), ("time_won", "時間が上回った")):
        for i, c in enumerate(sel["cross"][side]):
            lines.append(f"| {label if i == 0 else ''} | {c['project']} | {c['version']} | "
                         f"{constants.position_text(c['position'])}／p{c['position']} | {c['time_mean']:.3f} | "
                         f"{c['space_mean']:.3f} | {c['diff']:+.3f} | {c['n_days']} |")
    return "\n".join(lines) + "\n"


def run(release: pd.DataFrame, out_dir, k: int = constants.TOP_K) -> dict:
    sel = select(release, k)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_columns.json").write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "selected_columns.md").write_text(markdown(sel), encoding="utf-8")
    return sel


def load(out_dir) -> dict:
    """保存した selected_columns.json を読む。"""
    return json.loads((out_dir / "selected_columns.json").read_text(encoding="utf-8"))
