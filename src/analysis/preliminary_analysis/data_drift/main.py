"""data_drift（特徴量別の分布）の実行の入口（design.md §3〜§6）。

モード:
  --mode run（既定） : 評価期間の表を読み、評価日 × 行ごとに比べて保存し、図を描く
  --mode plot        : 保存済みの cells.csv・class_counts.csv から図だけ描き直す

図：格子一覧は全セル（行を固定したもの・評価時期を固定したもの）。個別セル図は --cells で指定したセルだけ（例 --cells d0-p8 d20-p8）。
寄与度分析の範囲（IG_TARGETS）が決まったら、そのセルを既定にする（design.md §5）。

実行:
    python3 -m src.analysis.preliminary_analysis.data_drift.main --project qtbase
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from src.analysis.background_problem.common.data_loader import load_release_dates
from src.analysis.preliminary_analysis.data_drift.io import writer
from src.analysis.preliminary_analysis.data_drift.pairing import time_distance
from src.analysis.preliminary_analysis.data_drift.stats.aggregate import to_cells
from src.analysis.preliminary_analysis.data_drift.stats.runner import run_version
from src.analysis.preliminary_analysis.data_drift.utils import constants
from src.analysis.preliminary_analysis.data_drift.visualization import plotter
from src.analysis.preliminary_analysis.pretrained_encoders.features.feature_set import MODEL_FEATURES
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

PAIRINGS = {time_distance.KIND: time_distance}      # RQ2（feature_distance）は距離版の設計が決まり次第足す（design.md §3.2）


def _auc_lookup(project: str, version: str):
    """同じセルの精度表の AUC（concept_drift_detection の出力）。なければ None を返す関数。"""
    from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants as det
    path = det.OUTPUT_ROOT / project / det.MODEL_NAME[0] / version / "probe" / "auc" / "drift_matrix.csv"
    if not path.exists():
        logger.warning(f"[{project} {version}] 精度表の AUC がありません（{path}）。題は「—」にします")
        return None
    m = pd.read_csv(path, index_col=0)

    def get(r, p):
        try:
            v = m.loc[f"d{r}", f"p{p}"]
            return float(v) if pd.notna(v) else None
        except KeyError:
            return None
    return get


def run(kind: str, project: str, versions: list[str]) -> None:
    from src.analysis.preliminary_analysis.feature_table.build import ensure_table
    from src.analysis.preliminary_analysis.feature_table.build.sets import day_sets_from_table

    rel_df = load_release_dates()
    table = ensure_table(project, "eval", rel_df=rel_df)
    mod = PAIRINGS[kind]
    meta = {}
    for v in versions:
        try:
            pairing = mod.for_version(rel_df, project, v)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {v}]: {e}")
            continue
        day_sets = {s.t.date(): s for s in day_sets_from_table(table, pairing.record_start, pairing.ce_R,
                                                                 max_set_size=pre.MAX_SET_SIZE)}
        logger.info(f"--- {project} {v}：段数 {pairing.rows}・窓長 {pairing.window}・刻み {pairing.step}・日の集合 {len(day_sets)}")
        daily, counts = run_version(day_sets, pairing, list(MODEL_FEATURES))
        cells = to_cells(daily)
        logger.info(f"保存: {writer.save(kind, project, v, daily, cells, counts)}（評価日ごと {len(daily)} 行）")
        meta[v] = {"window": pairing.window, "step": pairing.step, "rows": pairing.rows}
    writer.save_run_meta(kind, project, meta, constants.GROUPS)


def _parse_cells(cells: list[str] | None) -> list[tuple[int, int]]:
    out = []
    for c in cells or []:
        d, p = c.lower().replace("d", "").split("-p")
        out.append((int(d), int(p)))
    return out


def draw(kind: str, project: str, versions: list[str], cells_arg: list[str] | None, groups) -> None:
    plotter.configure()
    row_label = PAIRINGS[kind].ROW_LABEL
    for v in versions:
        try:
            cells = writer.load(kind, project, v, "cells")
            counts = writer.load(kind, project, v, "class_counts")
        except FileNotFoundError:
            logger.warning(f"結果がないのでスキップ: {project} {v}")
            continue
        aucs = _auc_lookup(project, v)
        fig_dir = writer.version_dir(kind, project, v) / "figures"
        rows = sorted(cells["row"].unique().tolist())
        positions = sorted(cells["position"].unique().tolist())
        for g in groups:
            for r in rows:
                plotter.plot_grid(cells, counts, aucs, project, v, row_label, g, fixed="row", fixed_value=r,
                                  panel_values=positions, out_path=fig_dir / "grid" / g / f"row{r}.png")
            for p in positions:
                plotter.plot_grid(cells, counts, aucs, project, v, row_label, g, fixed="position", fixed_value=p,
                                  panel_values=rows, out_path=fig_dir / "grid" / g / f"position{p}.png")
            for r, p in _parse_cells(cells_arg):
                auc = aucs(r, p) if aucs else None
                plotter.plot_cell(cells, project, v, row_label, r, p, g, plotter.cell_info(counts, r, p), auc,
                                  fig_dir / "cell" / g / f"d{r}_p{p}.png")
        logger.info(f"図: {fig_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="data_drift：特徴量別の分布（データのずれの確認）")
    ap.add_argument("--mode", choices=["run", "plot"], default="run")
    ap.add_argument("--kind", choices=list(PAIRINGS), default=time_distance.KIND)
    ap.add_argument("--project", default=None, help="対象（未指定・'all' なら PROJECTS の 6 件を順に）")
    ap.add_argument("--versions", nargs="*", default=None, help="対象の版（既定は 5 版）")
    ap.add_argument("--cells", nargs="*", default=None, help="個別セル図を描くセル（例 d0-p8 d20-p8）")
    ap.add_argument("--groups", nargs="*", default=list(constants.GROUPS), help="描く群（all pos neg）")
    args = ap.parse_args()
    projects = list(pre.PROJECTS) if args.project in (None, "all") else [args.project]
    for project in projects:
        versions = args.versions or list(pre.project_config(project)["versions"])
        logger.info(f"===== {args.kind} {project}（{len(versions)} 版）=====")
        if args.mode == "run":
            run(args.kind, project, versions)
        draw(args.kind, project, versions, args.cells, args.groups)


if __name__ == "__main__":
    main()
