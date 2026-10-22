"""data_drift（特徴量別の分布）の実行の入口（design.md §3〜§6）。

モード:
  --mode run（既定） : 評価期間の表を読み、評価日 × 行ごとに比べて保存し、図を描く
  --mode plot        : 保存済みの cells.csv・class_counts.csv から図だけ描き直す
  --mode compare     : RQ1 と RQ2 を比べる折れ線（両方の cells.csv を読むだけ。2026-10-22）
  --kind             : time_distance（RQ1・既定）／feature_distance（RQ2。精度表の train_days を読む。2026-10-22）

図：格子一覧は全セル（行を固定したもの・評価時期を固定したもの）。個別セル図は --cells で指定したセルだけ（例 --cells d0-p8 d20-p8）。
寄与度分析の範囲（IG_TARGETS）が決まったら、そのセルを既定にする（design.md §5）。

実行（2026-10-22：run は --variant 必須。posw ＝ 重みあり版の出力は data_drift/posw/ の下。design.md §10.1）:
    python3 -m src.analysis.preliminary_analysis.data_drift.main --variant posw
    python3 -m src.analysis.preliminary_analysis.data_drift.main --variant posw --kind feature_distance
    python3 -m src.analysis.preliminary_analysis.data_drift.main --mode compare          # 省略時は重みあり版
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from src.analysis.background_problem.common.data_loader import load_release_dates
from src.analysis.preliminary_analysis.data_drift.io import writer
from src.analysis.preliminary_analysis.data_drift.pairing import feature_distance, time_distance
from src.analysis.preliminary_analysis.data_drift.stats.aggregate import to_cells
from src.analysis.preliminary_analysis.data_drift.stats.runner import run_version
from src.analysis.preliminary_analysis.data_drift.utils import constants, parallel
from src.analysis.preliminary_analysis.data_drift.visualization import plotter
from src.analysis.preliminary_analysis.pretrained_encoders.features.feature_set import MODEL_FEATURES
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre
from src.analysis.preliminary_analysis.pretrained_encoders.utils import variant

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

PAIRINGS = {time_distance.KIND: time_distance,      # RQ1
            feature_distance.KIND: feature_distance}  # RQ2（2026-10-22 追加。design.md §3.2）


ACC_METRICS = ("recall@pos", "auc", "f1_plugin")   # 題の 2 行目に出す精度（この順。design.md §5。2026-10-21 改めた：旧版は AUC だけ）
# （2026-10-22：f1_plugin を足した。表がなければ題から外れ、表はあるのにセルがなければ「—」。design.md §10.1）


def _auc_lookup(project: str, version: str, kind: str = time_distance.KIND):
    """同じセルの精度表の recall@pos と AUC を返す関数 get(r, p) -> {指標: 値 or None}。どちらの表もなければ None。
    RQ1 は concept_drift_detection、RQ2 は feature_space_window の精度表（2026-10-22：kind を足した。パスは各部品の accuracy_path）。"""
    tables = {}
    for metric in ACC_METRICS:
        path = PAIRINGS[kind].accuracy_path(project, version, metric)
        if path.exists():
            tables[metric] = pd.read_csv(path, index_col=0)
        else:
            logger.warning(f"[{project} {version}] 精度表の {metric} がありません（{path}）。題は「—」にします")
    if not tables:
        return None

    def get(r, p):
        out = {}
        for metric, m in tables.items():
            try:
                v = m.loc[f"d{r}", f"p{p}"]
                out[metric] = float(v) if pd.notna(v) else None
            except KeyError:
                out[metric] = None
        return out
    return get


def run(kind: str, project: str, versions: list[str]) -> None:
    from src.analysis.preliminary_analysis.feature_table.build import ensure_table
    from src.analysis.preliminary_analysis.feature_table.build.sets import day_sets_from_table

    if kind == feature_distance.KIND:           # RQ2 は表のつなぎ方と部品の作り方が違う（2026-10-22）
        return _run_feature_distance(project, versions)
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


def _run_feature_distance(project: str, versions: list[str]) -> None:
    """RQ2（design.md §3.2）：精度表の train_days・daily_metrics から部品を作り、事前学習の表と評価期間の表をつないで比べる。"""
    from src.analysis.preliminary_analysis.data_drift.dataset.tables import load_day_sets

    kind = feature_distance.KIND
    pairings = {}
    for v in versions:
        try:
            pairings[v] = feature_distance.for_version(project, v)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"スキップ [{project} {v}]: {e}（先に feature_space_window の --mode matrix）")
    if not pairings:
        return
    keep = set().union(*(p.needed_days() for p in pairings.values()))
    day_sets, pretrain_days = load_day_sets(project, load_release_dates(), keep=keep)
    meta = {}
    for v, pairing in pairings.items():
        pairing.pretrain_days = pretrain_days
        n_pairs = len(pairing.positions) * pairing.rows
        logger.info(f"--- {project} {v}：段数 {pairing.rows}・k {pairing.window}・刻み {pairing.step}・評価日 {len(pairing.positions)}"
                    f"・seed を選べた（評価日, 行）{len(pairing.seeds)} / {n_pairs}")
        daily, counts = run_version(day_sets, pairing, list(MODEL_FEATURES))
        cells = to_cells(daily)
        logger.info(f"保存: {writer.save(kind, project, v, daily, cells, counts)}（評価日ごと {len(daily)} 行）")
        meta[v] = {"window": pairing.window, "step": pairing.step, "rows": pairing.rows}
    writer.save_run_meta(kind, project, meta, constants.GROUPS)


def compare(project: str, versions: list[str]) -> None:
    """RQ1 と RQ2 を比べる折れ線（design.md §5。2026-10-22）。両方の cells.csv を読むだけ（計算し直さない）。"""
    from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants as det
    from src.analysis.preliminary_analysis.data_drift.visualization import compare as cmp
    from src.analysis.preliminary_analysis.feature_space_window.utils import constants as fsw

    plotter.configure()
    k = (fsw.matrix_settings().get(project) or {}).get("size_days")   # 版の種類に合う k（2026-10-22）
    labels = {time_distance.KIND: f"RQ1：時期のずれ（直前 {det.window_for(project)} 日の窓）",
              feature_distance.KIND: f"RQ2：特徴のずれ（近い {k} 日）"}
    out_dir = writer.compare_dir(project)
    for v in versions:
        cells = {}
        for kind in (time_distance.KIND, feature_distance.KIND):
            try:
                cells[kind] = writer.load(kind, project, v, "cells")
            except FileNotFoundError:
                logger.warning(f"[{project} {v}] {kind} の cells.csv がありません（その線は描かない）")
        table = cmp.compare_table(cells)
        if table.empty:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / f"{v}.csv", index=False)
        logger.info(f"比べる折れ線: {cmp.plot_compare(table, project, v, labels, out_dir / f'{v}.png')}")


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
        aucs = _auc_lookup(project, v, kind)
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
                accs = aucs(r, p) if aucs else None
                plotter.plot_cell(cells, project, v, row_label, r, p, g, plotter.cell_info(counts, r, p), accs,
                                  fig_dir / "cell" / g / f"d{r}_p{p}.png")
        logger.info(f"図: {fig_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="data_drift：特徴量別の分布（データのずれの確認）")
    ap.add_argument("--mode", choices=["run", "plot", "compare"], default="run",
                    help="compare は RQ1 と RQ2 を比べる折れ線（両方の run の後。cells.csv を読むだけ。design.md §5）")
    ap.add_argument("--kind", choices=list(PAIRINGS), default=time_distance.KIND)
    ap.add_argument("--project", default=None, help="対象（未指定・'all' なら PROJECTS の 6 件を順に）")
    ap.add_argument("--versions", nargs="*", default=None, help="対象の版（既定は 5 版）")
    ap.add_argument("--cells", nargs="*", default=None, help="個別セル図を描くセル（例 d0-p8 d20-p8）")
    ap.add_argument("--groups", nargs="*", default=list(constants.GROUPS), help="描く群（all pos neg）")
    ap.add_argument("--workers", type=int, default=None,
                    help="リポジトリを並べて回す数（既定：CUDA が使える機械なら CPU の上限まで最大 6、Mac などは 1。§6）")
    variant.add_argument(ap)
    args = ap.parse_args()
    # 2026-10-22：run は --variant 必須（上書きを防ぐ）。plot・compare は省略すると重みあり版（design.md §10.1）。
    # 読み先（RQ1 の精度表・RQ2 の精度表）と出力先を、版の種類に合わせて切り替える
    from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants as det
    from src.analysis.preliminary_analysis.feature_space_window.utils import constants as fsw
    v = variant.apply(variant.resolve(args.variant, writing=(args.mode == "run")), constants, det, fsw)
    projects = list(pre.PROJECTS) if args.project in (None, "all") else [args.project]
    if args.mode == "run":      # GPU を使わないので CPU だけで並べる数を決める（design.md §6。2026-10-21）
        w, _frac = parallel.decide(len(projects), use_gpu=False, workers=args.workers)
        if w > 1:
            child = ["--mode", "run", "--kind", args.kind, "--variant", v, "--groups", *args.groups]
            if args.versions:
                child += ["--versions", *args.versions]
            if args.cells:
                child += ["--cells", *args.cells]
            failed = parallel.run_children("src.analysis.preliminary_analysis.data_drift.main", projects, child, w, None,
                                           variant.log_name(f"data_drift_{args.kind}", v))
            sys.exit(1 if failed else 0)
    for project in projects:
        versions = args.versions or list(pre.project_config(project)["versions"])
        if args.mode == "compare":
            logger.info(f"===== 比べる折れ線 {project}（{len(versions)} 版）=====")
            compare(project, versions)
            continue
        logger.info(f"===== {args.kind} {project}（{len(versions)} 版）=====")
        if args.mode == "run":
            run(args.kind, project, versions)
        draw(args.kind, project, versions, args.cells, args.groups)


if __name__ == "__main__":
    main()
