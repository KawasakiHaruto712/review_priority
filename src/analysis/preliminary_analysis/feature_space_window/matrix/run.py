"""特徴のずれ×評価時期の精度表を 1 リポジトリ分回して保存する（feature_space_window/design.md §9.2）。

出力：<OUTPUT_ROOT>/<project>/feature_distance/
  <版>/{probe,general,diff}/<metric>/drift_matrix.csv・.json・図、<版>/daily_metrics.csv.gz、<版>/train_days.csv.gz、
  probes/seed<k>.npz（すべての probe）、run_meta.json
"""
from __future__ import annotations

import json
import logging
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.background_problem.common.data_loader import load_release_dates
from src.analysis.preliminary_analysis.feature_space_window.dataset import candidates
from src.analysis.preliminary_analysis.feature_space_window.evaluation import metrics
from src.analysis.preliminary_analysis.feature_space_window.matrix import binning, builder, plotter, writer
from src.analysis.preliminary_analysis.feature_space_window.matrix.aggregate import assemble
from src.analysis.preliminary_analysis.feature_space_window.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.io import store
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre
from src.analysis.preliminary_analysis.pretrained_encoders.utils import variant

logger = logging.getLogger(__name__)
KIND = "feature_distance"


def out_root(project: str) -> Path:
    return constants.OUTPUT_ROOT / project / KIND


def settings_for(project: str) -> tuple[int, int]:
    """(k, 刻み)。どちらも本人が結果を見て constants.MATRIX_SETTINGS に書く（§9.1）。未記入なら止める。"""
    s = constants.matrix_settings().get(project) or {}        # 2026-10-22：版の種類で切り替える（重みあり版は MATRIX_SETTINGS_POSW）
    k, step = s.get("size_days"), s.get("step_days")
    if constants.fraction_grid():                              # 重みあり版の 3 × 3：刻みは版ごとに周期 ÷ 3 で決めるので k だけ要る（§10.1）
        if not k:
            raise ValueError(f"[{project}] constants.MATRIX_SETTINGS_POSW に size_days（大きさの分析（重みあり版）で選んだ k）を書いてください")
        return int(k), None
    if not k or not step:
        raise ValueError(f"[{project}] constants.MATRIX_SETTINGS（重みあり版は MATRIX_SETTINGS_POSW）に size_days（大きさの分析で選んだ k）と "
                         f"step_days（RQ1 の精度表と同じ刻み）を書いてください（いま size_days={k}・step_days={step}）")
    return int(k), int(step)


def version_grids(rel_df, project: str, versions: list[str], step: int, day_sets: dict) -> dict:
    """{版 -> (positions, N_v)}。列は RQ1 と同じ区切り（binning の複製）。集合のない日は除く。"""
    out = {}
    for v in versions:
        try:
            cs_R, ce_R, _ = binning.target_cycles(rel_df, project, v)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {v}]: {e}")
            continue
        grid = max(1, math.ceil(binning.cycle_days(cs_R, ce_R, constants.BIN_DAY_ALIGNED) / step))
        pos = binning.position_of_days(ce_R, step, grid, constants.BIN_DAY_ALIGNED)
        out[v] = ({d: p for d, p in pos.items() if d in day_sets}, grid)
    return out


def _should_plot(metric: str) -> bool:
    base = metric[:-5] if metric.endswith("_diff") else metric
    return base in constants.PLOT_METRICS


def save_probes(res: builder.SeedResults, project: str) -> list[Path]:
    """すべての probe を seed ごとに 1 ファイル（評価日・行・129 個の重み）。"""
    d = out_root(project) / "probes"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed_idx, items in res.probes.items():
        if not items:
            continue
        path = d / f"seed{seed_idx}.npz"
        np.savez(path, days=np.array([x.isoformat() for x, _g, _w in items]), rows=np.array([g for _x, g, _w in items]),
                 weights=np.stack([w for _x, _g, w in items]))
        paths.append(path)
    return paths


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def fraction_grids(rel_df, project: str, versions: list[str], day_sets: dict) -> dict:
    """重みあり版の 3 × 3（2026-10-22。§10.1）：{版 -> (positions, 3, 刻み)}。列は周期の 3 等分、刻み ＝ 周期 ÷ 3（版ごと）。"""
    out = {}
    parts = constants.FRACTION_PARTS
    for v in versions:
        try:
            cs_R, ce_R, _ = binning.target_cycles(rel_df, project, v)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {v}]: {e}")
            continue
        cycle = binning.cycle_days(cs_R, ce_R, constants.BIN_DAY_ALIGNED)
        pos = binning.position_by_fraction(ce_R, cycle, parts, constants.BIN_DAY_ALIGNED)
        out[v] = ({d: p for d, p in pos.items() if d in day_sets}, parts, binning.fraction_step(cycle, parts))
    return out


def _merge_into(acc: dict, vals: dict) -> None:
    """{指標 -> {セル -> {評価日 -> {seed -> 値}}}} を足し合わせる（5 版まとめ用。評価日は版の間で重ならない）。"""
    for m, by_cell in vals.items():
        for cell, by_day in by_cell.items():
            acc[m].setdefault(cell, {}).update(by_day)


def run_project(project: str, versions: list[str], n_seeds: int | None = None) -> None:
    k, step = settings_for(project)
    rel_df = load_release_dates()
    day_sets, _pre = candidates.load_all_day_sets(project, rel_df)
    thirds = constants.fraction_grid()
    row_offsets: dict | None = None
    if thirds:      # 重みあり版：行 g は近い順で「刻み × g」から k 日（刻みは X が属する版の周期 ÷ 3）
        fg = fraction_grids(rel_df, project, versions, day_sets)
        grids = {v: (pos, grid) for v, (pos, grid, _s) in fg.items()}
        row_offsets = {}
        for _v, (pos, grid, vstep) in fg.items():
            for x in pos:
                row_offsets[x] = [vstep * g for g in range(grid)]
    else:
        grids = version_grids(rel_df, project, versions, step, day_sets)
    eval_rows: dict = {}
    for _v, (pos, grid) in grids.items():
        for x in pos:
            eval_rows[x] = max(eval_rows.get(x, 0), grid)
    device = st.resolve_device()
    cutoff = constants.cutoff_for(project)
    seeds = store.list_seeds(project, cutoff)
    seeds = seeds if n_seeds is None else seeds[:n_seeds]
    if not seeds:
        raise RuntimeError(f"[{project}] 事前学習済みのエンコーダがありません（cutoff={cutoff}）")
    step_note = ("周期 ÷ 3（版ごと " + ", ".join(f"{v}:{s}" for v, (_p, _g, s) in fg.items()) + "）") if thirds else str(step)
    logger.info(f"[{project}] 特徴のずれ×評価時期の精度表：k={k}・刻み={step_note}・評価日 {len(eval_rows)}・seed {seeds}・device={device}")
    res = builder.SeedResults()
    names = metrics.metric_columns(constants.K_LIST)
    for s in seeds:
        enc, gh, scaler, _cfg = store.load_pretrained(project, cutoff, s, device)
        builder.run_seed(res, day_sets, eval_rows, enc, gh, scaler, s, k, device, names, row_offsets=row_offsets)
        del enc, gh
        st.release_device_cache(device)
    merged_p: dict = {m: {} for m in names}       # 5 版をまとめた 3 × 3 の材料（重みあり版。§10.1。2026-10-22）
    merged_g: dict = {m: {} for m in names}
    for v, (pos, grid) in grids.items():
        probe_vals, gen_vals, rows = builder.version_values(res, pos, grid, names)
        mats = assemble(probe_vals, gen_vals, grid, names)
        if thirds:
            _merge_into(merged_p, probe_vals)
            _merge_into(merged_g, gen_vals)
        base = out_root(project) / v
        variant.warn_if_exists(base, f"{project} {v}")                # 上書きの警告（design.md §10.1。2026-10-22）
        meta = {"project": project, "version": v, "kind": KIND, "size_days": k,
                "step_days": fg[v][2] if thirds else step,                    # 重みあり版は版ごとの刻み（周期 ÷ 3）
                "n_repeats": len(seeds), "device": str(device), "variant": pre.ENCODER_VARIANT}
        for kind in ("probe", "general", "diff"):
            for m, mr in mats[kind].items():
                writer.write_matrix(mr, meta, base / kind / m)
                if _should_plot(m):
                    plotter.plot_all(mr, base / kind / m, constants.PLOT_DPI)
        writer.write_daily_metrics(rows, {**meta, "grid": grid}, base)
        td = [(x.isoformat(), g, s, d.isoformat()) for (x, g, s, d) in res.train_days if x in pos and g < grid]
        pd.DataFrame(td, columns=["day", "row", "seed", "train_day"]).to_csv(base / "train_days.csv.gz", index=False,
                                                                           compression="gzip")
        logger.info(f"保存: {base}（{grid}×{grid}）")
    if thirds and grids:            # 5 版をまとめた 3 × 3（全評価日の平均 ± 標準偏差。§10.1）
        grid = constants.FRACTION_PARTS
        mats = assemble(merged_p, merged_g, grid, names)
        mbase = out_root(project) / constants.MERGED_DIRNAME
        mmeta = {"project": project, "version": constants.MERGED_DIRNAME, "versions": list(grids), "kind": KIND,
                 "size_days": k, "n_repeats": len(seeds), "device": str(device), "variant": pre.ENCODER_VARIANT}
        for kind in ("probe", "general", "diff"):
            for m, mr in mats[kind].items():
                writer.write_matrix(mr, mmeta, mbase / kind / m)
                if _should_plot(m):
                    plotter.plot_all(mr, mbase / kind / m, constants.PLOT_DPI)
        logger.info(f"5 版をまとめた表: {mbase}")
    logger.info(f"probe: {save_probes(res, project)}")
    with open(out_root(project) / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump({"project": project, "kind": KIND, "variant": pre.ENCODER_VARIANT, "size_days": k,
                   "step_days": ({v: s for v, (_p, _g, s) in fg.items()} if thirds else step), "seeds": seeds,
                   "versions": {v: g for v, (_p, g) in grids.items()}, "device": str(device),
                   "cuda_memory_fraction": _fraction(),
                   "machine": {"host": platform.node(), "platform": platform.platform()},
                   "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit()},
                  f, ensure_ascii=False, indent=2)


def _fraction():
    from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre
    return pre.CUDA_MEMORY_FRACTION
