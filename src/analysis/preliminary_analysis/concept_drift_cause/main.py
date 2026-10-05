"""特徴量の寄与度分析：精度変動の原因をモデル側から説明する（concept_drift_cause / design.md）。

実行:
    python -m src.analysis.preliminary_analysis.concept_drift_cause.main --mode compute
    python -m src.analysis.preliminary_analysis.concept_drift_cause.main --mode extract
    python -m src.analysis.preliminary_analysis.concept_drift_cause.main --mode plot
    python -m ...main --mode compute --project keystone            # 1 プロジェクトだけ

  対象セルは constants.IG_TARGETS（距離と位置の直積。"all" 可）。CLI で上書きできる。
  **距離×時期行列の分析を先に回し終えている必要がある**（daily_metrics と保存済み probe を使う）。

流れ:
  1. 距離×時期行列の分析の daily_metrics を読み、(評価日, 距離) ごとに **AUC が中央だった seed** を特定（§4）
  2. その seed の probe を距離×時期行列の分析の保存物から復元し、**実際の評価日のデータ**で IG を計算（§2）
  3. 正解ラベルで正例群／負例群に分け、15 次元プロファイルを ig_daily.csv.gz に追記（§3, §9）
  4. 同じ seed の汎用ヘッドでも計算（距離に依存しないので 1 日につき seed の種類数ぶん。§5）
  5. 抽出（--mode extract）で セル×特徴量の表を 4 つ書く（§9.3）
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.background_problem.common.data_loader import load_changes, load_release_dates
from src.analysis.preliminary_analysis.concept_drift_cause.attribution import integrated_gradients as ig
from src.analysis.preliminary_analysis.concept_drift_cause.extraction import tables
from src.analysis.preliminary_analysis.concept_drift_cause.io import ig_store
from src.analysis.preliminary_analysis.concept_drift_cause.utils import constants
from src.analysis.preliminary_analysis.concept_drift_cause.visualization import plotter, profiles
from src.analysis.preliminary_analysis.concept_drift_detection.dataset import binning
from src.analysis.preliminary_analysis.concept_drift_detection.io import probe_store, result_writer
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants as detection
from src.analysis.preliminary_analysis.pretrained_encoders.dataset import record_builder, set_builder
from src.analysis.preliminary_analysis.pretrained_encoders.features import feature_builder
from src.analysis.preliminary_analysis.pretrained_encoders.io import store
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.model.set_transformer import FEATURE_NAMES
from src.analysis.preliminary_analysis.pretrained_encoders.utils import review_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


# ──距離×時期行列の分析の結果から「そのセル値を出したモデル」を特定する（§4） ──────────────────
def median_seed(sub: pd.DataFrame, metric: str) -> int | None:
    """その (評価日, 距離) で metric が中央（5 個中 3 番目）だった seed。

    `N_REPEATS` が奇数なので中央値は実在する 1 個のモデルの値になる。
    同値のときは **seed 番号の小さい方**（`concept_drift_detection/design.md` §8.1 と同じ規則）。
    """
    s = sub.dropna(subset=[metric])
    if s.empty:
        return None
    med = s[metric].median()
    tied = s[s[metric] == med]
    if not tied.empty:
        return int(tied["seed"].min())
    # 偶数個などで中央値がどの行とも一致しない場合は、並べた真ん中の行を採る
    return int(s.sort_values([metric, "seed"], kind="stable").iloc[len(s) // 2]["seed"])


def select_seeds(daily: pd.DataFrame, metric: str) -> dict:
    """{(評価日(ISO), 距離(str)) -> seed}。欠測の行は使わない。"""
    d = daily[daily["missing"] == 0].copy()
    d["distance"] = d["distance"].astype(str)
    out = {}
    for (day, dist), sub in d.groupby(["day", "distance"]):
        seed = median_seed(sub, metric)
        if seed is not None:
            out[(str(day), str(dist))] = seed
    return out


def _seed_count_ok(daily: pd.DataFrame, project: str, version: str) -> bool:
    """距離×時期行列の分析の結果が**今の設定（奇数 seed）で作り直されたもの**かを確かめる。

    距離×時期行列の分析を回し直す前の古い結果（10 seed）が残っていると、中央値が 5 番目と 6 番目の平均に
    なり **どのモデルの値でもなくなる**。それに気づかないまま特徴量の寄与度分析を回すと、
    「そのセル値を出したモデルを見る」という前提が黙って崩れる（design.md §4）。
    """
    n = int(daily[daily["missing"] == 0]["seed"].nunique())
    if n == detection.N_REPEATS:
        return True
    logger.error(f"スキップ: {project} {version} の距離×時期行列の分析の結果は seed {n} 個で作られています"
                 f"（今の設定は {detection.N_REPEATS} 個）。距離×時期行列の分析を回し直してから特徴量の寄与度分析を実行してください。")
    return False


def _rows(run_id: str, day, position: int, distance, seed: int, head: str, prof: dict) -> list[dict]:
    """1 モデルのプロファイルを ig_daily の行（正例群・負例群の 2 行）にする。"""
    out = []
    for group, n_key in (("pos", "n_pos"), ("neg", "n_neg")):
        vec = prof[group]
        if np.all(np.isnan(vec)):
            continue
        row = {"run_id": run_id, "day": day.isoformat(), "position": int(position),
               "distance": distance, "seed": int(seed), "head": head, "group": group,
               "n_changes": int(prof[n_key])}
        row.update({f: float(v) for f, v in zip(FEATURE_NAMES, vec)})
        out.append(row)
    return out


def _resolve_encoders(project: str, device):
    """距離×時期行列の分析と同じ 5 個の (エンコーダ, 汎用ヘッド) を load する（作り直さない）。"""
    cutoff = detection.cutoff_for(project)
    saved = store.list_seeds(project, cutoff)[:detection.N_REPEATS]
    encoders, scaler = [], None
    for s in saved:
        enc, gh, sc, _cfg = store.load_pretrained(project, cutoff, s, device)
        st.freeze_for_gradients(enc, gh)     # 入力側だけ勾配を流す（§2）
        encoders.append((enc, gh))
        if scaler is None:
            scaler = sc
    logger.info(f"load したエンコーダ数: {len(encoders)}（seed={saved}）")
    return encoders, scaler


def compute_version(project: str, version: str, spec: dict, *, changes, rel_df, encoders,
                    scaler, device, bot_names, all_prs, run_id: str) -> int:
    """1 版ぶんの IG を計算して保存する。返り値は書いた行数。"""
    window, step, grid = detection.window_for(project), detection.step_for(project), detection.grid_for(project)
    base2 = constants.DETECTION_ROOT / project / constants.MODEL_NAME / version
    dpath = base2 / "daily_metrics.csv.gz"
    if not dpath.exists():
        logger.warning(f"スキップ（距離×時期行列の分析の結果がありません）: {project} {version} → {dpath}")
        return 0
    daily = result_writer.load_daily_metrics(dpath)
    if not _seed_count_ok(daily, project, version):
        return 0
    chosen = select_seeds(daily, constants.SELECT_METRIC)

    distances = constants.resolve_axis(spec.get("distances"), grid)
    pspec = spec.get("positions")
    target_pos = set(range(1, grid + 1)) if pspec in ("all", None) else {int(v) for v in pspec}

    # 距離×時期行列の分析と同じ手順で 日ごとの集合と位置を作る
    cs_R, ce_R, _prev = binning.target_cycles(rel_df, project, version)
    rec_start = binning.record_start(cs_R, window, step, grid, detection.RECORD_MARGIN_DAYS)
    logger.info(f"--- {project} {version}（サイクル {cs_R.date()}〜{ce_R.date()} / "
                f"レコード生成 {rec_start.date()} から / 距離 {len(distances)} × 位置 {len(target_pos)}）---")
    records = record_builder.build_records(changes, project, rec_start, ce_R, bot_names, all_prs, rel_df)
    day_sets = {s.t.date(): s for s in set_builder.build_sets(records, detection.MAX_SET_SIZE)}
    pos_of = binning.position_of_days(cs_R, ce_R, grid, detection.BIN_DAY_ALIGNED)

    probes = {k: probe_store.load_probes(constants.DETECTION_ROOT, project, window, k)
              for k in range(len(encoders))}
    if not any(probes.values()):
        logger.warning(f"スキップ（保存済み probe がありません）: {project} 窓長 {window} 日")
        return 0

    eval_days = [d for d in sorted(day_sets) if d in pos_of and (pos_of[d] + 1) in target_pos]
    logger.info(f"対象の評価日: {len(eval_days)} 日")
    rows, missing_probe = [], 0
    for n, X in enumerate(eval_days, 1):
        eval_set = day_sets[X]
        p1 = pos_of[X] + 1
        feats_std = st.standardize_feats(scaler, eval_set.feats)
        used_seeds = set()
        for d in distances:
            seed = chosen.get((X.isoformat(), str(d)))
            if seed is None:
                continue                      # 距離×時期行列の分析で欠測だった日・距離
            vec = probes.get(seed, {}).get(X - timedelta(days=1 + step * d))
            if vec is None:
                missing_probe += 1
                continue
            head = st.head_from_vector(vec, device)
            st.freeze_for_gradients(head)
            prof = ig.day_profiles(encoders[seed][0], head, feats_std, eval_set.labels,
                                   constants.IG_STEPS, device)
            rows += _rows(run_id, X, p1, d, seed, "probe", prof)
            used_seeds.add(seed)
        # 汎用ヘッドは距離に依存しないので、その日に使われた seed の種類ぶんだけ（§5.5）
        for seed in sorted(used_seeds):
            prof = ig.day_profiles(encoders[seed][0], encoders[seed][1], feats_std,
                                   eval_set.labels, constants.IG_STEPS, device)
            rows += _rows(run_id, X, p1, "general", seed, "general", prof)
        if n % 10 == 0 or n == len(eval_days):
            logger.info(f"  {n}/{len(eval_days)} 日（{len(rows)} 行）")
    if missing_probe:
        logger.warning(f"probe が見つからず飛ばした (日, 距離): {missing_probe} 件")

    out_dir = constants.OUTPUT_ROOT / project / constants.MODEL_NAME / version
    ig_store.write_daily(rows, out_dir, run_id,
                         ig_store.run_conditions({"project": project, "version": version,
                                                  "distances": distances,
                                                  "positions": sorted(target_pos)}))
    return len(rows)


def specs_for(project: str, versions=None, all_versions: bool = False,
              older_versions: bool = False) -> dict:
    """その project で回す {版 -> 対象セルの指定}。

    既定は `constants.IG_TARGETS`（各プロジェクトの最新版）。CLI で上書きできる：
      `--all-versions`   … 5 版すべて
      `--older-versions` … **最新版を除く 4 版**（最新版を回し終えたあとに足す用）
      `--version V ...`  … 指定した版だけ
    IG_TARGETS に載っていない版を指定したときは「全距離 × 全位置」を既定とする。
    """
    default = constants.targets_for(project)
    known = detection.versions_for(project)
    if all_versions:
        wanted = list(known)
    elif older_versions:
        wanted = constants.older_versions(project)
    elif versions:
        wanted = [v for v in known if v in set(versions)]
        for v in set(versions) - set(known):
            logger.warning(f"{project} に版 {v} はありません（無視します）")
    else:
        return default
    return {v: default.get(v, dict(constants.ALL_CELLS)) for v in wanted}


def compute(projects=None, versions=None, all_versions: bool = False,
            older_versions: bool = False) -> None:
    """constants.IG_TARGETS（または引数で絞ったもの）について IG を計算する。"""
    rel_df = load_release_dates()
    run_id = ig_store.new_run_id()
    targets = list(projects) if projects else list(constants.IG_TARGETS)
    logger.info(f"run_id = {run_id}")
    for project in targets:
        spec_by_version = specs_for(project, versions, all_versions, older_versions)
        if not spec_by_version:
            logger.warning(f"スキップ（対象の版がありません）: {project}")
            continue
        logger.info(f"===== プロジェクト: {project}（版 {', '.join(spec_by_version)}）=====")
        device = st.resolve_device()
        encoders, scaler = _resolve_encoders(project, device)
        changes = load_changes(project)
        bot_names = review_utils.load_bot_names()
        all_prs = feature_builder.build_all_prs_df(changes, bot_names)
        for version, spec in spec_by_version.items():
            compute_version(project, version, spec, changes=changes, rel_df=rel_df,
                            encoders=encoders, scaler=scaler, device=device,
                            bot_names=bot_names, all_prs=all_prs, run_id=run_id)


def extract(projects=None, versions=None, name: str = "default",
            all_versions: bool = False, older_versions: bool = False) -> None:
    """保存済みの ig_daily から セル×特徴量の表を書く（モデルは動かさない）。"""
    targets = list(projects) if projects else list(constants.IG_TARGETS)
    for project in targets:
        for version in specs_for(project, versions, all_versions, older_versions):
            out_dir = constants.OUTPUT_ROOT / project / constants.MODEL_NAME / version
            daily = ig_store.load_daily(out_dir)
            if daily.empty:
                logger.warning(f"スキップ（IG の結果がありません）: {project} {version}")
                continue
            dpath = constants.DETECTION_ROOT / project / constants.MODEL_NAME / version / "daily_metrics.csv.gz"
            metrics_df = result_writer.load_daily_metrics(dpath)
            paths = tables.write_tables(daily, metrics_df, out_dir / constants.EXTRACT_DIRNAME / name)
            logger.info(f"[{project} {version}] 表を書きました: {len(paths)} ファイル → "
                        f"{out_dir / constants.EXTRACT_DIRNAME / name}")


def plot(projects=None, versions=None, all_versions: bool = False,
         older_versions: bool = False) -> None:
    """保存済みの ig_daily から図を書く（モデルは動かさない。design.md §9.4）。

    個別セル図と格子一覧を**ペアで**出す。入力は `ig_daily.csv.gz` と距離×時期行列の
    `daily_metrics.csv.gz` の 2 つだけで、`--mode extract` の出力には依存しない（§9.4.9）。
    """
    plotter.configure()
    targets = list(projects) if projects else list(constants.IG_TARGETS)
    for project in targets:
        for version in specs_for(project, versions, all_versions, older_versions):
            daily = profiles.load_daily(project, version)
            metrics = profiles.load_metrics(project, version)
            if daily is None or daily.empty or metrics is None:
                logger.warning(f"スキップ: {project} {version}")
                continue
            base = (constants.OUTPUT_ROOT / project / constants.MODEL_NAME / version
                    / constants.FIGURE_DIRNAME)

            for series in constants.PLOT_SERIES:
                # 軸はプロジェクト × 系列ごとに共通にするため、群をまたいで先に決める（§9.4.5）
                built = {}
                for group in constants.PLOT_GROUPS:
                    rows = profiles.build_series(daily, series, group)
                    if rows is None:
                        continue
                    built[group] = (rows, profiles.cell_stats(rows),
                                    profiles.cell_info(metrics, rows, group))
                for group, (rows, stats, info) in built.items():
                    lim = profiles.axis_limit([stats])
                    tag = f"{series}_{group}"
                    dists = sorted(stats["distance"].astype(int).unique())
                    poss = sorted(stats["position"].astype(int).unique())

                    n_cell = 0
                    for d in dists:
                        for p in poss:
                            if plotter.plot_cell(stats, info, project, version, series, group,
                                                 d, p, lim,
                                                 base / "by_cell" / tag / f"d{d:02d}_p{p:02d}.png"):
                                n_cell += 1
                    n_grid = 0
                    for d in dists:
                        if plotter.plot_grid(stats, info, project, version, series, group,
                                             fixed="distance", fixed_value=d, panel_values=poss,
                                             lim=lim,
                                             out_path=base / "by_distance" / tag / f"d{d:02d}.png"):
                            n_grid += 1
                    for p in poss:
                        if plotter.plot_grid(stats, info, project, version, series, group,
                                             fixed="position", fixed_value=p, panel_values=dists,
                                             lim=lim,
                                             out_path=base / "by_position" / tag / f"p{p:02d}.png"):
                            n_grid += 1
                    logger.info(f"[{project} {version}] {tag}: 個別 {n_cell} 枚 / 一覧 {n_grid} 枚"
                                f"（軸 ±{lim:.3f}）→ {base}")


def _parse_args():
    ap = argparse.ArgumentParser(description="特徴量の寄与度分析：IG で精度変動の原因をモデル側から説明する")
    ap.add_argument("--mode", choices=["compute", "extract", "plot"], default="compute")
    ap.add_argument("--project", nargs="*", default=None, help="対象プロジェクト（既定 IG_TARGETS 全部）")
    ap.add_argument("--version", nargs="*", default=None, help="対象の版ラベル（既定 IG_TARGETS のもの）")
    ap.add_argument("--all-versions", action="store_true",
                    help="その project の 5 版すべてを対象にする（全セルだと全体で約 188 時間）")
    ap.add_argument("--older-versions", action="store_true",
                    help="**最新版を除く 4 版**を対象にする（最新版を回し終えたあとに足す用。約 173 時間）")
    ap.add_argument("--name", default="default", help="抽出の出力先サブディレクトリ名")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "compute":
        compute(projects=args.project, versions=args.version, all_versions=args.all_versions,
                older_versions=args.older_versions)
    elif args.mode == "plot":
        plot(projects=args.project, versions=args.version, all_versions=args.all_versions,
             older_versions=args.older_versions)
    else:
        extract(projects=args.project, versions=args.version, name=args.name,
                all_versions=args.all_versions, older_versions=args.older_versions)
