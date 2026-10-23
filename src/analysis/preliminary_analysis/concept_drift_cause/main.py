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
from src.analysis.preliminary_analysis.concept_drift_cause.attribution import sampling
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
from src.analysis.preliminary_analysis.pretrained_encoders.utils import variant
from src.utils.bot_detection import BotDetector, host_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


# ──距離×時期行列の分析の結果から「そのセル値を出したモデル」を特定する（§4） ──────────────────
def median_seed(sub: pd.DataFrame, metrics) -> int | None:
    """その (評価日, 行) で真ん中（5 個中 3 番目）の seed（design.md §4）。

    ~~metric が中央の seed。同値のときは seed 番号の小さい方。~~
    2026-10-21 改めた：metrics（既定 recall@pos → AUC）と seed 番号の順に、すべて小さい順に並べて 3 番目を選ぶ。
    recall@pos は飛び飛びの値で真ん中の値を持つ seed が複数ありうる（精度表では値が同じなので困らないが、IG はモデルの中身で
    値が変わるので 1 本に決める）。最初の指標で同じなら次の指標、それも同じなら番号で並びが決まる。metrics に文字列 1 つも渡せる。
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    s = sub.dropna(subset=[metrics[0]])
    if s.empty:
        return None
    s = s.sort_values(list(metrics) + ["seed"], kind="stable", na_position="last")
    return int(s.iloc[(len(s) - 1) // 2]["seed"])


def select_seeds(daily: pd.DataFrame, metrics=constants.SELECT_METRICS) -> dict:
    """{(評価日(ISO), 行(str)) -> seed}。欠測の行は使わない。"""
    d = daily[daily["missing"] == 0].copy()
    d["distance"] = d["distance"].astype(str)
    out = {}
    for (day, dist), sub in d.groupby(["day", "distance"]):
        seed = median_seed(sub, metrics)
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


def _sample_rows(run_id: str, day, eval_set, items) -> list[dict]:
    """その日の全 Change を ig_sample の行にする（used ＝ IG の計算に使ったか。items が None なら全件使用）。design.md §11.7。"""
    used = None if items is None else set(items)
    return [{"run_id": run_id, "day": day.isoformat(), "change_id": cid, "label": int(lab),
             "used": used is None or i in used}
            for i, (cid, lab) in enumerate(zip(eval_set.ids, eval_set.labels))]


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


def _load_feature_probes(project: str, seed_idx: int) -> dict:
    """RQ2 の probe（feature_space_window の出力 probes/seed<k>.npz）→ {(評価日, 行 g): 129 個の重み}（design.md §11.5）。"""
    path = constants.FSW_ROOT / project / "feature_distance" / "probes" / f"seed{seed_idx}.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    from datetime import date as _date
    return {(_date.fromisoformat(str(d)), int(g)): w for d, g, w in zip(z["days"], z["rows"], z["weights"])}


def compute_version(project: str, version: str, spec: dict, *, feat_table, rel_df, encoders,
                    scaler, device, run_id: str, kind: str = "time_distance") -> int:
    """1 版ぶんの IG を計算して保存する。返り値は書いた行数。

    kind：time_distance（RQ1）／feature_distance（RQ2。2026-10-21 追加。design.md §11.5）。違うのは日ごとの指標と probe の読み先・引き方だけ。
    """
    window, step = detection.window_for(project), detection.step_for(project)
    # 格子サイズ N_v は版のサイクル長から決まる（距離×時期行列の分析 §4.2 と同じ式。RQ2 も刻みは RQ1 の窓長と同じ）
    cs_R, ce_R, _prev = binning.target_cycles(rel_df, project, version)
    cycle = binning.cycle_days(cs_R, ce_R, detection.BIN_DAY_ALIGNED)
    thirds = detection.fraction_grid()     # 重みあり版は 3 × 3（刻み ＝ 周期 ÷ 3。精度表と同じセル。design.md §11.6。2026-10-22）
    if thirds:
        grid = detection.FRACTION_PARTS
        step = binning.fraction_step(cycle, grid)
    else:
        grid = detection.grid_for(project, cycle)
    dpath = constants.metrics_path(project, version, kind)
    if not dpath.exists():
        logger.warning(f"スキップ（精度表の結果がありません）: {project} {version} {kind} → {dpath}")
        return 0
    daily = result_writer.load_daily_metrics(dpath)
    if not _seed_count_ok(daily, project, version):
        return 0
    chosen = select_seeds(daily, constants.select_metrics())     # 2026-10-22：版の種類で並べる指標を変える（design.md §11.6）

    distances = constants.resolve_axis(spec.get("distances"), grid)
    pspec = spec.get("positions")
    target_pos = set(constants.resolve_axis(pspec, grid))   # 位置も 0 始まり（p0 ＝ リリースの直前）

    # 距離×時期行列の分析と同じ手順で 日ごとの集合と位置を作る
    rec_start = (binning.record_start_fraction(ce_R, cycle, window, step, grid, detection.RECORD_MARGIN_DAYS,
                                               detection.BIN_DAY_ALIGNED) if thirds
                 else binning.record_start(ce_R, window, step, grid, detection.RECORD_MARGIN_DAYS,
                                           detection.BIN_DAY_ALIGNED))
    logger.info(f"--- {project} {version}（サイクル {cs_R.date()}〜{ce_R.date()} / "
                f"レコード生成 {rec_start.date()} から / 距離 {len(distances)} × 位置 {len(target_pos)}）---")
    # 2026-10-18 改めた：feature_table の評価期間の表から、距離×時期行列の分析と同じ範囲を切り出す（design.md §0）
    from src.analysis.preliminary_analysis.feature_table.build.sets import day_sets_from_table
    day_sets = {s.t.date(): s for s in day_sets_from_table(feat_table, rec_start, ce_R,
                                                             max_set_size=detection.MAX_SET_SIZE)}
    pos_of = (binning.position_by_fraction(ce_R, cycle, grid, detection.BIN_DAY_ALIGNED) if thirds
              else binning.position_of_days(ce_R, step, grid, detection.BIN_DAY_ALIGNED))

    if kind == "feature_distance":     # （評価日, 行 g）で引く（design.md §11.5）
        probes = {k: _load_feature_probes(project, k) for k in range(len(encoders))}
    else:                              # 学習の窓の末尾の日で引く
        probes = {k: probe_store.load_probes(constants.DETECTION_ROOT, project, window, k)
                  for k in range(len(encoders))}
    if not any(probes.values()):
        logger.warning(f"スキップ（保存済み probe がありません）: {project} {kind} 窓長 {window} 日")
        return 0

    eval_days = [d for d in sorted(day_sets) if d in pos_of and pos_of[d] in target_pos]
    logger.info(f"対象の評価日: {len(eval_days)} 日（負例の標本 {'あり（n₀=' + str(constants.IG_SAMPLE_N0) + '）' if constants.IG_SAMPLE_NEG else 'なし'}"
                f"・汎用ヘッド {'計算する' if constants.IG_GENERAL_HEAD else '計算しない'}）")
    rows, sample_rows, missing_probe = [], [], 0
    for n, X in enumerate(eval_days, 1):
        eval_set = day_sets[X]
        p0 = pos_of[X]
        feats_std = st.standardize_feats(scaler, eval_set.feats)
        ctx_block = st.context_block_mask(eval_set)   # 文脈のマスク（pretrained_encoders/design.md §12.4）
        # 2026-10-23（design.md §11.7）：IG を計算する Change（正例は全件・負例は標本）。その日の全モデルで同じものを使う。文脈は全件のまま
        items = (sampling.select_items(eval_set.labels, project, X, constants.IG_SAMPLE_N0)
                 if constants.IG_SAMPLE_NEG else None)
        used_seeds = set()
        for d in distances:
            seed = chosen.get((X.isoformat(), str(d)))
            if seed is None:
                continue                      # 距離×時期行列の分析で欠測だった日・距離
            key = (X, int(d)) if kind == "feature_distance" else X - timedelta(days=1 + step * d)
            vec = probes.get(seed, {}).get(key)
            if vec is None:
                missing_probe += 1
                continue
            head = st.head_from_vector(vec, device)
            st.freeze_for_gradients(head)
            prof = ig.day_profiles(encoders[seed][0], head, feats_std, eval_set.labels,
                                   constants.IG_STEPS, device, ctx_block=ctx_block, step_chunk=constants.IG_STEP_CHUNK,
                                   items=items)
            rows += _rows(run_id, X, p0, d, seed, "probe", prof)
            used_seeds.add(seed)
        # 汎用ヘッドは距離に依存しないので、その日に使われた seed の種類ぶんだけ（§5.5）
        # 2026-10-23：計算するかは設定で選ぶ（IG_GENERAL_HEAD。§11.7）
        for seed in (sorted(used_seeds) if constants.IG_GENERAL_HEAD else []):
            prof = ig.day_profiles(encoders[seed][0], encoders[seed][1], feats_std,
                                   eval_set.labels, constants.IG_STEPS, device, ctx_block=ctx_block,
                                   step_chunk=constants.IG_STEP_CHUNK, items=items)
            rows += _rows(run_id, X, p0, "general", seed, "general", prof)
        if used_seeds:      # 1 モデルでも計算した日だけ、その日の全 Change と使ったかを残す（§11.7）
            sample_rows += _sample_rows(run_id, X, eval_set, items)
        if n % 10 == 0 or n == len(eval_days):
            logger.info(f"  {n}/{len(eval_days)} 日（{len(rows)} 行）")
    if missing_probe:
        logger.warning(f"probe が見つからず飛ばした (日, 距離): {missing_probe} 件")

    out_dir = constants.output_dir(project, version, kind)
    if out_dir.exists() and any(out_dir.iterdir()):     # IG の出力は追記式（run ごとに別の記録）なので、上書きではなく追記を知らせる（§11.6）
        logger.warning(f"既存の結果に追記します（版の種類 {variant.LABEL.get(constants.pre_variant(), '')}）：{out_dir}")
    ig_store.write_daily(rows, out_dir, run_id,
                         ig_store.run_conditions({"project": project, "version": version, "kind": kind,
                                                  "distances": distances,
                                                  "positions": sorted(target_pos),
                                                  "device": str(device),     # 回した機械（pretrained_encoders/design.md §14.5）
                                                  "variant": constants.pre_variant()}))   # 版の種類（2026-10-22）
    ig_store.write_sample(sample_rows, out_dir)     # 計算に使った Change の ID（2026-10-23。§11.7）
    return len(rows)


def specs_for(project: str, versions=None, all_versions: bool = False,
              older_versions: bool = False, kind: str = "time_distance") -> dict:
    """その project で回す {版 -> 対象セルの指定}。

    既定は `constants.IG_TARGETS`（各プロジェクトの最新版）。CLI で上書きできる：
      `--all-versions`   … 5 版すべて
      `--older-versions` … **最新版を除く 4 版**（最新版を回し終えたあとに足す用）
      `--version V ...`  … 指定した版だけ
    IG_TARGETS に載っていない版を指定したときは「全距離 × 全位置」を既定とする。
    2026-10-23：重みあり版は精度表の種類ごとの対象（IG_TARGETS_TIME／IG_TARGETS_FEATURE。design.md §11.7）。
    """
    default = constants.targets_for(project, kind)
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
            older_versions: bool = False, kind: str = "time_distance") -> None:
    """constants.IG_TARGETS（または引数で絞ったもの）について IG を計算する。"""
    rel_df = load_release_dates()
    run_id = ig_store.new_run_id()
    targets = list(projects) if projects else list(constants.targets(kind))
    logger.info(f"run_id = {run_id}")
    if not targets:
        logger.warning(f"対象がありません（{kind} の対象の設定が空です。utils/constants.py）")
    for project in targets:
        spec_by_version = specs_for(project, versions, all_versions, older_versions, kind)
        if not spec_by_version:
            logger.warning(f"スキップ（対象の版がありません）: {project}")
            continue
        logger.info(f"===== プロジェクト: {project}（版 {', '.join(spec_by_version)}）=====")
        device = st.resolve_device()
        encoders, scaler = _resolve_encoders(project, device)
        # 特徴量は feature_table の評価期間の表から読む（なければ作り、指紋が違えば止める。feature_table/design.md §2.6）
        from src.analysis.preliminary_analysis.feature_table.build import ensure_table
        feat_table = ensure_table(project, "eval", rel_df=rel_df)
        for version, spec in spec_by_version.items():
            compute_version(project, version, spec, feat_table=feat_table, rel_df=rel_df,
                            encoders=encoders, scaler=scaler, device=device, run_id=run_id, kind=kind)


def extract(projects=None, versions=None, name: str = "default",
            all_versions: bool = False, older_versions: bool = False, kind: str = "time_distance") -> None:
    """保存済みの ig_daily から セル×特徴量の表を書く（モデルは動かさない）。"""
    targets = list(projects) if projects else list(constants.targets(kind))
    for project in targets:
        for version, spec in specs_for(project, versions, all_versions, older_versions, kind).items():
            out_dir = constants.output_dir(project, version, kind)   # 2026-10-21：種類ごとに分ける（§11.5）
            daily = ig_store.load_daily(out_dir)
            if daily.empty:
                logger.warning(f"スキップ（IG の結果がありません）: {project} {version} {kind}")
                continue
            dpath = constants.metrics_path(project, version, kind)
            metrics_df = result_writer.load_daily_metrics(dpath)
            paths = tables.write_tables(daily, metrics_df, out_dir / constants.EXTRACT_DIRNAME / name)
            logger.info(f"[{project} {version}] 表を書きました: {len(paths)} ファイル → "
                        f"{out_dir / constants.EXTRACT_DIRNAME / name}")
            # 2026-10-23（design.md §11.7）：d0 と相手を並べた表（距離と位置を明示した対象だけ）
            cdir = out_dir / constants.EXTRACT_DIRNAME / name / constants.COMPARE_DIRNAME
            cpaths = tables.write_compare_tables(daily, spec, cdir, prefix="g" if kind == "feature_distance" else "d")
            if cpaths:
                logger.info(f"[{project} {version}] d0 と相手を並べた表: {len(cpaths)} ファイル → {cdir}")


def _series_to_plot(daily: pd.DataFrame, project: str, version: str) -> list[str]:
    """描く系列（constants.PLOT_SERIES）。汎用ヘッドの行がなければ、差分・汎用ヘッド単体の系列を probe 単体に切り替える
    （2026-10-23。design.md §11.7。汎用ヘッドを計算しなかった実行）。"""
    series = list(constants.PLOT_SERIES)
    if (daily["head"] == "general").any():
        return series
    out = []
    for s in series:
        s = "probe" if s in ("diff", "general") else s
        if s not in out:
            out.append(s)
    if out != series:
        logger.info(f"[{project} {version}] 汎用ヘッドの行がないため、差分・汎用ヘッドの系列は描かず probe 単体で描きます（{series} → {out}）")
    return out


def plot(projects=None, versions=None, all_versions: bool = False,
         older_versions: bool = False, kind: str = "time_distance") -> None:
    """保存済みの ig_daily から図を書く（モデルは動かさない。design.md §9.4）。

    個別セル図と格子一覧を**ペアで**出す。入力は `ig_daily.csv.gz` と距離×時期行列の
    `daily_metrics.csv.gz` の 2 つだけで、`--mode extract` の出力には依存しない（§9.4.9）。
    """
    plotter.configure()
    targets = list(projects) if projects else list(constants.targets(kind))
    for project in targets:
        for version in specs_for(project, versions, all_versions, older_versions, kind):
            daily = profiles.load_daily(project, version, kind)
            metrics = profiles.load_metrics(project, version, kind)
            if daily is None or daily.empty or metrics is None:
                logger.warning(f"スキップ: {project} {version} {kind}")
                continue
            base = constants.output_dir(project, version, kind) / constants.FIGURE_DIRNAME
            row_label = constants.ROW_LABEL[kind]

            for series in _series_to_plot(daily, project, version):
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
                                                 base / "by_cell" / tag / f"d{d:02d}_p{p:02d}.png",
                                                 row_label=row_label):
                                n_cell += 1
                    n_grid = 0
                    for d in dists:
                        if plotter.plot_grid(stats, info, project, version, series, group,
                                             fixed="distance", fixed_value=d, panel_values=poss,
                                             lim=lim, row_label=row_label,
                                             out_path=base / "by_distance" / tag / f"d{d:02d}.png"):
                            n_grid += 1
                    for p in poss:
                        if plotter.plot_grid(stats, info, project, version, series, group,
                                             fixed="position", fixed_value=p, panel_values=dists,
                                             lim=lim, row_label=row_label,
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
    variant.add_argument(ap)          # 2026-10-22：compute は必須、extract・plot は省略すると重みあり版（design.md §11.6）
    ap.add_argument("--kind", choices=list(constants.KINDS), default="time_distance",
                    help="time_distance＝RQ1（時期のずれ）／feature_distance＝RQ2（特徴のずれ。design.md §11.5。2026-10-21）")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _v = variant.apply(variant.resolve(args.variant, writing=(args.mode == "compute")), constants)
    constants.set_variant_roots(_v)   # 精度表の probe・daily_metrics の読み先も版の種類に合わせる（design.md §11.6）
    if args.mode == "compute":
        compute(projects=args.project, versions=args.version, all_versions=args.all_versions,
                older_versions=args.older_versions, kind=args.kind)
    elif args.mode == "plot":
        plot(projects=args.project, versions=args.version, all_versions=args.all_versions,
             older_versions=args.older_versions, kind=args.kind)
    else:
        extract(projects=args.project, versions=args.version, name=args.name,
                all_versions=args.all_versions, older_versions=args.older_versions, kind=args.kind)
