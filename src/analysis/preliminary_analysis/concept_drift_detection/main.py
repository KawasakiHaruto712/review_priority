"""レビュー優先順位ドリフト検出（Phase1 / step2）: オーケストレーション（design.md）。

実行:
    python -m src.analysis.preliminary_analysis.concept_drift_detection.main

流れ（上から順に追える構成）:
  1. データ読み込み（changes / release_dates / bot 名 / 特徴用 DataFrame）
  2. 事前学習エンコーダ＋汎用ヘッドを pretrained_encoders から load（不足 seed は自動作成。§6.1）
  3. 各対象リリースで:
       レコード生成（範囲は §4.4 で逆算）→ 日ごとの集合 → 位置（列）の割当（§4.3）
       → 距離×位置行列（probe / 汎用ヘッド / 差分）を**日次スライド学習**で構築（§4.1, §6.2, §8.3）
       → ドリフト検定（§8.2）→ 保存・作図（N×N ヒートマップ。§10）
出力: <project>/<model>/<version>/{probe,general,diff}/<metric>/ ＋ daily_metrics ＋ summary。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.analysis.background_problem.common.data_loader import load_changes, load_release_dates
from src.analysis.preliminary_analysis.concept_drift_detection.dataset import binning
from src.analysis.preliminary_analysis.concept_drift_detection.evaluation import (
    drift_detector, drift_matrix, metrics,
)
from src.analysis.preliminary_analysis.concept_drift_detection.io import result_writer
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants
from src.analysis.preliminary_analysis.concept_drift_detection.visualization import plotter
from src.analysis.preliminary_analysis.pretrained_encoders import build_encoders
from src.analysis.preliminary_analysis.pretrained_encoders.dataset import record_builder, set_builder
from src.analysis.preliminary_analysis.pretrained_encoders.features import feature_builder
from src.analysis.preliminary_analysis.pretrained_encoders.io import store
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.utils import review_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _resolve_encoders(project: str, device):
    """pretrained_encoders から N_REPEATS 個の (encoder, general_head) を load（不足は自動作成）。

    scaler は全 seed で同一（同じ事前学習データで fit）なので先頭のものを共有で使う。
    cutoff は project ごとに異なる（constants.PROJECTS）。
    """
    cutoff = constants.cutoff_for(project)
    saved = store.list_seeds(project, cutoff)
    target = constants.N_REPEATS
    missing = [k for k in range(target) if k not in saved]
    if missing:
        logger.info(f"事前学習モデルが不足（保存 {len(saved)} / 要求 {target}）。不足 seed{missing} を作成します…")
        build_encoders.build_and_save(project=project, seed_indices=missing)
        saved = store.list_seeds(project, cutoff)
    seeds = saved[:target]
    encoders, scaler = [], None
    for s in seeds:
        enc, gh, sc, _cfg = store.load_pretrained(project, cutoff, s, device)
        encoders.append((enc, gh))
        if scaler is None:
            scaler = sc
    logger.info(f"load したエンコーダ数: {len(encoders)}（seed={seeds}）")
    return encoders, scaler


def _should_plot(metric: str) -> bool:
    """この指標のヒートマップを描くか（既定は AUC のみ。差分 '*_diff' も base で判定）。"""
    base = metric[:-5] if metric.endswith("_diff") else metric
    return base in constants.PLOT_METRICS


def _save_kind(res_by_metric: dict, base_dir: Path, meta: dict) -> None:
    """probe/general/diff いずれかの {metric: MatrixResult} を保存（全指標）・作図（PLOT_METRICS のみ）。"""
    for m, mr in res_by_metric.items():
        out = base_dir / m
        result_writer.write_matrix(mr, meta, out)
        if _should_plot(m):
            plotter.plot_all(mr, out, constants.PLOT_DPI)


def analyze(changes, rel_df, project, versions, out_root, bot_names=None):
    """1 プロジェクトの全リリースを実行して出力する（テスト可能なコア）。"""
    out_root = Path(out_root)
    bot_names = review_utils.load_bot_names() if bot_names is None else bot_names
    all_prs = feature_builder.build_all_prs_df(changes)
    device = st.resolve_device()
    logger.info(f"device = {device}")
    model_name = constants.MODEL_NAME[0]

    # 2. 事前学習エンコーダ＋汎用ヘッドを load（不足は自動作成）
    encoders, scaler = _resolve_encoders(project, device)

    metric_names = metrics.metric_columns(constants.K_LIST)
    summary = {m: [] for m in constants.PLOT_METRICS}
    # 窓長・刻み・格子サイズは project ごと（§4.2。step1 で選んだ最良窓に基づく）
    window = constants.window_for(project)
    step = constants.step_for(project)
    grid = constants.grid_for(project)
    logger.info(f"格子: {grid}×{grid}（窓長 {window} 日・刻み {step} 日・スパン {step * (grid - 1)} 日）")
    for version in versions:
        try:
            cs_R, ce_R, _cs_prev = binning.target_cycles(rel_df, project, version)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {version}]: {e}")
            continue
        # 3. レコード生成（範囲は §4.4 で逆算）→ 日ごとの集合 → 位置（列）の割当
        rec_start = binning.record_start(cs_R, window, step, grid, constants.RECORD_MARGIN_DAYS)
        logger.info(f"--- {project} {version}（サイクル {cs_R.date()}〜{ce_R.date()} / "
                    f"レコード生成 {rec_start.date()} から）---")
        records = record_builder.build_records(changes, project, rec_start, ce_R,
                                               bot_names, all_prs, rel_df)
        day_sets = {s.t.date(): s for s in set_builder.build_sets(records, constants.MAX_SET_SIZE)}
        positions = binning.position_of_days(cs_R, ce_R, grid, constants.BIN_DAY_ALIGNED)
        eval_dates = [d for d in sorted(day_sets) if d in positions]
        logger.info(f"評価日数: {len(eval_dates)} / 全日集合: {len(day_sets)}")

        # 距離×位置行列（probe / 汎用 / 差分）を日次スライド学習で構築
        daily_sink = [] if constants.SAVE_DAILY_METRICS else None
        res = drift_matrix.build_matrices(day_sets, eval_dates, positions,
                                          encoders=encoders, scaler=scaler, grid=grid,
                                          window_days=window, step_days=step,
                                          daily_sink=daily_sink)

        base = out_root / project / model_name / version
        meta = {"project": project, "version": version, "model": model_name,
                "n_repeats": len(encoders)}
        _save_kind(res["probe"], base / "probe", meta)
        _save_kind(res["general"], base / "general", meta)
        _save_kind(res["diff"], base / "diff", meta)

        # ドリフト検定は主指標（PLOT_METRICS）の probe 行列に対して実施（§8.2）
        for m in constants.PLOT_METRICS:
            mr = res["probe"].get(m)
            if mr is None:
                continue
            dr = drift_detector.detect_drift(mr, constants.PERMUTATION_N, constants.SIGNIFICANCE)
            result_writer.write_drift_test(dr, base / "probe" / m)
            summary[m].append({"version": version, "drift_exists": dr["drift_exists"],
                               "min_p_value": dr["min_p_value"]})

        # (評価日, 距離, seed) ごとの指標を保存（位置のまとめ直し用。§10）
        if daily_sink is not None:
            path = result_writer.write_daily_metrics(daily_sink, {**meta, "grid": grid,
                                                                  "window_days": window,
                                                                  "step_days": step}, base)
            logger.info(f"日次指標を保存: {path}（{len(daily_sink)} 行）")

    # リリース横断の本数集計（§8.2）
    for m, per_version in summary.items():
        if per_version:
            result_writer.write_summary(per_version, {"project": project, "model": model_name, "metric": m},
                                        out_root / project / model_name / "summary" / m)


def run() -> None:
    """constants.PROJECTS の全プロジェクト×5版を実行する（§2）。"""
    rel_df = load_release_dates()
    for project in constants.PROJECTS:
        logger.info(f"===== プロジェクト: {project} =====")
        changes = load_changes(project)
        analyze(changes, rel_df, project, constants.versions_for(project), constants.OUTPUT_ROOT)


if __name__ == "__main__":
    run()
