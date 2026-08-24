"""レビュー優先順位ドリフト検出（Phase1 / step2）: オーケストレーション（design.md）。

実行:
    python -m src.analysis.preliminary_analysis.concept_drift_detection.main

流れ（上から順に追える構成）:
  1. データ読み込み（changes / release_dates / bot 名 / 特徴用 DataFrame）
  2. 事前学習エンコーダ＋汎用ヘッドを pretrained_encoders から load（不足 seed は自動作成。§6.1）
  3. 各対象リリースで:
       レコード生成 → per-release 26 分割ビン割当（§4）
       → 距離×位置行列（probe / 汎用ヘッド / 差分）を構築（§6.2, §8.3）
       → ドリフト検定（§8.2）→ 保存・作図（26×26 ヒートマップ。§10）
出力: <project>/<model>/<version>/{probe,general,diff}/<metric>/ ＋ predictions ＋ summary。
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
from src.analysis.preliminary_analysis.pretrained_encoders.dataset import record_builder
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
    """
    cutoff = constants.PRETRAINED_CUTOFF
    saved = store.list_seeds(project, cutoff)
    target = constants.N_REPEATS
    missing = [k for k in range(target) if k not in saved]
    if missing:
        logger.info(f"事前学習モデルが不足（保存 {len(saved)} / 要求 {target}）。不足 seed{missing} を作成します…")
        build_encoders.build_and_save(seed_indices=missing)
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
    for version in versions:
        try:
            cs_R, ce_R, _cs_prev = binning.target_cycles(rel_df, project, version)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {version}]: {e}")
            continue
        logger.info(f"--- {project} {version}（サイクル {cs_R.date()}〜{ce_R.date()}）---")
        # 3. レコード生成 → per-release ビン割当
        p_start = binning.pool_start(rel_df, project, version)
        records = record_builder.build_records(changes, project, p_start, ce_R, bot_names, all_prs, rel_df)
        bins = binning.make_local_bins(records, rel_df, project, version,
                                       constants.BIN_COUNT, constants.BIN_DAY_ALIGNED)

        # 距離×位置行列（probe / 汎用 / 差分）
        probe_sink = [] if constants.SAVE_PREDICTIONS else None
        general_sink = [] if constants.SAVE_PREDICTIONS else None
        res = drift_matrix.build_matrices(bins, encoders=encoders, scaler=scaler,
                                          probe_sink=probe_sink, general_sink=general_sink)

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

        if probe_sink is not None:
            result_writer.write_predictions(probe_sink, meta, base)
        if general_sink is not None:
            result_writer.write_general_predictions(general_sink, meta, base)

    # リリース横断の本数集計（§8.2）
    for m, per_version in summary.items():
        if per_version:
            result_writer.write_summary(per_version, {"project": project, "model": model_name, "metric": m},
                                        out_root / project / model_name / "summary" / m)


def run() -> None:
    rel_df = load_release_dates()
    for project, versions in constants.TARGET_PROJECTS.items():
        logger.info(f"===== プロジェクト: {project} =====")
        changes = load_changes(project)
        analyze(changes, rel_df, project, versions, constants.OUTPUT_ROOT)


if __name__ == "__main__":
    run()
