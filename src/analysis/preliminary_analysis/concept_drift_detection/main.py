"""レビュー優先順位ドリフト検出（Phase1）: オーケストレーション（design.md）。

実行:
    python -m src.analysis.preliminary_analysis.concept_drift_detection.main   # 事前学習＋probe＋評価＋作図

流れ（design.md §6, §8。上から順に追える構成にしている＝§14）:
  1. データ読み込み（changes / release_dates / bot 名 / 特徴用 DataFrame）
  2. 共有エンコーダの事前学習（プロジェクト初期〜分析対象の直前。N_REPEATS 個。§6.1）
  3. 各対象リリースで:
       レコード生成 → per-release ビン割当（§4）
       → 距離×時期行列（probe / 汎用ヘッド / 差分）を構築（§6.2, §8.3）
       → ドリフト検定（§8.2）→ 保存・作図（§10）
出力: <project>/<model>/<version>/{probe,general,diff}/<metric>/ ＋ predictions ＋ summary。
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.analysis.background_problem.common.data_loader import (
    load_changes, load_release_dates,
)
from src.analysis.preliminary_analysis.concept_drift_detection.dataset import (
    binning, record_builder, set_builder,
)
from src.analysis.preliminary_analysis.concept_drift_detection.evaluation import drift_detector, drift_matrix
from src.analysis.preliminary_analysis.concept_drift_detection.features import feature_builder
from src.analysis.preliminary_analysis.concept_drift_detection.io import result_writer
from src.analysis.preliminary_analysis.concept_drift_detection.model import set_transformer as st
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants, review_utils
from src.analysis.preliminary_analysis.concept_drift_detection.visualization import plotter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _pretrain_cutoff(rel_df, project, versions) -> datetime:
    """事前学習の締め切り（分析対象の直前）。PRETRAIN_CUTOFF 指定が無ければ最も早い対象の cycle_start。"""
    if constants.PRETRAIN_CUTOFF:
        return datetime.fromisoformat(constants.PRETRAIN_CUTOFF)
    cs_R, _ce, _cs_prev = binning.target_cycles(rel_df, project, versions[0])
    return cs_R


def build_shared_encoders(changes, project, versions, rel_df, bot_names, all_prs, device):
    """共有エンコーダを事前学習する（全リリース共通。N_REPEATS 個＝seed 違い。§6.1）。"""
    cutoff = _pretrain_cutoff(rel_df, project, versions)
    data_start = all_prs["created"].min().to_pydatetime()
    logger.info(f"事前学習データ: {data_start.date()} 〜 {cutoff.date()}（分析対象の直前まで）")
    pre_records = record_builder.build_records(changes, project, data_start, cutoff,
                                               bot_names, all_prs, rel_df)
    pre_sets = set_builder.build_sets(pre_records, constants.MAX_SET_SIZE)
    logger.info(f"事前学習用 集合数: {len(pre_sets)}（総レコード {set_builder.count_records(pre_sets)}）")
    scaler = st.Scaler.fit(pre_sets)
    encoders = []
    for k in range(constants.N_REPEATS):
        logger.info(f"共有エンコーダ 事前学習 {k + 1}/{constants.N_REPEATS}（seed={constants.RANDOM_SEED + k}）")
        enc, gh = st.pretrain(pre_sets, scaler, constants.RANDOM_SEED + k, device)
        encoders.append((enc, gh))
    return encoders, scaler


def _save_kind(res_by_metric: dict, base_dir: Path, meta: dict) -> None:
    """probe/general/diff いずれかの {metric: MatrixResult} を metric ごとに保存・作図する。"""
    for m, mr in res_by_metric.items():
        out = base_dir / m
        result_writer.write_matrix(mr, meta, out)
        plotter.plot_all(mr, out, constants.PLOT_DPI)


def analyze(changes, rel_df, project, versions, out_root, bot_names=None):
    """1 プロジェクトの全リリースを実行して出力する（テスト可能なコア）。"""
    out_root = Path(out_root)
    bot_names = review_utils.load_bot_names() if bot_names is None else bot_names
    all_prs = feature_builder.build_all_prs_df(changes)
    device = st.resolve_device()
    logger.info(f"device = {device}")
    model_name = constants.MODEL_NAME[0]  # 今回は set_transformer のみ

    # 2. 共有エンコーダを事前学習（全リリース共通で1回だけ作る）
    encoders, scaler = build_shared_encoders(changes, project, versions, rel_df, bot_names, all_prs, device)

    summary = {m: [] for m in constants.ENABLED_METRICS}
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

        # 距離×時期行列（probe / 汎用 / 差分）
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

        # ドリフト検定は probe 行列に対して指標ごとに実施（§8.2）
        for m, mr in res["probe"].items():
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
