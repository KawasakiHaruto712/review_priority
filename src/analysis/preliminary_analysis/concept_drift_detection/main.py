"""レビュー優先順位ドリフトの存在確認: オーケストレーション（§5）。

実行:
    python -m src.analysis.preliminary_analysis.concept_drift_detection.main            # 学習＋計算＋出力＋作図
    python -m src.analysis.preliminary_analysis.concept_drift_detection.main recompute  # 保存済み予測から再計算
    python -m src.analysis.preliminary_analysis.concept_drift_detection.main replot     # 保存済み行列から再描画

モデルを最外ループにして「1 モデルで全リリース → 次モデル」で回す。レコード/ビンはモデル非依存なので
リリースごとに 1 回だけ作ってモデル間で使い回す。目的変数は Δ以内2値（constants.TARGET）。
出力は <project>/<model>/<version>/<metric>/ と <project>/<model>/summary/<metric>/。
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from src.analysis.background_problem.common.data_loader import (
    get_release_cycle, load_changes, load_release_dates,
)
from src.analysis.preliminary_analysis.concept_drift_detection.dataset import binning, record_builder
from src.analysis.preliminary_analysis.concept_drift_detection.evaluation import drift_detector, drift_matrix
from src.analysis.preliminary_analysis.concept_drift_detection.features import feature_builder
from src.analysis.preliminary_analysis.concept_drift_detection.io import result_writer
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants, review_utils
from src.analysis.preliminary_analysis.concept_drift_detection.visualization import plotter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _build_bins_for_versions(changes, project, versions, rel_df, bot_names, all_prs):
    """各リリースのビンを 1 回だけ作る（特徴量計算はここで一度。モデル間で使い回す）。

    返り値: version -> (bins, cycle_start, bw_sec)
    """
    out = {}
    for version in versions:
        try:
            cs, ce = get_release_cycle(rel_df, project, version)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {version}]: {e}")
            continue
        pool_start = cs - (ce - cs)  # 当該リリース長ぶん前（§2.5）
        records = record_builder.build_records(changes, project, pool_start, ce,
                                               bot_names, all_prs, rel_df)
        bins = binning.make_bins(records, constants.BIN_COUNT, constants.BINNING, cs, ce)
        eff_cs, bw_sec = binning.resolve_bin_params(cs, ce, constants.BIN_COUNT)
        out[version] = (bins, eff_cs, bw_sec)
        logger.info(f"[{project} {version}] bins={len(bins)} records={len(records)}")
    return out


def analyze(changes, rel_df, project, versions, out_root, bot_names=None):
    """1 プロジェクトの全モデル×全リリース×全指標を実行して出力する（テスト可能なコア）。"""
    out_root = Path(out_root)
    bot_names = review_utils.load_bot_names() if bot_names is None else bot_names
    all_prs = feature_builder.build_all_prs_df(changes)
    metric_names = constants.ENABLED_METRICS

    bins_by_version = _build_bins_for_versions(changes, project, versions, rel_df, bot_names, all_prs)

    for model_name in constants.MODEL_NAME:
        logger.info(f"--- モデル: {model_name} ---")
        results_by_metric = {m: [] for m in metric_names}
        for version, (bins, cs, bw_sec) in bins_by_version.items():
            pred_sink = [] if constants.SAVE_PREDICTIONS else None
            matrices = drift_matrix.build_matrices(
                bins, model_name, cycle_start=cs, bw_sec=bw_sec,
                metric_names=metric_names, pred_sink=pred_sink)
            base_dir = out_root / project / model_name / version
            if pred_sink is not None:
                result_writer.write_predictions(
                    pred_sink,
                    {"project": project, "model": model_name, "version": version,
                     "target": constants.TARGET, "review_horizon_days": constants.REVIEW_HORIZON_DAYS,
                     "bin_count": constants.BIN_COUNT, "n_repeats": constants.N_REPEATS},
                    base_dir)
            for metric, res in matrices.items():
                drift = drift_detector.detect_drift(res, constants.PERMUTATION_N,
                                                    constants.SIGNIFICANCE, seed=constants.RANDOM_SEED)
                out_dir = base_dir / metric
                meta = {"project": project, "model": model_name, "version": version,
                        "target": constants.TARGET, "review_horizon_days": constants.REVIEW_HORIZON_DAYS,
                        "bin_count": constants.BIN_COUNT, "n_train": constants.N_TRAIN,
                        "n_eval": constants.N_EVAL, "n_repeats": constants.N_REPEATS,
                        "repeat_agg": constants.REPEAT_AGG, "save_per_repeat": constants.SAVE_PER_REPEAT}
                result_writer.write_matrix(res, meta, out_dir)
                result_writer.write_drift_test(drift, out_dir)
                plotter.plot_all(res, out_dir, dpi=constants.PLOT_DPI)
                results_by_metric[metric].append(
                    {"version": version, "drift_exists": drift["drift_exists"],
                     "min_p_value": drift["min_p_value"]})
        for metric, per_version in results_by_metric.items():
            result_writer.write_summary(
                per_version, {"project": project, "model": model_name, "metric": metric},
                out_root / project / model_name / "summary" / metric)


def replot(out_root=None) -> int:
    """保存済み drift_matrix.json から図だけ再描画する（再計算なし）。"""
    out_root = Path(out_root or constants.OUTPUT_ROOT)
    count = 0
    for json_path in sorted(out_root.rglob("drift_matrix.json")):
        res = result_writer.load_matrix(json_path)
        plotter.plot_all(res, json_path.parent, dpi=constants.PLOT_DPI)
        count += 1
        logger.info(f"再描画: {json_path.parent}")
    logger.info(f"完了。{count} 件の図を再描画しました（出力先: {out_root}）")
    return count


def recompute(out_root=None) -> int:
    """保存済み predictions.csv.gz から、再学習なしで行列・検定・図を作り直す（単一経路）。"""
    out_root = Path(out_root or constants.OUTPUT_ROOT)
    count = 0
    summary_acc: dict = defaultdict(lambda: defaultdict(list))
    for pred_path in sorted(out_root.rglob("predictions.csv.gz")):
        base = pred_path.parent                       # .../<version>
        meta_path = base / "predictions_meta.json"
        meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else {}
        project = meta.get("project") or base.parent.parent.name
        model_name = meta.get("model") or base.parent.name
        version = meta.get("version") or base.name

        df = result_writer.load_predictions(pred_path)
        rows = list(df.itertuples(index=False, name=None))
        matrices = drift_matrix.compute_matrices(rows, metric_names=constants.ENABLED_METRICS)
        for metric, res in matrices.items():
            drift = drift_detector.detect_drift(res, constants.PERMUTATION_N,
                                                constants.SIGNIFICANCE, seed=constants.RANDOM_SEED)
            out_dir = base / metric
            m = {"project": project, "model": model_name, "version": version,
                 "target": constants.TARGET, "bin_count": constants.BIN_COUNT,
                 "n_repeats": constants.N_REPEATS, "repeat_agg": constants.REPEAT_AGG,
                 "save_per_repeat": constants.SAVE_PER_REPEAT, "recomputed": True}
            result_writer.write_matrix(res, m, out_dir)
            result_writer.write_drift_test(drift, out_dir)
            plotter.plot_all(res, out_dir, dpi=constants.PLOT_DPI)
            summary_acc[(project, model_name)][metric].append(
                {"version": version, "drift_exists": drift["drift_exists"],
                 "min_p_value": drift["min_p_value"]})
        count += 1
        logger.info(f"再計算: {base}")
    for (project, model_name), metrics in summary_acc.items():
        for metric, per_version in metrics.items():
            result_writer.write_summary(
                per_version, {"project": project, "model": model_name, "metric": metric},
                out_root / project / model_name / "summary" / metric)
    logger.info(f"完了。{count} 件の予測から再計算しました（出力先: {out_root}）")
    return count


def run(projects=None):
    """全プロジェクト × 全モデルを実行する（既定は constants の設定）。"""
    rel_df = load_release_dates()
    bot_names = review_utils.load_bot_names()
    projects = projects or constants.TARGET_PROJECTS
    for project, versions in projects.items():
        logger.info(f"=== プロジェクト: {project} ===")
        changes = load_changes(project)
        if not changes:
            logger.warning(f"Change が無いためスキップ: {project}")
            continue
        analyze(changes, rel_df, project, versions, constants.OUTPUT_ROOT, bot_names)
    logger.info(f"完了。出力先: {constants.OUTPUT_ROOT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        replot()
    elif len(sys.argv) > 1 and sys.argv[1] == "recompute":
        recompute()
    else:
        run()
