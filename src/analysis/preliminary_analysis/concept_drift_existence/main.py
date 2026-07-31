"""事前分析① 変化区間の存在確認: オーケストレーション（§5.11）。

実行:
    python -m src.analysis.preliminary_analysis.concept_drift_existence.main

モデルを最外ループにして「1 モデルで全リリース → 次モデル」で回す。レコード/ビンはモデル非依存なので
リリースごとに 1 回だけ作って（特徴量計算はここで一度）モデル間で使い回す。
出力は <project>/<model>/<version>/<target>/<metric>/ と <project>/<model>/summary/<target>/<metric>/。
目的変数 (target) は constants.TARGETS で選ぶ（time_to_next_review / decision_result / 両方）。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.analysis.background_problem.common.data_loader import (
    get_release_cycle, load_changes, load_release_dates,
)
from src.analysis.preliminary_analysis.concept_drift_existence.dataset import binning, record_builder
from src.analysis.preliminary_analysis.concept_drift_existence.evaluation import drift_detector, drift_matrix
from src.analysis.preliminary_analysis.concept_drift_existence.features import feature_builder
from src.analysis.preliminary_analysis.concept_drift_existence.io import result_writer
from src.analysis.preliminary_analysis.concept_drift_existence.utils import constants, review_utils
from src.analysis.preliminary_analysis.concept_drift_existence.visualization import plotter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _build_bins_for_versions(changes, project, versions, rel_df, bot_names, all_prs, targets):
    """各リリースのビンを 1 回だけ作る（特徴量計算はここで一度。全目的変数・モデル間で使い回す）。"""
    bins_by_version = {}
    for version in versions:
        try:
            cs, ce = get_release_cycle(rel_df, project, version)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {version}]: {e}")
            continue
        pool_start = cs - (ce - cs)  # 当該リリース長ぶん前（§2.5, §6.1）
        records = record_builder.build_records(changes, project, pool_start, ce,
                                               bot_names, all_prs, rel_df, targets=targets)
        bins = binning.make_bins(records, constants.BIN_COUNT, constants.BINNING, cs, ce)
        bins_by_version[version] = bins
        logger.info(f"[{project} {version}] bins={len(bins)} records={len(records)}")
    return bins_by_version


def analyze(changes, rel_df, project, versions, out_root, bot_names=None, targets=None):
    """1 プロジェクトの全目的変数×全モデル×全リリース×全指標を実行して出力する（テスト可能なコア）。"""
    out_root = Path(out_root)
    bot_names = review_utils.load_bot_names() if bot_names is None else bot_names
    targets = targets or constants.TARGETS
    all_prs = feature_builder.build_all_prs_df(changes)

    # レコード/ビンは目的変数・モデル非依存（特徴量は共通）。全目的変数ぶんのラベルを付けて1回だけ作る。
    bins_by_version = _build_bins_for_versions(changes, project, versions, rel_df, bot_names, all_prs, targets)

    for target in targets:                          # ← 最外: 目的変数
        spec = constants.TARGET_SPEC[target]
        objective, metric_names, models = spec["objective"], spec["metrics"], spec["models"]
        logger.info(f"=== 目的変数: {target}（{objective}）===")
        for model_name in models:                   # モデル
            logger.info(f"--- モデル: {model_name} ---")
            results_by_metric = {m: [] for m in metric_names}
            for version, bins in bins_by_version.items():
                pred_sink = [] if constants.SAVE_PREDICTIONS else None
                matrices = drift_matrix.build_matrices(
                    bins, model_name, target=target, objective=objective,
                    metric_names=metric_names, pred_sink=pred_sink)
                base_dir = out_root / project / model_name / version / target
                if pred_sink is not None:
                    result_writer.write_predictions(
                        pred_sink,
                        {"project": project, "model": model_name, "version": version,
                         "target": target, "objective": objective,
                         "bin_count": constants.BIN_COUNT, "n_repeats": constants.N_REPEATS},
                        base_dir)
                for metric, res in matrices.items():
                    drift = drift_detector.detect_drift(res, constants.PERMUTATION_N,
                                                        constants.SIGNIFICANCE, seed=constants.RANDOM_SEED)
                    out_dir = base_dir / metric
                    meta = {"project": project, "model": model_name, "version": version,
                            "target": target, "objective": objective,
                            "bin_count": constants.BIN_COUNT, "n_train": constants.N_TRAIN,
                            "n_eval": constants.N_EVAL, "n_repeats": constants.N_REPEATS,
                            "t_agg": constants.T_AGG, "repeat_agg": constants.REPEAT_AGG,
                            "save_per_repeat": constants.SAVE_PER_REPEAT}
                    result_writer.write_matrix(res, meta, out_dir)
                    result_writer.write_drift_test(drift, out_dir)
                    plotter.plot_all(res, out_dir, dpi=constants.PLOT_DPI)
                    results_by_metric[metric].append(
                        {"version": version, "drift_exists": drift["drift_exists"],
                         "min_p_value": drift["min_p_value"]})
            for metric, per_version in results_by_metric.items():
                meta = {"project": project, "model": model_name, "target": target, "metric": metric}
                result_writer.write_summary(per_version, meta,
                                            out_root / project / model_name / "summary" / target / metric)


def replot(out_root=None) -> int:
    """保存済み結果（drift_matrix.json）から図だけ再描画する（再計算なし）。

    既に出力済みの `<...>/<metric>/drift_matrix.json` を全て探し、その場の png を描き直す。
    色やラベルの調整を再計算なしで反映したいときに使う。
    """
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
    """保存済み predictions.csv.gz から、再学習なしで行列・検定・図を作り直す（§単一経路）。

    評価指標を増やした／変えたあと（`constants.ENABLED_METRICS` と ranking_metrics を更新後）に
    これを呼べば、学習をやり直さずに全リリース・全モデルの結果を更新できる。
    """
    from collections import defaultdict

    out_root = Path(out_root or constants.OUTPUT_ROOT)
    count = 0
    # (project, model, target) -> metric -> [per_version]
    summary_acc: dict = defaultdict(lambda: defaultdict(list))
    for pred_path in sorted(out_root.rglob("predictions.csv.gz")):
        base = pred_path.parent                       # .../<version>/<target>
        # project/model/version/target/objective は predictions_meta.json から読む（パス深さ推定に依存しない）。
        meta_path = base / "predictions_meta.json"
        meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else {}
        target = meta.get("target")
        if not target:
            # 旧レイアウト（target 情報が無い予測）はスキップ。誤った階層に書き出さないため。
            logger.warning(f"target 情報が無い予測をスキップ（旧レイアウト？）: {pred_path}")
            continue
        objective = meta.get("objective", "regression")
        project = meta.get("project") or base.parent.parent.parent.name
        model_name = meta.get("model") or base.parent.parent.name
        version = meta.get("version") or base.parent.name
        metric_names = constants.TARGET_SPEC.get(target, {}).get("metrics")

        df = result_writer.load_predictions(pred_path)
        rows = list(df.itertuples(index=False, name=None))
        matrices = drift_matrix.compute_matrices(rows, metric_names=metric_names, objective=objective)
        for metric, res in matrices.items():
            drift = drift_detector.detect_drift(res, constants.PERMUTATION_N,
                                                constants.SIGNIFICANCE, seed=constants.RANDOM_SEED)
            out_dir = base / metric
            meta = {"project": project, "model": model_name, "version": version,
                    "target": target, "objective": objective,
                    "bin_count": constants.BIN_COUNT, "n_repeats": constants.N_REPEATS,
                    "t_agg": constants.T_AGG, "repeat_agg": constants.REPEAT_AGG,
                    "save_per_repeat": constants.SAVE_PER_REPEAT, "recomputed": True}
            result_writer.write_matrix(res, meta, out_dir)
            result_writer.write_drift_test(drift, out_dir)
            plotter.plot_all(res, out_dir, dpi=constants.PLOT_DPI)
            summary_acc[(project, model_name, target)][metric].append(
                {"version": version, "drift_exists": drift["drift_exists"],
                 "min_p_value": drift["min_p_value"]})
        count += 1
        logger.info(f"再計算: {base}")
    for (project, model_name, target), metrics in summary_acc.items():
        for metric, per_version in metrics.items():
            result_writer.write_summary(
                per_version,
                {"project": project, "model": model_name, "target": target, "metric": metric},
                out_root / project / model_name / "summary" / target / metric)
    logger.info(f"完了。{count} 件の予測から再計算しました（出力先: {out_root}）")
    return count


def run(projects=None, targets=None):
    """全プロジェクト × 全目的変数 × 全モデルを実行する（既定は constants の設定）。"""
    rel_df = load_release_dates()
    bot_names = review_utils.load_bot_names()
    projects = projects or constants.TARGET_PROJECTS
    targets = targets or constants.TARGETS
    for project, versions in projects.items():
        logger.info(f"=== プロジェクト: {project} ===")
        changes = load_changes(project)
        if not changes:
            logger.warning(f"Change が無いためスキップ: {project}")
            continue
        analyze(changes, rel_df, project, versions, constants.OUTPUT_ROOT, bot_names, targets=targets)
    logger.info(f"完了。出力先: {constants.OUTPUT_ROOT}")


if __name__ == "__main__":
    # 使い方:
    #   python -m ...main            # 全分析を実行（学習＋計算＋出力＋作図）
    #   python -m ...main replot     # 保存済み行列から図だけ再描画（再計算なし）
    #   python -m ...main recompute  # 保存済み予測から指標・検定・図を作り直す（再学習なし）
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        replot()
    elif len(sys.argv) > 1 and sys.argv[1] == "recompute":
        recompute()
    else:
        run()
