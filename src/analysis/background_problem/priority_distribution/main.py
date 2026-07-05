"""
priority_distribution 分析のエントリポイント（オーケストレーション, §5.8）。

実行:
    python -m src.analysis.background_problem.priority_distribution.main

constants.py の TARGET_PROJECTS × ENABLED_METRICS をすべて実行し、
リリースごと / 全リリース混合 の分布（csv / json / png）を OUTPUT_ROOT に保存する。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.analysis.background_problem.priority_distribution.io.result_writer import (
    write_csv,
    write_json,
)
from src.analysis.background_problem.priority_distribution.metrics import (
    distribution_builder,
)
from src.analysis.background_problem.priority_distribution.metrics.duration_calculator import (
    compute_change_records,
    created as change_created,
    get_metric,
)
from src.analysis.background_problem.priority_distribution.preprocessing.outlier_filter import (
    drop_outliers,
    drop_unfinished,
)
from src.analysis.background_problem.priority_distribution.utils import constants
from src.analysis.background_problem.priority_distribution.utils.data_loader import (
    get_release_cycle,
    load_bot_names,
    load_changes,
    load_release_dates,
)
from src.analysis.background_problem.priority_distribution.visualization.plotter import (
    plot_band,
    plot_percentiles,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# 縦軸ラベル（指標 → 表示名）
_Y_LABELS = {
    "time_to_next_review": "計測点から次のレビューまでの時間",
    "time_to_decision": "計測点からマージ/リジェクト判断までの時間",
}


def _xlabel(x_mode: str) -> str:
    if x_mode == "normalized":
        return "リリースサイクル内の相対位置 (0=開始, 1=リリース)"
    return "リリースまでの残り日数"


def _ylabel(metric_name: str, unit: str) -> str:
    base = _Y_LABELS.get(metric_name, metric_name)
    return f"{base} [{unit}]"


def _save_outputs(points, meta, out_dir: Path, title, xlabel, ylabel, include_release):
    """1 つの分布の出力（csv / json と 2 種類の png）をまとめて保存する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 統合データ（mean/std と分位点を併せ持つ）
    write_csv(points, out_dir / "distribution.csv", include_release=include_release)
    write_json(points, meta, out_dir / "distribution.json")
    # 図 2 枚
    plot_band(
        points,
        out_dir / "distribution_mean_std.png",
        title=f"{title}（mean±0.5std）",
        xlabel=xlabel,
        ylabel=ylabel,
        y_log=constants.Y_LOG_SCALE,
        dpi=constants.PLOT_DPI,
    )
    plot_percentiles(
        points,
        out_dir / "distribution_percentiles.png",
        title=f"{title}（percentiles）",
        xlabel=xlabel,
        ylabel=ylabel,
        y_log=constants.Y_LOG_SCALE,
        dpi=constants.PLOT_DPI,
    )


def analyze_project_metric(project: str, metric_name: str, changes, bot_names, rel_df):
    """1 プロジェクト × 1 メトリクスの分析を実行する。"""
    metric = get_metric(metric_name)
    xlabel = _xlabel(constants.X_AXIS_MODE)
    ylabel = _ylabel(metric_name, constants.DURATION_UNIT)
    all_points = []

    for version in constants.TARGET_PROJECTS[project]:
        try:
            cycle_start, cycle_end = get_release_cycle(rel_df, project, version)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {version}]: {e}")
            continue

        # 計測点は毎日 0 時の定点グリッド（build_distribution 側で生成）。各 T の
        # アクティブ集合の候補になるのは lookback さかのぼった投稿なので、
        # 候補プールは [cycle_start - lookback, cycle_end]。
        lookback = pd.Timedelta(days=constants.LOOKBACK_DAYS)
        pool_start = cycle_start - lookback
        cyc_changes = []
        for c in changes:
            cdt = change_created(c)
            if cdt is not None and pool_start <= cdt <= cycle_end:
                cyc_changes.append(c)

        records = compute_change_records(cyc_changes, metric, bot_names)
        n_total = len(records)
        records = drop_unfinished(records)
        n_after_unfinished = len(records)
        records, dropped = drop_outliers(
            records,
            method=constants.OUTLIER_METHOD,
            iqr_k=constants.OUTLIER_IQR_K,
            percentile=constants.OUTLIER_PERCENTILE,
        )

        points = distribution_builder.build_distribution(
            records,
            metric,
            bot_names,
            cycle_start,
            cycle_end,
            x_mode=constants.X_AXIS_MODE,
            duration_unit=constants.DURATION_UNIT,
            min_active=constants.MIN_ACTIVE_FOR_POINT,
            band_std_factor=constants.BAND_STD_FACTOR,
            lookback_days=constants.LOOKBACK_DAYS,
            percentiles=constants.PERCENTILES,
            release_version=version,
            step_days=constants.MEASUREMENT_STEP_DAYS,
        )

        meta = {
            "project": project,
            "version": version,
            "metric": metric_name,
            "cycle_start": cycle_start.date().isoformat(),
            "cycle_end": cycle_end.date().isoformat(),
            "duration_unit": constants.DURATION_UNIT,
            "band": f"mean +/- {constants.BAND_STD_FACTOR}*std",
            "percentiles": constants.PERCENTILES,
            "x_axis_mode": constants.X_AXIS_MODE,
            "y_log_scale": constants.Y_LOG_SCALE,
            "lookback_days": constants.LOOKBACK_DAYS,
            "n_changes_total": n_total,
            "n_dropped_unfinished": n_total - n_after_unfinished,
            "n_dropped_outliers": len(dropped),
            "outlier": {
                "method": constants.OUTLIER_METHOD,
                "k": constants.OUTLIER_IQR_K,
                "side": "both",
            },
        }

        out_dir = constants.OUTPUT_ROOT / project / version / metric_name
        title = f"{project} {version} : {metric_name}"
        _save_outputs(points, meta, out_dir, title, xlabel, ylabel, include_release=False)

        all_points.extend(points)

    # 全リリース混合
    mixed_points = all_points
    if constants.MIXED_PLOT_BINS:
        mixed_points = distribution_builder.bin_points(
            sorted(all_points, key=lambda p: p.x), constants.MIXED_PLOT_BINS
        )
    else:
        mixed_points = sorted(all_points, key=lambda p: p.x)

    mixed_meta = {
        "project": project,
        "version": "all_releases",
        "metric": metric_name,
        "duration_unit": constants.DURATION_UNIT,
        "band": f"mean +/- {constants.BAND_STD_FACTOR}*std",
        "percentiles": constants.PERCENTILES,
        "x_axis_mode": constants.X_AXIS_MODE,
        "y_log_scale": constants.Y_LOG_SCALE,
        "lookback_days": constants.LOOKBACK_DAYS,
        "n_points": len(mixed_points),
        "binned": bool(constants.MIXED_PLOT_BINS),
    }
    out_dir = constants.OUTPUT_ROOT / project / "all_releases" / metric_name
    title = f"{project} all_releases : {metric_name}"
    _save_outputs(
        mixed_points, mixed_meta, out_dir, title, xlabel, ylabel, include_release=True
    )


def run() -> None:
    """全プロジェクト × 全メトリクスを実行する。"""
    rel_df = load_release_dates()
    bot_names = load_bot_names()

    for project in constants.TARGET_PROJECTS:
        logger.info(f"=== プロジェクト: {project} ===")
        changes = load_changes(project)
        if not changes:
            logger.warning(f"Change が無いためスキップ: {project}")
            continue
        for metric_name in constants.ENABLED_METRICS:
            logger.info(f"--- メトリクス: {metric_name} ---")
            analyze_project_metric(project, metric_name, changes, bot_names, rel_df)

    logger.info(f"完了。出力先: {constants.OUTPUT_ROOT}")


if __name__ == "__main__":
    run()
