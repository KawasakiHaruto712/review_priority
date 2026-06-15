"""
change_lifetime 分析のエントリポイント（§5.4）。

実行:
    python -m src.analysis.background_problem.change_lifetime.main

各リリースの開発サイクル中に Open だった Change の生存期間（lifetime）を、
status グループ（all / merged / abandoned）ごとにヒストグラム・要約統計・箱ひげ図で出力する。
"""
from __future__ import annotations

import logging
import sys

from src.analysis.background_problem.common.data_loader import (
    get_release_cycle,
    load_changes,
    load_release_dates,
)
from src.analysis.background_problem.change_lifetime.io.result_writer import (
    write_histogram_csv,
    write_summary_json,
)
from src.analysis.background_problem.change_lifetime.metrics.lifetime_calculator import (
    extract_lifetimes,
    make_histogram,
    summarize,
)
from src.analysis.background_problem.change_lifetime.utils import constants
from src.analysis.background_problem.change_lifetime.visualization.plotter import (
    plot_boxplot,
    plot_histogram,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _xlabel() -> str:
    return f"生存期間（投稿〜終了）[{constants.DURATION_UNIT}]"


def _ylabel_count() -> str:
    return "頻度（Change 数）"


def _save_one(counts, edges, summary, meta, out_dir, title):
    """1 つのヒストグラム出力（csv / json / png）をまとめて保存する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_histogram_csv(counts, edges, out_dir / "histogram.csv")
    write_summary_json(meta, summary, counts, edges, out_dir / "histogram.json")
    plot_histogram(
        counts, edges, out_dir / "histogram.png",
        title=title, xlabel=_xlabel(), ylabel=_ylabel_count(),
        x_log=constants.X_LOG_SCALE, dpi=constants.PLOT_DPI,
    )


def analyze_project_group(project, group, statuses, changes, rel_df):
    """1 プロジェクト × 1 status グループの分析を実行する。"""
    statuses_sorted = sorted(statuses)
    values_by_release: dict = {}
    mixed_values: list = []
    seen_change_numbers: set = set()

    for version in constants.TARGET_PROJECTS[project]:
        try:
            cycle_start, cycle_end = get_release_cycle(rel_df, project, version)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {version}]: {e}")
            continue

        values, cnums, n_excl = extract_lifetimes(
            changes, cycle_start, cycle_end, statuses,
            constants.LOOKBACK_DAYS, constants.DURATION_UNIT,
        )
        counts, edges = make_histogram(
            values, constants.HIST_BINS, constants.HIST_LOG_BINS, constants.HIST_MIN_VALUE,
        )
        summary = summarize(values, constants.SUMMARY_PERCENTILES)
        meta = {
            "project": project,
            "version": version,
            "status_group": group,
            "statuses": statuses_sorted,
            "cycle_start": cycle_start.date().isoformat(),
            "cycle_end": cycle_end.date().isoformat(),
            "lookback_days": constants.LOOKBACK_DAYS,
            "duration_unit": constants.DURATION_UNIT,
            "hist_bins": constants.HIST_BINS,
            "hist_log_bins": constants.HIST_LOG_BINS,
            "x_log_scale": constants.X_LOG_SCALE,
            "n_changes": len(values),
            "n_excluded_unfinished": n_excl,
        }
        out_dir = constants.OUTPUT_ROOT / project / version / group
        title = f"{project} {version} : Change 生存期間（{group}）"
        _save_one(counts, edges, summary, meta, out_dir, title)

        values_by_release[version] = values
        # 全リリース混合は change_number で重複排除
        for v, cn in zip(values, cnums):
            if cn not in seen_change_numbers:
                seen_change_numbers.add(cn)
                mixed_values.append(v)

    # 全リリース混合（重複排除済み）
    counts, edges = make_histogram(
        mixed_values, constants.HIST_BINS, constants.HIST_LOG_BINS, constants.HIST_MIN_VALUE,
    )
    summary = summarize(mixed_values, constants.SUMMARY_PERCENTILES)
    meta = {
        "project": project,
        "version": "all_releases",
        "status_group": group,
        "statuses": statuses_sorted,
        "lookback_days": constants.LOOKBACK_DAYS,
        "duration_unit": constants.DURATION_UNIT,
        "hist_bins": constants.HIST_BINS,
        "hist_log_bins": constants.HIST_LOG_BINS,
        "x_log_scale": constants.X_LOG_SCALE,
        "n_changes": len(mixed_values),
        "deduplicated": True,
    }
    out_dir = constants.OUTPUT_ROOT / project / "all_releases" / group
    title = f"{project} all_releases : Change 生存期間（{group}）"
    _save_one(counts, edges, summary, meta, out_dir, title)

    # リリース横断の箱ひげ図
    if constants.MAKE_BOXPLOT:
        plot_boxplot(
            values_by_release,
            constants.OUTPUT_ROOT / project / f"{group}_boxplot.png",
            title=f"{project} : Change 生存期間のリリース別分布（{group}）",
            ylabel=_xlabel(),
            y_log=constants.BOXPLOT_LOG_SCALE,
            dpi=constants.PLOT_DPI,
        )


def run() -> None:
    """全プロジェクト × 全 status グループを実行する。"""
    rel_df = load_release_dates()
    for project in constants.TARGET_PROJECTS:
        logger.info(f"=== プロジェクト: {project} ===")
        changes = load_changes(project)
        if not changes:
            logger.warning(f"Change が無いためスキップ: {project}")
            continue
        for group, statuses in constants.STATUS_GROUPS.items():
            logger.info(f"--- status グループ: {group} ---")
            analyze_project_group(project, group, statuses, changes, rel_df)
    logger.info(f"完了。出力先: {constants.OUTPUT_ROOT}")


if __name__ == "__main__":
    run()
