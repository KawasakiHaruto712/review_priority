"""feature_space_window（RQ2）の実行の入口（design.md §4.1・§6・§10）。

モード:
  --mode run（既定） : 保存済みエンコーダを load → seed ごとに全日を埋め込み → 評価日ごとに近い k 日で probe → 保存 → 要約表
  --mode plot        : 保存済み metrics.csv から要約表だけ作り直す

実行:
    python3 -m src.analysis.preliminary_analysis.feature_space_window.main
    python3 -m src.analysis.preliminary_analysis.feature_space_window.main --project qtbase
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.analysis.background_problem.common.data_loader import load_release_dates
from src.analysis.preliminary_analysis.feature_space_window.dataset import candidates
from src.analysis.preliminary_analysis.feature_space_window.evaluation.runner import VersionRows, run_seed
from src.analysis.preliminary_analysis.feature_space_window.io import result_writer
from src.analysis.preliminary_analysis.feature_space_window.utils import constants
from src.analysis.preliminary_analysis.feature_space_window.visualization import table
from src.analysis.preliminary_analysis.pretrained_encoders.io import store
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _seeds(project: str, n_seeds: int | None) -> list[int]:
    """保存済みの seed。なければ止める（事前学習は pretrained_encoders で作る。ここでは作らない）。"""
    cutoff = constants.cutoff_for(project)
    saved = store.list_seeds(project, cutoff)
    if not saved:
        raise RuntimeError(f"[{project}] 事前学習済みのエンコーダがありません（cutoff={cutoff}）。先に "
                           f"python3 -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders --project {project}")
    return saved if n_seeds is None else saved[:n_seeds]


def run(project: str, versions: list[str], n_seeds: int | None, sizes: list[int]) -> None:
    rel_df = load_release_dates()
    day_sets, pretrain_days = candidates.load_all_day_sets(project, rel_df)
    version_eval = {}
    for v in versions:
        try:
            cs_R, ce_R = candidates.release_cycle(rel_df, project, v)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {v}]: {e}")
            continue
        version_eval[v] = candidates.eval_dates_for(day_sets, cs_R, ce_R)
        logger.info(f"{project} {v}：サイクル {cs_R.date()}〜{ce_R.date()}・評価日 {len(version_eval[v])}")
    device = st.resolve_device()
    seeds = _seeds(project, n_seeds)
    logger.info(f"device = {device}・seed {seeds}・大きさ {sizes}")
    cutoff = constants.cutoff_for(project)
    merged = {v: VersionRows() for v in version_eval}
    for i, k in enumerate(seeds):
        encoder, _general, scaler, _cfg = store.load_pretrained(project, cutoff, k, device)
        for v, rows in run_seed(day_sets, pretrain_days, version_eval, encoder, scaler, k, device, sizes).items():
            result_writer.append_predictions(rows.preds, project, v, first=(i == 0))   # 予測は seed ごとに書き足す
            rows.preds = []
            merged[v].extend(rows)
        del encoder
    for v, rows in merged.items():
        logger.info(f"保存: {result_writer.save_version(rows, project, v)}")
    result_writer.save_run_meta(project, seeds, sizes, list(version_eval))


def main() -> None:
    ap = argparse.ArgumentParser(description="feature_space_window：特徴空間で近い過去の日で probe を学習する（RQ2）")
    ap.add_argument("--mode", choices=["run", "plot"], default="run")
    ap.add_argument("--project", default=None, help="対象（未指定・'all' なら PROJECTS の 6 件を順に）")
    ap.add_argument("--versions", nargs="*", default=None, help="対象の版（既定は PROJECTS の 5 版）")
    ap.add_argument("--n-seeds", type=int, default=constants.N_SEEDS, help="使う seed の数（既定は保存済み全部）")
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="大きさ k（日。既定 1 3 7 14 30 60）")
    ap.add_argument("--metric", default=constants.PLOT_METRIC_DEFAULT, help="要約表の指標（既定 auc）")
    args = ap.parse_args()
    projects = list(constants.PROJECTS) if args.project in (None, "all") else [args.project]
    sizes = sorted(args.sizes) if args.sizes else list(constants.SIZES_DAYS)
    for project in projects:
        versions = args.versions if args.versions else constants.versions_for(project)
        logger.info(f"===== プロジェクト: {project}（{len(versions)} 版）=====")
        if args.mode == "run":
            run(project, versions, args.n_seeds, sizes)
        out = table.build_and_save(project, versions, args.metric, sizes)
        logger.info(f"要約表: {out}")


if __name__ == "__main__":
    main()
