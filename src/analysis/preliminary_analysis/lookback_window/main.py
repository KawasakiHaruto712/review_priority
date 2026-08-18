"""lookback_window オーケストレーション（design.md §10）。

モード:
  --mode run（既定）  : 保存済みエンコーダを load → 日次×窓長 probe → 生予測・指標を保存 → 作図（表＋折れ線）
  --mode plot         : 保存済み metrics.csv から描画のみ（モデルを回さない）
  --mode recompute    : 生予測 predictions.csv.gz から指標テーブルを作り直す（モデル再実行なし）

描画指標は --metric で選択（既定 auc）。--windows で描く窓長を選択。--no-general で基準線を外す。

実行:
    python -m src.analysis.preliminary_analysis.lookback_window.main --mode run
    python -m src.analysis.preliminary_analysis.lookback_window.main --mode plot --metric recall@10
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from src.analysis.background_problem.common.data_loader import load_changes, load_release_dates
from src.analysis.preliminary_analysis.lookback_window.evaluation import metrics as M
from src.analysis.preliminary_analysis.lookback_window.io import result_io
from src.analysis.preliminary_analysis.lookback_window.sweep import window_sweep
from src.analysis.preliminary_analysis.lookback_window.utils import constants
from src.analysis.preliminary_analysis.lookback_window.visualization import plotter, table
from src.analysis.preliminary_analysis.pretrained_encoders import build_encoders
from src.analysis.preliminary_analysis.pretrained_encoders.features import feature_builder
from src.analysis.preliminary_analysis.pretrained_encoders.io import store
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre_constants
from src.analysis.preliminary_analysis.pretrained_encoders.utils import review_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _resolve_seeds(project: str, n_seeds: int | None) -> list[int]:
    """使う事前学習 seed を返す。モデルが未作成なら、その場で build_encoders を実行して作る
    （lookback_window さえ実行すれば必ず probe 結果まで出せるようにするため）。"""
    cutoff = constants.PRETRAINED_CUTOFF
    seeds = store.list_seeds(project, cutoff)
    if not seeds:
        n_build = n_seeds if n_seeds else pre_constants.N_REPEATS
        logger.info(f"事前学習モデルが未作成（{project} cutoff={cutoff}）。"
                    f"先に build_encoders を実行します（{n_build} 個）…")
        build_encoders.build_and_save(n_build)
        seeds = store.list_seeds(project, cutoff)
        if not seeds:
            raise RuntimeError("事前学習モデルの作成に失敗しました。")
    return seeds if n_seeds is None else seeds[:n_seeds]


def run(project: str, versions: list[str], n_seeds: int | None) -> None:
    """モデルを走らせて生予測・指標を保存し、表＋折れ線を描く。"""
    rel_df = load_release_dates()
    changes = load_changes(project)
    bot_names = review_utils.load_bot_names()
    all_prs = feature_builder.build_all_prs_df(changes)
    device = st.resolve_device()
    logger.info(f"device = {device}")

    seeds = _resolve_seeds(project, n_seeds)
    logger.info(f"使用する事前学習 seed: {seeds}")
    # seed ごとに (encoder, general_head, scaler) を load
    loaded = [store.load_pretrained(project, constants.PRETRAINED_CUTOFF, s, device) for s in seeds]

    for version in versions:
        try:
            cs_R, ce_R = window_sweep.release_cycle(rel_df, project, version)
        except ValueError as e:
            logger.warning(f"スキップ [{project} {version}]: {e}")
            continue
        span_start = window_sweep.span_start_for(cs_R)
        logger.info(f"--- {project} {version}（サイクル {cs_R.date()}〜{ce_R.date()} / "
                    f"学習データ開始 {span_start.date()}）---")
        day_sets = window_sweep.build_day_sets(changes, project, span_start, ce_R,
                                               bot_names, all_prs, rel_df)
        eval_dates = [d for d in sorted(day_sets) if cs_R.date() <= d <= ce_R.date()]
        logger.info(f"評価日数: {len(eval_dates)} / 全日集合: {len(day_sets)}")

        pred_rows, metric_rows = [], []
        for seed_idx, (encoder, general_head, scaler, _cfg) in zip(seeds, loaded):
            pr, mr = window_sweep.run_sweep(day_sets, eval_dates, encoder, general_head, scaler,
                                            seed_idx, device)
            pred_rows.extend(pr)
            metric_rows.extend(mr)

        result_io.save_predictions(pred_rows, project, version)
        result_io.save_metrics(metric_rows, project, version)
        logger.info(f"保存: {result_io.version_dir(project, version)}")


def recompute(project: str, versions: list[str]) -> None:
    """生予測から指標テーブルを作り直す（モデル再実行なし）。"""
    for version in versions:
        try:
            pdf = result_io.load_predictions(project, version)
        except FileNotFoundError:
            logger.warning(f"predictions が無いためスキップ: {project} {version}")
            continue
        metric_rows = []
        for (day, window, seed), g in pdf.groupby(["day", "window", "seed"]):
            yt = g["y_true"].to_numpy(dtype=float)
            yp = g["score"].to_numpy(dtype=float)
            n_pos = int(yt.sum()); n_neg = int(len(yt) - yt.sum())
            mv = M.compute_day_metrics(yt, yp, constants.K_LIST, constants.CLASSIFY_THRESHOLD)
            row = {"day": day, "window": window, "seed": seed,
                   "n_pos_eval": n_pos, "n_neg_eval": n_neg,
                   "n_pos_train": np.nan, "n_neg_train": np.nan,
                   "missing": 0, "missing_reason": ""}
            row.update(mv)
            metric_rows.append(row)
        result_io.save_metrics(metric_rows, project, version)
        logger.info(f"再計算・保存: {result_io.version_dir(project, version)}")


def draw(project: str, versions: list[str], metric: str, windows, draw_general: bool) -> None:
    """保存済み metrics から 要約表＋バージョンごとの折れ線 を描く。"""
    table.build_and_save(project, versions, metric, constants.OUTPUT_ROOT / project / "summary")
    logger.info(f"要約表: {constants.OUTPUT_ROOT / project / 'summary'}")
    for version in versions:
        try:
            out = result_io.version_dir(project, version) / f"lines_{metric.replace('@', 'at')}.png"
            plotter.plot_version(project, version, metric, out, windows, draw_general)
            logger.info(f"折れ線: {out}")
        except FileNotFoundError:
            logger.warning(f"metrics が無いためスキップ: {project} {version}")


def main() -> None:
    ap = argparse.ArgumentParser(description="lookback_window：チューニング窓長の調査")
    ap.add_argument("--mode", choices=["run", "plot", "recompute"], default="run")
    ap.add_argument("--metric", default=constants.PLOT_METRIC_DEFAULT, help="描画する指標（既定 auc）")
    ap.add_argument("--n-seeds", type=int, default=None, help="使う事前学習 seed 数（既定 全部）")
    ap.add_argument("--windows", type=int, nargs="*", default=None, help="描く窓長（日）")
    ap.add_argument("--no-general", action="store_true", help="汎用ヘッド基準線を描かない")
    ap.add_argument("--versions", nargs="*", default=None, help="対象バージョン（既定 全部。例 30.0.0）")
    args = ap.parse_args()

    project = constants.PROJECT
    versions = args.versions if args.versions else constants.VERSIONS
    windows = args.windows if args.windows else None
    draw_general = not args.no_general

    if args.mode == "run":
        run(project, versions, args.n_seeds)
        draw(project, versions, args.metric, windows, draw_general)
    elif args.mode == "recompute":
        recompute(project, versions)
        draw(project, versions, args.metric, windows, draw_general)
    else:  # plot
        draw(project, versions, args.metric, windows, draw_general)


if __name__ == "__main__":
    main()
