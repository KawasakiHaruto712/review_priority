"""feature_space_window（RQ2）の実行の入口（design.md §4.1・§6・§10）。

モード:
  --mode run（既定） : 保存済みエンコーダを load → seed ごとに全日を埋め込み → 評価日ごとに近い k 日で probe → 保存 → 要約表
  --mode plot        : 保存済み metrics.csv から要約表だけ作り直す

実行（2026-10-22：run・matrix は --variant 必須。posw ＝ 重みあり版の出力は feature_space_window/posw/ の下。design.md §10.1）:
    python3 -m src.analysis.preliminary_analysis.feature_space_window.main --variant posw --gpu-share 1.0
    python3 -m src.analysis.preliminary_analysis.feature_space_window.main --mode rank --metric f1_plugin      # 省略時は重みあり版
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from src.analysis.background_problem.common.data_loader import load_release_dates
from src.analysis.preliminary_analysis.feature_space_window.dataset import candidates
from src.analysis.preliminary_analysis.feature_space_window.evaluation.runner import VersionRows, run_seed
from src.analysis.preliminary_analysis.feature_space_window.io import result_writer
from src.analysis.preliminary_analysis.feature_space_window.utils import constants, parallel
from src.analysis.preliminary_analysis.feature_space_window.visualization import table
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre_constants
from src.analysis.preliminary_analysis.pretrained_encoders.io import store
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.utils import variant

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _seeds(project: str, n_seeds: int | None) -> list[int]:
    """保存済みの seed。なければ止める（事前学習は pretrained_encoders で作る。ここでは作らない）。"""
    cutoff = constants.cutoff_for(project)
    saved = store.list_seeds(project, cutoff)
    if not saved:
        raise RuntimeError(f"[{project}] 事前学習済みのエンコーダがありません（cutoff={cutoff}・版の種類 {pre_constants.ENCODER_VARIANT}）。先に "
                           f"python3 -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders "
                           f"--variant {pre_constants.ENCODER_VARIANT} --project {project}")
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
    for v in version_eval:
        variant.warn_if_exists(result_writer.version_dir(project, v), f"{project} {v}")   # 上書きの警告（design.md §10.1）
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
    result_writer.save_run_meta(project, seeds, sizes, list(version_eval), device=str(device))


def main() -> None:
    ap = argparse.ArgumentParser(description="feature_space_window：特徴空間で近い過去の日で probe を学習する（RQ2）")
    ap.add_argument("--mode", choices=["run", "plot", "rank", "report", "matrix"], default="run",
                    help="rank は順位の集計（§5.1）、report は 6 件をまとめた報告用の表（run・rank の後。§5.2）、"
                         "matrix は特徴のずれ×評価時期の精度表（§9）")
    ap.add_argument("--project", default=None, help="対象（未指定・'all' なら PROJECTS の 6 件を順に）")
    ap.add_argument("--versions", nargs="*", default=None, help="対象の版（既定は PROJECTS の 5 版）")
    ap.add_argument("--n-seeds", type=int, default=constants.N_SEEDS, help="使う seed の数（既定は保存済み全部）")
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="大きさ k（日。既定 1 3 7 14 30 60）")
    ap.add_argument("--metric", default=constants.PLOT_METRIC_DEFAULT, help="要約表の指標（既定 auc）")
    ap.add_argument("--workers", type=int, default=None,
                    help="リポジトリを並べて回す数（既定：CUDA なら --gpu-share から自動、Mac などは 1。pretrained_encoders/design.md §14.7）")
    ap.add_argument("--gpu-share", type=float, default=parallel.DEFAULT_GPU_SHARE,
                    help="並べたプロセス全体で使ってよい GPU のメモリの割合（既定 0.25 ＝ 1/4。誰も使っていなければ 1.0 まで）")
    variant.add_argument(ap)
    args = ap.parse_args()
    # 2026-10-22：run・matrix は --variant 必須（上書きを防ぐ）。plot・rank・report は省略すると重みあり版（design.md §10.1）
    v = variant.apply(variant.resolve(args.variant, writing=args.mode in ("run", "matrix")), constants)
    projects = list(constants.PROJECTS) if args.project in (None, "all") else [args.project]
    sizes = sorted(args.sizes) if args.sizes else list(constants.SIZES_DAYS)
    if args.mode in ("run", "matrix"):
        if args.gpu_share > parallel.DEFAULT_GPU_SHARE:
            logger.warning(f"GPU のメモリを全体の {args.gpu_share:.0%} まで使います（既定の 1/4 を超えています）")
        w, frac = parallel.decide(len(projects), use_gpu=True, gpu_share=args.gpu_share, workers=args.workers)
        if w > 1:               # リポジトリごとに別々のプロセスで並べる（§14.7）。子は --project 1 件・--workers 1 で同じ入口を呼ぶ
            child = ["--mode", args.mode, "--metric", args.metric, "--variant", v]
            if args.versions:
                child += ["--versions", *args.versions]
            if args.n_seeds:
                child += ["--n-seeds", str(args.n_seeds)]
            if args.sizes:
                child += ["--sizes", *map(str, args.sizes)]
            failed = parallel.run_children("src.analysis.preliminary_analysis.feature_space_window.main", projects,
                                           child, w, frac, variant.log_name(f"feature_space_window_{args.mode}", v))
            sys.exit(1 if failed else 0)
        if frac is not None and not os.environ.get("RP_CUDA_MEMORY_FRACTION"):
            pre_constants.CUDA_MEMORY_FRACTION = frac           # 1 つで回すときは --gpu-share をそのまま使う
    if args.mode == "matrix":   # 特徴のずれ×評価時期の精度表（design.md §9。2026-10-21）
        from src.analysis.preliminary_analysis.feature_space_window.matrix.run import run_project
        failed = []
        for project in projects:
            versions = args.versions if args.versions else constants.versions_for(project)
            try:
                run_project(project, versions, args.n_seeds)
            except Exception:
                logger.exception(f"[{project}] 失敗しました（ほかは続けます）")
                failed.append(project)
        if failed:
            logger.error(f"失敗したリポジトリ：{failed}")
            sys.exit(1)
        return
    if args.mode == "report":   # 6 件をまとめて 1 回（design.md §5.2。2026-10-21）
        from src.analysis.preliminary_analysis.feature_space_window.evaluation.ranking import size_labels_desc
        from src.analysis.preliminary_analysis.feature_space_window.visualization import report
        out = report.build_report(projects, args.metric, constants.OUTPUT_ROOT, size_labels_desc(),
                                  index_col="size", row_name="大きさ",
                                  lower_is_better=args.metric in constants.LOWER_IS_BETTER)
        logger.info(f"報告用の表: {out}" if out else "報告用の表：結果がありません（先に --mode run と --mode rank）")
        return
    for project in projects:
        versions = args.versions if args.versions else constants.versions_for(project)
        logger.info(f"===== プロジェクト: {project}（{len(versions)} 版）=====")
        if args.mode == "rank":     # 順位の集計（design.md §5.1。2026-10-21）
            from src.analysis.preliminary_analysis.feature_space_window.evaluation.ranking import build_ranking
            from src.analysis.preliminary_analysis.feature_space_window.io.rank_io import save_ranking
            df, ndays = build_ranking(project, versions, args.metric)
            if not ndays.get("全体"):
                logger.warning(f"metrics がない・空のためスキップ（先に --mode run）: {project}")
                continue
            logger.info(f"順位の集計: {save_ranking(df, ndays, project, args.metric)}（対象日数 "
                        + ", ".join(f"{k}={v}" for k, v in ndays.items()) + "）")
            continue
        if args.mode == "run":
            run(project, versions, args.n_seeds, sizes)
        out = table.build_and_save(project, versions, args.metric, sizes)
        logger.info(f"要約表: {out}")
        if args.mode == "plot":     # 特徴のずれ×評価時期の精度表の図も、保存済みの json から描き直す（design.md §9.2。2026-10-21）
            from src.analysis.preliminary_analysis.feature_space_window.matrix import plotter as mplot, run as mrun, writer as mwriter
            for v in versions:
                for kind in ("probe", "general", "diff"):
                    for m in constants.PLOT_METRICS:
                        path = mrun.out_root(project) / v / kind / m / "drift_matrix.json"
                        if path.exists():
                            mplot.plot_all(mwriter.load_matrix(path), path.parent, constants.PLOT_DPI)


if __name__ == "__main__":
    main()
