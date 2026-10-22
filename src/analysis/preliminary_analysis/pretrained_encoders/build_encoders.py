"""事前学習エンコーダ＋汎用ヘッドの作成・保存（design.md）。

流れ（上から順に追える構成）:
  1. 締め切り（PROJECTS[project] の cutoff リリース日）を決める
  2. feature_table の事前学習の表（プロジェクト開始 〜 締め切り）から pretraining 集合を作る
     （2026-10-18 改めた。表がなければ作り、作った条件の指紋が違えば止める。特徴量は MODEL_FEATURES）
  3. N 個（seed 違い）のエンコーダ＋汎用ヘッドを事前学習して保存

実行（2026-10-22：--variant が必須になった。posw ＝ 重みあり版、base ＝ 重みなし版。design.md §15.2）:
    python -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders --variant posw                    # PROJECTS の 6 件、各 N_REPEATS 個
    python -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders --variant posw --project qtbase   # 対象プロジェクト指定
    python -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders --variant posw --gpu-share 1.0     # GPU を全部使って並べる（サーバ）
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from src.analysis.background_problem.common.data_loader import load_release_dates
from src.analysis.preliminary_analysis.pretrained_encoders.dataset import set_builder
from src.analysis.preliminary_analysis.pretrained_encoders.io import store
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants, parallel, variant

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _resolve_cutoff(rel_df: pd.DataFrame, project: str):
    """事前学習の締め切り (datetime, ラベル) を返す。

    ラベル = PROJECTS[project]["cutoff"] の版。日付はその版の release_date を
    major_releases_summary.csv（rel_df）から引く（日付は二重管理しない）。
    """
    cfg = constants.project_config(project)
    if cfg is None:
        raise ValueError(f"未対応のプロジェクトです（PROJECTS か OPENSTACK_PROJECTS に定義してください）: {project}")
    label = cfg["cutoff"]
    row = rel_df[(rel_df["project"] == project) & (rel_df["version"] == label)]
    if row.empty:
        raise ValueError(f"cutoff リリースが見つかりません: {project} {label}")
    dt = pd.to_datetime(row["release_date"].iloc[0]).to_pydatetime()
    return dt, label


def build_and_save(project: str, n: int | None = None, seed_indices=None) -> list[int]:
    """指定プロジェクトの共有エンコーダ＋汎用ヘッドを作って保存する。

    project: 対象プロジェクト（必須）。PROJECTS か OPENSTACK_PROJECTS に定義が必要。
             旧版の既定 constants.TARGET_PROJECT="nova" は廃止した（design.md §7）。
    seed_indices を渡すと、その index(k) だけを作る（不足分の追加ビルド用）。
    未指定なら 0..n-1（n 既定 N_REPEATS）。
    返り値は、非有限で止まって保存しなかった seed の index（design.md §4。2026-10-20）。
    """
    if seed_indices is None:
        n = constants.N_REPEATS if n is None else n
        seed_indices = list(range(n))
    else:
        seed_indices = list(seed_indices)

    # 1. 締め切り
    rel_df = load_release_dates()
    device = st.resolve_device()
    logger.info(f"device = {device}")
    cutoff_dt, cutoff_label = _resolve_cutoff(rel_df, project)

    # 2. pretraining 集合：feature_table の事前学習の表から作る（2026-10-18 改めた。design.md §2・feature_table/design.md §2.6・§2.9）。
    #    表がなければ作り、作った条件の指紋が違えば止める。build_records で全レコードをオブジェクトにしない（メモリのため）
    from src.analysis.preliminary_analysis.feature_table.build import ensure_table
    from src.analysis.preliminary_analysis.feature_table.build.sets import day_sets_from_table

    tab = ensure_table(project, "pretrain", rel_df=rel_df)
    pre_sets = day_sets_from_table(tab, max_set_size=constants.MAX_SET_SIZE)
    del tab
    logger.info(f"事前学習データ: 〜 {cutoff_dt.date()}（cutoff={cutoff_label}）。集合数: {len(pre_sets)}"
                f"（総レコード {set_builder.count_records(pre_sets)}、特徴量 {st.FEATURE_DIM} 個）")
    scaler = st.Scaler.fit(pre_sets)

    # 3. 版の種類（2026-10-22。design.md §15.2・§15.3）：重みあり版は正例の重み w ＝ 負例 ÷ 正例（事前学習の区間の Change × 日）
    v = constants.ENCODER_VARIANT
    pos_weight = st.class_balance_weight(pre_sets) if v == variant.POSW else 1.0
    logger.info(f"版の種類：{v}（{variant.LABEL[v]}）・正例の重み w = {pos_weight:.4f}")
    if store.list_seeds(project, cutoff_label):
        variant.warn_if_exists(store._cutoff_dir(project, cutoff_label), "事前学習エンコーダ")

    # 4. 指定 index のエンコーダ＋汎用ヘッドを事前学習して保存
    #    2026-10-20：途中で非有限になった seed は保存せず記録して、次の seed へ進む（design.md §4 の NaN の点検）
    failed: list[int] = []
    for i, k in enumerate(seed_indices):
        seed = constants.RANDOM_SEED + k
        logger.info(f"事前学習 {i + 1}/{len(seed_indices)}（seed{k}, torch_seed={seed}）")
        try:
            enc, gh = st.pretrain(pre_sets, scaler, seed, device, pos_weight=pos_weight)
        except st.NonFiniteError as e:
            rec = store.save_nonfinite_record(project, cutoff_label, k, {"torch_seed": seed, **e.info})
            logger.error(f"[{project}] seed{k} は保存しません：{e}（記録：{rec}）")
            failed.append(k)
            continue
        out = store.save_pretrained(project, cutoff_label, k, enc, gh, scaler,
                                    {"torch_seed": seed, "device": str(device),    # 回した機械（design.md §14.5）
                                     "variant": v, "pos_weight": pos_weight})      # 版の種類と正例の重み（§15.3。2026-10-22）
        logger.info(f"保存: {out}")

    logger.info("完了" if not failed else f"完了（非有限で保存しなかった seed：{failed}）")
    return failed


def main() -> None:
    ap = argparse.ArgumentParser(description="事前学習エンコーダ＋汎用ヘッドの作成・保存")
    ap.add_argument("--project", default=None,
                    help="対象プロジェクト（未指定なら constants.PROJECTS の全プロジェクトを順に。'all' も同義）")
    ap.add_argument("--n", type=int, default=None,
                    help="作るエンコーダ数（既定 constants.N_REPEATS）")
    variant.add_argument(ap)
    ap.add_argument("--workers", type=int, default=None,
                    help="リポジトリを並べて回す数（既定：CUDA なら --gpu-share から自動、Mac などは 1。design.md §14.7・§15.6）")
    ap.add_argument("--gpu-share", type=float, default=parallel.DEFAULT_GPU_SHARE,
                    help="並べたプロセス全体で使ってよい GPU のメモリの割合（既定 0.25 ＝ 1/4。誰も使っていなければ 1.0 まで）")
    args = ap.parse_args()
    v = variant.apply(variant.resolve(args.variant, writing=True))      # 事前学習は書き込みをするので --variant 必須（§15.2）
    projects = list(constants.PROJECTS) if args.project in (None, "all") else [args.project]
    if args.gpu_share > parallel.DEFAULT_GPU_SHARE:
        logger.warning(f"GPU のメモリを全体の {args.gpu_share:.0%} まで使います（既定の 1/4 を超えています）")
    w, frac = parallel.decide(len(projects), use_gpu=True, gpu_share=args.gpu_share, workers=args.workers)
    if w > 1:                   # リポジトリごとに別々のプロセスで並べる（§15.6）。子は --project 1 件・--workers 1 で同じ入口を呼ぶ
        child = ["--variant", v] + (["--n", str(args.n)] if args.n else [])
        bad = parallel.run_children("src.analysis.preliminary_analysis.pretrained_encoders.build_encoders", projects,
                                    child, w, frac, variant.log_name("pretrained_encoders", v))
        sys.exit(1 if bad else 0)
    if frac is not None and not os.environ.get("RP_CUDA_MEMORY_FRACTION"):
        constants.CUDA_MEMORY_FRACTION = frac                   # 1 つで回すときは --gpu-share をそのまま使う
    failed: dict[str, list[int]] = {}
    for project in projects:
        logger.info(f"===== プロジェクト: {project} =====")
        bad = build_and_save(project=project, n=args.n)
        if bad:
            failed[project] = bad
    if failed:   # 止めずに全部回したうえで、飛ばしたものを最後にまとめて知らせる（design.md §4）
        logger.error(f"非有限で保存しなかった seed：{failed}（記録は <OUTPUT>/<project>/cutoff_<cutoff>/seed<k>_nonfinite.json）")
        sys.exit(1)


if __name__ == "__main__":
    main()
