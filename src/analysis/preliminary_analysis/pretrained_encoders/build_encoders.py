"""事前学習エンコーダ＋汎用ヘッドの作成・保存（design.md）。

流れ（上から順に追える構成）:
  1. データ読み込み（changes / release_dates / bot 名 / 特徴用 DataFrame）
  2. 締め切り（PROJECTS[project] の cutoff リリース日）を決める
  3. プロジェクト開始 〜 締め切り の計測点で pretraining 集合を作る
  4. N 個（seed 違い）のエンコーダ＋汎用ヘッドを事前学習して保存

実行:
    python -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders                    # nova を N_REPEATS 個
    python -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders --project neutron  # 対象プロジェクト指定
    python -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders --n 3              # 個数を指定
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.analysis.background_problem.common.data_loader import load_changes, load_release_dates
from src.analysis.preliminary_analysis.pretrained_encoders.dataset import record_builder, set_builder
from src.analysis.preliminary_analysis.pretrained_encoders.features import feature_builder
from src.analysis.preliminary_analysis.pretrained_encoders.io import store
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants, review_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _resolve_cutoff(rel_df: pd.DataFrame, project: str):
    """事前学習の締め切り (datetime, ラベル) を返す。

    ラベル = PROJECTS[project]["cutoff"] の版。日付はその版の release_date を
    major_releases_summary.csv（rel_df）から引く（日付は二重管理しない）。
    """
    cfg = constants.PROJECTS.get(project)
    if cfg is None:
        raise ValueError(f"未対応のプロジェクトです（PROJECTS に定義してください）: {project}")
    label = cfg["cutoff"]
    row = rel_df[(rel_df["project"] == project) & (rel_df["version"] == label)]
    if row.empty:
        raise ValueError(f"cutoff リリースが見つかりません: {project} {label}")
    dt = pd.to_datetime(row["release_date"].iloc[0]).to_pydatetime()
    return dt, label


def build_and_save(project: str = None, n: int | None = None, seed_indices=None) -> None:
    """指定プロジェクトの共有エンコーダ＋汎用ヘッドを作って保存する。

    project: 対象プロジェクト（既定 constants.TARGET_PROJECT="nova"）。PROJECTS に定義が必要。
    seed_indices を渡すと、その index(k) だけを作る（不足分の追加ビルド用）。
    未指定なら 0..n-1（n 既定 N_REPEATS）。
    """
    project = constants.TARGET_PROJECT if project is None else project
    if seed_indices is None:
        n = constants.N_REPEATS if n is None else n
        seed_indices = list(range(n))
    else:
        seed_indices = list(seed_indices)

    # 1. データ読み込み
    rel_df = load_release_dates()
    changes = load_changes(project)
    bot_names = review_utils.load_bot_names()
    all_prs = feature_builder.build_all_prs_df(changes)
    device = st.resolve_device()
    logger.info(f"device = {device}")

    # 2. 締め切り
    cutoff_dt, cutoff_label = _resolve_cutoff(rel_df, project)
    data_start = all_prs["created"].min().to_pydatetime()
    logger.info(f"事前学習データ: {data_start.date()} 〜 {cutoff_dt.date()}（cutoff={cutoff_label}）")

    # 3. pretraining 集合
    pre_records = record_builder.build_records(changes, project, data_start, cutoff_dt,
                                               bot_names, all_prs, rel_df)
    pre_sets = set_builder.build_sets(pre_records, constants.MAX_SET_SIZE)
    logger.info(f"事前学習用 集合数: {len(pre_sets)}（総レコード {set_builder.count_records(pre_sets)}）")
    scaler = st.Scaler.fit(pre_sets)

    # 4. 指定 index のエンコーダ＋汎用ヘッドを事前学習して保存
    for i, k in enumerate(seed_indices):
        seed = constants.RANDOM_SEED + k
        logger.info(f"事前学習 {i + 1}/{len(seed_indices)}（seed{k}, torch_seed={seed}）")
        enc, gh = st.pretrain(pre_sets, scaler, seed, device)
        out = store.save_pretrained(project, cutoff_label, k, enc, gh, scaler, {"torch_seed": seed})
        logger.info(f"保存: {out}")

    logger.info("完了")


def main() -> None:
    ap = argparse.ArgumentParser(description="事前学習エンコーダ＋汎用ヘッドの作成・保存")
    ap.add_argument("--project", default=constants.TARGET_PROJECT,
                    help=f"対象プロジェクト（既定 {constants.TARGET_PROJECT}）")
    ap.add_argument("--n", type=int, default=None,
                    help="作るエンコーダ数（既定 constants.N_REPEATS）")
    args = ap.parse_args()
    build_and_save(project=args.project, n=args.n)


if __name__ == "__main__":
    main()
