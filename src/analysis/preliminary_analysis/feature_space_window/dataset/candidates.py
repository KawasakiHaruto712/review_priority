"""候補にする過去の日の集合を、事前学習の表と評価期間の表をつないで作る（design.md §2.1）。

- 候補：評価日 X より前のすべての日（事前学習の区間と、評価期間のうち X より前の日）。
- 2 つの表は重なることがある。重なった日は評価期間の表を使う（同じコード・同じ指紋で作った表なので値は同じ）。
- 事前学習の区間の日には印を付ける（選ばれた日の一覧の in_pretrain。エンコーダの学習に使った日なので偏りを数えるため）。
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd

from src.analysis.background_problem.common.data_loader import _drop_release_anomalies
from src.analysis.preliminary_analysis.pretrained_encoders.dataset.set_builder import TSet
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre

logger = logging.getLogger(__name__)


def merge_day_sets(pretrain: dict, evaluation: dict) -> dict:
    """{日付 -> TSet} を 2 つつなぐ。重なった日は evaluation を使う。日付の昇順で返す。"""
    out = dict(pretrain)
    out.update(evaluation)
    return dict(sorted(out.items()))


def _by_date(sets: list[TSet]) -> dict:
    return {s.t.date(): s for s in sets}


def load_all_day_sets(project: str, rel_df: pd.DataFrame) -> tuple[dict, set]:
    """(全日の {日付 -> TSet}, 事前学習の表にある日付の集合) を返す。表は ensure_table で読む（なければ作る・指紋が違えば止める）。"""
    from src.analysis.preliminary_analysis.feature_table.build import ensure_table
    from src.analysis.preliminary_analysis.feature_table.build.sets import day_sets_from_table

    tab = ensure_table(project, "pretrain", rel_df=rel_df)
    pre_sets = _by_date(day_sets_from_table(tab, max_set_size=pre.MAX_SET_SIZE))
    del tab
    tab = ensure_table(project, "eval", rel_df=rel_df)
    eval_sets = _by_date(day_sets_from_table(tab, max_set_size=pre.MAX_SET_SIZE))
    del tab
    overlap = set(pre_sets) & set(eval_sets)
    logger.info(f"[{project}] 事前学習の表 {min(pre_sets)}〜{max(pre_sets)}（{len(pre_sets)} 日）・"
                f"評価期間の表 {min(eval_sets)}〜{max(eval_sets)}（{len(eval_sets)} 日）・重なり {len(overlap)} 日"
                f"（評価期間の表を使う）")
    if max(pre_sets) < min(eval_sets):
        gap = (min(eval_sets) - max(pre_sets)).days - 1
        if gap > 0:
            logger.warning(f"[{project}] 2 つの表の間に {gap} 日のすき間があります（候補に入らない日）")
    return merge_day_sets(pre_sets, eval_sets), set(pre_sets)


def release_cycle(rel_df: pd.DataFrame, project: str, version: str) -> tuple[datetime, datetime]:
    """対象リリースの開発サイクル (cs_R＝直前リリース日, ce_R＝当該リリース日)。lookback_window と同じ決まりの複製（自己完結のため）。"""
    pdf = _drop_release_anomalies(rel_df[rel_df["project"] == project])
    dates = sorted(pd.to_datetime(pdf["release_date"]).tolist())
    tgt = rel_df[(rel_df["project"] == project) & (rel_df["version"] == version)]
    if tgt.empty:
        raise ValueError(f"リリースが見つかりません: {project} {version}")
    ce = pd.to_datetime(tgt["release_date"].iloc[0])
    earlier = [d for d in dates if d < ce]
    if not earlier:
        raise ValueError(f"直前リリースが見つかりません: {project} {version}")
    return earlier[-1].to_pydatetime(), ce.to_pydatetime()


def eval_dates_for(day_sets: dict, cs_R: datetime, ce_R: datetime) -> list[date]:
    """版のサイクルの評価日（集合がある日。lookback_window と同じ [cs_R, ce_R]）。"""
    return [d for d in day_sets if cs_R.date() <= d <= ce_R.date()]
