"""RQ2：事前学習の表と評価期間の表をつないで、日の集合を作る（design.md §3.2。2026-10-22）。

RQ2 の学習の日は評価期間より前（事前学習の区間）にもあるので、2 つの表をつなぐ。重なった日は評価期間の表を使う
（feature_space_window/dataset/candidates.py と同じ決まり。import はせず、data_drift の中で作る）。
メモリを減らすため、要る日（評価日と学習の日）だけを残す。
"""
from __future__ import annotations

import logging

import pandas as pd

from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre

logger = logging.getLogger(__name__)


def merge_day_sets(pretrain: dict, evaluation: dict) -> dict:
    """{日付 -> TSet} を 2 つつなぐ。重なった日は evaluation を使う。日付の昇順で返す。"""
    out = dict(pretrain)
    out.update(evaluation)
    return dict(sorted(out.items()))


def load_day_sets(project: str, rel_df: pd.DataFrame, keep: set | None = None) -> tuple[dict, set]:
    """(日の集合 {日付 -> TSet}, 事前学習の表にある日付の集合) を返す。keep を渡すと、その日だけを残す。

    表は ensure_table で読む（なければ作る・指紋が違えば警告して続ける）。
    """
    from src.analysis.preliminary_analysis.feature_table.build import ensure_table
    from src.analysis.preliminary_analysis.feature_table.build.sets import day_sets_from_table

    def _sets(period: str) -> dict:
        tab = ensure_table(project, period, rel_df=rel_df)
        out = {s.t.date(): s for s in day_sets_from_table(tab, max_set_size=pre.MAX_SET_SIZE)}
        del tab
        return out

    pre_sets = _sets("pretrain")
    pretrain_days = set(pre_sets)
    if keep is not None:
        pre_sets = {d: s for d, s in pre_sets.items() if d in keep}
    eval_sets = _sets("eval")
    if keep is not None:
        eval_sets = {d: s for d, s in eval_sets.items() if d in keep}
    merged = merge_day_sets(pre_sets, eval_sets)
    logger.info(f"[{project}] 日の集合：事前学習の表から {len(pre_sets)} 日・評価期間の表から {len(eval_sets)} 日"
                f"（重なりは評価期間の表を使う）→ {len(merged)} 日")
    return merged, pretrain_days
