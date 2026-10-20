"""1 seed 分のループ（design.md §4.1）：全日を 1 回だけ埋め込み → 評価日ごとに近い k 日を選ぶ → probe を学習 → 評価日で評価。

欠測（design.md §4。lookback_window §6 と同じ決まり）：学習の k 日に正例 ≥ 1 かつ負例 ≥ 1、評価日に正例 ≥ 1 かつ負例 ≥ 1。
満たさなければ欠測として記録し、次に近い日で補うことはしない。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.feature_space_window.evaluation import metrics as M
from src.analysis.preliminary_analysis.feature_space_window.selection.nearest_days import DayIndex, day_means
from src.analysis.preliminary_analysis.feature_space_window.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st

logger = logging.getLogger(__name__)


@dataclass
class VersionRows:
    """1 版分の結果（seed をまたいで足していく）。"""
    preds: list[pd.DataFrame] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    selected: list[dict] = field(default_factory=list)

    def extend(self, other: "VersionRows") -> None:
        self.preds += other.preds
        self.metrics += other.metrics
        self.selected += other.selected


def _counts(sets) -> tuple[int, int]:
    total = sum(len(s) for s in sets)
    pos = int(sum(sum(s.labels) for s in sets))
    return pos, total - pos


def _metric_row(day, size, seed, n_pos_e, n_neg_e, n_pos_t, n_neg_t, n_days_t,
                missing: bool, reason: str, mvals: dict | None) -> dict:
    row = {"day": day, "size": size, "seed": seed,
           "n_pos_eval": n_pos_e, "n_neg_eval": n_neg_e,
           "n_pos_train": n_pos_t, "n_neg_train": n_neg_t, "n_days_train": n_days_t,
           "missing": int(missing), "missing_reason": reason}
    for col in M.metric_columns(constants.K_LIST):
        row[col] = (mvals or {}).get(col, M.NAN)
    return row


def _pred_frame(day_iso: str, size: str, seed: int, rows) -> pd.DataFrame:
    return pd.DataFrame({"day": day_iso, "size": size, "seed": seed,
                         "change_id": [r[2] for r in rows],
                         "y_true": np.array([r[0] for r in rows], dtype=np.int8),
                         "score": np.array([r[1] for r in rows], dtype=np.float32)})


def run_seed(day_sets: dict, pretrain_days: set, version_eval: dict, encoder, scaler, seed: int, device,
             sizes=None) -> dict:
    """1 エンコーダ（seed）分。{版 -> VersionRows} を返す。

    day_sets     : {日付 -> TSet}（昇順。候補と評価日を含む全日。dataset.candidates.load_all_day_sets）
    pretrain_days: 事前学習の表にある日付（選ばれた日の in_pretrain の印）
    version_eval : {版 -> 評価日の列}
    """
    sizes = sorted(constants.SIZES_DAYS if sizes is None else sizes)
    max_k = max(sizes)
    dates = list(day_sets)
    embeds = st.embed_sets(encoder, scaler, [day_sets[d] for d in dates], device)   # 1 回だけ（§4.1）
    emb_by_date = dict(zip(dates, embeds))
    index = DayIndex(dates, day_means(embeds))
    pos_of = {d: i for i, d in enumerate(dates)}

    out: dict = {}
    for version, eval_dates in version_eval.items():
        res = VersionRows()
        for X in eval_dates:
            eval_set = day_sets.get(X)
            if eval_set is None or len(eval_set) == 0:
                continue
            day_iso = X.isoformat()
            yt = np.asarray(eval_set.labels, dtype=float)
            n_pos_e, n_neg_e = int(yt.sum()), int(len(yt) - yt.sum())
            eval_ok = n_pos_e >= 1 and n_neg_e >= 1

            near, dist, before = index.nearest(X, index.means[pos_of[X]], max_k)
            # 選ばれた日の一覧：大きさ max_k の分だけ残す（大きさごとの選び方は入れ子なので rank で分かる。design.md §6）
            for r, (d, ds, b) in enumerate(zip(near, dist, before), start=1):
                res.selected.append({"day": day_iso, "seed": seed, "rank": r, "selected_day": d.isoformat(),
                                     "days_before": int(b), "distance": float(ds), "in_pretrain": int(d in pretrain_days)})

            for k in sizes:
                label = constants.SIZE_LABELS.get(k, f"{k}d")
                tdates = near[:k]
                tsets = [day_sets[d] for d in tdates]
                n_pos_t, n_neg_t = _counts(tsets) if tsets else (0, 0)
                if not eval_ok:
                    reason = "eval_single_class"
                elif not tsets:
                    reason = "no_train_data"
                elif n_pos_t < 1 or n_neg_t < 1:
                    reason = "train_single_class"
                else:
                    reason = ""
                if reason:
                    res.metrics.append(_metric_row(day_iso, label, seed, n_pos_e, n_neg_e, n_pos_t, n_neg_t,
                                                   len(tdates), True, reason, None))
                    continue
                head = st.train_probe([emb_by_date[d] for d in tdates], tsets, seed, device)
                if head is None:
                    res.metrics.append(_metric_row(day_iso, label, seed, n_pos_e, n_neg_e, n_pos_t, n_neg_t,
                                                   len(tdates), True, "train_probe_none", None))
                    continue
                rows = st.predict(head, [emb_by_date[X]], [eval_set], device)
                mv = M.compute_day_metrics([r[0] for r in rows], [r[1] for r in rows],
                                           constants.K_LIST, constants.CLASSIFY_THRESHOLD)
                res.metrics.append(_metric_row(day_iso, label, seed, n_pos_e, n_neg_e, n_pos_t, n_neg_t,
                                               len(tdates), False, "", mv))
                res.preds.append(_pred_frame(day_iso, label, seed, rows))
        out[version] = res
        logger.info(f"seed{seed} {version}：評価日 {len(eval_dates)}・指標 {len(res.metrics)} 行")
    del embeds, emb_by_date
    return out
