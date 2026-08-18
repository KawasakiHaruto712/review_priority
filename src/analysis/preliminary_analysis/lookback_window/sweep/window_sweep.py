"""窓スイープ本体（design.md §4, §5, §6）。

- release_cycle : 対象リリースのサイクル [cs_R, ce_R] を返す
- build_day_sets: 必要な期間の (Change, T) レコード → 日ごとの TSet（1 日 1 集合）
- run_sweep     : 各 (評価日 × 窓長) で probe を貼り直して予測・全指標を算出（＋汎用ヘッド基準線）

学習窓 = 評価日の直前 W 日（[X−W, X−1]。末尾＝前日 X−1。評価日自身は予測対象なので学習に入れない）。
欠測ガード：probe は学習窓に正例≥1 かつ 負例≥1、評価は評価日に正例≥1 かつ 負例≥1。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.analysis.background_problem.common.data_loader import _drop_release_anomalies
from src.analysis.preliminary_analysis.lookback_window.evaluation import metrics as M
from src.analysis.preliminary_analysis.lookback_window.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.dataset import record_builder, set_builder
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre

logger = logging.getLogger(__name__)


def release_cycle(rel_df: pd.DataFrame, project: str, version: str):
    """対象リリースの開発サイクル (cs_R, ce_R) を返す（cs_R=直前リリース日, ce_R=当該リリース日）。"""
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


def build_day_sets(changes, project, span_start: datetime, ce_R: datetime,
                   bot_names, all_prs, rel_df) -> dict:
    """[span_start, ce_R] の (Change, T) レコードを日ごとの TSet にまとめる。{date -> TSet}。"""
    recs = record_builder.build_records(changes, project, span_start, ce_R, bot_names, all_prs, rel_df)
    sets = set_builder.build_sets(recs, pre.MAX_SET_SIZE)
    return {s.t.date(): s for s in sets}


def _counts(sets) -> tuple[int, int]:
    """集合群の (正例数, 負例数)。"""
    total = sum(len(s) for s in sets)
    pos = int(sum(sum(s.labels) for s in sets))
    return pos, total - pos


def _metric_row(day, window, seed, n_pos_e, n_neg_e, n_pos_t, n_neg_t,
                missing: bool, reason: str, mvals: dict | None) -> dict:
    row = {"day": day, "window": window, "seed": seed,
           "n_pos_eval": n_pos_e, "n_neg_eval": n_neg_e,
           "n_pos_train": n_pos_t, "n_neg_train": n_neg_t,
           "missing": int(missing), "missing_reason": reason}
    for col in M.metric_columns(constants.K_LIST):
        row[col] = (mvals or {}).get(col, M.NAN)
    return row


def run_sweep(day_sets_by_date: dict, eval_dates: list, encoder, general_head, scaler,
              seed: int, device, windows=None):
    """1 エンコーダ（seed）分のスイープ。(pred_rows, metric_rows) を返す。

    pred_rows:   (day, window, seed, change_id, y_true, score)
    metric_rows: 上記 _metric_row の dict（day×window×seed。汎用ヘッドは window='general'）
    """
    windows = constants.WINDOWS_DAYS if windows is None else windows

    # 全日の集合を 1 回だけ埋め込み（凍結エンコーダ。使い回す）
    all_dates = sorted(day_sets_by_date)
    all_sets = [day_sets_by_date[d] for d in all_dates]
    all_embeds = st.embed_sets(encoder, scaler, all_sets, device)
    emb_by_date = {d: e for d, e in zip(all_dates, all_embeds)}

    pred_rows: list[tuple] = []
    metric_rows: list[dict] = []

    for X in eval_dates:
        eval_set = day_sets_by_date.get(X)
        if eval_set is None or len(eval_set) == 0:
            continue
        eval_emb = emb_by_date[X]
        yt = np.array(eval_set.labels, dtype=float)
        n_pos_e, n_neg_e = int(yt.sum()), int(len(yt) - yt.sum())
        eval_ok = n_pos_e >= 1 and n_neg_e >= 1
        day_iso = X.isoformat()

        # ── 汎用ヘッド（チューニングなし）基準線: window='general' ──
        if not eval_ok:
            metric_rows.append(_metric_row(day_iso, "general", seed, n_pos_e, n_neg_e,
                                            -1, -1, True, "eval_single_class", None))
        else:
            grows = st.predict(general_head, [eval_emb], [eval_set], device)
            gm = M.compute_day_metrics([r[0] for r in grows], [r[1] for r in grows],
                                       constants.K_LIST, constants.CLASSIFY_THRESHOLD)
            metric_rows.append(_metric_row(day_iso, "general", seed, n_pos_e, n_neg_e,
                                           -1, -1, False, "", gm))
            for yt_i, yp_i, cid, _t in grows:
                pred_rows.append((day_iso, "general", seed, cid, yt_i, yp_i))

        # ── 各窓長で probe を貼り直す ──
        for W in windows:
            wlabel = constants.WINDOW_LABELS.get(W, f"{W}d")
            train_end = X - timedelta(days=1)          # 前日まで（評価日自身は予測対象なので入れない）
            # 注：Δ=1日 前提での「前日」。将来 Δ を変える場合はここを X−Δ（Δ日前）にする。
            train_start = train_end - timedelta(days=W - 1)
            tdates = [d for d in all_dates if train_start <= d <= train_end]
            tsets = [day_sets_by_date[d] for d in tdates]
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
                metric_rows.append(_metric_row(day_iso, wlabel, seed, n_pos_e, n_neg_e,
                                                n_pos_t, n_neg_t, True, reason, None))
                continue

            tembeds = [emb_by_date[d] for d in tdates]
            head = st.train_probe(tembeds, tsets, seed, device)
            if head is None:  # 念のため（ガード通過後でも単クラス等）
                metric_rows.append(_metric_row(day_iso, wlabel, seed, n_pos_e, n_neg_e,
                                                n_pos_t, n_neg_t, True, "train_probe_none", None))
                continue

            rows = st.predict(head, [eval_emb], [eval_set], device)
            mv = M.compute_day_metrics([r[0] for r in rows], [r[1] for r in rows],
                                       constants.K_LIST, constants.CLASSIFY_THRESHOLD)
            metric_rows.append(_metric_row(day_iso, wlabel, seed, n_pos_e, n_neg_e,
                                           n_pos_t, n_neg_t, False, "", mv))
            for yt_i, yp_i, cid, _t in rows:
                pred_rows.append((day_iso, wlabel, seed, cid, yt_i, yp_i))

    return pred_rows, metric_rows


def span_start_for(cs_R: datetime) -> datetime:
    """最長窓＋前日ぶん手前まで遡った学習データ開始点。"""
    back = 1 + max(constants.WINDOWS_DAYS)
    return cs_R - timedelta(days=back)
