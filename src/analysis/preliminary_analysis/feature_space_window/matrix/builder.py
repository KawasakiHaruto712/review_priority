"""特徴のずれ×評価時期の精度表の計算（feature_space_window/design.md §9.1）。

seed（エンコーダ）ごとに：
  1. 全日（候補と評価日）を 1 回だけ埋め込み、日の代表（埋め込みの平均）を出す。
  2. 評価日 X ごとに、候補（X より前の全日）を近い順に並べ、**連続した k 日ずつの群**を行 g にする（g0 ＝ 最も近い 1〜k 番目）。
  3. 行ごとに probe を学習し X で評価（RQ1 と同じ決まり：学習・評価のデータの最小件数、正例・負例が 1 件以上）。汎用ヘッドも X で評価。
評価日は版の間で重なりうる（列の区切りが前の版の日まで届くため）が、RQ2 の選び方は X だけで決まるので、X ごとに 1 回だけ計算して版で使い回す。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from src.analysis.preliminary_analysis.feature_space_window.evaluation import metrics
from src.analysis.preliminary_analysis.feature_space_window.selection.nearest_days import DayIndex, day_means
from src.analysis.preliminary_analysis.feature_space_window.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.dataset import set_builder
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st

logger = logging.getLogger(__name__)


@dataclass
class SeedResults:
    """全 seed の結果。probe[(X, g)][seed] ＝ (指標 or None, 理由, 件数)、general[X][seed] ＝ (指標 or None, 理由, 件数)。"""
    probe: dict = field(default_factory=dict)
    general: dict = field(default_factory=dict)
    probes: dict = field(default_factory=dict)        # seed -> [(X, g, 129 個の重み)]
    train_days: list = field(default_factory=list)   # (X, g, seed, 学習に使った日)


def _counts(sets) -> tuple[int, int]:
    total = sum(len(s) for s in sets)
    pos = int(sum(sum(s.labels) for s in sets))
    return pos, total - pos


def head_vector(head) -> np.ndarray:
    """線形ヘッド → 129 個（重み 128 ＋ 切片 1）。RQ1 の probe の保存と同じ形（`st.head_from_vector` で戻せる）。"""
    with torch.no_grad():
        w = head.net.weight.detach().cpu().numpy().reshape(-1)
        b = head.net.bias.detach().cpu().numpy().reshape(-1)
    return np.concatenate([w, b]).astype(np.float32)


def run_seed(res: SeedResults, day_sets: dict, eval_rows: dict, encoder, general_head, scaler, seed_idx: int,
             k: int, device, metric_names=None, row_offsets: dict | None = None) -> None:
    """1 seed 分を res に足す。eval_rows は {評価日 X -> その X を含む版の中で最大の行数 N_v}。

    row_offsets（2026-10-22。重みあり版の 3 × 3。design.md §10.1）：{評価日 X -> [行 g の始まりの順位（0 始まり）, ...]}。
    渡すと、行 g の学習の日は近い順で「始まり ＋ 1 〜 始まり ＋ k 番目」になる（始まり ＝ 刻み × g）。省くと今までどおり g × k。
    """
    metric_names = metric_names or metrics.metric_columns(constants.K_LIST)
    seed = constants.RANDOM_SEED + seed_idx
    thr = constants.CLASSIFY_THRESHOLD
    dates = list(day_sets)
    embeds = st.embed_sets(encoder, scaler, [day_sets[d] for d in dates], device)
    emb = dict(zip(dates, embeds))
    index = DayIndex(dates, day_means(embeds))
    pos_of = {d: i for i, d in enumerate(dates)}
    saved = res.probes.setdefault(seed_idx, [])
    for n, (x, rows) in enumerate(sorted(eval_rows.items()), start=1):
        es = day_sets.get(x)
        if es is None:
            continue
        yt = np.asarray(es.labels, dtype=float)
        npe, nne = int(yt.sum()), int(len(yt) - yt.sum())
        eval_ok = npe >= 1 and nne >= 1 and len(es) >= constants.MIN_EVAL
        if eval_ok:
            g_rows = st.predict(general_head, [emb[x]], [es], device)
            gm = metrics.compute_day_metrics([r[0] for r in g_rows], [r[1] for r in g_rows], constants.K_LIST, thr)
            res.general.setdefault(x, {})[seed_idx] = (gm, "", (npe, nne, -1, -1))
        else:
            res.general.setdefault(x, {})[seed_idx] = (None, "eval_guard", (npe, nne, -1, -1))
        offsets = row_offsets[x] if row_offsets is not None else [g * k for g in range(rows)]
        near, _dist, _before = index.nearest(x, index.means[pos_of[x]], max(offsets[:rows]) + k)
        for g in range(rows):
            tdays = near[offsets[g]:offsets[g] + k]
            tsets = [day_sets[d] for d in tdays]
            npt, nnt = _counts(tsets) if tsets else (0, 0)
            reason = ""
            if not eval_ok:
                reason = "eval_guard"
            elif len(tdays) < k:
                reason = "insufficient_candidates"      # 候補が g 行目の k 日に届かない（§9.1。まず起きない）
            elif npt < 1 or nnt < 1 or set_builder.count_records(tsets) < constants.MIN_TRAIN:
                reason = "train_guard"
            if reason:
                res.probe.setdefault((x, g), {})[seed_idx] = (None, reason, (npe, nne, npt, nnt))
                continue
            head = st.train_probe([emb[d] for d in tdays], tsets, seed, device)
            if head is None:
                res.probe.setdefault((x, g), {})[seed_idx] = (None, "train_guard", (npe, nne, npt, nnt))
                continue
            p_rows = st.predict(head, [emb[x]], [es], device)
            mv = metrics.compute_day_metrics([r[0] for r in p_rows], [r[1] for r in p_rows], constants.K_LIST, thr)
            res.probe.setdefault((x, g), {})[seed_idx] = (mv, "", (npe, nne, npt, nnt))
            saved.append((x, g, head_vector(head)))
            res.train_days.extend((x, g, seed_idx, d) for d in tdays)
        if n % 100 == 0:
            logger.info(f"seed{seed_idx}：評価日 {n}/{len(eval_rows)}")
    del embeds, emb


def version_values(res: SeedResults, positions: dict, grid: int, metric_names: list[str]):
    """1 版分の probe_vals・gen_vals（aggregate.assemble の入力）と、日ごとの指標の行。"""
    from src.analysis.preliminary_analysis.feature_space_window.matrix.aggregate import daily_row

    probe_vals = {m: {} for m in metric_names}
    gen_vals = {m: {} for m in metric_names}
    rows_out: list[dict] = []
    for x, p in positions.items():
        for seed_idx, (gm, reason, (npe, nne, _a, _b)) in res.general.get(x, {}).items():
            rows_out.append(daily_row(x, "general", seed_idx, p, npe, nne, -1, -1, gm is None, reason, gm))
            if gm is not None:
                for m in metric_names:
                    if not np.isnan(gm[m]):
                        gen_vals[m].setdefault(p, {}).setdefault(x, {})[seed_idx] = gm[m]
        for g in range(grid):
            for seed_idx, (mv, reason, (npe, nne, npt, nnt)) in res.probe.get((x, g), {}).items():
                rows_out.append(daily_row(x, g, seed_idx, p, npe, nne, npt, nnt, mv is None, reason, mv))
                if mv is not None:
                    for m in metric_names:
                        if not np.isnan(mv[m]):
                            probe_vals[m].setdefault((g, p), {}).setdefault(x, {})[seed_idx] = mv[m]
    return probe_vals, gen_vals, rows_out
