"""学習×予測行列（d×p）の構築（§5）。再設計版：目的変数は Δ以内2値、指標は precision/recall/f1。

セル (i, j)（i<j, 距離 d=j-i>0, 位置 p=j）ごとに n_repeats 回:
  - 学習ビン i から n_train Change を無作為抽出（混ぜ許容・層なし。sampler）、change 重みで学習。
  - 評価ビン j の全 Change（or n_eval）で予測 → precision/recall/f1 をプールして算出（1反復の代表値）。
  - 反復方向に repeat_agg（中央値）でセル値。ばらつき(IQR)と各反復値も保持。
Δ マージン：学習は末尾 Δ を除外（TRAIN_TAIL_MARGIN）、評価は既定で除外しない（EVAL_TAIL_MARGIN）。
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from src.analysis.preliminary_analysis.concept_drift_detection.dataset import sampler
from src.analysis.preliminary_analysis.concept_drift_detection.evaluation import binary_metrics
from src.analysis.preliminary_analysis.concept_drift_detection.model import classifier
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants

logger = logging.getLogger(__name__)

_AGG = {"median": np.median, "mean": np.mean}


@dataclass
class MatrixResult:
    """1 指標ぶんの四角行列（縦=距離 d、横=位置 p）。"""
    metric: str
    bin_count: int
    value: np.ndarray  # shape (bin_count, bin_count): [d-1, p]。欠測は NaN
    iqr: np.ndarray    # 反復のばらつき（同形）
    per_repeat: dict = field(default_factory=dict)  # (d, p) -> 各反復の代表値リスト


def _xy(records):
    x = np.array([r.features for r in records], dtype=float)
    y = np.array([r.labels[constants.TARGET] for r in records], dtype=float)
    return x, y


def _change_balanced_weights(records) -> np.ndarray:
    """各行に 1/(その Change のレコード数)。Change ごとの合計重みが 1 になる（長寿の水増し是正）。"""
    counts = defaultdict(int)
    for r in records:
        counts[r.change_id] += 1
    return np.array([1.0 / counts[r.change_id] for r in records], dtype=float)


def _tail_cutoff(cycle_start: datetime, bw_sec: float, bin_idx: int, delta: timedelta) -> datetime:
    """ビン bin_idx の終端から Δ 手前の時刻（この時刻以前の計測点だけ使う）。"""
    bin_end = cycle_start + timedelta(seconds=(bin_idx + 1) * bw_sec)
    return bin_end - delta


def _repeat_value_binary(records_pred, metric_names, threshold) -> dict:
    """1反復の二値指標。評価レコードをプールして算出。EVAL 重みが ON なら Change 均等重み。"""
    if not records_pred:
        return {m: np.nan for m in metric_names}
    yt = np.array([a for a, _, _ in records_pred], dtype=float)
    yp = np.array([b for _, b, _ in records_pred], dtype=float)
    sw = None
    if constants.EVAL_CHANGE_BALANCED_WEIGHT:
        counts = defaultdict(int)
        for _, _, cid in records_pred:
            counts[cid] += 1
        sw = np.array([1.0 / counts[cid] for _, _, cid in records_pred], dtype=float)
    return binary_metrics.compute(yt, yp, metric_names, threshold=threshold, sample_weight=sw)


def compute_matrices(rows, *, metric_names=None, n_repeats=None, repeat_agg=None,
                     bin_count=None, threshold=None) -> dict[str, MatrixResult]:
    """予測行から全指標の四角行列を作る（実行時・再計算の単一経路）。

    rows: (d, p, repeat, t, change_id, y_true(0/1), y_pred(正例確率)) のタプル列（d・p は 1 始まり）。
    """
    metric_names = metric_names or constants.ENABLED_METRICS
    n_repeats = constants.N_REPEATS if n_repeats is None else n_repeats
    repeat_agg = repeat_agg or constants.REPEAT_AGG
    bin_count = constants.BIN_COUNT if bin_count is None else bin_count
    threshold = constants.CLASSIFY_THRESHOLD if threshold is None else threshold

    shape = (bin_count, bin_count)
    results = {m: MatrixResult(m, bin_count, np.full(shape, np.nan), np.full(shape, np.nan))
               for m in metric_names}
    agg = _AGG[repeat_agg]

    # (d, p) -> {repeat -> [(y_true, y_pred, change_id), ...]}
    cells: dict = defaultdict(lambda: defaultdict(list))
    for d, p, k, _t, cid, y_true, y_pred in rows:
        cells[(int(d), int(p))][int(k)].append((float(y_true), float(y_pred), cid))

    for (d, p), reps_map in cells.items():
        reps = {m: [] for m in metric_names}
        for _k, recs in reps_map.items():
            vals = _repeat_value_binary(recs, metric_names, threshold)
            for m in metric_names:
                if not np.isnan(vals[m]):
                    reps[m].append(vals[m])
        col = p - 1  # 行列の列は 0 始まり（rows の p は 1 始まり）
        for m in metric_names:
            if reps[m]:
                results[m].value[d - 1, col] = float(agg(reps[m]))
                q75, q25 = np.percentile(reps[m], [75, 25])
                results[m].iqr[d - 1, col] = float(q75 - q25)
                results[m].per_repeat[(d, col)] = reps[m]
    return results


def build_matrices(bins: dict, model_name: str, *, cycle_start: datetime, bw_sec: float,
                   metric_names=None, n_train=None, n_eval=None, n_repeats=None,
                   repeat_agg=None, bin_count=None, base_seed=None, pred_sink=None
                   ) -> dict[str, MatrixResult]:
    """指定モデルで Δ以内2値の d×p 行列（precision/recall/f1）を構築する。

    cycle_start / bw_sec: Δ マージンの適用に使う（ビン境界の算出）。
    pred_sink: list を渡すと各評価レコードの予測を
        (d, p, repeat, t(ISO日付), change_id, y_true(0/1), y_pred(正例確率)) として追記する。
    """
    metric_names = metric_names or constants.ENABLED_METRICS
    n_train = constants.N_TRAIN if n_train is None else n_train
    n_eval = constants.N_EVAL if n_eval is None else n_eval
    n_repeats = constants.N_REPEATS if n_repeats is None else n_repeats
    repeat_agg = repeat_agg or constants.REPEAT_AGG
    bin_count = constants.BIN_COUNT if bin_count is None else bin_count
    base_seed = constants.RANDOM_SEED if base_seed is None else base_seed
    delta = timedelta(days=constants.REVIEW_HORIZON_DAYS)

    def _cut_train(i: int):
        recs = bins.get(i, [])
        if constants.TRAIN_TAIL_MARGIN and recs:
            cut = _tail_cutoff(cycle_start, bw_sec, i, delta)
            recs = [r for r in recs if r.t <= cut]
        return recs

    def _cut_eval(p: int):
        recs = bins.get(p, [])
        if constants.EVAL_TAIL_MARGIN and recs:
            cut = _tail_cutoff(cycle_start, bw_sec, p, delta)
            recs = [r for r in recs if r.t <= cut]
        return recs

    # 参照する学習ビン i = p - d の集合（前リリース側の負 index を含む）
    train_bin_ids = {p - d for p in range(bin_count) for d in range(1, bin_count + 1)}

    rows = []
    for k in range(n_repeats):
        seed = base_seed + k
        # 層(1/2/3)廃止により学習済みモデルは学習ビン i だけで決まる。
        # → 学習ビンごとに 1 回だけ学習し、距離 d 方向のセル間で使い回す（セルごと再学習しない）。
        models = {}
        for i in train_bin_ids:
            train_recs = sampler.sample_train(_cut_train(i), n_train, constants.MIN_TRAIN, seed)
            if not train_recs:
                continue
            x_tr, y_tr = _xy(train_recs)
            if len(np.unique(y_tr)) < 2:  # 片クラスのみは学習不可 → このビンは使わない
                continue
            w_tr = _change_balanced_weights(train_recs) if constants.CHANGE_BALANCED_WEIGHT else None
            models[i] = classifier.train(x_tr, y_tr, model_name, seed, sample_weight=w_tr)
        # 位置 p を評価。距離 d のセルは学習ビン p-d の使い回しモデルで予測
        for p in range(bin_count):
            eval_recs = sampler.sample_eval(_cut_eval(p), n_eval, constants.MIN_EVAL, seed)
            if not eval_recs:
                continue
            x_ev = _xy(eval_recs)[0]
            for d in range(1, bin_count + 1):
                model = models.get(p - d)
                if model is None:
                    continue
                y_pred = classifier.predict_proba(model, x_ev)
                for r, ypd in zip(eval_recs, y_pred):
                    rows.append((d, p + 1, k, r.t.date().isoformat(),
                                 r.change_id, float(r.labels[constants.TARGET]), float(ypd)))

    if pred_sink is not None:
        pred_sink.extend(rows)
    return compute_matrices(rows, metric_names=metric_names, n_repeats=n_repeats,
                            repeat_agg=repeat_agg, bin_count=bin_count)
