"""学習×予測行列（菱形→正方形）の構築（§6, §5.8）。

セル (i, j)（i<j, 距離 d=j-i>0）ごとに n_repeats 回繰り返す:
  - 各反復で層分割→学習→予測し、評価ビン内の各計測点 T で順位化して MAE/RMSE/NDCG@n を算出、
    T 方向に中央値でまとめて「その反復の代表値」とする（2段中央値の1段目。§5.7）。
  - 反復方向に中央値でまとめてセル値（2段目）。ばらつき(IQR)と各反復の代表値も保持。
行列は (距離 d=1..BIN_COUNT, 位置 p=0..BIN_COUNT-1) の四角。学習側ビン i=p-d は前リリースに及んでよい。
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.analysis.preliminary_analysis.concept_drift_existence.dataset import splitter
from src.analysis.preliminary_analysis.concept_drift_existence.evaluation import pooled_metrics, ranking_metrics
from src.analysis.preliminary_analysis.concept_drift_existence.model import regressor
from src.analysis.preliminary_analysis.concept_drift_existence.utils import constants

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
    y = np.array([r.label for r in records], dtype=float)
    return x, y


def _change_balanced_weights(records) -> np.ndarray:
    """各行に 1/(その Change の学習内レコード数) を与える（結果として Change ごとの合計重みが 1 になる）。

    同一 Change が複数の計測点 T で重複して現れることによる「長寿 Change への偏り」と
    「重複の水増し」を是正する（合計重み ≒ Change 数 = 実効標本数）。
    """
    counts = defaultdict(int)
    for r in records:
        counts[r.change_id] += 1
    return np.array([1.0 / counts[r.change_id] for r in records], dtype=float)


def _aggregate_per_t(groups: dict, metric_names, ndcg_n, t_agg) -> dict:
    """順位指標を計測点 T ごとに算出し、T 方向に集約（§2.8.1, 2.8.4）。

    groups: t -> [(y_true, y_pred), ...]。1反復ぶんの評価データ。
    """
    per_t = {m: [] for m in metric_names}
    for pairs in groups.values():
        if len(pairs) < 2:
            continue  # 順位が自明な T は除外（§2.8）
        yt = np.array([a for a, _ in pairs])
        yp = np.array([b for _, b in pairs])
        vals = ranking_metrics.compute(yt, yp, metric_names, ndcg_n)
        for m in metric_names:
            per_t[m].append(vals[m])
    agg = _AGG[t_agg]
    return {m: (float(agg(per_t[m])) if per_t[m] else np.nan) for m in metric_names}


def _aggregate_pooled(groups: dict, metric_names, buckets) -> dict:
    """回帰誤差・分類を、全レコードをプールして一発算出（§2.8.2/2.8.3, 2.8.4）。

    groups: t -> [(y_true, y_pred), ...]。T で区切らず全部まとめる（案1＝レコード重み）。
    """
    yt = np.array([a for pairs in groups.values() for a, _ in pairs], dtype=float)
    yp = np.array([b for pairs in groups.values() for _, b in pairs], dtype=float)
    if yt.size == 0:
        return {m: np.nan for m in metric_names}
    return pooled_metrics.compute(yt, yp, metric_names, buckets)


def _repeat_value(groups: dict, metric_names, ndcg_n, t_agg, buckets) -> dict:
    """1反復ぶんの代表値。順位は per-T 集約、回帰・分類はプール（§2.8.4）。"""
    rank_ms = [m for m in metric_names if m not in pooled_metrics.POOLED_METRICS]
    pool_ms = [m for m in metric_names if m in pooled_metrics.POOLED_METRICS]
    out = {}
    if rank_ms:
        out.update(_aggregate_per_t(groups, rank_ms, ndcg_n, t_agg))
    if pool_ms:
        out.update(_aggregate_pooled(groups, pool_ms, buckets))
    return out


def compute_matrices(rows, *, metric_names=None, n_repeats=None, t_agg=None,
                     repeat_agg=None, ndcg_n=None, bin_count=None, buckets=None) -> dict[str, MatrixResult]:
    """予測行から全指標の四角行列を作る（実行時・再計算の両方で使う単一経路）。

    rows: (d, p, repeat, t, change_id, y_true, y_pred) のタプル列。
        p・d は 1 始まり、t は計測点（同一 T で同じ値ならキーは何でもよい）、値は hours 単位。
    指標は系統で集約が違う（§2.8.4）: 順位は per-T 集約、回帰誤差・分類は評価ビン内プール。
    この関数だけが「予測 → 指標 → 行列（2段集約・IQR・per_repeat）」を担うので、
    実行時のメモリ上予測でも、保存済み predictions.csv.gz の読み直しでも同じ結果になる。
    """
    metric_names = metric_names or constants.ENABLED_METRICS
    n_repeats = constants.N_REPEATS if n_repeats is None else n_repeats
    t_agg = t_agg or constants.T_AGG
    repeat_agg = repeat_agg or constants.REPEAT_AGG
    ndcg_n = constants.NDCG_AT if ndcg_n is None else ndcg_n
    bin_count = constants.BIN_COUNT if bin_count is None else bin_count
    buckets = constants.CLASS_BUCKETS_HOURS if buckets is None else buckets

    shape = (bin_count, bin_count)
    results = {m: MatrixResult(m, bin_count, np.full(shape, np.nan), np.full(shape, np.nan))
               for m in metric_names}
    agg = _AGG[repeat_agg]

    # (d, p) -> {repeat -> {t -> [(y_true, y_pred), ...]}}
    cells: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d, p, k, t, _cid, y_true, y_pred in rows:
        cells[(int(d), int(p))][int(k)][t].append((float(y_true), float(y_pred)))

    for (d, p), reps_map in cells.items():
        reps = {m: [] for m in metric_names}
        for _k, groups in reps_map.items():
            vals = _repeat_value(groups, metric_names, ndcg_n, t_agg, buckets)
            for m in metric_names:
                if not np.isnan(vals[m]):
                    reps[m].append(vals[m])
        col = p - 1  # 行列の列は 0 始まり（rows の p は 1 始まり）
        for m in metric_names:
            if reps[m]:
                results[m].value[d - 1, col] = float(agg(reps[m]))
                q75, q25 = np.percentile(reps[m], [75, 25])
                results[m].iqr[d - 1, col] = float(q75 - q25)
                results[m].per_repeat[(d, col)] = reps[m]  # キーの p は 0 始まり（result_writer 互換）
    return results


def build_matrices(bins: dict, model_name: str, *, metric_names=None, n_train=None, n_eval=None,
                   n_repeats=None, t_agg=None, repeat_agg=None, ndcg_n=None,
                   bin_count=None, base_seed=None, pred_sink=None) -> dict[str, MatrixResult]:
    """指定モデルで全指標の四角行列をまとめて構築する。

    pred_sink: list を渡すと、各評価レコードの予測を
        (d, p, repeat, t(ISO日付), change_id, y_true(hours), y_pred(hours)) として追記する。
        後から再学習なしで別指標を計算し直すための生データ（None なら保存しない）。
    """
    metric_names = metric_names or constants.ENABLED_METRICS
    n_train = constants.N_TRAIN if n_train is None else n_train
    n_eval = constants.N_EVAL if n_eval is None else n_eval
    n_repeats = constants.N_REPEATS if n_repeats is None else n_repeats
    t_agg = t_agg or constants.T_AGG
    repeat_agg = repeat_agg or constants.REPEAT_AGG
    ndcg_n = constants.NDCG_AT if ndcg_n is None else ndcg_n
    bin_count = constants.BIN_COUNT if bin_count is None else bin_count
    base_seed = constants.RANDOM_SEED if base_seed is None else base_seed

    # 1) 学習・予測して生の予測行を作る。2) その行から compute_matrices で指標化する（単一経路）。
    rows = []
    for p in range(bin_count):            # 位置（当該リリース内ビン）
        for d in range(1, bin_count + 1):  # 距離（滞留期間）
            i = p - d                      # 学習側ビン（前リリースに及びうる）
            for k in range(n_repeats):
                out = splitter.split_train_eval(bins, i, p, n_train, n_eval, base_seed + k)
                if out is None:
                    continue
                train_recs, eval_recs = out
                if not train_recs or not eval_recs:
                    continue
                x_tr, y_tr = _xy(train_recs)
                x_ev, _ = _xy(eval_recs)
                # heavy-tail 対策: 学習 target を log1p 変換（評価は順位ベースで log は単調＝順位不変）。
                if constants.LABEL_LOG_TRANSFORM:
                    y_tr = np.log1p(y_tr)
                # 重複観測の偏り是正: 各行に 1/(その Change の重複数) を与える（合計重みが Change ごとに 1）。§2.6
                w_tr = _change_balanced_weights(train_recs) if constants.CHANGE_BALANCED_WEIGHT else None
                model = regressor.train(x_tr, y_tr, model_name, base_seed + k, sample_weight=w_tr)
                y_pred = regressor.predict(model, x_ev)
                # 予測は hours 単位に戻して保持（log は順位不変なので順位指標も同じ結果）。
                yp_hours = np.expm1(y_pred) if constants.LABEL_LOG_TRANSFORM else y_pred
                for r, ypd in zip(eval_recs, yp_hours):
                    rows.append((d, p + 1, k, r.t.date().isoformat(),
                                 r.change_id, float(r.label), float(ypd)))

    if pred_sink is not None:
        pred_sink.extend(rows)
    return compute_matrices(rows, metric_names=metric_names, n_repeats=n_repeats,
                            t_agg=t_agg, repeat_agg=repeat_agg, ndcg_n=ndcg_n, bin_count=bin_count)
