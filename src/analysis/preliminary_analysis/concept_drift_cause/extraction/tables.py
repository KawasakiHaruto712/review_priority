"""保存済みの IG 日次結果から表を組み立てる（design.md §8, §9.3）。

計算（重い）と抽出（軽い）を分けているので、ここはモデルを一切動かさない。
`ig_daily.csv.gz` と距離×時期行列の分析の `daily_metrics.csv.gz` を読むだけ。

集約（§8）:
    ① (評価日, 距離) ごとに 1 モデルのプロファイル ← ig_daily にそのまま入っている
    ② セル内の日で平均      → セルの値
       セル内の日で標準偏差  → std（**日間のばらつき**）
       有効日数              → 列に出す
精度（§7）は距離×時期行列の分析の daily_metrics から、**距離×時期行列の分析と同じ集約順**（日ごとに seed 中央値 → 日で平均）で
付ける。特徴量の寄与度分析が選ぶモデルは各日の中央モデルそのものなので、距離×時期行列の分析のセル値と一致する。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.concept_drift_cause.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.model.set_transformer import FEATURE_NAMES

logger = logging.getLogger(__name__)

ACC_COLS = ["auc", "recall@pos"] + [f"recall@{k}" for k in constants.K_LIST]   # recall@pos は 2026-10-21 追加（design.md §4）


def accuracy_by_cell(daily: pd.DataFrame, days_by_cell: dict) -> dict:
    """距離×時期行列の分析の daily_metrics から、セルごとの精度を返す（§7）。

    days_by_cell : {(distance, position) -> その セルで IG を計算した評価日の集合}
    **IG を計算した日に限定**して集計する（中断時に精度とプロファイルが別の日を見るのを防ぐ）。
    """
    d = daily[daily["missing"] == 0].copy()
    d["distance"] = d["distance"].astype(str)
    # ① 日ごとに seed 中央値（距離×時期行列の分析と同じ順序）
    med = d.groupby(["distance", "day"])[ACC_COLS].median()
    out = {}
    for (dist, pos), days in days_by_cell.items():
        key = [(str(dist), day) for day in days]
        sub = med.reindex(key).dropna(how="all")
        out[(dist, pos)] = {c: (float(sub[c].mean()) if len(sub) else np.nan) for c in ACC_COLS}
    return out


def _cell_frame(rows: pd.DataFrame) -> dict:
    """1 セルぶんの行（評価日ごと）→ 特徴量ごとの平均・std・寄与率 ＋ 日数。"""
    feats = rows[list(FEATURE_NAMES)]
    mean = feats.mean(axis=0)
    std = feats.std(axis=0, ddof=1) if len(feats) > 1 else pd.Series(np.nan, index=FEATURE_NAMES)
    total = float(mean.sum())
    # 寄与率は生の値から復元できるので保存はしないが、ここで出す（§9.1）
    share = mean / total if total != 0 else pd.Series(np.nan, index=FEATURE_NAMES)
    absmean = feats.abs().mean(axis=0)     # 評価日を通した寄与の絶対値の平均（図の上位の選び方。§9.4.3）
    top = list(absmean.sort_values(ascending=False, kind="stable").index[:constants.PLOT_TOP_N or len(FEATURE_NAMES)])
    out = {"n_days": int(len(rows)), "n_changes": float(rows["n_changes"].mean()),
           "sum_ig": total, "top_features": ";".join(top)}     # 図に並べる上位（順位の順）
    for f in FEATURE_NAMES:
        out[f] = float(mean[f])
        out[f"{f}_std"] = float(std[f])
        out[f"{f}_share"] = float(share[f])
        out[f"{f}_absmean"] = float(absmean[f])
    return out


def build_table(ig_daily: pd.DataFrame, daily_metrics: pd.DataFrame,
                head: str, group: str) -> pd.DataFrame:
    """1 つの (ヘッド, 群) についてセル×特徴量の表を作る。

    汎用ヘッドの行は `(評価日, seed)` で持っているので（§9.1）、probe 行の seed を見て
    同じ `(評価日, seed)` の行を引き当ててから距離ごとに並べ直す。
    """
    probe = ig_daily[(ig_daily["head"] == "probe") & (ig_daily["group"] == group)]
    if probe.empty:
        return pd.DataFrame()
    if head == "probe":
        rows = probe
    else:
        gen = ig_daily[(ig_daily["head"] == "general") & (ig_daily["group"] == group)]
        if gen.empty:
            return pd.DataFrame()
        # probe 側の (day, distance, seed, position) に、同じ (day, seed) の汎用ヘッドを付ける
        keys = probe[["day", "distance", "position", "seed"]]
        rows = keys.merge(gen.drop(columns=["distance", "position"]), on=["day", "seed"], how="inner")

    days_by_cell = {k: set(v["day"]) for k, v in rows.groupby(["distance", "position"])}
    acc = accuracy_by_cell(daily_metrics, days_by_cell)

    out = []
    for (dist, pos), sub in rows.groupby(["distance", "position"]):
        rec = {"distance": dist, "position": int(pos), **acc.get((dist, pos), {}),
               **_cell_frame(sub)}
        out.append(rec)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["_order"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df.sort_values(["_order", "position"], kind="stable").drop(columns="_order")
    cols = ["distance", "position", "n_days", "n_changes"] + ACC_COLS + ["sum_ig", "top_features"]
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest]


def build_compare(ig_daily: pd.DataFrame, position: int, ref: int, other: int, group: str,
                  prefix: str = "d") -> pd.DataFrame:
    """d0 と相手を並べた表（design.md §11.7。2026-10-23）。probe の生の値、1 つの列（評価時期 p）・1 つの群。

    行は全特徴量、|d0 − 相手| の大きい順。平均と標準偏差は評価日を通したもの（セルの表と同じ）。
    **平均をとる日は行ごとに値のある日をすべて使う**（両方の行にそろえない。精度表・特徴量別の分布と同じ扱い）。
    そのため日数を行ごとに載せる。どちらかの行に日がなければ空。
    """
    p = ig_daily[(ig_daily["head"] == "probe") & (ig_daily["group"] == group)
                 & (ig_daily["position"].astype(int) == int(position))]
    a = p[p["distance"].astype(str) == str(ref)]
    b = p[p["distance"].astype(str) == str(other)]
    if a.empty or b.empty:
        return pd.DataFrame()
    ra, rb = f"{prefix}{ref}", f"{prefix}{other}"

    def _std(rows):
        return rows[list(FEATURE_NAMES)].std(axis=0, ddof=1) if len(rows) > 1 else pd.Series(np.nan, index=FEATURE_NAMES)

    df = pd.DataFrame({"feature": list(FEATURE_NAMES),
                       f"{ra}_mean": a[list(FEATURE_NAMES)].mean(axis=0).values, f"{ra}_std": _std(a).values,
                       f"{rb}_mean": b[list(FEATURE_NAMES)].mean(axis=0).values, f"{rb}_std": _std(b).values})
    df[f"diff({ra}-{rb})"] = df[f"{ra}_mean"] - df[f"{rb}_mean"]
    df[f"{ra}_n_days"] = int(a["day"].nunique())
    df[f"{rb}_n_days"] = int(b["day"].nunique())
    order = df[f"diff({ra}-{rb})"].abs().sort_values(ascending=False, kind="stable").index
    return df.loc[order].reset_index(drop=True)


def pairs_from_spec(spec: dict) -> list[tuple[int, int]]:
    """対象の指定から、比べる（列, 相手の行）の組（相手は d0 以外の行）。d0 がない列は作らない。
    行 × 列の指定（"distances"・"positions"）と、組の一覧（"cells"。2026-10-24。design.md §11.9）の両方に対応する。
    行・列が "all" の指定（重みなし版の全セル）では作らない。"""
    if spec.get("cells") is not None:
        cells = {(int(d), int(p)) for d, p in spec["cells"]}
        return sorted((p, d) for d, p in cells if d != 0 and (0, p) in cells)
    dists, poss = spec.get("distances"), spec.get("positions")
    if dists in (None, "all") or poss in (None, "all") or 0 not in [int(d) for d in dists]:
        return []
    return [(int(p), int(d)) for p in poss for d in dists if int(d) != 0]


def write_compare_tables(ig_daily: pd.DataFrame, spec: dict, out_dir: Path, prefix: str = "d") -> list[Path]:
    """対象の指定（距離と位置を明示したもの）から、列ごとに d0 と d0 以外の行を並べた表を書く（design.md §11.7）。
    距離が "all" の指定（重みなし版の全セル）では作らない。2026-10-24：組の一覧（"cells"）にも対応（pairs_from_spec）。"""
    out_dir = Path(out_dir)
    paths = []
    for pos, other in pairs_from_spec(spec):
        for group in ("pos", "neg"):
            df = build_compare(ig_daily, pos, 0, other, group, prefix)
            if df.empty:
                logger.warning(f"該当データなしのため並べた表をスキップ: p{pos} {prefix}0 と {prefix}{other} {group}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"p{pos}_{prefix}0_vs_{prefix}{other}_{group}.csv"
            df.to_csv(path, index=False, encoding=constants.CSV_ENCODING)
            paths.append(path)
    return paths


# ── 判断値の表（design.md §11.9。2026-10-24。本人と合意した物差し） ──────────────────
def judgment_values(ig_daily: pd.DataFrame, distance, position: int) -> pd.DataFrame:
    """1 つのモデル（probe の行 distance・列 position）の、評価日 × 特徴量の判断値（正例での寄与の平均 − 負例での寄与の平均）。
    正例・負例の両方がある日だけ。"""
    p = ig_daily[(ig_daily["head"] == "probe") & (ig_daily["distance"].astype(str) == str(distance))
                 & (ig_daily["position"].astype(int) == int(position))]
    pos = p[p["group"] == "pos"].set_index(p[p["group"] == "pos"]["day"].astype(str))[list(FEATURE_NAMES)]
    neg = p[p["group"] == "neg"].set_index(p[p["group"] == "neg"]["day"].astype(str))[list(FEATURE_NAMES)]
    pos, neg = pos.align(neg, join="inner", axis=0)
    return pos - neg


def build_judgment(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    """2 つのモデルの判断値を並べた表。差 ＝ b − a。判断値の差の絶対値の大きい順（全特徴量）。
    平均・標準偏差はそれぞれのモデルに値のある日すべて、検定（ウィルコクソンの符号順位検定・両側）は両方に値のある日。"""
    from scipy.stats import wilcoxon
    if a.empty or b.empty:
        return pd.DataFrame()
    pa, pb = a.align(b, join="inner", axis=0)
    rows = []
    for f in FEATURE_NAMES:
        diff = (pb[f] - pa[f]).dropna()
        pv = float(wilcoxon(diff).pvalue) if len(diff) and (diff != 0).any() else np.nan
        rows.append({"feature": f,
                     f"{label_a}_mean": float(a[f].mean()), f"{label_a}_std": float(a[f].std(ddof=1)) if len(a) > 1 else np.nan,
                     f"{label_b}_mean": float(b[f].mean()), f"{label_b}_std": float(b[f].std(ddof=1)) if len(b) > 1 else np.nan,
                     f"diff({label_b}-{label_a})": float(b[f].mean() - a[f].mean()),
                     f"{label_a}_n_days": int(len(a)), f"{label_b}_n_days": int(len(b)), "n_paired": int(len(pa)),
                     "p_value": pv})
    df = pd.DataFrame(rows)
    col = f"diff({label_b}-{label_a})"
    return df.loc[df[col].abs().sort_values(ascending=False, kind="stable").index].reset_index(drop=True)


def write_judgment_tables(ig_daily: pd.DataFrame, spec: dict, out_dir: Path) -> list[Path]:
    """同じ選び方の中：列ごとに d0 と相手の判断値の表（例 p0_d0_vs_d1.csv。番号は保存物の向き。行は論文の記号 d）。"""
    out_dir = Path(out_dir)
    paths = []
    for pos, other in pairs_from_spec(spec):
        df = build_judgment(judgment_values(ig_daily, 0, pos), judgment_values(ig_daily, other, pos), "d0", f"d{other}")
        if df.empty:
            logger.warning(f"該当データなしのため判断値の表をスキップ: p{pos} d0 と d{other}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"p{pos}_d0_vs_d{other}.csv"
        df.to_csv(path, index=False, encoding=constants.CSV_ENCODING)
        paths.append(path)
    return paths


def write_cross_judgment(time_daily: pd.DataFrame, space_daily: pd.DataFrame, position: int, out_dir: Path) -> Path | None:
    """2 つの選び方の間：時間的な距離の d0 と特徴空間上の距離の d0 の判断値の表（p{列}_time_vs_space.csv）。差 ＝ 空間 − 時間。"""
    df = build_judgment(judgment_values(time_daily, 0, position), judgment_values(space_daily, 0, position), "time_d0", "space_d0")
    if df.empty:
        logger.warning(f"該当データなしのため 2 つの選び方の間の判断値の表をスキップ: p{position}")
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"p{int(position)}_time_vs_space.csv"
    df.to_csv(path, index=False, encoding=constants.CSV_ENCODING)
    return path


def write_tables(ig_daily: pd.DataFrame, daily_metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    """probe_pos / probe_neg / general_pos / general_neg の 4 表を書く（§9.3）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for head in ("probe", "general"):
        for group in ("pos", "neg"):
            df = build_table(ig_daily, daily_metrics, head, group)
            if df.empty:
                logger.warning(f"該当データなしのため表をスキップ: {head}_{group}")
                continue
            path = out_dir / f"{head}_{group}.csv"
            # Excel が UTF-8 を誤認して文字化けしないよう BOM 付きで保存
            df.to_csv(path, index=False, encoding=constants.CSV_ENCODING)
            paths.append(path)
    return paths
