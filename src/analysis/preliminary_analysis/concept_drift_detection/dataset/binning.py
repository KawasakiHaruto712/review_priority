"""サイクル境界・位置（列）の割当・レコード生成範囲の逆算（design.md §4.3, §4.4）。

学習は「評価日ごとに貼り直す日次スライド」なので、**学習期間は日付範囲で直接切り出す**。
したがってビンは **横軸（位置＝評価日をまとめる表示単位）を決めるためだけ**に使う。

- `target_cycles`     : 対象リリース R と直前リリースのサイクル境界
- `position_of_days`  : 対象サイクルを N 等分し {日付 -> 位置 p(0..N-1)} を返す（§4.3）
- `record_start`      : レコード生成の開始日を「窓長 ＋ 刻み×(N−1) ＋ 余裕」で逆算（§4.4）

旧方式にあった「直前リリースを N 等分してマイナス番号を振る仕組み」は廃止した
（学習期間を日付で切り出すため、前サイクルに届いても番号付けが不要）。
境界は 0 時に揃える（`BIN_DAY_ALIGNED`）。リリース日程は `_drop_release_anomalies` で異常エントリを除外。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.analysis.background_problem.common.data_loader import _drop_release_anomalies


def _midnight(dt: datetime) -> datetime:
    """最も近い 0 時に丸める（12 時以降は翌日 0 時）。"""
    base = datetime(dt.year, dt.month, dt.day)
    return base + timedelta(days=1) if dt.hour >= 12 else base


def _cleaned_release_dates(rel_df: pd.DataFrame, project: str) -> list[pd.Timestamp]:
    """プロジェクトの（異常除外済み）リリース日を昇順で返す。"""
    pdf = _drop_release_anomalies(rel_df[rel_df["project"] == project])
    return sorted(pd.to_datetime(pdf["release_date"]).tolist())


def target_cycles(rel_df: pd.DataFrame, project: str, version: str):
    """対象リリース R と直前リリースの開発サイクル境界を返す。

    返り値: (cs_R, ce_R, cs_prev)
      - cs_R  = 直前リリース日（R のサイクル開始）
      - ce_R  = R のリリース日（R のサイクル終了）
      - cs_prev = さらに 1 つ前のリリース日（直前リリースのサイクル開始）／無ければ None
    """
    dates = _cleaned_release_dates(rel_df, project)
    target = rel_df[(rel_df["project"] == project) & (rel_df["version"] == version)]
    if target.empty:
        raise ValueError(f"リリースが見つかりません: {project} {version}")
    ce_R = pd.to_datetime(target["release_date"].iloc[0])
    earlier = [d for d in dates if d < ce_R]
    if not earlier:
        raise ValueError(f"直前リリースが見つかりません: {project} {version}")
    cs_R = earlier[-1]
    before_cs_R = [d for d in earlier if d < cs_R]
    cs_prev = before_cs_R[-1] if before_cs_R else None
    return cs_R.to_pydatetime(), ce_R.to_pydatetime(), (cs_prev.to_pydatetime() if cs_prev is not None else None)


def bin_edges(cs: datetime, ce: datetime, bin_count: int, day_aligned: bool) -> list[datetime]:
    """[cs, ce] を bin_count 等分する境界（bin_count+1 個）。day_aligned なら各境界を 0 時に丸める。

    丸めで境界が潰れないよう、単調増加（前より最低 1 日後）を保証する。
    """
    span = (ce - cs).total_seconds()
    edges = [cs + timedelta(seconds=span * j / bin_count) for j in range(bin_count + 1)]
    if not day_aligned:
        return edges
    snapped: list[datetime] = []
    for e in edges:
        m = _midnight(e)
        if snapped and m <= snapped[-1]:
            m = snapped[-1] + timedelta(days=1)  # 単調増加を保証
        snapped.append(m)
    return snapped


def _which_bin(t: datetime, edges: list[datetime]) -> int | None:
    """t が edges のどの区間 [edges[j], edges[j+1]) に入るか（0..len-2）。範囲外は None。"""
    if t < edges[0] or t >= edges[-1]:
        return None
    for j in range(len(edges) - 1):
        if edges[j] <= t < edges[j + 1]:
            return j
    return None


def position_of_days(cs_R: datetime, ce_R: datetime, grid: int, day_aligned: bool) -> dict:
    """対象サイクルを grid 等分し、{日付 -> 位置 p（0..grid-1）} を返す（design.md §4.3）。

    位置は「評価日をまとめる表示単位」であり、学習には関与しない。
    キーは date（day_sets のキーと合わせる）。
    """
    edges = bin_edges(cs_R, ce_R, grid, day_aligned)
    out: dict = {}
    t = edges[0]
    one = timedelta(days=1)
    while t < edges[-1]:
        j = _which_bin(t, edges)
        if j is not None:
            out[t.date()] = j
        t += one
    return out


def record_start(cs_R: datetime, window_days: int, step_days: int, grid: int,
                 margin_days: int = 0) -> datetime:
    """レコード生成の開始日を逆算する（design.md §4.4）。

    最古の学習データが要るのは「サイクル初日 × 最大距離 d = grid−1」の組み合わせ：
        学習期間の開始 = 最初の評価日 − 刻み×(grid−1) − 窓長
    ＝ 遡り量は「窓長 ＋ 刻み×(grid−1)」（既定 7 + 7×25 = 182 日 ≒ 1 サイクル）。余裕を足して返す。
    旧方式の「直前リリース日から」では足りない（nova 26.0.0 で 7 日不足）ため、式で逆算する。
    """
    back = window_days + step_days * (grid - 1) + margin_days
    return cs_R - timedelta(days=back)
