"""（feature_space_window/design.md §9。concept_drift_detection からの複製。2026-10-21）サイクル境界・位置（列）の割当・レコード生成範囲の逆算（design.md §4.3, §4.4）。

学習は「評価日ごとに貼り直す日次スライド」なので、**学習期間は日付範囲で直接切り出す**。
したがってビンは **横軸（位置＝評価日をまとめる表示単位）を決めるためだけ**に使う。

- `target_cycles`     : 対象リリース R と直前リリースのサイクル境界
- `cycle_days`        : 対象サイクルの長さ（日）。格子サイズ N_v = ⌈サイクル長 ÷ 刻み⌉ の元（§4.2）
- `position_of_days`  : リリースの日 ce から刻みずつさかのぼり {日付 -> 位置 p(0..N_v-1)} を返す（§4.3）
- `record_start`      : レコード生成の開始日を「最初の評価日 − 窓長 − 刻み×(N_v−1) − 余裕」で逆算（§4.4）

列の幅は刻みで固定し、列の数 N_v を版ごとに変える（2026-10-15 改訂。旧版はサイクルを N 等分していた）。
旧方式にあった「直前リリースを N 等分してマイナス番号を振る仕組み」も廃止した
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


def _day_of(dt: datetime, day_aligned: bool):
    """リリース日時を日付にする。day_aligned なら最も近い 0 時に丸め、そうでなければ切り捨てる。"""
    return (_midnight(dt) if day_aligned else dt).date()


def cycle_days(cs_R: datetime, ce_R: datetime, day_aligned: bool) -> int:
    """対象サイクルの長さ（日）＝ ce − cs（design.md §4.2）。格子サイズ N_v はここから決める。"""
    return (_day_of(ce_R, day_aligned) - _day_of(cs_R, day_aligned)).days


def position_of_days(ce_R: datetime, step_days: int, grid: int, day_aligned: bool) -> dict:
    """リリースの日 ce から刻みずつさかのぼり、{日付 -> 位置 p（0..grid-1）} を返す（design.md §4.3）。

    評価日は [ce − 刻み×grid, ce − 1] の 刻み×grid 日。位置は p = ⌊(ce − 1 − x) ÷ 刻み⌋ で、
    p = 0 がリリースの直前の刻み日数分。どの列も刻みと同じ日数になる（いちばん遠い列は
    直前のリリースの日より前の日を含むことがある）。
    位置は「評価日をまとめる表示単位」であり、学習には関与しない。
    キーは date（day_sets のキーと合わせる）。
    """
    ce = _day_of(ce_R, day_aligned)
    out: dict = {}
    for back in range(1, step_days * grid + 1):      # back = ce − x（1 〜 刻み×grid 日）
        out[ce - timedelta(days=back)] = (back - 1) // step_days
    return out


def fraction_step(cycle_len_days: int, parts: int) -> int:
    """重みあり版の 3 × 3 の刻み ＝ 周期 ÷ parts を四捨五入した日数（1 以上。concept_drift_detection の binning と同じ。2026-10-22）。"""
    return max(1, int(round(cycle_len_days / parts)))


def position_by_fraction(ce_R: datetime, cycle_len_days: int, parts: int, day_aligned: bool) -> dict:
    """周期を parts 等分して {日付 -> 位置 p（0..parts−1）}（p ＝ 0 がリリース直前。concept_drift_detection の binning と同じ。2026-10-22）。"""
    ce = _day_of(ce_R, day_aligned)
    n = max(1, int(cycle_len_days))
    return {ce - timedelta(days=back): min(parts - 1, (back - 1) * parts // n) for back in range(1, n + 1)}


def record_start(ce_R: datetime, window_days: int, step_days: int, grid: int,
                 margin_days: int = 0, day_aligned: bool = True) -> datetime:
    """レコード生成の開始日を逆算する（design.md §4.4）。

    最古の学習データが要るのは「いちばん遠い列の最初の評価日 × 最大距離 d = grid−1」の組み合わせ：
        最初の評価日   = ce − 刻み×grid（直前のリリースの日以前のこともある。§4.3）
        学習期間の開始 = 最初の評価日 − 窓長 − 刻み×(grid−1)
    （例：窓長・刻み 7 日、grid 26 なら ce から 182 + 182 日）。余裕を足して返す。
    旧方式の「直前リリース日から」では足りない（nova 26.0.0 で 7 日不足）ため、式で逆算する。
    """
    ce = _day_of(ce_R, day_aligned)
    back = step_days * grid + window_days + step_days * (grid - 1) + margin_days
    start = ce - timedelta(days=back)
    return datetime(start.year, start.month, start.day)
