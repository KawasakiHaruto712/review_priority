"""developer/project 系特徴の高速インデックス（事前集計＋二分探索。§3.4）。

これらの特徴は「T までの件数」「直近◯日の合計」型の集計で、毎回 3.7万行を走査すると遅い
（実測 6.5ms/件 → 日次240万件で約4.3時間）。そこで全 Change から **ソート済み配列・累積和を 1 回だけ
事前集計**し、各 T を **二分探索 O(log n)** で答える。

定義は `src/features` の developer_metrics / project_metrics と同一（速くするだけ）。
同値は test_fast_index.py で `src/features` と突き合わせて保証する。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _ns(t: datetime) -> int:
    """datetime/Timestamp を int64 ナノ秒へ。"""
    return pd.Timestamp(t).value


def _to_ns_array(series: pd.Series) -> np.ndarray:
    """欠損(NaT)を除いた int64 ナノ秒の昇順配列。"""
    s = pd.to_datetime(series).dropna()
    arr = s.astype("int64").to_numpy()
    arr.sort()
    return arr


class FastFeatureIndex:
    """all_prs_df から developer/project 特徴を高速に引くための事前集計。

    必要列: owner_email, created, merged, decision_time, updated, lines_added, lines_deleted
    """

    def __init__(self, all_prs_df: pd.DataFrame):
        df = all_prs_df

        # ── グローバル（open_ticket_count 用） ──
        # open = created<=T かつ「決着(マージ/放棄)していない」。決着時刻 decision_time で判定（放棄も閉じた扱い）。
        self._created_all = _to_ns_array(df["created"])
        self._closed_all = _to_ns_array(df["decision_time"])

        # ── reviewed_lines_in_period 用: updated 昇順＋行数の累積和 ──
        u = df[["updated", "lines_added", "lines_deleted"]].dropna(subset=["updated"]).copy()
        u = u.sort_values("updated")
        self._updated_all = u["updated"].astype("int64").to_numpy()
        lines = (u["lines_added"].fillna(0).astype("int64")
                 + u["lines_deleted"].fillna(0).astype("int64")).to_numpy()
        self._lines_prefix = np.concatenate([[0], np.cumsum(lines)])  # prefix[i] = 先頭 i 件の合計

        # ── 開発者ごと: created 昇順、(created, merged) 昇順 ──
        self._created_by_dev: dict[str, np.ndarray] = {}
        self._merged_by_dev: dict[str, np.ndarray] = {}     # merge_rate 用（merged のみ昇順）
        self._cm_by_dev: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # recent_merge_rate 用
        for email, g in df.groupby("owner_email"):
            c = pd.to_datetime(g["created"]).dropna()
            c_sorted = np.sort(c.astype("int64").to_numpy())
            self._created_by_dev[email] = c_sorted
            self._merged_by_dev[email] = _to_ns_array(g["merged"])
            # created 昇順に並べた (created, merged) ペア（merged NaT は +inf 扱い）
            gg = g[["created", "merged"]].dropna(subset=["created"]).sort_values("created")
            cm_c = pd.to_datetime(gg["created"]).astype("int64").to_numpy()
            merged_ns = pd.to_datetime(gg["merged"]).astype("int64").to_numpy().astype("float64")
            merged_ns[pd.to_datetime(gg["merged"]).isna().to_numpy()] = np.inf
            self._cm_by_dev[email] = (cm_c, merged_ns)

    # ── developer 特徴 ──
    def past_report_count(self, email: str, t: datetime) -> int:
        arr = self._created_by_dev.get(email)
        if arr is None:
            return 0
        return int(np.searchsorted(arr, _ns(t), side="right"))

    def recent_report_count(self, email: str, t: datetime, lookback_months: int = 3) -> int:
        arr = self._created_by_dev.get(email)
        if arr is None:
            return 0
        start = _ns(t - timedelta(days=30 * lookback_months))
        return int(np.searchsorted(arr, _ns(t), side="right")
                   - np.searchsorted(arr, start, side="left"))

    def merge_rate(self, email: str, t: datetime) -> float:
        reported = self.past_report_count(email, t)
        if reported == 0:
            return 0.0
        m = self._merged_by_dev.get(email)
        merged = 0 if m is None else int(np.searchsorted(m, _ns(t), side="right"))
        return merged / reported

    def recent_merge_rate(self, email: str, t: datetime, lookback_months: int = 3) -> float:
        cm = self._cm_by_dev.get(email)
        if cm is None:
            return 0.0
        created_arr, merged_arr = cm
        t_ns = _ns(t)
        start = _ns(t - timedelta(days=30 * lookback_months))
        lo = int(np.searchsorted(created_arr, start, side="left"))
        hi = int(np.searchsorted(created_arr, t_ns, side="right"))
        if hi - lo == 0:
            return 0.0
        merged_recent = int(np.count_nonzero(merged_arr[lo:hi] <= t_ns))
        return merged_recent / (hi - lo)

    # ── project 特徴 ──
    def open_ticket_count(self, t: datetime) -> int:
        # open = (created<=T の数) - (T までに決着した数)。決着＝マージまたは放棄（decision_time）。
        t_ns = _ns(t)
        created_le = int(np.searchsorted(self._created_all, t_ns, side="right"))
        closed_le = int(np.searchsorted(self._closed_all, t_ns, side="right"))
        return created_le - closed_le

    def reviewed_lines_in_period(self, t: datetime, lookback_days: int = 14) -> int:
        t_ns = _ns(t)
        start = _ns(t - timedelta(days=lookback_days))
        hi = int(np.searchsorted(self._updated_all, t_ns, side="right"))
        lo = int(np.searchsorted(self._updated_all, start, side="left"))
        return int(self._lines_prefix[hi] - self._lines_prefix[lo])
