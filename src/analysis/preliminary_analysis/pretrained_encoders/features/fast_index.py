"""developer/project 系特徴の高速インデックス（事前集計＋二分探索。design.md §2）。

これらの特徴は「T までの件数」「直近◯日の合計」型の集計で、毎回全行を走査すると遅い。
そこで全 Change から **ソート済み配列・累積和を 1 回だけ事前集計**し、各 T を **二分探索 O(log n)** で答える。
定義は `src/features` の developer_metrics / project_metrics と同一（速くするだけ）。
ただし 2026-10 の改訂（design.md §11）で、次の 3 つは T より後の情報を使わない定義に変えたので、
`src/features` 側の旧定義とは一致しない：open_ticket_count（Open の期間）、merge_rate / recent_merge_rate
（マージの時刻＝submitted）、reviewed_lines_in_period（人間のレビュー × T の時点の版の行数）。
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

    必要列: owner_email, created, merged, open_periods, review_times, rev_times, rev_lines
    （feature_builder.build_all_prs_df が作る。design.md §11）
    """

    def __init__(self, all_prs_df: pd.DataFrame):
        df = all_prs_df

        # ── グローバル（open_ticket_count 用） ──
        # Open の期間（design.md §11.1）の開いた時刻・閉じた時刻をそれぞれ昇順に並べる。
        # T に Open な数 = (T までに開いた回数) − (T までに閉じた回数)。復活で開き直した分も数える。
        opens, closes = [], []
        for periods in df["open_periods"]:
            for s_, e_ in periods:
                opens.append(s_)
                if e_ is not None:
                    closes.append(e_)
        self._opens_all = np.sort(np.array([_ns(x) for x in opens], dtype=np.int64))
        self._closes_all = np.sort(np.array([_ns(x) for x in closes], dtype=np.int64))

        # ── reviewed_lines_in_period 用（design.md §11.2） ──
        # 人間のレビューの時刻を全 Change ぶん昇順に並べ、どの Change のものかを持つ。
        pairs = sorted((_ns(rt), i) for i, times in enumerate(df["review_times"]) for rt in times)
        self._review_ns = np.array([p[0] for p in pairs], dtype=np.int64)
        self._review_idx = np.array([p[1] for p in pairs], dtype=np.int64)
        # Change ごとの版の作成時刻（昇順）と行数。T の時点で最新の版の行数を引く
        self._rev_ns = [np.array([_ns(x) for x in times], dtype=np.int64) for times in df["rev_times"]]
        self._rev_lines = [list(lines) for lines in df["rev_lines"]]
        self._reviewed_lines_cache: dict[int, int] = {}  # T は日単位なので同じ T を何度も計算しない

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
        # T に Open な Change の数（design.md §11.1）。期間は [開いた時刻, 閉じた時刻) なので
        # 開いた時刻 <= T と、閉じた時刻 <= T を数えて引く
        t_ns = _ns(t)
        opened_le = int(np.searchsorted(self._opens_all, t_ns, side="right"))
        closed_le = int(np.searchsorted(self._closes_all, t_ns, side="right"))
        return opened_le - closed_le

    def reviewed_lines_in_period(self, t: datetime, lookback_days: int = 14) -> int:
        """[T − 14 日, T] に人間のレビューが 1 件以上ある Change の、T の時点で最新の版の行数の合計（design.md §11.2）。

        旧版は「updated が期間内にある Change」の「最終版の行数」で、どちらも T より後の情報を使っていた。
        """
        t_ns = _ns(t)
        cached = self._reviewed_lines_cache.get(t_ns)
        if cached is not None:
            return cached
        start = _ns(t - timedelta(days=lookback_days))
        lo = int(np.searchsorted(self._review_ns, start, side="left"))
        hi = int(np.searchsorted(self._review_ns, t_ns, side="right"))
        total = 0
        for i in np.unique(self._review_idx[lo:hi]):
            k = int(np.searchsorted(self._rev_ns[i], t_ns, side="right"))  # T までに出た版の数
            if k > 0:
                total += int(self._rev_lines[i][k - 1])
        self._reviewed_lines_cache[t_ns] = total
        return total
