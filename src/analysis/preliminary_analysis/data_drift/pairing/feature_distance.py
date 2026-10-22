"""RQ2（特徴のずれ）：評価日 X と特徴のずれ g から、学習に使う日を返す部品（design.md §3.2。2026-10-22）。

自分で選び直さず、特徴のずれ×評価時期の精度表（feature_space_window の --mode matrix）が保存したものを読む：
- train_days.csv.gz    ：（評価日, 行 g, seed）ごとの学習の日（精度表と必ず同じ日になる）
- daily_metrics.csv.gz ：（評価日, 行, seed）ごとの指標と評価時期 p（行の列名は distance。汎用ヘッドの行 general も入っている）
- run_meta.json        ：k・刻み・版ごとの段数 N_v
seed は評価日 × 行ごとに、寄与度分析（IG）と同じ決まりで 1 つ選ぶ（pairing/seed_rule.py）。選べない（全部欠測）ときは学習の日を返さない。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from src.analysis.preliminary_analysis.data_drift.pairing import seed_rule
from src.analysis.preliminary_analysis.feature_space_window.utils import constants as fsw
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre

KIND = "feature_distance"
ROW_LABEL = "特徴のずれ"


def matrix_root(project: str) -> Path:
    return fsw.OUTPUT_ROOT / project / KIND


def accuracy_path(project: str, version: str, metric: str) -> Path:
    """同じセルの精度表（RQ2）。図の題の recall@pos・AUC に使う。"""
    return matrix_root(project) / version / "probe" / metric / "drift_matrix.csv"


@dataclass
class FeatureDistancePairing:
    """1 版分。rows ＝ 段数 N_v（特徴のずれ g は 0..rows−1）、positions ＝ {評価日 -> 評価時期 p}、
    seeds ＝ {(評価日 ISO, 行 str) -> 選んだ seed}、train ＝ {(評価日, 行) -> 選んだ seed の学習の日（昇順）}。"""
    window: int
    step: int
    rows: int
    positions: dict
    seeds: dict
    train: dict
    pretrain_days: set | None = None
    kind: str = KIND
    row_label: str = ROW_LABEL
    _avail_ref: list | None = field(default=None, repr=False)
    _avail_set: set = field(default_factory=set, repr=False)

    def train_days(self, x: date, r: int, available: list) -> list[date]:
        """評価日 x・特徴のずれ r の学習の日（選んだ seed のもの。集合がある日だけ）。seed を選べなければ空。"""
        if self._avail_ref is not available:
            self._avail_ref, self._avail_set = available, set(available)
        return [d for d in self.train.get((x, r), []) if d in self._avail_set]

    def needed_days(self) -> set:
        """評価日と、選んだ seed の学習の日（表から残す日）。"""
        out = set(self.positions)
        for days in self.train.values():
            out.update(days)
        return out


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.date


def for_version(project: str, version: str) -> FeatureDistancePairing:
    """その版の部品を作る。精度表の出力がなければ FileNotFoundError。"""
    root = matrix_root(project)
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    if version not in meta.get("versions", {}):
        raise ValueError(f"RQ2 の精度表にこの版がありません: {project} {version}")
    rows = int(meta["versions"][version])
    daily = pd.read_csv(root / version / "daily_metrics.csv.gz", low_memory=False)   # missing_reason が空と文字の混在のため
    daily = daily[daily["distance"].astype(str) != "general"]           # 汎用ヘッドの行は使わない
    seeds = seed_rule.select_seeds(daily, seed_rule.metrics_for(pre.ENCODER_VARIANT))   # 版の種類で並べる指標を変える（2026-10-22）
    pos = daily[["day", "position"]].drop_duplicates("day")
    positions = dict(zip(_to_date(pos["day"]), pos["position"].astype(int)))

    td = pd.read_csv(root / version / "train_days.csv.gz")
    td["day"] = td["day"].astype(str)
    td["row"] = td["row"].astype(int)
    chosen = pd.DataFrame([(d, int(r), s) for (d, r), s in seeds.items()], columns=["day", "row", "seed"])
    td = td.merge(chosen, on=["day", "row", "seed"], how="inner")       # 選んだ seed の学習の日だけ
    td["day"] = _to_date(td["day"])
    td["train_day"] = _to_date(td["train_day"])
    train = {(d, int(r)): sorted(g["train_day"].tolist()) for (d, r), g in td.groupby(["day", "row"])}
    return FeatureDistancePairing(int(meta["size_days"]), int(meta["step_days"]), rows, positions, seeds, train)
