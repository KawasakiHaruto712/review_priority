"""IG の日次結果の保存・読み込み（design.md §9）。

集約する前の**最も細かい単位**（評価日 × 距離 × ヘッド × 群）で保存する。
こうしておくと、セル集約・std・寄与率をすべて抽出時に計算でき、
**後から集約の仕方を変えたくなっても計算し直さずに済む**。

- **追記式**：同じ `(day, distance, seed, head, group)` の行は新しい実行で置き換える。
- **メタ情報**は `run_id` ごとの履歴として貯める。知りたいのは「いつ計算したか」より
  「**どの条件で計算したか**」なので、条件を run_id で辿れるようにする。
- **寄与率と差分は保存しない**（生の値から復元できる。§9.1）。
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analysis.preliminary_analysis.concept_drift_cause.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.model.set_transformer import FEATURE_NAMES

logger = logging.getLogger(__name__)

KEY = ["day", "distance", "seed", "head", "group"]
COLUMNS = ["run_id", "day", "position", "distance", "seed", "head", "group",
           "n_changes"] + list(FEATURE_NAMES)
# 実行条件のうち、混ぜると結果が比較できなくなるもの（食い違ったら警告する）
_CRITICAL = ["ig_steps", "baseline", "target_output", "select_metric"]


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def run_conditions(extra: dict = None) -> dict:
    """この実行の条件（メタ情報に残すもの）。"""
    return {"ig_steps": constants.IG_STEPS,
            "baseline": "その評価日の集合の平均（対象 Change のみ移動）",
            "target_output": "logit",
            "select_metric": constants.SELECT_METRIC,
            "git": _git_commit(),
            **(extra or {})}


def daily_path(out_dir: Path) -> Path:
    return Path(out_dir) / constants.IG_DAILY_NAME


def load_daily(out_dir: Path) -> pd.DataFrame:
    """保存済み ig_daily を読む。無ければ空の DataFrame。"""
    path = daily_path(out_dir)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, compression="gzip")


def write_daily(rows: list[dict], out_dir: Path, run_id: str, conditions: dict) -> Path | None:
    """行を ig_daily.csv.gz に**追記**する（同じキーの行は置き換え）。メタ履歴も更新する。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return None
    new = pd.DataFrame(rows)
    old = load_daily(out_dir)
    if not old.empty:
        _warn_if_conditions_differ(out_dir, conditions)
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=KEY, keep="last")  # 後勝ち＝新しい実行
    else:
        merged = new
    merged = merged.sort_values(["day", "distance", "head", "group"], kind="stable")
    path = daily_path(out_dir)
    merged.to_csv(path, index=False, compression="gzip")
    _append_meta(out_dir, run_id, conditions, n_new=len(new), n_total=len(merged))
    logger.info(f"IG 日次結果を保存: {path}（新規 {len(new)} 行 / 累計 {len(merged)} 行）")
    return path


def _meta_path(out_dir: Path) -> Path:
    return Path(out_dir) / constants.IG_META_NAME


def _read_meta(out_dir: Path) -> dict:
    path = _meta_path(out_dir)
    if not path.exists():
        return {"runs": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_meta(out_dir: Path, run_id: str, conditions: dict, n_new: int, n_total: int) -> None:
    meta = _read_meta(out_dir)
    meta["runs"] = [r for r in meta.get("runs", []) if r.get("run_id") != run_id]
    meta["runs"].append({"run_id": run_id, "実行日時": datetime.now().isoformat(timespec="seconds"),
                         "新規行数": n_new, "累計行数": n_total, **conditions})
    with open(_meta_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _warn_if_conditions_differ(out_dir: Path, conditions: dict) -> None:
    """既存の行と条件が違う実行を検出して警告する（黙って混ざるのを防ぐ。§9.2）。"""
    for run in _read_meta(out_dir).get("runs", []):
        diff = {k: (run.get(k), conditions.get(k)) for k in _CRITICAL
                if k in run and run.get(k) != conditions.get(k)}
        if diff:
            logger.warning(f"既存の実行 {run.get('run_id')} と条件が違います: {diff}。"
                           f"条件の違う行が同じファイルに混ざります（run_id で区別できます）。")
