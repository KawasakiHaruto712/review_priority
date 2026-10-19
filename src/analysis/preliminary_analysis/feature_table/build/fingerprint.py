"""表の指紋（design.md §2.5）と、表の期間（§2.1）。

指紋は値に効くものだけから作る SHA-256：特徴量・レコード・ラベルを計算するソースファイルの中身、値に効く定数と表の期間、入力のデータの要約。
コードのコミットは記録だけにし、指紋には入れない（設計書を直しただけのコミットで「違う」にならないように）。
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.analysis.background_problem.common.data_loader import DEFAULT_CHANGES_DIR_TEMPLATE, DEFAULT_RELEASES_SUMMARY_CSV
from src.analysis.preliminary_analysis.feature_table import constants as C
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre
from src.utils.bot_detection import BOT_ACCOUNTS_CSV

REPO_ROOT = Path(__file__).resolve().parents[5]
_PE = "src/analysis/preliminary_analysis/pretrained_encoders"
# 指紋の材料のソース（§2.5）。ディレクトリは中の *.py をすべて（__pycache__ は除く）
SOURCE_PATHS = [f"{_PE}/features", f"{_PE}/dataset/record_builder.py", f"{_PE}/labeling",
                f"{_PE}/utils/review_utils.py", "src/features", "src/utils/bot_detection.py"]
# 表の値に効かないので除く（モデルに入れる特徴量の一覧。pretrained_encoders/design.md §12.5）
EXCLUDED_SOURCES = {f"{_PE}/features/feature_set.py"}
EVAL_MARGIN_DAYS = 7          # 評価期間の表の始まりの余裕（§2.1）


def _sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_files(root: Path = REPO_ROOT) -> list[str]:
    """指紋の材料のソースファイル（リポジトリからの相対パス、並べ替え済み）。"""
    out = []
    for p in SOURCE_PATHS:
        full = Path(root) / p
        files = [full] if full.is_file() else sorted(full.rglob("*.py"))
        for f in files:
            rel = f.relative_to(root).as_posix()
            if "__pycache__" not in rel and rel not in EXCLUDED_SOURCES:
                out.append(rel)
    return sorted(out)


def changes_summary(project: str) -> dict:
    """Change の JSON のディレクトリの要約（ファイル数・合計の大きさ・最新の更新時刻）。"""
    d = Path(str(DEFAULT_CHANGES_DIR_TEMPLATE).format(project=project))
    files = sorted(d.glob("*.json")) if d.exists() else []
    stats = [p.stat() for p in files]
    return {"path": str(d), "files": len(files), "total_bytes": int(sum(s.st_size for s in stats)),
            "latest_mtime": (datetime.fromtimestamp(max(s.st_mtime for s in stats), timezone.utc).isoformat()
                             if stats else None)}


# ── 表の期間（§2.1） ─────────────────────────────────────────
def pretrain_range(rel_df: pd.DataFrame, project: str) -> dict:
    """事前学習の表の期間。始まりはデータの始まり（データで決まるので入力の要約に含まれる）、終わりは cutoff の版の公開日。"""
    from src.analysis.preliminary_analysis.pretrained_encoders.build_encoders import _resolve_cutoff

    cutoff_dt, cutoff_label = _resolve_cutoff(rel_df, project)
    return {"period": C.PERIOD_PRETRAIN, "start": "data_start", "end": cutoff_dt.isoformat(), "cutoff": cutoff_label}


def eval_range(rel_df: pd.DataFrame, project: str) -> dict:
    """評価期間の表の期間（§2.1）：下流が使う最も早い日 − 余裕 〜 最後の評価の版の公開日。

    最も早い日は、5 版それぞれについて、窓長の調査（最大の窓長）の学習データの開始と、距離×時期行列・寄与度分析の
    レコード生成の開始（窓長の調査の候補の窓長すべてと、今の設定の窓長）を、下流と同じ関数で出した最小。
    """
    from src.analysis.preliminary_analysis.concept_drift_detection.dataset import binning
    from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants as det
    from src.analysis.preliminary_analysis.lookback_window.sweep import window_sweep
    from src.analysis.preliminary_analysis.lookback_window.utils import constants as lw

    versions = pre.project_config(project)["versions"]
    windows = sorted(set(lw.WINDOWS_DAYS) | {det.window_for(project)})
    starts, ends = [], []
    for v in versions:
        cs_lw, ce_lw = window_sweep.release_cycle(rel_df, project, v)
        starts.append(window_sweep.span_start_for(cs_lw))
        cs_R, ce_R, _prev = binning.target_cycles(rel_df, project, v)
        cycle = binning.cycle_days(cs_R, ce_R, det.BIN_DAY_ALIGNED)
        for w in windows:
            step = w if det.DISTANCE_STEP_DAYS is None else int(det.DISTANCE_STEP_DAYS)
            grid = max(1, math.ceil(cycle / step))
            starts.append(binning.record_start(ce_R, w, step, grid, det.RECORD_MARGIN_DAYS, det.BIN_DAY_ALIGNED))
        ends += [ce_lw, ce_R]
    start = min(starts) - timedelta(days=EVAL_MARGIN_DAYS)
    start = datetime(start.year, start.month, start.day)
    return {"period": C.PERIOD_EVAL, "start": start.isoformat(), "end": max(ends).isoformat(),
            "versions": list(versions), "windows": windows}


def period_range(rel_df: pd.DataFrame, project: str, period: str) -> dict:
    if period == C.PERIOD_PRETRAIN:
        return pretrain_range(rel_df, project)
    if period == C.PERIOD_EVAL:
        return eval_range(rel_df, project)
    raise ValueError(f"知らない期間: {period}")


# ── 指紋 ─────────────────────────────────────────────────
def fingerprint_parts(project: str, rng: dict, root: Path = REPO_ROOT) -> dict:
    """指紋の材料（どこが違うかを示せるように、材料ごとに持つ）。"""
    return {
        "sources": {rel: _sha256_file(Path(root) / rel) for rel in source_files(root)},
        "constants": {"MEASUREMENT_STEP_DAYS": pre.MEASUREMENT_STEP_DAYS, "LOOKBACK_DAYS": pre.LOOKBACK_DAYS,
                      "REVIEW_HORIZON_DAYS": pre.REVIEW_HORIZON_DAYS, "TARGET": pre.TARGET, "range": rng},
        "inputs": {"changes": changes_summary(project),
                   "major_releases_summary_sha256": _sha256_file(DEFAULT_RELEASES_SUMMARY_CSV),
                   "bot_accounts_sha256": _sha256_file(BOT_ACCOUNTS_CSV),
                   "main_branches_sha256": _sha256_file(_main_branches_csv())},
    }


def _main_branches_csv() -> Path:
    from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.ticket import MAIN_BRANCHES_CSV
    return MAIN_BRANCHES_CSV


def digest(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def compute(project: str, period: str, rel_df: pd.DataFrame, root: Path = REPO_ROOT) -> dict:
    """{"sha256": 指紋, "parts": 材料}。"""
    parts = fingerprint_parts(project, period_range(rel_df, project, period), root)
    return {"sha256": digest(parts), "parts": parts}


def differences(old_parts: dict, new_parts: dict) -> list[str]:
    """材料のどこが違うか（人が読む説明の列）。"""
    out = []
    os_, ns = old_parts.get("sources", {}), new_parts.get("sources", {})
    for rel in sorted(set(os_) | set(ns)):
        if os_.get(rel) != ns.get(rel):
            out.append(f"ソース {rel}")
    if old_parts.get("constants") != new_parts.get("constants"):
        out.append("定数・表の期間")
    oi, ni = old_parts.get("inputs", {}), new_parts.get("inputs", {})
    for k in sorted(set(oi) | set(ni)):
        if oi.get(k) != ni.get(k):
            out.append(f"入力 {k}")
    return out
