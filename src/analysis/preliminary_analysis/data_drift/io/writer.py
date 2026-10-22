"""出力の保存・読込（design.md §4）。出力先：<OUTPUT_ROOT>/<kind>/<project>/<版>/。回し直すと上書きする。"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.analysis.preliminary_analysis.data_drift.stats.aggregate import summarize_cells
from src.analysis.preliminary_analysis.data_drift.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.utils import variant


def version_dir(kind: str, project: str, version: str) -> Path:
    return constants.OUTPUT_ROOT / kind / project / version


def save(kind: str, project: str, version: str, daily: pd.DataFrame, cells: pd.DataFrame,
         class_counts: pd.DataFrame) -> Path:
    d = version_dir(kind, project, version)
    variant.warn_if_exists(d, f"{kind} {project} {version}")          # 上書きの警告（design.md §10.1。2026-10-22）
    d.mkdir(parents=True, exist_ok=True)
    daily.to_csv(d / "daily.csv.gz", index=False, compression="gzip")
    cells.to_csv(d / "cells.csv", index=False)
    summarize_cells(cells).to_csv(d / "cells_summary.csv", index=False)      # 全体のずれの大きさ（§3.4。2026-10-22）
    class_counts.to_csv(d / "class_counts.csv", index=False)
    return d


def compare_dir(project: str) -> Path:
    """RQ1 と RQ2 を比べる折れ線の出力先（§4。2026-10-22）。"""
    return constants.OUTPUT_ROOT / "compare" / project


def load(kind: str, project: str, version: str, name: str) -> pd.DataFrame:
    """name は cells・class_counts・daily のどれか。"""
    d = version_dir(kind, project, version)
    return pd.read_csv(d / ("daily.csv.gz" if name == "daily" else f"{name}.csv"))


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def save_run_meta(kind: str, project: str, versions: dict, groups) -> Path:
    """versions は {版 -> {window・step・rows}}。表の指紋・回した機械・日時・git commit も残す。
    RQ2（feature_distance）は事前学習の表もつなぐので、その指紋も残す（2026-10-22）。"""
    from src.analysis.preliminary_analysis.feature_table.build.table import table_paths

    def _fp(period: str):
        meta_path = table_paths(project, period)[1]
        if not meta_path.exists():
            return None
        return (json.loads(meta_path.read_text(encoding="utf-8")).get("fingerprint") or {}).get("sha256")

    fp = _fp("eval")
    extra = {"table_fingerprint_pretrain": _fp("pretrain")} if kind == "feature_distance" else {}
    d = constants.OUTPUT_ROOT / kind / project
    d.mkdir(parents=True, exist_ok=True)
    path = d / "run_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"kind": kind, "project": project, "versions": versions, "groups": list(groups),
                   "alpha": constants.ALPHA, "table_fingerprint_eval": fp, **extra,
                   "machine": {"host": platform.node(), "platform": platform.platform()},
                   "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit()},
                  f, ensure_ascii=False, indent=2)
    return path
