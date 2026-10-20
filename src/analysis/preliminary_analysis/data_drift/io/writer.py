"""出力の保存・読込（design.md §4）。出力先：<OUTPUT_ROOT>/<kind>/<project>/<版>/。回し直すと上書きする。"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.analysis.preliminary_analysis.data_drift.utils import constants


def version_dir(kind: str, project: str, version: str) -> Path:
    return constants.OUTPUT_ROOT / kind / project / version


def save(kind: str, project: str, version: str, daily: pd.DataFrame, cells: pd.DataFrame,
         class_counts: pd.DataFrame) -> Path:
    d = version_dir(kind, project, version)
    d.mkdir(parents=True, exist_ok=True)
    daily.to_csv(d / "daily.csv.gz", index=False, compression="gzip")
    cells.to_csv(d / "cells.csv", index=False)
    class_counts.to_csv(d / "class_counts.csv", index=False)
    return d


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
    """versions は {版 -> {window・step・rows}}。表の指紋・回した機械・日時・git commit も残す。"""
    from src.analysis.preliminary_analysis.feature_table.build.table import table_paths

    meta_path = table_paths(project, "eval")[1]
    fp = None
    if meta_path.exists():
        fp = (json.loads(meta_path.read_text(encoding="utf-8")).get("fingerprint") or {}).get("sha256")
    d = constants.OUTPUT_ROOT / kind / project
    d.mkdir(parents=True, exist_ok=True)
    path = d / "run_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"kind": kind, "project": project, "versions": versions, "groups": list(groups),
                   "alpha": constants.ALPHA, "table_fingerprint_eval": fp,
                   "machine": {"host": platform.node(), "platform": platform.platform()},
                   "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit()},
                  f, ensure_ascii=False, indent=2)
    return path
