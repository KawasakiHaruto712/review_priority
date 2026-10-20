"""出力の保存・読込（design.md §6）。出力先：<OUTPUT_ROOT>/<project>/<版>/ と <OUTPUT_ROOT>/<project>/summary/。"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.analysis.preliminary_analysis.feature_space_window.utils import constants


def version_dir(project: str, version: str) -> Path:
    return constants.OUTPUT_ROOT / project / version


def append_predictions(preds: list[pd.DataFrame], project: str, version: str, first: bool) -> None:
    """Change ごとの予測を predictions.csv.gz に seed ごとに書き足す（first なら作り直す）。全 seed 分をメモリに持たないため。"""
    d = version_dir(project, version)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "predictions.csv.gz"
    if first and path.exists():
        path.unlink()            # 前の実行の同じ出力（このディレクトリの生成物）を作り直す
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(path, index=False, compression="gzip",
                                                   mode="a", header=not path.exists())


def save_version(rows, project: str, version: str) -> Path:
    """1 版分（VersionRows）の metrics.csv・selected_days.csv.gz を書く（予測は append_predictions で書き足し済み）。"""
    d = version_dir(project, version)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows.metrics).to_csv(d / "metrics.csv", index=False)
    pd.DataFrame(rows.selected, columns=["day", "seed", "rank", "selected_day", "days_before", "distance",
                                         "in_pretrain"]).to_csv(d / "selected_days.csv.gz", index=False,
                                                                 compression="gzip")
    return d


def load_metrics(project: str, version: str) -> pd.DataFrame:
    return pd.read_csv(version_dir(project, version) / "metrics.csv")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def save_run_meta(project: str, seeds: list[int], sizes: list[int], versions: list[str]) -> Path:
    """実行条件（大きさ・距離の決まり・候補の範囲・seed・表の指紋・実行日時・git commit）を run_meta.json に書く。"""
    from src.analysis.preliminary_analysis.feature_table.build.table import table_paths

    prints = {}
    for period in ("pretrain", "eval"):
        meta_path = table_paths(project, period)[1]
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            prints[period] = (meta.get("fingerprint") or {}).get("sha256")
    d = constants.OUTPUT_ROOT / project
    d.mkdir(parents=True, exist_ok=True)
    path = d / "run_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"project": project, "versions": versions, "seeds": seeds, "sizes_days": sizes,
                   "distance": "各次元を標準化したユークリッド距離（日の代表＝埋め込みの平均。標準偏差は評価日ごとに X より前の全日から。"
                               f"{constants.STD_EPS} 未満は 1）",
                   "candidates": "評価日より前のすべての日（事前学習の表＋評価期間の表。重なりは評価期間の表）",
                   "table_fingerprints": prints, "created_at": datetime.now(timezone.utc).isoformat(),
                   "git_commit": _git_commit()}, f, ensure_ascii=False, indent=2)
    return path
