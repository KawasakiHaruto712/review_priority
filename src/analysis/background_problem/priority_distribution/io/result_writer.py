"""
分析結果（csv / json）の出力（§5.6, §6）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _percentile_keys(points: list) -> list:
    """点群に含まれる分位点キーを昇順で返す（最初の点から取得）。"""
    for p in points:
        if p.percentiles:
            return sorted(p.percentiles.keys())
    return []


def _base_row(point) -> dict:
    """平均±std 部分の共通項目。"""
    return {
        "x": point.x,
        "timestamp": point.timestamp.isoformat(),
        "n_active": point.n_active,
        "mean": point.mean,
        "std": point.std,
        "lower": point.lower,
        "upper": point.upper,
    }


def write_csv(points: list, path: Path, include_release: bool = False) -> None:
    """分布データを CSV で保存する（§6.1）。分位点は p10 / p30 … の列に展開する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pct_keys = _percentile_keys(points)
    pct_columns = [f"p{q}" for q in pct_keys]

    rows = []
    for p in points:
        row = _base_row(p)
        for q in pct_keys:
            row[f"p{q}"] = p.percentiles.get(q)
        if include_release:
            row["release_version"] = p.release_version
        rows.append(row)

    columns = ["x", "timestamp", "n_active", "mean", "std", "lower", "upper"]
    columns += pct_columns
    if include_release:
        columns.append("release_version")

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)
    logger.info(f"CSV を保存: {path}（{len(df)} 行）")


def write_json(points: list, meta: dict, path: Path) -> None:
    """分布データとメタ情報を JSON で保存する（§6.2）。分位点は points[].percentiles に格納。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_points = []
    for p in points:
        obj = _base_row(p)
        # JSON ではキーを文字列にして分位点を入れ子に保持する
        obj["percentiles"] = {str(q): v for q, v in p.percentiles.items()}
        obj["release_version"] = p.release_version
        json_points.append(obj)

    payload = {"meta": meta, "points": json_points}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON を保存: {path}（{len(points)} 点）")
