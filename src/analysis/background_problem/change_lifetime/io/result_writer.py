"""
ヒストグラム結果（csv / json）の出力（§5.2, §6）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_histogram_csv(counts, edges, path: Path) -> None:
    """ヒストグラムを CSV で保存する（1 行 = 1 ビン）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {"bin_left": float(edges[i]), "bin_right": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]
    df = pd.DataFrame(rows, columns=["bin_left", "bin_right", "count"])
    df.to_csv(path, index=False)
    logger.info(f"CSV を保存: {path}（{len(df)} ビン）")


def write_summary_json(meta: dict, summary: dict, counts, edges, path: Path) -> None:
    """メタ情報・要約統計・ヒストグラムを JSON で保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    histogram = {
        "bin_left": [float(e) for e in edges[:-1]] if len(edges) else [],
        "bin_right": [float(e) for e in edges[1:]] if len(edges) else [],
        "count": [int(c) for c in counts],
    }
    payload = {"meta": meta, "summary": summary, "histogram": histogram}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON を保存: {path}")
