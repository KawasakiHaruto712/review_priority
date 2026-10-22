"""重みあり版の大きさ k を自動で選ぶ（design.md §10.1・pretrained_encoders/design.md §15.5。2026-10-22）。

要約表 summary/summary_<指標>.csv の Ave（5 版の全評価日の平均）が最良の大きさを選ぶ。同点なら Ave_std（評価日の間の標準偏差）が小さい方。
リポジトリごとに「選んだ値（日数とラベル）と指標名」だけを report/selected_size.json に書く（次点との差は書かない。本人判断）。
特徴のずれ×評価時期の精度表は、手で書いた MATRIX_SETTINGS_POSW の size_days があればそれを、なければこのファイルを読む。
lookback_window/evaluation/select_best.py と同じ決まり（各ディレクトリ自己完結のため複製）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SELECTED_FILENAME = "selected_size.json"


def pick_best(summary: pd.DataFrame, lower_is_better: bool = False) -> str | None:
    """要約表（index ＝ ラベル、列 Ave・Ave_std）から最良のラベル。Ave が全部空なら None。"""
    d = summary.dropna(subset=["Ave"])
    if d.empty:
        return None
    ave = -d["Ave"] if lower_is_better else d["Ave"]
    best = ave.max()
    tied = d[ave == best]
    if len(tied) > 1 and "Ave_std" in tied.columns:
        tied = tied.sort_values("Ave_std", na_position="last", kind="stable")
    return str(tied.index[0])


def select(projects: list[str], metric: str, root: Path, labels: dict, lower_is_better: bool = False) -> Path | None:
    """各リポジトリの summary/summary_<metric>.csv から選び、root/report/selected_size.json に書く。labels は {日数 -> ラベル}。"""
    to_days = {v: k for k, v in labels.items()}
    out = {}
    for p in projects:
        path = Path(root) / p / "summary" / f"summary_{metric}.csv"
        if not path.exists():
            logger.warning(f"[{p}] 要約表がないので選べません：{path}（先に --mode run と --mode plot --metric {metric}）")
            continue
        label = pick_best(pd.read_csv(path, index_col=0), lower_is_better)
        if label is None or label not in to_days:
            logger.warning(f"[{p}] {metric} の平均が出ていないので選べません")
            continue
        out[p] = {"days": int(to_days[label]), "label": label, "metric": metric}
        logger.info(f"[{p}] 選んだ大きさ：{label}（{to_days[label]} 日。{metric} の平均が最良）")
    if not out:
        return None
    dest = Path(root) / "report" / SELECTED_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def load_selected(root: Path) -> dict:
    """{リポジトリ -> 日数}。ファイルがなければ空。"""
    path = Path(root) / "report" / SELECTED_FILENAME
    if not path.exists():
        return {}
    return {p: int(v["days"]) for p, v in json.loads(path.read_text(encoding="utf-8")).items()}
