"""リリースごとに適した日数 N（release_best_size/design.md）。ほかの分析の出力ファイルを読むだけ。

実行:
    python3 -m src.analysis.preliminary_analysis.release_best_size.main --variant posw
    読む → 日ごとの値（5 seed の中央値）→ マス（リポジトリ × 版 × 評価時期）× N の平均 → 最良の N → 出力
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.analysis.preliminary_analysis.pretrained_encoders.utils import variant
from src.analysis.preliminary_analysis.release_best_size.compute import best, summary
from src.analysis.preliminary_analysis.release_best_size.io import readers, writers
from src.analysis.preliminary_analysis.release_best_size.utils import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def positions(project: str, version: str) -> pd.Series:
    """評価時期の対応（精度表と同じ）。時間と特徴空間の精度表で一致しなければ止める（design.md §2）。"""
    t = readers.day_positions("time_distance", project, version)
    f = readers.day_positions("feature_distance", project, version)
    if not t.equals(f):
        raise ValueError(f"{project} {version}：時間と特徴空間の精度表で評価時期の対応が一致しません")
    return t


def build() -> pd.DataFrame:
    """全選び方 × 全リポジトリ × 全版の行（best_size.csv）。"""
    rows = []
    for method in constants.METHODS:
        selected = readers.selected_sizes(method)
        for project in constants.PROJECTS:
            for version in readers.versions(project):
                metrics = readers.size_metrics(method, project, version)
                missing_sizes = set(constants.SIZES) - set(metrics["size"])
                if missing_sizes:
                    raise ValueError(f"{method} {project} {version}：N の候補がそろっていません（{sorted(missing_sizes)}）")
                pos = positions(project, version)
                daily = best.daily_values(metrics)
                lost = sorted(set(pos.index) - set(metrics["day"]))
                if lost:
                    raise ValueError(f"{method} {project} {version}：精度表の評価日が metrics.csv にありません（{lost[:3]}）")
                dropped = sorted(set(metrics["day"]) - set(pos.index))
                logger.info(f"[{method}] {project} {version}：精度表にない日を {len(dropped)} 日除く（{dropped}）")
                rows += best.release_rows(method, project, version, best.attach_position(daily, pos), selected[project])
    return pd.DataFrame(rows)


def main() -> None:
    out = constants.OUTPUT_ROOT
    variant.warn_if_exists(out / "best_size.csv", "リリースごとに適した日数 N")
    table = build()
    s = summary.run(table)
    writers.write_all(table, s, out)
    for method, v in s.items():
        logger.info(f"[{method}] 5 リリースで最良の N がすべて同じ：{v['n_phase_all_same']}/{v['n_phase']}・"
                    f"4.1 の N が最良：{v['n_cell_selected_best']}/{v['n_cell']}")
    logger.info(f"{len(table)} 行 → {out}")


def _parse_args():
    ap = argparse.ArgumentParser(description="リリースごとに適した日数 N（時間・特徴空間による選び方）")
    variant.add_argument(ap)          # 省略すると重みあり版（design.md §7）
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    v = variant.apply(variant.resolve(args.variant, writing=False), constants)
    constants.set_variant_roots(v)
    main()
