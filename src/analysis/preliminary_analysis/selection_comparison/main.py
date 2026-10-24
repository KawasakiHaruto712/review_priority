"""学習に用いるチケットの選び方の比較（selection_comparison/design.md）。ほかの分析の出力ファイルを読むだけ。

実行:
    python3 -m src.analysis.preliminary_analysis.selection_comparison.main --mode all --variant posw
    --mode accuracy : A 精度の比較（時間の d0 と空間の d0 を同じ評価日で）→ accuracy_by_phase.*・accuracy_by_release.csv
    --mode select   : B 精度が大きく変わった列を選ぶ（accuracy の出力を読む）→ selected_columns.json・.md
    --mode drift    : C 選んだ列の分布の比較（select の出力を読む）→ drift_within.*・drift_cross.*
    --mode all      : 上の 3 つを順に
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.analysis.preliminary_analysis.pretrained_encoders.utils import variant
from src.analysis.preliminary_analysis.selection_comparison.compare import accuracy, drift, select
from src.analysis.preliminary_analysis.selection_comparison.utils import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def main(mode: str) -> None:
    out = constants.OUTPUT_ROOT
    if mode in ("accuracy", "all"):
        variant.warn_if_exists(out / "accuracy_by_release.csv", "精度の比較")
        phase, release = accuracy.run(out)
        logger.info(f"精度の比較：評価時期ごと {len(phase)} 行・リリースごと {len(release)} 行 → {out}")
    if mode in ("select", "all"):
        release = pd.read_csv(out / "accuracy_by_release.csv", encoding=constants.ENCODING)
        variant.warn_if_exists(out / "selected_columns.json", "選んだ列")
        sel = select.run(release, out)
        n = sum(len(s) for k in sel["within"].values() for s in k.values()) + sum(len(s) for s in sel["cross"].values())
        logger.info(f"選んだ列：{n} 列 → {out / 'selected_columns.json'}")
    if mode in ("drift", "all"):
        sel = select.load(out)
        variant.warn_if_exists(out / "drift_cross.csv", "分布の比較")
        within, cross = drift.run(sel, out)
        logger.info(f"分布の比較：同じ選び方の中 {len(within)} 行・2 つの選び方の間 {len(cross)} 行 → {out}")


def _parse_args():
    ap = argparse.ArgumentParser(description="学習に用いるチケットの選び方の比較（時間的な距離と特徴空間上の距離）")
    ap.add_argument("--mode", choices=["accuracy", "select", "drift", "all"], default="all")
    variant.add_argument(ap)          # 省略すると重みあり版（design.md §6）
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    v = variant.apply(variant.resolve(args.variant, writing=False), constants)
    constants.set_variant_roots(v)
    main(args.mode)
