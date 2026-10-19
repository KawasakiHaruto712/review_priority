"""相関で絞る（design.md §3）。表（build/）を先に作っておくこと。

実行： python3 -m src.analysis.preliminary_analysis.feature_table.selection [project ...]
project を指定しなければ pretrained_encoders の PROJECTS（6 件）。どれかの表がなければ止める（6 件そろえて比べるため）。
"""
import argparse
import logging
import sys

from src.analysis.preliminary_analysis.feature_table import constants as C
from src.analysis.preliminary_analysis.feature_table.selection.run import run
from src.analysis.preliminary_analysis.pretrained_encoders.utils.constants import PROJECTS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="特徴量の表を読んで相関で絞る")
    ap.add_argument("projects", nargs="*", help="対象（既定は PROJECTS の 6 件）")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    projects = args.projects or list(PROJECTS)
    missing = [p for p in projects if not (C.TABLE_ROOT / p / f"{C.PERIOD_PRETRAIN}.npz").exists()]
    if missing:
        print(f"表がありません：{', '.join(missing)}（先に python3 -m src.analysis.preliminary_analysis.feature_table.build）")
        return 1
    kept = run(projects)
    for rule, names in kept.items():
        print(f"{rule}: {len(names)} 個")
    print(f"結果：{C.SELECTION_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
