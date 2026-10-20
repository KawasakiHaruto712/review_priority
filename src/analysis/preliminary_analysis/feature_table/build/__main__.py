"""特徴量の表を作る（design.md §2）。

実行：
  python3 -m src.analysis.preliminary_analysis.feature_table.build [project ...]                 # 事前学習の表
  python3 -m src.analysis.preliminary_analysis.feature_table.build [project ...] --period eval   # 評価期間の表（§2.1）
  python3 -m src.analysis.preliminary_analysis.feature_table.build [project ...] --add-fingerprint  # 指紋のない表に指紋を書き足す（§2.5）
project を指定しなければ pretrained_encoders の PROJECTS（6 件）を回す。
すでに表があれば、指紋がそろっていれば飛ばし、違えば止めて知らせる（上書きは --overwrite のときだけ）。
1 件で失敗しても止めずに次へ進み、失敗したものを最後にまとめて出す（黙って飛ばさない）。
"""
import argparse
import logging
import sys

from src.analysis.preliminary_analysis.feature_table import constants as C
from src.analysis.preliminary_analysis.feature_table.build.table import (
    TableError, add_fingerprint, build_project, check_table, table_paths)
from src.analysis.preliminary_analysis.pretrained_encoders.utils.constants import PROJECTS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="特徴量の表を作る")
    ap.add_argument("projects", nargs="*", help="対象（既定は PROJECTS の 6 件）")
    ap.add_argument("--period", choices=C.PERIODS, default=C.PERIOD_PRETRAIN, help="表の期間（既定 pretrain）")
    ap.add_argument("--overwrite", action="store_true", help="すでにある表を作り直す（古い表は上書きされる）")
    ap.add_argument("--add-fingerprint", action="store_true", help="指紋のない表に、今のコード・データの指紋を書き足す")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    projects = args.projects or list(PROJECTS)
    failed, done = [], []
    for p in projects:
        try:
            if args.add_fingerprint:
                fp = add_fingerprint(p, args.period)
                done.append(f"{p}（指紋 {fp['sha256'][:12]}… を書き足した）")
                continue
            path, _meta = table_paths(p, args.period)
            if path.exists() and not args.overwrite:
                diffs = check_table(p, args.period)    # すでにあれば作らずに飛ばす（2026-10-21：指紋が違っても止めず、知らせるだけ）
                done.append(f"{p}（すでにあり、指紋もそろっている）" if not diffs else
                            f"{p}（すでにあり。指紋が違う：{', '.join(diffs)}。作り直すなら --overwrite）")
                continue
            build_project(p, args.period, overwrite=args.overwrite)
            done.append(p)
        except TableError as e:
            logging.error(str(e))
            failed.append((p, str(e)))
        except Exception as e:   # 止めずに次へ。最後にまとめて出す
            logging.exception(f"[{p}] 失敗しました")
            failed.append((p, repr(e)))
    for d in done:
        print(f"完了：{d}")
    if failed:
        print("失敗・止めたもの：")
        for p, e in failed:
            print(f"  {p}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
