"""本流の表（main_branches.csv）を作る（src/collectors/README.md「本流の表」、pretrained_encoders/design.md §12.2）。

分析の特徴量「本流か否か」に使う。手で書いた一覧を分析のコードに持たず、この表から引く（再現性のため）。
- 今の本流：Gerrit の API `/projects/<名前>/HEAD`（既定のブランチ）で取る（kind=current）
- 昔の本流：API では取れないので、`constants.GERRIT_PROJECTS` の `former_main_branches` に出どころつきで
  書いたものを入れる（kind=former。「昔の本流」）
分析は current も former も本流として扱う。

対象の既定は `constants.RELEASE_TAG_TARGETS`（区切りの表と同じ）。1 本ずつ、リポジトリの間を 30 秒空ける。
対象のリポジトリの行はすべて入れ替え、ほかの行は残す。

実行:
    python -m src.collectors.main_branch_collector                     # 既定の対象すべて
    python -m src.collectors.main_branch_collector --project qtbase    # 1 つだけ
ふだんは区切りの表と一緒に `python -m src.collectors.build_release_table` で作る。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.parse
from datetime import datetime, timezone

import pandas as pd

from src.collectors.release_tags_collector import PROJECT_INTERVAL, _parse, _session, merge_into_csv
from src.config.path import DEFAULT_DATA_DIR
from src.utils.constants import GERRIT_PROJECTS, RELEASE_TAG_TARGETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

OUTPUT_CSV = DEFAULT_DATA_DIR / "openstack_collected" / "main_branches.csv"
COLUMNS = ["project", "branch", "kind", "until", "source", "note", "fetched_at"]
HEADS_PREFIX = "refs/heads/"


def head_url(project_key: str) -> str:
    """そのリポジトリの既定のブランチ（HEAD）を返す API の URL。"""
    spec = GERRIT_PROJECTS[project_key]
    name = urllib.parse.quote(spec["path"], safe="")
    return f"{spec['host'].rstrip('/')}/projects/{name}/HEAD"


def fetch_head(project_key: str) -> str:
    """今の本流（既定のブランチ）の名前を返す（例 "dev"）。1 回だけ問い合わせる。"""
    spec = GERRIT_PROJECTS[project_key]
    r = _session(spec).get(head_url(project_key), timeout=(30, 120))
    r.raise_for_status()
    ref = str(_parse(r.text))
    return ref[len(HEADS_PREFIX):] if ref.startswith(HEADS_PREFIX) else ref


def build_rows(project_key: str, head: str, fetched_at: str) -> pd.DataFrame:
    """今の本流（current）と、登録表の昔の本流（former）の行を作る。"""
    rows = [{"project": project_key, "branch": head, "kind": "current", "until": "",
             "source": head_url(project_key), "note": "Gerrit の既定のブランチ（HEAD）", "fetched_at": fetched_at}]
    for f in GERRIT_PROJECTS[project_key].get("former_main_branches", []):
        if f["branch"] == head:
            logger.warning(f"[{project_key}] 昔の本流 {f['branch']} が今の本流と同じです。登録を確かめてください")
            continue
        rows.append({"project": project_key, "branch": f["branch"], "kind": "former",
                     "until": f.get("until", ""), "source": f.get("source", ""),
                     "note": "昔の本流。" + f.get("note", ""), "fetched_at": ""})
    return pd.DataFrame(rows, columns=COLUMNS)


def collect(projects: list[str]) -> pd.DataFrame:
    frames = []
    for i, p in enumerate(projects):
        if p not in GERRIT_PROJECTS:
            logger.warning(f"未登録のプロジェクト: {p}")
            continue
        if i > 0:
            time.sleep(PROJECT_INTERVAL)   # 1 本ずつ、リポジトリの間を空ける
        head = fetch_head(p)
        fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        df = build_rows(p, head, fetched_at)
        logger.info(f"[{p}] 今の本流 {head}"
                    + (f"／昔の本流 {', '.join(df[df.kind == 'former'].branch)}" if (df.kind == "former").any() else ""))
        frames.append(df)
    if not frames:
        logger.warning("対象がありません")
        return pd.DataFrame(columns=COLUMNS)
    new = pd.concat(frames, ignore_index=True)
    merge_into_csv(new, OUTPUT_CSV)
    return new


def _parse_args():
    ap = argparse.ArgumentParser(description="Gerrit の HEAD と登録表から本流の表を作る")
    ap.add_argument("--project", nargs="*", default=None,
                    help=f"対象（既定 {RELEASE_TAG_TARGETS}）")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collect(args.project or list(RELEASE_TAG_TARGETS))
