"""リリース情報を Gerrit の tags API から収集する。

`release_collector.py` は OpenStack 専用で、`opendev.org/openstack/releases`（リリース定義を
YAML で管理する専用リポジトリ）を clone して版と日付を得ている。他のプロジェクトには
そうしたリポジトリが無いため、**Gerrit 自身が持つタグ**から版と日付を組み立てる。

    GET /projects/<name>/tags/?n=1000&S=<offset>
    → [{"ref": "refs/tags/v6.8.0", "created": "2024-10-07 ...", ...}, ...]

どのタグを版とみなすかは `constants.GERRIT_PROJECTS` の `tag_pattern` / `tag_series` で決める。
RC やビルド番号まで刻むプロジェクト（LibreOffice の `libreoffice-25.2.0.3` など）があるため、
**同じ版シリーズに属するタグのうち最も古い日付**をそのリリース日として採用する。

出力は既存の `major_releases_summary.csv` と同じ形式（project, version, release_date, yaml_url）で、
**既存行は保持したまま併合**する（OpenStack の行を壊さない）。

実行:
    python -m src.collectors.release_tags_collector                     # 未収集の対象すべて
    python -m src.collectors.release_tags_collector --project qtbase    # 1 つだけ
    python -m src.collectors.release_tags_collector --dry-run           # 書き込まずに確認
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config.path import DEFAULT_DATA_DIR
from src.utils.constants import GERRIT_PROJECTS, NEW_COLLECT_TARGETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

OUTPUT_CSV = DEFAULT_DATA_DIR / "openstack_collected" / "major_releases_summary.csv"
COLUMNS = ["project", "version", "release_date", "yaml_url"]
PAGE = 1000


def _session(spec: dict) -> requests.Session:
    """接続用セッション（auth=True のホストだけ Basic 認証を付ける）。

    資格情報は Gerrit インスタンスごとに違うので、登録表の `env` が指す接頭辞から読む
    （既定 GERRIT_* = OpenStack、Qt は QT_GERRIT_*）。
    """
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    if spec.get("auth", True):
        load_dotenv()
        prefix = spec.get("env", "GERRIT")
        user, pwd = os.getenv(f"{prefix}_USERNAME"), os.getenv(f"{prefix}_PASSWORD")
        if not (user and pwd):
            raise ValueError(f"認証が必要なホストですが "
                             f"{prefix}_USERNAME / {prefix}_PASSWORD が未設定です")
        token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
        s.headers["Authorization"] = f"Basic {token}"
    return s


def _parse(body: str):
    """Gerrit の XSSI 対策プレフィックス `)]}'` を取り除いて JSON にする。"""
    if body.startswith(")]}'"):
        body = body.split("\n", 1)[1] if "\n" in body else body[4:]
    return json.loads(body)


def fetch_tags(project_key: str) -> list[dict]:
    """そのプロジェクトの全タグを取得する（1000 件ずつページング）。"""
    spec = GERRIT_PROJECTS[project_key]
    sess = _session(spec)
    name = urllib.parse.quote(spec["path"], safe="")
    base = spec["host"].rstrip("/")

    tags, start = [], 0
    while True:
        url = f"{base}/projects/{name}/tags/?n={PAGE}&S={start}"
        r = sess.get(url, timeout=(30, 120))
        r.raise_for_status()
        batch = _parse(r.text)
        if not batch:
            break
        tags.extend(batch)
        start += len(batch)
        if len(batch) < PAGE:
            break
    logger.info(f"[{project_key}] タグ {len(tags)} 件を取得")
    return tags


def extract_releases(project_key: str, tags: list[dict]) -> pd.DataFrame:
    """タグ一覧から (version, release_date) を組み立てる。

    同じ版シリーズに複数のタグ（RC・ビルド番号）がある場合は**最も古い日付**を採用する。
    """
    spec = GERRIT_PROJECTS[project_key]
    pattern, series = spec.get("tag_pattern"), spec.get("tag_series")
    if not pattern or not series:
        raise ValueError(f"[{project_key}] constants.GERRIT_PROJECTS に tag_pattern / tag_series がありません")

    rx = re.compile(pattern)
    first: dict[str, pd.Timestamp] = {}
    for t in tags:
        name = str(t.get("ref", "")).replace("refs/tags/", "")
        m = rx.match(name)
        if not m:
            continue
        # 注釈付きタグは tagger.date、軽量タグは created に日付が入る
        raw = (t.get("tagger") or {}).get("date") or t.get("created")
        date = pd.to_datetime(str(raw).replace(".000000000", ""), errors="coerce")
        if pd.isna(date):
            continue
        version = m.expand(series)
        if version not in first or date < first[version]:
            first[version] = date

    rows = [{"project": project_key, "version": v, "release_date": d.date().isoformat(),
             "yaml_url": f"{spec['host'].rstrip('/')}/projects/"
                         f"{urllib.parse.quote(spec['path'], safe='')}/tags/"}
            for v, d in sorted(first.items(), key=lambda kv: kv[1])]
    logger.info(f"[{project_key}] 版 {len(rows)} 件を抽出"
                + (f"（{rows[0]['version']} {rows[0]['release_date']} 〜 "
                   f"{rows[-1]['version']} {rows[-1]['release_date']}）" if rows else ""))
    return pd.DataFrame(rows, columns=COLUMNS)


def merge_into_csv(new: pd.DataFrame, csv_path: Path = OUTPUT_CSV) -> Path:
    """既存 CSV に併合して書き戻す（同じ (project, version) は新しい方で上書き）。"""
    csv_path = Path(csv_path)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["project", "version"], keep="last")
    else:
        merged = new
    merged = merged.sort_values(["project", "release_date"]).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)
    logger.info(f"保存: {csv_path}（全 {len(merged)} 行）")
    return csv_path


def collect(projects: list[str], dry_run: bool = False) -> pd.DataFrame:
    frames = []
    for p in projects:
        if p not in GERRIT_PROJECTS:
            logger.warning(f"未登録のプロジェクト: {p}")
            continue
        frames.append(extract_releases(p, fetch_tags(p)))
    if not frames:
        logger.warning("対象がありません")
        return pd.DataFrame(columns=COLUMNS)
    new = pd.concat(frames, ignore_index=True)
    if dry_run:
        logger.info("dry-run のため書き込みません")
        print(new.to_string(index=False))
    else:
        merge_into_csv(new)
    return new


def _parse_args():
    ap = argparse.ArgumentParser(description="Gerrit の tags API からリリース情報を収集する")
    ap.add_argument("--project", nargs="*", default=None,
                    help=f"対象（既定 {NEW_COLLECT_TARGETS}）")
    ap.add_argument("--dry-run", action="store_true", help="CSV に書き込まず内容だけ表示する")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collect(args.project or list(NEW_COLLECT_TARGETS), dry_run=args.dry_run)
