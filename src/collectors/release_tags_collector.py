"""リリース情報を Gerrit の tags API から収集する。

`release_collector.py` は OpenStack 専用で、`opendev.org/openstack/releases`（リリース定義を
YAML で管理する専用リポジトリ）を clone して版と日付を得ている。他のプロジェクトには
そうしたリポジトリが無いため、**Gerrit 自身が持つタグ**から版と日付を組み立てる。

    GET /projects/<name>/tags/?n=1000&S=<offset>
    → [{"ref": "refs/tags/v6.8.0", "created": "2024-10-07 ...", ...}, ...]

どのタグを版とみなすかは `constants.GERRIT_PROJECTS` の `tag_pattern` / `tag_series` で決める。
RC やビルド番号まで刻むプロジェクト（LibreOffice の `libreoffice-25.2.0.3` など）があるため、
**同じ版シリーズに属するタグのうち最も古い日付**をそのリリース日として採用する。
日付のないタグしかない版は飛ばし、ログに警告として残す（project_selection/design.md §5.2
「追記：区切りの日をタグから取る」）。

出力は既存の `major_releases_summary.csv` と同じ列（project, series, version, release_date, status, yaml_url）で、
**対象のプロジェクトの行をすべて入れ替え、ほかのプロジェクトの行と並びは変えない**（OpenStack の行を壊さない）。

既定の対象は `constants.RELEASE_TAG_TARGETS`（qtbase・qt-creator・mediawiki/core・qtdeclarative・go・TYPO3.CMS の 6 件）。表全体の作り直しは
`python -m src.collectors.build_release_table`（OpenStack の行を作る `major_release_storage` のあとに、これを呼ぶ）。

実行:
    python -m src.collectors.release_tags_collector                     # 既定の対象すべて
    python -m src.collectors.release_tags_collector --project qtbase    # 1 つだけ
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config.path import DEFAULT_DATA_DIR
from src.utils.constants import GERRIT_PROJECTS, RELEASE_TAG_TARGETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

OUTPUT_CSV = DEFAULT_DATA_DIR / "openstack_collected" / "major_releases_summary.csv"
# major_release_storage.py の表と同じ列（series はサイクル名で、タグから取る版には無いので空）
COLUMNS = ["project", "series", "version", "release_date", "status", "yaml_url"]
PAGE = 1000
REQUEST_DELAY = 1.0   # 問い合わせの間隔（秒）。収集と同じ既定値（project_selection/design.md §6.2）
PROJECT_INTERVAL = 30.0  # リポジトリとリポジトリの間に空ける秒数（project_selection/design.md §5.2「1 本ずつ、30 秒ほど空ける」）


def _session(spec: dict) -> requests.Session:
    """接続用セッション（auth=True のホストだけ Basic 認証を付ける）。

    資格情報は Gerrit インスタンスごとに違うので、登録表の `env` が指す接頭辞から読む
    （既定 GERRIT_* = OpenStack、Qt は QT_GERRIT_*）。
    """
    s = requests.Session()
    # 収集（change_collector）と同じ User-Agent。Wikimedia は連絡先を求める（project_selection/design.md §6.2）
    load_dotenv()
    contact = os.getenv("CONTACT") or "academic study of code review"
    s.headers.update({"Accept": "application/json",
                      "User-Agent": f"review-priority-research/1.0 ({contact})"})
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
        # 登録表に delay があるホスト（Wikimedia は 8 秒）はそれに従う
        time.sleep(float(spec.get("delay", REQUEST_DELAY)))
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
    undated: dict[str, list[str]] = {}   # 日付の取れなかったタグ（版 -> タグ名）
    for t in tags:
        name = str(t.get("ref", "")).replace("refs/tags/", "")
        m = rx.match(name)
        if not m:
            continue
        version = m.expand(series)
        # 注釈付きタグは tagger.date、軽量タグは created に日付が入る（入らないこともある）
        raw = (t.get("tagger") or {}).get("date") or t.get("created")
        date = pd.to_datetime(str(raw).replace(".000000000", ""), errors="coerce") if raw else pd.NaT
        if pd.isna(date):
            undated.setdefault(version, []).append(name)
            continue
        if version not in first or date < first[version]:
            first[version] = date

    # 日付のあるタグが 1 本も無い版は飛ばし、ログに残す（黙って抜けさせない）
    skipped = {v: names for v, names in undated.items() if v not in first}
    if skipped:
        logger.warning(f"[{project_key}] 日付の取れないタグしかない版を飛ばしました（{len(skipped)} 版）: "
                       + ", ".join(f"{v}（{'・'.join(n)}）" for v, n in sorted(skipped.items())))

    rows = [{"project": project_key, "series": "", "version": v, "release_date": d.date().isoformat(),
             "status": "released",
             "yaml_url": f"{spec['host'].rstrip('/')}/projects/"
                         f"{urllib.parse.quote(spec['path'], safe='')}/tags/"}
            for v, d in sorted(first.items(), key=lambda kv: kv[1])]
    logger.info(f"[{project_key}] 版 {len(rows)} 件を抽出"
                + (f"（{rows[0]['version']} {rows[0]['release_date']} 〜 "
                   f"{rows[-1]['version']} {rows[-1]['release_date']}）" if rows else ""))
    return pd.DataFrame(rows, columns=COLUMNS)


def merge_into_csv(new: pd.DataFrame, csv_path: Path = OUTPUT_CSV) -> Path:
    """既存 CSV に併合して書き戻す。

    new に含まれるプロジェクトの行は**すべて入れ替える**（古い版の行が残らないように）。
    ほかのプロジェクトの行（OpenStack など）は、中身も並びも変えない。
    """
    csv_path = Path(csv_path)
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        others = old[~old["project"].isin(new["project"].unique())]
        merged = pd.concat([others, new.astype(str)], ignore_index=True)
        cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
    else:
        merged = new
        cols = list(new.columns)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged[cols].to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"保存: {csv_path}（全 {len(merged)} 行）")
    return csv_path


def collect(projects: list[str]) -> pd.DataFrame:
    frames = []
    for i, p in enumerate(projects):
        if p not in GERRIT_PROJECTS:
            logger.warning(f"未登録のプロジェクト: {p}")
            continue
        if i > 0:
            time.sleep(PROJECT_INTERVAL)   # 1 本ずつ、リポジトリの間を空ける
        frames.append(extract_releases(p, fetch_tags(p)))
    if not frames:
        logger.warning("対象がありません")
        return pd.DataFrame(columns=COLUMNS)
    new = pd.concat(frames, ignore_index=True)
    merge_into_csv(new)
    return new


def _parse_args():
    ap = argparse.ArgumentParser(description="Gerrit の tags API からリリース情報を収集する")
    ap.add_argument("--project", nargs="*", default=None,
                    help=f"対象（既定 {RELEASE_TAG_TARGETS}）")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collect(args.project or list(RELEASE_TAG_TARGETS))
