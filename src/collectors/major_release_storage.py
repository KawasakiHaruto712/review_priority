"""メジャーリリースの表（major_releases_summary.csv）の生成。

OpenStack の公式のサイクル一覧（releases リポジトリの `data/series_status.yaml`）から、
**サイクル単位**の表を作る。分析側 `load_release_dates` はこの CSV を、特徴量 days_to_major_release と
分析期間の区切り（事前学習の締め・評価する版の期間）に使う。
設計：src/collectors/README.md の「メジャーリリースの表」、pretrained_encoders/design.md §11.4。

列（1 行 = 1 project × 1 サイクル）:
  project       nova など
  series        サイクル名（yoga など）。series_status.yaml の name
  version       その project がそのサイクルで最初に出した正式版（rc・b・-eom・-eol などは除く）。
                deliverables/<series>/<project>.yaml から取る。まだ出ていないサイクルは空
  release_date  series_status.yaml の initial-release（公式のサイクルのリリース日。6 件共通。swift もこの日付）
  status        released（正式版が出ている）/ planned（まだ出ていない。日付は予定日）
  yaml_url      そのサイクルの deliverables ファイルの URL

旧版（2026-09）は `releases_summary.csv` から版番号の形（nova 型は X.0.0、swift は 2.Y.0）で区切りを拾っていた。
版の付け方が途中で変わる（2011.2 → 2015.1.0 → 12.0.0）ため 2015 年秋より前が抜け、swift は 1 サイクルに
区切りが複数あった。サイクル単位にすると版番号を解釈しなくてよい。

OpenStack 以外の行（release_tags_collector が追記する qtbase・qtcreator・libreoffice）は、作り直すときも残す。
"""

import logging
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

try:
    from ..config import path as app_path
    from ..utils import constants
except ImportError:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    from src.config import path as app_path
    from src.utils import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# 正式版：数字とドットだけの版（2011.2 / 2015.1.0 / 25.0.0 / 2.29.0）。
# rc・b（試験版。例 20.0.0.0rc1）や -eom・-eol・-em（保守終了の印。例 yoga-eom）は除く
_FINAL_VERSION = re.compile(r"^\d+(\.\d+)+$")
_YAML_URL = "https://opendev.org/openstack/releases/src/branch/master/deliverables/{series}/{project}.yaml"
COLUMNS = ["project", "series", "version", "release_date", "status", "yaml_url"]


def _first_final_version(yaml_path: Path, project: str) -> str | None:
    """deliverables ファイルから、その project の最初の正式版を返す。無ければ None。"""
    if not yaml_path.exists():
        return None
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    for release in data.get("releases") or []:
        version = str(release.get("version", "")).strip()
        repos = {p.get("repo") for p in release.get("projects") or []}
        if f"openstack/{project}" in repos and _FINAL_VERSION.match(version):
            return version
    return None


class MajorReleaseStorage:
    """公式のサイクル一覧から、サイクル単位の `major_releases_summary.csv` を書き出す。"""

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: `releases_repo/`（openstack/releases の clone）があり、`major_releases_summary.csv` を出力するディレクトリ。
        """
        self.data_dir = Path(data_dir)
        self.repo_path = self.data_dir / "releases_repo"
        self.series_path = self.repo_path / "data" / "series_status.yaml"
        self.output_path = self.data_dir / "major_releases_summary.csv"
        self.projects = list(constants.OPENSTACK_CORE_COMPONENTS)

    def _series(self) -> list[tuple[str, date]]:
        """(サイクル名, 公式のリリース日) を日付の昇順で返す。"""
        if not self.series_path.exists():
            raise FileNotFoundError(f"サイクル一覧が見つかりません: {self.series_path}")
        rows = []
        for s in yaml.safe_load(self.series_path.read_text(encoding="utf-8")) or []:
            d = s.get("initial-release")
            if not d:
                logger.warning(f"サイクル {s.get('name')} に initial-release がないので除外します")
                continue
            rows.append((s["name"], pd.Timestamp(str(d)).date()))
        return sorted(rows, key=lambda r: r[1])

    def extract(self) -> pd.DataFrame:
        """OpenStack の対象 project ぶんの表を返す（列は COLUMNS）。"""
        series = self._series()
        deliverables = self.repo_path / "deliverables"
        today = date.today()
        rows = []
        for project in self.projects:
            # deliverables ファイルが最初に現れたサイクルから、一覧の最新のサイクル（予定を含む）まで
            first = next((i for i, (name, _) in enumerate(series)
                          if (deliverables / name / f"{project}.yaml").exists()), None)
            if first is None:
                logger.warning(f"[{project}] deliverables が 1 つも見つかりません")
                continue
            for name, release_date in series[first:]:
                yaml_path = deliverables / name / f"{project}.yaml"
                version = _first_final_version(yaml_path, project)
                if version:
                    status = "released"
                elif release_date > today:
                    status = "planned"
                else:
                    # 公式の日付は過ぎているのに正式版がない。区切りの日付は使うが、ログに残す
                    status = "planned"
                    logger.warning(f"[{project}] {name}（{release_date}）は日付を過ぎているが正式版がありません"
                                   f"（releases リポジトリが古い可能性）")
                rows.append({
                    "project": project, "series": name, "version": version or "",
                    "release_date": release_date.isoformat(), "status": status,
                    "yaml_url": _YAML_URL.format(series=name, project=project) if yaml_path.exists() else "",
                })
        df = pd.DataFrame(rows, columns=COLUMNS)
        return df.sort_values(by=["project", "release_date"], ascending=[True, False]).reset_index(drop=True)

    def save(self) -> Path:
        """作り直して保存し、パスを返す。OpenStack 以外の既存の行は残す。"""
        new = self.extract()
        if self.output_path.exists():
            old = pd.read_csv(self.output_path, dtype=str, keep_default_na=False)
            others = old[~old["project"].isin(self.projects)]
            if len(others):
                logger.info(f"OpenStack 以外の行を残します: "
                            + ", ".join(f"{p} {n} 件" for p, n in others["project"].value_counts().items()))
            new = pd.concat([new, others], ignore_index=True)
        cols = COLUMNS + [c for c in new.columns if c not in COLUMNS]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        new[cols].to_csv(self.output_path, index=False, encoding="utf-8")

        for project, sub in new[new["project"].isin(self.projects)].groupby("project"):
            sub = sub.sort_values("release_date")
            planned = sub[sub["status"] == "planned"]
            logger.info(f"  {project}: {len(sub)} サイクル（{sub.iloc[0]['series']} {sub.iloc[0]['release_date']} 〜 "
                        f"{sub.iloc[-1]['series']} {sub.iloc[-1]['release_date']}）"
                        + (f"、planned: {', '.join(planned['series'])}" if len(planned) else ""))
        logger.info(f"出力先: {self.output_path}")
        return self.output_path


if __name__ == "__main__":
    logging.info("===== メジャーリリース抽出を開始します =====")
    data_dir = app_path.DEFAULT_DATA_DIR / "openstack_collected"
    try:
        MajorReleaseStorage(data_dir=data_dir).save()
    except Exception as e:
        logging.critical(f"実行中にエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)
