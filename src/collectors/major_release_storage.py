"""メジャーリリース抽出（major_releases_summary.csv の生成）。

`releases_summary.csv`（全リリース：patch・EOL マーカー等も含む）から、各プロジェクトの
「サイクルごとの主要リリース」だけを抽出し、`major_releases_summary.csv` を生成する。
分析側 `load_release_dates` はこの CSV をサイクル境界として使う。

プロジェクト別ルール（版付け方式がプロジェクトで異なるため）:
  - nova 型（nova / neutron / cinder / glance / keystone）: `X.0.0`（サイクルごとに major を上げる）
  - swift                                                 : `2.Y.0`（major は 2 のまま minor を上げる）
    ※ swift は independent release model で `X.0.0` に該当するのが `2.0.0` の1つだけなので別ルール。

変換: `component` → `project` に読み替え、`version / release_date / yaml_url` はそのまま引き継ぐ。
"""

import logging
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from ..config import path as app_path
except ImportError:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    from src.config import path as app_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# 主要リリース判定の正規表現（プロジェクト別）
_NOVA_STYLE = re.compile(r"^\d+\.0\.0$")   # 例: 26.0.0（サイクルごとに major を上げる）
_SWIFT_STYLE = re.compile(r"^2\.\d+\.0$")  # 例: 2.34.0（major は 2 のまま minor を上げる）
_SWIFT = "swift"


def _is_major(component: str, version: str) -> bool:
    """その (component, version) がサイクルの主要リリースか。"""
    v = str(version).strip()
    if component == _SWIFT:
        return bool(_SWIFT_STYLE.match(v))
    return bool(_NOVA_STYLE.match(v))


class MajorReleaseStorage:
    """`releases_summary.csv` からメジャーリリースを抽出して `major_releases_summary.csv` を書き出す。"""

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: `releases_summary.csv` があり、`major_releases_summary.csv` を出力するディレクトリ。
        """
        self.data_dir = Path(data_dir)
        self.input_path = self.data_dir / "releases_summary.csv"
        self.output_path = self.data_dir / "major_releases_summary.csv"

    def extract(self) -> pd.DataFrame:
        """メジャーリリースだけを抽出した DataFrame を返す（`project, version, release_date, yaml_url`）。"""
        if not self.input_path.exists():
            raise FileNotFoundError(f"入力が見つかりません: {self.input_path}")
        df = pd.read_csv(self.input_path)

        mask = df.apply(lambda r: _is_major(r["component"], r["version"]), axis=1)
        major = df[mask].copy()
        major = major.rename(columns={"component": "project"})
        major = major[["project", "version", "release_date", "yaml_url"]]
        major = major.sort_values(by=["project", "release_date"], ascending=[True, False])
        return major.reset_index(drop=True)

    def save(self) -> Path:
        """抽出して `major_releases_summary.csv` に保存し、パスを返す。"""
        major = self.extract()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        major.to_csv(self.output_path, index=False, encoding="utf-8")
        logger.info(f"メジャーリリース {len(major)} 件を抽出しました。")
        for project, sub in major.groupby("project"):
            logger.info(f"  {project}: {len(sub)} 件")
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
