"""
Change / リリース日の読み込み（分析モジュール共通）。

`background_problem` 配下の各分析（priority_distribution / change_lifetime 等）で
共通して使う「読むだけ」のローダ。ボット判定など分析固有のロジックは含めない。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config.path import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# データの既定パス（プロジェクト共通）
DEFAULT_CHANGES_DIR_TEMPLATE = DEFAULT_DATA_DIR / "openstack_collected" / "{project}" / "changes"
DEFAULT_RELEASES_SUMMARY_CSV = DEFAULT_DATA_DIR / "openstack_collected" / "major_releases_summary.csv"


# ── Change の読み込み ───────────────────────────────────
def load_changes(project: str, changes_dir: Path | None = None) -> list[dict]:
    """指定プロジェクトの changes/*.json をすべて読み込む。

    1 ファイルが「単一 dict」でも「dict の list」でも吸収する。
    """
    if changes_dir is None:
        changes_dir = Path(str(DEFAULT_CHANGES_DIR_TEMPLATE).format(project=project))
    changes_dir = Path(changes_dir)

    if not changes_dir.exists():
        logger.warning(f"Change ディレクトリが見つかりません: {changes_dir}")
        return []

    changes: list[dict] = []
    for json_path in sorted(changes_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"読み込み失敗のためスキップ: {json_path} ({e})")
            continue
        if isinstance(data, list):
            changes.extend(d for d in data if isinstance(d, dict))
        elif isinstance(data, dict):
            changes.append(data)
    logger.info(f"[{project}] {len(changes)} 件の Change を読み込みました（{changes_dir}）")
    return changes


# ── リリース日の読み込み ─────────────────────────────────
def load_release_dates(csv_path: Path | None = None) -> pd.DataFrame:
    """major_releases_summary.csv を読み、release_date を datetime 化して返す。"""
    if csv_path is None:
        csv_path = DEFAULT_RELEASES_SUMMARY_CSV
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"リリース一覧 CSV が見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    return df


def _drop_release_anomalies(project_df: pd.DataFrame) -> pd.DataFrame:
    """版数と日付が矛盾する異常エントリを除外する。

    例: nova の "2010.1" / "2011.1" は release_date が 2018-02-21 と、
    年ベースの版数（2010 / 2011）から大きくずれている。こうした行が
    「直前リリース」判定を誤らせるため、年ベース版の年と release_date の年が
    2 年以上ずれる行を除外する。意味的バージョン（12.0.0 等）には影響しない。
    """
    keep_mask = []
    for _, row in project_df.iterrows():
        version = str(row["version"])
        ok = True
        head = version.split(".")[0]
        # 年ベース版（"2015" のように 4 桁で始まる）かどうか
        if len(head) == 4 and head.isdigit():
            version_year = int(head)
            release_year = row["release_date"].year
            if abs(release_year - version_year) > 1:
                ok = False
        keep_mask.append(ok)
    return project_df[keep_mask]


def get_release_cycle(
    df: pd.DataFrame, project: str, version: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """対象リリースと「1 つ前のリリース」から開発サイクル (cycle_start, cycle_end) を返す。

    - cycle_end = 対象リリースの release_date
    - cycle_start = 同プロジェクトで release_date が cycle_end より前のうち最大のもの
      （異常エントリは除外済み）

    直前リリースが見つからない場合は ValueError を送出する（呼び出し側でスキップ）。
    """
    project_df = df[df["project"] == project]
    target = project_df[project_df["version"] == version]
    if target.empty:
        raise ValueError(f"リリースが見つかりません: project={project}, version={version}")

    cycle_end = target["release_date"].iloc[0]

    cleaned = _drop_release_anomalies(project_df)
    earlier = cleaned[cleaned["release_date"] < cycle_end]
    if earlier.empty:
        raise ValueError(
            f"直前リリースが見つかりません: project={project}, version={version}"
        )
    cycle_start = earlier["release_date"].max()
    return cycle_start, cycle_end
