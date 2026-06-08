"""
shared_ranking キャッシュの I/O。

キャッシュキーは以下の組み合わせから生成される:
  project, release, period_start, period_end,
  min_query_size, max_censoring_seconds,
  feature_names, logic_version
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.analysis.shared_ranking.constants import (
    CACHE_FILENAME_PREFIX,
    FEATURE_NAMES,
    LOGIC_VERSION,
    MAX_CENSORING_SECONDS,
    MIN_QUERY_SIZE,
    SHARED_RANKING_CACHE_DIR,
)

logger = logging.getLogger(__name__)


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def compute_param_hash(
    project_name: str,
    release_version: str,
    period_start: datetime,
    period_end: datetime,
    min_query_size: int = MIN_QUERY_SIZE,
    max_censoring_seconds: int = MAX_CENSORING_SECONDS,
    feature_names: Optional[List[str]] = None,
    logic_version: str = LOGIC_VERSION,
) -> str:
    """パラメータから 16 桁の hex hash を生成する。"""
    payload = {
        "project": project_name,
        "release": release_version,
        "period_start": _to_iso(period_start),
        "period_end": _to_iso(period_end),
        "min_query_size": int(min_query_size),
        "max_censoring_seconds": int(max_censoring_seconds),
        "feature_names": list(feature_names or FEATURE_NAMES),
        "logic_version": logic_version,
    }
    serialized = json.dumps(payload, sort_keys=True, default=_to_iso)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


def get_cache_paths(
    project_name: str,
    release_version: str,
    param_hash: str,
    cache_dir: Path = SHARED_RANKING_CACHE_DIR,
) -> Tuple[Path, Path]:
    """(.pkl, .meta.json) のペアを返す。"""
    base = Path(cache_dir) / project_name / release_version
    pkl_path = base / f"{CACHE_FILENAME_PREFIX}_{param_hash}.pkl"
    meta_path = base / f"{CACHE_FILENAME_PREFIX}_{param_hash}.meta.json"
    return pkl_path, meta_path


def load_cache(pkl_path: Path) -> Optional[pd.DataFrame]:
    """キャッシュ pickle を読み込む。失敗時は None。"""
    if not pkl_path.exists():
        return None
    try:
        df = pd.read_pickle(pkl_path)
        if not isinstance(df, pd.DataFrame):
            logger.warning("キャッシュが DataFrame ではありません: %s", pkl_path)
            return None
        logger.info("shared_ranking キャッシュを読み込み: %s (rows=%d)", pkl_path, len(df))
        return df
    except Exception as exc:
        logger.warning("shared_ranking キャッシュ読み込みに失敗 (%s): %s", pkl_path, exc)
        return None


def save_cache(
    df: pd.DataFrame,
    pkl_path: Path,
    meta_path: Path,
    metadata: Dict[str, Any],
) -> None:
    """pickle と meta JSON を書き出す。"""
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.to_pickle(df, pkl_path)
        meta = dict(metadata)
        meta.update(
            {
                "logic_version": LOGIC_VERSION,
                "row_count": int(len(df)),
                "query_count": int(df["query_id"].nunique()) if "query_id" in df.columns else 0,
                "generated_at": datetime.now().isoformat(),
            }
        )
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=_to_iso),
            encoding="utf-8",
        )
        logger.info("shared_ranking キャッシュを保存: %s (rows=%d)", pkl_path, len(df))
    except Exception as exc:
        logger.warning("shared_ranking キャッシュ保存に失敗 (%s): %s", pkl_path, exc)
