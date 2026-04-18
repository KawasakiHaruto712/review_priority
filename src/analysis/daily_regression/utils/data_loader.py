"""
データ読み込みモジュール
リリース情報、Changeデータ、bot名を読み込む
trend_metrics の data_loader と同様のパターンで実装
"""

import json
import logging
import configparser
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.analysis.daily_regression.utils.constants import DATA_LOAD_CONFIG

logger = logging.getLogger(__name__)


def load_major_releases_summary(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    メジャーリリース情報を読み込む

    Args:
        csv_path: CSVファイルのパス（省略時はデフォルトパスを使用）

    Returns:
        pd.DataFrame: メジャーリリース情報
    """
    if csv_path is None:
        csv_path = Path(DATA_LOAD_CONFIG['major_releases_file'])
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"メジャーリリース情報ファイルが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)
    df['release_date'] = pd.to_datetime(df['release_date'])

    if 'project' in df.columns and 'component' not in df.columns:
        df = df.rename(columns={'project': 'component'})

    logger.info(f"メジャーリリース情報を読み込みました: {len(df)} 件")
    return df


def get_release_date(releases_df: pd.DataFrame, project: str, version: str) -> pd.Timestamp:
    """
    指定プロジェクト・バージョンのリリース日を取得

    Args:
        releases_df: リリース情報のDataFrame
        project: プロジェクト名
        version: バージョン番号

    Returns:
        pd.Timestamp: リリース日
    """
    project_col = 'project' if 'project' in releases_df.columns else 'component'
    mask = (releases_df[project_col] == project) & (releases_df['version'] == version)
    result = releases_df[mask]['release_date']

    if len(result) == 0:
        raise ValueError(f"Release not found: {project} {version}")

    release_date = result.iloc[0]
    logger.info(f"リリース日を取得: {project} {version} -> {release_date}")
    return release_date


def load_all_changes(project: str = None, changes_dir: Path = None) -> List[Dict]:
    """
    指定ディレクトリまたはプロジェクトから全Changeデータを読み込む

    Args:
        project: プロジェクト名
        changes_dir: Changeデータが格納されたディレクトリのパス

    Returns:
        List[Dict]: Changeデータのリスト
    """
    if changes_dir is not None:
        changes_dir = Path(changes_dir)
    elif project is not None:
        changes_dir = Path(DATA_LOAD_CONFIG['changes_dir_template'].format(project=project))
    else:
        raise ValueError("projectまたはchanges_dirのいずれかを指定してください")

    if not changes_dir.exists():
        raise FileNotFoundError(f"Changeデータディレクトリが見つかりません: {changes_dir}")

    changes = []
    change_files = list(changes_dir.glob('*.json'))

    logger.info(f"Changeファイルを読み込み中: {len(change_files)} 件")

    for change_file in change_files:
        try:
            with open(change_file, 'r', encoding='utf-8') as f:
                change_data = json.load(f)
                if isinstance(change_data, list):
                    changes.extend(change_data)
                else:
                    changes.append(change_data)
        except Exception as e:
            logger.warning(f"ファイル読み込みエラー: {change_file}, エラー: {e}")
            continue

    logger.info(f"Changeデータを読み込みました: {len(changes)} 件")
    return changes


def load_bot_names_from_config(config_path: Optional[Path] = None) -> List[str]:
    """
    設定ファイルからbot名のリストを読み込む

    Args:
        config_path: 設定ファイルのパス（省略時はデフォルトパスを使用）

    Returns:
        List[str]: bot名のリスト（小文字）
    """
    if config_path is None:
        config_path = Path(DATA_LOAD_CONFIG['bot_config_file'])
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Bot設定ファイルが見つかりません: {config_path}")
        return []

    try:
        config = configparser.ConfigParser()
        config.read(config_path)

        if 'organization' in config and 'bots' in config['organization']:
            bot_names_str = config['organization']['bots']
            bot_names = [name.strip().lower() for name in bot_names_str.split(',')]
        else:
            bot_names = []

        logger.info(f"Bot名を読み込みました: {len(bot_names)} 件")
        return bot_names

    except Exception as e:
        logger.warning(f"Bot設定ファイル読み込みエラー: {e}")
        return []


def is_bot(author_name: str, bot_names: List[str]) -> bool:
    """
    著者がbotかどうかを判定

    Args:
        author_name: 著者名
        bot_names: bot名のリスト

    Returns:
        bool: botの場合True
    """
    if not author_name or not bot_names:
        return False

    author_lower = author_name.lower()
    return any(bot_name in author_lower for bot_name in bot_names)
