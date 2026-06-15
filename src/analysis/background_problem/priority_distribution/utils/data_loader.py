"""
ボット/人間判定など、priority_distribution 固有の入力ヘルパー。

Change / リリース日の読み込み（`load_changes` / `load_release_dates` /
`get_release_cycle`）は共通モジュール `background_problem.common.data_loader` に集約し、
ここでは後方互換のために再エクスポートする。
"""
from __future__ import annotations

import configparser
import csv
import logging
from datetime import datetime
from pathlib import Path

from src.analysis.background_problem.common.data_loader import (  # noqa: F401
    get_release_cycle,
    load_changes,
    load_release_dates,
)
from src.analysis.background_problem.common.time_utils import parse_dt
from src.analysis.background_problem.priority_distribution.utils import constants

logger = logging.getLogger(__name__)


# ── ボット名の読み込み ─────────────────────────────────
def _load_gerrymander_bots(config_path: Path) -> set[str]:
    """gerrymanderconfig.ini の [organization] bots を読む（小文字化）。"""
    bot_names: set[str] = set()
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        if "organization" in config and "bots" in config["organization"]:
            bot_names = {
                name.strip().lower()
                for name in config["organization"]["bots"].split(",")
                if name.strip()
            }
        else:
            logger.warning(f"'{config_path}' に [organization] / bots が見つかりません。")
    except (configparser.Error, OSError) as e:
        logger.error(f"'{config_path}' の読み込みに失敗しました: {e}")
    return bot_names


def load_bot_accounts_csv(csv_path: Path | None = None) -> set[str]:
    """third_party_ci_accounts.csv（name,email）から識別子集合を読む（小文字化）。

    name と email の両方を照合候補として返す（どちらかが author に一致すればボット）。
    """
    if csv_path is None:
        csv_path = constants.BOT_ACCOUNTS_CSV
    csv_path = Path(csv_path)

    identifiers: set[str] = set()
    if not csv_path.exists():
        logger.warning(f"CI アカウント一覧 CSV が見つかりません: {csv_path}")
        return identifiers

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            if i == 0 and [c.strip().lower() for c in row[:2]] == ["name", "email"]:
                continue  # ヘッダ行をスキップ
            name = row[0].strip().lower() if len(row) > 0 else ""
            email = row[1].strip().lower() if len(row) > 1 else ""
            if name:
                identifiers.add(name)
            if email:
                identifiers.add(email)
    return identifiers


def load_extra_bots(path: Path | None = None) -> set[str]:
    """extra_bots.txt（1 行 1 名、'#' コメント可）からボット名集合を読む（小文字化）。"""
    if path is None:
        path = constants.EXTRA_BOTS_FILE
    path = Path(path)

    names: set[str] = set()
    if not path.exists():
        logger.warning(f"追加ボット一覧が見つかりません: {path}")
        return names

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(line.lower())
    return names


def load_bot_names(
    config_path: Path | None = None,
    csv_path: Path | None = None,
    extra_path: Path | None = None,
) -> set[str]:
    """ボット/自動アカウントの識別子集合を読み込む（すべて小文字、大文字小文字無視で照合）。

    出所は次の 3 つ（すべて src/config 配下で定義）:
    1. gerrymanderconfig.ini の [organization] bots（username）
    2. third_party_ci_accounts.csv（サードパーティ CI の name / email）
    3. extra_bots.txt（jenkins / zuul 等、上記未登録の補完）
    """
    if config_path is None:
        config_path = constants.GERRYMANDER_CONFIG

    bot_names = _load_gerrymander_bots(Path(config_path))
    bot_names |= load_bot_accounts_csv(csv_path)
    bot_names |= load_extra_bots(extra_path)
    logger.info(f"ボット/自動アカウント識別子数: {len(bot_names)}")
    return bot_names


# ── 人間判定・投稿者判定 ─────────────────────────────────
def _author_identifiers(author: dict) -> set[str]:
    """author dict から照合用の文字列集合（name / username / email / _account_id）を作る。"""
    if not isinstance(author, dict):
        return {str(author)} if author else set()
    ids: set[str] = set()
    for key in ("name", "username", "email"):
        val = author.get(key)
        if val:
            ids.add(str(val))
    account_id = author.get("_account_id")
    if account_id is not None:
        ids.add(str(account_id))
    return ids


def is_bot(author: dict, bot_names: set[str]) -> bool:
    """author がボットか（name / username / email のいずれかが bot_names に一致）。

    bot_names は小文字で保持されている前提で、大文字小文字を無視して照合する。
    """
    identifiers = {i.lower() for i in _author_identifiers(author)}
    return bool(identifiers & set(bot_names))


def is_owner(author: dict, change: dict) -> bool:
    """author が Change の投稿者本人か（_account_id → email → username/name の順で照合）。"""
    owner = change.get("owner", {})
    if not isinstance(owner, dict) or not isinstance(author, dict):
        return False
    # 最も信頼できる _account_id を優先
    if owner.get("_account_id") is not None and author.get("_account_id") is not None:
        if owner["_account_id"] == author["_account_id"]:
            return True
    for key in ("email", "username", "name"):
        if owner.get(key) and author.get(key) and owner[key] == author[key]:
            return True
    return False


def is_human_comment(message: dict, change: dict, bot_names: set[str]) -> bool:
    """メッセージが「人間のレビューコメント」か（§2.2）。

    次のいずれかに該当するものは人間ではないとみなす:
    - Gerrit が自動生成したメッセージ（tag が "autogenerated:" で始まる）
    - ボット（bot_names に一致。gerrymander + CI 一覧 CSV + EXTRA_BOT_NAMES）
    - 投稿者本人
    """
    tag = message.get("tag")
    if isinstance(tag, str) and tag.startswith("autogenerated:"):
        return False
    author = message.get("author", {})
    if is_bot(author, bot_names):
        return False
    if is_owner(author, change):
        return False
    return True


def human_comment_times(change: dict, bot_names: set[str]) -> list[datetime]:
    """Change の人間レビューコメントの時刻一覧（昇順）を返す。"""
    times: list[datetime] = []
    for message in change.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        if not is_human_comment(message, change, bot_names):
            continue
        dt = parse_dt(message.get("date"))
        if dt is not None:
            times.append(dt)
    times.sort()
    return times
