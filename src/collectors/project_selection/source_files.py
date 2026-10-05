"""
ファイルパスが実装言語のものかを判定する（E1 に使う。design.md §5.1）

「ソースコードとは何か」を自作の一覧で決めず、GitHub の linguist の定義に委ねる。
linguist は各言語に type を付けており、そのうち type: programming（実装言語）だけを
ソースコードとみなす。markup（HTML 等）・data（JSON/YAML 等）・prose（文書）は数えない。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, Set, Tuple

import yaml

from src.config.path import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

LINGUIST_PATH = DEFAULT_CONFIG / "linguist_languages.yml"

# Gerrit が変更ファイル一覧に混ぜる擬似パス（実在のファイルではない）
PSEUDO_PATHS = {"/COMMIT_MSG", "/MERGE_LIST", "/PATCHSET_LEVEL"}


@lru_cache(maxsize=1)
def _programming_definitions() -> Tuple[Set[str], Set[str]]:
    """linguist から (拡張子, 拡張子のないファイル名) を読む。いずれも小文字。

    1 つの拡張子を実装言語と非実装言語が取り合っている場合は、**実装言語とみなさない**。
    例: ".md" は Markdown（prose）と GCC Machine Description（programming）の両方が使う。
    linguist 本体はファイルの中身を見て区別するが、パスだけでは決められないため。

    E1 は「ソースコードを含まないリポジトリ」を落とす基準なので、控えめに判定してよい。
    本物の開発リポジトリは曖昧でない拡張子（.c / .py など）を必ず含むので落ちない。
    逆に文書だけのリポジトリを残してしまうほうが問題になる。
    """
    with open(LINGUIST_PATH, "r", encoding="utf-8") as f:
        languages = yaml.safe_load(f)

    programming: Set[str] = set()
    other: Set[str] = set()
    filenames: Set[str] = set()
    for spec in languages.values():
        is_programming = spec.get("type") == "programming"
        for ext in spec.get("extensions") or []:
            (programming if is_programming else other).add(str(ext).lower())
        if is_programming:
            for name in spec.get("filenames") or []:
                filenames.add(str(name).lower())

    ambiguous = programming & other
    extensions = programming - ambiguous
    logger.debug(f"linguist: 拡張子 {len(extensions)} 種（取り合い {len(ambiguous)} 種を除外）"
                 f"・ファイル名 {len(filenames)} 種")
    return extensions, filenames


def is_source_path(path: str) -> bool:
    """そのパスが実装言語のファイルか。

    拡張子は最も長いものから順に見る。linguist には ".8xp.txt" のように
    ドットを含む拡張子があり、末尾の 1 つだけでは判定できないため。
    """
    if path in PSEUDO_PATHS:
        return False
    extensions, filenames = _programming_definitions()

    name = path.rsplit("/", 1)[-1].lower()
    if name in filenames:          # Makefile / Rakefile など拡張子のないもの
        return True

    parts = name.split(".")
    for i in range(1, len(parts)):
        if "." + ".".join(parts[i:]) in extensions:
            return True
    return False


def has_source(change: Dict[str, Any]) -> bool:
    """その Change が実装言語のファイルを 1 つでも含むか。"""
    for revision in (change.get("revisions") or {}).values():
        for path in (revision.get("files") or {}):
            if is_source_path(path):
                return True
    return False
