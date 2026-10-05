"""
選定の母集団となる Gerrit インスタンスの一覧

既存の GERRIT_PROJECTS（src/utils/constants.py）は**リポジトリ単位**の表で、
「どのリポジトリを取るか」が決まっていることを前提にしている。
選定の第 1 段階ではそれがまだ決まっていないので、**インスタンス単位**の表が要る
（design.md §7.1）。

選定が終わったら、選ばれたリポジトリを GERRIT_PROJECTS に追記する。
そこから先は既存の収集の仕組みがそのまま使える。
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# **全インスタンス共通の既定の間隔（秒）。**
# 相手のサーバに負荷をかけないため、必ず間隔を空けてから次を投げる。
# 間隔なしで連続して投げると、短時間で数千リクエストに達し、
# 相手側の自動検出に掛かる（2026-10-02 に Google の確認画面が出た）。
# 個別に長くしたいインスタンスは HOSTS の delay で上書きする。
DEFAULT_REQUEST_DELAY = 1.0


# 母集団（design.md §2.3）
#   key        : 出力 CSV に載せるインスタンス名
#   base       : API のベース URL。認証するインスタンスは末尾に /a を付ける
#   auth       : True なら Basic 認証する
#   env        : 資格情報を読む .env のキー接頭辞（GERRIT_USERNAME / GERRIT_PASSWORD）
#   slice      : 走査の区切り（"day" / "week" / "month"）。design.md §3.3.1
#                深いオフセットで遅くなるインスタンスほど細かく切る
#   delay      : 1 リクエスト後に空ける秒数。頻度制限のあるインスタンス用
#   page_size  : 1 リクエストで要求する件数
#   skip_diffstat : SKIP_DIFFSTAT オプションを付けるか。古い Gerrit は非対応
#   exclude    : 走査から外すリポジトリ。既に全期間を収集済みのもの（design.md §3.3.3）
#                除外した分はローカルのデータから作成日で数える
HOSTS: Dict[str, Dict[str, Any]] = {
    "Android": {
        "base": "https://android-review.googlesource.com",
        "auth": False, "slice": "day", "page_size": 500,
    },
    "ChromiumOS": {
        "base": "https://chromium-review.googlesource.com",
        "auth": False, "slice": "day", "page_size": 500,
    },
    "Gerrit": {
        "base": "https://gerrit-review.googlesource.com",
        "auth": False, "slice": "month", "page_size": 500,
    },
    "Go": {
        "base": "https://go-review.googlesource.com",
        "auth": False, "slice": "month", "page_size": 500,
    },
    "Couchbase": {
        "base": "https://review.couchbase.org",
        "auth": False, "slice": "month", "page_size": 500,
    },
    "GerritHub": {
        "base": "https://review.gerrithub.io",
        "auth": False, "slice": "month", "page_size": 500,
    },
    "LibreOffice": {
        "base": "https://gerrit.libreoffice.org",
        "auth": False, "slice": "month", "page_size": 500,
        # core は全期間を収集済み（2000-2024）。ローカルから数える
        "exclude": ["core"],
        "local": {"core": "libreoffice"},
    },
    "OpenAFS": {
        "base": "https://gerrit.openafs.org",
        "auth": False, "slice": "month", "page_size": 500,
        # 古い Gerrit で SKIP_DIFFSTAT に対応しない（HTTP 400）
        "skip_diffstat": False,
    },
    "OpenDaylight": {
        "base": "https://git.opendaylight.org/gerrit",
        "auth": False, "slice": "month", "page_size": 500,
    },
    "OpenStack": {
        "base": "https://review.opendev.org/a",
        "auth": True, "env": "GERRIT", "slice": "month", "page_size": 500,
    },
    "Qt": {
        # 匿名だと 1 ページ 10 件しか返らない。認証すれば一覧取得は n=500 で通る
        "base": "https://codereview.qt-project.org/a",
        "auth": True, "env": "QT_GERRIT", "slice": "month", "page_size": 500,
        # qtbase / qtcreator は全期間を収集済み。ローカルから数える
        "exclude": ["qt/qtbase", "qt-creator/qt-creator"],
        "local": {"qt/qtbase": "qtbase", "qt-creator/qt-creator": "qtcreator"},
    },
    "RockBox": {
        "base": "https://gerrit.rockbox.org/r",
        "auth": False, "slice": "month", "page_size": 500,
    },
    "Typo3": {
        "base": "https://review.typo3.org",
        "auth": False, "slice": "month", "page_size": 500,
    },
    "Whamcloud": {
        "base": "https://review.whamcloud.com",
        "auth": False, "slice": "month", "page_size": 500,
    },
    "Wikimedia": {
        # 頻度制限あり。3 秒間隔でも連続すると 403 になったため 8 秒空ける
        "base": "https://gerrit.wikimedia.org/r",
        "auth": False, "slice": "month", "page_size": 500, "delay": 8.0,
    },
}


# 母集団から外したインスタンスと、その理由（design.md §2.3）。
# 「到達できなかった」ではなく理由まで書けるものは理由を書く。
EXCLUDED_HOSTS: Dict[str, str] = {
    "GWT": "到達できるが対象期間の Change が 0 件",
    "Eclipse": "2023-11 に GerritHub へ移行し、自前の Gerrit を停止",
    "Kitware": "2015 に GitLab へ移行",
    "Vaadin": "GitHub へ移行",
    "oVirt": "2021-12 に GitHub へ移行開始。対象期間は移行後",
    "SciLab": "接続を拒否される（サービス停止）",
    "ONAP": "対象とした文献が査読前の投稿のみで、査読付きの裏づけがない",
    "STOQ": "サービス停止",
    "Cyanogen": "サービス停止",
    "AOKP": "サービス停止",
    "Gluster": "サービス停止",
    "Tuleap": "サービス停止",
    "OpenSwitch": "サービス停止",
}


# Wikimedia のロボットポリシーは User-Agent に連絡先を含めることを求める。
# 登録先はなく HTTP ヘッダに書くだけでよい。連絡先は .env から読み、
# リポジトリに直書きしない（design.md §6.2）。
_UA_TEMPLATE = "review-priority-research/1.0 ({contact})"


def user_agent() -> str:
    """全インスタンス共通の User-Agent。連絡先は .env の CONTACT から読む。"""
    load_dotenv()
    contact = os.getenv("CONTACT")
    if not contact:
        logger.warning(
            "CONTACT が .env にありません。Wikimedia は User-Agent に連絡先を求めるため、"
            "メールアドレスか URL を CONTACT に設定してください"
        )
        contact = "academic study of code review"
    return _UA_TEMPLATE.format(contact=contact)


def credentials(env_prefix: str) -> tuple[Optional[str], Optional[str]]:
    """そのインスタンス用の資格情報を .env から読む。

    Gerrit の HTTP パスワードはインスタンスごとに発行されるため、
    OpenStack の資格情報を Qt に送っても認証できない。
    """
    load_dotenv()
    return os.getenv(f"{env_prefix}_USERNAME"), os.getenv(f"{env_prefix}_PASSWORD")


def create_session(host: Dict[str, Any]) -> requests.Session:
    """そのインスタンス用の HTTP セッションを作る。"""
    session = requests.Session()
    headers = {"Accept": "application/json", "User-Agent": user_agent()}
    if host.get("auth"):
        user, password = credentials(host.get("env", "GERRIT"))
        if not user or not password:
            raise RuntimeError(
                f"{host['base']} は認証が要りますが、"
                f"{host.get('env', 'GERRIT')}_USERNAME / _PASSWORD が .env にありません"
            )
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    session.headers.update(headers)
    return session


def query_options(host: Dict[str, Any]) -> List[str]:
    """一覧取得で付けるオプション。古い Gerrit は SKIP_DIFFSTAT に対応しない。"""
    return ["SKIP_DIFFSTAT"] if host.get("skip_diffstat", True) else []


def excluded_repositories(host: Dict[str, Any]) -> List[str]:
    """走査から外すリポジトリ（既に全期間を収集済みのもの）。"""
    return list(host.get("exclude", []))


def local_directory(host: Dict[str, Any], repository: str) -> Optional[str]:
    """走査から外したリポジトリの、ローカルデータのディレクトリ名。"""
    return (host.get("local") or {}).get(repository)
