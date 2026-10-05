"""
変更リスト取得エンドポイント

OpenStack Gerritの変更リストを取得します。
"""

from typing import List, Dict, Any
from src.collectors.base.base_api_client import BaseAPIClient


def _time_value(value: str) -> str:
    """after: / before: に書く値。時刻付き（"2026-01-05 06:00:00"）なら引用符で囲む。

    1 日でも 1 万件の上限に達した区間を、時刻で割って取り直すときに使う
    （project_selection/design.md §6.6）。日付だけならそのまま。時刻は UTC として扱われる。
    """
    value = str(value)
    return f'"{value}"' if " " in value else value


class ChangesEndpoint(BaseAPIClient):
    """変更リスト取得エンドポイント"""

    # 通常の取得で要求するオプション（解析が実際に読むフィールドだけ）。
    # サーバはオプションごとにデータを組み立てるので、余計な指定は応答時間を延ばす。
    # Qt の Gerrit は RestApi.timeout=5000ms でサーバ側から中断されるため、ここは軽い方がよい。
    #   ALL_REVISIONS     revisions（created / files / _number）→ lines_added ほか
    #   ALL_COMMITS       revisions[x].commit.message → bug_fix / refactoring の判定
    #   ALL_FILES         revisions[x].files → test_code_presence / files_changed
    #   MESSAGES          messages（レビューコメントの時刻）→ 目的変数
    #   DETAILED_ACCOUNTS owner / messages[].author の email・name → 開発者特徴とボット判定
    # 外したもの: CURRENT_REVISION / CURRENT_COMMIT / CURRENT_FILES（ALL_* に含まれる）、
    #             DETAILED_LABELS / REVIEWED / SUBMITTABLE（解析側で 1 度も参照していない）。
    FULL_OPTIONS = ["ALL_REVISIONS", "ALL_COMMITS", "ALL_FILES", "MESSAGES", "DETAILED_ACCOUNTS"]

    # ファイル一覧を外したもの。巨大な Change（例: 6,861 ファイル × 18 リビジョン）は
    # ALL_FILES があると 5 秒制限を超えるため、本体だけ取ってファイルは別途埋める。
    NO_FILES_OPTIONS = ["ALL_REVISIONS", "ALL_COMMITS", "MESSAGES", "DETAILED_ACCOUNTS"]

    def get_endpoint_path(self, **kwargs) -> str:
        """エンドポイントパスを返す"""
        return "changes/"

    def fetch_light(self, component: str, start_date: str, end_date: str,
                    limit: int = 1, skip: int = 0,
                    gerrit_path: str = None) -> List[Dict[str, Any]]:
        """オプションなしで変更リストを取得する（どの Change がそこにあるかを知るためだけ）。

        通常の取得が失敗した位置の Change を特定するのに使う。オプションが無ければ
        サーバの組み立てが最小なので、巨大な Change でも失敗しない。
        """
        project = gerrit_path or f"openstack/{component}"
        return self.make_request(self.get_endpoint_path(), {
            "q": f"project:{project} after:{_time_value(start_date)} before:{_time_value(end_date)}",
            "n": limit, "S": skip,
        })

    def fetch_change(self, change_number: int, options: List[str] = None) -> Dict[str, Any]:
        """Change 1 件を番号で取得する（既定はファイル一覧を外した軽い取得）。"""
        opts = self.NO_FILES_OPTIONS if options is None else options
        return self.make_request(f"changes/{change_number}/detail", {"o": opts})

    def fetch_revision_files(self, change_number: int, revision) -> Dict[str, Any]:
        """1 リビジョンのファイル一覧を専用エンドポイントから取得する。

        `ALL_FILES` は全リビジョンぶんをまとめて組み立てるため巨大な Change で時間切れになるが、
        こちらは 1 リビジョンずつなので通る（実測: 6,861 ファイルで 2.7 秒）。
        返る形は `ALL_FILES` と同じ（lines_inserted / lines_deleted など）。
        """
        return self.make_request(f"changes/{change_number}/revisions/{revision}/files/")


    def fetch(self, component: str, start_date: str, end_date: str,
              limit: int = 100, skip: int = 0,
              gerrit_path: str = None) -> List[Dict[str, Any]]:
        """
        変更リストを取得

        Args:
            component: 保存キー（data/openstack_collected/<component>/）
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            limit: 取得件数
            skip: スキップ件数
            gerrit_path: Gerrit 上のプロジェクト名（例 "qt/qtbase"）。
                         省略時は従来どおり "openstack/<component>" とみなす

        Returns:
            変更リスト
        """
        project = gerrit_path or f"openstack/{component}"
        # after/before は Gerrit では「最終更新日」で絞られる（作成日ではない）
        query = f"project:{project} after:{_time_value(start_date)} before:{_time_value(end_date)}"
        
        params = {"q": query, "n": limit, "S": skip, "o": self.FULL_OPTIONS}
        return self.make_request(self.get_endpoint_path(), params)
