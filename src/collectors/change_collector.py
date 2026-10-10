"""
変更データ収集オーケストレーター

OpenStack Gerritから変更データを収集するメインクラスです。
"""

import os
import json
import base64
import argparse
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
import requests

from src.collectors.config.collector_config import CollectorConfig
from src.collectors.base.base_api_client import BaseAPIClient
from src.collectors.base.retry_handler import RetryConfig, ServerDeadlineExceeded
from src.utils.constants import GERRIT_PROJECTS

# n=1 でも通らなかったときだけ待って投げ直す間隔（秒）。この先の復旧処理が重いので、
# 一時的な混雑ならここで回復させる。n>1 のうちは待たずに n を下げる方が速い。
DEADLINE_WAITS = (3.0, 8.0)

# Gerrit が 1 クエリで返す件数の上限。**超えてもエラーにならず、静かに打ち切られる**
# （`_more_changes` が落ちて「終端」として返る）。n を増やしても上がらない。
# 制限があるのは Google の 2 ホスト（chromium-review / android-review）だけだが、
# 判定は全ホスト共通にしておく（割り直して同数なら本当にその件数なので無害）。
QUERY_LIMIT = 10000

# 取れなかったものを、全リポジトリの後にもう一度取り直す前に待つ秒数（design.md §6.6）。
# 一時的な混雑や通信断なら、少し待てば通ることが多い
RETRY_PASS_WAIT = 300.0

# 1 秒の区間まで時刻で割っても失敗したときに、1 ページの件数をこの順に減らす（design.md §6.6）。
# 2026-10-06 までは 1 日の区間で失敗した時点で減らしていたが、深いページを読みに行くことは変わらず、
# 1 クエリ 100 ページの上限で HTTP 400 になって取れなかった（2022-10-11）。今は先に時刻で割る。
# 時間切れ 1 回に 120 秒かかるので、半分ずつではなく大きく減らす（半分ずつだと最悪 8 回・16 分）。
# 1 件でも時間切れなら、その Change だけを本体とファイル一覧に分けて取る（_recover_change）
TIMEOUT_PAGE_SIZES = (50, 5, 1)

# 通信できないこと（名前解決の失敗・接続できない・接続を切られた）で区間が失敗したときは、
# 先へ進まずに通信が戻るまで待つ（design.md §6.6「通信できない間は待つ」）。
# 以前は切断中に 2.3 分に 1 区間ずつ「取れなかった」にして先へ進み、2026-10-05〜06 の
# 約 5.5 時間の切断で 164 区間（取り直しに 10 時間以上）を読み飛ばした。
CONNECTION_PROBE_INTERVAL = 60.0        # 接続先に軽い問い合わせを送る間隔（秒）。待っている間も相手に問い合わせるので空ける
CONNECTION_WAIT_MAX = 12 * 3600.0       # これだけ待っても戻らなければ、記録して次の区間へ進む（一晩の切断まで含める）
CONNECTION_RECOVERY_RETRIES = 3         # 通信が戻っても同じ区間が通信のことで失敗し続けるときの打ち切り回数
CONNECTION_WAIT_NOTE_INTERVAL = 600.0   # 待っている間、この間隔ごとに「待っている」とログに出す（毎回は出さない）

# 1 日でも上限に達したときの割り直しの段階（project_selection/design.md §6.6。第 1 段階の
# ranking.SUBDAY_STEPS と同じ）。1 時間 → 10 分 → 1 分 → 1 秒。1 秒でも達したら記録して先へ進む。
# 第 1 段階で chromium-review の 5 日が 1 日でも 1 万件を超えた（ボットの一斉更新など）ため（2026-10-04）
SUBDAY_STEPS = (timedelta(hours=1), timedelta(minutes=10), timedelta(minutes=1), timedelta(seconds=1))

# **全ホスト共通の既定の間隔（秒）。** 相手のサーバに負荷をかけないため、
# 1 リクエストごとに必ず空ける。GERRIT_PROJECTS の delay で上書きできる。
DEFAULT_REQUEST_DELAY = 1.0

# アクセス元を明示する User-Agent。連絡先は .env の CONTACT から読む。
# 研究目的のアクセスであることを相手のログに残すため。
USER_AGENT_TEMPLATE = "review-priority-research/1.0 ({contact})"
from src.collectors.storage.change_storage import ChangeStorage
from src.collectors.storage.commit_storage import CommitStorage
from src.collectors.storage.collection_manifest import CollectionManifest

# エンドポイントクラスのインポート
from src.collectors.endpoints.changes_endpoint import ChangesEndpoint
from src.collectors.endpoints.change_detail_endpoint import ChangeDetailEndpoint
from src.collectors.endpoints.included_in_endpoint import IncludedInEndpoint
from src.collectors.endpoints.comments_endpoint import CommentsEndpoint
from src.collectors.endpoints.reviewers_endpoint import ReviewersEndpoint
from src.collectors.endpoints.file_content_endpoint import FileContentEndpoint
from src.collectors.endpoints.file_diff_endpoint import FileDiffEndpoint
from src.collectors.endpoints.commit_endpoint import CommitEndpoint
from src.collectors.endpoints.commit_parents_endpoint import CommitParentsEndpoint

logger = logging.getLogger(__name__)


# 終了日が未来でも、切り詰めずにそのまま問い合わせる（2026-10-03 に変更。選定の design.md §6.3）。
# 以前は clamp_end_date で「明日（UTC）」に切り詰めていたが、収集が日をまたぐと
# 終わりのほうで更新された Change が範囲の外に出る。未来の期間の空の問い合わせが増えるのは許容する。
# 代わりに、今年を含む区間の完了マーカーの名前が固定になる（例 2026-01-01_2027-01-01）ので、
# 後日もう一度実行しても、その区間は取り直されない。


def _is_connection_error(e: BaseException) -> bool:
    """通信できないことによる失敗か（名前解決の失敗・接続できない・接続を切られた）。

    requests の ConnectionError（接続の時間切れ ConnectTimeout を含む）で判定する。
    読み込みの時間切れ（ReadTimeout）は含めない。これはサーバの処理が重いことによるもので、
    待っても速くならないので、従来どおり区間を割って取り直す（design.md §6.6）。
    包み直された例外にも対応するため、原因（__cause__ / __context__）もたどる。
    """
    seen = set()
    while e is not None and id(e) not in seen:
        if isinstance(e, requests.exceptions.ConnectionError):
            return True
        seen.add(id(e))
        e = e.__cause__ or e.__context__
    return False


def date_chunks(start_date: str, end_date: str, years) -> List[Tuple[str, str]]:
    """[start_date, end_date] を years 年ごとの (start, end) 区間に分割する。

    years が None / 0 のときは分割せず [(start_date, end_date)] を返す（従来動作）。
    区間は隙間なく連続し、境界での二重取得は change_number キーで上書きされるため無害。
    """
    if not years:
        return [(start_date, end_date)]
    ys, ye = int(str(start_date)[:4]), int(str(end_date)[:4])
    chunks: List[Tuple[str, str]] = []
    y = ys
    while y <= ye:
        cs = start_date if y == ys else f"{y}-01-01"
        top = y + years
        ce = end_date if top > ye else f"{top}-01-01"
        chunks.append((cs, ce))
        y = top
    return chunks


def day_chunks(start_date: str, end_date: str, days: int) -> List[Tuple[str, str]]:
    """[start_date, end_date) を days 日ごとの (start, end) 区間に分割する。

    chunk_days を持つリポジトリ（件数の多いもの）は、この区間ごとに保存して完了マーカーを付ける
    （project_selection/design.md §6.6）。1 年ごとだと、途中で止まったときにその年の取得済みぶんを
    すべて失う（2026-10-04、chromium/src の 2022 年ぶん 1 時間 20 分を失った）。
    """
    s, e = date.fromisoformat(str(start_date)[:10]), date.fromisoformat(str(end_date)[:10])
    chunks: List[Tuple[str, str]] = []
    cur = s
    while cur < e:
        nxt = min(cur + timedelta(days=int(days)), e)
        chunks.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt
    return chunks


class ChangeCollector:
    """OpenStack Gerrit変更データ収集オーケストレーター"""
    
    # エンドポイントクラスマッピング
    ENDPOINT_CLASSES = {
        'ChangesEndpoint': ChangesEndpoint,
        'ChangeDetailEndpoint': ChangeDetailEndpoint,
        'IncludedInEndpoint': IncludedInEndpoint,
        'CommentsEndpoint': CommentsEndpoint,
        'ReviewersEndpoint': ReviewersEndpoint,
        'FileContentEndpoint': FileContentEndpoint,
        'FileDiffEndpoint': FileDiffEndpoint,
        'CommitEndpoint': CommitEndpoint,
        'CommitParentsEndpoint': CommitParentsEndpoint,
    }
    
    def __init__(self, config_path: Optional[Path] = None, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルパス
            username: Gerritユーザー名
            password: Gerritパスワード
        """
        load_dotenv()
        
        # 設定読み込み
        self.config = CollectorConfig(config_path)
        
        # 認証情報
        self.username = username or os.getenv("GERRIT_USERNAME")
        self.password = password or os.getenv("GERRIT_PASSWORD")
        
        # セッションとエンドポイントは収集対象ごとに作り直す（ホスト・認証が変わるため）。
        # collect_component の冒頭で _switch_to が設定する。
        self.session = None
        self.endpoints: Dict[str, Any] = {}

        # どうしても取得できずに飛ばした Change（マニフェストに記録し、終了時に警告する）。
        # 取れない原因は毎回違いうるので、1 件のために収集全体を止めない方針にしている。
        self._skipped_changes: List[Dict[str, Any]] = []

        # 1 リクエスト後に空ける秒数。収集対象ごとに _switch_to が設定する
        self._request_delay: float = 0.0

        # リトライ設定
        retry_config_dict = self.config.get_retry_config()
        self.retry_config = RetryConfig(**retry_config_dict)
        
        # ストレージ初期化
        storage_config = self.config.get_storage_config()
        self.change_storage = ChangeStorage(Path(storage_config['output_dir']))
        self.commit_storage = CommitStorage(Path(storage_config['output_dir']))

        # diff/commit を取る revision 範囲: "all"=全 revision / "first"=投稿時点(patch set 1)のみ
        self.revision_scope = self.config.get_collection_config().get("revision_scope", "all")

        logger.info(f"ChangeCollector初期化完了（revision_scope={self.revision_scope}）")
    
    def _credentials(self, env_prefix: str = "GERRIT") -> Tuple[Optional[str], Optional[str]]:
        """そのホスト用の資格情報を .env から読む。

        Gerrit の HTTP パスワードは**インスタンスごとに発行される**ので、OpenStack の資格情報を
        Qt に送っても認証できない。接頭辞で使い分ける
        （GERRIT_* = OpenStack / QT_GERRIT_* = Qt）。
        コンストラクタで明示指定された場合は既定の接頭辞のときだけそちらを優先する。
        """
        if env_prefix == "GERRIT" and self.username and self.password:
            return self.username, self.password
        return os.getenv(f"{env_prefix}_USERNAME"), os.getenv(f"{env_prefix}_PASSWORD")

    def _create_session(self, auth: bool = True, env_prefix: str = "GERRIT") -> requests.Session:
        """HTTPセッションを作成（auth=False なら Authorization ヘッダを付けない）。"""
        session = requests.Session()
        contact = os.getenv("CONTACT") or "academic study of code review"
        headers = {"Accept": "application/json",
                   "User-Agent": USER_AGENT_TEMPLATE.format(contact=contact)}

        if auth:
            user, pwd = self._credentials(env_prefix)
            if not (user and pwd):
                raise ValueError(f"認証が必要なホストですが "
                                 f"{env_prefix}_USERNAME / {env_prefix}_PASSWORD が未設定です")
            auth_string = base64.b64encode(f"{user}:{pwd}".encode()).decode()
            headers["Authorization"] = f"Basic {auth_string}"

        session.headers.update(headers)
        return session

    def _project_spec(self, component: str) -> Dict[str, Any]:
        """収集対象の接続情報を返す（未登録なら従来どおり OpenStack 扱い）。"""
        spec = GERRIT_PROJECTS.get(component)
        if spec is None:
            logger.warning(f"未登録のプロジェクト '{component}' → OpenStack として扱います")
            return {"host": BaseAPIClient.BASE_URL, "path": f"openstack/{component}", "auth": True}
        return spec

    def _initialize_endpoints(self, base_url: str = None, session: requests.Session = None,
                              auth: bool = True, env_prefix: str = "GERRIT") -> Dict[str, Any]:
        """エンドポイントインスタンスを初期化（接続先ごとに作り直す）。"""
        endpoints = {}
        enabled_endpoints = self.config.get_enabled_endpoints()
        session = session if session is not None else self.session
        user, pwd = self._credentials(env_prefix) if auth else (None, None)

        for endpoint_info in enabled_endpoints:
            name = endpoint_info['name']
            endpoint_config = endpoint_info['config']
            class_name = endpoint_config['class']

            if class_name in self.ENDPOINT_CLASSES:
                endpoint_class = self.ENDPOINT_CLASSES[class_name]
                endpoints[name] = endpoint_class(
                    username=user,
                    password=pwd,
                    session=session,
                    timeout=(30, 120),
                    base_url=base_url,
                )
                logger.info(f"エンドポイント初期化: {name} ({class_name})")
            else:
                logger.warning(f"未知のエンドポイントクラス: {class_name}")

        return endpoints

    def _switch_to(self, component: str) -> Dict[str, Any]:
        """収集対象を切り替える（ホスト・認証・エンドポイントを作り直す）。"""
        spec = self._project_spec(component)
        auth = bool(spec.get("auth", True))
        env_prefix = spec.get("env", "GERRIT")
        self.session = self._create_session(auth=auth, env_prefix=env_prefix)
        self.endpoints = self._initialize_endpoints(spec["host"], self.session, auth, env_prefix)
        # アクセス頻度に制限のあるインスタンス用（GERRIT_PROJECTS の delay）
        self._request_delay = float(spec.get("delay", DEFAULT_REQUEST_DELAY))
        logger.info(f"接続先: {spec['host']} / project:{spec['path']}"
                    f"（認証{'あり（' + env_prefix + '_*）' if auth else 'なし'}"
                    + (f"・間隔 {self._request_delay} 秒" if self._request_delay else "") + "）")
        return spec
    
    def collect_all_components(self) -> str:
        """全コンポーネントのデータを収集し、実行全体の状態（"completed" / "incomplete"）を返す。

        呼び出し元は、この状態に合わせて最後の 1 行を出す（design.md §6.6「終了時の表示」）。
        """
        collection_config = self.config.get_collection_config()
        components = collection_config['components']

        logger.info(f"データ収集開始: {len(components)}コンポーネント")

        # 収集マニフェスト（何を・いつからいつまで・全部取れたかを後から確認できるよう記録）
        manifest = CollectionManifest(
            self.change_storage.output_dir, collector="changes",
            requested={
                "start_date": collection_config.get('start_date'),
                "end_date": collection_config.get('end_date'),
                "components": list(components),
            },
        )
        try:
            # **1 つのリポジトリで失敗しても、残りは必ず取る**（design.md §6.6）。
            # 以前は例外が上に抜けて、残りのリポジトリを取らずに止まっていた（2026-10-04）
            for component in components:
                self._collect_component_safely(component, manifest)

            # 取れなかったものがあるリポジトリは、最後にもう一度だけ取り直す。
            # 完了マーカーの付いた区間は飛ばされるので、取り直すのは取れなかった区間だけ
            failed = [c for c in components
                      if any(d.get("component") == c for d in self._skipped_changes)]
            if failed:
                logger.warning(f"取れなかったものがある {len(failed)} 件を、{RETRY_PASS_WAIT:.0f} 秒待ってから"
                               f"取り直します: {', '.join(failed)}")
                time.sleep(RETRY_PASS_WAIT)
                for component in failed:
                    self._skipped_changes = [d for d in self._skipped_changes
                                             if d.get("component") != component]
                    self._collect_component_safely(component, manifest)

            remaining = sorted({d.get("component") for d in self._skipped_changes})
            if remaining:
                # 黙って抜けたままにしない。どこが取れていないかと、取り直す方法を必ず出す
                manifest.finish("incomplete")
                logger.error(f"**取り直しても取れなかったものがあります: {', '.join(remaining)}**。"
                             f"collection_manifest.jsonl の dropped_detail に位置があります。"
                             f"同じコマンドをもう一度実行すると、完了マーカーの無い区間だけを取り直します")
                return "incomplete"
            manifest.finish("completed")
            logger.info("全コンポーネントのデータ収集完了（取れなかったものはありません）")
            return "completed"
        except BaseException as e:
            # 途中で止まった/落ちた場合も「どこまで取れたか」を必ず残す
            manifest.finish("interrupted", error=f"{type(e).__name__}: {e}")
            raise

    def _collect_component_safely(self, component: str, manifest: CollectionManifest) -> None:
        """collect_component を呼び、例外が出ても記録して戻る（Ctrl+C などの中断だけは上に通す）。"""
        try:
            self.collect_component(component, manifest=manifest)
        except Exception as e:
            self._skipped_changes.append(
                {"component": component, "range": "（リポジトリ全体）", "offset": -1,
                 "reason": f"{type(e).__name__}: {e}"})
            logger.error(f"[{component}] 収集に失敗しました（{type(e).__name__}: {e}）。"
                         f"記録して次のリポジトリへ進みます")

    def collect_component(self, component: str, manifest: CollectionManifest = None):
        """特定コンポーネントのデータを収集（checkpoint_years ごとに途中保存＋レジューム）。"""
        logger.info(f"{component} の収集開始")

        # 接続先（ホスト・認証・Gerrit 上のプロジェクト名）をこの対象に合わせる
        spec = self._switch_to(component)
        gerrit_path = spec["path"]

        cfg = self.config.get_collection_config()
        # 要求件数はプロジェクト別の指定を優先する（Qt は 10 を超えると実行時間制限に触れる）
        batch_size = int(spec.get("batch_size") or cfg['batch_size'])
        if batch_size != cfg['batch_size']:
            logger.info(f"[{component}] 要求件数を {batch_size} に設定（既定 {cfg['batch_size']}）")
        end_date = str(cfg['end_date'])[:10]
        if spec.get("chunk_days"):
            # 件数の多いリポジトリは chunk_days ごとに保存・完了マーカー（design.md §6.6）
            chunks = day_chunks(cfg['start_date'], end_date, int(spec["chunk_days"]))
            chunked = True
        else:
            chunks = date_chunks(cfg['start_date'], end_date, cfg.get('checkpoint_years'))
            chunked = bool(cfg.get('checkpoint_years'))

        change_rows: List[Dict[str, Any]] = []  # summary 用の軽量行（全チャンク分を蓄積）
        total_saved = 0
        total_skipped = 0
        total_commits = 0
        created_all: List[str] = []
        status = "completed"
        error = None

        try:
            for cs, ce in chunks:
                marker = self._chunk_marker(component, cs, ce)
                if chunked and marker.exists():
                    logger.info(f"[{component}] 区間 {cs}〜{ce} は保存済み → スキップ（レジューム）")
                    continue

                drops_before = len(self._skipped_changes)
                try:
                    # 通信できないことで失敗したら、先へ進まずに戻るまで待って取り直す（design.md §6.6）
                    changes, commits, skipped = self._collect_chunk_waiting(
                        component, cs, ce, batch_size, gerrit_path, spec.get("chunk_days"))
                except Exception as e:
                    # 区間ごと取れなかった（通信が戻らない・戻っても失敗し続けた等）。**収集全体は止めない**。
                    # 記録して次の区間へ進み、完了マーカーは付けない（再実行でこの区間だけ取り直す）。
                    # 以前はここで例外を上に投げ、残りの区間・リポジトリを取らずに止まっていた
                    # （2026-10-04、chromium/src で発生。design.md §6.6）
                    self._skipped_changes.append(
                        {"component": component, "range": f"{cs}〜{ce}", "offset": -1,
                         "reason": f"区間ごと失敗: {type(e).__name__}: {e}"})
                    logger.error(f"[{component}] 区間 {cs}〜{ce} を取得できませんでした（{type(e).__name__}）。"
                                 f"記録して次の区間へ進みます（完了マーカーは付けません）")
                    continue

                # 途中保存: この区間ぶんを即ディスクへ（クラッシュしてもここまでは残る）
                change_rows.extend(self.change_storage.save_changes(component, changes))
                self.commit_storage.save_commits(component, commits)

                total_saved += len(changes)
                total_skipped += skipped
                total_commits += len(commits)
                created_all.extend(c.get("created") for c in changes if c.get("created"))
                if chunked:
                    if len(self._skipped_changes) > drops_before:
                        # 取れなかった Change や区間がある。**完了マーカーを付けない**ので、
                        # 再実行するとこの区間を取り直す（以前は付けてしまい、抜けたまま飛ばされていた）
                        logger.warning(f"[{component}] 区間 {cs}〜{ce} 保存: {len(changes)}変更。"
                                       f"取れなかったものが {len(self._skipped_changes) - drops_before} 件あるので、"
                                       f"完了マーカーは付けません（再実行で取り直します）")
                    else:
                        self._write_chunk_marker(marker, len(changes), len(commits))
                        logger.info(f"[{component}] 区間 {cs}〜{ce} 保存: {len(changes)}変更 / {len(commits)}コミット")
        except BaseException as e:
            status = "partial"
            error = f"{type(e).__name__}: {e}"
            logger.error(f"{component} の収集が中断しました: {error}")
            raise
        finally:
            # サマリー（全チャンク分をまとめて）とマニフェストを書く
            self.change_storage.write_summary(component, change_rows)
            self.commit_storage.write_summary(component, total_commits)
            # 取得できずに飛ばした Change（この対象ぶん）をマニフェストに残す
            dropped = [d for d in self._skipped_changes if d.get("component") == component]
            if status == "completed" and dropped:
                status = "incomplete"  # 取れなかったものがある（完了マーカーの無い区間が残っている）
            if manifest is not None:
                manifest.record_component(
                    component, status=status,
                    fetched=total_saved + total_skipped,
                    saved=total_saved, skipped=total_skipped,
                    earliest=min(created_all) if created_all else None,
                    latest=max(created_all) if created_all else None,
                    error=error,
                    extra={"commits_saved": total_commits,
                           "dropped_changes": len(dropped),
                           # 全件残す（以前は先頭 50 件だけで、それ以降は記録から消えていた）
                           "dropped_detail": dropped},
                )
            logger.info(
                f"{component} の収集{('完了' if status == 'completed' else '中断')}: "
                f"{total_saved}変更, {total_commits}コミット, スキップ {total_skipped}件"
            )
            if dropped:
                # 黙って欠けたまま解析へ進むのを防ぐ
                logger.warning(f"[{component}] **取得できずに飛ばした Change が {len(dropped)} 件あります**。"
                               f"collection_manifest.jsonl の dropped_detail で位置を確認できます")

    def _collect_chunk_waiting(self, component: str, start_date: str, end_date: str,
                               batch_size: int, gerrit_path: str, chunk_days: Optional[int]
                               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """_collect_chunk を呼び、通信できないことで失敗したら、戻るまで待って同じ区間を取り直す。

        design.md §6.6「通信できない間は待つ」。待つのは通信できないときだけで、時間切れ・HTTP エラー・
        上限は _collect_chunk の中でこれまでどおり区間を割って取り直す。
        待っても戻らない、または戻っても同じ区間が通信のことで CONNECTION_RECOVERY_RETRIES 回失敗したら、
        最後の例外をそのまま投げる（呼び出し元が記録して次の区間へ進み、完了マーカーは付かない）。
        """
        drops_before = len(self._skipped_changes)
        recoveries = 0
        while True:
            try:
                return self._collect_chunk(component, start_date, end_date,
                                           batch_size, gerrit_path, chunk_days)
            except Exception as e:
                if not _is_connection_error(e) or recoveries >= CONNECTION_RECOVERY_RETRIES:
                    raise
                logger.error(f"[{component}] 区間 {start_date}〜{end_date} で通信できません"
                             f"（{type(e).__name__}）。先へ進まずに、通信が戻るまで待ちます")
                if not self._wait_for_connection(component):
                    raise
                recoveries += 1
                # 途中まで取れていた分は捨てて、区間の最初から取り直す（上書き保存なので重複しない）。
                # その間に記録した「取れなかったもの」も、取り直しで改めて記録されるので消しておく
                del self._skipped_changes[drops_before:]
                logger.warning(f"[{component}] 区間 {start_date}〜{end_date} を最初から取り直します"
                               f"（通信が戻った後の取り直し {recoveries}/{CONNECTION_RECOVERY_RETRIES}）")

    def _wait_for_connection(self, component: str) -> bool:
        """接続先のサーバに軽い問い合わせを送り続け、応答が返ったら True を返す。

        CONNECTION_PROBE_INTERVAL ごとに /config/server/version を 1 回だけ問い合わせる
        （投げ直しの仕組みは通さない）。5xx 以外の応答が返れば通信は戻ったとみなす。
        CONNECTION_WAIT_MAX 待っても戻らなければ False（呼び出し元が記録して先へ進む）。
        待ち時間は time.monotonic で測るので、パソコンが眠っていた時間は数えない。
        """
        url = f"{self.endpoints['changes'].base_url}/config/server/version"
        start = time.monotonic()
        last_note = start
        while time.monotonic() - start < CONNECTION_WAIT_MAX:
            time.sleep(CONNECTION_PROBE_INTERVAL)
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code < 500:
                    logger.warning(f"[{component}] 通信が戻りました"
                                   f"（{(time.monotonic() - start) / 60:.0f} 分待ちました）")
                    return True
            except requests.exceptions.RequestException:
                pass  # まだ戻っていない。毎回はログに出さない（下で間隔を空けて出す）
            now = time.monotonic()
            if now - last_note >= CONNECTION_WAIT_NOTE_INTERVAL:
                logger.warning(f"[{component}] 通信が戻るのを待っています"
                               f"（{(now - start) / 60:.0f} 分経過。最大 {CONNECTION_WAIT_MAX / 3600:.0f} 時間）")
                last_note = now
        logger.error(f"[{component}] {CONNECTION_WAIT_MAX / 3600:.0f} 時間待っても通信が戻りません。"
                     f"この区間は記録して次へ進みます")
        return False

    def _collect_chunk(self, component: str, start_date: str, end_date: str,
                       batch_size: int, gerrit_path: str, chunk_days: Optional[int]
                       ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """完了マーカー 1 つぶんの区間を取得する。chunk_days があれば最初からその日数で割る。

        **件数の多いリポジトリでは、最初から区間を小さくして問い合わせる。**
        1 年ぶん（chromium/src で約 21 万件）を一度に問い合わせると、サーバの応答が
        120 秒を超えてタイムアウトする。タイムアウトすれば区間を割って取り直す仕組みは
        あるが（_collect_range_adaptive）、それに頼ると「1 年 → 半年 → 3 か月 …」と
        **小さくなるまで重いクエリを何度も相手のサーバに投げる**ことになる。
        件数は第 1 段階のランキングで分かっているので、最初から適切な大きさで始める。

        完了マーカーは従来どおり checkpoint_years 単位のまま（意味を変えない）。
        """
        if not chunk_days:
            return self._collect_range_adaptive(component, start_date, end_date,
                                                batch_size, gerrit_path=gerrit_path)
        s, e = date.fromisoformat(start_date), date.fromisoformat(end_date)
        changes: List[Dict[str, Any]] = []
        commits: List[Dict[str, Any]] = []
        skipped = 0
        cur = s
        while cur < e:
            nxt = min(cur + timedelta(days=int(chunk_days)), e)
            ch, cm, sk = self._collect_range_adaptive(component, cur.isoformat(), nxt.isoformat(),
                                                      batch_size, gerrit_path=gerrit_path)
            changes += ch; commits += cm; skipped += sk
            cur = nxt
        return changes, commits, skipped

    def _collect_range_adaptive(self, component: str, start_date: str, end_date: str,
                                batch_size: int, gerrit_path: str = None,
                                min_days: int = 1
                                ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """区間を取得する。サーバ側の制限で失敗したら**期間を半分に割って**取り直す。

        Gerrit は S（オフセット）を処理するのに先頭から走査するため、件数の多い区間では
        深いページほど時間がかかり、サーバ側の実行時間制限（Qt は RestApi.timeout=5000ms）に
        かかって 500 が返る。期間を割れば 1 区間あたりの件数が減り、オフセットが浅くなって収まる。

        区間は半開区間 [start, end) なので、割った 2 つを合わせても重複・欠落は生じない
        （境界で重複しても change_number キーで上書き保存されるため無害）。
        """
        try:
            got = self._collect_range(component, start_date, end_date, batch_size,
                                      gerrit_path=gerrit_path)
            # **Gerrit は 1 クエリ QUERY_LIMIT 件で打ち切る。しかもエラーにならず
            # `_more_changes` が落ちて「終端」として返るため、取得側からは
            # 「ちょうど取り終わった」のか「打ち切られた」のかが区別できない。**
            # 疑わしきは区間を割って取り直す（割り直して同数なら本当にその件数）。
            # 見逃すと欠損したまま完了マーカーが書かれる（2026-10-01 に chromium/src で発生）。
            if len(got[0]) + got[2] >= QUERY_LIMIT:
                s, t = date.fromisoformat(start_date), date.fromisoformat(end_date)
                if (t - s).days > min_days:
                    mid = (s + timedelta(days=(t - s).days // 2)).isoformat()
                    logger.warning(f"[{component}] 区間 {start_date}〜{end_date} が上限"
                                   f"（{QUERY_LIMIT:,}）に達しました。打ち切られた可能性があるため "
                                   f"{start_date}〜{mid} と {mid}〜{end_date} に割って取り直します")
                    a = self._collect_range_adaptive(component, start_date, mid,
                                                     batch_size, gerrit_path, min_days)
                    b = self._collect_range_adaptive(component, mid, end_date,
                                                     batch_size, gerrit_path, min_days)
                    return a[0] + b[0], a[1] + b[1], a[2] + b[2]
                # 1 日でも上限に達した。時刻で割り直す（project_selection/design.md §6.6。2026-10-04 追加）。
                # 以前はここで「これ以上割れない」として記録し、1 万件だけを残して先へ進んでいた
                logger.warning(f"[{component}] 区間 {start_date}〜{end_date} が 1 日でも上限"
                               f"（{QUERY_LIMIT:,}）に達しました。時刻で割って取り直します")
                return self._collect_subday(component, datetime.combine(s, datetime.min.time()),
                                            datetime.combine(t, datetime.min.time()),
                                            batch_size, gerrit_path, SUBDAY_STEPS)
            return got
        except (ServerDeadlineExceeded, requests.exceptions.HTTPError,
                requests.exceptions.ReadTimeout, ValueError) as e:
            # ReadTimeout も割る対象にする。該当件数の多い区間はサーバの応答が遅くなるため、
            # 区間を割って件数を減らせば応答が速くなる（retry_handler は投げ直さずに返してくる）
            s, t = date.fromisoformat(start_date), date.fromisoformat(end_date)
            span = (t - s).days
            if span <= min_days:
                # 1 日でも失敗した。**時刻で割って**取り直す（1 時間 → 10 分 → 1 分 → 1 秒。design.md §6.6）。
                # 1 ページの件数を減らすのは、1 秒の区間でも失敗したときだけ（_collect_subday の中）。
                # 2026-10-06 までは、ここで件数を減らしていた。しかし深いページを読みに行くことは変わらず、
                # 1 クエリ 100 ページの上限で HTTP 400 になり、2022-10-11 が取れずに残った。
                # （さらに前は例外を上に投げ、収集全体が止まっていた。2026-10-04、chromium/src の 2022-04-22）
                logger.warning(f"[{component}] 区間 {start_date}〜{end_date} は 1 日でも失敗しました"
                               f"（{type(e).__name__}）。時刻で割って取り直します")
                return self._collect_subday(component, datetime.combine(s, datetime.min.time()),
                                            datetime.combine(t, datetime.min.time()),
                                            batch_size, gerrit_path, SUBDAY_STEPS)
            mid = (s + timedelta(days=span // 2)).isoformat()
            logger.warning(f"[{component}] 区間 {start_date}〜{end_date}（{span}日）で失敗 → "
                           f"{start_date}〜{mid} と {mid}〜{end_date} に割って取り直します: "
                           f"{type(e).__name__}")
            ch_a, cm_a, sk_a = self._collect_range_adaptive(component, start_date, mid,
                                                           batch_size, gerrit_path, min_days)
            ch_b, cm_b, sk_b = self._collect_range_adaptive(component, mid, end_date,
                                                           batch_size, gerrit_path, min_days)
            return ch_a + ch_b, cm_a + cm_b, sk_a + sk_b

    def _collect_subday(self, component: str, start: datetime, end: datetime,
                        batch_size: int, gerrit_path: str, steps: Tuple[timedelta, ...]
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """[start, end) を steps[0] ごとに時刻で区切って取得する（1 日でも上限に達した・失敗したとき）。

        時刻は UTC で、問い合わせでは after:"2026-01-05 06:00:00" のように引用符で囲む
        （ChangesEndpoint が行う）。1 区間が上限に達したら、その区間だけ steps の次の段階で割り直す。
        最後の段階（1 秒）でも達したら、欠損の可能性として記録して先へ進む。
        **1 区間が時間切れ・HTTP エラー・壊れた応答で失敗したときも、その区間だけ次の段階で割り直す**
        （2026-10-06 追加。design.md §6.6）。1 ページの件数を減らすのは最後の段階（1 秒）の区間だけ。
        途中まで取れていた分は捨てて割り直した区間で取り直す（上書き保存なので重複しない）。
        境目は両方の区間に含まれうるが、change_number をキーに上書き保存されるので重複しない。
        """
        changes: List[Dict[str, Any]] = []
        commits: List[Dict[str, Any]] = []
        skipped = 0
        cursor = start
        while cursor < end:
            nxt = min(cursor + steps[0], end)
            a, b = f"{cursor:%Y-%m-%d %H:%M:%S}", f"{nxt:%Y-%m-%d %H:%M:%S}"
            last = len(steps) == 1
            try:
                # 最後の段階（1 秒）の区間だけ、失敗したら 1 ページの件数を減らす（design.md §6.6）。
                # それより粗い区間は例外を受けて、下で次の段階に割る
                got = self._collect_range(component, a, b, batch_size, gerrit_path=gerrit_path,
                                          reduce_on_failure=last)
            except (ServerDeadlineExceeded, requests.exceptions.HTTPError,
                    requests.exceptions.ReadTimeout, ValueError) as e:
                if last:
                    # 1 秒の区間は _collect_range の中で件数を減らして扱うので、ここには来ない想定。
                    # 来た場合は割りようがないので、上に投げる（区間ごと失敗として記録される）
                    raise
                logger.warning(f"[{component}] 区間 {a}〜{b} で失敗しました（{type(e).__name__}）。"
                               f"{steps[1]} ごとに割って取り直します")
                got = self._collect_subday(component, cursor, nxt, batch_size, gerrit_path, steps[1:])
                changes += got[0]; commits += got[1]; skipped += got[2]
                cursor = nxt
                continue
            if len(got[0]) + got[2] >= QUERY_LIMIT:
                if len(steps) > 1:
                    logger.warning(f"[{component}] 区間 {a}〜{b} が上限に達しました。"
                                   f"{steps[1]} ごとに割って取り直します")
                    got = self._collect_subday(component, cursor, nxt, batch_size, gerrit_path, steps[1:])
                else:
                    logger.error(f"[{component}] 区間 {a}〜{b} が上限に達しましたが、1 秒より細かく割れません。"
                                 f"**この区間は欠損している可能性があります**")
                    self._skipped_changes.append(
                        {"component": component, "range": f"{a}〜{b}",
                         "offset": -1, "reason": f"query_limit_{QUERY_LIMIT}"})
            changes += got[0]; commits += got[1]; skipped += got[2]
            cursor = nxt
        return changes, commits, skipped

    def _recover_change(self, component: str, start_date: str, end_date: str, offset: int,
                        gerrit_path: str = None) -> Optional[Dict[str, Any]]:
        """1 件でも通らない Change を、ファイル一覧を分けて取り直す。

        `ALL_FILES` は**全リビジョンぶんのファイル一覧をまとめて**組み立てるため、巨大な Change
        （実例: qtbase #104705「Update copyright headers」＝ 6,861 ファイル × 18 リビジョン）では
        サーバ側の 5 秒制限を超える。件数を 1 まで減らしても期間を 1 日まで割っても解決しない。

        そこで 3 段階に分ける。
          1. オプションなしで「その位置に居る Change」を特定する（最も軽い）
          2. ファイル一覧を外して本体を取る（リビジョン一覧は入る）
          3. リビジョンごとに専用エンドポイントでファイル一覧を取り、本体に埋め戻す

        返る中身は通常の取得と同じ形になるので、**解析側は何も変わらない**。
        取り直せなかったときは None（呼び出し元がその 1 件を飛ばして記録する）。
        """
        ep = self.endpoints['changes']
        try:
            head = ep.fetch_light(component=component, start_date=start_date,
                                  end_date=end_date, limit=1, skip=offset,
                                  gerrit_path=gerrit_path)
        except Exception as e:
            logger.error(f"[{component}] S={offset} の Change を特定できません: {type(e).__name__}")
            return None
        if not head:
            return None

        number = head[0].get("_number")
        logger.warning(f"[{component}] S={offset} の Change #{number} は通常の取得では重すぎます → "
                       f"ファイル一覧を分けて取り直します")
        try:
            change = ep.fetch_change(number)          # ファイル一覧なしの本体
        except Exception as e:
            logger.error(f"[{component}] #{number} の本体を取得できません: {type(e).__name__}")
            return None

        revisions = change.get("revisions") or {}
        for sha, meta in revisions.items():
            rev_no = meta.get("_number", sha)
            try:
                meta["files"] = ep.fetch_revision_files(number, rev_no)
            except Exception as e:
                # 1 つの版でもファイル一覧が取れなければ、この Change は取れなかったものとして扱う。
                # 以前はその版のファイル一覧を空にして保存しており、行数・ファイル数などの特徴量が
                # 黙って誤った値になっていた（2026-10-04 に修正。呼び出し元が記録して完了マーカーを付けない）
                logger.error(f"[{component}] #{number} rev{rev_no} のファイル一覧を取得できません"
                             f"（{type(e).__name__}）。この Change は取れなかったものとして記録します")
                return None
        logger.info(f"[{component}] #{number} を復旧しました（リビジョン {len(revisions)} 個すべての"
                    f"ファイル一覧を取得）")
        return change

    def _collect_range(self, component: str, start_date: str, end_date: str,
                       batch_size: int, gerrit_path: str = None,
                       reduce_on_failure: bool = False
                       ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """1 つの日付区間の変更・コミットを取得して返す（(changes, commits, skipped)）。

        reduce_on_failure=True（1 秒の時刻の区間など、期間ではもう割らないとき。2026-10-06 からは
        1 日の区間ではなく、時刻で 1 秒まで割った区間だけで使う）は、
        応答の時間切れ・HTTP エラー・壊れた応答で例外を投げず、1 ページの件数を
        TIMEOUT_PAGE_SIZES の順に減らして取り直す。1 件でも失敗したら、その Change だけ
        本体とファイル一覧に分けて取る（_recover_change）。それでも駄目なら記録して次へ進む
        （design.md §6.6）。
        """
        skip = 0
        changes_out: List[Dict[str, Any]] = []
        commits_out: List[Dict[str, Any]] = []
        skipped = 0
        if not self.config.is_endpoint_enabled('changes'):
            logger.warning("changesエンドポイントが無効です")
            return changes_out, commits_out, skipped

        # 要求件数。まず既定値で投げ、短い間隔での投げ直し（retry_handler）でも通らないときだけ
        # 半分に落とす。**ページごとに既定値へ戻す**のが重要で、戻さないと 1 回の一時的な混雑で
        # 以降ずっと小さい n のまま走り続けることになる（qtcreator が n=6 まで落ちて遅くなった原因）。
        n = batch_size

        while True:
            # サーバ側の上限で limit より少なく返ることがある（Qt は匿名だと 1 ページ 10 件）。
            # 返った件数ぶんだけ skip を進めるので、そのまま正しくページングできる。
            deadline_waits = 0
            while True:
                try:
                    changes = self.endpoints['changes'].fetch(
                        component=component, start_date=start_date, end_date=end_date,
                        limit=n, skip=skip, gerrit_path=gerrit_path,
                    )
                    # アクセス頻度に制限のあるインスタンスでは間隔を空ける。
                    # 制限に掛かってから再試行の待ち時間に入るより、最初から空けるほうが速い。
                    if self._request_delay:
                        time.sleep(self._request_delay)
                    break
                except ServerDeadlineExceeded:
                    if n > 1:
                        # **待たずに** n を半分にして即座に出し直す。
                        # 一時的な混雑なら軽い要求で通るし、要求が重いのが原因ならどのみち
                        # 待っても通らない。ここで待つと n を下げる段階ごとに積み上がる。
                        n = max(1, n // 2)
                        logger.warning(f"[{component}] {start_date}〜{end_date} S={skip} で"
                                       f"サーバ側の制限 → 要求件数を n={n} に落として再試行します")
                        continue

                    # n=1 でも通らない。この先の復旧は重いので、ここでだけ**待って投げ直す**
                    # （一時的な混雑だった場合はこれで回復する）
                    if deadline_waits < len(DEADLINE_WAITS):
                        delay = DEADLINE_WAITS[deadline_waits]
                        deadline_waits += 1
                        logger.warning(f"[{component}] {start_date}〜{end_date} S={skip} は n=1 でも"
                                       f"通りません → {delay:.0f}秒待って投げ直します "
                                       f"({deadline_waits}/{len(DEADLINE_WAITS)})")
                        time.sleep(delay)
                        continue

                    # 待っても駄目 ＝ その Change 単体が重い。ファイル一覧を分けて取り直す
                    one = self._recover_change(component, start_date, end_date, skip, gerrit_path)
                    if one is None:
                        # 復旧もできなかったので、この 1 件は飛ばして先へ進む（収集は止めない）
                        self._skipped_changes.append(
                            {"component": component, "range": f"{start_date}〜{end_date}",
                             "offset": skip, "reason": "ServerDeadlineExceeded"})
                        skip += 1
                        n, deadline_waits = batch_size, 0
                        continue
                    changes = [one]
                    break
                except (requests.exceptions.ReadTimeout, requests.exceptions.HTTPError, ValueError) as e:
                    # 応答の時間切れ・HTTP エラー（再試行しても通らなかったもの）・壊れた応答。
                    # 期間ではもう割らない区間のときだけ、ここで扱う（それ以外は呼び出し側が期間を割る）
                    if not reduce_on_failure:
                        raise
                    smaller = [m for m in TIMEOUT_PAGE_SIZES if m < n]
                    if smaller:
                        n = smaller[0]
                        logger.warning(f"[{component}] {start_date}〜{end_date} S={skip} で失敗"
                                       f"（{type(e).__name__}）→ 要求件数を n={n} に落として取り直します")
                        continue
                    # 1 件でも失敗 ＝ その Change 単体が重い。本体とファイル一覧に分けて取る
                    one = self._recover_change(component, start_date, end_date, skip, gerrit_path)
                    if one is None:
                        # それでも取れない。記録して次の 1 件へ進む（収集は止めない。完了マーカーも付かない）
                        self._skipped_changes.append(
                            {"component": component, "range": f"{start_date}〜{end_date}",
                             "offset": skip, "reason": f"{type(e).__name__}（n=1・分割取得とも失敗）"})
                        skip += 1
                        n = batch_size
                        continue
                    changes = [one]
                    break
            if not changes:
                break
            for change in changes:
                change_data = self._collect_change_details(change, component)
                if change_data:
                    changes_out.append(change_data['change'])
                    commits_out.extend(change_data['commits'])
                else:
                    skipped += 1
                    # 以前は件数を数えるだけで、どの Change かを記録していなかった
                    self._skipped_changes.append(
                        {"component": component, "range": f"{start_date}〜{end_date}",
                         "offset": -1, "change_number": change.get("_number"),
                         "reason": "Change の中身の処理に失敗"})
            skip += len(changes)
            if n < batch_size:
                # このページは通ったので次は既定値に戻す（混雑は一時的なので引きずらない）
                logger.info(f"[{component}] 要求件数を n={batch_size} に戻します")
                n = batch_size
        return changes_out, commits_out, skipped

    def _mode_sig(self) -> str:
        """収集モードの署名（diff の有無・revision 範囲）。区間マーカーに含め、モード変更時は再収集させる。"""
        if self.config.is_endpoint_enabled('file_diff') or self.config.is_endpoint_enabled('commit'):
            return f"diff-{self.revision_scope}"
        return "base"

    def _chunk_marker(self, component: str, start_date: str, end_date: str) -> Path:
        """区間の完了マーカーのパス（収集モードを含めるので、モードを変えると別マーカー＝再収集）。"""
        d = self.change_storage.output_dir / component / "changes"
        safe = f"{start_date}_{end_date}_{self._mode_sig()}".replace("/", "-")
        return d / f".chunk_{safe}.done"

    def _write_chunk_marker(self, marker: Path, n_changes: int, n_commits: int) -> None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"n_changes": n_changes, "n_commits": n_commits}),
                          encoding="utf-8")
    
    def _collect_change_details(self, change: Dict[str, Any], 
                                component: str) -> Optional[Dict[str, Any]]:
        """変更の詳細データを収集"""
        change_id = change['id']
        change_number = change['_number']
        
        try:
            result = {
                'change': change.copy(),
                'commits': []
            }
            
            # _number を change_number として保存（ストレージでの検索用）
            result['change']['change_number'] = change_number
            
            # 変更詳細
            if self.config.is_endpoint_enabled('change_detail'):
                detail = self.endpoints['change_detail'].fetch(change_id=change_id)
                result['change'].update(detail)

            # Included In 情報
            if self.config.is_endpoint_enabled('included_in'):
                included_in = self.endpoints['included_in'].fetch(change_id=change_id)
                result['change']['included_in'] = included_in
            
            # コメント
            if self.config.is_endpoint_enabled('comments'):
                comments = self.endpoints['comments'].fetch(change_id=change_id)
                result['change']['comments'] = comments
            
            # レビュワー
            if self.config.is_endpoint_enabled('reviewers'):
                reviewers = self.endpoints['reviewers'].fetch(change_id=change_id)
                result['change']['reviewers'] = reviewers
            
            # コミット情報（revision = patch set 単位。patch set 1 が投稿時点）
            if self.config.is_endpoint_enabled('commit'):
                revisions = change.get('revisions', {})
                # revision_scope="first" なら投稿時点（patch set 1）だけに絞る
                if self.revision_scope == "first":
                    revisions = {rid: rv for rid, rv in revisions.items()
                                 if rv.get("_number") == 1}
                for revision_id, revision in revisions.items():
                    # ファイル一覧は commit エンドポイントではなく change detail の
                    # revision に入っている（ALL_FILES で取得済み）。ここから渡す。
                    files = revision.get('files', {})
                    commit_data = self._collect_commit_details(
                        change_id, revision_id, change_number, files
                    )
                    if commit_data:
                        result['commits'].append(commit_data)
            
            logger.info(f"処理中: {component} - PR #{change_number}")
            
            return result
            
        except Exception as e:
            logger.error(f"変更 #{change_number} の収集エラー: {e}")
            return None
    
    def _collect_commit_details(self, change_id: str, revision_id: str,
                                change_number: int,
                                files: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """コミットの詳細データを収集

        files: その revision のファイル一覧（change detail の revision['files']）。
               commit エンドポイントは files を返さないため、呼び出し側から渡す。
        """
        try:
            commit_data = {
                'change_id': change_id,
                'revision_id': revision_id,
                'change_number': change_number,
            }

            # コミット情報
            commit_info = self.endpoints['commit'].fetch(
                change_id=change_id,
                revision_id=revision_id
            )
            commit_data['commit'] = commit_info

            # ファイル情報（change detail の revision から渡された一覧を使う）
            files = files or {}
            commit_data['files'] = list(files.keys())
            commit_data['file_changes'] = {
                path: {
                    "lines_inserted": info.get("lines_inserted", 0),
                    "lines_deleted": info.get("lines_deleted", 0),
                    "size_delta": info.get("size_delta", 0)
                } for path, info in files.items()
            }
            
            # 親コミット
            if self.config.is_endpoint_enabled('commit_parents'):
                parents = self.endpoints['commit_parents'].fetch(
                    change_id=change_id,
                    revision_id=revision_id
                )
                commit_data['commit']['parents'] = parents
            
            # ファイル差分
            if self.config.is_endpoint_enabled('file_diff'):
                file_diffs = {}
                
                for file_path in files.keys():
                    try:
                        diff = self.endpoints['file_diff'].fetch(
                            change_id=change_id,
                            revision_id=revision_id,
                            file_path=file_path
                        )
                        if diff:
                            file_diffs[file_path] = diff
                    except Exception as e:
                        logger.warning(f"ファイル差分取得エラー ({file_path}): {e}")
                
                commit_data['file_diffs'] = file_diffs
            
            return commit_data
            
        except Exception as e:
            logger.error(f"コミット {revision_id[:8]} の収集エラー: {e}")
            return None


def main():
    """メイン関数

    実行例:
        # endpoint_config.yaml の既定（OpenStack 6 コンポーネント）
        python -m src.collectors.change_collector
        # 新規プロジェクトだけ
        python -m src.collectors.change_collector --project qtbase libreoffice
        # 期間を指定（既定は endpoint_config.yaml / constants.py）
        python -m src.collectors.change_collector --project qtbase --start 2000-01-01 --end 2024-12-31
    """
    ap = argparse.ArgumentParser(description="Gerrit から Change を収集する")
    ap.add_argument("--project", nargs="*", default=None,
                    help=f"対象（既定は設定ファイル）。登録済み: {', '.join(GERRIT_PROJECTS)}")
    ap.add_argument("--start", default=None, help="開始日 YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="終了日 YYYY-MM-DD")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    logger.info("Gerrit データ収集を開始します")

    try:
        collector = ChangeCollector()
        cfg = collector.config.get_collection_config()
        if args.project:
            unknown = [p for p in args.project if p not in GERRIT_PROJECTS]
            if unknown:
                raise ValueError(f"未登録のプロジェクト: {unknown}（constants.GERRIT_PROJECTS に追加してください）")
            cfg["components"] = list(args.project)
        if args.start:
            cfg["start_date"] = args.start
        if args.end:
            cfg["end_date"] = args.end
        logger.info(f"対象={cfg['components']} 期間={cfg['start_date']}〜{cfg['end_date']}")
        status = collector.collect_all_components()
        # 最後の 1 行は実行の状態に合わせる（design.md §6.6「終了時の表示」）。以前は incomplete でも
        # 「正常に完了しました」と出し、直前の「取り直しても取れなかったものがあります」と食い違っていた
        if status == "completed":
            logger.info("データ収集が正常に完了しました")
        else:
            logger.warning("データ収集を終了しました。**取れなかったものがあります**（上のエラーに対象と位置。"
                           "同じコマンドをもう一度実行すると、取れなかった区間だけを取り直します）")
    except Exception as e:
        logger.error(f"データ収集中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
