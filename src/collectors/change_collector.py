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
from datetime import date, timedelta
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
        headers = {"Accept": "application/json"}

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
        logger.info(f"接続先: {spec['host']} / project:{spec['path']}"
                    f"（認証{'あり（' + env_prefix + '_*）' if auth else 'なし'}）")
        return spec
    
    def collect_all_components(self):
        """全コンポーネントのデータを収集"""
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
            for component in components:
                self.collect_component(component, manifest=manifest)
            manifest.finish("completed")
            logger.info("全コンポーネントのデータ収集完了")
        except BaseException as e:
            # 途中で止まった/落ちた場合も「どこまで取れたか」を必ず残す
            manifest.finish("interrupted", error=f"{type(e).__name__}: {e}")
            raise

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
        chunks = date_chunks(cfg['start_date'], cfg['end_date'], cfg.get('checkpoint_years'))
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

                changes, commits, skipped = self._collect_range_adaptive(
                    component, cs, ce, batch_size, gerrit_path=gerrit_path)

                # 途中保存: この区間ぶんを即ディスクへ（クラッシュしてもここまでは残る）
                change_rows.extend(self.change_storage.save_changes(component, changes))
                self.commit_storage.save_commits(component, commits)

                total_saved += len(changes)
                total_skipped += skipped
                total_commits += len(commits)
                created_all.extend(c.get("created") for c in changes if c.get("created"))
                if chunked:
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
                           "dropped_detail": dropped[:50]},
                )
            logger.info(
                f"{component} の収集{('完了' if status == 'completed' else '中断')}: "
                f"{total_saved}変更, {total_commits}コミット, スキップ {total_skipped}件"
            )
            if dropped:
                # 黙って欠けたまま解析へ進むのを防ぐ
                logger.warning(f"[{component}] **取得できずに飛ばした Change が {len(dropped)} 件あります**。"
                               f"collection_manifest.jsonl の dropped_detail で位置を確認できます")

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
            return self._collect_range(component, start_date, end_date, batch_size,
                                       gerrit_path=gerrit_path)
        except (ServerDeadlineExceeded, requests.exceptions.HTTPError, ValueError) as e:
            s, t = date.fromisoformat(start_date), date.fromisoformat(end_date)
            span = (t - s).days
            if span <= min_days:
                logger.error(f"[{component}] 区間 {start_date}〜{end_date} はこれ以上割れません: {e}")
                raise
            mid = (s + timedelta(days=span // 2)).isoformat()
            logger.warning(f"[{component}] 区間 {start_date}〜{end_date}（{span}日）で失敗 → "
                           f"{start_date}〜{mid} と {mid}〜{end_date} に割って取り直します: "
                           f"{type(e).__name__}")
            ch_a, cm_a, sk_a = self._collect_range_adaptive(component, start_date, mid,
                                                           batch_size, gerrit_path, min_days)
            ch_b, cm_b, sk_b = self._collect_range_adaptive(component, mid, end_date,
                                                           batch_size, gerrit_path, min_days)
            return ch_a + ch_b, cm_a + cm_b, sk_a + sk_b

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
        filled = 0
        for sha, meta in revisions.items():
            rev_no = meta.get("_number", sha)
            try:
                meta["files"] = ep.fetch_revision_files(number, rev_no)
                filled += 1
            except Exception as e:
                # このリビジョンだけ諦める（空にして続行）。件数はログに残す
                meta["files"] = {}
                logger.warning(f"[{component}] #{number} rev{rev_no} のファイル一覧を取得できません: "
                               f"{type(e).__name__}")
        logger.info(f"[{component}] #{number} を復旧しました（リビジョン {filled}/{len(revisions)} 個の"
                    f"ファイル一覧を取得）")
        return change

    def _collect_range(self, component: str, start_date: str, end_date: str,
                       batch_size: int,
                       gerrit_path: str = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """1 つの日付区間の変更・コミットを取得して返す（(changes, commits, skipped)）。"""
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
            if not changes:
                break
            for change in changes:
                change_data = self._collect_change_details(change, component)
                if change_data:
                    changes_out.append(change_data['change'])
                    commits_out.extend(change_data['commits'])
                else:
                    skipped += 1
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
        collector.collect_all_components()
        logger.info("データ収集が正常に完了しました")
    except Exception as e:
        logger.error(f"データ収集中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
