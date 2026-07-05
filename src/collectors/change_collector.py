"""
変更データ収集オーケストレーター

OpenStack Gerritから変更データを収集するメインクラスです。
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
import requests

from src.collectors.config.collector_config import CollectorConfig
from src.collectors.base.retry_handler import RetryConfig
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
        
        # セッション設定
        self.session = self._create_session()
        
        # リトライ設定
        retry_config_dict = self.config.get_retry_config()
        self.retry_config = RetryConfig(**retry_config_dict)
        
        # エンドポイントインスタンスを作成
        self.endpoints = self._initialize_endpoints()
        
        # ストレージ初期化
        storage_config = self.config.get_storage_config()
        self.change_storage = ChangeStorage(Path(storage_config['output_dir']))
        self.commit_storage = CommitStorage(Path(storage_config['output_dir']))

        # diff/commit を取る revision 範囲: "all"=全 revision / "first"=投稿時点(patch set 1)のみ
        self.revision_scope = self.config.get_collection_config().get("revision_scope", "all")

        logger.info(f"ChangeCollector初期化完了（revision_scope={self.revision_scope}）")
    
    def _create_session(self) -> requests.Session:
        """HTTPセッションを作成"""
        session = requests.Session()
        auth_string = base64.b64encode(
            f"{self.username}:{self.password}".encode()
        ).decode()
        
        session.headers.update({
            "Authorization": f"Basic {auth_string}",
            "Accept": "application/json"
        })
        
        return session
    
    def _initialize_endpoints(self) -> Dict[str, Any]:
        """エンドポイントインスタンスを初期化"""
        endpoints = {}
        enabled_endpoints = self.config.get_enabled_endpoints()
        
        for endpoint_info in enabled_endpoints:
            name = endpoint_info['name']
            endpoint_config = endpoint_info['config']
            class_name = endpoint_config['class']
            
            if class_name in self.ENDPOINT_CLASSES:
                endpoint_class = self.ENDPOINT_CLASSES[class_name]
                endpoints[name] = endpoint_class(
                    username=self.username,
                    password=self.password,
                    session=self.session,
                    timeout=(30, 120)
                )
                logger.info(f"エンドポイント初期化: {name} ({class_name})")
            else:
                logger.warning(f"未知のエンドポイントクラス: {class_name}")
        
        return endpoints
    
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

        cfg = self.config.get_collection_config()
        batch_size = cfg['batch_size']
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

                changes, commits, skipped = self._collect_range(component, cs, ce, batch_size)

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
            if manifest is not None:
                manifest.record_component(
                    component, status=status,
                    fetched=total_saved + total_skipped,
                    saved=total_saved, skipped=total_skipped,
                    earliest=min(created_all) if created_all else None,
                    latest=max(created_all) if created_all else None,
                    error=error,
                    extra={"commits_saved": total_commits},
                )
            logger.info(
                f"{component} の収集{('完了' if status == 'completed' else '中断')}: "
                f"{total_saved}変更, {total_commits}コミット, スキップ {total_skipped}件"
            )

    def _collect_range(self, component: str, start_date: str, end_date: str,
                       batch_size: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """1 つの日付区間の変更・コミットを取得して返す（(changes, commits, skipped)）。"""
        skip = 0
        changes_out: List[Dict[str, Any]] = []
        commits_out: List[Dict[str, Any]] = []
        skipped = 0
        if not self.config.is_endpoint_enabled('changes'):
            logger.warning("changesエンドポイントが無効です")
            return changes_out, commits_out, skipped

        while True:
            changes = self.endpoints['changes'].fetch(
                component=component, start_date=start_date, end_date=end_date,
                limit=batch_size, skip=skip,
            )
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
    """メイン関数"""
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger.info("OpenStack Gerritデータ収集を開始します（リファクタリング版）")
    
    try:
        collector = ChangeCollector()
        collector.collect_all_components()
        logger.info("データ収集が正常に完了しました")
    except Exception as e:
        logger.error(f"データ収集中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
