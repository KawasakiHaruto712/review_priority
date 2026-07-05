"""
収集マニフェスト（何を・いつからいつまで・全部取れたか）

収集を実行するたびに、コンパクトな 1 レコード（実行サマリー）を JSONL ファイルへ追記する。
実行ログそのものは残さない（巨大になるため）。代わりに「後から確認したい最小限」だけを残す:

  - いつ実行したか（started_at / finished_at）
  - どのプロジェクトを、どの期間（要求した start_date / end_date）対象にしたか
  - 実際に取得できた範囲（earliest / latest）と件数（fetched / saved / skipped）
  - 完走したか・途中で止まったか（status: completed / partial / interrupted / failed）

JSONL（1 実行 = 1 行）なので履歴が積み上がり、ファイルも小さいまま。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 既定のマニフェストファイル名（出力ディレクトリ直下に置く）
MANIFEST_FILENAME = "collection_manifest.jsonl"


class CollectionManifest:
    """1 回の収集実行を記録するマニフェスト。

    使い方:
        manifest = CollectionManifest(output_dir, collector="changes",
                                      requested={"start_date": ..., "end_date": ...,
                                                 "components": [...]})
        try:
            for component in components:
                ...  # 収集
                manifest.record_component(component, status="completed",
                                          fetched=.., saved=.., skipped=..,
                                          earliest=.., latest=.., extra={...})
            manifest.finish("completed")
        except BaseException as e:           # 中断・例外も必ず記録する
            manifest.finish("interrupted", error=str(e))
            raise

    finish() を呼ぶと JSONL に 1 行追記される（finally や except から必ず呼ぶこと）。
    """

    def __init__(self, output_dir: Path, collector: str,
                 requested: Optional[dict[str, Any]] = None,
                 filename: str = MANIFEST_FILENAME):
        self.path = Path(output_dir) / filename
        self.collector = collector
        self.requested = requested or {}
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.components: dict[str, dict[str, Any]] = {}

    def record_component(self, component: str, *, status: str,
                         fetched: Optional[int] = None, saved: Optional[int] = None,
                         skipped: Optional[int] = None,
                         earliest: Optional[str] = None, latest: Optional[str] = None,
                         error: Optional[str] = None,
                         extra: Optional[dict[str, Any]] = None) -> None:
        """1 コンポーネント分の収集結果を記録する。

        status: "completed"（全部取れた）/ "partial"（一部のみ・途中で止まった）/ "failed"（取得できず）
        fetched: 一覧取得した件数 / saved: 実際に保存できた件数 / skipped: 取得に失敗して飛ばした件数
        earliest, latest: 実際に取得できたデータの最古・最新（created 等）
        """
        entry: dict[str, Any] = {"status": status}
        if fetched is not None:
            entry["fetched"] = fetched
        if saved is not None:
            entry["saved"] = saved
        if skipped is not None:
            entry["skipped"] = skipped
        if earliest is not None:
            entry["earliest"] = earliest
        if latest is not None:
            entry["latest"] = latest
        if error is not None:
            entry["error"] = error
        if extra:
            entry.update(extra)
        self.components[component] = entry

    def finish(self, status: str, error: Optional[str] = None) -> Path:
        """実行全体の status を確定し、マニフェストを JSONL に 1 行追記する。

        status: "completed" / "interrupted" / "failed"
        """
        record = {
            "collector": self.collector,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "requested": self.requested,
            "components": self.components,
        }
        if error is not None:
            record["error"] = error
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"収集マニフェストを記録しました（status={status}）: {self.path}")
        return self.path
