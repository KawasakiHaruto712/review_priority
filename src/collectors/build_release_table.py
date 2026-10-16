"""リリースの区切りの表（major_releases_summary.csv）を作り直す入口。

次の 3 つを順に行う（project_selection/design.md §5.2「追記：区切りの日をタグから取る」）。
  1. OpenStack の行：major_release_storage（手元の openstack/releases の写しを読む。通信なし）
  2. qtbase・qt-creator・mediawiki/core・qtdeclarative・go・TYPO3.CMS の行：release_tags_collector
     （正式版のタグの日。通信は 6 回。対象は constants.RELEASE_TAG_TARGETS）
  3. 本流の表（main_branches.csv）：main_branch_collector（今の本流は Gerrit の HEAD、昔の本流は登録表から。
     通信は 6 回。2026-10-15 追加）

どちらも、自分の対象の行だけを入れ替え、ほかの行は残す。

実行:
    python -m src.collectors.build_release_table
"""
from __future__ import annotations

import logging
import sys
import time

from src.collectors import main_branch_collector, release_tags_collector
from src.collectors.major_release_storage import MajorReleaseStorage
from src.config.path import DEFAULT_DATA_DIR
from src.utils.constants import RELEASE_TAG_TARGETS

logger = logging.getLogger(__name__)


def build() -> None:
    logger.info("===== リリースの区切りの表を作り直します =====")
    MajorReleaseStorage(data_dir=DEFAULT_DATA_DIR / "openstack_collected").save()
    release_tags_collector.collect(list(RELEASE_TAG_TARGETS))
    # 本流の表（今の本流は Gerrit の HEAD、昔の本流は登録表から。2026-10-15 追加）。区切りの表と同じ対象
    logger.info("===== 本流の表を作り直します =====")
    time.sleep(release_tags_collector.PROJECT_INTERVAL)   # 区切りの表の最後の問い合わせと間を空ける
    main_branch_collector.collect(list(RELEASE_TAG_TARGETS))
    logger.info("===== 完了 =====")


if __name__ == "__main__":
    try:
        build()
    except Exception as e:
        logger.critical(f"実行中にエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)
