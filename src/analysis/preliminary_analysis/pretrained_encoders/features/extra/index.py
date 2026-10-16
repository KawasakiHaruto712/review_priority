"""リポジトリ全体の索引（design.md §13.2・§13.4）。ChangeFacts の一覧から 1 度だけ作り、各時点 t で引く。

段 1 では「中核の開発者」の索引だけを置く。開発状況の特徴量に要る索引（作成・閉じた出来事の並び、
ファイルの履歴など）は段 3 で足す。
"""
from __future__ import annotations

from datetime import datetime

from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.common import ChangeFacts


class CoreIndex:
    """中核の開発者（design.md §13.2）：t までに Code-Review の +2 か −2 を付けたことがある人（Ruangwan 2019）。

    投票はメッセージの 1 行目の記録から取る。ボットの投票は数えない（中核の開発者は人の役割のため）。
    「t までに」は t より前（t ちょうどの投票は含めない。その投票は t の時点ではまだ見えていない扱い）。
    """

    def __init__(self, facts: list[ChangeFacts]):
        self.first_pm2: dict[object, datetime] = {}
        for f in facts:
            for m in f.messages:
                if m.is_bot or m.author_id is None:
                    continue
                if any(lab == "Code-Review" and abs(val) == 2 for lab, val in m.votes):
                    prev = self.first_pm2.get(m.author_id)
                    if prev is None or m.date < prev:
                        self.first_pm2[m.author_id] = m.date

    def is_core(self, account_id: object, t: datetime) -> bool:
        first = self.first_pm2.get(account_id)
        return first is not None and first < t
