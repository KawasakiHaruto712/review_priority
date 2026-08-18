"""matplotlib の日本語フォント設定（豆腐□対策）。表・折れ線の両方で使う。"""
from __future__ import annotations

from matplotlib import font_manager
import matplotlib.pyplot as plt

_JP_FONT_CANDIDATES = [
    "Hiragino Sans", "Hiragino Kaku Gothic ProN", "YuGothic", "Yu Gothic",
    "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoGothic", "AppleGothic",
]


def configure_japanese_font() -> None:
    """利用可能な日本語フォントがあれば設定する（無ければ既定のまま）。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _JP_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
