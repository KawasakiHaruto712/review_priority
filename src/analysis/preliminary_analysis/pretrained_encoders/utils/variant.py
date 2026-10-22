"""エンコーダの版の種類（重みなし版 base／重みあり版 posw）の切り替え（design.md §15.2。2026-10-22）。

- base：重みなし版（今までのエンコーダと結果）。保存先・出力先は今のまま。
- posw：重みあり版（事前学習の損失で正例を w 倍）。エンコーダは cutoff_<版>_posw/、各分析の出力は出力先の下の posw/。
- 既定は重みあり版。ただし**書き込みをするモードでは --variant を必ず指定する**（付け忘れたら止める。上書きを防ぐため）。
  軽いモード（描き直し・順位・報告用の表など）は付けなければ既定で動く。
- 書き込みをするモードで出力先に結果があれば、上書きの警告をログに出して続ける。ログのファイル名に版の種類を入れる。

各分析の main は、引数を読んだら apply(variant, 各分析の constants) を呼ぶ。constants.OUTPUT_ROOT が版の出力先に切り替わり、
エンコーダの読み書き（io/store.py）も版のフォルダを使う。各分析のコードは呼ぶたびに constants.OUTPUT_ROOT を読むので、それだけで切り替わる。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants

logger = logging.getLogger(__name__)

BASE = "base"
POSW = "posw"
VARIANTS = (BASE, POSW)
DEFAULT_VARIANT = POSW
LABEL = {BASE: "重みなし版", POSW: "重みあり版"}


def subdir(root: Path, variant: str) -> Path:
    """出力先：重みなし版はそのまま、重みあり版は root/posw。"""
    check(variant)
    return Path(root) if variant == BASE else Path(root) / variant


def cutoff_dirname(cutoff: str, variant: str) -> str:
    """エンコーダのフォルダ名：重みなし版は cutoff_<版>、重みあり版は cutoff_<版>_posw。"""
    check(variant)
    return f"cutoff_{cutoff}" if variant == BASE else f"cutoff_{cutoff}_{variant}"


def check(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"版の種類は {VARIANTS} のどれかです（いま {variant!r}）")


def add_argument(ap: argparse.ArgumentParser) -> None:
    """--variant を足す（既定は None。resolve で決める）。"""
    ap.add_argument("--variant", choices=list(VARIANTS), default=None,
                    help=f"エンコーダの版の種類：base（重みなし版）／posw（重みあり版）。書き込みをするモードでは必須。"
                         f"軽いモードは省略すると {DEFAULT_VARIANT}（pretrained_encoders/design.md §15.2）")


def resolve(given: str | None, writing: bool) -> str:
    """書き込みをするモードで指定がなければ止める。軽いモードなら既定（重みあり版）。"""
    if given is None:
        if writing:
            raise SystemExit("書き込みをするモードでは --variant を指定してください（--variant posw か --variant base。"
                             "付け忘れで今の結果を上書きしないため。pretrained_encoders/design.md §15.2）")
        return DEFAULT_VARIANT
    check(given)
    return given


def apply(variant: str, *const_modules) -> str:
    """版の種類を、エンコーダの読み書きと各分析の出力先（constants.OUTPUT_ROOT）に当てる。何度呼んでも同じ結果になる。"""
    check(variant)
    constants.ENCODER_VARIANT = variant
    for mod in const_modules:
        if not hasattr(mod, "BASE_OUTPUT_ROOT"):
            mod.BASE_OUTPUT_ROOT = mod.OUTPUT_ROOT          # 最初の 1 回だけ、重みなし版の出力先を覚える
        mod.OUTPUT_ROOT = subdir(mod.BASE_OUTPUT_ROOT, variant)
    logger.info(f"版の種類：{variant}（{LABEL[variant]}）")
    return variant


def warn_if_exists(path: Path, what: str = "") -> bool:
    """出力先に結果がすでにあれば、上書きの警告をログに出す（止めない）。あれば True。"""
    p = Path(path)
    if p.exists() and (p.is_file() or any(p.iterdir())):
        logger.warning(f"既存の結果を上書きします{('（' + what + '）') if what else ''}：{p}")
        return True
    return False


def log_name(name: str, variant: str) -> str:
    """並べて回すときのログの名前に版の種類を入れる（例 lookback_window_posw）。"""
    return f"{name}_{variant}"
