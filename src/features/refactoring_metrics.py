import re
import logging

logger = logging.getLogger(__name__)

# ロギング設定の例 (必要に応じて調整)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SARパターンに基づいた正規表現キーワードのリスト
SAR_PATTERNS = [
    r"refactor(ing|ed)?",
    r"mov(e|ed|ing)",
    r"split(ting)?",
    r"fix(e(d|s)|ing)?",  # "fixing"も含むように修正
    r"introduc(e|ed|ing)",
    r"decompos(e|ed|ing|ition)",
    r"reorgani(z|s)(e|ed|ing|ation)?",
    r"extract(ed|ing)?",
    r"merg(e|ed|ing)",
    r"renam(e|ed|ing)",
    r"chang(e|ed|ing)",
    r"restructur(e|ed|ing)",
    r"reformat(ted|ting)?",
    r"extend(ed|ing)?",
    r"remov(e|ed|ing)",
    r"replac(e|ed|ing)",
    r"rewrit(e|ing)",
    r"simplify(ing|ied)?",
    r"creat(e|ed|ing)",
    r"improv(e|ed|ing|ement)",
    r"add(ed|ing)?",
    r"modify(ied|ing)?",
    r"enhanc(e|ed|ing)",
    r"rework(ed|ing)?",
    r"inlin(e|ed|ing)",
    r"redesign(ed|ing)?",
    r"cleanup", # 論文では "Cleanup"
    r"reduc(e|ed|ing)",
    r"encapsulat(e|ed|ing|ion)",
    r"removed poor coding practice",
    r"improve naming consistency",
    r"removing unused classes",
    r"pull some code up",
    r"use better name",
    r"replace it with",
    r"make maintenance easier",
    r"code cleanup",
    r"minor simplification",
    r"reorganize project structures",
    r"code maintenance for refactoring",
    r"remove redundant code",
    r"moved and gave clearer names to",
    r"refactor bad designed code",
    r"getting code out of",
    r"deleting a lot of old stuff",
    r"code revision",
    r"fix technical debt",
    r"fix quality issue",
    r"antipattern bad for performances",
    r"major structural changes",
    r"minor structural changes",
    r"clean up unnecessary code",
    r"code reformatting & reordering",
    r"nicer code",
    r"formatted",
    r"structure",
    r"simplify code redundancies",
    r"added more checks for quality factors",
    r"naming improvements",
    r"renamed for consistency",
    r"refactoring towards nicer name analysis",
    r"change design",
    r"modularize the code",
    r"code cosmetics",
    r"moved more code out of",
    r"remove dependency",
    r"enhanced code beauty",
    r"simplify internal design",
    r"change package structure",
    r"use a safer method",
    r"code improvements",
    r"minor enhancement",
    r"get rid of unused code",
    r"fixing naming convention",
    r"fix module structure",
    r"code optimization",
    r"fix a design flaw",
    r"nonfunctional code cleanup",
    r"improve code quality",
    r"fix code smell",
    r"use less code",
    r"avoid future confusion",
    r"more easily extended",
    r"polishing code",
    r"move unused file away",
    r"many cosmetic changes",
    r"inlined unnecessary classes",
    r"code cleansing",
    r"fix quality flaws",
    r"simplify the code"
]

# 全てのSARパターンをORで結合した単一の正規表現をコンパイル
SAR_REGEX = re.compile(
    r"\b(" + "|".join(pattern.replace(" ", r"\s+") for pattern in SAR_PATTERNS) + r")\b",
    re.IGNORECASE # 大文字・小文字を区別しない
)

def calculate_refactoring_confidence(pr_title: str, pr_description: str | None) -> int:
    """
    Pull Requestのタイトルと概要テキストに基づき、リファクタリング確信度を算出
    特定されたSARパターンにテキストがマッチするかどうかで0または1の値を返す

    Args:
        pr_title (str): Pull Requestのタイトル
        pr_description (str | None): Pull Requestの概要，存在しない場合はNone

    Returns:
        int: リファクタリングである確信度 (マッチすれば1、しなければ0)
    """
    # 分析対象となるすべてのテキストを結合
    combined_text = pr_title + " " + (pr_description if pr_description is not None else "")
    
    # 結合されたテキストがSARパターンにマッチするかをチェック
    if SAR_REGEX.search(combined_text):
        return 1 # マッチした場合、確信度1
    else:
        return 0 # マッチしない場合、確信度0