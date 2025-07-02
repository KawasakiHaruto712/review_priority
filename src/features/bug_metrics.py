import re
import logging

logger = logging.getLogger(__name__)

# ロギング設定の例 (必要に応じて調整)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_bug_fix_confidence(pr_title: str, pr_description: str | None) -> int:
    """
    Pull Requestのタイトルと概要テキストに基づき、バグ修正確信度を算出
    この確信度は、0-2の範囲でスコアリングされる

    スコアリングルール:
    初期スコアは0
    1. テキスト（タイトルまたは概要のいずれか）が
       バグ番号パターンにマッチした場合、+1点
    2. テキスト（タイトルまたは概要のいずれか）に
       キーワードが含まれている、または、テキストが数字と特定の記号のみで構成される場合、+1点

    Args:
        pr_title (str): Pull Requestのタイトル
        pr_description (str | None): Pull Requestの概要 存在しない場合はNone

    Returns:
        int: 算出されたバグ修正確信度スコア (0-2)。
    """
    score = 0
    
    # 分析対象となるすべてのテキストを結合
    combined_text = pr_title + " " + (pr_description if pr_description is not None else "")
    text_lower = combined_text.lower()

    # 1. バグ番号パターンにマッチした場合
    # 論文の正規表現パターンをPythonのreモジュール用に調整 
    bug_number_patterns = [
        re.compile(r"bug[#\s]*[0-9]+"),
        re.compile(r"pr[#\s]*[0-9]+"),
        re.compile(r"show_bug\.cgi\?id=[0-9]+"),
        re.compile(r"\[[0-9]+\]")
    ]
    
    found_bug_number_pattern = False
    for pattern in bug_number_patterns:
        if pattern.search(text_lower):
            score += 1
            found_bug_number_pattern = True
            break # 1つマッチすれば十分

    # 2. キーワードが含まれている、または数字と特定の記号のみのメッセージの場合
    # キーワードパターン 
    keyword_pattern = re.compile(r"\b(fix(e(d|s))?|bugs?|defects?|patch)\b")
    
    if keyword_pattern.search(text_lower):
        score += 1
    else:
        # 英字が含まれておらず、数字と一般的な区切り文字のみで構成されている場合にスコアを増加 
        if not re.search(r'[a-z]', text_lower) and re.fullmatch(r'[\d\s.,;:#\[\]/\\-]+', text_lower):
            # スコアが0または1の場合のみ増加 
            if score == 0 or score == 1:
                score += 1

    # 最終的なスコアは0から2の範囲に制限（意図しないスコアの増加を防ぐため）
    return min(score, 2)