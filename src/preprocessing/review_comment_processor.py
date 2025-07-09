import pandas as pd
import json
from collections import defaultdict
from pathlib import Path
import logging
import re
import numpy as np
import configparser

# NLTKのインポートとデータダウンロードの指示を削除
# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ロギング設定 (必要に応じて調整)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _generate_ngrams(words: list[str], n_min: int, n_max: int) -> list[str]:
    """
    単語のリストからN-gram (フレーズ) を生成

    Args:
        words (List[str]): 前処理済みの単語のリスト
        n_min (int): 生成するN-gramの最小の長さ
        n_max (int): 生成するN-gramの最大の長さ

    Returns:
        List[str]: 生成されたN-gramのリスト
    """
    ngrams = []
    num_words = len(words)
    for n in range(n_min, n_max + 1):
        for i in range(num_words - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.append(ngram)
    return ngrams

def _load_bot_names(config_path: Path) -> list[str]:
    """
    gerrymanderconfig.ini からボットのユーザー名を読み込む

    Args:
        config_path (Path): gerrymanderconfig.ini のパス

    Returns:
        List[str]: ボットのユーザー名リスト
    """
    bot_names = []
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        if 'organization' in config and 'bots' in config['organization']:
            # カンマで分割し、空白を除去
            bot_names = [name.strip() for name in config['organization']['bots'].split(',')]
            logger.info(f"ボット名が {config_path} からロードされました: {bot_names}")
        else:
            logger.warning(f"'{config_path}' に 'organization' セクションまたは 'bots' エントリが見つかりません。")
    except configparser.Error as e:
        logger.error(f"'{config_path}' のパース中にエラーが発生しました: {e}")
    except FileNotFoundError:
        logger.error(f"エラー: gerrymanderconfig.ini が {config_path} に見つかりません。")
    return bot_names

def _load_review_labels(label_path: Path) -> list[str]:
    """
    review_label.json からすべてのレビューラベルを読み込む

    Args:
        label_path (Path): review_label.json のパス

    Returns:
        List[str]: すべてのレビューラベルのリスト
    """
    all_labels = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        for category in labels_data.values():
            for sentiment in category.values():
                all_labels.extend(sentiment)
        # 長いラベルから順にソートすることで、短いラベルが長いラベルの一部である場合に
        # 長いラベルが先にマッチして削除されるようにする
        all_labels.sort(key=len, reverse=True)
        logger.info(f"レビューラベルが {label_path} からロードされました。合計 {len(all_labels)} 個。")
    except Exception as e:
        logger.error(f"エラー: レビューラベルファイルの読み込み中にエラーが発生しました: {e}")
    return all_labels

def summarize_keywords_by_inclusion(keywords: list[str]) -> list[str]:
    """
    キーワードのリストから包含関係にある冗長なキーワードを削除し、より一般的な（短い）キーワードを優先する。

    例: ['good', 'looks good', 'looks good to me'] -> ['good']
    """
    if not keywords:
        return []

    # 短いキーワードから順にソートする (昇順)
    sorted_keywords = sorted(keywords, key=len, reverse=False)
    
    # 最終的に残すキーワードのセット
    kept_keywords_set = set()

    for current_keyword in sorted_keywords:
        is_redundant = False
        # 現在のキーワードが、既に保持された（より短い）キーワードを含んでいるかチェック
        # もし current_keyword が kept_keyword を含むなら、current_keyword はより長く冗長なのでスキップ
        for kept_keyword in kept_keywords_set:
            if kept_keyword in current_keyword and current_keyword != kept_keyword:
                is_redundant = True
                break
        
        if not is_redundant:
            kept_keywords_set.add(current_keyword)
            
    # アルファベット順にソートして返す
    return sorted(list(kept_keywords_set))

def extract_and_save_review_keywords(
    checklist_path: Path,
    output_keywords_path: Path,
    gerrymander_config_path: Path,
    review_label_path: Path, # 新しい引数
    min_comment_count: int = 10,
    min_precision_ratio: float = 0.90,
    ngram_min: int = 1,
    ngram_max: int = 10
):
    """
    checklist.csvから修正要求/修正確認キーワード (N-gram) を抽出し, JSONファイルに保存

    Args:
        checklist_path (Path): checklist.csvファイルのパス
        output_keywords_path (Path): 抽出されたキーワードを保存するJSONファイルのパス
        gerrymander_config_path (Path): gerrymanderconfig.ini のパス
        review_label_path (Path): review_label.json のパス
        min_comment_count (int): 語彙を含むレビューコメントの最小数
        min_precision_ratio (float): 語彙を含むレビューコメントが修正要求/確認として分類される最小割合 (0.0〜1.0)
        ngram_min (int): 抽出するN-gramの最小の長さ (デフォルト: 1)
        ngram_max (int): 抽出するN-gramの最大の長さ (デフォルト: 10)
    """

    from src.utils.constants import LABELLED_CHANGE_COUNT

    if not checklist_path.exists():
        logger.error(f"エラー: checklist.csv が {checklist_path} に見つかりません.")
        return

    # ボット名をロード
    bot_names = _load_bot_names(gerrymander_config_path)
    # レビューラベルをロード
    review_labels = _load_review_labels(review_label_path)
    # レビューラベルから正規表現パターンを生成 (長いラベルを先にマッチさせるためソート済みのものを使用)
    # 単語境界 `\b` を使用して、部分マッチではなく完全な単語としてのラベルを削除
    review_labels_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(label) for label in review_labels) + r')\b', re.IGNORECASE)


    try:
        df = pd.read_csv(checklist_path)
        # カラム名が想定通りか確認。ボットフィルタリングのため'author'カラムが必要
        if 'comment' not in df.columns or '修正要求' not in df.columns or '修正確認' not in df.columns or 'author' not in df.columns:
            logger.error("エラー: checklist.csv に 'comment', '修正要求', '修正確認', 'author' カラムが必要です。")
            return
        
        # '修正要求'と'修正確認'カラムの値を文字列に統一し, 空白をNaNに変換して扱う
        df['修正要求'] = df['修正要求'].astype(str).replace(r'^\s*$', np.nan, regex=True)
        df['修正確認'] = df['修正確認'].astype(str).replace(r'^\s*$', np.nan, regex=True)

        # ラベルづけしたChange数でフィルタリング
        original_df_rows = len(df)
        df = df[df['PRNumber'] <= LABELLED_CHANGE_COUNT]
        logger.info(f"Change番号フィルター適用: {original_df_rows} 行から {len(df)} 行に削減 (PRNumber <= {LABELLED_CHANGE_COUNT})")

    except Exception as e:
        logger.error(f"エラー: checklist.csv の読み込み中にエラーが発生しました: {e}")
        return
    
    # 語彙ごとの集計
    vocab_counts = defaultdict(lambda: {'total': 0, 'is_request': 0, 'is_confirmation': 0})

    logger.info("キーワード (N-gram) 抽出のためにレビューコメントを分析中...")
    for index, row in df.iterrows():
        original_comment = str(row['comment'])
        comment_author = str(row['author'])

        # ボットのコメントをスキップ
        if comment_author in bot_names:
            logger.debug(f"ボット '{comment_author}' のコメントをスキップします。: {original_comment[:50]}...")
            continue

        # '修正要求'と'修正確認'の判定ロジック
        is_request_row = False
        request_val = row['修正要求']
        if pd.notna(request_val): # NaNでないことを確認
            # 文字列として '1' または '1.0' のどちらかに一致するかをチェック
            if str(request_val).strip() in ['1', '1.0']:
                is_request_row = True

        is_confirmation_row = False
        confirmation_val = row['修正確認']
        if pd.notna(confirmation_val): # NaNでないことを確認
            # 文字列として '1' または '1.0' のどちらかに一致するかをチェック
            if str(confirmation_val).strip() in ['1', '1.0']:
                is_confirmation_row = True

        # コメントの前処理
        processed_comment = original_comment.lower()
        processed_comment = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+|[^\s<>"]+\.[a-zA-Z]{2,}', " ", processed_comment) # URLを削除
        processed_comment = re.sub(r"patch set \d+:", " ", processed_comment, flags=re.IGNORECASE) # "Patch Set 数字:" のパターンを削除
        processed_comment = re.sub(r"\(\d+\s*(?:inline\s+)?comments?\)", " ", processed_comment) # インラインコメントのパターンを削除
        
        # review_label.jsonから読み込んだラベルを削除
        processed_comment = review_labels_pattern.sub(" ", processed_comment)
        
        processed_comment = re.sub(r"[^a-zA-Z'0-9\+-]+", " ", processed_comment) # 記号を削除
        
        # プロジェクト名やその他不要コメントを削除 (単語境界でマッチングを強化)
        processed_comment = re.sub(r"\b(nova|neutron|cinder|horizon|keystone|swift|glance|openstack|ci)\b", " ", processed_comment)
        
        # トークナイズ
        # NLTKのtokenizeの代わりに、簡単なスペース区切りを使用
        words = processed_comment.split() 

        # ストップワード除去, レマタイズ（NLTKがないため削除）
        # words = [word for word in words if word not in english_stopwords]                           
        # words = [lemmatizer.lemmatize(word) for word in words]                                      
        
        # 空文字列の単語を除外 (前処理でスペースになった部分など)
        words = [word for word in words if word]

        # N-gram (フレーズ) の生成
        phrases = _generate_ngrams(words, ngram_min, ngram_max)
        
        # フレーズごとのカウント (コメント内での重複を避けるためsetを使用)
        for phrase in set(phrases): 
            vocab_counts[phrase]['total'] += 1
            if is_request_row:
                vocab_counts[phrase]['is_request'] += 1
            if is_confirmation_row:
                vocab_counts[phrase]['is_confirmation'] += 1

    extracted_keywords = {
        '修正要求': [],
        '修正確認': []
    }

    for phrase, counts in vocab_counts.items():
        if counts['total'] >= min_comment_count:
            if counts['is_request'] / counts['total'] >= min_precision_ratio:
                extracted_keywords['修正要求'].append(phrase)
            if counts['is_confirmation'] / counts['total'] >= min_precision_ratio:
                extracted_keywords['修正確認'].append(phrase)

    extracted_keywords['修正要求'] = summarize_keywords_by_inclusion(extracted_keywords['修正要求'])
    extracted_keywords['修正確認'] = summarize_keywords_by_inclusion(extracted_keywords['修正確認'])

    # JSONファイルとして保存
    output_keywords_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_keywords_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_keywords, f, indent=2, ensure_ascii=False)
        logger.info(f"抽出されたキーワードが {output_keywords_path} に保存されました.")
        logger.info(f"修正要求キーワード数: {len(extracted_keywords['修正要求'])}")
        logger.info(f"修正確認キーワード数: {len(extracted_keywords['修正確認'])}")
    except Exception as e:
        logger.error(f"キーワードのJSONへの保存中にエラーが発生しました: {e}")

# このスクリプトを直接実行するためのエントリポイント
if __name__ == "__main__":
    from src.config.path import DEFAULT_DATA_DIR, DEFAULT_CONFIG
    checklist_csv_path = DEFAULT_DATA_DIR / "processed" / "checklist.csv"
    output_json_path = DEFAULT_DATA_DIR / "processed" / "review_keywords.json"
    gerrymander_config_path = DEFAULT_CONFIG / "gerrymanderconfig.ini"
    review_label_json_path = DEFAULT_DATA_DIR / "processed" / "review_label.json" # review_label.json のパスを追加

    extract_and_save_review_keywords(
        checklist_csv_path, 
        output_json_path,
        gerrymander_config_path,
        review_label_json_path, # 新しい引数を渡す
        min_comment_count=10,
        min_precision_ratio=0.90,
        ngram_min= 1,
        ngram_max=10
    )