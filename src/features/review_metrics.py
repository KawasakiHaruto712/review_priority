import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import re
import logging
import pandas as pd
import configparser

# 必要に応じて，NLTKのインポートとデータダウンロード
# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ロギング設定 (必要に応じて調整)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_bot_names(config_path: Path) -> List[str]:
    """
    gerrymanderconfig.ini からボットのユーザー名を読み込む
    (review_comment_processor.py と同じ関数を再利用)

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
            bot_names = [name.strip() for name in config['organization']['bots'].split(',')]
            logger.info(f"ボット数: {len(bot_names)}")
        else:
            logger.warning(f"'{config_path}' に 'organization' セクションまたは 'bots' エントリが見つかりません。")
    except configparser.Error as e:
        logger.error(f"'{config_path}' のパース中にエラーが発生しました: {e}")
    except FileNotFoundError:
        logger.error(f"エラー: gerrymanderconfig.ini が {config_path} に見つかりません。")
    return bot_names

class ReviewStatusAnalyzer:
    """
    レビューコメントの対応状況を分析し, メトリクスを算出するクラス
    修正要求と修正確認の抽出には、review_comment_processor.pyで抽出されたキーワードと、
    review_label.json のCode-Reviewラベルを使用する
    """
    def __init__(
        self, 
        extraction_keywords_path: Path, # review_keywords.json のパス
        gerrymander_config_path: Path, # gerrymanderconfig.ini のパス
        review_label_path: Path # review_label.json のパス
    ):
        self.extraction_keywords_path = extraction_keywords_path
        self.gerrymander_config_path = gerrymander_config_path
        self.review_label_path = review_label_path
        
        # 1. review_keywords.json からキーワードの読み込み
        try:
            with open(self.extraction_keywords_path, 'r', encoding='utf-8') as f:
                keywords_data = json.load(f)
            self.request_keywords = set(keywords_data.get('修正要求', [])) # 高速なルックアップのためsetに変換
            self.confirmation_keywords = set(keywords_data.get('修正確認', [])) # 高速なルックアップのためsetに変換
            logger.info(f"review_keywords.json からキーワードがロードされました.")
            logger.info(f"修正要求キーワード数: {len(self.request_keywords)}")
            logger.info(f"修正確認キーワード数: {len(self.confirmation_keywords)}")
        except Exception as e:
            logger.error(f"エラー: review_keywords.json ファイルの読み込み中にエラーが発生しました: {e}")
            self.request_keywords = set()
            self.confirmation_keywords = set()

        # review_label.json からレビューラベルを読み込む
        self.all_review_labels = [] # すべてのラベル（前処理で削除するため）
        self.code_review_minus_labels = set() # Code-Reviewのminusラベル（修正要求として使用）
        self.code_review_plus_labels = set() # Code-Reviewのplusラベル（修正確認として使用）
        self.non_code_review_labels_pattern = None # Code-Review以外のラベルを削除するための正規表現

        try:
            with open(self.review_label_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
            
            non_code_review_labels = [] # Code-Review以外のラベルを一時的に保持

            for category, sentiments in labels_data.items():
                for sentiment_type, labels in sentiments.items():
                    # すべてのラベルをall_review_labelsに追加
                    self.all_review_labels.extend(labels)

                    if category == "Code-Review":
                        if sentiment_type == "minus":
                            self.code_review_minus_labels.update(labels)
                        elif sentiment_type == "plus":
                            self.code_review_plus_labels.update(labels)
                    else: # Code-Review以外のカテゴリのラベルはnon_code_review_labelsに追加
                        non_code_review_labels.extend(labels)
            
            # 長いラベルから順にソートすることで、短いラベルが長いラベルの一部である場合に
            # 長いラベルが先にマッチして削除されるようにする
            self.all_review_labels.sort(key=len, reverse=True)
            non_code_review_labels.sort(key=len, reverse=True)

            # Code-Review以外のラベルを削除するための正規表現パターンを作成
            if non_code_review_labels:
                # 単語境界 `\b` を使用して、部分マッチではなく完全な単語としてのラベルを削除
                self.non_code_review_labels_pattern = re.compile(
                    r'\b(?:' + '|'.join(re.escape(label) for label in non_code_review_labels) + r')\b', 
                    re.IGNORECASE
                )
            logger.info(f"review_label.json からラベルがロードされました。")
            logger.info(f"Code-Review (minus) ラベル数: {len(self.code_review_minus_labels)}")
            logger.info(f"Code-Review (plus) ラベル数: {len(self.code_review_plus_labels)}")
        except Exception as e:
            logger.error(f"エラー: review_label.json ファイルの読み込み中にエラーが発生しました: {e}")

        # ボット名のロード
        self.bot_names = _load_bot_names(self.gerrymander_config_path)

        # NLTK関連のインスタンス化を削除
        # self.lemmatizer = WordNetLemmatizer()
        # self.english_stopwords = set(stopwords.words('english'))

    def _preprocess_comment_for_matching(self, comment_text: str) -> str:
        """
        キーワードマッチングのためにコメントを前処理する
        結果は単一の文字列として返す
        """
        processed_comment = comment_text.lower()
        processed_comment = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+|[^\s<>"]+\.[a-zA-Z]{2,}', " ", processed_comment) # URLを削除
        processed_comment = re.sub(r"patch set \d+:", " ", processed_comment, flags=re.IGNORECASE) # "Patch Set 数字:" を削除
        processed_comment = re.sub(r"\(\d+\s*(?:inline\s+)?comments?\)", " ", processed_comment) # インラインコメントのパターンを削除
        
        # --- 変更点: Code-Review以外のラベルを削除 ---
        if self.non_code_review_labels_pattern:
            processed_comment = self.non_code_review_labels_pattern.sub(" ", processed_comment)
        
        processed_comment = re.sub(r"[^a-zA-Z'0-9\+-]+", " ", processed_comment) # 記号を削除
        
        # プロジェクト名やその他不要コメントを削除 (review_comment_processor.pyと一致させる)
        processed_comment = re.sub(r"\b(nova|neutron|cinder|horizon|keystone|swift|glance|openstack|ci)\b", " ", processed_comment)
        
        # NLTKのtokenizeの代わりに、簡単なスペース区切りを使用
        words = processed_comment.split()

        # NLTK関連のストップワード除去, レマタイズを削除
        # words = [word for word in words if word not in self.english_stopwords]                           
        # words = [self.lemmatizer.lemmatize(word) for word in words]                                      
        words = [word for word in words if word] # 空文字列の単語を除外

        return " ".join(words) # 単語リストをスペースで結合して単一の文字列として返す

    def _classify_comment_as_request(self, comment_text: str) -> bool:
        """
        コメントを修正要求として分類する（キーワードベースとCode-Review minusラベルベース）
        """
        processed_comment_string = self._preprocess_comment_for_matching(comment_text)
        
        # review_keywords.jsonのキーワードとCode-Review (minus) ラベルの両方でチェック
        for keyword in self.request_keywords:
            if keyword in processed_comment_string:
                logger.debug(f"修正要求キーワード '{keyword}' がコメント内で見つかりました。")
                return True
        
        for label in self.code_review_minus_labels:
            # ラベルはすでに前処理で小文字化されていることを前提に、そのまま比較
            # ただし、完全な単語としてマッチするかを確認するため、単語境界を考慮
            if re.search(r'\b' + re.escape(label) + r'\b', processed_comment_string):
                logger.debug(f"Code-Review (minus) ラベル '{label}' がコメント内で見つかりました。")
                return True

        return False

    def _is_confirmation_comment(self, comment_text: str) -> bool:
        """
        コメントが修正確認であるかを判定する（キーワードベースとCode-Review plusラベルベース）
        """
        processed_comment_string = self._preprocess_comment_for_matching(comment_text)

        # --- 変更点: review_keywords.jsonのキーワードとCode-Review (plus) ラベルの両方でチェック ---
        for keyword in self.confirmation_keywords:
            if keyword in processed_comment_string:
                logger.debug(f"修正確認キーワード '{keyword}' がコメント内で見つかりました。")
                return True
        
        for label in self.code_review_plus_labels:
            # ラベルはすでに前処理で小文字化されていることを前提に、そのまま比較
            # ただし、完全な単語としてマッチするかを確認するため、単語境界を考慮
            if re.search(r'\b' + re.escape(label) + r'\b', processed_comment_string):
                logger.debug(f"Code-Review (plus) ラベル '{label}' がコメント内で見つかりました。")
                return True
        
        return False

    def analyze_pr_status(self, pr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        PRの各時点における修正要求の対応状況を分析する関数

        この関数は、与えられたプルリクエスト (PR) のデータ (`pr_data`) を解析し、
        PRのライフサイクルにおける修正要求の発生、完了、未完了の状態を追跡します
        具体的には以下のメトリクスを計算し、`pr_data` ディクショナリに追加して返します

        - `uncompleted_requests_counts`: 各イベント発生時点での未完了の修正要求数
        - `uncompleted_requests_latest_timestamps`: 各イベント発生時点での未完了の修正要求の中で最も新しいもののタイムスタンプ
        - `completed_requests_count`: PRのライフサイクル全体を通して完了した修正要求の総数
        - `uncompleted_requests_final_count`: PRクローズ時の最終的な未完了修正要求数

        処理のフロー:
        1. `pr_data` からすべてのメッセージ（コメント）とリビジョンアップロードイベントを抽出し、
           タイムスタンプに基づいてソート
        2. 各イベントを時系列順に処理
        3. メッセージが「修正確認」と分類された場合、それ以前のリビジョンアップロードより前に
           発生した未完了の修正要求を「完了」として処理し、`completed_requests_count` を更新
           これは、新しいリビジョンのアップロードとそれに続く修正確認コメントが、
           以前の修正要求に対応していると仮定
        4. メッセージが「修正要求」と分類された場合、その要求を未完了リストに追加
        5. リビジョンアップロードイベントがあった場合、そのタイムスタンプを記録
        6. PRの処理が完了した時点（すべてのイベントを処理した後）で、残っている未完了の修正要求の数と、
           最も古い未完了要求の経過時間 (`uncompleted_requests_age_at_close`) を計算

        Args:
            pr_data (Dict[str, Any]): 単一のプルリクエストに関するデータを含む辞書
                                       Gerritのシステムから取得された生データ形式を想定
                                       'messages' (コメント), 'revisions' (リビジョン情報),
                                       'last_updated' (最終更新日時) などのキーが含まれる

        Returns:
            uncompleted_requests_count (int): PRのライフサイクル全体を通しての未完了の修正要求数
        """
        
        pr_data['uncompleted_requests_counts'] = []
        pr_data['uncompleted_requests_latest_timestamps'] = []
        pr_data['completed_requests_count'] = 0

        events = []
        for msg in pr_data.get('messages', []):
            events.append({
                'type': 'message',
                'timestamp': datetime.fromisoformat(msg['date'].replace('Z', '+00:00')),
                'author': msg.get('author', {}).get('name'),
                'message_id': msg.get('id'),
                'comment_text': msg.get('message', ''),
                'revision_number': msg.get('revision_number')
            })
        for rev_id, rev_data in pr_data.get('revisions', {}).items():
            events.append({
                'type': 'revision_upload',
                'timestamp': datetime.fromisoformat(rev_data['created'].replace('Z', '+00:00')),
                'revision_number': rev_data.get('_number')
            })

        events.sort(key=lambda x: x['timestamp'])

        pending_requests: List[Dict[str, Any]] = []
        last_revision_upload_timestamp = None

        for event in events:
            if event['type'] == 'revision_upload':
                last_revision_upload_timestamp = event['timestamp']
                logger.debug(f"PR {pr_data.get('change_number')}: リビジョン {event.get('revision_number')} が {event['timestamp']} にアップロードされました.")
            
            elif event['type'] == 'message':
                comment_author = event.get('author')
                # ボットのコメントをスキップ
                if comment_author and comment_author in self.bot_names:
                    logger.debug(f"PR {pr_data.get('change_number')}: ボット '{comment_author}' のコメントをスキップします。")
                    continue

                comment_text = event['comment_text']
                
                is_request = self._classify_comment_as_request(comment_text)
                is_confirmation = self._is_confirmation_comment(comment_text)

                if is_confirmation:
                    logger.debug(f"PR {pr_data.get('change_number')}: 修正確認が {event['timestamp']} で識別されました: {comment_text[:50]}...")
                    # 最新のリビジョンアップロードより古い要求を完了済みとして処理
                    if last_revision_upload_timestamp:
                        requests_to_clear_indices = [
                            i for i, req in enumerate(pending_requests) 
                            if req['timestamp'] < last_revision_upload_timestamp
                        ]
                        for i in sorted(requests_to_clear_indices, reverse=True):
                            pending_requests.pop(i)
                            pr_data['completed_requests_count'] += 1

                        if requests_to_clear_indices:
                            logger.debug(f"修正確認により {len(requests_to_clear_indices)} 件の要求が完了しました.")
                        else:
                            logger.debug("修正確認が見つかりましたが, 完了可能な (最新リビジョン更新より前の) 要求はありません.")
                    else:
                        logger.debug("修正確認が見つかりましたが, それ以前にリビジョン更新がないため, 要求と紐付けできません.")
                        
                    # 修正確認コメント自体が新たな修正要求を含む場合があるため、そのチェックも行う
                    # is_requestがtrueの場合、新しい要求として追加
                    if is_request:
                        pending_requests.append({
                            'timestamp': event['timestamp'],
                            'comment_text': comment_text
                        })
                        logger.debug(f"PR {pr_data.get('change_number')}: コメントは修正確認かつ新たな修正要求です. 新たな要求を追加しました.")

                elif is_request:
                    # 修正要求の場合、未完了リストに追加
                    pending_requests.append({
                        'timestamp': event['timestamp'],
                        'comment_text': comment_text
                    })
                    logger.debug(f"PR {pr_data.get('change_number')}: 修正要求が {event['timestamp']} で識別されました: {comment_text[:50]}...\n未完了の要求: {len(pending_requests)}")
                
                else:
                    logger.debug(f"PR {pr_data.get('change_number')}: コメント {event['timestamp']} は修正要求でも修正確認でもありません.")

        # PRクローズ時の未完了要求数を計算
        uncompleted_requests_count = len(pending_requests)

        return uncompleted_requests_count