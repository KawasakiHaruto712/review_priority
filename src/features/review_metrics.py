import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import re
import logging
import pandas as pd # For pd.notna
import configparser # configparserをインポート

# NLTKのインポートとデータダウンロードの指示
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTKデータのダウンロード（初回のみ必要）
# 環境に合わせてコメントアウトを外し、一度実行してください
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

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
            logger.info(f"ボット名が {config_path} からロードされました: {bot_names}")
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
    修正要求と修正確認の抽出には、review_comment_processor.pyで抽出されたキーワードを使用
    """
    def __init__(
        self, 
        extraction_keywords_path: Path, # review_keywords.json のパスをコンストラクタに追加
        gerrymander_config_path: Path # gerrymanderconfig.ini のパスをコンストラクタに追加
    ):
        self.extraction_keywords_path = extraction_keywords_path
        self.gerrymander_config_path = gerrymander_config_path
        
        # 1. キーワードの読み込み
        try:
            with open(self.extraction_keywords_path, 'r', encoding='utf-8') as f:
                keywords_data = json.load(f)
            self.request_keywords = set(keywords_data.get('修正要求', [])) # 高速なルックアップのためsetに変換
            self.confirmation_keywords = set(keywords_data.get('修正確認', [])) # 高速なルックアップのためsetに変換
            logger.info(f"キーワードが {self.extraction_keywords_path} からロードされました.")
            logger.info(f"修正要求キーワード数: {len(self.request_keywords)}")
            logger.info(f"修正確認キーワード数: {len(self.confirmation_keywords)}")
        except Exception as e:
            logger.error(f"エラー: キーワードファイルの読み込み中にエラーが発生しました: {e}")
            self.request_keywords = set()
            self.confirmation_keywords = set()

        # ボット名のロード
        self.bot_names = _load_bot_names(self.gerrymander_config_path)

        # レマタイザーとストップワードのインスタンス化（利用したい場合はコメントアウトを外す）
        # self.lemmatizer = WordNetLemmatizer()
        # self.english_stopwords = set(stopwords.words('english'))

    def _preprocess_comment_for_matching(self, comment_text: str) -> str: # 戻り値を str に変更
        """
        キーワードマッチングのためにコメントを前処理する
        結果は単一の文字列として返す
        """
        processed_comment = comment_text.lower()
        processed_comment = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+|[^\s<>"]+\.[a-zA-Z]{2,}', " ", processed_comment) # URLを削除
        processed_comment = re.sub(r"patch set \d+:", " ", processed_comment, flags=re.IGNORECASE) # "Patch Set 数字:" を削除
        processed_comment = re.sub(r"\(\d+\s*(?:inline\s+)?comments?\)", " ", processed_comment) # インラインコメントのパターンを削除
        processed_comment = re.sub(r"[^a-zA-Z'0-9\+-]+", " ", processed_comment) # 記号を削除
        
        # プロジェクト名やその他不要コメントを削除 (review_comment_processor.pyと一致させる)
        processed_comment = re.sub(r"\b(nova|neutron|cinder|horizon|keystone|swift|glance|openstack|ci)\b", " ", processed_comment)
        
        # トークナイズ
        words = word_tokenize(processed_comment)

        # ストップワード除去, レマタイズ（利用したい場合はコメントアウトを外す）                                                  
        # words = [word for word in words if word not in self.english_stopwords]                           
        # words = [self.lemmatizer.lemmatize(word) for word in words]                                      
        words = [word for word in words if word] # 空文字列の単語を除外

        return " ".join(words) # 単語リストをスペースで結合して単一の文字列として返す

    def _classify_comment_as_request(self, comment_text: str) -> bool:
        """
        コメントを修正要求として分類する（キーワードベース）
        """
        processed_comment_string = self._preprocess_comment_for_matching(comment_text) # 処理済みコメント文字列を取得
        
        for keyword in self.request_keywords: # 抽出されたキーワードをイテレート
            # 処理済みコメント文字列がキーワードを部分文字列として含むかチェック
            if keyword in processed_comment_string: # シンプルな部分文字列チェック
                logger.debug(f"修正要求キーワード '{keyword}' がコメント内で見つかりました。")
                return True
        return False

    def _is_confirmation_comment(self, comment_text: str) -> bool:
        """
        コメントが修正確認であるかを判定する（キーワードベース）
        """
        processed_comment_string = self._preprocess_comment_for_matching(comment_text) # 処理済みコメント文字列を取得

        for keyword in self.confirmation_keywords: # 抽出されたキーワードをイテレート
            if keyword in processed_comment_string: # シンプルな部分文字列チェック
                logger.debug(f"修正確認キーワード '{keyword}' がコメント内で見つかりました。")
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
            Dict[str, Any]: 分析結果のメトリクスが追加された更新版の `pr_data` 辞書。
        """
        
        pr_data['uncompleted_requests_counts'] = []
        pr_data['uncompleted_requests_latest_timestamps'] = []
        pr_data['completed_requests_count'] = 0

        events = []
        for msg in pr_data.get('messages', []):
            events.append({
                'type': 'message',
                'timestamp': datetime.fromisoformat(msg['date'].replace('Z', '+00:00')),
                'author': msg.get('author', {}).get('name'), # author 情報も取得
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
                comment_author = event.get('author') # メッセージの著者を取得
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

        pr_data['uncompleted_requests_final_count'] = uncompleted_requests_count
        return pr_data