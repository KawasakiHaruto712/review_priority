import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import re
import logging
import pandas as pd # For pd.notna

logger = logging.getLogger(__name__)

# ロギング設定 (必要に応じて調整)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReviewStatusAnalyzer:
    """
    レビューコメントの対応状況を分析し, メトリクスを算出するクラス
    修正要求の抽出には学習済みBERTモデルを, 修正確認の抽出にはキーワードを使用
    """
    def __init__(
        self, 
        bert_model_path: Path, 
        confirmation_keywords_path: Path, 
        device: str = None # 'cuda' or 'cpu'
    ):
        self.bert_model_path = bert_model_path
        self.confirmation_keywords_path = confirmation_keywords_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. BERTモデルとトークナイザの読み込み
        try:
            # tokenizer_nameは'bert-base-uncased'が使われているため, それを採用
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_path) # モデルパスから直接ロード
            self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
            self.bert_model.to(self.device)
            self.bert_model.eval() # 推論モードに設定
            logger.info(f"BERTモデルが {bert_model_path} からロードされ, {self.device} に移動しました.")
        except Exception as e:
            logger.error(f"BERTモデルまたはトークナイザのロード中にエラーが発生しました: {bert_model_path}: {e}")
            self.tokenizer = None
            self.bert_model = None
            raise RuntimeError("BERTモデルのロードに失敗しました. 修正要求の分類を続行できません.")

        # 2. 修正確認キーワードの読み込みとパターンコンパイル
        self.confirmation_patterns = self._load_and_compile_confirmation_keywords(confirmation_keywords_path)
        if not self.confirmation_patterns:
            logger.warning(f"修正確認キーワードが {confirmation_keywords_path} からロードされませんでした. 確認検出が機能しない可能性があります.")

    def _load_and_compile_confirmation_keywords(self, path: Path) -> re.Pattern | None:
        """JSONファイルから修正確認キーワードを読み込み, 正規表現パターンにコンパイルする"""
        if not path.exists():
            logger.error(f"エラー: 修正確認キーワードファイルが {path} に見つかりません.")
            return None # Noneを返すように変更
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            keywords = []
            # AchieveComments.json の形式は [{"AchieveComments": "keyword"}]
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "AchieveComments" in item:
                        # キーワードの型が文字列であることを確認し, 小文字に変換してエスケープ
                        if isinstance(item["AchieveComments"], str):
                            keywords.append(re.escape(item["AchieveComments"].lower()))
            else:
                logger.error(f"{path} の予期せぬフォーマットです. 辞書のリストが期待されます.")
                return None # Noneを返すように変更

            if not keywords:
                return None # Noneを返すように変更
            
            # 全てのキーワードをORで結合し, 単語境界でマッチング (大文字小文字無視)
            pattern_str = r'\b(' + '|'.join(keywords) + r')\b'
            return re.compile(pattern_str, re.IGNORECASE)
        except Exception as e:
            logger.error(f"{path} からの修正確認キーワードのロードまたはコンパイル中にエラーが発生しました: {e}")
            return None # Noneを返すように変更

    def _classify_comment_as_request(self, comment_text: str) -> bool:
        """
        BERTモデルを使用して, コメントが修正要求であるかを分類
        """
        if not self.bert_model or not self.tokenizer:
            logger.error("BERTモデルまたはトークナイザがロードされていません. コメントを分類できません.")
            return False
        
        inputs = self.tokenizer(
            comment_text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(self.device)

        with torch.no_grad(): # 勾配計算を無効化し, メモリ使用量を削減
            outputs = self.bert_model(**inputs)
        
        # ロジットから予測クラス (0または1) を取得
        # 0: 修正要求ではない, 1: 修正要求である (この二値分類タスク向けにファインチューニングされていると仮定)
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
        
        return predicted_class_id == 1 # 1が修正要求を示すと仮定

    def _is_comment_confirmation(self, comment_text: str) -> bool:
        """
        キーワードマッチングにより, コメントが修正確認であるかを判定
        """
        if self.confirmation_patterns is None: # パターンがNoneの場合のチェックを追加
            return False
        return bool(self.confirmation_patterns.search(comment_text))

    def analyze_review_status(self, pr_data: Dict[str, Any], analysis_time: datetime) -> Dict[str, Any]:
        """
        PRのレビューコメント履歴とリビジョン更新状況に基づき, レビューの対応状況を分析

        Args:
            pr_data (Dict[str, Any]): 単一のPRの詳細データ (openstack.pyで収集されたJSONデータ)
                                      'messages' (コメント履歴), 'all_revisions_info' (リビジョン履歴) が必要
            analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻

        Returns:
            int: 未完了の修正要求の数
        """
        uncompleted_requests_count = 0
        completed_requests_count = 0
        
        # 1. 分析時点までのコメントとリビジョンを収集し, 時系列で統合
        event_stream = []

        # コメントイベントの追加
        for msg in pr_data.get('messages', []):
            # 'datetime'キーがNoneでないことを確認
            if msg.get('datetime') and msg['datetime'] <= analysis_time:
                event_stream.append({
                    'type': 'comment',
                    'timestamp': msg['datetime'],
                    'data': msg.get('message', '')
                })
        
        # リビジョン更新イベントの追加
        # pr_data['all_revisions_info']が利用可能であることを前提
        for rev_id, rev_info in pr_data.get('all_revisions_info', {}).items():
            if rev_info.get('created'):
                try:
                    # ISO 8601形式の文字列をdatetimeオブジェクトに変換
                    rev_datetime = datetime.fromisoformat(rev_info['created'].split('.')[0].replace(' ', 'T'))
                    if rev_datetime <= analysis_time:
                        event_stream.append({
                            'type': 'revision_update', # イベントタイプを'revision_update'に変更
                            'timestamp': rev_datetime,
                            'data': rev_id # リビジョンIDなど
                        })
                except ValueError as e:
                    logger.warning(f"PR {pr_data.get('change_number')}: リビジョン日時 {rev_info['created']} のパースに失敗しました: {e}")

        # イベントストリームを時系列でソート
        event_stream.sort(key=lambda x: x['timestamp'])

        # 修正要求の追跡リスト: 各要素は {'timestamp': datetime, 'comment_text': str}
        pending_requests = [] # ここには, まだ完了していない修正要求を格納
        
        # 最後に確認されたリビジョン更新のタイムスタンプ
        # これが「修正確認の投稿されたリビジョンより前のリビジョン」の基準となる
        last_revision_update_time = None # 変数名を変更

        for event in event_stream:
            if event['type'] == 'revision_update': # イベントタイプを'revision_update'に変更
                # リビジョン更新があった場合, その時刻を記録
                last_revision_update_time = event['timestamp']
                logger.debug(f"PR {pr_data.get('change_number')}: リビジョン更新が {event['timestamp']} に発生しました.")

            elif event['type'] == 'comment':
                comment_text = event['data']
                
                is_confirmation = self._is_comment_confirmation(comment_text)
                is_request = self._classify_comment_as_request(comment_text)

                if is_confirmation:
                    # 修正確認の投稿があった場合
                    logger.debug(f"PR {pr_data.get('change_number')}: 確認コメントが {event['timestamp']} で見つかりました: {comment_text[:50]}...")
                    
                    if not pending_requests: # 未完了の要求がなければスキップ
                        logger.debug("未完了の要求がないためスキップします.")
                        
                    # 修正確認が投稿されたリビジョンより前のリビジョンで投稿された修正要求を全て完了とみなす
                    # => req.timestamp < last_revision_update_time の条件を満たす要求をクリア
                    
                    if last_revision_update_time is not None:
                        requests_to_clear_indices = []
                        for i, req in enumerate(pending_requests):
                            if req['timestamp'] < last_revision_update_time: # リビジョン更新より前に投稿された要求
                                requests_to_clear_indices.append(i)
                        
                        # 後ろから削除することでインデックスのずれを防ぐ
                        for i in sorted(requests_to_clear_indices, reverse=True):
                            completed_requests_count += 1
                            pending_requests.pop(i)
                            
                        if requests_to_clear_indices:
                            logger.debug(f"修正確認により {len(requests_to_clear_indices)} 件の要求が完了しました.")
                        else:
                            logger.debug("修正確認が見つかりましたが, 完了可能な (最新リビジョン更新より前の) 要求はありません.")
                    else:
                        logger.debug("修正確認が見つかりましたが, それ以前にリビジョン更新がないため, 要求と紐付けできません.")
                        
                    # もしこのコメント自体が修正要求でもあれば, 新たな要求として追加
                    if is_request:
                        pending_requests.append({
                            'timestamp': event['timestamp'],
                            'comment_text': comment_text
                        })
                        logger.debug(f"PR {pr_data.get('change_number')}: コメントは修正確認かつ新たな修正要求です. 新たな要求を追加しました.")

                elif is_request: # 修正確認ではないが, 修正要求である場合
                    pending_requests.append({
                        'timestamp': event['timestamp'],
                        'comment_text': comment_text
                    })
                    logger.debug(f"PR {pr_data.get('change_number')}: 修正要求が {event['timestamp']} で識別されました: {comment_text[:50]}...")
                
                # どちらでもないコメントはスキップ
                else:
                    logger.debug(f"PR {pr_data.get('change_number')}: コメント {event['timestamp']} は修正要求でも修正確認でもありません.")


        # analysis_time 終了時点での未完了の修正要求の数をカウント
        uncompleted_requests_count = len(pending_requests)

        return uncompleted_requests_count