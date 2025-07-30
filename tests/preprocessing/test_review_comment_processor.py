"""
test_review_comment_processor.py

テストスイート: src.preprocessing.review_comment_processor

主な機能:
- N-gram生成
- ボット名の読み込み
- レビューラベルの読み込み
- キーワード包含関係での冗長性除去
- レビューコメントキーワード抽出

テスト対象:
- _generate_ngrams: N-gram生成機能
- _load_bot_names: configファイルからのボット名読み込み
- _load_review_labels: JSONからのレビューラベル読み込み
- summarize_keywords_by_inclusion: キーワードの冗長性除去
- extract_and_save_review_keywords: メイン機能のキーワード抽出
"""

import unittest
import pandas as pd
import json
import tempfile
import configparser
from pathlib import Path
from unittest.mock import patch, mock_open
import numpy as np

from src.preprocessing.review_comment_processor import (
    _generate_ngrams,
    _load_bot_names,
    _load_review_labels,
    summarize_keywords_by_inclusion,
    extract_and_save_review_keywords
)


class TestGenerateNgrams(unittest.TestCase):
    """_generate_ngrams関数のテスト"""

    def test_basic_unigrams(self):
        """基本的なunigram生成"""
        words = ["hello", "world", "test"]
        result = _generate_ngrams(words, 1, 1)
        expected = ["hello", "world", "test"]
        self.assertEqual(result, expected)

    def test_basic_bigrams(self):
        """基本的なbigram生成"""
        words = ["hello", "world", "test"]
        result = _generate_ngrams(words, 2, 2)
        expected = ["hello world", "world test"]
        self.assertEqual(result, expected)

    def test_mixed_ngrams(self):
        """unigram+bigramの混合生成"""
        words = ["a", "b", "c"]
        result = _generate_ngrams(words, 1, 2)
        expected = ["a", "b", "c", "a b", "b c"]
        self.assertEqual(result, expected)

    def test_empty_words(self):
        """空の単語リスト"""
        words = []
        result = _generate_ngrams(words, 1, 3)
        self.assertEqual(result, [])

    def test_single_word(self):
        """単一単語"""
        words = ["only"]
        result = _generate_ngrams(words, 1, 3)
        expected = ["only"]
        self.assertEqual(result, expected)

    def test_n_larger_than_words(self):
        """N-gramのNが単語数より大きい場合"""
        words = ["a", "b"]
        result = _generate_ngrams(words, 3, 5)  # 3-gram以上は生成されない
        self.assertEqual(result, [])

    def test_trigrams(self):
        """trigram生成"""
        words = ["one", "two", "three", "four"]
        result = _generate_ngrams(words, 3, 3)
        expected = ["one two three", "two three four"]
        self.assertEqual(result, expected)


class TestLoadBotNames(unittest.TestCase):
    """_load_bot_names関数のテスト"""

    def test_load_valid_config(self):
        """有効なconfigファイルからのボット名読み込み"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""
[organization]
bots = bot1, bot2, bot3
""")
            config_path = Path(f.name)

        try:
            result = _load_bot_names(config_path)
            expected = ["bot1", "bot2", "bot3"]
            self.assertEqual(result, expected)
        finally:
            config_path.unlink()

    def test_load_config_with_whitespace(self):
        """空白を含むボット名の処理"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""
[organization]
bots =  bot1 ,  bot2,bot3  
""")
            config_path = Path(f.name)

        try:
            result = _load_bot_names(config_path)
            expected = ["bot1", "bot2", "bot3"]
            self.assertEqual(result, expected)
        finally:
            config_path.unlink()

    def test_missing_file(self):
        """存在しないファイル"""
        config_path = Path("/nonexistent/path.ini")
        result = _load_bot_names(config_path)
        self.assertEqual(result, [])

    def test_missing_section(self):
        """organizationセクションが存在しない"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""
[other]
key = value
""")
            config_path = Path(f.name)

        try:
            result = _load_bot_names(config_path)
            self.assertEqual(result, [])
        finally:
            config_path.unlink()

    def test_missing_bots_key(self):
        """botsキーが存在しない"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""
[organization]
other_key = value
""")
            config_path = Path(f.name)

        try:
            result = _load_bot_names(config_path)
            self.assertEqual(result, [])
        finally:
            config_path.unlink()

    def test_empty_bots_value(self):
        """空のbotsの値"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""
[organization]
bots = 
""")
            config_path = Path(f.name)

        try:
            result = _load_bot_names(config_path)
            self.assertEqual(result, [""])  # 空文字列が1つ含まれる
        finally:
            config_path.unlink()


class TestLoadReviewLabels(unittest.TestCase):
    """_load_review_labels関数のテスト"""

    def test_load_valid_labels(self):
        """有効なレビューラベルファイルの読み込み"""
        labels_data = {
            "request": {
                "positive": ["please fix", "needs update"],
                "negative": ["not required"]
            },
            "confirmation": {
                "positive": ["looks good", "approved"],
                "negative": ["rejected"]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(labels_data, f)
            label_path = Path(f.name)

        try:
            result = _load_review_labels(label_path)
            expected = ["please fix", "needs update", "not required", 
                       "looks good", "approved", "rejected"]
            # 長さでソートされるので順序を確認
            expected_sorted = sorted(expected, key=len, reverse=True)
            self.assertEqual(result, expected_sorted)
        finally:
            label_path.unlink()

    def test_missing_file(self):
        """存在しないファイル"""
        label_path = Path("/nonexistent/labels.json")
        result = _load_review_labels(label_path)
        self.assertEqual(result, [])

    def test_invalid_json(self):
        """無効なJSONファイル"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json {")
            label_path = Path(f.name)

        try:
            result = _load_review_labels(label_path)
            self.assertEqual(result, [])
        finally:
            label_path.unlink()

    def test_empty_labels(self):
        """空のラベルデータ"""
        labels_data = {}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(labels_data, f)
            label_path = Path(f.name)

        try:
            result = _load_review_labels(label_path)
            self.assertEqual(result, [])
        finally:
            label_path.unlink()


class TestSummarizeKeywordsByInclusion(unittest.TestCase):
    """summarize_keywords_by_inclusion関数のテスト"""

    def test_basic_inclusion_removal(self):
        """基本的な包含関係の冗長性除去"""
        keywords = ["good", "looks good", "looks good to me"]
        result = summarize_keywords_by_inclusion(keywords)
        expected = ["good"]
        self.assertEqual(result, expected)

    def test_no_inclusion(self):
        """包含関係がない場合"""
        keywords = ["good", "bad", "neutral"]
        result = summarize_keywords_by_inclusion(keywords)
        expected = ["bad", "good", "neutral"]  # アルファベット順
        self.assertEqual(result, expected)

    def test_empty_keywords(self):
        """空のキーワードリスト"""
        keywords = []
        result = summarize_keywords_by_inclusion(keywords)
        self.assertEqual(result, [])

    def test_partial_inclusion(self):
        """一部包含関係がある場合"""
        keywords = ["fix", "please fix", "update", "needs update"]
        result = summarize_keywords_by_inclusion(keywords)
        expected = ["fix", "update"]
        self.assertEqual(result, expected)

    def test_identical_keywords(self):
        """同じキーワードが複数ある場合"""
        keywords = ["test", "test", "example"]
        result = summarize_keywords_by_inclusion(keywords)
        expected = ["example", "test"]  # 重複は除去される
        self.assertEqual(result, expected)

    def test_complex_inclusion(self):
        """複雑な包含関係"""
        keywords = ["a", "ab", "abc", "abcd", "xyz"]
        result = summarize_keywords_by_inclusion(keywords)
        expected = ["a", "xyz"]  # "a"が最短、"xyz"は独立
        self.assertEqual(result, expected)


class TestExtractAndSaveReviewKeywords(unittest.TestCase):
    """extract_and_save_review_keywords関数のテスト"""

    def setUp(self):
        """テスト用データの準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # テスト用CSVデータ
        self.test_data = pd.DataFrame({
            'PRNumber': [1, 2, 3, 4, 5],
            'comment': [
                'This looks good to me',
                'Please fix this issue',
                'Needs more work please',
                'LGTM looks good',
                'Bot automated comment'
            ],
            'author': ['user1', 'user2', 'user3', 'user4', 'bot1'],
            '修正要求': [np.nan, 1.0, 1.0, np.nan, np.nan],
            '修正確認': [1.0, np.nan, np.nan, 1.0, np.nan]
        })

        # テスト用設定ファイル
        self.config_content = """
[organization]
bots = bot1, bot2
"""

        # テスト用レビューラベル
        self.review_labels = {
            "confirmation": {
                "positive": ["lgtm", "looks good to me"]
            }
        }

    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.utils.constants.LABELLED_CHANGE_COUNT', 10)
    def test_basic_keyword_extraction(self):
        """基本的なキーワード抽出"""
        # テストファイルの作成
        checklist_path = self.temp_path / "checklist.csv"
        config_path = self.temp_path / "config.ini"
        labels_path = self.temp_path / "labels.json"
        output_path = self.temp_path / "keywords.json"

        self.test_data.to_csv(checklist_path, index=False)

        with open(config_path, 'w') as f:
            f.write(self.config_content)

        with open(labels_path, 'w') as f:
            json.dump(self.review_labels, f)

        # キーワード抽出の実行
        extract_and_save_review_keywords(
            checklist_path=checklist_path,
            output_keywords_path=output_path,
            gerrymander_config_path=config_path,
            review_label_path=labels_path,
            min_comment_count=1,
            min_precision_ratio=0.5,
            ngram_min=1,
            ngram_max=3
        )

        # 結果の確認
        self.assertTrue(output_path.exists())
        with open(output_path, 'r') as f:
            result = json.load(f)

        self.assertIn('修正要求', result)
        self.assertIn('修正確認', result)
        self.assertIsInstance(result['修正要求'], list)
        self.assertIsInstance(result['修正確認'], list)

    def test_missing_checklist_file(self):
        """checklistファイルが存在しない場合"""
        checklist_path = self.temp_path / "nonexistent.csv"
        config_path = self.temp_path / "config.ini" 
        labels_path = self.temp_path / "labels.json"
        output_path = self.temp_path / "keywords.json"

        with open(config_path, 'w') as f:
            f.write(self.config_content)

        with open(labels_path, 'w') as f:
            json.dump(self.review_labels, f)

        # エラーログが出力されるが、例外は発生しない
        extract_and_save_review_keywords(
            checklist_path=checklist_path,
            output_keywords_path=output_path,
            gerrymander_config_path=config_path,
            review_label_path=labels_path
        )

        # 出力ファイルは作成されない
        self.assertFalse(output_path.exists())

    @patch('src.utils.constants.LABELLED_CHANGE_COUNT', 10)
    def test_bot_filtering(self):
        """ボットコメントのフィルタリング"""
        # ボットのコメントのみのデータ
        bot_data = pd.DataFrame({
            'PRNumber': [1, 2],
            'comment': ['Bot comment 1', 'Bot comment 2'],
            'author': ['bot1', 'bot1'],
            '修正要求': [1.0, np.nan],
            '修正確認': [np.nan, 1.0]
        })

        checklist_path = self.temp_path / "checklist.csv"
        config_path = self.temp_path / "config.ini"
        labels_path = self.temp_path / "labels.json"
        output_path = self.temp_path / "keywords.json"

        bot_data.to_csv(checklist_path, index=False)

        with open(config_path, 'w') as f:
            f.write(self.config_content)

        with open(labels_path, 'w') as f:
            json.dump(self.review_labels, f)

        extract_and_save_review_keywords(
            checklist_path=checklist_path,
            output_keywords_path=output_path,
            gerrymander_config_path=config_path,
            review_label_path=labels_path,
            min_comment_count=1,
            min_precision_ratio=0.5
        )

        # ボットコメントは除外されるため、キーワードは空になる
        with open(output_path, 'r') as f:
            result = json.load(f)

        self.assertEqual(result['修正要求'], [])
        self.assertEqual(result['修正確認'], [])

    @patch('src.utils.constants.LABELLED_CHANGE_COUNT', 10)
    def test_invalid_csv_columns(self):
        """無効なCSVカラムの処理"""
        invalid_data = pd.DataFrame({
            'wrong_column': [1, 2, 3],
            'another_wrong': ['a', 'b', 'c']
        })

        checklist_path = self.temp_path / "checklist.csv"
        config_path = self.temp_path / "config.ini"
        labels_path = self.temp_path / "labels.json"
        output_path = self.temp_path / "keywords.json"

        invalid_data.to_csv(checklist_path, index=False)

        with open(config_path, 'w') as f:
            f.write(self.config_content)

        with open(labels_path, 'w') as f:
            json.dump(self.review_labels, f)

        # エラーログが出力されるが、例外は発生しない
        extract_and_save_review_keywords(
            checklist_path=checklist_path,
            output_keywords_path=output_path,
            gerrymander_config_path=config_path,
            review_label_path=labels_path
        )

        # 出力ファイルは作成されない
        self.assertFalse(output_path.exists())


if __name__ == '__main__':
    unittest.main()
