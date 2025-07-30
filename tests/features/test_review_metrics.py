"""
Tests for review_metrics.py module
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.features.review_metrics import ReviewStatusAnalyzer


class TestReviewStatusAnalyzer:
    """Test class for ReviewStatusAnalyzer"""
    
    def setup_method(self):
        """Setup test data for each test method"""
        # Sample keywords data
        self.sample_keywords = {
            "修正要求": ["fix", "change", "improve"],
            "修正確認": ["looks good", "approved", "lgtm"]
        }
        
        # Sample review labels data
        self.sample_labels = {
            "Code-Review": {
                "minus": ["-1", "-2"],
                "plus": ["+1", "+2"]
            },
            "Verified": {
                "minus": ["-1"],
                "plus": ["+1"]
            }
        }
        
        # Sample bot config
        self.sample_bot_config = """
[organization]
bots = bot1,bot2,cibot
"""

    def create_temp_files(self):
        """Create temporary files for testing"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create keywords file
        keywords_file = temp_path / "review_keywords.json"
        with open(keywords_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_keywords, f)
        
        # Create labels file  
        labels_file = temp_path / "review_label.json"
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_labels, f)
        
        # Create config file
        config_file = temp_path / "gerrymanderconfig.ini"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(self.sample_bot_config)
        
        return keywords_file, config_file, labels_file

    def test_init_loads_keywords_and_labels(self):
        """Test that ReviewStatusAnalyzer initializes with keywords and labels"""
        keywords_file, config_file, labels_file = self.create_temp_files()
        
        analyzer = ReviewStatusAnalyzer(
            extraction_keywords_path=keywords_file,
            gerrymander_config_path=config_file,
            review_label_path=labels_file
        )
        
        # Check keywords were loaded
        assert "fix" in analyzer.request_keywords
        assert "looks good" in analyzer.confirmation_keywords
        
        # Check labels were loaded
        assert "-1" in analyzer.code_review_minus_labels
        assert "+1" in analyzer.code_review_plus_labels

    def test_init_handles_missing_files(self):
        """Test that ReviewStatusAnalyzer handles missing files gracefully"""
        non_existent_path = Path("/non/existent/path.json")
        
        analyzer = ReviewStatusAnalyzer(
            extraction_keywords_path=non_existent_path,
            gerrymander_config_path=non_existent_path,
            review_label_path=non_existent_path
        )
        
        # Should have empty sets when files don't exist
        assert len(analyzer.request_keywords) == 0
        assert len(analyzer.confirmation_keywords) == 0

    def test_init_handles_invalid_json(self):
        """Test that ReviewStatusAnalyzer handles invalid JSON gracefully"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create invalid JSON file
        keywords_file = temp_path / "invalid_keywords.json"
        with open(keywords_file, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        
        labels_file = temp_path / "invalid_labels.json" 
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        
        config_file = temp_path / "config.ini"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("")
        
        analyzer = ReviewStatusAnalyzer(
            extraction_keywords_path=keywords_file,
            gerrymander_config_path=config_file,
            review_label_path=labels_file
        )
        
        # Should have empty sets when JSON is invalid
        assert len(analyzer.request_keywords) == 0
        assert len(analyzer.confirmation_keywords) == 0


if __name__ == "__main__":
    pytest.main([__file__])
