"""
Tests for change_metrics.py module
"""
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open

from src.features.change_metrics import (
    calculate_lines_added,
    calculate_lines_deleted,
    calculate_files_changed,
    calculate_elapsed_time,
    calculate_revision_count,
    check_test_code_presence
)


class TestChangeMetrics:
    """Test class for change metrics functions"""
    
    def setup_method(self):
        """Setup test data for each test method"""
        # Sample change data based on actual structure
        self.sample_change_data = {
            "change_number": 12345,
            "project": "test/project",
            "created": "2023-01-01 10:00:00.000000000",
            "updated": "2023-01-02 15:30:00.000000000",
            "subject": "Fix bug in authentication module",
            "message": "This change fixes authentication issues",
            "commit_ids": ["abc123def", "def456ghi"]
        }
        
        # Sample commit data
        self.sample_commit_data = {
            "commit": {
                "committer": {
                    "date": "2023-01-01 12:00:00.000000000"
                }
            }
        }
        
        self.analysis_time = datetime(2023, 1, 2, 16, 0, 0)

    def test_calculate_lines_added_returns_zero(self):
        """Test that lines added calculation returns 0 when no file_changes"""
        result = calculate_lines_added(self.sample_change_data)
        assert result == 0
        
    def test_calculate_lines_added_with_file_changes(self):
        """Test lines added calculation with actual file_changes data"""
        change_data_with_files = {
            **self.sample_change_data,
            "file_changes": {
                "file1.py": {"lines_inserted": 10, "lines_deleted": 2},
                "file2.py": {"lines_inserted": 5, "lines_deleted": 1},
                "file3.py": {"lines_inserted": 0, "lines_deleted": 3}
            }
        }
        result = calculate_lines_added(change_data_with_files)
        assert result == 15  # 10 + 5 + 0

    def test_calculate_lines_deleted_returns_zero(self):
        """Test that lines deleted calculation returns 0 when no file_changes"""
        result = calculate_lines_deleted(self.sample_change_data)
        assert result == 0
        
    def test_calculate_lines_deleted_with_file_changes(self):
        """Test lines deleted calculation with actual file_changes data"""
        change_data_with_files = {
            **self.sample_change_data,
            "file_changes": {
                "file1.py": {"lines_inserted": 10, "lines_deleted": 2},
                "file2.py": {"lines_inserted": 5, "lines_deleted": 1},
                "file3.py": {"lines_inserted": 0, "lines_deleted": 3}
            }
        }
        result = calculate_lines_deleted(change_data_with_files)
        assert result == 6  # 2 + 1 + 3

    def test_calculate_files_changed_returns_zero(self):
        """Test that files changed calculation returns 0 when no files"""
        result = calculate_files_changed(self.sample_change_data)
        assert result == 0
        
    def test_calculate_files_changed_with_files(self):
        """Test files changed calculation with actual files data"""
        change_data_with_files = {
            **self.sample_change_data,
            "files": ["file1.py", "file2.py", "file3.py"]
        }
        result = calculate_files_changed(change_data_with_files)
        assert result == 3

    def test_calculate_elapsed_time_valid_data(self):
        """Test elapsed time calculation with valid data"""
        result = calculate_elapsed_time(self.sample_change_data, self.analysis_time)
        # Expected: 2023-01-02 16:00:00 - 2023-01-01 10:00:00 = 30 hours = 1800 minutes
        expected = 30 * 60  # 30 hours in minutes
        assert result == expected

    def test_calculate_elapsed_time_missing_created(self):
        """Test elapsed time calculation with missing created timestamp"""
        invalid_data = {**self.sample_change_data}
        del invalid_data["created"]
        
        result = calculate_elapsed_time(invalid_data, self.analysis_time)
        assert result == -1.0

    def test_calculate_elapsed_time_future_analysis_time(self):
        """Test elapsed time calculation when analysis time is before creation"""
        past_analysis_time = datetime(2022, 12, 31, 10, 0, 0)
        
        result = calculate_elapsed_time(self.sample_change_data, past_analysis_time)
        assert result == -1.0

    def test_calculate_elapsed_time_invalid_created_format(self):
        """Test elapsed time calculation with invalid created timestamp format"""
        invalid_data = {**self.sample_change_data}
        invalid_data["created"] = "invalid-date-format"
        
        result = calculate_elapsed_time(invalid_data, self.analysis_time)
        assert result == -1.0

    @patch('src.features.change_metrics.DEFAULT_DATA_DIR')
    def test_calculate_revision_count_with_existing_commits(self, mock_data_dir):
        """Test revision count calculation with existing commit files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_data_dir.__truediv__ = lambda self, other: temp_path / other
            
            # Create test directory structure
            commits_dir = temp_path / "openstack" / "test" / "project" / "commits"
            commits_dir.mkdir(parents=True)
            
            # Create commit files
            commit_file1 = commits_dir / "commit_12345_abc123de.json"
            commit_file2 = commits_dir / "commit_12345_def456gh.json"
            
            commit_file1.write_text(json.dumps(self.sample_commit_data))
            commit_file2.write_text(json.dumps(self.sample_commit_data))
            
            # Mock the path resolution
            with patch('src.features.change_metrics.DEFAULT_DATA_DIR', temp_path):
                result = calculate_revision_count(self.sample_change_data, self.analysis_time)
                assert result == 2

    def test_calculate_revision_count_missing_data(self):
        """Test revision count calculation with missing project or change_number"""
        invalid_data = {**self.sample_change_data}
        del invalid_data["project"]
        
        result = calculate_revision_count(invalid_data, self.analysis_time)
        assert result == 0

    def test_check_test_code_presence_with_test_keywords(self):
        """Test test code presence detection with test-related keywords"""
        test_data = {
            **self.sample_change_data,
            "subject": "Add unit tests for authentication module",
            "message": "This change adds comprehensive unittest coverage"
        }
        
        result = check_test_code_presence(test_data)
        assert result == 1

    def test_check_test_code_presence_without_test_keywords(self):
        """Test test code presence detection without test-related keywords"""
        result = check_test_code_presence(self.sample_change_data)
        assert result == 0

    def test_check_test_code_presence_case_insensitive(self):
        """Test test code presence detection is case insensitive"""
        test_data = {
            **self.sample_change_data,
            "subject": "Add TESTING for new feature",
            "message": "This change includes PyTest cases"
        }
        
        result = check_test_code_presence(test_data)
        assert result == 1


if __name__ == "__main__":
    pytest.main([__file__])
