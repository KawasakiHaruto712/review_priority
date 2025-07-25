"""
Integration tests using actual data files
"""
import pytest
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.features.change_metrics import (
    calculate_elapsed_time,
    check_test_code_presence
)
from src.features.bug_metrics import calculate_bug_fix_confidence
from src.features.refactoring_metrics import calculate_refactoring_confidence
from src.features.developer_metrics import (
    calculate_past_report_count,
    calculate_merge_rate
)
from src.features.project_metrics import (
    calculate_predictive_target_ticket_count,
    calculate_reviewed_lines_in_period
)


class TestIntegrationWithRealData:
    """Integration tests using real data files"""
    
    def setup_method(self):
        """Setup paths to real data"""
        from src.config.path import DEFAULT_DATA_DIR
        self.data_dir = Path(DEFAULT_DATA_DIR) / "openstack"
        self.projects = ["neutron", "nova", "cinder", "glance", "keystone", "swift"]
        self.analysis_time = datetime(2023, 1, 1, 0, 0, 0)

    def test_data_directory_exists(self):
        """Test that the data directory exists"""
        assert self.data_dir.exists(), f"Data directory {self.data_dir} does not exist"

    def test_project_directories_exist(self):
        """Test that project directories exist"""
        for project in self.projects:
            project_dir = self.data_dir / project
            if project_dir.exists():
                changes_dir = project_dir / "changes"
                assert changes_dir.exists(), f"Changes directory for {project} does not exist"

    def test_change_files_format(self):
        """Test that change files have the expected JSON format"""
        for project in self.projects:
            changes_dir = self.data_dir / project / "changes"
            if not changes_dir.exists():
                continue
            
            # Test first few change files
            change_files = list(changes_dir.glob("change_*.json"))[:3]
            
            for change_file in change_files:
                try:
                    with open(change_file, 'r', encoding='utf-8') as f:
                        change_data = json.load(f)
                    
                    # Check required fields exist
                    assert "change_number" in change_data
                    assert "project" in change_data
                    assert "created" in change_data
                    
                    # Test elapsed time calculation with real data
                    result = calculate_elapsed_time(change_data, self.analysis_time)
                    assert isinstance(result, (int, float))
                    
                    # Test test code presence check
                    test_result = check_test_code_presence(change_data)
                    assert test_result in [0, 1]
                    
                    # Test bug fix confidence
                    subject = change_data.get("subject", "")
                    message = change_data.get("message", "")
                    bug_confidence = calculate_bug_fix_confidence(subject, message)
                    assert bug_confidence in [0, 1, 2]
                    
                    # Test refactoring confidence
                    refactor_confidence = calculate_refactoring_confidence(subject, message)
                    assert refactor_confidence in [0, 1]
                    
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in file {change_file}")
                except Exception as e:
                    pytest.fail(f"Error processing {change_file}: {e}")

    def test_sample_calculations_on_real_data(self):
        """Test metric calculations on a sample of real data"""
        neutron_changes = self.data_dir / "neutron" / "changes"
        
        if not neutron_changes.exists():
            pytest.skip("Neutron changes directory not found")
        
        # Get first change file
        change_files = list(neutron_changes.glob("change_*.json"))
        if not change_files:
            pytest.skip("No change files found in neutron directory")
        
        with open(change_files[0], 'r', encoding='utf-8') as f:
            change_data = json.load(f)
        
        # Test all metric functions
        elapsed_time = calculate_elapsed_time(change_data, self.analysis_time)
        test_presence = check_test_code_presence(change_data)
        
        # Test new metrics
        subject = change_data.get("subject", "")
        message = change_data.get("message", "")
        bug_confidence = calculate_bug_fix_confidence(subject, message)
        refactor_confidence = calculate_refactoring_confidence(subject, message)
        
        # Results should be valid
        assert isinstance(elapsed_time, (int, float))
        assert test_presence in [0, 1]
        assert bug_confidence in [0, 1, 2]
        assert refactor_confidence in [0, 1]
        
        print(f"Sample results for {change_files[0].name}:")
        print(f"  Elapsed time: {elapsed_time}")
        print(f"  Test presence: {test_presence}")
        print(f"  Bug fix confidence: {bug_confidence}")
        print(f"  Refactoring confidence: {refactor_confidence}")

    def test_developer_metrics_with_sample_data(self):
        """Test developer metrics with sample data"""
        # Create sample DataFrame for testing
        sample_data = {
            'owner_email': ['dev1@example.com', 'dev1@example.com', 'dev2@example.com'],
            'created': [
                datetime(2022, 6, 1),
                datetime(2022, 8, 1),
                datetime(2022, 10, 1)
            ],
            'merged': [
                datetime(2022, 6, 10),
                pd.NaT,
                datetime(2022, 10, 15)
            ]
        }
        sample_df = pd.DataFrame(sample_data)
        
        # Test developer metrics
        past_count = calculate_past_report_count('dev1@example.com', sample_df, self.analysis_time)
        merge_rate = calculate_merge_rate('dev1@example.com', sample_df, self.analysis_time)
        
        assert isinstance(past_count, int)
        assert isinstance(merge_rate, float)
        assert 0.0 <= merge_rate <= 1.0
        
        print(f"Developer metrics test:")
        print(f"  Past report count: {past_count}")
        print(f"  Merge rate: {merge_rate}")

    def test_project_metrics_with_sample_data(self):
        """Test project metrics with sample data"""
        # Create sample DataFrame for testing
        sample_prs = {
            'created': [datetime(2022, 6, 1), datetime(2022, 8, 1), datetime(2022, 10, 1)],
            'merged': [datetime(2022, 6, 10), pd.NaT, pd.NaT],
            'updated': [datetime(2022, 6, 5), datetime(2022, 8, 5), datetime(2022, 10, 5)]
        }
        sample_df = pd.DataFrame(sample_prs)
        
        # Test project metrics
        open_count = calculate_predictive_target_ticket_count(sample_df, self.analysis_time)
        reviewed_count = calculate_reviewed_lines_in_period(sample_df, self.analysis_time, lookback_days=365)
        
        assert isinstance(open_count, int)
        assert isinstance(reviewed_count, int)
        assert open_count >= 0
        assert reviewed_count >= 0
        
        print(f"Project metrics test:")
        print(f"  Open ticket count: {open_count}")
        print(f"  Reviewed lines count: {reviewed_count}")

    def test_change_metrics_with_real_data(self):
        """Test change metrics functions with real change data"""
        from src.features.change_metrics import (
            calculate_lines_added, 
            calculate_lines_deleted, 
            calculate_files_changed
        )
        from src.config.path import DEFAULT_DATA_DIR
        
        # Use a known change file
        change_file_path = os.path.join(DEFAULT_DATA_DIR, 'openstack', 'neutron', 'changes', 'change_289165.json')
        if os.path.exists(change_file_path):
            with open(change_file_path, 'r', encoding='utf-8') as f:
                change_data = json.load(f)
            
            lines_added = calculate_lines_added(change_data)
            lines_deleted = calculate_lines_deleted(change_data)
            files_changed = calculate_files_changed(change_data)
            
            # These should be positive values for a real change
            assert lines_added >= 0
            assert lines_deleted >= 0
            assert files_changed > 0
            
            print(f"Real change data - Lines added: {lines_added}, Lines deleted: {lines_deleted}, Files changed: {files_changed}")
        else:
            pytest.skip("Real change data not available")
            
    def test_project_metrics_with_lines_info(self):
        """Test project metrics with line information extraction"""
        from src.features.project_metrics import add_lines_info_to_dataframe
        
        # Create a small sample DataFrame
        sample_data = {
            'change_number': [289165, 318542],
            'created': [datetime(2016, 3, 7), datetime(2017, 1, 1)],
            'updated': [datetime(2021, 5, 8), datetime(2017, 2, 1)]
        }
        sample_df = pd.DataFrame(sample_data)
        
        # Test the helper function
        df_with_lines = add_lines_info_to_dataframe(sample_df, 'neutron')
        
        # Check that new columns were added
        assert 'lines_added' in df_with_lines.columns
        assert 'lines_deleted' in df_with_lines.columns
        assert 'files_changed' in df_with_lines.columns
        
        # Check that values are reasonable
        for idx, row in df_with_lines.iterrows():
            assert row['lines_added'] >= 0
            assert row['lines_deleted'] >= 0
            assert row['files_changed'] >= 0
            
        print(f"Sample DataFrame with lines info:\n{df_with_lines[['change_number', 'lines_added', 'lines_deleted', 'files_changed']]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
