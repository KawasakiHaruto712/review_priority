"""
Tests for developer_metrics.py module
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.features.developer_metrics import (
    calculate_past_report_count,
    calculate_recent_report_count,
    calculate_merge_rate,
    calculate_recent_merge_rate
)


class TestDeveloperMetrics:
    """Test class for developer metrics functions"""
    
    def setup_method(self):
        """Setup test data for each test method"""
        # Create sample dataframe with PR data
        self.sample_data = {
            'owner_email': [
                'dev1@example.com',
                'dev1@example.com', 
                'dev2@example.com',
                'dev1@example.com',
                'dev3@example.com'
            ],
            'created': [
                datetime(2023, 1, 1),
                datetime(2023, 1, 15),
                datetime(2023, 2, 1),
                datetime(2023, 2, 15),
                datetime(2023, 3, 1)
            ],
            'merged': [
                datetime(2023, 1, 5),    # merged
                pd.NaT,                  # not merged
                datetime(2023, 2, 10),   # merged
                datetime(2023, 2, 20),   # merged
                pd.NaT                   # not merged
            ]
        }
        self.df = pd.DataFrame(self.sample_data)
        self.analysis_time = datetime(2023, 3, 15)

    def test_calculate_past_report_count(self):
        """Test past report count calculation"""
        # dev1 has 3 PRs before analysis time
        result = calculate_past_report_count('dev1@example.com', self.df, self.analysis_time)
        assert result == 3
        
        # dev2 has 1 PR before analysis time
        result = calculate_past_report_count('dev2@example.com', self.df, self.analysis_time)
        assert result == 1
        
        # dev3 has 1 PR before analysis time
        result = calculate_past_report_count('dev3@example.com', self.df, self.analysis_time)
        assert result == 1
        
        # Non-existent developer
        result = calculate_past_report_count('nonexistent@example.com', self.df, self.analysis_time)
        assert result == 0

    def test_calculate_recent_report_count_default_3_months(self):
        """Test recent report count with default 3 months lookback"""
        # All PRs are within 3 months of analysis time
        result = calculate_recent_report_count('dev1@example.com', self.df, self.analysis_time)
        assert result == 3
        
        result = calculate_recent_report_count('dev2@example.com', self.df, self.analysis_time)
        assert result == 1

    def test_calculate_recent_report_count_custom_period(self):
        """Test recent report count with custom lookback period"""
        # Only PRs within 1 month (30 days) should be counted
        result = calculate_recent_report_count('dev1@example.com', self.df, self.analysis_time, lookback_months=1)
        assert result == 1  # Only the Feb 15 PR
        
        # No PRs within 1 week
        result = calculate_recent_report_count('dev1@example.com', self.df, self.analysis_time, lookback_months=0)
        assert result == 0

    def test_calculate_merge_rate(self):
        """Test merge rate calculation"""
        # dev1: 3 PRs total, 2 merged = 2/3 ≈ 0.667
        result = calculate_merge_rate('dev1@example.com', self.df, self.analysis_time)
        assert abs(result - 2/3) < 0.001
        
        # dev2: 1 PR, 1 merged = 1.0
        result = calculate_merge_rate('dev2@example.com', self.df, self.analysis_time)
        assert result == 1.0
        
        # dev3: 1 PR, 0 merged = 0.0
        result = calculate_merge_rate('dev3@example.com', self.df, self.analysis_time)
        assert result == 0.0
        
        # Non-existent developer should return 0.0
        result = calculate_merge_rate('nonexistent@example.com', self.df, self.analysis_time)
        assert result == 0.0

    def test_calculate_recent_merge_rate(self):
        """Test recent merge rate calculation"""
        # dev1 recent (within 3 months): 3 PRs, 2 merged
        result = calculate_recent_merge_rate('dev1@example.com', self.df, self.analysis_time)
        assert abs(result - 2/3) < 0.001
        
        # dev2 recent: 1 PR, 1 merged
        result = calculate_recent_merge_rate('dev2@example.com', self.df, self.analysis_time)
        assert result == 1.0

    def test_calculate_merge_rate_with_future_merge_dates(self):
        """Test merge rate when some merges happen after analysis time"""
        # Create data where some merges are after analysis time
        future_data = self.sample_data.copy()
        future_data['merged'] = [
            datetime(2023, 1, 5),    # merged before analysis
            datetime(2023, 4, 1),    # merged after analysis (should not count)
            datetime(2023, 2, 10),   # merged before analysis
            pd.NaT,                  # not merged
            pd.NaT                   # not merged
        ]
        future_df = pd.DataFrame(future_data)
        
        # dev1: 3 PRs total, only 1 merged before analysis time
        result = calculate_merge_rate('dev1@example.com', future_df, self.analysis_time)
        assert abs(result - 1/3) < 0.001

    def test_calculate_metrics_with_empty_dataframe(self):
        """Test all metrics with empty dataframe"""
        empty_df = pd.DataFrame({'owner_email': [], 'created': [], 'merged': []})
        
        result = calculate_past_report_count('dev1@example.com', empty_df, self.analysis_time)
        assert result == 0
        
        result = calculate_recent_report_count('dev1@example.com', empty_df, self.analysis_time)
        assert result == 0
        
        result = calculate_merge_rate('dev1@example.com', empty_df, self.analysis_time)
        assert result == 0.0
        
        result = calculate_recent_merge_rate('dev1@example.com', empty_df, self.analysis_time)
        assert result == 0.0

    def test_calculate_metrics_with_edge_dates(self):
        """Test metrics with edge case dates"""
        # PRs exactly at analysis time
        edge_data = {
            'owner_email': ['dev1@example.com', 'dev1@example.com'],
            'created': [self.analysis_time, self.analysis_time],
            'merged': [self.analysis_time, pd.NaT]
        }
        edge_df = pd.DataFrame(edge_data)
        
        # PRs created exactly at analysis time should be included
        result = calculate_past_report_count('dev1@example.com', edge_df, self.analysis_time)
        assert result == 2
        
        # Merges exactly at analysis time should be included
        result = calculate_merge_rate('dev1@example.com', edge_df, self.analysis_time)
        assert result == 0.5  # 1 merged out of 2 total

    def test_calculate_metrics_time_boundaries(self):
        """Test metrics at time boundaries for recent calculations"""
        # Test with analysis time exactly 3 months after some PRs
        boundary_analysis_time = datetime(2023, 4, 1)  # Exactly 3 months after Jan 1
        
        result = calculate_recent_report_count('dev1@example.com', self.df, boundary_analysis_time, lookback_months=3)
        # Should include PRs from Jan 1 onward (90 days back from Apr 1)
        assert result >= 2  # At least the Feb PRs should be included


if __name__ == "__main__":
    pytest.main([__file__])
