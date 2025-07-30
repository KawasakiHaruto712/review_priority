"""
Tests for project_metrics.py module
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.features.project_metrics import (
    _get_major_version,
    calculate_days_to_major_release,
    calculate_predictive_target_ticket_count,
    calculate_reviewed_lines_in_period
)


class TestProjectMetrics:
    """Test class for project metrics functions"""
    
    def setup_method(self):
        """Setup test data for each test method"""
        # Sample release data
        self.release_data = {
            'component': [
                'nova', 'nova', 'nova', 'nova',
                'neutron', 'neutron', 'neutron',
                'cinder', 'cinder'
            ],
            'version': [
                '13.0.0', '13.1.0', '14.0.0', '15.0.0',
                '10.0.0', '10.1.0', '11.0.0',
                '8.0.0', '9.0.0'
            ],
            'release_date': [
                datetime(2023, 1, 1),
                datetime(2023, 3, 1),
                datetime(2023, 6, 1),
                datetime(2023, 9, 1),
                datetime(2023, 2, 1),
                datetime(2023, 4, 1),
                datetime(2023, 7, 1),
                datetime(2023, 1, 15),
                datetime(2023, 5, 1)
            ]
        }
        self.releases_df = pd.DataFrame(self.release_data)
        
        # Sample PR data
        self.pr_data = {
            'created': [
                datetime(2023, 1, 1),
                datetime(2023, 1, 15),
                datetime(2023, 2, 1),
                datetime(2023, 2, 15),
                datetime(2023, 3, 1)
            ],
            'merged': [
                datetime(2023, 1, 10),   # merged
                pd.NaT,                  # not merged
                datetime(2023, 2, 20),   # merged
                pd.NaT,                  # not merged
                pd.NaT                   # not merged
            ],
            'updated': [
                datetime(2023, 1, 5),
                datetime(2023, 1, 20),
                datetime(2023, 2, 5),
                datetime(2023, 2, 25),
                datetime(2023, 3, 5)
            ]
        }
        self.prs_df = pd.DataFrame(self.pr_data)
        
        self.analysis_time = datetime(2023, 3, 15)

    def test_get_major_version(self):
        """Test major version extraction function"""
        assert _get_major_version("13.0.0") == 13
        assert _get_major_version("13.1.0") == 13
        assert _get_major_version("14.0.0.0b3") == 14
        assert _get_major_version("2.0") == 2
        assert _get_major_version("1") == 1
        
        # Invalid cases
        assert _get_major_version("invalid") is None
        assert _get_major_version("") is None
        assert _get_major_version(None) is None
        assert _get_major_version(123) is None

    def test_calculate_days_to_major_release(self):
        """Test days to major release calculation"""
        # Analysis time: 2023-03-15
        # For nova: current major should be 13 (released 2023-01-01, 13.1.0 on 2023-03-01)
        # Next major: 14.0.0 on 2023-06-01
        result = calculate_days_to_major_release(self.analysis_time, 'nova', self.releases_df)
        expected_days = (datetime(2023, 6, 1) - self.analysis_time).days
        assert result == expected_days

    def test_calculate_days_to_major_release_no_future_major(self):
        """Test when there's no future major release"""
        # Create a scenario where the component has no future major releases
        limited_releases = self.releases_df[self.releases_df['component'] == 'nova'].copy()
        limited_releases = limited_releases[limited_releases['release_date'] <= self.analysis_time]
        
        result = calculate_days_to_major_release(self.analysis_time, 'nova', limited_releases)
        assert result == -1.0

    def test_calculate_days_to_major_release_nonexistent_component(self):
        """Test with non-existent component"""
        result = calculate_days_to_major_release(self.analysis_time, 'nonexistent', self.releases_df)
        assert result == -1.0

    def test_calculate_days_to_major_release_analysis_after_release(self):
        """Test when analysis time is after the next major release"""
        # Use analysis time after the next major release
        future_analysis_time = datetime(2023, 10, 1)
        result = calculate_days_to_major_release(future_analysis_time, 'nova', self.releases_df)
        # Should return -1.0 since no future major release is found
        assert result == -1.0

    def test_calculate_predictive_target_ticket_count(self):
        """Test predictive target ticket count calculation"""
        # Analysis time: 2023-03-15
        # PRs created before analysis time: 5 (all of them)
        # PRs merged before analysis time: 2 (Jan 1 and Feb 1 PRs)
        # Open PRs: 3 (remaining unmerged)
        
        result = calculate_predictive_target_ticket_count(self.prs_df, self.analysis_time)
        assert result == 3

    def test_calculate_predictive_target_ticket_count_all_merged(self):
        """Test when all PRs are merged"""
        all_merged_data = self.pr_data.copy()
        all_merged_data['merged'] = [
            datetime(2023, 1, 10),
            datetime(2023, 1, 25),
            datetime(2023, 2, 20),
            datetime(2023, 2, 28),
            datetime(2023, 3, 10)
        ]
        all_merged_df = pd.DataFrame(all_merged_data)
        
        result = calculate_predictive_target_ticket_count(all_merged_df, self.analysis_time)
        assert result == 0

    def test_calculate_predictive_target_ticket_count_none_merged(self):
        """Test when no PRs are merged"""
        none_merged_data = self.pr_data.copy()
        none_merged_data['merged'] = [pd.NaT] * len(none_merged_data['merged'])
        none_merged_df = pd.DataFrame(none_merged_data)
        
        result = calculate_predictive_target_ticket_count(none_merged_df, self.analysis_time)
        assert result == 5  # All 5 PRs are open

    def test_calculate_predictive_target_ticket_count_future_created(self):
        """Test with PRs created after analysis time"""
        future_pr_data = self.pr_data.copy()
        future_pr_data['created'].append(datetime(2023, 4, 1))  # After analysis time
        future_pr_data['merged'].append(pd.NaT)
        future_pr_data['updated'].append(datetime(2023, 4, 5))
        future_df = pd.DataFrame(future_pr_data)
        
        # Should not count the future PR
        result = calculate_predictive_target_ticket_count(future_df, self.analysis_time)
        assert result == 3

    def test_calculate_reviewed_lines_in_period_default(self):
        """Test reviewed lines calculation with default 14-day period"""
        # Analysis time: 2023-03-15
        # 14 days back: 2023-03-01
        # Since self.prs_df doesn't have line info, should return 0
        
        result = calculate_reviewed_lines_in_period(self.prs_df, self.analysis_time)
        assert result == 0

    def test_calculate_reviewed_lines_in_period_custom_period(self):
        """Test reviewed lines calculation with custom period"""
        # Test with 30-day lookback period
        result = calculate_reviewed_lines_in_period(self.prs_df, self.analysis_time, lookback_days=30)
        # Since self.prs_df doesn't have line info, should return 0
        assert result == 0
        
        # Test with 7-day lookback period
        result = calculate_reviewed_lines_in_period(self.prs_df, self.analysis_time, lookback_days=7)
        # Since self.prs_df doesn't have line info, should return 0
        assert result == 0

    def test_calculate_reviewed_lines_in_period_no_activity(self):
        """Test when there's no activity in the specified period"""
        # Use analysis time far in the future
        future_analysis_time = datetime(2023, 12, 1)
        result = calculate_reviewed_lines_in_period(self.prs_df, future_analysis_time, lookback_days=14)
        assert result == 0

    def test_calculate_metrics_with_empty_dataframes(self):
        """Test all metrics with empty dataframes"""
        empty_releases_df = pd.DataFrame({'component': [], 'version': [], 'release_date': []})
        empty_prs_df = pd.DataFrame({'created': [], 'merged': [], 'updated': []})
        
        result = calculate_days_to_major_release(self.analysis_time, 'nova', empty_releases_df)
        assert result == -1.0
        
        result = calculate_predictive_target_ticket_count(empty_prs_df, self.analysis_time)
        assert result == 0
        
        result = calculate_reviewed_lines_in_period(empty_prs_df, self.analysis_time)
        assert result == 0

    def test_calculate_days_to_major_release_version_edge_cases(self):
        """Test edge cases for version parsing in major release calculation"""
        edge_release_data = {
            'component': ['test', 'test', 'test', 'test'],
            'version': ['1.0.0', 'invalid', '2.0.0', '1.5.0'],
            'release_date': [
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                datetime(2023, 6, 1),
                datetime(2023, 3, 1)
            ]
        }
        edge_df = pd.DataFrame(edge_release_data)
        
        # Should skip invalid version and find next major (2.0.0)
        result = calculate_days_to_major_release(self.analysis_time, 'test', edge_df)
        expected_days = (datetime(2023, 6, 1) - self.analysis_time).days
        assert result == expected_days

    def test_calculate_reviewed_lines_in_period_with_lines_info(self):
        """Test reviewed lines calculation when line information is available"""
        # Create test data with line information
        pr_data_with_lines = {
            'created': [
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                datetime(2023, 3, 1)
            ],
            'updated': [
                datetime(2023, 2, 25),  # within 30-day period
                datetime(2023, 3, 5),   # within 30-day period  
                datetime(2023, 1, 5)    # outside 30-day period
            ],
            'lines_added': [50, 30, 20],
            'lines_deleted': [10, 5, 8]
        }
        prs_df_with_lines = pd.DataFrame(pr_data_with_lines)
        
        # Test with 30-day lookback - should include first two PRs
        result = calculate_reviewed_lines_in_period(prs_df_with_lines, self.analysis_time, lookback_days=30)
        expected = (50 + 10) + (30 + 5)  # (lines_added + lines_deleted) for first two PRs
        assert result == expected  # 95 total lines
        
    def test_calculate_reviewed_lines_in_period_without_lines_info(self):
        """Test reviewed lines calculation returns 0 when no line info"""
        # Use existing PR data without line information
        result = calculate_reviewed_lines_in_period(self.prs_df, self.analysis_time, lookback_days=30)
        # Should return 0 when line info is not available
        assert result == 0
        
    def test_add_lines_info_to_dataframe_empty(self):
        """Test add_lines_info_to_dataframe with empty DataFrame"""
        from src.features.project_metrics import add_lines_info_to_dataframe
        
        empty_df = pd.DataFrame()
        result = add_lines_info_to_dataframe(empty_df, 'neutron')
        assert len(result) == 0
        
    def test_add_lines_info_to_dataframe_missing_change_numbers(self):
        """Test add_lines_info_to_dataframe with missing change numbers"""
        from src.features.project_metrics import add_lines_info_to_dataframe
        
        test_df = pd.DataFrame({
            'change_number': [None, 999999999],  # None and non-existent change
            'created': [datetime(2023, 1, 1), datetime(2023, 2, 1)]
        })
        
        result = add_lines_info_to_dataframe(test_df, 'neutron')
        assert 'lines_added' in result.columns
        assert 'lines_deleted' in result.columns
        assert 'files_changed' in result.columns
        assert result['lines_added'].iloc[0] == 0  # None case
        assert result['lines_added'].iloc[1] == 0  # Non-existent file case


if __name__ == "__main__":
    pytest.main([__file__])
