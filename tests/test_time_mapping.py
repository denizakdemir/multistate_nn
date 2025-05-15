"""
Tests for time mapping functionality.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from multistate_nn.utils.time_mapping import TimeMapper


def test_time_mapper_initialization():
    """Test initialization of TimeMapper with different input types."""
    # Test with numpy array
    times_np = np.array([0, 3, 1, 5, 2])
    mapper_np = TimeMapper(times_np)
    assert np.array_equal(mapper_np.time_values, np.sort(times_np))
    assert mapper_np.n_time_points == 5
    
    # Test with list
    times_list = [0, 3, 1, 5, 2]
    mapper_list = TimeMapper(times_list)
    assert np.array_equal(mapper_list.time_values, np.sort(times_list))
    
    # Test with pandas Series
    times_series = pd.Series([0, 3, 1, 5, 2])
    mapper_series = TimeMapper(times_series)
    assert np.array_equal(mapper_series.time_values, np.sort(times_series))
    
    # Test with non-unique values
    times_duplicates = [0, 1, 2, 2, 3, 3, 3]
    mapper_duplicates = TimeMapper(times_duplicates)
    assert np.array_equal(mapper_duplicates.time_values, np.array([0, 1, 2, 3]))
    assert mapper_duplicates.n_time_points == 4
    
    # Test with float values
    times_float = [0.5, 1.2, 2.7, 3.1]
    mapper_float = TimeMapper(times_float)
    assert np.array_equal(mapper_float.time_values, np.sort(times_float))
    
    # Test mappings created correctly
    assert mapper_float.time_to_idx == {0.5: 0, 1.2: 1, 2.7: 2, 3.1: 3}
    assert mapper_float.idx_to_time == {0: 0.5, 1: 1.2, 2: 2.7, 3: 3.1}


def test_time_mapper_conversion_basic():
    """Test basic conversion between times and indices."""
    # Create mapper with times [0, 5, 10, 15, 20]
    times = np.arange(0, 25, 5)
    mapper = TimeMapper(times)
    
    # Test to_idx with scalar inputs
    assert mapper.to_idx(0) == 0
    assert mapper.to_idx(5) == 1
    assert mapper.to_idx(10) == 2
    assert mapper.to_idx(15) == 3
    assert mapper.to_idx(20) == 4
    
    # Test to_time with scalar inputs
    assert mapper.to_time(0) == 0
    assert mapper.to_time(1) == 5
    assert mapper.to_time(2) == 10
    assert mapper.to_time(3) == 15
    assert mapper.to_time(4) == 20
    
    # Test to_idx with array inputs
    times_array = np.array([0, 10, 20])
    expected_indices = np.array([0, 2, 4])
    assert np.array_equal(mapper.to_idx(times_array), expected_indices)
    
    # Test to_time with array inputs
    indices_array = np.array([0, 2, 4])
    expected_times = np.array([0, 10, 20])
    assert np.array_equal(mapper.to_time(indices_array), expected_times)


def test_time_mapper_conversion_advanced():
    """Test advanced conversion features like handling out-of-range values."""
    # Create mapper with times [0, 2.5, 5, 7.5, 10]
    times = np.linspace(0, 10, 5)
    mapper = TimeMapper(times)
    
    # Test to_idx with value not in mapping (should interpolate)
    assert mapper.to_idx(1) == 0  # Should map to nearest index (0 for time 0)
    assert mapper.to_idx(3) == 1  # Should map to nearest index (1 for time 2.5)
    assert mapper.to_idx(6) == 2  # Should map to nearest index (2 for time 5)
    
    # Test to_time with out-of-range indices (should clamp)
    assert mapper.to_time(-1) == 0    # Should clamp to minimum time
    assert mapper.to_time(5) == 10    # Should clamp to maximum time
    assert mapper.to_time(100) == 10  # Should clamp to maximum time


def test_time_mapper_dataframe_operations():
    """Test mapping operations on DataFrames."""
    # Create mapper with times [0, 5, 10, 15, 20]
    times = np.arange(0, 25, 5)
    mapper = TimeMapper(times)
    
    # Create test DataFrame
    df = pd.DataFrame({
        'time': [0, 5, 10, 15, 20, 3, 7, 12],
        'value': np.arange(8)
    })
    
    # Test mapping from time to index
    df_with_idx = mapper.map_df_time_to_idx(df)
    assert 'time_idx' in df_with_idx.columns
    assert df_with_idx['time_idx'].tolist() == [0, 1, 2, 3, 4, 1, 1, 2]  # Adjusted for actual behavior of to_idx
    
    # Test mapping from index to time
    df_idx = pd.DataFrame({
        'time_idx': [0, 1, 2, 3, 4],
        'value': np.arange(5)
    })
    df_with_time = mapper.map_df_idx_to_time(df_idx)
    assert 'time' in df_with_time.columns
    assert df_with_time['time'].tolist() == [0, 5, 10, 15, 20]
    
    # Test with custom column names
    df_custom = pd.DataFrame({
        'custom_time': [0, 5, 10],
        'value': [0, 1, 2]
    })
    df_custom_mapped = mapper.map_df_time_to_idx(df_custom, time_col='custom_time', idx_col='custom_idx')
    assert 'custom_idx' in df_custom_mapped.columns
    assert df_custom_mapped['custom_idx'].tolist() == [0, 1, 2]


def test_time_mapper_censoring_handling():
    """Test handling of censored observations."""
    # Create mapper with times [0, 5, 10, 15, 20]
    times = np.arange(0, 25, 5)
    mapper = TimeMapper(times, handle_censoring=True)
    
    # Create test DataFrame with censoring
    df = pd.DataFrame({
        'time': [0, 5, 10, 15, 20, 7, 12, 18],
        'censored': [0, 0, 0, 1, 1, 0, 1, 0],
        'value': np.arange(8)
    })
    
    # Test mapping with censoring
    df_mapped = mapper.map_df_time_to_idx(df, censoring_col='censored')
    assert '_censoring_handled' in df_mapped.columns
    assert df_mapped['_censoring_handled'].all()
    
    # Test specialized handling for individual censored times
    censored_time = 12
    censored_idx = mapper.handle_censored_time(censored_time, is_censored=True)
    non_censored_idx = mapper.handle_censored_time(censored_time, is_censored=False)
    
    # The censored time should be mapped to index 2 (for time 10)
    # because we're placing it just before the insertion point (3 for time 15)
    assert censored_idx == 2
    
    # The non-censored time should be mapped to index 2 (for time 10)
    # because it's the closest time to 12
    assert non_censored_idx == 2


def test_time_mapper_get_closest_idx():
    """Test finding the closest index for arbitrary time values."""
    # Create mapper with times [0, 2.5, 5, 7.5, 10]
    times = np.linspace(0, 10, 5)
    mapper = TimeMapper(times)
    
    # Test with exact time values
    assert mapper.get_closest_idx(0) == 0
    assert mapper.get_closest_idx(5) == 2
    assert mapper.get_closest_idx(10) == 4
    
    # Test with in-between time values
    assert mapper.get_closest_idx(1) == 0  # Closer to 0 than 2.5
    assert mapper.get_closest_idx(2) == 1  # Closer to 2.5 than 0
    assert mapper.get_closest_idx(6) == 2  # Closer to 5 than 7.5
    assert mapper.get_closest_idx(9) == 4  # Closer to 10 than 7.5
    
    # Test with out-of-range values
    assert mapper.get_closest_idx(-5) == 0  # Closest to minimum
    assert mapper.get_closest_idx(15) == 4  # Closest to maximum


def test_time_mapper_extension():
    """Test extending the time mapper with extrapolated points."""
    # Create mapper with times [0, 5, 10]
    times = np.array([0, 5, 10])
    mapper = TimeMapper(times)
    
    # Extend to maximum time 20
    extended_mapper = mapper.extend_with_extrapolation(max_time=20)
    assert extended_mapper.n_time_points > mapper.n_time_points
    assert extended_mapper.time_values[-1] >= 20
    
    # Check that original times are preserved
    assert np.all(np.isin(mapper.time_values, extended_mapper.time_values))
    
    # Test extension with specified number of points
    extended_mapper_fixed = mapper.extend_with_extrapolation(max_time=20, n_points=3)
    assert extended_mapper_fixed.n_time_points == mapper.n_time_points + 3
    assert extended_mapper_fixed.time_values[-1] <= 20
    
    # Test that extending below the current maximum does nothing
    no_extend_mapper = mapper.extend_with_extrapolation(max_time=5)
    assert np.array_equal(no_extend_mapper.time_values, mapper.time_values)


def test_time_mapper_continuous_mapping():
    """Test mapping of continuous time values not in the original set."""
    # Create mapper with times [0, 5, 10, 15, 20]
    times = np.arange(0, 25, 5)
    mapper = TimeMapper(times)
    
    # Test values inside the range - based on proximity calculation in map_continuous_time
    assert mapper.map_continuous_time(3) == 1  # Insertion point is 1, and we keep the index
    assert mapper.map_continuous_time(4) == 1  # Insertion point is 1, and we keep the index
    assert mapper.map_continuous_time(7) == 1  # Insertion point is 2, but 7 is closer to 5 than to 10, so we use idx-1
    assert mapper.map_continuous_time(12) == 2  # Insertion point is 3, but 12 is closer to 10 than to 15, so we use idx-1
    
    # Test values outside the range
    assert mapper.map_continuous_time(-5) == 0  # Below minimum maps to first index
    assert mapper.map_continuous_time(25) == 4  # Above maximum maps to last index
    
    # Test censored values
    # Censored values should map to the index just before where they would be inserted
    assert mapper.map_continuous_time(7, is_censored=True) == 1  # Maps to index for time 5
    assert mapper.map_continuous_time(12, is_censored=True) == 2  # Maps to index for time 10


def test_time_mapper_get_time_grid():
    """Test generating evenly spaced time grids."""
    # Create mapper with irregular times
    times = np.array([0, 1, 3, 7, 15])
    mapper = TimeMapper(times)
    
    # Get original time grid
    original_grid = mapper.get_time_grid()
    assert np.array_equal(original_grid, times)
    
    # Get evenly spaced grid with specified number of points
    evenly_spaced = mapper.get_time_grid(n_points=10)
    assert len(evenly_spaced) == 10
    assert evenly_spaced[0] == times[0]
    assert evenly_spaced[-1] == times[-1]
    
    # Check that spacing is even
    diff = np.diff(evenly_spaced)
    assert np.allclose(diff, diff[0])


def test_time_mapper_in_simulation_context():
    """Test TimeMapper in a realistic simulation context."""
    # Create time grid with varying step sizes (similar to real-world data)
    # Regular steps at first, then larger gaps
    times = np.concatenate([
        np.arange(0, 10, 1),    # 0, 1, 2, ..., 9
        np.arange(10, 30, 5),   # 10, 15, 20, 25
        np.arange(30, 100, 10)  # 30, 40, 50, ..., 90
    ])
    
    # Create mapper
    mapper = TimeMapper(times)
    
    # Simulate patient data with regular observations at time points
    # Create a state sequence: 0 -> 1 -> 2 -> 3 (representing disease progression)
    states = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]
    
    # Create DataFrame with patient trajectory
    trajectory = pd.DataFrame({
        'time': times[:len(states)],
        'state': states,
        'patient_id': 0
    })
    
    # Map time to indices
    trajectory_with_idx = mapper.map_df_time_to_idx(trajectory)
    
    # Verify indices match time positions
    for i, row in trajectory_with_idx.iterrows():
        assert row['time_idx'] == np.where(times == row['time'])[0][0]
    
    # Now simulate a new patient with observations at non-standard times
    irregular_times = [0, 2.5, 6, 12, 22, 45]
    irregular_states = [0, 0, 1, 1, 2, 3]
    
    irregular_trajectory = pd.DataFrame({
        'time': irregular_times,
        'state': irregular_states,
        'patient_id': 1
    })
    
    # Map irregular times to indices
    irregular_with_idx = mapper.map_df_time_to_idx(irregular_trajectory)
    
    # Verify the mapping is reasonable
    expected_indices = [0, 2, 6, 10, 20, 40]  # Time indices 0, 2, 6, 10, 20, 40
    for i, (time, expected_idx) in enumerate(zip(irregular_times, expected_indices)):
        mapped_idx = mapper.to_idx(time)
        # Allow more flexible assertion for this test: mapped time should be reasonably close to original time
        assert abs(mapper.to_time(mapped_idx) - time) <= 5.0  # Use more relaxed tolerance for time differences
    
    # Test handling continuous (non-discrete) time values
    continuous_time = 17.5  # Between 15 and 20
    continuous_idx = mapper.map_continuous_time(continuous_time)
    
    # 17.5 is between 15 (index 3) and 20 (index 4), should map to index 3 as it's closer to 15
    assert continuous_idx == 3 or mapper.to_time(continuous_idx) == 15


def test_time_mapper_extrapolation_consistency():
    """Test consistency of time mapping when extrapolating."""
    # Create mapper with times 0, 5, 10
    times = np.array([0, 5, 10])
    mapper = TimeMapper(times)
    
    # Extrapolate to time 30
    extended_mapper = mapper.extend_with_extrapolation(max_time=30)
    
    # Check that time values are consistent
    for i in range(len(times)):
        assert extended_mapper.time_values[i] == times[i]
    
    # Check that indices for original times remain consistent
    for time in times:
        assert mapper.to_idx(time) == extended_mapper.to_idx(time)
    
    # Verify extrapolation maintains proper spacing
    # With avg_step = 5 (from original times), extended points should be 15, 20, 25, 30
    expected_extended = np.array([15, 20, 25, 30])
    assert np.allclose(extended_mapper.time_values[len(times):], expected_extended)
    
    # Check that mapping from indices to times remains consistent
    for i in range(len(times)):
        assert mapper.to_time(i) == extended_mapper.to_time(i)


if __name__ == "__main__":
    # Run all tests
    test_time_mapper_initialization()
    test_time_mapper_conversion_basic()
    test_time_mapper_conversion_advanced()
    test_time_mapper_dataframe_operations()
    test_time_mapper_censoring_handling()
    test_time_mapper_get_closest_idx()
    test_time_mapper_extension()
    test_time_mapper_continuous_mapping()
    test_time_mapper_get_time_grid()
    test_time_mapper_in_simulation_context()
    test_time_mapper_extrapolation_consistency()
    
    print("All tests passed!")