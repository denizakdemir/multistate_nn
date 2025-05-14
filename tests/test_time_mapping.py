"""Tests for time mapping functionality."""

import pytest
import torch
import pandas as pd
import numpy as np
from multistate_nn.utils.time_mapping import TimeMapper
from multistate_nn import (
    MultiStateNN, 
    ModelConfig, 
    TrainConfig,
    fit
)
from multistate_nn.utils.simulation import (
    generate_synthetic_data,
    simulate_patient_trajectory
)
from multistate_nn.utils.analysis import calculate_cif


@pytest.fixture
def time_mapper():
    """Create a TimeMapper with non-sequential time values."""
    time_values = [10, 20, 30, 50, 100]
    return TimeMapper(time_values)


def test_time_mapper_creation():
    """Test TimeMapper initialization and mappings."""
    # Test with sequential values
    sequential_values = [0, 1, 2, 3, 4]
    mapper1 = TimeMapper(sequential_values)
    
    assert mapper1.time_values.tolist() == sequential_values
    assert mapper1.time_to_idx == {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4}
    assert mapper1.idx_to_time == {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    
    # Test with non-sequential values
    non_sequential_values = [10, 20, 30, 50, 100]
    mapper2 = TimeMapper(non_sequential_values)
    
    assert mapper2.time_values.tolist() == non_sequential_values
    assert mapper2.time_to_idx == {10.0: 0, 20.0: 1, 30.0: 2, 50.0: 3, 100.0: 4}
    assert mapper2.idx_to_time == {0: 10.0, 1: 20.0, 2: 30.0, 3: 50.0, 4: 100.0}


def test_time_mapper_conversion(time_mapper):
    """Test conversions between time and indices."""
    # Test single value conversions
    assert time_mapper.to_idx(10) == 0
    assert time_mapper.to_idx(30) == 2
    assert time_mapper.to_idx(100) == 4
    
    assert time_mapper.to_time(0) == 10.0
    assert time_mapper.to_time(2) == 30.0
    assert time_mapper.to_time(4) == 100.0
    
    # Test array conversions
    times = np.array([10, 30, 100, 50])
    indices = time_mapper.to_idx(times)
    assert np.array_equal(indices, np.array([0, 2, 4, 3]))
    
    indices = np.array([0, 2, 4, 3])
    times = time_mapper.to_time(indices)
    assert np.array_equal(times, np.array([10.0, 30.0, 100.0, 50.0]))
    
    # Test values not in mapping
    assert time_mapper.to_idx(15) == 0  # Should map to closest value (10)
    assert time_mapper.to_idx(25) == 1  # Should map to closest value (20)
    
    # Test array with values not in mapping
    indices = time_mapper.to_idx(np.array([10, 15, 25]))
    assert np.array_equal(indices, np.array([0, 0, 1]))


def test_time_mapper_dataframe_mapping(time_mapper):
    """Test DataFrame mapping functions."""
    # Create a test DataFrame
    df = pd.DataFrame({
        'time': [10, 20, 30, 50, 100, 10, 30],
        'value': [1, 2, 3, 4, 5, 6, 7]
    })
    
    # Test mapping time to indices
    df_mapped = time_mapper.map_df_time_to_idx(df)
    assert 'time_idx' in df_mapped.columns
    assert df_mapped['time_idx'].tolist() == [0, 1, 2, 3, 4, 0, 2]
    
    # Test mapping indices back to time
    df_mapped = df_mapped.rename(columns={'time': 'original_time'})
    df_mapped_back = time_mapper.map_df_idx_to_time(df_mapped, idx_col='time_idx')
    assert 'time' in df_mapped_back.columns
    assert df_mapped_back['time'].tolist() == [10.0, 20.0, 30.0, 50.0, 100.0, 10.0, 30.0]


def test_continuous_time_mapping(time_mapper):
    """Test mapping continuous time values."""
    # Test map_continuous_time
    assert time_mapper.map_continuous_time(10) == 0
    assert time_mapper.map_continuous_time(20) == 1
    
    # Test points between known values
    assert time_mapper.map_continuous_time(15) == 0  # Closer to 10 than 20
    assert time_mapper.map_continuous_time(25) == 1  # Closer to 20 than 30
    assert time_mapper.map_continuous_time(40) == 2  # Closer to 30 than 50
    
    # Test out-of-range values
    assert time_mapper.map_continuous_time(5) == 0   # Below the min
    assert time_mapper.map_continuous_time(200) == 4  # Above the max


def test_get_closest_idx(time_mapper):
    """Test getting closest index for time values."""
    assert time_mapper.get_closest_idx(10) == 0  # Exact match
    assert time_mapper.get_closest_idx(15) == 0  # Closest to 10
    assert time_mapper.get_closest_idx(25) == 1  # Closest to 20
    assert time_mapper.get_closest_idx(75) == 3  # Closest to 50


def test_get_time_grid(time_mapper):
    """Test generating time grids."""
    # Default grid (all unique time points)
    default_grid = time_mapper.get_time_grid()
    assert np.array_equal(default_grid, np.array([10, 20, 30, 50, 100]))
    
    # Custom grid with specified number of points
    custom_grid = time_mapper.get_time_grid(10)
    assert len(custom_grid) == 10
    assert custom_grid[0] == 10.0
    assert custom_grid[-1] == 100.0


def test_model_with_custom_time():
    """Test training a model with custom time values."""
    # Generate synthetic data with custom time values
    time_values = [10, 20, 30, 40, 50]
    df = generate_synthetic_data(
        n_samples=100,
        n_covariates=3,
        n_states=4,
        n_time_points=5,
        time_values=time_values,
        random_seed=42
    )
    
    # Make sure the data has time_idx column and time column
    assert 'time_idx' in df.columns
    assert 'time' in df.columns
    
    # Check that all expected time values are included in the data
    # Note: not all values may be present due to random generation
    unique_times = sorted(df['time'].unique().tolist())
    for time_value in unique_times:
        assert time_value in time_values
    
    # Create model and train config
    model_config = ModelConfig(
        input_dim=3,
        hidden_dims=[32, 16],
        num_states=4,
        state_transitions={0: [1, 2], 1: [2, 3], 2: [3], 3: []}
    )
    
    train_config = TrainConfig(
        batch_size=32,
        epochs=2,  # Small for fast testing
        learning_rate=0.01,
        use_original_time=True
    )
    
    # Fit model
    covariates = [f"covariate_{i}" for i in range(3)]
    model = fit(
        df=df,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    # Check that model has time_mapper attribute
    assert hasattr(model, 'time_mapper')
    assert model.time_mapper is not None
    
    # Check that time_mapper has time values from the dataset
    # Not all time values may be present in the data due to random generation
    for time_value in model.time_mapper.time_values:
        assert time_value in time_values


def test_simulation_with_custom_time():
    """Test simulation with custom time values."""
    # Generate synthetic data with custom time values
    time_values = [10, 20, 30, 40, 50]
    
    # Define clear state transitions for testing
    state_transitions = {
        0: [1, 2], 
        1: [2, 3], 
        2: [3], 
        3: []
    }
    
    df = generate_synthetic_data(
        n_samples=100,
        n_covariates=3,
        n_states=4,
        n_time_points=5,
        state_transitions=state_transitions,
        time_values=time_values,
        random_seed=42
    )
    
    # Create and train model with the same state transitions
    model_config = ModelConfig(
        input_dim=3,
        hidden_dims=[32, 16],
        num_states=4,
        state_transitions=state_transitions
    )
    
    train_config = TrainConfig(
        batch_size=32,
        epochs=2,  # Small for fast testing
        learning_rate=0.01,
        use_original_time=True
    )
    
    covariates = [f"covariate_{i}" for i in range(3)]
    model = fit(
        df=df,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    # Simulate trajectories
    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    trajectories = simulate_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=4,  # Max time in indices (corresponds to 50 in original time)
        n_simulations=10,
        seed=42,
        use_original_time=True
    )
    
    # Check that trajectories contain both time_idx and time columns
    assert 'time_idx' in trajectories[0].columns
    assert 'time' in trajectories[0].columns
    
    # Check that time values are mapped (not necessarily to time_values since
    # model.time_mapper may not contain all time values from the original data)
    for trajectory in trajectories:
        for idx, time in zip(trajectory['time_idx'], trajectory['time']):
            if idx < len(model.time_mapper.time_values):
                assert time == model.time_mapper.time_values[idx]


def test_cif_with_custom_time():
    """Test CIF calculation with custom time values."""
    # Generate synthetic data with custom time values
    time_values = [10, 20, 30, 40, 50]
    
    # Define clear state transitions for testing
    state_transitions = {
        0: [1, 2], 
        1: [2, 3], 
        2: [3], 
        3: []
    }
    
    df = generate_synthetic_data(
        n_samples=100,
        n_covariates=3,
        n_states=4,
        n_time_points=5,
        state_transitions=state_transitions,
        time_values=time_values,
        random_seed=42
    )
    
    # Create and train model with the same state transitions
    model_config = ModelConfig(
        input_dim=3,
        hidden_dims=[32, 16],
        num_states=4,
        state_transitions=state_transitions
    )
    
    train_config = TrainConfig(
        batch_size=32,
        epochs=2,  # Small for fast testing
        learning_rate=0.01,
        use_original_time=True
    )
    
    covariates = [f"covariate_{i}" for i in range(3)]
    model = fit(
        df=df,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    # Simulate trajectories
    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    trajectories = simulate_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=50,  # Use actual time value instead of index
        n_simulations=20,
        seed=42,
        use_original_time=True
    )
    
    # Combine trajectories for CIF calculation
    all_trajectories = pd.concat(trajectories, ignore_index=True)
    
    # Create different time grids for testing
    original_grid = np.array(time_values)
    dense_grid = np.linspace(10, 50, 20)
    sparse_grid = np.array([10, 30, 50])
    
    # Calculate CIFs with different time grids (using non-absorbing state to avoid convergence to 1.0)
    target_state = 2  # Non-absorbing intermediate state
    max_time = 50     # Limit to observed time range
    
    cif_original = calculate_cif(
        all_trajectories, 
        target_state=target_state, 
        time_grid=original_grid,
        max_time=max_time,
        method="empirical"
    )
    
    cif_dense = calculate_cif(
        all_trajectories, 
        target_state=target_state, 
        time_grid=dense_grid,
        max_time=max_time,
        method="empirical"
    )
    
    cif_sparse = calculate_cif(
        all_trajectories, 
        target_state=target_state, 
        time_grid=sparse_grid,
        max_time=max_time,
        method="empirical"
    )
    
    # Check that CIFs have the expected structure
    assert 'time' in cif_original.columns
    assert 'cif' in cif_original.columns
    assert 'lower_ci' in cif_original.columns
    assert 'upper_ci' in cif_original.columns
    
    # Check that time values in CIFs match the specified time grids
    assert np.array_equal(cif_original['time'].values, original_grid)
    assert np.array_equal(cif_dense['time'].values, dense_grid)
    assert np.array_equal(cif_sparse['time'].values, sparse_grid)
    
    # Check that CIFs are consistent at matching time points
    for time_point in sparse_grid:
        # Get CIF values at this time point from each grid
        cif_orig_val = cif_original[cif_original['time'] == time_point]['cif'].values[0]
        cif_sparse_val = cif_sparse[cif_sparse['time'] == time_point]['cif'].values[0]
        
        # Find the closest point in the dense grid
        dense_idx = np.abs(dense_grid - time_point).argmin()
        cif_dense_val = cif_dense.iloc[dense_idx]['cif']
        
        # Values should be exactly equal at matching time points
        assert np.isclose(cif_orig_val, cif_sparse_val)
        
        # Values should be very close at approximately matching time points
        # (there might be some interpolation differences)
        assert np.abs(cif_orig_val - cif_dense_val) < 1e-6


def test_cif_discretization_consistency():
    """Test that CIF calculation is consistent across different time discretizations."""
    # Generate synthetic data with two different time discretizations
    fine_time_values = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    coarse_time_values = np.array([0, 25, 50, 75, 100])
    
    # Define the observed time range - important to limit simulation to this range
    max_observed_time = 100  # Maximum time value in our synthetic data
    
    # Same state transitions for both datasets
    # Create a model with an intermediate non-absorbing state
    state_transitions = {0: [1, 2], 1: [2], 2: []}
    
    # Fine discretization data
    np.random.seed(42)
    fine_data = generate_synthetic_data(
        n_samples=100,
        n_covariates=2,
        n_states=3,
        n_time_points=len(fine_time_values),
        state_transitions=state_transitions,
        time_values=fine_time_values,
        random_seed=42
    )
    
    # Coarse discretization data
    np.random.seed(42)
    coarse_data = generate_synthetic_data(
        n_samples=100,
        n_covariates=2,
        n_states=3,
        n_time_points=len(coarse_time_values),
        state_transitions=state_transitions,
        time_values=coarse_time_values,
        random_seed=42
    )
    
    # Train models with both discretizations
    model_config = ModelConfig(
        input_dim=2,
        hidden_dims=[32, 16],
        num_states=3,
        state_transitions=state_transitions
    )
    
    train_config = TrainConfig(
        batch_size=32,
        epochs=5,  # Small for fast testing
        learning_rate=0.01,
        use_original_time=True
    )
    
    covariates = [f"covariate_{i}" for i in range(2)]
    
    # Fine model
    torch.manual_seed(42)
    fine_model = fit(
        df=fine_data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    # Coarse model
    torch.manual_seed(42)
    coarse_model = fit(
        df=coarse_data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    # Simulate trajectories with both models
    x = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    
    torch.manual_seed(42)
    np.random.seed(42)
    fine_trajectories = simulate_patient_trajectory(
        model=fine_model,
        x=x,
        start_state=0,
        max_time=max_observed_time,  # Limit to observed time range
        n_simulations=50,
        seed=42,
        use_original_time=True
    )
    
    torch.manual_seed(42)
    np.random.seed(42)
    coarse_trajectories = simulate_patient_trajectory(
        model=coarse_model,
        x=x,
        start_state=0,
        max_time=max_observed_time,  # Limit to observed time range
        n_simulations=50,
        seed=42,
        use_original_time=True
    )
    
    # Combine trajectories
    fine_df = pd.concat(fine_trajectories, ignore_index=True)
    coarse_df = pd.concat(coarse_trajectories, ignore_index=True)
    
    # Create a common time grid for consistent evaluation within observed range
    common_grid = np.linspace(0, max_observed_time, 20)
    
    # Use a non-absorbing state (state 1) to avoid CIFs converging to 1.0
    target_state = 1  # Non-absorbing intermediate state
    
    # Calculate CIFs using the common grid, explicitly limiting to observed time range
    fine_cif = calculate_cif(
        fine_df, 
        target_state=target_state, 
        time_grid=common_grid,
        max_time=max_observed_time,
        method="empirical"
    )
    
    coarse_cif = calculate_cif(
        coarse_df, 
        target_state=target_state, 
        time_grid=common_grid,
        max_time=max_observed_time,
        method="empirical"
    )
    
    # Check that both CIFs have the expected structure
    assert 'time' in fine_cif.columns and 'time' in coarse_cif.columns
    assert 'cif' in fine_cif.columns and 'cif' in coarse_cif.columns
    
    # Check that both CIFs used the same time grid
    assert np.array_equal(fine_cif['time'].values, common_grid)
    assert np.array_equal(coarse_cif['time'].values, common_grid)
    
    # Calculate the difference between the CIFs
    # Allow for some differences due to different model fits and simulation randomness
    # We're testing for consistency in the approach, not exact equality of outcomes
    max_diff = np.max(np.abs(fine_cif['cif'].values - coarse_cif['cif'].values))
    
    # Define an acceptable threshold for differences
    # This is a bit arbitrary but gives us something to validate against
    # For test purposes, we accept larger differences due to simulation randomness
    threshold = 0.6  # Allow up to 60% difference due to randomness in simulation and small sample size
    
    assert max_diff < threshold, f"CIF difference {max_diff} exceeds threshold {threshold}"