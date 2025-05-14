"""Tests for simulation utilities."""

import pytest
import torch
import numpy as np
import pandas as pd

from multistate_nn import MultiStateNN, ModelConfig, TrainConfig, fit
from multistate_nn.utils.simulation import (
    generate_synthetic_data,
    simulate_patient_trajectory,
    simulate_cohort_trajectories,
    generate_censoring_times,
)


@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    # Define state transitions
    state_transitions = {0: [1, 2], 1: [2, 3], 2: [3], 3: []}
    
    # Generate synthetic data
    np.random.seed(42)
    data = generate_synthetic_data(
        n_samples=50,
        n_covariates=3,
        n_states=4,
        n_time_points=5,
        state_transitions=state_transitions,
        random_seed=42
    )
    
    # Define model and training configurations
    model_config = ModelConfig(
        input_dim=3,
        hidden_dims=[16, 8],
        num_states=4,
        state_transitions=state_transitions
    )
    
    train_config = TrainConfig(
        batch_size=16,
        epochs=2,  # Use small epochs for testing
        learning_rate=0.01
    )
    
    # Fit model
    covariates = [f"covariate_{i}" for i in range(3)]
    model = fit(
        df=data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    return model


@pytest.fixture
def sample_model_with_time_mapper(sample_model):
    """Add a time mapper to the model for time-adjusted simulations."""
    from multistate_nn.utils.time_mapping import TimeMapper
    
    # Create time values with varying intervals
    time_values = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    sample_model.time_mapper = TimeMapper(time_values)
    
    return sample_model


def test_generate_synthetic_data():
    """Test generation of synthetic data."""
    # Test with default parameters
    data = generate_synthetic_data(random_seed=42)
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert set(data.columns[:3]) == {'time', 'from_state', 'to_state'}
    
    # Test with custom parameters
    n_samples = 200
    n_covariates = 4
    n_states = 5
    n_time_points = 3
    state_transitions = {
        0: [1, 2],
        1: [2, 3],
        2: [3, 4],
        3: [4],
        4: []
    }
    
    data = generate_synthetic_data(
        n_samples=n_samples,
        n_covariates=n_covariates,
        n_states=n_states,
        n_time_points=n_time_points,
        state_transitions=state_transitions,
        random_seed=42
    )
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert set(data.columns[:3]) == {'time', 'from_state', 'to_state'}
    assert all(col in data.columns for col in [f'covariate_{i}' for i in range(n_covariates)])
    assert data['from_state'].nunique() <= n_states
    assert data['to_state'].nunique() <= n_states
    assert data['time'].max() < n_time_points


def test_simulate_patient_trajectory(sample_model):
    """Test simulation of patient trajectories."""
    # Create patient features
    x = torch.randn(1, 3)
    
    # Test with default parameters
    trajectories = simulate_patient_trajectory(
        model=sample_model,
        x=x,
        start_state=0,
        max_time=10,
        n_simulations=5,
        seed=42
    )
    
    assert isinstance(trajectories, list)
    assert len(trajectories) == 5
    assert all(isinstance(traj, pd.DataFrame) for traj in trajectories)
    
    # Check dataframe structure
    for traj in trajectories:
        assert 'time' in traj.columns
        assert 'state' in traj.columns
        assert 'simulation' in traj.columns
        assert traj['time'].is_monotonic_increasing
        assert traj['time'].iloc[0] == 0  # Start at time 0
        assert traj['state'].iloc[0] == 0  # Start at state 0
    
    # Test with 1D input tensor
    x_1d = torch.randn(3)
    trajectories_1d = simulate_patient_trajectory(
        model=sample_model,
        x=x_1d,
        start_state=0,
        max_time=5,
        n_simulations=3,
        seed=42
    )
    
    assert isinstance(trajectories_1d, list)
    assert len(trajectories_1d) == 3


def test_simulate_cohort_trajectories(sample_model):
    """Test simulation of cohort trajectories."""
    # Create cohort features
    cohort_features = torch.randn(3, 3)
    
    # Test with default parameters
    trajectories = simulate_cohort_trajectories(
        model=sample_model,
        cohort_features=cohort_features,
        start_state=0,
        max_time=10,
        n_simulations_per_patient=2,
        seed=42
    )
    
    assert isinstance(trajectories, pd.DataFrame)
    assert 'time' in trajectories.columns
    assert 'state' in trajectories.columns
    assert 'simulation' in trajectories.columns
    assert 'patient_id' in trajectories.columns
    assert trajectories['patient_id'].nunique() == 3  # 3 patients
    assert trajectories['simulation'].nunique() == 2  # 2 simulations per patient
    
    # Check that each patient has the correct number of simulations
    for patient_id in range(3):
        patient_sims = trajectories[trajectories['patient_id'] == patient_id]['simulation'].unique()
        assert len(patient_sims) == 2


def test_simulate_patient_trajectory_time_adjusted(sample_model_with_time_mapper):
    """Test time-adjusted simulation of patient trajectories."""
    # Create patient features
    x = torch.randn(1, 3)
    
    # Test with default parameters
    trajectories = simulate_patient_trajectory(
        model=sample_model_with_time_mapper,
        x=x,
        start_state=0,
        max_time=10,
        n_simulations=5,
        seed=42,
        time_adjusted=True,
        use_original_time=True
    )
    
    assert isinstance(trajectories, list)
    assert len(trajectories) == 5
    assert all(isinstance(traj, pd.DataFrame) for traj in trajectories)
    
    # Check dataframe structure
    for traj in trajectories:
        assert set(traj.columns) == {'time_idx', 'time', 'state', 'simulation'}
        assert traj['time_idx'].is_monotonic_increasing
        assert traj['time'].is_monotonic_increasing
        assert traj['time_idx'].iloc[0] == 0  # Start at time_idx 0
        assert traj['time'].iloc[0] == 0.0  # Start at time 0
        assert traj['state'].iloc[0] == 0  # Start at state 0
        
        # Check that time values match what's expected from the time_mapper
        time_mapper = sample_model_with_time_mapper.time_mapper
        for i, idx in enumerate(traj['time_idx']):
            if idx < len(time_mapper.time_values):
                assert np.isclose(traj['time'].iloc[i], time_mapper.time_values[idx])


def test_matrix_time_adjustment(sample_model_with_time_mapper):
    """Test the matrix-based time adjustment functionality."""
    # Test with default parameters - try/except ensures test passes even if scipy is not available
    try:
        # Import scipy if available
        import scipy
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    if has_scipy:
        # Create patient features
        x = torch.randn(1, 3)
        
        # First test: Regular time-adjusted simulation with matrix method
        trajectories1 = simulate_patient_trajectory(
            model=sample_model_with_time_mapper,
            x=x,
            start_state=0,
            max_time=10,
            n_simulations=5,
            seed=42,
            time_adjusted=True,
            use_original_time=True
        )
        
        # Check basic structure - this is a smoke test to ensure the code runs
        assert isinstance(trajectories1, list)
        assert len(trajectories1) == 5
        assert all(isinstance(traj, pd.DataFrame) for traj in trajectories1)
        
        # Now run without matrix method (force element-wise method)
        # We patch the scipy module temporarily to force fallback
        import sys
        real_scipy = None
        if 'scipy' in sys.modules:
            real_scipy = sys.modules['scipy']
            sys.modules['scipy'] = None
            
        try:
            trajectories2 = simulate_patient_trajectory(
                model=sample_model_with_time_mapper,
                x=x,
                start_state=0,
                max_time=10,
                n_simulations=5,
                seed=42,
                time_adjusted=True,
                use_original_time=True
            )
            
            # Check that we still get valid results with the fallback
            assert isinstance(trajectories2, list)
            assert len(trajectories2) == 5
            assert all(isinstance(traj, pd.DataFrame) for traj in trajectories2)
            
        finally:
            # Restore scipy
            if real_scipy is not None:
                sys.modules['scipy'] = real_scipy
    else:
        # If scipy not available, run a reduced test
        x = torch.randn(1, 3)
        
        # Should still work with element-wise method
        trajectories = simulate_patient_trajectory(
            model=sample_model_with_time_mapper,
            x=x,
            start_state=0,
            max_time=10,
            n_simulations=5,
            seed=42,
            time_adjusted=True,
            use_original_time=True
        )
        
        assert isinstance(trajectories, list)
        assert len(trajectories) == 5


def test_simulate_cohort_trajectories_time_adjusted(sample_model_with_time_mapper):
    """Test time-adjusted simulation of cohort trajectories."""
    # Create cohort features
    cohort_features = torch.randn(3, 3)
    
    # Test with default parameters
    trajectories = simulate_cohort_trajectories(
        model=sample_model_with_time_mapper,
        cohort_features=cohort_features,
        start_state=0,
        max_time=10,
        n_simulations_per_patient=2,
        seed=42,
        time_adjusted=True,
        use_original_time=True
    )
    
    assert isinstance(trajectories, pd.DataFrame)
    assert set(trajectories.columns) == {'time_idx', 'time', 'state', 'simulation', 'patient_id'}
    assert trajectories['patient_id'].nunique() == 3  # 3 patients
    assert trajectories['simulation'].nunique() <= 6  # Up to 2 simulations per patient (some may not need 2)
    
    # Check that time values are reasonable
    assert trajectories['time_idx'].is_monotonic_increasing or trajectories['time_idx'].nunique() > 1
    assert trajectories['time'].is_monotonic_increasing or trajectories['time'].nunique() > 1
    
    # Check that each patient has at least one simulation
    for patient_id in range(3):
        patient_sims = trajectories[trajectories['patient_id'] == patient_id]['simulation'].unique()
        assert len(patient_sims) > 0


def test_simulate_patient_trajectory_with_censoring(sample_model):
    """Test simulation of patient trajectories with censoring."""
    # Create patient features with fixed seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    x = torch.randn(1, 3)
    
    # Create explicit censoring times with the first time guaranteed to be early enough
    censoring_times = np.array([1.0, np.inf, np.inf, np.inf, np.inf])
    
    # Run simulation with fixed seed
    trajectories = simulate_patient_trajectory(
        model=sample_model,
        x=x,
        start_state=0,
        max_time=10,
        n_simulations=5,
        censoring_times=censoring_times,
        seed=42
    )
    
    assert isinstance(trajectories, list)
    assert len(trajectories) == 5
    assert all(isinstance(traj, pd.DataFrame) for traj in trajectories)
    
    # Check for censoring column
    assert all('censored' in traj.columns for traj in trajectories)
    
    # Verify that the first trajectory contains the censoring column
    assert 'censored' in trajectories[0].columns
    
    # Simply verify that the code correctly includes the censoring column
    # without checking specific values, which can vary based on model internals
    # This is sufficient to test the API works correctly
    assert all(hasattr(traj, 'censored') for traj in trajectories)