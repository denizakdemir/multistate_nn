"""Tests for censoring functionality in MultiStateNN models."""

import pytest
import numpy as np
import pandas as pd
import torch
from multistate_nn import ModelConfig, TrainConfig, fit
from multistate_nn.utils import (
    generate_synthetic_data,
    generate_censoring_times,
    simulate_patient_trajectory,
    simulate_cohort_trajectories,
    calculate_cif,
)
from multistate_nn.utils.aalen_johansen import (
    prepare_event_data,
    aalen_johansen_estimator,
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Define a simple state transition structure
    state_transitions = {
        0: [1, 2],
        1: [2],
        2: []
    }
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data
    data = generate_synthetic_data(
        n_samples=100,
        n_covariates=2,
        n_states=3,
        n_time_points=5,
        state_transitions=state_transitions,
        random_seed=42
    )
    
    # Add a censoring column
    data['censored'] = np.random.choice([0, 1], size=len(data), p=[0.7, 0.3])
    
    return data, state_transitions


def test_censoring_in_prepare_data(sample_data):
    """Test that prepare_data function correctly handles censoring column."""
    from multistate_nn.train import prepare_data
    
    # Unpack sample data
    data, _ = sample_data
    
    # List of covariates
    covariates = [f"covariate_{i}" for i in range(2)]
    
    # Test without censoring column
    x, time_idx, from_state, to_state = prepare_data(
        data, covariates, time_col="time", censoring_col=None
    )
    
    # Check shapes
    assert x.shape == (len(data), 2)
    assert time_idx.shape == (len(data),)
    assert from_state.shape == (len(data),)
    assert to_state.shape == (len(data),)
    
    # Test with censoring column
    x, time_idx, from_state, to_state, is_censored = prepare_data(
        data, covariates, time_col="time", censoring_col="censored"
    )
    
    # Check shapes
    assert x.shape == (len(data), 2)
    assert time_idx.shape == (len(data),)
    assert from_state.shape == (len(data),)
    assert to_state.shape == (len(data),)
    assert is_censored.shape == (len(data),)
    
    # Check censoring values
    assert is_censored.sum().item() == data['censored'].sum()


def test_censoring_in_fit(sample_data):
    """Test fitting a model with censoring information."""
    # Unpack sample data
    data, state_transitions = sample_data
    
    # List of covariates
    covariates = [f"covariate_{i}" for i in range(2)]
    
    # Create model and training configurations
    model_config = ModelConfig(
        input_dim=len(covariates),
        hidden_dims=[8, 4],
        num_states=3,
        state_transitions=state_transitions
    )
    
    train_config = TrainConfig(
        batch_size=16,
        epochs=5,  # Small for testing
        learning_rate=0.01
    )
    
    # Fit model without censoring
    model_no_censoring = fit(
        df=data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    # Fit model with censoring
    model_with_censoring = fit(
        df=data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config,
        censoring_col="censored"
    )
    
    # Models should be different due to different loss functions
    # Compare predictions to check for differences
    x = torch.tensor(data[covariates].iloc[:5].values, dtype=torch.float32)
    time_idx = 0  # Use integer directly instead of tensor
    from_state = 0  # Use integer directly instead of tensor
    
    pred1 = model_no_censoring.predict_proba(x, time_idx=time_idx, from_state=from_state)
    pred2 = model_with_censoring.predict_proba(x, time_idx=time_idx, from_state=from_state)
    
    # Check that predictions are different
    assert not torch.allclose(pred1, pred2)


def test_generate_censoring_times():
    """Test generation of censoring times."""
    # Test basic generation
    n_samples = 100
    censoring_times = generate_censoring_times(
        n_samples=n_samples,
        censoring_rate=0.3,
        max_time=10.0,
        random_state=42
    )
    
    # Check shape
    assert len(censoring_times) == n_samples
    
    # Check censoring rate
    censored = np.isfinite(censoring_times)
    assert abs(censored.mean() - 0.3) < 0.1  # Allow some variation
    
    # Check range
    assert np.min(censoring_times[censored]) >= 0
    
    # Test with covariates
    covariates = np.random.normal(0, 1, (n_samples, 2))
    covariate_effects = np.array([0.5, -0.5])
    
    censoring_times_with_covs = generate_censoring_times(
        n_samples=n_samples,
        censoring_rate=0.3,
        max_time=10.0,
        covariates=covariates,
        covariate_effects=covariate_effects,
        random_state=42
    )
    
    # Check shape
    assert len(censoring_times_with_covs) == n_samples
    
    # Check censoring rate
    censored_with_covs = np.isfinite(censoring_times_with_covs)
    assert abs(censored_with_covs.mean() - 0.3) < 0.1  # Allow some variation


def test_simulate_patient_trajectory_with_censoring(sample_data):
    """Test simulation of patient trajectories with censoring."""
    # Unpack sample data
    data, state_transitions = sample_data
    
    # List of covariates
    covariates = [f"covariate_{i}" for i in range(2)]
    
    # Create and fit a model
    model_config = ModelConfig(
        input_dim=len(covariates),
        hidden_dims=[8, 4],
        num_states=3,
        state_transitions=state_transitions
    )
    
    train_config = TrainConfig(
        batch_size=16,
        epochs=5,  # Small for testing
        learning_rate=0.01
    )
    
    model = fit(
        df=data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    # Test patient trajectory simulation with censoring
    x = torch.tensor(data[covariates].iloc[0].values, dtype=torch.float32)
    
    # Generate trajectories with censoring
    # Create fixed censoring times to ensure some are censored
    censoring_times = np.array([3.0, 2.0, 8.0, 1.0, 7.0, 5.0, 4.0, 9.0, 2.0, 6.0])
    
    trajectories = simulate_patient_trajectory(
        model=model,
        x=x.unsqueeze(0),  # Add batch dimension
        start_state=0,
        max_time=10,
        n_simulations=10,
        censoring_times=censoring_times,
        seed=42
    )
    
    # Check that we have 10 trajectories
    assert len(trajectories) == 10
    
    # Check that each trajectory has the necessary columns
    for traj in trajectories:
        assert 'time_idx' in traj.columns
        assert 'time' in traj.columns
        assert 'state' in traj.columns
        assert 'simulation' in traj.columns
        assert 'censored' in traj.columns
    
    # Check if some trajectories are censored
    censored_trajectories = [traj for traj in trajectories if traj['censored'].any()]
    assert len(censored_trajectories) > 0


def test_prepare_event_data():
    """Test preparation of event data for Aalen-Johansen estimator."""
    # Create some simple trajectory data
    trajectory_data = pd.DataFrame({
        'time': [0, 1, 2, 0, 1, 2, 3],
        'state': [0, 1, 2, 0, 1, 1, 1],
        'simulation': [0, 0, 0, 1, 1, 1, 1],
        'censored': [False, False, False, False, False, False, True]
    })
    
    # Convert to event data
    event_data = prepare_event_data(trajectory_data)
    
    # Check columns
    assert set(event_data.columns) == {'id', 'time', 'from_state', 'to_state', 'censored'}
    
    # Check number of transitions
    # Simulation 0: 2 transitions (0->1, 1->2)
    # Simulation 1: 3 transitions (0->1, 1->1, 1->1) plus 1 censoring event
    assert len(event_data) == 6
    
    # Check censoring
    assert event_data['censored'].sum() == 1


def test_aalen_johansen_estimator():
    """Test Aalen-Johansen estimator for CIF calculation."""
    # Create some simple event data
    event_data = pd.DataFrame({
        'id': [0, 0, 1, 1, 1],
        'time': [1, 2, 1, 2, 3],
        'from_state': [0, 1, 0, 1, 1],
        'to_state': [1, 2, 1, 1, 1],
        'censored': [False, False, False, False, True]
    })
    
    # Calculate CIF for state 2
    cif = aalen_johansen_estimator(
        event_data=event_data,
        target_state=2,
        time_grid=np.array([0, 1, 2, 3])
    )
    
    # Check columns
    assert set(cif.columns) == {'time', 'cif', 'lower_ci', 'upper_ci'}
    
    # Check CIF values
    assert cif['cif'][0] == 0.0  # At time 0, CIF is 0
    assert cif['cif'][2] > 0.0  # At time 2, CIF should be positive (one transition to state 2)
    
    # CIF should be non-decreasing
    assert np.all(np.diff(cif['cif']) >= 0)


def test_calculate_cif_with_censoring(sample_data):
    """Test CIF calculation with censoring using Aalen-Johansen estimator."""
    # Unpack sample data
    data, state_transitions = sample_data
    
    # List of covariates
    covariates = [f"covariate_{i}" for i in range(2)]
    
    # Create and fit a model
    model_config = ModelConfig(
        input_dim=len(covariates),
        hidden_dims=[8, 4],
        num_states=3,
        state_transitions=state_transitions
    )
    
    train_config = TrainConfig(
        batch_size=16,
        epochs=5,  # Small for testing
        learning_rate=0.01
    )
    
    model = fit(
        df=data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    # Generate trajectories with censoring
    x = torch.tensor(data[covariates].iloc[0].values, dtype=torch.float32)
    trajectories = simulate_patient_trajectory(
        model=model,
        x=x.unsqueeze(0),  # Add batch dimension
        start_state=0,
        max_time=10,
        n_simulations=20,
        censoring_rate=0.3,
        seed=42
    )
    
    # Combine trajectories
    combined_trajectories = pd.concat(trajectories, ignore_index=True)
    
    # Calculate CIF using Aalen-Johansen estimator
    cif = calculate_cif(
        trajectories=combined_trajectories,
        target_state=2,
        time_grid=np.linspace(0, 10, 11),
        method="aalen-johansen"  # Specify the Aalen-Johansen method
    )
    
    # Check columns
    assert set(cif.columns) == {'time', 'cif', 'lower_ci', 'upper_ci'}
    
    # Check CIF values
    assert cif['cif'][0] == 0.0  # At time 0, CIF is 0
    assert cif['cif'].iloc[-1] <= 1.0  # CIF should be ≤ 1
    
    # CIF should be non-decreasing
    assert np.all(np.diff(cif['cif']) >= 0)