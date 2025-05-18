"""Tests for continuous-time simulation utilities."""

import pytest
import torch
import numpy as np
import pandas as pd

# To avoid Pyro import errors, we'll add a try block
try:
    from multistate_nn.models_continuous import ContinuousMultiStateNN
    from multistate_nn.utils.continuous_simulation import (
        adjust_transitions_for_time,
        simulate_continuous_patient_trajectory,
        simulate_continuous_cohort_trajectories
    )
    IMPORTS_AVAILABLE = True
except (ImportError, AttributeError):
    IMPORTS_AVAILABLE = False

# Skip all tests if required imports aren't available
pytestmark = pytest.mark.skipif(not IMPORTS_AVAILABLE, 
                               reason="Required dependencies not available")


def get_test_state_transitions():
    """Get a standard test state transition structure."""
    return {
        0: [1, 2],  # State 0 can transition to states 1 or 2
        1: [2],     # State 1 can transition to state 2
        2: []       # State 2 is absorbing
    }


@pytest.fixture
def model():
    """Create a simple model for testing."""
    state_transitions = get_test_state_transitions()
    model = ContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[32, 16],
        num_states=3,
        state_transitions=state_transitions
    )
    return model


def test_adjust_transitions_for_time():
    """Test time adjustment for transition matrices."""
    # Create a simple transition matrix for testing
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.0, 0.6, 0.4],
        [0.0, 0.0, 1.0]
    ])
    
    # Case 1: Simple doubling of time
    P_double = adjust_transitions_for_time(P, 2.0)
    
    # Check shape
    assert P_double.shape == P.shape
    
    # Check probabilities constraints
    assert np.all(P_double >= 0)
    assert np.all(P_double <= 1)
    assert np.allclose(np.sum(P_double, axis=1), np.ones(3))
    
    # Check expected behavior: probability of staying in same state should decrease
    assert P_double[0, 0] < P[0, 0]
    assert P_double[1, 1] < P[1, 1]
    
    # Absorption state remains unchanged
    assert np.isclose(P_double[2, 2], 1.0)
    
    # Case 2: Time close to zero should be close to identity
    P_tiny = adjust_transitions_for_time(P, 0.001)
    assert np.allclose(P_tiny, np.eye(3), atol=0.1)
    
    # Case 3: Long time should lead to absorption
    P_long = adjust_transitions_for_time(P, 100.0)
    # After long time, all mass should be in absorbing state
    assert P_long[0, 2] > 0.9
    assert P_long[1, 2] > 0.9
    assert np.isclose(P_long[2, 2], 1.0)


def test_simulate_patient_trajectory(model):
    """Test simulation of individual patient trajectories."""
    # Create a single example
    x = torch.randn(1, 2)
    
    # Simulate 10 trajectories
    trajectories = simulate_continuous_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=10.0,
        n_simulations=10,
        seed=42
    )
    
    # Check output format
    assert isinstance(trajectories, list)
    assert len(trajectories) == 10
    assert all(isinstance(traj, pd.DataFrame) for traj in trajectories)
    
    # Check that all trajectories start at state 0
    for traj in trajectories:
        assert traj.iloc[0]['time'] == 0.0
        assert traj.iloc[0]['state'] == 0
    
    # Check that time is strictly increasing in each trajectory
    for traj in trajectories:
        assert (traj['time'].diff().dropna() >= 0).all()
    
    # Check trajectory constraints: only allowed transitions should occur
    state_transitions = get_test_state_transitions()
    for traj in trajectories:
        for i in range(len(traj) - 1):
            curr_state = traj.iloc[i]['state']
            next_state = traj.iloc[i+1]['state']
            if curr_state != next_state:  # If there was a transition
                assert next_state in state_transitions[curr_state]


def test_simulate_with_time_grid(model):
    """Test simulation with a custom time grid."""
    x = torch.randn(1, 2)
    
    # Create a custom time grid
    time_grid = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    
    trajectories = simulate_continuous_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=10.0,
        n_simulations=5,
        time_grid=time_grid,
        seed=42
    )
    
    # Check that the time grid points are included in the trajectories
    for traj in trajectories:
        grid_points = traj[traj['grid_point'] == True]
        assert set(grid_points['time'].values).issubset(time_grid)


def test_simulate_with_censoring(model):
    """Test simulation with censoring."""
    x = torch.randn(1, 2)
    
    # Case 1: With auto-generated censoring times
    trajectories_auto = simulate_continuous_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=10.0,
        n_simulations=20,
        censoring_rate=0.3,  # 30% censoring
        seed=42
    )
    
    # Check that some trajectories are censored
    censored_count = sum(traj['censored'].iloc[-1] for traj in trajectories_auto if 'censored' in traj.columns)
    assert censored_count > 0  # At least some should be censored
    
    # Case 2: With pre-specified censoring times
    manual_censoring_times = np.array([5.0] * 10)  # Censor all at time 5
    trajectories_manual = simulate_continuous_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=10.0,
        n_simulations=10,
        censoring_times=manual_censoring_times,
        seed=42
    )
    
    # Check that all trajectories end at or before time 5
    for traj in trajectories_manual:
        assert traj['time'].max() <= 5.0


def test_simulate_cohort_trajectories(model):
    """Test simulation for a cohort of patients."""
    # Create a small cohort
    cohort_size = 5
    x = torch.randn(cohort_size, 2)
    
    # Simulate cohort trajectories
    cohort_df = simulate_continuous_cohort_trajectories(
        model=model,
        cohort_features=x,
        start_state=0,
        max_time=10.0,
        n_simulations_per_patient=3,
        seed=42
    )
    
    # Check output format
    assert isinstance(cohort_df, pd.DataFrame)
    
    # Check that all patient IDs are present
    assert set(cohort_df['patient_id'].unique()) == set(range(cohort_size))
    
    # Check number of simulations
    assert len(cohort_df['simulation'].unique()) == 3 * cohort_size
    
    # Group by patient and simulation, and check that each trajectory starts at state 0
    for (patient, sim), group in cohort_df.groupby(['patient_id', 'simulation']):
        first_row = group.sort_values('time').iloc[0]
        assert first_row['time'] == 0.0
        assert first_row['state'] == 0


def test_long_term_behavior(model):
    """Test that long-term simulation behavior matches expected properties."""
    # Create a single example
    x = torch.randn(1, 2)
    
    # Simulate many trajectories over a long period
    trajectories = simulate_continuous_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=100.0,  # Long simulation time
        n_simulations=50,  # More simulations for statistical stability
        seed=42
    )
    
    # Count how many trajectories end in each state
    final_states = [traj.iloc[-1]['state'] for traj in trajectories]
    state_counts = {state: final_states.count(state) for state in range(3)}
    
    # Most trajectories should end in the absorbing state (state 2)
    # after a long enough simulation time
    assert state_counts[2] > 0.8 * len(trajectories)


def test_invalid_inputs(model):
    """Test that invalid inputs are handled properly."""
    # Test with multi-patient input (should raise ValueError)
    x_multi = torch.randn(2, 2)
    with pytest.raises(ValueError):
        simulate_continuous_patient_trajectory(
            model=model,
            x=x_multi,  # Multiple patients
            start_state=0,
            max_time=10.0
        )
    
    # Test with invalid censoring times
    x = torch.randn(1, 2)
    censoring_times = np.array([5.0, 5.0])  # 2 times, but 10 simulations
    with pytest.raises(ValueError):
        simulate_continuous_patient_trajectory(
            model=model,
            x=x,
            start_state=0,
            max_time=10.0,
            n_simulations=10,
            censoring_times=censoring_times
        )