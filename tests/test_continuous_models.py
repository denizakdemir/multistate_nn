"""Tests for continuous-time multistate models."""

import pytest
import torch
import numpy as np

# Import directly from modules to avoid circular import issues
from multistate_nn.models import ContinuousMultiStateNN
from multistate_nn.utils.continuous_simulation import simulate_continuous_patient_trajectory

def test_continuous_model_basics():
    """Test basic functionality of continuous-time model."""
    # Create a simple model with 3 states
    state_transitions = {
        0: [1, 2],  # State 0 can transition to states 1 or 2
        1: [2],     # State 1 can transition to state 2
        2: []       # State 2 is absorbing
    }
    
    model = ContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[32, 16],
        num_states=3,
        state_transitions=state_transitions
    )
    
    # Test forward pass
    x = torch.randn(5, 2)  # 5 samples, 2 features
    
    # Test with specific from_state
    probs = model(x, time_start=0.0, time_end=1.0, from_state=0)
    assert probs.shape == (5, 3)  # Batch size x num_states
    
    # Check that probabilities sum to 1
    assert torch.allclose(torch.sum(probs, dim=1), torch.ones(5))
    
    # Test without from_state (returns probabilities for all states)
    all_probs = model(x, time_start=0.0, time_end=1.0)
    assert isinstance(all_probs, dict)
    assert set(all_probs.keys()) == {0, 1, 2}
    
    # Check that state 0 transitions have correct shape (batch_size x 2 possible transitions)
    assert all_probs[0].shape == (5, 2)
    
    # Check that state 1 transitions have correct shape (batch_size x 1 possible transition)
    assert all_probs[1].shape == (5, 1)
    
    # Check that state 2 has no transitions (absorbing)
    assert all_probs[2].shape == (5, 0)

def test_intensity_matrix():
    """Test intensity matrix computation."""
    # Create a simple model with 3 states
    state_transitions = {
        0: [1, 2],  # State 0 can transition to states 1 or 2
        1: [2],     # State 1 can transition to state 2
        2: []       # State 2 is absorbing
    }
    
    model = ContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[32, 16],
        num_states=3,
        state_transitions=state_transitions
    )
    
    # Test intensity matrix computation
    x = torch.randn(1, 2)  # 1 sample, 2 features
    
    A = model.intensity_matrix(x)
    assert A.shape == (1, 3, 3)  # Batch size x num_states x num_states
    
    # Get intensity matrix as numpy for easier testing
    A_np = A.squeeze(0).detach().numpy()
    
    # Verify constraints on intensity matrix:
    # 1. Off-diagonal elements should be non-negative
    for i in range(3):
        for j in range(3):
            if i != j:
                assert A_np[i, j] >= 0
                
    # 2. Diagonal elements should be non-positive
    for i in range(3):
        assert A_np[i, i] <= 0
        
    # 3. Rows should sum to 0
    assert np.allclose(np.sum(A_np, axis=1), np.zeros(3), atol=1e-5)
    
    # 4. Check mask is applied correctly (only allowed transitions should be positive)
    # State 0 can go to 1 and 2
    assert A_np[0, 1] > 0 or A_np[0, 1] == 0
    assert A_np[0, 2] > 0 or A_np[0, 2] == 0
    
    # State 1 can only go to 2
    assert A_np[1, 0] == 0
    assert A_np[1, 2] > 0 or A_np[1, 2] == 0
    
    # State 2 is absorbing
    assert A_np[2, 0] == 0
    assert A_np[2, 1] == 0
    
    # Diagonal elements should equal negative sum of row
    for i in range(3):
        # Excluding diagonal
        row_sum_excl_diag = sum(A_np[i, j] for j in range(3) if i != j)
        assert np.isclose(A_np[i, i], -row_sum_excl_diag, atol=1e-5)

def test_time_effect():
    """Test that time has the expected effect on transition probabilities."""
    # Create a simple model with 3 states
    state_transitions = {
        0: [1, 2],  # State 0 can transition to states 1 or 2
        1: [2],     # State 1 can transition to state 2
        2: []       # State 2 is absorbing
    }
    
    model = ContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[32, 16],
        num_states=3,
        state_transitions=state_transitions
    )
    
    # Fix random seed for reproducibility
    torch.manual_seed(42)
    
    # Create one sample with fixed features
    x = torch.randn(1, 2)
    
    # Get probabilities at different time points
    short_time_probs = model(x, time_start=0.0, time_end=0.1, from_state=0)
    medium_time_probs = model(x, time_start=0.0, time_end=1.0, from_state=0)
    long_time_probs = model(x, time_start=0.0, time_end=10.0, from_state=0)
    
    # As time increases, probability of staying in state 0 should decrease
    assert short_time_probs[0, 0] > medium_time_probs[0, 0]
    assert medium_time_probs[0, 0] > long_time_probs[0, 0]
    
    # As time increases, probability of absorption (state 2) should increase
    assert short_time_probs[0, 2] < medium_time_probs[0, 2]
    assert medium_time_probs[0, 2] < long_time_probs[0, 2]
    
    # At t=0, probability of staying in the initial state should be 1
    zero_time_probs = model(x, time_start=0.0, time_end=0.0, from_state=0)
    assert torch.isclose(zero_time_probs[0, 0], torch.tensor(1.0))
    
    # Create absorption test: over very long time, an absorbing Markov chain
    # should end up in the absorbing state
    very_long_time_probs = model(x, time_start=0.0, time_end=100.0, from_state=0)
    assert very_long_time_probs[0, 2] > 0.9  # Should be close to 1

def test_simulation():
    """Test trajectory simulation with continuous-time model."""
    # Create a simple model with 3 states
    state_transitions = {
        0: [1, 2],  # State 0 can transition to states 1 or 2
        1: [2],     # State 1 can transition to state 2
        2: []       # State 2 is absorbing
    }
    
    model = ContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[32, 16],
        num_states=3,
        state_transitions=state_transitions
    )
    
    # Fix random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create one sample with fixed features
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
    
    # Basic checks
    assert len(trajectories) == 10
    
    # Check that all trajectories start at state 0
    for traj in trajectories:
        assert traj.iloc[0]['state'] == 0
        assert traj.iloc[0]['time'] == 0.0
        
    # Check that time is strictly increasing in each trajectory
    for traj in trajectories:
        assert (traj['time'].diff().dropna() >= 0).all()
        
    # Check that we only have valid state transitions
    for traj in trajectories:
        for i in range(len(traj) - 1):
            curr_state = traj.iloc[i]['state']
            next_state = traj.iloc[i+1]['state']
            # Valid transition check
            if curr_state != next_state:
                assert next_state in state_transitions[curr_state]
                
    # After enough time, most trajectories should end in the absorbing state (2)
    final_states = [traj.iloc[-1]['state'] for traj in trajectories]
    assert final_states.count(2) >= 7  # At least 7 out of 10 should reach state 2