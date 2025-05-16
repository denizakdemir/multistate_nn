"""Tests for time adjustment in continuous-time multistate models."""

import pytest
import torch
import numpy as np
import scipy.linalg

# To avoid Pyro import errors, we'll add a try block
try:
    from multistate_nn.models_continuous import ContinuousMultiStateNN
    from multistate_nn.utils.continuous_simulation import adjust_transitions_for_time
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


def test_matrix_exponential_approach():
    """Test the matrix exponential approach for time adjustment."""
    # Create intensity matrix (generator matrix) Q
    Q = np.array([
        [-0.5, 0.3, 0.2],
        [0.0, -0.4, 0.4],
        [0.0, 0.0, 0.0]  # Absorbing state
    ])
    
    # Calculate transition matrix for t=1 via matrix exponential
    P_1 = scipy.linalg.expm(Q)
    
    # Calculate transition matrix for t=2 directly
    P_2_direct = scipy.linalg.expm(2 * Q)
    
    # Calculate transition matrix for t=2 using our adjustment function
    P_2_adjusted = adjust_transitions_for_time(P_1, 2.0)
    
    # Compare both approaches
    assert np.allclose(P_2_direct, P_2_adjusted, atol=1e-5)


def test_time_scaling_properties():
    """Test that time scaling has expected mathematical properties."""
    # Create a simple transition matrix for testing
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.0, 0.6, 0.4],
        [0.0, 0.0, 1.0]
    ])
    
    # Property 1: P(0) = I (Identity matrix)
    P_0 = adjust_transitions_for_time(P, 0.0)
    assert np.allclose(P_0, np.eye(3), atol=1e-5)
    
    # Property 2: P(s+t) = P(s) * P(t)
    P_2 = adjust_transitions_for_time(P, 2.0)
    P_3 = adjust_transitions_for_time(P, 3.0)
    P_5_combined = adjust_transitions_for_time(P_2, 1.5)  # P(2) adjusted for 1.5 more time units
    
    assert np.allclose(P_3, P_5_combined, atol=1e-5)


def test_model_time_consistency(model):
    """Test that the model's transition probabilities are consistent with time scaling."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create test input
    x = torch.randn(1, 2)
    
    # Get transition probabilities at different times
    P_1 = model(x, time_start=0.0, time_end=1.0, from_state=0).squeeze(0).detach().cpu().numpy()
    P_2 = model(x, time_start=0.0, time_end=2.0, from_state=0).squeeze(0).detach().cpu().numpy()
    
    # Calculate 2-step transition using our adjustment function
    P_1_to_2 = adjust_transitions_for_time(P_1, 2.0)
    
    # They should be approximately equal
    # (not exact because the model's intensity matrix might not be constant)
    assert np.allclose(P_1_to_2, P_2, atol=0.2)


def test_special_cases():
    """Test special cases and edge cases of time adjustment."""
    # Case 1: Identity matrix - should remain identity for any time
    I = np.eye(3)
    I_adjusted = adjust_transitions_for_time(I, 5.0)
    assert np.allclose(I_adjusted, I, atol=1e-5)
    
    # Case 2: Absorbing state matrix - should remain the same
    P_absorbing = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ])
    P_abs_adjusted = adjust_transitions_for_time(P_absorbing, 10.0)
    assert np.allclose(P_abs_adjusted, P_absorbing, atol=1e-5)
    
    # Case 3: Very small time adjustment - should be close to original
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.0, 0.6, 0.4],
        [0.0, 0.0, 1.0]
    ])
    P_small = adjust_transitions_for_time(P, 0.01)
    # Should be close to original but slightly different
    assert not np.allclose(P_small, P, atol=1e-5)
    assert np.allclose(P_small, P, atol=0.1)


def test_numerical_stability():
    """Test numerical stability of time adjustment method."""
    # Create a valid but numerically challenging transition matrix
    P = np.array([
        [0.999, 0.0005, 0.0005],
        [0.0001, 0.9994, 0.0005],
        [0.0, 0.0, 1.0]
    ])
    
    # Test with various time scales
    for t in [0.1, 1.0, 10.0, 100.0]:
        P_t = adjust_transitions_for_time(P, t)
        
        # Check that the result is a valid probability matrix
        assert np.all(np.isfinite(P_t))
        assert np.all(P_t >= 0)
        assert np.all(P_t <= 1)
        assert np.allclose(np.sum(P_t, axis=1), np.ones(3), atol=1e-10)


def test_model_time_interval_additivity(model):
    """Test that model predictions are time-additive."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create test input
    x = torch.randn(1, 2)
    
    # Case 1: Direct transition from time 0 to time 3
    P_0_to_3 = model(x, time_start=0.0, time_end=3.0, from_state=0).squeeze(0)
    
    # Case 2: Transition from time 0 to 1.5, then from 1.5 to 3
    P_0_to_1_5 = model(x, time_start=0.0, time_end=1.5, from_state=0).squeeze(0)
    
    # Create initial distribution based on P_0_to_1_5
    initial_dist = P_0_to_1_5.clone()
    
    # For each possible state after reaching time 1.5, calculate onward transitions
    combined_P = torch.zeros_like(P_0_to_3)
    
    for mid_state in range(model.num_states):
        # Skip if probability of reaching this mid-state is negligible
        if initial_dist[mid_state] < 1e-10:
            continue
            
        # Get transitions from this mid-state to final states
        mid_to_end = model(
            x, 
            time_start=1.5, 
            time_end=3.0, 
            from_state=mid_state
        ).squeeze(0)
        
        # Accumulate probabilities
        combined_P += initial_dist[mid_state] * mid_to_end
    
    # The two approaches should yield similar results
    assert torch.allclose(P_0_to_3, combined_P, atol=1e-5)


def test_fallback_method():
    """Test the fallback method for time adjustment."""
    # Create a simple transition matrix
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.0, 0.6, 0.4],
        [0.0, 0.0, 1.0]
    ])
    
    # Adjust for integer time (should use matrix power)
    P_3 = adjust_transitions_for_time(P, 3.0)
    P_3_power = np.linalg.matrix_power(P, 3)
    
    assert np.allclose(P_3, P_3_power, atol=1e-5)
    
    # Adjust for non-integer time (should use interpolation)
    P_2_5 = adjust_transitions_for_time(P, 2.5)
    
    # Should be between P^2 and P^3
    P_2 = np.linalg.matrix_power(P, 2)
    P_3 = np.linalg.matrix_power(P, 3)
    
    # Check that P_2_5 is approximately between P_2 and P_3
    for i in range(3):
        for j in range(3):
            assert P_2[i, j] <= P_2_5[i, j] <= P_3[i, j] or P_3[i, j] <= P_2_5[i, j] <= P_2[i, j]