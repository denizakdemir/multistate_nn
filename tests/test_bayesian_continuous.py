"""Tests for Bayesian continuous-time multistate models."""

import sys
import pytest
import torch
import numpy as np

# Try to import required dependencies for Bayesian tests
try:
    import pyro
    import pyro.distributions as dist
    import pyro.nn as pynn
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False

# Import from modules
from multistate_nn.extensions.bayesian_continuous import (
    BayesianContinuousMultiStateNN, 
    train_bayesian_continuous
)

# Skip all tests if pyro is not available
pytestmark = pytest.mark.skipif(not PYRO_AVAILABLE, reason="Pyro not installed")


def get_test_state_transitions():
    """Get a standard test state transition structure."""
    return {
        0: [1, 2],  # State 0 can transition to states 1 or 2
        1: [2],     # State 1 can transition to state 2
        2: []       # State 2 is absorbing
    }


@pytest.fixture
def bayesian_model():
    """Create a small Bayesian model for testing."""
    if not PYRO_AVAILABLE:
        pytest.skip("Pyro not installed")
        
    state_transitions = get_test_state_transitions()
    model = BayesianContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[8, 4],  # Small hidden dims for faster tests
        num_states=3,
        state_transitions=state_transitions,
        prior_scale=1.0,
        use_lowrank_multivariate=False,
        solver="euler",  # Use simple solver for tests
        solver_options={"step_size": 0.1}
    )
    return model


def test_model_initialization(bayesian_model):
    """Test that Bayesian model initializes correctly."""
    assert bayesian_model.num_states == 3
    assert isinstance(bayesian_model.feature_net, pynn.PyroModule)
    
    # Test that the intensity network has PyroSample parameters
    assert isinstance(bayesian_model.intensity_net.weight, pyro.nn.pyro_sample.PyroSample)
    assert isinstance(bayesian_model.intensity_net.bias, pyro.nn.pyro_sample.PyroSample)


def test_forward_pass(bayesian_model):
    """Test forward pass with Bayesian model."""
    batch_size = 4
    x = torch.randn(batch_size, 2)
    
    # Test with specific from_state
    probs = bayesian_model(x, time_start=0.0, time_end=1.0, from_state=0)
    assert probs.shape == (batch_size, 3)
    
    # Check sum to 1
    assert torch.allclose(torch.sum(probs, dim=1), torch.ones(batch_size), atol=1e-5)
    
    # Test with all initial states
    all_probs = bayesian_model(x, time_start=0.0, time_end=1.0)
    assert isinstance(all_probs, dict)
    assert set(all_probs.keys()) == {0, 1, 2}


def test_intensity_matrix(bayesian_model):
    """Test intensity matrix calculation in Bayesian model."""
    batch_size = 2
    x = torch.randn(batch_size, 2)
    
    # Get intensity matrices
    A = bayesian_model.intensity_matrix(x)
    assert A.shape == (batch_size, 3, 3)
    
    # Check properties for each matrix
    for i in range(batch_size):
        A_i = A[i].detach().numpy()
        
        # Off-diagonal elements should be non-negative where transitions are allowed
        for from_state, to_states in bayesian_model.state_transitions.items():
            for to_state in to_states:
                assert A_i[from_state, to_state] >= 0
        
        # Diagonal elements should be non-positive
        for j in range(3):
            assert A_i[j, j] <= 0
        
        # Rows should sum to 0
        assert np.allclose(np.sum(A_i, axis=1), np.zeros(3), atol=1e-5)


def test_with_group_structure():
    """Test Bayesian model with group structure for hierarchical priors."""
    if not PYRO_AVAILABLE:
        pytest.skip("Pyro not installed")
        
    state_transitions = get_test_state_transitions()
    
    # Define group structure
    group_structure = {
        (0, 1): "group1",
        (0, 2): "group1",
        (1, 2): "group2"
    }
    
    model = BayesianContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[8, 4],
        num_states=3,
        state_transitions=state_transitions,
        group_structure=group_structure,
        prior_scale=1.0
    )
    
    # Check that group embeddings were created
    assert hasattr(model, "_group_emb")
    assert hasattr(model, "_group_index")
    assert len(model._group_index) == 2  # Two unique groups
    
    # Test forward pass with grouped model
    batch_size = 2
    x = torch.randn(batch_size, 2)
    probs = model(x, time_start=0.0, time_end=1.0, from_state=0)
    assert probs.shape == (batch_size, 3)


@pytest.mark.slow
def test_model_training():
    """Test training of Bayesian model (short version)."""
    if not PYRO_AVAILABLE:
        pytest.skip("Pyro not installed")
        
    # Create a small model for faster testing
    state_transitions = get_test_state_transitions()
    model = BayesianContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[8, 4],
        num_states=3,
        state_transitions=state_transitions,
        prior_scale=1.0,
        solver="euler",  # Use simple solver for tests
    )
    
    # Create synthetic data
    batch_size = 20
    torch.manual_seed(42)
    x = torch.randn(batch_size, 2)
    from_state = torch.zeros(batch_size, dtype=torch.long)
    to_state = torch.randint(0, 3, (batch_size,))
    time_start = torch.zeros(batch_size)
    time_end = torch.ones(batch_size)
    
    # Train for just a few epochs to ensure functionality
    losses = train_bayesian_continuous(
        model=model,
        x=x,
        time_start=time_start,
        time_end=time_end,
        from_state=from_state,
        to_state=to_state,
        epochs=5,  # Just a few epochs for testing
        batch_size=10,
        learning_rate=0.01
    )
    
    # Check that losses decrease
    assert losses[-1] < losses[0]
    
    # Check that all losses are finite
    assert all(np.isfinite(loss) for loss in losses)


def test_censoring_in_model():
    """Test that censoring is handled correctly in the model."""
    if not PYRO_AVAILABLE:
        pytest.skip("Pyro not installed")
        
    # Create model
    state_transitions = get_test_state_transitions()
    model = BayesianContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[8, 4],
        num_states=3,
        state_transitions=state_transitions,
        prior_scale=1.0,
        solver="euler",  # Use simple solver for tests
    )
    
    # Create data with censoring
    batch_size = 10
    x = torch.randn(batch_size, 2)
    from_state = torch.zeros(batch_size, dtype=torch.long)
    to_state = torch.randint(0, 3, (batch_size,))
    time_start = torch.zeros(batch_size)
    time_end = torch.ones(batch_size)
    is_censored = torch.zeros(batch_size, dtype=torch.bool)
    is_censored[0:5] = True  # First half are censored
    
    # Set up trace for testing
    from pyro.poutine import trace
    
    # Create two identical runs, one with and one without censoring
    trace_no_censor = trace(model.model).get_trace(
        x, time_start, time_end, from_state, to_state
    )
    
    trace_with_censor = trace(model.model).get_trace(
        x, time_start, time_end, from_state, to_state, is_censored
    )
    
    # Check that the traces are different (censoring changes the model)
    assert set(trace_no_censor.nodes.keys()) != set(trace_with_censor.nodes.keys())


def test_different_times():
    """Test model with different time values."""
    if not PYRO_AVAILABLE:
        pytest.skip("Pyro not installed")
        
    # Create model
    state_transitions = get_test_state_transitions()
    model = BayesianContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[8, 4],
        num_states=3,
        state_transitions=state_transitions,
        prior_scale=1.0,
        solver="euler",  # Use simple solver for tests
    )
    
    # Create a sample
    x = torch.randn(1, 2)
    
    # Test with different time intervals
    short_time = model(x, time_start=0.0, time_end=0.1, from_state=0)
    medium_time = model(x, time_start=0.0, time_end=1.0, from_state=0)
    long_time = model(x, time_start=0.0, time_end=10.0, from_state=0)
    
    # Check that the predictions are different
    assert not torch.allclose(short_time, medium_time)
    assert not torch.allclose(medium_time, long_time)
    
    # Check expected behavior: probability of staying in state 0 decreases with time
    assert short_time[0, 0] > medium_time[0, 0]
    assert medium_time[0, 0] > long_time[0, 0]