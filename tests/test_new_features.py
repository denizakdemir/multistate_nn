"""Tests for new features and refactored architecture."""

import pytest
import torch
import pandas as pd
import numpy as np
from multistate_nn import (
    BaseMultiStateNN,
    MultiStateNN, 
    fit_legacy, 
    ModelConfig, 
    TrainConfig
)
from multistate_nn import fit


@pytest.fixture
def sample_data():
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # State transition rules
    state_transitions = {0: [1, 2], 1: [2, 3], 2: [3], 3: []}
    
    # Generate valid transitions
    data = {
        "time": np.random.randint(0, 5, n_samples),
        "from_state": np.random.randint(0, 3, n_samples),  # No transitions from state 3 (absorbing)
        "age": np.random.normal(65, 10, n_samples),
        "sex": np.random.binomial(1, 0.5, n_samples),
        "biomarker": np.random.normal(1.0, 0.2, n_samples),
    }
    
    # Generate valid to_states based on transition rules
    to_states = []
    for from_state in data["from_state"]:
        possible_next_states = state_transitions[from_state]
        if possible_next_states:  # If not absorbing state
            to_states.append(np.random.choice(possible_next_states))
        else:  # Absorbing state (shouldn't happen with our from_state generation)
            to_states.append(from_state)
    
    data["to_state"] = to_states
    return pd.DataFrame(data)


@pytest.fixture
def model_config():
    return ModelConfig(
        input_dim=3,
        hidden_dims=[32, 16],
        num_states=4,
        state_transitions={0: [1, 2], 1: [2, 3], 2: [3], 3: []}
    )


@pytest.fixture
def train_config():
    return TrainConfig(
        batch_size=32,
        epochs=2,  # Small for fast testing
        learning_rate=0.01
    )


def test_base_multistate_nn_abstract():
    """Test that BaseMultiStateNN is an abstract class."""
    with pytest.raises(NotImplementedError):
        base_model = BaseMultiStateNN(
            input_dim=3,
            hidden_dims=[32, 16],
            num_states=4,
            state_transitions={0: [1, 2], 1: [2, 3], 2: [3], 3: []}
        )
        x = torch.randn(10, 3)
        base_model.forward(x)


def test_simplified_temporal_smoothing():
    """Test that the simplified temporal smoothing works correctly."""
    model = MultiStateNN(
        input_dim=3,
        hidden_dims=[32, 16],
        num_states=4,
        state_transitions={0: [1, 2], 1: [2, 3], 2: [3], 3: []}
    )
    
    gamma_t0 = model._temporal_smoothing(0)
    gamma_t1 = model._temporal_smoothing(1)
    
    assert gamma_t0.shape == (4, 4)
    assert gamma_t1.shape == (4, 4)
    
    # Check that temporal effect decays with time
    assert torch.all(gamma_t0.abs() >= gamma_t1.abs())


def test_config_objects(sample_data, model_config, train_config):
    """Test that model and training configuration objects work correctly."""
    covariates = ["age", "sex", "biomarker"]
    model = fit(
        df=sample_data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    assert isinstance(model, MultiStateNN)
    assert model.input_dim == model_config.input_dim
    assert model.num_states == model_config.num_states
    assert model.state_transitions == model_config.state_transitions


def test_legacy_compatibility(sample_data, model_config):
    """Test that the legacy fit function maintains backward compatibility."""
    covariates = ["age", "sex", "biomarker"]
    
    # Use legacy fit
    legacy_model = fit_legacy(
        df=sample_data,
        covariates=covariates,
        input_dim=model_config.input_dim,
        hidden_dims=model_config.hidden_dims,
        num_states=model_config.num_states,
        state_transitions=model_config.state_transitions,
        epochs=2  # Small for fast testing
    )
    
    assert isinstance(legacy_model, MultiStateNN)
    assert legacy_model.input_dim == model_config.input_dim
    assert legacy_model.num_states == model_config.num_states
    
    # Make predictions
    x = torch.randn(5, model_config.input_dim)
    probs = legacy_model.predict_proba(x, time_idx=0, from_state=0)
    
    assert probs.shape == (5, len(model_config.state_transitions[0]))
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(5))


def test_vectorized_training(sample_data, model_config, train_config):
    """Test that vectorized training produces valid results."""
    covariates = ["age", "sex", "biomarker"]
    model = fit(
        df=sample_data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    
    # Make predictions to ensure model is properly trained
    x = torch.randn(10, model_config.input_dim)
    for state in range(model_config.num_states - 1):  # Exclude absorbing state
        probs = model.predict_proba(x, time_idx=0, from_state=state)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(10))


try:
    from multistate_nn.extensions.bayesian import BayesianMultiStateNN
    
    def test_bayesian_extension(sample_data, model_config):
        """Test that the Bayesian extension works if installed."""
        bayesian_model = BayesianMultiStateNN(
            input_dim=model_config.input_dim,
            hidden_dims=model_config.hidden_dims,
            num_states=model_config.num_states,
            state_transitions=model_config.state_transitions
        )
        
        # Verify attributes
        assert bayesian_model.input_dim == model_config.input_dim
        assert bayesian_model.num_states == model_config.num_states
        
        # Test forward pass
        x = torch.randn(5, model_config.input_dim)
        logits = bayesian_model.forward(x, time_idx=0, from_state=0)
        
        assert logits.shape == (5, len(model_config.state_transitions[0]))
    
except ImportError:
    pass  # Skip Bayesian tests if Pyro is not installed