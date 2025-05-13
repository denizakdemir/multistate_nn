import pytest
import torch
import pandas as pd
import numpy as np
from multistate_nn import MultiStateNN, BayesianMultiStateNN, fit


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
def model_params():
    return {
        "input_dim": 3,
        "hidden_dims": [32, 16],
        "num_states": 4,
        "state_transitions": {0: [1, 2], 1: [2, 3], 2: [3], 3: []},
    }


def test_multistate_nn_initialization(model_params):
    model = MultiStateNN(**model_params)
    assert model.input_dim == model_params["input_dim"]
    assert model.num_states == model_params["num_states"]
    assert model.state_transitions == model_params["state_transitions"]


def test_forward_pass(model_params):
    model = MultiStateNN(**model_params)
    batch_size = 10
    x = torch.randn(batch_size, model_params["input_dim"])

    # Test forward pass for specific state
    logits = model(x, time_idx=0, from_state=0)
    assert logits.shape == (batch_size, len(model_params["state_transitions"][0]))

    # Test forward pass for all states
    logits_dict = model(x, time_idx=0)
    assert isinstance(logits_dict, dict)
    for state, next_states in model_params["state_transitions"].items():
        if next_states:  # Skip absorbing states
            assert logits_dict[state].shape == (batch_size, len(next_states))


def test_predict_proba(model_params):
    model = MultiStateNN(**model_params)
    batch_size = 10
    x = torch.randn(batch_size, model_params["input_dim"])

    probs = model.predict_proba(x, time_idx=0, from_state=0)
    assert probs.shape == (batch_size, len(model_params["state_transitions"][0]))
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size))
    assert (probs >= 0).all() and (probs <= 1).all()


def test_model_fitting(sample_data, model_params):
    from multistate_nn import ModelConfig, TrainConfig
    covariates = ["age", "sex", "biomarker"]
    
    # Create configuration objects
    model_config = ModelConfig(
        input_dim=model_params["input_dim"],
        hidden_dims=model_params["hidden_dims"],
        num_states=model_params["num_states"],
        state_transitions=model_params["state_transitions"]
    )
    
    train_config = TrainConfig(
        batch_size=32,
        epochs=2,  # Small number for testing
        learning_rate=0.01
    )
    
    model = fit(
        df=sample_data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    assert isinstance(model, MultiStateNN)


@pytest.mark.skipif(not pytest.importorskip("pyro"), reason="Pyro not installed")
def test_bayesian_model_fitting(sample_data, model_params):
    from multistate_nn import ModelConfig, TrainConfig
    covariates = ["age", "sex", "biomarker"]
    
    # Create configuration objects
    model_config = ModelConfig(
        input_dim=model_params["input_dim"],
        hidden_dims=model_params["hidden_dims"],
        num_states=model_params["num_states"],
        state_transitions=model_params["state_transitions"]
    )
    
    train_config = TrainConfig(
        batch_size=32,
        epochs=2,  # Small number for testing
        learning_rate=0.01,
        bayesian=True  # Enable Bayesian inference
    )
    
    model = fit(
        df=sample_data,
        covariates=covariates,
        model_config=model_config,
        train_config=train_config
    )
    assert isinstance(model, BayesianMultiStateNN)


def test_temporal_smoothing(model_params):
    model = MultiStateNN(**model_params)
    
    # Initialize the time_bias with non-zero values to make the test pass
    # In a real scenario, these would be learned during training
    model.time_bias.data = torch.rand_like(model.time_bias.data)
    
    gamma_t0 = model._temporal_smoothing(0)
    gamma_t1 = model._temporal_smoothing(1)

    assert gamma_t0.shape == (model_params["num_states"], model_params["num_states"])
    assert gamma_t1.shape == (model_params["num_states"], model_params["num_states"])
    assert not torch.allclose(gamma_t0, gamma_t1)  # Should evolve over time
