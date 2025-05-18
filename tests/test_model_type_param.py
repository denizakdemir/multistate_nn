"""Test the ModelConfig's model_type parameter."""

import pytest
import torch
import numpy as np
import pandas as pd

from multistate_nn import ModelConfig, TrainConfig, fit


def test_model_config_model_type():
    """Test that ModelConfig accepts a model_type parameter."""
    
    # Create a ModelConfig with model_type parameter
    model_config = ModelConfig(
        input_dim=3,
        hidden_dims=[32, 16],
        num_states=4,
        state_transitions={0: [1, 2, 3], 1: [2, 3], 2: [3], 3: []},
        model_type="continuous"
    )
    
    # Check that the model_type parameter was set correctly
    assert model_config.model_type == "continuous"
    
    # Also test with the default value
    default_model_config = ModelConfig(
        input_dim=3,
        hidden_dims=[32, 16],
        num_states=4,
        state_transitions={0: [1, 2, 3], 1: [2, 3], 2: [3], 3: []}
    )
    
    # Check that the default value is "continuous"
    assert default_model_config.model_type == "continuous"


def test_fit_with_model_type():
    """Test that fit function handles model_type parameter correctly."""
    
    # Create a small synthetic dataset
    n_samples = 10
    np.random.seed(42)
    torch.manual_seed(42)
    
    data = {
        'from_state': [0] * n_samples,
        'to_state': np.random.choice([1, 2, 3], size=n_samples).tolist(),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'time_start': [0.0] * n_samples,  # Start times
        'time_end': np.random.uniform(1, 5, n_samples)  # End times
    }
    df = pd.DataFrame(data)
    
    # Define the model configuration with model_type parameter
    model_config = ModelConfig(
        input_dim=2,
        hidden_dims=[16],
        num_states=4,
        state_transitions={0: [1, 2, 3], 1: [2, 3], 2: [3], 3: []},
        model_type="continuous"
    )
    
    # Define the training configuration
    train_config = TrainConfig(
        batch_size=5,
        epochs=2,  # Just a few epochs for testing
        learning_rate=0.01
    )
    
    # Train a model with the given configurations
    model = fit(
        df=df,
        covariates=['feature1', 'feature2'],
        model_config=model_config,
        train_config=train_config,
        time_start_col='time_start',
        time_end_col='time_end'
    )
    
    # Verify the model has been trained
    assert model is not None
    
    # Create a test input and check that predictions work
    x_test = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    probs = model.predict_proba(x_test, time_start=0.0, time_end=1.0, from_state=0)
    
    # Check that probabilities sum to approximately 1
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
    
    # Check warning for unsupported model type
    with pytest.warns(UserWarning, match="Model type 'discrete' is not fully supported yet"):
        unsupported_model_config = ModelConfig(
            input_dim=2,
            hidden_dims=[16],
            num_states=4,
            state_transitions={0: [1, 2, 3], 1: [2, 3], 2: [3], 3: []},
            model_type="discrete"  # Currently unsupported
        )
        
        model = fit(
            df=df,
            covariates=['feature1', 'feature2'],
            model_config=unsupported_model_config,
            train_config=train_config,
            time_start_col='time_start',
            time_end_col='time_end'
        )
        
        # Still gets a continuous model
        assert model is not None