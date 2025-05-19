"""Tests for model persistence functionality."""

import pytest
import torch
import tempfile
import os
import shutil

from multistate_nn.models import ContinuousMultiStateNN

def test_save_load_basic():
    """Test basic save and load functionality."""
    # Create a simple model with 3 states
    state_transitions = {
        0: [1, 2],  # State 0 can transition to states 1 or 2
        1: [2],     # State 1 can transition to state 2
        2: []       # State 2 is absorbing
    }
    
    # Create model
    model = ContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[32, 16],
        num_states=3,
        state_transitions=state_transitions
    )
    
    # Set fixed weights for deterministic testing
    torch.manual_seed(42)
    for param in model.parameters():
        param.data = torch.randn_like(param.data)
    
    # Create some test input
    x = torch.randn(1, 2)
    original_output = model(x, time_start=0.0, time_end=1.0, from_state=0)
    
    # Create a temporary directory for saving/loading
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the model
        save_path = model.save(temp_dir)
        
        # Verify files exist
        config_path = os.path.join(temp_dir, "model_config.json")
        state_dict_path = os.path.join(temp_dir, "model_state_dict.pt")
        assert os.path.exists(config_path)
        assert os.path.exists(state_dict_path)
        
        # Load the model
        loaded_model = ContinuousMultiStateNN.load(temp_dir)
        
        # Check model structure
        assert loaded_model.input_dim == model.input_dim
        assert loaded_model.num_states == model.num_states
        assert loaded_model.state_transitions == model.state_transitions
        assert loaded_model.solver == model.solver
        
        # Check model weights (outputs should be identical)
        loaded_output = loaded_model(x, time_start=0.0, time_end=1.0, from_state=0)
        assert torch.allclose(original_output, loaded_output, rtol=1e-5)

def test_save_load_custom_filename():
    """Test save and load with custom filename."""
    # Create a simple model
    state_transitions = {0: [1], 1: []}
    model = ContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[16],
        num_states=2,
        state_transitions=state_transitions
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save with custom filename
        save_path = model.save(temp_dir, filename="custom_model")
        
        # Verify files exist with custom name
        config_path = os.path.join(temp_dir, "custom_model_config.json")
        state_dict_path = os.path.join(temp_dir, "custom_model_state_dict.pt")
        assert os.path.exists(config_path)
        assert os.path.exists(state_dict_path)
        
        # Load with custom filename
        loaded_model = ContinuousMultiStateNN.load(temp_dir, filename="custom_model")
        
        # Check model structure
        assert loaded_model.input_dim == model.input_dim
        assert loaded_model.num_states == model.num_states

def test_save_load_with_device():
    """Test save and load with device specification."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping device test")
    
    # Create a simple model
    state_transitions = {0: [1], 1: []}
    model = ContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[16],
        num_states=2,
        state_transitions=state_transitions
    )
    
    # Move model to CUDA
    model = model.to("cuda")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        save_path = model.save(temp_dir)
        
        # Load to CPU
        loaded_model_cpu = ContinuousMultiStateNN.load(temp_dir, device=torch.device("cpu"))
        assert next(loaded_model_cpu.parameters()).device.type == "cpu"
        
        # Load to CUDA
        loaded_model_cuda = ContinuousMultiStateNN.load(temp_dir, device=torch.device("cuda"))
        assert next(loaded_model_cuda.parameters()).device.type == "cuda"

def test_error_handling():
    """Test error handling in load method."""
    # Create a simple model
    state_transitions = {0: [1], 1: []}
    model = ContinuousMultiStateNN(
        input_dim=2,
        hidden_dims=[16],
        num_states=2,
        state_transitions=state_transitions
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test non-existent directory
        with pytest.raises(FileNotFoundError):
            ContinuousMultiStateNN.load("/nonexistent/directory")
        
        # Save model
        model.save(temp_dir)
        
        # Directly call a non-existent model type
        # Create a mock class and hack its __name__ to force a type mismatch
        class MockClass(ContinuousMultiStateNN):
            pass
        
        # Set the name explicitly to ensure it's different
        MockClass.__name__ = "WrongModelClass"
        
        # This should raise ValueError due to model type mismatch
        with pytest.raises(ValueError):
            MockClass.load(temp_dir)