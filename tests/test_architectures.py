"""Tests for neural network architectures for intensity functions."""

import pytest
import torch
import numpy as np

# To avoid Pyro import errors, we'll add a try block
try:
    from multistate_nn.architectures import (
        IntensityNetwork,
        MLPIntensityNetwork,
        RecurrentIntensityNetwork,
        AttentionIntensityNetwork,
        create_intensity_network
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


def test_mlp_intensity_network():
    """Test MLPIntensityNetwork functionality."""
    # Create a simple network
    state_transitions = get_test_state_transitions()
    network = MLPIntensityNetwork(
        input_dim=5,
        hidden_dims=[32, 16],
        num_states=3,
        state_transitions=state_transitions,
        use_layernorm=True
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 5)
    A = network(x)
    
    # Check output shape
    assert A.shape == (batch_size, 3, 3)
    
    # Check intensity matrix properties
    for i in range(batch_size):
        A_i = A[i].detach().numpy()
        
        # 1. Off-diagonal elements should be non-negative
        for j in range(3):
            for k in range(3):
                if j != k:
                    assert A_i[j, k] >= 0
        
        # 2. Diagonal elements should be non-positive
        for j in range(3):
            assert A_i[j, j] <= 0
        
        # 3. Rows should sum to 0
        assert np.allclose(np.sum(A_i, axis=1), np.zeros(3), atol=1e-5)
        
        # 4. Check mask is applied correctly
        # State 0 can transition to states 1 and 2
        assert A_i[0, 1] >= 0 and A_i[0, 2] >= 0
        
        # State 1 can only transition to state 2
        assert A_i[1, 0] == 0 and A_i[1, 2] >= 0
        
        # State 2 is absorbing
        assert A_i[2, 0] == 0 and A_i[2, 1] == 0


def test_recurrent_intensity_network():
    """Test RecurrentIntensityNetwork functionality."""
    # Create a recurrent network
    state_transitions = get_test_state_transitions()
    network = RecurrentIntensityNetwork(
        input_dim=5,
        hidden_dim=32,
        num_states=3,
        state_transitions=state_transitions,
        cell_type="gru",
        num_layers=2
    )
    
    # Test forward pass without time
    batch_size = 4
    x = torch.randn(batch_size, 5)
    A1 = network(x)
    
    # Test forward pass with time
    t = torch.tensor([0.5, 1.0, 2.0, 5.0])
    A2 = network(x, t=t)
    
    # Outputs should be different with different time inputs
    assert not torch.allclose(A1, A2)
    
    # Check output shapes
    assert A1.shape == (batch_size, 3, 3)
    assert A2.shape == (batch_size, 3, 3)
    
    # Check intensity matrix properties
    for A in [A1, A2]:
        for i in range(batch_size):
            A_i = A[i].detach().numpy()
            
            # 1. Off-diagonal elements should be non-negative
            for j in range(3):
                for k in range(3):
                    if j != k:
                        assert A_i[j, k] >= 0
            
            # 2. Diagonal elements should be non-positive
            for j in range(3):
                assert A_i[j, j] <= 0
            
            # 3. Rows should sum to 0
            assert np.allclose(np.sum(A_i, axis=1), np.zeros(3), atol=1e-5)


def test_attention_intensity_network():
    """Test AttentionIntensityNetwork functionality."""
    # Create an attention network
    state_transitions = get_test_state_transitions()
    network = AttentionIntensityNetwork(
        input_dim=5,
        hidden_dim=32,
        num_states=3,
        state_transitions=state_transitions,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    # Test forward pass without time
    batch_size = 4
    x = torch.randn(batch_size, 5)
    A1 = network(x)
    
    # Test forward pass with time
    t = torch.tensor([0.5, 1.0, 2.0, 5.0])
    A2 = network(x, t=t)
    
    # Outputs should be different with different time inputs
    assert not torch.allclose(A1, A2)
    
    # Check output shapes
    assert A1.shape == (batch_size, 3, 3)
    assert A2.shape == (batch_size, 3, 3)
    
    # Check intensity matrix properties
    for A in [A1, A2]:
        for i in range(batch_size):
            A_i = A[i].detach().numpy()
            
            # 1. Off-diagonal elements should be non-negative
            for j in range(3):
                for k in range(3):
                    if j != k:
                        assert A_i[j, k] >= 0
            
            # 2. Diagonal elements should be non-positive
            for j in range(3):
                assert A_i[j, j] <= 0
            
            # 3. Rows should sum to 0
            assert np.allclose(np.sum(A_i, axis=1), np.zeros(3), atol=1e-5)


def test_factory_function():
    """Test the create_intensity_network factory function."""
    state_transitions = get_test_state_transitions()
    input_dim = 5
    num_states = 3
    
    # Test MLP creation
    mlp_net = create_intensity_network(
        arch_type="mlp",
        input_dim=input_dim,
        num_states=num_states,
        state_transitions=state_transitions,
        hidden_dims=[64, 32],
        use_layernorm=True
    )
    assert isinstance(mlp_net, MLPIntensityNetwork)
    
    # Test recurrent creation
    rnn_net = create_intensity_network(
        arch_type="recurrent",
        input_dim=input_dim,
        num_states=num_states,
        state_transitions=state_transitions,
        hidden_dim=64,
        cell_type="lstm",
        num_layers=2
    )
    assert isinstance(rnn_net, RecurrentIntensityNetwork)
    
    # Test attention creation
    attn_net = create_intensity_network(
        arch_type="attention",
        input_dim=input_dim,
        num_states=num_states,
        state_transitions=state_transitions,
        hidden_dim=64,
        num_heads=8,
        num_layers=3,
        dropout=0.2
    )
    assert isinstance(attn_net, AttentionIntensityNetwork)
    
    # Test invalid architecture type
    with pytest.raises(ValueError):
        create_intensity_network(
            arch_type="invalid",
            input_dim=input_dim,
            num_states=num_states,
            state_transitions=state_transitions
        )


def test_time_effect_on_intensity():
    """Test that time has an effect on time-dependent architectures."""
    state_transitions = get_test_state_transitions()
    
    # Create recurrent network which should be time-dependent
    rnn_net = RecurrentIntensityNetwork(
        input_dim=5,
        hidden_dim=32,
        num_states=3,
        state_transitions=state_transitions
    )
    
    # Create a batch with 1 sample
    x = torch.randn(1, 5)
    
    # Compute intensity matrices at different time points
    times = [0.0, 1.0, 10.0, 100.0]
    matrices = [rnn_net(x, t=torch.tensor([t])) for t in times]
    
    # Check that matrices are different
    for i in range(len(matrices)-1):
        assert not torch.allclose(matrices[i], matrices[i+1])


def test_different_architectures_with_same_input():
    """Test that different architectures produce different results with the same input."""
    state_transitions = get_test_state_transitions()
    input_dim = 5
    num_states = 3
    
    # Create networks with similar hidden dimensions
    mlp_net = create_intensity_network(
        arch_type="mlp",
        input_dim=input_dim,
        num_states=num_states,
        state_transitions=state_transitions,
        hidden_dims=[32]
    )
    
    rnn_net = create_intensity_network(
        arch_type="recurrent",
        input_dim=input_dim,
        num_states=num_states,
        state_transitions=state_transitions,
        hidden_dim=32
    )
    
    attn_net = create_intensity_network(
        arch_type="attention",
        input_dim=input_dim,
        num_states=num_states,
        state_transitions=state_transitions,
        hidden_dim=32
    )
    
    # Create a batch with consistent random seed
    torch.manual_seed(42)
    x = torch.randn(4, input_dim)
    
    # Get outputs
    mlp_out = mlp_net(x)
    rnn_out = rnn_net(x)
    attn_out = attn_net(x)
    
    # Check that outputs are different
    assert not torch.allclose(mlp_out, rnn_out)
    assert not torch.allclose(mlp_out, attn_out)
    assert not torch.allclose(rnn_out, attn_out)