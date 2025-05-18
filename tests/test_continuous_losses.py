"""Tests for continuous-time multistate model loss functions."""

import pytest
import torch
import numpy as np

# To avoid Pyro import errors, we'll add a try block
try:
    # Import directly from modules to avoid circular import issues
    from multistate_nn.models_continuous import ContinuousMultiStateNN
    from multistate_nn.losses import ContinuousTimeMultiStateLoss, CompetingRisksContinuousLoss
    IMPORTS_AVAILABLE = True
except (ImportError, AttributeError):
    IMPORTS_AVAILABLE = False
    
# Skip all tests if required imports aren't available
pytestmark = pytest.mark.skipif(not IMPORTS_AVAILABLE, 
                               reason="Required dependencies not available")


class MockModel(torch.nn.Module):
    """Mock model for testing loss functions."""
    
    def __init__(self, num_states=3):
        super().__init__()
        self.num_states = num_states
        self.state_transitions = {
            0: [1, 2],  # State 0 can transition to states 1 or 2
            1: [2],     # State 1 can transition to state 2
            2: []       # State 2 is absorbing
        }
    
    def forward(self, x, time_start=0.0, time_end=1.0, from_state=None):
        """Mock forward pass with predictable outputs for testing."""
        batch_size = x.shape[0]
        
        # Create deterministic but different probabilities based on inputs
        # This helps tests verify that loss calculation is correct
        if from_state == 0:
            if abs(time_end - time_start) < 0.01:  # Small time difference
                # Stay in same state with high probability
                probs = torch.tensor([[0.8, 0.1, 0.1]]).repeat(batch_size, 1)
            else:
                # More balanced probabilities for longer time intervals
                probs = torch.tensor([[0.4, 0.3, 0.3]]).repeat(batch_size, 1)
        elif from_state == 1:
            if abs(time_end - time_start) < 0.01:  # Small time difference
                probs = torch.tensor([[0.0, 0.9, 0.1]]).repeat(batch_size, 1)
            else:
                probs = torch.tensor([[0.0, 0.5, 0.5]]).repeat(batch_size, 1)
        elif from_state == 2:  # Absorbing state
            probs = torch.tensor([[0.0, 0.0, 1.0]]).repeat(batch_size, 1)
        else:
            # Default case for when from_state is None
            probs = torch.tensor([[0.4, 0.3, 0.3]]).repeat(batch_size, 1)
        
        return probs


def test_continuous_time_loss_basic():
    """Test basic functionality of ContinuousTimeMultiStateLoss."""
    # Create mock model and loss function
    model = MockModel()
    loss_fn = ContinuousTimeMultiStateLoss()
    
    # Create batch with 4 examples: 
    # 1. Transition 0->1
    # 2. Transition 0->2
    # 3. Transition 1->2
    # 4. Censored at state 0
    batch_size = 4
    x = torch.randn(batch_size, 2)  # 2 features
    time_start = torch.zeros(batch_size)
    time_end = torch.ones(batch_size)
    from_state = torch.tensor([0, 0, 1, 0])
    to_state = torch.tensor([1, 2, 2, 0])  # Last one doesn't matter when censored
    is_censored = torch.tensor([False, False, False, True])
    
    # Compute loss
    loss = loss_fn(
        model=model,
        x=x,
        time_start=time_start,
        time_end=time_end,
        from_state=from_state,
        to_state=to_state,
        is_censored=is_censored
    )
    
    # Check loss is a scalar and greater than zero
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    
    # Calculate expected loss manually
    # For this test we know the model returns fixed probabilities
    # 1. Transition 0->1: -log(0.3) = -log(P(state 1 | from state 0, t=1))
    # 2. Transition 0->2: -log(0.3) = -log(P(state 2 | from state 0, t=1))
    # 3. Transition 1->2: -log(0.5) = -log(P(state 2 | from state 1, t=1))
    # 4. Censored at state 0: -log(0.4) = -log(P(state 0 | from state 0, t=1))
    expected_loss = (-np.log(0.3) - np.log(0.3) - np.log(0.5) - np.log(0.4)) / 4
    assert np.isclose(loss.item(), expected_loss, rtol=1e-5)


def test_continuous_time_loss_empty_batch():
    """Test loss function with empty valid batch (all absorbing states)."""
    model = MockModel()
    loss_fn = ContinuousTimeMultiStateLoss()
    
    # Create batch with all samples in absorbing state
    batch_size = 2
    x = torch.randn(batch_size, 2)
    time_start = torch.zeros(batch_size)
    time_end = torch.ones(batch_size)
    from_state = torch.tensor([2, 2])  # Both in absorbing state
    to_state = torch.tensor([2, 2])
    
    # Compute loss - should return 0 since no valid samples
    loss = loss_fn(
        model=model,
        x=x,
        time_start=time_start,
        time_end=time_end,
        from_state=from_state,
        to_state=to_state
    )
    
    assert loss.item() == 0.0


def test_continuous_time_loss_with_real_model():
    """Test loss function with a real ContinuousMultiStateNN model."""
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
    
    loss_fn = ContinuousTimeMultiStateLoss()
    
    # Create batch with 3 examples
    batch_size = 3
    x = torch.randn(batch_size, 2)
    time_start = torch.zeros(batch_size)
    time_end = torch.ones(batch_size)
    from_state = torch.tensor([0, 0, 1])
    to_state = torch.tensor([1, 2, 2])
    
    # Compute loss
    loss = loss_fn(
        model=model,
        x=x,
        time_start=time_start,
        time_end=time_end,
        from_state=from_state,
        to_state=to_state
    )
    
    # Check loss is a scalar and greater than zero
    assert isinstance(loss.item(), float)
    assert not torch.isnan(loss).item()


def test_continuous_time_loss_censoring_effect():
    """Test that censoring has the expected effect on loss."""
    model = MockModel()
    loss_fn = ContinuousTimeMultiStateLoss()
    
    # Create two identical batches, one with censoring and one without
    batch_size = 1
    x = torch.randn(batch_size, 2)
    time_start = torch.zeros(batch_size)
    time_end = torch.ones(batch_size)
    from_state = torch.tensor([0])
    
    # Case 1: Uncensored transition 0->1
    to_state_uncensored = torch.tensor([1])
    loss_uncensored = loss_fn(
        model=model,
        x=x,
        time_start=time_start,
        time_end=time_end,
        from_state=from_state,
        to_state=to_state_uncensored,
        is_censored=torch.tensor([False])
    )
    
    # Case 2: Censored at state 0
    to_state_censored = torch.tensor([0])  # Doesn't matter when censored
    loss_censored = loss_fn(
        model=model,
        x=x,
        time_start=time_start,
        time_end=time_end,
        from_state=from_state,
        to_state=to_state_censored,
        is_censored=torch.tensor([True])
    )
    
    # Expected values based on the mock model's probabilities
    expected_loss_uncensored = -np.log(0.3)  # -log(P(state 1 | from state 0, t=1))
    expected_loss_censored = -np.log(0.4)    # -log(P(state 0 | from state 0, t=1))
    
    assert np.isclose(loss_uncensored.item(), expected_loss_uncensored, rtol=1e-5)
    assert np.isclose(loss_censored.item(), expected_loss_censored, rtol=1e-5)


def test_competing_risks_loss():
    """Test CompetingRisksContinuousLoss functionality."""
    model = MockModel()
    
    # Define states 1 and 2 as competing risks
    competing_risk_states = [1, 2]
    loss_fn = CompetingRisksContinuousLoss(competing_risk_states)
    
    # Create batch with 3 examples:
    # 1. Transition 0->1 (competing risk)
    # 2. Transition 0->2 (competing risk)
    # 3. Censored at state 0
    batch_size = 3
    x = torch.randn(batch_size, 2)
    time_start = torch.zeros(batch_size)
    time_end = torch.ones(batch_size)
    from_state = torch.tensor([0, 0, 0])
    to_state = torch.tensor([1, 2, 0])  # Last one doesn't matter when censored
    is_censored = torch.tensor([False, False, True])
    
    # Compute loss
    loss = loss_fn(
        model=model,
        x=x,
        time_start=time_start,
        time_end=time_end,
        from_state=from_state,
        to_state=to_state,
        is_censored=is_censored
    )
    
    # Check loss is a scalar and greater than zero
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    
    # For censored observation in competing risks, the loss is based on 
    # the probability of NOT transitioning to any competing risk state.
    # In our mock model for from_state=0, competing risks (1,2) have 
    # total probability 0.3 + 0.3 = 0.6, so survival is 1 - 0.6 = 0.4
    expected_loss = (-np.log(0.3) - np.log(0.3) - np.log(0.4)) / 3
    assert np.isclose(loss.item(), expected_loss, rtol=1e-5)