"""Tests for model summary and visualization methods."""

import pytest
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import io
from contextlib import redirect_stdout

# Import directly from modules to avoid circular import issues
from multistate_nn.models import ContinuousMultiStateNN

def test_model_summary_base():
    """Test the summary method on ContinuousMultiStateNN."""
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
    
    # Capture the printed output
    f = io.StringIO()
    with redirect_stdout(f):
        summary_dict = model.summary()
    
    # Check printed output
    output = f.getvalue()
    assert "ContinuousMultiStateNN Summary" in output
    assert "Input dimension: 2" in output
    assert "Number of states: 3" in output
    assert "State 0 → States [1, 2]" in output
    assert "State 1 → States [2]" in output
    assert "State 2 (absorbing state)" in output
    assert "ODE solver: dopri5" in output
    
    # Check summary dictionary
    assert summary_dict["model_type"] == "ContinuousMultiStateNN"
    assert summary_dict["input_dim"] == 2
    assert summary_dict["num_states"] == 3
    assert summary_dict["state_transitions"] == state_transitions
    assert summary_dict["ode_solver"] == "dopri5"
    assert "total_params" in summary_dict
    assert "trainable_params" in summary_dict
    assert "architecture" in summary_dict

def test_plot_transition_heatmap():
    """Test the plot_transition_heatmap method."""
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
    
    # Create test input
    x = torch.randn(5, 2)  # 5 samples, 2 features
    
    # Test with from_state specified
    ax = model.plot_transition_heatmap(x, time_start=0.0, time_end=1.0, from_state=0)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test without from_state (all states)
    ax = model.plot_transition_heatmap(x, time_start=0.0, time_end=1.0, from_state=None)
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_plot_transition_graph():
    """Test the plot_transition_graph method."""
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
    
    # Create test input
    x = torch.randn(1, 2)  # Single sample
    
    # Test the graph visualization
    fig, ax = model.plot_transition_graph(x, time_start=0.0, time_end=1.0, threshold=0.0)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close()

def test_plot_intensity_matrix():
    """Test the plot_intensity_matrix method."""
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
    
    # Create test input
    x = torch.randn(1, 2)  # Single sample
    
    # Test the intensity matrix visualization
    ax = model.plot_intensity_matrix(x)
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_plot_transition_probabilities_over_time():
    """Test the plot_transition_probabilities_over_time method."""
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
    
    # Create test input
    x = torch.randn(1, 2)  # Single sample
    
    # Test plotting probabilities over time
    ax = model.plot_transition_probabilities_over_time(x, from_state=0, max_time=5.0, num_points=10)
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_predict_trajectory():
    """Test the predict_trajectory method."""
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
    
    # Create test input
    x = torch.randn(1, 2)  # Single sample
    
    # Test trajectory prediction
    trajectories = model.predict_trajectory(x, start_state=0, max_time=5.0, n_simulations=3, seed=42)
    
    # Check result
    assert len(trajectories) == 3
    assert all(isinstance(traj, pd.DataFrame) for traj in trajectories)
    
    # Check columns
    for traj in trajectories:
        assert 'time' in traj.columns
        assert 'state' in traj.columns
        assert 'simulation' in traj.columns
        
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

def test_plot_state_distribution():
    """Test the plot_state_distribution method."""
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
    
    # Create test input
    x = torch.randn(1, 2)  # Single sample
    
    # Test state distribution plot
    ax = model.plot_state_distribution(x, start_state=0, max_time=5.0, n_simulations=5, n_time_points=10, seed=42)
    assert isinstance(ax, plt.Axes)
    plt.close()