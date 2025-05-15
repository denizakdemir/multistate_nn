"""Tests for visualization utilities."""

import pytest
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multistate_nn import MultiStateNN
from multistate_nn.utils.visualization import (
    plot_transition_heatmap,
    compute_transition_matrix,
    plot_transition_graph,
    plot_cif,
    compare_cifs,
)


@pytest.fixture
def sample_model():
    """Create a simple model for testing visualization functions."""
    state_transitions = {0: [1, 2], 1: [2, 3], 2: [3], 3: []}
    
    model = MultiStateNN(
        input_dim=3,
        hidden_dims=[32, 16],
        num_states=4,
        state_transitions=state_transitions
    )
    
    # Set some deterministic values for reproducible testing
    for i in range(4):
        if i < 3:  # Skip the last layer which is an absorbing state
            module = model.state_heads[str(i)]
            # Set weights to a simple pattern for deterministic output
            torch.nn.init.constant_(module.weight, 0.1)
            torch.nn.init.constant_(module.bias, 0.0)
    
    return model


@pytest.fixture
def sample_cif_data():
    """Create sample CIF data for testing plot functions."""
    time_points = np.arange(11)
    cif_values = np.array([0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 0.9])
    lower_ci = cif_values - 0.05
    upper_ci = cif_values + 0.05
    
    # Make sure CIs are within valid range
    lower_ci = np.maximum(0, lower_ci)
    upper_ci = np.minimum(1, upper_ci)
    
    cif_df = pd.DataFrame({
        'time': time_points,
        'cif': cif_values,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
    })
    
    return cif_df


def test_plot_transition_heatmap(sample_model):
    """Test plotting transition probabilities as a heatmap."""
    # Create sample input
    x = torch.randn(5, 3)
    
    # Test with new axis
    ax = plot_transition_heatmap(sample_model, x, time_idx=0, from_state=0)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with provided axis
    fig, ax = plt.subplots()
    ax = plot_transition_heatmap(sample_model, x, time_idx=0, from_state=0, ax=ax)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with different source state
    ax = plot_transition_heatmap(sample_model, x, time_idx=0, from_state=1)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with different colormap
    ax = plot_transition_heatmap(sample_model, x, time_idx=0, from_state=0, cmap="Blues")
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_compute_transition_matrix(sample_model):
    """Test computation of transition matrix."""
    # Create sample input
    x = torch.randn(5, 3)
    
    # Compute transition matrix
    P = compute_transition_matrix(sample_model, x, time_idx=0)
    
    # Check shape
    assert P.shape == (4, 4)
    
    # Check row sums
    assert np.allclose(P.sum(axis=1), np.ones(4))
    
    # Check absorbing state (state 3)
    assert P[3, 3] == 1.0
    assert np.sum(P[3, :3]) == 0.0


def test_plot_transition_graph(sample_model):
    """Test plotting transition graph."""
    # Skip if networkx is not installed
    pytest.importorskip("networkx")
    
    # Create sample input
    x = torch.randn(5, 3)
    
    # Test with default parameters
    fig, ax = plot_transition_graph(sample_model, x, time_idx=0)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with custom threshold
    fig, ax = plot_transition_graph(sample_model, x, time_idx=0, threshold=0.2)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with custom figsize
    fig, ax = plot_transition_graph(sample_model, x, time_idx=0, figsize=(8, 6))
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert fig.get_size_inches().tolist() == [8.0, 6.0]
    plt.close()


def test_plot_cif(sample_cif_data):
    """Test plotting cumulative incidence function."""
    # Test with new axis
    ax = plot_cif(sample_cif_data)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with provided axis
    fig, ax = plt.subplots()
    ax = plot_cif(sample_cif_data, ax=ax)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with custom parameters
    ax = plot_cif(
        sample_cif_data,
        color='red',
        label='Test CIF',
        show_ci=True,
        linestyle='--',
        alpha=0.3
    )
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test without confidence intervals
    ax = plot_cif(sample_cif_data, show_ci=False)
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_compare_cifs(sample_cif_data):
    """Test comparing multiple CIFs."""
    # Create a second CIF with different values
    cif2 = sample_cif_data.copy()
    cif2['cif'] = cif2['cif'] * 0.8
    cif2['lower_ci'] = cif2['cif'] - 0.05
    cif2['upper_ci'] = cif2['cif'] + 0.05
    
    # Make sure CIs are within valid range
    cif2['lower_ci'] = np.maximum(0, cif2['lower_ci'])
    cif2['upper_ci'] = np.minimum(1, cif2['upper_ci'])
    
    # Test with default parameters
    fig, ax = compare_cifs(
        [sample_cif_data, cif2],
        labels=['CIF 1', 'CIF 2']
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with custom parameters
    fig, ax = compare_cifs(
        [sample_cif_data, cif2],
        labels=['CIF 1', 'CIF 2'],
        colors=['blue', 'green'],
        title='Custom Title',
        figsize=(10, 6),
        show_ci=False
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert fig.get_size_inches().tolist() == [10.0, 6.0]
    plt.close()