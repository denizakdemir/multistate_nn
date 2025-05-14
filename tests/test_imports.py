"""Tests for package imports."""

import pytest

from multistate_nn import (
    plot_transition_heatmap,
    plot_transition_graph, 
    compute_transition_matrix,
    generate_synthetic_data,
    simulate_patient_trajectory,
    simulate_cohort_trajectories,
    calculate_cif,
    plot_cif,
    compare_cifs,
)
from multistate_nn.utils import (
    plot_transition_heatmap as utils_plot_transition_heatmap,
    plot_transition_graph as utils_plot_transition_graph,
    compute_transition_matrix as utils_compute_transition_matrix,
    generate_synthetic_data as utils_generate_synthetic_data,
    simulate_patient_trajectory as utils_simulate_patient_trajectory,
    simulate_cohort_trajectories as utils_simulate_cohort_trajectories,
    calculate_cif as utils_calculate_cif,
    plot_cif as utils_plot_cif,
    compare_cifs as utils_compare_cifs,
)


def test_imports_redirect():
    """Test that imports from the top level redirect to utils implementations."""
    # Test visualization functions
    assert plot_transition_heatmap is utils_plot_transition_heatmap
    assert plot_transition_graph is utils_plot_transition_graph
    assert compute_transition_matrix is utils_compute_transition_matrix
    assert plot_cif is utils_plot_cif
    assert compare_cifs is utils_compare_cifs
    
    # Test simulation functions
    assert generate_synthetic_data is utils_generate_synthetic_data
    assert simulate_patient_trajectory is utils_simulate_patient_trajectory
    assert simulate_cohort_trajectories is utils_simulate_cohort_trajectories
    
    # Test analysis functions
    assert calculate_cif is utils_calculate_cif