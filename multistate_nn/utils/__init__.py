"""Utility functions for MultiStateNN continuous-time models."""

from .continuous_simulation import (
    adjust_transitions_for_time,
    simulate_continuous_patient_trajectory,
    simulate_continuous_cohort_trajectories,
)

from .analysis import calculate_cif

from .visualization import (
    plot_transition_heatmap,
    plot_transition_graph,
    plot_intensity_matrix,
    plot_cif,
    compare_cifs,
)

__all__ = [
    # Continuous-time simulation utilities
    "adjust_transitions_for_time",
    "simulate_continuous_patient_trajectory",
    "simulate_continuous_cohort_trajectories",
    
    # Analysis utilities
    "calculate_cif",
    
    # Visualization utilities
    "plot_transition_heatmap",
    "plot_transition_graph",
    "plot_intensity_matrix",
    "plot_cif",
    "compare_cifs",
]