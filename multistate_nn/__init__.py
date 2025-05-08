"""MultiStateNN: Neural network-based multistate models."""

from .models import MultiStateNN, BayesianMultiStateNN
from .train import fit, prepare_data, train_deterministic, train_bayesian
from .utils import (
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

__version__ = "0.1.0"

__all__ = [
    # Core models
    "MultiStateNN",
    "BayesianMultiStateNN",
    
    # Training functions
    "fit",
    "prepare_data",
    "train_deterministic",
    "train_bayesian",
    
    # Visualization utilities
    "plot_transition_heatmap",
    "plot_transition_graph",
    "compute_transition_matrix",
    
    # Data generation
    "generate_synthetic_data",
    
    # Simulation and prediction 
    "simulate_patient_trajectory",
    "simulate_cohort_trajectories",
    "calculate_cif",
    "plot_cif",
    "compare_cifs",
]
