"""MultiStateNN: Neural Network Models for Multistate Processes."""

# Import core models
from .models import BaseMultiStateNN, MultiStateNN

# Import training utilities
from .train import (
    fit,
    ModelConfig,
    TrainConfig,
)

# Import utility functions
from .utils import (
    plot_transition_heatmap,
    plot_transition_graph, 
    compute_transition_matrix,
    generate_synthetic_data,
    simulate_patient_trajectory,
    simulate_cohort_trajectories,
    generate_censoring_times,
    calculate_cif,
    plot_cif,
    compare_cifs,
)

# Import time mapping utilities
from .utils.time_mapping import TimeMapper

# Try to import Bayesian extension
try:
    from .extensions.bayesian import BayesianMultiStateNN
    has_bayesian = True
except ImportError:
    has_bayesian = False

# fit is the primary interface for training models

__version__ = "0.2.0"

# Define exports
__all__ = [
    # Core models
    "BaseMultiStateNN",
    "MultiStateNN",
    
    # Training utilities
    "fit",
    "ModelConfig",
    "TrainConfig",
    
    # Visualization utilities
    "plot_transition_heatmap",
    "plot_transition_graph",
    "compute_transition_matrix",
    
    # Data generation
    "generate_synthetic_data",
    "generate_censoring_times",
    
    # Simulation and prediction 
    "simulate_patient_trajectory",
    "simulate_cohort_trajectories",
    "calculate_cif",
    "plot_cif",
    "compare_cifs",
    
    # Time mapping utilities
    "TimeMapper",
]

# Add BayesianMultiStateNN to exports if available
if has_bayesian:
    __all__.append("BayesianMultiStateNN")