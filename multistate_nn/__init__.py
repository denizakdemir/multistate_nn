"""MultiStateNN: Neural Network Models for Continuous-Time Multistate Processes."""

# First, apply patches to fix compatibility issues
from .patches import apply_all_patches
# This will fix issues like torchdiffeq rtol/atol duplication
patch_results = apply_all_patches()

# Import core models
from .models import BaseMultiStateNN, ContinuousMultiStateNN

# Import architectures
from .architectures import (
    IntensityNetwork,
    MLPIntensityNetwork,
    RecurrentIntensityNetwork,
    AttentionIntensityNetwork,
    create_intensity_network,
)

# Import loss functions
from .losses import (
    ContinuousTimeMultiStateLoss,
    CompetingRisksContinuousLoss,
    create_loss_function,
)

# Import training utilities
from .train import (
    fit,
    ModelConfig,
    TrainConfig,
)

# Import utility functions from utils package
from .utils import (
    # Simulation utilities
    adjust_transitions_for_time,
    simulate_continuous_patient_trajectory,
    simulate_continuous_cohort_trajectories,
    
    # Analysis utilities
    calculate_cif,
    
    # Visualization utilities
    plot_transition_heatmap,
    plot_transition_graph,
    plot_intensity_matrix,
    plot_cif,
    compare_cifs,
    
    # Example utilities
    setup_state_names_and_colors,
    create_patient_profile,
    create_covariate_profiles,
    analyze_covariate_effect,
    compare_treatment_effects,
    visualize_state_distribution,
    compare_models_cif,
    visualize_model_comparison,
)

# fit is the primary interface for training models

__version__ = "0.4.0"  # Updated version for continuous-time only

# Define exports
__all__ = [
    # Core models
    "BaseMultiStateNN",
    "ContinuousMultiStateNN",
    
    # Architecture components
    "IntensityNetwork",
    "MLPIntensityNetwork",
    "RecurrentIntensityNetwork",
    "AttentionIntensityNetwork",
    "create_intensity_network",
    
    # Loss functions
    "ContinuousTimeMultiStateLoss",
    "CompetingRisksContinuousLoss",
    "create_loss_function",
    
    # Training utilities
    "fit",
    "ModelConfig",
    "TrainConfig",
    
    # Simulation and prediction 
    "simulate_continuous_patient_trajectory",
    "simulate_continuous_cohort_trajectories",
    "adjust_transitions_for_time",
    
    # Analysis utilities
    "calculate_cif",
    
    # Visualization utilities
    "plot_transition_heatmap",
    "plot_transition_graph",
    "plot_intensity_matrix",
    "plot_cif",
    "compare_cifs",
    
    # Example utilities
    "setup_state_names_and_colors",
    "create_patient_profile",
    "create_covariate_profiles",
    "analyze_covariate_effect",
    "compare_treatment_effects",
    "visualize_state_distribution",
    "compare_models_cif",
    "visualize_model_comparison",
]

# Try to import Bayesian extensions - we will handle this differently 
# to avoid circular imports
has_bayesian = False
try:
    import pyro
    has_bayesian = True
except ImportError:
    pass

# Add Bayesian extensions to exports if available
if has_bayesian:
    try:
        from .extensions.bayesian import (
            BayesianContinuousMultiStateNN,
            train_bayesian_model,
        )
        __all__.extend([
            "BayesianContinuousMultiStateNN",
            "train_bayesian_model",
        ])
    except ImportError:
        pass