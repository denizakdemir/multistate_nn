"""MultiStateNN: Neural Network Models for Multistate Processes."""

# Import core models
from .models import BaseMultiStateNN
from .models_continuous import ContinuousMultiStateNN

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
    adjust_transitions_for_time,
    simulate_continuous_patient_trajectory,
    simulate_continuous_cohort_trajectories,
)

# fit is the primary interface for training models

__version__ = "0.3.0"

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
        from .extensions.bayesian_continuous import (
            BayesianContinuousMultiStateNN,
            train_bayesian_continuous,
        )
        __all__.extend([
            "BayesianContinuousMultiStateNN",
            "train_bayesian_continuous",
        ])
    except ImportError:
        pass