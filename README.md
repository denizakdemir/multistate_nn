# MultiStateNN: Neural Network Models for Multistate Processes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MultiStateNN is a PyTorch-based package implementing discrete-time multistate models using neural networks. It provides robust support for censored data as the default expectation in time-to-event analysis. The package supports both deterministic and Bayesian inference, making it suitable for modeling state transitions in various applications such as:

- Disease progression modeling with real-world censored patient data
- Survival analysis with competing risks
- Credit risk transitions with right-censored observations
- Career trajectory analysis with incomplete follow-up
- System degradation modeling with censored failure times

## Features

- Flexible neural network architectures with shared base classes
- Support for arbitrary state transition structures
- Simplified temporal effects modeling 
- Optional Bayesian inference using Pyro (via extensions)
- Hierarchical shrinkage for grouped transitions
- Built-in visualization tools
- Patient trajectory simulation
- Support for original time scales (days, years, etc.)
- Consistent CIF calculations across different time discretizations

### Advanced Censoring Support

- Comprehensive handling of right-censored observations as the default
- Modified loss function that properly accounts for censored transitions
- Specialized simulation functions that incorporate censoring
- Aalen-Johansen estimator for unbiased CIF calculation with censoring
- Competing risks analysis with proper censoring adjustments
- Inverse probability of censoring weighting (IPCW) for more accurate estimates

## Installation

Basic installation:
```bash
pip install multistate-nn
```

With Bayesian inference support:
```bash
pip install multistate-nn[bayesian]
```

With example notebook dependencies:
```bash
pip install multistate-nn[examples]
```

For development:
```bash
pip install -e ".[dev]"
```

## Data

The package includes scripts to download common multistate datasets from R packages. You need to have R and required packages installed on your system first.

To download datasets:

```bash
# 1. Make the scripts executable
chmod +x scripts/download_data.py scripts/setup_data.sh

# 2. Run the setup script to install dependencies and download datasets
./scripts/setup_data.sh

# Alternatively, you can run just the download script:
python scripts/download_data.py
```

This will create a `data` folder with datasets commonly used for multistate modeling:
- CAV (heart transplant data)
- Bladder cancer recurrence data
- Primary biliary cirrhosis data
- AIDS/SI switching data

## Quick Start

```python
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from multistate_nn import MultiStateNN, fit
from multistate_nn.train import ModelConfig, TrainConfig

# Prepare your data with censoring information
data = pd.DataFrame({
    'time': [0, 0, 1, 1, 1, 2, 2],
    'from_state': [0, 0, 1, 1, 2, 1, 2],
    'to_state': [1, 2, 2, 3, 2, 1, 2],  # Note: self-transitions can indicate censoring
    'age': [65, 70, 55, 75, 60, 62, 68],
    'biomarker': [1.2, 0.8, 1.5, 0.9, 1.1, 1.0, 1.3],
    'censored': [0, 0, 0, 0, 1, 1, 1]  # Censoring indicator (1=censored, 0=observed)
})

# Define state transitions
state_transitions = {
    0: [1, 2],    # From state 0, can transition to 1 or 2
    1: [1, 2, 3], # From state 1, can stay in 1 or go to 2 or 3
    2: [2, 3],    # From state 2, can stay in 2 or go to 3
    3: []         # State 3 is absorbing
}

# Define model configuration
model_config = ModelConfig(
    input_dim=2,              # Number of input features (age, biomarker)
    hidden_dims=[64, 32],     # Hidden layer dimensions
    num_states=4,             # Total number of states (0-3)
    state_transitions=state_transitions,
)

# Define training configuration
train_config = TrainConfig(
    batch_size=32,
    epochs=100,
    learning_rate=0.005,
    use_original_time=True    # Use actual time values rather than indices
)

# Fit the model with explicit censoring information
model = fit(
    df=data,
    covariates=['age', 'biomarker'],
    model_config=model_config,
    train_config=train_config,
    censoring_col='censored'  # Specify column containing censoring information
)

# Make predictions
x_new = torch.tensor([[70, 1.2], [65, 0.8]], dtype=torch.float32)
probs = model.predict_proba(x_new, time_idx=1, from_state=0)
print("Transition probabilities:", probs)

# Simulate trajectories with censoring
from multistate_nn.utils import simulate_patient_trajectory

trajectories = simulate_patient_trajectory(
    model=model,
    x=x_new[0:1],          # Features for a single patient
    start_state=0,
    max_time=5,
    n_simulations=100,
    censoring_rate=0.3     # 30% of simulated trajectories will be censored
)

# Calculate CIF with proper handling of censoring
from multistate_nn.utils.analysis import calculate_cif

cif = calculate_cif(
    trajectories=pd.concat(trajectories),
    target_state=3,        # Terminal state
    censoring_col='censored',
    competing_risk_states=[1, 2]  # States considered competing risks
)

# Plot the CIF
plt.figure(figsize=(8, 5))
plt.plot(cif['time'], cif['cif'], 'b-', label='CIF for State 3')
plt.fill_between(cif['time'], cif['lower_ci'], cif['upper_ci'], color='b', alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Cumulative Incidence')
plt.title('CIF with Censoring')
plt.legend()
plt.grid(alpha=0.3)
```


## Architecture

MultiStateNN has a modular architecture composed of:

### Core Model Components

- `BaseMultiStateNN`: Abstract base class providing shared functionality
- `MultiStateNN`: Deterministic implementation (main model class)

### Extensions

- `BayesianMultiStateNN`: Bayesian implementation (available with Pyro)

### Training Utilities

- `ModelConfig`: Configuration class for model architecture
- `TrainConfig`: Configuration class for training parameters
- `fit()`: Unified API for model training

### Utility Functions

The package includes utilities organized into logical groups:

- **Visualization**: Transition heatmaps, network graphs, and CIF plots
- **Simulation**: Generate synthetic data and trajectories
- **Analysis**: Cumulative Incidence Functions (CIFs)

## Detailed Documentation

### MultiStateNN Class

The core model class supporting deterministic inference with built-in censoring support:

```python
from multistate_nn import MultiStateNN

model = MultiStateNN(
    input_dim=2,            # Number of covariates
    hidden_dims=[64, 32],   # Architecture of hidden layers
    num_states=4,           # Total number of states
    state_transitions={...}, # Allowed transitions
    group_structure=None    # Optional grouping for hierarchical shrinkage
)

# Make predictions
x_test = torch.tensor([[65, 1.2]])
probs = model.predict_proba(x_test, time_idx=10, from_state=1)
```

### BayesianMultiStateNN Class

Extends MultiStateNN with Bayesian inference via Pyro and includes censoring support:

```python
from multistate_nn.extensions.bayesian import BayesianMultiStateNN

model = BayesianMultiStateNN(
    input_dim=2,
    hidden_dims=[64, 32],
    num_states=4,
    state_transitions={...}
)

# Training with censoring
from multistate_nn import fit

bayesian_model = fit(
    df=data,
    covariates=['age', 'biomarker'],
    model_config=model_config,
    train_config=TrainConfig(bayesian=True, epochs=200),
    censoring_col='censored'
)

# Inference with prediction
probs = bayesian_model.predict_proba(x, time_idx=5, from_state=1)
```

### Analysis and Visualization Tools

The package provides comprehensive tools for analyzing state transitions with proper handling of censoring:

```python
from multistate_nn.utils.visualization import (
    plot_transition_heatmap, 
    plot_transition_graph,
    plot_cif,
    compare_cifs
)

# Plot transition probabilities
plot_transition_heatmap(model, x, time_idx=0, from_state=0)

# Visualize transition network
plot_transition_graph(model, x, time_idx=0)

# Calculate and visualize cumulative incidence functions with censoring
from multistate_nn.utils import calculate_cif, simulate_cohort_trajectories

# Simulate trajectories with right-censoring by default
trajectories = simulate_cohort_trajectories(
    model, 
    cohort_features, 
    start_state=0, 
    max_time=100, 
    censoring_rate=0.3,  # 30% censoring rate
    use_original_time=True
)

# Calculate CIF with proper handling of censoring using Aalen-Johansen estimator
time_grid = np.linspace(0, 100, 50)  # 50 evenly spaced points
cif = calculate_cif(
    trajectories, 
    target_state=3, 
    time_grid=time_grid,
    censoring_col='censored',  # Specify the censoring column
    competing_risk_states=[1, 2],  # Optional: specify competing risks
    method="aalen-johansen"  # Recommended method for censored data
)

# Plot the censoring-adjusted CIF with confidence intervals
plot_cif(cif, label="With censoring", color="blue", show_ci=True)

# Compare CIFs from different models or scenarios
cif2 = calculate_cif(
    trajectories2, 
    target_state=3, 
    time_grid=time_grid,
    censoring_col='censored'
)

# Compare multiple CIFs on the same plot
compare_cifs(
    [cif, cif2],
    labels=["Model 1 (with censoring)", "Model 2 (with censoring)"],
    common_time_grid=True,  # Ensure consistent comparison
    show_ci=True  # Show confidence intervals
)

# You can also use the simpler method for comparison
cif_simple = calculate_cif(
    trajectories, 
    target_state=3, 
    time_grid=time_grid,
    censoring_col='censored',
    method="naive"  # Simpler method, less accurate with heavy censoring
)

# Compare the two estimation methods
compare_cifs(
    [cif, cif_simple],
    labels=["Aalen-Johansen estimator", "Simple method"],
    common_time_grid=True
)
```

## Examples

See the [examples](examples/) directory for detailed notebooks demonstrating:
- Disease progression modeling with synthetic data
- AIDS progression modeling with real data
- Trajectory simulation and cumulative incidence functions
- CIF consistency across different time discretizations
- Original time scale handling in multistate models
- Censoring handling and Aalen-Johansen estimation

For detailed documentation on advanced topics, see our specialized guides:

- [README_CENSORING.md](README_CENSORING.md): Detailed guide on handling right-censored observations
- [README_CIF_METHODOLOGY.md](README_CIF_METHODOLOGY.md): Explanation of CIF calculation methodologies and time discretization
- [README_TIME_ADJUSTMENT.md](README_TIME_ADJUSTMENT.md): Guide to time-adjusted simulations for consistent results across different time discretizations

These guides have been updated to use the consolidated API paths.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{multistate_nn2025,
    title={MultiStateNN: Neural Network Models for Multistate Processes},
    author={Akdemir, Deniz},
    year={2025},
    url={https://github.com/denizakdemir/multistate_nn}
}
```