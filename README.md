# MultiStateNN: Neural Network Models for Continuous-Time Multistate Processes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MultiStateNN is a PyTorch-based package implementing continuous-time multistate models using Neural Ordinary Differential Equations (Neural ODEs). It provides robust support for censored data as the default expectation in time-to-event analysis. The package supports both deterministic and Bayesian inference, making it suitable for modeling state transitions in various applications such as:

- Disease progression modeling with real-world censored patient data
- Survival analysis with competing risks
- Credit risk transitions with right-censored observations
- Career trajectory analysis with incomplete follow-up
- System degradation modeling with censored failure times

## Features

- Neural ODE-based implementation for continuous-time dynamics
- Specialized neural architectures for intensity functions (MLP, RNN, Attention)
- Support for arbitrary state transition structures
- Optional Bayesian inference using Pyro (via extensions)
- Proper handling of intensity matrix constraints
- Built-in visualization tools
- Patient trajectory simulation in continuous time
- Support for original time scales (days, years, etc.)

### Advanced Censoring Support

- Comprehensive handling of right-censored observations as the default
- Modified loss function that properly accounts for censored transitions
- Specialized simulation functions that incorporate censoring
- Competing risks analysis with proper censoring adjustments
- Continuous-time intensity matrix formulation for accurate estimates

## Version Note

**IMPORTANT**: Version 0.4.0+ of MultiStateNN has migrated to a fully continuous-time implementation using Neural ODEs. It is **NOT** backward compatible with earlier versions that used discrete-time models. If you need the discrete-time implementation, please use version 0.3.x or earlier.

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
from multistate_nn import ContinuousMultiStateNN, fit
from multistate_nn.train import ModelConfig, TrainConfig

# Prepare your data with censoring information
data = pd.DataFrame({
    'time_start': [0.0, 0.0, 1.2, 1.5, 1.7, 2.0, 2.3],
    'time_end': [1.2, 1.0, 1.8, 2.2, 2.5, 3.0, 3.2],
    'from_state': [0, 0, 1, 1, 2, 1, 2],
    'to_state': [1, 2, 2, 3, 2, 1, 2],
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
    model_type="continuous",  # Specify continuous-time model
)

# Define training configuration
train_config = TrainConfig(
    batch_size=32,
    epochs=100,
    learning_rate=0.005,
    solver="dopri5",          # ODE solver to use
    solver_options={"rtol": 1e-3, "atol": 1e-4}  # Solver tolerance options
)

# Fit the model with explicit censoring information
model = fit(
    df=data,
    covariates=['age', 'biomarker'],
    model_config=model_config,
    train_config=train_config,
    time_start_col='time_start',  # Specify column containing start times
    time_end_col='time_end',      # Specify column containing end times
    censoring_col='censored'      # Specify column containing censoring information
)

# Make predictions
x_new = torch.tensor([[70, 1.2], [65, 0.8]], dtype=torch.float32)
probs = model.predict_proba(x_new, time_start=1.0, time_end=2.5, from_state=0)
print("Transition probabilities:", probs)

# Simulate trajectories with censoring
from multistate_nn.utils import simulate_continuous_patient_trajectory

trajectories = simulate_continuous_patient_trajectory(
    model=model,
    x=x_new[0:1],          # Features for a single patient
    start_state=0,
    max_time=5.0,
    n_simulations=100,
    time_step=0.1,         # Time step for simulation grid
    censoring_rate=0.3     # 30% of simulated trajectories will be censored
)

# Calculate and plot transition probabilities over time
from_state = 0
to_state = 3
time_points = np.linspace(0, 5, 50)
probabilities = []

for t in time_points:
    prob = model.predict_proba(
        x_new[0:1], 
        time_start=0.0, 
        time_end=t, 
        from_state=from_state
    ).detach().numpy()[0, to_state]
    probabilities.append(prob)

# Plot the probability curve
plt.figure(figsize=(8, 5))
plt.plot(time_points, probabilities, 'b-', label=f'P({from_state} → {to_state})')
plt.xlabel('Time')
plt.ylabel('Transition Probability')
plt.title('Continuous-Time Transition Probability')
plt.legend()
plt.grid(alpha=0.3)
```


## Architecture

MultiStateNN has a modular architecture composed of:

### Core Model Components

- `BaseMultiStateNN`: Abstract base class providing shared functionality
- `ContinuousMultiStateNN`: Continuous-time implementation using Neural ODEs

### Extensions

- `BayesianContinuousMultiStateNN`: Bayesian implementation of continuous-time model (available with Pyro)

### Neural Architectures

- `IntensityNetwork`: Base class for intensity function networks
- `MLPIntensityNetwork`: Simple MLP architecture for intensity functions
- `RecurrentIntensityNetwork`: RNN-based architecture for time-dependent intensity
- `AttentionIntensityNetwork`: Transformer-based architecture for complex dependencies

### Training Utilities

- `ModelConfig`: Configuration class for model architecture
- `TrainConfig`: Configuration class for training parameters
- `fit()`: Unified API for model training

### Utility Functions

The package includes utilities organized into logical groups:

- **Visualization**: Transition heatmaps, network graphs, and probability curves
- **Simulation**: Generate synthetic trajectories in continuous time
- **Analysis**: Transition probability analysis in continuous time

## Detailed Documentation

### ContinuousMultiStateNN Class

The core model class implementing a continuous-time multistate model using Neural ODEs:

```python
from multistate_nn import ContinuousMultiStateNN

model = ContinuousMultiStateNN(
    input_dim=2,            # Number of covariates
    hidden_dims=[64, 32],   # Architecture of hidden layers
    num_states=4,           # Total number of states
    state_transitions={...}, # Allowed transitions
    solver="dopri5",        # ODE solver method
    solver_options={"rtol": 1e-3, "atol": 1e-4}  # Solver tolerance options
)

# Make predictions
x_test = torch.tensor([[65, 1.2]])
probs = model.predict_proba(x_test, time_start=0.0, time_end=2.5, from_state=1)
```

### BayesianContinuousMultiStateNN Class

Extends ContinuousMultiStateNN with Bayesian inference via Pyro:

```python
from multistate_nn.extensions.bayesian import BayesianContinuousMultiStateNN

model = BayesianContinuousMultiStateNN(
    input_dim=2,
    hidden_dims=[64, 32],
    num_states=4,
    state_transitions={...},
    prior_scale=1.0,          # Scale of prior distributions
    solver="dopri5",          # ODE solver method
    solver_options={"rtol": 1e-3, "atol": 1e-4}  # Solver tolerance options
)

# Training with censoring
from multistate_nn import fit

bayesian_model = fit(
    df=data,
    covariates=['age', 'biomarker'],
    time_start_col='time_start',
    time_end_col='time_end',
    model_config=model_config.replace(bayesian=True),
    train_config=TrainConfig(epochs=200),
    censoring_col='censored'
)

# Inference with prediction
probs = bayesian_model.predict_proba(x, time_start=0.0, time_end=2.5, from_state=1)
```

### Advanced Neural Architectures

The package provides specialized neural architectures for modeling intensity functions:

```python
from multistate_nn.architectures import (
    create_intensity_network,
    MLPIntensityNetwork,
    RecurrentIntensityNetwork,
    AttentionIntensityNetwork
)

# Create an MLP-based intensity network
intensity_net = create_intensity_network(
    arch_type="mlp",
    input_dim=2,
    num_states=4,
    state_transitions=state_transitions,
    hidden_dims=[64, 32],
    use_layernorm=True
)

# Create an RNN-based intensity network
rnn_intensity = create_intensity_network(
    arch_type="recurrent",
    input_dim=2,
    num_states=4,
    state_transitions=state_transitions,
    hidden_dim=64,
    cell_type="gru",
    num_layers=2
)

# Create an attention-based intensity network
attn_intensity = create_intensity_network(
    arch_type="attention",
    input_dim=2,
    num_states=4,
    state_transitions=state_transitions,
    hidden_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.1
)
```

### Analysis and Visualization Tools

The package provides tools for analyzing transition probabilities in continuous time:

```python
# Calculate transition probabilities over time
time_grid = np.linspace(0, 5, 50)
probs_over_time = []

for t in time_grid:
    prob = model.predict_proba(
        x_new[0:1], 
        time_start=0.0, 
        time_end=t, 
        from_state=0
    ).detach().numpy()[0]
    probs_over_time.append(prob)

# Plot transition probabilities
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i in range(model.num_states):
    plt.plot(time_grid, [p[i] for p in probs_over_time], label=f'State {i}')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Transition Probabilities from State 0')
plt.legend()
plt.grid(alpha=0.3)
```

### Simulation in Continuous Time

The package provides functions to simulate patient trajectories in continuous time:

```python
from multistate_nn.utils import (
    simulate_continuous_patient_trajectory,
    simulate_continuous_cohort_trajectories
)

# Simulate a single patient with fixed covariates
trajectories = simulate_continuous_patient_trajectory(
    model=model,
    x=x_new[0:1],          # Features for a single patient
    start_state=0,
    max_time=5.0,
    n_simulations=100,     # Number of trajectories to simulate
    time_step=0.1,         # Time step for simulation grid
    censoring_rate=0.3     # 30% of simulated trajectories will be censored
)

# Simulate a cohort of patients with different features
cohort_features = torch.tensor([
    [65, 1.2],  # Patient 1
    [70, 0.8],  # Patient 2
    [55, 1.5],  # Patient 3
], dtype=torch.float32)

cohort_trajectories = simulate_continuous_cohort_trajectories(
    model=model,
    cohort_features=cohort_features,
    start_state=0,
    max_time=5.0,
    n_simulations_per_patient=50,  # 50 trajectories per patient
    time_step=0.1,
    censoring_rate=0.3
)

# Visualize trajectories
import pandas as pd
import seaborn as sns

# Prepare data for plotting
traj_df = pd.concat(trajectories)
traj_df = traj_df[traj_df['grid_point'] == True]  # Use only grid points for cleaner plotting

# Plot the state distribution over time
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=traj_df,
    x='time',
    y='state',
    hue='simulation',
    alpha=0.3,
    palette='viridis'
)
plt.xlabel('Time')
plt.ylabel('State')
plt.title('Simulated Patient Trajectories')
```

## Mathematical Background

The continuous-time multistate model is based on the theory of continuous-time Markov processes. The key elements are:

1. **Intensity Matrix**: The core of the model is an intensity matrix A(t) that determines the rates of transition between states. For a system with m states, A(t) is an m×m matrix where the element a_ij(t) represents the instantaneous rate of transition from state i to state j at time t.

2. **Neural ODE Formulation**: We parameterize the intensity matrix using a neural network. The evolution of state probabilities follows the ODE:
   
   dp(t)/dt = p(t) · A(t)
   
   where p(t) is a probability vector over states at time t.

3. **Transition Probabilities**: The solution to this ODE gives the transition probability matrix P(s,t) from time s to time t:
   
   P(s,t) = exp(∫_s^t A(u) du)
   
   which is solved numerically using ODE solvers from the torchdiffeq package.

4. **Intensity Matrix Constraints**:
   - Off-diagonal elements must be non-negative (a_ij ≥ 0 for i≠j)
   - Rows must sum to zero (∑_j a_ij = 0 for all i)
   - The structure must respect the allowed state transitions

## Examples

See the [examples](examples/) directory for detailed notebooks demonstrating:
- Disease progression modeling with continuous-time models
- Bayesian inference in continuous time
- Trajectory simulation in continuous time
- Time-dependent intensity functions
- Censoring handling in continuous-time models

For detailed documentation on advanced topics, see our specialized guides:
- [README_CENSORING.md](README_CENSORING.md): Detailed guide on handling right-censored observations

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{multistate_nn2025,
    title={MultiStateNN: Neural Network Models for Continuous-Time Multistate Processes},
    author={Akdemir, Deniz, github: denizakdemir},
    year={2025},
    url={https://github.com/denizakdemir/multistate_nn}
}
```