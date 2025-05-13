# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation

Install the package with development dependencies:
```bash
pip install -e ".[dev]"
```

Add Bayesian inference support:
```bash
pip install -e ".[bayesian]"
```

Add example notebooks dependencies:
```bash
pip install -e ".[examples]"
```

### Testing

Run all tests:
```bash
pytest tests/
```

Run tests with coverage report:
```bash
pytest tests/ --cov=multistate_nn
```

Run a specific test:
```bash
pytest tests/test_multistate_nn.py::test_function_name
```

### Code Style

Format code with black:
```bash
black multistate_nn/ tests/
```

Run type checking with mypy:
```bash
mypy multistate_nn/
```

## Architecture

MultiStateNN is a PyTorch-based package implementing discrete-time multistate models using neural networks. The architecture is composed of three main components:

### 1. Core Model Classes (`models.py`)

- **MultiStateNN**: The main deterministic model class
  - Implements a neural network with shared feature extraction layers and state-specific output heads
  - Supports arbitrary state transition structures
  - Includes temporal smoothing via GRU cells
  - Supports hierarchical shrinkage for grouped transitions
  
- **BayesianMultiStateNN**: Extends MultiStateNN with Bayesian inference
  - Uses Pyro for variational inference
  - Implements model and guide functions for SVI

### 2. Training Utilities (`train.py`)

- **prepare_data**: Converts pandas DataFrame to PyTorch tensors
- **train_deterministic**: Trains a deterministic MultiStateNN model
- **train_bayesian**: Trains a Bayesian MultiStateNN model using SVI
- **fit**: Convenience function that handles data preparation and model training

### 3. Utility Functions (`utils.py`)

- Visualization tools:
  - `plot_transition_heatmap`: Displays transition probabilities
  - `plot_transition_graph`: Creates network diagram of transitions
  - `compute_transition_matrix`: Calculates average transition probabilities
  
- Simulation functions:
  - `generate_synthetic_data`: Creates synthetic multistate data
  - `simulate_patient_trajectory`: Simulates individual trajectories
  - `simulate_cohort_trajectories`: Simulates trajectories for a cohort
  
- Cumulative incidence functions:
  - `calculate_cif`: Computes cumulative incidence functions
  - `plot_cif`: Visualizes CIF curves
  - `compare_cifs`: Compares multiple CIFs

### Data Flow

1. Input data is provided as a pandas DataFrame with columns:
   - `time`: Time index
   - `from_state`: Source state
   - `to_state`: Target state
   - Additional covariates

2. The `fit` function:
   - Converts data to PyTorch tensors
   - Creates a DataLoader for batched training
   - Initializes and trains the appropriate model

3. Trained models can:
   - Predict transition probabilities with `predict_proba`
   - Generate visualizations of transitions
   - Simulate future trajectories
   - Calculate cumulative incidence functions