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

Install all dependencies at once:
```bash
pip install -e ".[dev,bayesian,examples]"
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

Run tests for new features:
```bash
pytest tests/test_new_features.py
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

### Data

Download example datasets:
```bash
python scripts/download_data.py
```

Or run the complete setup script (requires R):
```bash
chmod +x scripts/setup_data.sh
./scripts/setup_data.sh
```

### Examples

Run example notebooks to understand usage patterns:
```bash
jupyter notebook examples/
```

## Architecture

MultiStateNN is a PyTorch-based package implementing discrete-time multistate models using neural networks. The architecture is composed of three main components:

### 1. Core Model Classes (`models.py`)

- **BaseMultiStateNN**: Abstract base class providing shared functionality
  - Defines common interfaces and methods
  - Implements temporal smoothing functionality
  - Handles prediction of transition probabilities
  
- **MultiStateNN**: The main deterministic model class
  - Inherits from both `nn.Module` and `BaseMultiStateNN`
  - Implements a neural network with shared feature extraction layers and state-specific output heads
  - Supports arbitrary state transition structures
  - Includes temporal smoothing via decay factor mechanism
  - Supports hierarchical shrinkage for grouped transitions

### 2. Extensions

- **BayesianMultiStateNN** (in `extensions/bayesian.py`): Extends MultiStateNN with Bayesian inference
  - Uses Pyro for variational inference
  - Implements model and guide functions for SVI
  - Provides Bayesian uncertainty quantification

### 3. Training Utilities (`train.py`)

- **ModelConfig**: Configuration class for model architecture
- **TrainConfig**: Configuration class for training parameters
- **prepare_data**: Converts pandas DataFrame to PyTorch tensors
- **train_model**: Dispatcher function that routes to the appropriate training method
- **_train_deterministic**: Trains a deterministic MultiStateNN model
- **_train_bayesian**: Trains a Bayesian MultiStateNN model using SVI
- **fit**: Modern convenience function that handles data preparation and model training
- **fit_legacy**: Legacy function maintained for backward compatibility

### 4. Utility Modules

#### Visualization (`utils/visualization.py`)
- `plot_transition_heatmap`: Displays transition probabilities
- `plot_transition_graph`: Creates network diagram of transitions
- `compute_transition_matrix`: Calculates average transition probabilities
- `plot_cif`: Visualizes CIF curves
- `compare_cifs`: Compares multiple CIFs

#### Simulation (`utils/simulation.py`)
- `generate_synthetic_data`: Creates synthetic multistate data
- `simulate_patient_trajectory`: Simulates individual trajectories
- `simulate_cohort_trajectories`: Simulates trajectories for a cohort

#### Analysis (`utils/analysis.py`)
- `calculate_cif`: Computes cumulative incidence functions
- Supporting functions for statistical analysis

### Data Flow

1. Input data is provided as a pandas DataFrame with columns:
   - `time`: Time index
   - `from_state`: Source state
   - `to_state`: Target state
   - Additional covariates

2. The `fit` function:
   - Accepts model and training configurations
   - Converts data to PyTorch tensors
   - Creates a DataLoader for batched training
   - Initializes and trains the appropriate model

3. Trained models can:
   - Predict transition probabilities with `predict_proba`
   - Generate visualizations of transitions
   - Simulate future trajectories
   - Calculate cumulative incidence functions

## Design Patterns

### Configuration Objects
- The package uses dataclass-based configuration objects (`ModelConfig`, `TrainConfig`) to provide a clean interface for model creation and training.
- For backward compatibility, the legacy `fit` function can still be used.

### Factory Pattern
- The `fit` function serves as a factory that creates and returns the appropriate model type based on configuration parameters.
- The `train_model` function dispatches to the correct training implementation based on the model type.

### Inheritance Hierarchy
- The base `BaseMultiStateNN` class defines the interface
- `MultiStateNN` and `BayesianMultiStateNN` implement concrete models
- Extension system allows for adding new model types without modifying core functionality

## Common Development Tasks

### Adding a New Feature

When adding new functionality:

1. Determine if it belongs in core model, extensions, or utilities
2. Follow existing patterns for similar features
3. Add tests in appropriate test file
4. Update docstrings with proper type hints and descriptions
5. Run code formatting and type checking