# Censoring Support in MultiStateNN

This document explains how to work with right-censored observations in the MultiStateNN package.

## Overview

Censoring occurs when the event of interest does not happen during the observation period. In the context of multistate models, right-censoring happens when a patient is lost to follow-up before transitioning to another state. Properly handling censored observations is crucial for unbiased estimation of transition probabilities and cumulative incidence functions (CIFs).

The MultiStateNN package now provides explicit support for censoring through:

1. **Training with censoring information**
2. **Simulation with censoring**
3. **CIF calculation using the Aalen-Johansen estimator**

## Training with Censoring

### Data Preparation

When you have censoring information, your data should include a column indicating which observations are censored. By convention, we use:
- `1` or `True` for censored observations
- `0` or `False` for observed (non-censored) transitions

Example dataset:

```python
# Sample data with censoring
data = pd.DataFrame({
    'time': [0, 1, 2, 0, 1],
    'from_state': [0, 1, 2, 0, 1],
    'to_state': [1, 2, 3, 1, 1],  # Note: to_state = from_state for censored observations
    'censored': [0, 0, 0, 0, 1],  # Last observation is censored
    'covariate_0': [0.1, 0.2, 0.3, 0.4, 0.5],
    'covariate_1': [-0.1, -0.2, -0.3, -0.4, -0.5]
})
```

### Training the Model

To train a model with censoring information:

```python
from multistate_nn import fit, ModelConfig, TrainConfig

# Define state transitions
state_transitions = {
    0: [1, 2, 3],
    1: [2, 3],
    2: [3],
    3: []
}

# Create model and training configurations
model_config = ModelConfig(
    input_dim=2,  # Number of covariates
    hidden_dims=[32, 16],
    num_states=4,
    state_transitions=state_transitions
)

train_config = TrainConfig(
    batch_size=32,
    epochs=100,
    learning_rate=0.01
)

# Fit model with censoring information
model = fit(
    df=data,
    covariates=['covariate_0', 'covariate_1'],
    model_config=model_config,
    train_config=train_config,
    censoring_col='censored'  # Specify the column with censoring information
)
```

The loss function automatically adjusts for censored observations by using a partial likelihood approach. For censored observations, instead of penalizing specific transitions, the model maximizes the probability of the patient remaining in the observed state or transitioning to any future state.

## Simulation with Censoring

The package provides functions to simulate trajectories with censoring:

```python
from multistate_nn.utils import simulate_patient_trajectory

# Simulate trajectories with censoring
trajectories = simulate_patient_trajectory(
    model=model,
    x=patient_features,  # Feature tensor for a single patient
    start_state=0,
    max_time=10,
    n_simulations=100,
    censoring_rate=0.3,  # Target censoring rate
    seed=42,
    use_original_time=True
)
```

The simulated trajectories include a `censored` column indicating whether each trajectory was censored. You can control the censoring rate and even provide pre-generated censoring times.

For cohort simulation:

```python
from multistate_nn.utils import simulate_cohort_trajectories

# Simulate trajectories for a cohort with censoring
cohort_trajectories = simulate_cohort_trajectories(
    model=model,
    cohort_features=cohort_features,  # Feature tensor for multiple patients
    start_state=0,
    max_time=10,
    n_simulations_per_patient=10,
    censoring_rate=0.3,
    seed=42,
    use_original_time=True
)
```

## CIF Calculation with Censoring

The Aalen-Johansen estimator is a nonparametric method for estimating CIFs in the presence of competing risks and right-censoring. The package implements this estimator for accurate CIF calculation:

```python
from multistate_nn.utils import calculate_cif

# Calculate CIF using the Aalen-Johansen estimator
cif = calculate_cif(
    trajectories=trajectories,
    target_state=2,
    max_time=10,
    time_grid=np.linspace(0, 10, 101),  # Optional: specify evaluation time points
    ci_level=0.95,
    method="aalen-johansen"  # Specify the Aalen-Johansen estimator
)
```

The resulting CIF includes confidence intervals and properly accounts for censoring.

## Fine Control with Lower-Level Functions

For more advanced use cases, you can use the lower-level functions:

```python
from multistate_nn.utils import generate_censoring_times
from multistate_nn.utils import prepare_event_data, aalen_johansen_estimator

# Generate custom censoring times based on covariates
covariates = np.array([[0.1, 0.2], [0.3, 0.4]])
covariate_effects = np.array([0.5, -0.5])  # Positive values → longer censoring times
censoring_times = generate_censoring_times(
    n_samples=100,
    censoring_rate=0.4,
    max_time=10,
    covariates=covariates,
    covariate_effects=covariate_effects,
    random_state=42
)

# Prepare event data from trajectories for Aalen-Johansen estimator
event_data = prepare_event_data(trajectories)

# Calculate CIF using Aalen-Johansen estimator directly
cif = aalen_johansen_estimator(
    event_data=event_data,
    target_state=2,
    max_time=10,
    time_grid=np.linspace(0, 10, 101)
)
```

## Example: Real-World Data Analysis

### Step 1: Prepare data with censoring

```python
import pandas as pd
from multistate_nn import fit, ModelConfig, TrainConfig

# Load data with censoring information
data = pd.read_csv("patient_transitions.csv")

# Define state transitions based on your application
state_transitions = {
    0: [1, 2, 3],  # Healthy to mild/moderate/severe
    1: [2, 3],     # Mild to moderate/severe
    2: [3],        # Moderate to severe
    3: []          # Severe (absorbing state)
}

# Ensure censoring column exists (1 = censored, 0 = observed)
data['censored'] = data['event_type'].apply(lambda x: 1 if x == 'censored' else 0)
```

### Step 2: Train model with censoring

```python
# Define covariates
covariates = ['age', 'sex', 'biomarker1', 'biomarker2']

# Create model and training configurations
model_config = ModelConfig(
    input_dim=len(covariates),
    hidden_dims=[64, 32],
    num_states=4,
    state_transitions=state_transitions
)

train_config = TrainConfig(
    batch_size=64,
    epochs=200,
    learning_rate=0.005
)

# Fit model with censoring
model = fit(
    df=data,
    covariates=covariates,
    model_config=model_config,
    train_config=train_config,
    censoring_col='censored'
)
```

### Step 3: Simulate trajectories with censoring

```python
import torch
import numpy as np
from multistate_nn.utils import simulate_cohort_trajectories

# Create features for a typical patient
patient_features = torch.tensor([[
    data['age'].mean(),
    data['sex'].mode()[0],
    data['biomarker1'].mean(), 
    data['biomarker2'].mean()
]], dtype=torch.float32)

# Simulate multiple trajectories with censoring
trajectories = simulate_cohort_trajectories(
    model=model,
    cohort_features=patient_features,
    start_state=0,
    max_time=data['time'].max(),
    n_simulations_per_patient=1000,
    censoring_rate=0.3,
    seed=42
)
```

### Step 4: Calculate CIFs with proper censoring adjustment

```python
import matplotlib.pyplot as plt
from multistate_nn.utils import calculate_cif

# Calculate CIFs for each state using Aalen-Johansen estimator
states = [1, 2, 3]  # States of interest
cifs = []

for state in states:
    cif = calculate_cif(
        trajectories=trajectories,
        target_state=state,
        max_time=data['time'].max(),
        method="aalen-johansen"  # Use the Aalen-Johansen estimator for censored data
    )
    cifs.append(cif)

# Plot CIFs
plt.figure(figsize=(10, 6))
colors = ['blue', 'orange', 'red']

for i, (state, cif) in enumerate(zip(states, cifs)):
    plt.plot(cif['time'], cif['cif'], color=colors[i], label=f'State {state}')
    plt.fill_between(
        cif['time'], cif['lower_ci'], cif['upper_ci'],
        alpha=0.2, color=colors[i]
    )

plt.xlabel('Time')
plt.ylabel('Cumulative Incidence')
plt.title('Cumulative Incidence Functions with Censoring')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cifs_with_censoring.png', dpi=300)
plt.show()
```

## Technical Details

### Loss Function for Censored Observations

For censored observations, the likelihood contribution is different from observed transitions. Instead of penalizing specific transitions, we maximize the probability of all possible future states:

- For observed transitions: Standard cross-entropy loss on specific transitions
- For censored observations: Log-sum of probabilities across all possible transitions

This approach is implemented in the `_compute_batch_loss` function in `train.py`.

### Aalen-Johansen Estimator

The Aalen-Johansen estimator generalizes the Kaplan-Meier estimator to multistate models. It estimates the transition probability matrix P(s,t) for all time intervals and computes the CIF as the probability of being in the target state at time t, starting from the initial state.

The implementation in `aalen_johansen.py` follows these steps:

1. Convert trajectory data to event data format
2. Compute transition counts and at-risk counts at each time point
3. Calculate transition probability matrices for each time interval
4. Derive CIFs from the transition probability matrix
5. Calculate confidence intervals

## References

- Aalen, O. O., & Johansen, S. (1978). An empirical transition matrix for non-homogeneous Markov chains based on censored observations. *Scandinavian Journal of Statistics*, 5(3), 141-150.
- Andersen, P. K., & Keiding, N. (2002). Multi-state models for event history analysis. *Statistical Methods in Medical Research*, 11(2), 91-115.