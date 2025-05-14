# Time-Adjusted Simulations in MultiStateNN

This document explains the methodology for adjusting transition probabilities based on time window sizes in MultiStateNN models to achieve consistent Cumulative Incidence Functions (CIFs) across different time discretizations.

## The Problem

When working with multistate models using different time discretizations (e.g., monthly vs. quarterly data), the transition probabilities learned by the models can lead to inconsistent simulated trajectories and CIFs. This occurs because:

1. **Different time window sizes**: Coarser discretizations (e.g., quarterly) cover larger time windows than finer discretizations (e.g., monthly).

2. **Probability scaling**: The probability of transitioning from one state to another in a 3-month period is not equivalent to three independent 1-month probabilities.

3. **Cumulative effects**: Models trained on coarser discretizations learn "cumulative" probabilities over larger time windows, while finer discretizations learn more granular changes.

## The Solution: Time-Adjusted Simulation

The key insight is to adjust transition probabilities based on the time window sizes during simulation:

### Mathematical Foundation

For a transition probability p over a time step Δt, we can convert to a rate λ and then scale appropriately:

1. **Convert probability to rate**: λ = -log(1-p)
   - This represents the transition rate per unit time

2. **Scale rate by time difference**: λ' = λ × (Δt/Δt_ref)
   - Where Δt_ref is the reference time unit (usually the smallest time step)

3. **Convert back to probability**: p' = 1 - exp(-λ')
   - This gives the adjusted probability for the different time window

This approach is based on the relationship between discrete-time transition probabilities and continuous-time hazard rates, ensuring that different time discretizations produce comparable results.

### Implementation

The implementation is now integrated into the main simulation functions in `multistate_nn/utils/simulation.py` and accessible through:

1. `simulate_patient_trajectory`: Simulates individual patient trajectories with time-adjusted transition probabilities when the `time_adjusted` parameter is set to True.

2. `simulate_cohort_trajectories`: Extends the individual simulation to cohorts of patients with the same time adjustment capabilities.

The key adjustment code:

```python
# Apply time-based adjustment to transition probabilities
if time_diffs is not None and t < len(time_diffs):
    # Extract time difference (window size) for current time point
    time_diff = time_diffs[t]
    
    # Adjust probabilities for time window size
    if time_diff > 1.0 and len(next_states) > 1:
        # Convert to rates (per unit time) for proper scaling
        # Rate = -log(1-p)
        rates = -np.log(1.0 - probs)
        
        # Scale rates by time difference
        scaled_rates = rates * time_diff
        
        # Convert back to probabilities
        adjusted_probs = 1.0 - np.exp(-scaled_rates)
        
        # Renormalize to ensure sum = 1
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        probs = adjusted_probs
```

## Benefits

Using time-adjusted simulation provides several benefits:

1. **Consistent CIFs**: CIFs calculated from different time discretizations become much more comparable.

2. **Accurate temporal dynamics**: The simulations better reflect the true underlying process regardless of discretization.

3. **Reliable comparisons**: Results from models trained on different datasets with varying time granularity can be reliably compared.

4. **Theoretical foundation**: The approach is based on established relationships between discrete-time and continuous-time processes.

## Usage Example

```python
from multistate_nn.utils import simulate_cohort_trajectories, calculate_cif

# Simulate trajectories with time adjustment
trajectories = simulate_cohort_trajectories(
    model=model,
    cohort_features=features,
    start_state=0,
    max_time=max_observed_time,
    n_simulations_per_patient=1000,
    time_adjusted=True,  # Enable time adjustment
    seed=42,
    use_original_time=True
)

# Calculate CIF from time-adjusted trajectories
cif = calculate_cif(
    trajectories, 
    target_state=1,
    time_grid=np.linspace(0, max_observed_time, 100)
)
```

## Verification

To verify that the time adjustment works correctly, run:

```
python examples/time_adjusted_cif_comparison.py
```

This script demonstrates how CIFs from models trained on different time discretizations become significantly more consistent when using time-adjusted simulation.