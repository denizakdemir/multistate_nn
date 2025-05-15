# Recommendations for CIF Calculation

This document provides guidance on calculating Cumulative Incidence Functions (CIFs) in MultiStateNN models.

## Overview

The Cumulative Incidence Function (CIF) is a key metric in multi-state modeling, representing the probability of entering a specific state over time. When calculating CIFs, several factors can influence accuracy and consistency.

## Best Practices

Based on our testing, we recommend the following best practices for CIF calculation:

### 1. Use the Empirical Method

We've found that the `"empirical"` method generally produces more accurate and stable CIFs than the default `"aalen-johansen"` method, especially with simulated trajectories:

```python
cif = calculate_cif(
    trajectories=trajectories,
    target_state=target_state,
    time_grid=time_grid,
    method="empirical"  # Specify this explicitly
)
```

### 2. Consistent Time Grid

Always use a consistent time grid when comparing CIFs, especially across models with different time discretizations:

```python
time_grid = np.linspace(0, max_time, 100)  # Create a consistent grid

# Use the same grid for all CIF calculations you want to compare
cif1 = calculate_cif(..., time_grid=time_grid)
cif2 = calculate_cif(..., time_grid=time_grid)
```

### 3. Limit to Observed Time Range

Only calculate CIFs within the time range observed in your data:

```python
max_observed_time = 360  # Maximum time in your data
time_grid = np.linspace(0, max_observed_time, 100)

cif = calculate_cif(
    trajectories=trajectories,
    target_state=target_state,
    max_time=max_observed_time,  # Explicitly limit to observed range
    time_grid=time_grid
)
```

### 4. Use Time-Adjusted Simulation

When simulating trajectories, enable time adjustment for consistent results:

```python
trajectories = simulate_cohort_trajectories(
    model=model,
    cohort_features=features,
    start_state=0,
    max_time=max_time,
    n_simulations_per_patient=n_simulations,
    time_adjusted=True,  # Enable time adjustment
    use_original_time=True  # Use original time values
)
```

### 5. Original Time Values

Always use original time values rather than time indices:

```python
# In both simulation and CIF calculation
use_original_time=True
```

## Comparing Methods

If you want to compare the empirical and Aalen-Johansen methods:

```python
# Calculate CIF with empirical method
cif_empirical = calculate_cif(
    trajectories=trajectories,
    target_state=target_state,
    time_grid=time_grid,
    method="empirical"
)

# Calculate CIF with Aalen-Johansen method
cif_aj = calculate_cif(
    trajectories=trajectories,
    target_state=target_state,
    time_grid=time_grid,
    method="aalen-johansen"
)

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(cif_empirical['time'], cif_empirical['cif'], label='Empirical Method')
plt.plot(cif_aj['time'], cif_aj['cif'], label='Aalen-Johansen Method')
plt.fill_between(cif_empirical['time'], cif_empirical['lower_ci'], cif_empirical['upper_ci'], alpha=0.2)
plt.fill_between(cif_aj['time'], cif_aj['lower_ci'], cif_aj['upper_ci'], alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Cumulative Incidence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('CIF Method Comparison')
plt.show()
```

## Troubleshooting Flat CIFs

If your CIFs appear "flat at zero", consider the following:

1. **Check if your model has a time_mapper attribute**:
   ```python
   # Ensure your model has a TimeMapper instance
   model.time_mapper = TimeMapper(np.arange(0, max_time + 1))
   ```

2. **Try the empirical method instead of Aalen-Johansen**:
   ```python
   cif = calculate_cif(..., method="empirical")
   ```

3. **Verify transitions in your simulation data**:
   ```python
   # Check if simulations ever reach the target state
   reaches_state = (trajectories['state'] == target_state).any()
   print(f"Simulations reach state {target_state}: {reaches_state}")
   ```

4. **Examine the distribution of states in your trajectories**:
   ```python
   state_counts = trajectories['state'].value_counts().sort_index()
   print("State distribution:")
   print(state_counts)
   ```