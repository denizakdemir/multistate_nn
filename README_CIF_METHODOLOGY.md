# CIF Methodology in MultiStateNN

This document explains how Cumulative Incidence Functions (CIFs) are calculated and why different time discretizations can produce different results if not handled properly.

## Overview

A Cumulative Incidence Function (CIF) represents the probability of experiencing a specific event by a given time point, accounting for competing risks. In multistate modeling, CIFs help visualize the probability of transitioning to specific states over time.

## How CIFs are Calculated in MultiStateNN

The calculation process involves:

1. **Model Training**: The neural network learns transition probabilities between states at various time points.
2. **Trajectory Simulation**: Using these learned probabilities, we simulate multiple patient trajectories.
3. **Event Detection**: For each simulated trajectory, we identify the first occurrence of the target state.
4. **Probability Calculation**: At each time point, we calculate the proportion of trajectories that have reached the target state.

## Key Factors Affecting CIF Consistency

When comparing CIFs from models with different time discretizations (e.g., monthly vs. quarterly):

1. **Time Representation**:
   - **Problem**: Using time indices (`time_idx`) rather than actual time values (`time`) for event detection.
   - **Solution**: Always use actual time values for CIF calculation, not indices.

2. **Evaluation Time Grid**:
   - **Problem**: Evaluating CIFs at different time points based on discretization.
   - **Solution**: Use a consistent time grid for evaluation regardless of model discretization.

3. **Simulation Time Range**:
   - **Problem**: Simulating beyond observed time range causes extrapolation errors.
   - **Solution**: Limit simulation to observed time range in the original data.

4. **Absorbing States**:
   - **Problem**: CIFs for absorbing states eventually converge to 1.0 when simulating long enough.
   - **Solution**: For comparative analysis, use non-absorbing states or limit simulation time.

5. **Random Variability**:
   - **Problem**: Different random seeds introduce unnecessary variability.
   - **Solution**: Use consistent seeds and increase simulation count for stability.

## Implementation Details

The `calculate_cif` function in `multistate_nn/utils/__init__.py` (imported from `multistate_nn.utils.analysis`) handles these factors by:

```python
def calculate_cif(
    trajectories: pd.DataFrame,
    target_state: int,
    max_time: Optional[float] = None,
    by_patient: bool = False,
    ci_level: float = 0.95,
    use_original_time: bool = True,
    time_grid: Optional[np.ndarray] = None,
    n_grid_points: int = 100,
    method: str = "naive",  # Can also use "aalen-johansen" for censored data
    censoring_col: Optional[str] = "censored",
    competing_risk_states: Optional[List[int]] = None
) -> pd.DataFrame:
    """Calculate cumulative incidence function (CIF) from simulated trajectories."""
    # [...]
```

Key implementation features:

1. **Time Values**: Uses the `time` column for all calculations, not `time_idx`.
2. **Consistent Time Grid**: Accepts a custom `time_grid` parameter for consistent evaluation.
3. **Event Detection**: Identifies first occurrence based on actual time values.
4. **Confidence Intervals**: Provides confidence intervals for uncertainty quantification.
5. **Flexible Time Limitation**: Allows explicit limitation to specific time range.

## Recommended Practices

For consistent CIFs across different discretizations:

1. Always use `use_original_time=True` in training and simulation.
2. Provide a consistent `time_grid` for CIF calculation.
3. Explicitly set `max_time` to the observed time range.
4. Use non-absorbing states when comparing methodologies.
5. Use consistent random seeds for training and simulation.
6. Increase simulation count (1000+) for more stable estimates.

## Technical Explanation of Differences

When discrete-time models with different time discretizations show inconsistent CIFs:

1. **Learning Different Dynamics**: Models learn transition dynamics specific to their discretization:
   - Fine discretization: More frequent, smaller transition probabilities
   - Coarse discretization: Less frequent, larger transition probabilities

2. **Time Mapping Differences**: When discretizations don't align, the `TimeMapper` maps time points differently.

3. **First Occurrence Detection**: Without consistent time evaluation, first occurrences are detected at different points.

4. **Simulation Path Differences**: Different discretizations create different simulation trajectories, especially at boundaries.

The solution is to ensure that all time-dependent calculations use actual time values on a consistent grid, rather than relying on internal indices.