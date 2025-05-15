"""Analysis utilities for MultiStateNN models."""

from typing import Optional, Tuple, Union, Sequence, List, Dict
import numpy as np
import pandas as pd
import warnings


def calculate_cif(
    trajectories: pd.DataFrame,
    target_state: int,
    max_time: Optional[float] = None,
    by_patient: bool = False,
    ci_level: float = 0.95,
    use_original_time: bool = True,
    time_grid: Optional[np.ndarray] = None,
    n_grid_points: int = 100,
    censoring_col: Optional[str] = None,
    competing_risk_states: Optional[List[int]] = None
) -> pd.DataFrame:
    """Calculate cumulative incidence function (CIF) from simulated trajectories.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame of simulated trajectories from simulate_patient_trajectory
        or simulate_cohort_trajectories
    target_state : int
        The state for which to calculate cumulative incidence
    max_time : Optional[float], optional
        Maximum time value to include in results
    by_patient : bool, optional
        If True, calculate separate CIFs for each patient
    ci_level : float, optional
        Confidence interval level (0-1)
    use_original_time : bool, optional
        Whether to use original time values in the output
    time_grid : Optional[np.ndarray], optional
        Custom time grid for CIF calculation. If None, uses all unique time points 
        or an evenly spaced grid based on n_grid_points.
    n_grid_points : int, optional
        Number of points in the time grid if generating evenly spaced points
    censoring_col : Optional[str], optional
        Name of column indicating censoring status (True/1=censored, False/0=observed)
    competing_risk_states : Optional[List[int]], optional
        List of states considered competing risks to the target state
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time: Time values at which CIF is calculated
        - cif: Cumulative incidence function values
        - lower_ci: Lower confidence interval bound
        - upper_ci: Upper confidence interval bound
        - patient_id: (only if by_patient=True) Patient identifier
    """
    # Make a copy to avoid modifying the original
    trajectories = trajectories.copy()
    
    # Ensure we have a time column
    if 'time' not in trajectories.columns:
        if 'time_idx' in trajectories.columns:
            # Use time_idx as time if no time column exists
            trajectories['time'] = trajectories['time_idx'].copy()
        else:
            raise ValueError("Trajectories dataframe must have either 'time' or 'time_idx' column")
    
    # Determine the maximum time if not provided
    if max_time is None:
        max_time = trajectories['time'].max()
    
    # Filter to only include data up to max_time
    trajectories = trajectories[trajectories['time'] <= max_time].copy()
    
    # Create time grid if not provided
    if time_grid is None:
        if use_original_time:
            # Use unique time points from the data, plus max_time if not included
            time_points = sorted(trajectories['time'].unique())
            if max_time not in time_points:
                time_points = sorted(time_points + [max_time])
        else:
            # Create an evenly spaced grid with n_grid_points
            time_points = np.linspace(0, max_time, n_grid_points)
        time_grid = np.array(time_points)
    
    # Handle censoring if specified
    has_censoring = censoring_col is not None and censoring_col in trajectories.columns
    
    # Set up competing risks handling if specified
    has_competing_risks = competing_risk_states is not None and len(competing_risk_states) > 0
    
    # Group by patient_id if by_patient=True
    if by_patient:
        if 'patient_id' not in trajectories.columns:
            raise ValueError("Trajectories dataframe must have 'patient_id' column when by_patient=True")
        
        patient_groups = trajectories.groupby('patient_id')
        patient_cifs = []
        
        for patient_id, patient_data in patient_groups:
            # Calculate this patient's CIF with appropriate handling for censoring and competing risks
            patient_cif = _calculate_single_cif(
                patient_data, 
                target_state, 
                time_grid, 
                ci_level,
                censoring_col=censoring_col if has_censoring else None,
                competing_risk_states=competing_risk_states if has_competing_risks else None
            )
            patient_cif['patient_id'] = patient_id
            patient_cifs.append(patient_cif)
        
        return pd.concat(patient_cifs, ignore_index=True)
    else:
        # Calculate CIF for the entire cohort
        return _calculate_single_cif(
            trajectories, 
            target_state, 
            time_grid, 
            ci_level,
            censoring_col=censoring_col if has_censoring else None,
            competing_risk_states=competing_risk_states if has_competing_risks else None
        )


def _calculate_single_cif(
    trajectories: pd.DataFrame,
    target_state: int,
    time_grid: np.ndarray,
    ci_level: float = 0.95,
    censoring_col: Optional[str] = None,
    competing_risk_states: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Helper function to calculate CIF for a single group of trajectories.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame of trajectory data
    target_state : int
        Target state for CIF calculation
    time_grid : np.ndarray
        Array of time points at which to evaluate CIF
    ci_level : float
        Confidence interval level (0-1)
    censoring_col : Optional[str], optional
        Name of column indicating censoring status (True/1=censored, False/0=observed)
    competing_risk_states : Optional[List[int]], optional
        List of states considered competing risks to the target state
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time: Time points
        - cif: CIF values
        - lower_ci: Lower confidence interval bounds
        - upper_ci: Upper confidence interval bounds
    """
    # Ensure time_grid is sorted
    time_points = np.sort(time_grid)
    n_points = len(time_points)
    
    # Check if we have censoring information
    has_censoring = censoring_col is not None and censoring_col in trajectories.columns
    
    # Check if we have competing risks to handle
    has_competing_risks = competing_risk_states is not None and len(competing_risk_states) > 0
    competing_states = set(competing_risk_states) if has_competing_risks else set()
    
    # Group by simulation
    sim_groups = trajectories.groupby('simulation')
    n_sims = len(sim_groups)
    
    # Arrays for calculating CIF:
    # 1. Overall incidence of target event
    all_incidence = np.zeros((n_sims, n_points))
    # 2. If using competing risks, also track competing events
    if has_competing_risks:
        competing_incidence = np.zeros((n_sims, n_points))
    # 3. Track censoring status at each time for potential adjustments
    if has_censoring:
        censoring_status = np.zeros((n_sims, n_points), dtype=bool)
    
    for sim_idx, (_, sim_data) in enumerate(sim_groups):
        # Ensure data is sorted by time for correct identification of first occurrence
        sim_data = sim_data.sort_values('time')
        
        # Get censoring information for this simulation if available
        is_censored = False
        censoring_time = np.inf
        if has_censoring:
            censored_rows = sim_data[sim_data[censoring_col].astype(bool)]
            if len(censored_rows) > 0:
                is_censored = True
                censoring_time = censored_rows['time'].min()  # earliest censoring time
                
                # Mark all times after censoring as censored
                if censoring_time < np.inf:
                    for i, t in enumerate(time_points):
                        if t >= censoring_time:
                            censoring_status[sim_idx, i] = True
        
        # Find rows with target state (considering censoring if applicable)
        target_rows = sim_data[sim_data['state'] == target_state]
        
        # Handle target event occurrence
        if len(target_rows) > 0:
            # Get time of first occurrence of target state
            first_occurrence_time = target_rows['time'].iloc[0]
            
            # Check if the event happened before censoring
            if not is_censored or first_occurrence_time < censoring_time:
                # Valid event, mark incidence in all times >= event time
                for i, t in enumerate(time_points):
                    if t >= first_occurrence_time:
                        all_incidence[sim_idx, i] = 1
        
        # Handle competing risks if specified
        if has_competing_risks:
            # Find first occurrence of any competing risk state
            competing_rows = sim_data[sim_data['state'].isin(competing_states)]
            
            if len(competing_rows) > 0:
                # Get time of first competing event
                first_competing_time = competing_rows['time'].min()
                
                # Check if competing event happened before censoring
                if not is_censored or first_competing_time < censoring_time:
                    # Valid competing event, mark in all times >= competing event time
                    for i, t in enumerate(time_points):
                        if t >= first_competing_time:
                            competing_incidence[sim_idx, i] = 1
    
    # Handle censoring adjustments if needed (for more sophisticated methods)
    if has_censoring and censoring_status.any():
        # Basic implementation of inverse probability of censoring weighting (IPCW)
        # For more complex analyses, specialized libraries like 'lifelines' would be better
        
        # Calculate empirical probability of not being censored at each time
        censor_prob = 1.0 - np.mean(censoring_status, axis=0)
        
        # Avoid division by zero
        censor_prob = np.maximum(censor_prob, 1e-8)
        
        # Apply IPCW adjustment (simple version)
        # Each patient's contribution is weighted by inverse probability of not being censored
        weighted_incidence = all_incidence / censor_prob[np.newaxis, :]
        
        # Clip to valid range [0, 1]
        weighted_incidence = np.clip(weighted_incidence, 0, 1)
        
        # Use weighted incidence for CIF calculation
        cifs = np.mean(weighted_incidence, axis=0)
    else:
        # Standard CIF calculation without censoring adjustment
        cifs = np.mean(all_incidence, axis=0)
    
    # Adjust for competing risks if specified
    if has_competing_risks:
        # In the presence of competing risks, we need to account for their effect
        # Basic implementation of cumulative incidence function for competing risks
        
        # Calculate probability of each type of event
        p_target = np.mean(all_incidence, axis=0)
        p_competing = np.mean(competing_incidence, axis=0)
        
        # Calculate survival function (probability of no event)
        survival = 1.0 - (p_target + p_competing)
        
        # For true CIF with competing risks, we need to integrate
        # CIF(t) = integral from 0 to t of hazard(u) * S(u-) du
        # Simple implementation: calculate the cumulative incidence
        # iteratively at each time point
        
        cifs_cr = np.zeros_like(cifs)
        cumulative_hazard = 0.0
        
        for i in range(n_points):
            if i > 0:
                # Calculate hazard for target event at this time
                if p_target[i] > p_target[i-1]:
                    hazard = (p_target[i] - p_target[i-1]) / max(1 - p_target[i-1] - p_competing[i-1], 1e-8)
                    
                    # Update cumulative hazard
                    cumulative_hazard += hazard
                    
                    # Update CIF
                    cifs_cr[i] = cifs_cr[i-1] + hazard * max(1 - p_target[i-1] - p_competing[i-1], 1e-8)
                else:
                    cifs_cr[i] = cifs_cr[i-1]
            
        # Use competing risk adjusted CIFs
        cifs = cifs_cr
    
    # Calculate confidence intervals using normal approximation
    # Use fixed random seed for reproducibility
    np.random.seed(42)
    z = abs(np.percentile(np.random.normal(0, 1, 10000), [(1-ci_level)/2*100, (1+ci_level)/2*100]))
    
    # Handle division by zero in variance calculation
    var_terms = np.zeros_like(cifs)
    nonzero_mask = (cifs > 0) & (cifs < 1)
    var_terms[nonzero_mask] = np.sqrt(cifs[nonzero_mask] * (1 - cifs[nonzero_mask]) / n_sims)
    
    lower_ci = np.maximum(0, cifs - z[0] * var_terms)
    upper_ci = np.minimum(1, cifs + z[1] * var_terms)
    
    # Create result DataFrame
    cif_df = pd.DataFrame({
        'time': time_points,
        'cif': cifs,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
    })
    
    return cif_df