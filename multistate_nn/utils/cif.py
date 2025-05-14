"""Cumulative Incidence Function (CIF) calculation methods for MultiStateNN models."""

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
    competing_risk_states: Optional[List[int]] = None,
    method: str = "aalen-johansen",
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
    method : str, optional
        Method to use for CIF calculation. Options: "aalen-johansen" (default, recommended),
        "naive" (simpler method, less accurate with heavy censoring)
        
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
    
    # Check if we need to add censoring information (in case it's missing)
    if censoring_col is None and 'censored' in trajectories.columns:
        censoring_col = 'censored'
    
    # Choose the appropriate method
    if method.lower() == "aalen-johansen":
        # Use Aalen-Johansen estimator (recommended for censored data)
        # First, prepare event data from trajectories
        event_data = _prepare_event_data(trajectories, max_time, censoring_col)
        
        # Group by patient_id if by_patient=True
        if by_patient:
            if 'patient_id' not in trajectories.columns:
                raise ValueError("Trajectories dataframe must have 'patient_id' column when by_patient=True")
            
            patient_groups = event_data.groupby('patient_id')
            patient_cifs = []
            
            for patient_id, patient_data in patient_groups:
                # Calculate this patient's CIF using Aalen-Johansen
                patient_cif = _aalen_johansen_estimator(
                    patient_data, 
                    target_state, 
                    max_time, 
                    time_grid, 
                    ci_level=ci_level
                )
                patient_cif['patient_id'] = patient_id
                patient_cifs.append(patient_cif)
            
            return pd.concat(patient_cifs, ignore_index=True)
        else:
            # Calculate CIF for the entire cohort
            return _aalen_johansen_estimator(
                event_data, 
                target_state, 
                max_time, 
                time_grid, 
                ci_level=ci_level
            )
    else:
        # Use simpler method (less accurate with heavy censoring)
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


def _prepare_event_data(
    trajectories: pd.DataFrame,
    max_time: Optional[float] = None,
    censoring_col: Optional[str] = None,
) -> pd.DataFrame:
    """Prepare event data from simulated trajectories.
    
    This function transforms simulated trajectory data into event format
    required for calculation of the Aalen-Johansen estimator.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame of simulated trajectories
    max_time : Optional[float], optional
        Maximum time value to include in results
    censoring_col : Optional[str], optional
        Name of column indicating censoring status
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - id: Patient/simulation identifier
        - time: Event time
        - from_state: State before transition
        - to_state: State after transition
        - censored: Whether the trajectory was censored
    """
    # Make a copy to avoid modifying the original
    trajectories = trajectories.copy()
    
    # Ensure we have required columns
    required_cols = ['time', 'state', 'simulation']
    for col in required_cols:
        if col not in trajectories.columns:
            raise ValueError(f"Required column '{col}' is missing from trajectories")
    
    # Check for censoring column
    has_censoring = (censoring_col is not None and censoring_col in trajectories.columns) or 'censored' in trajectories.columns
    censoring_column = censoring_col if censoring_col is not None and censoring_col in trajectories.columns else 'censored'
    
    # Filter by max_time if specified
    if max_time is not None:
        trajectories = trajectories[trajectories['time'] <= max_time]
    
    # Initialize event data list
    events = []
    
    # Process each simulation separately
    for sim_id, sim_data in trajectories.groupby('simulation'):
        # Sort by time to ensure correct transition order
        sim_data = sim_data.sort_values('time')
        
        # Get patient ID if available
        patient_id = sim_data['patient_id'].iloc[0] if 'patient_id' in sim_data.columns else None
        
        # Determine if trajectory was censored
        was_censored = sim_data[censoring_column].iloc[-1] if has_censoring and censoring_column in sim_data.columns else False
        
        # Iterate through rows to extract transitions
        for i in range(1, len(sim_data)):
            from_state = sim_data['state'].iloc[i-1]
            to_state = sim_data['state'].iloc[i]
            event_time = sim_data['time'].iloc[i]
            
            # Add transition event
            event = {
                'id': sim_id,
                'time': event_time,
                'from_state': from_state,
                'to_state': to_state,
                'censored': False  # Actual transitions are not censored
            }
            
            # Add patient ID if available
            if patient_id is not None:
                event['patient_id'] = patient_id
                
            events.append(event)
        
        # Add censoring event if trajectory was censored
        if was_censored:
            # Last observed state
            last_state = sim_data['state'].iloc[-1]
            last_time = sim_data['time'].iloc[-1]
            
            # Add censoring event
            censoring_event = {
                'id': sim_id,
                'time': last_time,
                'from_state': last_state,
                'to_state': last_state,  # No transition for censoring
                'censored': True
            }
            
            # Add patient ID if available
            if patient_id is not None:
                censoring_event['patient_id'] = patient_id
                
            events.append(censoring_event)
    
    # Create DataFrame from events
    if events:
        return pd.DataFrame(events)
    else:
        return pd.DataFrame(columns=['id', 'time', 'from_state', 'to_state', 'censored'])


def _aalen_johansen_estimator(
    event_data: pd.DataFrame,
    target_state: int,
    max_time: Optional[float] = None,
    time_grid: Optional[np.ndarray] = None,
    n_grid_points: int = 100,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Calculate CIF using the Aalen-Johansen estimator.
    
    The Aalen-Johansen estimator is a nonparametric estimator for the 
    transition probability matrix in a multistate model with right-censoring.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        DataFrame with transition events from prepare_event_data
    target_state : int
        The state for which to calculate cumulative incidence
    max_time : Optional[float], optional
        Maximum time value to include in results
    time_grid : Optional[np.ndarray], optional
        Custom time grid for CIF calculation. If None, uses all unique event times
        or an evenly spaced grid based on n_grid_points.
    n_grid_points : int, optional
        Number of points in the time grid if generating evenly spaced points
    ci_level : float, optional
        Confidence interval level (0-1)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time: Time values at which CIF is calculated
        - cif: Cumulative incidence function values
        - lower_ci: Lower confidence interval bound
        - upper_ci: Upper confidence interval bound
    """
    # Make a copy to avoid modifying the original
    event_data = event_data.copy()
    
    # Ensure we have required columns
    required_cols = ['id', 'time', 'from_state', 'to_state', 'censored']
    for col in required_cols:
        if col not in event_data.columns:
            raise ValueError(f"Required column '{col}' is missing from event_data")
    
    # Filter by max_time if specified
    if max_time is not None:
        event_data = event_data[event_data['time'] <= max_time]
    else:
        max_time = event_data['time'].max()
    
    # Create time grid if not provided
    if time_grid is None:
        if n_grid_points <= 0:
            # Use unique event times
            time_points = np.sort(event_data['time'].unique())
            # Add 0 if not present
            if time_points[0] > 0:
                time_points = np.insert(time_points, 0, 0.0)
        else:
            # Create evenly spaced grid
            time_points = np.linspace(0, max_time, n_grid_points)
        
        time_grid = time_points
    
    # Ensure time grid starts with 0
    if time_grid[0] > 0:
        time_grid = np.insert(time_grid, 0, 0.0)
    
    # Get all unique states
    all_states = np.unique(np.concatenate([
        event_data['from_state'].unique(), 
        event_data['to_state'].unique()
    ]))
    n_states = len(all_states)
    
    # Map states to indices (0 to n_states-1)
    state_to_idx = {state: i for i, state in enumerate(all_states)}
    
    # Initialize transition probability matrix P(0,t) for all time points
    # P[t, i, j] = P(being in state j at time t | being in state i at time 0)
    P = np.zeros((len(time_grid), n_states, n_states))
    
    # At time 0, P is the identity matrix (probability 1 of staying in the same state)
    P[0] = np.eye(n_states)
    
    # Sort event data by time
    event_data = event_data.sort_values('time')
    
    # Group events by unique times
    time_groups = event_data.groupby('time')
    
    # Get total number of individuals at risk at the beginning
    n_individuals = event_data['id'].nunique()
    
    # Track individuals at risk for each state at each time
    at_risk = np.zeros((len(time_grid), n_states), dtype=int)
    
    # Initialize at_risk at time 0 with everyone in state 0
    at_risk[0, state_to_idx[all_states[0]]] = n_individuals
    
    # Setup for counting transitions
    current_states = {id_: all_states[0] for id_ in event_data['id'].unique()}
    
    # For each transition time, update P
    for time_idx in range(1, len(time_grid)):
        t = time_grid[time_idx]
        
        # Find events at times <= t
        events_until_t = event_data[event_data['time'] <= t]
        
        # Update current states for all individuals
        for _, event in events_until_t.iterrows():
            id_ = event['id']
            current_states[id_] = event['to_state']
        
        # Count individuals in each state at time t
        state_counts = {state: 0 for state in all_states}
        for state in current_states.values():
            state_counts[state] += 1
        
        # Update at_risk for current time
        for state, count in state_counts.items():
            at_risk[time_idx, state_to_idx[state]] = count
        
        # Find events at exactly time t
        if t in time_groups.groups:
            events_at_t = time_groups.get_group(t)
            
            # Calculate transition matrix A for this time point
            A = np.eye(n_states)
            
            # For each transition type at this time
            transition_counts = events_at_t.groupby(['from_state', 'to_state']).size()
            
            for (from_state, to_state), count in transition_counts.items():
                # Skip censoring transitions (where from_state = to_state)
                if from_state == to_state:
                    continue
                
                # Convert states to indices
                from_idx = state_to_idx[from_state]
                to_idx = state_to_idx[to_state]
                
                # Get number at risk in from_state
                n_at_risk = at_risk[time_idx-1, from_idx]
                
                if n_at_risk > 0:
                    # Probability of transition
                    A[from_idx, from_idx] -= count / n_at_risk
                    A[from_idx, to_idx] += count / n_at_risk
            
            # Update P(0,t) = P(0,t-1) * A
            P[time_idx] = np.dot(P[time_idx-1], A)
        else:
            # No events at this time, so P(0,t) = P(0,t-1)
            P[time_idx] = P[time_idx-1]
    
    # Extract CIF for target state (from initial state)
    initial_state_idx = state_to_idx[all_states[0]]
    target_state_idx = state_to_idx[target_state]
    cif_values = P[:, initial_state_idx, target_state_idx]
    
    # Calculate confidence intervals using Greenwood's formula
    var_terms = np.zeros_like(cif_values)
    for i in range(1, len(time_grid)):
        # Variance calculation similar to Kaplan-Meier but adapted for CIF
        if cif_values[i] > 0 and cif_values[i] < 1:
            var_terms[i] = cif_values[i] * (1 - cif_values[i]) / n_individuals
    
    # Apply z-score for desired confidence level
    np.random.seed(42)  # For reproducibility
    z = abs(np.percentile(np.random.normal(0, 1, 10000), [(1-ci_level)/2*100, (1+ci_level)/2*100]))
    
    lower_ci = np.maximum(0, cif_values - z[0] * np.sqrt(var_terms))
    upper_ci = np.minimum(1, cif_values + z[1] * np.sqrt(var_terms))
    
    # Create result DataFrame
    cif_df = pd.DataFrame({
        'time': time_grid,
        'cif': cif_values,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
    })
    
    return cif_df


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