"""Aalen-Johansen estimator for MultiStateNN models."""

from typing import Optional, Tuple, Union, Sequence, List, Dict
import numpy as np
import pandas as pd
import warnings


def prepare_event_data(
    trajectories: pd.DataFrame,
    max_time: Optional[float] = None,
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
    has_censoring = 'censored' in trajectories.columns
    
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
        was_censored = sim_data['censored'].iloc[-1] if has_censoring else False
        
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


def aalen_johansen_estimator(
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


def calculate_cif_with_censoring(
    trajectories: pd.DataFrame,
    target_state: int,
    max_time: Optional[float] = None,
    by_patient: bool = False,
    ci_level: float = 0.95,
    time_grid: Optional[np.ndarray] = None,
    n_grid_points: int = 100,
) -> pd.DataFrame:
    """Calculate cumulative incidence function (CIF) with proper handling of censoring.
    
    This function uses the Aalen-Johansen estimator to calculate CIFs
    accounting for right-censoring.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame of simulated trajectories
    target_state : int
        The state for which to calculate cumulative incidence
    max_time : Optional[float], optional
        Maximum time value to include in results
    by_patient : bool, optional
        If True, calculate separate CIFs for each patient
    ci_level : float, optional
        Confidence interval level (0-1)
    time_grid : Optional[np.ndarray], optional
        Custom time grid for CIF calculation. If None, uses all unique time points 
        or an evenly spaced grid based on n_grid_points.
    n_grid_points : int, optional
        Number of points in the time grid if generating evenly spaced points
        
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
    # Prepare event data for Aalen-Johansen estimator
    event_data = prepare_event_data(trajectories, max_time)
    
    # Group by patient_id if by_patient=True
    if by_patient:
        if 'patient_id' not in trajectories.columns:
            raise ValueError("Trajectories dataframe must have 'patient_id' column when by_patient=True")
        
        patient_groups = event_data.groupby('patient_id')
        patient_cifs = []
        
        for patient_id, patient_data in patient_groups:
            # Calculate this patient's CIF using Aalen-Johansen
            patient_cif = aalen_johansen_estimator(
                patient_data, 
                target_state, 
                max_time, 
                time_grid, 
                n_grid_points, 
                ci_level
            )
            patient_cif['patient_id'] = patient_id
            patient_cifs.append(patient_cif)
        
        return pd.concat(patient_cifs, ignore_index=True)
    else:
        # Calculate CIF for the entire cohort
        return aalen_johansen_estimator(
            event_data, 
            target_state, 
            max_time, 
            time_grid, 
            n_grid_points, 
            ci_level
        )