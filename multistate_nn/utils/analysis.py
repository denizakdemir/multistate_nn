"""Analysis utilities for MultiStateNN models."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def calculate_cif(
    trajectories: pd.DataFrame,
    target_state: int,
    max_time: Optional[int] = None,
    by_patient: bool = False,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Calculate cumulative incidence function (CIF) from simulated trajectories.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame of simulated trajectories from simulate_patient_trajectory
        or simulate_cohort_trajectories
    target_state : int
        The state for which to calculate cumulative incidence
    max_time : Optional[int], optional
        Maximum time point to include in results
    by_patient : bool, optional
        If True, calculate separate CIFs for each patient
    ci_level : float, optional
        Confidence interval level (0-1)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time: Time points
        - cif: Cumulative incidence function values
        - lower_ci: Lower confidence interval bound
        - upper_ci: Upper confidence interval bound
        - patient_id: (only if by_patient=True) Patient identifier
    """
    # Determine the maximum time if not provided
    if max_time is None:
        max_time = trajectories['time'].max()
    
    # Filter to only include data up to max_time
    trajectories = trajectories[trajectories['time'] <= max_time].copy()
    
    # Group by patient_id if by_patient=True
    if by_patient:
        if 'patient_id' not in trajectories.columns:
            raise ValueError("Trajectories dataframe must have 'patient_id' column when by_patient=True")
        
        patient_groups = trajectories.groupby('patient_id')
        patient_cifs = []
        
        for patient_id, patient_data in patient_groups:
            # Calculate this patient's CIF
            patient_cif = _calculate_single_cif(
                patient_data, 
                target_state, 
                max_time, 
                ci_level
            )
            patient_cif['patient_id'] = patient_id
            patient_cifs.append(patient_cif)
        
        return pd.concat(patient_cifs, ignore_index=True)
    else:
        # Calculate CIF for the entire cohort
        return _calculate_single_cif(trajectories, target_state, max_time, ci_level)


def _calculate_single_cif(
    trajectories: pd.DataFrame,
    target_state: int,
    max_time: int,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Helper function to calculate CIF for a single group of trajectories."""
    # Initialize arrays to store results
    times = list(range(max_time + 1))
    cifs = np.zeros(max_time + 1)
    
    # For each simulation, find the first occurrence of target state (if any)
    sim_groups = trajectories.groupby('simulation')
    n_sims = len(sim_groups)
    
    # For calculating confidence intervals
    all_incidence = np.zeros((n_sims, max_time + 1))
    
    for sim_idx, (_, sim_data) in enumerate(sim_groups):
        # Sort by time (should already be sorted, but just to be safe)
        sim_data = sim_data.sort_values('time')
        
        # Find first occurrence of target state (if any)
        target_rows = sim_data[sim_data['state'] == target_state]
        
        if len(target_rows) > 0:
            first_occurrence = target_rows['time'].iloc[0]
            
            # Set incidence to 1 for all times >= first occurrence
            if first_occurrence <= max_time:
                all_incidence[sim_idx, int(first_occurrence):] = 1
    
    # Calculate mean CIF and confidence intervals
    cifs = np.mean(all_incidence, axis=0)
    
    # Calculate confidence intervals
    z = abs(np.percentile(np.random.normal(0, 1, 10000), [(1-ci_level)/2*100, (1+ci_level)/2*100]))
    lower_ci = np.maximum(0, cifs - z[0] * np.sqrt(cifs * (1 - cifs) / n_sims))
    upper_ci = np.minimum(1, cifs + z[1] * np.sqrt(cifs * (1 - cifs) / n_sims))
    
    # Create result DataFrame
    cif_df = pd.DataFrame({
        'time': times,
        'cif': cifs,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
    })
    
    return cif_df