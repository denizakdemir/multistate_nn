"""Simulation utilities for MultiStateNN models."""

from typing import Dict, List, Optional, Union, Sequence
import numpy as np
import pandas as pd
import torch
from ..models import BaseMultiStateNN


def generate_synthetic_data(
    n_samples: int = 1000,
    n_covariates: int = 3,
    n_states: int = 4,
    n_time_points: int = 5,
    state_transitions: Optional[Dict[int, List[int]]] = None,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic multistate data for testing.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples
    n_covariates : int, optional
        Number of covariates
    n_states : int, optional
        Number of states
    n_time_points : int, optional
        Number of time points
    state_transitions : dict[int, list[int]], optional
        State transition structure. If None, uses a forward-only structure.
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    df : pd.DataFrame
        Synthetic dataset with columns:
        - time: Time index
        - from_state: Source state
        - to_state: Target state
        - covariate_0, ...: Covariates
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Default forward-only transitions if none provided
    if state_transitions is None:
        state_transitions = {
            i: [j for j in range(i + 1, min(i + 3, n_states))] for i in range(n_states)
        }

    # Generate covariates
    X = np.random.normal(0, 1, (n_samples, n_covariates))

    # Generate transitions based on a simple logistic model
    records = []
    for i in range(n_samples):
        current_state = 0
        for t in range(n_time_points):
            if not state_transitions[current_state]:  # Absorbing state
                break

            # Transition probabilities influenced by covariates
            next_states = state_transitions[current_state]
            logits = np.dot(X[i], np.random.normal(0, 1, (n_covariates, len(next_states))))
            probs = np.exp(logits) / np.sum(np.exp(logits))
            next_state = np.random.choice(next_states, p=probs)

            records.append(
                {
                    "time": t,
                    "from_state": current_state,
                    "to_state": next_state,
                    **{f"covariate_{j}": X[i, j] for j in range(n_covariates)},
                }
            )

            current_state = next_state

    return pd.DataFrame(records)


def simulate_patient_trajectory(
    model: BaseMultiStateNN,
    x: torch.Tensor,
    start_state: int,
    max_time: int,
    n_simulations: int = 100,
    seed: Optional[int] = None,
) -> List[pd.DataFrame]:
    """Simulate patient trajectories through the multistate model.
    
    Parameters
    ----------
    model : BaseMultiStateNN
        Trained multistate model
    x : torch.Tensor
        Patient features (shape: [1, n_features])
    start_state : int
        Initial state
    max_time : int
        Maximum number of time points to simulate
    n_simulations : int, optional
        Number of simulations to run
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames containing simulated trajectories.
        Each DataFrame has columns:
        - time: Time index
        - state: Patient state at that time
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Add batch dimension if needed
    
    # Make sure we only have one patient
    if x.shape[0] != 1:
        raise ValueError("x should represent a single patient (shape: [1, n_features])")
    
    trajectories = []
    
    for sim_idx in range(n_simulations):
        # Start with initial state
        current_state = start_state
        states = [current_state]
        times = [0]
        
        # Simulate until max_time or absorbing state
        t = 0
        while t < max_time - 1 and model.state_transitions[current_state]:
            t += 1
            
            # Get transition probabilities
            probs = model.predict_proba(x, time_idx=t, from_state=current_state)
            probs = probs.squeeze().detach().cpu().numpy()
            
            # Sample next state
            next_states = model.state_transitions[current_state]
            
            if len(next_states) == 0:  # Check again for absorbing state
                break
                
            # Choose next state based on probabilities
            next_state_idx = np.random.choice(len(next_states), p=probs)
            current_state = next_states[next_state_idx]
            
            states.append(current_state)
            times.append(t)
            
        # Create trajectory dataframe
        trajectory_df = pd.DataFrame({
            'time': times,
            'state': states,
            'simulation': sim_idx
        })
        
        trajectories.append(trajectory_df)
    
    return trajectories


def simulate_cohort_trajectories(
    model: BaseMultiStateNN,
    cohort_features: torch.Tensor,
    start_state: int,
    max_time: int,
    n_simulations_per_patient: int = 10,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate trajectories for a cohort of patients.
    
    Parameters
    ----------
    model : BaseMultiStateNN
        Trained multistate model
    cohort_features : torch.Tensor
        Features for each patient in the cohort (shape: [n_patients, n_features])
    start_state : int
        Initial state for all patients
    max_time : int
        Maximum number of time points to simulate
    n_simulations_per_patient : int, optional
        Number of simulations to run per patient
    seed : Optional[int], optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing all simulated trajectories with columns:
        - patient_id: Identifier for the patient
        - simulation: Simulation run number
        - time: Time index
        - state: Patient state at that time
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    all_trajectories = []
    
    for patient_idx in range(cohort_features.shape[0]):
        patient_x = cohort_features[patient_idx:patient_idx+1]
        
        # Simulate trajectories for this patient
        patient_trajectories = simulate_patient_trajectory(
            model=model,
            x=patient_x,
            start_state=start_state,
            max_time=max_time,
            n_simulations=n_simulations_per_patient,
            seed=seed + patient_idx if seed is not None else None
        )
        
        # Add patient ID to each trajectory
        for sim_idx, traj in enumerate(patient_trajectories):
            traj['patient_id'] = patient_idx
        
        all_trajectories.extend(patient_trajectories)
    
    # Combine all trajectories into a single DataFrame
    return pd.concat(all_trajectories, ignore_index=True)