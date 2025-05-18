"""Simulation utilities for continuous-time multistate models."""

from typing import Dict, List, Optional, Union, Sequence, Any
import numpy as np
import pandas as pd
import torch
import warnings
import scipy.linalg

from ..models import ContinuousMultiStateNN


def generate_censoring_times(
    n_samples: int,
    censoring_rate: float = 0.3,
    max_time: float = 10.0,
    covariates: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Generate random censoring times.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    censoring_rate : float, optional
        Target censoring rate (0-1)
    max_time : float, optional
        Maximum time value
    covariates : Optional[np.ndarray], optional
        Optional covariates to influence censoring times
    random_state : Optional[int], optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of censoring times
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Basic censoring: Exponential distribution scaled to achieve target censoring rate
    # If censoring_rate = 0, return infinities (no censoring)
    if censoring_rate <= 0:
        return np.ones(n_samples) * np.inf
    
    # Higher rate means more censoring (shorter times)
    # Scale to get a certain % of censoring times below max_time
    rate = -np.log(1 - censoring_rate) / max_time
    
    # Generate censoring times from exponential distribution
    censoring_times = np.random.exponential(scale=1/rate, size=n_samples)
    
    # If covariates are provided, adjust censoring times based on them
    if covariates is not None:
        # Simple adjustment: multiply by scaled norm of covariates
        # This makes censoring dependent on covariates (informative censoring)
        if covariates.ndim == 2 and covariates.shape[0] == n_samples:
            # Normalize covariate norms to mean=1 to preserve the overall censoring rate
            covariate_factors = np.linalg.norm(covariates, axis=1)
            covariate_factors = covariate_factors / np.mean(covariate_factors) if np.mean(covariate_factors) > 0 else 1.0
            
            # Apply adjustment (clip to avoid extreme values)
            censoring_times = censoring_times * np.clip(covariate_factors, 0.5, 2.0)
    
    return censoring_times


def adjust_transitions_for_time(P: np.ndarray, time_diff: float) -> np.ndarray:
    """
    Adjust transition matrix P for a different time step using
    the standard continuous-time approach.
    
    P(t) = exp(Q * t) where Q = log(P)
    P(k*t) = exp(Q * k*t) = exp(k * Q*t) = (P(t))^k
    
    Parameters
    ----------
    P : np.ndarray
        Transition probability matrix
    time_diff : float
        Time scaling factor
        
    Returns
    -------
    np.ndarray
        Adjusted transition matrix
    """
    try:
        # Method 1: Matrix logarithm and exponential
        # Compute rate matrix Q = log(P)
        Q = scipy.linalg.logm(P)
        
        # Compute adjusted P = exp(time_diff * Q)
        P_adjusted = scipy.linalg.expm(time_diff * Q)
        
        # Ensure probabilities are valid
        if not np.all(np.isfinite(P_adjusted)) or np.any(P_adjusted < 0) or np.any(P_adjusted > 1):
            raise ValueError("Matrix method produced invalid probabilities")
            
        return P_adjusted
        
    except Exception as e:
        # Fallback: Use a simpler approximation with direct power
        # For time_diff > 1, P(n*t) ≈ P^n where n is the closest integer
        n = round(time_diff)
        if n > 0:
            P_power = np.linalg.matrix_power(P, n)
            
            # For non-integer time_diff, interpolate
            if not np.isclose(n, time_diff):
                alpha = time_diff - np.floor(time_diff)
                P_power_ceil = np.linalg.matrix_power(P, n+1)
                P_adjusted = (1-alpha) * P_power + alpha * P_power_ceil
            else:
                P_adjusted = P_power
                
            return P_adjusted
        elif n == 0:
            # For very small time steps, use identity matrix
            return np.eye(P.shape[0])
        else:
            # Should not happen with positive time_diff
            raise ValueError(f"Invalid time_diff: {time_diff}")


def simulate_continuous_patient_trajectory(
    model: ContinuousMultiStateNN,
    x: torch.Tensor,
    start_state: int,
    max_time: float,
    n_simulations: int = 100,
    time_grid: Optional[np.ndarray] = None,
    time_step: float = 0.1,
    censoring_times: Optional[np.ndarray] = None,
    censoring_rate: float = 0.0,
    seed: Optional[int] = None,
) -> List[pd.DataFrame]:
    """Simulate patient trajectories through the continuous-time multistate model.
    
    Parameters
    ----------
    model : ContinuousMultiStateNN
        Trained continuous-time multistate model
    x : torch.Tensor
        Patient features (shape: [1, input_dim])
    start_state : int
        Initial state
    max_time : float
        Maximum time value to simulate until
    n_simulations : int, optional
        Number of simulations to run
    time_grid : Optional[np.ndarray], optional
        Custom time grid for simulation. If None, uses regular steps of time_step.
    time_step : float, optional
        Time step size for default time grid
    censoring_times : Optional[np.ndarray], optional
        Pre-generated censoring times, shape (n_simulations,). If None and 
        censoring_rate > 0, censoring times will be generated automatically.
    censoring_rate : float, optional
        Target censoring rate if censoring_times not provided
    seed : Optional[int], optional
        Random seed for reproducibility
        
    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames containing simulated trajectories
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Add batch dimension if needed
    
    # Make sure we only have one patient
    if x.shape[0] != 1:
        raise ValueError("x should represent a single patient (shape: [1, n_features])")
    
    # Generate time grid if not provided
    if time_grid is None:
        time_grid = np.arange(0, max_time + time_step, time_step)
    
    # Generate censoring times if they should be used but weren't provided
    enable_censoring = censoring_rate > 0 or censoring_times is not None
    if enable_censoring and censoring_times is None:
        censoring_times = generate_censoring_times(
            n_samples=n_simulations,
            censoring_rate=censoring_rate,
            max_time=max_time,
            covariates=x.squeeze().cpu().numpy()[np.newaxis, :].repeat(n_simulations, axis=0),
            random_state=seed
        )
    elif censoring_times is not None and len(censoring_times) != n_simulations:
        raise ValueError(f"Expected {n_simulations} censoring times, got {len(censoring_times)}")
    
    trajectories = []
    
    # Pre-compute intensity matrix for efficiency
    intensity_matrix = model.intensity_matrix(x).squeeze(0).detach().cpu().numpy()
    
    for sim_idx in range(n_simulations):
        # Start with initial state
        current_state = start_state
        states = [current_state]
        times = [0.0]
        
        # Get censoring time for this simulation if censoring is enabled
        censoring_time = None
        censored = False
        if enable_censoring:
            censoring_time = censoring_times[sim_idx]
            if np.isfinite(censoring_time) and censoring_time <= max_time:
                censored = True
        
        # Generate realization of continuous-time Markov chain
        current_time = 0.0
        
        while current_time < max_time and model.state_transitions[current_state]:
            # Check if we've reached censoring time
            if censored and current_time >= censoring_time:
                break
            
            # Get transition rates from current state (the row of intensity matrix)
            rate_row = intensity_matrix[current_state].copy()
            
            # Set diagonal to 0 to get only outgoing rates
            rate_row[current_state] = 0
            
            # Total exit rate from current state
            total_rate = np.sum(rate_row)
            
            if total_rate <= 0:
                # No transitions possible, stay in current state until max_time
                # This can happen with numerical issues or with almost-absorbing states
                if current_time < max_time:
                    states.append(current_state)
                    times.append(max_time)
                break
            
            # Time until next transition follows exponential distribution
            time_to_next = np.random.exponential(scale=1.0/total_rate)
            
            # Update current time
            new_time = current_time + time_to_next
            
            # Check if we exceed max_time or censoring time
            if new_time > max_time:
                # Include the final state at max_time
                states.append(current_state)
                times.append(max_time)
                break
            elif censored and new_time > censoring_time:
                # Include the final state at censoring time
                states.append(current_state)
                times.append(censoring_time)
                break
            
            # Choose the next state based on transition rates
            transition_probs = rate_row / total_rate
            next_state = np.random.choice(len(rate_row), p=transition_probs)
            
            # Record transition
            current_time = new_time
            current_state = next_state
            states.append(current_state)
            times.append(current_time)
            
            # Check if we reached an absorbing state
            if not model.state_transitions[current_state]:
                break
        
        # Create trajectory dataframe
        trajectory_data = {
            'time': times,
            'state': states,
            'simulation': sim_idx,
        }
        
        # Add censoring information if enabled
        if enable_censoring:
            trajectory_data['censored'] = censored
        
        # Convert to dataframe
        trajectory_df = pd.DataFrame(trajectory_data)
        
        # Add time grid observations if needed
        # This ensures we have observations at specific time points for visualization/analysis
        if len(time_grid) > 0:
            grid_observations = []
            
            for grid_time in time_grid:
                if grid_time > max_time or (censored and grid_time > censoring_time):
                    # Skip times beyond max_time or censoring time
                    continue
                    
                # Find the state at this grid time
                # (last state with time <= grid_time)
                idx = np.searchsorted(times, grid_time, side='right') - 1
                if idx >= 0:
                    grid_state = states[idx]
                    grid_observations.append({
                        'time': grid_time,
                        'state': grid_state,
                        'simulation': sim_idx,
                        'censored': censored if enable_censoring else False,
                        'grid_point': True
                    })
            
            # Add grid observations if any were created
            if grid_observations:
                grid_df = pd.DataFrame(grid_observations)
                
                # Add indicator for original trajectory points
                trajectory_df['grid_point'] = False
                
                # Combine original and grid trajectories
                trajectory_df = pd.concat([trajectory_df, grid_df], ignore_index=True)
                
                # Sort by time
                trajectory_df = trajectory_df.sort_values(['simulation', 'time'])
            
        trajectories.append(trajectory_df)
    
    return trajectories


def simulate_continuous_cohort_trajectories(
    model: ContinuousMultiStateNN,
    cohort_features: torch.Tensor,
    start_state: int,
    max_time: float,
    n_simulations_per_patient: int = 10,
    time_grid: Optional[np.ndarray] = None,
    time_step: float = 0.1,
    censoring_rate: float = 0.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate trajectories for a cohort of patients using a continuous-time model.
    
    Parameters
    ----------
    model : ContinuousMultiStateNN
        Trained continuous-time multistate model
    cohort_features : torch.Tensor
        Features for each patient in the cohort (shape: [n_patients, input_dim])
    start_state : int
        Initial state for all patients
    max_time : float
        Maximum time value to simulate until
    n_simulations_per_patient : int, optional
        Number of simulations to run per patient
    time_grid : Optional[np.ndarray], optional
        Custom time grid for simulation. If None, uses regular steps of time_step.
    time_step : float, optional
        Time step size for default time grid
    censoring_rate : float, optional
        Target censoring rate if censoring_times not provided
    seed : Optional[int], optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing all simulated trajectories
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    all_trajectories = []
    
    for patient_idx in range(cohort_features.shape[0]):
        patient_x = cohort_features[patient_idx:patient_idx+1]
        
        # Generate censoring times for this patient if censoring is enabled
        patient_seed = seed + patient_idx if seed is not None else None
        censoring_times = None
        if censoring_rate > 0:
            censoring_times = generate_censoring_times(
                n_samples=n_simulations_per_patient,
                censoring_rate=censoring_rate,
                max_time=max_time,
                random_state=patient_seed
            )
        
        # Simulate trajectories for this patient
        patient_trajectories = simulate_continuous_patient_trajectory(
            model=model,
            x=patient_x,
            start_state=start_state,
            max_time=max_time,
            n_simulations=n_simulations_per_patient,
            time_grid=time_grid,
            time_step=time_step,
            censoring_times=censoring_times,
            censoring_rate=censoring_rate,
            seed=patient_seed
        )
        
        # Add patient ID to each trajectory
        for traj in patient_trajectories:
            traj['patient_id'] = patient_idx
        
        all_trajectories.extend(patient_trajectories)
    
    # Combine all trajectories into a single DataFrame
    return pd.concat(all_trajectories, ignore_index=True)