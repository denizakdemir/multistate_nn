"""Simulation utilities for MultiStateNN models."""

from typing import Dict, List, Optional, Union, Sequence, Any
import numpy as np
import pandas as pd
import torch
import warnings
from ..models import BaseMultiStateNN


def generate_synthetic_data(
    n_samples: int = 1000,
    n_covariates: int = 3,
    n_states: int = 4,
    n_time_points: int = 5,
    state_transitions: Optional[Dict[int, List[int]]] = None,
    random_seed: Optional[int] = None,
    time_values: Optional[Sequence[float]] = None,
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
    time_values : Sequence[float], optional
        Custom time values to use instead of 0, 1, 2, ... If provided, must be of length n_time_points.

    Returns
    -------
    df : pd.DataFrame
        Synthetic dataset with columns:
        - time: Time values (custom or indices)
        - time_idx: Time indices (only if custom time_values provided)
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

    # Handle custom time values
    if time_values is not None:
        if len(time_values) != n_time_points:
            raise ValueError(f"time_values must have length {n_time_points}, but got {len(time_values)}")
        # Create mapping from index to time value
        idx_to_time = {i: t for i, t in enumerate(time_values)}
    else:
        # Use default time indices
        idx_to_time = {i: i for i in range(n_time_points)}

    # Generate covariates
    X = np.random.normal(0, 1, (n_samples, n_covariates))

    # Generate transitions based on a simple logistic model
    records = []
    for i in range(n_samples):
        current_state = 0
        for t_idx in range(n_time_points):
            if not state_transitions[current_state]:  # Absorbing state
                break

            # Transition probabilities influenced by covariates
            next_states = state_transitions[current_state]
            logits = np.dot(X[i], np.random.normal(0, 1, (n_covariates, len(next_states))))
            probs = np.exp(logits) / np.sum(np.exp(logits))
            next_state = np.random.choice(next_states, p=probs)

            record = {
                "time": idx_to_time[t_idx],
                "from_state": current_state,
                "to_state": next_state,
                **{f"covariate_{j}": X[i, j] for j in range(n_covariates)},
            }
            
            # Add time_idx column if using custom time values
            if time_values is not None:
                record["time_idx"] = t_idx
                
            records.append(record)

            current_state = next_state

    return pd.DataFrame(records)


def generate_censoring_times(
    n_samples: int = 100,
    censoring_rate: float = 0.3,
    max_time: float = 10.0,
    covariates: Optional[np.ndarray] = None,
    covariate_effects: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate random censoring times.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate
    censoring_rate : float, optional
        Target censoring rate (proportion of samples with censoring times)
    max_time : float, optional
        Maximum time
    covariates : Optional[np.ndarray], optional
        Array of covariates, shape (n_samples, n_covariates)
    covariate_effects : Optional[np.ndarray], optional
        Effects of covariates on censoring times, shape (n_covariates,)
    random_state : Optional[int], optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Array of censoring times, shape (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Base censoring times from exponential distribution
    scale = max_time * 1.5  # Slightly larger than max_time to ensure reasonable censoring rates
    censoring_times = np.random.exponential(scale=scale, size=n_samples)
    
    # Apply covariate effects if provided
    if covariates is not None and covariate_effects is not None:
        if covariates.shape[0] != n_samples:
            raise ValueError(f"Expected {n_samples} samples in covariates, got {covariates.shape[0]}")
        if covariate_effects.shape[0] != covariates.shape[1]:
            raise ValueError(
                f"Expected {covariates.shape[1]} covariate effects, got {covariate_effects.shape[0]}"
            )
        
        # Compute linear predictor
        linear_pred = np.dot(covariates, covariate_effects)
        
        # Scale censoring times by exp(linear_pred)
        # Positive effects -> longer censoring times
        # Negative effects -> shorter censoring times
        censoring_times *= np.exp(linear_pred)
    
    # Ensure some censoring by setting a proportion of samples to have no censoring
    # Invert the logic: we want `censoring_rate` proportion of samples to have finite censoring times
    no_censoring_mask = np.random.random(n_samples) >= censoring_rate
    censoring_times[no_censoring_mask] = np.inf
    
    return censoring_times


def simulate_patient_trajectory(
    model: BaseMultiStateNN,
    x: torch.Tensor,
    start_state: int,
    max_time: int,
    n_simulations: int = 100,
    censoring_times: Optional[np.ndarray] = None,
    censoring_rate: float = 0.0,
    time_adjusted: bool = False,
    seed: Optional[int] = None,
    use_original_time: bool = True,
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
    censoring_times : Optional[np.ndarray], optional
        Pre-generated censoring times, shape (n_simulations,). If None and censoring_rate > 0, 
        censoring times will be generated automatically.
    censoring_rate : float, optional
        Target censoring rate if censoring_times not provided. Set to 0 to disable censoring.
    time_adjusted : bool, optional
        Whether to adjust transition probabilities based on time window sizes.
    seed : Optional[int], optional
        Random seed for reproducibility
    use_original_time : bool, optional
        Whether to use original time values in the output
        
    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames containing simulated trajectories.
        Each DataFrame has columns:
        - time_idx: Time index (internal representation)
        - time: Time value (original scale, if available)
        - state: Patient state at that time
        - simulation: Simulation index
        - censored: Whether the trajectory was censored (only if censoring_rate > 0 or censoring_times provided)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Add batch dimension if needed
    
    # Make sure we only have one patient
    if x.shape[0] != 1:
        raise ValueError("x should represent a single patient (shape: [1, n_features])")
    
    # Check if model has time_mapper
    has_time_mapper = hasattr(model, 'time_mapper') and model.time_mapper is not None
    
    # Handle time mapping
    if use_original_time and not has_time_mapper:
        if time_adjusted:
            raise ValueError("Model does not have a time_mapper attribute. Cannot use time-adjusted simulation.")
        else:
            warnings.warn("Model does not have a time_mapper attribute. Using index-based time instead.")
            use_original_time = False
    
    # When using original time, max_time should be the index not the original time value
    # If max_time exceeds our available time points, we extend the time mapper
    original_max_time = max_time
    if use_original_time and has_time_mapper:
        # First, check if max_time is a time value or an index
        if max_time > 100:  # Heuristic: large values likely represent real time, not an index
            # max_time is likely a time value (e.g., 365 days), not an index
            # Try to find or create an index for this time value
            max_time_value = float(max_time)
            
            # Check if max_time is beyond our current range
            if max_time_value > model.time_mapper.time_values[-1]:
                # Extend the time mapper
                extended_mapper = model.time_mapper.extend_with_extrapolation(max_time_value)
                # Update model's time mapper
                model.time_mapper = extended_mapper
                # Set max_time to the maximum index
                max_time = len(model.time_mapper.time_values) - 1
            else:
                # Find the closest index for this time value
                max_time = model.time_mapper.get_closest_idx(max_time_value)
        
        # If max_time is still an index that exceeds our available points
        elif max_time >= model.time_mapper.n_time_points:
            # Extend time mapper to accommodate more time points
            # First, estimate what the time value would be
            last_time = model.time_mapper.time_values[-1]
            avg_step = 1.0
            if len(model.time_mapper.time_values) > 1:
                avg_step = (model.time_mapper.time_values[-1] - model.time_mapper.time_values[0]) / (len(model.time_mapper.time_values) - 1)
            
            # Estimate max time value needed
            estimated_max_time_value = last_time + (max_time - model.time_mapper.n_time_points + 1) * avg_step
            # Extend the time mapper
            extended_mapper = model.time_mapper.extend_with_extrapolation(estimated_max_time_value, n_points=max_time - model.time_mapper.n_time_points + 1)
            # Update model's time mapper
            model.time_mapper = extended_mapper
            
    # Generate censoring times if they should be used but weren't provided
    enable_censoring = censoring_rate > 0 or censoring_times is not None
    if enable_censoring and censoring_times is None:
        censoring_times = generate_censoring_times(
            n_samples=n_simulations,
            censoring_rate=censoring_rate,
            max_time=original_max_time if use_original_time and has_time_mapper else max_time,
            covariates=x.squeeze().cpu().numpy()[np.newaxis, :].repeat(n_simulations, axis=0),
            random_state=seed
        )
    elif censoring_times is not None and len(censoring_times) != n_simulations:
        raise ValueError(f"Expected {n_simulations} censoring times, got {len(censoring_times)}")
    
    # Get time differences for time-adjusted transition probabilities
    time_diffs = None
    if time_adjusted and has_time_mapper:
        time_values = model.time_mapper.time_values
        time_diffs = np.diff(time_values, prepend=time_values[0])
        # Normalize to make the smallest time step = 1.0
        min_diff = np.min(time_diffs[time_diffs > 0])  # Only consider positive differences
        if min_diff > 0:
            time_diffs = time_diffs / min_diff
    
    trajectories = []
    
    for sim_idx in range(n_simulations):
        # Start with initial state
        current_state = start_state
        states = [current_state]
        time_indices = [0]
        
        # Get censoring time for this simulation if censoring is enabled
        censored = False
        censoring_idx = None
        if enable_censoring:
            censoring_time = censoring_times[sim_idx]
            
            # Convert censoring time to index if needed
            if has_time_mapper and use_original_time and np.isfinite(censoring_time):
                censoring_idx = model.time_mapper.get_closest_idx(censoring_time)
            else:
                censoring_idx = int(censoring_time) if np.isfinite(censoring_time) else None
        
        # Simulate until max_time, absorbing state, or censoring
        t = 0
        while t < max_time - 1 and model.state_transitions[current_state]:
            t += 1
            
            # Check for censoring
            if enable_censoring and censoring_idx is not None and t >= censoring_idx:
                censored = True
                break
            
            # Get transition probabilities
            probs = model.predict_proba(x, time_idx=t, from_state=current_state)
            probs = probs.squeeze().detach().cpu().numpy()
            
            # Sample next state
            next_states = model.state_transitions[current_state]
            
            if len(next_states) == 0:  # Check again for absorbing state
                break
                
            # Make sure probs is 1D array for np.random.choice
            if not isinstance(probs, np.ndarray) or probs.ndim == 0:
                probs = np.array([1.0])
            
            # Apply time-based adjustment to transition probabilities if needed
            if time_adjusted and time_diffs is not None and t < len(time_diffs):
                # For time-windows larger than the minimum, we use adjusted probabilities
                # If time_diff = 3.0 (e.g., quarterly vs monthly data), we reduce the probability
                # of staying in the current state and increase transition probabilities
                
                # Extract time difference (window size) for current time point
                time_diff = time_diffs[t]
                
                # Adjust probabilities for time window size
                if time_diff > 1.0 and len(next_states) > 1:
                    try:
                        # Try using matrix exponential method for more accurate time adjustment
                        # This requires scipy, but will fall back to element-wise methods if unavailable
                        from scipy import linalg
                        
                        # Construct full transition matrix for all states in the model
                        # This is more accurate than just using the visible next states
                        all_states = set([current_state] + next_states)
                        for s in model.state_transitions:
                            all_states.add(s)
                            all_states.update(model.state_transitions[s])
                        
                        all_states = sorted(list(all_states))
                        n_all_states = len(all_states)
                        state_to_idx = {s: i for i, s in enumerate(all_states)}
                        
                        # Create transition matrix P (initialize as identity matrix for all states)
                        P = np.eye(n_all_states)
                        
                        # Fill in transitions from current state based on calculated probs
                        curr_idx = state_to_idx[current_state]
                        for i, next_state in enumerate(next_states):
                            next_idx = state_to_idx[next_state]
                            P[curr_idx, next_idx] = probs[i]
                        
                        # Convert P to rate matrix Q using matrix logarithm
                        try:
                            Q = linalg.logm(P)
                            
                            # Scale Q by time difference
                            Q_scaled = Q * time_diff
                            
                            # Convert back to probability matrix using matrix exponential
                            P_adjusted = linalg.expm(Q_scaled)
                            
                            # Extract adjusted probabilities for next states
                            adjusted_probs = np.array([
                                P_adjusted[curr_idx, state_to_idx[ns]] for ns in next_states
                            ])
                            
                            # Ensure valid probability vector
                            adjusted_probs = np.clip(adjusted_probs, 0, 1)
                            if adjusted_probs.sum() > 0:
                                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                            
                            probs = adjusted_probs
                        except (np.linalg.LinAlgError, ValueError, OverflowError) as e:
                            # Fall back to element-wise method on matrix method failure
                            # This happens most commonly with ill-conditioned matrices
                            raise RuntimeError("Matrix method failed, falling back to element-wise") from e
                    
                    except (ImportError, RuntimeError):
                        # Fall back to element-wise methods when scipy is not available
                        # or when the matrix method fails (numerical issues, etc.)
                        
                        # Check if we have a self-transition
                        self_transition_idx = None
                        for i, next_state in enumerate(next_states):
                            if next_state == current_state:
                                self_transition_idx = i
                                break
                        
                        # Use different adjustment methods based on transition structure
                        if self_transition_idx is not None:
                            # Extract the probability of staying in current state
                            p_stay = probs[self_transition_idx]
                            
                            # Method for models with self-transitions
                            if p_stay < 1.0:  # Only adjust if there's a non-zero probability of leaving
                                # Convert self-transition to exit rate
                                exit_rate = -np.log(p_stay)
                                
                                # Scale rate by time difference
                                scaled_exit_rate = exit_rate * time_diff
                                
                                # New probability of staying
                                new_p_stay = np.exp(-scaled_exit_rate)
                                
                                # Total probability of leaving current state
                                total_leaving_prob = 1.0 - new_p_stay
                                
                                # Create new probability vector
                                adjusted_probs = np.zeros_like(probs)
                                adjusted_probs[self_transition_idx] = new_p_stay
                                
                                # Distribute remaining probability mass for non-self transitions
                                # proportionally to their original values
                                remaining_probs = np.array([
                                    probs[i] for i in range(len(probs)) if i != self_transition_idx
                                ])
                                
                                if sum(remaining_probs) > 0:  # Avoid division by zero
                                    remaining_ratios = remaining_probs / sum(remaining_probs)
                                    idx = 0
                                    for i in range(len(probs)):
                                        if i != self_transition_idx:
                                            adjusted_probs[i] = total_leaving_prob * remaining_ratios[idx]
                                            idx += 1
                                
                                probs = adjusted_probs
                        else:
                            # Method for models without self-transitions
                            # Convert to rates (per unit time)
                            rates = -np.log(1.0 - np.array(probs))
                            
                            # Scale rates by time difference
                            scaled_rates = rates * time_diff
                            
                            # Convert back to probabilities
                            adjusted_probs = 1.0 - np.exp(-scaled_rates)
                            
                            # Ensure probabilities sum to 1
                            if adjusted_probs.sum() > 0:
                                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                            
                            probs = adjusted_probs
                    
                    # The old complex method is commented out below:
                    """
                    # First, identify the self-transition probability (if any)
                    self_transition_idx = None
                    for i, next_state in enumerate(next_states):
                        if next_state == current_state:
                            self_transition_idx = i
                            break
                    
                    # Apply different time-adjustment methods based on the structure
                    if self_transition_idx is not None:
                        # Method 1: When there are self-transitions, we use the continuous-time
                        # Markov process conversion to adjust probabilities
                        
                        # Extract the probability of staying in current state
                        p_stay = probs[self_transition_idx]
                        
                        # Calculate the effective exit rate from this state
                        if p_stay < 1.0:
                            # Convert to rate using lambda = -ln(p_stay)
                            exit_rate = -np.log(p_stay)
                            
                            # Scale rate by time difference
                            scaled_exit_rate = exit_rate * time_diff
                            
                            # Calculate new staying probability
                            new_p_stay = np.exp(-scaled_exit_rate)
                            
                            # Calculate scaling factor for other transitions
                            scaling_factor = (1.0 - new_p_stay) / (1.0 - p_stay)
                            
                            # Create adjusted probabilities
                            adjusted_probs = probs.copy()
                            adjusted_probs[self_transition_idx] = new_p_stay
                            
                            # Scale other transition probabilities
                            for i in range(len(probs)):
                                if i != self_transition_idx:
                                    adjusted_probs[i] = probs[i] * scaling_factor
                            
                            # Use adjusted probabilities
                            probs = adjusted_probs
                    else:
                        # Method 2: For transitions without self-transitions, we use a different approach
                        # Convert to rates using -log(1-p) formula
                        rates = -np.log(1.0 - probs)
                        
                        # Scale rates by time difference
                        scaled_rates = rates * time_diff
                        
                        # Convert back to probabilities
                        adjusted_probs = 1.0 - np.exp(-scaled_rates)
                        
                        # Renormalize to ensure sum = 1
                        if adjusted_probs.sum() > 0:
                            adjusted_probs = adjusted_probs / adjusted_probs.sum()
                        
                        probs = adjusted_probs
                    """
                
            # Choose next state based on probabilities
            next_state_idx = np.random.choice(len(next_states), p=probs)
            current_state = next_states[next_state_idx]
            
            states.append(current_state)
            time_indices.append(t)
            
        # Create trajectory dataframe with time indices
        trajectory_data = {
            'time_idx': time_indices,
            'state': states,
            'simulation': sim_idx,
        }
        
        # Add censoring information if enabled
        if enable_censoring:
            trajectory_data['censored'] = censored
        
        trajectory_df = pd.DataFrame(trajectory_data)
        
        # Add original time values if available
        if use_original_time and has_time_mapper:
            # For each time index, convert to original time
            # Make sure we handle indices beyond our mapping safely
            safe_indices = np.array([
                min(idx, model.time_mapper.n_time_points - 1) for idx in time_indices
            ])
            
            # Convert to original time values using the model's time_mapper
            original_times = model.time_mapper.to_time(safe_indices)
            trajectory_df['time'] = original_times
        else:
            # If no time mapper, use indices as time
            trajectory_df['time'] = trajectory_df['time_idx']
        
        trajectories.append(trajectory_df)
    
    return trajectories


def simulate_cohort_trajectories(
    model: BaseMultiStateNN,
    cohort_features: torch.Tensor,
    start_state: int,
    max_time: int,
    n_simulations_per_patient: int = 10,
    censoring_rate: float = 0.0,
    time_adjusted: bool = False,
    seed: Optional[int] = None,
    use_original_time: bool = True,
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
    censoring_rate : float, optional
        Target censoring rate. Set to 0 to disable censoring.
    time_adjusted : bool, optional
        Whether to adjust transition probabilities based on time window sizes.
    seed : Optional[int], optional
        Random seed for reproducibility
    use_original_time : bool, optional
        Whether to use original time values in the output
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing all simulated trajectories with columns:
        - patient_id: Identifier for the patient
        - simulation: Simulation run number
        - time_idx: Time index (internal representation)
        - time: Time value (original scale, if available)
        - state: Patient state at that time
        - censored: Whether the trajectory was censored (only if censoring_rate > 0)
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
        patient_trajectories = simulate_patient_trajectory(
            model=model,
            x=patient_x,
            start_state=start_state,
            max_time=max_time,
            n_simulations=n_simulations_per_patient,
            censoring_times=censoring_times,
            censoring_rate=censoring_rate,
            time_adjusted=time_adjusted,
            seed=patient_seed,
            use_original_time=use_original_time
        )
        
        # Add patient ID to each trajectory
        for traj in patient_trajectories:
            traj['patient_id'] = patient_idx
        
        all_trajectories.extend(patient_trajectories)
    
    # Combine all trajectories into a single DataFrame
    return pd.concat(all_trajectories, ignore_index=True)


# Alias old function names for backward compatibility if needed
# These may be removed in future versions
simulate_patient_trajectory_with_censoring = simulate_patient_trajectory
simulate_cohort_trajectories_with_censoring = simulate_cohort_trajectories
simulate_patient_trajectory_time_adjusted = lambda *args, **kwargs: simulate_patient_trajectory(*args, time_adjusted=True, **kwargs)
simulate_cohort_trajectories_time_adjusted = lambda *args, **kwargs: simulate_cohort_trajectories(*args, time_adjusted=True, **kwargs)