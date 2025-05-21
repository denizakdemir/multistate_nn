"""Utility functions for working with multistate_nn models.

This module provides standardized functions for common operations
used with multistate models, including:
- Data preparation
- Patient profile creation
- Visualization wrappers
- Analysis helpers
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from typing import Dict, List, Tuple, Optional, Union, Any, cast
import networkx as nx
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Import multistate_nn utilities
from multistate_nn.utils.visualization import (
    plot_transition_heatmap,
    plot_transition_graph,
    plot_intensity_matrix,
    plot_cif,
    compare_cifs
)
from multistate_nn.utils.analysis import calculate_cif
from multistate_nn.utils.continuous_simulation import (
    simulate_continuous_patient_trajectory,
    simulate_continuous_cohort_trajectories
)


def setup_state_names_and_colors(num_states: int, 
                                state_names: Optional[List[str]] = None, 
                                colors: Optional[List[str]] = None,
                                cmap_name: str = 'viridis') -> Tuple[Dict[int, str], Dict[int, str]]:
    """Create standardized state names and colors.
    
    Parameters
    ----------
    num_states : int
        Number of states in the model
    state_names : Optional[List[str]], optional
        Custom state names, defaults to "State 0", "State 1", etc.
    colors : Optional[List[str]], optional
        Custom state colors, defaults to colors from the specified colormap
    cmap_name : str, optional
        Name of colormap to use if colors not provided
        
    Returns
    -------
    Tuple[Dict[int, str], Dict[int, str]]
        Dictionaries mapping state indices to names and colors
    """
    # Create state names if not provided
    if state_names is None:
        state_names = [f"State {i}" for i in range(num_states)]
    
    if colors is None:
        # Generate colors from a colormap
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i / max(1, num_states - 1)) for i in range(num_states)]
    
    # Create dictionaries mapping state indices to names and colors
    state_name_dict = {i: state_names[i] for i in range(min(num_states, len(state_names)))}
    state_color_dict = {i: colors[i] for i in range(min(num_states, len(colors)))}
    
    return state_name_dict, state_color_dict


def create_patient_profile(covariates: Union[Dict[str, float], Dict[str, List[float]]], 
                          feature_order: Optional[List[str]] = None,
                          normalize_values: Optional[Dict[str, Tuple[float, float]]] = None,
                          as_tensor: bool = True) -> Union[torch.Tensor, np.ndarray]:
    """Create a patient profile tensor with specified covariate values.
    
    Parameters
    ----------
    covariates : Union[Dict[str, float], Dict[str, List[float]]]
        Dictionary mapping covariate names to values. If values are lists, multiple profiles are created.
    feature_order : Optional[List[str]]
        Order of features in the resulting tensor. If None, alphabetical order is used.
    normalize_values : Optional[Dict[str, Tuple[float, float]]]
        Dictionary mapping feature names to (mean, std) tuples for normalization.
    as_tensor : bool, optional
        Whether to return as a PyTorch tensor, by default True
        
    Returns
    -------
    Union[torch.Tensor, np.ndarray]
        Patient profile tensor ready for model input
    """
    # Check if values are scalar or lists
    is_list_values = any(isinstance(v, list) for v in covariates.values())
    
    if is_list_values:
        # Get list size from first list value
        list_size = len(next(v for v in covariates.values() if isinstance(v, list)))
        
        # Make sure all lists have the same length
        for k, v in covariates.items():
            if isinstance(v, list) and len(v) != list_size:
                raise ValueError(f"All lists must have the same length. {k} has length {len(v)} but expected {list_size}")
        
        # Convert scalar values to lists
        covariates_lists = {}
        for k, v in covariates.items():
            covariates_lists[k] = v if isinstance(v, list) else [v] * list_size
            
        # Create ordered feature list
        if feature_order is None:
            feature_order = sorted(covariates_lists.keys())
            
        # Create profiles
        profiles = []
        for i in range(list_size):
            profile = [covariates_lists[feat][i] for feat in feature_order]
            profiles.append(profile)
            
        # Convert to tensor
        profiles_array = np.array(profiles, dtype=np.float32)
        
        # Normalize if needed
        if normalize_values:
            for i, feat in enumerate(feature_order):
                if feat in normalize_values:
                    mean, std = normalize_values[feat]
                    profiles_array[:, i] = (profiles_array[:, i] - mean) / std
                    
        return torch.tensor(profiles_array, dtype=torch.float32) if as_tensor else profiles_array
    
    else:
        # Create ordered feature list
        if feature_order is None:
            feature_order = sorted(covariates.keys())
            
        # Create profile
        # Cast the result to List[float] to satisfy mypy
        profile = cast(List[float], [covariates[feat] for feat in feature_order])
        
        # Convert to tensor
        profile_array = np.array(profile, dtype=np.float32)
        
        # Normalize if needed
        if normalize_values:
            for i, feat in enumerate(feature_order):
                if feat in normalize_values:
                    mean, std = normalize_values[feat]
                    profile_array[i] = (profile_array[i] - mean) / std
        
        if as_tensor:
            return torch.tensor(profile_array, dtype=torch.float32).unsqueeze(0)
        else:
            return profile_array.reshape(1, -1)


def create_fixed_profile(value: float, feature_order: List[str], fixed_covariates: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """Create a patient profile tensor with all features set to a fixed value.
    
    Parameters
    ----------
    value : float
        Value to set for all features
    feature_order : List[str]
        Order of features in the resulting tensor
    fixed_covariates : Optional[Dict[str, float]]
        Dictionary of features to override with specific values
        
    Returns
    -------
    torch.Tensor
        Patient profile tensor with specified values
    """
    profile = torch.ones(1, len(feature_order)) * value
    
    if fixed_covariates:
        for feat, val in fixed_covariates.items():
            if feat in feature_order:
                idx = feature_order.index(feat)
                profile[0, idx] = val
                
    return profile


def create_covariate_profiles(**covariates: List[Any]) -> Dict[str, Any]:
    """Create a grid of patient profiles for different combinations of covariates.
    
    Parameters
    ----------
    **covariates : dict
        Dictionary of covariate names and their possible values.
        Example: age_std=[-1.5, 1.5], treatment=[0, 1]
        
    Returns
    -------
    dict
        Dictionary containing the profiles tensor and labels
    """
    # Extract covariate names and values
    covariate_names = list(covariates.keys())
    covariate_values = list(covariates.values())
    
    # Generate all combinations
    combinations = list(itertools.product(*covariate_values))
    
    # Create profiles tensor
    profiles = torch.tensor(combinations, dtype=torch.float32)
    
    # Create descriptive labels
    labels = []
    for combo in combinations:
        label_parts = []
        for i, name in enumerate(covariate_names):
            value = combo[i]
            if name == 'age_std':
                if value < -1.0:
                    label_parts.append("Young")
                elif value > 1.0:
                    label_parts.append("Elderly")
                else:
                    label_parts.append("Middle-aged")
            elif name == 'treatment':
                if value == 0:
                    label_parts.append("Untreated")
                else:
                    label_parts.append("Treated")
            elif name == 'biomarker':
                if value < 0:
                    label_parts.append("Low Biomarker")
                else:
                    label_parts.append("High Biomarker")
            else:
                label_parts.append(f"{name}={value}")
        
        labels.append(", ".join(label_parts))
    
    return {
        'profiles': profiles,
        'labels': labels,
        'covariate_names': covariate_names
    }


def analyze_covariate_distribution(df: pd.DataFrame, 
                                 covariate: str, 
                                 by_state: Optional[str] = None,
                                 state_names: Optional[Dict[int, str]] = None,
                                 figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    """Analyze the distribution of a covariate, optionally grouped by state.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing patient data
    covariate : str
        Name of the covariate to analyze
    by_state : Optional[str]
        Name of the state column for grouping. If None, no grouping is performed.
    state_names : Optional[Dict[int, str]]
        Dictionary mapping state indices to names
    figsize : Tuple[int, int]
        Figure size for the plots
        
    Returns
    -------
    Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]
        Figure and axes with the distribution plots
    """
    if covariate not in df.columns:
        raise ValueError(f"Covariate {covariate} not found in DataFrame")
    
    if by_state and by_state not in df.columns:
        raise ValueError(f"State column {by_state} not found in DataFrame")
    
    if by_state:
        # Distribution by state
        unique_states = sorted(df[by_state].unique())
        n_states = len(unique_states)
        
        if state_names is None:
            state_names = {i: f"State {i}" for i in unique_states}
            
        fig, axes = plt.subplots(1, n_states, figsize=figsize)
        
        for i, state in enumerate(unique_states):
            state_data = df[df[by_state] == state][covariate]
            ax = axes[i] if n_states > 1 else axes
            
            # Histogram and kernel density
            sns.histplot(state_data, kde=True, ax=ax)
            
            # Add statistics
            mean = state_data.mean()
            median = state_data.median()
            std = state_data.std()
            
            ax.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
            ax.axvline(median, color='g', linestyle='-.', label=f'Median: {median:.2f}')
            
            ax.set_title(f"{covariate} in {state_names[state]}")
            ax.set_xlabel(covariate)
            ax.set_ylabel("Density")
            ax.legend()
            
        plt.tight_layout()
        return fig, axes
        
    else:
        # Overall distribution
        fig, ax = plt.subplots(figsize=figsize)
        
        # Histogram and kernel density
        sns.histplot(df[covariate], kde=True, ax=ax)
        
        # Add statistics
        mean = df[covariate].mean()
        median = df[covariate].median()
        std = df[covariate].std()
        
        ax.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
        ax.axvline(median, color='g', linestyle='-.', label=f'Median: {median:.2f}')
        
        ax.set_title(f"Distribution of {covariate}")
        ax.set_xlabel(covariate)
        ax.set_ylabel("Density")
        ax.legend()
        
        return fig, ax


def plot_transition_curves(model: Any, 
                         profile: torch.Tensor,
                         from_state: int,
                         max_time: float = 5.0,
                         num_points: int = 100,
                         state_names: Optional[Dict[int, str]] = None,
                         figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, Axes]:
    """Plot transition probability curves over time from a specific state.
    
    Parameters
    ----------
    model : Any
        Trained MultiStateNN model
    profile : torch.Tensor
        Patient profile tensor (shape: [1, input_dim])
    from_state : int
        Starting state for transitions
    max_time : float
        Maximum time to plot
    num_points : int
        Number of time points to evaluate
    state_names : Optional[Dict[int, str]]
        Dictionary mapping state indices to names
    figsize : Tuple[int, int]
        Figure size for the plot
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes with the transition probability curves
    """
    # Get default state names if not provided
    if state_names is None:
        state_names = {i: f"State {i}" for i in range(model.num_states)}
        
    # Get possible next states
    next_states = model.state_transitions[from_state]
    
    # Create time grid
    time_grid = np.linspace(0, max_time, num_points)
    
    # Calculate probabilities at each time point
    probs = []
    for t in time_grid:
        p = model.predict_proba(profile, time_start=0.0, time_end=t, from_state=from_state).detach().numpy()[0]
        probs.append(p)
    
    # Convert to array for easier indexing
    probs_array = np.array(probs)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, state in enumerate(next_states):
        ax.plot(time_grid, probs_array[:, i], label=f'To {state_names[state]}', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Transition Probability')
    ax.set_title(f'Transition Probabilities from {state_names[from_state]} Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def visualize_state_distribution_over_time(trajectories: List[pd.DataFrame],
                                        max_time: float = 5.0,
                                        num_points: int = 11,  # Default to 11 points (0, 0.5, 1, ..., 5)
                                        state_names: Optional[Dict[int, str]] = None,
                                        state_colors: Optional[Dict[int, str]] = None,
                                        figsize: Tuple[int, int] = (12, 6)) -> Tuple[Figure, Axes]:
    """Visualize state distribution over time from simulated trajectories.
    
    Parameters
    ----------
    trajectories : List[pd.DataFrame]
        List of DataFrames containing simulated trajectories
    max_time : float
        Maximum time to show
    num_points : int
        Number of time points to show distribution
    state_names : Optional[Dict[int, str]]
        Dictionary mapping state indices to names
    state_colors : Optional[Dict[int, str]]
        Dictionary mapping state indices to colors
    figsize : Tuple[int, int]
        Figure size for the plot
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes with the state distribution plot
    """
    # Combine all trajectories
    combined_df = pd.concat(trajectories, ignore_index=True)
    
    # Create time points
    time_points = np.linspace(0, max_time, num_points)
    
    # Get all possible states
    all_states = sorted(combined_df['state'].unique())
    
    # Create state names and colors if not provided
    if state_names is None:
        state_names = {i: f"State {i}" for i in all_states}
    
    if state_colors is None:
        cmap = plt.cm.get_cmap('viridis', len(all_states))
        state_colors = {i: plt.rgb2hex(cmap(i/len(all_states))) for i in all_states}
    
    # Calculate distribution at each time point
    distributions = []
    
    for t in time_points:
        # Find states at time t for each simulation
        dist = {'time': t}
        
        for sim_id in combined_df['simulation'].unique():
            sim_data = combined_df[combined_df['simulation'] == sim_id]
            
            # Find state at time t (or closest previous time)
            mask = sim_data['time'] <= t
            if mask.any():
                latest_idx = sim_data[mask]['time'].idxmax()
                state = sim_data.loc[latest_idx, 'state']
                
                # Count this state
                state_key = int(state)
                # Use string keys instead of int keys to make mypy happy
                state_key_str = str(state_key)
                if state_key_str in dist:
                    dist[state_key_str] = dist[state_key_str] + 1
                else:
                    dist[state_key_str] = 1
        
        # Convert counts to proportions
        # Use string keys for the dictionary
        total = sum(dist.get(str(state), 0) for state in all_states)
        for state in all_states:
            state_str = str(state)
            if state_str in dist:
                dist[state_str] = dist[state_str] / total
            else:
                dist[state_str] = 0
                
        # Add numeric state keys back for the result
        for state in all_states:
            state_str = str(state)
            if state_str in dist and state_str != 'time':
                dist[state] = dist[state_str]
                
        distributions.append(dist)
    
    # Convert to DataFrame
    dist_df = pd.DataFrame(distributions)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot state distribution as stacked bars
    bottom = np.zeros(len(dist_df))
    
    for state in all_states:
        if state in dist_df.columns:
            ax.bar(
                dist_df['time'], dist_df[state], bottom=bottom,
                label=state_names.get(state, f"State {state}"),
                color=state_colors.get(state)
            )
            bottom += dist_df[state].values
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Proportion')
    ax.set_title('State Distribution Over Time')
    ax.legend(title='State')
    
    return fig, ax


def visualize_state_distribution(model: Any, 
                                patient_profile: torch.Tensor, 
                                from_state: int = 0, 
                                max_time: float = 5.0, 
                                num_points: int = 100, 
                                state_names: Optional[List[str]] = None, 
                                state_colors: Optional[List[str]] = None,
                                ax: Optional[plt.Axes] = None,
                                title: Optional[str] = None) -> plt.Axes:
    """Visualize how state probabilities change over time.
    
    Parameters
    ----------
    model : Any
        Trained multistate model
    patient_profile : torch.Tensor
        Patient profile tensor of shape (1, input_dim)
    from_state : int, optional
        Starting state, by default 0
    max_time : float, optional
        Maximum time to visualize, by default 5.0
    num_points : int, optional
        Number of time points to calculate, by default 100
    state_names : Optional[List[str]], optional
        List of state names, by default None
    state_colors : Optional[List[str]], optional
        List of state colors, by default None
    ax : Optional[plt.Axes], optional
        Matplotlib axes to plot on, by default None
    title : Optional[str], optional
        Plot title, by default None
        
    Returns
    -------
    plt.Axes
        The matplotlib axes with the visualization
    """
    if state_names is None:
        state_names = [f"State {i}" for i in range(model.num_states)]
    if state_colors is None:
        cmap = plt.get_cmap('viridis')
        state_colors = [cmap(i / max(1, model.num_states - 1)) for i in range(model.num_states)]
        
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
        
    # Create time grid
    times = np.linspace(0, max_time, num_points)
    
    # Initialize probability matrix
    probs = np.zeros((num_points, len(state_names)))
    time_start = torch.tensor([0.0], dtype=torch.float32)
    
    # Calculate probabilities at each time point
    for i, t in enumerate(times):
        if i == 0:
            # At t=0, probability of being in the initial state is 1
            probs[i, from_state] = 1.0
        else:
            time_end = torch.tensor([float(t)], dtype=torch.float32)
            p = model.predict_proba(patient_profile, time_start=time_start, time_end=time_end, from_state=from_state).detach().numpy()
            probs[i] = p.squeeze()
            
    # Plot the state distribution over time
    ax.stackplot(times, probs.T, labels=state_names, colors=state_colors, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    
    if title:
        ax.set_title(title)
        
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    return ax


def analyze_covariate_effect(model: Any,
                            base_profile: Optional[torch.Tensor] = None,
                            covariate_idx: Optional[int] = None,
                            covariate_values: Optional[List[float]] = None,
                            covariate_name: Optional[str] = None,
                            time_end: float = 1.0,
                            from_state: int = 0,
                            state_names: Optional[Dict[int, str]] = None,
                            figsize: Tuple[int, int] = (15, 10)) -> Tuple[Figure, Axes]:
    """Analyze and visualize the effect of varying a covariate on transition probabilities.
    
    This function supports two calling conventions:
    1. Legacy keyword-based API (base_profile, covariate_idx, covariate_values required)
    2. New direct-value API where model expects individual covariate names/values
    
    Parameters
    ----------
    model : Any
        Trained multistate model
    base_profile : torch.Tensor, optional
        Base patient profile tensor of shape (1, input_dim)
    covariate_idx : int, optional 
        Index of the covariate to vary
    covariate_values : List[float], optional
        List of values to assign to the covariate
    covariate_name : str, optional
        Name of the covariate for plotting
    time_end : float, optional
        Time horizon for transition probabilities, by default 1.0
    from_state : int, optional
        Starting state, by default 0
    state_names : Optional[Dict[int, str]], optional
        Dictionary mapping state indices to names, by default None
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 10)
        
    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes with the plots
    """
    # Check if required parameters are provided for legacy API
    if base_profile is not None and covariate_idx is not None and covariate_values is not None:
        # Legacy API using specified profile and index
        profiles = []
        for value in covariate_values:
            profile_copy = base_profile.clone()
            profile_copy[0, covariate_idx] = value
            profiles.append(profile_copy)
            
        # Stack the profiles
        all_profiles = torch.cat(profiles, dim=0)
        
        # Get transition probabilities
        time_start = torch.tensor([0.0], dtype=torch.float32)
        time_end_tensor = torch.tensor([float(time_end)], dtype=torch.float32)
        probs = model.predict_proba(all_profiles, time_start=time_start, time_end=time_end_tensor, from_state=from_state).detach().numpy()
        
        # Create figure for visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        # Convert to ndarray to handle indexing
        axes = cast(np.ndarray, axes)
        
        # If no state names provided, create generic ones
        if state_names is None:
            state_names = {i: f"State {i}" for i in range(model.num_states)}
            
        # Get possible next states
        next_states = model.state_transitions[from_state] if from_state in model.state_transitions else []
        if not next_states:  # If this is an absorbing state or invalid state
            next_states = list(range(model.num_states))
            next_states.remove(from_state)
        
        # Plot transition probabilities
        for i, state in enumerate(next_states):
            if i < len(axes):  # In case we have more states than axes
                ax = axes[i]
                ax.plot(covariate_values, probs[:, state], 'o-', linewidth=2)
                ax.set_xlabel(covariate_name)
                ax.set_ylabel(f'Probability of transition to {state_names[state]}')
                ax.set_title(f'Effect of {covariate_name} on transition to {state_names[state]}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Return the results for further analysis if needed
        results = {
            'covariate_values': covariate_values,
            'probs': probs,
            'covariate_name': covariate_name,
            'from_state': from_state
        }
        
        return fig, axes
    else:
        # Error if required parameters are missing
        raise ValueError("analyze_covariate_effect requires base_profile, covariate_idx, and covariate_values")


def compare_treatment_effects(model: Any,
                             base_profile: torch.Tensor,
                             treatment_idx: int,
                             treatment_values: List[float],
                             treatment_labels: Optional[List[str]] = None,
                             target_states: Optional[List[int]] = None,
                             time_end: float = 5.0,
                             from_state: int = 0,
                             state_names: Optional[Dict[int, str]] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> Tuple[Figure, Axes]:
    """Compare and visualize treatment effects on transition probabilities.
    
    Parameters
    ----------
    model : MultiStateNN or BayesianMultiStateNN
        Trained multistate model
    base_profile : torch.Tensor
        Base patient profile tensor (shape: [1, input_dim])
    treatment_idx : int
        Index of the treatment variable
    treatment_values : List[float]
        Values for the treatment variable (typically 0/1 for untreated/treated)
    treatment_labels : Optional[List[str]], optional
        Labels for treatment values (e.g., ["Untreated", "Treated"])
    target_states : List[int], optional
        States to include in comparison. If None, uses all next states from from_state.
    time_end : float, optional
        Time horizon for transition probabilities
    from_state : int, optional
        Starting state for transitions
    state_names : Optional[Dict[int, str]], optional
        Dictionary mapping state indices to names
    figsize : Tuple[int, int], optional
        Figure size for the plot
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes with the comparison plot
    """
    if treatment_labels is None:
        treatment_labels = [f"Treatment={v}" for v in treatment_values]
    
    if len(treatment_values) != len(treatment_labels):
        raise ValueError("Length of treatment_values and treatment_labels must match")
    
    # Create profiles for each treatment value
    profiles = []
    for value in treatment_values:
        profile = base_profile.clone()
        profile[0, treatment_idx] = value
        profiles.append(profile)
    
    # Get default state names if not provided
    if state_names is None:
        state_names = {i: f"State {i}" for i in range(model.num_states)}
    
    # If target_states not specified, use all next states from from_state
    if target_states is None:
        target_states = model.state_transitions[from_state]
    
    # Calculate transition probabilities for each profile
    all_probs = []
    for profile in profiles:
        probs = model.predict_proba(
            profile, time_start=0.0, time_end=time_end, from_state=from_state
        ).detach().numpy()[0]
        all_probs.append(probs)
    
    # Extract probabilities for target states
    target_probs = []
    for probs in all_probs:
        target_probs.append([probs[i] for i in target_states])
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(target_states))
    width = 0.8 / len(profiles)
    
    for i, (probs, label) in enumerate(zip(target_probs, treatment_labels)):
        offset = (i - len(profiles)/2 + 0.5) * width
        ax.bar(x + offset, probs, width, label=label)
    
    # Add labels and title
    target_state_names = [state_names[i] for i in target_states]
    ax.set_xticks(x)
    ax.set_xticklabels(target_state_names)
    ax.set_ylabel("Transition Probability")
    ax.set_xlabel("Target State")
    ax.set_title(f"Effect of Treatment on Transition Probabilities\nFrom {state_names[from_state]} at time t={time_end}")
    ax.legend()
    
    return fig, ax


def compare_models_cif(models: Dict[str, Any],
                      profile: torch.Tensor,
                      from_state: int,
                      target_state: int,
                      max_time: float = 5.0,
                      n_simulations: int = 100,
                      time_step: float = 0.1,
                      state_names: Optional[Dict[int, str]] = None,
                      colors: Optional[Dict[str, str]] = None,
                      figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, Axes]:
    """Compare cumulative incidence functions (CIFs) across different models.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary mapping model names to trained models
    profile : torch.Tensor
        Patient profile tensor (shape: [1, input_dim])
    from_state : int
        Starting state for the simulation
    target_state : int
        Target state for CIF calculation
    max_time : float, optional
        Maximum time for simulation and CIF calculation
    n_simulations : int, optional
        Number of simulations per model
    time_step : float, optional
        Time step for simulation
    state_names : Optional[Dict[int, str]], optional
        Dictionary mapping state indices to names
    colors : Optional[Dict[str, str]], optional
        Dictionary mapping model names to colors
    figsize : Tuple[int, int], optional
        Figure size for the plot
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes with the CIF comparison plot
    """
    # Get default state names if not provided
    if state_names is None:
        max_states = max([model.num_states for model in models.values()])
        state_names = {i: f"State {i}" for i in range(max_states)}
    
    # Calculate CIFs for each model
    cifs = {}
    
    for model_name, model in models.items():
        # Simulate trajectories
        trajectories = simulate_continuous_patient_trajectory(
            model=model,
            x=profile,
            start_state=from_state,
            max_time=max_time,
            n_simulations=n_simulations,
            time_step=time_step
        )
        
        # Combine trajectories
        combined_trajectories = pd.concat(trajectories, ignore_index=True)
        
        # Calculate CIF
        cif = calculate_cif(combined_trajectories, target_state=target_state, max_time=max_time)
        
        # Add to dictionary
        cifs[model_name] = cif
    
    # Compare CIFs
    title = f"Comparison of Cumulative Incidence to {state_names[target_state]}"
    fig, ax = compare_cifs(cifs, colors=colors, title=title, figsize=figsize)
    
    return fig, ax


def visualize_model_comparison(models: Dict[str, Any],
                              profile: torch.Tensor,
                              time_end: float = 1.0,
                              from_state: int = 0,
                              state_names: Optional[Dict[int, str]] = None,
                              figsize: Tuple[int, int] = (15, 8)) -> Tuple[Figure, np.ndarray]:
    """Compare intensity matrices and transition probabilities across different models.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary mapping model names to trained models
    profile : torch.Tensor
        Patient profile tensor (shape: [1, input_dim])
    time_end : float, optional
        Time horizon for transition probabilities
    from_state : int, optional
        Starting state for transitions
    state_names : Optional[Dict[int, str]], optional
        Dictionary mapping state indices to names
    figsize : Tuple[int, int], optional
        Figure size for the plot
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and array of axes with the comparison plots
    """
    num_models = len(models)
    
    # Create figure with 2 rows (top: intensity matrices, bottom: transition probabilities)
    fig, axes = plt.subplots(2, num_models, figsize=figsize)
    
    # Get default state names if not provided
    if state_names is None:
        max_states = max([model.num_states for model in models.values()])
        state_names = {i: f"State {i}" for i in range(max_states)}
    
    # For each model, plot intensity matrix and transition probabilities
    for i, (model_name, model) in enumerate(models.items()):
        # Plot intensity matrix
        plot_intensity_matrix(model, profile, ax=axes[0, i], annot=True)
        axes[0, i].set_title(f"Intensity Matrix\n{model_name}")
        
        # Plot transition probabilities
        plot_transition_heatmap(
            model, profile, time_start=0.0, time_end=time_end, 
            from_state=from_state, ax=axes[1, i], annot=True
        )
        axes[1, i].set_title(f"Transition Probabilities from {state_names[from_state]}\n{model_name}")
    
    plt.tight_layout()
    return fig, axes