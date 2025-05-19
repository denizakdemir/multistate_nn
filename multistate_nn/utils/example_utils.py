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
from typing import Dict, List, Tuple, Optional, Union, Any
import networkx as nx
from matplotlib import cm

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
    if state_names is None or len(state_names) != num_states:
        state_names = [f"State {i}" for i in range(num_states)]
    
    # Create state colors if not provided
    if colors is None or len(colors) != num_states:
        cmap = cm.get_cmap(cmap_name, num_states)
        colors = [plt.rgb2hex(cmap(i)) for i in range(num_states)]
    
    # Create dictionaries
    state_name_dict = {i: name for i, name in enumerate(state_names)}
    state_color_dict = {i: color for i, color in enumerate(colors)}
    
    return state_name_dict, state_color_dict


def create_patient_profile(feature_names: List[str], 
                           feature_values: List[Union[float, int]], 
                           normalize_values: Optional[Dict[str, Tuple[float, float]]] = None) -> torch.Tensor:
    """Create a patient profile tensor with specified feature values.
    
    Parameters
    ----------
    feature_names : List[str]
        Names of features in the model's input
    feature_values : List[Union[float, int]]
        Values for each feature
    normalize_values : Optional[Dict[str, Tuple[float, float]]], optional
        Dictionary mapping feature names to (mean, std) tuples for normalization.
        If provided, features will be normalized using these values.
        
    Returns
    -------
    torch.Tensor
        Patient profile tensor ready for model input
    """
    if len(feature_names) != len(feature_values):
        raise ValueError("Length of feature_names and feature_values must match")
    
    # Create dictionary of feature values
    profile_dict = {name: value for name, value in zip(feature_names, feature_values)}
    
    # Convert to tensor and normalize if needed
    profile = np.array(feature_values, dtype=np.float32)
    
    if normalize_values:
        for i, name in enumerate(feature_names):
            if name in normalize_values:
                mean, std = normalize_values[name]
                profile[i] = (profile[i] - mean) / std
    
    return torch.tensor(profile, dtype=torch.float32).unsqueeze(0)


def create_covariate_profiles(base_profile: torch.Tensor,
                             covariate_idx: int,
                             covariate_values: List[float],
                             covariate_name: str = "Covariate") -> Tuple[List[torch.Tensor], List[str]]:
    """Create a set of patient profiles by varying a single covariate.
    
    Parameters
    ----------
    base_profile : torch.Tensor
        Base patient profile tensor (shape: [1, input_dim])
    covariate_idx : int
        Index of the covariate to vary
    covariate_values : List[float]
        Values for the covariate
    covariate_name : str, optional
        Name of the covariate for labels
        
    Returns
    -------
    Tuple[List[torch.Tensor], List[str]]
        List of patient profiles and corresponding labels
    """
    profiles = []
    labels = []
    
    for value in covariate_values:
        # Create a copy of the base profile
        profile = base_profile.clone()
        
        # Update the covariate value
        profile[0, covariate_idx] = value
        
        # Add to list
        profiles.append(profile)
        labels.append(f"{covariate_name}={value:.1f}")
    
    return profiles, labels


def analyze_covariate_effect(model: Any,
                            base_profile: torch.Tensor,
                            covariate_idx: int,
                            covariate_values: List[float],
                            covariate_name: str,
                            time_end: float = 1.0,
                            from_state: int = 0,
                            state_names: Optional[Dict[int, str]] = None,
                            figsize: Tuple[int, int] = (15, 10)):
    """Analyze and visualize the effect of a covariate on transition intensities and probabilities.
    
    Parameters
    ----------
    model : MultiStateNN or BayesianMultiStateNN
        Trained multistate model
    base_profile : torch.Tensor
        Base patient profile tensor (shape: [1, input_dim])
    covariate_idx : int
        Index of the covariate to vary
    covariate_values : List[float]
        Values for the covariate
    covariate_name : str
        Name of the covariate
    time_end : float, optional
        Time horizon for transition probabilities
    from_state : int, optional
        Starting state for transitions
    state_names : Optional[Dict[int, str]], optional
        Dictionary mapping state indices to names
    figsize : Tuple[int, int], optional
        Figure size for the plots
        
    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and array of axes
    """
    profiles, labels = create_covariate_profiles(
        base_profile, covariate_idx, covariate_values, covariate_name
    )
    
    num_profiles = len(profiles)
    
    # Create figure with 2 rows (top: intensity matrices, bottom: transition probabilities)
    fig, axes = plt.subplots(2, num_profiles, figsize=figsize)
    
    # Get default state names if not provided
    if state_names is None:
        state_names = {i: f"State {i}" for i in range(model.num_states)}
        
    # For each profile, plot intensity matrix and transition probabilities
    for i, (profile, label) in enumerate(zip(profiles, labels)):
        # Plot intensity matrix
        plot_intensity_matrix(model, profile, ax=axes[0, i], annot=True)
        axes[0, i].set_title(f"Intensity Matrix\n{label}")
        
        # Plot transition probabilities
        plot_transition_heatmap(
            model, profile, time_start=0.0, time_end=time_end, 
            from_state=from_state, ax=axes[1, i], annot=True
        )
        axes[1, i].set_title(f"Transition Probabilities from {state_names[from_state]}\n{label}")
    
    plt.tight_layout()
    return fig, axes


def compare_treatment_effects(model: Any,
                             base_profile: torch.Tensor,
                             treatment_idx: int,
                             treatment_values: List[float],
                             treatment_labels: Optional[List[str]] = None,
                             target_states: List[int] = None,
                             time_end: float = 5.0,
                             from_state: int = 0,
                             state_names: Optional[Dict[int, str]] = None,
                             figsize: Tuple[int, int] = (12, 8)):
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


def visualize_state_distribution(trajectories: pd.DataFrame,
                                time_points: Optional[List[float]] = None,
                                state_names: Optional[Dict[int, str]] = None,
                                state_colors: Optional[Dict[int, str]] = None,
                                figsize: Tuple[int, int] = (12, 6)):
    """Visualize state distribution over time from simulated trajectories.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame of simulated trajectories
    time_points : Optional[List[float]], optional
        Time points at which to show distribution. If None, uses 5 evenly spaced points.
    state_names : Optional[Dict[int, str]], optional
        Dictionary mapping state indices to names
    state_colors : Optional[Dict[int, str]], optional
        Dictionary mapping state indices to colors
    figsize : Tuple[int, int], optional
        Figure size for the plot
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes with the state distribution plot
    """
    # Set up time points if not provided
    if time_points is None:
        max_time = trajectories['time'].max()
        time_points = np.linspace(0, max_time, 5)
    
    # Filter to trajectories with grid_point=True if that column exists
    if 'grid_point' in trajectories.columns:
        plot_data = trajectories[trajectories['grid_point'] == True].copy()
    else:
        plot_data = trajectories.copy()
    
    # Make sure all required time points are included
    all_states = plot_data['state'].unique()
    all_sims = plot_data['simulation'].unique()
    
    # Calculate state distribution at each time point
    distributions = []
    
    for time in time_points:
        # Get closest time point (rounded to 2 decimal places)
        rounded_time = np.round(time, 2)
        close_times = np.round(plot_data['time'], 2)
        mask = close_times == rounded_time
        
        if mask.sum() == 0:
            # No exact match, skip this time point
            continue
        
        # Get state distribution
        time_data = plot_data[mask]
        state_counts = time_data['state'].value_counts(normalize=True)
        
        # Create record for this time
        dist = {'time': time}
        for state in all_states:
            if state in state_counts:
                dist[int(state)] = state_counts[state]
            else:
                dist[int(state)] = 0.0
        distributions.append(dist)
    
    # Convert to DataFrame
    dist_df = pd.DataFrame(distributions)
    
    # Set up state names and colors if not provided
    if state_names is None:
        state_names = {i: f"State {i}" for i in range(len(all_states))}
    
    if state_colors is None:
        cmap = cm.get_cmap('viridis', len(all_states))
        state_colors = {i: plt.rgb2hex(cmap(i)) for i in range(len(all_states))}
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot state distribution as stacked bars
    state_cols = [col for col in dist_df.columns if isinstance(col, int)]
    bottom = np.zeros(len(dist_df))
    
    for state in state_cols:
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


def compare_models_cif(models: Dict[str, Any],
                      profile: torch.Tensor,
                      from_state: int,
                      target_state: int,
                      max_time: float = 5.0,
                      n_simulations: int = 100,
                      time_step: float = 0.1,
                      state_names: Optional[Dict[int, str]] = None,
                      colors: Optional[Dict[str, str]] = None,
                      figsize: Tuple[int, int] = (10, 6)):
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
                              figsize: Tuple[int, int] = (15, 8)):
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