"""Utility functions for MultiStateNN models.

This module is deprecated and will be removed in a future version.
All functions have been moved to specialized modules:
- Visualization: multistate_nn.utils.visualization
- Simulation: multistate_nn.utils.simulation
- Analysis: multistate_nn.utils.analysis
"""

import warnings
import functools
from typing import Optional, Dict, List, Tuple, Union, Sequence, Callable, Any
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from .models import MultiStateNN

def deprecated(func: Callable) -> Callable:
    """Decorator to mark functions as deprecated."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"Function {func.__name__} in multistate_nn.utils is deprecated and will be "
            f"removed in a future version. Use multistate_nn.utils.visualization, "
            f"multistate_nn.utils.simulation, or multistate_nn.utils.analysis instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper


@deprecated
def plot_transition_heatmap(
    model: MultiStateNN,
    x: torch.Tensor,
    time_idx: int,
    from_state: int,
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlOrRd",
) -> plt.Axes:
    """Plot transition probabilities as a heatmap.

    Parameters
    ----------
    model : MultiStateNN
        Trained model
    x : torch.Tensor
        Input features
    time_idx : int
        Time index
    from_state : int
        Source state
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    cmap : str, optional
        Colormap to use

    Returns
    -------
    ax : plt.Axes
        The axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    probs = model.predict_proba(x, time_idx, from_state).detach().cpu().numpy()
    next_states = model.state_transitions[from_state]

    sns.heatmap(
        probs,
        ax=ax,
        cmap=cmap,
        xticklabels=[f"State {s}" for s in next_states],
        yticklabels=[f"Sample {i}" for i in range(len(x))],
    )
    ax.set_title(f"Transition Probabilities from State {from_state}")

    return ax


@deprecated
def compute_transition_matrix(
    model: MultiStateNN,
    x: torch.Tensor,
    time_idx: int,
) -> np.ndarray:
    """Compute average transition probability matrix.

    Parameters
    ----------
    model : MultiStateNN
        Trained model
    x : torch.Tensor
        Input features
    time_idx : int
        Time index

    Returns
    -------
    P : np.ndarray
        Transition probability matrix of shape (num_states, num_states)
    """
    num_states = model.num_states
    P = np.zeros((num_states, num_states))

    for i in range(num_states):
        if not model.state_transitions[i]:  # Absorbing state
            P[i, i] = 1.0
            continue

        probs = model.predict_proba(x, time_idx, i).mean(dim=0).detach().cpu().numpy()
        for j, next_state in enumerate(model.state_transitions[i]):
            P[i, next_state] = probs[j]

    return P


@deprecated
def plot_transition_graph(
    model: MultiStateNN,
    x: torch.Tensor,
    time_idx: int,
    threshold: float = 0.1,
    figsize: Tuple[float, float] = (10, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot transition graph using networkx.

    Parameters
    ----------
    model : MultiStateNN
        Trained model
    x : torch.Tensor
        Input features
    time_idx : int
        Time index
    threshold : float, optional
        Minimum probability to show transition
    figsize : tuple[float, float], optional
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure
    ax : plt.Axes
        The axes with the plot
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for plotting transition graphs")

    P = compute_transition_matrix(model, x, time_idx)
    G = nx.DiGraph()

    # Add nodes
    for i in range(model.num_states):
        G.add_node(i)

    # Add edges with probability weights
    for i in range(model.num_states):
        for j in range(model.num_states):
            if P[i, j] > threshold:
                G.add_edge(i, j, weight=P[i, j])

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", ax=ax)
    nx.draw_networkx_labels(G, pos, {i: f"State {i}" for i in range(model.num_states)})

    # Draw edges with varying thickness based on probability
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[w * 3 for w in edge_weights], edge_color="gray", ax=ax)

    # Add probability labels on edges
    edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax.set_title(f"State Transition Graph (t={time_idx})")
    ax.axis("off")

    return fig, ax


@deprecated
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


@deprecated
def simulate_patient_trajectory(
    model: MultiStateNN,
    x: torch.Tensor,
    start_state: int,
    max_time: int,
    n_simulations: int = 100,
    seed: Optional[int] = None,
) -> List[pd.DataFrame]:
    """Simulate patient trajectories through the multistate model.
    
    Parameters
    ----------
    model : MultiStateNN
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


@deprecated
def simulate_cohort_trajectories(
    model: MultiStateNN,
    cohort_features: torch.Tensor,
    start_state: int,
    max_time: int,
    n_simulations_per_patient: int = 10,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate trajectories for a cohort of patients.
    
    Parameters
    ----------
    model : MultiStateNN
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


@deprecated
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


@deprecated
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


@deprecated
def plot_cif(
    cif_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    color: str = 'blue',
    label: Optional[str] = None,
    show_ci: bool = True,
    linestyle: str = '-',
    alpha: float = 0.2,
) -> plt.Axes:
    """Plot cumulative incidence function.
    
    Parameters
    ----------
    cif_df : pd.DataFrame
        DataFrame from calculate_cif
    ax : Optional[plt.Axes], optional
        Matplotlib axes to plot on
    color : str, optional
        Line color
    label : Optional[str], optional
        Line label for legend
    show_ci : bool, optional
        If True, show confidence intervals
    linestyle : str, optional
        Line style
    alpha : float, optional
        Transparency for confidence interval
        
    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the CIF
    ax.plot(cif_df['time'], cif_df['cif'], color=color, linestyle=linestyle, label=label)
    
    # Plot confidence intervals if requested
    if show_ci and 'lower_ci' in cif_df.columns and 'upper_ci' in cif_df.columns:
        ax.fill_between(
            cif_df['time'],
            cif_df['lower_ci'],
            cif_df['upper_ci'],
            color=color,
            alpha=alpha
        )
    
    # Add labels and grid
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Incidence')
    ax.grid(True, alpha=0.3)
    
    if label is not None:
        ax.legend()
    
    return ax


@deprecated
def compare_cifs(
    cif_list: List[pd.DataFrame],
    labels: List[str],
    colors: Optional[List[str]] = None,
    title: str = 'Comparison of Cumulative Incidence Functions',
    figsize: Tuple[float, float] = (12, 8),
    show_ci: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Compare multiple CIFs on the same plot.
    
    Parameters
    ----------
    cif_list : List[pd.DataFrame]
        List of DataFrames from calculate_cif
    labels : List[str]
        Labels for each CIF
    colors : Optional[List[str]], optional
        Colors for each CIF. If None, uses default color cycle.
    title : str, optional
        Plot title
    figsize : Tuple[float, float], optional
        Figure size
    show_ci : bool, optional
        If True, show confidence intervals
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes with the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = colors[:len(cif_list)]
    
    for i, (cif_df, label) in enumerate(zip(cif_list, labels)):
        color = colors[i % len(colors)]
        plot_cif(
            cif_df=cif_df,
            ax=ax,
            color=color,
            label=label,
            show_ci=show_ci,
            alpha=0.1,  # Lower alpha when comparing multiple CIFs
        )
    
    ax.set_title(title)
    ax.legend()
    
    return fig, ax
