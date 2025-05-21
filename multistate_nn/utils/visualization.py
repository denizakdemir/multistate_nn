"""Visualization utilities for continuous-time MultiStateNN models."""

from typing import Optional, Dict, List, Union, Tuple, Any, cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import torch

def plot_transition_heatmap(
    model: Any,
    x: torch.Tensor,
    time_start: Union[float, torch.Tensor] = 0.0,
    time_end: Union[float, torch.Tensor] = 1.0,
    from_state: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'YlGnBu',
    annot: bool = True,
) -> plt.Axes:
    """Plot a heatmap of transition probabilities for each input in x.
    
    Parameters
    ----------
    model : MultiStateNN or BayesianMultiStateNN
        The trained model
    x : torch.Tensor
        Input tensor with shape (batch_size, input_dim)
    time_start : Union[float, torch.Tensor], optional
        Start time for transition probability calculation
    time_end : Union[float, torch.Tensor], optional
        End time for transition probability calculation
    from_state : int, optional
        Starting state. If None, shows transitions from all states
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure
    figsize : tuple, optional
        Figure size if creating a new figure
    cmap : str, optional
        Colormap for heatmap
    annot : bool, optional
        Whether to annotate cells with values
        
    Returns
    -------
    plt.Axes
        The matplotlib axes with the heatmap
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    # Convert time values to tensors with float32 dtype if they're not already tensors
    device = x.device
    if isinstance(time_start, (int, float)):
        time_start = torch.tensor([time_start], dtype=torch.float32, device=device)
    elif isinstance(time_start, torch.Tensor) and time_start.dtype != torch.float32:
        time_start = time_start.to(dtype=torch.float32)
        
    if isinstance(time_end, (int, float)):
        time_end = torch.tensor([time_end], dtype=torch.float32, device=device)
    elif isinstance(time_end, torch.Tensor) and time_end.dtype != torch.float32:
        time_end = time_end.to(dtype=torch.float32)
    
    # Get probabilities for each input
    all_probs: List[np.ndarray] = []
    all_states: List[List[int]] = []
    
    if from_state is not None:
        # Single starting state
        probs = model.predict_proba(x, time_start=time_start, time_end=time_end, from_state=from_state)
        probs = probs.detach().numpy()
        
        # Get next states
        next_states = model.state_transitions[from_state]
        
        # Create column names
        col_names = [f"State {s}" for s in next_states]
        
        # Extract only the columns for valid transitions
        filtered_probs = probs[:, next_states]
        
        # Create heatmap data
        df = pd.DataFrame(filtered_probs, columns=col_names)
        
        # Plot
        sns.heatmap(df, ax=ax, cmap=cmap, annot=annot, fmt=".3f")
        ax.set_title(f"Transition Probabilities from State {from_state}")
        ax.set_xlabel("To State")
        ax.set_ylabel("Sample Index")
    else:
        # All starting states
        # This is not typically used with continuous-time models
        # but we'll implement it for compatibility
        all_rows = []
        all_cols = []
        all_values = []
        
        num_states = model.num_states
        
        for state in range(num_states):
            if not model.state_transitions[state]:
                continue
                
            probs = model.predict_proba(x[0:1], time_start=time_start, time_end=time_end, from_state=state)
            probs = probs.detach().numpy()[0]
            
            next_states = model.state_transitions[state]
            
            for i, next_state in enumerate(next_states):
                all_rows.append(state)
                all_cols.append(next_state)
                all_values.append(probs[i])
        
        # Reshape into a matrix
        matrix = np.zeros((num_states, num_states))
        for row, col, val in zip(all_rows, all_cols, all_values):
            matrix[row, col] = val
        
        # Plot
        sns.heatmap(matrix, ax=ax, cmap=cmap, annot=annot, fmt=".3f")
        ax.set_title("Transition Probabilities")
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        
    return ax

def plot_transition_graph(
    model: Any,
    x: torch.Tensor,
    time_start: Union[float, torch.Tensor] = 0.0,
    time_end: Union[float, torch.Tensor] = 1.0,
    threshold: float = 0.01,
    figsize: Tuple[int, int] = (12, 10),
    node_size: int = 2000,
    font_size: int = 12,
    cmap: str = 'YlOrRd',
) -> Tuple[Figure, Axes]:
    """Plot a directed graph of transition probabilities.
    
    Parameters
    ----------
    model : MultiStateNN or BayesianMultiStateNN
        The trained model
    x : torch.Tensor
        Input tensor with shape (1, input_dim)
    time_start : Union[float, torch.Tensor], optional
        Start time for transition probability calculation
    time_end : Union[float, torch.Tensor], optional
        End time for transition probability calculation
    threshold : float, optional
        Minimum transition probability to include in graph
    figsize : Tuple[int, int], optional
        Figure size
    node_size : int, optional
        Size of nodes in graph
    font_size : int, optional
        Font size for labels
    cmap : str, optional
        Colormap for edge colors
        
    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes with the graph plot
    """
    if x.shape[0] != 1:
        x = x[0:1]  # Use only the first sample
    
    # Convert time values to tensors with float32 dtype if they're not already tensors
    device = x.device
    if isinstance(time_start, (int, float)):
        time_start = torch.tensor([time_start], dtype=torch.float32, device=device)
    elif isinstance(time_start, torch.Tensor) and time_start.dtype != torch.float32:
        time_start = time_start.to(dtype=torch.float32)
        
    if isinstance(time_end, (int, float)):
        time_end = torch.tensor([time_end], dtype=torch.float32, device=device)
    elif isinstance(time_end, torch.Tensor) and time_end.dtype != torch.float32:
        time_end = time_end.to(dtype=torch.float32)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    num_states = model.num_states
    for state in range(num_states):
        G.add_node(state, label=f"State {state}")
    
    # Add edges based on transition probabilities
    max_prob = 0.0
    edges = []
    
    for from_state in range(num_states):
        if not model.state_transitions[from_state]:
            continue
            
        probs = model.predict_proba(x, time_start=time_start, time_end=time_end, from_state=from_state)
        probs = probs.detach().numpy()[0]
        
        next_states = model.state_transitions[from_state]
        
        # Extract only probabilities for valid next states
        filtered_probs = probs[next_states]
        
        for i, to_state in enumerate(next_states):
            prob = filtered_probs[i]
            if prob > threshold:
                G.add_edge(from_state, to_state, weight=prob, label=f"{prob:.3f}")
                edges.append((from_state, to_state, prob))
                max_prob = max(max_prob, prob)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Node positions - use spring layout or circular layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)
    
    # Draw edges with colors based on probability
    colormap = plt.get_cmap(cmap)
    
    for u, v, p in edges:
        # Normalize probability to 0-1 range for color mapping
        color = colormap(p / max_prob if max_prob > 0 else 0)
        # Width based on probability
        width = 1 + 5 * (p / max_prob if max_prob > 0 else 0)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, edge_color=[color], 
                               alpha=0.7, arrows=True, arrowsize=20, ax=ax)
    
    # Draw edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size-2, ax=ax)
    
    plt.title(f"Transition Probabilities Network (Threshold={threshold})")
    plt.axis('off')
    
    return fig, ax

def plot_intensity_matrix(
    model: Any,
    x: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'coolwarm',
    annot: bool = True,
) -> plt.Axes:
    """Plot a heatmap of the intensity matrix for a given input.
    
    Parameters
    ----------
    model : ContinuousMultiStateNN or BayesianContinuousMultiStateNN
        The trained continuous-time model
    x : torch.Tensor
        Input tensor with shape (1, input_dim)
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure
    figsize : tuple, optional
        Figure size if creating a new figure
    cmap : str, optional
        Colormap for heatmap
    annot : bool, optional
        Whether to annotate cells with values
        
    Returns
    -------
    plt.Axes
        The matplotlib axes with the heatmap
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    if x.shape[0] != 1:
        x = x[0:1]  # Use only the first sample
    
    # Get intensity matrix - using the correct method name 'intensity_matrix'
    intensity_matrix = model.intensity_matrix(x).detach().numpy()[0]
    
    # For better visualization, we replace diagonal elements with zero
    # (diagonal elements are typically negative and can dominate the color scale)
    for i in range(intensity_matrix.shape[0]):
        intensity_matrix[i, i] = 0
    
    # Create heatmap
    sns.heatmap(intensity_matrix, ax=ax, cmap=cmap, annot=annot, fmt=".3f")
    ax.set_title("Transition Intensity Matrix")
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    
    return ax

def plot_cif(
    cif_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = 'blue',
    alpha: float = 0.2,
    label: Optional[str] = None,
    show_ci: bool = True,
) -> plt.Axes:
    """Plot cumulative incidence function (CIF) with confidence intervals.
    
    Parameters
    ----------
    cif_df : pd.DataFrame
        DataFrame with CIF data (must have columns: time, cif, lower_ci, upper_ci)
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure
    figsize : tuple, optional
        Figure size if creating a new figure
    color : str, optional
        Color for the CIF line and confidence interval
    alpha : float, optional
        Alpha for confidence interval shading
    label : str, optional
        Label for legend
    show_ci : bool, optional
        Whether to show confidence intervals
        
    Returns
    -------
    plt.Axes
        The matplotlib axes with the CIF plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    # Plot CIF
    ax.plot(cif_df['time'], cif_df['cif'], color=color, label=label)
    
    # Plot confidence intervals
    if show_ci and 'lower_ci' in cif_df.columns and 'upper_ci' in cif_df.columns:
        ax.fill_between(cif_df['time'], cif_df['lower_ci'], cif_df['upper_ci'], 
                       color=color, alpha=alpha)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Incidence')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    if label is not None:
        ax.legend()
    
    return ax

def compare_cifs(
    cifs: Dict[str, pd.DataFrame],
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[Dict[str, str]] = None,
    alpha: float = 0.2,
    title: str = 'Comparison of Cumulative Incidence Functions',
    show_ci: bool = True,
) -> Tuple[Figure, Axes]:
    """Compare multiple cumulative incidence functions on a single plot.
    
    Parameters
    ----------
    cifs : Dict[str, pd.DataFrame]
        Dictionary mapping labels to CIF DataFrames
    figsize : tuple, optional
        Figure size
    colors : Dict[str, str], optional
        Dictionary mapping labels to colors
    alpha : float, optional
        Alpha for confidence interval shading
    title : str, optional
        Plot title
    show_ci : bool, optional
        Whether to show confidence intervals
        
    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes with the comparison plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        # Generate colors using default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors_cycle = prop_cycle.by_key()['color']
        colors = {label: colors_cycle[i % len(colors_cycle)] for i, label in enumerate(cifs.keys())}
    
    for label, cif_df in cifs.items():
        color = colors.get(label, 'blue')
        plot_cif(cif_df, ax=ax, color=color, alpha=alpha, label=label, show_ci=show_ci)
    
    ax.set_title(title)
    ax.legend()
    
    return fig, ax