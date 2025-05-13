"""Visualization utilities for MultiStateNN models."""

from typing import Optional, Dict, List, Tuple, Union, Sequence
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ..models import BaseMultiStateNN


def plot_transition_heatmap(
    model: BaseMultiStateNN,
    x: torch.Tensor,
    time_idx: int,
    from_state: int,
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlOrRd",
) -> plt.Axes:
    """Plot transition probabilities as a heatmap.

    Parameters
    ----------
    model : BaseMultiStateNN
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


def compute_transition_matrix(
    model: BaseMultiStateNN,
    x: torch.Tensor,
    time_idx: int,
) -> np.ndarray:
    """Compute average transition probability matrix.

    Parameters
    ----------
    model : BaseMultiStateNN
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


def plot_transition_graph(
    model: BaseMultiStateNN,
    x: torch.Tensor,
    time_idx: int,
    threshold: float = 0.1,
    figsize: Tuple[float, float] = (10, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot transition graph using networkx.

    Parameters
    ----------
    model : BaseMultiStateNN
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


def plot_cif(
    cif_df: 'pd.DataFrame',
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


def compare_cifs(
    cif_list: List['pd.DataFrame'],
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