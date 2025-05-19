"""Core continuous-time model definitions."""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, Hashable, cast, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import networkx as nx
import os
import json
from torchdiffeq import odeint

__all__ = [
    "BaseMultiStateNN",
    "ContinuousMultiStateNN",
]


class BaseMultiStateNN:
    """Base class for multistate models with shared functionality."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_states: int,
        state_transitions: Dict[int, List[int]],
        group_structure: Optional[Dict[tuple[int, int], Hashable]] = None,
    ) -> None:
        """Initialize base model parameters and attributes.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input features
        hidden_dims : List[int]
            List of hidden layer dimensions
        num_states : int
            Number of states in the model
        state_transitions : Dict[int, List[int]]
            Dictionary mapping source states to possible target states
        group_structure : Optional[Dict[tuple[int, int], Hashable]]
            Optional grouping structure for regularization
        """
        self.input_dim = input_dim
        self.num_states = num_states
        self.state_transitions = state_transitions
        self.group_structure = group_structure or {}

        # Initialize optional attributes for group structure
        self._group_index: Dict[Hashable, int] = {}
        self._group_emb: Optional[nn.Embedding] = None
        self._log_lambda: Optional[nn.Parameter] = None
    
    def forward(
        self,
        x: torch.Tensor,
        time_start: Union[float, torch.Tensor] = 0.0,
        time_end: Union[float, torch.Tensor] = 1.0,
        from_state: Optional[int] = None,
    ) -> Union[Dict[int, torch.Tensor], torch.Tensor]:
        """Forward pass computing transition probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        time_start : Union[float, torch.Tensor]
            Start time for probability calculation
        time_end : Union[float, torch.Tensor]
            End time for probability calculation
        from_state : Optional[int]
            Source state, if None returns probabilities for all states
            
        Returns
        -------
        Union[Dict[int, torch.Tensor], torch.Tensor]
            Dictionary of probabilities by state or probabilities for specific state
        """
        # To be implemented by subclasses
        raise NotImplementedError
    
    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        time_start: Union[float, torch.Tensor] = 0.0,
        time_end: Union[float, torch.Tensor] = 1.0,
        from_state: int = 0,
    ) -> torch.Tensor:
        """Predict transition probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        time_start : Union[float, torch.Tensor]
            Start time for probability calculation
        time_end : Union[float, torch.Tensor]
            End time for probability calculation
        from_state : int
            Source state
            
        Returns
        -------
        torch.Tensor
            Transition probabilities
        """
        return self.forward(x, time_start, time_end, from_state)
    
    def summary(self, print_fn=print) -> Dict[str, Any]:
        """Generate a summary of the model architecture and configuration.
        
        Parameters
        ----------
        print_fn : callable
            Function used for printing the summary (default: print)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing model summary information
        """
        # Create header
        model_type = self.__class__.__name__
        print_fn(f"==== {model_type} Summary ====")
        print_fn(f"Input dimension: {self.input_dim}")
        
        # Architecture details - will be extended by subclasses
        if hasattr(self, 'hidden_dims'):
            print_fn(f"Hidden dimensions: {getattr(self, 'hidden_dims')}")
        
        # State transition structure
        print_fn(f"Number of states: {self.num_states}")
        print_fn("State transition structure:")
        for from_state, to_states in self.state_transitions.items():
            if to_states:
                print_fn(f"  State {from_state} → States {to_states}")
            else:
                print_fn(f"  State {from_state} (absorbing state)")
        
        # Group structure if present
        if self.group_structure:
            print_fn("Group structure present: Yes")
            print_fn(f"Number of groups: {len(set(self.group_structure.values()))}")
        else:
            print_fn("Group structure present: No")
        
        # Create a summary dict to return
        summary_dict = {
            "model_type": model_type,
            "input_dim": self.input_dim,
            "num_states": self.num_states,
            "state_transitions": self.state_transitions.copy(),
            "has_group_structure": bool(self.group_structure)
        }
        
        # Add additional information specific to subclasses
        if hasattr(self, 'hidden_dims'):
            summary_dict["hidden_dims"] = getattr(self, 'hidden_dims')
        
        # Print parameter count if it's a torch.nn.Module
        if isinstance(self, nn.Module):
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print_fn(f"Total parameters: {total_params:,}")
            print_fn(f"Trainable parameters: {trainable_params:,}")
            summary_dict["total_params"] = total_params
            summary_dict["trainable_params"] = trainable_params
        
        print_fn("=" * (len(model_type) + 14))  # Match the header length
        
        return summary_dict
    
    def plot_transition_heatmap(
        self,
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
        all_probs = []
        all_states = []
        
        if from_state is not None:
            # Single starting state
            probs = self.predict_proba(x, time_start=time_start, time_end=time_end, from_state=from_state)
            probs = probs.detach().numpy()
            
            # Get next states
            next_states = self.state_transitions[from_state]
            
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
            
            num_states = self.num_states
            
            for state in range(num_states):
                if not self.state_transitions[state]:
                    continue
                    
                probs = self.predict_proba(x[0:1], time_start=time_start, time_end=time_end, from_state=state)
                probs = probs.detach().numpy()[0]
                
                next_states = self.state_transitions[state]
                
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
        self,
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
        num_states = self.num_states
        for state in range(num_states):
            G.add_node(state, label=f"State {state}")
        
        # Add edges based on transition probabilities
        max_prob = 0.0
        edges = []
        
        for from_state in range(num_states):
            if not self.state_transitions[from_state]:
                continue
                
            probs = self.predict_proba(x, time_start=time_start, time_end=time_end, from_state=from_state)
            probs = probs.detach().numpy()[0]
            
            next_states = self.state_transitions[from_state]
            
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
        cmap_obj = plt.get_cmap(cmap)
        
        for u, v, p in edges:
            # Normalize probability to 0-1 range for color mapping
            color = cmap_obj(p / max_prob if max_prob > 0 else 0)
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
    
    def plot_transition_probabilities_over_time(
        self,
        x: torch.Tensor,
        from_state: int = 0,
        max_time: float = 5.0,
        num_points: int = 50,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Axes:
        """Plot transition probabilities over time for a given set of input features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (1, input_dim)
        from_state : int, optional
            Starting state, default is 0
        max_time : float, optional
            Maximum time to plot, default is 5.0
        num_points : int, optional
            Number of time points to calculate, default is 50
        ax : plt.Axes, optional
            Matplotlib axes to plot on. If None, creates a new figure
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns
        -------
        plt.Axes
            Axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        if x.shape[0] != 1:
            x = x[0:1]  # Use only the first sample
        
        # Create time grid
        times = np.linspace(0, max_time, num_points)
        
        # Initialize arrays for probabilities
        probs = np.zeros((num_points, self.num_states))
        
        # Create tensor for time_start with proper dtype
        time_start = torch.tensor([0.0], dtype=torch.float32, device=x.device)
        
        # Calculate probabilities at each time point
        for i, t in enumerate(times):
            if i == 0:
                # At t=0, probability of being in the initial state is 1
                probs[i, from_state] = 1.0
            else:
                # Get probabilities from model
                time_end = torch.tensor([float(t)], dtype=torch.float32, device=x.device)
                p = self.predict_proba(x, time_start=time_start, time_end=time_end, from_state=from_state).detach()
                probs[i] = p.squeeze().cpu().numpy()
        
        # Plot probabilities
        for state in range(self.num_states):
            ax.plot(times, probs[:, state], label=f"State {state}", linewidth=2)
        
        ax.set_title(f"Transition Probabilities from State {from_state}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.grid(alpha=0.3)
        
        return ax
    
    def save(self, directory: str, filename: str = "model") -> str:
        """Save the model to disk.
        
        The method saves the model's state dict and configuration separately to 
        ensure easy loading and compatibility across versions.
        
        Parameters
        ----------
        directory : str
            Directory path where model will be saved
        filename : str, optional
            Base filename for saved model files, default is "model"
            
        Returns
        -------
        str
            Path to the model's state dictionary file
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save model configuration
        config = {
            "model_type": self.__class__.__name__,
            "input_dim": self.input_dim,
            "num_states": self.num_states,
            "state_transitions": {str(k): v for k, v in self.state_transitions.items()},
            "has_group_structure": bool(self.group_structure),
        }
        
        # Add class-specific config - to be overridden by subclasses
        self._add_specific_config(config)
        
        # Save configuration
        config_path = os.path.join(directory, f"{filename}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save state dict if the model is a torch.nn.Module
        if isinstance(self, nn.Module):
            state_dict_path = os.path.join(directory, f"{filename}_state_dict.pt")
            torch.save(self.state_dict(), state_dict_path)
            return state_dict_path
        else:
            return config_path
    
    def _add_specific_config(self, config: Dict[str, Any]) -> None:
        """Add class-specific configuration for serialization.
        
        This method is meant to be overridden by subclasses to add their specific
        configuration items to the provided config dictionary.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to be updated with class-specific items
        """
        # Base implementation does nothing, subclasses should override
        pass
    
    @classmethod
    def load(cls, directory: str, filename: str = "model", device: Optional[torch.device] = None) -> "BaseMultiStateNN":
        """Load a model from disk.
        
        Parameters
        ----------
        directory : str
            Directory path where model was saved
        filename : str, optional
            Base filename for saved model files, default is "model"
        device : Optional[torch.device], optional
            Device to load the model to, defaults to current device
            
        Returns
        -------
        BaseMultiStateNN
            Loaded model instance
        
        Raises
        ------
        FileNotFoundError
            If configuration file is not found
        ValueError
            If saved model type doesn't match the requested class
        """
        # Load configuration
        config_path = os.path.join(directory, f"{filename}_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model configuration file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Convert state_transitions back to proper format (str keys -> int keys)
        state_transitions = {int(k): v for k, v in config["state_transitions"].items()}
        
        # Check model type
        model_type = config["model_type"]
        caller_class_name = cls.__name__
        
        if model_type != caller_class_name:
            raise ValueError(f"Saved model is of type {model_type}, not {caller_class_name}")
        
        # Import the correct model class dynamically
        # This allows loading any model type from the file
        from multistate_nn import models as model_module
        model_class = getattr(model_module, model_type)
        
        # Create model instance - this will be delegated to subclass implementations
        return model_class._create_from_config(config, directory, filename, device)


class ContinuousMultiStateNN(nn.Module, BaseMultiStateNN):
    """Neural ODE-based continuous-time multistate model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_states: int,
        state_transitions: Dict[int, List[int]],
        group_structure: Optional[Dict[tuple[int, int], Hashable]] = None,
        solver: str = "dopri5",
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize continuous-time model.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input features
        hidden_dims : List[int]
            List of hidden layer dimensions
        num_states : int
            Number of states in the model
        state_transitions : Dict[int, List[int]]
            Dictionary mapping source states to possible target states
        group_structure : Optional[Dict[tuple[int, int], Hashable]]
            Optional grouping structure for regularization
        solver : str
            ODE solver to use ('dopri5', 'rk4', etc.)
        solver_options : Optional[Dict[str, Any]]
            Additional options for the ODE solver
        """
        nn.Module.__init__(self)
        BaseMultiStateNN.__init__(
            self,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_states=num_states,
            state_transitions=state_transitions,
            group_structure=group_structure,
        )
        
        # Feature extractor (shared across all states)
        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend([
                nn.Linear(prev, width),
                nn.ReLU(inplace=True),
                nn.LayerNorm(width)
            ])
            prev = width
        self.feature_net = nn.Sequential(*layers)
        self.output_dim = prev
        
        # Create intensity network that outputs transition rates
        self.intensity_net = nn.Linear(prev, num_states * num_states)
        
        # ODE solver settings
        self.solver = solver
        self.solver_options = solver_options or {}
        
        # Optional group embeddings
        if group_structure:
            # Convert hashable values to strings for sorting
            groups_str = sorted(str(g) for g in set(group_structure.values()))
            # Map back to original values maintaining sorted order
            groups = [next(g for g in set(group_structure.values()) 
                     if str(g) == g_str) for g_str in groups_str]
            self._group_index = {g: i for i, g in enumerate(groups)}
            self._group_emb = nn.Embedding(len(groups), prev)
            self._log_lambda = nn.Parameter(torch.zeros(num_states, num_states))
            
    def intensity_matrix(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute intensity matrix A(t) for given covariates x at time t.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [batch_size, input_dim]
        t : Optional[torch.Tensor]
            Time points, shape [batch_size, 1]
            
        Returns
        -------
        torch.Tensor
            Intensity matrix, shape [batch_size, num_states, num_states]
        """
        batch_size = x.shape[0]
        
        # Extract features
        h = self.feature_net(x)
        
        # Get raw intensity values
        outputs = self.intensity_net(h)
        
        # Reshape to batch_size x num_states x num_states
        A = outputs.view(batch_size, self.num_states, self.num_states)
        
        # Apply constraints: non-negative off-diagonal, zero-sum rows
        # Create mask for allowed transitions
        mask = torch.zeros(self.num_states, self.num_states, device=x.device)
        for from_state, to_states in self.state_transitions.items():
            for to_state in to_states:
                mask[from_state, to_state] = 1.0
        
        # Apply softplus to ensure non-negative off-diagonal elements
        A = F.softplus(A) * mask
        
        # Set diagonal to ensure rows sum to zero (Intensity matrix requirement)
        A_diag = -torch.sum(A, dim=2)
        A = A + torch.diag_embed(A_diag)
        
        return A
    
    def ode_func(self, t: torch.Tensor, p: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """ODE function: dp/dt = p·A.
        
        Parameters
        ----------
        t : torch.Tensor
            Time point
        p : torch.Tensor
            Current probabilities
        A : torch.Tensor
            Intensity matrix
            
        Returns
        -------
        torch.Tensor
            Time derivative of probabilities
        """
        # Handle different cases based on dimensions
        if p.dim() == 2:  # From specific state (batch_size x num_states)
            # Add batch dimension for bmm
            p_batch = p.unsqueeze(1)  # batch_size x 1 x num_states
            result = torch.bmm(p_batch, A).squeeze(1)  # batch_size x num_states
            return result
        else:  # Full transition matrix (batch_size x num_states x num_states)
            return torch.bmm(p, A)
    
    def forward(
        self,
        x: torch.Tensor,
        time_start: Union[float, torch.Tensor] = 0.0,
        time_end: Union[float, torch.Tensor] = 1.0,
        from_state: Optional[int] = None,
    ) -> Union[Dict[int, torch.Tensor], torch.Tensor]:
        """Compute transition probabilities from time_start to time_end.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        time_start : Union[float, torch.Tensor]
            Start time for probability calculation
        time_end : Union[float, torch.Tensor]
            End time for probability calculation
        from_state : Optional[int]
            Source state, if None returns probabilities for all states
            
        Returns
        -------
        Union[Dict[int, torch.Tensor], torch.Tensor]
            Dictionary of probabilities by state or probabilities for specific state
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Convert times to tensors if they're scalars
        if isinstance(time_start, (int, float)):
            time_start = torch.tensor([time_start], device=device)
        if isinstance(time_end, (int, float)):
            time_end = torch.tensor([time_end], device=device)
        
        # Special case: If time_start equals time_end, return identity matrix
        if torch.allclose(time_start, time_end):
            if from_state is not None:
                # Return probabilities for transitions from from_state
                probs = torch.zeros(batch_size, self.num_states, device=device)
                probs[:, from_state] = 1.0  # Identity matrix (stay in current state)
                return probs
            else:
                # Return dictionary with identity matrix for each state
                return {i: torch.zeros(batch_size, len(self.state_transitions[i]), device=device)
                       if self.state_transitions[i] else torch.zeros((batch_size, 0), device=device)
                       for i in self.state_transitions}
            
        # Compute intensity matrix
        A = self.intensity_matrix(x)
        
        # Set up initial condition based on from_state
        if from_state is not None:
            p0 = torch.zeros(batch_size, self.num_states, device=device)
            p0[:, from_state] = 1.0
        else:
            # Return full transition matrix for all states
            p0 = torch.eye(self.num_states, device=device).repeat(batch_size, 1, 1)
            p0 = p0.reshape(batch_size, self.num_states, self.num_states)
        
        # Ensure times are strictly increasing
        if time_end.item() <= time_start.item():
            time_end = time_start + 1e-6
        
        # Solve ODE to get transition probabilities
        times = torch.tensor([time_start.item(), time_end.item()], device=device)
        
        ode_result = odeint(
            lambda t, p: self.ode_func(t, p, A),
            p0,
            times,
            method=self.solver,
            options=self.solver_options
        )
        
        # Return final state
        p_final = ode_result[-1]
        
        if from_state is not None:
            # Get probabilities for transitions from from_state to all states
            return p_final
        else:
            # Return dictionary with transitions for each state
            return {i: p_final[:, i, self.state_transitions[i]] 
                   if self.state_transitions[i] else torch.zeros((batch_size, 0), device=device) 
                   for i in self.state_transitions}
    
    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        time_start: Union[float, torch.Tensor] = 0.0,
        time_end: Union[float, torch.Tensor] = 1.0,
        from_state: int = 0,
    ) -> torch.Tensor:
        """Predict transition probabilities for continuous-time model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        time_start : Union[float, torch.Tensor]
            Start time for probability calculation
        time_end : Union[float, torch.Tensor]
            End time for probability calculation
        from_state : int
            Source state
            
        Returns
        -------
        torch.Tensor
            Transition probabilities
        """
        return self.forward(x, time_start, time_end, from_state)
        
    def summary(self, print_fn=print) -> Dict[str, Any]:
        """Generate a detailed summary of the continuous-time model.
        
        Extends the base summary method with continuous-time specific info.
        
        Parameters
        ----------
        print_fn : callable
            Function used for printing the summary (default: print)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing model summary information
        """
        # Get the base summary
        summary_dict = super().summary(print_fn=print_fn)
        
        # Add continuous-time specific information
        print_fn("\nContinuous-time model specific:")
        print_fn(f"ODE solver: {self.solver}")
        
        # Print solver options if present
        if self.solver_options:
            print_fn("Solver options:")
            for key, value in self.solver_options.items():
                print_fn(f"  {key}: {value}")
        
        # Add to the summary dictionary
        summary_dict["model_type"] = "ContinuousMultiStateNN"
        summary_dict["ode_solver"] = self.solver
        summary_dict["ode_solver_options"] = self.solver_options.copy() if self.solver_options else {}
        summary_dict["hidden_dims"] = self.output_dim
        
        # Network architecture details
        layers_info = []
        for i, layer in enumerate(self.feature_net):
            if isinstance(layer, nn.Linear):
                layers_info.append(f"Linear({layer.in_features}, {layer.out_features})")
            elif isinstance(layer, nn.ReLU):
                layers_info.append("ReLU()")
            elif isinstance(layer, nn.LayerNorm):
                layers_info.append(f"LayerNorm({layer.normalized_shape[0]})")
                
        print_fn("\nNetwork architecture:")
        print_fn(f"Feature extraction: {' -> '.join(layers_info)}")
        print_fn(f"Intensity network: Linear({self.output_dim}, {self.num_states * self.num_states})")
        
        # Add to the summary dictionary
        summary_dict["architecture"] = {
            "feature_extraction": layers_info,
            "intensity_network": f"Linear({self.output_dim}, {self.num_states * self.num_states})"
        }
        
        return summary_dict
        
    def plot_intensity_matrix(
        self,
        x: torch.Tensor,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'coolwarm',
        annot: bool = True,
    ) -> plt.Axes:
        """Plot a heatmap of the intensity matrix for a given input.
        
        Parameters
        ----------
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
        
        # Get intensity matrix
        intensity_matrix = self.intensity_matrix(x).detach().cpu().numpy()[0]
        
        # For better visualization, we replace diagonal elements with zero
        # (diagonal elements are typically negative and can dominate the color scale)
        intensity_matrix_vis = intensity_matrix.copy()
        for i in range(intensity_matrix_vis.shape[0]):
            intensity_matrix_vis[i, i] = 0
        
        # Create heatmap
        sns.heatmap(intensity_matrix_vis, ax=ax, cmap=cmap, annot=annot, fmt=".3f")
        ax.set_title("Transition Intensity Matrix")
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        
        return ax
    
    def predict_trajectory(
        self,
        x: torch.Tensor,
        start_state: int = 0,
        max_time: float = 5.0,
        n_simulations: int = 10,
        time_step: float = 0.1,
        seed: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """Simulate state trajectories for a given input over time.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (1, input_dim)
        start_state : int, optional
            Initial state for the simulation
        max_time : float, optional
            Maximum time to simulate
        n_simulations : int, optional
            Number of trajectories to simulate
        time_step : float, optional
            Time step for the simulation
        seed : Optional[int], optional
            Random seed for reproducibility
            
        Returns
        -------
        List[pd.DataFrame]
            List of DataFrames containing simulated trajectories
        """
        if x.shape[0] != 1:
            x = x[0:1]  # Use only the first sample
            
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Create list for trajectories
        trajectories = []
        
        for sim_idx in range(n_simulations):
            # Initialize trajectory
            times = [0.0]
            states = [start_state]
            current_state = start_state
            current_time = 0.0
            
            # Simulate until max_time
            while current_time < max_time:
                # Get intensity matrix at current time
                intensity = self.intensity_matrix(x).detach().cpu().numpy()[0]
                
                # Get out-transition rate from current state (sum of off-diagonal elements in row)
                if current_state in self.state_transitions and self.state_transitions[current_state]:
                    # Sum the rates of all possible transitions
                    total_out_rate = -intensity[current_state, current_state]
                    
                    # If absorbing state or very low rate, stay in current state
                    if total_out_rate < 1e-10:
                        # Add final point at max_time if we're in an absorbing state
                        if current_time < max_time:
                            times.append(max_time)
                            states.append(current_state)
                        break
                    
                    # Sample time to next transition (exponential distribution)
                    time_to_next = np.random.exponential(scale=1.0/total_out_rate)
                    next_time = current_time + time_to_next
                    
                    # If next transition happens after max_time, truncate
                    if next_time > max_time:
                        times.append(max_time)
                        states.append(current_state)
                        break
                    
                    # Sample next state based on transition probabilities
                    next_states = self.state_transitions[current_state]
                    transition_probs = []
                    
                    for next_state in next_states:
                        # Get transition rate and normalize by total out-rate
                        rate = intensity[current_state, next_state]
                        transition_probs.append(rate / total_out_rate)
                    
                    # Normalize to ensure sum to 1 (handle numerical precision issues)
                    sum_probs = sum(transition_probs)
                    if sum_probs > 0:
                        transition_probs = [p / sum_probs for p in transition_probs]
                    else:
                        # If all probabilities are zero, use uniform distribution
                        transition_probs = [1.0 / len(next_states)] * len(next_states)
                    
                    # Sample next state
                    next_state = np.random.choice(next_states, p=transition_probs)
                    
                    # Update trajectory
                    times.append(next_time)
                    states.append(next_state)
                    
                    # Update current state and time
                    current_state = next_state
                    current_time = next_time
                else:
                    # Absorbing state, stay here until max_time
                    times.append(max_time)
                    states.append(current_state)
                    break
            
            # Create trajectory DataFrame
            trajectory = pd.DataFrame({
                'time': times,
                'state': states,
                'simulation': sim_idx
            })
            
            trajectories.append(trajectory)
            
        return trajectories
    
    def plot_state_distribution(
        self,
        x: torch.Tensor,
        start_state: int = 0,
        max_time: float = 5.0,
        n_simulations: int = 100,
        n_time_points: int = 50,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 6),
        seed: Optional[int] = None,
    ) -> plt.Axes:
        """Plot the distribution of states over time using simulations.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (1, input_dim)
        start_state : int, optional
            Initial state for the simulation
        max_time : float, optional
            Maximum time to simulate
        n_simulations : int, optional
            Number of trajectories to simulate
        n_time_points : int, optional
            Number of time points to evaluate
        ax : Optional[plt.Axes], optional
            Matplotlib axes to plot on
        figsize : Tuple[int, int], optional
            Figure size
        seed : Optional[int], optional
            Random seed for reproducibility
            
        Returns
        -------
        plt.Axes
            Axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        # Simulate trajectories
        trajectories = self.predict_trajectory(
            x=x,
            start_state=start_state,
            max_time=max_time,
            n_simulations=n_simulations,
            seed=seed,
        )
        
        # Create time grid for evaluation
        time_grid = np.linspace(0, max_time, n_time_points)
        
        # Calculate state distributions over time
        distributions = np.zeros((len(time_grid), self.num_states))
        
        # For each time point in the grid
        for i, t in enumerate(time_grid):
            # Get state at or before this time for each simulation
            for sim_idx, traj in enumerate(trajectories):
                # Find the index of the time point at or before t
                idx = np.searchsorted(traj['time'].values, t)
                if idx == 0:
                    # If t is before the first time point, use the first state
                    state = traj.iloc[0]['state']
                else:
                    # Otherwise use the state at or before t
                    state = traj.iloc[idx-1]['state']
                distributions[i, int(state)] += 1
            
            # Normalize
            distributions[i] = distributions[i] / n_simulations
        
        # Plot state distributions over time
        state_names = [f"State {s}" for s in range(self.num_states)]
        for state in range(self.num_states):
            ax.plot(time_grid, distributions[:, state], label=state_names[state], linewidth=2)
        
        ax.set_title(f"State Distribution Over Time (Starting from State {start_state})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.grid(alpha=0.3)
        
        return ax
        
    def _add_specific_config(self, config: Dict[str, Any]) -> None:
        """Add ContinuousMultiStateNN-specific configuration."""
        config["hidden_dims"] = [layer.out_features for layer in self.feature_net 
                               if isinstance(layer, nn.Linear)]
        config["solver"] = self.solver
        config["solver_options"] = self.solver_options
        
        # Save architecture details
        layers_info = []
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                layers_info.append({
                    "type": "Linear",
                    "in_features": layer.in_features,
                    "out_features": layer.out_features
                })
            elif isinstance(layer, nn.ReLU):
                layers_info.append({"type": "ReLU"})
            elif isinstance(layer, nn.LayerNorm):
                if isinstance(layer.normalized_shape, (list, tuple)):
                    norm_shape = layer.normalized_shape[0]
                else:
                    norm_shape = layer.normalized_shape
                layers_info.append({
                    "type": "LayerNorm",
                    "normalized_shape": norm_shape
                })
                
        config["architecture"] = {
            "feature_net": layers_info,
            "output_dim": self.output_dim
        }
    
    @classmethod
    def _create_from_config(cls, config: Dict[str, Any], directory: str, 
                          filename: str, device: Optional[torch.device] = None) -> "ContinuousMultiStateNN":
        """Create a ContinuousMultiStateNN instance from a saved configuration."""
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Convert state_transitions back to proper format (str keys -> int keys)
        state_transitions = {int(k): v for k, v in config["state_transitions"].items()}
        
        # Create model
        model = cls(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            num_states=config["num_states"],
            state_transitions=state_transitions,
            solver=config.get("solver", "dopri5"),
            solver_options=config.get("solver_options", {})
        )
        
        # Load state dict
        state_dict_path = os.path.join(directory, f"{filename}_state_dict.pt")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict)
            
        # Move model to device
        model = model.to(device)
        
        return model