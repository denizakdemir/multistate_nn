"""Core continuous-time model definitions."""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, Hashable, cast, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
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