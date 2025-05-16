"""Core model definitions."""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, Hashable, cast, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BaseMultiStateNN",
    "MultiStateNN",
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

    def _temporal_smoothing(self, t: int) -> torch.Tensor:
        """Simple temporal smoothing function.
        
        Parameters
        ----------
        t : int
            Time index
            
        Returns
        -------
        torch.Tensor
            Smoothing factors for time t
        """
        # Return the temporal factors for time t
        # This simpler approach uses a learnable bias for each time point
        # with a reasonable decay factor as time increases
        decay_factor = torch.exp(torch.tensor(-0.1 * float(t), device=self.time_bias.device))
        gamma = self.time_bias * decay_factor
        return gamma

    def forward(
        self,
        x: torch.Tensor,
        time_idx: Optional[int] = None,
        from_state: Optional[int] = None,
    ) -> Union[Dict[int, torch.Tensor], torch.Tensor]:
        """Forward pass computing transition logits.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        time_idx : Optional[int]
            Time index for temporal effects
        from_state : Optional[int]
            Source state, if None returns logits for all states
            
        Returns
        -------
        Union[Dict[int, torch.Tensor], torch.Tensor]
            Dictionary of logits by state or logits for specific state
        """
        # To be implemented by subclasses
        raise NotImplementedError
    
    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        time_idx: int,
        from_state: int,
    ) -> torch.Tensor:
        """Predict transition probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        time_idx : int
            Time index
        from_state : int
            Source state
            
        Returns
        -------
        torch.Tensor
            Transition probabilities
        """
        logits = self.forward(x, time_idx, from_state)
        if isinstance(logits, dict):
            raise ValueError("from_state must be specified for predict_proba")
        if logits.size(1) == 0:  # Absorbing state
            return torch.ones((x.size(0), 1), device=x.device)
        
        # Convert logits to probabilities using softmax
        return F.softmax(logits, dim=1)


class MultiStateNN(nn.Module, BaseMultiStateNN):
    """Discrete‑time multistate neural network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_states: int,
        state_transitions: Dict[int, List[int]],
        group_structure: Optional[Dict[tuple[int, int], Hashable]] = None,
    ) -> None:
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
                # Use LayerNorm instead of BatchNorm
                nn.LayerNorm(width)
            ])
            prev = width
        self.feature_net = nn.Sequential(*layers)
        self.output_dim = prev

        # State‑specific heads
        self.state_heads = nn.ModuleDict()
        for i, nexts in state_transitions.items():
            if nexts:  # Skip absorbing states
                self.state_heads[str(i)] = nn.Linear(prev, len(nexts))

        # Simplified temporal smoothing
        self.time_bias = nn.Parameter(torch.zeros(num_states, num_states))

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

    @property
    def group_emb(self) -> Optional[nn.Embedding]:
        """Get group embeddings."""
        return self._group_emb

    @property
    def log_lambda(self) -> Optional[nn.Parameter]:
        """Get log lambda parameters."""
        return self._log_lambda

    def forward(
        self,
        x: torch.Tensor,
        time_idx: Optional[int] = None,
        from_state: Optional[int] = None,
    ) -> Union[Dict[int, torch.Tensor], torch.Tensor]:
        """Forward pass computing transition logits."""
        h = self.feature_net(x)

        def _one(i: int) -> torch.Tensor:
            if not self.state_transitions[i]:
                return torch.zeros((x.size(0), 0), device=x.device)
            logits = cast(torch.Tensor, self.state_heads[str(i)](h))
            if time_idx is not None:
                gamma = self._temporal_smoothing(time_idx)
                idx = torch.tensor(self.state_transitions[i], device=x.device)
                logits = logits + gamma[i, idx]
            return logits

        if from_state is not None:
            return _one(from_state)
        return {i: _one(i) for i in self.state_transitions}


# Bayesian model is moved to extensions module
# Import is handled in __init__.py to avoid circular import issues