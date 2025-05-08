"""Core model definitions."""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, Hashable, cast, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyro
    import pyro.distributions as dist
    import pyro.nn as pynn
    PYRO_AVAILABLE = True
except ModuleNotFoundError:
    PYRO_AVAILABLE = False
    pyro = None  # type: ignore
    dist = None  # type: ignore
    pynn = None  # type: ignore

__all__ = [
    "MultiStateNN",
    "BayesianMultiStateNN",
]


class MultiStateNN(nn.Module):
    """Discrete‑time multistate neural network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_states: int,
        state_transitions: Dict[int, List[int]],
        group_structure: Optional[Dict[tuple[int, int], Hashable]] = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_states = num_states
        self.state_transitions = state_transitions
        self.group_structure = group_structure or {}

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

        # Temporal smoothness
        self.gamma_init = nn.Parameter(torch.zeros(num_states, num_states))
        self._gru = nn.GRUCell(num_states * num_states, num_states * num_states)

        # Initialize optional attributes
        self._group_index: Dict[Hashable, int] = {}
        self._group_emb: Optional[nn.Embedding] = None
        self._log_lambda: Optional[nn.Parameter] = None

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

    def _temporal_gamma(self, t: int) -> torch.Tensor:
        """Return temporal smoothing matrix."""
        gamma_flat = self.gamma_init.flatten()
        for _ in range(t):
            gamma_flat = self._gru(gamma_flat, gamma_flat)
        return gamma_flat.view(self.num_states, self.num_states)

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
                gamma = self._temporal_gamma(time_idx)
                idx = torch.tensor(self.state_transitions[i], device=x.device)
                logits = logits + gamma[i, idx]
            return logits

        if from_state is not None:
            return _one(from_state)
        return {i: _one(i) for i in self.state_transitions}

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        time_idx: int,
        from_state: int,
    ) -> torch.Tensor:
        """Predict transition probabilities."""
        logits = self.forward(x, time_idx, from_state)
        if isinstance(logits, dict):
            raise ValueError("from_state must be specified for predict_proba")
        if logits.size(1) == 0:  # Absorbing state
            return torch.ones((x.size(0), 1), device=x.device)
        return F.softmax(logits, dim=1)


class BayesianMultiStateNN(pynn.PyroModule):
    """Bayesian extension with variational inference."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_states: int,
        state_transitions: Dict[int, List[int]],
        group_structure: Optional[Dict[tuple[int, int], Hashable]] = None,
    ) -> None:
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro must be installed for BayesianMultiStateNN")
        super().__init__()

        self.input_dim = input_dim
        self.num_states = num_states
        self.state_transitions = state_transitions
        self.group_structure = group_structure or {}

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
        self.feature_net = pynn.PyroModule[nn.Sequential](*layers)
        self.output_dim = prev

        # State‑specific heads
        self.state_heads = pynn.PyroModule[nn.ModuleDict]()
        for i, nexts in state_transitions.items():
            if nexts:  # Skip absorbing states
                self.state_heads[str(i)] = pynn.PyroModule[nn.Linear](prev, len(nexts))

        # Temporal smoothness
        self.gamma_init = pynn.PyroParam(torch.zeros(num_states, num_states))
        self._gru = pynn.PyroModule[nn.GRUCell](num_states * num_states, num_states * num_states)

        # Initialize optional attributes
        self._group_index: Dict[Hashable, int] = {}
        self._group_emb: Optional[nn.Embedding] = None
        self._log_lambda: Optional[nn.Parameter] = None

        # Optional group embeddings
        if group_structure:
            # Convert hashable values to strings for sorting
            groups_str = sorted(str(g) for g in set(group_structure.values()))
            # Map back to original values maintaining sorted order
            groups = [next(g for g in set(group_structure.values()) 
                         if str(g) == g_str) for g_str in groups_str]
            self._group_index = {g: i for i, g in enumerate(groups)}
            self._group_emb = pynn.PyroModule[nn.Embedding](len(groups), prev)
            self._log_lambda = pynn.PyroParam(torch.zeros(num_states, num_states))

    @property
    def group_emb(self) -> Optional[nn.Embedding]:
        """Get group embeddings."""
        return self._group_emb

    @property
    def log_lambda(self) -> Optional[nn.Parameter]:
        """Get log lambda parameters."""
        return self._log_lambda

    def _temporal_gamma(self, t: int) -> torch.Tensor:
        """Return temporal smoothing matrix."""
        gamma_flat = self.gamma_init.flatten()
        for _ in range(t):
            gamma_flat = self._gru(gamma_flat, gamma_flat)
        return gamma_flat.view(self.num_states, self.num_states)

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
                gamma = self._temporal_gamma(time_idx)
                idx = torch.tensor(self.state_transitions[i], device=x.device)
                logits = logits + gamma[i, idx]
            return logits

        if from_state is not None:
            return _one(from_state)
        return {i: _one(i) for i in self.state_transitions}

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        time_idx: int,
        from_state: int,
    ) -> torch.Tensor:
        """Predict transition probabilities."""
        logits = self.forward(x, time_idx, from_state)
        if isinstance(logits, dict):
            raise ValueError("from_state must be specified for predict_proba")
        if logits.size(1) == 0:  # Absorbing state
            return torch.ones((x.size(0), 1), device=x.device)
        return F.softmax(logits, dim=1)

    def model(
        self,
        x: torch.Tensor,
        time_idx: torch.Tensor,
        from_state: torch.Tensor,
        to_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pyro model for variational inference."""
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro must be installed")

        batch_size = x.size(0)
        max_transitions = max(len(nexts) for nexts in self.state_transitions.values() if nexts)

        # Pre-compute all logits and mask invalid transitions
        logits = torch.zeros(batch_size, max_transitions, device=x.device)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        obs = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        for i in range(batch_size):
            state_int = int(from_state[i].item())
            next_states = self.state_transitions[state_int]
            
            if not next_states:  # Skip absorbing states
                continue

            # Compute logits for this sample
            curr_logits = self.forward(
                x[i:i+1],
                time_idx[i].item(),
                state_int
            )
            if isinstance(curr_logits, dict):
                raise ValueError("from_state must be specified")
            
            # Store logits and mark as valid
            logits[i, :curr_logits.size(1)] = curr_logits
            valid_mask[i] = True

            # Set observation index if available
            if to_state is not None:
                to_state_i = int(to_state[i].item())
                try:
                    obs[i] = next_states.index(to_state_i)
                    valid_mask[i] = True
                except ValueError:
                    valid_mask[i] = False

        # Only sample valid transitions
        if valid_mask.any() and to_state is not None:
            valid_logits = logits[valid_mask]
            valid_obs = obs[valid_mask]

            with pyro.poutine.block(hide=["obs"]):
                with pyro.plate("obs", len(valid_logits)):
                    pyro.sample(
                        "obs",
                        dist.Categorical(probs=F.softmax(valid_logits, dim=1)),
                        obs=valid_obs
                    )

        return logits

    def guide(
        self,
        x: torch.Tensor,
        time_idx: torch.Tensor,
        from_state: torch.Tensor,
        to_state: Optional[torch.Tensor] = None,
    ) -> None:
        """Pyro guide for variational inference."""
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro must be installed")
        # Guide is empty since we're using MAP estimation
