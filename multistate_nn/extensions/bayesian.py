"""Bayesian extensions for MultiStateNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Any, Union, Tuple, Hashable, cast

try:
    import pyro
    import pyro.distributions as dist
    import pyro.nn as pynn
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam as PyroAdam
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    pyro = None  # type: ignore
    dist = None  # type: ignore
    pynn = None  # type: ignore

from ..models import BaseMultiStateNN


class BayesianMultiStateNN(pynn.PyroModule, BaseMultiStateNN):
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
        pynn.PyroModule.__init__(self)
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
        self.feature_net = pynn.PyroModule[nn.Sequential](*layers)
        self.output_dim = prev

        # State‑specific heads
        self.state_heads = pynn.PyroModule[nn.ModuleDict]()
        for i, nexts in state_transitions.items():
            if nexts:  # Skip absorbing states
                self.state_heads[str(i)] = pynn.PyroModule[nn.Linear](prev, len(nexts))

        # Simplified temporal smoothing
        self.time_bias = pynn.PyroParam(torch.zeros(num_states, num_states))

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

        # Process the batch together for efficiency
        all_logits = []
        valid_indices = []
        for i, (x_i, time_i, state_i) in enumerate(zip(x, time_idx, from_state)):
            state_int = int(state_i.item())
            next_states = self.state_transitions[state_int]
            
            if not next_states:  # Skip absorbing states
                continue
            
            # Add this sample
            valid_indices.append(i)
            
            # Compute logits for this sample
            curr_logits = self.forward(
                x_i.unsqueeze(0),
                time_i.item(),
                state_int
            )
            if isinstance(curr_logits, dict):
                raise ValueError("from_state must be specified")
            
            # Store logits
            all_logits.append(curr_logits.squeeze(0))
        
        # Create a tensor for all valid logits
        if not valid_indices:
            # No valid transitions in this batch
            return torch.zeros((0, 0), device=x.device)
        
        # Initialize tensors for all items
        logits = torch.zeros(batch_size, max_transitions, device=x.device)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        obs = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Fill in valid entries
        for idx, i in enumerate(valid_indices):
            state_int = int(from_state[i].item())
            next_states = self.state_transitions[state_int]
            curr_logits = all_logits[idx]
            
            # Store logits and mark as valid
            logits[i, :curr_logits.size(0)] = curr_logits
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


def train_bayesian(
    model: BayesianMultiStateNN,
    x: torch.Tensor,
    time_idx: torch.Tensor,
    from_state: torch.Tensor,
    to_state: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Train a Bayesian MultiStateNN model using SVI.
    
    Parameters
    ----------
    model : BayesianMultiStateNN
        Model to train
    x : torch.Tensor
        Input features
    time_idx : torch.Tensor
        Time indices
    from_state : torch.Tensor
        Source states
    to_state : torch.Tensor
        Target states
    epochs : int, optional
        Number of training epochs
    batch_size : int, optional
        Batch size
    learning_rate : float, optional
        Learning rate
    device : Optional[torch.device], optional
        Device to use for training
        
    Returns
    -------
    List[float]
        Training losses
    """
    if not PYRO_AVAILABLE:
        raise ImportError("Pyro must be installed for Bayesian training.")
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    x = x.to(device)
    time_idx = time_idx.to(device)
    from_state = from_state.to(device) 
    to_state = to_state.to(device)
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(x, time_idx, from_state, to_state)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = PyroAdam({"lr": learning_rate})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
    
    from tqdm.auto import tqdm
    losses: List[float] = []
    for _ in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_time, batch_from, batch_to in train_loader:
            loss = svi.step(batch_x, batch_time, batch_from, batch_to)
            epoch_loss += loss
            n_batches += 1
            
        losses.append(epoch_loss / n_batches)
        
    return losses