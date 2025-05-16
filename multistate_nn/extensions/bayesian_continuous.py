"""Bayesian extensions for continuous-time multistate models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Any, Union, Tuple, Hashable, cast
from torchdiffeq import odeint

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
from ..models_continuous import ContinuousMultiStateNN


class BayesianContinuousMultiStateNN(pynn.PyroModule, BaseMultiStateNN):
    """Bayesian extension of continuous-time MultiStateNN using variational inference."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_states: int,
        state_transitions: Dict[int, List[int]],
        group_structure: Optional[Dict[tuple[int, int], Hashable]] = None,
        prior_scale: float = 1.0,
        use_lowrank_multivariate: bool = False,
        solver: str = "dopri5",
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize Bayesian continuous-time multistate model.
        
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
        prior_scale : float
            Scale of prior distributions
        use_lowrank_multivariate : bool
            Whether to use low-rank multivariate priors for correlation
        solver : str
            ODE solver to use ('dopri5', 'rk4', etc.)
        solver_options : Optional[Dict[str, Any]]
            Additional options for the ODE solver
        """
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro must be installed for BayesianContinuousMultiStateNN")
        pynn.PyroModule.__init__(self)
        BaseMultiStateNN.__init__(
            self,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_states=num_states,
            state_transitions=state_transitions,
            group_structure=group_structure,
        )
        
        self.prior_scale = prior_scale
        self.use_lowrank_multivariate = use_lowrank_multivariate
        self.solver = solver
        self.solver_options = solver_options or {}

        # Feature extractor (shared across all states)
        self.feature_net = self._create_bayesian_feature_net(input_dim, hidden_dims)
        self.output_dim = hidden_dims[-1]

        # Create intensity network with Bayesian layers
        self.intensity_net = pynn.PyroModule[nn.Linear](self.output_dim, num_states * num_states)
        self.intensity_net.weight = pynn.PyroSample(
            dist.Normal(0.0, self.prior_scale).expand([num_states * num_states, self.output_dim]).to_event(2)
        )
        self.intensity_net.bias = pynn.PyroSample(
            dist.Normal(0.0, self.prior_scale).expand([num_states * num_states]).to_event(1)
        )
        
        # Optional group embeddings
        if group_structure:
            # Convert hashable values to strings for sorting
            groups_str = sorted(str(g) for g in set(group_structure.values()))
            # Map back to original values maintaining sorted order
            groups = [next(g for g in set(group_structure.values()) 
                     if str(g) == g_str) for g_str in groups_str]
            self._group_index = {g: i for i, g in enumerate(groups)}
            self._group_emb = self._create_bayesian_embedding(
                len(groups), self.output_dim, "group_embedding"
            )
            self._log_lambda = pynn.PyroParam(
                torch.zeros(num_states, num_states)
            )

    def _create_bayesian_feature_net(self, input_dim: int, hidden_dims: List[int]) -> pynn.PyroModule:
        """Create Bayesian feature extraction network."""
        layers = []
        prev = input_dim
        
        for i, width in enumerate(hidden_dims):
            # Linear layer with priors on weights and biases
            linear = pynn.PyroModule[nn.Linear](prev, width)
            
            # Register priors for weights and biases
            linear.weight = pynn.PyroSample(
                dist.Normal(0.0, self.prior_scale).expand([width, prev]).to_event(2)
            )
            linear.bias = pynn.PyroSample(
                dist.Normal(0.0, self.prior_scale).expand([width]).to_event(1)
            )
            
            layers.append(linear)
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(width))
            
            prev = width
            
        return pynn.PyroModule[nn.Sequential](*layers)
        
    def _create_bayesian_embedding(self, num_embeddings: int, embedding_dim: int, name: str) -> pynn.PyroModule:
        """Create Bayesian embedding layer."""
        embedding = pynn.PyroModule[nn.Embedding](num_embeddings, embedding_dim)
        
        # Register prior for embeddings
        embedding.weight = pynn.PyroSample(
            dist.Normal(0.0, self.prior_scale).expand([num_embeddings, embedding_dim]).to_event(2)
        )
        
        return embedding

    def intensity_matrix(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute intensity matrix with Bayesian parameters."""
        batch_size = x.shape[0]
        h = self.feature_net(x)
        outputs = self.intensity_net(h)
        
        # Reshape and apply constraints
        A = outputs.view(batch_size, self.num_states, self.num_states)
        
        # Create mask for allowed transitions
        mask = torch.zeros(self.num_states, self.num_states, device=x.device)
        for from_state, to_states in self.state_transitions.items():
            for to_state in to_states:
                mask[from_state, to_state] = 1.0
        
        # Apply softplus for non-negative rates
        A = torch.softplus(A) * mask
        
        # Set diagonal to ensure rows sum to zero
        A_diag = -torch.sum(A, dim=2)
        A = A + torch.diag_embed(A_diag)
        
        return A
        
    def ode_func(self, t: torch.Tensor, p: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """ODE function: dp/dt = p·A."""
        return torch.bmm(p, A)
    
    def forward(
        self,
        x: torch.Tensor,
        time_start: Union[float, torch.Tensor] = 0.0,
        time_end: Union[float, torch.Tensor] = 1.0,
        from_state: Optional[int] = None,
    ) -> Union[Dict[int, torch.Tensor], torch.Tensor]:
        """Forward pass computing transition probabilities."""
        batch_size = x.shape[0]
        device = x.device
        
        # Convert times to tensors if they're scalars
        if isinstance(time_start, (int, float)):
            time_start = torch.tensor([time_start], device=device)
        if isinstance(time_end, (int, float)):
            time_end = torch.tensor([time_end], device=device)
        
        # Compute intensity matrix
        A = self.intensity_matrix(x)
        
        # Set up initial condition
        if from_state is not None:
            p0 = torch.zeros(batch_size, self.num_states, device=device)
            p0[:, from_state] = 1.0
        else:
            p0 = torch.eye(self.num_states, device=device).repeat(batch_size, 1, 1)
            p0 = p0.reshape(batch_size, self.num_states, self.num_states)
        
        # Solve ODE
        times = torch.cat([time_start.view(1), time_end.view(1)], dim=0)
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
            return p_final
        else:
            return {i: p_final[:, i, self.state_transitions[i]] 
                  if self.state_transitions[i] else torch.zeros((batch_size, 0), device=device) 
                  for i in self.state_transitions}

    def model(
        self,
        x: torch.Tensor,
        time_start: torch.Tensor,
        time_end: torch.Tensor,
        from_state: torch.Tensor,
        to_state: Optional[torch.Tensor] = None,
        is_censored: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pyro model for continuous-time variational inference with censoring support."""
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro must be installed")

        batch_size = x.size(0)
        
        # Process each sample
        for i in range(batch_size):
            x_i = x[i:i+1]
            from_state_i = from_state[i].item()
            time_start_i = time_start[i].item()
            time_end_i = time_end[i].item()
            
            # Skip absorbing states
            if not self.state_transitions[from_state_i]:
                continue
                
            # Compute transition probabilities
            probs = self.forward(
                x_i,
                time_start=time_start_i,
                time_end=time_end_i,
                from_state=from_state_i
            ).squeeze(0)
            
            # For uncensored observations
            if to_state is not None and i < len(to_state) and not (is_censored is not None and is_censored[i].item()):
                to_state_i = to_state[i].item()
                
                # Use categorical distribution for the observation
                pyro.sample(
                    f"obs_{i}",
                    dist.Categorical(probs=probs),
                    obs=torch.tensor(to_state_i, device=x.device)
                )
            # For censored observations
            elif is_censored is not None and i < len(is_censored) and is_censored[i].item():
                # For censored data, condition on survival (staying in current state)
                pyro.factor(
                    f"censored_{i}",
                    torch.log(torch.clamp(probs[from_state_i], min=1e-8))
                )
        
        return probs

    def guide(
        self,
        x: torch.Tensor,
        time_start: torch.Tensor,
        time_end: torch.Tensor,
        from_state: torch.Tensor,
        to_state: Optional[torch.Tensor] = None,
        is_censored: Optional[torch.Tensor] = None,
    ) -> None:
        """Pyro guide for variational inference."""
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro must be installed")
            
        # The guide is automatically created by AutoNormal
        pass


def train_bayesian_continuous(
    model: BayesianContinuousMultiStateNN,
    x: torch.Tensor,
    time_start: torch.Tensor,
    time_end: torch.Tensor,
    from_state: torch.Tensor,
    to_state: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    is_censored: Optional[torch.Tensor] = None,
) -> List[float]:
    """Train a Bayesian continuous-time multistate model using SVI.
    
    Parameters
    ----------
    model : BayesianContinuousMultiStateNN
        Model to train
    x : torch.Tensor
        Input features
    time_start : torch.Tensor
        Start times
    time_end : torch.Tensor
        End times
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
    is_censored : Optional[torch.Tensor], optional
        Binary indicator for censored observations
        
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
    time_start = time_start.to(device)
    time_end = time_end.to(device)
    from_state = from_state.to(device) 
    to_state = to_state.to(device)
    
    if is_censored is not None:
        is_censored = is_censored.to(device)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(x, time_start, time_end, from_state, to_state, is_censored)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(x, time_start, time_end, from_state, to_state)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Use AutoNormal guide factory with init_loc_fn=init_to_median 
    # for stable initialization
    guide = pyro.infer.autoguide.AutoNormal(model.model, init_loc_fn=pyro.infer.autoguide.init_to_median)
    
    optimizer = PyroAdam({"lr": learning_rate})
    svi = SVI(model.model, guide, optimizer, loss=Trace_ELBO())
    
    from tqdm.auto import tqdm
    losses: List[float] = []
    
    for _ in tqdm(range(epochs), desc="Training Bayesian model"):
        epoch_loss = 0.0
        n_batches = 0
        
        if is_censored is not None:
            for batch_x, batch_time_start, batch_time_end, batch_from, batch_to, batch_censored in train_loader:
                loss = svi.step(
                    batch_x, batch_time_start, batch_time_end, batch_from, batch_to, batch_censored
                )
                epoch_loss += loss
                n_batches += 1
        else:
            for batch_x, batch_time_start, batch_time_end, batch_from, batch_to in train_loader:
                loss = svi.step(
                    batch_x, batch_time_start, batch_time_end, batch_from, batch_to
                )
                epoch_loss += loss
                n_batches += 1
                
        losses.append(epoch_loss / n_batches)
        
    return losses