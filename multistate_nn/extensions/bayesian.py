"""Bayesian extensions for continuous-time multistate models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from typing import Optional, Dict, List, Any, Union, Tuple, Hashable, cast, Type
from torchdiffeq import odeint

# Try importing Pyro components with fallbacks for type checking
try:
    import pyro
    import pyro.distributions as dist
    import pyro.nn as pynn
    from pyro.infer import SVI, Trace_ELBO
    
    # Try different ways to get PyroAdam
    PyroAdam: Any = None
    try:
        # Try newer location
        from pyro.optim.optim import Adam as PyroAdam  # type: ignore
    except ImportError:
        try:
            # Try older location
            from pyro.optim import Adam as PyroAdam  # type: ignore
        except ImportError:
            # Fallback to a wrapper around torch's Adam
            import torch.optim
            class PyroAdamWrapper:
                def __init__(self, options: Dict[str, Any]):
                    self.options = options
                
                def __call__(self, params: Dict[str, Any]) -> torch.optim.Adam:
                    return torch.optim.Adam(params.values(), lr=self.options.get('lr', 0.001))
            
            PyroAdam = PyroAdamWrapper
            
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    pyro = None  # type: ignore
    dist = None  # type: ignore
    pynn = None  # type: ignore
    PyroAdam = None  # type: ignore

from ..models import BaseMultiStateNN, ContinuousMultiStateNN

__all__ = ["BayesianContinuousMultiStateNN"]


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
        
        # Get solver parameters from train_config during model creation in train.py
        self.solver = solver
        # Delete rtol and atol from options to avoid duplication
        self.solver_options = solver_options.copy() if solver_options else {}
        if 'rtol' in self.solver_options:
            del self.solver_options['rtol']
        if 'atol' in self.solver_options:
            del self.solver_options['atol']

        # Feature extractor (shared across all states)
        self.feature_net = self._create_bayesian_feature_net(input_dim, hidden_dims)
        self.output_dim = hidden_dims[-1]

        # Create intensity network with Bayesian layers
        # Create with string name first to avoid mypy error
        self.intensity_net = pynn.PyroModule("intensity_net")  # type: ignore
        # Then set up as linear layer
        self.intensity_net = cast(Any, pynn.PyroModule[nn.Linear](self.output_dim, num_states * num_states))  # type: ignore
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
            # Use cast to tell mypy this is an embedding
            self._group_emb = cast(nn.Embedding, self._create_bayesian_embedding(
                len(groups), self.output_dim, "group_embedding"
            ))
            # Use cast to tell mypy this is a parameter
            self._log_lambda = cast(nn.Parameter, pynn.PyroParam(
                torch.zeros(num_states, num_states)
            ))

    def _create_bayesian_feature_net(self, input_dim: int, hidden_dims: List[int]) -> pynn.PyroModule:
        """Create Bayesian feature extraction network."""
        layers = []
        prev = input_dim
        
        for i, width in enumerate(hidden_dims):
            # Linear layer with priors on weights and biases
            # Create with string name first to avoid mypy errors
            linear_name = f"linear_{i}"
            linear = pynn.PyroModule(linear_name)  # type: ignore
            # Then set up as linear layer
            linear = cast(Any, pynn.PyroModule[nn.Linear](prev, width))  # type: ignore
            
            # Register priors for weights and biases
            linear.weight = pynn.PyroSample(
                dist.Normal(0.0, self.prior_scale).expand([width, prev]).to_event(2)
            )
            linear.bias = pynn.PyroSample(
                dist.Normal(0.0, self.prior_scale).expand([width]).to_event(1)
            )
            
            layers.append(linear)
            # Cast regular PyTorch modules to Any to satisfy mypy
            layers.append(cast(Any, nn.ReLU(inplace=True)))
            layers.append(cast(Any, nn.LayerNorm(width)))
            
            prev = width
            
        # Create with string name first
        seq = pynn.PyroModule("feature_net_seq")  # type: ignore
        # Then set up as sequential with explicit cast
        seq = cast(Any, pynn.PyroModule[nn.Sequential](*layers))  # type: ignore
        return seq
        
    def _create_bayesian_embedding(self, num_embeddings: int, embedding_dim: int, name: str) -> Any:
        """Create Bayesian embedding layer."""
        # Create with string name first
        embedding = pynn.PyroModule(name)  # type: ignore
        # Then set up as embedding with explicit cast
        embedding = cast(Any, pynn.PyroModule[nn.Embedding](num_embeddings, embedding_dim))  # type: ignore
        
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
        
        # Apply softplus for non-negative rates and add small epsilon to avoid exact zeros
        A = F.softplus(A) * mask + 1e-10
        
        # Set diagonal to ensure rows sum to zero
        A_diag = -torch.sum(A, dim=2)
        A = A + torch.diag_embed(A_diag)
        
        # Apply additional stability checks - clamp very small negative values that might
        # arise from floating point errors to zero
        A = torch.where(A < -1e-8, A, torch.clamp(A, min=0.0))
        
        # Cast the result to tensor to satisfy mypy
        return cast(torch.Tensor, A)
        
    def ode_func(self, t: torch.Tensor, p: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """ODE function: dp/dt = p·A.
        
        Parameters
        ----------
        t : torch.Tensor
            Time point
        p : torch.Tensor
            Current probabilities (may be 1D, 2D or 3D)
        A : torch.Tensor
            Intensity matrix (3D: batch_size x num_states x num_states)
            
        Returns
        -------
        torch.Tensor
            Time derivative of probabilities
        """
        # First check p's dimensions and print shapes for debugging
        p_dim = p.dim()
        
        # Handle different input dimensions
        if p_dim == 2:
            # p is (batch_size x num_states)
            # Need to ensure A is 3D (batch x state x state)
            if A.dim() != 3:
                raise ValueError(f"A must be 3D tensor when p is 2D, got A.dim()={A.dim()}")
            
            # Need to unsqueeze to use batch matrix multiplication
            p_batch = p.unsqueeze(1)  # Shape: (batch_size, 1, num_states)
            result = torch.bmm(p_batch, A).squeeze(1)  # Shape: (batch_size, num_states)
            return result
            
        elif p_dim == 3:
            # p is already (batch_size x something x num_states)
            # Can directly use bmm
            return torch.bmm(p, A)
            
        elif p_dim == 1:
            # p is (num_states), need to reshape for batch matrix multiplication
            # First ensure A is at least 3D for bmm
            if A.dim() != 3:
                # If A is 2D, add batch dimension
                A_batched = A.unsqueeze(0)  # Shape: (1, state, state)
            else:
                A_batched = A
                
            # Reshape p to (1, 1, num_states) for bmm
            p_batched = p.view(1, 1, -1)
            
            # Perform bmm and reshape back to 1D
            result = torch.bmm(p_batched, A_batched).squeeze(0).squeeze(0)
            return result
            
        else:
            raise ValueError(f"Unexpected tensor dimension: p.dim()={p_dim}, expected 1, 2 or 3")
    
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
        
        # Use Pyro's block handler to prevent sample site registration during forward pass
        # This prevents name collisions when the forward method is called multiple times
        with pyro.poutine.block() if PYRO_AVAILABLE else contextlib.nullcontext():
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
            
            # Set default tolerance values
            rtol = 1e-7  # Default rtol
            atol = 1e-9  # Default atol
            
            # Define ODE function (without debug printing)
            def clean_ode_func(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
                return cast(torch.Tensor, self.ode_func(t, p, A))
            
            # Call odeint with properly structured arguments
            try:
                ode_result = odeint(
                    clean_ode_func,
                    p0,
                    times,
                    method=self.solver,
                    rtol=rtol,
                    atol=atol,
                    options=self.solver_options
                )
            except Exception as e:
                # If we encounter errors, we'll provide a more helpful message
                error_msg = str(e)
                if "Multiple sample sites named" in error_msg:
                    raise RuntimeError("Pyro sampling site name collision. Try using pyro.poutine.block() in the forward method.")
                raise
            
            # Return final state
            p_final = ode_result[-1]
            
            # Ensure probabilities are valid (non-negative and normalized)
            # This can fix issues with small negative values from numerical errors
            if from_state is not None:
                # For single-state predictions, make probabilities valid
                # First ensure non-negative values
                p_final_valid = torch.clamp(p_final, min=0.0)
                # Then normalize each row to sum to 1
                row_sums = p_final_valid.sum(dim=1, keepdim=True)
                # Avoid division by zero
                row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
                p_final_valid = p_final_valid / row_sums
                
                # Cast the result to tensor to satisfy mypy
                return cast(torch.Tensor, p_final_valid)
            else:
                # For multi-state predictions (dictionary output)
                # Apply the same corrections to each row
                p_final_valid = torch.clamp(p_final, min=0.0)
                # Normalize if needed
                row_sums = p_final_valid.sum(dim=2, keepdim=True)
                # Avoid division by zero
                row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
                p_final_valid = p_final_valid / row_sums
                
                # Create result dictionary with valid probabilities
                result = {i: p_final_valid[:, i, self.state_transitions[i]] 
                        if self.state_transitions[i] else torch.zeros((batch_size, 0), device=device) 
                        for i in self.state_transitions}
                return cast(Union[Dict[int, torch.Tensor], torch.Tensor], result)

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
            
            # Skip absorbing states - cast to int to satisfy mypy
            from_state_int = int(from_state_i)
            if not self.state_transitions[from_state_int]:
                continue
                
            # Compute transition probabilities with blocking parameter registration during the forward pass
            # This prevents sample site name collisions when called multiple times
            with pyro.poutine.block(hide_types=["param"]) if PYRO_AVAILABLE else contextlib.nullcontext():
                # Call forward with explicit int for from_state
                forward_result = self.forward(
                    x_i,
                    time_start=time_start_i,
                    time_end=time_end_i,
                    from_state=int(from_state_i)
                )
            
            # Handle tensor vs dict result
            if isinstance(forward_result, torch.Tensor):
                probs = forward_result.squeeze(0)
            else:
                # This shouldn't happen when from_state is provided, but handle it
                raise TypeError("Expected tensor output but got dictionary when from_state is provided")
            
            # For uncensored observations
            if to_state is not None and i < len(to_state) and not (is_censored is not None and is_censored[i].item()):
                to_state_i = to_state[i].item()
                
                # Ensure probabilities are valid (non-negative and sum to 1)
                # First, clamp to ensure non-negative values
                valid_probs = torch.clamp(probs, min=1e-10)
                # Then normalize to ensure they sum to 1
                valid_probs = valid_probs / valid_probs.sum()
                
                # Use categorical distribution for the observation with valid probabilities
                pyro.sample(
                    f"obs_{i}",
                    dist.Categorical(probs=valid_probs),
                    obs=torch.tensor(to_state_i, device=x.device)
                )
            # For censored observations
            elif is_censored is not None and i < len(is_censored) and is_censored[i].item():
                # For censored data, condition on survival (staying in current state)
                # Cast indices to int to satisfy mypy
                from_state_idx = int(from_state_i)
                
                # Handle potential invalid probabilities
                # Ensure the probability is valid and non-zero (for log operation)
                valid_prob = torch.clamp(probs[from_state_idx], min=1e-10)
                
                pyro.factor(
                    f"censored_{i}",
                    torch.log(valid_prob)
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


def train_bayesian_model(
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