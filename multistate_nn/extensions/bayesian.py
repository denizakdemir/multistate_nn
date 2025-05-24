"""Bayesian extensions for continuous-time multistate models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from typing import Optional, Dict, List, Any, Union, Tuple, Hashable, cast, Type
from torchdiffeq import odeint

# Try importing Pyro components
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
        
        # Create mask for allowed transitions (exclude diagonal)
        mask = torch.zeros(self.num_states, self.num_states, device=x.device)
        for from_state, to_states in self.state_transitions.items():
            for to_state in to_states:
                if from_state != to_state:  # Only off-diagonal elements
                    mask[from_state, to_state] = 1.0
        
        # Apply softplus for non-negative off-diagonal rates where transitions are allowed
        # Zero out diagonal and disallowed transitions
        A_offdiag = F.softplus(A) * mask
        
        # Add small epsilon only to non-zero elements to avoid exact zeros in valid transitions
        epsilon = 1e-10
        A_offdiag = A_offdiag + epsilon * mask
        
        # Set diagonal elements to ensure rows sum to zero (Q-matrix property)
        A_diag = -torch.sum(A_offdiag, dim=2)
        
        # Create final intensity matrix
        A_final = A_offdiag.clone()
        # Set diagonal elements
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
        state_indices = torch.arange(self.num_states, device=x.device).unsqueeze(0)
        A_final[batch_indices, state_indices, state_indices] = A_diag
        
        # Verify mathematical constraints (optional, can be removed in production)
        row_sums = torch.sum(A_final, dim=2)
        max_row_sum_error = torch.max(torch.abs(row_sums))
        if max_row_sum_error > 1e-5:  # More lenient threshold for floating point precision
            # This should rarely happen with correct implementation
            import warnings
            warnings.warn(f"Intensity matrix row sums not zero (max error: {max_row_sum_error:.2e})")
        
        return A_final
        
    def ode_func(self, t: torch.Tensor, p: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """ODE function: dp/dt = p·A.
        
        Parameters
        ----------
        t : torch.Tensor
            Time point (unused but required by ODE solver interface)
        p : torch.Tensor
            Current probabilities 
        A : torch.Tensor
            Intensity matrix (3D: batch_size x num_states x num_states)
            
        Returns
        -------
        torch.Tensor
            Time derivative of probabilities with same shape as p
        """
        # Store original shape for output
        original_shape = p.shape
        
        # Ensure we have the right dimensions for batch matrix multiplication
        # Convert p to shape (batch_size, 1, num_states) for bmm
        if p.dim() == 1:
            # Single probability vector: (num_states,) -> (1, 1, num_states)
            p_bmm = p.unsqueeze(0).unsqueeze(0)
            A_bmm = A[:1]  # Use first batch element
        elif p.dim() == 2:
            # Batch of probability vectors: (batch_size, num_states) -> (batch_size, 1, num_states)
            p_bmm = p.unsqueeze(1)
            A_bmm = A
        elif p.dim() == 3:
            # Already in correct format for bmm: (batch_size, num_rows, num_states)
            p_bmm = p
            A_bmm = A
        else:
            raise ValueError(f"Unsupported tensor dimension: p.dim()={p.dim()}")
        
        # Ensure A has the right shape
        if A_bmm.dim() != 3:
            raise ValueError(f"A must be 3D tensor, got shape {A_bmm.shape}")
            
        # Check dimension compatibility
        if p_bmm.shape[-1] != A_bmm.shape[-2]:
            raise ValueError(f"Incompatible dimensions: p.shape[-1]={p_bmm.shape[-1]}, A.shape[-2]={A_bmm.shape[-2]}")
        
        # Perform matrix multiplication: p·A
        result = torch.bmm(p_bmm, A_bmm)
        
        # Reshape back to original format
        if len(original_shape) == 1:
            result = result.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            result = result.squeeze(1)
        # For 3D, result is already in correct shape
        
        return result
    
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
            
            # Use more conservative tolerance values for numerical stability
            rtol = 1e-6  # Slightly relaxed for stability
            atol = 1e-8  # More conservative than before
            
            # Define ODE function
            def clean_ode_func(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
                return self.ode_func(t, p, A)
            
            # Validate initial conditions
            if torch.any(p0 < -1e-10):
                raise ValueError("Initial probabilities contain significantly negative values")
            if torch.any(torch.isnan(p0)) or torch.any(torch.isinf(p0)):
                raise ValueError("Initial probabilities contain NaN or Inf values")
            
            # Call odeint with robust error handling
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
                error_msg = str(e)
                if "Multiple sample sites named" in error_msg:
                    raise RuntimeError("Pyro sampling site name collision. Use pyro.poutine.block() to prevent this.")
                elif "NaN" in error_msg or "nan" in error_msg:
                    raise RuntimeError(f"NaN encountered during ODE integration. This may indicate numerical instability. Original error: {e}")
                else:
                    raise RuntimeError(f"ODE integration failed: {e}")
            
            # Get final state and validate
            p_final = ode_result[-1]
            
            # Check for numerical issues
            if torch.any(torch.isnan(p_final)) or torch.any(torch.isinf(p_final)):
                raise RuntimeError("ODE solution contains NaN or Inf values")
            
            # Handle probability validation more carefully
            if from_state is not None:
                # For single-state predictions
                # Set tolerance for numerical precision issues
                neg_tolerance = 1e-8  # Allow small negative values due to floating point precision
                sum_tolerance = 1e-5  # Tolerance for sum deviation from 1.0
                
                # Check if probabilities are already valid (within tolerance)
                min_prob = p_final.min()
                max_prob = p_final.max()
                row_sums = p_final.sum(dim=1)
                
                if min_prob >= -neg_tolerance and torch.allclose(row_sums, torch.ones(batch_size), atol=sum_tolerance):
                    # Probabilities are valid within tolerance, just fix minor numerical errors
                    p_final_valid = torch.clamp(p_final, min=0.0)
                    # Renormalize only if necessary
                    current_sums = p_final_valid.sum(dim=1, keepdim=True)
                    if not torch.allclose(current_sums, torch.ones_like(current_sums), atol=sum_tolerance):
                        p_final_valid = p_final_valid / current_sums
                else:
                    # Only warn for significant violations (beyond reasonable floating point precision)
                    if min_prob < -1e-6 or not torch.allclose(row_sums, torch.ones(batch_size), atol=1e-4):
                        import warnings
                        warnings.warn(f"Significant probability violations detected. Min: {min_prob:.2e}, Max: {max_prob:.2e}, Row sum range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")
                    
                    # Apply corrections
                    p_final_valid = torch.clamp(p_final, min=0.0)
                    current_sums = p_final_valid.sum(dim=1, keepdim=True)
                    current_sums = torch.where(current_sums > 1e-12, current_sums, torch.ones_like(current_sums))
                    p_final_valid = p_final_valid / current_sums
                
                return p_final_valid
            else:
                # For multi-state predictions (transition matrices)
                neg_tolerance = 1e-8
                sum_tolerance = 1e-5
                
                # Apply similar validation to each transition matrix
                min_prob = p_final.min()
                max_prob = p_final.max()
                
                if min_prob >= -neg_tolerance:
                    p_final_valid = torch.clamp(p_final, min=0.0)
                    # Check and fix normalization
                    row_sums = p_final_valid.sum(dim=2, keepdim=True)
                    # Only renormalize if significantly off
                    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=sum_tolerance):
                        row_sums = torch.where(row_sums > 1e-12, row_sums, torch.ones_like(row_sums))
                        p_final_valid = p_final_valid / row_sums
                else:
                    # Only warn for significant violations
                    if min_prob < -1e-6:
                        import warnings
                        warnings.warn(f"Significant probability violations in transition matrix. Min: {min_prob:.2e}, Max: {max_prob:.2e}")
                    
                    p_final_valid = torch.clamp(p_final, min=0.0)
                    row_sums = p_final_valid.sum(dim=2, keepdim=True)
                    row_sums = torch.where(row_sums > 1e-12, row_sums, torch.ones_like(row_sums))
                    p_final_valid = p_final_valid / row_sums
                
                # Create result dictionary
                result = {}
                for i in self.state_transitions:
                    if self.state_transitions[i]:
                        result[i] = p_final_valid[:, i, self.state_transitions[i]]
                    else:
                        result[i] = torch.zeros((batch_size, 0), device=device)
                
                return result

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
                
                # Ensure probabilities are valid for categorical distribution
                epsilon = 1e-10
                valid_probs = torch.clamp(probs, min=epsilon)
                # Normalize to ensure they sum to 1
                valid_probs = valid_probs / valid_probs.sum()
                
                # Validate the target state index is within bounds
                if to_state_i >= len(valid_probs):
                    raise ValueError(f"Target state {to_state_i} is out of bounds for probability vector of length {len(valid_probs)}")
                
                # Use categorical distribution for the observation
                pyro.sample(
                    f"obs_{i}",
                    dist.Categorical(probs=valid_probs),
                    obs=torch.tensor(to_state_i, device=x.device, dtype=torch.long)
                )
            # For censored observations
            elif is_censored is not None and i < len(is_censored) and is_censored[i].item():
                # For censored data, condition on survival (staying in current state)
                # Cast indices to int to satisfy mypy
                from_state_idx = int(from_state_i)
                
                # For censored observations, we condition on the probability of being in the observed state
                # Ensure the probability is valid and non-zero (for log operation)
                if from_state_idx >= len(probs):
                    raise ValueError(f"From state {from_state_idx} is out of bounds for probability vector of length {len(probs)}")
                
                survival_prob = torch.clamp(probs[from_state_idx], min=1e-10, max=1.0)
                
                pyro.factor(
                    f"censored_{i}",
                    torch.log(survival_prob)
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