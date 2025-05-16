# Detailed Implementation Plan for MultiStateNN Transition to Continuous Time

## 1. Framework Transition to Continuous Time Models

### 1.1 Replace Discrete-Time with Continuous-Time Neural ODE Models

- **Description**: Completely replace the current discrete-time implementation with a continuous-time framework using Neural ODEs to better represent continuous-time processes.
- **Mathematical Formulation**:
  ```
  dh(t)/dt = f_θ(h(t), x, t)
  P(s, t) = exp(∫_s^t A(u) du)
  ```
  where `h(t)` is the hidden state, `f_θ` is a neural network with parameters θ, `P(s,t)` is the transition probability matrix from time s to t, and `A` is the intensity matrix.

- **Implementation Steps**:
  1. Create a new core class `ContinuousMultiStateNN` to replace the discrete-time one
  2. Add Neural ODE dependencies (torchdiffeq)
  3. Implement intensity network that outputs state transition rates
  4. Create ODE solver integration for transition probability calculation
  5. Modify simulation code to handle continuous paths
  6. Remove all discrete-time specific implementation

```python
# New file: multistate_nn/models_continuous.py
import torch
import torch.nn as nn
from torchdiffeq import odeint
from .models import BaseMultiStateNN

class ContinuousMultiStateNN(nn.Module, BaseMultiStateNN):
    """Neural ODE-based continuous-time multistate model."""
    
    def __init__(self, input_dim, hidden_dims, num_states, state_transitions, **kwargs):
        super().__init__(input_dim, hidden_dims, num_states, state_transitions, **kwargs)
        
        # Create intensity network that outputs transition rates
        self.intensity_net = nn.Sequential(
            self.feature_net,
            nn.Linear(hidden_dims[-1], num_states * num_states)
        )
        
    def intensity_matrix(self, x, t=None):
        """Compute intensity matrix A(t) for given covariates x at time t."""
        batch_size = x.shape[0]
        outputs = self.intensity_net(x)
        
        # Reshape to batch_size x num_states x num_states
        A = outputs.view(batch_size, self.num_states, self.num_states)
        
        # Apply constraints: non-negative off-diagonal, zero-sum rows
        # Create mask for allowed transitions
        mask = torch.zeros(self.num_states, self.num_states, device=x.device)
        for from_state, to_states in self.state_transitions.items():
            for to_state in to_states:
                mask[from_state, to_state] = 1.0
        
        # Apply softplus to ensure non-negative off-diagonal elements
        A = torch.softplus(A) * mask
        
        # Set diagonal to ensure rows sum to zero
        A_diag = -torch.sum(A, dim=2)
        A = A + torch.diag_embed(A_diag)
        
        return A
    
    def ode_func(self, t, p, A):
        """ODE function: dp/dt = p·A."""
        return torch.bmm(p, A)
    
    def forward(self, x, time_start=0.0, time_end=1.0, from_state=None):
        """Compute transition probabilities from time_start to time_end."""
        batch_size = x.shape[0]
        
        # Compute intensity matrix
        A = self.intensity_matrix(x)
        
        # Set up initial condition based from_state
        if from_state is not None:
            p0 = torch.zeros(batch_size, self.num_states, device=x.device)
            p0[:, from_state] = 1.0
        else:
            # Return full transition matrix for all states
            p0 = torch.eye(self.num_states, device=x.device).repeat(batch_size, 1, 1)
        
        # Solve ODE to get transition probabilities
        times = torch.tensor([time_start, time_end], device=x.device)
        ode_result = odeint(
            lambda t, p: self.ode_func(t, p, A),
            p0,
            times
        )
        
        # Return final state
        p_final = ode_result[-1]
        
        if from_state is not None:
            return p_final
        else:
            # Return dictionary with transitions for each state
            return {i: p_final[:, i, self.state_transitions[i]] for i in self.state_transitions}
```

### 1.2 Improved Censoring Mechanism for Continuous Time

- **Description**: Implement proper censoring handling in the continuous-time framework.
- **Implementation Steps**:
  1. Create a censoring-aware loss function for continuous-time models
  2. Implement right censoring, left truncation, and interval censoring support
  3. Add options for different informative censoring models

```python
# Add to multistate_nn/losses.py
class ContinuousTimeMultiStateLoss(nn.Module):
    """Loss function for continuous-time multistate models with censoring support."""
    
    def forward(
        self, 
        model,
        x: torch.Tensor,
        time_start: torch.Tensor,
        time_end: torch.Tensor,
        from_state: torch.Tensor,
        to_state: torch.Tensor,
        is_censored: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss with proper censoring handling for continuous time."""
        device = x.device
        batch_size = x.size(0)
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        for i in range(batch_size):
            from_i = from_state[i].item()
            to_i = to_state[i].item()
            time_start_i = time_start[i].item()
            time_end_i = time_end[i].item()
            censored_i = is_censored[i].item() if is_censored is not None else False
            
            # Get transition probabilities
            probs = model(
                x[i:i+1],
                time_start=time_start_i,
                time_end=time_end_i,
                from_state=from_i
            ).squeeze(0)
            
            if not censored_i:
                # For observed transitions, maximize probability of observed transition
                loss = loss - torch.log(torch.clamp(probs[to_i], min=1e-8))
                valid_samples += 1
            else:
                # For censored data, we know the subject stayed in from_state
                # Maximize probability of staying in the same state
                loss = loss - torch.log(torch.clamp(probs[from_i], min=1e-8))
                valid_samples += 1
        
        # Return mean loss
        return loss / max(1, valid_samples)
```

### 1.3 Replace Time-Adjustment Logic with Continuous-Time Approach

- **Description**: Replace all time-adjustment and discretization logic with proper continuous-time methods.
- **Implementation Steps**:
  1. Remove all discrete time binning code
  2. Use matrix exponential for time scaling
  3. Implement better validation methods for continuous time models

```python
# Replace in multistate_nn/utils/simulation.py
def adjust_transitions_for_time(P, time_diff):
    """
    Adjust transition matrix P for a different time step using
    the standard continuous-time approach.
    
    P(t) = exp(Q * t) where Q = log(P)
    P(k*t) = exp(Q * k*t) = exp(k * Q*t) = (P(t))^k
    
    Args:
        P: Transition probability matrix
        time_diff: Time scaling factor
        
    Returns:
        P_adjusted: Adjusted transition matrix
    """
    import scipy.linalg
    
    try:
        # Method 1: Matrix logarithm and exponential
        # Compute rate matrix Q = log(P)
        Q = scipy.linalg.logm(P)
        
        # Compute adjusted P = exp(time_diff * Q)
        P_adjusted = scipy.linalg.expm(time_diff * Q)
        
        # Ensure probabilities are valid
        if not np.all(np.isfinite(P_adjusted)) or np.any(P_adjusted < 0) or np.any(P_adjusted > 1):
            raise ValueError("Matrix method produced invalid probabilities")
            
        return P_adjusted
        
    except Exception as e:
        # Fallback: Use a simpler approximation with direct power
        # For time_diff > 1, P(n*t) ≈ P^n where n is the closest integer
        n = round(time_diff)
        if n > 0:
            P_power = np.linalg.matrix_power(P, n)
            
            # For non-integer time_diff, interpolate
            if not np.isclose(n, time_diff):
                alpha = time_diff - np.floor(time_diff)
                P_power_ceil = np.linalg.matrix_power(P, n+1)
                P_adjusted = (1-alpha) * P_power + alpha * P_power_ceil
            else:
                P_adjusted = P_power
                
            return P_adjusted
        elif n == 0:
            # For very small time steps, use identity matrix
            return np.eye(P.shape[0])
        else:
            # Should not happen with positive time_diff
            raise ValueError(f"Invalid time_diff: {time_diff}")
```

## 2. Bayesian Extensions for Continuous Time Models

### 2.1 Enhanced Bayesian Continuous-Time Model

- **Description**: Develop a Bayesian version of the continuous-time model with proper variational inference.
- **Implementation Steps**:
  1. Create a new `BayesianContinuousMultiStateNN` class
  2. Implement proper Bayesian inference for intensity matrices
  3. Add prior distributions for transition rates
  4. Support posterior sampling and uncertainty quantification

```python
# Add to multistate_nn/extensions/bayesian.py
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
    ) -> None:
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

    def intensity_matrix(self, x, t=None):
        """Compute intensity matrix with Bayesian parameters."""
        batch_size = x.shape[0]
        h = self.feature_net(x)
        outputs = self.intensity_net(h)
        
        # Reshape and apply constraints as in the non-Bayesian version
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
        
    def ode_func(self, t, p, A):
        """ODE function: dp/dt = p·A."""
        return torch.bmm(p, A)
    
    def forward(self, x, time_start=0.0, time_end=1.0, from_state=None):
        """Forward pass computing transition probabilities."""
        batch_size = x.shape[0]
        
        # Compute intensity matrix
        A = self.intensity_matrix(x)
        
        # Set up initial condition
        if from_state is not None:
            p0 = torch.zeros(batch_size, self.num_states, device=x.device)
            p0[:, from_state] = 1.0
        else:
            p0 = torch.eye(self.num_states, device=x.device).repeat(batch_size, 1, 1)
        
        # Solve ODE
        times = torch.tensor([time_start, time_end], device=x.device)
        ode_result = odeint(
            lambda t, p: self.ode_func(t, p, A),
            p0,
            times
        )
        
        # Return final state
        p_final = ode_result[-1]
        
        if from_state is not None:
            return p_final
        else:
            return {i: p_final[:, i, self.state_transitions[i]] for i in self.state_transitions}

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
```

### 2.2 Posterior Analysis Tools for Bayesian Models

- **Description**: Add comprehensive tools for Bayesian posterior analysis.
- **Implementation Steps**:
  1. Implement posterior predictive checks
  2. Add uncertainty quantification for predictions
  3. Create visualizations for posterior distributions

```python
# Add to multistate_nn/extensions/bayesian_utils.py
def generate_posterior_samples(
    model: BayesianContinuousMultiStateNN,
    x: torch.Tensor,
    time_grid: torch.Tensor,
    from_state: int,
    num_samples: int = 100
) -> Dict[str, np.ndarray]:
    """
    Generate posterior samples of transition probabilities.
    
    Args:
        model: Bayesian model
        x: Covariates
        time_grid: Grid of time points
        from_state: Starting state
        num_samples: Number of posterior samples
        
    Returns:
        Dictionary of posterior samples
    """
    import pyro
    
    # Initialize storage for samples
    n_times = len(time_grid)
    n_states = model.num_states
    
    samples = {
        'time': time_grid.numpy(),
        'transition_probs': np.zeros((num_samples, n_times, n_states)),
        'intensity_matrices': np.zeros((num_samples, n_states, n_states))
    }
    
    # Get posterior predictive
    for i in range(num_samples):
        # Get single posterior sample trace
        trace = pyro.poutine.trace(model).get_trace(
            x, 
            time_start=torch.tensor(0.0),
            time_end=torch.tensor(1.0),
            from_state=torch.tensor(from_state)
        )
        
        # For each time point
        for t_idx, t in enumerate(time_grid):
            # Get transition probabilities at this time
            probs = model(
                x,
                time_start=torch.tensor(0.0),
                time_end=t,
                from_state=from_state
            ).squeeze(0)
            
            samples['transition_probs'][i, t_idx] = probs.detach().numpy()
        
        # Get intensity matrix sample
        intensity = model.intensity_matrix(x).squeeze(0).detach().numpy()
        samples['intensity_matrices'][i] = intensity
    
    return samples

def plot_posterior_transition_probs(
    samples: Dict[str, np.ndarray],
    target_states: List[int] = None,
    credible_interval: float = 0.95
) -> plt.Figure:
    """
    Plot posterior distributions of transition probabilities.
    
    Args:
        samples: Posterior samples from generate_posterior_samples
        target_states: States to plot (defaults to all)
        credible_interval: Width of credible interval
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    times = samples['time']
    probs = samples['transition_probs']
    
    n_samples, n_times, n_states = probs.shape
    
    if target_states is None:
        target_states = list(range(n_states))
    
    # Create figure
    fig, axes = plt.subplots(1, len(target_states), figsize=(4 * len(target_states), 4))
    if len(target_states) == 1:
        axes = [axes]
    
    # Calculate credible intervals
    alpha = (1 - credible_interval) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    # Plot each target state
    for i, state in enumerate(target_states):
        ax = axes[i]
        
        # Calculate statistics
        mean_probs = np.mean(probs[:, :, state], axis=0)
        lower_ci = np.percentile(probs[:, :, state], lower_percentile, axis=0)
        upper_ci = np.percentile(probs[:, :, state], upper_percentile, axis=0)
        
        # Plot mean
        ax.plot(times, mean_probs, label=f'Mean', lw=2)
        
        # Plot CI
        ax.fill_between(times, lower_ci, upper_ci, alpha=0.3, 
                         label=f'{credible_interval:.0%} CI')
        
        ax.set_title(f'P(State {state} | t)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_posterior_cifs(
    model: BayesianContinuousMultiStateNN,
    x: torch.Tensor,
    time_grid: torch.Tensor,
    from_state: int,
    target_states: List[int],
    num_samples: int = 100
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Generate posterior samples of cumulative incidence functions.
    
    Args:
        model: Bayesian model
        x: Covariates
        time_grid: Grid of time points
        from_state: Starting state
        target_states: Target states for CIF calculation
        num_samples: Number of posterior samples
        
    Returns:
        Dictionary of posterior CIF samples for each target state
    """
    # First get transition probability samples
    trans_samples = generate_posterior_samples(
        model, x, time_grid, from_state, num_samples
    )
    
    # Initialize storage for CIFs
    cifs = {}
    for target in target_states:
        cifs[target] = np.zeros((num_samples, len(time_grid)))
    
    # Calculate CIF for each sample
    for i in range(num_samples):
        for t_idx, _ in enumerate(time_grid):
            # For each target state
            for target in target_states:
                if t_idx == 0:
                    # CIF is 0 at t=0
                    cifs[target][i, t_idx] = 0.0
                else:
                    # CIF at time t
                    # For continuous-time: P(absorption into state j by time t)
                    cifs[target][i, t_idx] = trans_samples['transition_probs'][i, t_idx, target]
    
    return {
        'time': time_grid.numpy(),
        'cifs': cifs
    }

def plot_posterior_cifs(
    cif_samples: Dict[str, Dict[int, np.ndarray]],
    credible_interval: float = 0.95
) -> plt.Figure:
    """
    Plot posterior distributions of cumulative incidence functions.
    
    Args:
        cif_samples: Posterior CIF samples from generate_posterior_cifs
        credible_interval: Width of credible interval
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    times = cif_samples['time']
    cifs = cif_samples['cifs']
    target_states = list(cifs.keys())
    
    # Create figure
    fig, axes = plt.subplots(1, len(target_states), figsize=(4 * len(target_states), 4))
    if len(target_states) == 1:
        axes = [axes]
    
    # Calculate credible intervals
    alpha = (1 - credible_interval) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    # Plot each target state
    for i, state in enumerate(target_states):
        ax = axes[i]
        cif_state = cifs[state]
        
        # Calculate statistics
        mean_cif = np.mean(cif_state, axis=0)
        lower_ci = np.percentile(cif_state, lower_percentile, axis=0)
        upper_ci = np.percentile(cif_state, upper_percentile, axis=0)
        
        # Plot mean
        ax.plot(times, mean_cif, label=f'Mean', lw=2)
        
        # Plot CI
        ax.fill_between(times, lower_ci, upper_ci, alpha=0.3, 
                         label=f'{credible_interval:.0%} CI')
        
        ax.set_title(f'CIF for State {state}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Incidence')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## 3. Advanced Neural Architectures for Continuous-Time Models

### 3.1 Specialized Architectures for Intensity Functions

- **Description**: Implement specialized neural architectures designed for intensity functions.
- **Implementation Steps**:
  1. Create recurrent and attention architectures for temporal dependencies
  2. Implement architectures that respect intensity function constraints
  3. Add specialized building blocks for hazard functions

```python
# Add to multistate_nn/architectures.py
class IntensityNetwork(nn.Module):
    """Base class for intensity function networks."""
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute intensity matrix."""
        raise NotImplementedError
        
class MLPIntensityNetwork(IntensityNetwork):
    """Simple MLP for intensity functions."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int],
        num_states: int,
        state_transitions: Dict[int, List[int]],
        use_layernorm: bool = True
    ):
        super().__init__()
        
        # Feature extractor
        layers = []
        prev = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU(inplace=True))
            if use_layernorm:
                layers.append(nn.LayerNorm(width))
            prev = width
        self.feature_net = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev, num_states * num_states)
        
        # Store state transitions for masking
        self.num_states = num_states
        self.state_transitions = state_transitions
        
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute intensity matrix."""
        batch_size = x.shape[0]
        
        # Extract features
        h = self.feature_net(x)
        
        # Get intensities
        A_flat = self.output_layer(h)
        A = A_flat.view(batch_size, self.num_states, self.num_states)
        
        # Apply constraints
        # 1. Create mask for allowed transitions
        mask = torch.zeros(self.num_states, self.num_states, device=x.device)
        for from_state, to_states in self.state_transitions.items():
            for to_state in to_states:
                mask[from_state, to_state] = 1.0
        
        # 2. Apply softplus for non-negative rates
        A = torch.softplus(A) * mask
        
        # 3. Set diagonal to ensure rows sum to zero
        A_diag = -torch.sum(A, dim=2)
        A = A + torch.diag_embed(A_diag)
        
        return A
        
class RecurrentIntensityNetwork(IntensityNetwork):
    """Recurrent network for time-dependent intensity functions."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_states: int,
        state_transitions: Dict[int, List[int]],
        cell_type: str = "gru",
        num_layers: int = 1
    ):
        super().__init__()
        
        # RNN for feature extraction
        if cell_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
        elif cell_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_states * num_states)
        
        # Time embedding
        self.time_embedding = nn.Linear(1, input_dim)
        
        # Store parameters
        self.num_states = num_states
        self.state_transitions = state_transitions
        self.input_dim = input_dim
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute intensity matrix with time dependency."""
        batch_size = x.shape[0]
        
        # Handle time embedding if provided
        if t is not None:
            # Convert to tensor if scalar
            if isinstance(t, (int, float)):
                t = torch.tensor([[t]], device=x.device)
            
            # Ensure correct shape
            if t.dim() == 1:
                t = t.unsqueeze(1)
                
            # Embed time
            t_emb = self.time_embedding(t)
            
            # Concatenate with input
            x_t = torch.cat([x, t_emb], dim=1)
        else:
            # No time information
            x_t = x
            
        # Apply RNN
        # Add sequence dimension if not present
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
            
        output, _ = self.rnn(x_t)
        
        # Get last output
        h = output[:, -1, :]
        
        # Get intensities
        A_flat = self.output_layer(h)
        A = A_flat.view(batch_size, self.num_states, self.num_states)
        
        # Apply constraints as in MLP version
        mask = torch.zeros(self.num_states, self.num_states, device=x.device)
        for from_state, to_states in self.state_transitions.items():
            for to_state in to_states:
                mask[from_state, to_state] = 1.0
        
        A = torch.softplus(A) * mask
        A_diag = -torch.sum(A, dim=2)
        A = A + torch.diag_embed(A_diag)
        
        return A
        
class AttentionIntensityNetwork(IntensityNetwork):
    """Attention-based network for time-dependent intensity functions."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_states: int,
        state_transitions: Dict[int, List[int]],
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Time embedding
        self.time_embedding = nn.Linear(1, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_states * num_states)
        
        # Store parameters
        self.num_states = num_states
        self.state_transitions = state_transitions
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute intensity matrix with attention mechanism."""
        batch_size = x.shape[0]
        
        # Handle time embedding if provided
        if t is not None:
            # Convert to tensor if scalar
            if isinstance(t, (int, float)):
                t = torch.tensor([[t]], device=x.device)
            
            # Ensure correct shape
            if t.dim() == 1:
                t = t.unsqueeze(1)
                
            # Embed time
            t_emb = self.time_embedding(t)
            
            # Create sequence with x and time
            x_seq = torch.cat([
                self.input_proj(x).unsqueeze(1),
                t_emb.unsqueeze(1)
            ], dim=1)
        else:
            # No time information, just use input
            x_seq = self.input_proj(x).unsqueeze(1)
            
        # Apply transformer
        h_seq = self.transformer(x_seq)
        
        # Use the first token output
        h = h_seq[:, 0, :]
        
        # Get intensities
        A_flat = self.output_layer(h)
        A = A_flat.view(batch_size, self.num_states, self.num_states)
        
        # Apply constraints
        mask = torch.zeros(self.num_states, self.num_states, device=x.device)
        for from_state, to_states in self.state_transitions.items():
            for to_state in to_states:
                mask[from_state, to_state] = 1.0
        
        A = torch.softplus(A) * mask
        A_diag = -torch.sum(A, dim=2)
        A = A + torch.diag_embed(A_diag)
        
        return A

# Factory function to create intensity networks
def create_intensity_network(
    arch_type: str,
    input_dim: int,
    num_states: int,
    state_transitions: Dict[int, List[int]],
    **kwargs
) -> IntensityNetwork:
    """Create intensity network based on architecture type."""
    if arch_type == "mlp":
        hidden_dims = kwargs.get("hidden_dims", [64, 32])
        use_layernorm = kwargs.get("use_layernorm", True)
        return MLPIntensityNetwork(
            input_dim, hidden_dims, num_states, state_transitions, use_layernorm
        )
    elif arch_type == "recurrent":
        hidden_dim = kwargs.get("hidden_dim", 64)
        cell_type = kwargs.get("cell_type", "gru")
        num_layers = kwargs.get("num_layers", 1)
        return RecurrentIntensityNetwork(
            input_dim, hidden_dim, num_states, state_transitions,
            cell_type, num_layers
        )
    elif arch_type == "attention":
        hidden_dim = kwargs.get("hidden_dim", 64)
        num_heads = kwargs.get("num_heads", 4)
        num_layers = kwargs.get("num_layers", 2)
        dropout = kwargs.get("dropout", 0.1)
        return AttentionIntensityNetwork(
            input_dim, hidden_dim, num_states, state_transitions,
            num_heads, num_layers, dropout
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")
```

## 4. Validation and Documentation

### 4.1 Comprehensive Validation Framework for Continuous-Time Models

- **Description**: Create validation tools for continuous-time models against established methods.
- **Implementation Steps**:
  1. Implement comparative benchmarks with R's msm and mstate packages for continuous-time models
  2. Add validation against theoretical solutions for simple cases
  3. Create visualization tools for model validation

### 4.2 End-to-End Workflow Examples

- **Description**: Create end-to-end examples for continuous-time model applications.
- **Implementation Steps**:
  1. Develop comprehensive notebooks for disease progression modeling
  2. Create competing risks analysis examples
  3. Implement practical examples with real-world datasets

### 4.3 Documentation Updates

- **Description**: Update all documentation to reflect the transition to continuous-time models.
- **Implementation Steps**:
  1. Create detailed API documentation for continuous-time models
  2. Update README and tutorials
  3. Add mathematical background for continuous-time processes

## 5. Implementation Timeline

- Week 1: Core continuous-time model implementation and ODE integration
- Week 2: Bayesian extensions and advanced architectures
- Week 3: Validation, examples, and benchmarking
- Week 4: Documentation, testing, and polishing

## Dependencies

- `torchdiffeq`: Required for Neural ODE implementation
- `pyro-ppl`: Required for Bayesian inference
- `networkx`: For transition graph visualization
- `rpy2`: Optional for benchmarking against R packages

The implementation will completely replace the current discrete-time approach with a continuous-time framework, providing a more theoretically sound foundation for multistate modeling.