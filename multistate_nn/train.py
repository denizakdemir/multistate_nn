"""Training utilities for MultiStateNN models."""

from typing import Any, Optional, List, Dict, Union, cast, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm.auto import tqdm

try:
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam as PyroAdam
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    pyro = None  # type: ignore

from .models import BaseMultiStateNN, MultiStateNN, BayesianMultiStateNN


@dataclass
class ModelConfig:
    """Configuration for MultiStateNN models.
    
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
    group_structure : Optional[Dict[tuple[int, int], Any]]
        Optional grouping structure for regularization
    """
    input_dim: int
    hidden_dims: List[int]
    num_states: int
    state_transitions: Dict[int, List[int]]
    group_structure: Optional[Dict[tuple[int, int], Any]] = None

    
@dataclass
class TrainConfig:
    """Configuration for training MultiStateNN models.
    
    Parameters
    ----------
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        Weight decay (L2 regularization) for optimizer
    device : Optional[torch.device]
        Device to use for training
    bayesian : bool
        Whether to use Bayesian inference
    """
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: Optional[torch.device] = None
    bayesian: bool = False


def prepare_data(
    df: pd.DataFrame,
    covariates: List[str],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare training data from a pandas DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data
    covariates : List[str]
        List of covariate column names to include
    device : Optional[torch.device]
        Device to place tensors on
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Tensors for x, time_idx, from_state, to_state
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    x = torch.tensor(df[covariates].values, dtype=torch.float32, device=device)
    time_idx = torch.tensor(df["time"].values, dtype=torch.int64, device=device)
    from_state = torch.tensor(df["from_state"].values, dtype=torch.int64, device=device)
    to_state = torch.tensor(df["to_state"].values, dtype=torch.int64, device=device)
    
    return x, time_idx, from_state, to_state


def train_model(
    model: BaseMultiStateNN,
    train_loader: DataLoader,
    train_config: TrainConfig,
) -> List[float]:
    """Train a MultiStateNN model.
    
    Parameters
    ----------
    model : BaseMultiStateNN
        Model to train
    train_loader : DataLoader
        DataLoader for training data
    train_config : TrainConfig
        Training configuration
        
    Returns
    -------
    List[float]
        Training losses per epoch
    """
    if train_config.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = train_config.device
    
    model_type = type(model)
    if model_type is MultiStateNN:
        return _train_deterministic(
            cast(MultiStateNN, model),
            train_loader,
            train_config,
        )
    elif model_type is BayesianMultiStateNN:
        return _train_bayesian(
            cast(BayesianMultiStateNN, model),
            train_loader, 
            train_config,
        )
    else:
        raise TypeError(f"Unsupported model type: {model_type}")


def _train_deterministic(
    model: MultiStateNN,
    train_loader: DataLoader,
    train_config: TrainConfig,
) -> List[float]:
    """Train a deterministic MultiStateNN model."""
    if train_config.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = train_config.device
    
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    
    losses: List[float] = []
    for _ in tqdm(range(train_config.epochs), desc="Training"):
        epoch_loss = 0.0
        n_batches = 0
        
        for x, time_idx, from_state, to_state in train_loader:
            x = x.to(device)
            time_idx = time_idx.to(device)
            from_state = from_state.to(device)
            to_state = to_state.to(device)
            
            optimizer.zero_grad()
            batch_loss = _compute_batch_loss(model, x, time_idx, from_state, to_state)
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            n_batches += 1
            
        losses.append(epoch_loss / n_batches)
    
    return losses


def _compute_batch_loss(
    model: MultiStateNN,
    x: torch.Tensor,
    time_idx: torch.Tensor,
    from_state: torch.Tensor,
    to_state: torch.Tensor,
) -> torch.Tensor:
    """Compute loss for a batch of data using vectorized operations."""
    batch_size = x.size(0)
    device = x.device
    
    # Map from_state and to_state to indices in state_transitions
    all_logits = []
    target_indices = []
    valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Process all states in the batch
    unique_from_states = torch.unique(from_state).tolist()
    
    for state in unique_from_states:
        # Get samples for this state
        state_mask = from_state == state
        if not state_mask.any():
            continue
            
        # Get next states for this state
        next_states = model.state_transitions[state]
        if not next_states:  # Skip absorbing states
            continue
            
        # Compute logits for all samples in this state
        x_state = x[state_mask]
        t_state = time_idx[state_mask]
        
        # Process each time index separately if needed
        for t in torch.unique(t_state).tolist():
            t_mask = t_state == t
            if not t_mask.any():
                continue
                
            x_state_t = x_state[t_mask]
            logits = model(x_state_t, t, state)
            
            # Get target indices
            to_state_t = to_state[state_mask][t_mask]
            targets = []
            for s in to_state_t:
                try:
                    targets.append(next_states.index(s.item()))
                except ValueError:
                    # Invalid transition - will be masked out
                    targets.append(-1)
            
            target_tensor = torch.tensor(targets, device=device)
            valid = target_tensor >= 0
            
            # Store logits and targets for valid transitions
            if valid.any():
                all_logits.append(logits[valid])
                target_indices.append(target_tensor[valid])
                
                # Update valid mask
                orig_indices = torch.where(state_mask)[0][t_mask][valid]
                valid_mask[orig_indices] = True
    
    # Compute loss for all valid transitions
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)
    
    loss = torch.tensor(0.0, device=device)
    for logits, targets in zip(all_logits, target_indices):
        loss = loss + nn.CrossEntropyLoss()(logits, targets)
    
    return loss / valid_mask.sum()


def _train_bayesian(
    model: BayesianMultiStateNN,
    train_loader: DataLoader,
    train_config: TrainConfig,
) -> List[float]:
    """Train a Bayesian MultiStateNN model using SVI."""
    if not PYRO_AVAILABLE:
        raise ImportError("Pyro must be installed for Bayesian training.")
        
    if train_config.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = train_config.device
    
    model = model.to(device)
    
    optimizer = PyroAdam({"lr": train_config.learning_rate})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
    
    losses: List[float] = []
    for _ in tqdm(range(train_config.epochs), desc="Training"):
        epoch_loss = 0.0
        n_batches = 0
        
        for x, time_idx, from_state, to_state in train_loader:
            x = x.to(device)
            time_idx = time_idx.to(device)
            from_state = from_state.to(device) 
            to_state = to_state.to(device)
            
            loss = svi.step(x, time_idx, from_state, to_state)
            epoch_loss += loss
            n_batches += 1
            
        losses.append(epoch_loss / n_batches)
        
    return losses


def fit(
    df: pd.DataFrame,
    covariates: List[str],
    model_config: ModelConfig,
    train_config: Optional[TrainConfig] = None,
) -> Union[MultiStateNN, BayesianMultiStateNN]:
    """Convenience function to fit a MultiStateNN model.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    covariates : List[str]
        List of covariate column names
    model_config : ModelConfig
        Model configuration
    train_config : Optional[TrainConfig]
        Training configuration, defaults to standard parameters
        
    Returns
    -------
    Union[MultiStateNN, BayesianMultiStateNN]
        Trained model
    """
    # Use default train config if not provided
    if train_config is None:
        train_config = TrainConfig()
    
    # Prepare data
    x, time_idx, from_state, to_state = prepare_data(
        df, covariates, train_config.device
    )
    dataset = TensorDataset(x, time_idx, from_state, to_state)
    train_loader = DataLoader(
        dataset, batch_size=train_config.batch_size, shuffle=True
    )
    
    # Initialize model
    model_cls = BayesianMultiStateNN if train_config.bayesian else MultiStateNN
    model = model_cls(
        input_dim=model_config.input_dim,
        hidden_dims=model_config.hidden_dims,
        num_states=model_config.num_states,
        state_transitions=model_config.state_transitions,
        group_structure=model_config.group_structure,
    )
    
    # Train
    train_model(model, train_loader, train_config)
    
    return model


