"""Training utilities for MultiStateNN models."""

from typing import Any, Optional, List, Dict, Union, cast

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

from .models import MultiStateNN, BayesianMultiStateNN


def prepare_data(
    df: pd.DataFrame,
    covariates: List[str],
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare training data from a pandas DataFrame."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    x = torch.tensor(df[covariates].values, dtype=torch.float32, device=device)
    time_idx = torch.tensor(df["time"].values, dtype=torch.int64, device=device)
    from_state = torch.tensor(df["from_state"].values, dtype=torch.int64, device=device)
    to_state = torch.tensor(df["to_state"].values, dtype=torch.int64, device=device)
    
    return x, time_idx, from_state, to_state


def train_deterministic(
    model: MultiStateNN,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Train a deterministic MultiStateNN model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    losses: List[float] = []
    for _ in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0.0
        n_batches = 0
        
        for x, time_idx, from_state, to_state in train_loader:
            x = x.to(device)
            time_idx = time_idx.to(device)
            from_state = from_state.to(device)
            to_state = to_state.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass batch by batch
            batch_loss = torch.tensor(0.0, device=device)
            for i in range(len(x)):
                logits = model(
                    x[i:i+1],
                    time_idx[i].item(),
                    from_state[i].item()
                )
                if isinstance(logits, dict):
                    raise ValueError("from_state must be specified")
                next_states = model.state_transitions[from_state[i].item()]
                if not next_states:  # Absorbing state
                    continue
                idx = next_states.index(to_state[i].item())
                batch_loss = batch_loss + nn.CrossEntropyLoss()(
                    logits,
                    torch.tensor([idx], device=device)
                )
            
            loss = batch_loss / len(x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
        losses.append(epoch_loss / n_batches)
    
    return losses


def train_bayesian(
    model: BayesianMultiStateNN,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Train a Bayesian MultiStateNN model using SVI."""
    if not PYRO_AVAILABLE:
        raise ImportError("Pyro must be installed for Bayesian training.")
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = PyroAdam({"lr": learning_rate})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
    
    losses: List[float] = []
    for _ in tqdm(range(epochs), desc="Training"):
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
    input_dim: int,
    hidden_dims: List[int],
    num_states: int,
    state_transitions: Dict[int, List[int]],
    group_structure: Optional[Dict[tuple[int, int], Any]] = None,
    bayesian: bool = False,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Union[MultiStateNN, BayesianMultiStateNN]:
    """Convenience function to fit a MultiStateNN model."""
    # Prepare data
    x, time_idx, from_state, to_state = prepare_data(df, covariates, device)
    dataset = TensorDataset(x, time_idx, from_state, to_state)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model_cls = BayesianMultiStateNN if bayesian else MultiStateNN
    model = model_cls(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_states=num_states,
        state_transitions=state_transitions,
        group_structure=group_structure,
    )
    
    # Train
    if bayesian:
        model_bayes = cast(BayesianMultiStateNN, model)
        train_bayesian(
            model_bayes,
            train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )
    else:
        train_deterministic(
            model,
            train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )
        
    return model
