"""Training utilities for MultiStateNN models."""

from typing import Any, Optional, List, Dict, Union, cast, Tuple
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

try:
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam as PyroAdam
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    pyro = None  # type: ignore

from .models import BaseMultiStateNN, MultiStateNN
# Handle Bayesian model import conditionally
if PYRO_AVAILABLE:
    try:
        from .extensions.bayesian import BayesianMultiStateNN
    except ImportError:
        BayesianMultiStateNN = None
else:
    BayesianMultiStateNN = None
from .utils.time_mapping import TimeMapper


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
    use_original_time : bool
        Whether to preserve and use original time values
    """
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: Optional[torch.device] = None
    bayesian: bool = False
    use_original_time: bool = True


def prepare_data(
    df: pd.DataFrame,
    covariates: List[str],
    time_col: str = "time",
    censoring_col: Optional[str] = None,
    device: Optional[torch.device] = None,
    handle_missing: bool = True,
    impute_strategy: str = 'mean',
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Prepare training data from a pandas DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data
    covariates : List[str]
        List of covariate column names to include
    time_col : str, optional
        Name of the column containing time indices
    censoring_col : Optional[str], optional
        Name of the column containing censoring information (1=censored, 0=observed)
    device : Optional[torch.device]
        Device to place tensors on
    handle_missing : bool, optional
        Whether to handle missing values in covariates
    impute_strategy : str, optional
        Strategy for imputing missing values ('mean', 'median', 'mode', or 'zero')
        
    Returns
    -------
    Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        Tensors for x, time_idx, from_state, to_state, and optionally is_censored
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Handle missing values in covariates if requested
    if handle_missing:
        for col in covariates:
            if df[col].isna().any():
                if impute_strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif impute_strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif impute_strategy == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif impute_strategy == 'zero':
                    df[col].fillna(0, inplace=True)
                else:
                    raise ValueError(f"Unknown imputation strategy: {impute_strategy}")
    
    # Convert covariates to tensor, handling any data type issues
    try:
        x = torch.tensor(df[covariates].values, dtype=torch.float32, device=device)
    except (ValueError, TypeError) as e:
        # Try to handle non-numeric values
        warnings.warn(f"Error converting covariates to tensor: {e}. Attempting to convert to numeric.")
        for col in covariates:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col].fillna(df[col].mean(), inplace=True)  # Fill any NaNs created
                except:
                    raise ValueError(f"Could not convert column {col} to numeric type.")
        
        x = torch.tensor(df[covariates].values, dtype=torch.float32, device=device)
    
    # Convert core model values to tensors
    time_idx = torch.tensor(df[time_col].values, dtype=torch.int64, device=device)
    from_state = torch.tensor(df["from_state"].values, dtype=torch.int64, device=device)
    to_state = torch.tensor(df["to_state"].values, dtype=torch.int64, device=device)
    
    # Handle censoring if provided
    if censoring_col is not None:
        if censoring_col in df.columns:
            # Convert censoring values to boolean tensor, handling various input formats
            censoring_vals = df[censoring_col].values
            
            # Handle different input types (0/1, True/False, etc.)
            if pd.api.types.is_bool_dtype(df[censoring_col]):
                is_censored = torch.tensor(censoring_vals, dtype=torch.bool, device=device)
            elif pd.api.types.is_numeric_dtype(df[censoring_col]):
                is_censored = torch.tensor(censoring_vals > 0, dtype=torch.bool, device=device)
            else:
                # Try to interpret string values or other types
                try:
                    is_censored = torch.tensor(
                        [str(val).lower() in ('1', 'true', 't', 'yes', 'y') for val in censoring_vals],
                        dtype=torch.bool, 
                        device=device
                    )
                except:
                    raise ValueError(f"Could not convert censoring column {censoring_col} to boolean values.")
            
            return x, time_idx, from_state, to_state, is_censored
        else:
            warnings.warn(f"Censoring column '{censoring_col}' not found in the data. Proceeding without censoring.")
    
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
        
        # Check if the dataloader contains censoring information
        has_censoring = False
        for batch in train_loader:
            if len(batch) > 4:  # x, time_idx, from_state, to_state, is_censored
                has_censoring = True
            break
        
        for batch in train_loader:
            if has_censoring:
                x, time_idx, from_state, to_state, is_censored = batch
            else:
                x, time_idx, from_state, to_state = batch
                is_censored = None
            
            x = x.to(device)
            time_idx = time_idx.to(device)
            from_state = from_state.to(device)
            to_state = to_state.to(device)
            if is_censored is not None:
                is_censored = is_censored.to(device)
            
            optimizer.zero_grad()
            batch_loss = _compute_batch_loss(model, x, time_idx, from_state, to_state, is_censored)
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
    is_censored: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute loss for a batch of data using vectorized operations.
    
    For censored observations, we use a different loss function. Rather than penalizing
    specific transitions, we account for all possible future states by using a partial likelihood
    approach for right-censored observations.
    
    Parameters
    ----------
    model : MultiStateNN
        The model to compute loss for
    x : torch.Tensor
        Batch of features
    time_idx : torch.Tensor
        Batch of time indices
    from_state : torch.Tensor
        Batch of source states
    to_state : torch.Tensor
        Batch of target states
    is_censored : Optional[torch.Tensor], optional
        Batch of censoring indicators (True=censored, False=observed)
        
    Returns
    -------
    torch.Tensor
        Loss value for the batch
    """
    batch_size = x.size(0)
    device = x.device
    
    # Prepare data structures to collect logits and targets by state and time
    # This improves readability and performance by avoiding nested loops
    all_logits = []  # Will hold logits for valid transitions
    target_indices = []  # Will hold target indices for valid transitions
    valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Track information needed for censoring handling
    valid_indices = []  # Original indices for valid samples
    valid_from_states = []  # From states for valid samples
    valid_time_indices = []  # Time indices for valid samples
    
    # Process all states in the batch at once by grouping by unique from_states
    unique_from_states = torch.unique(from_state).tolist()
    
    for state in unique_from_states:
        # Skip processing if this state has no transitions
        next_states = model.state_transitions[state]
        if not next_states:  # Skip absorbing states
            continue
            
        # Get all samples for this state
        state_mask = from_state == state
        if not state_mask.any():
            continue
            
        # Get features and time points for this state
        x_state = x[state_mask]
        t_state = time_idx[state_mask]
        
        # Process each time index separately
        for t in torch.unique(t_state).tolist():
            t_mask = t_state == t
            if not t_mask.any():
                continue
                
            # Get features for samples at this time point
            x_state_t = x_state[t_mask]
            
            # Get model predictions (logits)
            logits = model(x_state_t, t, state)
            
            # Convert target states to indices in the state_transitions list
            to_state_t = to_state[state_mask][t_mask]
            targets = torch.tensor([
                next_states.index(s.item()) if s.item() in next_states else -1 
                for s in to_state_t
            ], device=device)
            
            # Keep only valid transitions
            valid = targets >= 0
            if valid.any():
                all_logits.append(logits[valid])
                target_indices.append(targets[valid])
                
                # Track original indices for censoring handling
                orig_indices = torch.where(state_mask)[0][t_mask][valid]
                valid_mask[orig_indices] = True
                
                # Store information needed for censoring
                valid_indices.append(orig_indices)
                valid_from_states.append(torch.full_like(orig_indices, state))
                valid_time_indices.append(torch.full_like(orig_indices, t))
    
    # No valid transitions in the batch
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)
    
    # Initialize loss
    loss = torch.tensor(0.0, device=device)
    
    # Handle censored and uncensored observations separately
    if is_censored is not None and valid_indices:
        # Process each batch of samples (grouped by state and time)
        for i, (logits, targets) in enumerate(zip(all_logits, target_indices)):
            if i >= len(valid_indices):
                continue

            # Get censoring status for this batch
            batch_censored = is_censored[valid_indices[i]]
            batch_uncensored = ~batch_censored
            
            # --- Process uncensored observations with standard cross-entropy ---
            if batch_uncensored.any():
                loss = loss + nn.CrossEntropyLoss(reduction='sum')(
                    logits[batch_uncensored], 
                    targets[batch_uncensored]
                )
            
            # --- Process censored observations with modified loss ---
            if batch_censored.any():
                # Get state and transition information
                cens_state = valid_from_states[i][0].item()
                next_possible_states = model.state_transitions[cens_state]
                
                # Get model predictions for censored observations
                cens_logits = logits[batch_censored]
                cens_probs = torch.softmax(cens_logits, dim=1)
                
                # Calculate censoring-aware loss
                # For right-censored data, we want to maximize the probability
                # of remaining in the current state (if possible)
                if cens_state in next_possible_states:
                    # With self-transition: maximize probability of staying in current state
                    self_idx = next_possible_states.index(cens_state)
                    stay_probs = cens_probs[:, self_idx]
                    censoring_loss = -torch.log(torch.clamp(stay_probs, min=1e-8)).sum()
                else:
                    # Without self-transition: equal probability for all transitions
                    # (maximum entropy approach)
                    uniform_probs = torch.ones_like(cens_probs) / cens_probs.size(1)
                    censoring_loss = F.kl_div(
                        F.log_softmax(cens_logits, dim=1),
                        uniform_probs,
                        reduction='sum'
                    )
                
                loss = loss + censoring_loss
        
        # Normalize the loss by the number of valid samples
        return loss / valid_mask.sum()
    
    # If no censoring information, use standard cross-entropy for all samples
    for logits, targets in zip(all_logits, target_indices):
        loss = loss + nn.CrossEntropyLoss(reduction='sum')(logits, targets)
    
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
        
        # Check if the dataloader contains censoring information
        has_censoring = False
        for batch in train_loader:
            if len(batch) > 4:  # x, time_idx, from_state, to_state, is_censored
                has_censoring = True
            break
        
        for batch in train_loader:
            if has_censoring:
                x, time_idx, from_state, to_state, is_censored = batch
            else:
                x, time_idx, from_state, to_state = batch
                is_censored = None
                
            x = x.to(device)
            time_idx = time_idx.to(device)
            from_state = from_state.to(device) 
            to_state = to_state.to(device)
            if is_censored is not None:
                is_censored = is_censored.to(device)
            
            # Pass censoring information to SVI step
            if is_censored is not None:
                loss = svi.step(x, time_idx, from_state, to_state, is_censored)
            else:
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
    censoring_col: Optional[str] = None,
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
    censoring_col : Optional[str], optional
        Name of the column containing censoring information (True=censored, False=observed)
        
    Returns
    -------
    Union[MultiStateNN, BayesianMultiStateNN]
        Trained model
    """
    # Use default train config if not provided
    if train_config is None:
        train_config = TrainConfig()
    
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Create time mapper if using original time
    time_mapper = None
    time_col = "time"
    
    if train_config.use_original_time:
        # Create time mapper from original time values
        time_mapper = TimeMapper(df["time"].values)
        
        # Map original time to indices
        df_copy = time_mapper.map_df_time_to_idx(df_copy, 
                                               time_col="time", 
                                               idx_col="time_idx")
        time_col = "time_idx"
    
    # Prepare data with or without censoring information
    if censoring_col is not None:
        data_tensors = prepare_data(
            df_copy, covariates, time_col=time_col, censoring_col=censoring_col, 
            device=train_config.device
        )
        if len(data_tensors) == 5:  # With censoring
            x, time_idx, from_state, to_state, is_censored = data_tensors
            dataset = TensorDataset(x, time_idx, from_state, to_state, is_censored)
        else:  # Censoring column not found
            x, time_idx, from_state, to_state = data_tensors
            dataset = TensorDataset(x, time_idx, from_state, to_state)
            print(f"Warning: Censoring column '{censoring_col}' not found in the data.")
    else:
        # Without censoring
        x, time_idx, from_state, to_state = prepare_data(
            df_copy, covariates, time_col=time_col, device=train_config.device
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
    
    # Attach time mapper to the model for later use in predictions and simulations
    if time_mapper is not None:
        model.time_mapper = time_mapper
    
    # Train
    train_model(model, train_loader, train_config)
    
    return model




