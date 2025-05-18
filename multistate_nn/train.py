"""Training utilities for continuous-time MultiStateNN models."""

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

# Import base and continuous-time models
from .models import BaseMultiStateNN, ContinuousMultiStateNN
from .losses import ContinuousTimeMultiStateLoss, CompetingRisksContinuousLoss, create_loss_function
from .architectures import create_intensity_network

# Handle Bayesian model imports conditionally
if PYRO_AVAILABLE:
    try:
        from .extensions.bayesian import BayesianContinuousMultiStateNN, train_bayesian_model
    except ImportError:
        BayesianContinuousMultiStateNN = None
        train_bayesian_model = None
else:
    BayesianContinuousMultiStateNN = None
    train_bayesian_model = None


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
    model_type : str
        Type of model to use ('continuous', 'discrete')
    bayesian : bool
        Whether to use Bayesian inference
    """
    input_dim: int
    hidden_dims: List[int]
    num_states: int
    state_transitions: Dict[int, List[int]]
    group_structure: Optional[Dict[tuple[int, int], Any]] = None
    model_type: str = "continuous"
    bayesian: bool = False

    
@dataclass
class TrainConfig:
    """Configuration for training continuous-time MultiStateNN models.
    
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
    architecture_type : str
        Type of architecture for continuous-time intensity network ('mlp', 'recurrent', 'attention')
    loss_type : str
        Type of loss function for continuous-time model ('standard', 'competing_risks')
    competing_risk_states : List[int]
        States that represent competing risks (used only if loss_type='competing_risks')
    ode_solver : str
        ODE solver for continuous-time model ('dopri5', 'rk4', etc.)
    ode_solver_options : Optional[Dict[str, Any]]
        Additional options for the ODE solver
    architecture_options : Optional[Dict[str, Any]]
        Additional options for the neural architecture
    """
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: Optional[torch.device] = None
    bayesian: bool = False
    architecture_type: str = "mlp"
    loss_type: str = "standard"
    competing_risk_states: List[int] = None
    ode_solver: str = "dopri5"
    ode_solver_options: Optional[Dict[str, Any]] = None
    architecture_options: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Initialize empty lists/dicts if None provided
        if self.competing_risk_states is None:
            self.competing_risk_states = []
        if self.ode_solver_options is None:
            self.ode_solver_options = {}
        if self.architecture_options is None:
            self.architecture_options = {}


def prepare_data(
    df: pd.DataFrame,
    covariates: List[str],
    time_start_col: Optional[str] = None,
    time_end_col: Optional[str] = None,
    censoring_col: Optional[str] = None,
    device: Optional[torch.device] = None,
    handle_missing: bool = True,
    impute_strategy: str = 'mean',
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """Prepare training data from a pandas DataFrame for continuous-time models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data
    covariates : List[str]
        List of covariate column names to include
    time_start_col : Optional[str], optional
        Name of the column containing start times
    time_end_col : Optional[str], optional
        Name of the column containing end times
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
    Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]
        Tensors for x, time_start, time_end, from_state, to_state, and optionally is_censored
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
    
    # Get from_state and to_state tensors
    from_state = torch.tensor(df["from_state"].values, dtype=torch.int64, device=device)
    to_state = torch.tensor(df["to_state"].values, dtype=torch.int64, device=device)
    
    # For continuous-time models, we need time intervals
    if time_start_col is None or time_end_col is None:
        # If explicit time intervals aren't provided, try to infer them
        if "time" in df.columns:
            # Create time intervals from time points
            # This assumes that each row represents a transition at the end time
            # and the start time is the previous time for the same entity
            entity_col = None
            for col in ["id", "patient_id", "subject_id", "entity_id"]:
                if col in df.columns:
                    entity_col = col
                    break
            
            if entity_col is None:
                # Without entity information, use general approach
                warnings.warn("No entity identifier found. Using time points directly.")
                # Use time directly as both start and end times
                time_vals = df["time"].values.astype(np.float32)
                # For start times, shift each time value back by 1 (or use 0 for the first time)
                time_start = np.zeros_like(time_vals)
                time_start[1:] = time_vals[:-1]
            else:
                # Use entity information to create proper intervals
                df = df.sort_values([entity_col, "time"])
                time_vals = df["time"].values.astype(np.float32)
                entity_vals = df[entity_col].values
                
                # Initialize start times
                time_start = np.zeros_like(time_vals)
                
                # Set start times based on previous time for the same entity
                for i in range(1, len(df)):
                    if entity_vals[i] == entity_vals[i-1]:
                        time_start[i] = time_vals[i-1]
            
            # Convert to tensors
            time_start_tensor = torch.tensor(time_start, dtype=torch.float32, device=device)
            time_end_tensor = torch.tensor(time_vals, dtype=torch.float32, device=device)
        else:
            raise ValueError("For continuous-time models, either time_start_col and time_end_col "
                          "must be provided, or 'time' column must be present in the DataFrame.")
    else:
        # Use the provided start and end time columns
        time_start_tensor = torch.tensor(df[time_start_col].values, dtype=torch.float32, device=device)
        time_end_tensor = torch.tensor(df[time_end_col].values, dtype=torch.float32, device=device)
    
    # Handle censoring
    if censoring_col is not None and censoring_col in df.columns:
        # Convert censoring values to boolean tensor
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
        
        return x, time_start_tensor, time_end_tensor, from_state, to_state, is_censored
    
    # Without censoring
    return x, time_start_tensor, time_end_tensor, from_state, to_state


def get_loss_function(train_config: TrainConfig) -> nn.Module:
    """Create the appropriate loss function for model training.
    
    Parameters
    ----------
    train_config : TrainConfig
        Training configuration with loss parameters
        
    Returns
    -------
    nn.Module
        Loss function module for training
    """
    return create_loss_function(
        loss_type=train_config.loss_type,
        competing_risk_states=train_config.competing_risk_states
    )


def train_model(
    model: Union[ContinuousMultiStateNN, BayesianContinuousMultiStateNN],
    train_loader: DataLoader,
    train_config: TrainConfig,
) -> List[float]:
    """Train a continuous-time MultiStateNN model.
    
    Parameters
    ----------
    model : Union[ContinuousMultiStateNN, BayesianContinuousMultiStateNN]
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
    
    # Handle deterministic continuous-time model
    if model_type is ContinuousMultiStateNN:
        # Create loss function for continuous-time model
        loss_fn = get_loss_function(train_config)
        
        return _train_continuous(
            cast(ContinuousMultiStateNN, model),
            train_loader,
            loss_fn,
            train_config,
        )
    # Handle Bayesian continuous-time model
    elif model_type is BayesianContinuousMultiStateNN:
        # Bayesian training for continuous-time
        return _train_bayesian(
            cast(BayesianContinuousMultiStateNN, model),
            train_loader,
            train_config,
        )
    else:
        raise TypeError(f"Unsupported model type: {model_type}")


def _train_continuous(
    model: ContinuousMultiStateNN,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    train_config: TrainConfig,
) -> List[float]:
    """Train a continuous-time MultiStateNN model.
    
    Parameters
    ----------
    model : ContinuousMultiStateNN
        Continuous-time model to train
    train_loader : DataLoader
        DataLoader for training data
    loss_fn : nn.Module
        Loss function for continuous-time training
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
    
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    
    losses: List[float] = []
    for _ in tqdm(range(train_config.epochs), desc="Training continuous-time model"):
        epoch_loss = 0.0
        n_batches = 0
        
        # Check if the dataloader contains censoring information
        has_censoring = False
        for batch in train_loader:
            if len(batch) > 5:  # x, time_start, time_end, from_state, to_state, is_censored
                has_censoring = True
            break
        
        for batch in train_loader:
            if has_censoring:
                x, time_start, time_end, from_state, to_state, is_censored = batch
            else:
                x, time_start, time_end, from_state, to_state = batch
                is_censored = None
            
            # Move tensors to device
            x = x.to(device)
            time_start = time_start.to(device)
            time_end = time_end.to(device)
            from_state = from_state.to(device)
            to_state = to_state.to(device)
            if is_censored is not None:
                is_censored = is_censored.to(device)
            
            optimizer.zero_grad()
            
            # Compute loss using the provided loss function
            batch_loss = loss_fn(
                model=model,
                x=x,
                time_start=time_start,
                time_end=time_end,
                from_state=from_state,
                to_state=to_state,
                is_censored=is_censored
            )
            
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            n_batches += 1
            
        losses.append(epoch_loss / n_batches)
    
    return losses


def _train_bayesian(
    model: BayesianContinuousMultiStateNN,
    train_loader: DataLoader,
    train_config: TrainConfig,
) -> List[float]:
    """Train a Bayesian continuous-time model using SVI.
    
    Parameters
    ----------
    model : BayesianContinuousMultiStateNN
        Bayesian continuous-time model to train
    train_loader : DataLoader
        DataLoader for training data
    train_config : TrainConfig
        Training configuration
        
    Returns
    -------
    List[float]
        Training losses per epoch
    """
    if not PYRO_AVAILABLE:
        raise ImportError("Pyro must be installed for Bayesian training.")
        
    if train_config.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = train_config.device
    
    model = model.to(device)
    
    optimizer = PyroAdam({"lr": train_config.learning_rate})
    
    # Use AutoNormal guide factory with init_to_median for stable initialization
    guide = pyro.infer.autoguide.AutoNormal(model.model, init_loc_fn=pyro.infer.autoguide.init_to_median)
    
    svi = SVI(model.model, guide, optimizer, loss=Trace_ELBO())
    
    losses: List[float] = []
    for _ in tqdm(range(train_config.epochs), desc="Training Bayesian continuous-time model"):
        epoch_loss = 0.0
        n_batches = 0
        
        # Check if the dataloader contains censoring information
        has_censoring = False
        for batch in train_loader:
            if len(batch) > 5:  # x, time_start, time_end, from_state, to_state, is_censored
                has_censoring = True
            break
        
        for batch in train_loader:
            if has_censoring:
                x, time_start, time_end, from_state, to_state, is_censored = batch
            else:
                x, time_start, time_end, from_state, to_state = batch
                is_censored = None
            
            # Move tensors to device
            x = x.to(device)
            time_start = time_start.to(device)
            time_end = time_end.to(device)
            from_state = from_state.to(device)
            to_state = to_state.to(device)
            if is_censored is not None:
                is_censored = is_censored.to(device)
            
            # Pass time interval and censoring information to SVI step
            if is_censored is not None:
                loss = svi.step(x, time_start, time_end, from_state, to_state, is_censored)
            else:
                loss = svi.step(x, time_start, time_end, from_state, to_state)
                
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
    time_start_col: Optional[str] = None,
    time_end_col: Optional[str] = None,
) -> Union[ContinuousMultiStateNN, BayesianContinuousMultiStateNN]:
    """Convenience function to fit a MultiStateNN model.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    covariates : List[str]
        List of covariate column names
    model_config : ModelConfig
        Model configuration, which can include:
        - model_type: Type of model ('continuous', 'discrete'). Currently, only 'continuous' is fully supported.
        - bayesian: Whether to use Bayesian inference (requires Pyro)
    train_config : Optional[TrainConfig]
        Training configuration, defaults to standard parameters
    censoring_col : Optional[str], optional
        Name of the column containing censoring information (True=censored, False=observed)
    time_start_col : Optional[str], optional
        Name of the column containing start times (required for continuous-time models)
    time_end_col : Optional[str], optional
        Name of the column containing end times (required for continuous-time models)
        
    Returns
    -------
    Union[ContinuousMultiStateNN, BayesianContinuousMultiStateNN]
        Trained model
    """
    # Use default train config if not provided
    if train_config is None:
        train_config = TrainConfig()
    
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Determine model type based on configuration
    # Use model_type from ModelConfig, but fallback to the bayesian parameter in TrainConfig for backward compatibility
    if model_config.bayesian or train_config.bayesian:
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro must be installed for Bayesian models.")
        model_cls = BayesianContinuousMultiStateNN
    else:
        # For now, only continuous model type is supported, but this allows for future expansion
        if model_config.model_type != "continuous":
            warnings.warn(f"Model type '{model_config.model_type}' is not fully supported yet. Using continuous model.")
        model_cls = ContinuousMultiStateNN
    
    # Prepare data tensors for continuous-time models
    data_tensors = prepare_data(
        df_copy, 
        covariates, 
        time_start_col=time_start_col,
        time_end_col=time_end_col,
        censoring_col=censoring_col, 
        device=train_config.device
    )
    
    # Create dataset based on whether censoring information is available
    if len(data_tensors) == 6:  # With censoring
        x, time_start, time_end, from_state, to_state, is_censored = data_tensors
        dataset = TensorDataset(x, time_start, time_end, from_state, to_state, is_censored)
    else:  # Without censoring
        x, time_start, time_end, from_state, to_state = data_tensors
        dataset = TensorDataset(x, time_start, time_end, from_state, to_state)
    
    # Create DataLoader
    train_loader = DataLoader(
        dataset, batch_size=train_config.batch_size, shuffle=True
    )
    
    # Initialize the model
    if model_config.bayesian or train_config.bayesian:
        model = model_cls(
            input_dim=model_config.input_dim,
            hidden_dims=model_config.hidden_dims,
            num_states=model_config.num_states,
            state_transitions=model_config.state_transitions,
            group_structure=model_config.group_structure,
            prior_scale=1.0,  # Default prior scale
            solver=train_config.ode_solver,
            solver_options=train_config.ode_solver_options,
        )
    else:
        # For deterministic continuous-time model, optionally create an IntensityNetwork
        # based on the architecture type if not using default
        intensity_net = None
        if train_config.architecture_type != "mlp" or train_config.architecture_options:
            intensity_net = create_intensity_network(
                arch_type=train_config.architecture_type,
                input_dim=model_config.input_dim,
                num_states=model_config.num_states,
                state_transitions=model_config.state_transitions,
                **train_config.architecture_options
            )
        
        # Initialize the model with additional parameters specific to continuous-time models
        model = model_cls(
            input_dim=model_config.input_dim,
            hidden_dims=model_config.hidden_dims,
            num_states=model_config.num_states,
            state_transitions=model_config.state_transitions,
            group_structure=model_config.group_structure,
            solver=train_config.ode_solver,
            solver_options=train_config.ode_solver_options,
        )
        
        # TODO: Once IntensityNetwork can be passed to ContinuousMultiStateNN,
        # uncomment and update this section
        # if intensity_net:
        #     model.intensity_net = intensity_net
    
    # Train the model
    train_model(model, train_loader, train_config)
    
    return model