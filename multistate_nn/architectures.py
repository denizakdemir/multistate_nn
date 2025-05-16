"""Neural architectures for intensity functions in continuous-time multistate models."""

from typing import Dict, List, Optional, Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntensityNetwork(nn.Module):
    """Base class for intensity function networks."""
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute intensity matrix.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [batch_size, input_dim]
        t : Optional[torch.Tensor]
            Time points, shape [batch_size, 1] or scalar
            
        Returns
        -------
        torch.Tensor
            Intensity matrix, shape [batch_size, num_states, num_states]
        """
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
        """Initialize MLP intensity network.
        
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
        use_layernorm : bool
            Whether to use layer normalization
        """
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
        """Compute intensity matrix.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [batch_size, input_dim]
        t : Optional[torch.Tensor]
            Time points (not used in this architecture)
            
        Returns
        -------
        torch.Tensor
            Intensity matrix, shape [batch_size, num_states, num_states]
        """
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
        """Initialize recurrent intensity network.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input features
        hidden_dim : int
            Dimension of hidden state
        num_states : int
            Number of states in the model
        state_transitions : Dict[int, List[int]]
            Dictionary mapping source states to possible target states
        cell_type : str
            Type of recurrent cell ('lstm' or 'gru')
        num_layers : int
            Number of recurrent layers
        """
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
        """Compute intensity matrix with time dependency.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [batch_size, input_dim]
        t : Optional[torch.Tensor]
            Time points, shape [batch_size, 1] or scalar
            
        Returns
        -------
        torch.Tensor
            Intensity matrix, shape [batch_size, num_states, num_states]
        """
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
        """Initialize attention-based intensity network.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input features
        hidden_dim : int
            Dimension of hidden state
        num_states : int
            Number of states in the model
        state_transitions : Dict[int, List[int]]
            Dictionary mapping source states to possible target states
        num_heads : int
            Number of attention heads
        num_layers : int
            Number of transformer layers
        dropout : float
            Dropout rate
        """
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
        """Compute intensity matrix with attention mechanism.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [batch_size, input_dim]
        t : Optional[torch.Tensor]
            Time points, shape [batch_size, 1] or scalar
            
        Returns
        -------
        torch.Tensor
            Intensity matrix, shape [batch_size, num_states, num_states]
        """
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
    """Create intensity network based on architecture type.
    
    Parameters
    ----------
    arch_type : str
        Architecture type ('mlp', 'recurrent', or 'attention')
    input_dim : int
        Dimension of input features
    num_states : int
        Number of states in the model
    state_transitions : Dict[int, List[int]]
        Dictionary mapping source states to possible target states
    **kwargs
        Additional architecture-specific parameters
        
    Returns
    -------
    IntensityNetwork
        Initialized intensity network
    """
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