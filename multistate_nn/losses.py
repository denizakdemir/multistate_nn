"""Loss functions for continuous-time multistate models."""

from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ContinuousTimeMultiStateLoss",
    "CompetingRisksContinuousLoss"
]


class ContinuousTimeMultiStateLoss(nn.Module):
    """Loss function for continuous-time multistate models with censoring support."""
    
    def forward(
        self, 
        model: nn.Module,
        x: torch.Tensor,
        time_start: torch.Tensor,
        time_end: torch.Tensor,
        from_state: torch.Tensor,
        to_state: torch.Tensor,
        is_censored: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss with proper censoring handling for continuous time.
        
        Parameters
        ----------
        model : nn.Module
            Continuous-time multistate model
        x : torch.Tensor
            Input features
        time_start : torch.Tensor
            Start times for each sample
        time_end : torch.Tensor
            End times for each sample
        from_state : torch.Tensor
            Source states
        to_state : torch.Tensor
            Target states
        is_censored : Optional[torch.Tensor]
            Binary indicator for censored observations
            
        Returns
        -------
        torch.Tensor
            Computed loss value
        """
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
            
            # Skip absorbing states
            if not hasattr(model, 'state_transitions') or from_i not in getattr(model, 'state_transitions', {}):
                continue
                
            # Get transition probabilities
            probs = model(
                x[i:i+1],
                time_start=time_start_i,
                time_end=time_end_i,
                from_state=from_i
            ).squeeze(0)
            
            if not censored_i:
                # For observed transitions, maximize probability of observed transition
                if to_i < probs.size(0):  # Ensure to_i is valid
                    loss = loss - torch.log(torch.clamp(probs[to_i], min=1e-8))
                    valid_samples += 1
            else:
                # For censored data, we know the subject was still in from_state at censoring time
                # So we maximize the probability of staying in the same state
                loss = loss - torch.log(torch.clamp(probs[from_i], min=1e-8))
                valid_samples += 1
        
        # Return mean loss
        return loss / max(1, valid_samples)


class CompetingRisksContinuousLoss(nn.Module):
    """Loss function for continuous-time competing risks models."""
    
    def __init__(self, competing_risk_states: List[int]):
        """Initialize competing risks loss function.
        
        Parameters
        ----------
        competing_risk_states : List[int]
            List of states that represent competing risks
        """
        super().__init__()
        self.competing_risk_states = competing_risk_states
    
    def forward(
        self, 
        model: nn.Module,
        x: torch.Tensor,
        time_start: torch.Tensor,
        time_end: torch.Tensor,
        from_state: torch.Tensor,
        to_state: torch.Tensor,
        is_censored: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss for competing risks scenario in continuous time.
        
        Parameters
        ----------
        model : nn.Module
            Continuous-time multistate model
        x : torch.Tensor
            Input features
        time_start : torch.Tensor
            Start times for each sample
        time_end : torch.Tensor
            End times for each sample
        from_state : torch.Tensor
            Source states
        to_state : torch.Tensor
            Target states
        is_censored : Optional[torch.Tensor]
            Binary indicator for censored observations
            
        Returns
        -------
        torch.Tensor
            Computed loss value
        """
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
            
            # Skip absorbing states
            if not hasattr(model, 'state_transitions') or from_i not in getattr(model, 'state_transitions', {}):
                continue
                
            # Get transition probabilities
            probs = model(
                x[i:i+1],
                time_start=time_start_i,
                time_end=time_end_i,
                from_state=from_i
            ).squeeze(0)
            
            if not censored_i:
                # For observed transitions, check if it's to a competing risk state
                if to_i in self.competing_risk_states:
                    # For competing risks, use cause-specific likelihood
                    if to_i < probs.size(0):  # Ensure to_i is valid
                        loss = loss - torch.log(torch.clamp(probs[to_i], min=1e-8))
                        valid_samples += 1
                else:
                    # For non-competing risk, standard likelihood
                    if to_i < probs.size(0):  # Ensure to_i is valid
                        loss = loss - torch.log(torch.clamp(probs[to_i], min=1e-8))
                        valid_samples += 1
            else:
                # For censored observations
                # Overall survival: probability of not transitioning to any competing risk
                competing_risk_prob = torch.sum(
                    probs[torch.tensor(
                        [s for s in self.competing_risk_states if s < probs.size(0)],
                        device=device
                    )]
                )
                survival_prob = 1.0 - competing_risk_prob
                loss = loss - torch.log(torch.clamp(survival_prob, min=1e-8))
                valid_samples += 1
        
        # Return mean loss
        return loss / max(1, valid_samples)


# Factory function to create appropriate loss function
def create_loss_function(
    loss_type: str = "standard",
    **kwargs: Any
) -> nn.Module:
    """Create an appropriate loss function based on the specified type.
    
    Parameters
    ----------
    loss_type : str
        Type of loss function to create ('standard' or 'competing_risks')
    **kwargs : Any
        Additional keyword arguments to pass to the loss function constructor
        
    Returns
    -------
    nn.Module
        Instance of the requested loss function
    
    Raises
    ------
    ValueError
        If an invalid loss_type is provided
    """
    """Create loss function based on type.
    
    Parameters
    ----------
    loss_type : str
        Type of loss function ('standard' or 'competing_risks')
    **kwargs
        Additional arguments for specific loss functions
    
    Returns
    -------
    nn.Module
        Loss function module
    """
    if loss_type == "standard":
        return ContinuousTimeMultiStateLoss()
    elif loss_type == "competing_risks":
        competing_risk_states = kwargs.get("competing_risk_states", [])
        return CompetingRisksContinuousLoss(competing_risk_states)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")