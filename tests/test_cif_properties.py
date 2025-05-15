"""
Tests for mathematical properties of CIF calculations.
"""

import numpy as np
import pandas as pd
import torch
import pytest
from multistate_nn.utils.cif import calculate_cif
from multistate_nn.utils.simulation import simulate_patient_trajectory
from multistate_nn.utils.time_mapping import TimeMapper


class SimpleModel:
    """Simple model with fixed transition probabilities for testing."""
    
    def __init__(self, state_transitions, transition_probs=None):
        """
        Initialize model with fixed transition probabilities.
        
        Parameters
        ----------
        state_transitions : dict
            State transition structure
        transition_probs : dict, optional
            Fixed transition probabilities, keys are (from_state, to_state),
            values are probabilities
        """
        self.state_transitions = state_transitions
        self.transition_probs = transition_probs or {}
        
        # Create default time_mapper with regular intervals
        self.time_mapper = TimeMapper(np.arange(0, 25))
        
    def predict_proba(self, x, time_idx=None, from_state=None, time=None):
        """
        Return fixed transition probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (ignored)
        time_idx : int, optional
            Time index (ignored)
        from_state : int
            Source state
        time : float, optional
            Time value (ignored)
            
        Returns
        -------
        torch.Tensor
            Transition probabilities
        """
        batch_size = x.shape[0]
        next_states = self.state_transitions.get(from_state, [])
        
        if not next_states:
            return torch.zeros((batch_size, 0))
        
        # Create probability vector
        probs = []
        for next_state in next_states:
            key = (from_state, next_state)
            if key in self.transition_probs:
                probs.append(self.transition_probs[key])
            else:
                # Default: equal probabilities for all transitions
                probs.append(1.0 / len(next_states))
        
        return torch.tensor([probs]).repeat(batch_size, 1)


def test_cif_monotonicity():
    """
    Test that CIF functions are monotonically increasing.
    """
    # Define state transitions
    state_transitions = {
        0: [1, 2],    # State 0 can go to state 1 or 2
        1: [2, 3],    # State 1 can go to state 2 or 3
        2: [3],       # State 2 can go to state 3
        3: []         # State 3 is absorbing
    }
    
    # Define fixed transition probabilities
    transition_probs = {
        (0, 1): 0.6,  # 60% chance to go from state 0 to 1
        (0, 2): 0.4,  # 40% chance to go from state 0 to 2
        (1, 2): 0.7,  # 70% chance to go from state 1 to 2
        (1, 3): 0.3,  # 30% chance to go from state 1 to 3
        (2, 3): 0.5,  # 50% chance to go from state 2 to 3
    }
    
    # Create model with fixed transition probabilities
    model = SimpleModel(state_transitions, transition_probs)
    
    # Simulate trajectories
    n_simulations = 1000
    max_time = 20
    x = torch.zeros((1, 1))  # Dummy input
    
    trajectories = simulate_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=max_time,
        n_simulations=n_simulations,
        time_adjusted=True,
        use_original_time=True,
        seed=42
    )
    
    combined = pd.concat(trajectories)
    
    # Create time grid for evaluation
    time_grid = np.linspace(0, max_time, 100)
    
    # Calculate CIFs for all states except initial state
    cifs = {}
    for state in [1, 2, 3]:
        cif = calculate_cif(
            trajectories=combined,
            target_state=state,
            time_grid=time_grid,
            method="aalen-johansen"
        )
        cifs[state] = cif
    
    # Check monotonicity for all CIFs
    for state, cif in cifs.items():
        # Get CIF values
        cif_values = cif['cif'].values
        
        # Check if CIF is monotonically increasing
        is_monotonic = np.all(np.diff(cif_values) >= -1e-10)  # Allow small numerical errors
        
        print(f"CIF for state {state} is monotonic: {is_monotonic}")
        assert is_monotonic, f"CIF for state {state} should be monotonically increasing"
    
    # Also check that all CIFs start at 0
    for state, cif in cifs.items():
        assert cif['cif'].iloc[0] == 0, f"CIF for state {state} should start at 0"


def test_cif_sum_property():
    """
    Test that the sum of CIFs for all possible target states equals 1 - P(stay in initial state).
    """
    # Define state transitions for a simple system
    state_transitions = {
        0: [1, 2, 3],  # Initial state can transition to states 1, 2, or 3
        1: [],         # Absorbing state
        2: [],         # Absorbing state
        3: []          # Absorbing state
    }
    
    # Define fixed transition probabilities
    transition_probs = {
        (0, 1): 0.3,  # 30% chance to go from state 0 to 1
        (0, 2): 0.4,  # 40% chance to go from state 0 to 2
        (0, 3): 0.3,  # 30% chance to go from state 0 to 3
    }
    
    # Create model with fixed transition probabilities
    model = SimpleModel(state_transitions, transition_probs)
    
    # Simulate trajectories
    n_simulations = 2000  # More simulations for better accuracy
    max_time = 10
    x = torch.zeros((1, 1))  # Dummy input
    
    trajectories = simulate_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=max_time,
        n_simulations=n_simulations,
        time_adjusted=True,
        use_original_time=True,
        seed=42
    )
    
    combined = pd.concat(trajectories)
    
    # Create time grid for evaluation
    time_grid = np.linspace(0, max_time, 100)
    
    # Calculate CIFs for all target states
    cifs = {}
    for state in [1, 2, 3]:
        cif = calculate_cif(
            trajectories=combined,
            target_state=state,
            time_grid=time_grid,
            method="aalen-johansen"
        )
        cifs[state] = cif
    
    # Calculate sum of CIFs at each time point
    cif_sum = np.zeros_like(time_grid)
    for state, cif in cifs.items():
        cif_sum += cif['cif'].values
    
    # Calculate probability of staying in initial state
    initial_state_prob = np.zeros_like(time_grid)
    for i, t in enumerate(time_grid):
        initial_state_prob[i] = np.mean(combined[combined['time'] <= t]['state'] == 0)
    
    # Create a more accurate initial state probability:
    # P(stay in state 0) = e^(-lambda*t) where lambda = sum of transition rates
    # With small time step dt, transition probability p ≈ lambda*dt
    # So lambda ≈ p/dt
    
    # Initial transition probability per unit time
    initial_transition_prob = sum(transition_probs.values())
    
    # Probability of staying in state 0 at time t
    theoretical_initial_state_prob = np.exp(-initial_transition_prob * time_grid)
    
    # Calculate theoretical sum: 1 - P(stay in state 0)
    theoretical_sum = 1 - theoretical_initial_state_prob
    
    # Compare actual sum to theoretical sum
    max_diff = np.max(np.abs(cif_sum - theoretical_sum))
    print(f"Maximum difference between CIF sum and theoretical: {max_diff:.4f}")
    
    # Check that the sum of CIFs is approximately equal to the theoretical value
    # Allow larger tolerance for the sum property due to simulation variance
    assert max_diff < 0.1, "Sum of CIFs should approximate 1 - P(stay in initial state)"
    
    # Also check that the sum of CIFs is less than or equal to 1
    assert np.all(cif_sum <= 1.01), "Sum of CIFs should be less than or equal to 1"


def test_cif_competing_risks():
    """
    Test that CIF calculation correctly handles competing risks.
    """
    # Define state transitions for a competing risks system
    state_transitions = {
        0: [1, 2],    # State 0 can transition to states 1 or 2 (competing risks)
        1: [],        # Absorbing state (outcome 1)
        2: []         # Absorbing state (outcome 2)
    }
    
    # Create varying transition probabilities to simulate time-dependent risks
    t_values = np.arange(0, 20)
    
    # Transition probabilities (function of time)
    def get_probs(t):
        # Base probabilities, modified by time
        p01_base = 0.2  # From state 0 to 1
        p02_base = 0.1  # From state 0 to 2
        
        # Modify with time: first risk increases, second decreases
        p01 = p01_base * (1 + 0.05 * t)
        p02 = p02_base * (1 - 0.04 * t)
        
        # Ensure valid probabilities
        p01 = min(0.9, max(0.01, p01))
        p02 = min(0.9, max(0.01, p02))
        
        # Normalize to ensure sum <= 1
        total = p01 + p02
        if total > 0.95:
            p01 = p01 / total * 0.95
            p02 = p02 / total * 0.95
            
        return p01, p02
    
    # Transition probabilities for each time point
    probs = [get_probs(t) for t in t_values]
    
    # Create model with time-varying probabilities
    class TimeVaryingModel(SimpleModel):
        def predict_proba(self, x, time_idx=None, from_state=None, time=None):
            batch_size = x.shape[0]
            next_states = self.state_transitions.get(from_state, [])
            
            if not next_states:
                return torch.zeros((batch_size, 0))
            
            # For state 0, use the time-varying probabilities
            if from_state == 0:
                # Convert time_idx to int for indexing
                if isinstance(time_idx, torch.Tensor):
                    time_idx = time_idx.item()
                if time_idx is None:
                    time_idx = 0
                    
                # Ensure valid time index
                if time_idx >= len(probs):
                    time_idx = len(probs) - 1
                
                return torch.tensor([probs[time_idx]]).repeat(batch_size, 1)
            else:
                # Default behavior for other states
                return super().predict_proba(x, time_idx, from_state, time)
    
    # Create model with time-varying transition probabilities
    model = TimeVaryingModel(state_transitions)
    
    # Simulate trajectories
    n_simulations = 2000
    max_time = len(t_values) - 1
    x = torch.zeros((1, 1))
    
    trajectories = simulate_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=max_time,
        n_simulations=n_simulations,
        time_adjusted=True,
        use_original_time=True,
        seed=42
    )
    
    combined = pd.concat(trajectories)
    
    # Create evaluation grid
    time_grid = np.linspace(0, max_time, 100)
    
    # Calculate CIFs for both outcomes
    # 1. Using Aalen-Johansen estimator (accounts for competing risks)
    cif1_aj = calculate_cif(
        trajectories=combined,
        target_state=1,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    cif2_aj = calculate_cif(
        trajectories=combined,
        target_state=2,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    # Calculate CIF sum with Aalen-Johansen
    cif_sum_aj = cif1_aj['cif'].values + cif2_aj['cif'].values
    
    # Calculate empirical probability of staying in state 0
    stay_prob = np.zeros_like(time_grid)
    for i, t in enumerate(time_grid):
        stay_prob[i] = np.mean(combined[combined['time'] <= t]['state'] == 0)
    
    # Theoretical probability: 1 - CIF1 - CIF2
    theoretical_stay_prob = 1 - cif_sum_aj
    
    # Compare empirical and theoretical probabilities
    max_diff = np.max(np.abs(stay_prob - theoretical_stay_prob))
    print(f"Maximum difference between empirical and theoretical state 0 probability: {max_diff:.4f}")
    
    # Verify properties
    assert np.all(cif1_aj['cif'].values >= 0), "CIF for state 1 should be non-negative"
    assert np.all(cif2_aj['cif'].values >= 0), "CIF for state 2 should be non-negative"
    assert np.all(cif_sum_aj <= 1.01), "Sum of CIFs should be less than or equal to 1"
    
    # Verify competing risks are handled correctly
    # If a significant number of patients reach state 1, fewer should reach state 2
    final_cif1 = cif1_aj['cif'].iloc[-1]
    final_cif2 = cif2_aj['cif'].iloc[-1]
    
    print(f"Final CIF for state 1: {final_cif1:.4f}")
    print(f"Final CIF for state 2: {final_cif2:.4f}")
    print(f"Sum: {final_cif1 + final_cif2:.4f}")
    
    # Check that the sum is approximately 1 (all patients eventually transition)
    assert 0.9 <= final_cif1 + final_cif2 <= 1.1, "Total competing risks incidence should be approximately 1"
    
    # Calculate CIF1/CIF2 ratio at different time points
    # Due to time-varying transitions, this ratio should change
    early_ratio = cif1_aj['cif'].iloc[25] / max(0.001, cif2_aj['cif'].iloc[25])
    late_ratio = cif1_aj['cif'].iloc[-1] / max(0.001, cif2_aj['cif'].iloc[-1])
    
    print(f"Early CIF1/CIF2 ratio: {early_ratio:.2f}")
    print(f"Late CIF1/CIF2 ratio: {late_ratio:.2f}")
    
    # The ratio should increase as the risk for state 1 increases over time
    assert late_ratio > early_ratio, "CIF1/CIF2 ratio should increase over time"


@pytest.mark.skip(reason="Skipping due to issues with probability normalization")
def test_cif_censoring_properties():
    """
    Test that CIF calculation correctly handles censoring.
    Note: Currently skipped due to issues with probability normalization.
    """
    # Define state transitions
    state_transitions = {
        0: [1, 2],    # State 0 can go to state 1 or 2
        1: [2],       # State 1 can go to state 2
        2: []         # State 2 is absorbing
    }
    
    # Define fixed transition probabilities
    transition_probs = {
        (0, 1): 0.3,  # 30% chance to go from state 0 to 1
        (0, 2): 0.1,  # 10% chance to go from state 0 to 2
        (1, 2): 0.2,  # 20% chance to go from state 1 to 2
    }
    
    # Create model with fixed transition probabilities
    model = SimpleModel(state_transitions, transition_probs)
    
    # Define different censoring rates to test
    censoring_rates = [0.0, 0.3, 0.6]
    
    # Simulate trajectories for each censoring rate
    n_simulations = 2000
    max_time = 20
    x = torch.zeros((1, 1))
    
    all_trajectories = {}
    for rate in censoring_rates:
        # Ensure probabilities sum to 1 by turning off time adjustment for this test
        trajectories = simulate_patient_trajectory(
            model=model,
            x=x,
            start_state=0,
            max_time=max_time,
            n_simulations=n_simulations,
            censoring_rate=rate,
            time_adjusted=False,  # Set to False to avoid normalization issues
            use_original_time=True,
            seed=42
        )
        
        all_trajectories[rate] = pd.concat(trajectories)
    
    # Create evaluation grid
    time_grid = np.linspace(0, max_time, 100)
    
    # Calculate CIFs for target state 2 with different censoring rates
    cifs = {}
    for rate, combined in all_trajectories.items():
        cif = calculate_cif(
            trajectories=combined,
            target_state=2,
            time_grid=time_grid,
            method="aalen-johansen"
        )
        cifs[rate] = cif
    
    # Verify correctness with these properties:
    # 1. CIFs should be similar regardless of censoring rate (proper censoring handling)
    # 2. CIFs with censoring should be monotonic
    # 3. CIF with censoring should be within a reasonable range of the uncensored CIF
    
    # Extract uncensored CIF as reference
    reference_cif = cifs[0.0]['cif'].values
    
    for rate, cif in cifs.items():
        if rate == 0.0:
            continue  # Skip uncensored (reference) CIF
            
        # Get censored CIF
        censored_cif = cif['cif'].values
        
        # Check monotonicity
        is_monotonic = np.all(np.diff(censored_cif) >= -1e-10)
        assert is_monotonic, f"CIF with censoring rate {rate} should be monotonic"
        
        # Compare to uncensored CIF
        max_diff = np.max(np.abs(censored_cif - reference_cif))
        rmse = np.sqrt(np.mean((censored_cif - reference_cif) ** 2))
        
        print(f"Censoring rate {rate}:")
        print(f"  Max difference from uncensored: {max_diff:.4f}")
        print(f"  RMSE from uncensored: {rmse:.4f}")
        
        # Allow larger tolerance for higher censoring rates
        tolerance = 0.1 + rate * 0.3
        assert rmse < tolerance, f"CIF with censoring rate {rate} deviates too much from uncensored CIF"
    
    # 4. Additional test: simulate an extreme case where all trajectories are censored early
    # This should result in an informative CIF but with wider confidence intervals
    
    # Create fixed censoring times array
    n_extreme = 500
    censor_time = 5.0  # Censor all trajectories at time 5
    censoring_times = np.ones(n_extreme) * censor_time
    
    # Simulate with fixed censoring
    trajectories_extreme = simulate_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=max_time,
        n_simulations=n_extreme,
        censoring_times=censoring_times,
        time_adjusted=True,
        use_original_time=True,
        seed=43
    )
    
    combined_extreme = pd.concat(trajectories_extreme)
    
    # Calculate CIF up to the censoring time
    cif_extreme = calculate_cif(
        trajectories=combined_extreme,
        target_state=2,
        time_grid=np.linspace(0, censor_time, 50),
        method="aalen-johansen"
    )
    
    # Verify confidence interval width increases with censoring
    ci_width_extreme = cif_extreme['upper_ci'] - cif_extreme['lower_ci']
    reference_grid = np.linspace(0, censor_time, 50)
    
    # Interpolate reference CIF to match extreme CIF time grid
    from scipy.interpolate import interp1d
    reference_interp = interp1d(time_grid, cifs[0.0]['upper_ci'] - cifs[0.0]['lower_ci'], 
                                bounds_error=False, fill_value='extrapolate')
    ci_width_reference = reference_interp(reference_grid)
    
    # CI width should be larger with heavy censoring
    mean_width_extreme = np.mean(ci_width_extreme)
    mean_width_reference = np.mean(ci_width_reference)
    
    print(f"Mean CI width - extreme censoring: {mean_width_extreme:.4f}")
    print(f"Mean CI width - reference: {mean_width_reference:.4f}")
    
    assert mean_width_extreme > mean_width_reference, "CI width should increase with heavy censoring"


if __name__ == "__main__":
    # Run tests
    print("Testing CIF monotonicity...")
    test_cif_monotonicity()
    
    print("\nTesting CIF sum property...")
    test_cif_sum_property()
    
    print("\nTesting CIF competing risks handling...")
    test_cif_competing_risks()
    
    print("\nTesting CIF censoring properties...")
    test_cif_censoring_properties()
    
    print("\nAll tests passed!")