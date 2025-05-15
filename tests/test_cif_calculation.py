"""
Test suite for verifying CIF calculation functionality.
This focuses specifically on debugging the issue of flat zero CIFs.
"""

import numpy as np
import pandas as pd
import torch
import pytest
from multistate_nn.utils import calculate_cif, simulate_patient_trajectory
from multistate_nn import MultiStateNN, ModelConfig, TrainConfig, fit


def test_basic_cif_from_known_trajectories():
    """
    Test CIF calculation with manually constructed trajectories
    where we know the expected output.
    """
    # Create a simple deterministic set of trajectories
    # All patients start in state 0 and eventually reach state 1
    trajectories = []
    n_patients = 100
    
    for i in range(n_patients):
        # Patient transitions at time i % 10 + 1
        transition_time = i % 10 + 1
        
        # Create before transition
        before_df = pd.DataFrame({
            'time': np.arange(transition_time),
            'state': np.zeros(transition_time),
            'simulation': i
        })
        
        # Create after transition
        after_df = pd.DataFrame({
            'time': np.arange(transition_time, 15),
            'state': np.ones(15 - transition_time),
            'simulation': i
        })
        
        # Combine
        traj = pd.concat([before_df, after_df])
        trajectories.append(traj)
    
    # Convert to format expected by calculate_cif
    all_trajectories = pd.concat(trajectories)
    
    # Calculate CIF for target state 1
    cif = calculate_cif(
        trajectories=all_trajectories,
        target_state=1,
        max_time=15,
        method="aalen-johansen"
    )
    
    # Check resulting CIF
    print("CIF shape:", cif.shape)
    print("CIF head:", cif.head())
    print("CIF tail:", cif.tail())
    
    # Verify CIF increases over time as expected
    # At time 1, 10% of patients have transitioned
    # At time 5, 50% of patients have transitioned
    # At time 10, 100% of patients have transitioned
    assert cif.loc[cif['time'] >= 1, 'cif'].iloc[0] > 0, "CIF should be positive at time 1"
    assert cif.loc[cif['time'] >= 5, 'cif'].iloc[0] >= 0.5, "CIF should be at least 0.5 at time 5"
    assert cif.loc[cif['time'] >= 10, 'cif'].iloc[0] >= 0.99, "CIF should be approximately 1 at time 10"
    
    # Verify CIF increases monotonically
    assert np.all(np.diff(cif['cif']) >= 0), "CIF should never decrease"


def test_cif_with_time_adjusted_simulation():
    """
    Test CIF calculation with trajectories from time-adjusted simulation.
    """
    # Create a simple model with known transition probabilities
    # Simple 3-state model: 0 -> 1 -> 2
    state_transitions = {
        0: [1],    # State 0 can only go to state 1
        1: [2],    # State 1 can only go to state 2
        2: []      # State 2 is absorbing
    }
    
    # Define a very simple neural network that always outputs fixed probabilities
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Create a single parameter (never used) to satisfy PyTorch
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.state_transitions = state_transitions
        
        def forward(self, x, time=None):
            batch_size = x.shape[0]
            # Always return a fixed probability of 0.2 for all transitions
            # Shape: [batch_size, sum(len(transitions) for transitions in state_transitions.values())]
            return torch.ones(batch_size, 2) * 0.2
        
        def predict_proba(self, x, time_idx=None, from_state=None, time=None):
            batch_size = x.shape[0]
            if from_state == 0:
                # 100% probability of going to state 1
                return torch.ones(batch_size, 1)
            elif from_state == 1:
                # 100% probability of going to state 2
                return torch.ones(batch_size, 1)
            else:
                # No transitions from state 2 (absorbing)
                return torch.zeros(batch_size, 0)
    
    # Create model instance
    model = SimpleNN()
    
    # Dummy input for simulation
    x = torch.zeros((1, 3))  # One patient with 3 features
    
    # Simulate trajectories with time adjustment
    n_simulations = 100
    max_time = 10
    
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
    
    # Convert to format expected by calculate_cif
    all_trajectories = pd.concat(trajectories)
    
    # Calculate CIF for target state 1
    cif_state1 = calculate_cif(
        trajectories=all_trajectories,
        target_state=1,
        max_time=max_time,
        method="aalen-johansen"
    )
    
    # Check resulting CIF
    print("State 1 CIF shape:", cif_state1.shape)
    print("State 1 CIF head:", cif_state1.head())
    print("State 1 CIF tail:", cif_state1.tail())
    
    # Calculate CIF for target state 2
    cif_state2 = calculate_cif(
        trajectories=all_trajectories,
        target_state=2,
        max_time=max_time,
        method="aalen-johansen"
    )
    
    # Check resulting CIF
    print("State 2 CIF shape:", cif_state2.shape)
    print("State 2 CIF head:", cif_state2.head())
    print("State 2 CIF tail:", cif_state2.tail())
    
    # Verify CIFs are not flat at zero
    assert not np.all(cif_state1['cif'] == 0), "CIF for state 1 should not be all zeros"
    assert not np.all(cif_state2['cif'] == 0), "CIF for state 2 should not be all zeros"
    
    # Since we have deterministic transitions, check timing
    # With time_adjusted=True, patients should progress to state 1 after 1 time unit and to state 2 after 2 time units
    # But since patients might exit state 1 as soon as they enter it, we need to check for at least some non-zero value
    assert cif_state1.loc[cif_state1['time'] >= 1, 'cif'].iloc[0] > 0, "CIF for state 1 should be positive after time 1"
    assert cif_state2.loc[cif_state2['time'] >= 2, 'cif'].iloc[0] > 0, "CIF for state 2 should be positive after time 2"


def test_full_pipeline_with_synthetic_data():
    """
    Test the full pipeline from model training to CIF calculation with synthetic data.
    This test is closer to how the examples use the library.
    """
    # Create synthetic data
    # Use a simple 3-state model: 0 -> 1 -> 2
    state_transitions = {
        0: [1],    # State 0 can only go to state 1
        1: [2],    # State 1 can only go to state 2
        2: []      # State 2 is absorbing
    }
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 100
    
    # Feature: age between 20-80
    age = np.random.uniform(20, 80, n_samples)
    
    # Create transitions 0->1 with time based on age (older transitions faster)
    time_to_state1 = 10 - (age - 20) * 8 / 60  # Range: 2-10
    
    # Create DataFrame for model training
    train_data = []
    for i in range(n_samples):
        # Transition from state 0 to state 1
        train_data.append({
            'patient': i,
            'time': 0,
            'from_state': 0,
            'to_state': 1,
            'age': age[i],
            'time_diff': time_to_state1[i]
        })
        
        # Transition from state 1 to state 2
        train_data.append({
            'patient': i,
            'time': 1,
            'from_state': 1,
            'to_state': 2,
            'age': age[i],
            'time_diff': 5.0  # Fixed time for simplicity
        })
    
    # Create DataFrame
    df = pd.DataFrame(train_data)
    
    # Create and train model
    model_config = ModelConfig(
        input_dim=1,  # Just age
        hidden_dims=[16, 8],
        num_states=3,
        state_transitions=state_transitions
    )
    
    train_config = TrainConfig(
        epochs=50,
        batch_size=32,
        learning_rate=0.01
    )
    
    # Train model
    model = fit(
        df=df,
        covariates=['age'],
        model_config=model_config,
        train_config=train_config
    )
    
    # Test young and old patients
    young_patient = torch.tensor([[30.0]])
    old_patient = torch.tensor([[70.0]])
    
    # Simulate trajectories for young patient (with time adjustment)
    trajectories_young = simulate_patient_trajectory(
        model=model,
        x=young_patient,
        start_state=0,
        max_time=20,
        n_simulations=100,
        time_adjusted=True,
        use_original_time=True,
        seed=42
    )
    
    # Simulate trajectories for old patient (with time adjustment)
    trajectories_old = simulate_patient_trajectory(
        model=model,
        x=old_patient,
        start_state=0,
        max_time=20,
        n_simulations=100,
        time_adjusted=True,
        use_original_time=True,
        seed=43
    )
    
    # Create time grid for consistent CIF evaluation
    time_grid = np.linspace(0, 20, 100)
    
    # Calculate CIFs for state 1 with time grid
    young_cif_state1 = calculate_cif(
        trajectories=pd.concat(trajectories_young),
        target_state=1,
        max_time=20,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    old_cif_state1 = calculate_cif(
        trajectories=pd.concat(trajectories_old),
        target_state=1,
        max_time=20,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    # Calculate CIFs for state 2 with time grid
    young_cif_state2 = calculate_cif(
        trajectories=pd.concat(trajectories_young),
        target_state=2,
        max_time=20,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    old_cif_state2 = calculate_cif(
        trajectories=pd.concat(trajectories_old),
        target_state=2,
        max_time=20,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    # Print CIF statistics
    print("Young patient - State 1 CIF:", young_cif_state1['cif'].describe())
    print("Old patient - State 1 CIF:", old_cif_state1['cif'].describe())
    print("Young patient - State 2 CIF:", young_cif_state2['cif'].describe())
    print("Old patient - State 2 CIF:", old_cif_state2['cif'].describe())
    
    # Verify CIFs are not flat at zero
    assert not np.all(young_cif_state1['cif'] == 0), "CIF for young patient and state 1 should not be all zeros"
    assert not np.all(old_cif_state1['cif'] == 0), "CIF for old patient and state 1 should not be all zeros"
    assert not np.all(young_cif_state2['cif'] == 0), "CIF for young patient and state 2 should not be all zeros"
    assert not np.all(old_cif_state2['cif'] == 0), "CIF for old patient and state 2 should not be all zeros"
    
    # Verify old patients progress faster than young patients
    # Get time to reach 50% probability
    def time_to_reach(cif, prob=0.5):
        times = cif.loc[cif['cif'] >= prob, 'time']
        return times.iloc[0] if not times.empty else float('inf')
    
    young_t50_state1 = time_to_reach(young_cif_state1, 0.5)
    old_t50_state1 = time_to_reach(old_cif_state1, 0.5)
    young_t50_state2 = time_to_reach(young_cif_state2, 0.5)
    old_t50_state2 = time_to_reach(old_cif_state2, 0.5)
    
    print(f"Time to 50% probability for state 1: Young = {young_t50_state1}, Old = {old_t50_state1}")
    print(f"Time to 50% probability for state 2: Young = {young_t50_state2}, Old = {old_t50_state2}")
    
    # Assert that old patients progress faster (if CIFs reach 50%)
    if old_t50_state1 != float('inf') and young_t50_state1 != float('inf'):
        assert old_t50_state1 < young_t50_state1, "Old patients should progress to state 1 faster"
    
    if old_t50_state2 != float('inf') and young_t50_state2 != float('inf'):
        assert old_t50_state2 < young_t50_state2, "Old patients should progress to state 2 faster"


def test_debug_time_adjusted_simulation():
    """
    Detailed debug test to trace through time-adjusted simulation and understand
    why the trajectories might be producing flat CIFs.
    """
    # Create a simple fixed-probability model
    state_transitions = {
        0: [1, 2],  # State 0 can go to state 1 or 2
        1: [2],     # State 1 can go to state 2
        2: []       # State 2 is absorbing
    }
    
    class DebugModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.state_transitions = state_transitions
        
        def forward(self, x, time=None):
            batch_size = x.shape[0]
            # Not used, but needed for model definition
            return torch.ones(batch_size, 3) * 0.3
        
        def predict_proba(self, x, time_idx=None, from_state=None, time=None):
            batch_size = x.shape[0]
            if from_state == 0:
                # 80% to state 1, 20% to state 2
                return torch.tensor([[0.8, 0.2]]).repeat(batch_size, 1)
            elif from_state == 1:
                # 100% to state 2
                return torch.ones(batch_size, 1)
            else:
                return torch.zeros(batch_size, 0)
    
    # Create model instance
    model = DebugModel()
    
    # Output fixed probabilities for debugging
    x = torch.zeros((1, 1))
    print("State 0 transitions:", model.predict_proba(x, from_state=0))
    print("State 1 transitions:", model.predict_proba(x, from_state=1))
    
    # Simulate with and without time adjustment
    trajectories_no_adj = simulate_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=10,
        n_simulations=100,
        time_adjusted=False,
        seed=42
    )
    
    trajectories_adj = simulate_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=10,
        n_simulations=100,
        time_adjusted=True,
        seed=42
    )
    
    # Inspect the trajectories in detail
    print("\nFirst trajectory without time adjustment:")
    print(trajectories_no_adj[0])
    
    print("\nFirst trajectory with time adjustment:")
    print(trajectories_adj[0])
    
    # Check for state 1 occurrences in trajectories
    def state_count(trajectories, state):
        counts = []
        for traj in trajectories:
            count = sum(traj['state'] == state)
            counts.append(count)
        return counts
    
    state1_counts_no_adj = state_count(trajectories_no_adj, 1)
    state1_counts_adj = state_count(trajectories_adj, 1)
    
    print(f"\nState 1 occurrences without time adjustment: "
          f"Mean = {np.mean(state1_counts_no_adj):.2f}, "
          f"Max = {np.max(state1_counts_no_adj)}, "
          f"Sum = {np.sum(state1_counts_no_adj)}")
    
    print(f"State 1 occurrences with time adjustment: "
          f"Mean = {np.mean(state1_counts_adj):.2f}, "
          f"Max = {np.max(state1_counts_adj)}, "
          f"Sum = {np.sum(state1_counts_adj)}")
    
    # Calculate CIFs
    cif_state1_no_adj = calculate_cif(
        trajectories=pd.concat(trajectories_no_adj),
        target_state=1,
        max_time=10,
        method="aalen-johansen"
    )
    
    cif_state1_adj = calculate_cif(
        trajectories=pd.concat(trajectories_adj),
        target_state=1,
        max_time=10,
        method="aalen-johansen"
    )
    
    # Print CIF values
    print("\nCIF values for state 1 without time adjustment:")
    print(cif_state1_no_adj)
    
    print("\nCIF values for state 1 with time adjustment:")
    print(cif_state1_adj)
    
    # Check if CIFs are flat at zero
    is_flat_no_adj = np.all(cif_state1_no_adj['cif'] == 0)
    is_flat_adj = np.all(cif_state1_adj['cif'] == 0)
    
    print(f"CIF is flat at zero without time adjustment: {is_flat_no_adj}")
    print(f"CIF is flat at zero with time adjustment: {is_flat_adj}")
    
    # Basic tests
    assert not is_flat_no_adj, "CIF without time adjustment should not be flat at zero"
    assert not is_flat_adj, "CIF with time adjustment should not be flat at zero"


def test_trajectory_conversion_for_cif():
    """
    Test how trajectories are prepared for CIF calculation,
    specifically focusing on the data structure and how calculate_cif
    processes the trajectories DataFrame.
    """
    # Create a simple trajectory dataset manually
    # 10 patients, 5 of whom transition to state 1 at different times
    trajectories = []
    n_patients = 10
    
    for i in range(n_patients):
        # Time ranges from 0 to 9
        times = np.arange(10)
        
        # For half the patients, transition at time i, others stay in state 0
        states = np.zeros(10)
        if i < 5:
            states[i+1:] = 1
        
        # Create DataFrame for this trajectory
        trajectory_df = pd.DataFrame({
            'time': times,
            'state': states,
            'simulation': i
        })
        
        trajectories.append(trajectory_df)
    
    # Now we have 10 trajectories, 5 of which include a transition to state 1
    # Inspect them for sanity check
    print("Trajectories generated:", len(trajectories))
    print("Sample trajectory:")
    print(trajectories[2])  # Patient who transitions at time 2
    
    # Combine into one DataFrame as calculate_cif expects
    all_trajectories = pd.concat(trajectories)
    
    # Try calculating CIF with and without time grid
    # Without time grid
    cif_no_grid = calculate_cif(
        trajectories=all_trajectories,
        target_state=1,
        max_time=10,
        method="aalen-johansen"
    )
    
    # With time grid
    time_grid = np.linspace(0, 9, 100)
    cif_with_grid = calculate_cif(
        trajectories=all_trajectories,
        target_state=1,
        max_time=10,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    print("\nCIF without time grid:")
    print(cif_no_grid.head())
    print(cif_no_grid.tail())
    
    print("\nCIF with time grid:")
    print(cif_with_grid.head())
    print(cif_with_grid.tail())
    
    # Check final values - should be 0.5 since half the patients transition to state 1
    print(f"Final CIF value without grid: {cif_no_grid['cif'].iloc[-1]}")
    print(f"Final CIF value with grid: {cif_with_grid['cif'].iloc[-1]}")
    
    # Assert CIFs aren't flat at zero
    assert not np.all(cif_no_grid['cif'] == 0), "CIF without grid should not be flat at zero"
    assert not np.all(cif_with_grid['cif'] == 0), "CIF with grid should not be flat at zero"
    
    # Assert final value is approximately 0.5
    assert 0.45 <= cif_no_grid['cif'].iloc[-1] <= 0.55, "Final CIF value should be approximately 0.5"
    assert 0.45 <= cif_with_grid['cif'].iloc[-1] <= 0.55, "Final CIF value should be approximately 0.5"


def test_examine_aalen_johansen_implementation():
    """
    Test specifically focused on the Aalen-Johansen implementation
    to check if there are any issues with its logic.
    """
    # First, we'll create an even simpler set of trajectories to test with
    # One patient, clear transition from state 0 to state 1 at time 5
    simple_trajectory = pd.DataFrame({
        'time': np.arange(10),
        'state': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        'simulation': 0
    })
    
    # Try with and without time grid, all combinations of parameters
    # The most basic calculation without any optional parameters
    cif_basic = calculate_cif(
        trajectories=simple_trajectory,
        target_state=1,
        max_time=10,
        method="aalen-johansen"
    )
    
    print("Basic CIF calculation:")
    print(cif_basic)
    
    # With time grid
    time_grid = np.linspace(0, 9, 20)
    cif_with_grid = calculate_cif(
        trajectories=simple_trajectory,
        target_state=1,
        max_time=10,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    print("\nCIF with time grid:")
    print(cif_with_grid)
    
    # With landmark time
    cif_with_landmark = calculate_cif(
        trajectories=simple_trajectory,
        target_state=1,
        max_time=10,
        landmark_time=2,
        method="aalen-johansen"
    )
    
    print("\nCIF with landmark time:")
    print(cif_with_landmark)
    
    # With use_original_time
    cif_with_original_time = calculate_cif(
        trajectories=simple_trajectory,
        target_state=1,
        max_time=10,
        use_original_time=True,
        method="aalen-johansen"
    )
    
    print("\nCIF with use_original_time:")
    print(cif_with_original_time)
    
    # No time adjustment (just emission probabilities)
    cif_empirical = calculate_cif(
        trajectories=simple_trajectory,
        target_state=1,
        max_time=10,
        method="empirical"
    )
    
    print("\nEmpirical CIF (no Aalen-Johansen):")
    print(cif_empirical)
    
    # Basic assertions for all variants
    for name, cif in [
        ("basic", cif_basic),
        ("with grid", cif_with_grid),
        ("with landmark", cif_with_landmark),
        ("with original time", cif_with_original_time),
        ("empirical", cif_empirical)
    ]:
        assert not cif.empty, f"CIF {name} should not be empty"
        assert not np.all(cif['cif'] == 0), f"CIF {name} should not be flat at zero"
        
        # For the single trajectory case, the final CIF should be 1
        # (except for landmark, which starts at a later time)
        if name != "with landmark":
            assert cif['cif'].iloc[-1] >= 0.99, f"Final CIF value for {name} should be 1"


def test_direct_plotting_trajectory():
    """
    Test the most basic case by directly plotting a
    trajectory DataFrame to see if it visualizes as expected.
    """
    # Create a very simple trajectory for one patient
    times = np.arange(10)
    states = np.zeros(10)
    states[5:] = 1  # Transition to state 1 at time 5
    
    trajectory = pd.DataFrame({
        'time': times,
        'state': states,
        'simulation': 0
    })
    
    print("Simple trajectory for visualization:")
    print(trajectory)
    
    # Calculate CIF
    cif = calculate_cif(
        trajectories=trajectory,
        target_state=1,
        max_time=10,
        method="aalen-johansen"
    )
    
    print("\nCIF data for plotting:")
    print(cif)
    
    # Assert the CIF is not flat at zero
    assert not np.all(cif['cif'] == 0), "CIF should not be flat at zero"
    assert cif['cif'].iloc[-1] >= 0.99, "Final CIF value should be approximately 1"


if __name__ == "__main__":
    # Run the most basic test first
    print("Testing direct plotting trajectory...")
    test_direct_plotting_trajectory()
    
    print("\nTesting basic CIF from known trajectories...")
    test_basic_cif_from_known_trajectories()
    
    print("\nTesting examine Aalen-Johansen implementation...")
    test_examine_aalen_johansen_implementation()
    
    print("\nTesting trajectory conversion for CIF...")
    test_trajectory_conversion_for_cif()
    
    print("\nTesting time-adjusted simulation...")
    test_debug_time_adjusted_simulation()
    
    print("\nTesting CIF with time-adjusted simulation...")
    test_cif_with_time_adjusted_simulation()
    
    print("\nTesting full pipeline with synthetic data...")
    test_full_pipeline_with_synthetic_data()
    
    print("\nAll tests passed!")