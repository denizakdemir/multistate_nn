"""
Minimal example script to demonstrate and debug flat CIF issues.
This script isolates the key components - simulate and CIF calculation - 
in a minimal reproducible example.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from multistate_nn.utils import calculate_cif, simulate_patient_trajectory

# Import the TimeMapper class
from multistate_nn.utils.time_mapping import TimeMapper

# Create a minimal class that emulates a MultiStateNN model
class MinimalModel:
    def __init__(self):
        # Define a simple 3-state model: 0 -> 1 -> 2
        self.state_transitions = {
            0: [1],    # State 0 can only transition to state 1
            1: [2],    # State 1 can only transition to state 2
            2: []      # State 2 is absorbing
        }
        
        # Create a proper TimeMapper object for time-adjusted simulation
        # Initialize with time points 0-10
        self.time_mapper = TimeMapper(np.arange(0, 11))
    
    def predict_proba(self, x, time_idx=None, from_state=None, time=None):
        batch_size = x.shape[0]
        if from_state == 0:
            # 100% probability of going to state 1
            return torch.ones(batch_size, 1)
        elif from_state == 1:
            # 100% probability of going to state 2
            return torch.ones(batch_size, 1)
        else:
            # State 2 is absorbing - no transitions
            return torch.zeros(batch_size, 0)


def create_simple_trajectories():
    """
    Create a simple dataframe of trajectories manually
    without using the simulation function.
    """
    # Create 5 trajectories:
    # - All start in state 0
    # - Transition to state 1 at times 1, 2, 3, 4, 5 respectively
    # - Then transition to state 2 one time unit later
    trajectories = []
    
    for i in range(5):
        transition_time1 = i + 1
        transition_time2 = transition_time1 + 1
        max_time = 10
        
        # Before first transition (state 0)
        state0_times = np.arange(transition_time1)
        state0_states = np.zeros_like(state0_times)
        
        # Between transitions (state 1)
        state1_times = np.arange(transition_time1, transition_time2)
        state1_states = np.ones_like(state1_times)
        
        # After second transition (state 2)
        state2_times = np.arange(transition_time2, max_time)
        state2_states = np.full_like(state2_times, 2)
        
        # Combine all segments
        times = np.concatenate([state0_times, state1_times, state2_times])
        states = np.concatenate([state0_states, state1_states, state2_states])
        simulation = np.full_like(times, i)
        
        # Create dataframe
        traj_df = pd.DataFrame({
            'time': times,
            'state': states,
            'simulation': simulation
        })
        
        trajectories.append(traj_df)
    
    return trajectories


def main():
    # Let's test both approaches:
    # 1. Manually created trajectories
    # 2. Simulated trajectories using simulate_patient_trajectory
    
    # 1. Manually created trajectories
    manual_trajectories = create_simple_trajectories()
    print("Manually created trajectories:")
    print(manual_trajectories[0])  # Show the first trajectory
    
    # 2. Simulated trajectories
    model = MinimalModel()
    x = torch.zeros((1, 1))  # Dummy input
    
    # Simulate with and without time adjustment
    print("\nSimulating trajectories...")
    trajectories_no_adj = simulate_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=10,
        n_simulations=5,
        time_adjusted=False,
        seed=42
    )
    
    trajectories_adj = simulate_patient_trajectory(
        model=model,
        x=x,
        start_state=0,
        max_time=10,
        n_simulations=5,
        time_adjusted=True,
        seed=42
    )
    
    print("Simulated trajectory without time adjustment:")
    print(trajectories_no_adj[0])
    
    print("\nSimulated trajectory with time adjustment:")
    print(trajectories_adj[0])
    
    # Calculate CIF for state 1 for all three sets of trajectories
    all_manual = pd.concat(manual_trajectories)
    all_no_adj = pd.concat(trajectories_no_adj)
    all_adj = pd.concat(trajectories_adj)
    
    # Create time grid for consistent evaluation
    time_grid = np.linspace(0, 10, 100)
    
    # Calculate CIFs for state 1
    print("\nCalculating CIFs for state 1...")
    cif_manual = calculate_cif(
        trajectories=all_manual,
        target_state=1,
        max_time=10,
        time_grid=time_grid,
        method="empirical"  # Use empirical method instead of aalen-johansen
    )
    
    cif_no_adj = calculate_cif(
        trajectories=all_no_adj,
        target_state=1,
        max_time=10,
        time_grid=time_grid,
        method="empirical"  # Use empirical method instead of aalen-johansen
    )
    
    cif_adj = calculate_cif(
        trajectories=all_adj,
        target_state=1,
        max_time=10,
        time_grid=time_grid,
        method="empirical"  # Use empirical method instead of aalen-johansen
    )
    
    print("Manual trajectories CIF shape:", cif_manual.shape)
    print("Manual trajectories CIF non-zero values:", len(cif_manual[cif_manual['cif'] > 0]))
    
    print("No time adjust CIF shape:", cif_no_adj.shape)
    print("No time adjust CIF non-zero values:", len(cif_no_adj[cif_no_adj['cif'] > 0]))
    
    print("Time adjusted CIF shape:", cif_adj.shape)
    print("Time adjusted CIF non-zero values:", len(cif_adj[cif_adj['cif'] > 0]))
    
    # Plot all three CIFs for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(cif_manual['time'], cif_manual['cif'], 'g-', label='Manual trajectories')
    plt.plot(cif_no_adj['time'], cif_no_adj['cif'], 'b-', label='Simulated (no time adj)')
    plt.plot(cif_adj['time'], cif_adj['cif'], 'r-', label='Simulated (time adj)')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Incidence')
    plt.title('Comparison of CIF Calculation Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig("cif_comparison.png")
    print("Plot saved to cif_comparison.png")
    
    # Calculate CIFs for state 2 for comparison
    print("\nCalculating CIFs for state 2...")
    cif_manual_2 = calculate_cif(
        trajectories=all_manual,
        target_state=2,
        max_time=10,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    cif_no_adj_2 = calculate_cif(
        trajectories=all_no_adj,
        target_state=2,
        max_time=10,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    cif_adj_2 = calculate_cif(
        trajectories=all_adj,
        target_state=2,
        max_time=10,
        time_grid=time_grid,
        method="aalen-johansen"
    )
    
    print("Manual trajectories CIF shape:", cif_manual_2.shape)
    print("Manual trajectories CIF non-zero values:", len(cif_manual_2[cif_manual_2['cif'] > 0]))
    
    print("No time adjust CIF shape:", cif_no_adj_2.shape)
    print("No time adjust CIF non-zero values:", len(cif_no_adj_2[cif_no_adj_2['cif'] > 0]))
    
    print("Time adjusted CIF shape:", cif_adj_2.shape)
    print("Time adjusted CIF non-zero values:", len(cif_adj_2[cif_adj_2['cif'] > 0]))
    
    # Plot CIFs for state 2
    plt.figure(figsize=(10, 6))
    plt.plot(cif_manual_2['time'], cif_manual_2['cif'], 'g-', label='Manual trajectories')
    plt.plot(cif_no_adj_2['time'], cif_no_adj_2['cif'], 'b-', label='Simulated (no time adj)')
    plt.plot(cif_adj_2['time'], cif_adj_2['cif'], 'r-', label='Simulated (time adj)')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Incidence')
    plt.title('Comparison of CIF Calculation Methods (State 2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig("cif_comparison_state2.png")
    print("Plot saved to cif_comparison_state2.png")
    
    # Try different method for CIF calculation
    print("\nTrying empirical method for CIF calculation...")
    cif_manual_emp = calculate_cif(
        trajectories=all_manual,
        target_state=1,
        max_time=10,
        time_grid=time_grid,
        method="empirical"
    )
    
    cif_no_adj_emp = calculate_cif(
        trajectories=all_no_adj,
        target_state=1,
        max_time=10,
        time_grid=time_grid,
        method="empirical"
    )
    
    cif_adj_emp = calculate_cif(
        trajectories=all_adj,
        target_state=1,
        max_time=10,
        time_grid=time_grid,
        method="empirical"
    )
    
    # Plot empirical CIFs
    plt.figure(figsize=(10, 6))
    plt.plot(cif_manual_emp['time'], cif_manual_emp['cif'], 'g-', label='Manual trajectories')
    plt.plot(cif_no_adj_emp['time'], cif_no_adj_emp['cif'], 'b-', label='Simulated (no time adj)')
    plt.plot(cif_adj_emp['time'], cif_adj_emp['cif'], 'r-', label='Simulated (time adj)')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Incidence')
    plt.title('Comparison of Empirical CIF Calculation Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig("cif_comparison_empirical.png")
    print("Plot saved to cif_comparison_empirical.png")
    
    # Check if CIFs are flat at zero
    print("\nChecking if CIFs are flat at zero:")
    print("Manual trajectories (AJ):", np.all(cif_manual['cif'] == 0))
    print("No time adjust (AJ):", np.all(cif_no_adj['cif'] == 0))
    print("Time adjusted (AJ):", np.all(cif_adj['cif'] == 0))
    print("Manual trajectories (empirical):", np.all(cif_manual_emp['cif'] == 0))
    print("No time adjust (empirical):", np.all(cif_no_adj_emp['cif'] == 0))
    print("Time adjusted (empirical):", np.all(cif_adj_emp['cif'] == 0))


if __name__ == "__main__":
    main()