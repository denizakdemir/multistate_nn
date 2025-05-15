"""
Test script to verify CIF calculation consistency across different time discretizations.

This script demonstrates that CIFs should be comparable regardless of the time
discretization used in the model, when properly implemented.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from multistate_nn import fit, ModelConfig, TrainConfig
from multistate_nn.utils import (
    generate_synthetic_data,
    simulate_cohort_trajectories,
    calculate_cif,
    compare_cifs,
    TimeMapper
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Generate synthetic data with different time discretizations
def generate_test_data():
    """Generate synthetic data with two different time discretizations from the same continuous-time process."""
    # Define parameters
    n_samples = 1000
    n_covariates = 2
    n_states = 4
    
    # Return these values for later use
    global_params = {
        'n_covariates': n_covariates,
        'n_states': n_states
    }
    
    # Define state transition structure
    state_transitions = {
        0: [1, 0],     # Allow self-transitions and transitions to state 1
        1: [2, 1],     # Allow self-transitions and transitions to state 2
        2: [3, 2],     # Allow self-transitions and transitions to state 3
        3: []          # Absorbing state
    }
    
    # Define time grids for discretization
    fine_time_values = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
    coarse_time_values = np.array([0, 90, 180, 270, 360])
    
    # Set consistent random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate covariates
    X = np.random.normal(0, 1, (n_samples, n_covariates))
    
    # Define hazard rates (continuous-time transition rates)
    # These determine the probability of transitioning in the continuous-time process
    hazard_rates = {
        0: 0.003,  # Rate of transitioning from state 0 to state 1
        1: 0.005,  # Rate of transitioning from state 1 to state 2
        2: 0.007   # Rate of transitioning from state 2 to state 3
    }
    
    # Generate continuous-time trajectories for each patient
    continuous_trajectories = []
    
    for i in range(n_samples):
        current_state = 0
        current_time = 0
        patient_trajectory = [(current_time, current_state, current_state)]  # (time, from_state, to_state)
        
        while current_time < 360 and current_state < 3:  # Until max time or absorbing state
            # Time until next event follows exponential distribution with rate parameter
            rate = hazard_rates[current_state]
            time_to_event = np.random.exponential(1/rate)
            
            # New event time
            next_time = current_time + time_to_event
            
            # If we're still within the time horizon
            if next_time <= 360:
                # Determine next state (deterministic transition to next state in this simple case)
                next_state = current_state + 1
                
                # Record transition
                patient_trajectory.append((next_time, current_state, next_state))
                
                # Update state
                current_state = next_state
                current_time = next_time
            else:
                # We've reached the time horizon without another event
                break
        
        continuous_trajectories.append(patient_trajectory)
    
    # Discretize continuous trajectories to fine and coarse grids
    fine_records = []
    coarse_records = []
    
    for i, trajectory in enumerate(continuous_trajectories):
        # Extract patient covariates
        patient_covariates = {f"covariate_{j}": X[i, j] for j in range(n_covariates)}
        
        # Discretize to fine grid
        fine_discretized = discretize_trajectory(trajectory, fine_time_values, patient_covariates)
        fine_records.extend(fine_discretized)
        
        # Discretize to coarse grid
        coarse_discretized = discretize_trajectory(trajectory, coarse_time_values, patient_covariates)
        coarse_records.extend(coarse_discretized)
    
    # Create DataFrames
    fine_data = pd.DataFrame(fine_records)
    coarse_data = pd.DataFrame(coarse_records)
    
    # Ensure all states are represented (for training stability)
    fine_data = ensure_all_states_present(fine_data, n_states, fine_time_values[0], n_covariates)
    coarse_data = ensure_all_states_present(coarse_data, n_states, coarse_time_values[0], n_covariates)
    
    # Set maximum observed time
    max_observed_time = 360
    
    return fine_data, coarse_data, max_observed_time, state_transitions, global_params

def discretize_trajectory(trajectory, time_grid, covariates):
    """Discretize a continuous-time trajectory to a specific time grid."""
    records = []
    
    # Extract continuous event times and states
    event_times = [event[0] for event in trajectory]
    states = [event[2] for event in trajectory]
    
    # For each time point in the grid (except the last one)
    for t_idx in range(len(time_grid) - 1):
        current_time = time_grid[t_idx]
        next_grid_time = time_grid[t_idx + 1]
        
        # Find the state at this time point
        state_idx = np.searchsorted(event_times, current_time, side='right') - 1
        state_idx = max(0, state_idx)  # Ensure non-negative index
        current_state = states[state_idx]
        
        # Find if any state change happens before the next grid time
        next_event_idx = np.searchsorted(event_times, next_grid_time, side='left')
        
        # Handle index error by checking array bounds
        if next_event_idx > state_idx + 1 and state_idx + 1 < len(states):
            # Multiple events before next grid time
            # Record transition to the first new state
            next_state = states[state_idx + 1]
        elif next_event_idx == state_idx + 1 and state_idx + 1 < len(states) and event_times[state_idx + 1] < next_grid_time:
            # One event before next grid time
            next_state = states[state_idx + 1]
        else:
            # No events before next grid time - self-transition
            next_state = current_state
        
        # Create record
        record = {
            "time": current_time,
            "from_state": current_state,
            "to_state": next_state,
            **covariates
        }
        records.append(record)
    
    return records

def ensure_all_states_present(df, n_states, initial_time, n_covariates):
    """Ensure all states are present in the dataset for training stability."""
    if not df.empty:
        all_states_present = set(df["from_state"].unique()) | set(df["to_state"].unique())
        
        for s in range(n_states):
            if s not in all_states_present:
                # Add a dummy record for each missing state
                dummy_record = {
                    "time": initial_time,
                    "from_state": max(s-1, 0),
                    "to_state": s,
                    **{f"covariate_{j}": 0.0 for j in range(n_covariates)},
                }
                df = pd.concat([df, pd.DataFrame([dummy_record])], ignore_index=True)
    
    return df

# 2. Fit models with each discretization
def fit_models(fine_data, coarse_data, state_transitions, n_covariates):
    """Fit models with both time discretizations."""
    # Common parameters    
    covariates = [f"covariate_{i}" for i in range(n_covariates)]
    
    # Define model and training configurations
    def create_configs():
        model_config = ModelConfig(
            input_dim=len(covariates),
            hidden_dims=[16, 8],  # Smaller network for faster training
            num_states=4,
            state_transitions=state_transitions
        )
        
        train_config = TrainConfig(
            batch_size=32,    # Small batch size
            epochs=30,        # Minimum epochs for convergence
            learning_rate=0.01,  # Higher learning rate for faster convergence 
            weight_decay=1e-4,   # L2 regularization for better generalization
            use_original_time=True  # Critical: use original time values, not indices
        )
        
        return model_config, train_config
    
    # Fit model with fine discretization
    model_config_fine, train_config_fine = create_configs()
    # Use consistent seed for training
    torch.manual_seed(42)
    np.random.seed(42)
    
    fine_model = fit(
        df=fine_data,
        covariates=covariates,
        model_config=model_config_fine,
        train_config=train_config_fine
    )
    
    # Fit model with coarse discretization
    model_config_coarse, train_config_coarse = create_configs()
    # Use consistent seed for training
    torch.manual_seed(42)
    np.random.seed(42)
    
    coarse_model = fit(
        df=coarse_data,
        covariates=covariates,
        model_config=model_config_coarse,
        train_config=train_config_coarse
    )
    
    return fine_model, coarse_model

# 3. Simulate trajectories from each model
def simulate_trajectories(fine_model, coarse_model, max_observed_time, n_covariates):
    """Simulate patient trajectories using both models."""
    # Create a test patient
    test_features = torch.zeros((1, n_covariates))
    
    # Simulate within the observed time range (important!)
    n_simulations = 10000  # More simulations for better statistical comparison
    
    # Use same random seed for both simulations to reduce variability
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # For the fine model
    fine_trajectories = simulate_cohort_trajectories(
        model=fine_model,
        cohort_features=test_features,
        start_state=0,
        max_time=max_observed_time,  # Explicitly use observed time range
        n_simulations_per_patient=n_simulations,
        seed=1234,
        time_adjusted=True,  # Enable time adjustment
        use_original_time=True
    )
    
    # Reset seed for consistent simulation
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # For the coarse model
    coarse_trajectories = simulate_cohort_trajectories(
        model=coarse_model,
        cohort_features=test_features,
        start_state=0,
        max_time=max_observed_time,  # Explicitly use observed time range
        n_simulations_per_patient=n_simulations,
        seed=1234,
        time_adjusted=True,  # Enable time adjustment
        use_original_time=True
    )
    
    return fine_trajectories, coarse_trajectories

# 4. Calculate and compare CIFs
def calculate_and_compare_cifs(fine_trajectories, coarse_trajectories, max_observed_time, fine_data, coarse_data):
    """Calculate CIFs for both trajectories and compare them."""
    # Create a time grid for evaluation within observed time range
    time_grid = np.linspace(0, max_observed_time, 60)  # fewer points for faster computation
    
    # Let's calculate CIFs for multiple states to demonstrate the approach
    # Focus on non-absorbing states to prevent all CIFs converging to 1.0
    states_to_check = [1, 2, 3]  # Include absorbing state 3 as well
    
    # Create subplots with more space for legends
    plt.figure(figsize=(18, 12))
    
    # Improved confidence interval settings - 95% CI for better visualization
    ci_level = 0.95

    for i, target_state in enumerate(states_to_check):
        # Calculate CIFs using the very fine time grid with improved confidence intervals
        fine_cif = calculate_cif(
            fine_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time,  # Explicitly limit to observed time range
            ci_level=ci_level,  # Explicitly set confidence level
            method="empirical"  # Use empirical method
        )
        
        coarse_cif = calculate_cif(
            coarse_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time,  # Explicitly limit to observed time range
            ci_level=ci_level,  # Explicitly set confidence level
            method="empirical"  # Use empirical method
        )
        
        # Plot in a subplot
        ax = plt.subplot(1, len(states_to_check), i+1)
        
        # Plot the CIFs with smoother, more attractive lines
        plt.plot(fine_cif['time'], fine_cif['cif'], 'b-', linewidth=3, 
                label=f'Fine Discretization (Weekly)')
        plt.plot(coarse_cif['time'], coarse_cif['cif'], 'r-', linewidth=3, 
                label=f'Coarse Discretization (Monthly)')
        
        # Add confidence intervals with slightly more opacity for better visibility
        plt.fill_between(fine_cif['time'], fine_cif['lower_ci'], fine_cif['upper_ci'], 
                        color='blue', alpha=0.15, label=f'Fine 95% CI')
        plt.fill_between(coarse_cif['time'], coarse_cif['lower_ci'], coarse_cif['upper_ci'], 
                        color='red', alpha=0.15, label=f'Coarse 95% CI')
        
        # Add reference lines
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Improve title and labels
        state_name = "Absorbing" if target_state == 3 else f"Intermediate {target_state}"
        plt.title(f'CIF for State {target_state} ({state_name})', fontsize=14)
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Cumulative Incidence', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='best')
        
        # Set y-axis limits for better visualization
        plt.ylim(0, 1.05)
        
        # Calculate max difference for this state
        max_diff = np.max(np.abs(fine_cif['cif'].values - coarse_cif['cif'].values))
        mean_diff = np.mean(np.abs(fine_cif['cif'].values - coarse_cif['cif'].values))
        plt.text(0.05, 0.95, f'Max Diff: {max_diff:.3f}\nMean Diff: {mean_diff:.3f}', 
                transform=ax.transAxes, backgroundcolor='white', fontsize=12)
                
        # Add time discretization information
        if target_state == 1:
            plt.text(0.05, 0.05, 
                    f'Fine: {len(np.unique(fine_data["time"]))} time points\nCoarse: {len(np.unique(coarse_data["time"]))} time points',
                    transform=ax.transAxes, backgroundcolor='white', fontsize=10)

    plt.suptitle('CIF Comparison with Different Time Discretizations', fontsize=18)
    plt.tight_layout()
    
    # Save high resolution figure
    plt.savefig('cif_discretization_comparison.png', dpi=600, bbox_inches='tight')
    
    # Also save a version for including in papers
    plt.savefig('cif_discretization_comparison.pdf', format='pdf', bbox_inches='tight')
    
    plt.show()
    
    # Print statistical comparison for all states
    print("\nCIF Comparison Statistics Summary:")
    print("=" * 40)
    
    for target_state in states_to_check:
        fine_cif = calculate_cif(
            fine_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time,
            method="empirical"  # Use empirical method
        )
        
        coarse_cif = calculate_cif(
            coarse_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time,
            method="empirical"  # Use empirical method
        )
        
        mean_diff = np.mean(np.abs(fine_cif['cif'].values - coarse_cif['cif'].values))
        max_diff = np.max(np.abs(fine_cif['cif'].values - coarse_cif['cif'].values))
        
        print(f"State {target_state}:")
        print(f"  Mean absolute difference: {mean_diff:.4f}")
        print(f"  Maximum absolute difference: {max_diff:.4f}")
    
    # Also check absorbing state for comparison
    absorbing_state = 3
    fine_cif_abs = calculate_cif(
        fine_trajectories, 
        target_state=absorbing_state, 
        time_grid=time_grid,
        max_time=max_observed_time,
        method="empirical"  # Use empirical method
    )
    
    coarse_cif_abs = calculate_cif(
        coarse_trajectories, 
        target_state=absorbing_state, 
        time_grid=time_grid,
        max_time=max_observed_time,
        method="empirical"  # Use empirical method
    )
    
    mean_diff_abs = np.mean(np.abs(fine_cif_abs['cif'].values - coarse_cif_abs['cif'].values))
    max_diff_abs = np.max(np.abs(fine_cif_abs['cif'].values - coarse_cif_abs['cif'].values))
    
    print(f"State {absorbing_state} (absorbing):")
    print(f"  Mean absolute difference: {mean_diff_abs:.4f}")
    print(f"  Maximum absolute difference: {max_diff_abs:.4f}")
    
    return fine_cif, coarse_cif

# Main function
def main():
    print("1. Generating synthetic data with different time discretizations...")
    fine_data, coarse_data, max_observed_time, state_transitions, global_params = generate_test_data()
    
    print(f"   Generated data with time range: 0 to {max_observed_time}")
    print(f"   Fine discretization has {len(np.unique(fine_data['time']))} unique time points: {np.unique(fine_data['time'])}")
    print(f"   Coarse discretization has {len(np.unique(coarse_data['time']))} unique time points: {np.unique(coarse_data['time'])}")
    
    n_covariates = global_params['n_covariates']
    
    print("\n2. Fitting models...")
    fine_model, coarse_model = fit_models(fine_data, coarse_data, state_transitions, n_covariates)
    
    print("\n3. Simulating trajectories...")
    fine_trajectories, coarse_trajectories = simulate_trajectories(
        fine_model, coarse_model, max_observed_time, n_covariates
    )
    
    print("\n4. Calculating and comparing CIFs...")
    fine_cif, coarse_cif = calculate_and_compare_cifs(
        fine_trajectories, coarse_trajectories, max_observed_time, fine_data, coarse_data
    )
    
    print("\nTest complete! If the CIFs are similar, the time discretization no longer affects the results.")
    print("\nKey factors for consistent CIFs:")
    print("1. Using original time values rather than indices")
    print("2. Evaluating on a consistent time grid")
    print("3. Limiting simulation to the observed time range")
    print("4. Using non-absorbing states to prevent convergence to 1.0")
    print("5. Using consistent random seeds for reproducibility")
    print("6. Enabling time adjustment for simulations")
    print("7. Simulating from the same underlying continuous-time process")

if __name__ == "__main__":
    main()