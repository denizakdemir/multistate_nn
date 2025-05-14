"""
Test script to demonstrate time-adjusted simulation for consistent CIFs 
across different time discretizations.

This script shows how time-adjusted simulation resolves issues with CIF 
calculation when comparing different time discretizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from multistate_nn import fit, ModelConfig, TrainConfig
from multistate_nn.utils import (
    generate_synthetic_data,
    calculate_cif,
    compare_cifs,
    simulate_cohort_trajectories,
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Generate synthetic data with different time discretizations
def generate_test_data():
    """Generate synthetic data with two different time discretizations."""
    # Define common parameters
    n_samples = 1000
    n_covariates = 3
    n_states = 4
    
    # Define a strongly connected transition structure
    # This ensures intermediate states are reliably reached in simulation
    state_transitions = {
        0: [1, 2, 3],  # Allow direct transitions to all states
        1: [2, 3],     # Allow transitions from intermediate states
        2: [3],        # Final transition to absorbing state
        3: []          # Absorbing state
    }
    
    # Fine time discretization (e.g., monthly)
    fine_time_values = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
    
    # Use consistent random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    fine_data = generate_synthetic_data(
        n_samples=n_samples, 
        n_covariates=n_covariates, 
        n_states=n_states, 
        n_time_points=len(fine_time_values),
        time_values=fine_time_values,
        state_transitions=state_transitions,
        random_seed=42
    )
    
    # Coarse time discretization (e.g., quarterly)
    coarse_time_values = np.array([0, 90, 180, 270, 360])
    
    # Generate new dataset with coarse time discretization
    np.random.seed(42)
    torch.manual_seed(42)
    
    coarse_data = generate_synthetic_data(
        n_samples=n_samples, 
        n_covariates=n_covariates, 
        n_states=n_states, 
        n_time_points=len(coarse_time_values),
        time_values=coarse_time_values,
        state_transitions=state_transitions,
        random_seed=42
    )
    
    # Store the max observed time for use in simulations
    max_observed_time = 360  # This is the maximum time value in our data
    
    return fine_data, coarse_data, max_observed_time, state_transitions

# 2. Fit models with each discretization
def fit_models(fine_data, coarse_data, state_transitions):
    """Fit models with both time discretizations."""
    # Common parameters    
    covariates = [f"covariate_{i}" for i in range(3)]
    
    # Define model and training configurations
    def create_configs():
        model_config = ModelConfig(
            input_dim=len(covariates),
            hidden_dims=[32, 16],
            num_states=4,
            state_transitions=state_transitions
        )
        
        train_config = TrainConfig(
            batch_size=64,
            epochs=100,  # Increase epochs for better convergence
            learning_rate=0.01,
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
    
    return fine_model, coarse_model, covariates

# 3. Simulate trajectories from each model with time adjustment
def simulate_trajectories(fine_model, coarse_model, max_observed_time):
    """Simulate patient trajectories using both models with time adjustment."""
    # Create a test patient
    test_features = torch.zeros((1, 3))
    
    # Simulate within the observed time range (important!)
    n_simulations = 2000  # Increase for more stable results
    
    # Use same random seed for both simulations to reduce variability
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # For the fine model with time adjustment
    print("Simulating with fine model (time-adjusted)...")
    fine_trajectories = simulate_cohort_trajectories_time_adjusted(
        model=fine_model,
        cohort_features=test_features,
        start_state=0,
        max_time=max_observed_time,  # Explicitly use observed time range
        n_simulations_per_patient=n_simulations,
        seed=1234,
        use_original_time=True
    )
    
    # Reset seed for consistent simulation
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # For the coarse model with time adjustment
    print("Simulating with coarse model (time-adjusted)...")
    coarse_trajectories = simulate_cohort_trajectories_time_adjusted(
        model=coarse_model,
        cohort_features=test_features,
        start_state=0,
        max_time=max_observed_time,  # Explicitly use observed time range
        n_simulations_per_patient=n_simulations,
        seed=1234,
        use_original_time=True
    )
    
    return fine_trajectories, coarse_trajectories

# 4. Calculate and compare CIFs
def calculate_and_compare_cifs(fine_trajectories, coarse_trajectories, max_observed_time):
    """Calculate CIFs for both trajectories and compare them."""
    # Create consistent time grid for evaluation within observed time range
    time_grid = np.linspace(0, max_observed_time, 100)
    
    # Let's calculate CIFs for multiple states to demonstrate the approach
    # Focus on non-absorbing states to prevent all CIFs converging to 1.0
    states_to_check = [1, 2]  # Non-absorbing states

    plt.figure(figsize=(15, 10))

    for i, target_state in enumerate(states_to_check):
        # Calculate CIFs using the consistent time grid
        fine_cif = calculate_cif(
            fine_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time  # Explicitly limit to observed time range
        )
        
        coarse_cif = calculate_cif(
            coarse_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time  # Explicitly limit to observed time range
        )
        
        # Plot in a subplot
        ax = plt.subplot(1, len(states_to_check), i+1)
        
        # Plot the CIFs
        plt.plot(fine_cif['time'], fine_cif['cif'], 'b-', linewidth=2, 
                label=f'Fine Discretization')
        plt.plot(coarse_cif['time'], coarse_cif['cif'], 'r-', linewidth=2, 
                label=f'Coarse Discretization')
        
        # Add confidence intervals
        plt.fill_between(fine_cif['time'], fine_cif['lower_ci'], fine_cif['upper_ci'], 
                        color='blue', alpha=0.1)
        plt.fill_between(coarse_cif['time'], coarse_cif['lower_ci'], coarse_cif['upper_ci'], 
                        color='red', alpha=0.1)
        
        plt.title(f'CIF for State {target_state}')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Incidence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Calculate max difference for this state
        max_diff = np.max(np.abs(fine_cif['cif'].values - coarse_cif['cif'].values))
        mean_diff = np.mean(np.abs(fine_cif['cif'].values - coarse_cif['cif'].values))
        plt.text(0.05, 0.95, f'Max Diff: {max_diff:.3f}\nMean Diff: {mean_diff:.3f}', 
                transform=ax.transAxes, backgroundcolor='white', fontsize=10)

    plt.suptitle('CIF Comparison with Time-Adjusted Simulation', fontsize=16)
    plt.tight_layout()
    
    plt.savefig('time_adjusted_cif_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistical comparison for all states
    print("\nCIF Comparison Statistics with Time Adjustment:")
    print("=" * 50)
    
    for target_state in states_to_check:
        fine_cif = calculate_cif(
            fine_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time
        )
        
        coarse_cif = calculate_cif(
            coarse_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time
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
        max_time=max_observed_time
    )
    
    coarse_cif_abs = calculate_cif(
        coarse_trajectories, 
        target_state=absorbing_state, 
        time_grid=time_grid,
        max_time=max_observed_time
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
    fine_data, coarse_data, max_observed_time, state_transitions = generate_test_data()
    
    print(f"   Generated data with time range: 0 to {max_observed_time}")
    print(f"   Fine discretization: {len(np.unique(fine_data['time']))} time points")
    print(f"   Coarse discretization: {len(np.unique(coarse_data['time']))} time points")
    print(f"   Time step sizes:")
    print(f"     Fine: {np.unique(fine_data['time'])[1] - np.unique(fine_data['time'])[0]} time units")
    print(f"     Coarse: {np.unique(coarse_data['time'])[1] - np.unique(coarse_data['time'])[0]} time units")
    print(f"   Ratio: {(np.unique(coarse_data['time'])[1] - np.unique(coarse_data['time'])[0]) / (np.unique(fine_data['time'])[1] - np.unique(fine_data['time'])[0]):.1f}x")
    
    print("\n2. Fitting models...")
    fine_model, coarse_model, covariates = fit_models(fine_data, coarse_data, state_transitions)
    
    print("\n3. Simulating trajectories with time adjustment...")
    fine_trajectories, coarse_trajectories = simulate_trajectories(
        fine_model, coarse_model, max_observed_time
    )
    
    print("\n4. Calculating and comparing CIFs...")
    fine_cif, coarse_cif = calculate_and_compare_cifs(
        fine_trajectories, coarse_trajectories, max_observed_time
    )
    
    print("\nTest complete!")
    print("\nKey factors for consistent CIFs with time-adjusted simulation:")
    print("1. Using original time values rather than indices")
    print("2. Adjusting transition probabilities based on time window size")
    print("3. Converting probabilities to rates and scaling by time difference")
    print("4. Evaluating on a consistent time grid")
    print("5. Limiting simulation to the observed time range")

if __name__ == "__main__":
    main()