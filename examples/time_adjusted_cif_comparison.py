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
    # MODIFIED: Added self-transitions to better test time adjustment
    state_transitions = {
        0: [0, 1, 2, 3],  # Allow self-transition and transitions to all states
        1: [1, 2, 3],     # Allow self-transition and forward transitions
        2: [2, 3],        # Allow self-transition and to absorbing state
        3: []             # Absorbing state
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
def simulate_trajectories(fine_model, coarse_model, max_observed_time, n_sims=2000):
    """Simulate patient trajectories using both models with time adjustment."""
    # Create a test patient
    test_features = torch.zeros((1, 3))
    
    # Simulate within the observed time range (important!)
    n_simulations = n_sims  # Use parameter, defaults to 2000 for stable results
    
    # Use same random seed for both simulations to reduce variability
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # For the fine model with time adjustment
    print("Simulating with fine model (time-adjusted)...")
    fine_trajectories = simulate_cohort_trajectories(
        model=fine_model,
        cohort_features=test_features,
        start_state=0,
        max_time=max_observed_time,  # Explicitly use observed time range
        n_simulations_per_patient=n_simulations,
        time_adjusted=True,  # Enable time adjustment
        seed=1234,
        use_original_time=True
    )
    
    # Add debugging info - check if we have any transitions to target states
    state_counts = fine_trajectories['state'].value_counts().sort_index()
    print("\nFine model state distribution:")
    print(state_counts)
    print(f"Percentage of trajectories reaching state 1: {(fine_trajectories['state'] == 1).any().mean() * 100:.2f}%")
    print(f"Percentage of trajectories reaching state 2: {(fine_trajectories['state'] == 2).any().mean() * 100:.2f}%")
    
    # Reset seed for consistent simulation
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # For the coarse model with time adjustment
    print("\nSimulating with coarse model (time-adjusted)...")
    coarse_trajectories = simulate_cohort_trajectories(
        model=coarse_model,
        cohort_features=test_features,
        start_state=0,
        max_time=max_observed_time,  # Explicitly use observed time range
        n_simulations_per_patient=n_simulations,
        time_adjusted=True,  # Enable time adjustment
        seed=1234,
        use_original_time=True
    )
    
    # Add debugging info for coarse model
    state_counts = coarse_trajectories['state'].value_counts().sort_index()
    print("\nCoarse model state distribution:")
    print(state_counts)
    print(f"Percentage of trajectories reaching state 1: {(coarse_trajectories['state'] == 1).any().mean() * 100:.2f}%")
    print(f"Percentage of trajectories reaching state 2: {(coarse_trajectories['state'] == 2).any().mean() * 100:.2f}%")
    
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
        print(f"\nCalculating CIF for state {target_state}...")
        
        # Calculate CIFs using the consistent time grid and empirical method
        fine_cif = calculate_cif(
            fine_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time,  # Explicitly limit to observed time range
            method="empirical"  # Use empirical method which works better with simulated trajectories
        )
        
        coarse_cif = calculate_cif(
            coarse_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time,  # Explicitly limit to observed time range
            method="empirical"  # Use empirical method which works better with simulated trajectories
        )
        
        # Print CIF values for debugging
        print(f"\nCIF statistics for State {target_state}:")
        print(f"Fine CIF: min={fine_cif['cif'].min():.6f}, max={fine_cif['cif'].max():.6f}, " 
              f"mean={fine_cif['cif'].mean():.6f}")
        print(f"Coarse CIF: min={coarse_cif['cif'].min():.6f}, max={coarse_cif['cif'].max():.6f}, "
              f"mean={coarse_cif['cif'].mean():.6f}")
        
        # For detailed debugging, print a sample of values
        print("\nSample CIF values (every 10th point):")
        for j in range(0, len(fine_cif), 10):
            print(f"Time {fine_cif['time'].iloc[j]:.1f}: Fine={fine_cif['cif'].iloc[j]:.6f}, "
                  f"Coarse={coarse_cif['cif'].iloc[j]:.6f}")
        
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
        
        # Find max CIF value across both datasets for better y-axis scaling
        max_cif = max(fine_cif['cif'].max(), coarse_cif['cif'].max())
        plt.ylim(0, max_cif * 1.1)  # Add 10% padding
        
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
    
    # Use absolute path for saving to avoid path issues
    import os
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'time_adjusted_cif_comparison.png')
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Test if file was saved
    if os.path.exists(save_path):
        print(f"File saved successfully! Size: {os.path.getsize(save_path)} bytes")
    else:
        print("Error: File was not saved properly!")
    
    # Show plot in interactive mode
    plt.show()
    
    # Print statistical comparison for all states
    print("\nCIF Comparison Statistics with Time Adjustment:")
    print("=" * 50)
    
    for target_state in states_to_check:
        fine_cif = calculate_cif(
            fine_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time,
            method="empirical"  # Use empirical method for better results with simulated trajectories
        )
        
        coarse_cif = calculate_cif(
            coarse_trajectories, 
            target_state=target_state, 
            time_grid=time_grid,
            max_time=max_observed_time,
            method="empirical"  # Use empirical method for better results with simulated trajectories
        )
        
        # Actual CIF values for debugging
        print(f"\nState {target_state} CIF values (first 5):")
        for i in range(min(5, len(fine_cif))):
            print(f"  Time {fine_cif['time'].iloc[i]:.1f}: Fine={fine_cif['cif'].iloc[i]:.4f}, "
                  f"Coarse={coarse_cif['cif'].iloc[i]:.4f}, "
                  f"Diff={abs(fine_cif['cif'].iloc[i] - coarse_cif['cif'].iloc[i]):.4f}")
        
        mean_diff = np.mean(np.abs(fine_cif['cif'].values - coarse_cif['cif'].values))
        max_diff = np.max(np.abs(fine_cif['cif'].values - coarse_cif['cif'].values))
        
        print(f"\nState {target_state} summary:")
        print(f"  Mean absolute difference: {mean_diff:.4f}")
        print(f"  Maximum absolute difference: {max_diff:.4f}")
    
    # Also check absorbing state for comparison
    absorbing_state = 3
    fine_cif_abs = calculate_cif(
        fine_trajectories, 
        target_state=absorbing_state, 
        time_grid=time_grid,
        max_time=max_observed_time,
        method="empirical"  # Use empirical method for better results with simulated trajectories
    )
    
    coarse_cif_abs = calculate_cif(
        coarse_trajectories, 
        target_state=absorbing_state, 
        time_grid=time_grid,
        max_time=max_observed_time,
        method="empirical"  # Use empirical method for better results with simulated trajectories
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

def fit_models(fine_data, coarse_data, state_transitions, epochs=100):
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
            epochs=epochs,  # Use the epochs parameter
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

def main(dry_run=False):
    """Main function with optional dry-run mode."""
    # Use smaller computation for dry-run mode
    n_simulations = 200 if dry_run else 2000
    train_epochs = 20 if dry_run else 100
    
    print("\nIMPORTANT: Using enhanced matrix-based time adjustment for more consistent results")
    
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
    # Override the number of epochs during model fitting if in dry-run mode
    if dry_run:
        print(f"   Using reduced epochs for dry run: {train_epochs}")
    
    fine_model, coarse_model, covariates = fit_models(
        fine_data, coarse_data, state_transitions, epochs=train_epochs
    )
    
    print("\n3. Simulating trajectories with time adjustment...")
    # Simulate with appropriate number of simulations based on dry_run mode
    fine_trajectories, coarse_trajectories = simulate_trajectories(
        fine_model, coarse_model, max_observed_time, n_sims=n_simulations
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Time-adjusted CIF comparison')
    parser.add_argument('--dry-run', action='store_true', help='Run with minimal computation for debugging')
    args = parser.parse_args()
    
    if args.dry_run:
        print("Running in dry-run mode with minimal computation...")
    
    main(dry_run=args.dry_run)