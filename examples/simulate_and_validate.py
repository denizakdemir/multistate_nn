#!/usr/bin/env python
"""
Simulation and Validation Example for MultiStateNN

This example demonstrates:
1. Generating synthetic data with known transition patterns
2. Training a MultiStateNN model on the data
3. Validating that the model correctly learns the underlying patterns
4. Showing how time discretization affects model behavior
5. Demonstrating original time scale handling

The script uses a simple 4-state model:
- State 0: Healthy
- State 1: Mild disease
- State 2: Severe disease
- State 3: Death (absorbing state)

Transitions are influenced by:
- Age: Older patients progress faster
- Treatment: Reduces progression rates
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple

from multistate_nn import (
    MultiStateNN, 
    ModelConfig, 
    TrainConfig, 
    fit
)
from multistate_nn.utils import (
    simulate_patient_trajectory,
    calculate_cif,
    plot_cif
)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define model parameters
N_SAMPLES = 2000
N_STATES = 4
MAX_TIME = 24  # months

# State transition structure
STATE_TRANSITIONS = {
    0: [1, 2],    # Healthy -> Mild or Severe
    1: [0, 2, 3], # Mild -> Healthy, Severe, or Death
    2: [1, 3],    # Severe -> Mild or Death
    3: []         # Death (absorbing state)
}

# Define two different time scales to demonstrate discretization effects
TIME_SCALES = {
    "coarse": [0, 6, 12, 18, 24],                      # 6-month intervals
    "fine": [0, 3, 6, 9, 12, 15, 18, 21, 24]           # 3-month intervals
}


def sigmoid(x: float) -> float:
    """Sigmoid function to convert linear combinations to probabilities."""
    return 1 / (1 + np.exp(-x))


def generate_transition_probs(age: float, treatment: int, time: float) -> Dict[Tuple[int, int], float]:
    """
    Generate transition probabilities based on covariates.
    
    Parameters
    ----------
    age : float
        Patient age (normalized to mean 0, std 1)
    treatment : int
        Treatment indicator (0=untreated, 1=treated)
    time : float
        Current time in months
    
    Returns
    -------
    Dict[Tuple[int, int], float]
        Mapping from (source, target) state pairs to transition probabilities
    """
    # Time factor: transitions become more likely with time (disease progression)
    time_factor = time / MAX_TIME
    
    # Base transition probabilities
    transition_probs = {
        # Format: (from_state, to_state): probability
        (0, 1): sigmoid(0.1 + 0.5 * age - 0.8 * treatment + 0.3 * time_factor),  # Healthy -> Mild
        (0, 2): sigmoid(-1.5 + 0.6 * age - 0.7 * treatment + 0.4 * time_factor), # Healthy -> Severe
        
        (1, 0): sigmoid(0.0 - 0.3 * age + 1.0 * treatment - 0.5 * time_factor),  # Mild -> Healthy (recovery)
        (1, 2): sigmoid(-0.5 + 0.7 * age - 0.9 * treatment + 0.6 * time_factor), # Mild -> Severe
        (1, 3): sigmoid(-2.0 + 0.8 * age - 0.8 * treatment + 0.7 * time_factor), # Mild -> Death
        
        (2, 1): sigmoid(-1.0 - 0.4 * age + 1.1 * treatment - 0.6 * time_factor), # Severe -> Mild (improvement)
        (2, 3): sigmoid(-0.5 + 0.9 * age - 0.7 * treatment + 0.8 * time_factor)  # Severe -> Death
    }
    
    return transition_probs


def normalize_and_sample_next_state(
    current_state: int, 
    transition_probs: Dict[Tuple[int, int], float]
) -> int:
    """
    Sample next state based on transition probabilities.
    
    Parameters
    ----------
    current_state : int
        Current state
    transition_probs : Dict[Tuple[int, int], float]
        Transition probabilities
    
    Returns
    -------
    int
        Next state
    """
    possible_next_states = STATE_TRANSITIONS[current_state]
    
    if not possible_next_states:  # Absorbing state
        return current_state
    
    # Extract probabilities for current state
    probs = [transition_probs.get((current_state, next_state), 0) 
             for next_state in possible_next_states]
    
    # Normalize probabilities
    total_prob = sum(probs)
    if total_prob > 0:
        probs = [p / total_prob for p in probs]
    else:
        # Equal probabilities if all are zero
        probs = [1 / len(possible_next_states)] * len(possible_next_states)
    
    # Sample next state
    return np.random.choice(possible_next_states, p=probs)


def generate_synthetic_data(
    n_samples: int, 
    time_points: List[float]
) -> pd.DataFrame:
    """
    Generate synthetic data for multistate model.
    
    Parameters
    ----------
    n_samples : int
        Number of patient trajectories to generate
    time_points : List[float]
        Time points to include
    
    Returns
    -------
    pd.DataFrame
        Synthetic data with columns:
        - time: Time point
        - from_state: Source state
        - to_state: Target state
        - age: Patient age (normalized)
        - treatment: Treatment indicator
        - patient_id: Patient identifier
    """
    records = []
    
    for patient_id in range(n_samples):
        # Generate patient characteristics
        age = np.random.normal(0, 1)  # Normalized age
        treatment = np.random.binomial(1, 0.5)  # 50% treated
        
        # Start in healthy state
        current_state = 0
        
        # Generate transitions over time
        for i, time in enumerate(time_points[:-1]):
            next_time = time_points[i + 1]
            
            # Skip if in absorbing state
            if not STATE_TRANSITIONS[current_state]:
                break
                
            # Generate transition probabilities
            transition_probs = generate_transition_probs(age, treatment, time)
            
            # Sample next state
            next_state = normalize_and_sample_next_state(current_state, transition_probs)
            
            # Add record
            records.append({
                "time": time,
                "from_state": current_state,
                "to_state": next_state,
                "age": age,
                "treatment": treatment,
                "patient_id": patient_id
            })
            
            # Update current state
            current_state = next_state
    
    return pd.DataFrame(records)


def calculate_true_probabilities(
    covariates: Dict[str, float], 
    time_points: List[float]
) -> Dict[Tuple[int, int, float], float]:
    """
    Calculate true transition probabilities based on our generating model.
    
    Parameters
    ----------
    covariates : Dict[str, float]
        Covariate values (age, treatment)
    time_points : List[float]
        Time points to calculate probabilities for
    
    Returns
    -------
    Dict[Tuple[int, int, float], float]
        Mapping from (from_state, to_state, time) to probability
    """
    age = covariates["age"]
    treatment = covariates["treatment"]
    
    # Calculate probabilities for each time point
    all_probs = {}
    for time in time_points:
        transition_probs = generate_transition_probs(age, treatment, time)
        
        # Normalize probabilities for each from_state
        for state in range(N_STATES):
            if not STATE_TRANSITIONS[state]:
                continue
                
            # Get probs for transitions from this state
            state_probs = {
                next_state: transition_probs.get((state, next_state), 0)
                for next_state in STATE_TRANSITIONS[state]
            }
            
            # Normalize
            total = sum(state_probs.values())
            if total > 0:
                for next_state, prob in state_probs.items():
                    all_probs[(state, next_state, time)] = prob / total
    
    return all_probs


def validate_model(
    model: MultiStateNN, 
    covariates: Dict[str, float], 
    time_points: List[float]
) -> Tuple[Dict[Tuple[int, int, float], float], Dict[Tuple[int, int, float], float]]:
    """
    Compare model predictions with true probabilities.
    
    Parameters
    ----------
    model : MultiStateNN
        Trained model
    covariates : Dict[str, float]
        Covariate values
    time_points : List[float]
        Time points to validate at
    
    Returns
    -------
    Tuple[Dict, Dict]
        True probabilities and model predictions
    """
    # Calculate true probabilities
    true_probs = calculate_true_probabilities(covariates, time_points)
    
    # Convert covariates to tensor for model input
    x = torch.tensor(
        [[covariates["age"], covariates["treatment"]]], 
        dtype=torch.float32
    )
    
    # Get model predictions
    model_probs = {}
    for time_idx, time in enumerate(time_points):
        for state in range(N_STATES):
            if not STATE_TRANSITIONS[state]:
                continue
                
            # Get model predictions
            probs = model.predict_proba(
                x, 
                time_idx=time_idx if not hasattr(model, 'time_mapper') else time, 
                from_state=state
            ).detach().cpu().numpy()[0]
            
            # Store predictions
            for i, next_state in enumerate(STATE_TRANSITIONS[state]):
                model_probs[(state, next_state, time)] = probs[i]
    
    return true_probs, model_probs


def plot_validation_results(
    true_probs: Dict[Tuple[int, int, float], float], 
    model_probs: Dict[Tuple[int, int, float], float]
) -> plt.Figure:
    """
    Plot comparison of true vs. predicted probabilities.
    
    Parameters
    ----------
    true_probs : Dict[Tuple[int, int, float], float]
        True probabilities
    model_probs : Dict[Tuple[int, int, float], float]
        Model predictions
    
    Returns
    -------
    plt.Figure
        Figure with plot
    """
    # Extract values
    keys = list(true_probs.keys())
    true_values = [true_probs[k] for k in keys]
    pred_values = [model_probs[k] for k in keys]
    
    # Calculate R² score
    r2 = r2_score(true_values, pred_values)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_values, pred_values, alpha=0.6)
    
    # Add diagonal line
    min_val = min(min(true_values), min(pred_values))
    max_val = max(max(true_values), max(pred_values))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    ax.set_xlabel('True Probability')
    ax.set_ylabel('Predicted Probability')
    ax.set_title(f'True vs. Predicted Transition Probabilities (R² = {r2:.3f})')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig


def run_simulation_and_validation(time_scale: str):
    """
    Run full simulation, training, and validation for a given time scale.
    
    Parameters
    ----------
    time_scale : str
        Which time scale to use ("coarse" or "fine")
    """
    print(f"\n{'='*80}\nRunning simulation with {time_scale} time discretization\n{'='*80}")
    
    # Get time points for this scale
    time_points = TIME_SCALES[time_scale]
    print(f"Time points: {time_points}")
    
    # Generate data
    print("Generating synthetic data...")
    df = generate_synthetic_data(N_SAMPLES, time_points)
    print(f"Generated {len(df)} transitions from {N_SAMPLES} patients")
    
    # Examine distribution of states
    state_counts = df.groupby(['time', 'from_state']).size().unstack(fill_value=0)
    print("\nState distribution over time:")
    print(state_counts)
    
    # Define model configuration
    model_config = ModelConfig(
        input_dim=2,  # age, treatment
        hidden_dims=[64, 32],
        num_states=N_STATES,
        state_transitions=STATE_TRANSITIONS
    )
    
    # Define training configuration
    train_config = TrainConfig(
        batch_size=64,
        epochs=50,
        learning_rate=0.01,
        use_original_time=True
    )
    
    # Train model
    print("\nTraining model...")
    model = fit(
        df=df,
        covariates=["age", "treatment"],
        model_config=model_config,
        train_config=train_config
    )
    
    # Validate model on different patient profiles
    patient_profiles = [
        {"age": -1.0, "treatment": 1},  # Young, treated
        {"age": 1.0, "treatment": 1},   # Elderly, treated
        {"age": -1.0, "treatment": 0},  # Young, untreated
        {"age": 1.0, "treatment": 0},   # Elderly, untreated
    ]
    
    profile_names = [
        "Young, Treated",
        "Elderly, Treated", 
        "Young, Untreated",
        "Elderly, Untreated"
    ]
    
    # Validate probabilities
    print("\nValidating transition probabilities...")
    validation_results = []
    for profile, name in zip(patient_profiles, profile_names):
        true_probs, model_probs = validate_model(model, profile, time_points)
        validation_results.append((name, true_probs, model_probs))
    
    # Plot validation for one profile (elderly, untreated - should have highest risk)
    _, true_probs, model_probs = validation_results[3]
    fig = plot_validation_results(true_probs, model_probs)
    fig.savefig(f"validation_{time_scale}.png")
    print(f"Saved validation plot to validation_{time_scale}.png")
    
    # Simulate trajectories for each profile
    print("\nSimulating patient trajectories...")
    all_cifs = []
    
    for profile, name in zip(patient_profiles, profile_names):
        # Create input tensor
        x = torch.tensor([[profile["age"], profile["treatment"]]], dtype=torch.float32)
        
        # Simulate trajectories
        trajectories = simulate_patient_trajectory(
            model=model,
            x=x,
            start_state=0,
            max_time=time_points[-1],
            n_simulations=1000,
            use_original_time=True
        )
        
        # Calculate CIF for death (state 3)
        cif = calculate_cif(
            trajectories=pd.concat(trajectories),
            target_state=3,
            use_original_time=True
        )
        
        all_cifs.append((name, cif))
    
    # Plot CIFs
    plt.figure(figsize=(10, 6))
    for name, cif in all_cifs:
        ax = plot_cif(cif, label=name, use_original_time=True)
    
    plt.title(f"Cumulative Incidence of Death ({time_scale} time discretization)")
    plt.xlabel("Time (months)")
    plt.ylabel("Cumulative Incidence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"cif_{time_scale}.png")
    print(f"Saved CIF plot to cif_{time_scale}.png")
    
    # Print some insights
    print("\nInsights from simulations:")
    for name, cif in all_cifs:
        final_cif = cif['cif'].iloc[-1]
        print(f"- {name}: Final death probability after {time_points[-1]} months: {final_cif:.2f}")
    
    return model, all_cifs


def compare_discretizations(coarse_cifs, fine_cifs):
    """
    Compare CIFs from different time discretizations.
    
    Parameters
    ----------
    coarse_cifs : List[Tuple[str, pd.DataFrame]]
        CIFs from coarse time discretization
    fine_cifs : List[Tuple[str, pd.DataFrame]]
        CIFs from fine time discretization
    """
    # Plot comparison for elderly untreated (highest risk group)
    plt.figure(figsize=(12, 7))
    
    # Plot coarse discretization
    elderly_untreated_coarse = coarse_cifs[3][1]
    ax = plot_cif(
        elderly_untreated_coarse, 
        label="Coarse Discretization (6-month)",
        color='blue',
        use_original_time=True
    )
    
    # Plot fine discretization
    elderly_untreated_fine = fine_cifs[3][1]
    plot_cif(
        elderly_untreated_fine, 
        label="Fine Discretization (3-month)",
        color='red',
        linestyle='--',
        ax=ax,
        use_original_time=True
    )
    
    plt.title("Effect of Time Discretization on CIF (Elderly Untreated)")
    plt.xlabel("Time (months)")
    plt.ylabel("Cumulative Incidence of Death")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("discretization_comparison.png")
    print("\nSaved discretization comparison to discretization_comparison.png")


def main():
    """Run the full simulation and validation."""
    # Run simulation and validation for each time scale
    coarse_model, coarse_cifs = run_simulation_and_validation("coarse")
    fine_model, fine_cifs = run_simulation_and_validation("fine")
    
    # Compare discretizations
    compare_discretizations(coarse_cifs, fine_cifs)
    
    print("\nSimulation and validation complete!")


if __name__ == "__main__":
    main()