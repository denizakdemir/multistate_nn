"""
Example of continuous-time multistate modeling with Neural ODEs.

This script demonstrates how to:
1. Create a continuous-time multistate model using Neural ODEs
2. Visualize the intensity matrix and transition probabilities
3. Simulate trajectories from the model
4. Calculate cumulative incidence functions (CIFs)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import directly from modules to avoid import issues
# In a real application, you would use normal imports from multistate_nn
from multistate_nn.models_continuous import ContinuousMultiStateNN
from multistate_nn.utils.continuous_simulation import simulate_continuous_patient_trajectory


def generate_synthetic_data(n_samples=100, n_covariates=2):
    """Generate synthetic data for testing."""
    np.random.seed(42)
    
    # Generate covariates
    x = np.random.normal(0, 1, (n_samples, n_covariates))
    
    # Generate synthetic time-to-event data
    # This is a simple 3-state model: Healthy -> Diseased -> Death
    times = np.random.exponential(scale=5, size=n_samples)
    
    # Generate transitions based on covariates
    # Higher value of first covariate -> faster progression to disease
    # Higher value of second covariate -> faster mortality
    disease_risk = 1 / (1 + np.exp(-x[:, 0]))  # Sigmoid transform
    death_risk = 1 / (1 + np.exp(-x[:, 1]))    # Sigmoid transform
    
    # Generate outcomes
    state = np.zeros(n_samples, dtype=int)
    time_to_disease = np.random.exponential(scale=10 / (1 + np.exp(x[:, 0])))
    time_to_death_from_healthy = np.random.exponential(scale=20 / (1 + np.exp(x[:, 1])))
    time_to_death_from_disease = np.random.exponential(scale=5 / (1 + np.exp(x[:, 1])))
    
    # Determine final state and time
    data = []
    for i in range(n_samples):
        # Case 1: Disease occurs first, then possibly death
        if time_to_disease[i] < time_to_death_from_healthy[i]:
            disease_time = time_to_disease[i]
            # Record the transition to disease state
            data.append({
                'id': i,
                'time': disease_time,
                'from_state': 0,  # Healthy
                'to_state': 1,    # Diseased
                'feature_0': x[i, 0],
                'feature_1': x[i, 1]
            })
            
            # Determine if death occurs after disease
            if disease_time + time_to_death_from_disease[i] < times[i]:
                death_time = disease_time + time_to_death_from_disease[i]
                # Record the transition to death state
                data.append({
                    'id': i,
                    'time': death_time,
                    'from_state': 1,  # Diseased
                    'to_state': 2,    # Death
                    'feature_0': x[i, 0],
                    'feature_1': x[i, 1]
                })
        
        # Case 2: Death occurs before disease
        elif time_to_death_from_healthy[i] < times[i]:
            death_time = time_to_death_from_healthy[i]
            # Record the direct transition to death state
            data.append({
                'id': i,
                'time': death_time,
                'from_state': 0,  # Healthy
                'to_state': 2,    # Death
                'feature_0': x[i, 0],
                'feature_1': x[i, 1]
            })
    
    return pd.DataFrame(data)


def plot_intensity_heatmap(model, x, title="Intensity Matrix"):
    """Plot a heatmap of the intensity matrix."""
    A = model.intensity_matrix(x).squeeze(0).detach().numpy()
    
    # Create a mask for zero elements (not allowed transitions)
    mask = A == 0
    
    # Handle diagonal elements differently
    diag_mask = np.eye(A.shape[0], dtype=bool)
    
    # Combine masks
    combined_mask = mask & ~diag_mask
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(A, annot=True, fmt=".3f", cmap="coolwarm_r", 
                     mask=combined_mask, center=0, 
                     linewidths=0.5, cbar_kws={'label': 'Rate'})
    
    # Add state labels
    state_labels = ["Healthy", "Diseased", "Death"]
    ax.set_xticks(np.arange(len(state_labels)) + 0.5)
    ax.set_yticks(np.arange(len(state_labels)) + 0.5)
    ax.set_xticklabels(state_labels)
    ax.set_yticklabels(state_labels)
    
    plt.title(title)
    plt.tight_layout()
    return ax


def plot_transition_probs(model, x, from_state=0, max_time=20, num_points=100):
    """Plot transition probabilities over time from a given state."""
    times = np.linspace(0, max_time, num_points)
    state_labels = ["Healthy", "Diseased", "Death"]
    
    # Initialize array to store probabilities
    probs = np.zeros((num_points, model.num_states))
    
    # Calculate probabilities at each time point
    for i, t in enumerate(times):
        if i == 0:
            # At t=0, probability of being in the initial state is 1
            probs[i, from_state] = 1.0
        else:
            p = model(x, time_start=0.0, time_end=t, from_state=from_state)
            probs[i] = p.squeeze().detach().numpy()
    
    # Plot the probabilities
    plt.figure(figsize=(10, 6))
    for j in range(model.num_states):
        plt.plot(times, probs[:, j], label=f"P(State {j}: {state_labels[j]})")
    
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.title(f"Transition Probabilities from State {from_state} ({state_labels[from_state]})")
    plt.legend()
    plt.grid(alpha=0.3)
    return plt.gca()


def main():
    """Run the continuous-time multistate example."""
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=500)
    print(f"Generated {len(data)} transitions")
    
    # Define the state transition structure
    # 3-state illness-death model: Healthy -> Diseased -> Death, with direct transition Healthy -> Death
    state_transitions = {
        0: [1, 2],  # Healthy can transition to Diseased or Death
        1: [2],     # Diseased can transition to Death
        2: []       # Death is an absorbing state
    }
    
    # Create and initialize a continuous-time multistate model
    print("Creating continuous-time model...")
    model = ContinuousMultiStateNN(
        input_dim=2,                 # 2 features
        hidden_dims=[32, 16],        # 2 hidden layers
        num_states=3,                # 3 states (Healthy, Diseased, Death)
        state_transitions=state_transitions
    )
    
    # Manually set some weights to create interesting dynamics
    # (In practice, you would train the model on data)
    with torch.no_grad():
        # Set weights in the feature network for demonstration
        for i, layer in enumerate(model.feature_net):
            if isinstance(layer, torch.nn.Linear):
                # Initialize with small random values
                layer.weight.data = torch.randn_like(layer.weight) * 0.1
                layer.bias.data = torch.randn_like(layer.bias) * 0.01
        
        # Set weights in the intensity network
        model.intensity_net.weight.data = torch.randn_like(model.intensity_net.weight) * 0.1
        model.intensity_net.bias.data = torch.randn_like(model.intensity_net.bias) * 0.01
        
        # Ensure positive rates for allowed transitions
        # This is just for demonstration; normally these would be learned
        # First covariate increases risk of disease, second increases risk of death
        model.intensity_net.bias.data = model.intensity_net.bias.data.reshape(3, 3)
        model.intensity_net.bias.data[0, 1] = 0.3  # Healthy -> Diseased
        model.intensity_net.bias.data[0, 2] = 0.1  # Healthy -> Death
        model.intensity_net.bias.data[1, 2] = 0.5  # Diseased -> Death
        model.intensity_net.bias.data = model.intensity_net.bias.data.reshape(-1)
    
    # Create example patients with different risk profiles
    low_risk = torch.tensor([[-1.0, -1.0]])   # Low on both risk factors
    medium_risk = torch.tensor([[0.0, 0.0]])  # Average risk
    high_risk = torch.tensor([[1.0, 1.0]])    # High on both risk factors
    disease_risk = torch.tensor([[1.0, -1.0]])  # High disease risk, low death risk
    death_risk = torch.tensor([[-1.0, 1.0]])    # Low disease risk, high death risk
    
    # Plot intensity matrices for different risk profiles
    print("Plotting intensity matrices...")
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    plot_intensity_heatmap(model, low_risk, "Low Risk Patient")
    
    plt.subplot(2, 3, 2)
    plot_intensity_heatmap(model, medium_risk, "Medium Risk Patient")
    
    plt.subplot(2, 3, 3)
    plot_intensity_heatmap(model, high_risk, "High Risk Patient")
    
    plt.subplot(2, 3, 4)
    plot_intensity_heatmap(model, disease_risk, "High Disease, Low Death Risk")
    
    plt.subplot(2, 3, 5)
    plot_intensity_heatmap(model, death_risk, "Low Disease, High Death Risk")
    
    plt.tight_layout()
    plt.savefig("intensity_matrices.png", dpi=300)
    
    # Plot transition probabilities over time
    print("Plotting transition probabilities...")
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    plot_transition_probs(model, low_risk, from_state=0, max_time=20)
    plt.title("Low Risk Patient")
    
    plt.subplot(2, 3, 2)
    plot_transition_probs(model, medium_risk, from_state=0, max_time=20)
    plt.title("Medium Risk Patient")
    
    plt.subplot(2, 3, 3)
    plot_transition_probs(model, high_risk, from_state=0, max_time=20)
    plt.title("High Risk Patient")
    
    plt.subplot(2, 3, 4)
    plot_transition_probs(model, disease_risk, from_state=0, max_time=20)
    plt.title("High Disease, Low Death Risk")
    
    plt.subplot(2, 3, 5)
    plot_transition_probs(model, death_risk, from_state=0, max_time=20)
    plt.title("Low Disease, High Death Risk")
    
    plt.tight_layout()
    plt.savefig("transition_probabilities.png", dpi=300)
    
    # Simulate trajectories
    print("Simulating patient trajectories...")
    high_risk_trajectories = simulate_continuous_patient_trajectory(
        model=model,
        x=high_risk,
        start_state=0,
        max_time=20,
        n_simulations=100,
        time_step=0.1,
        seed=42
    )
    
    # Plot a few example trajectories
    plt.figure(figsize=(10, 6))
    state_colors = ['green', 'orange', 'red']
    state_labels = ["Healthy", "Diseased", "Death"]
    
    for i in range(min(10, len(high_risk_trajectories))):
        traj = high_risk_trajectories[i]
        plt.step(traj['time'], traj['state'], where='post', alpha=0.7, 
                 label=f"Patient {i}" if i < 3 else None)
        
        # Add colored markers at state changes
        for state in range(3):
            state_points = traj[traj['state'] == state]
            plt.scatter(state_points['time'], state_points['state'], 
                        color=state_colors[state], s=30)
    
    plt.yticks([0, 1, 2], state_labels)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("Example Patient Trajectories (High Risk)")
    plt.grid(alpha=0.3)
    if len(high_risk_trajectories) >= 3:
        plt.legend()
    plt.savefig("patient_trajectories.png", dpi=300)
    
    # Calculate and plot proportions in each state over time
    print("Analyzing state occupancy probabilities...")
    time_points = np.linspace(0, 20, 100)
    state_counts = np.zeros((len(time_points), 3))
    
    for traj in high_risk_trajectories:
        for i, t in enumerate(time_points):
            # Find state at time t
            state = traj.iloc[np.searchsorted(traj['time'].values, t) - 1]['state']
            state_counts[i, int(state)] += 1
    
    # Convert to proportions
    state_props = state_counts / len(high_risk_trajectories)
    
    # Plot state occupancy
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(time_points, state_props[:, i], 
                 label=f"State {i}: {state_labels[i]}", 
                 color=state_colors[i], linewidth=2)
    
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.title("State Occupancy Probabilities Over Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("state_occupancy.png", dpi=300)
    
    print("Example completed! Results saved as PNG files.")


if __name__ == "__main__":
    main()