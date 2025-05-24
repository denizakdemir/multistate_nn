"""
Simple Bayesian MultiStateNN Example
===================================

This example demonstrates how to use the Bayesian extension of MultiStateNN
for uncertainty quantification in multistate modeling.

We'll model a simple 3-state system:
- State 0: Healthy
- State 1: Sick  
- State 2: Recovered (absorbing)

The Bayesian model provides uncertainty estimates for predictions.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import MultiStateNN components
from multistate_nn import fit, ModelConfig, TrainConfig

# Check if Pyro is available for Bayesian modeling
try:
    import pyro
    from multistate_nn.extensions.bayesian import BayesianContinuousMultiStateNN
    PYRO_AVAILABLE = True
    print("✓ Pyro is available - we can use Bayesian models!")
except ImportError:
    PYRO_AVAILABLE = False
    print("✗ Pyro not available. Install with: pip install pyro-ppl")
    exit(1)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
pyro.set_rng_seed(42)

print("=" * 60)
print("Simple Bayesian MultiStateNN Example")
print("=" * 60)

# 1. CREATE SYNTHETIC DATA
print("\n1. Creating synthetic data...")

# Define our simple 3-state system
state_transitions = {
    0: [1],     # Healthy → Sick
    1: [0, 2],  # Sick → Healthy or Recovered  
    2: []       # Recovered (absorbing state)
}

# Generate synthetic patient data
n_patients = 100
data = []

for patient_id in range(n_patients):
    # Patient features: age (normalized), treatment (0=control, 1=treatment)
    age = np.random.normal(50, 15)  # Age around 50
    treatment = np.random.choice([0, 1])  # Random treatment assignment
    age_normalized = (age - 50) / 15  # Normalize age
    
    # Simulate transitions based on features
    current_state = 0  # Start healthy
    current_time = 0.0
    
    # Simulate up to 3 transitions or 12 months
    for _ in range(3):
        if current_state == 2:  # Recovered (absorbing)
            break
            
        # Transition rates depend on age and treatment
        if current_state == 0:  # Healthy → Sick
            base_rate = 0.2
            rate = base_rate * (1 + 0.3 * age_normalized) * (0.7 if treatment else 1.0)
            time_to_event = np.random.exponential(1/rate)
            
            if current_time + time_to_event < 12.0:  # Within 12 months
                data.append({
                    'patient_id': patient_id,
                    'time_start': current_time,
                    'time_end': current_time + time_to_event,
                    'from_state': current_state,
                    'to_state': 1,  # Sick
                    'age_norm': age_normalized,
                    'treatment': treatment,
                    'censored': 0
                })
                current_state = 1
                current_time += time_to_event
            else:
                # Censored observation
                data.append({
                    'patient_id': patient_id,
                    'time_start': current_time,
                    'time_end': 12.0,
                    'from_state': current_state,
                    'to_state': current_state,
                    'age_norm': age_normalized,
                    'treatment': treatment,
                    'censored': 1
                })
                break
                
        elif current_state == 1:  # Sick → Healthy or Recovered
            # Recovery rate (higher with treatment)
            recovery_rate = 0.3 * (1.5 if treatment else 1.0)
            # Return to healthy rate
            return_rate = 0.1
            
            total_rate = recovery_rate + return_rate
            time_to_event = np.random.exponential(1/total_rate)
            
            if current_time + time_to_event < 12.0:
                # Decide which transition
                if np.random.random() < recovery_rate / total_rate:
                    next_state = 2  # Recovered
                else:
                    next_state = 0  # Back to healthy
                    
                data.append({
                    'patient_id': patient_id,
                    'time_start': current_time,
                    'time_end': current_time + time_to_event,
                    'from_state': current_state,
                    'to_state': next_state,
                    'age_norm': age_normalized,
                    'treatment': treatment,
                    'censored': 0
                })
                current_state = next_state
                current_time += time_to_event
            else:
                # Censored
                data.append({
                    'patient_id': patient_id,
                    'time_start': current_time,
                    'time_end': 12.0,
                    'from_state': current_state,
                    'to_state': current_state,
                    'age_norm': age_normalized,
                    'treatment': treatment,
                    'censored': 1
                })
                break

# Convert to DataFrame
df = pd.DataFrame(data)
print(f"Generated {len(df)} transition records for {n_patients} patients")
print(f"Censoring rate: {df['censored'].mean():.1%}")

# 2. TRAIN BAYESIAN MODEL
print("\n2. Training Bayesian model...")

# Model configuration
model_config = ModelConfig(
    input_dim=2,  # age_norm, treatment
    hidden_dims=[16, 8],  # Small network for quick training
    num_states=3,
    state_transitions=state_transitions,
    model_type="continuous",
    bayesian=True  # Enable Bayesian inference
)

# Training configuration - fewer epochs for quick demo
train_config = TrainConfig(
    epochs=5,  # Quick training for demo
    batch_size=32,
    learning_rate=0.01
)

# Train the Bayesian model
bayesian_model = fit(
    df=df,
    covariates=['age_norm', 'treatment'],
    model_config=model_config,
    train_config=train_config,
    time_start_col='time_start',
    time_end_col='time_end',
    censoring_col='censored'
)

print("✓ Bayesian model training completed!")

# 3. MAKE BAYESIAN PREDICTIONS WITH UNCERTAINTY
print("\n3. Making predictions with uncertainty quantification...")

# Define test profiles
test_profiles = torch.tensor([
    [-1.0, 0],   # Young patient, no treatment
    [-1.0, 1],   # Young patient, with treatment  
    [1.0, 0],    # Older patient, no treatment
    [1.0, 1],    # Older patient, with treatment
], dtype=torch.float32)

profile_names = [
    "Young, No Treatment",
    "Young, With Treatment", 
    "Older, No Treatment",
    "Older, With Treatment"
]

# Predict probabilities at 6 months
print(f"\nPredicting state probabilities at 6 months from initial state:")
print("-" * 60)

predictions = []
for i, (profile, name) in enumerate(zip(test_profiles, profile_names)):
    # Get prediction (single forward pass - point estimate)
    probs = bayesian_model.predict_proba(
        profile.unsqueeze(0), 
        time_start=0.0, 
        time_end=6.0, 
        from_state=0
    ).detach().numpy()[0]
    
    predictions.append(probs)
    
    print(f"{name}:")
    print(f"  P(Healthy)   = {probs[0]:.3f}")
    print(f"  P(Sick)      = {probs[1]:.3f}")
    print(f"  P(Recovered) = {probs[2]:.3f}")
    print()

# 4. UNCERTAINTY QUANTIFICATION THROUGH SAMPLING
print("4. Quantifying prediction uncertainty...")

def get_prediction_samples(model, x, n_samples=50):
    """Get multiple prediction samples from Bayesian model."""
    samples = []
    
    for _ in range(n_samples):
        with pyro.poutine.trace() as tr:
            # Sample from posterior by doing forward passes
            probs = model.predict_proba(x, time_start=0.0, time_end=6.0, from_state=0)
            samples.append(probs.detach().numpy())
    
    return np.array(samples)

# Get uncertainty estimates for the young patient with treatment
young_treated = test_profiles[1:2]  # Young, with treatment
print(f"Getting uncertainty estimates for: {profile_names[1]}")

# Get prediction samples
samples = get_prediction_samples(bayesian_model, young_treated, n_samples=30)

# Calculate statistics
mean_pred = np.mean(samples, axis=0)[0]
std_pred = np.std(samples, axis=0)[0]
q05 = np.percentile(samples, 5, axis=0)[0]
q95 = np.percentile(samples, 95, axis=0)[0]

print(f"\nUncertainty Analysis for {profile_names[1]}:")
print("-" * 50)
print(f"State          Mean    Std     5%      95%")
print(f"Healthy      {mean_pred[0]:.3f}   {std_pred[0]:.3f}   {q05[0]:.3f}   {q95[0]:.3f}")
print(f"Sick         {mean_pred[1]:.3f}   {std_pred[1]:.3f}   {q05[1]:.3f}   {q95[1]:.3f}")
print(f"Recovered    {mean_pred[2]:.3f}   {std_pred[2]:.3f}   {q05[2]:.3f}   {q95[2]:.3f}")

# 5. VISUALIZE PREDICTIONS AND UNCERTAINTY
print("\n5. Creating visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Point predictions comparison
states = ['Healthy', 'Sick', 'Recovered']
x_pos = np.arange(len(profile_names))
width = 0.25

for i, state in enumerate(states):
    values = [pred[i] for pred in predictions]
    ax1.bar(x_pos + i*width, values, width, label=state, alpha=0.8)

ax1.set_xlabel('Patient Profile')
ax1.set_ylabel('Probability')
ax1.set_title('Predicted State Probabilities at 6 Months')
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels(profile_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Uncertainty for young treated patient
state_names = ['Healthy', 'Sick', 'Recovered']
x_pos2 = np.arange(len(state_names))

# Plot means with error bars (90% confidence intervals)
ax2.bar(x_pos2, mean_pred, yerr=[mean_pred - q05, q95 - mean_pred], 
        capsize=5, alpha=0.7, color=['lightblue', 'orange', 'lightgreen'])

ax2.set_xlabel('State')
ax2.set_ylabel('Probability')
ax2.set_title(f'Uncertainty Estimates\n{profile_names[1]}')
ax2.set_xticks(x_pos2)
ax2.set_xticklabels(state_names)
ax2.grid(True, alpha=0.3)

# Add uncertainty information as text
for i, (mean, std, low, high) in enumerate(zip(mean_pred, std_pred, q05, q95)):
    ax2.text(i, mean + 0.05, f'±{std:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('bayesian_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. TREATMENT EFFECT ANALYSIS
print("\n6. Analyzing treatment effect...")

# Compare young patients with and without treatment
young_no_treatment = test_profiles[0:1]
young_with_treatment = test_profiles[1:2]

# Get predictions
prob_no_tx = bayesian_model.predict_proba(young_no_treatment, time_start=0.0, time_end=6.0, from_state=0).detach().numpy()[0]
prob_with_tx = bayesian_model.predict_proba(young_with_treatment, time_start=0.0, time_end=6.0, from_state=0).detach().numpy()[0]

print("Treatment Effect Analysis (Young Patients):")
print("-" * 45)
print(f"Outcome          No Treatment    With Treatment    Difference")
print(f"P(Healthy)         {prob_no_tx[0]:.3f}           {prob_with_tx[0]:.3f}          {prob_with_tx[0]-prob_no_tx[0]:+.3f}")
print(f"P(Sick)            {prob_no_tx[1]:.3f}           {prob_with_tx[1]:.3f}          {prob_with_tx[1]-prob_no_tx[1]:+.3f}")
print(f"P(Recovered)       {prob_no_tx[2]:.3f}           {prob_with_tx[2]:.3f}          {prob_with_tx[2]-prob_no_tx[2]:+.3f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("✓ Successfully trained a Bayesian multistate model")
print("✓ Made predictions with uncertainty quantification")
print("✓ Analyzed treatment effects")
print("✓ Visualized results with confidence intervals")
print("\nKey Benefits of Bayesian Approach:")
print("• Provides uncertainty estimates for predictions")
print("• Enables confidence intervals and risk assessment")
print("• Helps identify when more data is needed")
print("• Supports robust decision-making under uncertainty")

print(f"\nVisualization saved as 'bayesian_predictions.png'")