# Handling Censoring in Continuous-Time MultiStateNN

This guide explains how MultiStateNN handles right-censored observations in continuous-time multistate models.

## What is Censoring?

Censoring occurs when the exact time of a state transition is not observed, but we know it occurs after a certain time point. In medical studies, this typically happens when:

- A patient drops out of the study
- The study ends before the event of interest occurs
- A patient is lost to follow-up

In continuous-time multistate models, proper handling of censoring is essential for obtaining unbiased estimates of transition probabilities.

## Types of Censoring Supported

MultiStateNN provides robust support for:

- **Right censoring**: When a subject's event time is only known to be greater than a certain value
- **Independent censoring**: When the censoring mechanism is independent of the transition process

## How MultiStateNN Handles Censoring

### 1. Data Representation

Censored observations can be specified in your data using a dedicated censoring column:

```python
data = pd.DataFrame({
    'time_start': [0.0, 0.0, 1.2, 1.5, 1.7],
    'time_end': [1.2, 1.0, 1.8, 2.2, 2.5],
    'from_state': [0, 0, 1, 1, 2],
    'to_state': [1, 2, 2, 3, 2],  # State at time_end
    'covariates': [...],
    'censored': [0, 0, 0, 0, 1]   # 1 indicates censoring
})
```

When an observation is censored:
- `time_end` represents the last observation time
- `to_state` represents the last observed state
- `censored` column is set to 1

### 2. Modified Loss Function

The continuous-time model uses a specialized loss function that properly accounts for censored observations:

```python
class ContinuousTimeMultiStateLoss(nn.Module):
    """Loss function for continuous-time multistate models with censoring support."""
    
    def forward(
        self, 
        model,
        x: torch.Tensor,
        time_start: torch.Tensor,
        time_end: torch.Tensor,
        from_state: torch.Tensor,
        to_state: torch.Tensor,
        is_censored: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss with proper censoring handling."""
        # ...
        
        for i in range(batch_size):
            # ... 
            
            if not censored_i:
                # For observed transitions, maximize probability of observed transition
                loss = loss - torch.log(torch.clamp(probs[to_i], min=1e-8))
            else:
                # For censored data, we know the subject stayed in from_state
                # Maximize probability of staying in the same state
                loss = loss - torch.log(torch.clamp(probs[from_i], min=1e-8))
```

### 3. Training with Censoring Information

When training a model, you can specify the censoring column:

```python
model = fit(
    df=data,
    covariates=['age', 'biomarker'],
    model_config=model_config,
    train_config=train_config,
    time_start_col='time_start',
    time_end_col='time_end',
    censoring_col='censored'  # Specify column containing censoring information
)
```

### 4. Bayesian Handling of Censoring

For Bayesian models, censoring is handled within the probabilistic model:

```python
def model(
    self,
    x: torch.Tensor,
    time_start: torch.Tensor,
    time_end: torch.Tensor,
    from_state: torch.Tensor,
    to_state: Optional[torch.Tensor] = None,
    is_censored: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pyro model with censoring support."""
    # ...
    
    # For uncensored observations
    if not (is_censored is not None and is_censored[i].item()):
        pyro.sample(
            f"obs_{i}",
            dist.Categorical(probs=probs),
            obs=torch.tensor(to_state_i, device=x.device)
        )
    # For censored observations
    else:
        # For censored data, condition on survival (staying in current state)
        pyro.factor(
            f"censored_{i}",
            torch.log(torch.clamp(probs[from_state_i], min=1e-8))
        )
```

### 5. Simulation with Censoring

The simulation functions support generating censored trajectories:

```python
trajectories = simulate_continuous_patient_trajectory(
    model=model,
    x=x_new[0:1],
    start_state=0,
    max_time=5.0,
    n_simulations=100,
    censoring_rate=0.3  # 30% of simulated trajectories will be censored
)
```

The censoring time is generated from an exponential distribution calibrated to achieve the target censoring rate.

## Mathematical Background

In continuous-time Markov models, censoring is handled by considering the likelihood contribution for censored individuals.

For a subject observed in state i at time s and censored at time t:
- We know the subject was in state i at time t
- The likelihood contribution is the probability of staying in state i from time s to time t: P_ii(s,t)

For a subject observed transitioning from state i to state j between times s and t:
- The likelihood contribution is the transition probability: P_ij(s,t)

The continuous-time model with Neural ODEs allows us to calculate these probabilities directly by solving the intensity matrix ODE.

## Checking for Censoring

If you're unsure whether your dataset contains censored observations, you can check with:

```python
# Check if there are any censored observations
has_censoring = data['censored'].sum() > 0
print(f"Dataset contains censored observations: {has_censoring}")

# Check censoring rate
censoring_rate = data['censored'].mean()
print(f"Censoring rate: {censoring_rate:.2%}")
```

## Best Practices

1. **Always include censoring information** when available
2. **Use the appropriate loss function** for your data (the default handles censoring)
3. **Ensure your censoring mechanism is independent** of the event process
4. **Validate your model with simulated data** including censoring
5. **Compare results with and without censoring** to understand its impact

## Informative Censoring

The current implementation assumes independent censoring. For informative censoring (where the censoring mechanism depends on the event process), specialized techniques may be required. Future versions of MultiStateNN will address this case.

## References

For more details on censoring in continuous-time Markov models:

1. Andersen, P. K., & Keiding, N. (2002). Multi-state models for event history analysis. Statistical Methods in Medical Research, 11(2), 91-115.
2. Aalen, O., Borgan, O., & Gjessing, H. (2008). Survival and event history analysis: a process point of view. Springer Science & Business Media.