Quick Start
==========

This guide will help you get started with MultiStateNN by showing a complete example of setting up, training, and using a continuous-time multistate model.

Basic Example
-----------

Here's a complete example of training a continuous-time multistate model on sample data with censoring:

.. code-block:: python

    import pandas as pd
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from multistate_nn import ContinuousMultiStateNN, fit
    from multistate_nn.train import ModelConfig, TrainConfig

    # Prepare your data with censoring information
    data = pd.DataFrame({
        'time_start': [0.0, 0.0, 1.2, 1.5, 1.7, 2.0, 2.3],
        'time_end': [1.2, 1.0, 1.8, 2.2, 2.5, 3.0, 3.2],
        'from_state': [0, 0, 1, 1, 2, 1, 2],
        'to_state': [1, 2, 2, 3, 2, 1, 2],
        'age': [65, 70, 55, 75, 60, 62, 68],
        'biomarker': [1.2, 0.8, 1.5, 0.9, 1.1, 1.0, 1.3],
        'censored': [0, 0, 0, 0, 1, 1, 1]  # Censoring indicator (1=censored, 0=observed)
    })

    # Define state transitions
    state_transitions = {
        0: [1, 2],    # From state 0, can transition to 1 or 2
        1: [1, 2, 3], # From state 1, can stay in 1 or go to 2 or 3
        2: [2, 3],    # From state 2, can stay in 2 or go to 3
        3: []         # State 3 is absorbing
    }

    # Define model configuration
    model_config = ModelConfig(
        input_dim=2,              # Number of input features (age, biomarker)
        hidden_dims=[64, 32],     # Hidden layer dimensions
        num_states=4,             # Total number of states (0-3)
        state_transitions=state_transitions,
        model_type="continuous",  # Specify continuous-time model
    )

    # Define training configuration
    train_config = TrainConfig(
        batch_size=32,
        epochs=100,
        learning_rate=0.005,
        solver="dopri5",          # ODE solver to use
        solver_options={"rtol": 1e-3, "atol": 1e-4}  # Solver tolerance options
    )

    # Fit the model with explicit censoring information
    model = fit(
        df=data,
        covariates=['age', 'biomarker'],
        model_config=model_config,
        train_config=train_config,
        time_start_col='time_start',  # Specify column containing start times
        time_end_col='time_end',      # Specify column containing end times
        censoring_col='censored'      # Specify column containing censoring information
    )

    # Make predictions
    x_new = torch.tensor([[70, 1.2], [65, 0.8]], dtype=torch.float32)
    probs = model.predict_proba(x_new, time_start=1.0, time_end=2.5, from_state=0)
    print("Transition probabilities:", probs)

    # Calculate transition probabilities over time
    from_state = 0
    to_state = 3
    time_points = np.linspace(0, 5, 50)
    probabilities = []

    for t in time_points:
        prob = model.predict_proba(
            x_new[0:1], 
            time_start=0.0, 
            time_end=t, 
            from_state=from_state
        ).detach().numpy()[0, to_state]
        probabilities.append(prob)

    # Plot the probability curve
    plt.figure(figsize=(8, 5))
    plt.plot(time_points, probabilities, 'b-', label=f'P({from_state} → {to_state})')
    plt.xlabel('Time')
    plt.ylabel('Transition Probability')
    plt.title('Continuous-Time Transition Probability')
    plt.legend()
    plt.grid(alpha=0.3)

Simulating Patient Trajectories
-----------------------------

You can simulate patient trajectories through the multistate model:

.. code-block:: python

    from multistate_nn.utils import simulate_continuous_patient_trajectory

    # Simulate trajectories with censoring
    trajectories = simulate_continuous_patient_trajectory(
        model=model,
        x=x_new[0:1],          # Features for a single patient
        start_state=0,
        max_time=5.0,
        n_simulations=100,
        time_step=0.1,         # Time step for simulation grid
        censoring_rate=0.3     # 30% of simulated trajectories will be censored
    )

    # Plot simulated patient trajectories
    import pandas as pd
    import seaborn as sns

    # Combine trajectories for visualization
    traj_df = pd.concat(trajectories)
    traj_df = traj_df[traj_df['grid_point'] == True]  # Use only grid points for cleaner plotting

    # Plot the state distribution over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=traj_df,
        x='time',
        y='state',
        hue='simulation',
        alpha=0.3,
        palette='viridis'
    )
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('Simulated Patient Trajectories')

For more detailed examples, check out the example notebooks in the `examples` directory of the package repository.