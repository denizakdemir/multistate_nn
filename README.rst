MultiStateNN: Neural Network Models for Continuous-Time Multistate Processes
=================================================================

|license|

.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

MultiStateNN is a PyTorch-based package implementing continuous-time multistate models using Neural Ordinary Differential Equations (Neural ODEs). It provides robust support for censored data as the default expectation in time-to-event analysis. The package supports both deterministic and Bayesian inference, making it suitable for modeling state transitions in various applications such as:

- Disease progression modeling with real-world censored patient data
- Survival analysis with competing risks
- Credit risk transitions with right-censored observations
- Career trajectory analysis with incomplete follow-up
- System degradation modeling with censored failure times

Features
--------

- Neural ODE-based implementation for continuous-time dynamics
- Specialized neural architectures for intensity functions (MLP, RNN, Attention)
- Support for arbitrary state transition structures
- Optional Bayesian inference using Pyro (via extensions)
- Proper handling of intensity matrix constraints
- Built-in visualization tools
- Patient trajectory simulation in continuous time
- Support for original time scales (days, years, etc.)

Advanced Censoring Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Comprehensive handling of right-censored observations as the default
- Modified loss function that properly accounts for censored transitions
- Specialized simulation functions that incorporate censoring
- Competing risks analysis with proper censoring adjustments
- Continuous-time intensity matrix formulation for accurate estimates

Version Note
-----------

**IMPORTANT**: Version 0.4.0+ of MultiStateNN has migrated to a fully continuous-time implementation using Neural ODEs. It is **NOT** backward compatible with earlier versions that used discrete-time models. If you need the discrete-time implementation, please use version 0.3.x or earlier.

Installation
-----------

Basic installation::

    pip install multistate-nn

With Bayesian inference support::

    pip install multistate-nn[bayesian]

With example notebook dependencies::

    pip install multistate-nn[examples]

For development::

    pip install -e ".[dev]"

Quick Start
----------

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

    # Fit the model with explicit censoring information
    model = fit(
        df=data,
        covariates=['age', 'biomarker'],
        model_config=model_config,
        train_config=TrainConfig(epochs=100),
        time_start_col='time_start',  # Specify column containing start times
        time_end_col='time_end',      # Specify column containing end times
        censoring_col='censored'      # Specify column containing censoring information
    )

    # Make predictions
    x_new = torch.tensor([[70, 1.2], [65, 0.8]], dtype=torch.float32)
    probs = model.predict_proba(x_new, time_start=1.0, time_end=2.5, from_state=0)
    print("Transition probabilities:", probs)

For more details, see the `full documentation <https://github.com/denizakdemir/multistate_nn>`_.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Citation
--------

If you use this package in your research, please cite:

.. code-block:: bibtex

    @software{multistate_nn2025,
        title={MultiStateNN: Neural Network Models for Continuous-Time Multistate Processes},
        author={Akdemir, Deniz, github: denizakdemir},
        year={2025},
        url={https://github.com/denizakdemir/multistate_nn}
    }