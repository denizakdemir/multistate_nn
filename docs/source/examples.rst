Examples
========

MultiStateNN comes with several example notebooks demonstrating different use cases and features. You can find these notebooks in the `examples` directory of the repository.

Available Examples
----------------

Disease Progression Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **PBC Analysis Example**: Analysis of Primary Biliary Cirrhosis (PBC) disease progression using continuous-time multistate models.
- **PBC Advanced Architectures**: Advanced neural architectures for modeling PBC disease progression.
- **AIDS/SI Real Data Example**: Analysis of AIDS/SI switching data with continuous-time models.
- **Bladder Cancer Bayesian Example**: Bayesian analysis of bladder cancer recurrence using continuous-time models.

Methodological Examples
^^^^^^^^^^^^^^^^^^^^

- **Disease Progression Example**: Fundamental approach to disease progression modeling with continuous-time models.
- **Survival Analysis with Censoring**: Handling right-censored observations in survival analysis.

Running Examples
--------------

To run the examples, you'll need to install MultiStateNN with the examples extras:

.. code-block:: bash

    pip install multistate-nn[examples]

You'll also need to download the example datasets:

.. code-block:: bash

    chmod +x scripts/download_data.py scripts/setup_data.sh
    ./scripts/setup_data.sh

This will create a `data` folder with commonly used multistate datasets:
- CAV (heart transplant data)
- Bladder cancer recurrence data
- Primary biliary cirrhosis data
- AIDS/SI switching data