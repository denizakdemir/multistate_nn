Contributing
===========

We welcome contributions to MultiStateNN! This document provides guidelines for contributing to the project.

Getting Started
-------------

1. Fork the repository on GitHub.
2. Clone your fork:

   .. code-block:: bash

       git clone https://github.com/yourusername/multistate_nn.git
       cd multistate_nn

3. Create a virtual environment and install development dependencies:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # or venv\\Scripts\\activate on Windows
       pip install -e ".[dev]"

4. Install additional dependencies for Bayesian features:

   .. code-block:: bash

       pip install -e ".[dev,bayesian]"

Development Guidelines
-------------------

Code Style
^^^^^^^^^

- Follow PEP 8 style guidelines
- Use type hints for function arguments and return values
- Write docstrings in NumPy format
- Keep functions focused and concise
- Format code with black before submitting:

  .. code-block:: bash

      black multistate_nn/ tests/

Running Tests
^^^^^^^^^^^

Run the test suite with pytest:

.. code-block:: bash

    pytest tests/

Run tests with coverage:

.. code-block:: bash

    pytest tests/ --cov=multistate_nn

For specific test categories:

.. code-block:: bash

    # Run only continuous-time model tests
    pytest tests/test_continuous_models.py

    # Run only Bayesian model tests
    pytest tests/test_bayesian_continuous.py

Pull Request Process
-----------------

1. Create a new branch for your feature/fix
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a pull request with a clear description of changes

Feature Categories
---------------

Focus your contributions in these areas:

1. **Advanced ODE solvers**: Improved numerical methods for solving the continuous-time equations
2. **Specialized neural architectures**: New architectures for the intensity function
3. **Bayesian extensions**: Enhancements to the Bayesian continuous-time model
4. **Validation tools**: Methods to validate models against established approaches
5. **Visualization**: Better tools for visualizing continuous-time processes
6. **Performance improvements**: Optimizations for faster training and prediction

Code Structure
------------

Maintain the package's structure:

- `models.py`: Core continuous-time model classes
- `train.py`: Training utilities for continuous-time models
- `extensions/`: Bayesian and other extensions
- `utils/`: Supporting utilities
- `architectures.py`: Neural architectures for intensity functions
- `losses.py`: Loss functions for different modeling scenarios

For more details, see the full [Contributing Guidelines](https://github.com/denizakdemir/multistate_nn/blob/main/CONTRIBUTING.md) in the repository.