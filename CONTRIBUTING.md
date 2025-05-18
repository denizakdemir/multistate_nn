# Contributing to MultiStateNN

Thank you for your interest in contributing to MultiStateNN! This document provides guidelines and instructions for contributing to our continuous-time multistate modeling package.

## Important Note on Model Architecture

As of version 0.4.0, MultiStateNN has fully transitioned to continuous-time models using Neural ODEs. All new contributions should focus on the continuous-time implementation.

- **DO NOT** add discrete-time model functionality
- **DO** focus on enhancing the continuous-time capabilities
- **DO** align with the Neural ODE architecture and principles

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/multistate_nn.git
cd multistate_nn
```

2. Create a virtual environment and install development dependencies:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e ".[dev]"
```

3. Install additional dependencies for Bayesian features:
```bash
pip install -e ".[dev,bayesian]"
```

## Running Tests

Run the test suite with pytest:
```bash
pytest tests/
```

To run tests with coverage:
```bash
pytest tests/ --cov=multistate_nn
```

For specific test categories:
```bash
# Run only continuous-time model tests
pytest tests/test_continuous_models.py

# Run only Bayesian model tests
pytest tests/test_bayesian_continuous.py

# Run simulation tests
pytest tests/test_continuous_simulation.py
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function arguments and return values
- Write docstrings in NumPy format
- Keep functions focused and concise
- Add tests for new functionality
- Format code with black before submitting

```bash
black multistate_nn/ tests/
```

## Pull Request Process

1. Create a new branch for your feature/fix
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a pull request with a clear description of changes

## Adding New Features

When adding new features:

1. Start with a test case in `tests/`
2. Implement the feature
3. Add example usage in docstrings
4. Update the example notebook if relevant
5. Add any new dependencies to `pyproject.toml`

### Feature Categories

Focus your contributions in these areas:

1. **Advanced ODE solvers**: Improved numerical methods for solving the continuous-time equations
2. **Specialized neural architectures**: New architectures for the intensity function
3. **Bayesian extensions**: Enhancements to the Bayesian continuous-time model
4. **Validation tools**: Methods to validate models against established approaches
5. **Visualization**: Better tools for visualizing continuous-time processes
6. **Performance improvements**: Optimizations for faster training and prediction

## Code Structure

Maintain the package's structure:

- `models.py`: Core continuous-time model classes
- `train.py`: Training utilities for continuous-time models
- `extensions/`: Bayesian and other extensions
- `utils/`: Supporting utilities
- `architectures.py`: Neural architectures for intensity functions
- `losses.py`: Loss functions for different modeling scenarios

## Mathematical Correctness

When contributing to mathematical components:

1. Ensure intensity matrices satisfy Markov process constraints:
   - Non-negative off-diagonal elements
   - Rows sum to zero
2. Validate ODE solutions against analytical solutions for simple cases
3. Confirm consistency across different time scales
4. Test with both small and large time intervals

## Questions and Support

- Open an issue for bug reports
- Use discussions for feature requests and questions
- Tag issues appropriately

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Follow project guidelines
- Help others learn and grow

Thank you for contributing!