# Contributing to MultiStateNN

Thank you for your interest in contributing to MultiStateNN! This document provides guidelines and instructions for contributing.

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

## Running Tests

Run the test suite with pytest:
```bash
pytest tests/
```

To run tests with coverage:
```bash
pytest tests/ --cov=multistate_nn

```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function arguments and return values
- Write docstrings in NumPy format
- Keep functions focused and concise
- Add tests for new functionality

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