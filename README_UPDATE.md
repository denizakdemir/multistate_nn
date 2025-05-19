# Package Structure Update: Moved Example Utilities

As of the latest updates, the example utility functions have been moved into the main package structure for easier access and better integration. This means you no longer need to copy or import custom utility functions from the examples directory.

## Import Changes

Previously, you might have used:
```python
from example_utils import visualize_state_distribution, create_fixed_profile
```

Now, you should use:
```python
from multistate_nn.utils.example_utils import visualize_state_distribution, create_fixed_profile
```

## Available Utilities

The `multistate_nn.utils.example_utils` module contains:

- `visualize_state_distribution`: Visualize state distributions over time
- `visualize_state_distribution_over_time`: Enhanced visualization with more customization options
- `create_fixed_profile`: Create a profile with fixed feature values for model prediction
- `create_covariate_profiles`: Generate profiles with different covariate combinations
- `analyze_covariate_effect`: Analyze and visualize how a covariate affects transitions
- `plot_transition_curves`: Plot transition probabilities over time
- And more helpful utilities for analysis and visualization

## Benefits of This Change

1. **Consistent imports**: All package functionality is now available through standard imports
2. **Better discoverability**: Tools are more visible as part of the package's public API
3. **Improved maintenance**: Example notebooks now use the same code paths as users
4. **Documentation support**: Functions are now part of the documented API

## Example Notebooks

All example notebooks have been updated to use the new import paths. If you have created your own notebooks based on the examples, you should update your imports as shown above.

## How to Check Your Notebooks

If you see errors like:
```
ModuleNotFoundError: No module named 'example_utils'
```

You should update your imports to use the new path.