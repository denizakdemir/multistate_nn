# MultiStateNN Testing Framework

This document describes the testing framework for the MultiStateNN package and provides guidance on how to use and extend the tests.

## Test Suite Structure

The test suite is organized into several modules that test different components of the package:

1. **Core Model Tests**
   - `test_multistate_nn.py`: Tests core functionality of the MultiStateNN model
   - `test_model_architecture.py`: Tests model architecture and initialization
   - `test_model_variations.py`: Tests model with varying configurations

2. **Utility Tests**
   - `test_simulation.py`: Tests simulation capabilities
   - `test_time_mapping.py`: Tests time mapping functionality
   - `test_censoring.py`: Tests censoring mechanisms
   - `test_cif_calculation.py`: Tests cumulative incidence function calculation
   - `test_cif_discretization.py`: Tests CIF calculation with different time discretizations
   - `test_cif_properties.py`: Tests mathematical properties of CIFs

3. **Analysis Tests**
   - `test_analysis.py`: Tests analysis functions
   - `test_visualization.py`: Tests visualization functions

4. **End-to-End Workflow Tests**
   - `test_end_to_end_workflow.py`: Tests complete workflow from data preparation to visualization
   - `test_bayesian_workflow.py`: Tests Bayesian model workflow with uncertainty quantification
   - `test_real_data_import.py`: Tests model training with real-world datasets

## Test Categories

The tests can be categorized as follows:

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test the interaction between components
3. **Validation Tests**: Test against known ground truth or analytical solutions
4. **Property-Based Tests**: Test that mathematical properties hold
5. **Edge Case Tests**: Test behavior in extreme scenarios

## Running Tests

### Running All Tests

```bash
pytest tests/
```

### Running Tests with Coverage

```bash
pytest tests/ --cov=multistate_nn
```

### Running Specific Test Categories

```bash
# Run only CIF-related tests
pytest tests/test_cif*

# Run only time mapping tests
pytest tests/test_time_mapping.py

# Run a specific test function
pytest tests/test_cif_calculation.py::test_basic_cif_from_known_trajectories
```

## Test Data

Tests use a combination of:

1. **Synthetic Data**: Generated using `generate_synthetic_data()`
2. **Analytical Solutions**: Calculated for simple test cases using known formulas
3. **Reference Implementations**: Compared with established methods where available

## Key Testing Principles

1. **Ground Truth Validation**: Tests verify that CIF calculation and simulation methods produce results consistent with analytical solutions for simple test cases.

2. **Time Discretization Robustness**: Tests ensure that CIF calculations produce similar results regardless of the time discretization used.

3. **Censoring Handling**: Tests verify that censoring is correctly handled in both simulation and CIF calculation.

4. **Mathematical Properties**: Tests ensure that CIFs are monotonic and satisfies other mathematical properties.

5. **Model Configuration Flexibility**: Tests ensure that the model can be configured with different architectures and parameters.

## Extending the Tests

When adding new features to the package, you should add corresponding tests that verify:

1. **Correctness**: The feature produces the expected results
2. **Robustness**: The feature handles edge cases and invalid inputs
3. **Integration**: The feature works correctly with existing components
4. **Performance**: The feature performs efficiently

## Known Limitations

1. **Test Performance**: Some tests with simulation can be time-consuming. Use the optimized versions for quick testing.

2. **Randomization**: Some tests involve randomization, which can occasionally lead to false failures in edge cases, especially with small simulation sizes.

3. **Coverage Gaps**: Some edge cases and complex interactions may not be covered by the current tests.

## Future Test Improvements

1. **Reference Implementation Comparison**: Add tests comparing with R's msm and mstate packages.

2. **Property-Based Testing**: Extend property-based tests using frameworks like Hypothesis.

3. **Benchmark Tests**: Add performance benchmark tests.

✅ **End-to-End Workflow Tests**: Comprehensive tests covering the entire workflow from data preparation to visualization have been implemented in `test_end_to_end_workflow.py`, `test_bayesian_workflow.py`, and `test_real_data_import.py`.

## Test Debugging

If a test fails, you can use the following approaches to debug:

1. Run the test with increased verbosity:
   ```bash
   pytest tests/test_failed.py -v
   ```

2. Add print statements to the test to see intermediate values.

3. Check if the failure is due to randomization by running with different seeds.

4. For simulation-based tests, increase the simulation size to reduce variance.

5. For tests with analytical solutions, verify the formulas and parameters used.

## Contribution Guide for Testing

When contributing to the testing framework:

1. Follow the existing test organization
2. Use clear, descriptive test names
3. Include tests for both normal and edge cases
4. Add comments explaining test logic and expected outcomes
5. Keep tests independent and idempotent
6. Include fixtures for reusable test setup
7. Use parameterization for testing multiple similar cases
8. For slow tests, add markers to allow skipping during quick runs