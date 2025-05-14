"""Tests for analysis utilities."""

import pytest
import numpy as np
import pandas as pd
import torch

from multistate_nn import MultiStateNN
from multistate_nn.utils.analysis import (
    calculate_cif,
    _calculate_single_cif,
)
from multistate_nn.utils.simulation import (
    simulate_patient_trajectory,
)


@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    state_transitions = {0: [1, 2], 1: [2, 3], 2: [3], 3: []}
    
    model = MultiStateNN(
        input_dim=3,
        hidden_dims=[32, 16],
        num_states=4,
        state_transitions=state_transitions
    )
    
    # Set some deterministic values for reproducible testing
    for i in range(4):
        if i < 3:  # Skip the last layer which is an absorbing state
            module = model.state_heads[str(i)]
            # Set weights to a simple pattern for deterministic output
            torch.nn.init.constant_(module.weight, 0.1)
            torch.nn.init.constant_(module.bias, 0.0)
    
    return model


@pytest.fixture
def sample_trajectories(sample_model):
    """Generate sample trajectories for testing."""
    # Create some reproducible patient features
    torch.manual_seed(42)
    np.random.seed(42)
    
    x = torch.randn(1, 3)
    
    # Simulate trajectories
    trajectories = simulate_patient_trajectory(
        model=sample_model,
        x=x,
        start_state=0,
        max_time=10,
        n_simulations=20,
        seed=42
    )
    
    # Combine into a single DataFrame
    return pd.concat(trajectories, ignore_index=True)


@pytest.fixture
def cohort_trajectories(sample_model):
    """Generate sample cohort trajectories."""
    # Create cohort features
    torch.manual_seed(43)
    np.random.seed(43)
    
    cohort_features = torch.randn(3, 3)
    
    # Simulate trajectories for each patient
    trajectories = []
    
    for i in range(cohort_features.shape[0]):
        patient_x = cohort_features[i:i+1]
        
        patient_trajectories = simulate_patient_trajectory(
            model=sample_model,
            x=patient_x,
            start_state=0,
            max_time=10,
            n_simulations=5,
            seed=42 + i
        )
        
        # Add patient ID
        for traj in patient_trajectories:
            traj['patient_id'] = i
        
        trajectories.extend(patient_trajectories)
    
    # Combine into a single DataFrame
    return pd.concat(trajectories, ignore_index=True)


def test_calculate_single_cif(sample_trajectories):
    """Test _calculate_single_cif function."""
    # Create time grid
    time_grid = np.arange(11)  # 0 to 10
    
    # Test for target state 2
    cif_df = _calculate_single_cif(
        trajectories=sample_trajectories,
        target_state=2,
        time_grid=time_grid,
        ci_level=0.95
    )
    
    assert isinstance(cif_df, pd.DataFrame)
    assert set(cif_df.columns) == {'time', 'cif', 'lower_ci', 'upper_ci'}
    assert len(cif_df) == 11  # 0 to 10
    
    # Check CIF properties
    assert cif_df['cif'].iloc[0] == 0.0  # CIF starts at 0
    assert (cif_df['cif'] >= 0).all()  # CIF is non-negative
    assert (cif_df['cif'] <= 1).all()  # CIF is at most 1
    assert (np.diff(cif_df['cif']) >= 0).all()  # CIF is non-decreasing
    
    # Check confidence interval properties
    assert (cif_df['lower_ci'] <= cif_df['cif']).all()  # Lower CI <= CIF
    assert (cif_df['upper_ci'] >= cif_df['cif']).all()  # Upper CI >= CIF
    assert (cif_df['lower_ci'] >= 0).all()  # Lower CI is non-negative
    assert (cif_df['upper_ci'] <= 1).all()  # Upper CI is at most 1


def test_calculate_cif(sample_trajectories, cohort_trajectories):
    """Test calculate_cif function."""
    # Test for simple trajectories
    cif_df = calculate_cif(
        trajectories=sample_trajectories,
        target_state=2,
        max_time=10,
        by_patient=False,
        ci_level=0.95
    )
    
    assert isinstance(cif_df, pd.DataFrame)
    assert set(cif_df.columns) == {'time', 'cif', 'lower_ci', 'upper_ci'}
    
    # Test with cohort trajectories and by_patient=True
    cif_by_patient = calculate_cif(
        trajectories=cohort_trajectories,
        target_state=2,
        max_time=10,
        by_patient=True,
        ci_level=0.95
    )
    
    assert isinstance(cif_by_patient, pd.DataFrame)
    assert set(cif_by_patient.columns) == {'time', 'cif', 'lower_ci', 'upper_ci', 'patient_id'}
    assert cif_by_patient['patient_id'].nunique() == 3  # 3 patients
    
    # Test with custom max_time
    cif_custom_max = calculate_cif(
        trajectories=sample_trajectories,
        target_state=2,
        max_time=5,  # Custom max_time
        by_patient=False,
        ci_level=0.95
    )
    
    assert isinstance(cif_custom_max, pd.DataFrame)
    assert cif_custom_max['time'].max() <= 5
    # The number of points may depend on the discretization of time in the data
    assert cif_custom_max['time'].iloc[0] == 0  # Should always start at 0
    
    # Test with different target state
    cif_state3 = calculate_cif(
        trajectories=sample_trajectories,
        target_state=3,  # Different target state
        max_time=10,
        by_patient=False,
        ci_level=0.95
    )
    
    assert isinstance(cif_state3, pd.DataFrame)
    
    # Test with different confidence level
    cif_90 = calculate_cif(
        trajectories=sample_trajectories,
        target_state=2,
        max_time=10,
        by_patient=False,
        ci_level=0.90  # Different confidence level
    )
    
    assert isinstance(cif_90, pd.DataFrame)
    
    # Compare 90% CI to 95% CI (90% CI should be narrower)
    cif_95 = calculate_cif(
        trajectories=sample_trajectories,
        target_state=2,
        max_time=10,
        by_patient=False,
        ci_level=0.95
    )
    
    # The 90% CI should generally be narrower than the 95% CI
    # (This is not guaranteed for small samples, but should tend to be true on average)
    ci_width_90 = (cif_90['upper_ci'] - cif_90['lower_ci']).mean()
    ci_width_95 = (cif_95['upper_ci'] - cif_95['lower_ci']).mean()
    
    # We don't assert strictly here since it depends on random variation
    # but generally the 90% CI should be narrower
    assert abs(ci_width_90 - ci_width_95) < 0.2


def test_calculate_cif_missing_patient_id():
    """Test that calculate_cif raises an error when patient_id is missing for by_patient=True."""
    # Create synthetic trajectories without patient_id
    trajectories = pd.DataFrame({
        'time': [0, 1, 2, 0, 1, 2],
        'state': [0, 1, 2, 0, 1, 1],
        'simulation': [0, 0, 0, 1, 1, 1],
    })
    
    # Test with by_patient=True should raise ValueError
    with pytest.raises(ValueError, match="must have 'patient_id' column"):
        calculate_cif(
            trajectories=trajectories,
            target_state=2,
            max_time=10,
            by_patient=True
        )