# Understanding the PBC data structure:
# - status: 0=censored, 1=transplant, 2=dead
# - stage: 1-4, histologic stage of disease (1=best, 4=worst)

# We'll define our states as:
# 0: Stage 1 (early PBC)
# 1: Stage 2 (moderate PBC)
# 2: Stage 3 (advanced PBC)
# 3: Stage 4 (cirrhosis)
# 4: Liver transplant
# 5: Death

import pandas as pd
import numpy as np

def fix_pbc_data(pbc):
    """Fix PBC dataset with proper handling of invalid values."""
    # First, let's identify and fix the stage column
    # Replace implausible values with NaN in all columns
    invalid_value = -2147483648
    for col in pbc.columns:
        pbc[col] = pbc[col].replace(invalid_value, np.nan)
    
    # Fill missing stage values with median
    pbc['stage'] = pbc['stage'].fillna(pbc['stage'].median())
    
    # Create a robust mapping function to convert stage to state
    def stage_to_state(row):
        if row['status'] == 2:  # Death
            return 5
        elif row['status'] == 1:  # Transplant
            return 4
        else:  # Stage-based state
            try:
                # Make sure stage is a valid integer between 1-4
                stage = int(row['stage'])
                # Handle any out-of-range values
                if stage < 1:
                    stage = 1
                elif stage > 4:
                    stage = 4
                return stage - 1  # Convert stage 1-4 to state 0-3
            except (ValueError, TypeError):
                # If conversion fails, default to median stage
                return 2  # Default to Stage 3 (most common)
    
    # Add state column
    pbc['state'] = pbc.apply(stage_to_state, axis=1)
    
    # Show distribution of states
    print("State distribution:")
    print(pbc['state'].value_counts().sort_index())
    
    return pbc