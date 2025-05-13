#!/usr/bin/env python3
"""
Script to download multistate datasets from R packages using rpy2.
This script creates a data folder and downloads common multistate datasets.
"""

import os
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
except ImportError:
    logging.error("rpy2 is required for this script. Please install it with: pip install rpy2")
    exit(1)

# Create data directory
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
logging.info(f"Data will be saved to {data_dir.absolute()}")

# Install necessary R packages if not already installed
def install_r_package(package_name):
    """Install R package if not already installed."""
    try:
        ro.r(f"""
        if (!requireNamespace("{package_name}", quietly = TRUE)) {{
            install.packages("{package_name}", repos="https://cloud.r-project.org")
        }}
        """)
        return True
    except Exception as e:
        logging.error(f"Failed to install R package {package_name}: {e}")
        return False

# Import R packages and save datasets
datasets = []

# 1. EBMT dataset from mstate package
if install_r_package("mstate"):
    try:
        mstate = importr('mstate')
        ro.r("data(ebmt, package='mstate')")
        ebmt = ro.r("ebmt")
        ebmt_df = pandas2ri.rpy2py(ebmt)
        ebmt_df.to_csv(data_dir / "ebmt.csv", index=False)
        datasets.append(("ebmt", "Bone marrow transplant data from the EBMT registry"))
        logging.info("EBMT dataset saved successfully")
    except Exception as e:
        logging.error(f"Error saving EBMT dataset: {e}")

# 2. CAV dataset from msm package
if install_r_package("msm"):
    try:
        msm = importr('msm')
        ro.r("data(cav, package='msm')")
        cav = ro.r("cav")
        cav_df = pandas2ri.rpy2py(cav)
        cav_df.to_csv(data_dir / "cav.csv", index=False)
        datasets.append(("cav", "Heart transplant coronary allograft vasculopathy data"))
        logging.info("CAV dataset saved successfully")
    except Exception as e:
        logging.error(f"Error saving CAV dataset: {e}")

# 3. Bladder cancer data from survival package
if install_r_package("survival"):
    try:
        survival = importr('survival')
        ro.r("data(bladder, package='survival')")
        bladder = ro.r("bladder")
        bladder_df = pandas2ri.rpy2py(bladder)
        bladder_df.to_csv(data_dir / "bladder.csv", index=False)
        datasets.append(("bladder", "Bladder cancer recurrence data"))
        logging.info("Bladder cancer dataset saved successfully")
    except Exception as e:
        logging.error(f"Error saving Bladder cancer dataset: {e}")

# 4. PBC dataset from survival package
if install_r_package("survival"):
    try:
        ro.r("data(pbc, package='survival')")
        pbc = ro.r("pbc")
        pbc_df = pandas2ri.rpy2py(pbc)
        pbc_df.to_csv(data_dir / "pbc.csv", index=False)
        datasets.append(("pbc", "Primary biliary cirrhosis data"))
        logging.info("PBC dataset saved successfully")
    except Exception as e:
        logging.error(f"Error saving PBC dataset: {e}")

# 5. AIDS SI switching data from mstate
if 'mstate' in ro.r("(.packages())"):
    try:
        ro.r("data(aidssi, package='mstate')")
        aidssi = ro.r("aidssi")
        aidssi_df = pandas2ri.rpy2py(aidssi)
        aidssi_df.to_csv(data_dir / "aidssi.csv", index=False)
        datasets.append(("aidssi", "AIDS and SI switching data"))
        logging.info("AIDSSI dataset saved successfully")
    except Exception as e:
        logging.error(f"Error saving AIDSSI dataset: {e}")

# Create a README file in the data directory
readme_content = """# Multistate Dataset Collection

This directory contains datasets commonly used for multistate modeling:

"""
for name, description in datasets:
    readme_content += f"- **{name}.csv**: {description}\n"

readme_content += """
## Dataset Details

### ebmt
Bone marrow transplant data with states:
- Transplant
- Recovery (platelet recovery)
- Adverse event (acute graft-versus-host disease)
- Relapse
- Death

### cav
Heart transplant data with coronary allograft vasculopathy (CAV) states:
- No CAV
- Mild/moderate CAV
- Severe CAV
- Death

### bladder
Bladder cancer recurrence data with multiple recurrence events.

### pbc
Primary biliary cirrhosis data with disease progression states.

### aidssi
AIDS and SI switching data with:
- HIV infection
- SI virus appearance
- AIDS diagnosis
- Death

## Usage with MultiStateNN

These datasets need preprocessing before use with MultiStateNN. See the examples directory for notebooks demonstrating proper data preparation.
"""

with open(data_dir / "README.md", "w") as f:
    f.write(readme_content)

logging.info(f"Successfully downloaded {len(datasets)} datasets to {data_dir.absolute()}")
logging.info("See data/README.md for dataset details")