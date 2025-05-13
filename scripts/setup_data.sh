#!/bin/bash
# Script to install necessary dependencies and download datasets

set -e  # Exit on error

echo "Setting up datasets for MultiStateNN..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
else
  source venv/bin/activate
fi

# Install the package in development mode
echo "Installing multistate_nn with development dependencies..."
pip install -e ".[dev]"

# Install rpy2 for data download
echo "Installing rpy2 for R integration..."
pip install rpy2

# Create scripts directory if it doesn't exist
mkdir -p scripts

# Make the download script executable
chmod +x scripts/download_data.py

# Run the download script
echo "Downloading datasets from R packages..."
python scripts/download_data.py

echo "Setup complete! Datasets are available in the 'data' directory."