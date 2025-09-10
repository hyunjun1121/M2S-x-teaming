#!/bin/bash

# M2S X-Teaming Conda Environment Setup Script
# This script sets up the conda environment for running M2S experiments on GPU servers

set -e  # Exit on any error

echo "Setting up M2S X-Teaming environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Environment name
ENV_NAME="m2s_xteaming"

# Remove existing environment if it exists
echo "Checking for existing environment: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing environment: $ENV_NAME"
    conda env remove -n $ENV_NAME -y
fi

# Create new conda environment with Python 3.10 (required for aisuite)
echo "Creating new conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo "Activating environment: $ENV_NAME"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install pip packages
echo "Installing Python packages..."
pip install --upgrade pip

# Try installing full requirements first
echo "Attempting to install from requirements.txt..."
if ! pip install -r requirements.txt; then
    echo "Full requirements installation failed. Trying minimal requirements..."
    if [ -f "requirements_minimal.txt" ]; then
        pip install -r requirements_minimal.txt
        echo "Minimal requirements installed successfully."
        echo "Note: Some optional packages like aisuite were not installed."
    else
        echo "Error: Neither requirements.txt nor requirements_minimal.txt worked."
        exit 1
    fi
else
    echo "Full requirements installed successfully."
fi

# Install additional GPU-specific packages if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing additional GPU packages..."
    # Add any GPU-specific packages here if needed
fi

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the multi-model evaluation:"
echo "  python run_multi_model_custom.py --config ./config/multi_model_config.yaml --templates ./templates_for_multi_model.json"
echo ""
echo "For tmux session usage:"
echo "  tmux new-session -d -s m2s_eval"
echo "  tmux send-keys -t m2s_eval 'conda activate $ENV_NAME && python run_multi_model_custom.py --config ./config/multi_model_config.yaml --templates ./templates_for_multi_model.json' Enter"
echo "  tmux attach -t m2s_eval"
