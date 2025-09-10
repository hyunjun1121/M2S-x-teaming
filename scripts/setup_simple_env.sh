#!/bin/bash

# Simple Conda Environment Setup for run_multi_model_evaluation.py
# This script creates a minimal environment for running the multi-model evaluation

set -e  # Exit on any error

echo "Setting up simple M2S evaluation environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Environment name
ENV_NAME="m2s_simple"

# Remove existing environment if it exists
echo "Checking for existing environment: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing environment: $ENV_NAME"
    conda env remove -n $ENV_NAME -y
fi

# Create new conda environment with Python 3.9 (more stable)
echo "Creating new conda environment: $ENV_NAME with Python 3.9"
conda create -n $ENV_NAME python=3.9 -y

# Activate environment
echo "Activating environment: $ENV_NAME"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install pip packages
echo "Installing minimal Python packages..."
pip install --upgrade pip

# Install packages one by one to catch any issues
echo "Installing core packages..."
pip install openai>=1.0.0
pip install google-generativeai>=0.3.0
pip install requests>=2.25.0
pip install pandas>=1.3.0
pip install numpy>=1.21.0
pip install openpyxl>=3.0.0
pip install pyyaml>=6.0
pip install tqdm>=4.60.0
pip install colorama>=0.4.0

# Test import of critical modules
echo "Testing critical imports..."
python -c "import openai; print('✓ OpenAI')"
python -c "import google.generativeai; print('✓ Google GenAI')"
python -c "import pandas; print('✓ Pandas')"
python -c "import yaml; print('✓ PyYAML')"

echo ""
echo "Simple environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the multi-model evaluation:"
echo "  python run_multi_model_evaluation.py --samples 100 --output-dir ./experiments/multi_model_results"
echo ""
echo "For tmux session usage:"
echo "  tmux new-session -d -s m2s_simple"
echo "  tmux send-keys -t m2s_simple 'conda activate $ENV_NAME && python run_multi_model_evaluation.py --samples 100' Enter"
echo "  tmux attach -t m2s_simple"
echo ""
echo "Environment name: $ENV_NAME"
