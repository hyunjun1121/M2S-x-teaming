#!/bin/bash

# Example: Run M2S Template Evolution Pipeline
# This script demonstrates how to reproduce the M2S evolution experiment

echo "M2S Template Evolution Example"
echo "=============================="

# 1. Setup environment
echo "Setting up environment..."
cd scripts/
./setup_simple_env.sh

# 2. Activate conda environment
echo "Activating conda environment..."
conda activate m2s_simple

# 3. Configure API settings (you need to update config/config.yaml with your API keys)
echo "Make sure to configure your API keys in config/config.yaml"

# 4. Run evolution pipeline
echo "Running M2S evolution pipeline..."
cd ../
python scripts/enhanced_experiment_tracker.py

# 5. The results will be saved in evolution_results/
echo "Evolution completed! Check evolution_results/ for generated templates."

# 6. Optional: Run multi-model evaluation with evolved templates
echo "Optional: Run multi-model evaluation..."
python scripts/run_multi_model_custom.py --config ./config/multi_model_config.yaml --templates ./templates_for_multi_model.json

echo "Pipeline execution completed!"