# M2S X-Teaming Evolution Pipeline

> **Multi-turn to Single-turn Jailbreak Template Evolution using Evolutionary Algorithms**

📄 **Paper**: [X-Teaming Evolutionary M2S: Automated Discovery of Multi-turn to Single-turn Jailbreak Templates](https://arxiv.org/abs/2509.08729)  
🏷️ **arXiv**: 2509.08729 [cs.CL]

This repository contains the complete implementation of our M2S (Multi-turn to Single-turn) template evolution pipeline, which uses evolutionary algorithms to automatically discover effective jailbreak templates that convert multi-turn conversations into single-turn attacks.

## 🎯 **Project Overview**

Our pipeline combines evolutionary algorithms with multi-turn jailbreak template optimization to automatically discover effective single-turn attack vectors. The system demonstrates how evolutionary computation can improve the efficiency and effectiveness of AI safety testing.

### 🔬 **Three Main Experimental Components**
1. **M2S Template Evolution (Threshold=0.25)** (`evolution_results/threshold_025_high_success/`)
   - High success case: 63.5% success rate → 4 generations (200 experiments)
   - Demonstrates meaningful evolution with relaxed threshold
   
2. **M2S Template Evolution (Threshold=0.70)** (`evolution_results/threshold_070_five_generations/`)
   - Meaningful evolution case: 44.8% success rate → 5 generations
   - Shows successful template evolution with challenging threshold
   
3. **Multi-Model Evaluation Results** (`evaluation_results/`)
   - Cross-model transfer testing of evolved templates across 5 SOTA models
   - 2,500 total experiments with statistical analysis
   - Publication-ready figures and detailed performance matrices

### 🎯 **Research Contributions**

1. **Automated Template Evolution**: First application of evolutionary algorithms to M2S template optimization
2. **Cross-Model Generalization**: Comprehensive evaluation across multiple state-of-the-art language models  
3. **Statistical Rigor**: Robust evaluation using StrongReject framework with Wilson confidence intervals
4. **Reproducible Results**: Complete codebase and experimental data for full reproduction

## 📁 Repository Structure

```
M2S-x-teaming-pipeline/
├── agents/                     # Core evolution agents
│   ├── evolutionary_m2s_generator.py    # Evolution algorithm implementation
│   ├── correct_m2s_converter.py         # Template conversion logic
│   ├── strongreject_evaluator.py        # Evaluation framework
│   └── lightweight_agent.py             # Base agent class
├── config/                     # Configuration files
│   ├── config.yaml            # Evolution pipeline config
│   └── multi_model_config.yaml # Multi-model evaluation config  
├── utils/                      # Utility functions
│   └── smart_data_loader.py   # Dataset loading and sampling
├── scripts/                    # Execution scripts
│   ├── enhanced_experiment_tracker.py   # Main evolution pipeline
│   ├── run_multi_model_custom.py        # Multi-model evaluation
│   ├── setup_simple_env.sh              # Environment setup
│   └── requirements*.txt                # Dependencies
├── examples/                   # Usage examples
│   └── run_evolution_example.sh         # Complete pipeline example
├── evolution_results/          # Evolution experiment outputs
│   ├── threshold_025_high_success/           # Threshold=0.25: High success evolution (63.5% SR)
│   └── threshold_070_five_generations/       # Threshold=0.70: Five-generation evolution (44.8% SR)
├── evaluation_results/         # Multi-model evaluation outputs  
│   └── multi_model_evaluation_2025-09-09/  # Cross-model transfer results (2,500 experiments)
├── templates_for_multi_model.json       # Final evolved templates
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ 
- Conda or Miniconda
- API access to evaluation models (OpenAI, etc.)

### 1. Environment Setup
```bash
cd scripts/
chmod +x setup_simple_env.sh
./setup_simple_env.sh
conda activate m2s_simple
```

### 2. Configuration
Edit configuration files with your API credentials:
```bash
# Evolution pipeline config
vim config/config.yaml

# Multi-model evaluation config  
vim config/multi_model_config.yaml
```

### 3. Run Evolution Pipeline
```bash
python scripts/enhanced_experiment_tracker.py
```

### 4. Evaluate Evolved Templates
```bash
python scripts/run_multi_model_custom.py --config ./config/multi_model_config.yaml --templates ./templates_for_multi_model.json
```

## 📊 Key Results

Our evolutionary pipeline successfully generated improved M2S templates with the following findings:

### Template Performance Ranking (Multi-Model Average)
1. **evolved_template_1**: 36.6% success rate (🏆 **Best Overall**)
2. **numberize**: 36.0% success rate  
3. **pythonize**: 35.8% success rate
4. **hyphenize**: 34.6% success rate
5. **evolved_template_2**: 33.2% success rate

### Evolution Threshold Comparison
- **Threshold=0.25**: 63.5% success rate → 4 generations of meaningful evolution (200 experiments)
- **Threshold=0.70**: 44.8% success rate → 5 generations with challenging threshold (230 experiments)

### Model Vulnerability Analysis
- **Complete Immunity**: GPT-5, Gemini-2.5-Pro (0% success rate)
- **Highly Vulnerable**: Qwen3-235B (64.3-65.5% success rate)
- **Moderately Vulnerable**: GPT-4.1 (62.7-67.0%), Claude-4-Sonnet (42.0-50.3%)

### Evolution Effectiveness
- **GPT-4.1**: +6.9% improvement with evolved templates
- **Qwen3-235B**: +1.8% improvement  
- **Claude-4-Sonnet**: -16.6% (evolution had negative effect)

## 🔬 Experimental Details

### Evolution Pipeline
- **Algorithm**: Multi-objective evolutionary optimization
- **Generations**: Up to 5 generations with convergence detection
- **Population Size**: Dynamic based on performance
- **Selection**: Tournament selection with elitism
- **Mutation**: LLM-guided template modifications
- **Evaluation**: StrongReject framework (thresholds: 0.25, 0.70)

### Multi-Model Evaluation  
- **Models**: 5 SOTA LLMs (GPT-4.1, GPT-5, Gemini-2.5-Pro, Qwen3-235B, Claude-4-Sonnet)
- **Templates**: 5 total (3 baseline + 2 evolved)
- **Samples**: 2,500 total experiments (100 per model-template pair)
- **Dataset**: Balanced sampling across SafeMT, MHJ, HH-RLHF, Guard
- **Duration**: 1h 47min with 8 parallel workers

## 📈 Results Analysis

### Statistical Significance
- Effect sizes indicate meaningful practical differences
- GPT-4.1 shows largest positive response (Cohen's d = +2.109)
- Wilson confidence intervals provide robust uncertainty estimates

### Key Insights
1. **Template Evolution Works**: evolved_template_1 outperforms all baselines
2. **Model-Specific Optimization**: Evolution effectiveness varies by architecture
3. **Safety Advances**: Latest models (GPT-5, Gemini-2.5-Pro) show complete immunity
4. **Vulnerability Persistence**: Some models remain consistently vulnerable

## 📁 Result Files

### Evolution Results (`evolution_results/`)
#### Threshold=0.25 High Success (`threshold_025_high_success/`)
- **m2s_evolution_pipeline_results.json**: Four-generation evolution results (63.5% SR)
- **m2s_evolution_analysis.json**: Complete evolutionary analysis with relaxed threshold
- Demonstrates successful template discovery with higher success rates

#### Threshold=0.70 Five Generations (`threshold_070_five_generations/`)
- **m2s_evolution_pipeline_results.json**: Five-generation evolution history
- **m2s_evolution_analysis.json**: Complete evolutionary analysis (44.8% SR)
- **detailed_analysis_report.md**: Comprehensive statistical analysis
- Generation-by-generation performance tracking and template discovery

### Evaluation Results (`evaluation_results/`)
- **multi_model_results.json**: Complete 2,500-experiment dataset
- **success_rate_matrix.csv**: Model-template performance matrix
- **summary_statistics.json**: Aggregated performance metrics  
- **M2S_Evolution_Analysis_Report.md**: Comprehensive analysis report
- **figs/**: Publication-ready figures
  - `m2s_crossmodel_heatmap.pdf`: Main paper heatmap
  - `m2s_model_vulnerability.pdf`: Appendix vulnerability analysis
  - `m2s_template_ranking.pdf`: Appendix template ranking

## 🔧 Advanced Usage

### Custom Evolution Parameters
Modify `config/config.yaml` to adjust:
- Population size and selection pressure
- Mutation rates and strategies  
- Evaluation thresholds and metrics
- Early stopping criteria

### Multi-Processing Configuration
Enable parallel processing in `config/multi_model_config.yaml`:
```yaml
experiment:
  parallel_requests: true
  max_workers: 8  # Adjust based on your hardware
```

### Custom Dataset Integration
Use `utils/smart_data_loader.py` to integrate your own datasets:
- Supports Excel (.xlsx) format
- Balanced group sampling
- Configurable source mapping

## 🔍 **Key Experiment Summary**

### **Experiment 1: M2S Template Evolution (Dual Threshold Analysis)**
#### **Case 1A: Threshold=0.25**
- **Location**: `evolution_results/threshold_025_high_success/`
- **Key Finding**: 63.5% success rate enables meaningful 4-generation evolution
- **Insight**: Relaxed threshold allows sustained template development

#### **Case 1B: Threshold=0.70** 
- **Location**: `evolution_results/threshold_070_five_generations/`
- **Key Finding**: 44.8% success rate enables 5 generations of meaningful evolution
- **Best Templates**: `evolved_template_1` and `evolved_template_2` discovered
- **Evidence**: Complete evolution logs, statistical analysis, generation-by-generation metrics

### **Experiment 2: Cross-Model Transfer Protocol**
- **Location**: `evaluation_results/multi_model_evaluation_2025-09-09/`  
- **Scope**: 2,500 experiments across 5 models × 5 templates × 100 samples
- **Templates Tested**: 3 baseline + 2 evolved (from threshold=0.70 experiment)
- **Key Findings**:
  - `evolved_template_1`: **36.6%** success rate (🏆 best overall)
  - **GPT-5 & Gemini-2.5-Pro**: Complete immunity (0% success)
  - **GPT-4.1**: +6.9% improvement with evolved templates
  - **Qwen3-235B**: Most vulnerable (64.3-65.5% success rate)
- **Evidence**: Cross-model transferability validation of evolved templates

### **Publication-Ready Results**
- **Figures**: `evaluation_results/multi_model_evaluation_2025-09-09/figs/`
  - Main paper heatmap: `m2s_crossmodel_heatmap.pdf`
  - Appendix figures: `m2s_model_vulnerability.pdf`, `m2s_template_ranking.pdf`
- **Statistical Analysis**: Wilson CIs, Cohen's d effect sizes
- **Complete Dataset**: 77,966 lines of experimental data

## 🚀 **Quick Start Guide**

### Option 1: View Results Only
```bash
# Threshold comparison analysis
cd evolution_results/
ls threshold_025_high_success/       # 63.5% SR → Meaningful evolution
ls threshold_070_five_generations/   # 44.8% SR → Meaningful evolution

# Cross-model transfer results
cd evaluation_results/multi_model_evaluation_2025-09-09/
ls -la  # View all result files
open figs/*.pdf  # View publication figures
```

### Option 2: Threshold-Specific Analysis
```bash
# Analyze threshold=0.25 experiment (high success)
cat evolution_results/threshold_025_high_success/m2s_evolution_analysis.json

# Analyze threshold=0.70 experiment (five generations)
cat evolution_results/threshold_070_five_generations/detailed_analysis_report.md

# Compare evolved templates performance
cat evaluation_results/multi_model_evaluation_2025-09-09/success_rate_matrix.csv
```

### Option 3: Full Reproduction  
```bash
# Setup environment
cd scripts/ && ./setup_simple_env.sh
conda activate m2s_simple

# Run evolution with different thresholds
python scripts/enhanced_experiment_tracker.py --threshold 0.25  # High success evolution
python scripts/enhanced_experiment_tracker.py --threshold 0.70  # Meaningful evolution

# Run multi-model evaluation  
python scripts/run_multi_model_custom.py --config ./config/multi_model_config.yaml --templates ./templates_for_multi_model.json
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Ethical Use

This research tool is intended for:
- Academic research in AI safety
- Red-teaming and vulnerability assessment
- Defensive AI development

**Please use responsibly and in compliance with relevant AI safety guidelines.**

## 💬 Support

For questions or issues:
1. Check the [examples/](examples/) directory
2. Review result files in [evaluation_results/](evaluation_results/)
3. Open a GitHub issue
4. Contact the maintainers

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{kim2025xteamingevolutionarym2sautomated,
      title={X-Teaming Evolutionary M2S: Automated Discovery of Multi-turn to Single-turn Jailbreak Templates}, 
      author={Hyunjun Kim and Junwoo Ha and Sangyoon Yu and Haon Park},
      year={2025},
      eprint={2509.08729},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.08729}, 
}
```

---

**Generated by M2S X-Teaming Pipeline v1.0**  
**Last Updated**: September 2025