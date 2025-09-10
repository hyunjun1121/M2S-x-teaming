# M2S Evolution Pipeline Technical Guide

## Pipeline Architecture

The M2S X-Teaming pipeline consists of three main phases:

### Phase 1: Template Evolution
**Script**: `scripts/enhanced_experiment_tracker.py`

1. **Initialization**: Load baseline templates (hyphenize, numberize, pythonize)
2. **Population Generation**: Create initial population of template variants
3. **Evaluation**: Test each template on sample dataset using StrongReject
4. **Selection**: Tournament selection with elitism preservation
5. **Mutation**: LLM-guided template modifications and improvements
6. **Iteration**: Repeat until convergence or max generations

**Key Parameters**:
- `max_generations`: Maximum evolution cycles (default: 10)
- `population_size`: Templates per generation (dynamic)
- `threshold`: StrongReject success threshold (0.70)
- `early_stopping`: Stop if no improvement for N generations

### Phase 2: Template Validation
**Intermediate Step**: Quality assurance of evolved templates

1. **Template Parsing**: Validate generated template syntax
2. **Compatibility Check**: Ensure templates work across models
3. **Performance Filtering**: Remove low-performing variants
4. **Final Selection**: Choose best templates for evaluation

### Phase 3: Multi-Model Evaluation  
**Script**: `scripts/run_multi_model_custom.py`

1. **Model Configuration**: Setup API clients for all target models
2. **Dataset Sampling**: Balanced sampling across 4 groups (100 per model-template)
3. **Parallel Execution**: 8-worker parallel processing for efficiency
4. **Evaluation**: StrongReject scoring for each experiment
5. **Statistical Analysis**: Wilson confidence intervals and effect sizes

## File Formats

### Evolution Results
```json
{
  "metadata": {
    "timestamp": "2025-09-08T19:06:08",
    "total_generations": 1,
    "threshold": 0.7
  },
  "generations": [
    {
      "generation": 1,
      "templates": [...],
      "performance": {...}
    }
  ],
  "final_templates": [...],
  "statistics": {...}
}
```

### Multi-Model Results
```json
[
  {
    "template_name": "evolved_template_1",
    "model_name": "GPT-4.1", 
    "conversation_id": "conv_001",
    "success_rate": 0.67,
    "evaluation_score": 0.85,
    "attack_success": true,
    "timestamp": "2025-09-09T14:51:59"
  }
]
```

## Reproducibility Checklist

To reproduce our results:

- [ ] Use identical model versions (specified in config)
- [ ] Set random seeds for consistent sampling
- [ ] Use same StrongReject threshold (0.70)
- [ ] Maintain balanced dataset groups (25 samples each)
- [ ] Apply identical evaluation criteria

## Performance Optimization

### For High-Performance Servers
```yaml
experiment:
  parallel_requests: true
  max_workers: 16        # Scale with CPU cores
  batch_size: 50         # Larger batches for efficiency
```

### For Resource-Constrained Environments
```yaml
experiment:
  parallel_requests: false
  max_workers: 2
  batch_size: 10
```

## Troubleshooting

### Common Issues

1. **API Rate Limiting**
   - Reduce `max_workers`
   - Increase delay between requests
   - Check API quota and billing

2. **Memory Issues**
   - Reduce `batch_size`
   - Limit `max_generations`
   - Monitor system resources

3. **Template Generation Failures**
   - Check LLM connectivity
   - Verify prompt templates
   - Review mutation parameters

### Debug Mode
Enable verbose logging in config:
```yaml
logging:
  level: DEBUG
  save_to_file: true
```