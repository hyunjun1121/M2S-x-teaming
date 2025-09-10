#!/usr/bin/env python3
"""
Generate M2S Cross-Model Analysis Figures
- Main paper: Cross-model success rate heatmap
- Appendix: Model vulnerability & Template ranking with Wilson CIs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def wilson_ci(k, n, z=1.96):
    """Calculate Wilson confidence interval for binomial proportion"""
    if n == 0:
        return 0.0, 0.0, 0.0
    
    p_hat = k / n
    center = (p_hat + z**2 / (2*n)) / (1 + z**2 / n)
    half = z * sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / (1 + z**2 / n)
    
    return center, center - half, center + half


def main():
    # Load data
    df = pd.read_csv('success_rate_matrix.csv')
    
    # Define order
    model_order = ["Claude-4-Sonnet", "GPT-4.1", "Qwen3-235B", "GPT-5", "Gemini-2.5-Pro"]
    template_order = ["hyphenize", "numberize", "pythonize", "evolved_template_1", "evolved_template_2"]
    
    # Pivot data
    pivot = df.set_index('Template').reindex(template_order)[model_order]
    
    # [Figure 1: Main paper heatmap]
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(model_order)))
    ax.set_yticks(range(len(template_order)))
    ax.set_xticklabels(model_order, rotation=45, ha='right')
    ax.set_yticklabels(template_order)
    
    # Add value annotations
    for i in range(len(template_order)):
        for j in range(len(model_order)):
            value = pivot.iloc[i, j]
            if value == 0.0:
                text = "IMMUNE"
                color = 'white'
            else:
                text = f"{value:.2f}"
                color = 'white' if value < 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('Target Models')
    ax.set_ylabel('Templates')
    ax.set_title('Cross-model success rates (θ=0.70, judge=GPT-4.1; 100 per cell)')
    
    # Colorbar
    plt.colorbar(im, ax=ax, label='Success Rate')
    
    plt.tight_layout()
    plt.savefig('figs/m2s_crossmodel_heatmap.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: figs/m2s_crossmodel_heatmap.pdf")
    
    # [Figure A1: Model vulnerability]
    model_avg_rates = pivot.mean(axis=0).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate Wilson CIs
    model_stats = []
    for model in model_avg_rates.index:
        avg_rate = model_avg_rates[model]
        k = round(avg_rate * 500)  # 5 templates * 100 samples
        center, lower, upper = wilson_ci(k, 500)
        model_stats.append({
            'model': model,
            'avg_rate': avg_rate,
            'k': k,
            'center': center,
            'lower': lower,
            'upper': upper,
            'ci_width': upper - lower
        })
    
    models = [s['model'] for s in model_stats]
    centers = [s['center'] for s in model_stats]
    errors = [[s['center'] - s['lower'] for s in model_stats], 
              [s['upper'] - s['center'] for s in model_stats]]
    
    # Horizontal bar plot
    bars = ax.barh(range(len(models)), centers)
    ax.errorbar(centers, range(len(models)), xerr=errors, fmt='none', color='black', capsize=3)
    
    # Add value labels and IMMUNE text
    for i, stats in enumerate(model_stats):
        ax.text(stats['center'] + 0.02, i, f"{stats['avg_rate']:.3f}", 
                va='center', ha='left', fontweight='bold')
        if stats['k'] == 0:
            ax.text(0.01, i, "IMMUNE", va='center', ha='left', 
                   fontweight='bold', color='red')
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Average Success Rate (95% Wilson CI)')
    ax.set_ylabel('Models')
    ax.set_title('Model Vulnerability Analysis\n(θ=0.70, judge fixed=GPT-4.1, 100 prompts per cell)')
    ax.set_xlim(0, max(centers) * 1.1)
    
    plt.tight_layout()
    plt.savefig('figs/m2s_model_vulnerability.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: figs/m2s_model_vulnerability.pdf")
    
    # [Figure A2: Template ranking]
    template_avg_rates = pivot.mean(axis=1).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate Wilson CIs for templates
    template_stats = []
    for template in template_avg_rates.index:
        avg_rate = template_avg_rates[template]
        k = round(avg_rate * 500)  # 5 models * 100 samples
        center, lower, upper = wilson_ci(k, 500)
        template_stats.append({
            'template': template,
            'avg_rate': avg_rate,
            'k': k,
            'center': center,
            'lower': lower,
            'upper': upper
        })
    
    templates = [s['template'] for s in template_stats]
    centers = [s['center'] for s in template_stats]
    errors = [[s['center'] - s['lower'] for s in template_stats], 
              [s['upper'] - s['center'] for s in template_stats]]
    
    # Vertical bar plot
    bars = ax.bar(range(len(templates)), centers, yerr=errors, capsize=5)
    
    # Add value labels
    for i, stats in enumerate(template_stats):
        ax.text(i, stats['center'] + 0.01, f"{stats['avg_rate']:.3f}", 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(range(len(templates)))
    ax.set_xticklabels(templates, rotation=45, ha='right')
    ax.set_ylabel('Average Success Rate (95% Wilson CI)')
    ax.set_xlabel('Templates')
    ax.set_title('Template Performance Ranking\n(θ=0.70, judge fixed=GPT-4.1, 100 prompts per cell)')
    ax.set_ylim(0, max(centers) * 1.1)
    
    plt.tight_layout()
    plt.savefig('figs/m2s_template_ranking.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: figs/m2s_template_ranking.pdf")
    
    print("\nAll figures generated successfully!")
    print("- Main paper: figs/m2s_crossmodel_heatmap.pdf")
    print("- Appendix A1: figs/m2s_model_vulnerability.pdf")
    print("- Appendix A2: figs/m2s_template_ranking.pdf")


if __name__ == "__main__":
    main()