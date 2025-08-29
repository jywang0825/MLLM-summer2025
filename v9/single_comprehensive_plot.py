#!/usr/bin/env python3
"""
Single Comprehensive Plot - All metrics in one graph for easy comparison.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_evaluation_results(file_path: str) -> pd.DataFrame:
    """
    Load evaluation results and convert to DataFrame for analysis.
    """
    print(f"Loading evaluation results from {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract detailed results
    if 'detailed_results' in data:
        results = data['detailed_results']
    else:
        results = data
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    print(f"Loaded {len(df)} video results")
    return df

def create_single_comprehensive_plot(baseline_df: pd.DataFrame, 
                                   finetuned_df: pd.DataFrame,
                                   output_dir: str = "v9/analysis_plots"):
    """
    Create a single comprehensive plot showing all metrics together.
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define metrics to analyze
    metrics = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1']
    
    # Add model identifier
    baseline_df['model'] = 'Before LoRA'
    finetuned_df['model'] = 'After LoRA'
    
    # Combine dataframes
    combined_df = pd.concat([baseline_df, finetuned_df], ignore_index=True)
    
    print("Creating comprehensive single plot...")
    
    # Create a single large plot with all metrics
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Per-Video Metric Distribution: Before vs After LoRA Finetuning', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Color palette for better distinction
    colors = ['#FF6B6B', '#4ECDC4']  # Red for before, Teal for after
    
    for i, metric in enumerate(metrics):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Create violin plot with box plot inside
        sns.violinplot(data=combined_df, x='model', y=metric, ax=ax, palette=colors)
        
        # Add box plot on top
        sns.boxplot(data=combined_df, x='model', y=metric, ax=ax, 
                   width=0.3, palette=colors, boxprops=dict(alpha=0.7))
        
        # Customize appearance
        ax.set_title(f'{metric.upper().replace("_", "-")}', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.upper().replace("_", "-"), fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        
        # Add statistics
        baseline_mean = baseline_df[metric].mean()
        finetuned_mean = finetuned_df[metric].mean()
        improvement = finetuned_mean - baseline_mean
        improvement_pct = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        # Color code the improvement
        color = 'green' if improvement > 0 else 'red'
        
        stats_text = f'Before: {baseline_mean:.4f}\nAfter: {finetuned_mean:.4f}\nΔ: {improvement:+.4f} ({improvement_pct:+.1f}%)'
        ax.text(0.5, 0.95, stats_text, transform=ax.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                fontsize=10, color=color, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create improvement summary plot
    create_improvement_summary(baseline_df, finetuned_df, output_dir)
    
    print(f"✓ Comprehensive plot saved to {output_dir}")

def create_improvement_summary(baseline_df: pd.DataFrame, 
                             finetuned_df: pd.DataFrame, 
                             output_dir: str):
    """
    Create a summary plot showing improvements across all metrics.
    """
    
    metrics = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1']
    
    # Calculate improvements
    improvements = []
    improvement_pcts = []
    
    for metric in metrics:
        baseline_mean = baseline_df[metric].mean()
        finetuned_mean = finetuned_df[metric].mean()
        improvement = finetuned_mean - baseline_mean
        improvement_pct = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        improvements.append(improvement)
        improvement_pcts.append(improvement_pct)
    
    # Create improvement summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute improvements
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars1 = ax1.bar(metrics, improvements, color=colors, alpha=0.7)
    ax1.set_title('Absolute Improvement', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Improvement', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                f'{value:+.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # Percentage improvements
    colors = ['green' if x > 0 else 'red' for x in improvement_pcts]
    bars2 = ax2.bar(metrics, improvement_pcts, color=colors, alpha=0.7)
    ax2.set_title('Percentage Improvement', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, improvement_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -1),
                f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_statistics(baseline_df: pd.DataFrame, 
                              finetuned_df: pd.DataFrame,
                              output_file: str = "v9/analysis_plots/summary_statistics.json"):
    """
    Generate comprehensive summary statistics.
    """
    
    metrics = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1']
    
    summary = {
        'baseline_stats': {},
        'finetuned_stats': {},
        'improvement_stats': {},
        'overall_summary': {}
    }
    
    for metric in metrics:
        baseline_scores = baseline_df[metric].values
        finetuned_scores = finetuned_df[metric].values
        
        # Baseline statistics
        summary['baseline_stats'][metric] = {
            'mean': float(np.mean(baseline_scores)),
            'std': float(np.std(baseline_scores)),
            'min': float(np.min(baseline_scores)),
            'max': float(np.max(baseline_scores)),
            'median': float(np.median(baseline_scores))
        }
        
        # Finetuned statistics
        summary['finetuned_stats'][metric] = {
            'mean': float(np.mean(finetuned_scores)),
            'std': float(np.std(finetuned_scores)),
            'min': float(np.min(finetuned_scores)),
            'max': float(np.max(finetuned_scores)),
            'median': float(np.median(finetuned_scores))
        }
        
        # Improvement statistics
        if len(baseline_scores) == len(finetuned_scores):
            improvements = finetuned_scores - baseline_scores
            summary['improvement_stats'][metric] = {
                'mean_improvement': float(np.mean(improvements)),
                'std_improvement': float(np.std(improvements)),
                'improvement_rate': float(np.sum(improvements > 0) / len(improvements) * 100),
                'videos_improved': int(np.sum(improvements > 0)),
                'total_videos': len(improvements)
            }
    
    # Overall summary
    summary['overall_summary'] = {
        'total_videos': len(baseline_df),
        'analysis_date': pd.Timestamp.now().isoformat(),
        'metrics_analyzed': metrics
    }
    
    # Save summary
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary statistics saved to {output_file}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Single comprehensive plot analysis")
    parser.add_argument("baseline_results", help="Path to baseline evaluation results JSON")
    parser.add_argument("finetuned_results", help="Path to finetuned evaluation results JSON")
    parser.add_argument("--output_dir", default="v9/analysis_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load data
    baseline_df = load_evaluation_results(args.baseline_results)
    finetuned_df = load_evaluation_results(args.finetuned_results)
    
    # Create comprehensive plot
    create_single_comprehensive_plot(baseline_df, finetuned_df, args.output_dir)
    
    # Generate summary statistics
    summary = generate_summary_statistics(baseline_df, finetuned_df, 
                                        f"{args.output_dir}/summary_statistics.json")
    
    print(f"\n🎯 Analysis complete! Check {args.output_dir} for:")
    print(f"- comprehensive_metric_comparison.png: All metrics in one view")
    print(f"- improvement_summary.png: Improvement summary")
    print(f"- summary_statistics.json: Detailed statistics")

if __name__ == "__main__":
    main() 