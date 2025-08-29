#!/usr/bin/env python3
"""
Per Video Distribution Analysis - Boxplots and Violin Plots
Shows the spread of metrics across all videos in the test set, both before and after LoRA finetuning.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
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

def create_per_video_distribution_plots(baseline_df: pd.DataFrame, 
                                       finetuned_df: pd.DataFrame,
                                       output_dir: str = "v9/analysis_plots"):
    """
    Create comprehensive per-video distribution plots.
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define metrics to analyze
    metrics = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1']
    
    # Add model identifier
    baseline_df['model'] = 'Baseline'
    finetuned_df['model'] = 'LoRA Finetuned'
    
    # Combine dataframes
    combined_df = pd.concat([baseline_df, finetuned_df], ignore_index=True)
    
    print("Creating per-video distribution plots...")
    
    # 1. Boxplots for each metric
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Per-Video Metric Distribution: Baseline vs LoRA Finetuned', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Create boxplot
        sns.boxplot(data=combined_df, x='model', y=metric, ax=ax)
        ax.set_title(f'{metric.upper().replace("_", "-")} Distribution')
        ax.set_ylabel(metric.upper().replace("_", "-"))
        ax.set_xlabel('Model')
        
        # Add statistics
        baseline_mean = baseline_df[metric].mean()
        finetuned_mean = finetuned_df[metric].mean()
        improvement = finetuned_mean - baseline_mean
        
        ax.text(0.5, 0.95, f'Baseline: {baseline_mean:.4f}\nFinetuned: {finetuned_mean:.4f}\nΔ: {improvement:+.4f}', 
                transform=ax.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_video_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Violin plots for each metric
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Per-Video Metric Distribution (Violin Plots): Baseline vs LoRA Finetuned', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Create violin plot
        sns.violinplot(data=combined_df, x='model', y=metric, ax=ax)
        ax.set_title(f'{metric.upper().replace("_", "-")} Distribution')
        ax.set_ylabel(metric.upper().replace("_", "-"))
        ax.set_xlabel('Model')
        
        # Add statistics
        baseline_mean = baseline_df[metric].mean()
        finetuned_mean = finetuned_df[metric].mean()
        improvement = finetuned_mean - baseline_mean
        
        ax.text(0.5, 0.95, f'Baseline: {baseline_mean:.4f}\nFinetuned: {finetuned_mean:.4f}\nΔ: {improvement:+.4f}', 
                transform=ax.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_video_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Improvement analysis
    create_improvement_analysis(baseline_df, finetuned_df, output_dir)
    
    # 4. Per-video improvement scatter plots
    create_per_video_improvement_scatter(baseline_df, finetuned_df, output_dir)
    
    print(f"✓ All plots saved to {output_dir}")

def create_improvement_analysis(baseline_df: pd.DataFrame, 
                              finetuned_df: pd.DataFrame, 
                              output_dir: str):
    """
    Create improvement analysis plots.
    """
    
    metrics = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1']
    
    # Calculate improvements
    improvements = {}
    for metric in metrics:
        baseline_scores = baseline_df[metric].values
        finetuned_scores = finetuned_df[metric].values
        
        # Ensure same order
        if len(baseline_scores) == len(finetuned_scores):
            improvements[metric] = finetuned_scores - baseline_scores
    
    # Create improvement distribution plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Per-Video Improvement Distribution', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        if metric in improvements:
            improvements_array = improvements[metric]
            
            # Create histogram
            ax.hist(improvements_array, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='No Change')
            ax.axvline(np.mean(improvements_array), color='green', linestyle='-', alpha=0.8, 
                      label=f'Mean: {np.mean(improvements_array):.4f}')
            
            ax.set_title(f'{metric.upper().replace("_", "-")} Improvement')
            ax.set_xlabel('Improvement')
            ax.set_ylabel('Number of Videos')
            ax.legend()
            
            # Add statistics
            positive_improvements = np.sum(improvements_array > 0)
            total_videos = len(improvements_array)
            improvement_rate = positive_improvements / total_videos * 100
            
            ax.text(0.02, 0.98, f'Improved: {positive_improvements}/{total_videos} ({improvement_rate:.1f}%)', 
                   transform=ax.transAxes, ha='left', va='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_per_video_improvement_scatter(baseline_df: pd.DataFrame, 
                                       finetuned_df: pd.DataFrame, 
                                       output_dir: str):
    """
    Create scatter plots showing per-video improvements.
    """
    
    metrics = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1']
    
    # Create scatter plots for each metric
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Per-Video Performance: Baseline vs LoRA Finetuned', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        baseline_scores = baseline_df[metric].values
        finetuned_scores = finetuned_df[metric].values
        
        if len(baseline_scores) == len(finetuned_scores):
            # Create scatter plot
            ax.scatter(baseline_scores, finetuned_scores, alpha=0.6, s=30)
            
            # Add diagonal line (no change)
            min_val = min(baseline_scores.min(), finetuned_scores.min())
            max_val = max(baseline_scores.max(), finetuned_scores.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='No Change')
            
            # Add regression line
            z = np.polyfit(baseline_scores, finetuned_scores, 1)
            p = np.poly1d(z)
            ax.plot(baseline_scores, p(baseline_scores), "g-", alpha=0.8, label='Trend')
            
            ax.set_xlabel('Baseline Score')
            ax.set_ylabel('Finetuned Score')
            ax.set_title(f'{metric.upper().replace("_", "-")}')
            ax.legend()
            
            # Add correlation
            correlation = np.corrcoef(baseline_scores, finetuned_scores)[0, 1]
            ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, ha='left', va='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_video_scatter_plots.png', dpi=300, bbox_inches='tight')
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
    parser = argparse.ArgumentParser(description="Per-video distribution analysis")
    parser.add_argument("baseline_results", help="Path to baseline evaluation results JSON")
    parser.add_argument("finetuned_results", help="Path to finetuned evaluation results JSON")
    parser.add_argument("--output_dir", default="v9/analysis_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load data
    baseline_df = load_evaluation_results(args.baseline_results)
    finetuned_df = load_evaluation_results(args.finetuned_results)
    
    # Create plots
    create_per_video_distribution_plots(baseline_df, finetuned_df, args.output_dir)
    
    # Generate summary statistics
    summary = generate_summary_statistics(baseline_df, finetuned_df, 
                                        f"{args.output_dir}/summary_statistics.json")
    
    print(f"\n🎯 Analysis complete! Check {args.output_dir} for all plots and statistics.")
    print(f"Key insights:")
    print(f"- Boxplots show metric distributions before/after finetuning")
    print(f"- Violin plots show density distributions")
    print(f"- Improvement plots show which videos benefited most")
    print(f"- Scatter plots show correlation between baseline and finetuned performance")

if __name__ == "__main__":
    main() 