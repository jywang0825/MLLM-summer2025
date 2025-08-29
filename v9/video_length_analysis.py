#!/usr/bin/env python3
"""
Video Length vs BLEU-1 Analysis
For each video, plot its length (in seconds or frames) on the x axis and its BLEU-1 score on the y axis,
with different markers/colors for baseline vs LoRA finetuned.
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

def extract_video_length(video_path: str) -> int:
    """
    Extract video length from video path or other metadata.
    For now, we'll use uniform_sampled_frames as a proxy for video length.
    """
    # This is a placeholder - you might need to adjust based on your data structure
    return 32  # Default uniform sampling

def create_video_length_analysis(baseline_df: pd.DataFrame, 
                               finetuned_df: pd.DataFrame,
                               output_dir: str = "v9/analysis_plots"):
    """
    Create video length vs BLEU-1 analysis plots.
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Add video length information (using uniform_sampled_frames as proxy)
    if 'uniform_sampled_frames' in baseline_df.columns:
        baseline_df['video_length'] = baseline_df['uniform_sampled_frames']
        finetuned_df['video_length'] = finetuned_df['uniform_sampled_frames']
    else:
        # If no frame info, use a default length
        baseline_df['video_length'] = 32
        finetuned_df['video_length'] = 32
    
    # Add model identifier
    baseline_df['model'] = 'Baseline'
    finetuned_df['model'] = 'LoRA Finetuned'
    
    print("Creating video length vs BLEU-1 analysis...")
    
    # 1. Scatter plot: Video Length vs BLEU-1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Scatter plot
    ax1.scatter(baseline_df['video_length'], baseline_df['bleu_1'], 
                alpha=0.6, s=50, color='red', label='Baseline')
    ax1.scatter(finetuned_df['video_length'], finetuned_df['bleu_1'], 
                alpha=0.6, s=50, color='blue', label='LoRA Finetuned')
    
    # Add trend lines
    z_baseline = np.polyfit(baseline_df['video_length'], baseline_df['bleu_1'], 1)
    p_baseline = np.poly1d(z_baseline)
    ax1.plot(baseline_df['video_length'], p_baseline(baseline_df['video_length']), 
             "r--", alpha=0.8, linewidth=2)
    
    z_finetuned = np.polyfit(finetuned_df['video_length'], finetuned_df['bleu_1'], 1)
    p_finetuned = np.poly1d(z_finetuned)
    ax1.plot(finetuned_df['video_length'], p_finetuned(finetuned_df['video_length']), 
             "b--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Video Length (Frames)', fontsize=12)
    ax1.set_ylabel('BLEU-1 Score', fontsize=12)
    ax1.set_title('Video Length vs BLEU-1 Score', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficients
    corr_baseline = np.corrcoef(baseline_df['video_length'], baseline_df['bleu_1'])[0, 1]
    corr_finetuned = np.corrcoef(finetuned_df['video_length'], finetuned_df['bleu_1'])[0, 1]
    
    ax1.text(0.02, 0.98, f'Baseline Corr: {corr_baseline:.3f}\nFinetuned Corr: {corr_finetuned:.3f}', 
             transform=ax1.transAxes, ha='left', va='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 2: Improvement vs Video Length
    # Calculate improvement for each video
    improvements = []
    video_lengths = []
    
    for i, video_uid in enumerate(baseline_df['video_uid']):
        if video_uid in finetuned_df['video_uid'].values:
            baseline_score = baseline_df.loc[baseline_df['video_uid'] == video_uid, 'bleu_1'].iloc[0]
            finetuned_score = finetuned_df.loc[finetuned_df['video_uid'] == video_uid, 'bleu_1'].iloc[0]
            video_length = baseline_df.loc[baseline_df['video_uid'] == video_uid, 'video_length'].iloc[0]
            
            improvement = finetuned_score - baseline_score
            improvements.append(improvement)
            video_lengths.append(video_length)
    
    # Color code improvements
    colors = ['green' if x > 0 else 'red' for x in improvements]
    
    ax2.scatter(video_lengths, improvements, c=colors, alpha=0.6, s=50)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='No Change')
    
    # Add trend line for improvements
    if len(improvements) > 1:
        z_improvement = np.polyfit(video_lengths, improvements, 1)
        p_improvement = np.poly1d(z_improvement)
        ax2.plot(video_lengths, p_improvement(video_lengths), "k--", alpha=0.8, linewidth=2, label='Trend')
    
    ax2.set_xlabel('Video Length (Frames)', fontsize=12)
    ax2.set_ylabel('BLEU-1 Improvement', fontsize=12)
    ax2.set_title('Video Length vs BLEU-1 Improvement', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    positive_improvements = sum(1 for x in improvements if x > 0)
    total_videos = len(improvements)
    improvement_rate = positive_improvements / total_videos * 100 if total_videos > 0 else 0
    
    ax2.text(0.02, 0.98, f'Improved: {positive_improvements}/{total_videos} ({improvement_rate:.1f}%)', 
             transform=ax2.transAxes, ha='left', va='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/video_length_vs_bleu1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create length-based analysis
    create_length_based_analysis(baseline_df, finetuned_df, output_dir)
    
    print(f"✓ Video length analysis saved to {output_dir}")

def create_length_based_analysis(baseline_df: pd.DataFrame, 
                               finetuned_df: pd.DataFrame, 
                               output_dir: str):
    """
    Create analysis based on video length categories.
    """
    
    # Create length categories
    baseline_df['length_category'] = pd.cut(baseline_df['video_length'], 
                                          bins=[0, 20, 40, 60, 100], 
                                          labels=['Short', 'Medium', 'Long', 'Very Long'])
    finetuned_df['length_category'] = pd.cut(finetuned_df['video_length'], 
                                            bins=[0, 20, 40, 60, 100], 
                                            labels=['Short', 'Medium', 'Long', 'Very Long'])
    
    # Add model identifier
    baseline_df['model'] = 'Baseline'
    finetuned_df['model'] = 'LoRA Finetuned'
    
    # Combine dataframes
    combined_df = pd.concat([baseline_df, finetuned_df], ignore_index=True)
    
    # Create boxplot by length category
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: BLEU-1 by length category
    sns.boxplot(data=combined_df, x='length_category', y='bleu_1', hue='model', ax=axes[0])
    axes[0].set_title('BLEU-1 Score by Video Length Category', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Video Length Category', fontsize=12)
    axes[0].set_ylabel('BLEU-1 Score', fontsize=12)
    
    # Plot 2: Improvement by length category
    # Calculate improvements for each category
    categories = ['Short', 'Medium', 'Long', 'Very Long']
    improvements_by_category = {}
    
    for category in categories:
        baseline_scores = baseline_df[baseline_df['length_category'] == category]['bleu_1']
        finetuned_scores = finetuned_df[finetuned_df['length_category'] == category]['bleu_1']
        
        if len(baseline_scores) > 0 and len(finetuned_scores) > 0:
            improvements = finetuned_scores.values - baseline_scores.values
            improvements_by_category[category] = improvements
    
    # Create improvement boxplot
    improvement_data = []
    category_labels = []
    
    for category, improvements in improvements_by_category.items():
        for improvement in improvements:
            improvement_data.append(improvement)
            category_labels.append(category)
    
    if improvement_data:
        improvement_df = pd.DataFrame({
            'length_category': category_labels,
            'improvement': improvement_data
        })
        
        sns.boxplot(data=improvement_df, x='length_category', y='improvement', ax=axes[1])
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Change')
        axes[1].set_title('BLEU-1 Improvement by Video Length Category', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Video Length Category', fontsize=12)
        axes[1].set_ylabel('BLEU-1 Improvement', fontsize=12)
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/length_category_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_length_statistics(baseline_df: pd.DataFrame, 
                            finetuned_df: pd.DataFrame,
                            output_file: str = "v9/analysis_plots/length_statistics.json"):
    """
    Generate statistics about video length and performance.
    """
    
    # Calculate statistics by length category
    baseline_df['length_category'] = pd.cut(baseline_df['video_length'], 
                                          bins=[0, 20, 40, 60, 100], 
                                          labels=['Short', 'Medium', 'Long', 'Very Long'])
    finetuned_df['length_category'] = pd.cut(finetuned_df['video_length'], 
                                            bins=[0, 20, 40, 60, 100], 
                                            labels=['Short', 'Medium', 'Long', 'Very Long'])
    
    stats = {
        'length_categories': {},
        'overall_stats': {}
    }
    
    categories = ['Short', 'Medium', 'Long', 'Very Long']
    
    for category in categories:
        baseline_category = baseline_df[baseline_df['length_category'] == category]
        finetuned_category = finetuned_df[finetuned_df['length_category'] == category]
        
        if len(baseline_category) > 0:
            baseline_mean = baseline_category['bleu_1'].mean()
            finetuned_mean = finetuned_category['bleu_1'].mean()
            improvement = finetuned_mean - baseline_mean
            
            stats['length_categories'][category] = {
                'baseline_mean_bleu1': float(baseline_mean),
                'finetuned_mean_bleu1': float(finetuned_mean),
                'improvement': float(improvement),
                'improvement_percentage': float((improvement / baseline_mean) * 100) if baseline_mean > 0 else 0,
                'video_count': len(baseline_category)
            }
    
    # Overall statistics
    stats['overall_stats'] = {
        'total_videos': len(baseline_df),
        'mean_video_length': float(baseline_df['video_length'].mean()),
        'std_video_length': float(baseline_df['video_length'].std()),
        'analysis_date': pd.Timestamp.now().isoformat()
    }
    
    # Save statistics
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Length statistics saved to {output_file}")
    return stats

def main():
    parser = argparse.ArgumentParser(description="Video length vs BLEU-1 analysis")
    parser.add_argument("baseline_results", help="Path to baseline evaluation results JSON")
    parser.add_argument("finetuned_results", help="Path to finetuned evaluation results JSON")
    parser.add_argument("--output_dir", default="v9/analysis_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load data
    baseline_df = load_evaluation_results(args.baseline_results)
    finetuned_df = load_evaluation_results(args.finetuned_results)
    
    # Create video length analysis
    create_video_length_analysis(baseline_df, finetuned_df, args.output_dir)
    
    # Generate length statistics
    stats = generate_length_statistics(baseline_df, finetuned_df, 
                                     f"{args.output_dir}/length_statistics.json")
    
    print(f"\n🎯 Video length analysis complete! Check {args.output_dir} for:")
    print(f"- video_length_vs_bleu1.png: Scatter plots of length vs BLEU-1")
    print(f"- length_category_analysis.png: Performance by length category")
    print(f"- length_statistics.json: Detailed length-based statistics")

if __name__ == "__main__":
    main() 