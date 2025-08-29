#!/usr/bin/env python3
"""
Metric Correlation Analysis
Compute pairwise correlations between evaluation metrics (BLEU-1, BLEU-2, METEOR, ROUGE, etc.)
and create a correlation heatmap.
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
    Load evaluation results and extract per-video metrics.
    """
    print(f"Loading evaluation results from {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract detailed results for per-video analysis
    if 'detailed_results' in data:
        results = data['detailed_results']
    else:
        results = data
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    print(f"Loaded {len(df)} video results with {len(df.columns)} metrics")
    return df

def extract_metrics_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and clean metrics for correlation analysis.
    """
    # Define the metrics we want to analyze
    metric_columns = [
        'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
        'meteor', 
        'rouge1_f1', 'rouge1_precision', 'rouge1_recall',
        'rouge2_f1', 'rouge2_precision', 'rouge2_recall',
        'rougeL_f1', 'rougeL_precision', 'rougeL_recall'
    ]
    
    # Filter to only include available metrics
    available_metrics = [col for col in metric_columns if col in df.columns]
    print(f"Found {len(available_metrics)} metrics: {available_metrics}")
    
    # Extract metrics data
    metrics_df = df[available_metrics].copy()
    
    # Clean data (remove any NaN values)
    metrics_df = metrics_df.dropna()
    
    print(f"Final dataset: {len(metrics_df)} videos with complete metric data")
    return metrics_df

def create_correlation_heatmap(metrics_df: pd.DataFrame, output_dir: str = "v9/analysis_plots"):
    """
    Create correlation heatmap between all metrics.
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Computing pairwise correlations...")
    
    # Compute correlation matrix
    correlation_matrix = metrics_df.corr()
    
    # Create the heatmap
    plt.figure(figsize=(14, 12))
    
    # Create mask for upper triangle (optional - to show only lower triangle)
    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Uncomment to show only lower triangle
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                # mask=mask,  # Comment this line to show full matrix
                annot=True, 
                fmt='.3f', 
                cmap='RdBu_r', 
                center=0,
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'},
                linewidths=0.5)
    
    plt.title('Pairwise Correlation Between Evaluation Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metric_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Correlation heatmap saved to {output_dir}/metric_correlation_heatmap.png")
    
    return correlation_matrix

def create_metric_statistics(metrics_df: pd.DataFrame, correlation_matrix: pd.DataFrame, output_dir: str):
    """
    Create detailed statistics about the metrics.
    """
    
    # Basic statistics
    stats = {
        'metric_statistics': metrics_df.describe().to_dict(),
        'correlation_summary': {
            'highest_correlations': [],
            'lowest_correlations': [],
            'metric_variance': metrics_df.var().to_dict()
        }
    }
    
    # Find highest and lowest correlations (excluding self-correlations)
    correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            metric1 = correlation_matrix.columns[i]
            metric2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            correlations.append((metric1, metric2, corr_value))
    
    # Sort by correlation value
    correlations.sort(key=lambda x: x[2], reverse=True)
    
    stats['correlation_summary']['highest_correlations'] = correlations[:5]
    stats['correlation_summary']['lowest_correlations'] = correlations[-5:]
    
    # Save statistics
    stats_file = f"{output_dir}/metric_correlation_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"✓ Metric statistics saved to {stats_file}")
    return stats

def create_metric_distribution_plots(metrics_df: pd.DataFrame, output_dir: str):
    """
    Create distribution plots for each metric.
    """
    
    # Create subplots for metric distributions
    n_metrics = len(metrics_df.columns)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, metric in enumerate(metrics_df.columns):
        ax = axes[i]
        
        # Create histogram
        ax.hist(metrics_df[metric], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        
        # Add mean line
        mean_val = metrics_df[metric].mean()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metric distribution plots saved to {output_dir}/metric_distributions.png")

def create_scatter_matrix(metrics_df: pd.DataFrame, output_dir: str):
    """
    Create scatter matrix for key metrics.
    """
    
    # Select key metrics for scatter matrix
    key_metrics = ['bleu_1', 'meteor', 'rouge1_f1', 'rougeL_f1']
    available_key_metrics = [m for m in key_metrics if m in metrics_df.columns]
    
    if len(available_key_metrics) >= 2:
        # Create scatter matrix
        fig = sns.pairplot(metrics_df[available_key_metrics], 
                           diag_kind='hist', 
                           plot_kws={'alpha': 0.6, 's': 20})
        
        plt.suptitle('Scatter Matrix of Key Metrics', y=1.02, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metric_scatter_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Scatter matrix saved to {output_dir}/metric_scatter_matrix.png")
    else:
        print("Not enough key metrics available for scatter matrix")

def create_metric_clustering_analysis(metrics_df: pd.DataFrame, output_dir: str):
    """
    Create clustering analysis to group similar metrics.
    """
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Standardize the data
        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform(metrics_df)
        
        # Perform clustering
        n_clusters = min(5, len(metrics_df.columns))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(metrics_scaled.T)  # Transpose to cluster metrics
        
        # Create clustering visualization
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot of first two principal components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        metrics_pca = pca.fit_transform(metrics_scaled.T)
        
        scatter = plt.scatter(metrics_pca[:, 0], metrics_pca[:, 1], 
                            c=cluster_labels, cmap='viridis', s=100, alpha=0.7)
        
        # Add metric labels
        for i, metric in enumerate(metrics_df.columns):
            plt.annotate(metric.replace('_', '\n'), 
                        (metrics_pca[i, 0], metrics_pca[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, ha='left', va='bottom')
        
        plt.title('Metric Clustering Analysis', fontsize=14, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metric_clustering.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Metric clustering analysis saved to {output_dir}/metric_clustering.png")
        
    except ImportError:
        print("scikit-learn not available, skipping clustering analysis")

def main():
    parser = argparse.ArgumentParser(description="Metric correlation analysis")
    parser.add_argument("evaluation_results", help="Path to evaluation results JSON")
    parser.add_argument("--output_dir", default="v9/analysis_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load evaluation results
    df = load_evaluation_results(args.evaluation_results)
    
    # Extract metrics for correlation analysis
    metrics_df = extract_metrics_for_correlation(df)
    
    if len(metrics_df) == 0:
        print("No metrics data available for correlation analysis")
        return
    
    # Create correlation heatmap
    correlation_matrix = create_correlation_heatmap(metrics_df, args.output_dir)
    
    # Create additional analyses
    stats = create_metric_statistics(metrics_df, correlation_matrix, args.output_dir)
    create_metric_distribution_plots(metrics_df, args.output_dir)
    create_scatter_matrix(metrics_df, args.output_dir)
    create_metric_clustering_analysis(metrics_df, args.output_dir)
    
    # Print summary
    print(f"\n🎯 Metric correlation analysis complete!")
    print(f"Analyzed {len(metrics_df)} videos with {len(metrics_df.columns)} metrics")
    print(f"\nTop correlations:")
    for metric1, metric2, corr in stats['correlation_summary']['highest_correlations'][:3]:
        print(f"- {metric1} ↔ {metric2}: {corr:.3f}")
    print(f"\nCheck {args.output_dir} for detailed visualizations and statistics.")

if __name__ == "__main__":
    main() 