#!/usr/bin/env python3
"""
Create volcano plot from embeddings to visualize the impact of LoRA finetuning.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse

def load_embeddings(embedding_file: str) -> np.ndarray:
    """Load embeddings from pickle file."""
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings

def load_metadata(metadata_file: str) -> dict:
    """Load embedding metadata."""
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    # Ensure vectors are normalized
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def create_volcano_plot(baseline_embeddings: np.ndarray, finetuned_embeddings: np.ndarray, 
                       metadata: dict, output_file: str = "v9/analysis_plots/volcano_plot.png"):
    """
    Create a proper volcano plot showing the impact of LoRA finetuning on embeddings.
    
    X-axis: log2 fold-change in cosine similarity (effect size)
    Y-axis: -log10(p-value) (statistical significance)
    """
    print("Creating volcano plot...")
    
    # Load summaries from metadata
    baseline_summaries = metadata['baseline_summaries']
    finetuned_summaries = metadata['finetuned_summaries']
    
    # Split embeddings into ground truth and generated
    n_baseline = len(baseline_summaries['ground_truth'])
    n_finetuned = len(finetuned_summaries['ground_truth'])
    
    baseline_gt = baseline_embeddings[:n_baseline]
    baseline_gen = baseline_embeddings[n_baseline:]
    finetuned_gt = finetuned_embeddings[:n_finetuned]
    finetuned_gen = finetuned_embeddings[n_finetuned:]
    
    print(f"Baseline: {len(baseline_gt)} GT, {len(baseline_gen)} generated")
    print(f"Finetuned: {len(finetuned_gen)} generated")
    
    # Calculate cosine similarities for each pair
    baseline_similarities = []
    finetuned_similarities = []
    
    # Process each generated summary
    for i in range(len(finetuned_gen)):
        if i >= len(baseline_gen):
            continue
            
        gt_emb = finetuned_gt[i] if i < len(finetuned_gt) else finetuned_gt[0]
        baseline_gen_emb = baseline_gen[i]
        finetuned_gen_emb = finetuned_gen[i]
        
        # Calculate cosine similarities
        baseline_sim = cosine_similarity(gt_emb, baseline_gen_emb)
        finetuned_sim = cosine_similarity(gt_emb, finetuned_gen_emb)
        
        baseline_similarities.append(baseline_sim)
        finetuned_similarities.append(finetuned_sim)
        
        # Debug: print first few comparisons
        if i < 5:
            print(f"Sample {i}:")
            print(f"  GT: {metadata['finetuned_summaries']['ground_truth'][i][:100]}...")
            print(f"  Baseline sim: {baseline_sim:.3f}")
            print(f"  Finetuned sim: {finetuned_sim:.3f}")
            print(f"  Change: {finetuned_sim - baseline_sim:.3f}")
            print()
    
    baseline_similarities = np.array(baseline_similarities)
    finetuned_similarities = np.array(finetuned_similarities)
    
    print(f"Baseline similarities - Mean: {np.mean(baseline_similarities):.3f}, Std: {np.std(baseline_similarities):.3f}")
    print(f"Finetuned similarities - Mean: {np.mean(finetuned_similarities):.3f}, Std: {np.std(finetuned_similarities):.3f}")
    print(f"Overall improvement: {np.mean(finetuned_similarities) - np.mean(baseline_similarities):.3f}")
    
    # 1. Compute effect-size (log2 fold-change)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    log2fc = np.log2((finetuned_similarities + epsilon) / (baseline_similarities + epsilon))
    
    # Handle invalid values (replace inf/nan with reasonable bounds)
    log2fc = np.nan_to_num(log2fc, nan=0.0, posinf=5.0, neginf=-5.0)
    
    # 2. Compute statistical significance based on magnitude of change
    # Use the absolute change in cosine similarity as our "significance" measure
    abs_changes = np.abs(finetuned_similarities - baseline_similarities)
    
    # Define significance based on percentiles of the absolute changes
    # Top 25% of changes are considered "significant"
    significance_threshold = np.percentile(abs_changes, 75)
    
    # Use the absolute changes directly as the Y-axis (more intuitive)
    # Larger changes = higher Y values = more significant
    significance_values = abs_changes
    
    # 3. Create the volcano plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define thresholds
    log2fc_threshold = 0.5  # 1.4-fold change (more reasonable for cosine similarities)
    
    # Color points based on significance and effect size
    colors = []
    for l, sig in zip(log2fc, abs_changes):
        if l > log2fc_threshold and sig > significance_threshold:
            colors.append('red')  # Significant improvements
        elif l < -log2fc_threshold and sig > significance_threshold:
            colors.append('blue')  # Significant regressions
        else:
            colors.append('grey')  # Not significant or small effect
    
    # Plot points
    ax.scatter(log2fc, significance_values, c=colors, alpha=0.6, s=50)
    
    # Add threshold lines
    ax.axhline(significance_threshold, color='black', linestyle='--', alpha=0.7)
    ax.axvline(log2fc_threshold, color='black', linestyle='--', alpha=0.7)
    ax.axvline(-log2fc_threshold, color='black', linestyle='--', alpha=0.7)
    
    # Create custom legend for colors
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Significant Improvements'),
        Patch(facecolor='blue', alpha=0.6, label='Significant Regressions'),
        Patch(facecolor='grey', alpha=0.6, label='Small Changes'),
        Line2D([0], [0], color='black', linestyle='--', label=f'Significance threshold'),
        Line2D([0], [0], color='black', linestyle='--', label=f'log₂FC = ±{log2fc_threshold}')
    ]
    
    # Customize the plot
    ax.set_xlabel('log₂ Fold-Change in Cosine Similarity', fontsize=12)
    ax.set_ylabel('Magnitude of Change (|Δ Similarity|)', fontsize=12)
    ax.set_title('LoRA Finetuning Impact on Embedding Similarity', fontsize=14, fontweight='bold')
    
    # Add combined legend
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    significant_up = sum(1 for c in colors if c == 'red')
    significant_down = sum(1 for c in colors if c == 'blue')
    total_significant = significant_up + significant_down
    
    stats_text = f'Total samples: {len(log2fc)}\n'
    stats_text += f'Significant improvements: {significant_up}\n'
    stats_text += f'Significant regressions: {significant_down}\n'
    stats_text += f'Significant changes: {total_significant}/{len(log2fc)} ({total_significant/len(log2fc)*100:.1f}%)\n'
    stats_text += f'Mean log₂FC: {np.mean(log2fc):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Volcano plot saved to {output_file}")
    
    # Print statistics
    print(f"\n=== Volcano Plot Statistics ===")
    print(f"Total samples: {len(log2fc)}")
    print(f"Mean log₂ fold-change: {np.mean(log2fc):.3f}")
    print(f"Median log₂ fold-change: {np.median(log2fc):.3f}")
    print(f"Significant improvements (red): {significant_up}")
    print(f"Significant regressions (blue): {significant_down}")
    print(f"Non-significant changes (grey): {len(log2fc) - total_significant}")
    print(f"Max improvement: {max(log2fc):.3f}")
    print(f"Max regression: {min(log2fc):.3f}")
    print(f"Mean absolute change: {np.mean(abs_changes):.3f}")
    print(f"Significance threshold: {significance_threshold:.3f}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Create volcano plot from embeddings")
    parser.add_argument("--embeddings_dir", default="v9/embeddings", help="Directory containing embeddings")
    parser.add_argument("--output_dir", default="v9/analysis_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load embeddings
    baseline_file = f"{args.embeddings_dir}/baseline_embeddings.pkl"
    finetuned_file = f"{args.embeddings_dir}/finetuned_embeddings.pkl"
    metadata_file = f"{args.embeddings_dir}/embedding_metadata.json"
    
    print("Loading embeddings...")
    baseline_embeddings = load_embeddings(baseline_file)
    finetuned_embeddings = load_embeddings(finetuned_file)
    metadata = load_metadata(metadata_file)
    
    # Create volcano plot
    output_file = f"{args.output_dir}/volcano_plot.png"
    create_volcano_plot(baseline_embeddings, finetuned_embeddings, metadata, output_file)
    
    print("✓ Volcano plot analysis completed!")

if __name__ == "__main__":
    main() 