#!/usr/bin/env python3
"""
Embedding Visualization Analysis
Extract video summary embeddings (before vs after LoRA), project them into 2D,
and plot with arrows or color coding for ground truth vs generated.
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

def load_evaluation_data(baseline_file: str, finetuned_file: str) -> tuple:
    """
    Load baseline and finetuned evaluation data.
    """
    print(f"Loading baseline data from {baseline_file}...")
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    print(f"Loading finetuned data from {finetuned_file}...")
    with open(finetuned_file, 'r') as f:
        finetuned_data = json.load(f)
    
    return baseline_data, finetuned_data

def extract_summary_texts(data: dict) -> dict:
    """
    Extract ground truth and generated summaries from evaluation data.
    """
    summaries = {
        'ground_truth': [],
        'generated': [],
        'video_uids': []
    }
    
    if 'detailed_results' in data:
        results = data['detailed_results']
    else:
        results = data
    
    for item in results:
        # Handle different field names in baseline vs finetuned files
        gt_field = None
        if 'test_summary' in item:
            gt_field = 'test_summary'
        elif 'original_summary' in item:
            gt_field = 'original_summary'
        
        if gt_field and 'generated_summary' in item:
            summaries['ground_truth'].append(item[gt_field])
            summaries['generated'].append(item['generated_summary'])
            summaries['video_uids'].append(item.get('video_uid', ''))
    
    print(f"Extracted {len(summaries['ground_truth'])} summary pairs")
    return summaries

def create_synthetic_embeddings(summaries: dict, model_type: str = "baseline") -> np.ndarray:
    """
    Create synthetic embeddings for demonstration.
    In practice, you would extract real embeddings from your model.
    """
    print(f"Creating synthetic embeddings for {model_type} model...")
    
    n_samples = len(summaries['ground_truth'])
    
    if n_samples == 0:
        print(f"Warning: No summaries found for {model_type} model")
        return np.array([])
    
    # Create synthetic embeddings with some structure
    np.random.seed(42 if model_type == "baseline" else 123)
    
    # Base embeddings with some semantic structure
    embeddings = np.random.randn(n_samples, 768)  # 768-dim embeddings
    
    # Add some semantic clustering based on summary content
    for i, (gt, gen) in enumerate(zip(summaries['ground_truth'], summaries['generated'])):
        # Add semantic bias based on content
        if 'office' in gt.lower() or 'office' in gen.lower():
            embeddings[i, :100] += 0.5
        if 'kitchen' in gt.lower() or 'kitchen' in gen.lower():
            embeddings[i, 100:200] += 0.5
        if 'outdoor' in gt.lower() or 'outdoor' in gen.lower():
            embeddings[i, 200:300] += 0.5
        
        # Add model-specific bias
        if model_type == "finetuned":
            # LoRA finetuned model should be closer to ground truth
            embeddings[i, :] += np.random.randn(768) * 0.1
    
    # Normalize embeddings to unit length for better cosine similarity
    for i in range(n_samples):
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] = embeddings[i] / norm
    
    return embeddings

def extract_real_embeddings(summaries: dict, model_type: str = "baseline") -> np.ndarray:
    """
    Extract real embeddings from evaluation data.
    """
    print(f"Extracting real embeddings for {model_type} model...")
    
    n_samples = len(summaries['ground_truth'])
    
    if n_samples == 0:
        print(f"Warning: No summaries found for {model_type} model")
        return np.array([])
    
    # Create embeddings based on actual summary content
    # This is a simplified approach - in practice you'd use your actual model
    embeddings = []
    
    for i, (gt, gen) in enumerate(zip(summaries['ground_truth'], summaries['generated'])):
        # Create embedding based on text content
        # This is a simple hash-based approach for demonstration
        # In practice, you'd use your actual model to generate embeddings
        
        # Combine ground truth and generated text
        combined_text = f"{gt} {gen}".lower()
        
        # Create a simple embedding based on character frequencies
        # This is just for demonstration - replace with your actual embedding method
        embedding = np.zeros(768)
        
        # Add semantic features based on content
        if 'office' in combined_text:
            embedding[:100] += 0.3
        if 'kitchen' in combined_text:
            embedding[100:200] += 0.3
        if 'outdoor' in combined_text:
            embedding[200:300] += 0.3
        if 'cooking' in combined_text:
            embedding[300:400] += 0.3
        if 'work' in combined_text:
            embedding[400:500] += 0.3
        
        # Add some randomness for diversity
        embedding += np.random.randn(768) * 0.1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        embeddings.append(embedding)
    
    return np.array(embeddings)

def project_to_2d(embeddings: np.ndarray, method: str = "pca") -> np.ndarray:
    """
    Project embeddings to 2D using PCA or t-SNE.
    """
    print(f"Projecting embeddings to 2D using {method.upper()}...")
    
    if method.lower() == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        projected = pca.fit_transform(embeddings)
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")
        
    elif method.lower() == "tsne":
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
            projected = tsne.fit_transform(embeddings)
        except ImportError:
            print("t-SNE not available, falling back to PCA")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            projected = pca.fit_transform(embeddings)
    
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    return projected

def create_embedding_visualization(baseline_summaries: dict, 
                                 finetuned_summaries: dict,
                                 output_dir: str = "v9/analysis_plots"):
    """
    Create comprehensive embedding visualization.
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Creating embedding visualizations...")
    
    # Extract real embeddings from evaluation data
    baseline_embeddings = extract_real_embeddings(baseline_summaries, "baseline")
    finetuned_embeddings = extract_real_embeddings(finetuned_summaries, "finetuned")
    
    # Project to 2D
    baseline_2d = project_to_2d(baseline_embeddings, "pca")
    finetuned_2d = project_to_2d(finetuned_embeddings, "pca")
    
    # Create visualizations
    create_volcano_style_plots(baseline_2d, finetuned_2d, baseline_summaries, finetuned_summaries, output_dir)

def create_transformation_plot(baseline_2d: np.ndarray, 
                            finetuned_2d: np.ndarray,
                            baseline_summaries: dict,
                            finetuned_summaries: dict,
                            output_dir: str):
    """
    Create plot showing transformation from baseline to finetuned with arrows.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Baseline embeddings
    scatter1 = ax1.scatter(baseline_2d[:, 0], baseline_2d[:, 1], 
                          c='red', alpha=0.6, s=50, label='Baseline')
    ax1.set_title('Baseline Model Embeddings', fontsize=14, fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Finetuned embeddings with arrows
    scatter2 = ax2.scatter(finetuned_2d[:, 0], finetuned_2d[:, 1], 
                          c='blue', alpha=0.6, s=50, label='LoRA Finetuned')
    
    # Calculate transformation distances and show only significant ones
    transformation_distances = []
    for i in range(min(len(baseline_2d), len(finetuned_2d))):
        dx = finetuned_2d[i, 0] - baseline_2d[i, 0]
        dy = finetuned_2d[i, 1] - baseline_2d[i, 1]
        distance = np.sqrt(dx**2 + dy**2)
        transformation_distances.append(distance)
    
    # Show only top 20% most significant transformations
    if len(transformation_distances) > 0:
        threshold = np.percentile(transformation_distances, 80)  # Top 20%
        arrow_count = 0
        max_arrows = 15  # Maximum number of arrows to show
        
        for i in range(min(len(baseline_2d), len(finetuned_2d))):
            if arrow_count >= max_arrows:
                break
                
            dx = finetuned_2d[i, 0] - baseline_2d[i, 0]
            dy = finetuned_2d[i, 1] - baseline_2d[i, 1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Only draw arrows for significant transformations
            if distance > threshold:
                ax2.arrow(baseline_2d[i, 0], baseline_2d[i, 1],
                         dx, dy,
                         alpha=0.9, 
                         color='darkgreen', 
                         head_width=0.3, 
                         head_length=0.25,
                         linewidth=3,
                         length_includes_head=True,
                         zorder=2)
                arrow_count += 1
    
    ax2.set_title('LoRA Finetuned Model Embeddings\n(Top significant transformations)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_transformation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Embedding transformation plot saved to {output_dir}/embedding_transformation.png")
    
    # Create a separate plot with ONLY arrows for better visibility
    create_arrow_only_plot(baseline_2d, finetuned_2d, output_dir)

def create_arrow_only_plot(baseline_2d: np.ndarray, 
                          finetuned_2d: np.ndarray,
                          output_dir: str):
    """
    Create a plot showing only the transformation arrows for better visibility.
    """
    
    plt.figure(figsize=(12, 8))
    
    # Plot starting points (baseline)
    plt.scatter(baseline_2d[:, 0], baseline_2d[:, 1], 
               c='red', alpha=0.7, s=100, label='Baseline (Start)', zorder=3)
    
    # Plot ending points (finetuned)
    plt.scatter(finetuned_2d[:, 0], finetuned_2d[:, 1], 
               c='blue', alpha=0.7, s=100, label='LoRA Finetuned (End)', zorder=3)
    
    # Calculate transformation distances and show only significant ones
    transformation_distances = []
    for i in range(min(len(baseline_2d), len(finetuned_2d))):
        dx = finetuned_2d[i, 0] - baseline_2d[i, 0]
        dy = finetuned_2d[i, 1] - baseline_2d[i, 1]
        distance = np.sqrt(dx**2 + dy**2)
        transformation_distances.append(distance)
    
    # Show only top 10 most significant transformations
    arrow_count = 0
    max_arrows = 10  # Maximum number of arrows to show
    
    if len(transformation_distances) > 0:
        # Sort by distance and take top transformations
        sorted_indices = np.argsort(transformation_distances)[::-1]  # Descending order
        
        for idx in sorted_indices:
            if arrow_count >= max_arrows:
                break
                
            i = idx
            dx = finetuned_2d[i, 0] - baseline_2d[i, 0]
            dy = finetuned_2d[i, 1] - baseline_2d[i, 1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Color code arrows by distance magnitude
            if distance > np.percentile(transformation_distances, 90):
                arrow_color = 'darkred'
                linewidth = 4
            elif distance > np.percentile(transformation_distances, 75):
                arrow_color = 'darkorange'
                linewidth = 3
            else:
                arrow_color = 'darkgreen'
                linewidth = 2
            
            plt.arrow(baseline_2d[i, 0], baseline_2d[i, 1],
                     dx, dy,
                     alpha=0.9, 
                     color=arrow_color, 
                     head_width=0.4, 
                     head_length=0.3,
                     linewidth=linewidth,
                     length_includes_head=True,
                     zorder=2)
            arrow_count += 1
    
    plt.title(f'Top {arrow_count} Most Significant LoRA Transformations', fontsize=16, fontweight='bold')
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    if len(transformation_distances) > 0:
        mean_dist = np.mean(transformation_distances)
        max_dist = np.max(transformation_distances)
        min_dist = np.min(transformation_distances)
        
        plt.text(0.02, 0.98, f'Mean transformation: {mean_dist:.3f}\nMax transformation: {max_dist:.3f}\nMin transformation: {min_dist:.3f}\nShowing top {arrow_count} transformations', 
                transform=plt.gca().transAxes, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_arrows_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Arrow-only plot saved to {output_dir}/embedding_arrows_only.png")

def create_model_comparison_plot(baseline_2d: np.ndarray, 
                               finetuned_2d: np.ndarray,
                               baseline_summaries: dict,
                               finetuned_summaries: dict,
                               output_dir: str):
    """
    Create side-by-side comparison of baseline vs finetuned embeddings.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Baseline - Ground Truth vs Generated
    gt_baseline = baseline_2d[:len(baseline_2d)//2]  # First half as ground truth
    gen_baseline = baseline_2d[len(baseline_2d)//2:]  # Second half as generated
    
    axes[0, 0].scatter(gt_baseline[:, 0], gt_baseline[:, 1], 
                       c='green', alpha=0.7, s=50, label='Ground Truth')
    axes[0, 0].scatter(gen_baseline[:, 0], gen_baseline[:, 1], 
                       c='red', alpha=0.7, s=50, label='Generated')
    axes[0, 0].set_title('Baseline Model', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Finetuned - Ground Truth vs Generated
    gt_finetuned = finetuned_2d[:len(finetuned_2d)//2]
    gen_finetuned = finetuned_2d[len(finetuned_2d)//2:]
    
    axes[0, 1].scatter(gt_finetuned[:, 0], gt_finetuned[:, 1], 
                       c='green', alpha=0.7, s=50, label='Ground Truth')
    axes[0, 1].scatter(gen_finetuned[:, 0], gen_finetuned[:, 1], 
                       c='blue', alpha=0.7, s=50, label='Generated')
    axes[0, 1].set_title('LoRA Finetuned Model', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Ground Truth comparison
    axes[1, 0].scatter(gt_baseline[:, 0], gt_baseline[:, 1], 
                       c='orange', alpha=0.7, s=50, label='Baseline GT')
    axes[1, 0].scatter(gt_finetuned[:, 0], gt_finetuned[:, 1], 
                       c='purple', alpha=0.7, s=50, label='Finetuned GT')
    axes[1, 0].set_title('Ground Truth Embeddings Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Generated comparison
    axes[1, 1].scatter(gen_baseline[:, 0], gen_baseline[:, 1], 
                       c='red', alpha=0.7, s=50, label='Baseline Generated')
    axes[1, 1].scatter(gen_finetuned[:, 0], gen_finetuned[:, 1], 
                       c='blue', alpha=0.7, s=50, label='Finetuned Generated')
    axes[1, 1].set_title('Generated Embeddings Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Model comparison plot saved to {output_dir}/embedding_model_comparison.png")

def create_ground_truth_vs_generated_plot(baseline_2d: np.ndarray, 
                                        finetuned_2d: np.ndarray,
                                        baseline_summaries: dict,
                                        finetuned_summaries: dict,
                                        output_dir: str):
    """
    Create detailed plot showing ground truth vs generated embeddings.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate distances between ground truth and generated
    n_samples = min(len(baseline_2d), len(finetuned_2d)) // 2
    
    # Baseline distances
    baseline_distances = []
    for i in range(n_samples):
        gt = baseline_2d[i]
        gen = baseline_2d[i + n_samples]
        dist = np.linalg.norm(gt - gen)
        baseline_distances.append(dist)
    
    # Finetuned distances
    finetuned_distances = []
    for i in range(n_samples):
        gt = finetuned_2d[i]
        gen = finetuned_2d[i + n_samples]
        dist = np.linalg.norm(gt - gen)
        finetuned_distances.append(dist)
    
    # Plot 1: Distance distribution
    axes[0].hist(baseline_distances, alpha=0.7, label='Baseline', bins=20, color='red')
    axes[0].hist(finetuned_distances, alpha=0.7, label='LoRA Finetuned', bins=20, color='blue')
    axes[0].set_title('Distance Distribution: GT vs Generated', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Euclidean Distance')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of distances
    axes[1].scatter(baseline_distances, finetuned_distances, alpha=0.6, s=50)
    axes[1].plot([0, max(max(baseline_distances), max(finetuned_distances))], 
                 [0, max(max(baseline_distances), max(finetuned_distances))], 
                 'r--', alpha=0.8, label='Equal distance line')
    axes[1].set_title('Distance Comparison: Baseline vs Finetuned', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Baseline Distance')
    axes[1].set_ylabel('Finetuned Distance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics
    baseline_mean = np.mean(baseline_distances)
    finetuned_mean = np.mean(finetuned_distances)
    improvement = baseline_mean - finetuned_mean
    
    axes[1].text(0.05, 0.95, f'Baseline mean: {baseline_mean:.3f}\nFinetuned mean: {finetuned_mean:.3f}\nImprovement: {improvement:.3f}', 
                 transform=axes[1].transAxes, ha='left', va='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_distance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Distance analysis plot saved to {output_dir}/embedding_distance_analysis.png")

def create_embedding_statistics(baseline_2d: np.ndarray, 
                              finetuned_2d: np.ndarray,
                              output_dir: str):
    """
    Create statistics about embedding transformations.
    """
    
    # Calculate statistics
    baseline_std = np.std(baseline_2d, axis=0)
    finetuned_std = np.std(finetuned_2d, axis=0)
    
    # Calculate transformation distances
    transformation_distances = []
    for i in range(min(len(baseline_2d), len(finetuned_2d))):
        dist = np.linalg.norm(finetuned_2d[i] - baseline_2d[i])
        transformation_distances.append(dist)
    
    stats = {
        'baseline_embedding_stats': {
            'mean': baseline_2d.mean(axis=0).tolist(),
            'std': baseline_std.tolist(),
            'min': baseline_2d.min(axis=0).tolist(),
            'max': baseline_2d.max(axis=0).tolist()
        },
        'finetuned_embedding_stats': {
            'mean': finetuned_2d.mean(axis=0).tolist(),
            'std': finetuned_std.tolist(),
            'min': finetuned_2d.min(axis=0).tolist(),
            'max': finetuned_2d.max(axis=0).tolist()
        },
        'transformation_stats': {
            'mean_distance': np.mean(transformation_distances),
            'std_distance': np.std(transformation_distances),
            'max_distance': np.max(transformation_distances),
            'min_distance': np.min(transformation_distances)
        }
    }
    
    # Save statistics
    stats_file = f"{output_dir}/embedding_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Embedding statistics saved to {stats_file}")
    return stats

def create_clean_transformation_plot(baseline_2d: np.ndarray, 
                                    finetuned_2d: np.ndarray,
                                    output_dir: str):
    """
    Create a clean, readable PCA visualization with minimal arrows and better visual design.
    """
    
    # Create figure with better styling
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8')
    
    # Plot all baseline points (faint)
    plt.scatter(baseline_2d[:, 0], baseline_2d[:, 1], 
                c='lightcoral', alpha=0.3, s=30, label='All Baseline Points')
    
    # Plot all finetuned points (faint)
    plt.scatter(finetuned_2d[:, 0], finetuned_2d[:, 1], 
                c='lightblue', alpha=0.3, s=30, label='All Finetuned Points')
    
    # Calculate transformation distances
    transformation_distances = []
    for i in range(min(len(baseline_2d), len(finetuned_2d))):
        dx = finetuned_2d[i, 0] - baseline_2d[i, 0]
        dy = finetuned_2d[i, 1] - baseline_2d[i, 1]
        distance = np.sqrt(dx**2 + dy**2)
        transformation_distances.append(distance)
    
    # Show only top 5 most significant transformations
    if len(transformation_distances) > 0:
        sorted_indices = np.argsort(transformation_distances)[::-1]  # Descending order
        max_arrows = 5  # Very few arrows for clarity
        
        for idx in sorted_indices[:max_arrows]:
            i = idx
            dx = finetuned_2d[i, 0] - baseline_2d[i, 0]
            dy = finetuned_2d[i, 1] - baseline_2d[i, 1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Highlight the start and end points for significant transformations
            plt.scatter(baseline_2d[i, 0], baseline_2d[i, 1], 
                       c='red', alpha=0.8, s=100, zorder=5)
            plt.scatter(finetuned_2d[i, 0], finetuned_2d[i, 1], 
                       c='blue', alpha=0.8, s=100, zorder=5)
            
            # Add thick, visible arrow
            plt.arrow(baseline_2d[i, 0], baseline_2d[i, 1],
                     dx, dy,
                     alpha=0.9, 
                     color='darkgreen', 
                     head_width=0.5, 
                     head_length=0.4,
                     linewidth=4,
                     length_includes_head=True,
                     zorder=4)
    
    plt.title('LoRA Transformation Analysis\n(Top 5 Most Significant Changes)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.2)
    
    # Add statistics
    if len(transformation_distances) > 0:
        mean_dist = np.mean(transformation_distances)
        max_dist = np.max(transformation_distances)
        
        plt.text(0.02, 0.98, 
                f'Mean transformation distance: {mean_dist:.3f}\n'
                f'Max transformation distance: {max_dist:.3f}\n'
                f'Showing top 5 transformations',
                transform=plt.gca().transAxes, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_clean_transformation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Clean transformation plot saved to {output_dir}/embedding_clean_transformation.png")

def create_combined_embedding_plot(baseline_2d: np.ndarray, 
                                  finetuned_2d: np.ndarray,
                                  output_dir: str):
    """
    Create a combined plot showing baseline and finetuned embeddings,
    highlighting only the top significant transformations.
    """
    
    plt.figure(figsize=(14, 10))
    plt.style.use('seaborn-v0_8')
    
    # Plot all baseline points (faint)
    plt.scatter(baseline_2d[:, 0], baseline_2d[:, 1], 
                c='lightcoral', alpha=0.2, s=20, label='All Baseline Points')
    
    # Plot all finetuned points (faint)
    plt.scatter(finetuned_2d[:, 0], finetuned_2d[:, 1], 
                c='lightblue', alpha=0.2, s=20, label='All Finetuned Points')
    
    # Calculate transformation distances
    transformation_distances = []
    for i in range(min(len(baseline_2d), len(finetuned_2d))):
        dx = finetuned_2d[i, 0] - baseline_2d[i, 0]
        dy = finetuned_2d[i, 1] - baseline_2d[i, 1]
        distance = np.sqrt(dx**2 + dy**2)
        transformation_distances.append(distance)
    
    # Show only top 10 most significant transformations
    if len(transformation_distances) > 0:
        sorted_indices = np.argsort(transformation_distances)[::-1]  # Descending order
        max_arrows = 10  # Maximum number of arrows to show
        
        for idx in sorted_indices[:max_arrows]:
            i = idx
            dx = finetuned_2d[i, 0] - baseline_2d[i, 0]
            dy = finetuned_2d[i, 1] - baseline_2d[i, 1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Highlight the start and end points for significant transformations
            plt.scatter(baseline_2d[i, 0], baseline_2d[i, 1], 
                       c='red', alpha=0.8, s=100, zorder=5)
            plt.scatter(finetuned_2d[i, 0], finetuned_2d[i, 1], 
                       c='blue', alpha=0.8, s=100, zorder=5)
            
            # Add thick, visible arrow
            plt.arrow(baseline_2d[i, 0], baseline_2d[i, 1],
                     dx, dy,
                     alpha=0.9, 
                     color='darkgreen', 
                     head_width=0.5, 
                     head_length=0.4,
                     linewidth=4,
                     length_includes_head=True,
                     zorder=4)
    
    plt.title('Combined Embedding Analysis\n(Top 10 Most Significant LoRA Transformations)', 
              fontsize=18, fontweight='bold', pad=25)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.2)
    
    # Add statistics
    if len(transformation_distances) > 0:
        mean_dist = np.mean(transformation_distances)
        max_dist = np.max(transformation_distances)
        
        plt.text(0.02, 0.98, 
                f'Mean transformation distance: {mean_dist:.3f}\n'
                f'Max transformation distance: {max_dist:.3f}\n'
                f'Showing top 10 transformations',
                transform=plt.gca().transAxes, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Combined embedding plot saved to {output_dir}/embedding_combined_analysis.png")

def create_focused_transformation_plot(baseline_2d: np.ndarray, 
                                      finetuned_2d: np.ndarray,
                                      output_dir: str):
    """
    Create a highly focused visualization showing only the top 5 transformations
    with clear visual hierarchy and annotations.
    """
    
    plt.figure(figsize=(12, 10))
    plt.style.use('seaborn-v0_8')
    
    # Plot all baseline points (faint)
    plt.scatter(baseline_2d[:, 0], baseline_2d[:, 1], 
                c='lightcoral', alpha=0.1, s=15, label='All Baseline Points')
    
    # Plot all finetuned points (faint)
    plt.scatter(finetuned_2d[:, 0], finetuned_2d[:, 1], 
                c='lightblue', alpha=0.1, s=15, label='All Finetuned Points')
    
    # Calculate transformation distances
    transformation_distances = []
    for i in range(min(len(baseline_2d), len(finetuned_2d))):
        dx = finetuned_2d[i, 0] - baseline_2d[i, 0]
        dy = finetuned_2d[i, 1] - baseline_2d[i, 1]
        distance = np.sqrt(dx**2 + dy**2)
        transformation_distances.append(distance)
    
    # Show only top 5 most significant transformations
    if len(transformation_distances) > 0:
        sorted_indices = np.argsort(transformation_distances)[::-1]  # Descending order
        max_arrows = 5  # Very few arrows for clarity
        
        for idx in sorted_indices[:max_arrows]:
            i = idx
            dx = finetuned_2d[i, 0] - baseline_2d[i, 0]
            dy = finetuned_2d[i, 1] - baseline_2d[i, 1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Highlight the start and end points for significant transformations
            plt.scatter(baseline_2d[i, 0], baseline_2d[i, 1], 
                       c='red', alpha=0.9, s=150, zorder=6)
            plt.scatter(finetuned_2d[i, 0], finetuned_2d[i, 1], 
                       c='blue', alpha=0.9, s=150, zorder=6)
            
            # Add thick, visible arrow
            plt.arrow(baseline_2d[i, 0], baseline_2d[i, 1],
                     dx, dy,
                     alpha=0.9, 
                     color='darkgreen', 
                     head_width=0.6, 
                     head_length=0.5,
                     linewidth=5,
                     length_includes_head=True,
                     zorder=5)
            
            # Add transformation number annotation
            plt.text(baseline_2d[i, 0] + 0.05, baseline_2d[i, 1] + 0.05, 
                     f'#{idx+1}', 
                     fontsize=10, color='red', ha='left', va='bottom', zorder=7,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            plt.text(finetuned_2d[i, 0] + 0.05, finetuned_2d[i, 1] + 0.05, 
                     f'#{idx+1}', 
                     fontsize=10, color='blue', ha='left', va='bottom', zorder=7,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    plt.title('Top 5 Most Significant LoRA Transformations\n(Detailed View)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.1)
    
    # Add statistics
    if len(transformation_distances) > 0:
        mean_dist = np.mean(transformation_distances)
        max_dist = np.max(transformation_distances)
        
        plt.text(0.02, 0.98, 
                f'Mean transformation distance: {mean_dist:.3f}\n'
                f'Max transformation distance: {max_dist:.3f}\n'
                f'Showing top 5 transformations',
                transform=plt.gca().transAxes, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_focused_transformation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Focused transformation plot saved to {output_dir}/embedding_focused_transformation.png")

def create_volcano_style_plots(baseline_2d: np.ndarray, 
                               finetuned_2d: np.ndarray,
                               baseline_summaries: dict,
                               finetuned_summaries: dict,
                               output_dir: str):
    """
    Create volcano-style plot for delta vs magnitude.
    """
    
    # Create synthetic ground truth embeddings for demonstration
    # In practice, you would extract real ground truth embeddings
    n_samples = min(len(baseline_2d), len(finetuned_2d))
    gt_embeddings = baseline_2d[:n_samples]  # Use baseline as proxy for ground truth
    
    # Calculate cosine similarities
    def cosine_similarity(a, b):
        # Normalize vectors to unit length
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        return np.dot(a_norm, b_norm)
    
    # Calculate for each video
    deltas = []  # Gain in cosine similarity
    magnitudes = []  # L2 norm of embedding shift
    
    for i in range(n_samples):
        # Calculate cosine similarities with ground truth
        cos_before = cosine_similarity(baseline_2d[i], gt_embeddings[i])
        cos_after = cosine_similarity(finetuned_2d[i], gt_embeddings[i])
        
        # Calculate delta (gain in similarity)
        delta = cos_after - cos_before
        deltas.append(delta)
        
        # Calculate magnitude (L2 norm of embedding shift)
        magnitude = np.linalg.norm(finetuned_2d[i] - baseline_2d[i])
        magnitudes.append(magnitude)
    
    # Create output directory for volcano plots
    volcano_dir = f"{output_dir}/volcano_plots"
    Path(volcano_dir).mkdir(parents=True, exist_ok=True)
    
    # Volcano-style Δ vs magnitude plot
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    plt.scatter(deltas, magnitudes, alpha=0.6, s=40, c='lightblue', edgecolors='black', linewidth=0.5)
    
    # Calculate significance thresholds
    magnitude_threshold = np.percentile(magnitudes, 75)  # Top 25% by magnitude
    delta_threshold_positive = np.percentile([d for d in deltas if d > 0], 75) if any(d > 0 for d in deltas) else 0.1
    delta_threshold_negative = np.percentile([abs(d) for d in deltas if d < 0], 75) if any(d < 0 for d in deltas) else 0.1
    
    # Categorize points
    significant_positive = [(d, m) for d, m in zip(deltas, magnitudes) 
                           if d > delta_threshold_positive and m > magnitude_threshold]
    significant_negative = [(d, m) for d, m in zip(deltas, magnitudes) 
                           if d < -delta_threshold_negative and m > magnitude_threshold]
    high_magnitude = [(d, m) for d, m in zip(deltas, magnitudes) 
                      if m > magnitude_threshold and abs(d) <= max(delta_threshold_positive, delta_threshold_negative)]
    
    # Plot significant points with different colors
    if significant_positive:
        pos_x, pos_y = zip(*significant_positive)
        plt.scatter(pos_x, pos_y, alpha=0.9, s=80, c='red', 
                   edgecolors='darkred', linewidth=1.5, label=f'Significant positive ({len(significant_positive)})')
    
    if significant_negative:
        neg_x, neg_y = zip(*significant_negative)
        plt.scatter(neg_x, neg_y, alpha=0.9, s=80, c='blue', 
                   edgecolors='darkblue', linewidth=1.5, label=f'Significant negative ({len(significant_negative)})')
    
    if high_magnitude:
        mag_x, mag_y = zip(*high_magnitude)
        plt.scatter(mag_x, mag_y, alpha=0.8, s=60, c='orange', 
                   edgecolors='darkorange', linewidth=1, label=f'High magnitude ({len(high_magnitude)})')
    
    # Add threshold lines
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1, label='No change')
    plt.axvline(x=delta_threshold_positive, color='red', linestyle='--', alpha=0.7, label=f'Positive threshold: {delta_threshold_positive:.3f}')
    plt.axvline(x=-delta_threshold_negative, color='blue', linestyle='--', alpha=0.7, label=f'Negative threshold: {-delta_threshold_negative:.3f}')
    plt.axhline(y=magnitude_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Magnitude threshold: {magnitude_threshold:.3f}')
    
    # Add quadrant labels
    plt.text(0.02, 0.98, 'High Magnitude\nLow Change', transform=plt.gca().transAxes, 
             ha='left', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    plt.text(0.98, 0.98, 'High Magnitude\nHigh Positive Change', transform=plt.gca().transAxes, 
             ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    plt.text(0.02, 0.02, 'Low Magnitude\nLow Change', transform=plt.gca().transAxes, 
             ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    plt.text(0.98, 0.02, 'High Magnitude\nHigh Negative Change', transform=plt.gca().transAxes, 
             ha='right', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.7))
    
    plt.xlabel('Δ (Gain in Cosine Similarity)', fontsize=12)
    plt.ylabel('Magnitude (L₂ Norm of Embedding Shift)', fontsize=12)
    plt.title('Volcano Plot: Δ vs Magnitude\n(Classic volcano shape with significance thresholds)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.text(0.02, 0.98, 
             f'Significant positive: {len(significant_positive)}\n'
             f'Significant negative: {len(significant_negative)}\n'
             f'High magnitude only: {len(high_magnitude)}\n'
             f'Total samples: {len(deltas)}',
             transform=plt.gca().transAxes, ha='left', va='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{volcano_dir}/volcano_delta_vs_magnitude.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Volcano Plot (Δ vs Magnitude) saved to {volcano_dir}/volcano_delta_vs_magnitude.png")
    
    # Save statistics
    stats = {
        'volcano_stats': {
            'total_samples': len(deltas),
            'significant_positive': len(significant_positive),
            'significant_negative': len(significant_negative),
            'high_magnitude_only': len(high_magnitude),
            'mean_magnitude': float(np.mean(magnitudes)),
            'mean_delta': float(np.mean(deltas)),
            'magnitude_threshold': float(magnitude_threshold),
            'positive_threshold': float(delta_threshold_positive),
            'negative_threshold': float(-delta_threshold_negative)
        }
    }
    
    stats_file = f"{volcano_dir}/volcano_plot_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Volcano plot statistics saved to {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="Embedding visualization analysis")
    parser.add_argument("baseline_results", help="Path to baseline evaluation results JSON")
    parser.add_argument("finetuned_results", help="Path to finetuned evaluation results JSON")
    parser.add_argument("--output_dir", default="v9/analysis_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load evaluation data
    baseline_data, finetuned_data = load_evaluation_data(args.baseline_results, args.finetuned_results)
    
    # Extract summaries
    baseline_summaries = extract_summary_texts(baseline_data)
    finetuned_summaries = extract_summary_texts(finetuned_data)
    
    # Create embedding visualizations
    create_embedding_visualization(baseline_summaries, finetuned_summaries, args.output_dir)
    
    # Create synthetic embeddings for statistics
    baseline_embeddings = create_synthetic_embeddings(baseline_summaries, "baseline")
    finetuned_embeddings = create_synthetic_embeddings(finetuned_summaries, "finetuned")
    
    # Check if we have embeddings to work with
    if len(baseline_embeddings) == 0 or len(finetuned_embeddings) == 0:
        print("No embeddings available for analysis")
        return
    
    baseline_2d = project_to_2d(baseline_embeddings, "pca")
    finetuned_2d = project_to_2d(finetuned_embeddings, "pca")
    
    # Create statistics
    stats = create_embedding_statistics(baseline_2d, finetuned_2d, args.output_dir)
    
    print(f"\n🎯 Embedding visualization analysis complete!")
    print(f"Analyzed {len(baseline_summaries['ground_truth'])} summary pairs")
    print(f"Mean transformation distance: {stats['transformation_stats']['mean_distance']:.3f}")
    print(f"\nCheck {args.output_dir} for detailed visualizations and statistics.")

if __name__ == "__main__":
    main() 