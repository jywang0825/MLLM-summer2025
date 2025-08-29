#!/usr/bin/env python3
"""
Hyperparameter Heatmap Analysis
Create heatmaps showing BLEU-1 scores or validation loss for different combinations of:
- LoRA rank vs learning rate
- LoRA rank vs LoRA alpha
- Learning rate vs LoRA alpha
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

def load_training_logs(log_file: str) -> pd.DataFrame:
    """
    Load training logs and extract hyperparameter information.
    """
    print(f"Loading training logs from {log_file}...")
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    # Extract log history
    log_history = data.get('log_history', [])
    
    # Convert to DataFrame
    df = pd.DataFrame(log_history)
    
    print(f"Loaded {len(df)} training steps")
    return df

def extract_hyperparameters_from_config(config_files: list) -> pd.DataFrame:
    """
    Extract hyperparameter information from config files.
    """
    hyperparams = []
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Extract relevant hyperparameters
            lora_rank = config.get('lora_r', config.get('lora_rank', None))
            lora_alpha = config.get('lora_alpha', None)
            learning_rate = config.get('learning_rate', None)
            
            # Extract from training args if available
            if 'training_args' in config:
                training_args = config['training_args']
                lora_rank = lora_rank or training_args.get('lora_r', None)
                lora_alpha = lora_alpha or training_args.get('lora_alpha', None)
                learning_rate = learning_rate or training_args.get('learning_rate', None)
            
            hyperparams.append({
                'config_file': config_file,
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
                'learning_rate': learning_rate
            })
            
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    return pd.DataFrame(hyperparams)

def create_hyperparameter_heatmap(hyperparams_df: pd.DataFrame, 
                                 evaluation_results: dict,
                                 output_dir: str = "v9/analysis_plots"):
    """
    Create heatmaps for different hyperparameter combinations.
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Creating hyperparameter heatmaps...")
    
    # Prepare data for heatmaps
    heatmap_data = []
    
    for _, row in hyperparams_df.iterrows():
        config_file = row['config_file']
        lora_rank = row['lora_rank']
        lora_alpha = row['lora_alpha']
        learning_rate = row['learning_rate']
        
        # Find corresponding evaluation results
        # You might need to adjust this based on your file naming convention
        config_name = Path(config_file).stem
        
        # Look for evaluation results that match this config
        eval_key = None
        for key in evaluation_results.keys():
            if config_name in key or key in config_name:
                eval_key = key
                break
        
        if eval_key:
            eval_data = evaluation_results[eval_key]
            bleu_1 = eval_data.get('average_scores', {}).get('bleu_1', 0)
            eval_loss = eval_data.get('average_scores', {}).get('eval_loss', 0)
        else:
            bleu_1 = 0
            eval_loss = 0
        
        heatmap_data.append({
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'learning_rate': learning_rate,
            'bleu_1': bleu_1,
            'eval_loss': eval_loss,
            'config_name': config_name
        })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    if len(heatmap_df) == 0:
        print("No hyperparameter data found. Creating example heatmap...")
        create_example_heatmap(output_dir)
        return
    
    # Create heatmaps
    create_heatmap_plots(heatmap_df, output_dir)

def create_heatmap_plots(df: pd.DataFrame, output_dir: str):
    """
    Create various heatmap plots for hyperparameter analysis.
    """
    
    # 1. LoRA Rank vs Learning Rate (BLEU-1)
    if 'lora_rank' in df.columns and 'learning_rate' in df.columns:
        create_rank_lr_heatmap(df, 'bleu_1', output_dir, 'BLEU-1 Score')
        create_rank_lr_heatmap(df, 'eval_loss', output_dir, 'Validation Loss')
    
    # 2. LoRA Rank vs LoRA Alpha (BLEU-1)
    if 'lora_rank' in df.columns and 'lora_alpha' in df.columns:
        create_rank_alpha_heatmap(df, 'bleu_1', output_dir, 'BLEU-1 Score')
        create_rank_alpha_heatmap(df, 'eval_loss', output_dir, 'Validation Loss')
    
    # 3. Learning Rate vs LoRA Alpha (BLEU-1)
    if 'learning_rate' in df.columns and 'lora_alpha' in df.columns:
        create_lr_alpha_heatmap(df, 'bleu_1', output_dir, 'BLEU-1 Score')
        create_lr_alpha_heatmap(df, 'eval_loss', output_dir, 'Validation Loss')
    
    # 4. 3D scatter plot
    create_3d_scatter(df, output_dir)
    
    # 5. Summary statistics
    create_hyperparameter_summary(df, output_dir)

def create_rank_lr_heatmap(df: pd.DataFrame, metric: str, output_dir: str, title: str):
    """
    Create heatmap for LoRA Rank vs Learning Rate.
    """
    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        values=metric, 
        index='lora_rank', 
        columns='learning_rate', 
        aggfunc='mean'
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': title})
    plt.title(f'LoRA Rank vs Learning Rate - {title}', fontsize=14, fontweight='bold')
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('LoRA Rank', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lora_rank_vs_lr_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created LoRA Rank vs Learning Rate heatmap for {metric}")

def create_rank_alpha_heatmap(df: pd.DataFrame, metric: str, output_dir: str, title: str):
    """
    Create heatmap for LoRA Rank vs LoRA Alpha.
    """
    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        values=metric, 
        index='lora_rank', 
        columns='lora_alpha', 
        aggfunc='mean'
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': title})
    plt.title(f'LoRA Rank vs LoRA Alpha - {title}', fontsize=14, fontweight='bold')
    plt.xlabel('LoRA Alpha', fontsize=12)
    plt.ylabel('LoRA Rank', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lora_rank_vs_alpha_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created LoRA Rank vs LoRA Alpha heatmap for {metric}")

def create_lr_alpha_heatmap(df: pd.DataFrame, metric: str, output_dir: str, title: str):
    """
    Create heatmap for Learning Rate vs LoRA Alpha.
    """
    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        values=metric, 
        index='learning_rate', 
        columns='lora_alpha', 
        aggfunc='mean'
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': title})
    plt.title(f'Learning Rate vs LoRA Alpha - {title}', fontsize=14, fontweight='bold')
    plt.xlabel('LoRA Alpha', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lr_vs_alpha_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created Learning Rate vs LoRA Alpha heatmap for {metric}")

def create_3d_scatter(df: pd.DataFrame, output_dir: str):
    """
    Create 3D scatter plot for hyperparameter analysis.
    """
    if len(df) < 3:
        print("Not enough data points for 3D scatter plot")
        return
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: LoRA Rank vs Learning Rate vs BLEU-1
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(df['lora_rank'], df['learning_rate'], df['bleu_1'], 
                           c=df['bleu_1'], cmap='viridis', s=50)
    ax1.set_xlabel('LoRA Rank')
    ax1.set_ylabel('Learning Rate')
    ax1.set_zlabel('BLEU-1 Score')
    ax1.set_title('LoRA Rank vs LR vs BLEU-1')
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot 2: LoRA Rank vs LoRA Alpha vs BLEU-1
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(df['lora_rank'], df['lora_alpha'], df['bleu_1'], 
                           c=df['bleu_1'], cmap='viridis', s=50)
    ax2.set_xlabel('LoRA Rank')
    ax2.set_ylabel('LoRA Alpha')
    ax2.set_zlabel('BLEU-1 Score')
    ax2.set_title('LoRA Rank vs Alpha vs BLEU-1')
    plt.colorbar(scatter2, ax=ax2)
    
    # Plot 3: Learning Rate vs LoRA Alpha vs BLEU-1
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(df['learning_rate'], df['lora_alpha'], df['bleu_1'], 
                           c=df['bleu_1'], cmap='viridis', s=50)
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('LoRA Alpha')
    ax3.set_zlabel('BLEU-1 Score')
    ax3.set_title('LR vs Alpha vs BLEU-1')
    plt.colorbar(scatter3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hyperparameter_3d_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created 3D scatter plots")

def create_hyperparameter_summary(df: pd.DataFrame, output_dir: str):
    """
    Create summary statistics for hyperparameter analysis.
    """
    summary = {
        'best_bleu1_config': df.loc[df['bleu_1'].idxmax()].to_dict() if len(df) > 0 else {},
        'best_eval_loss_config': df.loc[df['eval_loss'].idxmin()].to_dict() if len(df) > 0 else {},
        'hyperparameter_ranges': {
            'lora_rank': {'min': df['lora_rank'].min(), 'max': df['lora_rank'].max()} if 'lora_rank' in df.columns else {},
            'lora_alpha': {'min': df['lora_alpha'].min(), 'max': df['lora_alpha'].max()} if 'lora_alpha' in df.columns else {},
            'learning_rate': {'min': df['learning_rate'].min(), 'max': df['learning_rate'].max()} if 'learning_rate' in df.columns else {}
        },
        'performance_stats': {
            'bleu1_mean': df['bleu_1'].mean(),
            'bleu1_std': df['bleu_1'].std(),
            'eval_loss_mean': df['eval_loss'].mean(),
            'eval_loss_std': df['eval_loss'].std()
        },
        'total_experiments': len(df)
    }
    
    # Save summary
    with open(f'{output_dir}/hyperparameter_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Created hyperparameter summary")
    return summary

def create_example_heatmap(output_dir: str):
    """
    Create an example heatmap with synthetic data for demonstration.
    """
    print("Creating example heatmap with synthetic data...")
    
    # Create synthetic data
    lora_ranks = [8, 16, 32, 64, 128]
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
    
    # Create synthetic BLEU-1 scores (higher for moderate rank and LR)
    data = []
    for rank in lora_ranks:
        for lr in learning_rates:
            # Synthetic score: optimal around rank=32, lr=5e-5
            score = 0.3 + 0.4 * np.exp(-((rank-32)**2 + (lr-5e-5)**2*1e10) / 1000)
            data.append({'lora_rank': rank, 'learning_rate': lr, 'bleu_1': score})
    
    df = pd.DataFrame(data)
    
    # Create heatmap
    pivot_df = df.pivot_table(values='bleu_1', index='lora_rank', columns='learning_rate')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'BLEU-1 Score'})
    plt.title('Example: LoRA Rank vs Learning Rate - BLEU-1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('LoRA Rank', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example_hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created example hyperparameter heatmap")

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter heatmap analysis")
    parser.add_argument("--config_files", nargs='+', help="Paths to config files")
    parser.add_argument("--evaluation_results", help="Path to evaluation results JSON")
    parser.add_argument("--output_dir", default="v9/analysis_plots", help="Output directory for plots")
    parser.add_argument("--example", action="store_true", help="Create example heatmap")
    
    args = parser.parse_args()
    
    if args.example:
        create_example_heatmap(args.output_dir)
        return
    
    # Load evaluation results
    evaluation_results = {}
    if args.evaluation_results:
        with open(args.evaluation_results, 'r') as f:
            evaluation_results = json.load(f)
    
    # Extract hyperparameters from config files
    if args.config_files:
        hyperparams_df = extract_hyperparameters_from_config(args.config_files)
        
        # Create hyperparameter heatmaps
        create_hyperparameter_heatmap(hyperparams_df, evaluation_results, args.output_dir)
    else:
        print("No config files provided. Creating example heatmap...")
        create_example_heatmap(args.output_dir)
    
    print(f"\n🎯 Hyperparameter analysis complete! Check {args.output_dir} for:")
    print(f"- lora_rank_vs_lr_bleu_1.png: LoRA Rank vs Learning Rate heatmap")
    print(f"- lora_rank_vs_alpha_bleu_1.png: LoRA Rank vs LoRA Alpha heatmap")
    print(f"- lr_vs_alpha_bleu_1.png: Learning Rate vs LoRA Alpha heatmap")
    print(f"- hyperparameter_3d_scatter.png: 3D scatter plots")
    print(f"- hyperparameter_summary.json: Detailed statistics")

if __name__ == "__main__":
    main() 