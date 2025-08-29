#!/usr/bin/env python3
"""
Extract LoRA Hyperparameters from Training Setup
Extract LoRA rank, learning rate, and other hyperparameters from config files and trainer state.
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

def extract_lora_config(config_file: str) -> dict:
    """
    Extract LoRA configuration from config.json file.
    """
    print(f"Extracting LoRA config from {config_file}...")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract LoRA settings
    lora_config = {
        'lora_rank': config.get('use_llm_lora', None),  # This is the LoRA rank
        'backbone_lora': config.get('use_backbone_lora', None),
        'config_file': config_file
    }
    
    print(f"Found LoRA rank: {lora_config['lora_rank']}")
    return lora_config

def extract_learning_rate_from_trainer_state(trainer_state_file: str) -> float:
    """
    Extract initial learning rate from trainer state.
    """
    print(f"Extracting learning rate from {trainer_state_file}...")
    
    with open(trainer_state_file, 'r') as f:
        trainer_state = json.load(f)
    
    # Get the initial learning rate (first entry in log_history)
    log_history = trainer_state.get('log_history', [])
    if log_history:
        initial_lr = log_history[0].get('learning_rate', None)
        print(f"Found initial learning rate: {initial_lr}")
        return initial_lr
    else:
        print("No log history found in trainer state")
        return None

def create_single_experiment_analysis(config_file: str, 
                                    trainer_state_file: str,
                                    evaluation_results: dict,
                                    output_dir: str = "v9/analysis_plots"):
    """
    Create analysis for a single experiment with extracted hyperparameters.
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract hyperparameters
    lora_config = extract_lora_config(config_file)
    learning_rate = extract_learning_rate_from_trainer_state(trainer_state_file)
    
    # Create summary
    experiment_summary = {
        'lora_rank': lora_config['lora_rank'],
        'learning_rate': learning_rate,
        'backbone_lora': lora_config['backbone_lora'],
        'config_file': config_file,
        'trainer_state_file': trainer_state_file
    }
    
    # Add evaluation results if available
    if evaluation_results:
        # The evaluation_results is already the data we need
        eval_data = evaluation_results
        
        experiment_summary.update({
            'bleu_1': eval_data.get('average_scores', {}).get('bleu_1', {}).get('mean', 0),
            'bleu_2': eval_data.get('average_scores', {}).get('bleu_2', {}).get('mean', 0),
            'bleu_3': eval_data.get('average_scores', {}).get('bleu_3', {}).get('mean', 0),
            'bleu_4': eval_data.get('average_scores', {}).get('bleu_4', {}).get('mean', 0),
            'meteor': eval_data.get('average_scores', {}).get('meteor', {}).get('mean', 0),
            'rouge_l': eval_data.get('average_scores', {}).get('rougeL_f1', {}).get('mean', 0),
            'rouge_1': eval_data.get('average_scores', {}).get('rouge1_f1', {}).get('mean', 0),
            'rouge_2': eval_data.get('average_scores', {}).get('rouge2_f1', {}).get('mean', 0)
        })
    
    # Save experiment summary
    summary_file = f"{output_dir}/experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    print(f"✓ Experiment summary saved to {summary_file}")
    
    # Create visualization
    create_hyperparameter_visualization(experiment_summary, output_dir)
    
    return experiment_summary

def create_hyperparameter_visualization(experiment_summary: dict, output_dir: str):
    """
    Create visualization for the single experiment.
    """
    
    # Create a summary plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Hyperparameter values
    hyperparams = ['LoRA Rank', 'Learning Rate', 'Backbone LoRA']
    values = [experiment_summary.get('lora_rank', 0), 
              experiment_summary.get('learning_rate', 0),
              experiment_summary.get('backbone_lora', 0)]
    
    bars = axes[0].bar(hyperparams, values, color=['blue', 'green', 'orange'])
    axes[0].set_title('Hyperparameter Configuration', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}' if value > 0.01 else f'{value}',
                    ha='center', va='bottom')
    
    # Plot 2: Evaluation metrics
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'ROUGE-1', 'ROUGE-2']
    metric_values = [experiment_summary.get('bleu_1', 0),
                    experiment_summary.get('bleu_2', 0),
                    experiment_summary.get('bleu_3', 0),
                    experiment_summary.get('bleu_4', 0),
                    experiment_summary.get('meteor', 0),
                    experiment_summary.get('rouge_l', 0),
                    experiment_summary.get('rouge_1', 0),
                    experiment_summary.get('rouge_2', 0)]
    
    bars2 = axes[1].bar(metrics, metric_values, color='skyblue')
    axes[1].set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, metric_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/experiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Experiment visualization saved to {output_dir}/experiment_analysis.png")

def create_grid_search_template(output_dir: str = "v9/analysis_plots"):
    """
    Create a template for grid search analysis with your current setup.
    """
    
    # Your current configuration
    current_config = {
        'lora_rank': 16,
        'learning_rate': 2e-5,  # Approximate from trainer state
        'backbone_lora': 0
    }
    
    # Suggested grid search parameters
    grid_search_template = {
        'current_experiment': current_config,
        'suggested_grid_search': {
            'lora_ranks': [8, 16, 32, 64, 128],
            'learning_rates': [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
            'total_experiments': 25
        },
        'grid_search_script': """
# Example grid search script
for lora_rank in [8, 16, 32, 64, 128]:
    for learning_rate in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]:
        # Update config.json with new lora_rank
        # Update training script with new learning_rate
        # Run training
        # Run evaluation
        # Collect results
        """,
        'expected_analysis': [
            'LoRA Rank vs Learning Rate heatmap',
            'Performance comparison across ranks',
            'Learning rate sensitivity analysis',
            'Optimal hyperparameter combination'
        ]
    }
    
    # Save template
    template_file = f"{output_dir}/grid_search_template.json"
    with open(template_file, 'w') as f:
        json.dump(grid_search_template, f, indent=2)
    
    print(f"✓ Grid search template saved to {template_file}")
    
    # Create example heatmap
    create_example_grid_search_heatmap(output_dir)

def create_example_grid_search_heatmap(output_dir: str):
    """
    Create an example heatmap showing what a grid search would look like.
    """
    
    # Create synthetic grid search data
    lora_ranks = [8, 16, 32, 64, 128]
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
    
    # Create synthetic BLEU-1 scores (optimal around rank=32, lr=5e-5)
    data = []
    for rank in lora_ranks:
        for lr in learning_rates:
            # Synthetic score: optimal around rank=32, lr=5e-5
            score = 0.3 + 0.4 * np.exp(-((rank-32)**2 + (lr-5e-5)**2*1e10) / 1000)
            data.append({'lora_rank': rank, 'learning_rate': lr, 'bleu_1': score})
    
    df = pd.DataFrame(data)
    
    # Create heatmap
    pivot_df = df.pivot_table(values='bleu_1', index='lora_rank', columns='learning_rate')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'BLEU-1 Score'})
    plt.title('Example Grid Search: LoRA Rank vs Learning Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('LoRA Rank', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example_grid_search_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Example grid search heatmap saved to {output_dir}/example_grid_search_heatmap.png")

def main():
    parser = argparse.ArgumentParser(description="Extract LoRA hyperparameters from training setup")
    parser.add_argument("--config_file", default="v8/work_dirs/internvl3_8b_single_gpu_aggressive/config.json", 
                       help="Path to config.json file")
    parser.add_argument("--trainer_state", default="v8/work_dirs/internvl3_8b_single_gpu_aggressive/trainer_state.json",
                       help="Path to trainer_state.json file")
    parser.add_argument("--evaluation_results", help="Path to evaluation results JSON")
    parser.add_argument("--output_dir", default="v9/analysis_plots", help="Output directory for plots")
    parser.add_argument("--create_template", action="store_true", help="Create grid search template")
    
    args = parser.parse_args()
    
    # Load evaluation results if provided
    evaluation_results = {}
    if args.evaluation_results:
        with open(args.evaluation_results, 'r') as f:
            evaluation_results = json.load(f)
    
    # Extract hyperparameters from your current setup
    experiment_summary = create_single_experiment_analysis(
        args.config_file, 
        args.trainer_state, 
        evaluation_results, 
        args.output_dir
    )
    
    # Create grid search template if requested
    if args.create_template:
        create_grid_search_template(args.output_dir)
    
    print(f"\n🎯 LoRA hyperparameter analysis complete!")
    print(f"Current configuration:")
    print(f"- LoRA Rank: {experiment_summary.get('lora_rank', 'N/A')}")
    print(f"- Learning Rate: {experiment_summary.get('learning_rate', 'N/A'):.2e}")
    print(f"- Backbone LoRA: {experiment_summary.get('backbone_lora', 'N/A')}")
    print(f"\nCheck {args.output_dir} for detailed analysis and visualizations.")

if __name__ == "__main__":
    main() 