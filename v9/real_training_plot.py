#!/usr/bin/env python3
"""
Real Training Data Visualization for LoRA Training
Uses actual training data from trainer_state.json
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def load_training_data():
    """Load actual training data from trainer_state.json"""
    trainer_state_path = "v8/work_dirs/internvl3_8b_single_gpu_aggressive/trainer_state.json"
    
    try:
        with open(trainer_state_path, 'r') as f:
            data = json.load(f)
        
        # Extract training history
        log_history = data['log_history']
        
        # Create dictionaries to store data by step
        lr_data = {}
        loss_data = {}
        eval_data = {}
        
        for entry in log_history:
            if 'step' in entry:
                step = entry['step']
                
                if 'learning_rate' in entry:
                    lr_data[step] = entry['learning_rate']
                
                if 'loss' in entry:
                    loss_data[step] = entry['loss']
                
                if 'eval_loss' in entry:
                    eval_data[step] = entry['eval_loss']
        
        # Get all unique steps and sort them
        all_steps = sorted(set(lr_data.keys()) | set(loss_data.keys()) | set(eval_data.keys()))
        
        # Create aligned arrays
        steps = []
        learning_rates = []
        losses = []
        eval_steps = []
        eval_losses = []
        
        for step in all_steps:
            steps.append(step)
            
            if step in lr_data:
                learning_rates.append(lr_data[step])
            else:
                learning_rates.append(None)
            
            if step in loss_data:
                losses.append(loss_data[step])
            else:
                losses.append(None)
            
            if step in eval_data:
                eval_steps.append(step)
                eval_losses.append(eval_data[step])
        
        # Filter out None values for plotting
        lr_steps = [s for s, lr in zip(steps, learning_rates) if lr is not None]
        lr_values = [lr for lr in learning_rates if lr is not None]
        
        loss_steps = [s for s, loss in zip(steps, losses) if loss is not None]
        loss_values = [loss for loss in losses if loss is not None]
        
        return lr_steps, lr_values, loss_steps, loss_values, eval_steps, eval_losses
        
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None, None, None, None, None, None

def create_complete_training_plot():
    """Create a comprehensive training plot with real data."""
    
    # Load real training data
    steps, learning_rates, losses, eval_steps, eval_losses = load_training_data()
    
    if not steps:
        print("No training data found")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Learning Rate
    ax1.plot(steps, learning_rates, 'b-', linewidth=2, label='Learning Rate')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('LoRA Training: Complete Learning Rate & Loss Progression', fontsize=16, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add epoch markers
    total_steps = max(steps)
    steps_per_epoch = total_steps // 3
    for epoch in range(1, 4):
        step = epoch * steps_per_epoch
        ax1.axvline(x=step, color='gray', linestyle=':', alpha=0.5)
        ax1.text(step, ax1.get_ylim()[1] * 0.9, f'Epoch {epoch}', 
                rotation=90, ha='right', va='top', fontsize=10)
    
    # Plot 2: Loss
    if losses:
        ax2.plot(steps, losses, 'r-', linewidth=2, label='Training Loss', alpha=0.8)
    
    if eval_steps and eval_losses:
        ax2.plot(eval_steps, eval_losses, 'g-', linewidth=2, label='Validation Loss', alpha=0.8)
    
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Training Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add epoch markers to loss plot too
    for epoch in range(1, 4):
        step = epoch * steps_per_epoch
        ax2.axvline(x=step, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "v9/complete_training_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Complete training plot saved to: {output_path}")
    
    # Print training statistics
    print("\n" + "="*60)
    print("COMPLETE TRAINING STATISTICS")
    print("="*60)
    print(f"Total Training Steps: {max(steps)}")
    print(f"Final Epoch: {max(steps) / steps_per_epoch:.2f}")
    
    if learning_rates:
        print(f"\nLearning Rate:")
        print(f"  Initial LR: {learning_rates[0]:.2e}")
        print(f"  Final LR: {learning_rates[-1]:.2e}")
        print(f"  Min LR: {min(learning_rates):.2e}")
        print(f"  Max LR: {max(learning_rates):.2e}")
    
    if losses:
        print(f"\nTraining Loss:")
        print(f"  Initial Loss: {losses[0]:.4f}")
        print(f"  Final Loss: {losses[-1]:.4f}")
        print(f"  Min Loss: {min(losses):.4f}")
        print(f"  Max Loss: {max(losses):.4f}")
        print(f"  Loss Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    
    if eval_losses:
        print(f"\nValidation Loss:")
        print(f"  Initial Eval Loss: {eval_losses[0]:.4f}")
        print(f"  Final Eval Loss: {eval_losses[-1]:.4f}")
        print(f"  Min Eval Loss: {min(eval_losses):.4f}")
        print(f"  Max Eval Loss: {max(eval_losses):.4f}")
    
    plt.show()

def create_zoom_plots():
    """Create zoomed plots for different phases of training."""
    
    steps, learning_rates, losses, eval_steps, eval_losses = load_training_data()
    
    if not steps:
        return
    
    # Create subplots for different phases
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Phase 1: Early training (first 20%)
    early_end = int(len(steps) * 0.2)
    early_steps = steps[:early_end]
    early_lrs = learning_rates[:early_end]
    early_losses = losses[:early_end]
    
    axes[0, 0].plot(early_steps, early_lrs, 'b-', linewidth=2)
    axes[0, 0].set_title('Early Training (First 20%)', fontweight='bold')
    axes[0, 0].set_ylabel('Learning Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(early_steps, early_losses, 'r-', linewidth=2)
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Phase 2: Middle training (20-80%)
    mid_start = int(len(steps) * 0.2)
    mid_end = int(len(steps) * 0.8)
    mid_steps = steps[mid_start:mid_end]
    mid_lrs = learning_rates[mid_start:mid_end]
    mid_losses = losses[mid_start:mid_end]
    
    axes[0, 1].plot(mid_steps, mid_lrs, 'b-', linewidth=2)
    axes[0, 1].set_title('Middle Training (20-80%)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(mid_steps, mid_losses, 'r-', linewidth=2)
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Phase 3: Late training (last 20%)
    late_start = int(len(steps) * 0.8)
    late_steps = steps[late_start:]
    late_lrs = learning_rates[late_start:]
    late_losses = losses[late_start:]
    
    axes[0, 2].plot(late_steps, late_lrs, 'b-', linewidth=2)
    axes[0, 2].set_title('Late Training (Last 20%)', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 2].plot(late_steps, late_losses, 'r-', linewidth=2)
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add x-labels
    for ax in axes[1, :]:
        ax.set_xlabel('Training Steps')
    
    plt.tight_layout()
    
    output_path = "v9/training_phase_plots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training phase plots saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating complete training visualization with real data...")
    
    # Create main training plot
    create_complete_training_plot()
    
    # Create zoomed phase plots
    create_zoom_plots()
    
    print("\nDone! Check the generated PNG files for your complete training analysis.") 