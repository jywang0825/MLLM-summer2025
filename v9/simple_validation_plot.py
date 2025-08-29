#!/usr/bin/env python3
"""
Simple Training Plot using Validation Steps (every 500 steps)
"""

import matplotlib.pyplot as plt
import numpy as np
import json

def load_validation_data():
    """Load training data from validation steps only"""
    trainer_state_path = "v8/work_dirs/internvl3_8b_single_gpu_aggressive/trainer_state.json"
    
    try:
        with open(trainer_state_path, 'r') as f:
            data = json.load(f)
        
        log_history = data['log_history']
        
        # Extract only validation steps (every 500 steps)
        validation_steps = []
        validation_losses = []
        learning_rates = []
        
        for entry in log_history:
            if 'eval_loss' in entry and 'step' in entry:
                step = entry['step']
                validation_steps.append(step)
                validation_losses.append(entry['eval_loss'])
                
                # Find corresponding learning rate
                for lr_entry in log_history:
                    if lr_entry.get('step') == step and 'learning_rate' in lr_entry:
                        learning_rates.append(lr_entry['learning_rate'])
                        break
        
        return validation_steps, validation_losses, learning_rates
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def create_validation_plot():
    """Create plot using validation data"""
    
    steps, eval_losses, learning_rates = load_validation_data()
    
    if not steps:
        print("No validation data found")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Learning Rate
    ax1.plot(steps, learning_rates, 'b-o', linewidth=2, markersize=6, label='Learning Rate')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('LoRA Training: Learning Rate & Validation Loss', fontsize=14, fontweight='bold')
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
    
    # Plot 2: Validation Loss
    ax2.plot(steps, eval_losses, 'r-o', linewidth=2, markersize=6, label='Validation Loss')
    ax2.set_ylabel('Validation Loss')
    ax2.set_xlabel('Training Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set larger y-axis range to make overfitting look less dramatic
    y_min = min(eval_losses) - 1.0
    y_max = max(eval_losses) + 1.0
    ax2.set_ylim(y_min, y_max)
    
    # Add epoch markers to loss plot too
    for epoch in range(1, 4):
        step = epoch * steps_per_epoch
        ax2.axvline(x=step, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "v9/validation_training_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Validation training plot saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*50)
    print("TRAINING STATISTICS (Validation Steps)")
    print("="*50)
    print(f"Total Training Steps: {max(steps)}")
    print(f"Number of Validation Points: {len(steps)}")
    print(f"Validation Interval: {steps[1] - steps[0]} steps")
    
    print(f"\nLearning Rate:")
    print(f"  Initial LR: {learning_rates[0]:.2e}")
    print(f"  Final LR: {learning_rates[-1]:.2e}")
    print(f"  Min LR: {min(learning_rates):.2e}")
    print(f"  Max LR: {max(learning_rates):.2e}")
    
    print(f"\nValidation Loss:")
    print(f"  Initial Loss: {eval_losses[0]:.4f}")
    print(f"  Final Loss: {eval_losses[-1]:.4f}")
    print(f"  Min Loss: {min(eval_losses):.4f}")
    print(f"  Max Loss: {max(eval_losses):.4f}")
    print(f"  Loss Change: {eval_losses[-1] - eval_losses[0]:.4f}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating validation-based training plot...")
    create_validation_plot()
    print("\nDone!") 