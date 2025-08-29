#!/usr/bin/env python3
"""
Learning Rate Visualization for LoRA Training
Shows the cosine annealing schedule with warmup used in your training.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import os
from pathlib import Path
import json

def create_theoretical_lr_schedule():
    """Create the theoretical learning rate schedule based on your training config."""
    
    # Your training parameters
    initial_lr = 2e-5  # 0.00002
    num_epochs = 3
    warmup_ratio = 0.03  # 3%
    
    # Calculate total steps (approximate)
    # Assuming ~1000 steps per epoch based on your data
    steps_per_epoch = 1000  # This is approximate
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Generate step numbers
    steps = np.arange(total_steps)
    
    # Calculate learning rates
    lrs = []
    for step in steps:
        if step < warmup_steps:
            # Linear warmup
            lr = initial_lr * (step / warmup_steps)
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
        lrs.append(lr)
    
    return steps, lrs, warmup_steps, total_steps

def extract_tensorboard_lr_data(log_dir):
    """Extract actual learning rate data from TensorBoard logs if available."""
    try:
        import tensorboard as tb
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("TensorBoard not available for extracting actual LR data")
        return None, None
    
    # Look for TensorBoard logs
    tb_dirs = list(Path(log_dir).glob("**/runs/*"))
    if not tb_dirs:
        print(f"No TensorBoard logs found in {log_dir}")
        return None, None
    
    # Use the most recent run
    latest_run = max(tb_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Using TensorBoard logs from: {latest_run}")
    
    try:
        ea = EventAccumulator(str(latest_run))
        ea.Reload()
        
        # Look for learning rate scalars
        if 'train/learning_rate' in ea.Tags()['scalars']:
            lr_events = ea.Scalars('train/learning_rate')
            steps = [event.step for event in lr_events]
            lrs = [event.value for event in lr_events]
            return steps, lrs
        else:
            print("No learning rate data found in TensorBoard logs")
            return None, None
            
    except Exception as e:
        print(f"Error reading TensorBoard logs: {e}")
        return None, None

def create_lr_visualization():
    """Create comprehensive learning rate visualization."""
    
    # Get theoretical schedule
    steps, lrs, warmup_steps, total_steps = create_theoretical_lr_schedule()
    
    # Try to get actual data from TensorBoard
    actual_steps, actual_lrs = extract_tensorboard_lr_data("work_dirs")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Full schedule
    ax1.plot(steps, lrs, 'b-', linewidth=2, label='Theoretical Schedule')
    
    if actual_steps and actual_lrs:
        ax1.plot(actual_steps, actual_lrs, 'r--', linewidth=2, alpha=0.7, label='Actual (from logs)')
    
    # Add warmup region
    ax1.axvspan(0, warmup_steps, alpha=0.2, color='orange', label='Warmup Period')
    ax1.axvline(x=warmup_steps, color='orange', linestyle='--', alpha=0.7)
    
    # Add epoch markers
    steps_per_epoch = total_steps // 3
    for epoch in range(1, 4):
        step = epoch * steps_per_epoch
        ax1.axvline(x=step, color='gray', linestyle=':', alpha=0.5)
        ax1.text(step, ax1.get_ylim()[1] * 0.9, f'Epoch {epoch}', 
                rotation=90, ha='right', va='top', fontsize=8)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('LoRA Training Learning Rate Schedule\n(Cosine Annealing with 3% Warmup)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed in view of first 20% of training
    zoom_steps = int(total_steps * 0.2)
    zoom_mask = steps <= zoom_steps
    ax2.plot(steps[zoom_mask], np.array(lrs)[zoom_mask], 'b-', linewidth=2, label='Theoretical Schedule')
    
    if actual_steps and actual_lrs:
        actual_zoom_mask = [s <= zoom_steps for s in actual_steps]
        actual_zoom_steps = [s for s, m in zip(actual_steps, actual_zoom_mask) if m]
        actual_zoom_lrs = [lr for lr, m in zip(actual_lrs, actual_zoom_mask) if m]
        if actual_zoom_steps:
            ax2.plot(actual_zoom_steps, actual_zoom_lrs, 'r--', linewidth=2, alpha=0.7, label='Actual (from logs)')
    
    # Add warmup region
    ax2.axvspan(0, warmup_steps, alpha=0.2, color='orange', label='Warmup Period')
    ax2.axvline(x=warmup_steps, color='orange', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Zoomed View: First 20% of Training', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "v9/lr_schedule_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Learning rate visualization saved to: {output_path}")
    
    # Print key statistics
    print("\n" + "="*50)
    print("LEARNING RATE SCHEDULE STATISTICS")
    print("="*50)
    print(f"Initial Learning Rate: {2e-5:.2e}")
    print(f"Total Training Steps: {total_steps}")
    print(f"Warmup Steps: {warmup_steps} ({0.03*100:.1f}%)")
    print(f"Warmup Duration: {warmup_steps/(total_steps//3):.1f} epochs")
    print(f"Cosine Annealing Steps: {total_steps - warmup_steps}")
    print(f"Final Learning Rate: {lrs[-1]:.2e}")
    print(f"Min Learning Rate: {min(lrs):.2e}")
    print(f"Max Learning Rate: {max(lrs):.2e}")
    
    if actual_lrs:
        print(f"\nActual LR Statistics (from logs):")
        print(f"Actual Min LR: {min(actual_lrs):.2e}")
        print(f"Actual Max LR: {max(actual_lrs):.2e}")
    
    plt.show()

def create_lr_comparison_plot():
    """Create a comparison plot showing different LR schedules."""
    
    # Your current schedule
    steps, lrs_current, _, _ = create_theoretical_lr_schedule()
    
    # Alternative schedules for comparison
    lrs_constant = [2e-5] * len(steps)
    lrs_linear_decay = [2e-5 * (1 - i/len(steps)) for i in range(len(steps))]
    lrs_step_decay = []
    for i, step in enumerate(steps):
        if step < len(steps) // 3:
            lrs_step_decay.append(2e-5)
        elif step < 2 * len(steps) // 3:
            lrs_step_decay.append(2e-5 * 0.1)
        else:
            lrs_step_decay.append(2e-5 * 0.01)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(steps, lrs_current, 'b-', linewidth=3, label='Your Schedule (Cosine + Warmup)')
    ax.plot(steps, lrs_constant, 'r--', linewidth=2, alpha=0.7, label='Constant LR')
    ax.plot(steps, lrs_linear_decay, 'g--', linewidth=2, alpha=0.7, label='Linear Decay')
    ax.plot(steps, lrs_step_decay, 'm--', linewidth=2, alpha=0.7, label='Step Decay')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule Comparison\n(Your LoRA Training vs. Alternatives)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = "v9/lr_schedule_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"LR comparison plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating learning rate visualizations for your LoRA training...")
    
    # Create main visualization
    create_lr_visualization()
    
    # Create comparison plot
    create_lr_comparison_plot()
    
    print("\nDone! Check the generated PNG files for your learning rate schedules.") 