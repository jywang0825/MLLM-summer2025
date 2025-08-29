#!/usr/bin/env python3
"""
Simple Learning Rate and Loss Plot for LoRA Training
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
import json

def create_simple_lr_plot():
    """Create a simple learning rate decline plot."""
    
    # Your training parameters
    initial_lr = 2e-5
    num_epochs = 3
    warmup_ratio = 0.03
    
    # Calculate steps (approximate)
    steps_per_epoch = 1000
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Generate learning rate schedule
    steps = np.arange(total_steps)
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
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot learning rate
    ax.plot(steps, lrs, 'b-', linewidth=2, label='Learning Rate')
    
    # Add warmup region
    ax.axvspan(0, warmup_steps, alpha=0.2, color='orange', label='Warmup')
    ax.axvline(x=warmup_steps, color='orange', linestyle='--', alpha=0.7)
    
    # Add epoch markers
    for epoch in range(1, 4):
        step = epoch * steps_per_epoch
        ax.axvline(x=step, color='gray', linestyle=':', alpha=0.5)
        ax.text(step, ax.get_ylim()[1] * 0.9, f'Epoch {epoch}', 
                rotation=90, ha='right', va='top', fontsize=10)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('LoRA Training: Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "v9/simple_lr_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Learning rate plot saved to: {output_path}")
    
    # Print key info
    print(f"\nLearning Rate Schedule:")
    print(f"Initial LR: {initial_lr:.2e}")
    print(f"Warmup: {warmup_steps} steps ({warmup_ratio*100:.1f}%)")
    print(f"Final LR: {lrs[-1]:.2e}")
    print(f"Min LR: {min(lrs):.2e}")
    
    plt.show()

def extract_loss_from_logs():
    """Try to extract loss data from training logs."""
    log_files = list(Path(".").glob("**/training_log.txt"))
    
    if not log_files:
        print("No training logs found")
        return None, None
    
    # Use the most recent log
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    print(f"Using log file: {latest_log}")
    
    steps = []
    losses = []
    
    try:
        with open(latest_log, 'r') as f:
            for line in f:
                # Look for lines with loss information
                if any(keyword in line.lower() for keyword in ['loss:', 'train_loss', 'loss =']):
                    parts = line.split()
                    
                    # Try different patterns for step and loss
                    step = None
                    loss = None
                    
                    for i, part in enumerate(parts):
                        # Find step number
                        if part.isdigit() and i > 0 and 'step' in parts[i-1].lower():
                            try:
                                step = int(part)
                            except ValueError:
                                continue
                        
                        # Find loss value
                        if 'loss' in part.lower() and i + 1 < len(parts):
                            try:
                                loss = float(parts[i + 1])
                            except ValueError:
                                continue
                        
                        # Alternative: look for loss after colon
                        if 'loss:' in part and i + 1 < len(parts):
                            try:
                                loss = float(parts[i + 1])
                            except ValueError:
                                continue
                    
                    if step is not None and loss is not None:
                        steps.append(step)
                        losses.append(loss)
                        
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None
    
    if steps and losses:
        print(f"Found {len(steps)} loss data points")
        return steps, losses
    else:
        print("No loss data found in logs")
        return None, None

def create_combined_plot():
    """Create a combined plot showing both learning rate and loss."""
    
    # Get learning rate data
    initial_lr = 2e-5
    num_epochs = 3
    warmup_ratio = 0.03
    steps_per_epoch = 1000
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Generate learning rate schedule
    lr_steps = np.arange(total_steps)
    lrs = []
    
    for step in lr_steps:
        if step < warmup_steps:
            lr = initial_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
        lrs.append(lr)
    
    # Get loss data
    loss_steps, losses = extract_loss_from_logs()
    
    # Create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Learning Rate
    ax1.plot(lr_steps, lrs, 'b-', linewidth=2, label='Learning Rate')
    ax1.axvspan(0, warmup_steps, alpha=0.2, color='orange', label='Warmup')
    ax1.axvline(x=warmup_steps, color='orange', linestyle='--', alpha=0.7)
    
    # Add epoch markers
    for epoch in range(1, 4):
        step = epoch * steps_per_epoch
        ax1.axvline(x=step, color='gray', linestyle=':', alpha=0.5)
        ax1.text(step, ax1.get_ylim()[1] * 0.9, f'Epoch {epoch}', 
                rotation=90, ha='right', va='top', fontsize=8)
    
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('LoRA Training: Learning Rate & Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss (if available)
    if loss_steps and losses:
        ax2.plot(loss_steps, losses, 'r-', linewidth=2, label='Training Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add epoch markers to loss plot too
        for epoch in range(1, 4):
            step = epoch * steps_per_epoch
            ax2.axvline(x=step, color='gray', linestyle=':', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, 'No loss data available\nfrom training logs', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_ylabel('Loss')
    
    ax2.set_xlabel('Training Steps')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "v9/combined_lr_loss_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {output_path}")
    
    # Print statistics
    print(f"\nLearning Rate Schedule:")
    print(f"Initial LR: {initial_lr:.2e}")
    print(f"Warmup: {warmup_steps} steps ({warmup_ratio*100:.1f}%)")
    print(f"Final LR: {lrs[-1]:.2e}")
    
    if losses:
        print(f"\nLoss Statistics:")
        print(f"Initial Loss: {losses[0]:.4f}")
        print(f"Final Loss: {losses[-1]:.4f}")
        print(f"Min Loss: {min(losses):.4f}")
        print(f"Max Loss: {max(losses):.4f}")
    
    plt.show()

def create_loss_plot():
    """Create a simple loss plot if data is available."""
    steps, losses = extract_loss_from_logs()
    
    if not steps or not losses:
        print("No loss data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps, losses, 'r-', linewidth=2, label='Training Loss')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('LoRA Training: Loss Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = "v9/simple_loss_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to: {output_path}")
    
    print(f"\nLoss Statistics:")
    print(f"Initial Loss: {losses[0]:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Min Loss: {min(losses):.4f}")
    print(f"Max Loss: {max(losses):.4f}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating combined learning rate and loss plot...")
    
    # Create combined plot (shows both LR and loss)
    create_combined_plot()
    
    print("\nDone!") 