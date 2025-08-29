#!/usr/bin/env python3
"""
Example script to run per-video distribution analysis.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """
    Run per-video distribution analysis on your evaluation results.
    """
    
    # Example usage - replace with your actual file paths
    baseline_results = "v3/my_new_output_final_checkpoint_evaluation.json"  # Your baseline results
    finetuned_results = "v3/my_new_output_final_checkpoint_combined_optimized_evaluation.json"  # Your finetuned results
    
    # Check if files exist
    if not Path(baseline_results).exists():
        print(f"❌ Baseline results file not found: {baseline_results}")
        print("Please run evaluation on your baseline model first:")
        print(f"python v7/evaluate_summaries_simple.py v3/my_new_output_final_checkpoint.json --output {baseline_results}")
        return
    
    if not Path(finetuned_results).exists():
        print(f"❌ Finetuned results file not found: {finetuned_results}")
        print("Please run evaluation on your finetuned model first:")
        print(f"python v7/evaluate_summaries_simple.py v3/my_new_output_final_checkpoint_combined_optimized.json --output {finetuned_results}")
        return
    
    print("🚀 Running per-video distribution analysis...")
    print(f"Baseline: {baseline_results}")
    print(f"Finetuned: {finetuned_results}")
    
    # Run the analysis
    cmd = [
        "python", "v9/per_video_distribution_analysis.py",
        baseline_results,
        finetuned_results,
        "--output_dir", "v9/analysis_plots"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Analysis completed successfully!")
        print("\n📊 Generated plots:")
        print("- per_video_boxplots.png: Boxplots showing metric distributions")
        print("- per_video_violin_plots.png: Violin plots showing density distributions")
        print("- improvement_distribution.png: Histograms of improvements")
        print("- per_video_scatter_plots.png: Scatter plots of baseline vs finetuned")
        print("- summary_statistics.json: Detailed statistics")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        print(f"Error output: {e.stderr}")
        return
    
    print(f"\n📁 Check v9/analysis_plots/ for all generated files")

if __name__ == "__main__":
    main() 