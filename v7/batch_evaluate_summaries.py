#!/usr/bin/env python3
"""
Batch Summary Evaluation Script
Evaluates multiple caption files and generates comparison reports.
"""
import os
import json
import glob
import pandas as pd
from datetime import datetime
import argparse
from evaluate_summaries import evaluate_summaries, calculate_average_scores

def find_caption_files(directory, pattern="*captions*.json"):
    """Find all caption files in the directory."""
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    return sorted(files)

def extract_model_info(filename):
    """Extract model and method information from filename."""
    basename = os.path.basename(filename)
    
    # Extract model name
    if "internvl" in basename.lower():
        model = "InternVL"
    elif "qwen" in basename.lower():
        model = "Qwen"
    elif "llava" in basename.lower():
        model = "LLaVA-Video"
    elif "mimo" in basename.lower():
        model = "MiMo-VL"
    elif "intern2.5" in basename.lower():
        model = "Intern2.5"
    else:
        model = "Unknown"
    
    # Extract method
    if "uniform" in basename.lower():
        method = "Uniform"
    elif "aks" in basename.lower():
        method = "AKS"
    else:
        method = "Unknown"
    
    return model, method

def batch_evaluate(caption_files, frames_dir="../v1/ego4d_aks_full/frames", output_dir="evaluation_results"):
    """Evaluate multiple caption files and generate comparison reports."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    comparison_data = []
    
    print(f"Found {len(caption_files)} caption files to evaluate")
    
    for caption_file in caption_files:
        print(f"\n{'='*80}")
        print(f"Evaluating: {caption_file}")
        print(f"{'='*80}")
        
        try:
            # Extract model and method info
            model, method = extract_model_info(caption_file)
            
            # Run evaluation
            results = evaluate_summaries(caption_file, frames_dir)
            
            if not results:
                print(f"No valid results for {caption_file}")
                continue
            
            # Calculate averages
            averages = calculate_average_scores(results)
            
            # Save individual results
            basename = os.path.splitext(os.path.basename(caption_file))[0]
            individual_output = os.path.join(output_dir, f"{basename}_evaluation.json")
            
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'caption_file': caption_file,
                'model': model,
                'method': method,
                'total_videos': len(results),
                'average_scores': averages,
                'detailed_results': results
            }
            
            with open(individual_output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Individual results saved to {individual_output}")
            
            # Add to comparison data
            comparison_entry = {
                'caption_file': caption_file,
                'model': model,
                'method': method,
                'total_videos': len(results),
                **averages
            }
            comparison_data.append(comparison_entry)
            
            all_results.append({
                'file': caption_file,
                'model': model,
                'method': method,
                'results': results,
                'averages': averages
            })
            
        except Exception as e:
            print(f"Error evaluating {caption_file}: {e}")
            continue
    
    # Generate comparison report
    if comparison_data:
        generate_comparison_report(comparison_data, output_dir)
    
    return all_results

def generate_comparison_report(comparison_data, output_dir):
    """Generate a comprehensive comparison report."""
    
    # Create DataFrame for easy analysis
    df = pd.DataFrame(comparison_data)
    
    # Save detailed comparison
    comparison_file = os.path.join(output_dir, "comparison_report.csv")
    df.to_csv(comparison_file, index=False)
    print(f"\nComparison report saved to {comparison_file}")
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(df)
    
    # Save summary
    summary_file = os.path.join(output_dir, "summary_statistics.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Summary statistics saved to {summary_file}")
    
    # Print comparison table
    print_comparison_table(df)
    
    # Generate model-wise comparison
    generate_model_comparison(df, output_dir)

def generate_summary_statistics(df):
    """Generate summary statistics for the comparison."""
    
    # Get all metric columns
    metric_columns = [col for col in df.columns if col not in ['caption_file', 'model', 'method', 'total_videos']]
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_files_evaluated': len(df),
        'models_evaluated': df['model'].unique().tolist(),
        'methods_evaluated': df['method'].unique().tolist(),
        'total_videos_processed': df['total_videos'].sum(),
        'metric_summaries': {}
    }
    
    # Calculate statistics for each metric
    for metric in metric_columns:
        values = df[metric].dropna()
        if len(values) > 0:
            summary['metric_summaries'][metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median())
            }
    
    return summary

def print_comparison_table(df):
    """Print a formatted comparison table."""
    
    print("\n" + "="*120)
    print("COMPARISON TABLE")
    print("="*120)
    
    # Select key metrics for display
    display_metrics = ['bleu_1', 'bleu_4', 'meteor', 'rougeL_f1', 'semantic_similarity', 'clip_score']
    available_metrics = [m for m in display_metrics if m in df.columns]
    
    # Create display DataFrame
    display_df = df[['model', 'method', 'total_videos'] + available_metrics].copy()
    
    # Round numeric columns
    for col in available_metrics:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)
    
    # Print table
    print(display_df.to_string(index=False))
    
    print("\n" + "="*120)

def generate_model_comparison(df, output_dir):
    """Generate model-wise comparison charts and analysis."""
    
    # Group by model and calculate averages
    model_comparison = df.groupby('model').agg({
        'bleu_1': 'mean',
        'bleu_4': 'mean', 
        'meteor': 'mean',
        'rougeL_f1': 'mean',
        'semantic_similarity': 'mean',
        'clip_score': 'mean',
        'total_videos': 'sum'
    }).round(4)
    
    # Save model comparison
    model_file = os.path.join(output_dir, "model_comparison.csv")
    model_comparison.to_csv(model_file)
    print(f"Model comparison saved to {model_file}")
    
    # Print model comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON (Averages)")
    print("="*80)
    print(model_comparison.to_string())
    
    # Method comparison
    method_comparison = df.groupby('method').agg({
        'bleu_1': 'mean',
        'bleu_4': 'mean',
        'meteor': 'mean', 
        'rougeL_f1': 'mean',
        'semantic_similarity': 'mean',
        'clip_score': 'mean',
        'total_videos': 'sum'
    }).round(4)
    
    # Save method comparison
    method_file = os.path.join(output_dir, "method_comparison.csv")
    method_comparison.to_csv(method_file)
    print(f"\nMethod comparison saved to {method_file}")
    
    print("\n" + "="*80)
    print("METHOD COMPARISON (Averages)")
    print("="*80)
    print(method_comparison.to_string())

def main():
    parser = argparse.ArgumentParser(description='Batch evaluate multiple caption files')
    parser.add_argument('--directory', default='.', help='Directory containing caption files')
    parser.add_argument('--pattern', default='*captions*.json', help='File pattern to match')
    parser.add_argument('--frames_dir', default='../v1/ego4d_aks_full/frames', 
                       help='Path to frames directory for CLIP scoring')
    parser.add_argument('--output_dir', default='evaluation_results', 
                       help='Output directory for evaluation results')
    parser.add_argument('--files', nargs='+', help='Specific caption files to evaluate')
    
    args = parser.parse_args()
    
    if args.files:
        caption_files = args.files
    else:
        caption_files = find_caption_files(args.directory, args.pattern)
    
    if not caption_files:
        print(f"No caption files found in {args.directory} matching pattern '{args.pattern}'")
        return
    
    print("Batch Summary Evaluation")
    print("="*50)
    print(f"Directory: {args.directory}")
    print(f"Pattern: {args.pattern}")
    print(f"Frames directory: {args.frames_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found {len(caption_files)} files to evaluate")
    
    # Run batch evaluation
    results = batch_evaluate(caption_files, args.frames_dir, args.output_dir)
    
    print(f"\nBatch evaluation completed!")
    print(f"Results saved in: {args.output_dir}")
    print(f"Successfully evaluated {len(results)} files")

if __name__ == "__main__":
    main() 