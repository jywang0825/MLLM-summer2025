#!/usr/bin/env python3
"""
Script to create a formatted table from evaluation results, excluding videos with all metrics zero
"""

import json
import pandas as pd
from tabulate import tabulate

def is_all_metrics_zero(entry):
    # Check if all main metrics are zero
    return (
        entry.get('bleu_1', 0) == 0 and
        entry.get('bleu_2', 0) == 0 and
        entry.get('bleu_3', 0) == 0 and
        entry.get('bleu_4', 0) == 0 and
        entry.get('meteor', 0) == 0 and
        entry.get('rougeL_f1', 0) == 0
    )

def create_results_table(json_file_path):
    """Create a formatted table from evaluation results, excluding all-zero-metric videos"""
    
    # Load the JSON results
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    detailed = results['detailed_results']
    # Exclude videos where all main metrics are zero
    filtered = [entry for entry in detailed if not is_all_metrics_zero(entry)]
    excluded_count = len(detailed) - len(filtered)
    
    # Recompute averages
    def get_metric_list(metric):
        return [entry[metric] for entry in filtered if metric in entry]
    
    def avg_std_min_max(metric):
        vals = get_metric_list(metric)
        if not vals:
            return (None, None, None, None)
        return (float(pd.Series(vals).mean()), float(pd.Series(vals).std()), float(pd.Series(vals).min()), float(pd.Series(vals).max()))
    
    table_data = []
    for i in range(1, 5):
        bleu_key = f'bleu_{i}'
        mean, std, minv, maxv = avg_std_min_max(bleu_key)
        if mean is not None:
            table_data.append([
                f'BLEU-{i}',
                f"{mean:.4f}",
                f"{std:.4f}",
                f"{minv:.4f}",
                f"{maxv:.4f}"
            ])
    mean, std, minv, maxv = avg_std_min_max('meteor')
    if mean is not None:
        table_data.append(['METEOR', f"{mean:.4f}", f"{std:.4f}", f"{minv:.4f}", f"{maxv:.4f}"])
    mean, std, minv, maxv = avg_std_min_max('rouge1_f1')
    if mean is not None:
        table_data.append(['ROUGE-1 F1', f"{mean:.4f}", f"{std:.4f}", f"{minv:.4f}", f"{maxv:.4f}"])
    mean, std, minv, maxv = avg_std_min_max('rouge2_f1')
    if mean is not None:
        table_data.append(['ROUGE-2 F1', f"{mean:.4f}", f"{std:.4f}", f"{minv:.4f}", f"{maxv:.4f}"])
    mean, std, minv, maxv = avg_std_min_max('rougeL_f1')
    if mean is not None:
        table_data.append(['ROUGE-L F1', f"{mean:.4f}", f"{std:.4f}", f"{minv:.4f}", f"{maxv:.4f}"])
    mean, std, minv, maxv = avg_std_min_max('claire_score')
    if mean is not None:
        table_data.append(['CLAIRE Score', f"{mean:.4f}", f"{std:.4f}", f"{minv:.4f}", f"{maxv:.4f}"])
    
    headers = ['Metric', 'Mean', 'Std Dev', 'Min', 'Max']
    table = tabulate(table_data, headers=headers, tablefmt='grid', floatfmt='.4f')
    print(f"Evaluation Results Summary (excluding {excluded_count} all-zero-metric videos)")
    print(f"Total Videos (after exclusion): {len(filtered)}")
    print(f"Timestamp: {results['timestamp']}")
    print()
    print(table)
    print("\n" + "="*80)
    print("MARKDOWN TABLE:")
    print("="*80)
    print()
    markdown_table = tabulate(table_data, headers=headers, tablefmt='pipe', floatfmt='.4f')
    print(markdown_table)
    with open('evaluation_results_table_excluded.txt', 'w') as f:
        f.write(f"Evaluation Results Summary (excluding {excluded_count} all-zero-metric videos)\n")
        f.write(f"Total Videos (after exclusion): {len(filtered)}\n")
        f.write(f"Timestamp: {results['timestamp']}\n\n")
        f.write(table)
        f.write("\n\nMARKDOWN TABLE:\n")
        f.write("="*80 + "\n")
        f.write(markdown_table)
    print(f"\nTable saved to: evaluation_results_table_excluded.txt")
    print(f"Number of excluded videos: {excluded_count}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        create_results_table(sys.argv[1])
    else:
        create_results_table('v1_evaluation_results.json') 