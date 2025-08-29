#!/usr/bin/env python3
"""
Script to find videos with all-zero metrics and display their summaries
"""

import json

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

def find_zero_videos(json_file_path):
    """Find videos with all-zero metrics and display their summaries"""
    
    # Load the JSON results
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    detailed = results['detailed_results']
    
    # Find videos with all-zero metrics
    zero_videos = [entry for entry in detailed if is_all_metrics_zero(entry)]
    
    print(f"Found {len(zero_videos)} videos with all-zero metrics:")
    print("=" * 80)
    
    for i, video in enumerate(zero_videos, 1):
        print(f"\n{i}. Video UID: {video['video_uid']}")
        print("-" * 40)
        print(f"Original Summary: {video['original_summary']}")
        print(f"Generated Summary: {video['generated_summary']}")
        print()
        
        # Also show the actual metric values
        print("Metrics:")
        print(f"  BLEU-1: {video.get('bleu_1', 0):.4f}")
        print(f"  BLEU-2: {video.get('bleu_2', 0):.4f}")
        print(f"  BLEU-3: {video.get('bleu_3', 0):.4f}")
        print(f"  BLEU-4: {video.get('bleu_4', 0):.4f}")
        print(f"  METEOR: {video.get('meteor', 0):.4f}")
        print(f"  ROUGE-L F1: {video.get('rougeL_f1', 0):.4f}")
        print(f"  Semantic Similarity: {video.get('semantic_similarity', 0):.4f}")
        print(f"  CLAIRE Score: {video.get('claire_score', 0):.4f}")
        print("=" * 80)
    
    return zero_videos

if __name__ == "__main__":
    zero_videos = find_zero_videos('v1_evaluation_results.json')
    
    # Save to file as well
    with open('zero_metric_videos.txt', 'w') as f:
        f.write(f"Found {len(zero_videos)} videos with all-zero metrics:\n")
        f.write("=" * 80 + "\n")
        
        for i, video in enumerate(zero_videos, 1):
            f.write(f"\n{i}. Video UID: {video['video_uid']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Original Summary: {video['original_summary']}\n")
            f.write(f"Generated Summary: {video['generated_summary']}\n\n")
            
            f.write("Metrics:\n")
            f.write(f"  BLEU-1: {video.get('bleu_1', 0):.4f}\n")
            f.write(f"  BLEU-2: {video.get('bleu_2', 0):.4f}\n")
            f.write(f"  BLEU-3: {video.get('bleu_3', 0):.4f}\n")
            f.write(f"  BLEU-4: {video.get('bleu_4', 0):.4f}\n")
            f.write(f"  METEOR: {video.get('meteor', 0):.4f}\n")
            f.write(f"  ROUGE-L F1: {video.get('rougeL_f1', 0):.4f}\n")
            f.write(f"  Semantic Similarity: {video.get('semantic_similarity', 0):.4f}\n")
            f.write(f"  CLAIRE Score: {video.get('claire_score', 0):.4f}\n")
            f.write("=" * 80 + "\n")
    
    print(f"\nResults also saved to: zero_metric_videos.txt") 