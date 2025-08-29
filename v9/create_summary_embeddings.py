#!/usr/bin/env python3
"""
Create embeddings from video summaries using a pre-trained model.
"""

import json
import torch
import numpy as np
from pathlib import Path
import argparse
from sentence_transformers import SentenceTransformer
import pickle

def load_evaluation_data(baseline_file: str, finetuned_file: str) -> tuple:
    """Load baseline and finetuned evaluation data."""
    print(f"Loading baseline data from {baseline_file}...")
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    print(f"Loading finetuned data from {finetuned_file}...")
    with open(finetuned_file, 'r') as f:
        finetuned_data = json.load(f)
    
    return baseline_data, finetuned_data

def extract_summary_texts(data: dict) -> dict:
    """Extract ground truth and generated summaries from evaluation data."""
    if 'detailed_results' in data:
        results = data['detailed_results']
    else:
        results = data
    
    ground_truth = []
    generated = []
    
    for result in results:
        if 'ground_truth' in result:
            ground_truth.append(result['ground_truth'])
        if 'generated' in result:
            generated.append(result['generated'])
    
    return {
        'ground_truth': ground_truth,
        'generated': generated
    }

def create_embeddings_from_summaries(summaries: dict, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Create embeddings from summary texts using a pre-trained model.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Combine ground truth and generated summaries
    all_texts = summaries['ground_truth'] + summaries['generated']
    
    print(f"Creating embeddings for {len(all_texts)} summaries...")
    embeddings = model.encode(all_texts, convert_to_numpy=True)
    
    print(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings

def save_embeddings(embeddings: np.ndarray, output_file: str):
    """Save embeddings to file."""
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create embeddings from video summaries")
    parser.add_argument("baseline_results", help="Path to baseline evaluation results JSON")
    parser.add_argument("finetuned_results", help="Path to finetuned evaluation results JSON")
    parser.add_argument("--output_dir", default="v9/embeddings", help="Output directory for embeddings")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model to use")
    
    args = parser.parse_args()
    
    # Load evaluation data
    baseline_data, finetuned_data = load_evaluation_data(args.baseline_results, args.finetuned_results)
    
    # Extract summaries
    baseline_summaries = extract_summary_texts(baseline_data)
    finetuned_summaries = extract_summary_texts(finetuned_data)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create embeddings
    print("Creating baseline embeddings...")
    baseline_embeddings = create_embeddings_from_summaries(baseline_summaries, args.model)
    
    print("Creating finetuned embeddings...")
    finetuned_embeddings = create_embeddings_from_summaries(finetuned_summaries, args.model)
    
    # Save embeddings
    save_embeddings(baseline_embeddings, f"{args.output_dir}/baseline_embeddings.pkl")
    save_embeddings(finetuned_embeddings, f"{args.output_dir}/finetuned_embeddings.pkl")
    
    # Save summary metadata
    metadata = {
        'baseline_summaries': baseline_summaries,
        'finetuned_summaries': finetuned_summaries,
        'embedding_model': args.model,
        'baseline_embedding_shape': baseline_embeddings.shape,
        'finetuned_embedding_shape': finetuned_embeddings.shape
    }
    
    with open(f"{args.output_dir}/embedding_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Embeddings created successfully!")
    print(f"✓ Baseline embeddings: {baseline_embeddings.shape}")
    print(f"✓ Finetuned embeddings: {finetuned_embeddings.shape}")
    print(f"✓ Files saved to {args.output_dir}")

if __name__ == "__main__":
    main() 