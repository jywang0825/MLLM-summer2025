#!/usr/bin/env python3
"""
Create embeddings from video summaries using a simple approach.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import pickle
import hashlib
from collections import Counter
import re

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
        if 'test_summary' in result:
            ground_truth.append(result['test_summary'])
        if 'generated_summary' in result:
            generated.append(result['generated_summary'])
    
    return {
        'ground_truth': ground_truth,
        'generated': generated
    }

def create_simple_embeddings(summaries: dict, embedding_dim: int = 768) -> np.ndarray:
    """
    Create embeddings from summary texts using a simple hash-based approach.
    """
    print(f"Creating simple embeddings with dimension {embedding_dim}...")
    
    # Combine all texts to build vocabulary
    all_texts = summaries['ground_truth'] + summaries['generated']
    
    # Create a simple vocabulary from all texts
    vocab = set()
    for text in all_texts:
        # Simple tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        vocab.update(words)
    
    vocab = list(vocab)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create embeddings based on word frequencies and semantic features
    embeddings = []
    
    for text in all_texts:
        # Create embedding based on word frequencies
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        
        # Initialize embedding
        embedding = np.zeros(embedding_dim)
        
        # Add semantic features based on content
        text_lower = text.lower()
        
        # Kitchen/cooking related
        if any(word in text_lower for word in ['kitchen', 'cooking', 'cook', 'food', 'meal']):
            embedding[:100] += 0.3
        if any(word in text_lower for word in ['stove', 'oven', 'pan', 'pot']):
            embedding[100:200] += 0.3
            
        # Office/work related
        if any(word in text_lower for word in ['office', 'work', 'desk', 'computer', 'meeting']):
            embedding[200:300] += 0.3
        if any(word in text_lower for word in ['typing', 'writing', 'document']):
            embedding[300:400] += 0.3
            
        # Outdoor/activity related
        if any(word in text_lower for word in ['outdoor', 'outside', 'walking', 'running', 'exercise']):
            embedding[400:500] += 0.3
        if any(word in text_lower for word in ['park', 'street', 'road', 'path']):
            embedding[500:600] += 0.3
            
        # Social interaction related
        if any(word in text_lower for word in ['conversation', 'talking', 'speaking', 'discussion']):
            embedding[600:700] += 0.3
        if any(word in text_lower for word in ['people', 'person', 'group', 'friend']):
            embedding[700:768] += 0.3
        
        # Add word frequency features
        for i, word in enumerate(vocab):
            if word in word_counts:
                # Use hash to distribute words across embedding dimensions
                hash_val = hash(word) % embedding_dim
                embedding[hash_val] += word_counts[word] * 0.1
        
        # Add some randomness for diversity
        embedding += np.random.randn(embedding_dim) * 0.05
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
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
    parser.add_argument("--embedding_dim", type=int, default=768, help="Embedding dimension")
    
    args = parser.parse_args()
    
    # Load evaluation data
    # Note: File names are misleading:
    # - v3 file contains FINETUNED results
    # - v7 file contains BASELINE results
    finetuned_data, baseline_data = load_evaluation_data(args.baseline_results, args.finetuned_results)
    
    # Extract summaries
    baseline_summaries = extract_summary_texts(baseline_data)  # v7 file = baseline
    finetuned_summaries = extract_summary_texts(finetuned_data)  # v3 file = finetuned
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create embeddings
    print("Creating baseline embeddings...")
    baseline_embeddings = create_simple_embeddings(baseline_summaries, args.embedding_dim)
    
    print("Creating finetuned embeddings...")
    finetuned_embeddings = create_simple_embeddings(finetuned_summaries, args.embedding_dim)
    
    # Save embeddings
    save_embeddings(baseline_embeddings, f"{args.output_dir}/baseline_embeddings.pkl")
    save_embeddings(finetuned_embeddings, f"{args.output_dir}/finetuned_embeddings.pkl")
    
    # Save summary metadata
    metadata = {
        'baseline_summaries': baseline_summaries,
        'finetuned_summaries': finetuned_summaries,
        'embedding_method': 'simple_hash_based',
        'embedding_dimension': args.embedding_dim,
        'baseline_embedding_shape': baseline_embeddings.shape,
        'finetuned_embedding_shape': finetuned_embeddings.shape,
        'note': 'File names were swapped: baseline file contains finetuned results, finetuned file contains baseline results'
    }
    
    with open(f"{args.output_dir}/embedding_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Embeddings created successfully!")
    print(f"✓ Baseline embeddings: {baseline_embeddings.shape}")
    print(f"✓ Finetuned embeddings: {finetuned_embeddings.shape}")
    print(f"✓ Files saved to {args.output_dir}")

if __name__ == "__main__":
    main() 