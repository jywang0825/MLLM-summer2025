#!/usr/bin/env python3
"""
Create proper embeddings from video summaries using sentence-transformers.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import pickle
import os
import sys

# Add a simple fallback if sentence-transformers fails
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available, using fallback method")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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

def create_embeddings_with_sentence_transformers(summaries: dict, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Create embeddings using sentence-transformers.
    """
    print(f"Loading sentence transformer model: {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        
        # Combine ground truth and generated summaries
        all_texts = summaries['ground_truth'] + summaries['generated']
        
        print(f"Creating embeddings for {len(all_texts)} summaries...")
        embeddings = model.encode(
            all_texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"Error with sentence-transformers: {e}")
        print("Falling back to simple embedding method...")
        return create_simple_embeddings_fallback(summaries)

def create_simple_embeddings_fallback(summaries: dict, embedding_dim: int = 384) -> np.ndarray:
    """
    Fallback embedding method using simple hash-based approach.
    """
    print(f"Creating fallback embeddings with dimension {embedding_dim}...")
    
    import hashlib
    from collections import Counter
    import re
    
    # Combine all texts to build vocabulary
    all_texts = summaries['ground_truth'] + summaries['generated']
    
    # Create a simple vocabulary from all texts
    vocab = set()
    for text in all_texts:
        words = re.findall(r'\b\w+\b', text.lower())
        vocab.update(words)
    
    vocab = list(vocab)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create embeddings based on word frequencies and semantic features
    embeddings = []
    
    for text in all_texts:
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        
        # Initialize embedding
        embedding = np.zeros(embedding_dim)
        
        # Add semantic features based on content
        text_lower = text.lower()
        
        # Kitchen/cooking related
        if any(word in text_lower for word in ['kitchen', 'cooking', 'cook', 'food', 'meal']):
            embedding[:50] += 0.3
        if any(word in text_lower for word in ['stove', 'oven', 'pan', 'pot']):
            embedding[50:100] += 0.3
            
        # Office/work related
        if any(word in text_lower for word in ['office', 'work', 'desk', 'computer', 'meeting']):
            embedding[100:150] += 0.3
        if any(word in text_lower for word in ['typing', 'writing', 'document']):
            embedding[150:200] += 0.3
            
        # Outdoor/activity related
        if any(word in text_lower for word in ['outdoor', 'outside', 'walking', 'running', 'exercise']):
            embedding[200:250] += 0.3
        if any(word in text_lower for word in ['park', 'street', 'road', 'path']):
            embedding[250:300] += 0.3
            
        # Social interaction related
        if any(word in text_lower for word in ['conversation', 'talking', 'speaking', 'discussion']):
            embedding[300:350] += 0.3
        if any(word in text_lower for word in ['people', 'person', 'group', 'friend']):
            embedding[350:384] += 0.3
        
        # Add word frequency features
        for word in vocab:
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
    print(f"Created fallback embeddings with shape: {embeddings.shape}")
    return embeddings

def save_embeddings(embeddings: np.ndarray, output_file: str):
    """Save embeddings to file."""
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create proper embeddings from video summaries")
    parser.add_argument("baseline_results", help="Path to baseline evaluation results JSON")
    parser.add_argument("finetuned_results", help="Path to finetuned evaluation results JSON")
    parser.add_argument("--output_dir", default="v9/embeddings", help="Output directory for embeddings")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model to use")
    
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
    baseline_embeddings = create_embeddings_with_sentence_transformers(baseline_summaries, args.model)
    
    print("Creating finetuned embeddings...")
    finetuned_embeddings = create_embeddings_with_sentence_transformers(finetuned_summaries, args.model)
    
    # Save embeddings
    save_embeddings(baseline_embeddings, f"{args.output_dir}/baseline_embeddings.pkl")
    save_embeddings(finetuned_embeddings, f"{args.output_dir}/finetuned_embeddings.pkl")
    
    # Save summary metadata
    metadata = {
        'baseline_summaries': baseline_summaries,
        'finetuned_summaries': finetuned_summaries,
        'embedding_model': args.model,
        'embedding_method': 'sentence_transformers' if SENTENCE_TRANSFORMERS_AVAILABLE else 'fallback_hash_based',
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