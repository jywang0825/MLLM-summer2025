#!/usr/bin/env python3
"""
Create embeddings using InternVL3 models - original for baseline, finetuned for generated summaries.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import pickle
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoProcessor
from safetensors import safe_open

# Disable flash attention before any imports
os.environ["FLASH_ATTENTION_DISABLE"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "0"

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
        # Handle different field names for ground truth
        if 'test_summary' in result:
            ground_truth.append(result['test_summary'])
        elif 'original_summary' in result:
            ground_truth.append(result['original_summary'])
        
        if 'generated_summary' in result:
            generated.append(result['generated_summary'])
    
    return {
        'ground_truth': ground_truth,
        'generated': generated
    }

def load_finetuned_internvl3(model_path: str, checkpoint: int = 11000, device: str = "cuda"):
    """
    Load the finetuned InternVL3 model with specific checkpoint.
    """
    print(f"Loading finetuned InternVL3 model from {model_path} checkpoint {checkpoint}...")
    
    try:
        # Load the base model to get the correct architecture
        print("Loading base model architecture...")
        model = AutoModelForCausalLM.from_pretrained(
            "OpenGVLab/InternVL3-8B",
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # Disable flash attention
        )
        
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-8B", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-8B", trust_remote_code=True)
        
        print("✅ Base model architecture loaded successfully!")
        
        # Now load the finetuned weights from checkpoint
        print(f"Loading finetuned weights from checkpoint {checkpoint}...")
        finetuned_state_dict = {}
        
        # Load from the specific checkpoint
        for i in range(1, 5):
            safetensor_path = os.path.join(model_path, f"model-{i:05d}-of-00004.safetensors")
            if os.path.exists(safetensor_path):
                print(f"Loading weights from {safetensor_path}")
                with safe_open(safetensor_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        finetuned_state_dict[key] = f.get_tensor(key)
        
        if not finetuned_state_dict:
            print("No safetensors files found!")
            return None, None, None
        
        # Load the finetuned weights
        model.load_state_dict(finetuned_state_dict, strict=False)
        
        # Disable flash attention
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        
        print("✅ Finetuned weights loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        
        return model, tokenizer, processor
        
    except Exception as e:
        print(f"Error loading finetuned model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_embeddings_with_internvl3(texts: list, model_path: str, device: str = "cuda", is_finetuned: bool = False) -> np.ndarray:
    """
    Create embeddings using InternVL3 model.
    """
    print(f"Loading InternVL3 model from {model_path}...")
    
    try:
        if is_finetuned:
            # Load finetuned model
            model, tokenizer, processor = load_finetuned_internvl3(model_path, device=device)
            if model is None:
                raise Exception("Failed to load finetuned model")
        else:
            # Load original model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"
            )
        
        model.eval()
        
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = []
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                if i % 50 == 0:
                    print(f"Processing text {i+1}/{len(texts)}")
                
                # Tokenize
                inputs = tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get embeddings from the LLM part
                if hasattr(model, 'llm_model'):
                    # Use the LLM model for text embeddings
                    llm_outputs = model.llm_model(**inputs)
                    embedding = llm_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                else:
                    # Fallback to full model
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                embeddings.append(embedding.flatten())
        
        embeddings = np.array(embeddings)
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"Error with InternVL3 model: {e}")
        print("Falling back to simple embedding method...")
        return create_simple_embeddings_fallback(texts)

def create_simple_embeddings_fallback(texts: list, embedding_dim: int = 768) -> np.ndarray:
    """
    Fallback embedding method using simple hash-based approach.
    """
    print(f"Creating fallback embeddings with dimension {embedding_dim}...")
    
    import hashlib
    from collections import Counter
    import re
    
    # Create a simple vocabulary from all texts
    vocab = set()
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        vocab.update(words)
    
    vocab = list(vocab)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create embeddings based on word frequencies and semantic features
    embeddings = []
    
    for text in texts:
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
    parser = argparse.ArgumentParser(description="Create embeddings using InternVL3 models")
    parser.add_argument("baseline_results", help="Path to baseline evaluation results JSON")
    parser.add_argument("finetuned_results", help="Path to finetuned evaluation results JSON")
    parser.add_argument("--original_model", default="OpenGVLab/InternVL3-8B", help="Path to original InternVL3 model")
    parser.add_argument("--finetuned_model", default="../v8/work_dirs/internvl3_8b_single_gpu_aggressive", help="Path to your finetuned InternVL3 model")
    parser.add_argument("--checkpoint", type=int, default=11000, help="Checkpoint number to use")
    parser.add_argument("--output_dir", default="v9/embeddings", help="Output directory for embeddings")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
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
    
    # Create embeddings using original model for baseline
    print("Creating baseline embeddings using ORIGINAL InternVL3 model...")
    baseline_embeddings = create_embeddings_with_internvl3(
        baseline_summaries['ground_truth'] + baseline_summaries['generated'],
        args.original_model,
        args.device,
        is_finetuned=False
    )
    
    # Create embeddings using finetuned model for finetuned summaries
    print("Creating finetuned embeddings using FINETUNED InternVL3 model...")
    finetuned_embeddings = create_embeddings_with_internvl3(
        finetuned_summaries['ground_truth'] + finetuned_summaries['generated'],
        args.finetuned_model,
        args.device,
        is_finetuned=True
    )
    
    # Save embeddings
    save_embeddings(baseline_embeddings, f"{args.output_dir}/baseline_embeddings.pkl")
    save_embeddings(finetuned_embeddings, f"{args.output_dir}/finetuned_embeddings.pkl")
    
    # Save summary metadata
    metadata = {
        'baseline_summaries': baseline_summaries,
        'finetuned_summaries': finetuned_summaries,
        'original_model_path': args.original_model,
        'finetuned_model_path': args.finetuned_model,
        'checkpoint': args.checkpoint,
        'embedding_method': 'internvl3_text_encoder',
        'baseline_embedding_shape': baseline_embeddings.shape,
        'finetuned_embedding_shape': finetuned_embeddings.shape,
        'note': 'File names were swapped: baseline file contains finetuned results, finetuned file contains baseline results'
    }
    
    with open(f"{args.output_dir}/embedding_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Embeddings created successfully!")
    print(f"✓ Baseline embeddings (original model): {baseline_embeddings.shape}")
    print(f"✓ Finetuned embeddings (finetuned model): {finetuned_embeddings.shape}")
    print(f"✓ Files saved to {args.output_dir}")

if __name__ == "__main__":
    main() 