#!/usr/bin/env python3
"""
Comprehensive Summary Evaluation Script
Compares test summaries (from test_summaries_minimal.json) to generated summaries using multiple metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L, ROUGE-1, ROUGE-2
- Sentence Transformers (semantic similarity)
- CLIP Score (visual-semantic alignment)
"""
import os
import json
import numpy as np
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# NLTK for BLEU and METEOR
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# ROUGE
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Installing rouge-score...")
    import subprocess
    subprocess.run(["pip", "install", "rouge-score"], check=True)
    from rouge_score import rouge_scorer

# Sentence Transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.run(["pip", "install", "sentence-transformers"], check=True)
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

# CLIP for visual-semantic alignment
try:
    import torch
    from transformers.models.clip.processing_clip import CLIPProcessor
    from transformers.models.clip.modeling_clip import CLIPModel
    from PIL import Image
    import cv2
except ImportError:
    print("Installing transformers...")
    import subprocess
    subprocess.run(["pip", "install", "transformers"], check=True)
    import torch
    from transformers.models.clip.processing_clip import CLIPProcessor
    from transformers.models.clip.modeling_clip import CLIPModel
    from PIL import Image
    import cv2

# CLAIR for semantic evaluation
try:
    from clair import clair
except ImportError:
    print("CLAIR not available. Install with: pip install git+https://github.com/DavidMChan/clair.git")
    clair = None

def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

def load_test_summaries(test_file="test_summaries_minimal.json"):
    """Load test summaries from the minimal file to use as ground truth."""
    print(f"Loading test summaries from {test_file}...")
    
    if not os.path.exists(test_file):
        print(f"Warning: Test summaries file {test_file} not found. Using original summaries as fallback.")
        return {}
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a mapping from video_id to test summary
    test_summaries = {}
    for entry in data:
        video_id = entry.get('video_id')
        if video_id:
            test_summaries[video_id] = entry.get('original_summary', '')
    
    print(f"Loaded {len(test_summaries)} test summaries as ground truth")
    return test_summaries

def load_clip_model():
    """Load CLIP model for visual-semantic scoring."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if model is not None:
            model.to(device)
        return model, processor, device
    except Exception as e:
        print(f"Warning: Could not load CLIP model: {e}")
        return None, None, None

def load_sentence_transformer():
    """Load sentence transformer for semantic similarity."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        print(f"Warning: Could not load sentence transformer: {e}")
        return None

def calculate_bleu_scores(reference, candidate):
    """Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores."""
    try:
        # Tokenize
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        
        # Calculate BLEU scores
        smoothing = SmoothingFunction().method1
        
        bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu_3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu_4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        return {
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4
        }
    except Exception as e:
        print(f"Error calculating BLEU scores: {e}")
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

def calculate_meteor_score(reference, candidate):
    """Calculate METEOR score."""
    try:
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        
        # METEOR expects list of references
        meteor = meteor_score([ref_tokens], cand_tokens)
        return meteor
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0

def calculate_rouge_scores(reference, candidate):
    """Calculate ROUGE scores."""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall
        }
    except Exception as e:
        print(f"Error calculating ROUGE scores: {e}")
        return {
            'rouge1_f1': 0.0, 'rouge1_precision': 0.0, 'rouge1_recall': 0.0,
            'rouge2_f1': 0.0, 'rouge2_precision': 0.0, 'rouge2_recall': 0.0,
            'rougeL_f1': 0.0, 'rougeL_precision': 0.0, 'rougeL_recall': 0.0
        }

def calculate_semantic_similarity(reference, candidate, sentence_model):
    """Calculate semantic similarity using sentence transformers."""
    try:
        if sentence_model is None:
            return 0.0
        
        # Encode sentences
        embeddings = sentence_model.encode([reference, candidate])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0

def calculate_clair_score(reference, candidate, model='gpt-3.5-turbo'):
    """Calculate CLAIR score for semantic evaluation."""
    try:
        if clair is None:
            print("CLAIR not available, skipping CLAIR score")
            return 0.0
        
        # Set up OpenAI API key if available
        if 'OPENAI_API_KEY' not in os.environ:
            print("OPENAI_API_KEY not set. Set it to use CLAIR scoring.")
            return 0.0
        
        # Calculate CLAIR score
        clair_score = clair([candidate], [reference], model=model)
        return clair_score
        
    except Exception as e:
        print(f"Error calculating CLAIR score: {e}")
        return 0.0

def calculate_clip_score(reference, candidate, frames_dir, video_uid, clip_model, clip_processor, device):
    """Calculate CLIP score for visual-semantic alignment."""
    try:
        if clip_model is None:
            return 0.0
        
        # Load a sample frame for visual representation
        video_frames_dir = os.path.join(frames_dir, video_uid)
        if not os.path.exists(video_frames_dir):
            return 0.0
        
        frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
        if not frame_files:
            return 0.0
        
        # Use the first frame as representative
        frame_path = os.path.join(video_frames_dir, frame_files[0])
        image = Image.open(frame_path).convert('RGB')
        
        # Process image and text with CLIP
        inputs = clip_processor(
            text=[candidate],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            # Calculate similarity
            similarity = torch.nn.functional.cosine_similarity(
                logits_per_image, logits_per_text, dim=-1
            ).item()
            
            # Convert to 0-1 scale
            clip_score = (similarity + 1) / 2
            
        return clip_score
    except Exception as e:
        print(f"Error calculating CLIP score: {e}")
        return 0.0

def evaluate_summaries(caption_file, frames_dir="../v1/ego4d_aks_full/frames", test_file="test_summaries_minimal.json", clair_model="gpt-3.5-turbo"):
    """Evaluate summaries using test summaries as ground truth."""
    print(f"Loading caption file: {caption_file}")
    
    with open(caption_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} video entries")
    
    # Load test summaries as ground truth
    test_summaries = load_test_summaries(test_file)
    
    # Initialize models
    download_nltk_data()
    clip_model, clip_processor, device = load_clip_model()
    sentence_model = load_sentence_transformer()
    
    results = []
    matched_count = 0
    
    for entry in tqdm(data, desc="Evaluating summaries"):
        video_uid = entry.get('video_uid')
        generated_summary = entry.get('generated_summary', '')
        
        # Get test summary as reference (ground truth)
        test_summary = test_summaries.get(video_uid, '')
        
        # If no test summary available, try to use original summary as fallback
        if not test_summary:
            test_summary = entry.get('original_summary', '')
        
        if not test_summary or not generated_summary:
            print(f"Skipping {video_uid}: missing summaries")
            continue
        
        matched_count += 1
        print(f"\nEvaluating {video_uid}")
        print(f"Test Summary (Ground Truth): {test_summary}")
        print(f"Generated: {generated_summary}")
        
        # Calculate all metrics
        bleu_scores = calculate_bleu_scores(test_summary, generated_summary)
        meteor = calculate_meteor_score(test_summary, generated_summary)
        rouge_scores = calculate_rouge_scores(test_summary, generated_summary)
        semantic_sim = calculate_semantic_similarity(test_summary, generated_summary, sentence_model)
        clair_score = calculate_clair_score(test_summary, generated_summary, clair_model)
        clip_score = calculate_clip_score(test_summary, generated_summary, frames_dir, video_uid, 
                                        clip_model, clip_processor, device)
        
        # Combine all scores
        result = {
            'video_uid': video_uid,
            'test_summary': test_summary,
            'generated_summary': generated_summary,
            **bleu_scores,
            'meteor': meteor,
            **rouge_scores,
            'semantic_similarity': semantic_sim,
            'clair_score': clair_score,
            'clip_score': clip_score
        }
        
        results.append(result)
        
        # Print current scores
        print(f"BLEU-1: {bleu_scores['bleu_1']:.4f}, BLEU-4: {bleu_scores['bleu_4']:.4f}")
        print(f"METEOR: {meteor:.4f}")
        print(f"ROUGE-L F1: {rouge_scores['rougeL_f1']:.4f}")
        print(f"Semantic Similarity: {semantic_sim:.4f}")
        print(f"CLAIR Score: {clair_score:.4f}")
        print(f"CLIP Score: {clip_score:.4f}")
    
    print(f"\nTotal videos with test summaries: {matched_count}")
    return results

def calculate_average_scores(results):
    """Calculate average scores across all videos."""
    if not results:
        return {}
    
    # Get all metric names (excluding non-numeric fields)
    metric_names = [k for k in results[0].keys() if k not in ['video_uid', 'test_summary', 'generated_summary']]
    
    averages = {}
    for metric in metric_names:
        values = [r[metric] for r in results if isinstance(r[metric], (int, float))]
        if values:
            averages[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return averages

def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_results(results, averages, output_file):
    """Save evaluation results to file."""
    # Convert numpy types to Python types
    results_converted = convert_numpy_types(results)
    averages_converted = convert_numpy_types(averages)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_videos': len(results),
        'average_scores': averages_converted,
        'detailed_results': results_converted
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_file}")

def print_summary(averages):
    """Print a summary of the evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    if not averages:
        print("No results to display")
        return
    
    # BLEU scores
    print("\nBLEU Scores:")
    for i in range(1, 5):
        metric = f'bleu_{i}'
        if metric in averages:
            score = averages[metric]
            print(f"  BLEU-{i}: {score['mean']:.4f} ± {score['std']:.4f} (range: {score['min']:.4f}-{score['max']:.4f})")
    
    # METEOR
    if 'meteor' in averages:
        score = averages['meteor']
        print(f"\nMETEOR: {score['mean']:.4f} ± {score['std']:.4f} (range: {score['min']:.4f}-{score['max']:.4f})")
    
    # ROUGE scores
    print("\nROUGE Scores:")
    for metric in ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']:
        if metric in averages:
            score = averages[metric]
            print(f"  {metric.upper()}: {score['mean']:.4f} ± {score['std']:.4f} (range: {score['min']:.4f}-{score['max']:.4f})")
    
    # Semantic similarity
    if 'semantic_similarity' in averages:
        score = averages['semantic_similarity']
        print(f"\nSemantic Similarity: {score['mean']:.4f} ± {score['std']:.4f} (range: {score['min']:.4f}-{score['max']:.4f})")
    
    # CLAIR score
    if 'clair_score' in averages:
        score = averages['clair_score']
        print(f"CLAIR Score: {score['mean']:.4f} ± {score['std']:.4f} (range: {score['min']:.4f}-{score['max']:.4f})")
    
    # CLIP score
    if 'clip_score' in averages:
        score = averages['clip_score']
        print(f"CLIP Score: {score['mean']:.4f} ± {score['std']:.4f} (range: {score['min']:.4f}-{score['max']:.4f})")

def main():
    parser = argparse.ArgumentParser(description='Evaluate generated summaries against test summaries (ground truth)')
    parser.add_argument('caption_file', help='Path to the caption JSON file')
    parser.add_argument('--frames_dir', default='../v1/ego4d_aks_full/frames', 
                       help='Path to frames directory for CLAIRE scoring')
    parser.add_argument('--output', default='evaluation_results.json', 
                       help='Output file for evaluation results')
    parser.add_argument('--test_file', default='test_summaries_minimal.json',
                       help='Path to test summaries file to use as ground truth')
    parser.add_argument('--clair_model', default='gpt-3.5-turbo',
                       help='OpenAI model to use for CLAIR scoring (e.g., gpt-3.5-turbo, gpt-4)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.caption_file):
        print(f"Error: Caption file not found: {args.caption_file}")
        return
    
    print("Starting comprehensive summary evaluation...")
    print(f"Caption file: {args.caption_file}")
    print(f"Test file (ground truth): {args.test_file}")
    print(f"Frames directory: {args.frames_dir}")
    print(f"Output file: {args.output}")
    print(f"CLAIR model: {args.clair_model}")
    
    # Run evaluation
    results = evaluate_summaries(args.caption_file, args.frames_dir, args.test_file, args.clair_model)
    
    if not results:
        print("No valid results to evaluate")
        return
    
    # Calculate averages
    averages = calculate_average_scores(results)
    
    # Save results
    save_results(results, averages, args.output)
    
    # Print summary
    print_summary(averages)
    
    print(f"\nEvaluation completed! Results saved to {args.output}")

if __name__ == "__main__":
    main() 