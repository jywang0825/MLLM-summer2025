#!/usr/bin/env python3
"""
Comprehensive Summary Evaluation Script with CLAIR
Compares original summaries to generated summaries using multiple metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L, ROUGE-1, ROUGE-2
- CLAIR (semantic evaluation using LLMs)
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
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

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
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_f1': scores['rougeL'].fmeasure
        }
    except Exception as e:
        print(f"Error calculating ROUGE scores: {e}")
        return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}

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

def evaluate_summaries(input_file, output_file=None, max_samples=None):
    """Evaluate summaries using multiple metrics."""
    print(f"Loading data from {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Processing first {len(data)} samples")
    
    results = []
    
    for i, entry in enumerate(tqdm(data, desc="Evaluating summaries")):
        video_uid = entry.get('video_uid')
        original_summary = entry.get('original_summary', '')
        generated_summary = entry.get('generated_summary', '') or entry.get('generated_caption', '')
        
        if not original_summary or not generated_summary:
            print(f"Skipping {video_uid}: missing summaries")
            continue
        
        print(f"\nProcessing {i+1}/{len(data)}: {video_uid}")
        print(f"Original: {original_summary}")
        print(f"Generated: {generated_summary}")
        
        # Calculate all metrics
        bleu_scores = calculate_bleu_scores(original_summary, generated_summary)
        meteor = calculate_meteor_score(original_summary, generated_summary)
        rouge_scores = calculate_rouge_scores(original_summary, generated_summary)
        clair_score = calculate_clair_score(original_summary, generated_summary)
        
        # Combine all scores
        result = {
            'video_uid': video_uid,
            'original_summary': original_summary,
            'generated_summary': generated_summary,
            **bleu_scores,
            'meteor': meteor,
            **rouge_scores,
            'clair_score': clair_score
        }
        
        results.append(result)
        
        # Print current scores
        print(f"BLEU-1: {bleu_scores['bleu_1']:.4f}, BLEU-4: {bleu_scores['bleu_4']:.4f}")
        print(f"METEOR: {meteor:.4f}")
        print(f"ROUGE-L F1: {rouge_scores['rougeL_f1']:.4f}")
        print(f"CLAIR Score: {clair_score:.4f}")
    
    # Calculate averages
    if results:
        averages = {}
        for metric in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'clair_score']:
            values = [r[metric] for r in results if metric in r]
            if values:
                averages[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # BLEU scores
        for i in range(1, 5):
            metric = f'bleu_{i}'
            if metric in averages:
                score = averages[metric]
                print(f"BLEU-{i}: {score['mean']:.4f} ± {score['std']:.4f} (range: {score['min']:.4f}-{score['max']:.4f})")
        
        # METEOR
        if 'meteor' in averages:
            score = averages['meteor']
            print(f"METEOR: {score['mean']:.4f} ± {score['std']:.4f} (range: {score['min']:.4f}-{score['max']:.4f})")
        
        # ROUGE scores
        print("\nROUGE Scores:")
        for metric in ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']:
            if metric in averages:
                score = averages[metric]
                print(f"  {metric.upper()}: {score['mean']:.4f} ± {score['std']:.4f} (range: {score['min']:.4f}-{score['max']:.4f})")
        
        # CLAIR score
        if 'clair_score' in averages:
            score = averages['clair_score']
            print(f"\nCLAIR Score: {score['mean']:.4f} ± {score['std']:.4f} (range: {score['min']:.4f}-{score['max']:.4f})")
        
        print("="*80)
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate summaries using multiple metrics including CLAIR')
    parser.add_argument('input_file', help='Input JSON file with summaries')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--max_samples', '-m', type=int, help='Maximum number of samples to process')
    parser.add_argument('--clair_model', default='gpt-3.5-turbo', help='CLAIR model to use')
    
    args = parser.parse_args()
    
    print("Comprehensive Summary Evaluation with CLAIR")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"CLAIR model: {args.clair_model}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Initialize models
    download_nltk_data()
    
    # Evaluate summaries
    results = evaluate_summaries(args.input_file, args.output, args.max_samples)
    
    print(f"\nEvaluation completed!")
    print(f"Total samples processed: {len(results)}")

if __name__ == "__main__":
    main() 