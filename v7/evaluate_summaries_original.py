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
            # Use original_summary instead of generated_summary
            test_summaries[video_id] = entry.get('original_summary', '')
    
    print(f"Loaded {len(test_summaries)} test summaries as ground truth")
    return test_summaries 