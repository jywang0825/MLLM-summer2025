# Ego4D AKS Finetuning Project

This repository contains code for adaptive keyframe selection (AKS) on the Ego4D NLQ dataset for video understanding and summarization tasks.

## Project Overview

The project implements adaptive keyframe selection algorithms to extract the most relevant frames from Ego4D videos based on natural language queries and video summaries. This is useful for video understanding, summarization, and efficient video processing.

## Files Structure

### Main Scripts
- `ego4d_aks_full.py` - Full AKS implementation for Ego4D NLQ dataset with progress saving
- `ego4d_aks.py` - Basic AKS implementation
- `ego4d_aks_test_small.py` - Small-scale testing version
- `ego4d_aks_with_video_summary.py` - AKS with video summary integration
- `extract_narrations.py` - Extract video narrations
- `extract_nlq_summaries.py` - Extract NLQ summaries from dataset

### Data Files
- `nlq_val_summaries.json` - NLQ validation summaries
- `requirements.txt` - Python dependencies

### Directories
- `AKS/` - Core AKS algorithm implementations
- `ego4d_aks_full/` - Output directory for full AKS results
- `ego4d_aks_test_small/` - Output directory for small test results
- `ego4d_aks_test/` - Test output directory
- `nlq_summaries/` - Extracted NLQ summaries
- `nlq_summary_data/` - Summary data
- `extract/` - Extraction utilities
- `benchmark/` - Benchmarking scripts

## Features

- **Adaptive Keyframe Selection (AKS)**: Intelligent frame selection based on content relevance
- **Multiple Model Support**: CLIP and BLIP for feature extraction
- **Progress Saving**: Resume processing from where it left off
- **Frame Extraction**: Save selected frames as images
- **Ego4D Integration**: Works with Ego4D NLQ dataset
- **Remote Dataset Support**: Can work with remotely mounted Ego4D dataset

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic AKS Processing
```bash
python ego4d_aks_full.py
```

### Small Scale Testing
```bash
python ego4d_aks_test_small.py
```

### Extract NLQ Summaries
```bash
python extract_nlq_summaries.py
```

## Configuration

The scripts are configured to work with:
- Ego4D dataset mounted at `~/remote_ego4d`
- CLIP model for feature extraction (can be changed to BLIP)
- Maximum 32 frames per video
- AKS parameters: t1=0.8, t2=-100, depth=5

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Decord (for video processing)
- OpenCV
- PIL
- NumPy
- tqdm

## License

This project is for research purposes.

## Citation

If you use this code, please cite the relevant papers:
- Ego4D: Around the World in 3,000 Hours of Egocentric Video
- Adaptive Keyframe Selection for Video Understanding 