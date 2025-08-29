# InternVL3-8B Finetuned Model - LoRA Memory Optimized

## Model Information
- **Base Model**: OpenGVLab/InternVL3-8B
- **LLM Backbone**: Qwen2.5-32B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) with memory optimization
- **Training Date**: August 14, 2025

## Training Results
- **Final Training Loss**: 0.8904
- **Training Duration**: 1 day, 9 hours, 55 minutes (33+ hours)
- **Total Steps**: 11,433 steps
- **Training Samples**: 30,492 samples
- **Epochs**: ~3.0 epochs
- **Training Runtime**: 122,135.18 seconds
- **Samples per Second**: 0.749
- **Steps per Second**: 0.094

## Model Configuration
- **Architecture**: InternVLChatModel
- **Hidden Size**: 3584
- **Image Size**: 448px (dynamic)
- **Downsample Ratio**: 0.5
- **Dynamic Image Size**: Enabled
- **Attention Implementation**: Flash Attention 2
- **Model Type**: qwen2
- **Layers**: 28 attention layers, 28 hidden layers
- **Attention Heads**: 28
- **Key-Value Heads**: 4

## Files Included
- `all_results.json` - Complete training metrics
- `train_results.json` - Training results summary
- `trainer_state.json` - Trainer state and checkpoint information
- `training_args.bin` - Training arguments and configuration
- `config.json` - Model configuration

## Usage Notes
This model was fine-tuned using LoRA for memory efficiency and is optimized for video-language understanding tasks. The training completed successfully with a good final loss, indicating effective learning from the training data.

## Original Location
Source: `InternVL/internvl_chat/work_dirs/internvl3_8b_official_finetune_lora_memory_optimized/`
