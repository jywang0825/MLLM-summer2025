# Benchmarking Results Directory

This directory contains the results and artifacts from various model fine-tuning experiments for benchmarking and comparison purposes.

## Available Models

### 1. InternVL3-8B LoRA Memory Optimized
- **Location**: `internvl3_8b_finetuned_lora_memory_optimized/`
- **Training Date**: August 14, 2025
- **Base Model**: OpenGVLab/InternVL3-8B
- **LLM**: Qwen2.5-32B-Instruct
- **Method**: LoRA fine-tuning with memory optimization
- **Final Loss**: 0.8904
- **Training Time**: 33+ hours

## Directory Structure
```
benchmarking_results/
├── README.md (this file)
└── internvl3_8b_finetuned_lora_memory_optimized/
    ├── README.md (model-specific details)
    ├── all_results.json (complete training metrics)
    ├── train_results.json (training results summary)
    ├── trainer_state.json (trainer state)
    ├── training_args.bin (training arguments)
    └── config.json (model configuration)
```

## Adding New Results
When adding new fine-tuning results:
1. Create a new subdirectory with descriptive name
2. Copy relevant result files (metrics, configs, etc.)
3. Create a README.md with model details
4. Update this main README.md file

## Notes
- Original model files remain in their source locations
- This directory contains only results and metadata for benchmarking
- Full model weights are not duplicated here to save space
