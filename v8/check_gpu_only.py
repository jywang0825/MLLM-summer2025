#!/usr/bin/env python3
"""
Check GPU-only execution for InternVL3 finetuning
Verifies that PyTorch is using CUDA and not falling back to CPU
"""

import torch
import os
import sys

def check_gpu_only():
    """Check if PyTorch is configured for GPU-only execution"""
    
    print("🔍 Checking GPU-only execution configuration...")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ CUDA not available!")
        return False
    
    # Check environment variables
    print("\n📋 Environment Variables:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
    print(f"DS_ACCELERATOR: {os.environ.get('DS_ACCELERATOR', 'Not set')}")
    
    # Test tensor operations on GPU
    print("\n🧪 Testing GPU tensor operations:")
    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # Perform operations
        z = torch.matmul(x, y)
        
        print(f"✅ Tensor device: {x.device}")
        print(f"✅ Matrix multiplication successful on {z.device}")
        print(f"✅ Result shape: {z.shape}")
        
        # Check memory usage
        print(f"✅ GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"✅ GPU memory cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        
    except Exception as e:
        print(f"❌ GPU tensor operation failed: {e}")
        return False
    
    # Test model loading on GPU
    print("\n🤖 Testing model loading on GPU:")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load a small model for testing
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        
        # Move to GPU
        model = model.cuda()
        
        print(f"✅ Model device: {next(model.parameters()).device}")
        print(f"✅ Model loaded successfully on GPU")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False
    
    print("\n✅ All GPU-only checks passed!")
    print("🚀 Ready for GPU-only InternVL3 finetuning")
    return True

if __name__ == "__main__":
    success = check_gpu_only()
    sys.exit(0 if success else 1) 