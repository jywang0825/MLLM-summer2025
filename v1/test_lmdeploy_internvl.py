#!/usr/bin/env python3
"""
Test script for LMDeploy with InternVL3 model
"""
import torch
from PIL import Image
import numpy as np

def test_lmdeploy_internvl():
    """Test LMDeploy with InternVL3 model."""
    print("Testing LMDeploy with InternVL3...")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")
    
    try:
        from lmdeploy import pipeline
        
        print("Loading InternVL3 model with LMDeploy...")
        pipe = pipeline(
            'OpenGVLab/InternVL3-8B',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("Model loaded successfully!")
        
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color=0xFF0000)
        
        # Test captioning
        print("Testing captioning...")
        prompt = "What color is this image?"
        response = pipe([(prompt, [test_image])])
        print(f"Response: {response}")
        
        # Test text-only generation
        print("Testing text-only generation...")
        text_prompt = "Hello, how are you?"
        text_response = pipe([text_prompt])
        print(f"Text response: {text_response}")
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_lmdeploy_internvl() 