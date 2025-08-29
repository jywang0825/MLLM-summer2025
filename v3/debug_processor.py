#!/usr/bin/env python3
"""
Debug script to understand the processor structure
"""
import os
os.environ["FLASH_ATTENTION_DISABLE"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "0"

import sys
sys.path.append('/shared/ssd_14T/home/wangj/your-repo/InternVL/internvl_chat/internvl/model/internvl_chat')
sys.path.append('/shared/ssd_14T/home/wangj/your-repo/InternVL')

from transformers import AutoProcessor
from PIL import Image
import numpy as np

# Load the processor
processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-8B", trust_remote_code=True)

print("Processor type:", type(processor))
print("Processor attributes:", dir(processor))

# Check if it has image_processor
if hasattr(processor, 'image_processor'):
    print("Has image_processor:", type(processor.image_processor))
    print("Image processor attributes:", dir(processor.image_processor))

# Check if it has tokenizer
if hasattr(processor, 'tokenizer'):
    print("Has tokenizer:", type(processor.tokenizer))

# Test with a dummy image
dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
prompt = "Describe this image."

print("\nTesting processor calls...")

try:
    # Test 1: Direct call
    print("Test 1: Direct call with text and images")
    result1 = processor(text=prompt, images=dummy_image, return_tensors="pt")
    print("Success:", list(result1.keys()))
except Exception as e:
    print("Test 1 failed:", e)

try:
    # Test 2: Separate calls
    print("Test 2: Separate text and image calls")
    text_result = processor(text=prompt, return_tensors="pt")
    print("Text result keys:", list(text_result.keys()))
    
    if hasattr(processor, 'image_processor'):
        image_result = processor.image_processor(dummy_image, return_tensors="pt")
        print("Image result keys:", list(image_result.keys()))
    else:
        image_result = processor(images=dummy_image, return_tensors="pt")
        print("Image result keys:", list(image_result.keys()))
        
except Exception as e:
    print("Test 2 failed:", e) 