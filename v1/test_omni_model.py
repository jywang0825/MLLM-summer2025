#!/usr/bin/env python3
"""
Test script for Qwen2.5-Omni-7B model
Simple test to verify the model loads and can process images correctly.
"""
import os
import sys
import torch
from PIL import Image
import numpy as np

def test_omni_model():
    """Test the Qwen2.5-Omni-7B model with a simple image captioning task."""
    
    print("Testing Qwen2.5-Omni-7B model...")
    
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        
        # Model path
        model_path = "models/Qwen2.5-Omni-7B"
        
        print(f"Loading model from: {model_path}")
        
        # Load processor and model
        processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Disable talker to save memory (we only need text output)
        model.disable_talker()
        
        print("✅ Model loaded successfully!")
        
        # Create a simple test image (random noise for testing)
        print("Creating test image...")
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test text-only generation first
        print("Testing text-only generation...")
        text_conversation = [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
        
        text = processor.apply_chat_template(text_conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        
        text_ids = model.generate(
            **inputs, 
            return_audio=False,
            max_new_tokens=20,
            do_sample=False,
            temperature=0.0
        )
        
        text_response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"Text response: {text_response[0]}")
        
        # Test image captioning
        print("Testing image captioning...")
        image_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": "Describe this image briefly."}
                ]
            }
        ]
        
        text = processor.apply_chat_template(image_conversation, add_generation_prompt=True, tokenize=False)
        
        # Process multimodal inputs
        audios, images, videos = [], [test_image], []
        inputs = processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=False
        )
        inputs = inputs.to(model.device).to(model.dtype)
        
        text_ids = model.generate(
            **inputs, 
            use_audio_in_video=False, 
            return_audio=False,
            max_new_tokens=30,
            do_sample=False,
            temperature=0.0
        )
        
        response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"Image caption: {response[0]}")
        
        print("✅ All tests passed! Model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_omni_model()
    if success:
        print("\n🎉 Model test successful! You can now run the full captioning script.")
    else:
        print("\n💥 Model test failed! Please check the error messages above.") 