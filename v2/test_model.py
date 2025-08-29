#!/usr/bin/env python3
"""
Simple test script to verify Qwen2.5-Omni model can generate
"""
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from PIL import Image
import numpy as np

def test_model():
    print("Testing Qwen2.5-Omni model generation...")
    
    # Load model
    model_path = "./qwen2.5-omni-7b"
    print(f"Loading model from: {model_path}")
    
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map={"": 7},  # Use GPU 7
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    )
    
    print(f"Model loaded on: {next(model.parameters()).device}")
    
    # Create a simple test image (random noise)
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Test conversation
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a helpful assistant."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "Describe this image in one sentence."}
            ]
        }
    ]
    
    # Process inputs
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)
    
    print("Generating response...")
    
    # Generate with strict limits
    text_ids = model.generate(
        **inputs,
        use_audio_in_video=False,
        return_audio=False,
        max_new_tokens=16,  # Very short for testing
        do_sample=True,
        temperature=0.7,
        pad_token_id=processor.tokenizer.eos_token_id
    )
    
    response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    caption = response[0].split('<|im_end|>')[-1].strip()
    
    print(f"✅ SUCCESS! Generated: {caption}")
    print("Model is working correctly!")
    
    # Clean up
    del model, processor, inputs, text_ids
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_model() 