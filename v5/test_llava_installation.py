#!/usr/bin/env python3
"""
Test script to verify LLaVA-NeXT installation and basic functionality.
"""

import torch
import warnings
warnings.filterwarnings("ignore")

def test_llava_installation():
    """Test if LLaVA-NeXT can be imported and basic functionality works."""
    print("Testing LLaVA-NeXT installation...")
    
    try:
        # Test imports
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
        from llava.conversation import conv_templates, SeparatorStyle
        
        print("✓ All LLaVA-NeXT imports successful")
        
        # Test GPU availability
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ No GPU available, will use CPU")
        
        # Test conversation templates
        conv_template = "qwen_1_5"
        if conv_template in conv_templates:
            print(f"✓ Conversation template '{conv_template}' available")
        else:
            print(f"⚠ Conversation template '{conv_template}' not found")
            print(f"Available templates: {list(conv_templates.keys())}")
        
        print("\n🎉 LLaVA-NeXT installation test passed!")
        print("You can now run the video captioning script.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run: bash install_llava_next.sh")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_llava_installation() 