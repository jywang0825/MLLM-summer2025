#!/usr/bin/env python3
"""
Test InternVL3 Official 4-bit LoRA Finetuning Setup
Validates all components for official InternVL3 training with 4-bit optimization
"""

import os
import sys
import json
import subprocess
import importlib
from pathlib import Path
import torch

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is too old. Need Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_gpu_availability():
    """Check GPU availability and memory"""
    print("\n🖥️  Checking GPU availability...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check if it's suitable for official InternVL3 training
        if "Ada" in gpu_name and gpu_memory >= 40:
            print(f"   ✅ GPU {i} is excellent for official InternVL3 training")
        elif gpu_memory >= 24:
            print(f"   ✅ GPU {i} is suitable for official InternVL3 training")
        else:
            print(f"   ⚠️  GPU {i} may need reduced batch size for official training")
    
    return True

def check_required_packages():
    """Check required packages for official InternVL3 training"""
    print("\n📦 Checking required packages...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'deepspeed': 'DeepSpeed',
        'accelerate': 'Accelerate',
        'tensorboard': 'TensorBoard',
        'pillow': 'Pillow (PIL)',
        'opencv-python': 'OpenCV',
        'sentencepiece': 'SentencePiece',
        'protobuf': 'Protobuf',
        'huggingface_hub': 'Hugging Face Hub',
        'bitsandbytes': 'BitsAndBytes (4-bit optimization)',
        'peft': 'PEFT (LoRA support)'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {description} ({package})")
        except ImportError:
            print(f"❌ {description} ({package}) - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_official.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_internvl_repository():
    """Check if InternVL repository is available"""
    print("\n📁 Checking InternVL repository...")
    
    internvl_paths = [
        "InternVL",
        "../InternVL"
    ]
    
    for path in internvl_paths:
        if os.path.exists(path):
            print(f"✅ InternVL repository found at {path}")
            
            # Check for key training files
            training_script = os.path.join(path, "internvl_chat/internvl/train/internvl_chat_finetune.py")
            if os.path.exists(training_script):
                print(f"✅ Official training script found: {training_script}")
                return True
            else:
                print(f"❌ Official training script not found in {path}")
    
    print("❌ InternVL repository not found or incomplete")
    print("Please ensure you're in the InternVL directory or it's properly cloned")
    return False

def check_deepspeed_config():
    """Check DeepSpeed configuration"""
    print("\n⚡ Checking DeepSpeed configuration...")
    
    config_file = "zero_stage3_config.json"
    if not os.path.exists(config_file):
        print(f"❌ {config_file} not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("✅ DeepSpeed config file exists")
        
        # Check key configurations
        if config.get('zero_optimization', {}).get('stage') == 3:
            print("✅ ZeRO Stage 3 configured")
        else:
            print("⚠️  ZeRO Stage 3 not configured")
        
        if config.get('bf16', {}).get('enabled'):
            print("✅ BF16 mixed precision configured")
        else:
            print("⚠️  BF16 mixed precision not configured")
        
        if config.get('zero_optimization', {}).get('offload_optimizer'):
            print("✅ Optimizer offloading configured")
        else:
            print("⚠️  Optimizer offloading not configured")
        
    except Exception as e:
        print(f"❌ Error reading DeepSpeed config: {e}")
        return False
    
    return True

def check_training_scripts():
    """Check training scripts exist"""
    print("\n📜 Checking training scripts...")
    
    required_scripts = [
        'finetune_internvl3_8b_4bit_lora.sh',
        'quick_start_internvl3_4bit_lora.sh',
        'download_internvl3_models.sh'
    ]
    
    missing_scripts = []
    
    for script in required_scripts:
        if os.path.exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - MISSING")
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"\n⚠️  Missing scripts: {', '.join(missing_scripts)}")
        return False
    
    return True

def check_dataset_files():
    """Check dataset files exist"""
    print("\n📊 Checking dataset files...")
    
    dataset_dir = "internvl3_data"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory {dataset_dir} not found")
        return False
    
    required_files = [
        'meta.json',
        'video_summary_dataset_train.jsonl',
        'video_summary_dataset_val.jsonl'
    ]
    
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            if file == 'meta.json':
                # Check meta file format
                try:
                    with open(file_path, 'r') as f:
                        meta = json.load(f)
                    if 'video_summary_dataset' in meta:
                        print(f"✅ {file} (valid format)")
                    else:
                        print(f"⚠️  {file} (invalid format)")
                        missing_files.append(file)
                except:
                    print(f"❌ {file} (invalid JSON)")
                    missing_files.append(file)
            else:
                # Count lines to estimate dataset size
                with open(file_path, 'r') as f:
                    line_count = sum(1 for line in f)
                print(f"✅ {file} ({line_count} samples)")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing dataset files: {', '.join(missing_files)}")
        print("Run: python prepare_internvl3_finetuning.py")
        return False
    
    return True

def check_model_directory():
    """Check if InternVL3 model directory exists"""
    print("\n🤖 Checking InternVL3 model...")
    
    model_path = "models/InternVL3-8B"
    if not os.path.exists(model_path):
        print(f"❌ Model directory {model_path} not found")
        print("Download with: ./download_internvl3_models.sh")
        return False
    
    # Check for key model files
    required_files = ['config.json', 'tokenizer.json', 'pytorch_model.bin']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing model files: {', '.join(missing_files)}")
        return False
    
    print("✅ InternVL3-8B model is ready")
    return True

def test_official_internvl3_loading():
    """Test loading InternVL3 model using official method"""
    print("\n🧪 Testing official InternVL3 model loading...")
    
    try:
        # Add InternVL to path
        sys.path.insert(0, 'InternVL')
        
        from internvl_chat.internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
        from transformers import AutoTokenizer
        
        model_path = "models/InternVL3-8B"
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded")
        
        print("Loading model with official InternVL3 method...")
        config = InternVLChatConfig.from_pretrained(model_path)
        config.llm_config.attn_implementation = 'flash_attention_2'
        config.template = "internvl2_5"
        config.select_layer = -1
        config.dynamic_image_size = True
        config.use_thumbnail = True
        config.ps_version = 'v2'
        config.min_dynamic_patch = 1
        config.max_dynamic_patch = 12
        
        model = InternVLChatModel.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            config=config
        )
        print("✅ Official InternVL3 model loaded successfully")
        
        # Test LoRA wrapping
        print("Testing LoRA wrapping...")
        model.wrap_llm_lora(r=32, lora_alpha=64, lora_dropout=0.05)
        print("✅ LoRA wrapping successful")
        
        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   Memory allocated: {memory_allocated:.2f} GB")
            print(f"   Memory reserved: {memory_reserved:.2f} GB")
            
            if memory_allocated < 35:
                print("✅ Memory usage is reasonable for official training")
            else:
                print("⚠️  Memory usage is high, consider reducing batch size")
        
        # Clean up
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Official InternVL3 model loading failed: {e}")
        return False

def test_official_training_script():
    """Test if official training script can be imported"""
    print("\n🧪 Testing official training script...")
    
    try:
        # Add InternVL to path
        sys.path.insert(0, 'InternVL')
        
        # Try to import the training script
        import internvl_chat.internvl.train.internvl_chat_finetune
        print("✅ Official training script can be imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Official training script import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 InternVL3 Official 4-bit LoRA Finetuning Setup Test")
    print("=" * 60)
    
    tests = [
        ("Python Version", check_python_version),
        ("GPU Availability", check_gpu_availability),
        ("Required Packages", check_required_packages),
        ("InternVL Repository", check_internvl_repository),
        ("DeepSpeed Configuration", check_deepspeed_config),
        ("Training Scripts", check_training_scripts),
        ("Dataset Files", check_dataset_files),
        ("Model Directory", check_model_directory),
        ("Official InternVL3 Loading", test_official_internvl3_loading),
        ("Official Training Script", test_official_training_script)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your official InternVL3 4-bit LoRA setup is ready.")
        print("\n🚀 Start training with:")
        print("   ./quick_start_internvl3_4bit_lora.sh")
        print("\n📊 Monitor training with:")
        print("   tensorboard --logdir work_dirs/")
        print("\n💡 Official InternVL3 advantages:")
        print("   - Proven training methodology")
        print("   - Optimized for InternVL3 architecture")
        print("   - LoRA rank 32 with official implementation")
        print("   - DeepSpeed ZeRO Stage 3 optimization")
        print("   - Dynamic image size support")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Install missing packages: pip install -r requirements_official.txt")
        print("   - Download model: ./download_internvl3_models.sh")
        print("   - Prepare dataset: python prepare_internvl3_finetuning.py")
        print("   - Ensure InternVL repository is properly set up")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 