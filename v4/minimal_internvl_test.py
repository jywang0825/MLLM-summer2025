import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.append("../models/InternVL2_5-8B")

# Use the official loader if available, else fallback to HF loader
try:
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    model_path = "../models/InternVL2_5-8B"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True
    )
except Exception as e:
    print("Error loading model:", e)
    sys.exit(1)

print("Model loaded!")

# Create a dummy image
image = Image.fromarray((torch.rand(448, 448, 3).numpy() * 255).astype('uint8'))
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = next(model.parameters()).device
pixel_values = transform(image).unsqueeze(0).to(device)
pixel_values = pixel_values.half()

prompt = "Describe this image briefly."
generation_config = {
    'max_new_tokens': 100,
    'do_sample': True,
    'temperature': 0.7
}

try:
    print("Running model.chat on dummy image...")
    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=prompt,
        generation_config=generation_config
    )
    print("Model response:", response)
except Exception as e:
    print("Error during model forward:", e)
    import traceback; traceback.print_exc() 