from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    image_b64 = data.get("images", [None])[0]  # expects a list of images
    model = data.get("model", "llama4:scout")

    if image_b64 is None:
        print("[DEBUG] No image provided in request.")
        return jsonify({"error": "No image provided"}), 400

    # Forward the request to Ollama
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }
    print("[DEBUG] Payload sent to Ollama:", payload)
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        print("[DEBUG] Ollama raw response:", response.text)
        response.raise_for_status()
        result = response.json()
        print("[DEBUG] Ollama parsed response:", result)
        return jsonify({"response": result.get("response", "")})
    except Exception as e:
        print("[DEBUG] Exception occurred:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)