import requests
import sys

# Usage: python benchmark_deepseek.py "Your prompt here"
if len(sys.argv) < 2:
    print("Usage: python benchmark_deepseek.py 'Your prompt here'")
    sys.exit(1)

prompt = sys.argv[1]

url = "http://localhost:5000/benchmark"
data = {"prompt": prompt}

response = requests.post(url, json=data)

if response.ok:
    print("Response:", response.json().get("response"))
else:
    print("Error:", response.text) 