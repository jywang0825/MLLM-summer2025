# Video Captioning with Vision-Language Models

A comprehensive video captioning system that uses various state-of-the-art vision-language models (VLM) to generate detailed captions for video frames and entire videos.

## Features

- **Multi-Model Support**: Integration with multiple VLMs including:
  - LLaVA (Large Language and Vision Assistant)
  - Qwen2.5-VL (Qwen2.5 Vision Language)
  - Llama4-Scout (Llama4 with vision capabilities)
  - DeepSeek-VL2

- **Flexible Processing**: 
  - Frame-level captioning with timestamp extraction
  - Video-level captioning for entire videos
  - Batch processing for large datasets

- **Ollama Integration**: Local model serving through Ollama API
- **Flask API**: RESTful API endpoints for captioning services
- **Benchmarking Tools**: Comprehensive evaluation scripts for model comparison

## Project Structure

```
finetuning/
├── benchmarking/           # Model benchmarking and evaluation scripts
│   ├── llava_caption.py   # LLaVA captioning script
│   ├── qwen_caption.py    # Qwen2.5-VL captioning script
│   └── llama_caption.py   # Llama4-Scout captioning script
├── data_files/            # Dataset and annotation files
├── models/                # Model files and configurations
├── extract.py             # Video frame extraction utility
├── deepseek_vl2_flask_app.py  # DeepSeek-VL2 Flask API
├── llava_server.py        # LLaVA server implementation
└── llama_server.py        # Llama server implementation
```

## Prerequisites

- Python 3.8+
- Ollama (for local model serving)
- FFmpeg (for video processing)
- Sufficient GPU memory for vision-language models

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd finetuning
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and setup Ollama**:
   ```bash
   # Follow instructions at https://ollama.ai
   # Pull required models
   ollama pull llava
   ollama pull qwen2.5-vl
   ollama pull llama4-scout
   ```

## Usage

### 1. Frame-Level Captioning

Extract frames from videos and generate captions:

```bash
python benchmarking/llava_caption.py
```

### 2. Video-Level Captioning

Generate captions for entire videos:

```bash
python benchmarking/qwen_caption.py
```

### 3. Flask API Server

Start the captioning API server:

```bash
python deepseek_vl2_flask_app.py
```

### 4. Ollama Integration

Use the server scripts to interact with Ollama models:

```bash
python llava_server.py
```

## API Endpoints

### POST /caption
Generate captions for uploaded images or videos.

**Request**:
```json
{
  "image": "base64_encoded_image",
  "prompt": "Describe this image in detail"
}
```

**Response**:
```json
{
  "caption": "Generated caption text",
  "model": "model_name",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Configuration

### Model Configuration

Each model can be configured in its respective script:

- **LLaVA**: Configure in `benchmarking/llava_caption.py`
- **Qwen2.5-VL**: Configure in `benchmarking/qwen_caption.py`
- **Llama4-Scout**: Configure in `benchmarking/llama_caption.py`

### Environment Variables

Create a `.env` file for configuration:

```env
OLLAMA_HOST=http://localhost:11434
MODEL_NAME=qwen2.5-vl
BATCH_SIZE=4
MAX_TOKENS=512
```

## Benchmarking

Run comprehensive benchmarks to compare model performance:

```bash
# LLaVA benchmark
python llava_benchmark.py

# DeepSeek benchmark
python benchmark_deepseek.py
```

## Data Processing

### Frame Extraction

Extract frames from videos at specific intervals:

```bash
python extract.py --video_path /path/to/video --output_dir ./frames --interval 1
```

### Dataset Preparation

The system supports various video datasets including Ego4D. Place your dataset files in the `data_files/` directory.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai) for local model serving
- [LLaVA](https://github.com/haotian-liu/LLaVA) for the vision-language model
- [Qwen](https://github.com/QwenLM/Qwen-VL) for Qwen2.5-VL
- [Ego4D](https://ego4d-data.org/) for the video dataset

## Support

For issues and questions:
1. Check the [Issues](https://github.com/your-username/your-repo/issues) page
2. Create a new issue with detailed information
3. Include error logs and system information

## Roadmap

- [ ] Support for more vision-language models
- [ ] Real-time video captioning
- [ ] Web interface for easy interaction
- [ ] Model fine-tuning capabilities
- [ ] Multi-language support 