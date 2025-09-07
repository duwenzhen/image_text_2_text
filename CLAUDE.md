# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for testing and comparing vision-language models with metadata-enhanced image description. The project focuses on combining visual AI analysis with photo metadata (location, time, camera settings) to create rich, contextual descriptions.

## Development Environment

- **Python Version**: >=3.12,<3.14
- **Package Manager**: Poetry
- **Virtual Environment**: Configured to use in-project virtual environments (`.venv/`)
- **Models Tested**: Google Gemma-3-4B-IT, Qwen2.5-VL-3B-Instruct

## Key Dependencies

- **transformers**: Hugging Face transformers library
- **torch**: PyTorch for model inference
- **pillow**: Image processing
- **pillow-heif**: HEIC image format support
- **qwen-vl-utils**: Utilities for Qwen vision models
- **accelerate**: Multi-GPU and optimization support
- **python-dotenv**: Environment variable management

## Common Commands

### Model Testing
```bash
# Test Gemma-3-4B-IT model
poetry run python test_gemma.py

# Test Qwen2.5-VL-3B-Instruct model
poetry run python test_qwen.py

# Compare both models side-by-side
poetry run python compare_models.py
```

### Image Processing
```bash
# Convert HEIC to JPEG with metadata preservation
poetry run python convert_heic.py

# Extract comprehensive metadata from images
poetry run python extract_metadata.py
```

### Poetry Commands
```bash
# Install all dependencies
poetry install

# Add a new dependency
poetry add <package_name>

# Run commands in the virtual environment
poetry run <command>

# Activate the shell
poetry shell

# Show installed packages
poetry show
```

## Project Structure

### Core Scripts
- `test_gemma.py` - Test Gemma-3-4B-IT model with metadata integration
- `test_qwen.py` - Test Qwen2.5-VL-3B-Instruct model with metadata integration
- `compare_models.py` - Compare both models side-by-side
- `convert_heic.py` - Convert HEIC files to JPEG with metadata preservation
- `extract_metadata.py` - Extract comprehensive metadata from images

### Configuration
- `pyproject.toml` - Poetry configuration and project metadata
- `poetry.toml` - Poetry settings (in-project virtual environments)
- `.env` - Environment variables (not tracked, contains HUGGINGFACE_TOKEN)

### Data Directory
- `data/` - Contains test images and generated outputs
  - `IMG_1854.jpg` - Sample image file
  - `*_metadata.json` - Extracted metadata files
  - `model_comparison_*.txt` - Model comparison results

## Development Setup

1. **Clone and Install**:
   ```bash
   git clone <repository>
   cd image_text_2_text
   poetry install
   ```

2. **Environment Setup**:
   - Create `.env` file with `HUGGINGFACE_TOKEN=your_token_here`
   - Ensure sufficient disk space for model downloads (~15GB)
   - GPU recommended but not required

3. **Model Cache**:
   - Models are cached to `/media/wenzhen/SSD1T/huggingface_models/`
   - First run will download models (may take 10-15 minutes)

## Key Features

### Metadata Integration
- Extracts GPS coordinates and converts to human-readable locations
- Processes date/time information from EXIF data
- Includes camera information (make, model, settings)
- Creates context-aware prompts for AI models

### HEIC Support
- Converts Apple HEIC format to JPEG
- Preserves all metadata during conversion
- Exports detailed metadata to JSON files

### Natural Language Descriptions
- Generates storytelling-style descriptions
- Combines visual analysis with metadata context
- Avoids technical jargon, focuses on human-readable narratives
- Supports up to 1000 tokens for detailed descriptions

## Model Comparison

The project supports testing two different approaches:
- **Gemma-3-4B-IT**: Text model with image support via chat templates
- **Qwen2.5-VL-3B-Instruct**: Dedicated vision-language model with smart memory management

Both models receive identical metadata-enhanced prompts for fair comparison.

## Environment Variables

Create a `.env` file with:
```
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

## Notes

### GPU Memory Requirements
- **Gemma-3-4B-IT**: Works well on RTX 2070 (8GB)
- **Qwen2.5-VL-3B-Instruct**: Auto-detects GPU capacity and falls back to CPU for RTX 2070
- **High-end GPUs** (12GB+): Can run both models with quantization for optimal performance

### Performance Tips
- CPU inference is supported but significantly slower
- RTX 2070 users: Gemma model recommended for GPU acceleration
- First model download requires internet connection and time
- HEIC files require `pillow-heif` for processing
- Metadata extraction works with most JPEG/HEIC files with EXIF data

### Memory Management
- Automatic GPU memory detection and clearing
- Smart fallback to CPU when GPU memory insufficient
- 4-bit/8-bit quantization support for high-memory GPUs
- `bitsandbytes` integration for memory optimization