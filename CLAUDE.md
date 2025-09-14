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
- **piexif**: EXIF metadata writing and editing
- **numpy**: Array operations and batch processing
- **opencv-python**: Video frame extraction and processing (NEW!)
- **pymediainfo**: Video metadata extraction (optional, fallback available)

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

# Convert HEIC/MOV/MP4 files to JPEG with metadata preservation (NEW!)
poetry run python convert_media.py
```

### Batch Photo Processing
```bash
# Process thousands of photos with AI descriptions (Simple version)
poetry run python simple_batch_processor.py /path/to/photos

# Process with advanced optimizations (memory management, progress tracking)
poetry run python batch_process_photos.py /path/to/photos

# Run interactive demo
poetry run python demo_batch.py

# Organize photos by camera type from zip files (with auto-conversion)
poetry run python organize_by_camera.py /path/to/zip/folder
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
- `convert_media.py` - **NEW!** Convert HEIC/MOV/MP4 to JPEG with metadata preservation

### Batch Processing Scripts
- `simple_batch_processor.py` - Simple batch processor for thousands of photos
- `batch_process_photos.py` - Advanced batch processor with optimizations
- `demo_batch.py` - Interactive demo and testing script
- `organize_by_camera.py` - **UPDATED!** Organize photos by camera from ZIP files with auto-conversion

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

### Multi-Format Media Support
- **HEIC Support**: Converts Apple HEIC format to JPEG with metadata preservation
- **Video Support (NEW!)**: Extracts frames from MOV/MP4 (iPhone Live Photos) to JPEG
- **Smart Conversion**: Automatically detects and converts HEIC/MOV/MP4 files
- **Space Optimization**: Deletes large originals after successful conversion
- **Metadata Preservation**: GPS, timestamps, camera info preserved across formats

### Natural Language Descriptions
- Generates storytelling-style descriptions
- Combines visual analysis with metadata context
- Avoids technical jargon, focuses on human-readable narratives
- Supports up to 1000 tokens for detailed descriptions

### Batch Processing for Thousands of Photos
- **Simple Processor**: Load model once, process many images sequentially
- **Advanced Processor**: Memory optimization, batch inference, parallel pipeline
- **EXIF Integration**: Writes AI descriptions directly to original image metadata
- **Memory Management**: Smart image resizing, temp files in memory only
- **Progress Tracking**: SQLite database, resume capability, real-time ETA
- **Quality Control**: Description validation, automatic retry for poor quality
- **Performance**: 4-6 hours for thousands of photos on RTX 2070

### Photo Organization by Camera
- **ZIP Processing**: Extract and organize photos from multiple ZIP archives
- **Camera Detection**: Automatically identifies camera make/model from EXIF data
- **Auto-Conversion**: Converts HEIC/MOV/MP4 to space-efficient JPEG format
- **Folder Organization**: Creates camera-specific folders (e.g., "Canon EOS R5", "iPhone 13 Pro")
- **Metadata Preservation**: Saves detailed metadata alongside organized photos
- **Space Savings**: 90%+ space reduction by converting videos to key frames

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
- **Gemma-3-4B-IT**: Works well on RTX 2070 (8GB) with full GPU acceleration
- **Qwen2.5-VL-3B-Instruct**: Successfully runs on RTX 2070 with INT4 quantization + image resizing
- **High-end GPUs** (12GB+): Can run both models with quantization for optimal performance

### Performance Tips
- **RTX 2070 Users**: Both models now work on GPU with optimization
  - Gemma: Direct GPU acceleration, fastest performance
  - Qwen: INT4 quantization + automatic image resizing for memory efficiency
- CPU inference is supported but significantly slower
- First model download requires internet connection and time
- Large images (>2M pixels) are automatically resized to fit GPU memory
- HEIC files require `pillow-heif` for processing
- MOV/MP4 processing requires `opencv-python` for frame extraction
- Video metadata extraction uses `ffprobe` when available (graceful fallback without it)
- Metadata extraction works with most JPEG/HEIC/MOV/MP4 files with EXIF data

### Memory Management
- **Smart Image Resizing**: Automatically reduces large images while preserving EXIF metadata
- **INT4 Quantization**: Reduces Qwen model memory usage to ~3-4GB on RTX 2070
- Automatic GPU memory detection and clearing
- PyTorch memory allocator optimization (`expandable_segments:True`)
- `bitsandbytes` integration for memory optimization
- Intelligent fallback strategies for different GPU configurations