#!/usr/bin/env python3
"""
Test script for Qwen2.5-VL-7B-Instruct vision-language model with metadata integration.
"""

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import torch
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN')

def extract_image_metadata_for_prompt(image_path):
    """
    Extract key metadata from image for use in AI model prompt.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        str: Formatted metadata string for prompt
    """
    try:
        with Image.open(image_path) as img:
            metadata_parts = []
            
            # Extract EXIF data
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif_raw = img._getexif()
                exif_data = {}
                gps_info = {}
                
                for tag_id, value in exif_raw.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'GPSInfo':
                        gps_info = value
                    else:
                        exif_data[tag] = value
                
                # Camera information
                camera_info = []
                if 'Make' in exif_data:
                    camera_info.append(exif_data['Make'])
                if 'Model' in exif_data:
                    camera_info.append(exif_data['Model'])
                if camera_info:
                    metadata_parts.append(f"Captured with: {' '.join(camera_info)}")
                
                # Date and time
                if 'DateTimeOriginal' in exif_data:
                    try:
                        dt_str = exif_data['DateTimeOriginal']
                        dt = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
                        formatted_time = dt.strftime('%B %d, %Y at %I:%M %p')
                        metadata_parts.append(f"Taken on: {formatted_time}")
                    except:
                        pass
                
                # GPS information
                if gps_info:
                    gps_readable = {}
                    for key, value in gps_info.items():
                        tag_name = GPSTAGS.get(key, key)
                        gps_readable[tag_name] = value
                    
                    # Extract coordinates
                    lat, lon = None, None
                    if 'GPSLatitude' in gps_readable and 'GPSLatitudeRef' in gps_readable:
                        try:
                            lat_dms = gps_readable['GPSLatitude']
                            lat_ref = gps_readable['GPSLatitudeRef']
                            lat = float(lat_dms[0]) + (float(lat_dms[1]) / 60.0) + (float(lat_dms[2]) / 3600.0)
                            if lat_ref == 'S':
                                lat = -lat
                        except:
                            pass
                    
                    if 'GPSLongitude' in gps_readable and 'GPSLongitudeRef' in gps_readable:
                        try:
                            lon_dms = gps_readable['GPSLongitude']
                            lon_ref = gps_readable['GPSLongitudeRef']
                            lon = float(lon_dms[0]) + (float(lon_dms[1]) / 60.0) + (float(lon_dms[2]) / 3600.0)
                            if lon_ref == 'W':
                                lon = -lon
                        except:
                            pass
                    
                    if lat is not None and lon is not None:
                        # Determine general location based on coordinates
                        location_hint = "Unknown location"
                        if 22 <= lat <= 26 and 51 <= lon <= 57:
                            location_hint = "United Arab Emirates"
                        elif 24 <= lat <= 26 and 54 <= lon <= 57:
                            location_hint = "Dubai/Abu Dhabi area, UAE"
                        
                        metadata_parts.append(f"Location: {location_hint}")
            
            if metadata_parts:
                return " | ".join(metadata_parts)
            else:
                return "No metadata available"
    
    except Exception as e:
        return f"Could not extract metadata: {str(e)}"

def clear_gpu_memory():
    """Clear GPU memory cache to free up space."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def main():
    """Test the Qwen2.5-VL model with metadata-enhanced prompts."""
    print("=== Qwen2.5-VL-3B-Instruct Test ===\n")
    
    # Clear any existing GPU memory
    clear_gpu_memory()
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_free = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU memory: {gpu_total:.1f} GB total, {gpu_free:.1f} GB free")
    print()
    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    cache_dir = "/media/wenzhen/SSD1T/huggingface_models"
    image_path = "./data/IMG_1854.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    try:
        print(f"Loading Qwen2.5-VL model: {model_id}")
        print("This may take several minutes...")
        
        # Load the model with aggressive memory optimization for RTX 2070
        if torch.cuda.is_available():
            gpu_free = torch.cuda.mem_get_info()[0] / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU memory available: {gpu_free:.1f} GB / {gpu_total:.1f} GB")
            
            # RTX 2070 with 7.8GB needs very aggressive optimization
            if gpu_total < 10:  # RTX 2070 case
                print("🔧 RTX 2070 detected: Using CPU inference for reliability")
                print("💡 Vision models are very memory-intensive, CPU will be slower but stable")
                quantization_config = None
                device_map = "cpu"
                torch_dtype = torch.float32
                use_quantization = False
            else:
                print("✅ High-end GPU detected, using GPU with quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                device_map = "auto"
                torch_dtype = None
                use_quantization = True
        else:
            print("⚠️  No GPU available, using CPU inference")
            quantization_config = None
            device_map = "cpu"
            torch_dtype = torch.float32
            use_quantization = False
        
        # Load model with appropriate settings
        model_kwargs = {
            "cache_dir": cache_dir,
            "token": hf_token,
            "low_cpu_mem_usage": True,
            "device_map": device_map
        }
        
        if use_quantization:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype
            
        model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=hf_token
        )
        
        print("✓ Model and processor loaded successfully")
        if use_quantization:
            print("📊 Model loaded with 4-bit quantization")
        
        # Get model device info
        try:
            device_info = next(model.parameters()).device
            print(f"Model device: {device_info}")
        except:
            print("Model device: Distributed across multiple devices")
        
        # Extract metadata
        metadata_info = extract_image_metadata_for_prompt(image_path)
        print(f"Metadata extracted: {metadata_info}")
        
        # Create enhanced prompt
        enhanced_prompt = f"""Please describe this image as if you were telling someone about a memorable photo. Combine what you can see in the image with the context provided to create a natural, flowing description. 

Context: {metadata_info}

Describe this scene naturally, mentioning where and when it was taken, what's happening in the image, and paint a picture that brings this moment to life. Be detailed and descriptive, but write it as a human would naturally describe a photo to a friend. Don't focus on technical details - instead tell the story of this moment."""
        
        # Prepare messages in Qwen format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": enhanced_prompt},
                ],
            }
        ]
        
        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device if using GPU
        if device_map != "cpu":
            inputs = inputs.to(model.device)
        
        print("\n" + "="*60)
        print("PROMPT SENT TO MODEL:")
        print("="*60)
        print(enhanced_prompt)
        print("="*60)
        print("\nGenerating response...\n")
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print("="*60)
        print("QWEN2.5-VL MODEL RESPONSE:")
        print("="*60)
        print(output_text[0])
        print("="*60)
        
    except Exception as e:
        print(f"Error running Qwen model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()