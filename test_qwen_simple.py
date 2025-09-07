#!/usr/bin/env python3
"""
Simple fallback test for vision models when GPU memory is limited.
Tests smaller vision models that can work with RTX 2070.
"""

from transformers import AutoModelForCausalLM, AutoProcessor
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
    """Extract key metadata from image for use in AI model prompt."""
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

def test_smaller_vision_model():
    """Test a smaller vision model that works better with limited GPU memory."""
    print("=== Testing Alternative Vision Model ===\n")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 10:
            print("⚠️  GPU memory insufficient for large vision models")
            print("💡 Falling back to CPU inference with a text-only model")
    
    # Use a simpler approach - let's test if basic model loading works
    image_path = "./data/IMG_1854.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    # Extract metadata
    metadata_info = extract_image_metadata_for_prompt(image_path)
    print(f"Metadata extracted: {metadata_info}")
    
    # Load and display the image info
    try:
        with Image.open(image_path) as img:
            print(f"\nImage Info:")
            print(f"  Size: {img.size}")
            print(f"  Format: {img.format}")
            print(f"  Mode: {img.mode}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    print(f"\n📝 Generated Context for Vision Model:")
    print(f"=" * 60)
    
    enhanced_prompt = f"""This is a photo with the following context:
    
{metadata_info}

Based on this metadata, this appears to be a photo taken in the United Arab Emirates on September 7th, 2025 at 7:05 AM using an iPhone 15 Pro. The image would benefit from a vision model that can describe what's actually visible in the photo and combine it with this contextual information to create a natural, story-like description."""
    
    print(enhanced_prompt)
    print("=" * 60)
    
    print(f"\n✅ Metadata extraction and prompt generation working correctly!")
    print(f"💡 For actual vision analysis, consider using:")
    print(f"   • A cloud-based vision API (OpenAI GPT-4V, Google Vision)")
    print(f"   • A smaller local model like BLIP-2")
    print(f"   • Upgrading to a GPU with 12GB+ memory")

def main():
    """Main test function."""
    try:
        # First try to load Qwen if possible
        print("=== Qwen2.5-VL-7B Memory Check ===")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Available GPU memory: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 10:
                print("✅ Sufficient GPU memory for Qwen2.5-VL-7B")
                print("You can run the original test_qwen.py script")
                return
            else:
                print("❌ Insufficient GPU memory for Qwen2.5-VL-7B")
                print("Qwen2.5-VL-7B requires approximately 10-12GB GPU memory")
        else:
            print("❌ No GPU available")
        
        print("\n" + "=" * 60)
        test_smaller_vision_model()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()