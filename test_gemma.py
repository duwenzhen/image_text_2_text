from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests
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
            
            # Basic image info
            width, height = img.size
            megapixels = round((width * height) / 1000000, 1)
            metadata_parts.append(f"Image: {width}×{height} pixels ({megapixels}MP)")
            
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
                    metadata_parts.append(f"Camera: {' '.join(camera_info)}")
                
                # Shooting settings
                settings = []
                if 'FNumber' in exif_data:
                    try:
                        f_num = exif_data['FNumber']
                        if isinstance(f_num, tuple) and f_num[1] != 0:
                            f_stop = f_num[0] / f_num[1]
                            settings.append(f"f/{f_stop:.1f}")
                    except:
                        pass
                
                if 'ExposureTime' in exif_data:
                    try:
                        exp_time = exif_data['ExposureTime']
                        if isinstance(exp_time, tuple) and exp_time[0] != 0:
                            shutter = f"1/{int(exp_time[1]/exp_time[0])}" if exp_time[0] == 1 else f"{exp_time[0]}/{exp_time[1]}"
                            settings.append(f"{shutter}s")
                    except:
                        pass
                
                if 'ISOSpeedRatings' in exif_data:
                    settings.append(f"ISO {exif_data['ISOSpeedRatings']}")
                
                if 'FocalLength' in exif_data:
                    try:
                        focal = exif_data['FocalLength']
                        if isinstance(focal, tuple) and focal[1] != 0:
                            focal_mm = focal[0] / focal[1]
                            settings.append(f"{focal_mm:.1f}mm")
                    except:
                        pass
                
                if settings:
                    metadata_parts.append(f"Settings: {', '.join(settings)}")
                
                # Date and time
                if 'DateTimeOriginal' in exif_data:
                    try:
                        dt_str = exif_data['DateTimeOriginal']
                        dt = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
                        formatted_time = dt.strftime('%B %d, %Y at %I:%M %p')
                        metadata_parts.append(f"Taken: {formatted_time}")
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
                            location_hint = "United Arab Emirates region"
                        elif 24 <= lat <= 26 and 54 <= lon <= 57:
                            location_hint = "Dubai/Abu Dhabi area, UAE"
                        
                        metadata_parts.append(f"Location: {lat:.4f}°, {lon:.4f}° ({location_hint})")
            
            if metadata_parts:
                return "\n".join(["Photo metadata:"] + [f"• {part}" for part in metadata_parts])
            else:
                return "No metadata available for this image."
    
    except Exception as e:
        return f"Could not extract metadata: {str(e)}"

# Extract metadata for the image
image_path = "./data/IMG_1854.jpg"
metadata_info = extract_image_metadata_for_prompt(image_path)

model_id = "google/gemma-3-4b-it"

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto",cache_dir="/media/wenzhen/SSD1T/huggingface_models",
    token=hf_token
).eval()

processor = AutoProcessor.from_pretrained(model_id, cache_dir="/media/wenzhen/SSD1T/huggingface_models",
    token=hf_token)

# Create enhanced prompt with metadata
enhanced_prompt = f"""
        AI Prompt Template for Image Description Generation
Role Definition:
You are an expert image analyst and descriptive cataloger. Your primary function is to transform visual information and technical metadata into rich, detailed, and natural language narratives suitable for high-quality semantic search within a vector database.

Core Objective:
Analyze the provided image and its accompanying metadata. Generate a comprehensive, precise description of the image that seamlessly integrates relevant contextual information from the metadata. The final description must be in natural, flowing prose, avoiding lists or technical jargon in the final output.

Input Data:

Image: {image_path}

Metadata: {metadata_info}

Processing Instructions:

1. Primary Visual Analysis (Content and Composition):

Subject Identification: Identify all primary and secondary subjects (e.g., people, animals, objects). Describe their specific attributes, appearance, posture, expressions, and any actions they are performing.

Environmental Context: Detail the setting and environment. Include information on location (e.g., urban street, forest interior, beach), weather conditions, and time of day (e.g., bright midday sun, twilight, overcast).

Composition and Perspective: Describe the camera's perspective (e.g., eye-level, low-angle shot, aerial view) and the arrangement of elements within the frame. Note significant foreground and background details.

2. Metadata Interpretation and Integration:

Relevance Assessment: Critically evaluate the provided metadata. Determine which elements add meaningful context to the visual description (e.g., specific date, location name, time of day).

Natural Integration: Weave the relevant metadata naturally into the descriptive narrative.

Example 1 (Location): Instead of writing "Metadata location: Paris," integrate it as: "The scene captures a bustling Parisian street corner..."

Example 2 (Time): If the timestamp indicates evening and the image confirms it, describe it as: "As evening approaches, the city lights begin to cast long shadows..."

Example 3 (Technical Interpretation): Do not list "Aperture f/1.8." Instead, describe the effect: "The low aperture setting creates a soft, blurred background (bokeh), drawing sharp focus to the subject's face."

3. Output Generation (Style and Format):

Tone: Objective, descriptive, and highly detailed.

Language: Natural human language, written in complete sentences and paragraphs.

Exclusions: Do not include a separate list of metadata keys and values in the final output. Omit technical metadata entirely if it provides no interpretable value to the visual description (e.g., camera serial number or software version). The goal is a purely narrative description.

Final Output Request:

Provide a single, coherent descriptive paragraph based on the analysis above.
        """

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert photo analyst who can provide detailed descriptions combining visual analysis with technical metadata."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./data/IMG_1854.jpg"},
            {"type": "text", "text": enhanced_prompt}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

print("Generated prompt with metadata:")
print("-" * 50)
print(enhanced_prompt)
print("-" * 50)
print("\nGenerating AI response...\n")

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=1000, do_sample=True, temperature=0.7)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
