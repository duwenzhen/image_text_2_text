#!/usr/bin/env python3
"""
Convert HEIC files to JPEG while preserving metadata.
"""

from PIL import Image
from PIL.ExifTags import TAGS
import pillow_heif
import os
import json

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

def convert_heic_to_jpeg(heic_path, jpeg_path=None, quality=95):
    """
    Convert HEIC file to JPEG while preserving all metadata.
    
    Args:
        heic_path: Path to the HEIC file
        jpeg_path: Output path for JPEG (optional, will auto-generate if None)
        quality: JPEG quality (1-100, default 95 for high quality)
    
    Returns:
        str: Path to the converted JPEG file
    """
    if jpeg_path is None:
        # Auto-generate output filename
        base_name = os.path.splitext(heic_path)[0]
        jpeg_path = f"{base_name}.jpg"
    
    try:
        # Open HEIC image
        print(f"Opening HEIC file: {heic_path}")
        with Image.open(heic_path) as img:
            # Print image info
            print(f"Image mode: {img.mode}")
            print(f"Image size: {img.size}")
            print(f"Image format: {img.format}")
            
            # Extract and display EXIF data
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
                    
                print(f"Found {len(exif_data)} EXIF tags")
            
            # Convert to RGB if needed (HEIC might be in different color space)
            if img.mode != 'RGB':
                print(f"Converting from {img.mode} to RGB")
                img = img.convert('RGB')
            
            # Save as JPEG with metadata preserved
            print(f"Saving as JPEG: {jpeg_path}")
            
            # Prepare save parameters
            save_params = {
                'format': 'JPEG',
                'quality': quality,
                'optimize': True
            }
            
            # Preserve EXIF data if it exists
            if hasattr(img, 'info') and 'exif' in img.info:
                save_params['exif'] = img.info['exif']
                print("Preserving EXIF data")
            
            img.save(jpeg_path, **save_params)
            
            print(f"✓ Successfully converted to: {jpeg_path}")
            
            # Verify the conversion by opening the JPEG
            with Image.open(jpeg_path) as jpeg_img:
                print(f"JPEG size: {jpeg_img.size}")
                print(f"JPEG mode: {jpeg_img.mode}")
                
                # Check if EXIF was preserved
                if hasattr(jpeg_img, '_getexif') and jpeg_img._getexif() is not None:
                    jpeg_exif = jpeg_img._getexif()
                    print(f"JPEG EXIF tags preserved: {len(jpeg_exif) if jpeg_exif else 0}")
                
            return jpeg_path
            
    except Exception as e:
        print(f"Error converting HEIC to JPEG: {e}")
        raise

def extract_metadata_to_json(image_path, json_path=None):
    """
    Extract all metadata from an image and save to JSON file.
    
    Args:
        image_path: Path to the image file
        json_path: Output path for JSON (optional, will auto-generate if None)
    
    Returns:
        dict: Metadata dictionary
    """
    if json_path is None:
        base_name = os.path.splitext(image_path)[0]
        json_path = f"{base_name}_metadata.json"
    
    metadata = {}
    
    try:
        with Image.open(image_path) as img:
            # Basic image info
            metadata['basic'] = {
                'filename': os.path.basename(image_path),
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height
            }
            
            # EXIF data
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = img._getexif()
                exif_readable = {}
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    # Convert bytes to string if needed
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except UnicodeDecodeError:
                            value = str(value)
                    exif_readable[tag] = value
                
                metadata['exif'] = exif_readable
            
            # Other metadata from img.info
            info_data = {}
            for key, value in img.info.items():
                if key != 'exif':  # Already handled above
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except UnicodeDecodeError:
                            value = str(value)
                    info_data[key] = value
            
            if info_data:
                metadata['info'] = info_data
        
        # Save metadata to JSON
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Metadata saved to: {json_path}")
        return metadata
        
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        raise

def main():
    """Test the conversion with the HEIC file."""
    heic_file = "./data/IMG_1854.heic"
    
    if not os.path.exists(heic_file):
        print(f"HEIC file not found: {heic_file}")
        print("Please make sure the file exists at the specified path.")
        return
    
    print("=== HEIC to JPEG Conversion ===\n")
    
    try:
        # Extract metadata first
        print("Extracting metadata...")
        metadata = extract_metadata_to_json(heic_file)
        
        # Convert to JPEG
        print("\nConverting HEIC to JPEG...")
        jpeg_file = convert_heic_to_jpeg(heic_file, quality=95)
        
        print(f"\n✓ Conversion completed successfully!")
        print(f"Original HEIC: {heic_file}")
        print(f"Converted JPEG: {jpeg_file}")
        print(f"Metadata JSON: {os.path.splitext(heic_file)[0]}_metadata.json")
        
        # Display some key metadata if available
        if 'exif' in metadata and metadata['exif']:
            print("\nKey EXIF data:")
            key_tags = ['Make', 'Model', 'DateTime', 'GPS']
            for tag in key_tags:
                if tag in metadata['exif']:
                    print(f"  {tag}: {metadata['exif'][tag]}")
        
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    main()