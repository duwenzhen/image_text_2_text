#!/usr/bin/env python3
"""
Organize photos and videos from zip files by camera type based on metadata.
This script processes multiple zip files, extracts photos and videos, reads their metadata,
and organizes them into folders by camera model.
"""

import os
import zipfile
import shutil
from pathlib import Path
import json
from datetime import datetime
import tempfile
import argparse
from PIL import Image
from PIL.ExifTags import TAGS

# Import existing metadata extraction functions
from extract_metadata import analyze_image_metadata, extract_camera_info
from convert_media import convert_media_file, extract_comprehensive_metadata

def get_video_metadata(video_path):
    """
    Extract comprehensive metadata from video files using the new convert_media functionality.

    Args:
        video_path: Path to the video file

    Returns:
        dict: Comprehensive video metadata including camera info if available
    """
    try:
        # Use the new comprehensive metadata extraction
        return extract_comprehensive_metadata(video_path)

    except Exception as e:
        # Fallback to basic file info if comprehensive extraction fails
        return {
            'file_info': {
                'filename': os.path.basename(video_path),
                'error': str(e),
                'file_type': 'video'
            },
            'camera_info': {},
            'basic_info': {'format': 'video', 'type': 'video'}
        }

def extract_camera_from_metadata(metadata):
    """
    Extract camera make and model from metadata for folder naming.
    
    Args:
        metadata: Metadata dictionary from analyze_image_metadata or get_video_metadata
        
    Returns:
        str: Clean camera name for folder creation
    """
    camera_info = metadata.get('camera_info', {})
    
    # For images with proper EXIF data
    make = None
    model = None
    
    if 'Make' in camera_info and 'value' in camera_info['Make']:
        make = str(camera_info['Make']['value']).strip()
    if 'Model' in camera_info and 'value' in camera_info['Model']:
        model = str(camera_info['Model']['value']).strip()
    
    # For videos or when EXIF data is limited
    if not make and 'probable_make' in camera_info:
        make = camera_info['probable_make']
    
    # Create a clean folder name
    if make and model:
        # Remove make from model if it's redundant (e.g., "Canon" in "Canon EOS 5D")
        if make.upper() in model.upper():
            folder_name = model
        else:
            folder_name = f"{make} {model}"
    elif make:
        folder_name = make
    elif model:
        folder_name = model
    else:
        folder_name = "Unknown_Camera"
    
    # Clean up the folder name for filesystem compatibility
    folder_name = folder_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    folder_name = ''.join(c for c in folder_name if c.isalnum() or c in (' ', '-', '_')).strip()
    folder_name = folder_name.replace('  ', ' ')  # Remove double spaces
    
    return folder_name if folder_name else "Unknown_Camera"

def is_image_file(filename):
    """Check if file is a supported image format."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.heic', '.webp'}
    return Path(filename).suffix.lower() in image_extensions

def is_video_file(filename):
    """Check if file is a supported video format."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mts', '.m2ts'}
    return Path(filename).suffix.lower() in video_extensions

def process_media_file(file_path, output_base_dir, stats):
    """
    Process a single media file (image or video) and organize it by camera type.
    Automatically converts HEIC/MOV/MP4 files to JPEG to save space.

    Args:
        file_path: Path to the media file
        output_base_dir: Base directory for organized output
        stats: Dictionary to track processing statistics

    Returns:
        bool: True if processed successfully, False otherwise
    """
    try:
        filename = os.path.basename(file_path)
        original_path = file_path

        # Check if this is a file that needs conversion (HEIC, MOV, MP4)
        file_ext = Path(filename).suffix.lower()
        needs_conversion = file_ext in {'.heic', '.mov', '.mp4'}

        if needs_conversion:
            print(f"   Converting {filename} to JPEG...")
            # Convert the file to JPEG and extract metadata
            try:
                jpeg_path, metadata_json_path = convert_media_file(
                    file_path,
                    delete_original=False  # Don't delete original yet, we'll handle it after organizing
                )

                # Update file path to point to converted JPEG
                file_path = jpeg_path
                filename = os.path.basename(jpeg_path)

                # Read the metadata from the JSON file created during conversion
                with open(metadata_json_path, 'r') as f:
                    metadata = json.load(f)

                # Clean up the metadata JSON file as we'll create our own
                os.remove(metadata_json_path)

                stats['files_converted'] += 1

            except Exception as e:
                print(f"   Error converting {filename}: {str(e)}")
                stats['files_error'] += 1
                return False
        else:
            # Handle regular image/video files
            if is_image_file(filename):
                metadata = analyze_image_metadata(file_path)
                stats['images_processed'] += 1
            elif is_video_file(filename):
                metadata = get_video_metadata(file_path)
                stats['videos_processed'] += 1
            else:
                print(f"   Skipping unsupported file: {filename}")
                stats['files_skipped'] += 1
                return False

        # Extract camera information for folder naming
        camera_folder = extract_camera_from_metadata(metadata)

        # Create camera-specific directory
        camera_dir = Path(output_base_dir) / camera_folder
        camera_dir.mkdir(parents=True, exist_ok=True)

        # Create destination path
        dest_path = camera_dir / filename

        # Handle filename conflicts
        counter = 1
        original_dest = dest_path
        while dest_path.exists():
            stem = original_dest.stem
            suffix = original_dest.suffix
            dest_path = original_dest.parent / f"{stem}_{counter}{suffix}"
            counter += 1

        # Move the file (converted or original)
        shutil.move(file_path, dest_path)

        # If we converted the file, delete the original now
        if needs_conversion and original_path != file_path:
            try:
                os.remove(original_path)
                print(f"   ✓ Original {Path(original_path).suffix.upper()[1:]} file deleted")
            except OSError as e:
                print(f"   ⚠️ Could not delete original {original_path}: {e}")

        # Save metadata alongside the file
        metadata_file = dest_path.with_suffix(dest_path.suffix + '.metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"   → {camera_folder}/{dest_path.name}")
        stats['files_organized'] += 1

        # Track camera types
        if camera_folder not in stats['cameras_found']:
            stats['cameras_found'][camera_folder] = 0
        stats['cameras_found'][camera_folder] += 1

        return True

    except Exception as e:
        filename = os.path.basename(file_path) if 'filename' not in locals() else filename
        print(f"   Error processing {filename}: {str(e)}")
        stats['files_error'] += 1
        return False

def process_zip_file(zip_path, output_base_dir, stats):
    """
    Process a single zip file, extracting and organizing its media files.
    
    Args:
        zip_path: Path to the zip file
        output_base_dir: Base directory for organized output
        stats: Dictionary to track processing statistics
    """
    zip_name = Path(zip_path).stem
    print(f"\n📦 Processing: {zip_name}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in the zip
            file_list = zip_ref.namelist()
            media_files = [f for f in file_list if is_image_file(f) or is_video_file(f)]
            
            if not media_files:
                print(f"   No media files found in {zip_name}")
                stats['empty_zips'] += 1
                return
            
            print(f"   Found {len(media_files)} media files")
            
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract media files
                for file_path in media_files:
                    try:
                        # Extract file to temporary directory
                        zip_ref.extract(file_path, temp_dir)
                        extracted_path = os.path.join(temp_dir, file_path)
                        
                        # Process the extracted file
                        process_media_file(extracted_path, output_base_dir, stats)
                        
                    except Exception as e:
                        print(f"   Error extracting {file_path}: {str(e)}")
                        stats['files_error'] += 1
            
    except Exception as e:
        print(f"   Error processing zip file {zip_name}: {str(e)}")
        stats['zip_errors'] += 1

def main():
    """Main function to organize media files from zip archives by camera type."""
    parser = argparse.ArgumentParser(
        description="Organize photos and videos from zip files by camera type based on metadata"
    )
    parser.add_argument(
        "zip_folder",
        help="Path to folder containing zip files"
    )
    parser.add_argument(
        "-o", "--output",
        default="./organized_media",
        help="Output directory for organized files (default: ./organized_media)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually organizing files"
    )
    
    args = parser.parse_args()
    
    zip_folder = Path(args.zip_folder)
    output_dir = Path(args.output)
    
    # Validate input directory
    if not zip_folder.exists() or not zip_folder.is_dir():
        print(f"Error: Zip folder '{zip_folder}' does not exist or is not a directory")
        return 1
    
    # Find all zip files
    zip_files = list(zip_folder.glob("*.zip"))
    if not zip_files:
        print(f"No zip files found in '{zip_folder}'")
        return 1
    
    print(f"📁 Input directory: {zip_folder}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📦 Found {len(zip_files)} zip files")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be moved")
    
    # Create output directory
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize statistics
    stats = {
        'zip_files_processed': 0,
        'empty_zips': 0,
        'zip_errors': 0,
        'images_processed': 0,
        'videos_processed': 0,
        'files_converted': 0,  # New: track HEIC/MOV/MP4 conversions
        'files_organized': 0,
        'files_skipped': 0,
        'files_error': 0,
        'cameras_found': {}
    }
    
    # Process each zip file
    start_time = datetime.now()
    
    for zip_path in zip_files:
        if not args.dry_run:
            process_zip_file(zip_path, output_dir, stats)
        else:
            # In dry-run mode, just show what would be processed
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    media_files = [f for f in file_list if is_image_file(f) or is_video_file(f)]
                    print(f"\n📦 Would process: {zip_path.name} ({len(media_files)} media files)")
            except Exception as e:
                print(f"\n📦 Error reading: {zip_path.name} - {str(e)}")
        
        stats['zip_files_processed'] += 1
    
    # Print final statistics
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("📊 ORGANIZATION COMPLETE")
    print("="*60)
    print(f"⏱️  Processing time: {duration}")
    print(f"📦 Zip files processed: {stats['zip_files_processed']}")
    print(f"🖼️  Images processed: {stats['images_processed']}")
    print(f"🎥 Videos processed: {stats['videos_processed']}")
    print(f"🔄 Files converted (HEIC/MOV/MP4→JPEG): {stats['files_converted']}")
    print(f"✅ Files organized: {stats['files_organized']}")
    print(f"⚠️  Files skipped: {stats['files_skipped']}")
    print(f"❌ Files with errors: {stats['files_error']}")
    
    if stats['cameras_found']:
        print(f"\n📷 Camera types found:")
        for camera, count in sorted(stats['cameras_found'].items()):
            print(f"   {camera}: {count} files")
    
    if not args.dry_run and stats['files_organized'] > 0:
        print(f"\n📂 Organized files are available in: {output_dir}")
        
        # Create a summary report
        report_file = output_dir / "organization_report.json"
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': str(zip_folder),
            'output_directory': str(output_dir),
            'processing_duration': str(duration),
            'statistics': stats
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved to: {report_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())