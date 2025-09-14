#!/usr/bin/env python3
"""
Convert HEIC, MOV, and MP4 files to JPEG with metadata preservation.
Supports iPhone Live Photos (MOV/MP4) by extracting the best frame.
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Image processing imports
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pillow_heif

# Video processing imports
import cv2
import piexif
import numpy as np

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

def get_file_type(file_path):
    """
    Determine the file type based on extension.

    Args:
        file_path: Path to the file

    Returns:
        str: 'heic', 'video', or 'unknown'
    """
    ext = Path(file_path).suffix.lower()

    if ext == '.heic':
        return 'heic'
    elif ext in ['.mov', '.mp4']:
        return 'video'
    else:
        return 'unknown'

def extract_video_frame(video_path, output_path=None, frame_number=0):
    """
    Extract a specific frame from a video file (MOV/MP4).

    Args:
        video_path: Path to the video file
        output_path: Output path for the frame (optional, will auto-generate if None)
        frame_number: Frame number to extract (0 = first frame, -1 = middle frame)

    Returns:
        str: Path to the extracted frame
    """
    if output_path is None:
        base_name = os.path.splitext(video_path)[0]
        output_path = f"{base_name}_frame.jpg"

    print(f"Extracting frame from video: {video_path}")

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        print(f"Video properties: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")

        # Determine which frame to extract
        if frame_number == -1:
            # Extract middle frame
            target_frame = total_frames // 2
        else:
            target_frame = min(frame_number, total_frames - 1)

        print(f"Extracting frame {target_frame}")

        # Set video position to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        # Read the frame
        ret, frame = cap.read()

        if not ret:
            raise Exception(f"Could not read frame {target_frame} from video")

        # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Save as JPEG
        pil_image.save(output_path, 'JPEG', quality=95, optimize=True)

        print(f"✓ Frame extracted to: {output_path}")

        return output_path

    finally:
        cap.release()

def extract_video_metadata_ffprobe(video_path):
    """
    Extract metadata from video file using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        dict: Video metadata
    """
    try:
        # Run ffprobe to get metadata in JSON format
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)

        return metadata

    except subprocess.CalledProcessError as e:
        print(f"ffprobe failed: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Could not parse ffprobe output: {e}")
        return {}
    except FileNotFoundError:
        print("ffprobe not found. Install ffmpeg to extract video metadata.")
        return {}

def convert_video_metadata_to_exif(video_metadata, video_path):
    """
    Convert video metadata to EXIF-like format for consistency.

    Args:
        video_metadata: Video metadata from ffprobe
        video_path: Path to the video file

    Returns:
        dict: EXIF-like metadata
    """
    exif_like = {}

    format_info = video_metadata.get('format', {})
    streams = video_metadata.get('streams', [])

    # Find video stream
    video_stream = None
    for stream in streams:
        if stream.get('codec_type') == 'video':
            video_stream = stream
            break

    if video_stream:
        # Basic video properties
        exif_like['ImageWidth'] = video_stream.get('width')
        exif_like['ImageLength'] = video_stream.get('height')  # ImageLength is EXIF term for height
        exif_like['VideoFrameRate'] = video_stream.get('r_frame_rate', '').split('/')[0] if '/' in str(video_stream.get('r_frame_rate', '')) else video_stream.get('r_frame_rate')
        exif_like['VideoCodec'] = video_stream.get('codec_name')
        exif_like['VideoDuration'] = float(video_stream.get('duration', 0))

        # Creation time from stream
        if 'tags' in video_stream and 'creation_time' in video_stream['tags']:
            creation_time = video_stream['tags']['creation_time']
            try:
                # Parse ISO format datetime
                dt = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                exif_like['DateTime'] = dt.strftime('%Y:%m:%d %H:%M:%S')
                exif_like['DateTimeOriginal'] = dt.strftime('%Y:%m:%d %H:%M:%S')
            except ValueError:
                exif_like['DateTime'] = creation_time

    # Format-level metadata
    if format_info:
        exif_like['FileFormat'] = format_info.get('format_name')
        exif_like['FileSize'] = format_info.get('size')

        # Creation time from format
        if 'tags' in format_info and 'creation_time' in format_info['tags']:
            creation_time = format_info['tags']['creation_time']
            if 'DateTime' not in exif_like:  # Use format time if stream time not available
                try:
                    dt = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                    exif_like['DateTime'] = dt.strftime('%Y:%m:%d %H:%M:%S')
                    exif_like['DateTimeOriginal'] = dt.strftime('%Y:%m:%d %H:%M:%S')
                except ValueError:
                    exif_like['DateTime'] = creation_time

        # Look for location data (common in iPhone videos)
        if 'tags' in format_info:
            tags = format_info['tags']

            # iPhone location metadata
            if 'location' in tags:
                location_str = tags['location']
                # Parse location string like "+37.7749-122.4194+010.000/"
                try:
                    if location_str.startswith('+') or location_str.startswith('-'):
                        # Extract latitude and longitude
                        parts = location_str.replace('+', ' +').replace('-', ' -').split()
                        if len(parts) >= 2:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            exif_like['GPSLatitude'] = lat
                            exif_like['GPSLongitude'] = lon
                            exif_like['GPSLatitudeRef'] = 'N' if lat >= 0 else 'S'
                            exif_like['GPSLongitudeRef'] = 'E' if lon >= 0 else 'W'
                except (ValueError, IndexError):
                    pass

            # Other common metadata
            for key in ['make', 'model', 'software', 'artist', 'comment']:
                if key in tags:
                    exif_key = key.title().replace('_', '')
                    exif_like[exif_key] = tags[key]

    return exif_like

def create_exif_dict_for_transfer(exif_like_metadata):
    """
    Create a proper EXIF dictionary that can be embedded in JPEG.

    Args:
        exif_like_metadata: Dictionary with EXIF-like metadata

    Returns:
        dict: Proper EXIF dictionary for piexif
    """
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    # Map common tags to their EXIF IDs
    tag_mapping = {
        'ImageWidth': piexif.ImageIFD.ImageWidth,
        'ImageLength': piexif.ImageIFD.ImageLength,
        'Make': piexif.ImageIFD.Make,
        'Model': piexif.ImageIFD.Model,
        'Software': piexif.ImageIFD.Software,
        'DateTime': piexif.ImageIFD.DateTime,
        'DateTimeOriginal': piexif.ExifIFD.DateTimeOriginal,
    }

    # GPS tag mapping
    gps_mapping = {
        'GPSLatitude': piexif.GPSIFD.GPSLatitude,
        'GPSLatitudeRef': piexif.GPSIFD.GPSLatitudeRef,
        'GPSLongitude': piexif.GPSIFD.GPSLongitude,
        'GPSLongitudeRef': piexif.GPSIFD.GPSLongitudeRef,
    }

    # Process regular EXIF tags
    for key, value in exif_like_metadata.items():
        if key in tag_mapping and value is not None:
            tag_id = tag_mapping[key]

            # Convert to appropriate format
            if key in ['ImageWidth', 'ImageLength']:
                exif_dict["0th"][tag_id] = int(value)
            elif key in ['Make', 'Model', 'Software']:
                exif_dict["0th"][tag_id] = str(value)
            elif key in ['DateTime', 'DateTimeOriginal']:
                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = str(value)
                exif_dict["0th"][piexif.ImageIFD.DateTime] = str(value)

    # Process GPS tags
    for key, value in exif_like_metadata.items():
        if key in gps_mapping and value is not None:
            tag_id = gps_mapping[key]

            if key in ['GPSLatitudeRef', 'GPSLongitudeRef']:
                exif_dict["GPS"][tag_id] = str(value)
            elif key in ['GPSLatitude', 'GPSLongitude']:
                # Convert decimal degrees to DMS format for EXIF
                decimal_degrees = abs(float(value))
                degrees = int(decimal_degrees)
                minutes = int((decimal_degrees - degrees) * 60)
                seconds = ((decimal_degrees - degrees) * 60 - minutes) * 60

                # EXIF GPS coordinates are stored as tuples of (numerator, denominator)
                exif_dict["GPS"][tag_id] = [
                    (degrees, 1),
                    (minutes, 1),
                    (int(seconds * 1000000), 1000000)  # Store with microsecond precision
                ]

    return exif_dict

def convert_heic_to_jpeg(heic_path, jpeg_path=None, quality=95):
    """
    Convert HEIC file to JPEG while preserving all metadata.
    (Same as original convert_heic.py)
    """
    if jpeg_path is None:
        base_name = os.path.splitext(heic_path)[0]
        jpeg_path = f"{base_name}.jpg"

    try:
        print(f"Converting HEIC to JPEG: {heic_path}")
        with Image.open(heic_path) as img:
            print(f"Image size: {img.size}, mode: {img.mode}, format: {img.format}")

            # Convert to RGB if needed
            if img.mode != 'RGB':
                print(f"Converting from {img.mode} to RGB")
                img = img.convert('RGB')

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
            print(f"✓ HEIC converted to: {jpeg_path}")

            return jpeg_path

    except Exception as e:
        print(f"Error converting HEIC to JPEG: {e}")
        raise

def convert_video_to_jpeg(video_path, jpeg_path=None, quality=95, frame_number=0):
    """
    Convert video file (MOV/MP4) to JPEG by extracting a frame and transferring metadata.

    Args:
        video_path: Path to the video file
        jpeg_path: Output path for JPEG (optional, will auto-generate if None)
        quality: JPEG quality (1-100, default 95)
        frame_number: Frame to extract (0 = first, -1 = middle)

    Returns:
        str: Path to the converted JPEG file
    """
    if jpeg_path is None:
        base_name = os.path.splitext(video_path)[0]
        jpeg_path = f"{base_name}.jpg"

    try:
        print(f"Converting video to JPEG: {video_path}")

        # Extract frame from video
        temp_frame_path = f"{jpeg_path}.temp_frame.jpg"
        extract_video_frame(video_path, temp_frame_path, frame_number)

        # Extract video metadata
        video_metadata = extract_video_metadata_ffprobe(video_path)

        # Convert video metadata to EXIF-like format
        exif_like = convert_video_metadata_to_exif(video_metadata, video_path)

        # Load the extracted frame
        with Image.open(temp_frame_path) as img:
            print(f"Frame size: {img.size}, mode: {img.mode}")

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Create EXIF data with video metadata
            exif_dict = create_exif_dict_for_transfer(exif_like)
            exif_bytes = piexif.dump(exif_dict) if any(exif_dict.values()) else None

            # Save with embedded metadata
            save_params = {
                'format': 'JPEG',
                'quality': quality,
                'optimize': True
            }

            if exif_bytes:
                save_params['exif'] = exif_bytes
                print("Embedding video metadata as EXIF data")

            img.save(jpeg_path, **save_params)

            print(f"✓ Video converted to: {jpeg_path}")

        # Clean up temp frame
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)

        return jpeg_path

    except Exception as e:
        print(f"Error converting video to JPEG: {e}")
        # Clean up temp frame on error
        temp_frame_path = f"{jpeg_path}.temp_frame.jpg"
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
        raise

def extract_comprehensive_metadata(file_path):
    """
    Extract comprehensive metadata from HEIC, MOV, or MP4 files.

    Args:
        file_path: Path to the file

    Returns:
        dict: Complete metadata analysis
    """
    file_type = get_file_type(file_path)

    if file_type == 'heic':
        # Use existing PIL-based extraction for HEIC
        from extract_metadata import analyze_image_metadata
        return analyze_image_metadata(file_path)

    elif file_type == 'video':
        # Extract video metadata and format it similar to image metadata
        metadata = {
            'file_info': {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'file_path': os.path.abspath(file_path),
                'file_type': 'video'
            },
            'video_info': {},
            'gps_info': {},
            'datetime_info': {},
            'technical_info': {}
        }

        # Get video metadata using ffprobe
        video_metadata = extract_video_metadata_ffprobe(file_path)

        if video_metadata:
            format_info = video_metadata.get('format', {})
            streams = video_metadata.get('streams', [])

            # Find video stream
            video_stream = None
            for stream in streams:
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break

            if video_stream:
                metadata['video_info'] = {
                    'codec': video_stream.get('codec_name'),
                    'width': video_stream.get('width'),
                    'height': video_stream.get('height'),
                    'duration': float(video_stream.get('duration', 0)),
                    'frame_rate': video_stream.get('r_frame_rate'),
                    'bit_rate': video_stream.get('bit_rate'),
                    'pixel_format': video_stream.get('pix_fmt')
                }

            # Process datetime info
            creation_time = None
            if video_stream and 'tags' in video_stream and 'creation_time' in video_stream['tags']:
                creation_time = video_stream['tags']['creation_time']
            elif format_info and 'tags' in format_info and 'creation_time' in format_info['tags']:
                creation_time = format_info['tags']['creation_time']

            if creation_time:
                try:
                    dt = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                    metadata['datetime_info'] = {
                        'DateTimeOriginal': {
                            'raw': creation_time,
                            'parsed': dt.strftime('%Y-%m-%d %H:%M:%S'),
                            'formatted': dt.strftime('%B %d, %Y at %I:%M:%S %p'),
                            'description': 'Video creation time'
                        }
                    }
                except ValueError:
                    metadata['datetime_info'] = {
                        'DateTimeOriginal': {
                            'raw': creation_time,
                            'parsed': 'Could not parse',
                            'description': 'Video creation time'
                        }
                    }

            # Process GPS info (iPhone videos)
            if format_info and 'tags' in format_info and 'location' in format_info['tags']:
                location_str = format_info['tags']['location']
                try:
                    if location_str.startswith('+') or location_str.startswith('-'):
                        parts = location_str.replace('+', ' +').replace('-', ' -').split()
                        if len(parts) >= 2:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            metadata['gps_info'] = {
                                'latitude': lat,
                                'longitude': lon,
                                'latitude_ref': 'N' if lat >= 0 else 'S',
                                'longitude_ref': 'E' if lon >= 0 else 'W',
                                'google_maps_link': f"https://maps.google.com/?q={lat},{lon}",
                                'source': 'video_metadata'
                            }
                except (ValueError, IndexError):
                    pass

            # Technical info
            metadata['technical_info'] = {
                'format': format_info.get('format_name'),
                'file_size': format_info.get('size'),
                'all_streams': len(streams),
                'video_streams': len([s for s in streams if s.get('codec_type') == 'video']),
                'audio_streams': len([s for s in streams if s.get('codec_type') == 'audio'])
            }

        return metadata

    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def convert_media_file(file_path, output_path=None, quality=95, frame_number=0, delete_original=True):
    """
    Convert HEIC, MOV, or MP4 files to JPEG with metadata preservation.

    Args:
        file_path: Path to the input file
        output_path: Output path for JPEG (optional, will auto-generate if None)
        quality: JPEG quality (1-100, default 95)
        frame_number: For videos, which frame to extract (0 = first, -1 = middle)
        delete_original: Whether to delete the original file after conversion (default True)

    Returns:
        tuple: (jpeg_path, metadata_json_path)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = get_file_type(file_path)

    if file_type == 'unknown':
        raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")

    # Generate output paths
    if output_path is None:
        base_name = os.path.splitext(file_path)[0]
        output_path = f"{base_name}.jpg"

    metadata_json_path = f"{os.path.splitext(output_path)[0]}_metadata.json"

    print(f"=" * 60)
    print(f"CONVERTING {file_type.upper()} FILE TO JPEG")
    print(f"=" * 60)
    print(f"Input: {file_path}")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_json_path}")
    print()

    try:
        # Convert based on file type
        if file_type == 'heic':
            jpeg_path = convert_heic_to_jpeg(file_path, output_path, quality)
        elif file_type == 'video':
            jpeg_path = convert_video_to_jpeg(file_path, output_path, quality, frame_number)

        # Extract comprehensive metadata
        print("\nExtracting metadata...")
        metadata = extract_comprehensive_metadata(file_path)

        # Save metadata to JSON
        with open(metadata_json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"✓ Metadata saved to: {metadata_json_path}")

        print(f"\n{'='*60}")
        print("CONVERSION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"JPEG file: {jpeg_path}")
        print(f"Metadata file: {metadata_json_path}")

        # Print quick summary
        if file_type == 'video' and 'video_info' in metadata:
            video_info = metadata['video_info']
            print(f"\nVideo Summary:")
            print(f"  Resolution: {video_info.get('width')}x{video_info.get('height')}")
            print(f"  Duration: {video_info.get('duration', 0):.2f} seconds")
            print(f"  Codec: {video_info.get('codec')}")

        if 'gps_info' in metadata and metadata['gps_info']:
            gps = metadata['gps_info']
            if 'google_maps_link' in gps:
                print(f"  📍 Location: {gps['google_maps_link']}")

        # Delete original file if conversion was successful and delete_original is True
        if delete_original:
            try:
                os.remove(file_path)
                print(f"✓ Original file deleted: {file_path}")
            except OSError as e:
                print(f"⚠️ Could not delete original file {file_path}: {e}")

        return jpeg_path, metadata_json_path

    except Exception as e:
        print(f"Conversion failed: {e}")
        raise

def main():
    """Test the conversion with different file types."""
    test_files = [
        "./data/IMG_1854.heic",  # HEIC file
        # Add test MOV/MP4 files here when available
        # "./data/test_video.mov",
        # "./data/test_video.mp4",
    ]

    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nTesting with: {test_file}")
            try:
                jpeg_path, metadata_path = convert_media_file(test_file, delete_original=False)  # Don't delete during testing
                print(f"Success: {test_file} -> {jpeg_path}")
            except Exception as e:
                print(f"Failed: {test_file} - {e}")
        else:
            print(f"Test file not found: {test_file}")

if __name__ == "__main__":
    main()