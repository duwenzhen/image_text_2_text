#!/usr/bin/env python3
"""
Extract comprehensive metadata from image files including location, time, and camera settings.
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
from datetime import datetime
import json

def convert_gps_coordinates(gps_coords, gps_coords_ref):
    """
    Convert GPS coordinates from DMS (Degrees, Minutes, Seconds) to decimal degrees.
    
    Args:
        gps_coords: Tuple of (degrees, minutes, seconds) as Rational numbers
        gps_coords_ref: Reference direction ('N', 'S', 'E', 'W')
    
    Returns:
        float: Decimal degrees
    """
    try:
        degrees = float(gps_coords[0])
        minutes = float(gps_coords[1])
        seconds = float(gps_coords[2])
        
        decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # Apply negative sign for South and West
        if gps_coords_ref in ['S', 'W']:
            decimal_degrees = -decimal_degrees
            
        return decimal_degrees
    except (TypeError, IndexError, ZeroDivisionError):
        return None

def extract_gps_info(gps_info):
    """
    Extract GPS information from EXIF data.
    
    Args:
        gps_info: GPS info dictionary from EXIF
    
    Returns:
        dict: Processed GPS information
    """
    gps_data = {}
    
    if not gps_info:
        return gps_data
    
    # Convert GPS tag IDs to readable names
    gps_readable = {}
    for key, value in gps_info.items():
        tag_name = GPSTAGS.get(key, key)
        gps_readable[tag_name] = value
    
    # Extract latitude
    if 'GPSLatitude' in gps_readable and 'GPSLatitudeRef' in gps_readable:
        lat = convert_gps_coordinates(
            gps_readable['GPSLatitude'],
            gps_readable['GPSLatitudeRef']
        )
        if lat is not None:
            gps_data['latitude'] = lat
            gps_data['latitude_ref'] = gps_readable['GPSLatitudeRef']
    
    # Extract longitude
    if 'GPSLongitude' in gps_readable and 'GPSLongitudeRef' in gps_readable:
        lon = convert_gps_coordinates(
            gps_readable['GPSLongitude'],
            gps_readable['GPSLongitudeRef']
        )
        if lon is not None:
            gps_data['longitude'] = lon
            gps_data['longitude_ref'] = gps_readable['GPSLongitudeRef']
    
    # Extract altitude
    if 'GPSAltitude' in gps_readable:
        try:
            altitude = float(gps_readable['GPSAltitude'])
            gps_data['altitude'] = altitude
            
            # Check altitude reference (0 = above sea level, 1 = below sea level)
            if 'GPSAltitudeRef' in gps_readable:
                alt_ref = gps_readable['GPSAltitudeRef']
                if alt_ref == 1:
                    gps_data['altitude'] = -altitude
                gps_data['altitude_ref'] = 'above sea level' if alt_ref == 0 else 'below sea level'
        except (TypeError, ValueError):
            pass
    
    # Extract timestamp
    if 'GPSTimeStamp' in gps_readable and 'GPSDateStamp' in gps_readable:
        try:
            time_stamp = gps_readable['GPSTimeStamp']
            date_stamp = gps_readable['GPSDateStamp']
            
            # Convert time stamp (hours, minutes, seconds)
            hours = int(time_stamp[0])
            minutes = int(time_stamp[1])
            seconds = int(time_stamp[2])
            
            # Combine date and time
            gps_datetime = f"{date_stamp} {hours:02d}:{minutes:02d}:{seconds:02d} UTC"
            gps_data['timestamp'] = gps_datetime
        except (TypeError, ValueError, IndexError):
            pass
    
    # Add Google Maps link if coordinates are available
    if 'latitude' in gps_data and 'longitude' in gps_data:
        gps_data['google_maps_link'] = f"https://maps.google.com/?q={gps_data['latitude']},{gps_data['longitude']}"
    
    # Add raw GPS data for reference
    gps_data['raw_gps_tags'] = gps_readable
    
    return gps_data

def extract_datetime_info(exif_data):
    """
    Extract and parse datetime information from EXIF data.
    
    Args:
        exif_data: EXIF data dictionary
    
    Returns:
        dict: Processed datetime information
    """
    datetime_info = {}
    
    # Common datetime tags
    datetime_tags = {
        'DateTime': 'Last modified',
        'DateTimeOriginal': 'Date taken (original)',
        'DateTimeDigitized': 'Date digitized',
        'SubSecTime': 'Subseconds for DateTime',
        'SubSecTimeOriginal': 'Subseconds for DateTimeOriginal',
        'SubSecTimeDigitized': 'Subseconds for DateTimeDigitized'
    }
    
    for tag, description in datetime_tags.items():
        if tag in exif_data:
            datetime_str = exif_data[tag]
            datetime_info[tag] = {
                'raw': datetime_str,
                'description': description
            }
            
            # Try to parse the datetime
            try:
                if tag.startswith('SubSec'):
                    # Subsecond fields are just numbers
                    datetime_info[tag]['parsed'] = datetime_str
                else:
                    # Parse datetime string
                    dt = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
                    datetime_info[tag]['parsed'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                    datetime_info[tag]['formatted'] = dt.strftime('%B %d, %Y at %I:%M:%S %p')
            except ValueError:
                datetime_info[tag]['parsed'] = 'Could not parse'
    
    return datetime_info

def extract_camera_info(exif_data):
    """
    Extract camera and shooting information from EXIF data.
    
    Args:
        exif_data: EXIF data dictionary
    
    Returns:
        dict: Camera and shooting information
    """
    camera_info = {}
    
    # Camera identification
    camera_tags = {
        'Make': 'Camera manufacturer',
        'Model': 'Camera model',
        'Software': 'Software used',
        'LensModel': 'Lens model',
        'LensMake': 'Lens manufacturer'
    }
    
    # Shooting parameters
    shooting_tags = {
        'FNumber': 'Aperture (f-stop)',
        'ExposureTime': 'Shutter speed',
        'ISOSpeedRatings': 'ISO sensitivity',
        'FocalLength': 'Focal length',
        'Flash': 'Flash used',
        'WhiteBalance': 'White balance',
        'ExposureProgram': 'Exposure program',
        'MeteringMode': 'Metering mode',
        'ExposureBiasValue': 'Exposure compensation'
    }
    
    # Image properties
    image_tags = {
        'ImageWidth': 'Image width',
        'ImageLength': 'Image height',
        'Orientation': 'Orientation',
        'ColorSpace': 'Color space',
        'ResolutionUnit': 'Resolution unit',
        'XResolution': 'X resolution',
        'YResolution': 'Y resolution'
    }
    
    all_tags = {**camera_tags, **shooting_tags, **image_tags}
    
    for tag, description in all_tags.items():
        if tag in exif_data:
            value = exif_data[tag]
            camera_info[tag] = {
                'value': value,
                'description': description
            }
            
            # Add human-readable interpretations for specific tags
            if tag == 'ExposureTime' and isinstance(value, tuple):
                # Convert fraction to readable format
                if value[1] != 0:
                    shutter_speed = f"1/{int(value[1]/value[0])}" if value[0] == 1 else f"{value[0]}/{value[1]}"
                    camera_info[tag]['readable'] = f"{shutter_speed} sec"
            
            elif tag == 'FNumber' and isinstance(value, tuple):
                # Convert fraction to f-stop
                if value[1] != 0:
                    f_stop = value[0] / value[1]
                    camera_info[tag]['readable'] = f"f/{f_stop:.1f}"
            
            elif tag == 'FocalLength' and isinstance(value, tuple):
                # Convert fraction to mm
                if value[1] != 0:
                    focal_length = value[0] / value[1]
                    camera_info[tag]['readable'] = f"{focal_length:.1f}mm"
            
            elif tag == 'Flash':
                # Interpret flash value
                flash_modes = {
                    0: 'Flash did not fire',
                    1: 'Flash fired',
                    5: 'Strobe return light not detected',
                    7: 'Strobe return light detected',
                    9: 'Flash fired, compulsory flash mode',
                    13: 'Flash fired, compulsory flash mode, return light not detected',
                    15: 'Flash fired, compulsory flash mode, return light detected',
                    16: 'Flash did not fire, compulsory flash mode',
                    24: 'Flash did not fire, auto mode',
                    25: 'Flash fired, auto mode',
                    29: 'Flash fired, auto mode, return light not detected',
                    31: 'Flash fired, auto mode, return light detected',
                    32: 'No flash function',
                    65: 'Flash fired, red-eye reduction mode',
                    69: 'Flash fired, red-eye reduction mode, return light not detected',
                    71: 'Flash fired, red-eye reduction mode, return light detected',
                    73: 'Flash fired, compulsory flash mode, red-eye reduction mode',
                    77: 'Flash fired, compulsory flash mode, red-eye reduction mode, return light not detected',
                    79: 'Flash fired, compulsory flash mode, red-eye reduction mode, return light detected',
                    89: 'Flash fired, auto mode, red-eye reduction mode',
                    93: 'Flash fired, auto mode, return light not detected, red-eye reduction mode',
                    95: 'Flash fired, auto mode, return light detected, red-eye reduction mode'
                }
                camera_info[tag]['readable'] = flash_modes.get(value, f'Unknown flash mode ({value})')
    
    return camera_info

def analyze_image_metadata(image_path):
    """
    Comprehensive metadata analysis of an image file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        dict: Complete metadata analysis
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    analysis = {
        'file_info': {
            'filename': os.path.basename(image_path),
            'file_size': os.path.getsize(image_path),
            'file_path': os.path.abspath(image_path)
        },
        'basic_image_info': {},
        'gps_info': {},
        'datetime_info': {},
        'camera_info': {},
        'all_exif_tags': {}
    }
    
    try:
        with Image.open(image_path) as img:
            # Basic image information
            analysis['basic_image_info'] = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'megapixels': round((img.width * img.height) / 1000000, 1)
            }
            
            # Extract EXIF data
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif_raw = img._getexif()
                
                # Convert EXIF tag IDs to readable names
                exif_data = {}
                gps_info = {}
                
                for tag_id, value in exif_raw.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Handle GPS info separately
                    if tag == 'GPSInfo':
                        gps_info = value
                    else:
                        exif_data[tag] = value
                
                analysis['all_exif_tags'] = exif_data
                
                # Process GPS information
                analysis['gps_info'] = extract_gps_info(gps_info)
                
                # Process datetime information
                analysis['datetime_info'] = extract_datetime_info(exif_data)
                
                # Process camera information
                analysis['camera_info'] = extract_camera_info(exif_data)
            
            else:
                analysis['message'] = 'No EXIF data found in this image'
    
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis

def print_metadata_summary(analysis):
    """
    Print a human-readable summary of the metadata analysis.
    
    Args:
        analysis: Metadata analysis dictionary
    """
    print("=" * 80)
    print(f"METADATA ANALYSIS: {analysis['file_info']['filename']}")
    print("=" * 80)
    
    # File information
    print(f"\n📁 FILE INFORMATION:")
    print(f"   File size: {analysis['file_info']['file_size']:,} bytes")
    print(f"   Full path: {analysis['file_info']['file_path']}")
    
    # Basic image info
    basic = analysis['basic_image_info']
    if basic:
        print(f"\n🖼️  IMAGE PROPERTIES:")
        print(f"   Format: {basic.get('format', 'Unknown')}")
        print(f"   Dimensions: {basic.get('width', 0)} × {basic.get('height', 0)} pixels")
        print(f"   Megapixels: {basic.get('megapixels', 0):.1f} MP")
        print(f"   Color mode: {basic.get('mode', 'Unknown')}")
    
    # GPS information
    gps = analysis['gps_info']
    if gps and ('latitude' in gps or 'longitude' in gps):
        print(f"\n📍 LOCATION INFORMATION:")
        if 'latitude' in gps and 'longitude' in gps:
            print(f"   Coordinates: {gps['latitude']:.6f}, {gps['longitude']:.6f}")
            print(f"   Location: {gps['latitude']:.6f}° {gps.get('latitude_ref', 'N')}, {gps['longitude']:.6f}° {gps.get('longitude_ref', 'E')}")
        if 'altitude' in gps:
            print(f"   Altitude: {gps['altitude']:.1f}m {gps.get('altitude_ref', 'above sea level')}")
        if 'timestamp' in gps:
            print(f"   GPS timestamp: {gps['timestamp']}")
        if 'google_maps_link' in gps:
            print(f"   📍 View on Google Maps: {gps['google_maps_link']}")
    else:
        print(f"\n📍 LOCATION INFORMATION: No GPS data found")
    
    # DateTime information
    datetime_info = analysis['datetime_info']
    if datetime_info:
        print(f"\n⏰ DATE & TIME INFORMATION:")
        for tag, info in datetime_info.items():
            if 'formatted' in info:
                print(f"   {info['description']}: {info['formatted']}")
            elif 'parsed' in info:
                print(f"   {info['description']}: {info['parsed']}")
    else:
        print(f"\n⏰ DATE & TIME INFORMATION: No datetime data found")
    
    # Camera information
    camera = analysis['camera_info']
    if camera:
        print(f"\n📷 CAMERA INFORMATION:")
        
        # Camera identification
        if 'Make' in camera:
            print(f"   Manufacturer: {camera['Make']['value']}")
        if 'Model' in camera:
            print(f"   Model: {camera['Model']['value']}")
        if 'LensModel' in camera:
            print(f"   Lens: {camera['LensModel']['value']}")
        
        # Shooting settings
        print(f"\n⚙️  SHOOTING SETTINGS:")
        settings = ['FNumber', 'ExposureTime', 'ISOSpeedRatings', 'FocalLength']
        for setting in settings:
            if setting in camera:
                readable = camera[setting].get('readable', camera[setting]['value'])
                print(f"   {camera[setting]['description']}: {readable}")
        
        if 'Flash' in camera and 'readable' in camera['Flash']:
            print(f"   Flash: {camera['Flash']['readable']}")
    
    # Summary of available tags
    all_tags = analysis.get('all_exif_tags', {})
    if all_tags:
        print(f"\n📊 METADATA SUMMARY:")
        print(f"   Total EXIF tags found: {len(all_tags)}")
        print(f"   GPS data available: {'Yes' if gps else 'No'}")
        print(f"   Datetime data available: {'Yes' if datetime_info else 'No'}")
        print(f"   Camera data available: {'Yes' if camera else 'No'}")

def main():
    """Analyze the IMG_1854.jpg file."""
    image_path = "./data/IMG_1854.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        print("Please make sure the file exists at the specified path.")
        return
    
    try:
        # Perform comprehensive metadata analysis
        analysis = analyze_image_metadata(image_path)
        
        # Print human-readable summary
        print_metadata_summary(analysis)
        
        # Save detailed analysis to JSON file
        output_json = "./data/IMG_1854_detailed_metadata.json"
        with open(output_json, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Detailed metadata saved to: {output_json}")
        
    except Exception as e:
        print(f"Error analyzing image: {e}")

if __name__ == "__main__":
    main()