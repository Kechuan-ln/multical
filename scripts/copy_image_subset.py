#!/usr/bin/env python3
"""
Image subset copying utility for multical calibration datasets.

This script copies a subset of images from camera subfolders (cam0, cam1, cam2, etc.)
based on specified frame IDs. It's useful for creating smaller datasets for testing
or when you only need specific frames for calibration.

Usage:
    python copy_image_subset.py --image_path ./assets/extr_620_sync --dest_path ./assets/extr_620_sync_subset --frames 0,10,20,30,40
    
    Or modify the constants in the script and run directly:
    python copy_image_subset.py
"""

import os
import shutil
import argparse
from pathlib import Path
import re
from typing import List, Optional
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration - modify these as needed
DEFAULT_SOURCE_DIR = "/Volumes/FastACIS/gorpos-2-calibration/videos/original"
DEFAULT_DEST_DIR = "/Volumes/FastACIS/gorpos-2-calibration/videos_stable"
DEFAULT_FRAME_IDS =  [20, 29, 35, 48, 69, 80, 87, 94, 99, 114, 126, 131, 137, 144, 149, 157, 167, 184, 192, 203, 216, 221, 226, 236, 249, 259, 266, 273, 279, 288, 298, 307, 312, 326, 335, 340, 345, 351, 361, 372, 377, 383, 388, 404, 410, 415, 421, 428, 433, 443, 454, 459, 470, 479, 487, 493, 504, 516, 529, 534, 543, 548, 556, 565, 571, 582, 589, 594, 599, 604, 609, 621, 626, 633, 645, 657, 671, 680, 685, 693, 699, 707, 722, 728, 733, 747, 752, 757, 762, 767, 772, 778, 783, 788, 793, 806, 817, 828, 838, 843, 848, 857, 868, 875, 880, 891, 897, 905, 917, 927, 934, 940, 945, 953, 960, 968, 973, 985, 1000, 1009, 1014, 1030, 1046, 1051, 1056, 1061, 1066, 1073, 1081, 1086, 1091, 1096, 1105, 1118, 1129, 1144, 1154, 1162, 1172, 1177, 1183, 1190, 1199, 1207, 1214, 1219, 1230, 1236, 1249, 1259, 1267, 1283]



# Camera folder pattern (matches cam0, cam1, cam2, etc.)
CAMERA_FOLDER_PATTERN = re.compile(r'^cam\d+$')

# Common image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def is_camera_folder(folder_name: str) -> bool:
    """Check if a folder name matches the camera folder pattern (cam0, cam1, etc.)"""
    return CAMERA_FOLDER_PATTERN.match(folder_name) is not None


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files in a directory, sorted by name."""
    if not directory.exists():
        return []
    
    image_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file_path)
    
    # Sort by filename to ensure consistent ordering
    return sorted(image_files)


def extract_frame_id_from_filename(filename: str) -> Optional[int]:
    """
    Extract frame ID from filename. Assumes filenames contain frame numbers.
    
    Common patterns:
    - frame_000001.jpg -> 1
    - img_0010.png -> 10
    - 000025.jpg -> 25
    - image_frame_050.jpg -> 50
    """
    # Try to find numbers in the filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # Usually the last or largest number is the frame ID
        # You might need to adjust this logic based on your naming convention
        return int(numbers[-1])  # Take the last number found
    return None


def copy_image_subset(source_dir: str, dest_dir: str, frame_ids: List[int], 
                     dry_run: bool = False) -> None:
    """
    Copy a subset of images from source to destination directory.
    
    Args:
        source_dir: Source directory path
        dest_dir: Destination directory path  
        frame_ids: List of frame IDs to copy
        dry_run: If True, only print what would be copied without actually copying
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_path}")
        return
    
    if not source_path.is_dir():
        logger.error(f"Source path is not a directory: {source_path}")
        return
    
    logger.info(f"Source directory: {source_path}")
    logger.info(f"Destination directory: {dest_path}")
    logger.info(f"Frame IDs to copy: {frame_ids}")
    logger.info(f"Dry run: {dry_run}")
    
    # Create destination directory if it doesn't exist
    if not dry_run:
        dest_path.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    total_skipped = 0
    
    # Process each subdirectory in source
    for item in source_path.iterdir():
        if not item.is_dir():
            logger.debug(f"Skipping non-directory: {item.name}")
            continue
            
        if not is_camera_folder(item.name):
            logger.info(f"Skipping non-camera folder: {item.name}")
            continue
        
        logger.info(f"Processing camera folder: {item.name}")
        
        # Get all image files in this camera folder
        image_files = get_image_files(item)
        logger.info(f"Found {len(image_files)} images in {item.name}")
        
        if not image_files:
            logger.warning(f"No images found in {item.name}")
            continue
        
        # Create destination camera folder
        dest_camera_dir = dest_path / item.name
        if not dry_run:
            dest_camera_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images based on frame IDs
        copied_count = 0
        
        # Method 1: Copy by frame ID extracted from filename
        for image_file in image_files:
            frame_id = extract_frame_id_from_filename(image_file.name)
            
            if frame_id is not None and frame_id in frame_ids:
                dest_file = dest_camera_dir / image_file.name
                
                if dry_run:
                    logger.info(f"Would copy: {image_file} -> {dest_file}")
                else:
                    try:
                        shutil.copy2(image_file, dest_file)
                        logger.debug(f"Copied: {image_file.name} (frame {frame_id})")
                        copied_count += 1
                    except Exception as e:
                        logger.error(f"Failed to copy {image_file}: {e}")
            else:
                total_skipped += 1
        
        logger.info(f"Copied {copied_count} images from {item.name}")
        total_copied += copied_count
    
    logger.info(f"Summary: {total_copied} images copied, {total_skipped} images skipped")
    
    if dry_run:
        logger.info("This was a dry run. No files were actually copied.")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Copy a subset of images from camera subfolders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Copy specific frame IDs
    python copy_image_subset.py --image_path ./data/calib --dest_path ./data/calib_subset --frames 0,10,20,30
    
    # Dry run to see what would be copied
    python copy_image_subset.py --image_path ./data/calib --dest_path ./data/calib_subset --frames 0,5,10 --dry_run
    
    # Use default settings (modify DEFAULT_* constants in script)
    python copy_image_subset.py
        """
    )
    
    parser.add_argument(
        '--image_path', '-s',
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help=f'Source directory path (default: {DEFAULT_SOURCE_DIR})'
    )
    
    parser.add_argument(
        '--dest_path', '-d', 
        type=str,
        default=DEFAULT_DEST_DIR,
        help=f'Destination directory path (default: {DEFAULT_DEST_DIR})'
    )
    
    parser.add_argument(
        '--frames', '-f',
        type=str,
        default=','.join(map(str, DEFAULT_FRAME_IDS)),
        help=f'Comma-separated list of frame IDs to copy (default: {",".join(map(str, DEFAULT_FRAME_IDS))})'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be copied without actually copying files'
    )
    
    
    args = parser.parse_args()
    
    
    # Parse frame IDs
    try:
        frame_ids = [int(x.strip()) for x in args.frames.split(',') if x.strip()]
    except ValueError as e:
        logger.error(f"Invalid frame IDs format: {e}")
        return
    
    if not frame_ids:
        logger.error("No valid frame IDs provided")
        return
    
    # Perform the copy operation
    copy_image_subset(args.image_path, args.dest_path, frame_ids, args.dry_run)


if __name__ == '__main__':
    main()
