#!/usr/bin/env python3

import cv2
import numpy as np
import tqdm
import yaml
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import glob
from tqdm import tqdm
import re
import os
import sys
sys.path.append('..')  # Add parent directory to path for imports
from utils.constants import PATH_ASSETS_VIDEOS


class CharucoBoardDetector:
    def __init__(self, config_path: str):
        """Initialize the detector with board configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        board_config = config['common']
        self.size = tuple(board_config['size'])
        self.square_length = board_config['square_length']
        self.marker_length = board_config['marker_length']
        self.min_points = board_config.get('min_points', 40)
        
        # Create ArUco dictionary
        aruco_dict_name = board_config['aruco_dict']
        if aruco_dict_name == '7X7_250':
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
        elif aruco_dict_name == '4X4_100':
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        else:
            raise ValueError(f"Unsupported ArUco dictionary: {aruco_dict_name}")
        
        # Create ChArUco board
        width, height = self.size
        self.board = cv2.aruco.CharucoBoard_create(
            width, height, self.square_length, self.marker_length, self.aruco_dict
        )
        
        # ArUco detection parameters
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        if 'aruco_params' in config:
            aruco_config = config['aruco_params']
            if 'adaptiveThreshWinSizeMax' in aruco_config:
                self.aruco_params.adaptiveThreshWinSizeMax = aruco_config['adaptiveThreshWinSizeMax']
            if 'adaptiveThreshWinSizeStep' in aruco_config:
                self.aruco_params.adaptiveThreshWinSizeStep = aruco_config['adaptiveThreshWinSizeStep']

    def detect_board(self, image: np.ndarray) -> Optional[Dict]:
        """Detect ChArUco board in image."""
        # Detect ArUco markers
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            image, self.aruco_dict, parameters=self.aruco_params
        )
        
        if marker_ids is None:
            return None
        
        # Interpolate ChArUco corners
        _, corners, ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, image, self.board
        )
        
        if ids is None or len(ids) < self.min_points:
            return None
        
        return {
            'corners': corners.squeeze(1),
            'ids': ids.squeeze(1),
            'num_points': len(ids)
        }


def calculate_frame_stability(detections: List[Optional[Dict]]) -> List[float]:
    """Calculate stability scores by comparing with previous frame."""
    stability_scores = []
    
    for i in range(len(detections)):
        if detections[i] is None:
            stability_scores.append(1000)
            continue
        
        # First frame has no previous frame to compare
        if i == 0:
            stability_scores.append(1000)
            continue
            
        # Previous frame must exist and have detection
        if detections[i-1] is None:
            stability_scores.append(1000)
            continue
        
        current_corners = detections[i]['corners']
        current_ids = detections[i]['ids']
        prev_corners = detections[i-1]['corners']
        prev_ids = detections[i-1]['ids']
        
        # Find common corner points between current and previous frame
        common_ids = np.intersect1d(current_ids, prev_ids)
        if len(common_ids) < 10:  # Need sufficient common points
            stability_scores.append(1000)
            continue
        
        # Get corner positions for common IDs
        current_mask = np.isin(current_ids, common_ids)
        prev_mask = np.isin(prev_ids, common_ids)
        
        current_common = current_corners[current_mask]
        prev_common = prev_corners[prev_mask]
        
        # Sort by IDs to ensure correspondence
        current_sorted_ids = current_ids[current_mask]
        prev_sorted_ids = prev_ids[prev_mask]
        
        current_sort_idx = np.argsort(current_sorted_ids)
        prev_sort_idx = np.argsort(prev_sorted_ids)
        
        current_common = current_common[current_sort_idx]
        prev_common = prev_common[prev_sort_idx]
        
        # Calculate movement (lower is more stable)
        movements = np.linalg.norm(current_common - prev_common, axis=1)
        avg_movement = np.mean(movements)
        
        # Convert to stability score (higher is more stable)
        stability_score = avg_movement
        stability_scores.append(stability_score)
    
    return stability_scores


def downsample_consecutive_frames(indices: List[int], min_gap: int = 5) -> List[int]:
    """
    Downsample consecutive frame indices to avoid selecting too many similar frames.
    
    Args:
        indices: List of frame indices
        min_gap: Minimum gap between selected frames
        
    Returns:
        Downsampled list of indices
    """
    if not indices:
        return indices
    
    downsampled = [indices[0]]  # Always keep first frame
    
    for idx in indices[1:]:
        # Only add if it's sufficiently far from the last selected frame
        if idx - downsampled[-1] >= min_gap:
            downsampled.append(idx)
    
    return downsampled




def is_camera_folder(folder_name: str) -> bool:
    """Check if a folder name matches the camera folder pattern (cam0, cam1, etc.)"""
    camera_folder_pattern = re.compile(r'^cam\d+$')
    return camera_folder_pattern.match(folder_name) is not None

def find_stable_boards(data_dir: str, board_config_path: str, 
                      movement_threshold: float = 10.0, 
                      min_detection_quality: int = 40) -> Dict[str, List[int]]:
    """
    Find stable board detections across camera folders.
    
    Args:
        data_dir: Root directory containing cam1, cam2, etc. folders
        board_config_path: Path to board configuration YAML
        stability_threshold: Minimum stability score for selection
        min_detection_quality: Minimum number of detected points
        
    Returns:
        Dictionary mapping camera names to lists of stable frame indices
    """
    detector = CharucoBoardDetector(board_config_path)
    results = {}
    
    # Find all camera directories
    data_path = Path(data_dir)
    cam_dirs = [d for d in data_path.iterdir() if d.is_dir() and is_camera_folder(d.name)]
    
    if not cam_dirs:
        print(f"No camera directories found in {data_dir}")
        return results
    
    for cam_dir in sorted(cam_dirs):
        print(f"Processing {cam_dir.name}...")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(cam_dir / ext)))
        
        image_files.sort()  # Ensure consistent ordering
        
        if not image_files:
            print(f"  No images found in {cam_dir}")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Detect boards in all images
        detections = []
        for i, img_path in tqdm(enumerate(image_files), total=len(image_files)):
            
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                detections.append(None)
                continue
            
            detection = detector.detect_board(image)
            
            # Filter by detection quality
            if detection and detection['num_points'] >= min_detection_quality:
                detections.append(detection)
            else:
                detections.append(None)
        
        # Calculate stability scores
        stability_scores = calculate_frame_stability(detections)
        
        # Select stable frames
        stable_indices = []
        for i, (detection, stability) in enumerate(zip(detections, stability_scores)):
            if detection is not None and stability < movement_threshold:
                cimg_idx = int(image_files[i].split('/')[-1].split('_')[-1].split('.')[0])
                print(image_files[i],cimg_idx, stability)
                stable_indices.append(cimg_idx)
        
        results[cam_dir.name] = stable_indices
        print(f"  Found {len(stable_indices)} stable boards out of {len(image_files)} images")
        
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Find stable calibration board detections')
    parser.add_argument('--recording_tag', default='sync_9122/original', help='Indicate the recording folder under PATH_ASSETS_VIDEOS')
    parser.add_argument('--boards', default='../multical/asset/charuco_b1_2.yaml',
                       help='Path to board configuration file')
    parser.add_argument('--movement_threshold', type=float, default=10,
                       help='Minimum movement threshold for selection')
    parser.add_argument('--min_detection_quality', type=int, default=40,
                       help='Minimum number of detected corner points')
    parser.add_argument('--downsample_rate',default=5, type=int, help='Downsample rate for frame selection')

    args = parser.parse_args()
    
    # Find stable boards
    path_data = os.path.join(PATH_ASSETS_VIDEOS, args.recording_tag)
    results = find_stable_boards(
        path_data,
        args.boards,
        args.movement_threshold,
        args.min_detection_quality
    )
    
    # Downsample consecutive frames
    indices_set = set()
    for camera, indices in results.items():
        if indices:
            results[camera] = downsample_consecutive_frames(indices, min_gap=args.downsample_rate)
            indices_set.update(results[camera])

    # Print results
    print("\n=== RESULTS ===")
    print(results)
    #print indices_set as list
    print("Total stable frames found:", len(indices_set))

    indices_set = list(indices_set)
    indices_set.sort()
    indices_set = downsample_consecutive_frames(indices_set, min_gap=args.downsample_rate)
    print("Stable frame indices:", sorted(list(indices_set)))


if __name__ == '__main__':
    main()