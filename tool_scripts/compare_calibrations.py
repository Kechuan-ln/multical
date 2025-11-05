#!/usr/bin/env python3

import json
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import sys
import os

sys.path.append('..')

def load_calibration(file_path):
    """Load calibration file and return the data."""
    with open(file_path, 'r') as f:
        return json.load(f)

def rotation_matrix_to_axis_angle_degrees(R_matrix):
    """Convert rotation matrix to axis-angle representation in degrees."""
    r = R.from_matrix(R_matrix)
    axis_angle = r.as_rotvec()
    angle_degrees = np.linalg.norm(axis_angle) * 180.0 / np.pi
    return angle_degrees

def compute_rotation_difference(R1, R2):
    """Compute the angular difference between two rotation matrices in degrees."""
    R1 = np.array(R1, dtype=np.float64)
    R2 = np.array(R2, dtype=np.float64)
    
    # Compute relative rotation: R_rel = R2 * R1^T
    R_rel = R2 @ R1.T
    
    # Convert to angle
    angle_diff = rotation_matrix_to_axis_angle_degrees(R_rel)
    return angle_diff

def compute_translation_difference(T1, T2):
    """Compute the Euclidean distance between two translation vectors."""
    T1 = np.array(T1, dtype=np.float64).flatten()
    T2 = np.array(T2, dtype=np.float64).flatten()
    
    diff = T2 - T1
    distance = np.linalg.norm(diff)
    return distance, diff

def compare_calibrations(calib1_path, calib2_path):
    """Compare two calibration files and return the differences."""
    
    # Load calibration files
    print(f"Loading {calib1_path}")
    calib1 = load_calibration(calib1_path)
    
    print(f"Loading {calib2_path}")
    calib2 = load_calibration(calib2_path)
    
    # Check if both files have camera_base2cam section
    if 'camera_base2cam' not in calib1:
        print(f"No 'camera_base2cam' section found in {calib1_path}")
        return
    
    if 'camera_base2cam' not in calib2:
        print(f"No 'camera_base2cam' section found in {calib2_path}")
        return
    
    base2cam1 = calib1['camera_base2cam']
    base2cam2 = calib2['camera_base2cam']
    
    # Get common cameras
    cameras1 = set(base2cam1.keys())
    cameras2 = set(base2cam2.keys())
    common_cameras = cameras1.intersection(cameras2)
    
    if not common_cameras:
        print("No common cameras found between the two calibration files")
        return
    
    print(f"Found {len(common_cameras)} common cameras: {sorted(common_cameras)}")
    print("=" * 80)
    
    results = {}
    
    for camera in sorted(common_cameras):
        print(f"Camera: {camera}")
        
        # Get rotation matrices and translation vectors
        R1 = base2cam1[camera]['R']
        T1 = base2cam1[camera]['T']
        R2 = base2cam2[camera]['R']
        T2 = base2cam2[camera]['T']
        
        # Compute differences
        angle_diff = compute_rotation_difference(R1, R2)
        trans_distance, trans_diff = compute_translation_difference(T1, T2)
        
        results[camera] = {
            'rotation_angle_diff_degrees': angle_diff,
            'translation_distance': trans_distance,
            'translation_diff_vector': trans_diff.tolist()
        }
        
        print(f"  Rotation angle difference: {angle_diff:.6f} degrees")
        print(f"  Translation distance: {trans_distance:.6f}")
        print(f"  Translation difference vector: [{trans_diff[0]:.6f}, {trans_diff[1]:.6f}, {trans_diff[2]:.6f}]")
        print("-" * 40)
    
    # Summary statistics
    angle_diffs = [results[cam]['rotation_angle_diff_degrees'] for cam in results]
    trans_distances = [results[cam]['translation_distance'] for cam in results]
    
    print("Summary Statistics:")
    print(f"  Average rotation difference: {np.mean(angle_diffs):.6f} ± {np.std(angle_diffs):.6f} degrees")
    print(f"  Max rotation difference: {np.max(angle_diffs):.6f} degrees")
    print(f"  Average translation distance: {np.mean(trans_distances):.6f} ± {np.std(trans_distances):.6f}")
    print(f"  Max translation distance: {np.max(trans_distances):.6f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare two calibration files")
    parser.add_argument("--path_calib1", default="assets/videos/extr_623_sync2/calibration.json", 
                        help="Path to first calibration file")
    parser.add_argument("--path_calib2", default="assets/videos/extr_623_sync2/calibration0.json",
                        help="Path to second calibration file")
    parser.add_argument("--output", help="Optional JSON file to save comparison results")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.path_calib1):
        print(f"Error: File {args.path_calib1} does not exist")
        return
    
    if not os.path.exists(args.path_calib2):
        print(f"Error: File {args.path_calib2} does not exist")
        return
    
    # Compare calibrations
    results = compare_calibrations(args.path_calib1, args.path_calib2)
    
if __name__ == "__main__":
    main()