#!/usr/bin/env python3
"""
Merge calibration files to compute primecolor_to_cam4 transform.

Given:
- primecolor_to_cam1 (from motion/calibration.json)
- cam4_to_cam1 (from gopro_extrinsic/calibration.json)

Compute:
- primecolor_to_cam4 = inverse(cam4_to_cam1) * primecolor_to_cam1
"""

import numpy as np
import json
from scipy.spatial.transform import Rotation

def compose_transforms(R1, T1, R2, T2):
    """Compose two transforms: T_result = T2 * T1"""
    R_result = R2 @ R1
    T_result = R2 @ T1 + T2
    return R_result, T_result

def invert_transform(R, T):
    """Invert a transform: [R|T] -> [R^T | -R^T*T]"""
    R_inv = R.T
    T_inv = -R_inv @ T
    return R_inv, T_inv

# Load motion/calibration.json (has primecolor_to_cam1)
with open('/Volumes/FastACIS/GoPro/motion/calibration.json', 'r') as f:
    motion_calib = json.load(f)

# Load gopro_extrinsic/calibration.json (has cam4_to_cam1)
with open('/Volumes/FastACIS/GoPro/gopro_extrinsic /calibration.json', 'r') as f:
    gopro_calib = json.load(f)

print("="*80)
print("MERGING CALIBRATIONS FOR CAM4")
print("="*80)

# Get primecolor_to_cam1
p2c1 = motion_calib['camera_base2cam']['primecolor_to_cam1']
R_p2c1 = np.array(p2c1['R'], dtype=np.float64)
T_p2c1 = np.array(p2c1['T'], dtype=np.float64)

print("\n1. primecolor_to_cam1:")
print(f"   R:\n{R_p2c1}")
print(f"   T: {T_p2c1}")

# Get cam4_to_cam1
c42c1 = gopro_calib['camera_base2cam']['cam4_to_cam1']
R_c42c1 = np.array(c42c1['R'], dtype=np.float64)
T_c42c1 = np.array(c42c1['T'], dtype=np.float64)

print("\n2. cam4_to_cam1:")
print(f"   R:\n{R_c42c1}")
print(f"   T: {T_c42c1}")

# Compute cam1_to_cam4 = inverse(cam4_to_cam1)
R_c12c4, T_c12c4 = invert_transform(R_c42c1, T_c42c1)

print("\n3. cam1_to_cam4 (inverted):")
print(f"   R:\n{R_c12c4}")
print(f"   T: {T_c12c4}")

# Compute primecolor_to_cam4 = cam1_to_cam4 * primecolor_to_cam1
R_p2c4, T_p2c4 = compose_transforms(R_p2c1, T_p2c1, R_c12c4, T_c12c4)

print("\n4. primecolor_to_cam4 (computed):")
print(f"   R:\n{R_p2c4}")
print(f"   T: {T_p2c4}")

# Get cam4 intrinsics
cam4_intrinsics = gopro_calib['cameras']['cam4']

print("\n5. cam4 intrinsics:")
K_cam4 = np.array(cam4_intrinsics['K'])
print(f"   fx={K_cam4[0,0]:.2f}, fy={K_cam4[1,1]:.2f}")
print(f"   cx={K_cam4[0,2]:.2f}, cy={K_cam4[1,2]:.2f}")
print(f"   Image size: {cam4_intrinsics['image_size']}")

# Create merged calibration file
merged_calib = {
    "cameras": {
        "cam4": cam4_intrinsics,
        "primecolor": motion_calib['cameras']['primecolor']
    },
    "camera_base2cam": {
        "cam4": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "T": [0.0, 0.0, 0.0]
        },
        "primecolor_to_cam4": {
            "R": R_p2c4.tolist(),
            "T": T_p2c4.tolist()
        }
    }
}

output_path = '/Volumes/FastACIS/GoPro/motion/calibration_cam4.json'
with open(output_path, 'w') as f:
    json.dump(merged_calib, f, indent=2)

print(f"\n✓ Merged calibration saved to: {output_path}")
print("\nYou can now use this file for projection to cam4:")
print("  --calibration /Volumes/FastACIS/GoPro/motion/calibration_cam4.json")

# Also check the distance between cameras
distance = np.linalg.norm(T_c42c1)
print(f"\n6. Camera separation:")
print(f"   Distance cam4→cam1: {distance:.3f}m")
print(f"   This explains why the person appears smaller in cam4")