#!/usr/bin/env python3
"""
Fix the X-axis mirroring issue.
Based on Method 4: R^T, tvec = -R^T @ T
"""

import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path


def load_camera_raw(mcal_path):
    """Load raw camera parameters."""
    tree = ET.parse(mcal_path)
    root = tree.getroot()

    for cam in root.findall('.//Camera'):
        if cam.get('Serial') == 'C11764':
            intrinsic = cam.find('IntrinsicStandardCameraModel')
            cx = float(intrinsic.get('LensCenterX'))
            cy = float(intrinsic.get('LensCenterY'))
            fx = float(intrinsic.get('HorizontalFocalLength'))
            fy = float(intrinsic.get('VerticalFocalLength'))
            k1 = float(intrinsic.get('k1'))
            k2 = float(intrinsic.get('k2'))
            k3 = float(intrinsic.get('k3'))
            p1 = float(intrinsic.get('TangentialX'))
            p2 = float(intrinsic.get('TangentialY'))

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

            extrinsic = cam.find('Extrinsic')
            T = np.array([
                float(extrinsic.get('X')),
                float(extrinsic.get('Y')),
                float(extrinsic.get('Z'))
            ], dtype=np.float64)

            R_flat = [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(9)]
            R = np.array(R_flat, dtype=np.float64).reshape(3, 3)

            return K, dist, R, T

    raise ValueError("Camera not found")


def load_markers_from_frame(csv_path, frame_idx):
    """Load markers from frame."""
    with open(csv_path, 'r') as f:
        for _ in range(3):
            f.readline()
        names_line = f.readline()

    names_raw = names_line.strip().split(',')[2:]
    names = [names_raw[i] for i in range(0, len(names_raw), 3) if names_raw[i]]

    df = pd.read_csv(csv_path, skiprows=7, header=None, low_memory=False)

    markers = []
    for i, name in enumerate(names):
        col = 2 + i*3
        x = pd.to_numeric(df.iloc[frame_idx, col], errors='coerce')
        y = pd.to_numeric(df.iloc[frame_idx, col+1], errors='coerce')
        z = pd.to_numeric(df.iloc[frame_idx, col+2], errors='coerce')

        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            markers.append([x, y, z])

    return np.array(markers) if markers else np.array([]).reshape(0, 3)


def project_with_flip(markers_3d_mm, K, dist, R, T, flip_x_world=False,
                     flip_x_cam=False, flip_x_pixel=False, negate_fx=False):
    """
    Project using Method 4 + X flip options.
    """
    # Convert mm to meters
    markers_3d = markers_3d_mm / 1000.0

    # Flip X in world coordinates
    if flip_x_world:
        markers_3d = markers_3d * np.array([-1, 1, 1])
        T_use = T * np.array([-1, 1, 1])
    else:
        T_use = T

    # Method 4: R^T, tvec = -R^T @ T
    rvec, _ = cv2.Rodrigues(R.T)
    tvec = -R.T @ T_use

    # Project using OpenCV
    points_2d, _ = cv2.projectPoints(
        markers_3d.reshape(-1, 1, 3),
        rvec,
        tvec,
        K if not negate_fx else np.array([[-K[0,0], 0, K[0,2]], [0, K[1,1], K[1,2]], [0, 0, 1]]),
        dist
    )
    points_2d = points_2d.reshape(-1, 2)

    # Flip X in pixel coordinates
    if flip_x_pixel:
        points_2d[:, 0] = 1920 - points_2d[:, 0]

    return points_2d


def main():
    mcal_path = "/Volumes/FastACIS/annotation_pipeline/optitrack.mcal"
    csv_path = "/Volumes/FastACIS/csldata/csl/mocap.csv"
    video_path = "/Volumes/FastACIS/csldata/video/mocap.avi"

    print("="*70)
    print("Testing X-axis Mirror Fixes")
    print("="*70)

    # Load camera
    K, dist, R, T = load_camera_raw(mcal_path)

    # Load markers
    test_frame = 4000
    markers = load_markers_from_frame(csv_path, test_frame)
    print(f"\nLoaded {len(markers)} markers from frame {test_frame}")

    # Load video frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, frame_base = cap.read()
    cap.release()

    if not ret:
        print("Cannot read frame!")
        return

    # Test different X-flip methods
    methods = [
        ("Method4_Original", False, False, False, False),
        ("Method4_FlipX_World", True, False, False, False),
        ("Method4_FlipX_Pixel", False, False, True, False),
        ("Method4_NegativeFx", False, False, False, True),
        ("Method4_FlipWorld_FlipPixel", True, False, True, False),
    ]

    output_dir = Path("/Volumes/FastACIS/csldata/video")

    for method_name, flip_x_world, flip_x_cam, flip_x_pixel, negate_fx in methods:
        print(f"\nTesting: {method_name}")

        try:
            points_2d = project_with_flip(markers, K, dist, R, T,
                                         flip_x_world, flip_x_cam, flip_x_pixel, negate_fx)

            # Check bounds
            in_bounds = (points_2d[:,0] >= 0) & (points_2d[:,0] < 1920) & \
                        (points_2d[:,1] >= 0) & (points_2d[:,1] < 1080)

            print(f"  Markers in image: {np.sum(in_bounds)}/{len(markers)}")

            if np.sum(in_bounds) > 0:
                # Draw on frame
                frame = frame_base.copy()
                for i, in_img in enumerate(in_bounds):
                    if in_img:
                        x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        cv2.circle(frame, (x, y), 6, (0, 0, 0), 1)

                output_path = output_dir / f"mirror_fix_{method_name}.jpg"
                cv2.putText(frame, method_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                           0.9, (0, 255, 255), 2)
                cv2.imwrite(str(output_path), frame)
                print(f"  Saved: {output_path.name}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*70}")
    print("Results saved to /Volumes/FastACIS/csldata/video/mirror_fix_*.jpg")
    print("Find the image where markers correctly align with the person.")
    print("="*70)


if __name__ == "__main__":
    main()
