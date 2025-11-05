#!/usr/bin/env python3
"""
Correct projection using proper OptiTrack extrinsics interpretation.
Based on the analysis: R is World-to-Camera, T is camera position in world.
"""

import numpy as np
import cv2
import pandas as pd
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def load_optitrack_calibration(mcal_path, camera_serial='C11764'):
    """
    Load BOTH intrinsics and extrinsics from OptiTrack .mcal file.

    This ensures data consistency - using calibration from the same source.

    Returns:
        K: 3x3 intrinsic camera matrix
        dist: Distortion coefficients [k1, k2, p1, p2, k3]
        R_w2c: 3x3 World-to-Camera rotation matrix
        T_world: 3x1 Camera position in world coordinates
        img_size: [width, height]
    """
    tree = ET.parse(mcal_path)
    root = tree.getroot()

    for cam in root.findall('.//Camera'):
        if cam.get('Serial') == camera_serial:
            # ===== IMAGE SIZE (from Attributes) =====
            attributes = cam.find('Attributes')
            if attributes is None:
                raise ValueError(f"No Attributes found for camera {camera_serial}")

            width = int(attributes.get('ImagerPixelWidth'))
            height = int(attributes.get('ImagerPixelHeight'))

            # ===== INTRINSICS =====
            # Prefer IntrinsicStandardCameraModel (standard model)
            intrinsic = cam.find('.//IntrinsicStandardCameraModel')
            if intrinsic is None:
                # Fallback to Intrinsic (OptiTrack internal model)
                intrinsic = cam.find('.//Intrinsic')

            if intrinsic is None:
                raise ValueError(f"No intrinsic data found for camera {camera_serial}")

            # Focal lengths
            fx = float(intrinsic.get('HorizontalFocalLength'))
            fy = float(intrinsic.get('VerticalFocalLength'))

            # Principal point
            cx = float(intrinsic.get('LensCenterX'))
            cy = float(intrinsic.get('LensCenterY'))

            # Distortion coefficients
            # OptiTrack uses: k1, k2, k3, TangentialX (p1), TangentialY (p2)
            # OpenCV expects: [k1, k2, p1, p2, k3]
            k1 = float(intrinsic.get('k1', 0.0))
            k2 = float(intrinsic.get('k2', 0.0))
            k3 = float(intrinsic.get('k3', 0.0))
            p1 = float(intrinsic.get('TangentialX', 0.0))
            p2 = float(intrinsic.get('TangentialY', 0.0))

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

            # ===== EXTRINSICS =====
            extrinsic = cam.find('Extrinsic')

            if extrinsic is None:
                raise ValueError(f"No extrinsic data found for camera {camera_serial}")

            # Camera position in world
            T_world = np.array([
                float(extrinsic.get('X')),
                float(extrinsic.get('Y')),
                float(extrinsic.get('Z'))
            ], dtype=np.float64)

            # World-to-Camera rotation matrix
            R_w2c = np.array([
                [float(extrinsic.get('OrientMatrix0')),
                 float(extrinsic.get('OrientMatrix1')),
                 float(extrinsic.get('OrientMatrix2'))],
                [float(extrinsic.get('OrientMatrix3')),
                 float(extrinsic.get('OrientMatrix4')),
                 float(extrinsic.get('OrientMatrix5'))],
                [float(extrinsic.get('OrientMatrix6')),
                 float(extrinsic.get('OrientMatrix7')),
                 float(extrinsic.get('OrientMatrix8'))]
            ], dtype=np.float64)

            return K, dist, R_w2c, T_world, [width, height]

    raise ValueError(f"Camera {camera_serial} not found in .mcal file")


def load_markers_from_frame(csv_path, frame_idx):
    """Load markers from frame."""
    with open(csv_path, 'r') as f:
        for _ in range(3):
            f.readline()
        names_line = f.readline()

    names_raw = names_line.strip().split(',')[2:]
    names = [names_raw[i] for i in range(0, len(names_raw), 3) if names_raw[i]]

    df = pd.read_csv(csv_path, skiprows=7, header=None, low_memory=False)

    if frame_idx >= len(df):
        return np.array([]).reshape(0, 3), []

    markers = []
    marker_names = []
    for i, name in enumerate(names):
        col = 2 + i*3
        x = pd.to_numeric(df.iloc[frame_idx, col], errors='coerce')
        y = pd.to_numeric(df.iloc[frame_idx, col+1], errors='coerce')
        z = pd.to_numeric(df.iloc[frame_idx, col+2], errors='coerce')

        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            markers.append([x, y, z])
            marker_names.append(name)

    return np.array(markers) if markers else np.array([]).reshape(0, 3), marker_names


def main():
    mcal_path = "/Volumes/FastACIS/annotation_pipeline/optitrack.mcal"
    csv_path = "/Volumes/FastACIS/csldata/csl/mocap.csv"
    video_path = "/Volumes/FastACIS/csldata/video/mocap.avi"
    output_dir = Path("/Volumes/FastACIS/csldata/video")

    print("="*70)
    print("Projection using ONLY OptiTrack .mcal Calibration")
    print("(Intrinsics + Extrinsics from same source)")
    print("="*70)

    # Load BOTH intrinsics and extrinsics from .mcal file
    K, dist, R_w2c, T_world, img_size = load_optitrack_calibration(mcal_path)

    print(f"\nOptiTrack Intrinsics (from .mcal):")
    print(f"  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"  cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    print(f"  Distortion: k1={dist[0]:.6f}, k2={dist[1]:.6f}, p1={dist[2]:.6f}, p2={dist[3]:.6f}")
    print(f"  Image size: {img_size[0]}x{img_size[1]}")

    print(f"\nOptiTrack Extrinsics (from .mcal):")
    print(f"  T_world (camera position): {T_world} m")
    print(f"  R_w2c (World-to-Camera):")
    print(f"    det(R) = {np.linalg.det(R_w2c):.6f}")
    print(f"    Z-axis direction (camera forward): {R_w2c[:, 2]}")

    # For OpenCV projectPoints, we need:
    # rvec: rotation from world to camera
    # tvec: translation from world to camera
    # Standard W2C transform: P_cam = R_w2c @ P_world + t_w2c
    # where t_w2c = -R_w2c @ T_world

    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = -R_w2c @ T_world

    print(f"\nOpenCV projection parameters:")
    print(f"  rvec (from R_w2c): {rvec.flatten()}")
    print(f"  tvec = -R_w2c @ T_world: {tvec}")

    # Find a frame with markers
    test_frames = [100, 500, 1000, 1500, 2000, 3000, 5000]
    test_frame = None
    markers = None
    names = None

    for frame_idx in test_frames:
        m, n = load_markers_from_frame(csv_path, frame_idx)
        if len(m) > 0:
            test_frame = frame_idx
            markers = m
            names = n
            break

    if test_frame is None or len(markers) == 0:
        print("\nERROR: No markers found in any test frame!")
        return

    print(f"\nTesting frame {test_frame} with {len(markers)} markers")
    print(f"First marker '{names[0]}': {markers[0]} mm")

    # Load video frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Cannot read frame!")
        return

    # Convert markers from mm to meters
    markers_m = markers / 1000.0

    # Project using correct extrinsics
    print(f"\nProjecting with CORRECT extrinsics...")

    points_2d, _ = cv2.projectPoints(
        markers_m.reshape(-1, 1, 3),
        rvec,
        tvec,
        K,
        dist
    )
    points_2d = points_2d.reshape(-1, 2)

    # Check which are in front of camera
    R, _ = cv2.Rodrigues(rvec)
    points_cam = (R @ markers_m.T).T + tvec.T
    valid = points_cam[:, 2] > 0

    print(f"  Markers in front (Z>0): {np.sum(valid)}/{len(markers)}")

    if np.sum(valid) > 0:
        valid_2d = points_2d[valid]
        in_bounds = (valid_2d[:,0] >= 0) & (valid_2d[:,0] < img_size[0]) & \
                    (valid_2d[:,1] >= 0) & (valid_2d[:,1] < img_size[1])

        print(f"  In image bounds: {np.sum(in_bounds)}/{np.sum(valid)}")
        print(f"  First marker cam coords: {points_cam[0]}")
        print(f"  First marker 2D: ({points_2d[0,0]:.1f}, {points_2d[0,1]:.1f})")

        if np.sum(in_bounds) > 0:
            # Draw markers
            for i, is_valid in enumerate(valid):
                if is_valid:
                    x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
                    if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        cv2.circle(frame, (x, y), 6, (0, 0, 0), 1)

            output_path = output_dir / f"correct_projection_mcal_only_frame_{test_frame}.jpg"
            cv2.putText(frame, f"Frame {test_frame} - OptiTrack .mcal Calibration",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(str(output_path), frame)
            print(f"\n  SUCCESS! Saved: {output_path.name}")
        else:
            print("\n  WARNING: No markers in image bounds!")
    else:
        print("\n  ERROR: No markers in front of camera (all Z < 0)!")
        print("\n  DIAGNOSIS:")
        print("  - Camera position:", T_world, "m")
        print("  - Camera Z-axis (forward):", R_w2c[:, 2])
        print("  - First marker position:", markers[0], "mm =", markers_m[0], "m")
        print("  - Vector from camera to marker:", markers_m[0] - T_world)
        print("\n  CONCLUSION:")
        print("  The camera is FACING AWAY from the markers!")
        print("\n  POSSIBLE CAUSES:")
        print("  1. OptiTrack calibration is incorrect/corrupted")
        print("  2. This .mcal file is from a different capture session")
        print("  3. World coordinate system was reset after calibration")
        print("\n  RECOMMENDATION:")
        print("  - Re-calibrate the OptiTrack system with PrimeColor camera")
        print("  - Ensure .mcal and mocap.csv are from the SAME session")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
