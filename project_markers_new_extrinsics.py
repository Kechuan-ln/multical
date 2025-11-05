#!/usr/bin/env python3
"""
Project mocap markers to video using .mcal intrinsics + calibrated extrinsics
"""

import cv2
import numpy as np
import pandas as pd
import json
import xml.etree.ElementTree as ET
from pathlib import Path

def load_mcal_intrinsics(mcal_path, camera_serial='C11764'):
    """Load intrinsics from .mcal file."""
    tree = ET.parse(mcal_path)
    root = tree.getroot()

    for camera in root.findall('.//Camera'):
        serial = camera.get('Serial')
        props = camera.find('.//Properties')

        if serial == camera_serial or (props is not None and props.get('CameraID') == '13'):
            intrinsic_elem = camera.find('.//Intrinsic')

            # Extract intrinsic parameters (OptiTrack format)
            fx = float(intrinsic_elem.get('HorizontalFocalLength'))
            fy = float(intrinsic_elem.get('VerticalFocalLength'))
            cx = float(intrinsic_elem.get('LensCenterX'))
            cy = float(intrinsic_elem.get('LensCenterY'))

            # Build K matrix
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)

            # Get distortion (OptiTrack format)
            k1 = float(intrinsic_elem.get('k1', 0))
            k2 = float(intrinsic_elem.get('k2', 0))
            k3 = float(intrinsic_elem.get('k3', 0))
            p1 = float(intrinsic_elem.get('TangentialX', 0))
            p2 = float(intrinsic_elem.get('TangentialY', 0))
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

            # Get resolution from Properties
            width = 1920  # Default for PrimeColor
            height = 1080
            if props is not None:
                width = int(props.get('Width', width))
                height = int(props.get('Height', height))

            print(f"✓ Loaded .mcal intrinsics for {camera_serial}")
            print(f"  K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
            print(f"  Distortion: {dist}")

            return K, dist, width, height

    raise ValueError(f"Camera {camera_serial} not found in .mcal file")

def load_calibrated_extrinsics(json_path):
    """Load calibrated extrinsics from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    rvec = np.array(data['rvec'], dtype=np.float64).reshape(3, 1)
    tvec = np.array(data['tvec'], dtype=np.float64).reshape(3, 1)

    print(f"✓ Loaded calibrated extrinsics")
    print(f"  rvec: {rvec.flatten()}")
    print(f"  tvec: {tvec.flatten()}")
    print(f"  Camera position (world): {data['camera_position_world']}")

    return rvec, tvec

def load_mocap_data(csv_path):
    """Load mocap marker data from CSV (OptiTrack format with 7-row header)."""
    # Read header rows
    with open(csv_path, 'r') as f:
        header_lines = [next(f) for _ in range(7)]

    # Parse marker names from row 3 (Name row)
    # Each marker appears 3 times (X, Y, Z)
    name_row = header_lines[2].strip().split(',')
    all_names = name_row[2:]  # Skip first 2 columns (Frame, Time)

    # Extract unique marker names (every 3rd entry is the same marker)
    marker_names = []
    for i in range(0, len(all_names), 3):
        if i < len(all_names) and all_names[i]:
            marker_names.append(all_names[i])

    # Read data starting from row 8
    df = pd.read_csv(csv_path, skiprows=7, header=None, low_memory=False)

    # Build column names: Frame, Time, then marker1:X, marker1:Y, marker1:Z, ...
    columns = ['Frame', 'Time']
    for marker in marker_names:
        columns.extend([f'{marker}:X', f'{marker}:Y', f'{marker}:Z'])

    # Trim columns to actual dataframe size
    df.columns = columns[:len(df.columns)]

    print(f"✓ Loaded mocap data: {len(df)} frames, {len(marker_names)} markers")

    return df, marker_names

def get_markers_3d(df, marker_names, frame_idx):
    """Get 3D marker positions for a specific frame."""
    # Use iloc for positional indexing (frame_idx is row number, not frame ID)
    markers_3d = []
    for marker in marker_names:
        try:
            x = float(df.iloc[frame_idx][f'{marker}:X'])
            y = float(df.iloc[frame_idx][f'{marker}:Y'])
            z = float(df.iloc[frame_idx][f'{marker}:Z'])
            markers_3d.append([x, y, z])
        except (ValueError, KeyError, IndexError, TypeError):
            # If conversion fails or column missing, use NaN
            markers_3d.append([np.nan, np.nan, np.nan])

    result = np.array(markers_3d, dtype=np.float64)
    return result

def project_markers(markers_3d, rvec, tvec, K, dist):
    """Project 3D markers to 2D using OpenCV (OptiTrack convention with negative fx)."""
    # Convert mm to meters
    markers_3d_m = markers_3d / 1000.0

    # Ensure shape is (N, 3)
    if markers_3d_m.ndim == 1:
        markers_3d_m = markers_3d_m.reshape(-1, 3)

    # Use negative fx for OptiTrack Y-up coordinate system
    K_neg = K.copy()
    K_neg[0, 0] = -K_neg[0, 0]

    # Remove invalid markers (NaN)
    valid_mask = ~np.isnan(markers_3d_m).any(axis=1)
    valid_markers = markers_3d_m[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    if len(valid_markers) == 0:
        return np.array([]).reshape(0, 2), np.array([])

    # Project to 2D
    # Note: cv2.projectPoints expects shape (N, 1, 3) for proper broadcasting
    points_2d, _ = cv2.projectPoints(
        valid_markers.reshape(-1, 1, 3), rvec, tvec, K_neg, dist
    )
    points_2d = points_2d.reshape(-1, 2)

    return points_2d, valid_indices

def visualize_frame(video_path, csv_path, mcal_path, extrinsics_path,
                   output_path, frame_idx=8465, marker_names_to_show=None):
    """Visualize a single frame with projected markers."""

    # Load data
    K, dist, img_width, img_height = load_mcal_intrinsics(mcal_path)
    rvec, tvec = load_calibrated_extrinsics(extrinsics_path)
    df, marker_names = load_mocap_data(csv_path)

    # Read video frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_idx}")
        return

    # Find mocap row for this video frame (Frame column contains frame numbers)
    # For now, assume 1:1 mapping (video frame = mocap row index)
    # This works if mocap starts at frame 0 and has no gaps
    mocap_row = frame_idx
    if mocap_row >= len(df):
        print(f"Warning: Frame {frame_idx} out of mocap range (max {len(df)-1})")
        mocap_row = len(df) - 1

    # Get 3D markers for this frame
    markers_3d = get_markers_3d(df, marker_names, mocap_row)

    # Project to 2D
    points_2d, valid_indices = project_markers(markers_3d, rvec, tvec, K, dist)

    # Check if we have any points
    if len(points_2d) == 0 or len(valid_indices) == 0:
        print(f"Warning: No valid markers found in frame {frame_idx}")
        cv2.imwrite(str(output_path), frame)
        return

    # Filter points within image bounds
    in_bounds = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height)
    )

    # Draw markers
    print(f"\nProjecting {len(points_2d)} markers to frame {frame_idx}:")
    for i, (pt, valid_idx) in enumerate(zip(points_2d, valid_indices)):
        if in_bounds[i]:
            marker_name = marker_names[valid_idx]

            # Only show selected markers if specified
            if marker_names_to_show and marker_name not in marker_names_to_show:
                continue

            x, y = int(pt[0]), int(pt[1])

            # Draw marker
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)  # White outline

            # Draw marker name
            cv2.putText(frame, marker_name, (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            print(f"  {marker_name}: ({x}, {y})")

    # Save output
    cv2.imwrite(str(output_path), frame)
    print(f"\n✓ Saved visualization to {output_path}")

def create_video_with_projections(video_path, csv_path, mcal_path, extrinsics_path,
                                 output_path, start_frame=0, num_frames=100):
    """Create video with projected markers."""

    # Load data
    K, dist, img_width, img_height = load_mcal_intrinsics(mcal_path)
    rvec, tvec = load_calibrated_extrinsics(extrinsics_path)
    df, marker_names = load_mocap_data(csv_path)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (img_width, img_height))

    print(f"\nCreating video with projections...")
    print(f"  Input: {video_path}")
    print(f"  Output: {output_path}")
    print(f"  Frames: {start_frame} to {min(start_frame + num_frames, total_frames)}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(num_frames):
        frame_idx = start_frame + i
        if frame_idx >= total_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Use frame_idx as mocap row (assuming 1:1 mapping)
        mocap_row = min(frame_idx, len(df) - 1)

        # Get 3D markers for this frame
        markers_3d = get_markers_3d(df, marker_names, mocap_row)

        # Project to 2D
        points_2d, valid_indices = project_markers(markers_3d, rvec, tvec, K, dist)

        # Filter points within image bounds
        in_bounds = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height)
        )

        # Draw markers
        for pt, valid_idx, visible in zip(points_2d, valid_indices, in_bounds):
            if visible:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")

    cap.release()
    out.release()

    print(f"✓ Video created successfully!")

if __name__ == '__main__':
    # Paths
    CSV_PATH = "/Volumes/FastACIS/GoPro/motion/mocap/mocap.csv"
    VIDEO_PATH = "/Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi"
    MCAL_PATH = "/Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal"
    EXTRINSICS_PATH = "/Volumes/FastACIS/annotation_pipeline/extrinsics_calibrated.json"

    # Test on single frame first
    print("=" * 60)
    print("Testing projection on single frame...")
    print("=" * 60)

    visualize_frame(
        VIDEO_PATH, CSV_PATH, MCAL_PATH, EXTRINSICS_PATH,
        output_path="test_projection_new_extrinsics.jpg",
        frame_idx=8465  # Use frame from annotation session (has markers)
    )

    # Optionally create video
    create_choice = input("\nCreate video with projections? (y/n): ")
    if create_choice.lower() == 'y':
        create_video_with_projections(
            VIDEO_PATH, CSV_PATH, MCAL_PATH, EXTRINSICS_PATH,
            output_path="primecolor_projected_new_extrinsics.mp4",
            start_frame=8400,
            num_frames=200
        )
