#!/usr/bin/env python3
"""
Synchronize mocap markers with video and project 3D markers to 2D images.

This script:
1. Loads mocap CSV data (120fps or other frame rate)
2. Loads video and detects video frame rate
3. Synchronizes mocap frames to video frames (resampling if needed)
4. Projects 3D markers to 2D image coordinates using camera intrinsics
5. Generates visualization with markers overlaid on video frames
"""

import os
import sys
import argparse
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from utils.constants import PATH_ASSETS_VIDEOS


def parse_mocap_csv(csv_path):
    """
    Parse OptiTrack mocap CSV export format.

    Returns:
        dict: {
            'fps': float,
            'total_frames': int,
            'marker_names': list of str,
            'frames': list of dicts with frame_id, time, markers (N x 3 array)
        }
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Read header lines
        header_line = f.readline().strip()
        headers = dict(zip(header_line.split(',')[::2], header_line.split(',')[1::2]))

        fps = float(headers.get('Export Frame Rate', 120.0))
        total_frames = int(headers.get('Total Exported Frames', 0))

        print(f"Mocap CSV Info:")
        print(f"  - Export FPS: {fps}")
        print(f"  - Total Frames: {total_frames}")
        print(f"  - Capture Start: {headers.get('Capture Start Time', 'Unknown')}")

        # Skip to marker name row (row 3)
        f.readline()  # Type row
        name_line = f.readline().strip()
        marker_cols = name_line.split(',')[2:]  # Skip "Frame" and "Time (Seconds)"

        # Extract unique marker names (each marker has X, Y, Z columns)
        marker_names = []
        for i in range(0, len(marker_cols), 3):
            if i < len(marker_cols):
                marker_names.append(marker_cols[i])

        print(f"  - Number of markers: {len(marker_names)}")

        # Skip ID and Parent rows
        f.readline()
        f.readline()

        # Read column headers (should be Frame, Time, X, Y, Z, X, Y, Z, ...)
        col_headers = f.readline().strip().split(',')

        # Parse data rows
        frames_data = []
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 2:
                continue

            try:
                frame_id = int(row[0])
                time_sec = float(row[1])

                # Extract marker positions (groups of 3: X, Y, Z)
                markers = []
                for i in range(2, len(row), 3):
                    if i + 2 < len(row):
                        x_str, y_str, z_str = row[i], row[i+1], row[i+2]

                        # Handle missing markers (empty strings)
                        if x_str and y_str and z_str:
                            x = float(x_str)
                            y = float(y_str)
                            z = float(z_str)
                            markers.append([x, y, z])
                        else:
                            markers.append([np.nan, np.nan, np.nan])

                frames_data.append({
                    'frame_id': frame_id,
                    'time': time_sec,
                    'markers': np.array(markers, dtype=np.float32)
                })
            except (ValueError, IndexError) as e:
                continue

    return {
        'fps': fps,
        'total_frames': total_frames,
        'marker_names': marker_names,
        'frames': frames_data
    }


def get_video_info(video_path):
    """Get video FPS and frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    print(f"\nVideo Info:")
    print(f"  - FPS: {fps}")
    print(f"  - Total Frames: {frame_count}")
    print(f"  - Resolution: {width}x{height}")

    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height
    }


def synchronize_mocap_to_video(mocap_data, video_info, offset_frames=0):
    """
    Synchronize mocap frames to video frames.

    Args:
        mocap_data: Parsed mocap data dict
        video_info: Video info dict
        offset_frames: Offset in video frames (positive = mocap starts later)

    Returns:
        list: Synchronized marker data for each video frame
    """
    mocap_fps = mocap_data['fps']
    video_fps = video_info['fps']

    print(f"\nSynchronization:")
    print(f"  - Mocap FPS: {mocap_fps}")
    print(f"  - Video FPS: {video_fps}")
    print(f"  - Offset: {offset_frames} video frames")

    # Resample mocap to match video FPS
    synced_data = []
    fps_ratio = mocap_fps / video_fps

    for video_frame_idx in range(video_info['frame_count']):
        # Account for offset
        adjusted_frame_idx = video_frame_idx - offset_frames

        if adjusted_frame_idx < 0:
            # Before mocap starts
            synced_data.append(None)
            continue

        # Find corresponding mocap frame
        mocap_frame_idx = int(adjusted_frame_idx * fps_ratio)

        if mocap_frame_idx >= len(mocap_data['frames']):
            # After mocap ends
            synced_data.append(None)
            continue

        synced_data.append(mocap_data['frames'][mocap_frame_idx])

    print(f"  - Synced {len([d for d in synced_data if d is not None])} frames")

    return synced_data


def load_camera_intrinsics(intrinsic_path, camera_name='cam0'):
    """Load camera intrinsics from JSON file."""
    with open(intrinsic_path, 'r') as f:
        calib_data = json.load(f)

    cam_data = calib_data['cameras'][camera_name]
    K = np.array(cam_data['K'], dtype=np.float32)
    dist = np.array(cam_data['dist'], dtype=np.float32).flatten()

    print(f"\nCamera Intrinsics ({camera_name}):")
    print(f"  - K matrix:\n{K}")
    print(f"  - Distortion: {dist}")

    return K, dist


def project_markers_to_image(markers_3d, K, dist, extrinsics=None):
    """
    Project 3D markers to 2D image coordinates.

    Args:
        markers_3d: (N, 3) array of 3D positions in world coordinates (mm)
        K: (3, 3) camera intrinsic matrix
        dist: (5,) distortion coefficients
        extrinsics: Optional dict with 'R' and 'T' for world-to-camera transform

    Returns:
        (N, 2) array of 2D image coordinates, or None for invalid markers
    """
    # Filter out invalid markers (NaN)
    valid_mask = ~np.isnan(markers_3d).any(axis=1)
    valid_markers = markers_3d[valid_mask]

    if len(valid_markers) == 0:
        return None, None

    # Convert from mm to meters (OpenCV typically uses meters)
    markers_3d_m = valid_markers / 1000.0

    # Apply extrinsics if provided (world to camera transform)
    if extrinsics is not None:
        R = np.array(extrinsics['R'], dtype=np.float32).reshape(3, 3)
        T = np.array(extrinsics['T'], dtype=np.float32).reshape(3, 1)
        markers_cam = (R @ markers_3d_m.T + T).T
    else:
        # Assume markers are already in camera coordinates
        # For visualization: assume camera at origin, looking down +Z axis
        markers_cam = markers_3d_m

    # Project to image using cv2.projectPoints
    rvec = np.zeros(3, dtype=np.float32)  # No additional rotation
    tvec = np.zeros(3, dtype=np.float32)  # No additional translation

    points_2d, _ = cv2.projectPoints(markers_cam, rvec, tvec, K, dist)
    points_2d = points_2d.reshape(-1, 2)

    return points_2d, valid_mask


def visualize_markers_on_video(video_path, synced_markers, intrinsic_path,
                                output_path, marker_names,
                                camera_name='cam0', extrinsics=None,
                                max_frames=None):
    """
    Generate video with projected markers overlaid.

    Args:
        video_path: Input video path
        synced_markers: List of synced marker data per frame
        intrinsic_path: Path to intrinsic calibration JSON
        output_path: Output video path
        marker_names: List of marker names
        camera_name: Camera name in intrinsic JSON
        extrinsics: Optional extrinsics dict
        max_frames: Maximum frames to process (for testing)
    """
    # Load camera parameters
    K, dist = load_camera_intrinsics(intrinsic_path, camera_name)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nGenerating visualization video...")
    print(f"  - Output: {output_path}")

    frame_idx = 0
    pbar = tqdm(total=min(len(synced_markers), max_frames or float('inf')))

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(synced_markers):
            break

        if max_frames and frame_idx >= max_frames:
            break

        # Get synced markers for this frame
        marker_data = synced_markers[frame_idx]

        if marker_data is not None:
            markers_3d = marker_data['markers']

            # Project markers to 2D
            points_2d, valid_mask = project_markers_to_image(markers_3d, K, dist, extrinsics)

            if points_2d is not None:
                # Draw markers on frame
                for i, pt in enumerate(points_2d):
                    x, y = int(pt[0]), int(pt[1])

                    # Check if point is within image bounds
                    if 0 <= x < width and 0 <= y < height:
                        # Draw marker as circle
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        cv2.circle(frame, (x, y), 7, (255, 255, 255), 2)

                        # Draw marker ID (optional, can be cluttered)
                        # valid_idx = np.where(valid_mask)[0][i]
                        # cv2.putText(frame, str(valid_idx), (x+10, y),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"Done! Processed {frame_idx} frames.")


def main():
    parser = argparse.ArgumentParser(description="Sync mocap markers with video and project to 2D")
    parser.add_argument("--mocap_csv", required=True, help="Path to mocap CSV file")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--intrinsic", required=True, help="Path to intrinsic calibration JSON")
    parser.add_argument("--output", default="output_with_markers.mp4", help="Output video path")
    parser.add_argument("--camera_name", default="cam0", help="Camera name in intrinsic JSON")
    parser.add_argument("--offset_frames", type=int, default=0,
                        help="Offset in video frames (positive = mocap starts later)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames to process (for testing)")
    parser.add_argument("--extrinsics", help="Path to extrinsics JSON (optional)")

    args = parser.parse_args()

    # Parse mocap data
    print("=" * 60)
    print("STEP 1: Parsing Mocap CSV")
    print("=" * 60)
    mocap_data = parse_mocap_csv(args.mocap_csv)

    # Get video info
    print("\n" + "=" * 60)
    print("STEP 2: Analyzing Video")
    print("=" * 60)
    video_info = get_video_info(args.video)

    # Synchronize
    print("\n" + "=" * 60)
    print("STEP 3: Synchronizing Mocap to Video")
    print("=" * 60)
    synced_markers = synchronize_mocap_to_video(mocap_data, video_info, args.offset_frames)

    # Load extrinsics if provided
    extrinsics = None
    if args.extrinsics:
        with open(args.extrinsics, 'r') as f:
            extrinsics_data = json.load(f)
            if 'camera_base2cam' in extrinsics_data:
                extrinsics = extrinsics_data['camera_base2cam'][args.camera_name]
            else:
                extrinsics = extrinsics_data

    # Generate visualization
    print("\n" + "=" * 60)
    print("STEP 4: Generating Visualization")
    print("=" * 60)
    visualize_markers_on_video(
        args.video,
        synced_markers,
        args.intrinsic,
        args.output,
        mocap_data['marker_names'],
        args.camera_name,
        extrinsics,
        args.max_frames
    )

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
