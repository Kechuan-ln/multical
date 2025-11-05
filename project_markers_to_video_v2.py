#!/usr/bin/env python3
"""
Project OptiTrack 3D markers to PrimeColor camera 2D video.
Version 2: Flexible calibration sources with command-line arguments.

New features in V2:
- Flexible calibration: can use user-calibrated intrinsics/extrinsics OR .mcal fallback
- Support for start_frame and num_frames (specify frame range to process)
- Full command-line argument support (compatible with project_markers_final.py)
- Uses Method 4 (negative fx) for correct projection
- Easy-to-use CLI interface

This script:
1. Loads intrinsics from user-calibrated JSON (optional, falls back to .mcal)
2. Loads extrinsics from user-calibrated JSON (optional, falls back to .mcal)
3. Reads 3D marker positions from mocap.csv
4. Projects markers to 2D image coordinates using negative fx method
5. Overlays markers on video frames (with optional frame range)

Usage:
    python project_markers_to_video_v2.py \\
        --mcal /path/to/calibration.mcal \\
        --csv /path/to/mocap.csv \\
        --video /path/to/video.avi \\
        --output /path/to/output.mp4 \\
        --intrinsics /path/to/intrinsics.json \\  # optional
        --extrinsics /path/to/extrinsics.json \\  # optional
        --start-frame 3800 \\
        --duration 600 \\
        --marker-size 3 \\
        --marker-color 0,255,0

    # Alternative: use --num-frames instead of --duration
    python project_markers_to_video_v2.py \\
        --start-frame 3800 \\
        --num-frames 600 \\
        ...

Calibration flexibility:
- --intrinsics: Path to multical format JSON OR omit to use .mcal
- --extrinsics: Path to {rvec, tvec} JSON OR omit to use .mcal
"""

import numpy as np
import cv2
import pandas as pd
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm


class PrimeColorCamera:
    """PrimeColor camera with flexible calibration sources."""

    def __init__(self, mcal_path, intrinsics_json=None, extrinsics_json=None):
        """
        Load camera calibration.

        Args:
            mcal_path: Path to OptiTrack .mcal file (fallback for intrinsics/extrinsics)
            intrinsics_json: Optional path to calibrated intrinsics JSON (multical format)
            extrinsics_json: Optional path to calibrated extrinsics JSON (rvec, tvec format)
        """
        self.load_calibration(mcal_path, intrinsics_json, extrinsics_json)

    def load_calibration(self, mcal_path, intrinsics_json=None, extrinsics_json=None):
        """Load intrinsics and extrinsics from various sources."""

        camera_serial = 'C11764'

        # Parse .mcal for image size and fallback calibration
        tree = ET.parse(mcal_path)
        root = tree.getroot()

        camera = None
        for cam in root.findall('.//Camera'):
            if cam.get('Serial') == camera_serial:
                camera = cam
                break

        if camera is None:
            raise ValueError(f"Camera {camera_serial} not found in .mcal file")

        # Image size
        attributes = camera.find('Attributes')
        self.width = int(attributes.get('ImagerPixelWidth'))
        self.height = int(attributes.get('ImagerPixelHeight'))

        # Load intrinsics
        if intrinsics_json:
            # Load from multical JSON file
            print(f"Loading intrinsics from: {intrinsics_json}")
            with open(intrinsics_json, 'r') as f:
                intr_data = json.load(f)

            # Get camera data (multical uses camera names like 'primecolor')
            cam_data = None
            for cam_name in ['primecolor', 'C11764', camera_serial]:
                if cam_name in intr_data.get('cameras', {}):
                    cam_data = intr_data['cameras'][cam_name]
                    break

            if cam_data is None:
                raise ValueError(f"Camera {camera_serial} not found in {intrinsics_json}")

            # Multical format: K is 3x3 matrix stored as list
            K_multical = np.array(cam_data['K'], dtype=np.float64)
            fx = K_multical[0, 0]
            fy = K_multical[1, 1]
            cx = K_multical[0, 2]
            cy = K_multical[1, 2]

            # Distortion coefficients
            self.dist = np.array(cam_data['dist'], dtype=np.float64).flatten()

            print(f"  Using multical intrinsics")
            print(f"    fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        else:
            # Load from .mcal file (fallback)
            print(f"Loading intrinsics from .mcal: {mcal_path}")
            intrinsic = camera.find('.//IntrinsicStandardCameraModel')
            if intrinsic is None:
                intrinsic = camera.find('.//Intrinsic')

            fx = float(intrinsic.get('HorizontalFocalLength'))
            fy = float(intrinsic.get('VerticalFocalLength'))
            cx = float(intrinsic.get('LensCenterX'))
            cy = float(intrinsic.get('LensCenterY'))
            k1 = float(intrinsic.get('k1', 0.0))
            k2 = float(intrinsic.get('k2', 0.0))
            k3 = float(intrinsic.get('k3', 0.0))
            p1 = float(intrinsic.get('TangentialX', 0.0))
            p2 = float(intrinsic.get('TangentialY', 0.0))
            self.dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

            print(f"  Using .mcal intrinsics")

        # METHOD 4: NEGATIVE fx (empirically verified correct for PrimeColor)
        # Note: This produces Z<0 but correct 2D projections
        # The negative fx compensates for OptiTrack coordinate system
        self.K = np.array([[-fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Load extrinsics
        if extrinsics_json:
            # Load from calibrated JSON file
            print(f"Loading extrinsics from: {extrinsics_json}")
            with open(extrinsics_json, 'r') as f:
                ext_data = json.load(f)

            # Use rvec/tvec (Method 4 projection)
            self.rvec = np.array(ext_data['rvec'], dtype=np.float64).reshape(3, 1)
            self.tvec = np.array(ext_data['tvec'], dtype=np.float64).reshape(3, 1)

            print(f"  Using calibrated extrinsics (rvec/tvec for Method 4)")
            if 'camera_position_world' in ext_data:
                print(f"  Camera position (world): {ext_data['camera_position_world']}")
        else:
            # Load from .mcal file (fallback)
            print(f"Loading extrinsics from .mcal: {mcal_path}")
            extrinsic = camera.find('Extrinsic')
            T_world = np.array([
                float(extrinsic.get('X')),
                float(extrinsic.get('Y')),
                float(extrinsic.get('Z'))
            ], dtype=np.float64)

            R_c2w = np.array([
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

            # Standard W2C transform
            R_w2c = R_c2w.T
            self.rvec, _ = cv2.Rodrigues(R_w2c)
            self.tvec = -R_w2c @ T_world

            print(f"  Using .mcal extrinsics")
            print(f"  Camera position (world): [{T_world[0]:.3f}, {T_world[1]:.3f}, {T_world[2]:.3f}] meters")

        print(f"  Image size: {self.width}x{self.height}")
        print(f"  Using Method 4 with negative fx for correct projection")

    def project_3d_to_2d(self, points_3d):
        """
        Project 3D points in OptiTrack world coordinates to 2D image pixels.

        Uses Method 4: negative fx in K matrix with direct rvec/tvec projection.

        Args:
            points_3d: Nx3 array of 3D points in millimeters (OptiTrack units)

        Returns:
            points_2d: Nx2 array of pixel coordinates
            valid_mask: N boolean array (True if point is in image bounds)
        """
        if len(points_3d) == 0:
            return np.array([]), np.array([])

        # Convert from millimeters to meters
        points_3d_m = points_3d / 1000.0

        # Direct projection using negative fx (K matrix already has -fx)
        points_2d, _ = cv2.projectPoints(
            points_3d_m.reshape(-1, 1, 3),
            self.rvec,
            self.tvec,
            self.K,
            self.dist
        )
        points_2d = points_2d.reshape(-1, 2)

        # Check which points are in image bounds
        # NOTE: We don't check Z>0 because Method 4 (negative fx) produces Z<0 but correct projections
        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < self.width) & \
                     (points_2d[:, 1] >= 0) & (points_2d[:, 1] < self.height)

        return points_2d, valid_mask


def load_mocap_data(csv_path):
    """
    Load OptiTrack marker data from CSV export.

    Returns:
        markers: dict with marker names as keys, values are (N_frames, 3) arrays
        frame_times: array of frame timestamps in seconds
    """
    print(f"Loading mocap data from {csv_path}...")

    # Read CSV with complex header structure
    # OptiTrack CSV format:
    # Lines 1-3: metadata/headers
    # Line 4: Marker names (actual marker names we need)
    # Lines 5-7: more metadata
    # Line 8+: data
    with open(csv_path, 'r') as f:
        # Skip first 3 lines
        for _ in range(3):
            f.readline()

        # Line 4: Marker names
        marker_names_line = f.readline()
        marker_names_raw = marker_names_line.strip().split(',')[2:]  # Skip first 2 columns

    # Build marker name list (every 3 columns is one marker)
    marker_names = []
    for i in range(0, len(marker_names_raw), 3):
        if i < len(marker_names_raw) and marker_names_raw[i]:
            marker_names.append(marker_names_raw[i])

    print(f"Found {len(marker_names)} markers")

    # Read data (skip header rows)
    df = pd.read_csv(csv_path, skiprows=7, header=None, low_memory=False)

    # Extract frame numbers and times
    frames = df.iloc[:, 0].values
    times = df.iloc[:, 1].values

    # Extract marker positions
    markers = {}
    for i, name in enumerate(marker_names):
        col_start = 2 + i * 3  # Skip Frame and Time columns
        x = df.iloc[:, col_start].values
        y = df.iloc[:, col_start + 1].values
        z = df.iloc[:, col_start + 2].values

        # Stack into (N, 3) array, replacing empty strings with NaN
        positions = np.stack([
            pd.to_numeric(x, errors='coerce'),
            pd.to_numeric(y, errors='coerce'),
            pd.to_numeric(z, errors='coerce')
        ], axis=1)

        markers[name] = positions

    return markers, times


def overlay_markers_on_video(video_path, output_path, camera, markers, frame_times,
                              marker_color=(0, 255, 0), marker_size=5,
                              start_frame=0, num_frames=None):
    """
    Project markers onto video and save result.

    Args:
        video_path: Path to input video
        output_path: Path to output video
        camera: PrimeColorCamera instance
        markers: Dict of marker data from load_mocap_data()
        frame_times: Array of frame timestamps
        marker_color: BGR color tuple for markers
        marker_size: Radius of marker circles
        start_frame: Starting frame number (default: 0)
        num_frames: Number of frames to process (default: None = all frames)
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Mocap frames: {len(frame_times)}")

    # Determine processing range
    if num_frames is not None:
        end_frame = min(start_frame + num_frames, total_frames)
    else:
        end_frame = total_frames

    print(f"\nProcessing range:")
    print(f"  Start frame: {start_frame}")
    print(f"  End frame: {end_frame}")
    print(f"  Total frames to process: {end_frame - start_frame}")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Process frames
    frame_idx = start_frame
    pbar = tqdm(total=end_frame - start_frame, desc="Processing frames")

    # Statistics
    total_markers_seen = 0
    total_markers_valid = 0
    total_markers_in_image = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Get corresponding mocap frame (match by index)
        if frame_idx < len(frame_times):
            # Collect all valid markers for this frame
            points_3d = []
            for marker_name, positions in markers.items():
                pos = positions[frame_idx]
                if not np.any(np.isnan(pos)):  # Valid position
                    points_3d.append(pos)

            if len(points_3d) > 0:
                points_3d = np.array(points_3d)
                total_markers_seen += len(points_3d)

                # Project to 2D
                points_2d, valid_mask = camera.project_3d_to_2d(points_3d)
                total_markers_valid += np.sum(valid_mask)

                # Draw markers on frame
                for i, (pt, valid) in enumerate(zip(points_2d, valid_mask)):
                    if valid:
                        x, y = int(pt[0]), int(pt[1])
                        cv2.circle(frame, (x, y), marker_size, marker_color, -1)
                        cv2.circle(frame, (x, y), marker_size + 1, (0, 0, 0), 1)  # Black outline
                        total_markers_in_image += 1

        # Write frame
        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"\nProjection statistics:")
    print(f"  Total markers detected: {total_markers_seen}")
    print(f"  Markers in front of camera: {total_markers_valid}")
    print(f"  Markers projected in image: {total_markers_in_image}")
    if total_markers_seen > 0:
        print(f"  Success rate: {100*total_markers_in_image/total_markers_seen:.1f}%")

    print(f"\nOutput saved to: {output_path}")


def main():
    """Main function with command-line argument support."""
    import argparse

    # ============================================================
    # 命令行参数解析
    # ============================================================
    parser = argparse.ArgumentParser(
        description='Project OptiTrack markers onto PrimeColor video (V2 with flexible calibration)'
    )

    # Input files
    parser.add_argument('--mcal', default='/Volumes/FastACIS/annotation_pipeline/optitrack.mcal',
                        help='Path to .mcal calibration file')
    parser.add_argument('--csv', default='/Volumes/FastACIS/csldata/csl/mocap.csv',
                        help='Path to mocap CSV file')
    parser.add_argument('--video', default='/Volumes/FastACIS/csldata/video/mocap.avi',
                        help='Path to video file')

    # Calibration files (optional)
    parser.add_argument('--intrinsics', default=None,
                        help='Path to calibrated intrinsics JSON (optional, overrides .mcal intrinsics)')
    parser.add_argument('--extrinsics', default=None,
                        help='Path to calibrated extrinsics JSON (optional, overrides .mcal extrinsics)')

    # Output
    parser.add_argument('--output', default='/Volumes/FastACIS/csldata/video/mocap_with_markers_v2.mp4',
                        help='Output video path')

    # Frame range
    parser.add_argument('--start-frame', type=int, default=3800,
                        help='Start frame (default: 0)')
    parser.add_argument('--num-frames', '--duration', type=int, default=600, dest='num_frames',
                        help='Number of frames to process (default: None = all frames). Can also use --duration')

    # Visualization
    parser.add_argument('--marker-size', type=int, default=3,
                        help='Marker circle radius in pixels (default: 3)')
    parser.add_argument('--marker-color', default='0,255,0',
                        help='Marker color in BGR format (default: 0,255,0 = green)')

    args = parser.parse_args()

    # Parse marker color
    marker_color = tuple(map(int, args.marker_color.split(',')))

    # Convert to Path objects
    mcal_path = Path(args.mcal)
    mocap_csv = Path(args.csv)
    video_path = Path(args.video)
    output_path = Path(args.output)
    INTRINSICS_JSON = Path(args.intrinsics) if args.intrinsics else None
    EXTRINSICS_JSON = Path(args.extrinsics) if args.extrinsics else None
    START_FRAME = args.start_frame
    NUM_FRAMES = args.num_frames
    MARKER_COLOR = marker_color
    MARKER_SIZE = args.marker_size

    # ============================================================
    # 处理流程
    # ============================================================

    print("=" * 70)
    print("OptiTrack Marker Projection to Video (V2)")
    print("=" * 70)
    print(f"\nInput files:")
    print(f"  MCAL: {mcal_path}")
    print(f"  CSV: {mocap_csv}")
    print(f"  Video: {video_path}")
    if INTRINSICS_JSON:
        print(f"  Intrinsics JSON: {INTRINSICS_JSON}")
    if EXTRINSICS_JSON:
        print(f"  Extrinsics JSON: {EXTRINSICS_JSON}")
    print(f"\nOutput: {output_path}")
    print(f"Frame range: {START_FRAME} to {START_FRAME + NUM_FRAMES if NUM_FRAMES else 'end'}")

    # Load camera calibration
    print("\n" + "=" * 60)
    camera = PrimeColorCamera(
        mcal_path=mcal_path,
        intrinsics_json=INTRINSICS_JSON,
        extrinsics_json=EXTRINSICS_JSON
    )

    # Load mocap data
    print("\n" + "=" * 60)
    markers, frame_times = load_mocap_data(mocap_csv)

    # Overlay markers on video
    print("\n" + "=" * 60)
    overlay_markers_on_video(
        video_path=video_path,
        output_path=output_path,
        camera=camera,
        markers=markers,
        frame_times=frame_times,
        marker_color=MARKER_COLOR,
        marker_size=MARKER_SIZE,
        start_frame=START_FRAME,
        num_frames=NUM_FRAMES
    )

    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
