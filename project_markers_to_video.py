#!/usr/bin/env python3
"""
Project OptiTrack 3D markers to PrimeColor camera 2D video.

This script:
1. Loads PrimeColor camera calibration from optitrack.mcal
2. Reads 3D marker positions from mocap.csv
3. Projects markers to 2D image coordinates
4. Overlays markers on video frames
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm


class PrimeColorCamera:
    """PrimeColor camera with OptiTrack calibration parameters."""

    def __init__(self, mcal_path):
        """
        Load camera calibration from OptiTrack .mcal file.

        Args:
            mcal_path: Path to optitrack.mcal file
        """
        self.load_calibration(mcal_path)

    def load_calibration(self, mcal_path):
        """Parse .mcal XML file and extract PrimeColor camera C11764 parameters."""
        # Parse UTF-16LE encoded XML
        tree = ET.parse(mcal_path)
        root = tree.getroot()

        # Find PrimeColor camera (Serial="C11764")
        camera = None
        for cam in root.findall('.//Camera'):
            if cam.get('Serial') == 'C11764':
                camera = cam
                break

        if camera is None:
            raise ValueError("PrimeColor camera C11764 not found in calibration file")

        # Extract intrinsics (Standard Camera Model)
        intrinsic = camera.find('IntrinsicStandardCameraModel')
        cx = float(intrinsic.get('LensCenterX'))
        cy = float(intrinsic.get('LensCenterY'))
        fx = float(intrinsic.get('HorizontalFocalLength'))
        fy = float(intrinsic.get('VerticalFocalLength'))
        k1 = float(intrinsic.get('k1'))
        k2 = float(intrinsic.get('k2'))
        k3 = float(intrinsic.get('k3'))
        p1 = float(intrinsic.get('TangentialX'))
        p2 = float(intrinsic.get('TangentialY'))

        # Camera matrix K
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients (k1, k2, p1, p2, k3)
        self.dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

        # Extract extrinsics
        extrinsic = camera.find('Extrinsic')
        tx = float(extrinsic.get('X'))
        ty = float(extrinsic.get('Y'))
        tz = float(extrinsic.get('Z'))

        # Rotation matrix (row-major in XML)
        R_flat = [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(9)]
        R = np.array(R_flat).reshape(3, 3)

        # OptiTrack extrinsics: Method 4 with negative fx
        # rvec from R^T, tvec = -R^T @ T, use -fx in camera matrix
        self.rvec, _ = cv2.Rodrigues(R.T)
        self.tvec = -R.T @ np.array([tx, ty, tz])

        # Modified camera matrix with negative fx (to fix X-axis mirroring)
        self.K_negfx = np.array([
            [-fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Image resolution
        attrs = camera.find('Attributes')
        self.width = int(attrs.get('ImagerPixelWidth'))
        self.height = int(attrs.get('ImagerPixelHeight'))

        print(f"Loaded PrimeColor camera C11764:")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  Focal length: fx={fx:.2f}, fy={fy:.2f}")
        print(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
        print(f"  Camera position (world frame): [{tx:.3f}, {ty:.3f}, {tz:.3f}] meters")
        print(f"  Using Method 4 with negative fx for correct projection")

    def project_3d_to_2d(self, points_3d):
        """
        Project 3D points in OptiTrack world coordinates to 2D image pixels.
        Uses OpenCV's cv2.projectPoints with Method 4 (negative fx).

        Args:
            points_3d: Nx3 array of 3D points in millimeters (OptiTrack units)

        Returns:
            points_2d: Nx2 array of pixel coordinates
            valid_mask: N boolean array (True if point is in front of camera)
        """
        if len(points_3d) == 0:
            return np.array([]), np.array([])

        # Convert from millimeters to meters
        points_3d_m = points_3d / 1000.0

        # Use OpenCV's projectPoints with negative fx camera matrix
        points_2d, _ = cv2.projectPoints(
            points_3d_m.reshape(-1, 1, 3),
            self.rvec,
            self.tvec,
            self.K_negfx,
            self.dist
        )
        points_2d = points_2d.reshape(-1, 2)

        # Create valid mask (all points are valid since OpenCV handles projection)
        # But we should check if they're in front of camera
        # Transform to camera coords to check Z
        R, _ = cv2.Rodrigues(self.rvec)
        points_cam = (R @ points_3d_m.T).T + self.tvec.T
        valid_mask = points_cam[:, 2] > 0

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
    with open(csv_path, 'r') as f:
        # Line 1: metadata
        line1 = f.readline()

        # Line 2: Type (all "Marker")
        line2 = f.readline()

        # Line 3: Marker names
        line3 = f.readline()
        marker_names_raw = line3.strip().split(',')[2:]  # Skip first 2 columns

        # Line 4: Marker IDs
        line4 = f.readline()

        # Line 5: Parent
        line5 = f.readline()

        # Line 6: Position labels (X,Y,Z repeated)
        line6 = f.readline()

        # Line 7: Column headers (Frame, Time, X, Y, Z, ...)
        line7 = f.readline()

    # Build marker name list (every 3 columns is one marker)
    marker_names = []
    for i in range(0, len(marker_names_raw), 3):
        if i < len(marker_names_raw) and marker_names_raw[i]:
            marker_names.append(marker_names_raw[i])

    print(f"Found {len(marker_names)} markers")

    # Read data (skip header rows)
    df = pd.read_csv(csv_path, skiprows=7, header=None)

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
                              marker_color=(0, 255, 0), marker_size=5):
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

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Process each frame
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing frames")

    # Statistics
    total_markers_seen = 0
    total_markers_valid = 0
    total_markers_in_image = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get corresponding mocap frame (match by index or timestamp)
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
                        # Check if in image bounds
                        if 0 <= x < width and 0 <= y < height:
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
    """Main function."""
    # Paths
    mcal_path = Path("/Volumes/FastACIS/annotation_pipeline/optitrack.mcal")
    mocap_csv = Path("/Volumes/FastACIS/csldata/csl/mocap.csv")
    video_path = Path("/Volumes/FastACIS/csldata/video/mocap.avi")
    output_path = Path("/Volumes/FastACIS/csldata/video/mocap_with_markers.mp4")

    # Load camera calibration
    print("=" * 60)
    camera = PrimeColorCamera(mcal_path)

    # Load mocap data
    print("=" * 60)
    markers, frame_times = load_mocap_data(mocap_csv)

    # Overlay markers on video
    print("=" * 60)
    overlay_markers_on_video(
        video_path=video_path,
        output_path=output_path,
        camera=camera,
        markers=markers,
        frame_times=frame_times,
        marker_color=(0, 255, 0),  # Green
        marker_size=3
    )

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
