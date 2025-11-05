#!/usr/bin/env python3
"""
Final marker projection script using OptiTrack calibration.

Key findings:
1. OptiTrack R matrix is Camera-to-World (R_c2w)
2. OptiTrack T is camera position in world coordinates
3. OptiTrack camera: -Z points forward
4. OpenCV camera: +Z points forward
5. Solution: Use W2C transform with NEGATIVE fx in K matrix
"""

import numpy as np
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


def load_optitrack_calibration(mcal_path, camera_serial='C11764', extrinsics_json=None, intrinsics_json=None):
    """Load intrinsics and extrinsics from OptiTrack .mcal file.

    Args:
        mcal_path: Path to .mcal file (for intrinsics if intrinsics_json not provided)
        camera_serial: Camera serial number
        extrinsics_json: Optional path to JSON file with calibrated extrinsics
        intrinsics_json: Optional path to JSON file with calibrated intrinsics (multical format)
    """
    import json

    tree = ET.parse(mcal_path)
    root = tree.getroot()

    for cam in root.findall('.//Camera'):
        if cam.get('Serial') == camera_serial:
            # Image size
            attributes = cam.find('Attributes')
            width = int(attributes.get('ImagerPixelWidth'))
            height = int(attributes.get('ImagerPixelHeight'))

            # Load intrinsics
            if intrinsics_json:
                # Load from multical JSON file
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
                dist = np.array(cam_data['dist'], dtype=np.float64).flatten()

                print(f"  Using multical intrinsics from: {intrinsics_json}")
                print(f"    fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            else:
                # Load from .mcal file (original method)
                intrinsic = cam.find('.//IntrinsicStandardCameraModel')
                if intrinsic is None:
                    intrinsic = cam.find('.//Intrinsic')

                fx = float(intrinsic.get('HorizontalFocalLength'))
                fy = float(intrinsic.get('VerticalFocalLength'))
                cx = float(intrinsic.get('LensCenterX'))
                cy = float(intrinsic.get('LensCenterY'))
                k1 = float(intrinsic.get('k1', 0.0))
                k2 = float(intrinsic.get('k2', 0.0))
                k3 = float(intrinsic.get('k3', 0.0))
                p1 = float(intrinsic.get('TangentialX', 0.0))
                p2 = float(intrinsic.get('TangentialY', 0.0))
                dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

                print(f"  Using .mcal intrinsics")

            # METHOD 4: NEGATIVE fx (empirically verified correct)
            # Note: This produces Z<0 but correct 2D projections
            # The negative fx compensates for OptiTrack coordinate system
            K = np.array([[-fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

            # Load extrinsics
            if extrinsics_json:
                # Load from calibrated JSON file
                with open(extrinsics_json, 'r') as f:
                    ext_data = json.load(f)

                rvec = np.array(ext_data['rvec'], dtype=np.float64).reshape(3, 1)
                tvec = np.array(ext_data['tvec'], dtype=np.float64).reshape(3, 1)

                print(f"  Using calibrated extrinsics from: {extrinsics_json}")
                print(f"  Camera position (world): {ext_data['camera_position_world']}")

                # Check if extrinsics JSON contains intrinsics (from updated annotation tool)
                if 'intrinsics' in ext_data and not intrinsics_json:
                    print(f"  ⚠️  Found intrinsics in extrinsics JSON (source: {ext_data.get('intrinsics_source', 'unknown')})")
                    print(f"  ⚠️  But --intrinsics was specified, using that instead")
                elif 'intrinsics' in ext_data and intrinsics_json:
                    # Warn if there's a mismatch
                    intr_from_ext = ext_data['intrinsics']
                    print(f"  ℹ️  Extrinsics were calibrated using: {ext_data.get('intrinsics_source', 'unknown')} intrinsics")
                    print(f"     (fx={intr_from_ext['fx']:.1f}, fy={intr_from_ext['fy']:.1f})")
            else:
                # Load from .mcal file (original method)
                extrinsic = cam.find('Extrinsic')
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
                rvec, _ = cv2.Rodrigues(R_w2c)
                tvec = -R_w2c @ T_world

                print(f"  Using .mcal extrinsics")

            return K, dist, rvec, tvec, [width, height]

    raise ValueError(f"Camera {camera_serial} not found")


def load_all_markers(csv_path):
    """Load all markers from CSV file."""
    # Parse header to get marker names
    with open(csv_path, 'r') as f:
        for _ in range(3):
            f.readline()
        names_line = f.readline()

    names_raw = names_line.strip().split(',')[2:]
    marker_names = [names_raw[i] for i in range(0, len(names_raw), 3) if names_raw[i]]

    # Load data
    df = pd.read_csv(csv_path, skiprows=7, header=None, low_memory=False)

    all_markers = []
    for frame_idx in range(len(df)):
        frame_markers = []
        for i, name in enumerate(marker_names):
            col = 2 + i*3
            x = pd.to_numeric(df.iloc[frame_idx, col], errors='coerce')
            y = pd.to_numeric(df.iloc[frame_idx, col+1], errors='coerce')
            z = pd.to_numeric(df.iloc[frame_idx, col+2], errors='coerce')

            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                frame_markers.append([x, y, z])
            else:
                frame_markers.append([np.nan, np.nan, np.nan])

        all_markers.append(np.array(frame_markers))

    return all_markers, marker_names


def project_markers_to_frame(markers_mm, K, dist, rvec, tvec, img_size):
    """Project 3D markers to 2D image coordinates."""
    # Filter out NaN markers
    valid_mask = ~np.isnan(markers_mm[:, 0])
    if not valid_mask.any():
        return None

    valid_markers = markers_mm[valid_mask]

    # Convert mm to meters
    markers_m = valid_markers / 1000.0

    # Project
    points_2d, _ = cv2.projectPoints(
        markers_m.reshape(-1, 1, 3),
        rvec, tvec, K, dist
    )
    points_2d = points_2d.reshape(-1, 2)

    # Check which are in image bounds
    # NOTE: We don't check Z>0 because Method 4 (negative fx) produces Z<0 but correct projections
    in_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_size[0]) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_size[1])

    if not in_bounds.any():
        return None

    return points_2d[in_bounds]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Project OptiTrack markers onto PrimeColor video')
    parser.add_argument('--mcal', default='/Volumes/FastACIS/annotation_pipeline/optitrack.mcal',
                        help='Path to .mcal calibration file (for intrinsics)')
    parser.add_argument('--csv', default='/Volumes/FastACIS/csldata/csl/mocap.csv',
                        help='Path to mocap CSV file')
    parser.add_argument('--video', default='/Volumes/FastACIS/csldata/video/mocap.avi',
                        help='Path to video file')
    parser.add_argument('--output', default='/Volumes/FastACIS/csldata/video/mocap_with_markers.mp4',
                        help='Output video path')
    parser.add_argument('--intrinsics', default=None,
                        help='Path to calibrated intrinsics JSON file (optional, overrides .mcal intrinsics)')
    parser.add_argument('--extrinsics', default=None,
                        help='Path to calibrated extrinsics JSON file (optional, overrides .mcal extrinsics)')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Start frame (default: 0)')
    parser.add_argument('--num-frames', type=int, default=-1,
                        help='Number of frames to process (-1 for all)')
    parser.add_argument('--marker-size', type=int, default=3,
                        help='Marker circle radius in pixels')
    parser.add_argument('--marker-color', default='0,255,0',
                        help='Marker color in BGR (default: green)')

    args = parser.parse_args()

    print("="*70)
    print("OptiTrack Marker Projection to Video")
    print("="*70)

    # Parse marker color
    marker_color = tuple(map(int, args.marker_color.split(',')))

    # Load calibration
    print("\nLoading calibration...")
    K, dist, rvec, tvec, img_size = load_optitrack_calibration(
        args.mcal,
        extrinsics_json=args.extrinsics,
        intrinsics_json=args.intrinsics
    )
    print(f"  Camera intrinsics: fx={-K[0,0]:.2f} (negative for coord conversion), fy={K[1,1]:.2f}")
    print(f"  Image size: {img_size[0]}x{img_size[1]}")

    # Load markers
    print("\nLoading mocap data...")
    all_markers, marker_names = load_all_markers(args.csv)
    print(f"  Total frames: {len(all_markers)}")
    print(f"  Total markers: {len(marker_names)}")

    # Open video
    print("\nOpening video...")
    cap = cv2.VideoCapture(args.video)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video frames: {total_video_frames}")
    print(f"  FPS: {fps}")

    # Determine frame range
    start_frame = args.start_frame
    if args.num_frames == -1:
        num_frames = min(len(all_markers), total_video_frames) - start_frame
    else:
        num_frames = min(args.num_frames, len(all_markers) - start_frame, total_video_frames - start_frame)

    print(f"\nProcessing frames {start_frame} to {start_frame + num_frames - 1}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, tuple(img_size))

    # Set start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Process frames
    projection_count = 0
    for i in tqdm(range(num_frames), desc="Projecting markers"):
        frame_idx = start_frame + i

        ret, frame = cap.read()
        if not ret:
            print(f"\nWarning: Could not read frame {frame_idx}")
            break

        # Get markers for this frame
        markers = all_markers[frame_idx]

        # Project markers
        points_2d = project_markers_to_frame(markers, K, dist, rvec, tvec, img_size)

        if points_2d is not None:
            projection_count += 1
            # Draw markers
            for pt in points_2d:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), args.marker_size, marker_color, -1)
                cv2.circle(frame, (x, y), args.marker_size + 1, (0, 0, 0), 1)

        # Write frame
        out.write(frame)

    cap.release()
    out.release()

    print(f"\n✓ Done!")
    print(f"  Frames with projections: {projection_count}/{num_frames}")
    print(f"  Output saved to: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()
