#!/usr/bin/env python3
"""
Project 3D skeleton + prosthesis onto video using OptiTrack calibration.

Based on project_markers_final.py with skeleton rendering.
"""

import numpy as np
import cv2
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


# H36M skeleton connections (17 joints)
H36M_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),  # Pelvis to hips and spine
    (3, 4), (4, 5), (5, 6),  # Spine to neck to head to jaw
    (4, 7), (7, 8), (8, 9),  # Neck to left shoulder to elbow to wrist
    (4, 10), (10, 11), (11, 12),  # Neck to right shoulder to elbow to wrist
    (1, 13), (13, 14),  # Left hip to knee to ankle
    (2, 15), (15, 16),  # Right hip to knee to ankle
]

H36M_JOINT_NAMES = [
    "Pelvis", "LHip", "RHip", "Spine1", "Neck", "Head", "Jaw",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
    "LKnee", "LAnkle", "RKnee", "RAnkle"
]


def load_optitrack_calibration(mcal_path, camera_serial='C11764', extrinsics_json=None, intrinsics_json=None):
    """Load intrinsics and extrinsics from OptiTrack .mcal file.

    Args:
        mcal_path: Path to .mcal file (for intrinsics if intrinsics_json not provided)
        camera_serial: Camera serial number
        extrinsics_json: Optional path to JSON file with calibrated extrinsics
        intrinsics_json: Optional path to JSON file with calibrated intrinsics (multical format)
    """
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

            # NEGATIVE fx for OptiTrack coordinate system (-Z forward)
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


def load_skeleton_data(json_path):
    """Load skeleton + prosthesis data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    frames = data['frames']

    # Convert list to dict keyed by frame number for quick access
    frames_dict = {}
    for frame_info in frames:
        frame_num = frame_info['frame']
        frames_dict[frame_num] = frame_info

    return metadata, frames_dict


def project_skeleton_to_2d(frame_data, K, dist, rvec, tvec, img_size):
    """Project 3D skeleton joints to 2D image coordinates."""
    skeleton = frame_data['skeleton']

    # Extract joint positions
    joints_3d = []
    joint_valid = []

    for joint_name in H36M_JOINT_NAMES:
        if joint_name in skeleton:
            joint_info = skeleton[joint_name]
            if joint_info['valid']:
                pos = joint_info['position']  # in mm
                joints_3d.append(pos)
                joint_valid.append(True)
            else:
                joints_3d.append([0, 0, 0])
                joint_valid.append(False)
        else:
            joints_3d.append([0, 0, 0])
            joint_valid.append(False)

    if not any(joint_valid):
        return None, None

    # Convert mm to meters
    joints_3d = np.array(joints_3d, dtype=np.float64) / 1000.0

    # Project all joints (including invalid ones, will filter later)
    joints_2d, _ = cv2.projectPoints(
        joints_3d.reshape(-1, 1, 3),
        rvec, tvec, K, dist
    )
    joints_2d = joints_2d.reshape(-1, 2)

    return joints_2d, joint_valid


def project_prosthesis_to_2d(frame_data, K, dist, rvec, tvec):
    """Project prosthesis mesh vertices to 2D."""
    prosthesis = frame_data.get('prosthesis_transform')
    if not prosthesis or not prosthesis.get('valid'):
        return None

    mesh_vertices = frame_data.get('prosthesis_mesh')
    if not mesh_vertices:
        return None

    # Convert mm to meters
    vertices_3d = np.array(mesh_vertices, dtype=np.float64) / 1000.0

    # Project
    vertices_2d, _ = cv2.projectPoints(
        vertices_3d.reshape(-1, 1, 3),
        rvec, tvec, K, dist
    )
    vertices_2d = vertices_2d.reshape(-1, 2)

    return vertices_2d


def draw_skeleton_on_frame(frame, joints_2d, joint_valid, img_size,
                           line_thickness=2, point_radius=4):
    """Draw skeleton bones and joints on frame."""
    h, w = img_size[1], img_size[0]

    # Draw bones
    for joint_i, joint_j in H36M_CONNECTIONS:
        if joint_valid[joint_i] and joint_valid[joint_j]:
            pt1 = joints_2d[joint_i]
            pt2 = joints_2d[joint_j]

            # Check if both points are in bounds
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(frame,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        (0, 255, 0), line_thickness)

    # Draw joints
    for i, (pt, valid) in enumerate(zip(joints_2d, joint_valid)):
        if valid and 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(frame, (int(pt[0]), int(pt[1])),
                      point_radius, (255, 0, 0), -1)
            cv2.circle(frame, (int(pt[0]), int(pt[1])),
                      point_radius + 1, (0, 0, 0), 1)

    return frame


def draw_prosthesis_on_frame(frame, vertices_2d, img_size,
                             color=(0, 255, 255), alpha=0.3):
    """Draw prosthesis mesh on frame."""
    if vertices_2d is None or len(vertices_2d) == 0:
        return frame

    h, w = img_size[1], img_size[0]
    overlay = frame.copy()

    # Draw mesh as triangles (assuming vertices are triangulated)
    n_triangles = len(vertices_2d) // 3
    for i in range(n_triangles):
        pts = vertices_2d[i*3:(i+1)*3]

        # Check if all points are in bounds
        valid = all(0 <= pt[0] < w and 0 <= pt[1] < h for pt in pts)
        if valid:
            pts = pts.astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)

    # Blend overlay with original frame
    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Project 3D skeleton + prosthesis onto video using OptiTrack calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with calibrated extrinsics
  python project_skeleton_with_prosthesis.py \\
    --mcal Primecolor.mcal \\
    --skeleton skeleton_with_prosthesis.json \\
    --video primecolor.avi \\
    --extrinsics extrinsics_calibrated.json \\
    --output skeleton_video.mp4 \\
    --start-frame 3820 \\
    --num-frames 62
        """
    )

    parser.add_argument('--mcal', required=True,
                        help='Path to .mcal calibration file')
    parser.add_argument('--skeleton', required=True,
                        help='Path to skeleton JSON file (from markers_to_skeleton_with_prosthesis.py)')
    parser.add_argument('--video', required=True,
                        help='Path to input video file')
    parser.add_argument('--output', required=True,
                        help='Path to output video file')
    parser.add_argument('--extrinsics', default=None,
                        help='Path to calibrated extrinsics JSON (optional)')
    parser.add_argument('--intrinsics', default=None,
                        help='Path to calibrated intrinsics JSON (optional)')
    parser.add_argument('--camera-serial', default='C11764',
                        help='PrimeColor camera serial number (default: C11764)')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Start frame index (default: 0)')
    parser.add_argument('--num-frames', type=int, default=-1,
                        help='Number of frames to process (-1 for all)')
    parser.add_argument('--line-thickness', type=int, default=2,
                        help='Skeleton bone line thickness in pixels')
    parser.add_argument('--point-radius', type=int, default=4,
                        help='Joint point radius in pixels')
    parser.add_argument('--prosthesis-alpha', type=float, default=0.3,
                        help='Prosthesis mesh transparency (0-1)')
    parser.add_argument('--no-prosthesis', action='store_true',
                        help='Do not draw prosthesis mesh')

    args = parser.parse_args()

    print("="*70)
    print("Skeleton + Prosthesis Projection to Video")
    print("="*70)

    # Load calibration
    print("\nLoading calibration...")
    K, dist, rvec, tvec, img_size = load_optitrack_calibration(
        args.mcal,
        camera_serial=args.camera_serial,
        extrinsics_json=args.extrinsics,
        intrinsics_json=args.intrinsics
    )
    print(f"  Camera intrinsics: fx={-K[0,0]:.2f} (negative for coord conversion), fy={K[1,1]:.2f}")
    print(f"  Image size: {img_size[0]}x{img_size[1]}")

    # Load skeleton data
    print("\nLoading skeleton data...")
    metadata, frames_dict = load_skeleton_data(args.skeleton)
    print(f"  Total frames in skeleton data: {len(frames_dict)}")
    available_frames = sorted(frames_dict.keys())
    print(f"  Frame range: {available_frames[0]} - {available_frames[-1]}")

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
        num_frames = total_video_frames - start_frame
    else:
        num_frames = min(args.num_frames, total_video_frames - start_frame)

    print(f"\nProcessing video frames {start_frame} to {start_frame + num_frames - 1}")
    print(f"  (These correspond to mocap frames {start_frame} to {start_frame + num_frames - 1})")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, tuple(img_size))

    # Set start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Process frames
    projection_count = 0
    for i in tqdm(range(num_frames), desc="Projecting skeleton"):
        frame_idx = start_frame + i

        ret, frame = cap.read()
        if not ret:
            print(f"\nWarning: Could not read frame {frame_idx}")
            break

        # Get skeleton data for this mocap frame
        if frame_idx in frames_dict:
            frame_data = frames_dict[frame_idx]

            # Project skeleton
            joints_2d, joint_valid = project_skeleton_to_2d(
                frame_data, K, dist, rvec, tvec, img_size
            )

            if joints_2d is not None:
                projection_count += 1

                # Draw skeleton
                frame = draw_skeleton_on_frame(
                    frame, joints_2d, joint_valid, img_size,
                    line_thickness=args.line_thickness,
                    point_radius=args.point_radius
                )

                # Draw prosthesis
                if not args.no_prosthesis:
                    vertices_2d = project_prosthesis_to_2d(
                        frame_data, K, dist, rvec, tvec
                    )
                    frame = draw_prosthesis_on_frame(
                        frame, vertices_2d, img_size,
                        alpha=args.prosthesis_alpha
                    )

        # Write frame
        out.write(frame)

    cap.release()
    out.release()

    print(f"\n✓ Done!")
    print(f"  Frames with skeleton projections: {projection_count}/{num_frames}")
    print(f"  Output saved to: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()
