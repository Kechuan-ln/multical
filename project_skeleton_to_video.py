#!/usr/bin/env python3
"""
Project 3D skeleton (17 joints) onto PrimeColor video using OptiTrack calibration.

This script uses the CORRECT projection method with negative fx to handle
OptiTrack's -Z forward coordinate system.

Usage:
    python project_skeleton_to_video.py \
        --mcal optitrack.mcal \
        --skeleton skeleton_joints.json \
        --video video.avi \
        --output skeleton_video.mp4 \
        --start-frame 0 \
        --num-frames -1
"""

import numpy as np
import cv2
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


def load_optitrack_calibration(mcal_path, camera_serial='C11764'):
    """
    Load camera calibration from OptiTrack .mcal file.

    Returns:
        K: Camera intrinsic matrix (3x3) with NEGATIVE fx
        dist: Distortion coefficients [k1, k2, p1, p2, k3]
        rvec: Rotation vector (World-to-Camera)
        tvec: Translation vector (World-to-Camera)
        img_size: [width, height]
    """
    tree = ET.parse(mcal_path)
    root = tree.getroot()

    for cam in root.findall('.//Camera'):
        if cam.get('Serial') == camera_serial:
            # Image size
            attributes = cam.find('Attributes')
            width = int(attributes.get('ImagerPixelWidth'))
            height = int(attributes.get('ImagerPixelHeight'))

            # Intrinsics
            intrinsic = cam.find('.//IntrinsicStandardCameraModel')
            fx = float(intrinsic.get('HorizontalFocalLength'))
            fy = float(intrinsic.get('VerticalFocalLength'))
            cx = float(intrinsic.get('LensCenterX'))
            cy = float(intrinsic.get('LensCenterY'))
            k1 = float(intrinsic.get('k1'))
            k2 = float(intrinsic.get('k2'))
            k3 = float(intrinsic.get('k3'))
            p1 = float(intrinsic.get('TangentialX'))
            p2 = float(intrinsic.get('TangentialY'))

            # CRITICAL: Use NEGATIVE fx for OptiTrack coordinate system
            K = np.array([[-fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float64)
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

            # Extrinsics
            extrinsic = cam.find('Extrinsic')
            T_world = np.array([
                float(extrinsic.get('X')),
                float(extrinsic.get('Y')),
                float(extrinsic.get('Z'))
            ])

            R_c2w = np.array([
                [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(3)],
                [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(3, 6)],
                [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(6, 9)]
            ])

            # Convert Camera-to-World to World-to-Camera
            R_w2c = R_c2w.T
            rvec, _ = cv2.Rodrigues(R_w2c)
            tvec = -R_w2c @ T_world

            print(f"  Camera intrinsics: fx={-fx:.2f} (negative for coord conversion), fy={fy:.2f}")
            print(f"  Image size: {width}x{height}")
            print(f"  Camera position (world): {T_world}")

            return K, dist, rvec, tvec, [width, height]

    raise ValueError(f"Camera with Serial='{camera_serial}' not found in {mcal_path}")


def load_skeleton_data(json_path):
    """
    Load skeleton joint data from JSON file (new format with prosthesis).

    Returns:
        metadata: Skeleton metadata
        frames_data: Frame-by-frame joint positions + prosthesis mesh
        joint_names: List of joint names
        parents: Parent indices for skeleton topology
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    frames_data_list = data['frames']

    # Convert list to dict keyed by frame number
    frames_data = {}
    for frame_info in frames_data_list:
        frame_num = frame_info['frame']
        frames_data[str(frame_num)] = frame_info

    # H36M joint names
    joint_names = [
        "Pelvis", "LHip", "RHip", "Spine1", "Neck", "Head", "Jaw",
        "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
        "LKnee", "LAnkle", "RKnee", "RAnkle"
    ]

    # H36M skeleton topology
    parents = [-1, 0, 0, 0, 3, 4, 5, 4, 7, 8, 4, 10, 11, 1, 13, 2, 15]

    print(f"  Skeleton: {len(joint_names)} joints")
    print(f"  Frames: {len(frames_data_list)}")
    if len(frames_data_list) > 0:
        print(f"  Frame range: {min(f['frame'] for f in frames_data_list)} - {max(f['frame'] for f in frames_data_list)}")
        # Check if prosthesis data exists
        has_prosthesis = frames_data_list[0].get('prosthesis_mesh') is not None
        print(f"  Has prosthesis: {has_prosthesis}")

    return metadata, frames_data, joint_names, parents


def project_skeleton_to_2d(frame_data, joint_names, K, dist, rvec, tvec, img_size):
    """
    Project 3D skeleton joints to 2D image coordinates.

    Args:
        frame_data: Single frame data with joint positions (new format)
        joint_names: List of joint names
        K, dist, rvec, tvec: Camera parameters
        img_size: [width, height]

    Returns:
        joints_2d: Dict {joint_name: [x, y]} for valid projections
        None if no valid joints
    """
    # Collect 3D joint positions (in mm)
    joints_3d = []
    valid_joint_names = []

    skeleton_data = frame_data['skeleton']
    for joint_name in joint_names:
        if joint_name in skeleton_data:
            joint_info = skeleton_data[joint_name]
            if joint_info['valid']:
                pos = joint_info['position']
                joints_3d.append(pos)
                valid_joint_names.append(joint_name)

    if len(joints_3d) == 0:
        return None

    # Convert to numpy array and meters
    joints_3d = np.array(joints_3d, dtype=np.float64)  # (N, 3) in mm
    joints_3d_m = joints_3d / 1000.0  # Convert to meters

    # Project using OpenCV
    points_2d, _ = cv2.projectPoints(
        joints_3d_m.reshape(-1, 1, 3),
        rvec, tvec, K, dist
    )
    points_2d = points_2d.reshape(-1, 2)

    # Filter: only keep points within image boundaries
    # NOTE: Do NOT check Z > 0 (negative fx causes Z < 0)
    width, height = img_size
    in_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)

    # Create result dict
    joints_2d = {}
    for i, (joint_name, in_bound) in enumerate(zip(valid_joint_names, in_bounds)):
        if in_bound:
            joints_2d[joint_name] = points_2d[i]

    return joints_2d if len(joints_2d) > 0 else None


def project_prosthesis_to_2d(frame_data, K, dist, rvec, tvec, img_size):
    """
    Project prosthesis mesh vertices to 2D.

    Args:
        frame_data: Frame data with prosthesis_mesh
        K, dist, rvec, tvec: Camera parameters
        img_size: [width, height]

    Returns:
        vertices_2d: Projected 2D vertices (N, 2)
        None if no prosthesis data
    """
    mesh_vertices = frame_data.get('prosthesis_mesh')
    if mesh_vertices is None or len(mesh_vertices) == 0:
        return None

    # Convert to numpy array and meters
    vertices_3d = np.array(mesh_vertices, dtype=np.float64)  # (N, 3) in mm
    vertices_3d_m = vertices_3d / 1000.0  # Convert to meters

    # Project using OpenCV
    vertices_2d, _ = cv2.projectPoints(
        vertices_3d_m.reshape(-1, 1, 3),
        rvec, tvec, K, dist
    )
    vertices_2d = vertices_2d.reshape(-1, 2)

    return vertices_2d


def get_bone_color(parent_idx, child_idx, joint_names):
    """Get bone color based on body part."""
    parent_name = joint_names[parent_idx]
    child_name = joint_names[child_idx]

    if 'Spine' in parent_name or 'Spine' in child_name or 'Neck' in parent_name or 'Neck' in child_name:
        return (255, 0, 0)  # Blue
    elif 'Head' in child_name or 'Jaw' in child_name:
        return (255, 0, 255)  # Magenta
    elif 'L' in child_name and ('Shoulder' in child_name or 'Elbow' in child_name or 'Wrist' in child_name):
        return (0, 255, 0)  # Green
    elif 'R' in child_name and ('Shoulder' in child_name or 'Elbow' in child_name or 'Wrist' in child_name):
        return (0, 0, 255)  # Red
    elif 'L' in child_name and ('Hip' in child_name or 'Knee' in child_name or 'Ankle' in child_name):
        return (255, 255, 0)  # Cyan
    elif 'R' in child_name and ('Hip' in child_name or 'Knee' in child_name or 'Ankle' in child_name):
        return (0, 165, 255)  # Orange
    else:
        return (128, 128, 128)  # Gray


def draw_skeleton_on_frame(frame, joints_2d, joint_names, parents, line_thickness=2, point_radius=4):
    """
    Draw skeleton (bones + joints) on video frame.

    Args:
        frame: Video frame (BGR image)
        joints_2d: Dict {joint_name: [x, y]}
        joint_names: List of joint names
        parents: Parent indices
        line_thickness: Bone line thickness
        point_radius: Joint point radius

    Returns:
        Annotated frame
    """
    # Draw bones (lines)
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:  # Skip root
            continue

        parent_name = joint_names[parent_idx]
        child_name = joint_names[child_idx]

        if parent_name in joints_2d and child_name in joints_2d:
            pt1 = tuple(joints_2d[parent_name].astype(int))
            pt2 = tuple(joints_2d[child_name].astype(int))
            color = get_bone_color(parent_idx, child_idx, joint_names)

            cv2.line(frame, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    # Draw joints (circles)
    for joint_name, pos in joints_2d.items():
        center = tuple(pos.astype(int))
        cv2.circle(frame, center, point_radius, (255, 255, 255), -1, cv2.LINE_AA)  # White fill
        cv2.circle(frame, center, point_radius, (0, 0, 0), 1, cv2.LINE_AA)  # Black outline

    return frame


def draw_prosthesis_on_frame(frame, vertices_2d, color=(0, 255, 0), alpha=0.4):
    """
    Draw prosthesis mesh on video frame.

    Args:
        frame: Video frame (BGR image)
        vertices_2d: Projected 2D vertices (N, 2)
        color: Fill color (BGR)
        alpha: Transparency (0-1)

    Returns:
        Annotated frame
    """
    if vertices_2d is None or len(vertices_2d) == 0:
        return frame

    # Create overlay
    overlay = frame.copy()

    # Draw triangles
    n_triangles = len(vertices_2d) // 3
    width, height = frame.shape[1], frame.shape[0]

    for i in range(n_triangles):
        pts = vertices_2d[i*3:(i+1)*3].astype(np.int32)

        # Check if all points are within frame bounds
        if all(0 <= pt[0] < width and 0 <= pt[1] < height for pt in pts):
            cv2.fillPoly(overlay, [pts], color)

    # Blend with original frame
    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    return frame


def process_video(
    mcal_path,
    skeleton_json,
    video_path,
    output_path,
    start_frame=0,
    num_frames=-1,
    camera_serial='C11764',
    line_thickness=2,
    point_radius=4,
    show_frame_info=True
):
    """
    Main processing function: project skeleton onto video.
    """
    print("="*70)
    print("Skeleton Projection to Video")
    print("="*70)

    # Load calibration
    print("\nLoading calibration...")
    K, dist, rvec, tvec, img_size = load_optitrack_calibration(mcal_path, camera_serial)

    # Load skeleton data
    print("\nLoading skeleton data...")
    metadata, frames_data, joint_names, parents = load_skeleton_data(skeleton_json)

    # Open video
    print("\nOpening video...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video frames: {video_frames}")
    print(f"  FPS: {video_fps}")
    print(f"  Resolution: {video_width}x{video_height}")

    # Check resolution match
    if [video_width, video_height] != img_size:
        print(f"  WARNING: Video resolution doesn't match calibration!")
        print(f"    Video: {video_width}x{video_height}")
        print(f"    Calibration: {img_size[0]}x{img_size[1]}")

    # Determine frame range
    skeleton_frames = len(frames_data)
    max_frames = min(video_frames, skeleton_frames)

    if num_frames < 0:
        num_frames = max_frames - start_frame
    else:
        num_frames = min(num_frames, max_frames - start_frame)

    end_frame = start_frame + num_frames

    print(f"\nProcessing frames {start_frame} to {end_frame}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (video_width, video_height))

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Process frames
    frames_with_skeleton = 0
    pbar = tqdm(total=num_frames, desc="Projecting skeleton")

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"\nWarning: Failed to read frame {frame_idx}")
            break

        # Get skeleton data for this frame
        skeleton_frame_idx = frame_idx  # Assuming 1:1 correspondence
        if str(skeleton_frame_idx) in frames_data:
            frame_data = frames_data[str(skeleton_frame_idx)]

            # Project prosthesis to 2D first (draw as background)
            vertices_2d = project_prosthesis_to_2d(
                frame_data, K, dist, rvec, tvec, img_size
            )
            if vertices_2d is not None:
                frame = draw_prosthesis_on_frame(frame, vertices_2d)

            # Project skeleton to 2D
            joints_2d = project_skeleton_to_2d(
                frame_data, joint_names, K, dist, rvec, tvec, img_size
            )

            if joints_2d is not None:
                # Draw skeleton on frame
                frame = draw_skeleton_on_frame(
                    frame, joints_2d, joint_names, parents,
                    line_thickness, point_radius
                )
                frames_with_skeleton += 1

                # Optionally show frame info
                if show_frame_info:
                    has_pros = vertices_2d is not None
                    info_text = f"Frame {frame_idx} | Joints: {len(joints_2d)}/17"
                    if has_pros:
                        info_text += " | Prosthesis: ✓"
                    cv2.putText(frame, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write frame
        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"\n✓ Done!")
    print(f"  Frames with skeleton: {frames_with_skeleton}/{num_frames}")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Project 3D skeleton onto video using OptiTrack calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire video
  python project_skeleton_to_video.py --skeleton skeleton_joints.json --video video.avi

  # Process specific frame range
  python project_skeleton_to_video.py --start-frame 1000 --num-frames 500

  # Custom styling
  python project_skeleton_to_video.py --line-thickness 3 --point-radius 5
        """
    )

    parser.add_argument('--mcal', type=str, default='optitrack.mcal',
                        help='OptiTrack .mcal calibration file')
    parser.add_argument('--skeleton', type=str, default='skeleton_joints.json',
                        help='Skeleton JSON file (from markers_to_skeleton.py)')
    parser.add_argument('--video', type=str, default='video.avi',
                        help='Input video file')
    parser.add_argument('--output', type=str, default='skeleton_video.mp4',
                        help='Output video file')
    parser.add_argument('--camera-serial', type=str, default='C11764',
                        help='PrimeColor camera serial number')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Start frame index')
    parser.add_argument('--num-frames', type=int, default=-1,
                        help='Number of frames to process (-1 = all)')
    parser.add_argument('--line-thickness', type=int, default=2,
                        help='Skeleton bone line thickness (pixels)')
    parser.add_argument('--point-radius', type=int, default=4,
                        help='Joint point radius (pixels)')
    parser.add_argument('--no-frame-info', action='store_true',
                        help='Do not show frame info overlay')

    args = parser.parse_args()

    # Validate paths
    mcal_path = Path(args.mcal)
    skeleton_path = Path(args.skeleton)
    video_path = Path(args.video)

    if not mcal_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {mcal_path}")
    if not skeleton_path.exists():
        raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Run processing
    process_video(
        mcal_path=mcal_path,
        skeleton_json=skeleton_path,
        video_path=video_path,
        output_path=args.output,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        camera_serial=args.camera_serial,
        line_thickness=args.line_thickness,
        point_radius=args.point_radius,
        show_frame_info=not args.no_frame_info
    )


if __name__ == '__main__':
    main()
