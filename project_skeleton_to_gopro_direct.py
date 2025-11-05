#!/usr/bin/env python3
"""
Project skeleton to GoPro using OptiTrack-style extrinsics (with negative fx).
"""

import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm

# H36M skeleton connections
H36M_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),
    (3, 4), (4, 5), (5, 6),
    (4, 7), (7, 8), (8, 9),
    (4, 10), (10, 11), (11, 12),
    (1, 13), (13, 14),
    (2, 15), (15, 16),
]

H36M_JOINT_NAMES = [
    "Pelvis", "LHip", "RHip", "Spine1", "Neck", "Head", "Jaw",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
    "LKnee", "LAnkle", "RKnee", "RAnkle"
]


def load_gopro_calibration(calib_json, gopro_extrinsics_json):
    """Load GoPro calibration."""
    # Load intrinsics
    with open(calib_json, 'r') as f:
        data = json.load(f)

    gopro_cam = data['cameras']['cam1']
    K_gopro = np.array(gopro_cam['K'], dtype=np.float64)
    dist_gopro = np.array(gopro_cam['dist'], dtype=np.float64).flatten()
    img_size = gopro_cam['image_size']

    # Load OptiTrack-style extrinsics
    with open(gopro_extrinsics_json, 'r') as f:
        ext_data = json.load(f)

    rvec = np.array(ext_data['rvec'], dtype=np.float64).reshape(3, 1)
    tvec = np.array(ext_data['tvec'], dtype=np.float64).reshape(3, 1)

    # Use NEGATIVE fx for OptiTrack coordinate system
    fx = K_gopro[0, 0]
    fy = K_gopro[1, 1]
    cx = K_gopro[0, 2]
    cy = K_gopro[1, 2]

    K_gopro_optitrack = np.array([[-fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    return K_gopro_optitrack, dist_gopro, rvec, tvec, img_size


def load_skeleton_data(json_path):
    """Load skeleton data."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    frames_dict = {}
    for frame_info in frames:
        frame_num = frame_info['frame']
        frames_dict[frame_num] = frame_info

    return frames_dict


def project_skeleton_to_2d(frame_data, K, dist, rvec, tvec, img_size):
    """Project skeleton joints to 2D."""
    skeleton = frame_data['skeleton']

    joints_3d = []
    joint_valid = []

    for joint_name in H36M_JOINT_NAMES:
        if joint_name in skeleton:
            joint_info = skeleton[joint_name]
            if joint_info['valid']:
                pos = joint_info['position']
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

    # Project
    joints_2d, _ = cv2.projectPoints(
        joints_3d.reshape(-1, 1, 3),
        rvec, tvec, K, dist
    )
    joints_2d = joints_2d.reshape(-1, 2)

    return joints_2d, joint_valid


def project_prosthesis_to_2d(frame_data, K, dist, rvec, tvec):
    """Project prosthesis mesh to 2D."""
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
                           line_thickness=3, point_radius=6):
    """Draw skeleton."""
    h, w = img_size[1], img_size[0]

    # Draw bones
    for joint_i, joint_j in H36M_CONNECTIONS:
        if joint_valid[joint_i] and joint_valid[joint_j]:
            pt1 = joints_2d[joint_i]
            pt2 = joints_2d[joint_j]

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
                      point_radius + 1, (0, 0, 0), 2)

    return frame


def draw_prosthesis_on_frame(frame, vertices_2d, img_size,
                             color=(0, 255, 255), radius=3):
    """Draw prosthesis."""
    if vertices_2d is None or len(vertices_2d) == 0:
        return frame

    h, w = img_size[1], img_size[0]

    for pt in vertices_2d:
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--calibration', required=True)
    parser.add_argument('--gopro-extrinsics', required=True)
    parser.add_argument('--skeleton', required=True)
    parser.add_argument('--gopro-video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mocap-fps', type=float, default=120.0)
    parser.add_argument('--sync-offset', type=float, default=0.0)
    parser.add_argument('--no-prosthesis', action='store_true')

    args = parser.parse_args()

    print("="*70)
    print("GoPro Skeleton Projection (OptiTrack Coordinate System)")
    print("="*70)

    # Load calibration
    print("\nLoading calibration...")
    K, dist, rvec, tvec, img_size = load_gopro_calibration(
        args.calibration, args.gopro_extrinsics
    )
    print(f"  GoPro K (with negative fx): fx={-K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"  Image size: {img_size[0]}x{img_size[1]}")

    # Load skeleton
    print("\nLoading skeleton...")
    frames_dict = load_skeleton_data(args.skeleton)
    available_frames = sorted(frames_dict.keys())
    print(f"  Frames: {len(frames_dict)} ({available_frames[0]}-{available_frames[-1]})")

    # Open video
    print("\nOpening video...")
    cap = cv2.VideoCapture(args.gopro_video)
    gopro_fps = cap.get(cv2.CAP_PROP_FPS)
    gopro_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {gopro_fps}")
    print(f"  Frames: {gopro_frames}")

    # Calculate frame range
    first_mocap = available_frames[0]
    last_mocap = available_frames[-1]
    first_time = first_mocap / args.mocap_fps + args.sync_offset
    last_time = last_mocap / args.mocap_fps + args.sync_offset
    start_frame = int(first_time * gopro_fps)
    end_frame = int(last_time * gopro_fps) + 1

    print(f"\nOutput range:")
    print(f"  Mocap {first_mocap}-{last_mocap}")
    print(f"  GoPro frames {start_frame}-{end_frame} (time {first_time:.2f}s-{last_time:.2f}s)")
    print(f"  Total: {end_frame - start_frame} frames")

    # Build frame mapping
    gopro_to_mocap = {}
    for gf in range(start_frame, end_frame):
        gt = gf / gopro_fps
        mt = gt - args.sync_offset
        mf = int(mt * args.mocap_fps + 0.5)
        if mf in frames_dict:
            gopro_to_mocap[gf] = mf

    print(f"  Mapped: {len(gopro_to_mocap)} frames")

    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, gopro_fps, tuple(img_size))

    # Process
    projection_count = 0
    print(f"\nProcessing...")
    for gf in tqdm(range(start_frame, end_frame), desc="Rendering"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, gf)
        ret, frame = cap.read()
        if not ret:
            break

        if gf in gopro_to_mocap:
            mf = gopro_to_mocap[gf]
            frame_data = frames_dict[mf]

            # Project skeleton
            joints_2d, joint_valid = project_skeleton_to_2d(
                frame_data, K, dist, rvec, tvec, img_size
            )

            if joints_2d is not None:
                projection_count += 1
                frame = draw_skeleton_on_frame(frame, joints_2d, joint_valid, img_size)

                if not args.no_prosthesis:
                    vertices_2d = project_prosthesis_to_2d(frame_data, K, dist, rvec, tvec)
                    frame = draw_prosthesis_on_frame(frame, vertices_2d, img_size)

        out.write(frame)

    cap.release()
    out.release()

    print(f"\n✓ Done!")
    print(f"  Total frames: {end_frame - start_frame}")
    print(f"  With skeleton: {projection_count}")
    print(f"  Saved: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()
