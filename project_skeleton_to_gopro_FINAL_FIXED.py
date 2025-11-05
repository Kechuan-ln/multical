#!/usr/bin/env python3
"""
Final GoPro skeleton projection with coordinate system conversion.

Key insight: OptiTrack uses -Z forward, but primecolor_to_cam1 was calibrated
in standard camera coordinates (+Z forward). Must flip coordinate system.
"""

import numpy as np
import cv2
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# Coordinate system flip: OptiTrack (-Z forward) → Standard (+Z forward)
# Based on comprehensive testing with inverse transform: YZ flip [1,-1,-1] is optimal.
# Achieved 9/10 score: correct Z>0, upright, left-right, size, and position.
R_OPTI_TO_STD = np.array([
    [1, 0, 0],   # Keep X (no flip)
    [0, -1, 0],  # Flip Y (corrects upside-down)
    [0, 0, -1],  # Flip Z (ensures Z > 0)
], dtype=np.float64)

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


def load_calibration_and_extrinsics(calib_json, mcal_path, extrinsics_json=None):
    """
    Load all calibration data.

    Args:
        calib_json: Path to calibration.json (contains PrimeColor→GoPro extrinsics)
        mcal_path: Path to .mcal file (contains Mocap→PrimeColor extrinsics, used if extrinsics_json is None)
        extrinsics_json: Optional path to custom extrinsics JSON (contains Mocap→PrimeColor extrinsics)
    """
    # Load calibration.json
    with open(calib_json, 'r') as f:
        calib = json.load(f)

    # GoPro intrinsics (auto-detect cam1 or cam4)
    gopro_cam_key = 'cam1' if 'cam1' in calib['cameras'] else 'cam4'
    gopro_cam = calib['cameras'][gopro_cam_key]
    K_gopro = np.array(gopro_cam['K'], dtype=np.float64)
    dist_gopro = np.array(gopro_cam['dist'], dtype=np.float64).flatten()
    img_size = gopro_cam['image_size']

    # PrimeColor → GoPro transform (in standard coords)
    p2g_key = 'primecolor_to_cam1' if 'primecolor_to_cam1' in calib['camera_base2cam'] else 'primecolor_to_cam4'
    p2g = calib['camera_base2cam'][p2g_key]
    R_p2g_std = np.array(p2g['R'], dtype=np.float64)
    T_p2g_std = np.array(p2g['T'], dtype=np.float64)

    print(f"  Using camera: {gopro_cam_key}")
    print(f"  Using transform: {p2g_key}")

    # Load Mocap → PrimeColor extrinsics
    if extrinsics_json is not None:
        # Load from custom JSON file
        print(f"  Loading Mocap→PrimeColor from custom JSON: {extrinsics_json}")
        with open(extrinsics_json, 'r') as f:
            extr_data = json.load(f)

        R_w2p_opti = np.array(extr_data['rotation_matrix'], dtype=np.float64)
        T_w2p_opti = np.array(extr_data['tvec'], dtype=np.float64).flatten()

        print(f"  ✓ Loaded custom extrinsics")
    else:
        # Load from .mcal file (OptiTrack calibration)
        print(f"  Loading Mocap→PrimeColor from .mcal: {mcal_path}")
        tree = ET.parse(mcal_path)
        root = tree.getroot()

        for cam in root.findall('.//Camera'):
            if cam.get('Serial') == 'C11764':
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

                R_w2p_opti = R_c2w.T
                T_w2p_opti = -R_w2p_opti @ T_world

                print(f"  ✓ Loaded OptiTrack extrinsics")
                break
        else:
            raise ValueError("PrimeColor camera not found in .mcal")

    return K_gopro, dist_gopro, img_size, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std


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


def project_3d_to_gopro(points_3d_world_opti, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std, K_gopro, dist_gopro):
    """
    Project 3D OptiTrack world points to GoPro 2D.

    Pipeline:
    1. OptiTrack world → PrimeColor OptiTrack cam
    2. Flip coordinate system → PrimeColor standard cam
    3. PrimeColor standard → GoPro standard
    4. Project to 2D
    """
    # Step 1: OptiTrack world → PrimeColor OptiTrack cam
    points_prime_opti = (R_w2p_opti @ points_3d_world_opti.T).T + T_w2p_opti.reshape(1, 3)

    # Step 2: Flip coordinate system
    points_prime_std = (R_OPTI_TO_STD @ points_prime_opti.T).T

    # Step 3: PrimeColor standard → GoPro standard
# Step 3: PrimeColor standard → GoPro standard
    #
    # --- BEGIN MODIFICATION ---
    #
    # 假设: T_p2g (primecolor_to_cam1) 实际上定义的是 T_gopro_to_prime
    # 我们需要应用它的逆变换 (T_prime_to_gopro) 才能得到正确的位置。
    #
    # R_inv = R^T
    R_inv_p2g = R_p2g_std.T
    # T_inv = -R_inv * T
    T_inv_p2g = -R_inv_p2g @ T_p2g_std

    # 应用逆变换
    points_gopro_std = (R_inv_p2g @ points_prime_std.T).T + T_inv_p2g.reshape(1, 3)
    #
    # --- END MODIFICATION ---
    # Step 4: Project (standard projection with positive fx)
    rvec_identity = np.zeros(3, dtype=np.float64)
    tvec_identity = np.zeros(3, dtype=np.float64)

    points_2d, _ = cv2.projectPoints(
        points_gopro_std.reshape(-1, 1, 3),
        rvec_identity, tvec_identity,
        K_gopro, dist_gopro
    )

    return points_2d.reshape(-1, 2), points_gopro_std


def project_skeleton_to_2d(frame_data, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std, K_gopro, dist_gopro):
    """Project skeleton joints."""
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
    joints_2d, joints_gopro = project_3d_to_gopro(
        joints_3d, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std, K_gopro, dist_gopro
    )

    # Filter out points behind camera (Z < 0 in GoPro standard coords)
    for i in range(len(joint_valid)):
        if joint_valid[i] and joints_gopro[i, 2] < 0:
            joint_valid[i] = False

    return joints_2d, joint_valid


def project_prosthesis_to_2d(frame_data, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std, K_gopro, dist_gopro):
    """Project prosthesis mesh."""
    prosthesis = frame_data.get('prosthesis_transform')
    if not prosthesis or not prosthesis.get('valid'):
        return None

    mesh_vertices = frame_data.get('prosthesis_mesh')
    if not mesh_vertices:
        return None

    # Convert mm to meters
    vertices_3d = np.array(mesh_vertices, dtype=np.float64) / 1000.0

    # Project
    vertices_2d, vertices_gopro = project_3d_to_gopro(
        vertices_3d, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std, K_gopro, dist_gopro
    )

    # Filter behind camera
    valid_mask = vertices_gopro[:, 2] >= 0
    if not valid_mask.any():
        return None

    return vertices_2d[valid_mask]


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
    parser = argparse.ArgumentParser(description='GoPro projection with coordinate system fix')

    parser.add_argument('--calibration', required=True,
                        help='Path to calibration.json (PrimeColor→GoPro extrinsics)')
    parser.add_argument('--mcal', required=True,
                        help='Path to .mcal file (Mocap→PrimeColor extrinsics, used if --extrinsics-json not provided)')
    parser.add_argument('--extrinsics-json', default=None,
                        help='Optional: Path to custom extrinsics JSON (Mocap→PrimeColor extrinsics)')
    parser.add_argument('--skeleton', required=True)
    parser.add_argument('--gopro-video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mocap-fps', type=float, default=120.0)
    parser.add_argument('--sync-offset', type=float, default=0.0)
    parser.add_argument('--no-prosthesis', action='store_true')

    args = parser.parse_args()

    print("="*70)
    print("GoPro Skeleton Projection (WITH COORDINATE SYSTEM FIX)")
    print("="*70)

    # Load calibration
    print("\nLoading calibration...")
    K_gopro, dist_gopro, img_size, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std = \
        load_calibration_and_extrinsics(args.calibration, args.mcal, args.extrinsics_json)
    print(f"  GoPro: fx={K_gopro[0,0]:.2f}, fy={K_gopro[1,1]:.2f}")
    print(f"  Image size: {img_size[0]}x{img_size[1]}")
    print(f"  ✓ Coordinate system conversion enabled")

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

    # Calculate timeline
    first_mocap = available_frames[0]
    last_mocap = available_frames[-1]

    # Convert mocap frames to mocap time
    first_mocap_time = first_mocap / args.mocap_fps
    last_mocap_time = last_mocap / args.mocap_fps

    # Convert mocap time to GoPro time using sync offset
    first_time = first_mocap_time + args.sync_offset
    last_time = last_mocap_time + args.sync_offset
    start_frame = max(0, int(first_time * gopro_fps))
    end_frame = int(last_time * gopro_fps) + 1
    actual_start_time = start_frame / gopro_fps
    last_render_frame = max(start_frame, end_frame - 1)
    actual_end_time = last_render_frame / gopro_fps

    print(f"\nOutput range:")
    print(f"  Mocap {first_mocap}-{last_mocap} (time {first_mocap_time:.2f}s-{last_mocap_time:.2f}s)")
    print(f"  GoPro frames {start_frame}-{end_frame} (time {actual_start_time:.2f}s-{actual_end_time:.2f}s)")

    # Build frame mapping
    gopro_to_mocap = {}
    for gf in range(start_frame, end_frame):
        gt = gf / gopro_fps  # GoPro absolute time
        mt = gt - args.sync_offset  # Mocap absolute time
        if mt < 0:
            continue
        mocap_frame_num = int(mt * args.mocap_fps + 0.5)  # Mocap absolute frame
        if first_mocap <= mocap_frame_num <= last_mocap and mocap_frame_num in frames_dict:
            gopro_to_mocap[gf] = mocap_frame_num

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

        mocap_frame = gopro_to_mocap.get(gf)
        if mocap_frame is not None:
            frame_data = frames_dict[mocap_frame]

            # Project skeleton
            joints_2d, joint_valid = project_skeleton_to_2d(
                frame_data, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std, K_gopro, dist_gopro
            )

            if joints_2d is not None:
                projection_count += 1
                frame = draw_skeleton_on_frame(frame, joints_2d, joint_valid, img_size)

                if not args.no_prosthesis:
                    vertices_2d = project_prosthesis_to_2d(
                        frame_data, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std, K_gopro, dist_gopro
                    )
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
