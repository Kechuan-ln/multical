#!/usr/bin/env python3
"""
Project 3D skeleton + prosthesis onto GoPro video (continuous output).

This version outputs a continuous video segment with all frames.
"""

import numpy as np
import cv2
import json
import xml.etree.ElementTree as ET
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


def load_calibration(calib_json):
    """Load calibration from multical JSON format."""
    with open(calib_json, 'r') as f:
        data = json.load(f)

    gopro_cam = data['cameras']['cam1']
    K_gopro = np.array(gopro_cam['K'], dtype=np.float64)
    dist_gopro = np.array(gopro_cam['dist'], dtype=np.float64).flatten()
    img_size_gopro = gopro_cam['image_size']

    prime_cam = data['cameras']['primecolor']
    K_prime = np.array(prime_cam['K'], dtype=np.float64)
    dist_prime = np.array(prime_cam['dist'], dtype=np.float64).flatten()

    prime_to_gopro = data['camera_base2cam']['primecolor_to_cam1']
    R_p2g = np.array(prime_to_gopro['R'], dtype=np.float64)
    T_p2g = np.array(prime_to_gopro['T'], dtype=np.float64)

    return K_gopro, dist_gopro, img_size_gopro, K_prime, dist_prime, R_p2g, T_p2g


def load_optitrack_extrinsics(mcal_path, camera_serial='C11764'):
    """Load PrimeColor extrinsics from .mcal file."""
    tree = ET.parse(mcal_path)
    root = tree.getroot()

    for cam in root.findall('.//Camera'):
        if cam.get('Serial') == camera_serial:
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

            R_w2c = R_c2w.T
            rvec_prime, _ = cv2.Rodrigues(R_w2c)
            tvec_prime = -R_w2c @ T_world

            return rvec_prime, tvec_prime

    raise ValueError(f"Camera {camera_serial} not found in .mcal")


def load_skeleton_data(json_path):
    """Load skeleton + prosthesis data."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    frames = data['frames']

    frames_dict = {}
    for frame_info in frames:
        frame_num = frame_info['frame']
        frames_dict[frame_num] = frame_info

    return metadata, frames_dict


def project_3d_to_gopro(points_3d_world, rvec_prime, tvec_prime, R_p2g, T_p2g, K_gopro, dist_gopro):
    """Project 3D world points to GoPro 2D image."""
    R_w2p, _ = cv2.Rodrigues(rvec_prime)
    points_prime_cam = (R_w2p @ points_3d_world.T).T + tvec_prime.reshape(1, 3)
    points_gopro_cam = (R_p2g @ points_prime_cam.T).T + T_p2g.reshape(1, 3)

    rvec_identity = np.zeros(3, dtype=np.float64)
    tvec_identity = np.zeros(3, dtype=np.float64)

    points_2d, _ = cv2.projectPoints(
        points_gopro_cam.reshape(-1, 1, 3),
        rvec_identity, tvec_identity,
        K_gopro, dist_gopro
    )

    return points_2d.reshape(-1, 2), points_gopro_cam


def project_skeleton_to_2d(frame_data, rvec_prime, tvec_prime, R_p2g, T_p2g, K_gopro, dist_gopro):
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

    joints_3d = np.array(joints_3d, dtype=np.float64) / 1000.0
    joints_2d, joints_cam = project_3d_to_gopro(
        joints_3d, rvec_prime, tvec_prime, R_p2g, T_p2g, K_gopro, dist_gopro
    )

    for i in range(len(joint_valid)):
        if joint_valid[i] and joints_cam[i, 2] < 0:
            joint_valid[i] = False

    return joints_2d, joint_valid


def project_prosthesis_to_2d(frame_data, rvec_prime, tvec_prime, R_p2g, T_p2g, K_gopro, dist_gopro):
    """Project prosthesis mesh to 2D."""
    prosthesis = frame_data.get('prosthesis_transform')
    if not prosthesis or not prosthesis.get('valid'):
        return None

    mesh_vertices = frame_data.get('prosthesis_mesh')
    if not mesh_vertices:
        return None

    vertices_3d = np.array(mesh_vertices, dtype=np.float64) / 1000.0
    vertices_2d, vertices_cam = project_3d_to_gopro(
        vertices_3d, rvec_prime, tvec_prime, R_p2g, T_p2g, K_gopro, dist_gopro
    )

    valid_mask = vertices_cam[:, 2] >= 0
    if not valid_mask.any():
        return None

    return vertices_2d[valid_mask]


def draw_skeleton_on_frame(frame, joints_2d, joint_valid, img_size,
                           line_thickness=2, point_radius=4):
    """Draw skeleton bones and joints."""
    h, w = img_size[1], img_size[0]

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

    for i, (pt, valid) in enumerate(zip(joints_2d, joint_valid)):
        if valid and 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(frame, (int(pt[0]), int(pt[1])),
                      point_radius, (255, 0, 0), -1)
            cv2.circle(frame, (int(pt[0]), int(pt[1])),
                      point_radius + 1, (0, 0, 0), 1)

    return frame


def draw_prosthesis_on_frame(frame, vertices_2d, img_size,
                             color=(0, 255, 255), alpha=0.3):
    """Draw prosthesis mesh."""
    if vertices_2d is None or len(vertices_2d) == 0:
        return frame

    h, w = img_size[1], img_size[0]
    overlay = frame.copy()

    for pt in vertices_2d:
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 2, color, -1)

    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Project skeleton to GoPro (continuous output)')

    parser.add_argument('--calibration', required=True)
    parser.add_argument('--mcal', required=True)
    parser.add_argument('--skeleton', required=True)
    parser.add_argument('--gopro-video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mocap-fps', type=float, default=120.0)
    parser.add_argument('--sync-offset', type=float, default=0.0)
    parser.add_argument('--line-thickness', type=int, default=2)
    parser.add_argument('--point-radius', type=int, default=4)
    parser.add_argument('--no-prosthesis', action='store_true')

    args = parser.parse_args()

    print("="*70)
    print("Skeleton + Prosthesis Projection to GoPro (Continuous)")
    print("="*70)

    # Load calibration
    print("\nLoading calibration...")
    K_gopro, dist_gopro, img_size_gopro, K_prime, dist_prime, R_p2g, T_p2g = load_calibration(args.calibration)
    print(f"  GoPro intrinsics: fx={K_gopro[0,0]:.2f}, fy={K_gopro[1,1]:.2f}")
    print(f"  GoPro image size: {img_size_gopro[0]}x{img_size_gopro[1]}")

    # Load extrinsics
    print("\nLoading PrimeColor extrinsics...")
    rvec_prime, tvec_prime = load_optitrack_extrinsics(args.mcal)

    # Load skeleton data
    print("\nLoading skeleton data...")
    metadata, frames_dict = load_skeleton_data(args.skeleton)
    available_frames = sorted(frames_dict.keys())
    print(f"  Total frames: {len(frames_dict)}")
    print(f"  Mocap frame range: {available_frames[0]} - {available_frames[-1]}")

    # Open video
    print("\nOpening GoPro video...")
    cap = cv2.VideoCapture(args.gopro_video)
    gopro_fps = cap.get(cv2.CAP_PROP_FPS)
    gopro_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  GoPro FPS: {gopro_fps}")
    print(f"  GoPro frames: {gopro_frames}")

    # Calculate GoPro frame range
    first_mocap = available_frames[0]
    last_mocap = available_frames[-1]
    first_prime_time = first_mocap / args.mocap_fps
    last_prime_time = last_mocap / args.mocap_fps
    first_gopro_time = first_prime_time + args.sync_offset
    last_gopro_time = last_prime_time + args.sync_offset
    start_gopro_frame = int(first_gopro_time * gopro_fps)
    end_gopro_frame = int(last_gopro_time * gopro_fps) + 1

    print(f"\nGoPro output range:")
    print(f"  Mocap {first_mocap}-{last_mocap} → GoPro frames {start_gopro_frame}-{end_gopro_frame}")
    print(f"  GoPro time: {first_gopro_time:.2f}s - {last_gopro_time:.2f}s")
    print(f"  Output frames: {end_gopro_frame - start_gopro_frame}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, gopro_fps, tuple(img_size_gopro))

    # Build mocap frame lookup (GoPro frame → closest mocap frame)
    print(f"\nBuilding frame mapping...")
    gopro_to_mocap = {}
    for gopro_frame in range(start_gopro_frame, end_gopro_frame):
        gopro_time = gopro_frame / gopro_fps
        prime_time = gopro_time - args.sync_offset
        mocap_frame = int(prime_time * args.mocap_fps + 0.5)  # Round to nearest
        if mocap_frame in frames_dict:
            gopro_to_mocap[gopro_frame] = mocap_frame

    print(f"  Mapped {len(gopro_to_mocap)} GoPro frames to mocap data")

    # Process frames
    projection_count = 0
    print(f"\nProcessing {end_gopro_frame - start_gopro_frame} GoPro frames...")

    for gopro_frame in tqdm(range(start_gopro_frame, end_gopro_frame), desc="Rendering"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, gopro_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"\nWarning: Could not read frame {gopro_frame}")
            break

        # Get skeleton data if available
        if gopro_frame in gopro_to_mocap:
            mocap_frame = gopro_to_mocap[gopro_frame]
            frame_data = frames_dict[mocap_frame]

            # Project skeleton
            joints_2d, joint_valid = project_skeleton_to_2d(
                frame_data, rvec_prime, tvec_prime, R_p2g, T_p2g, K_gopro, dist_gopro
            )

            if joints_2d is not None:
                projection_count += 1
                frame = draw_skeleton_on_frame(
                    frame, joints_2d, joint_valid, img_size_gopro,
                    line_thickness=args.line_thickness,
                    point_radius=args.point_radius
                )

                if not args.no_prosthesis:
                    vertices_2d = project_prosthesis_to_2d(
                        frame_data, rvec_prime, tvec_prime, R_p2g, T_p2g, K_gopro, dist_gopro
                    )
                    frame = draw_prosthesis_on_frame(
                        frame, vertices_2d, img_size_gopro
                    )

        out.write(frame)

    cap.release()
    out.release()

    print(f"\n✓ Done!")
    print(f"  Output frames: {end_gopro_frame - start_gopro_frame}")
    print(f"  Frames with skeleton: {projection_count}")
    print(f"  Output saved to: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()
