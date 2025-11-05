#!/usr/bin/env python3
"""
同时在GoPro和PrimeColor视频上投影Mocap markers，生成堆叠视频。

解决时间同步问题：
- GoPro使用sync-offset来对齐mocap时间
- PrimeColor直接使用frame index对齐
- 生成stacked video进行对比验证

Key differences:
- GoPro: 坐标系转换(YZ flip) + positive fx
- PrimeColor: Method 4 (negative fx) + direct projection
"""

import numpy as np
import cv2
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import argparse

# 坐标系转换矩阵 (用于GoPro)
R_OPTI_TO_STD = np.array([
    [1, 0, 0],   # Keep X
    [0, -1, 0],  # Flip Y
    [0, 0, -1],  # Flip Z
], dtype=np.float64)


class GoproProjector:
    """GoPro投影器（使用坐标系转换方法）"""

    def __init__(self, K_gopro, dist_gopro, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std):
        self.K = K_gopro
        self.dist = dist_gopro
        self.R_w2p_opti = R_w2p_opti
        self.T_w2p_opti = T_w2p_opti
        self.R_p2g_std = R_p2g_std
        self.T_p2g_std = T_p2g_std

    def project(self, points_3d_world):
        """
        投影3D点到GoPro 2D。
        Pipeline: OptiTrack world → PrimeColor Opti → PrimeColor Std → GoPro Std → 2D
        """
        if len(points_3d_world) == 0:
            return np.array([]), np.array([])

        # Step 1: OptiTrack world → PrimeColor OptiTrack cam
        points_prime_opti = (self.R_w2p_opti @ points_3d_world.T).T + self.T_w2p_opti.reshape(1, 3)

        # Step 2: Coordinate system flip
        points_prime_std = (R_OPTI_TO_STD @ points_prime_opti.T).T

        # Step 3: PrimeColor standard → GoPro standard (inverse transform)
        R_inv = self.R_p2g_std.T
        T_inv = -R_inv @ self.T_p2g_std
        points_gopro = (R_inv @ points_prime_std.T).T + T_inv.reshape(1, 3)

        # Step 4: Project
        rvec_identity = np.zeros(3, dtype=np.float64)
        tvec_identity = np.zeros(3, dtype=np.float64)

        points_2d, _ = cv2.projectPoints(
            points_gopro.reshape(-1, 1, 3),
            rvec_identity, tvec_identity,
            self.K, self.dist
        )

        return points_2d.reshape(-1, 2), points_gopro[:, 2]  # Return Z values


class PrimeColorProjector:
    """PrimeColor投影器（使用Method 4: negative fx）"""

    def __init__(self, K_prime, dist_prime, rvec, tvec):
        # Method 4: Use NEGATIVE fx
        fx = K_prime[0, 0]
        self.K = np.array([[-fx, 0, K_prime[0, 2]],
                           [0, K_prime[1, 1], K_prime[1, 2]],
                           [0, 0, 1]], dtype=np.float64)
        self.dist = dist_prime
        self.rvec = rvec
        self.tvec = tvec

    def project(self, points_3d_world):
        """
        投影3D点到PrimeColor 2D。
        Uses Method 4: negative fx with direct projection.
        """
        if len(points_3d_world) == 0:
            return np.array([]), np.array([])

        # Direct projection (K already has -fx)
        points_2d, _ = cv2.projectPoints(
            points_3d_world.reshape(-1, 1, 3),
            self.rvec, self.tvec,
            self.K, self.dist
        )

        # Method 4 produces Z<0, so we return dummy Z values (don't check Z>0)
        return points_2d.reshape(-1, 2), np.ones(len(points_3d_world))


def load_all_calibrations(calib_json, mcal_path, extrinsics_json, intrinsics_json):
    """加载所有标定数据"""

    # 1. Load GoPro calibration from calibration.json
    with open(calib_json, 'r') as f:
        calib = json.load(f)

    gopro_cam_key = 'cam1' if 'cam1' in calib['cameras'] else 'cam4'
    gopro_cam = calib['cameras'][gopro_cam_key]
    K_gopro = np.array(gopro_cam['K'], dtype=np.float64)
    dist_gopro = np.array(gopro_cam['dist'], dtype=np.float64).flatten()
    gopro_size = gopro_cam['image_size']

    # PrimeColor → GoPro transform
    p2g_key = 'primecolor_to_cam1' if 'primecolor_to_cam1' in calib['camera_base2cam'] else 'primecolor_to_cam4'
    p2g = calib['camera_base2cam'][p2g_key]
    R_p2g_std = np.array(p2g['R'], dtype=np.float64)
    T_p2g_std = np.array(p2g['T'], dtype=np.float64)

    print(f"  GoPro camera: {gopro_cam_key}")
    print(f"  GoPro size: {gopro_size}")

    # 2. Load PrimeColor intrinsics
    tree = ET.parse(mcal_path)
    root = tree.getroot()
    camera = None
    for cam in root.findall('.//Camera'):
        if cam.get('Serial') == 'C11764':
            camera = cam
            break

    if camera is None:
        raise ValueError("PrimeColor camera not found in .mcal")

    attributes = camera.find('Attributes')
    prime_width = int(attributes.get('ImagerPixelWidth'))
    prime_height = int(attributes.get('ImagerPixelHeight'))
    prime_size = [prime_width, prime_height]

    # Load intrinsics (from JSON or .mcal)
    if intrinsics_json:
        with open(intrinsics_json, 'r') as f:
            intr_data = json.load(f)

        cam_data = None
        for cam_name in ['primecolor', 'C11764']:
            if cam_name in intr_data.get('cameras', {}):
                cam_data = intr_data['cameras'][cam_name]
                break

        if cam_data is None:
            raise ValueError("PrimeColor not found in intrinsics JSON")

        K_prime_orig = np.array(cam_data['K'], dtype=np.float64)
        dist_prime = np.array(cam_data['dist'], dtype=np.float64).flatten()
        print(f"  PrimeColor intrinsics from: {intrinsics_json}")
    else:
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

        K_prime_orig = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_prime = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        print(f"  PrimeColor intrinsics from: {mcal_path}")

    print(f"  PrimeColor size: {prime_size}")

    # 3. Load Mocap → PrimeColor extrinsics
    if extrinsics_json:
        with open(extrinsics_json, 'r') as f:
            ext_data = json.load(f)

        # For GoPro (uses rotation_matrix)
        R_w2p_opti = np.array(ext_data['rotation_matrix'], dtype=np.float64)
        T_w2p_opti = np.array(ext_data['tvec'], dtype=np.float64).flatten()

        # For PrimeColor (uses rvec)
        rvec_prime = np.array(ext_data['rvec'], dtype=np.float64).reshape(3, 1)
        tvec_prime = np.array(ext_data['tvec'], dtype=np.float64).reshape(3, 1)

        print(f"  Mocap→PrimeColor from: {extrinsics_json}")
    else:
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

        R_w2p_opti = R_c2w.T
        T_w2p_opti = -R_w2p_opti @ T_world

        R_w2c = R_c2w.T
        rvec_prime, _ = cv2.Rodrigues(R_w2c)
        tvec_prime = -R_w2c @ T_world

        print(f"  Mocap→PrimeColor from: {mcal_path}")

    return {
        'gopro': (K_gopro, dist_gopro, gopro_size, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std),
        'primecolor': (K_prime_orig, dist_prime, prime_size, rvec_prime, tvec_prime)
    }


def parse_mocap_csv_header(mocap_csv):
    """解析mocap CSV header"""
    with open(mocap_csv, 'r') as f:
        lines = [f.readline() for _ in range(7)]

    # Line 4: marker IDs
    id_line = lines[3].strip().split(',')

    marker_columns = {}
    for i in range(2, len(id_line), 3):
        if i >= len(id_line):
            break
        marker_id = id_line[i].strip()
        if marker_id and marker_id != 'ID':
            marker_columns[marker_id] = i

    return marker_columns


def load_mocap_frame(mocap_csv, frame_num, marker_columns):
    """加载指定帧的mocap数据"""
    markers_3d = {}

    with open(mocap_csv, 'r') as f:
        for _ in range(7):
            f.readline()

        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue

            try:
                frame = int(row[0])
            except ValueError:
                continue

            if frame == frame_num:
                for marker_id, col_idx in marker_columns.items():
                    try:
                        x = float(row[col_idx])
                        y = float(row[col_idx + 1])
                        z = float(row[col_idx + 2])
                        markers_3d[marker_id] = np.array([x, y, z], dtype=np.float64)
                    except (ValueError, IndexError):
                        pass
                break

    return markers_3d


def draw_markers(frame, points_2d, z_values, img_size, color=(0, 255, 0), radius=3):
    """绘制markers"""
    h, w = img_size[1], img_size[0]

    for (x, y), z in zip(points_2d, z_values):
        # PrimeColor: don't check Z (Method 4 produces Z<0)
        # GoPro: check Z >= 0
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
            cv2.circle(frame, (int(x), int(y)), radius + 1, (0, 0, 0), 1)

    return frame


def stack_frames(frame1, frame2, mode='horizontal'):
    """堆叠两个帧"""
    if mode == 'horizontal':
        # Resize to same height
        h = min(frame1.shape[0], frame2.shape[0])
        w1 = int(frame1.shape[1] * h / frame1.shape[0])
        w2 = int(frame2.shape[1] * h / frame2.shape[0])

        frame1_resized = cv2.resize(frame1, (w1, h))
        frame2_resized = cv2.resize(frame2, (w2, h))

        return np.hstack([frame1_resized, frame2_resized])
    else:  # vertical
        # Resize to same width
        w = min(frame1.shape[1], frame2.shape[1])
        h1 = int(frame1.shape[0] * w / frame1.shape[1])
        h2 = int(frame2.shape[0] * w / frame2.shape[1])

        frame1_resized = cv2.resize(frame1, (w, h1))
        frame2_resized = cv2.resize(frame2, (w, h2))

        return np.vstack([frame1_resized, frame2_resized])


def main():
    parser = argparse.ArgumentParser(description='同时在GoPro和PrimeColor上投影markers')

    parser.add_argument('--calibration', required=True, help='calibration.json路径')
    parser.add_argument('--mcal', required=True, help='.mcal文件路径')
    parser.add_argument('--extrinsics-json', default=None, help='自定义外参JSON')
    parser.add_argument('--intrinsics-json', default=None, help='PrimeColor内参JSON')
    parser.add_argument('--mocap-csv', required=True, help='mocap.csv路径')
    parser.add_argument('--gopro-video', required=True, help='GoPro视频路径')
    parser.add_argument('--primecolor-video', required=True, help='PrimeColor视频路径')
    parser.add_argument('--output', required=True, help='输出stacked视频路径')

    parser.add_argument('--mocap-fps', type=float, default=120.0, help='Mocap帧率')
    parser.add_argument('--gopro-sync-offset', type=float, default=0.0,
                       help='GoPro同步偏移（秒），mocap_time = gopro_time - offset')
    parser.add_argument('--primecolor-sync-offset', type=float, default=0.0,
                       help='PrimeColor同步偏移（秒），mocap_time = primecolor_time - offset')

    parser.add_argument('--start-frame', type=int, required=True,
                       help='起始帧号（两个视频使用相同的frame index）')
    parser.add_argument('--num-frames', type=int, required=True,
                       help='处理帧数')

    parser.add_argument('--stack-mode', choices=['horizontal', 'vertical'],
                       default='horizontal', help='堆叠模式')
    parser.add_argument('--marker-color', default='0,255,0', help='Marker颜色(BGR)')
    parser.add_argument('--marker-size', type=int, default=3, help='Marker半径')

    args = parser.parse_args()

    print("="*70)
    print("Dual Video Marker Projection (GoPro + PrimeColor)")
    print("="*70)

    # Parse marker color
    marker_color = tuple(map(int, args.marker_color.split(',')))

    # Load calibrations
    print("\n加载标定...")
    calib_data = load_all_calibrations(
        args.calibration, args.mcal,
        args.extrinsics_json, args.intrinsics_json
    )

    # Extract data
    K_gopro, dist_gopro, gopro_size, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std = calib_data['gopro']
    K_prime, dist_prime, prime_size, rvec_prime, tvec_prime = calib_data['primecolor']

    # Create projectors (without size)
    gopro_proj = GoproProjector(K_gopro, dist_gopro, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std)
    prime_proj = PrimeColorProjector(K_prime, dist_prime, rvec_prime, tvec_prime)

    print(f"\n✓ GoPro投影器: 坐标系转换方法")
    print(f"✓ PrimeColor投影器: Method 4 (negative fx)")

    # Parse mocap
    print(f"\n解析mocap: {args.mocap_csv}")
    marker_columns = parse_mocap_csv_header(args.mocap_csv)
    print(f"  Markers: {len(marker_columns)}")

    # Open videos
    print(f"\n打开视频...")
    cap_gopro = cv2.VideoCapture(args.gopro_video)
    cap_prime = cv2.VideoCapture(args.primecolor_video)

    gopro_fps = cap_gopro.get(cv2.CAP_PROP_FPS)
    prime_fps = cap_prime.get(cv2.CAP_PROP_FPS)

    print(f"  GoPro: {gopro_fps} fps")
    print(f"  PrimeColor: {prime_fps} fps")

    # Determine output range
    start_gopro = args.start_frame
    end_gopro = start_gopro + args.num_frames

    print(f"\n处理范围:")
    print(f"  GoPro帧: {start_gopro} - {end_gopro}")
    print(f"  GoPro sync offset: {args.gopro_sync_offset}s")
    print(f"  PrimeColor sync offset: {args.primecolor_sync_offset}s")

    # Create sample stacked frame to get output size
    cap_gopro.set(cv2.CAP_PROP_POS_FRAMES, start_gopro)
    ret1, frame1 = cap_gopro.read()
    cap_prime.set(cv2.CAP_PROP_POS_FRAMES, start_gopro)
    ret2, frame2 = cap_prime.read()

    if not ret1 or not ret2:
        raise ValueError("Cannot read initial frames")

    stacked = stack_frames(frame1, frame2, args.stack_mode)
    output_size = (stacked.shape[1], stacked.shape[0])

    print(f"  Output size: {output_size[0]}x{output_size[1]}")

    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, gopro_fps, output_size)

    # Process
    gopro_count = 0
    prime_count = 0

    print(f"\n投影中...")
    for gf in tqdm(range(start_gopro, end_gopro), desc="渲染"):
        # Read GoPro frame
        cap_gopro.set(cv2.CAP_PROP_POS_FRAMES, gf)
        ret_g, frame_gopro = cap_gopro.read()
        if not ret_g:
            break

        # Calculate mocap frame for GoPro (using sync offset)
        gt = gf / gopro_fps
        mt = gt - args.gopro_sync_offset

        if mt < 0:
            # Before mocap starts, write empty frames
            stacked = stack_frames(frame_gopro, np.zeros_like(frame_gopro), args.stack_mode)
            out.write(stacked)
            continue

        mocap_frame = int(mt * args.mocap_fps + 0.5)

        # PrimeColor: frame index = mocap frame (they are synchronized)
        # Apply PrimeColor sync offset
        pf = int(mocap_frame + args.primecolor_sync_offset)

        cap_prime.set(cv2.CAP_PROP_POS_FRAMES, pf)
        ret_p, frame_prime = cap_prime.read()
        if not ret_p:
            # If PrimeColor frame not available, use black frame
            frame_prime = np.zeros((prime_size[1], prime_size[0], 3), dtype=np.uint8)

        # Load mocap data
        markers_3d = load_mocap_frame(args.mocap_csv, mocap_frame, marker_columns)

        if len(markers_3d) > 0:
            # Convert to array (mm to meters)
            points_3d = np.array(list(markers_3d.values())) / 1000.0

            # Project to GoPro
            points_2d_gopro, z_gopro = gopro_proj.project(points_3d)
            if len(points_2d_gopro) > 0:
                frame_gopro = draw_markers(
                    frame_gopro, points_2d_gopro, z_gopro,
                    gopro_size, marker_color, args.marker_size
                )
                gopro_count += 1

            # Project to PrimeColor (using same mocap data)
            points_2d_prime, z_prime = prime_proj.project(points_3d)
            if len(points_2d_prime) > 0:
                frame_prime = draw_markers(
                    frame_prime, points_2d_prime, z_prime,
                    prime_size, marker_color, args.marker_size
                )
                prime_count += 1

        # Add labels
        gopro_label = f"GoPro F{gf}"
        cv2.putText(frame_gopro, gopro_label, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame_gopro, gopro_label, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        prime_label = f"PrimeColor F{pf}"
        cv2.putText(frame_prime, prime_label, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame_prime, prime_label, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        mocap_label = f"Mocap F{mocap_frame}"
        cv2.putText(frame_gopro, mocap_label, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_gopro, mocap_label, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)

        # Stack and write
        stacked = stack_frames(frame_gopro, frame_prime, args.stack_mode)
        out.write(stacked)

    cap_gopro.release()
    cap_prime.release()
    out.release()

    print(f"\n✓ 完成!")
    print(f"  总帧数: {end_gopro - start_gopro}")
    print(f"  GoPro投影帧数: {gopro_count}")
    print(f"  PrimeColor投影帧数: {prime_count}")
    print(f"  输出: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()
