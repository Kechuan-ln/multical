#!/usr/bin/env python3
"""
将Mocap marker点投影到GoPro视频上，用于测试投影算法。

这是一个简化版本，只投影marker点，不涉及skeleton和mesh。
"""

import numpy as np
import cv2
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import argparse

# YZ翻转：OptiTrack (-Z前向) → 标准OpenCV (+Z前向)
R_OPTI_TO_STD = np.array([
    [1, 0, 0],   # Keep X
    [0, -1, 0],  # Flip Y
    [0, 0, -1],  # Flip Z
], dtype=np.float64)


def load_calibration_and_extrinsics(calib_json, mcal_path, extrinsics_json=None):
    """加载所有标定数据"""
    # Load calibration.json
    with open(calib_json, 'r') as f:
        calib = json.load(f)

    # GoPro intrinsics
    gopro_cam_key = 'cam1' if 'cam1' in calib['cameras'] else 'cam4'
    gopro_cam = calib['cameras'][gopro_cam_key]
    K_gopro = np.array(gopro_cam['K'], dtype=np.float64)
    dist_gopro = np.array(gopro_cam['dist'], dtype=np.float64).flatten()
    img_size = gopro_cam['image_size']

    # PrimeColor → GoPro transform
    p2g_key = 'primecolor_to_cam1' if 'primecolor_to_cam1' in calib['camera_base2cam'] else 'primecolor_to_cam4'
    p2g = calib['camera_base2cam'][p2g_key]
    R_p2g_std = np.array(p2g['R'], dtype=np.float64)
    T_p2g_std = np.array(p2g['T'], dtype=np.float64)

    print(f"  GoPro camera: {gopro_cam_key}")
    print(f"  Transform: {p2g_key}")

    # Load Mocap → PrimeColor extrinsics
    if extrinsics_json is not None:
        print(f"  Loading Mocap→PrimeColor from: {extrinsics_json}")
        with open(extrinsics_json, 'r') as f:
            extr_data = json.load(f)
        R_w2p_opti = np.array(extr_data['rotation_matrix'], dtype=np.float64)
        T_w2p_opti = np.array(extr_data['tvec'], dtype=np.float64).flatten()
        print(f"  ✓ Loaded custom extrinsics")
    else:
        print(f"  Loading Mocap→PrimeColor from: {mcal_path}")
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


def load_marker_labels(marker_labels_csv):
    """加载marker标签"""
    labels = {}
    with open(marker_labels_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            marker_id = row['marker_id']
            label = row['label']
            labels[marker_id] = label
    return labels


def parse_mocap_csv_header(mocap_csv):
    """解析mocap.csv的header，获取marker列映射"""
    with open(mocap_csv, 'r') as f:
        lines = [f.readline() for _ in range(7)]  # 读取前7行header

    # OptiTrack CSV格式:
    # Line 1: 元数据
    # Line 2: Type
    # Line 3: Name (marker名称)
    # Line 4: ID (marker ID)
    # Line 5: Parent
    # Line 6: Position (重复)
    # Line 7: X/Y/Z (坐标轴，重复)
    # Line 8+: 数据

    # Line 3: marker names
    name_line = lines[2].strip().split(',')

    # Line 4: marker IDs
    id_line = lines[3].strip().split(',')

    # 构建marker到列索引的映射
    marker_columns = {}

    # 从第3列开始（跳过Frame和Time），每个marker占3列（X,Y,Z）
    for i in range(2, len(id_line), 3):
        if i >= len(id_line):
            break

        marker_id = id_line[i].strip()
        if marker_id and marker_id != 'ID':  # 跳过header
            marker_columns[marker_id] = i

    return marker_columns


def load_mocap_frame(mocap_csv, frame_num, marker_columns):
    """加载指定帧的mocap数据"""
    markers_3d = {}

    with open(mocap_csv, 'r') as f:
        # 跳过header（前7行）
        for _ in range(7):
            f.readline()

        # 读取数据行
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue

            try:
                frame = int(row[0])
            except ValueError:
                # 跳过非数字行（如重复的header）
                continue

            if frame == frame_num:
                # 提取所有marker的3D坐标
                for marker_id, col_idx in marker_columns.items():
                    try:
                        x = float(row[col_idx])
                        y = float(row[col_idx + 1])
                        z = float(row[col_idx + 2])
                        markers_3d[marker_id] = np.array([x, y, z], dtype=np.float64)
                    except (ValueError, IndexError):
                        # 缺失数据
                        pass
                break

    return markers_3d


def project_3d_to_gopro(points_3d_world_opti, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std, K_gopro, dist_gopro):
    """
    投影3D点到GoPro 2D。

    Pipeline:
    1. OptiTrack world → PrimeColor OptiTrack cam
    2. Flip coordinate system → PrimeColor standard cam
    3. PrimeColor standard → GoPro standard (使用逆变换)
    4. Project to 2D
    """
    if len(points_3d_world_opti) == 0:
        return np.array([]), np.array([])

    # Step 1: OptiTrack world → PrimeColor OptiTrack cam
    points_prime_opti = (R_w2p_opti @ points_3d_world_opti.T).T + T_w2p_opti.reshape(1, 3)

    # Step 2: Flip coordinate system
    points_prime_std = (R_OPTI_TO_STD @ points_prime_opti.T).T

    # Step 3: PrimeColor standard → GoPro standard (使用逆变换)
    R_inv_p2g = R_p2g_std.T
    T_inv_p2g = -R_inv_p2g @ T_p2g_std
    points_gopro_std = (R_inv_p2g @ points_prime_std.T).T + T_inv_p2g.reshape(1, 3)

    # Step 4: Project
    rvec_identity = np.zeros(3, dtype=np.float64)
    tvec_identity = np.zeros(3, dtype=np.float64)

    points_2d, _ = cv2.projectPoints(
        points_gopro_std.reshape(-1, 1, 3),
        rvec_identity, tvec_identity,
        K_gopro, dist_gopro
    )

    return points_2d.reshape(-1, 2), points_gopro_std


def draw_markers_on_frame(frame, markers_2d, marker_labels, markers_3d_gopro, img_size,
                          point_radius=8, text_size=0.5):
    """在帧上绘制marker点"""
    h, w = img_size[1], img_size[0]

    for (marker_id, pt_2d), z_gopro in zip(markers_2d.items(), markers_3d_gopro):
        label = marker_labels.get(marker_id, marker_id[:10])
        x, y = pt_2d

        # 只绘制在相机前方且在图像范围内的点
        if z_gopro >= 0 and 0 <= x < w and 0 <= y < h:
            # 绘制圆点
            cv2.circle(frame, (int(x), int(y)), point_radius, (0, 255, 0), -1)
            cv2.circle(frame, (int(x), int(y)), point_radius + 1, (0, 0, 0), 2)

            # 绘制标签
            cv2.putText(frame, label, (int(x) + 10, int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 2)
            cv2.putText(frame, label, (int(x) + 10, int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description='将Mocap marker投影到GoPro视频')

    parser.add_argument('--calibration', required=True,
                       help='calibration.json路径')
    parser.add_argument('--mcal', required=True,
                       help='.mcal文件路径')
    parser.add_argument('--extrinsics-json', default=None,
                       help='可选：自定义外参JSON路径')
    parser.add_argument('--mocap-csv', required=True,
                       help='mocap.csv文件路径')
    parser.add_argument('--marker-labels', default=None,
                       help='可选：marker_labels.csv文件路径')
    parser.add_argument('--gopro-video', required=True,
                       help='GoPro视频路径')
    parser.add_argument('--output', required=True,
                       help='输出视频路径')
    parser.add_argument('--mocap-fps', type=float, default=120.0,
                       help='Mocap帧率 (默认120)')
    parser.add_argument('--sync-offset', type=float, default=0.0,
                       help='同步偏移（秒），mocap_time = gopro_time - offset')
    parser.add_argument('--start-frame', type=int, default=None,
                       help='GoPro起始帧号（可选）')
    parser.add_argument('--num-frames', type=int, default=None,
                       help='处理帧数（可选）')

    args = parser.parse_args()

    print("="*70)
    print("Mocap Marker投影到GoPro")
    print("="*70)

    # Load calibration
    print("\n加载标定...")
    K_gopro, dist_gopro, img_size, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std = \
        load_calibration_and_extrinsics(args.calibration, args.mcal, args.extrinsics_json)
    print(f"  GoPro: fx={K_gopro[0,0]:.2f}, fy={K_gopro[1,1]:.2f}")
    print(f"  Image size: {img_size[0]}x{img_size[1]}")

    # Load marker labels
    marker_labels = {}
    if args.marker_labels:
        print(f"\n加载marker标签: {args.marker_labels}")
        marker_labels = load_marker_labels(args.marker_labels)
        print(f"  标签数: {len(marker_labels)}")

    # Parse mocap CSV header
    print(f"\n解析mocap CSV: {args.mocap_csv}")
    marker_columns = parse_mocap_csv_header(args.mocap_csv)
    print(f"  Marker数: {len(marker_columns)}")

    # Open video
    print(f"\n打开GoPro视频: {args.gopro_video}")
    cap = cv2.VideoCapture(args.gopro_video)
    gopro_fps = cap.get(cv2.CAP_PROP_FPS)
    gopro_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {gopro_fps}")
    print(f"  总帧数: {gopro_total_frames}")

    # 确定处理范围
    start_frame = args.start_frame if args.start_frame is not None else 0
    if args.num_frames is not None:
        end_frame = min(start_frame + args.num_frames, gopro_total_frames)
    else:
        end_frame = gopro_total_frames

    print(f"\n处理范围: GoPro帧 {start_frame} - {end_frame}")

    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, gopro_fps, tuple(img_size))

    # Process
    projection_count = 0
    print(f"\n投影中...")

    for gf in tqdm(range(start_frame, end_frame), desc="渲染"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, gf)
        ret, frame = cap.read()
        if not ret:
            break

        # 计算对应的mocap帧
        gt = gf / gopro_fps  # GoPro时间
        mt = gt - args.sync_offset  # Mocap时间

        if mt >= 0:
            mocap_frame_num = int(mt * args.mocap_fps + 0.5)

            # 加载mocap数据
            markers_3d_world = load_mocap_frame(args.mocap_csv, mocap_frame_num, marker_columns)

            if len(markers_3d_world) > 0:
                # 转换为数组（mm to meters）
                marker_ids = list(markers_3d_world.keys())
                points_3d = np.array([markers_3d_world[mid] / 1000.0 for mid in marker_ids])

                # 投影
                points_2d, points_gopro = project_3d_to_gopro(
                    points_3d, R_w2p_opti, T_w2p_opti, R_p2g_std, T_p2g_std, K_gopro, dist_gopro
                )

                if len(points_2d) > 0:
                    # 构建marker_id到2D点的映射
                    markers_2d = {mid: pt for mid, pt in zip(marker_ids, points_2d)}

                    # 绘制
                    frame = draw_markers_on_frame(
                        frame, markers_2d, marker_labels, points_gopro[:, 2],
                        img_size
                    )
                    projection_count += 1

        out.write(frame)

    cap.release()
    out.release()

    print(f"\n✓ 完成!")
    print(f"  总帧数: {end_frame - start_frame}")
    print(f"  投影帧数: {projection_count}")
    print(f"  输出: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()
