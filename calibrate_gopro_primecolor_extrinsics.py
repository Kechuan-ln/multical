#!/usr/bin/env python3
"""
GoPro + PrimeColor 外参标定脚本
- 使用同步后的视频
- 合并两个相机的内参
- 计算相对外参
"""

import os
import sys
import cv2
import json
import argparse
import subprocess
import shutil


def extract_sync_frames(gopro_video: str, prime_video: str, output_dir: str,
                       fps: float = 1.0, max_frames: int = 100, gopro_cam_name: str = 'cam1'):
    """从同步视频中提取帧"""
    print(f"\n提取同步帧...")

    videos = {
        gopro_cam_name: gopro_video,
        'primecolor': prime_video
    }

    for cam_name, video_path in videos.items():
        print(f"\n  {cam_name}: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    ❌ 无法打开")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(video_fps / fps)

        cam_dir = os.path.join(output_dir, cam_name)
        os.makedirs(cam_dir, exist_ok=True)

        extracted = 0
        frame_idx = 0

        while frame_idx < total_frames and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # ============ PrimeColor暗图像增强 (CLAHE) ============
                # 测试显示CLAHE可将检测成功率从25%提升至91.2% (+66%)
                if cam_name == 'primecolor':
                    # 转换到LAB色彩空间
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)

                    # 对L通道应用CLAHE（对比度限制自适应直方图均衡化）
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l_enhanced = clahe.apply(l)

                    # 合并通道并转回BGR
                    enhanced_lab = cv2.merge([l_enhanced, a, b])
                    frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                # ====================================================

                output_path = os.path.join(cam_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(output_path, frame)
                extracted += 1

            frame_idx += 1

        cap.release()
        print(f"    ✅ 提取 {extracted} 帧")

    print(f"\n✅ 所有帧已提取到: {output_dir}")


def merge_intrinsics(gopro_intrinsic: str, prime_intrinsic: str, output_path: str, frames_dir: str = None):
    """合并两个内参文件，自动添加 image_size"""
    print(f"\n合并内参文件...")

    # 读取GoPro内参
    with open(gopro_intrinsic, 'r') as f:
        gopro_data = json.load(f)

    # 读取PrimeColor内参
    with open(prime_intrinsic, 'r') as f:
        prime_data = json.load(f)

    # 创建合并的内参
    merged = {
        'cameras': {}
    }

    # 添加GoPro（自动检测相机名：cam1, cam4等）
    gopro_cam_name = None
    gopro_cameras = gopro_data.get('cameras', {})

    if gopro_cameras:
        # 使用第一个（也应该是唯一的）相机
        gopro_cam_name = list(gopro_cameras.keys())[0]
        merged['cameras'][gopro_cam_name] = gopro_cameras[gopro_cam_name]
        print(f"  ✅ 添加 {gopro_cam_name} (GoPro)")
    else:
        print(f"  ❌ GoPro内参中没有找到相机")
        return None

    # 添加primecolor
    if 'primecolor' in prime_data.get('cameras', {}):
        merged['cameras']['primecolor'] = prime_data['cameras']['primecolor']
        print(f"  ✅ 添加 primecolor")
    else:
        print(f"  ⚠️  PrimeColor内参中没有找到primecolor，尝试其他键...")
        # 尝试第一个相机
        if prime_data.get('cameras'):
            first_cam = list(prime_data['cameras'].keys())[0]
            merged['cameras']['primecolor'] = prime_data['cameras'][first_cam]
            print(f"  ✅ 使用 {first_cam} 作为 primecolor")
        else:
            print(f"  ❌ 无法找到PrimeColor内参")
            return None

    # 添加 image_size（如果缺失）
    if frames_dir and os.path.exists(frames_dir):
        print(f"\n  检查并添加 image_size...")
        for cam_name in [gopro_cam_name, 'primecolor']:
            if cam_name in merged['cameras']:
                cam_data = merged['cameras'][cam_name]

                # 如果已有 image_size，跳过
                if 'image_size' in cam_data:
                    print(f"    {cam_name}: 已有 image_size = {cam_data['image_size']}")
                    continue

                # 从图像帧读取尺寸
                cam_frame_dir = os.path.join(frames_dir, cam_name)
                if os.path.exists(cam_frame_dir):
                    images = [f for f in os.listdir(cam_frame_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    if images:
                        first_image = os.path.join(cam_frame_dir, images[0])
                        img = cv2.imread(first_image)
                        if img is not None:
                            h, w = img.shape[:2]
                            merged['cameras'][cam_name]['image_size'] = [w, h]
                            print(f"    {cam_name}: 从帧读取 image_size = [{w}, {h}]")
                        else:
                            print(f"    ⚠️  {cam_name}: 无法读取图像 {first_image}")
                    else:
                        print(f"    ⚠️  {cam_name}: 目录中没有图像")
                else:
                    print(f"    ⚠️  {cam_name}: 帧目录不存在 {cam_frame_dir}")

    # 保存合并文件
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\n✅ 合并内参已保存: {output_path}")
    print(f"包含相机: {list(merged['cameras'].keys())}")

    return output_path, gopro_cam_name


def run_extrinsic_calibration(image_dir: str, intrinsic_path: str,
                              board_yaml: str):
    """运行外参标定"""
    print(f"\n运行外参标定...")
    print(f"图像目录: {image_dir}")
    print(f"内参文件: {intrinsic_path}")
    print(f"标定板: {board_yaml}")

    multical_dir = "/Volumes/FastACIS/annotation_pipeline/multical"

    # 命令
    cmd = [
        sys.executable, 'calibrate.py',
        '--boards', board_yaml,
        '--image_path', image_dir,
        '--calibration', intrinsic_path,
        '--fix_intrinsic',
        '--limit_images', '100',
        '--vis'
    ]

    print(f"\n执行: cd {multical_dir} && {' '.join(cmd)}")

    # 运行
    result = subprocess.run(
        cmd,
        cwd=multical_dir,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ 外参标定失败")
        return False

    # 检查输出
    calib_json = os.path.join(image_dir, 'calibration.json')
    if os.path.exists(calib_json):
        print(f"\n✅ 外参标定完成: {calib_json}")

        with open(calib_json, 'r') as f:
            data = json.load(f)

        print(f"\n外参结果:")
        base2cam = data.get('camera_base2cam', {})
        for cam, params in base2cam.items():
            R = params.get('R', [])
            T = params.get('T', [])
            print(f"  {cam}:")
            print(f"    R: {R[:3]}")
            print(f"       {R[3:6]}")
            print(f"       {R[6:]}")
            print(f"    T: {T}")

        rms = data.get('rms', 0)
        print(f"\n  RMS误差: {rms:.4f} 像素")

        if rms < 1.0:
            print(f"  ✅ 标定质量良好 (RMS < 1.0)")
        else:
            print(f"  ⚠️  标定质量一般 (建议RMS < 1.0)")

        return True
    else:
        print(f"❌ 未找到输出文件: {calib_json}")
        return False


def main():
    parser = argparse.ArgumentParser(description='GoPro+PrimeColor外参标定')
    parser.add_argument('--gopro-video', required=True, help='GoPro视频（原始或同步）')
    parser.add_argument('--prime-video', required=True, help='PrimeColor同步视频')
    parser.add_argument('--gopro-intrinsic', required=True, help='GoPro内参JSON')
    parser.add_argument('--prime-intrinsic', required=True, help='PrimeColor内参JSON')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--board', default='./asset/charuco_b3.yaml',
                       help='标定板配置')
    parser.add_argument('--fps', type=float, default=1.0,
                       help='提取帧率')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='最多提取帧数')

    args = parser.parse_args()

    print("=" * 80)
    print("GoPro + PrimeColor 外参标定")
    print("=" * 80)

    # 检查输入文件
    files_to_check = [
        args.gopro_video,
        args.prime_video,
        args.gopro_intrinsic,
        args.prime_intrinsic
    ]

    for path in files_to_check:
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # 0. 预读GoPro内参以获取相机名
    print("\n预读GoPro内参以获取相机名...")
    with open(args.gopro_intrinsic, 'r') as f:
        gopro_data = json.load(f)
    gopro_cameras = gopro_data.get('cameras', {})
    if gopro_cameras:
        gopro_cam_name = list(gopro_cameras.keys())[0]
        print(f"  检测到GoPro相机名: {gopro_cam_name}")
    else:
        print("  ❌ 无法读取GoPro相机名")
        return 1

    # 1. 提取同步帧
    frames_dir = os.path.join(args.output_dir, 'frames')
    extract_sync_frames(
        args.gopro_video, args.prime_video,
        frames_dir, args.fps, args.max_frames, gopro_cam_name
    )

    # 2. 合并内参
    merged_intrinsic = os.path.join(args.output_dir, 'intrinsic_merged.json')
    intrinsic_path, _ = merge_intrinsics(
        args.gopro_intrinsic, args.prime_intrinsic,
        merged_intrinsic, frames_dir  # 传入 frames_dir 以自动添加 image_size
    )

    if not intrinsic_path:
        return 1

    # 3. 运行外参标定
    success = run_extrinsic_calibration(
        frames_dir, intrinsic_path, args.board
    )

    if not success:
        return 1

    print("\n" + "=" * 80)
    print("✅ 外参标定完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  外参: {os.path.join(frames_dir, 'calibration.json')}")
    print(f"  内参(合并): {merged_intrinsic}")
    print(f"  提取帧: {frames_dir}")
    print(f"  可视化: {os.path.join(frames_dir, 'vis')}")

    print(f"\n下一步:")
    print(f"  1. 查看 vis/ 目录验证检测质量")
    print(f"  2. 确认RMS误差 < 1.0像素")
    print(f"  3. 使用 calibration.json 进行3D重建")

    return 0


if __name__ == "__main__":
    exit(main())
