#!/usr/bin/env python3
"""
PrimeColor相机内参标定脚本
- 从标定视频中提取帧
- 使用multical标定内参
"""

import os
import cv2
import argparse
import subprocess
import json


def extract_frames_from_video(video_path: str, output_dir: str,
                              fps: float = 2.0, max_frames: int = 200):
    """从视频中提取帧"""
    print(f"\n从视频提取帧: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频FPS: {video_fps:.2f}, 总帧数: {total_frames}")
    print(f"提取策略: 每{video_fps/fps:.1f}帧取1帧，最多{max_frames}帧")

    os.makedirs(output_dir, exist_ok=True)

    frame_interval = int(video_fps / fps)
    extracted = 0
    frame_idx = 0

    while frame_idx < total_frames and extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(output_path, frame)
            extracted += 1

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  进度: {frame_idx}/{total_frames}, 已提取: {extracted}", end='\r')

    cap.release()
    print(f"\n✅ 提取了 {extracted} 帧到 {output_dir}")
    return extracted


def run_multical_intrinsic(image_dir: str, output_dir: str,
                           board_yaml: str, limit_images: int = 200):
    """运行multical内参标定"""
    print(f"\n运行multical内参标定...")
    print(f"图像目录: {image_dir}")
    print(f"标定板: {board_yaml}")

    multical_dir = "/Volumes/FastACIS/annotation_pipeline/multical"

    # 构建命令
    cmd = [
        'python', 'intrinsic.py',
        '--boards', board_yaml,
        '--image_path', image_dir,
        '--limit_images', str(limit_images),
        '--vis'
    ]

    print(f"命令: cd {multical_dir} && {' '.join(cmd)}")

    # 运行
    result = subprocess.run(
        cmd,
        cwd=multical_dir,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ multical失败")
        return False

    # 检查输出
    intrinsic_json = os.path.join(image_dir, 'intrinsic.json')
    if os.path.exists(intrinsic_json):
        print(f"\n✅ 内参标定完成: {intrinsic_json}")

        # 显示结果
        with open(intrinsic_json, 'r') as f:
            data = json.load(f)

        print(f"\n内参结果:")
        for cam, params in data.get('cameras', {}).items():
            K = params.get('K', [])
            dist = params.get('dist', [])
            rms = params.get('rms', 0)
            print(f"  相机: {cam}")
            print(f"    K: {K[:3]}")
            print(f"       {K[3:6]}")
            print(f"       {K[6:]}")
            print(f"    畸变: {dist}")
            print(f"    RMS: {rms:.4f} 像素")

        return True
    else:
        print(f"❌ 未找到输出文件: {intrinsic_json}")
        return False


def main():
    parser = argparse.ArgumentParser(description='PrimeColor内参标定')
    parser.add_argument('--video', required=True, help='标定视频路径')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--board', default='./asset/charuco_b1_2.yaml',
                       help='标定板配置文件（相对于multical目录）')
    parser.add_argument('--fps', type=float, default=2.0,
                       help='提取帧率（每秒几帧）')
    parser.add_argument('--max-frames', type=int, default=200,
                       help='最多提取多少帧')
    parser.add_argument('--limit-images', type=int, default=200,
                       help='multical使用多少张图像')

    args = parser.parse_args()

    print("=" * 80)
    print("PrimeColor相机内参标定")
    print("=" * 80)
    print(f"视频: {args.video}")
    print(f"输出: {args.output_dir}")

    # 检查视频文件
    if not os.path.exists(args.video):
        print(f"❌ 视频文件不存在: {args.video}")
        return 1

    # 1. 提取帧
    frames_dir = os.path.join(args.output_dir, 'frames')
    num_frames = extract_frames_from_video(
        args.video, frames_dir,
        args.fps, args.max_frames
    )

    if num_frames == 0:
        print("❌ 未提取到任何帧")
        return 1

    # 2. 运行multical
    success = run_multical_intrinsic(
        frames_dir, args.output_dir,
        args.board, args.limit_images
    )

    if not success:
        return 1

    print("\n" + "=" * 80)
    print("✅ 内参标定完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  内参JSON: {os.path.join(frames_dir, 'intrinsic.json')}")
    print(f"  图像帧: {frames_dir}")
    print(f"  可视化: {os.path.join(frames_dir, 'vis')}")

    print(f"\n下一步:")
    print(f"  1. 检查 vis/ 目录中的可视化结果")
    print(f"  2. 确认RMS误差 < 0.5像素")
    print(f"  3. 使用此内参进行外参标定")

    return 0


if __name__ == "__main__":
    exit(main())
