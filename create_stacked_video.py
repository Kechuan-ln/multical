#!/usr/bin/env python3
"""
创建Stacked对比视频工具

用途：
- 将两个视频左右或上下拼接
- 用于验证视频同步效果
- 可独立使用，也可作为sync_with_qr_anchor.py的一部分

使用示例：
  # 左右拼接（默认）
  python create_stacked_video.py \
    --video1 gopro.MP4 \
    --video2 primecolor_synced.mp4 \
    --output verify_sync.mp4 \
    --duration 15

  # 上下拼接
  python create_stacked_video.py \
    --video1 gopro.MP4 \
    --video2 primecolor_synced.mp4 \
    --output verify_sync.mp4 \
    --layout vstack \
    --duration 10
"""

import argparse
import subprocess
import os
import cv2


def get_video_info(video_path):
    """获取视频信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


def create_stacked_video(video1_path, video2_path, output_path,
                         layout="hstack", duration=10.0,
                         scale_width=960, add_labels=True):
    """
    创建stacked对比视频

    Args:
        video1_path: 视频1路径
        video2_path: 视频2路径
        output_path: 输出路径
        layout: 布局 ("hstack"=左右, "vstack"=上下)
        duration: 输出时长（秒）
        scale_width: 缩放宽度
        add_labels: 是否添加标签文字

    Returns:
        是否成功
    """
    print("=" * 80)
    print("创建Stacked对比视频")
    print("=" * 80)
    print(f"视频1: {os.path.basename(video1_path)}")
    print(f"视频2: {os.path.basename(video2_path)}")
    print(f"输出: {os.path.basename(output_path)}")
    print(f"布局: {layout} ({'左右拼接' if layout == 'hstack' else '上下拼接'})")
    print(f"时长: {duration:.1f}s")
    print("=" * 80)

    # 获取视频信息
    video1_info = get_video_info(video1_path)
    video2_info = get_video_info(video2_path)

    print(f"\n视频1信息: {video1_info['width']}x{video1_info['height']}, "
          f"{video1_info['fps']:.2f}fps, {video1_info['duration']:.2f}s")
    print(f"视频2信息: {video2_info['width']}x{video2_info['height']}, "
          f"{video2_info['fps']:.2f}fps, {video2_info['duration']:.2f}s")

    # 构建filter_complex
    if add_labels:
        # 添加标签
        if layout == "hstack":
            filter_complex = (
                f"[0:v]scale={scale_width}:-1,"
                f"drawtext=text='Video 1':fontsize=30:fontcolor=white:x=10:y=10[v0];"
                f"[1:v]scale={scale_width}:-1,"
                f"drawtext=text='Video 2 (Synced)':fontsize=30:fontcolor=white:x=10:y=10[v1];"
                f"[v0][v1]hstack=inputs=2"
            )
        else:
            filter_complex = (
                f"[0:v]scale={scale_width}:-1,"
                f"drawtext=text='Video 1':fontsize=30:fontcolor=white:x=10:y=10[v0];"
                f"[1:v]scale={scale_width}:-1,"
                f"drawtext=text='Video 2 (Synced)':fontsize=30:fontcolor=white:x=10:y=10[v1];"
                f"[v0][v1]vstack=inputs=2"
            )
    else:
        # 不添加标签
        if layout == "hstack":
            filter_complex = (
                f"[0:v]scale={scale_width}:-1[v0];"
                f"[1:v]scale={scale_width}:-1[v1];"
                f"[v0][v1]hstack=inputs=2"
            )
        else:
            filter_complex = (
                f"[0:v]scale={scale_width}:-1[v0];"
                f"[1:v]scale={scale_width}:-1[v1];"
                f"[v0][v1]vstack=inputs=2"
            )

    print(f"\n开始生成...")

    cmd = [
        'ffmpeg', '-y',
        '-i', video1_path,
        '-i', video2_path,
        '-filter_complex', filter_complex,
        '-t', str(duration),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and os.path.exists(output_path):
        output_info = get_video_info(output_path)
        print(f"\n✅ 创建成功！")
        print(f"输出: {output_path}")
        print(f"分辨率: {output_info['width']}x{output_info['height']}")
        print(f"时长: {output_info['duration']:.2f}s")
        print(f"FPS: {output_info['fps']:.2f}")

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"文件大小: {file_size_mb:.2f} MB")

        print("\n" + "=" * 80)
        print("💡 提示：播放视频来验证同步效果")
        print(f"   ffplay {output_path}")
        print(f"   或")
        print(f"   open {output_path}")
        print("=" * 80)

        return True
    else:
        print(f"\n❌ 创建失败")
        if result.stderr:
            print(f"错误信息: {result.stderr[:500]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='创建Stacked对比视频工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用场景:
  验证两个视频的同步效果（特别是QR码同步）

示例:
  # 基本用法（左右拼接）
  python create_stacked_video.py \\
    --video1 gopro.MP4 \\
    --video2 primecolor_synced.mp4 \\
    --output verify_sync.mp4

  # 上下拼接，15秒
  python create_stacked_video.py \\
    --video1 gopro.MP4 \\
    --video2 primecolor_synced.mp4 \\
    --output verify_sync.mp4 \\
    --layout vstack \\
    --duration 15

  # 不添加标签，更高分辨率
  python create_stacked_video.py \\
    --video1 gopro.MP4 \\
    --video2 primecolor_synced.mp4 \\
    --output verify_sync.mp4 \\
    --scale 1280 \\
    --no-labels
        """
    )

    parser.add_argument('--video1', required=True,
                       help='视频1路径（参考视频）')
    parser.add_argument('--video2', required=True,
                       help='视频2路径（同步后的视频）')
    parser.add_argument('--output', required=True,
                       help='输出stacked视频路径')

    parser.add_argument('--layout', default='hstack',
                       choices=['hstack', 'vstack'],
                       help='布局方式: hstack=左右拼接, vstack=上下拼接，默认hstack')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='输出时长（秒），默认10秒')
    parser.add_argument('--scale', type=int, default=960,
                       help='缩放宽度（像素），默认960')
    parser.add_argument('--no-labels', action='store_true',
                       help='不添加标签文字')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.video1):
        print(f"❌ 错误: 视频1不存在: {args.video1}")
        return 1

    if not os.path.exists(args.video2):
        print(f"❌ 错误: 视频2不存在: {args.video2}")
        return 1

    # 创建stacked视频
    success = create_stacked_video(
        args.video1,
        args.video2,
        args.output,
        layout=args.layout,
        duration=args.duration,
        scale_width=args.scale,
        add_labels=not args.no_labels
    )

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
