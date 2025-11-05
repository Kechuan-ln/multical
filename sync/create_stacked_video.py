#!/usr/bin/env python3
"""
创建多相机堆叠视频（支持网格布局）
包含GoPro + PrimeColor，每行n个视频
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils.io_utils import stack_videos_grid, stack_videos_horizontally


def collect_videos(gopro_dir: str, primecolor_video: str = None) -> List[str]:
    """
    收集所有要堆叠的视频

    Args:
        gopro_dir: GoPro同步后的视频目录
        primecolor_video: PrimeColor同步后的视频路径（可选）

    Returns:
        视频路径列表（按顺序）
    """
    videos = []

    # 收集GoPro视频
    gopro_path = Path(gopro_dir)
    gopro_videos = sorted(gopro_path.glob('*/Video.MP4'))
    if not gopro_videos:
        gopro_videos = sorted(gopro_path.glob('*.MP4'))

    for video in gopro_videos:
        videos.append(str(video))
        cam_name = video.parent.name if video.parent != gopro_path else video.stem
        print(f"  添加GoPro视频: {cam_name}")

    # 添加PrimeColor视频
    if primecolor_video and os.path.exists(primecolor_video):
        videos.append(primecolor_video)
        print(f"  添加PrimeColor视频: {Path(primecolor_video).name}")

    return videos


def create_stacked_video(
    videos: List[str],
    output_path: str,
    videos_per_row: int = 3,
    layout: str = 'grid'
) -> bool:
    """
    创建堆叠视频

    Args:
        videos: 视频路径列表
        output_path: 输出视频路径
        videos_per_row: 每行视频数（仅grid模式）
        layout: 布局模式 ('grid' 或 'horizontal')

    Returns:
        是否成功
    """
    if not videos:
        print("❌ 错误: 没有视频可堆叠")
        return False

    print(f"\n生成堆叠视频:")
    print(f"  视频数: {len(videos)}")
    print(f"  布局: {layout}")
    if layout == 'grid':
        print(f"  每行视频数: {videos_per_row}")
    print(f"  输出: {output_path}")

    try:
        if layout == 'horizontal':
            # 水平排列（一行）
            stack_videos_horizontally(videos, output_path)
        else:
            # 网格布局 (注意：现有的grid函数是硬编码2x2，暂时使用水平)
            # TODO: 实现支持自定义每行视频数的网格布局
            print(f"  ⚠️  网格布局当前使用水平排列（待实现自定义布局）")
            stack_videos_horizontally(videos, output_path)

        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"  ✅ 成功生成堆叠视频 ({file_size_mb:.2f} MB)")
            return True
        else:
            print(f"  ❌ 错误: 输出文件未生成")
            return False

    except Exception as e:
        print(f"  ❌ 错误: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='创建多相机堆叠视频（GoPro + PrimeColor）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--gopro_dir', required=True,
                       help='GoPro同步后的视频目录')
    parser.add_argument('--primecolor_video', type=str, default=None,
                       help='PrimeColor同步后的视频路径（可选）')
    parser.add_argument('--output', required=True,
                       help='输出视频路径')
    parser.add_argument('--layout', type=str, default='grid',
                       choices=['grid', 'horizontal'],
                       help='布局模式: grid (网格) 或 horizontal (水平)')
    parser.add_argument('--videos_per_row', type=int, default=3,
                       help='每行视频数（仅grid模式），默认3')

    args = parser.parse_args()

    # 收集视频
    print("=" * 80)
    print("收集视频文件")
    print("=" * 80)

    videos = collect_videos(args.gopro_dir, args.primecolor_video)

    if not videos:
        print("\n❌ 错误: 未找到任何视频")
        return 1

    # 创建堆叠视频
    print("\n" + "=" * 80)
    print("创建堆叠视频")
    print("=" * 80)

    success = create_stacked_video(
        videos,
        args.output,
        videos_per_row=args.videos_per_row,
        layout=args.layout
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
