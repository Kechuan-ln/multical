#!/usr/bin/env python3
"""
从完整的内参JSON中提取指定相机的内参

用法:
  python filter_intrinsics.py --input intrinsic_full.json --output intrinsic_filtered.json --cameras cam1,cam2,cam3
"""

import json
import argparse
from pathlib import Path


def filter_intrinsics(input_json, output_json, camera_list):
    """
    从完整内参JSON中提取指定相机（支持大小写不敏感匹配）

    Args:
        input_json: 输入JSON文件路径
        output_json: 输出JSON文件路径
        camera_list: 相机列表 (如 ['cam1', 'Cam2', 'CAM3'])
    """
    # 读取原始JSON
    print(f"读取: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)

    # 显示原始相机列表
    if 'cameras' in data:
        original_cameras = sorted(data['cameras'].keys())
        print(f"\n原始内参包含 {len(original_cameras)} 个相机:")
        print(f"  {', '.join(original_cameras)}")

    # 创建大小写不敏感的映射
    camera_map = {}
    if 'cameras' in data:
        for cam_key in data['cameras'].keys():
            camera_map[cam_key.lower()] = cam_key

    # 创建新的数据结构
    filtered_data = {}

    # 过滤 cameras 部分
    if 'cameras' in data:
        filtered_data['cameras'] = {}
        for cam in camera_list:
            cam_lower = cam.lower()
            if cam_lower in camera_map:
                original_key = camera_map[cam_lower]
                filtered_data['cameras'][original_key] = data['cameras'][original_key]
                if cam != original_key:
                    print(f"  ✓ 包含 {original_key} (匹配 {cam})")
                else:
                    print(f"  ✓ 包含 {cam}")
            else:
                print(f"  ⚠️  警告: {cam} 不在原始内参中")

    # 如果有其他顶级字段（如 camera_base2cam），也需要过滤
    for key in data.keys():
        if key != 'cameras':
            # 检查是否是字典且键是相机名
            if isinstance(data[key], dict):
                # 检查键是否看起来像相机名
                sample_keys = list(data[key].keys())[:3]
                if sample_keys and all(k.startswith('cam') for k in sample_keys):
                    # 过滤这个字段
                    filtered_data[key] = {}
                    for cam in camera_list:
                        if cam in data[key]:
                            filtered_data[key][cam] = data[key][cam]
                else:
                    # 不是相机相关的字段，直接复制
                    filtered_data[key] = data[key]
            else:
                # 不是字典，直接复制
                filtered_data[key] = data[key]

    # 保存过滤后的JSON
    print(f"\n保存到: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\n✅ 完成！过滤后包含 {len(filtered_data['cameras'])} 个相机")
    print(f"  {', '.join(sorted(filtered_data['cameras'].keys()))}")


def auto_detect_cameras(video_dir):
    """
    从视频目录自动检测相机列表（支持Cam1/cam1文件夹和Video.MP4文件）

    Args:
        video_dir: 视频目录路径

    Returns:
        相机名称列表
    """
    video_dir = Path(video_dir)
    cameras = []

    # 方法1: 查找所有 camX.MP4 文件
    for video_file in video_dir.glob('cam*.MP4'):
        cam_name = video_file.stem  # 去掉 .MP4 后缀
        cameras.append(cam_name)

    # 方法2: 查找所有 Cam*/Video.MP4 文件夹结构
    for cam_dir in video_dir.glob('*'):
        if cam_dir.is_dir() and cam_dir.name.lower().startswith('cam'):
            # 检查是否有Video.MP4
            video_file = cam_dir / 'Video.MP4'
            if video_file.exists():
                cameras.append(cam_dir.name)

    # 方法3: 查找所有 cam*/video.MP4 (小写)
    for cam_dir in video_dir.glob('*'):
        if cam_dir.is_dir() and cam_dir.name.lower().startswith('cam'):
            video_file = cam_dir / 'video.MP4'
            if not video_file.exists():
                video_file = cam_dir / 'video.mp4'
            if video_file.exists() and cam_dir.name not in cameras:
                cameras.append(cam_dir.name)

    return sorted(set(cameras))


def main():
    parser = argparse.ArgumentParser(
        description='从完整内参JSON中提取指定相机的内参',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 手动指定相机
  python filter_intrinsics.py \\
    --input intrinsic_hyperoff_linear_60fps.json \\
    --output intrinsic_filtered.json \\
    --cameras cam1,cam2,cam3,cam5

  # 自动从视频目录检测相机
  python filter_intrinsics.py \\
    --input intrinsic_hyperoff_linear_60fps.json \\
    --output intrinsic_filtered.json \\
    --auto-detect /Volumes/FastACIS/gorpos-2-sync/gorpos-2
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入的完整内参JSON文件'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出的过滤后内参JSON文件'
    )

    parser.add_argument(
        '--cameras', '-c',
        help='要保留的相机列表（逗号分隔），如: cam1,cam2,cam3'
    )

    parser.add_argument(
        '--auto-detect', '-a',
        help='自动从视频目录检测相机（视频文件名如 cam1.MP4）'
    )

    args = parser.parse_args()

    # 确定相机列表
    if args.auto_detect:
        print(f"自动检测相机从: {args.auto_detect}")
        camera_list = auto_detect_cameras(args.auto_detect)
        if not camera_list:
            print(f"❌ 错误: 在 {args.auto_detect} 中未找到 cam*.MP4 文件")
            return 1
        print(f"检测到 {len(camera_list)} 个相机: {', '.join(camera_list)}")
    elif args.cameras:
        camera_list = [cam.strip() for cam in args.cameras.split(',')]
    else:
        print("❌ 错误: 必须指定 --cameras 或 --auto-detect")
        parser.print_help()
        return 1

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return 1

    # 过滤内参
    try:
        filter_intrinsics(args.input, args.output, camera_list)
        return 0
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
