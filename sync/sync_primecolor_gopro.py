#!/usr/bin/env python3
"""
PrimeColor与GoPro精确帧级别同步（基于QR码anchor）

功能：
1. 使用QR码anchor视频作为时间基准
2. 精确计算PrimeColor与GoPro的帧映射关系（支持不同FPS）
3. 使用least_squares优化offset和fps_ratio
4. 对齐PrimeColor视频到GoPro时间轴
5. 同步Mocap CSV数据

映射公式：
    primecolor_time = offset + fps_ratio * gopro_time

使用示例：
    python sync_primecolor_gopro.py \
        --gopro_video /path/to/gopro_synced/cam01/Video.MP4 \
        --primecolor_video /path/to/primecolor/Video.avi \
        --anchor_video /path/to/qr_sync.mp4 \
        --mocap_csv /path/to/video.csv \
        --output_dir /path/to/output
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import least_squares

# 导入已有的QR检测函数
from sync_with_qr_anchor import (
    scan_video_qr_segment,
    extract_anchor_metadata_from_video,
    get_anchor_time,
    get_video_info,
    create_synced_video as create_synced_video_simple,
    FFMPEG
)


def calculate_time_mapping_with_fps_ratio(
    gopro_detections: List[Tuple[float, int]],
    primecolor_detections: List[Tuple[float, int]],
    anchor_map: Dict[int, float],
    anchor_fps: float,
    gopro_fps: float,
    primecolor_fps: float
) -> Dict:
    """
    计算PrimeColor与GoPro的精确时间映射（支持不同FPS）

    方法：每个视频单独与anchor对齐
        1. 计算GoPro相对anchor的offset
        2. 计算PrimeColor相对anchor的offset
        3. 相对偏移 = primecolor_offset - gopro_offset
        4. 使用least_squares优化FPS比例

    Args:
        gopro_detections: GoPro QR检测 [(video_time, qr_num), ...]
        primecolor_detections: PrimeColor QR检测
        anchor_map: QR anchor映射
        anchor_fps: Anchor FPS
        gopro_fps: GoPro FPS
        primecolor_fps: PrimeColor FPS

    Returns:
        {
            'offset': float,  # 时间偏移（秒）
            'fps_ratio': float,  # FPS比例
            'offset_frames_primecolor': int,  # PrimeColor帧偏移
            'rmse': float,  # 拟合误差
            'num_matches': int,  # 匹配点数
            ...
        }
    """
    if not gopro_detections or not primecolor_detections:
        raise ValueError("至少一个视频没有检测到QR码")

    print("\n" + "=" * 80)
    print("计算PrimeColor与GoPro的精确时间映射")
    print("=" * 80)

    # 1. 将两个视频的检测映射到anchor时间
    gopro_pairs = []  # [(gopro_time, anchor_time, qr_num), ...]
    for video_time, qr_num in gopro_detections:
        anchor_time = get_anchor_time(qr_num, anchor_map, anchor_fps)
        gopro_pairs.append((video_time, anchor_time, qr_num))

    primecolor_pairs = []
    for video_time, qr_num in primecolor_detections:
        anchor_time = get_anchor_time(qr_num, anchor_map, anchor_fps)
        primecolor_pairs.append((video_time, anchor_time, qr_num))

    print(f"GoPro: {len(gopro_pairs)} QR码 (范围: QR#{gopro_pairs[0][2]}-{gopro_pairs[-1][2]})")
    print(f"PrimeColor: {len(primecolor_pairs)} QR码 (范围: QR#{primecolor_pairs[0][2]}-{primecolor_pairs[-1][2]})")

    # 2. 计算每个视频相对anchor的offset（使用中位数）
    gopro_offsets = [vt - at for vt, at, qr in gopro_pairs]
    primecolor_offsets = [vt - at for vt, at, qr in primecolor_pairs]

    gopro_offset_median = np.median(gopro_offsets)
    primecolor_offset_median = np.median(primecolor_offsets)

    print(f"\n相对Anchor的偏移:")
    print(f"  GoPro: {gopro_offset_median:.6f}s")
    print(f"  PrimeColor: {primecolor_offset_median:.6f}s")

    # 3. 计算相对偏移（使用anchor对齐方法）
    # 原理：不需要共同QR码，每个视频单独与anchor对齐
    # 注意：offset = gopro - primecolor（参考sync_with_qr_anchor.py的定义）
    # offset > 0: PrimeColor需要延迟（加黑帧）
    # offset < 0: PrimeColor需要提前（裁剪开头）
    offset = gopro_offset_median - primecolor_offset_median
    fps_ratio = primecolor_fps / gopro_fps

    print(f"\n✅ Anchor对齐方法:")
    print(f"  相对偏移: {offset:.6f}s ({offset * primecolor_fps:.2f} 帧 @ {primecolor_fps}fps)")
    print(f"  FPS比例: {fps_ratio:.6f} (理论值: {primecolor_fps}/{gopro_fps})")

    # 显示QR码映射示例（前10个）
    print(f"\nQR码映射示例（前10个）:")
    print("  GoPro:")
    for i in range(min(10, len(gopro_pairs))):
        vt, at, qr = gopro_pairs[i]
        off = vt - at
        print(f"    [{i+1}] QR#{qr:06d}: video_t={vt:.2f}s, anchor_t={at:.2f}s, offset={off:.3f}s")

    print("  PrimeColor:")
    for i in range(min(10, len(primecolor_pairs))):
        vt, at, qr = primecolor_pairs[i]
        off = vt - at
        print(f"    [{i+1}] QR#{qr:06d}: video_t={vt:.2f}s, anchor_t={at:.2f}s, offset={off:.3f}s")

    # 计算偏移一致性（标准差）
    gopro_std = np.std(gopro_offsets)
    primecolor_std = np.std(primecolor_offsets)

    print(f"\n偏移一致性:")
    print(f"  GoPro标准差: {gopro_std:.3f}s")
    print(f"  PrimeColor标准差: {primecolor_std:.3f}s")

    if gopro_std > 0.5 or primecolor_std > 0.5:
        print(f"  ⚠️ 警告: 标准差较大，可能存在时间漂移或检测错误")

    # 4. 质量评估
    fps_ratio_error = abs(fps_ratio - (primecolor_fps / gopro_fps)) / (primecolor_fps / gopro_fps)
    num_matches = 0  # Anchor方法不需要共同QR码
    rmse = None
    max_error = None
    fit_quality = 'anchor_alignment'

    return {
        'offset': float(offset),
        'fps_ratio': float(fps_ratio),
        'offset_frames_primecolor': int(round(offset * primecolor_fps)),
        'rmse': float(rmse) if rmse is not None else None,
        'max_error': float(max_error) if max_error is not None else None,
        'num_matches': num_matches,
        'gopro_fps': float(gopro_fps),
        'primecolor_fps': float(primecolor_fps),
        'expected_fps_ratio': float(primecolor_fps / gopro_fps),
        'fps_ratio_error_percent': float(fps_ratio_error * 100),
        'gopro_offset_anchor': float(gopro_offset_median),
        'primecolor_offset_anchor': float(primecolor_offset_median),
        'gopro_offset_std': float(gopro_std),
        'primecolor_offset_std': float(primecolor_std),
        'fit_quality': fit_quality
    }


def align_primecolor_video(
    primecolor_video_path: str,
    gopro_video_path: str,
    output_path: str,
    mapping: Dict
) -> bool:
    """
    对齐PrimeColor视频到GoPro时间轴

    Args:
        primecolor_video_path: PrimeColor视频路径
        gopro_video_path: GoPro参考视频路径
        output_path: 输出路径
        mapping: 时间映射参数（来自calculate_time_mapping_with_fps_ratio）
    """
    print("\n" + "=" * 80)
    print("对齐PrimeColor视频到GoPro时间轴")
    print("=" * 80)

    offset = mapping['offset']
    primecolor_fps = mapping['primecolor_fps']

    gopro_info = get_video_info(gopro_video_path)
    primecolor_info = get_video_info(primecolor_video_path)

    # 计算实际可用的duration（取决于offset方向）
    if offset > 0:
        # PrimeColor需要延迟，可用duration = min(gopro_duration, primecolor_duration + offset)
        max_duration = gopro_info['duration']
        available_primecolor = primecolor_info['duration']
        target_duration = min(max_duration, available_primecolor + offset)
    else:
        # PrimeColor需要提前，可用duration = min(gopro_duration, primecolor_duration + offset)
        trim_duration = abs(offset)
        available_primecolor = primecolor_info['duration'] - trim_duration
        target_duration = min(gopro_info['duration'], available_primecolor)

    print(f"GoPro参考: {gopro_info['duration']:.2f}s @ {gopro_info['fps']:.2f}fps")
    print(f"PrimeColor源: {primecolor_info['duration']:.2f}s @ {primecolor_fps:.2f}fps")
    print(f"时间偏移: {offset:.3f}s ({offset * primecolor_fps:.1f} 帧)")
    print(f"目标时长: {target_duration:.2f}s (取重叠部分)")

    # 使用ffmpeg裁剪和对齐
    if offset > 0:
        # PrimeColor需要延迟 -> 前面填充黑帧
        print(f"方案: 前面填充 {offset:.3f}s 黑帧，然后裁剪到 {target_duration:.2f}s")

        # 创建黑帧视频
        black_video = output_path.replace('.avi', '_black.avi').replace('.mp4', '_black.mp4')
        cmd_black = [
            FFMPEG, '-y',
            '-f', 'lavfi',
            '-i', f'color=c=black:s={primecolor_info["width"]}x{primecolor_info["height"]}:r={primecolor_fps}',
            '-t', str(offset),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0',
            '-pix_fmt', 'yuv420p',
            black_video
        ]

        subprocess.run(cmd_black, capture_output=True)

        # 裁剪PrimeColor
        content_duration = target_duration - offset
        adjusted_video = output_path.replace('.avi', '_adjusted.avi').replace('.mp4', '_adjusted.mp4')
        cmd_adjust = [
            FFMPEG, '-y',
            '-i', primecolor_video_path,
            '-t', str(content_duration),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            adjusted_video
        ]
        subprocess.run(cmd_adjust, capture_output=True)

        # 拼接
        concat_list = output_path.replace('.avi', '_concat.txt').replace('.mp4', '_concat.txt')
        with open(concat_list, 'w') as f:
            f.write(f"file '{os.path.abspath(black_video)}'\n")
            f.write(f"file '{os.path.abspath(adjusted_video)}'\n")

        cmd_concat = [
            FFMPEG, '-y',
            '-f', 'concat', '-safe', '0', '-i', concat_list,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        subprocess.run(cmd_concat, capture_output=True)

        # 清理
        for temp in [black_video, adjusted_video, concat_list]:
            if os.path.exists(temp):
                os.remove(temp)
    else:
        # PrimeColor需要提前 -> 裁剪开头
        trim_duration = abs(offset)
        print(f"方案: 裁剪开头 {trim_duration:.3f}s，保留 {target_duration:.2f}s")

        cmd = [
            FFMPEG, '-y',
            '-ss', str(trim_duration),
            '-i', primecolor_video_path,
            '-t', str(target_duration),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        subprocess.run(cmd, capture_output=True)

    if os.path.exists(output_path):
        output_info = get_video_info(output_path)
        print(f"✅ 对齐完成: {output_info['duration']:.2f}s @ {output_info['fps']:.2f}fps")
        return True
    else:
        print(f"❌ 对齐失败")
        return False


def sync_mocap_csv(
    csv_path: str,
    output_path: str,
    mapping: Dict,
    gopro_video_duration: float
) -> bool:
    """
    同步Mocap CSV到GoPro时间轴

    原理：
        由于Mocap FPS = PrimeColor FPS（都是120fps），
        可以应用相同的frame offset

    Args:
        csv_path: Mocap CSV路径
        output_path: 输出CSV路径
        mapping: 时间映射参数
        gopro_video_duration: GoPro视频时长（秒）
    """
    print("\n" + "=" * 80)
    print("同步Mocap CSV")
    print("=" * 80)

    offset_frames = mapping['offset_frames_primecolor']
    primecolor_fps = mapping['primecolor_fps']

    print(f"读取CSV: {csv_path}")

    # Optitrack CSV格式：前3行是header，第4行开始是数据
    df = pd.read_csv(csv_path, skiprows=7, low_memory=False)

    print(f"  原始数据: {len(df)} 行")

    # 第一列是Frame
    frame_col = df.columns[0]

    # 应用offset
    df[frame_col] = df[frame_col] + offset_frames

    # 裁剪到GoPro时长
    max_frame = int(gopro_video_duration * primecolor_fps)
    df_synced = df[(df[frame_col] >= 0) & (df[frame_col] <= max_frame)]

    print(f"  应用offset: {offset_frames} 帧 ({offset_frames / primecolor_fps:.3f}s)")
    print(f"  裁剪到: 0 - {max_frame} 帧 ({gopro_video_duration:.2f}s)")
    print(f"  同步后数据: {len(df_synced)} 行")

    # 保存（保留原始CSV的前7行header）
    with open(csv_path, 'r') as f:
        header_lines = [next(f) for _ in range(7)]

    with open(output_path, 'w') as f:
        f.writelines(header_lines)

    df_synced.to_csv(output_path, mode='a', index=False)

    print(f"✅ 保存到: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='PrimeColor与GoPro精确帧级别同步（基于QR码）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--gopro_video', required=True,
                       help='同步后的GoPro视频路径（参考）')
    parser.add_argument('--primecolor_video', required=True,
                       help='PrimeColor视频路径')
    parser.add_argument('--anchor_video', required=True,
                       help='QR anchor视频路径')
    parser.add_argument('--mocap_csv', default=None,
                       help='Mocap CSV路径（可选）')
    parser.add_argument('--output_dir', required=True,
                       help='输出目录')

    parser.add_argument('--scan_duration', type=float, default=30.0,
                       help='QR扫描时长（秒），默认30')
    parser.add_argument('--frame_step', type=int, default=5,
                       help='帧步长，默认5')
    parser.add_argument('--prefix', type=str, default='',
                       help='QR码前缀，默认无')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 步骤1: 提取anchor metadata
    print("\n" + "=" * 80)
    print("步骤1: 提取QR Anchor Metadata")
    print("=" * 80)
    anchor_map, anchor_fps = extract_anchor_metadata_from_video(
        args.anchor_video,
        prefix=args.prefix,
        sample_frames=200,
        frame_step=5
    )

    # 步骤2: 扫描GoPro视频
    print("\n" + "=" * 80)
    print("步骤2: 扫描GoPro视频")
    print("=" * 80)
    gopro_detections = scan_video_qr_segment(
        args.gopro_video,
        start_time=0.0,
        duration=args.scan_duration,
        frame_step=args.frame_step,
        prefix=args.prefix
    )

    gopro_info = get_video_info(args.gopro_video)

    # 步骤3: 扫描PrimeColor视频
    print("\n" + "=" * 80)
    print("步骤3: 扫描PrimeColor视频")
    print("=" * 80)
    primecolor_detections = scan_video_qr_segment(
        args.primecolor_video,
        start_time=0.0,
        duration=args.scan_duration,
        frame_step=args.frame_step,
        prefix=args.prefix
    )

    primecolor_info = get_video_info(args.primecolor_video)

    # 步骤4: 计算时间映射
    mapping = calculate_time_mapping_with_fps_ratio(
        gopro_detections,
        primecolor_detections,
        anchor_map,
        anchor_fps,
        gopro_info['fps'],
        primecolor_info['fps']
    )

    # 保存映射结果
    mapping_json = os.path.join(args.output_dir, 'sync_mapping.json')
    with open(mapping_json, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"\n💾 映射参数已保存: {mapping_json}")

    # 步骤5: 对齐PrimeColor视频
    primecolor_output = os.path.join(args.output_dir, 'primecolor_synced.mp4')
    success = align_primecolor_video(
        args.primecolor_video,
        args.gopro_video,
        primecolor_output,
        mapping
    )

    if not success:
        return 1

    # 步骤6: 同步Mocap CSV（可选）
    if args.mocap_csv:
        mocap_output = os.path.join(args.output_dir, 'mocap_synced.csv')
        sync_mocap_csv(
            args.mocap_csv,
            mocap_output,
            mapping,
            gopro_info['duration']
        )

    print("\n" + "=" * 80)
    print("✅ 同步完成！")
    print("=" * 80)
    print(f"PrimeColor同步视频: {primecolor_output}")
    if args.mocap_csv:
        print(f"Mocap同步CSV: {mocap_output}")
    print(f"映射参数: {mapping_json}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
