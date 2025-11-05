#!/usr/bin/env python3
"""
验证GoPro官方timecode同步质量（使用QR码anchor视频）

工作原理：
1. 扫描所有同步后的GoPro视频的开始段和结尾段QR码
2. 对比QR anchor视频，计算各相机之间的相对偏移
3. 检查偏移是否一致（理想情况下应该都是0）
4. 计算时间漂移：drift = abs(end_offset - start_offset)

使用示例：
    python verify_gopro_sync_with_qr.py \
        --gopro_dir /path/to/gopro_synced \
        --anchor_video /path/to/qr_sync.mp4 \
        --start_duration 30 \
        --end_duration 30
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 导入已有的QR检测函数
from sync_with_qr_anchor import (
    scan_video_qr_segment,
    extract_anchor_metadata_from_video,
    get_anchor_time,
    get_video_info
)


def scan_gopro_dual_segments(
    gopro_video_path: str,
    anchor_map: Dict[int, float],
    anchor_fps: float,
    start_duration: float = 30.0,
    end_duration: float = 30.0,
    frame_step: int = 5,
    prefix: str = ""
) -> Dict:
    """
    扫描GoPro视频的开始段和结尾段QR码

    Args:
        gopro_video_path: GoPro视频路径
        anchor_map: QR anchor映射
        anchor_fps: Anchor FPS
        start_duration: 开始段扫描时长（秒）
        end_duration: 结尾段扫描时长（秒）
        frame_step: 帧间隔
        prefix: QR码前缀

    Returns:
        {
            'start_segment': {
                'detections': [(video_time, qr_frame_num), ...],
                'anchor_offset': float,  # 相对anchor的偏移
                'qr_count': int
            },
            'end_segment': {...},
            'drift': float  # abs(end_offset - start_offset)
        }
    """
    video_info = get_video_info(gopro_video_path)
    video_duration = video_info['duration']

    print(f"\n扫描: {os.path.basename(gopro_video_path)}")
    print(f"  视频时长: {video_duration:.2f}s")

    # 扫描开始段
    print(f"  [开始段] 0s - {start_duration}s")
    start_detections = scan_video_qr_segment(
        gopro_video_path,
        start_time=0.0,
        duration=start_duration,
        frame_step=frame_step,
        prefix=prefix
    )

    # 扫描结尾段
    end_start_time = max(0, video_duration - end_duration)
    print(f"  [结尾段] {end_start_time:.1f}s - {video_duration:.1f}s")
    end_detections = scan_video_qr_segment(
        gopro_video_path,
        start_time=end_start_time,
        duration=end_duration,
        frame_step=frame_step,
        prefix=prefix
    )

    # 计算相对anchor的偏移（使用中位数）
    def calc_anchor_offset(detections: List[Tuple[float, int]]) -> Optional[float]:
        if not detections:
            return None
        offsets = []
        for video_time, qr_frame_num in detections:
            anchor_time = get_anchor_time(qr_frame_num, anchor_map, anchor_fps)
            offset = video_time - anchor_time
            offsets.append(offset)
        return float(np.median(offsets))

    start_offset = calc_anchor_offset(start_detections)
    end_offset = calc_anchor_offset(end_detections)

    # 计算漂移
    drift = None
    if start_offset is not None and end_offset is not None:
        drift = abs(end_offset - start_offset)

    result = {
        'video_path': gopro_video_path,
        'video_duration': video_duration,
        'start_segment': {
            'detections': start_detections,
            'anchor_offset': start_offset,
            'qr_count': len(start_detections),
            'qr_range': [start_detections[0][1], start_detections[-1][1]] if start_detections else None
        },
        'end_segment': {
            'detections': end_detections,
            'anchor_offset': end_offset,
            'qr_count': len(end_detections),
            'qr_range': [end_detections[0][1], end_detections[-1][1]] if end_detections else None
        },
        'drift': drift
    }

    print(f"  开始段: {len(start_detections)} QR码, offset={start_offset:.3f}s" if start_offset else "  开始段: 无QR码")
    print(f"  结尾段: {len(end_detections)} QR码, offset={end_offset:.3f}s" if end_offset else "  结尾段: 无QR码")
    if drift is not None:
        print(f"  时间漂移: {drift:.3f}s")

    return result


def verify_gopro_sync_quality(
    gopro_results: Dict[str, Dict],
    max_offset_threshold: float = 0.1  # 100ms
) -> Dict:
    """
    验证所有GoPro相机之间的同步质量

    Args:
        gopro_results: {cam_name: scan_result, ...}
        max_offset_threshold: 最大允许偏移（秒）

    Returns:
        {
            'sync_quality': 'excellent' | 'good' | 'poor',
            'max_offset_frames': float,  # 换算为帧数（假设60fps）
            'problem_cameras': List[str],
            'report': str
        }
    """
    print("\n" + "=" * 80)
    print("验证GoPro同步质量")
    print("=" * 80)

    # 提取所有相机的开始段offset
    cam_names = list(gopro_results.keys())
    start_offsets = {}
    end_offsets = {}

    for cam_name, result in gopro_results.items():
        start_offset = result['start_segment']['anchor_offset']
        end_offset = result['end_segment']['anchor_offset']

        if start_offset is not None:
            start_offsets[cam_name] = start_offset
        if end_offset is not None:
            end_offsets[cam_name] = end_offset

    # 有QR码的相机
    cameras_with_qr = list(start_offsets.keys())
    # 没有QR码的相机
    cameras_without_qr = [cam for cam in cam_names if cam not in cameras_with_qr]

    if not start_offsets:
        # 所有相机都没有QR码
        return {
            'sync_quality': 'unknown',
            'max_offset_frames': None,
            'max_offset_seconds': None,
            'verified_cameras': [],
            'unverified_cameras': list(cam_names),
            'failed_cameras': [],
            'relative_offsets': {},
            'cameras_with_qr': [],
            'report': f"⊘ 所有{len(cam_names)}个相机都没有检测到QR码\n建议: GoPro官方timecode同步通常是可靠的"
        }

    # 计算相对偏移（以第一个有QR码的相机为参考）
    ref_cam = cameras_with_qr[0]
    ref_offset = start_offsets[ref_cam]

    relative_offsets = {}
    for cam_name, offset in start_offsets.items():
        relative_offsets[cam_name] = offset - ref_offset

    # 检查最大偏移
    max_relative_offset = max(abs(v) for v in relative_offsets.values())
    max_offset_frames = max_relative_offset * 60  # 假设60fps

    print(f"\n相对偏移分析（以{ref_cam}为参考）:")
    for cam_name, offset in relative_offsets.items():
        frames = offset * 60
        print(f"  {cam_name}: {offset:+.3f}s ({frames:+.1f} 帧)")

    # 根据偏移大小分类相机
    verified_cameras = []  # 有QR码且偏移≤2帧
    failed_cameras = []    # 有QR码但偏移>2帧

    for cam_name, offset in relative_offsets.items():
        frames = abs(offset * 60)
        if frames <= 2.0:
            verified_cameras.append(cam_name)
        else:
            failed_cameras.append(cam_name)

    # 判断整体质量
    if max_offset_frames <= 1.0:
        sync_quality = 'excellent'
        quality_text = "✅ 优秀（≤1帧）"
    elif max_offset_frames <= 2.0:
        sync_quality = 'good'
        quality_text = "✅ 良好（≤2帧）"
    else:
        sync_quality = 'poor'
        quality_text = f"⚠️ 较差（{max_offset_frames:.1f}帧）"

    print(f"\n同步质量: {quality_text}")
    print(f"最大偏移: {max_offset_frames:.2f} 帧 ({max_relative_offset:.3f}s)")
    print(f"\n分类统计:")
    print(f"  ✅ 验证成功: {len(verified_cameras)} 个相机 - {', '.join(verified_cameras)}")
    if failed_cameras:
        print(f"  ❌ 验证失败: {len(failed_cameras)} 个相机 - {', '.join(failed_cameras)}")
    print(f"  ⊘  未验证: {len(cameras_without_qr)} 个相机 - {', '.join(cameras_without_qr) if cameras_without_qr else '无'}")

    # 生成报告
    report_lines = [
        f"同步质量: {sync_quality}",
        f"最大偏移: {max_offset_frames:.2f} 帧 ({max_relative_offset:.3f}s)",
        f"参考相机: {ref_cam}",
        "",
        f"✅ 验证成功: {len(verified_cameras)} 个相机"
    ]

    for cam_name in verified_cameras:
        offset = relative_offsets[cam_name]
        frames = offset * 60
        report_lines.append(f"  {cam_name}: {offset:+.3f}s ({frames:+.1f} 帧)")

    if failed_cameras:
        report_lines.append("")
        report_lines.append(f"❌ 验证失败: {len(failed_cameras)} 个相机（偏移>2帧）")
        for cam_name in failed_cameras:
            offset = relative_offsets[cam_name]
            frames = offset * 60
            report_lines.append(f"  {cam_name}: {offset:+.3f}s ({frames:+.1f} 帧)")
        report_lines.append("建议: 检查这些相机的timecode设置或使用QR码重新同步")

    if cameras_without_qr:
        report_lines.append("")
        report_lines.append(f"⊘  未验证: {len(cameras_without_qr)} 个相机（无QR码）")
        for cam_name in cameras_without_qr:
            report_lines.append(f"  {cam_name}")

    return {
        'sync_quality': sync_quality,
        'max_offset_frames': float(max_offset_frames),
        'max_offset_seconds': float(max_relative_offset),
        'verified_cameras': verified_cameras,
        'failed_cameras': failed_cameras,
        'unverified_cameras': cameras_without_qr,
        'relative_offsets': {k: float(v) for k, v in relative_offsets.items()},
        'cameras_with_qr': cameras_with_qr,
        'report': '\n'.join(report_lines)
    }


def main():
    parser = argparse.ArgumentParser(
        description='验证GoPro官方timecode同步质量（使用QR码）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--gopro_dir', required=True,
                       help='同步后的GoPro视频目录')
    parser.add_argument('--anchor_video', required=True,
                       help='QR anchor视频路径')
    parser.add_argument('--start_duration', type=float, default=30.0,
                       help='开始段扫描时长（秒），默认30')
    parser.add_argument('--end_duration', type=float, default=30.0,
                       help='结尾段扫描时长（秒），默认30')
    parser.add_argument('--frame_step', type=int, default=5,
                       help='帧步长，默认5')
    parser.add_argument('--prefix', type=str, default='',
                       help='QR码前缀，默认无')
    parser.add_argument('--save_json', type=str, default=None,
                       help='保存验证结果到JSON文件')

    args = parser.parse_args()

    # 检查目录
    if not os.path.isdir(args.gopro_dir):
        print(f"❌ 错误: GoPro目录不存在: {args.gopro_dir}")
        return 1

    if not os.path.exists(args.anchor_video):
        print(f"❌ 错误: Anchor视频不存在: {args.anchor_video}")
        return 1

    # 提取anchor metadata
    print("\n" + "=" * 80)
    print("步骤1: 提取QR Anchor Metadata")
    print("=" * 80)
    anchor_map, anchor_fps = extract_anchor_metadata_from_video(
        args.anchor_video,
        prefix=args.prefix,
        sample_frames=200,
        frame_step=5
    )

    # 查找所有GoPro视频
    gopro_videos = []
    for cam_dir in sorted(Path(args.gopro_dir).iterdir()):
        if cam_dir.is_dir() and cam_dir.name.lower().startswith('cam'):
            video_path = cam_dir / 'Video.MP4'
            if not video_path.exists():
                video_path = cam_dir / 'video.mp4'
            if video_path.exists():
                gopro_videos.append((cam_dir.name, str(video_path)))

    if not gopro_videos:
        print(f"❌ 错误: 在{args.gopro_dir}中未找到GoPro视频")
        return 1

    print(f"\n找到 {len(gopro_videos)} 个GoPro相机")

    # 扫描所有GoPro视频
    print("\n" + "=" * 80)
    print("步骤2: 扫描所有GoPro视频（开始段+结尾段）")
    print("=" * 80)

    gopro_results = {}
    for cam_name, video_path in gopro_videos:
        result = scan_gopro_dual_segments(
            video_path,
            anchor_map,
            anchor_fps,
            start_duration=args.start_duration,
            end_duration=args.end_duration,
            frame_step=args.frame_step,
            prefix=args.prefix
        )
        gopro_results[cam_name] = result

    # 验证同步质量
    print("\n" + "=" * 80)
    print("步骤3: 验证同步质量")
    print("=" * 80)

    verification = verify_gopro_sync_quality(gopro_results)

    print("\n" + "=" * 80)
    print("验证报告")
    print("=" * 80)
    print(verification['report'])

    # 保存结果
    if args.save_json:
        result_data = {
            'gopro_dir': args.gopro_dir,
            'anchor_video': args.anchor_video,
            'verification': verification,
            'gopro_results': {
                cam_name: {
                    'video_path': result['video_path'],
                    'video_duration': result['video_duration'],
                    'start_segment': {
                        'qr_count': result['start_segment']['qr_count'],
                        'anchor_offset': result['start_segment']['anchor_offset'],
                        'qr_range': result['start_segment']['qr_range']
                    },
                    'end_segment': {
                        'qr_count': result['end_segment']['qr_count'],
                        'anchor_offset': result['end_segment']['anchor_offset'],
                        'qr_range': result['end_segment']['qr_range']
                    },
                    'drift': result['drift']
                }
                for cam_name, result in gopro_results.items()
            }
        }

        with open(args.save_json, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\n💾 验证结果已保存: {args.save_json}")

    # 返回状态码
    if verification['sync_quality'] in ['excellent', 'good']:
        print("\n✅ 验证通过：GoPro官方同步质量良好")
        return 0
    else:
        print("\n⚠️ 验证失败：建议使用QR码重新同步")
        return 1


if __name__ == '__main__':
    sys.exit(main())
