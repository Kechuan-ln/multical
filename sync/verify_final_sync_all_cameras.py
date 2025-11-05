#!/usr/bin/env python3
"""
最终同步验证：使用视频结尾的QR码验证所有相机（GoPro + PrimeColor）的同步质量

验证逻辑：
1. 扫描所有同步后视频的最后n秒
2. 检测QR码并映射到anchor时间
3. 计算所有相机之间的相对偏移
4. 分类：验证成功、验证失败、未验证
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 导入QR检测函数
from sync.sync_with_qr_anchor import detect_qr_fast, extract_anchor_metadata_from_video


def get_video_info(video_path: str) -> Dict:
    """获取视频基本信息"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    return {
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration
    }


def scan_video_end_segment(
    video_path: str,
    end_duration: float,
    frame_step: int = 5,
    qr_prefix: str = ""
) -> List[Tuple[float, int]]:
    """
    扫描视频结尾段的QR码

    Args:
        video_path: 视频路径
        end_duration: 从结尾往前扫描的时长（秒）
        frame_step: 帧步长
        qr_prefix: QR码前缀

    Returns:
        [(video_time, qr_frame_num), ...]
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # 计算开始帧（从duration - end_duration开始）
    start_time = max(0, duration - end_duration)
    start_frame = int(start_time * fps)

    detections = []
    detected_qrs = set()

    print(f"  扫描结尾段: {start_time:.1f}s - {duration:.1f}s")
    print(f"  视频信息: {fps:.2f}fps, {duration:.2f}s, {total_frames}帧")

    frame_idx = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测QR码
        qr_codes = detect_qr_fast(frame)

        for qr_data in qr_codes:
            # 去除前缀
            if qr_prefix and qr_data.startswith(qr_prefix):
                qr_data = qr_data[len(qr_prefix):]

            try:
                qr_num = int(qr_data)
                if qr_num not in detected_qrs:
                    video_time = frame_idx / fps
                    detections.append((video_time, qr_num))
                    detected_qrs.add(qr_num)
            except ValueError:
                continue

        frame_idx += frame_step

    cap.release()

    detections.sort(key=lambda x: x[1])  # 按QR码编号排序

    if detections:
        qr_range = f"{detections[0][1]} - {detections[-1][1]}"
        print(f"  ✅ 检测到 {len(detections)} 个唯一QR码（QR#{qr_range}）")
    else:
        print(f"  ⊘  未检测到QR码")

    return detections


def get_anchor_time(qr_num: int, anchor_map: Dict[int, float], anchor_fps: float) -> float:
    """根据QR码编号获取anchor时间"""
    if qr_num in anchor_map:
        return anchor_map[qr_num]
    else:
        # 线性插值（假设QR码是逐帧的）
        return qr_num / anchor_fps


def verify_all_cameras_sync(
    camera_results: Dict[str, Dict],
    anchor_map: Dict[int, float],
    anchor_fps: float,
    fps_threshold: float = 2.0
) -> Dict:
    """
    验证所有相机的同步质量

    Args:
        camera_results: {camera_name: {'detections': [...], 'fps': ...}, ...}
        anchor_map: QR码到anchor时间的映射
        anchor_fps: Anchor视频FPS
        fps_threshold: 帧偏移阈值（≤此值为成功）

    Returns:
        验证结果字典
    """
    print("\n" + "=" * 80)
    print("验证所有相机的同步质量")
    print("=" * 80)

    # 计算每个相机相对anchor的offset
    camera_offsets = {}

    for cam_name, cam_data in camera_results.items():
        detections = cam_data['detections']
        if not detections:
            continue

        # 计算offset（video_time - anchor_time）的中位数
        offsets = []
        for video_time, qr_num in detections:
            anchor_time = get_anchor_time(qr_num, anchor_map, anchor_fps)
            offsets.append(video_time - anchor_time)

        camera_offsets[cam_name] = {
            'offset_median': float(np.median(offsets)),
            'fps': cam_data['fps'],
            'qr_count': len(detections),
            'qr_range': (detections[0][1], detections[-1][1])
        }

    if not camera_offsets:
        return {
            'sync_quality': 'unknown',
            'verified_cameras': [],
            'failed_cameras': [],
            'unverified_cameras': list(camera_results.keys()),
            'report': "❌ 所有相机都没有检测到QR码"
        }

    # 以第一个有QR码的相机为参考
    ref_cam = list(camera_offsets.keys())[0]
    ref_offset = camera_offsets[ref_cam]['offset_median']

    print(f"\n参考相机: {ref_cam}")
    print(f"\n相对偏移分析（以{ref_cam}为参考）:")

    # 计算相对偏移
    relative_offsets = {}
    verified_cameras = []
    failed_cameras = []

    for cam_name, cam_data in camera_offsets.items():
        offset = cam_data['offset_median'] - ref_offset
        fps = cam_data['fps']
        frames = abs(offset * fps)

        relative_offsets[cam_name] = {
            'offset_seconds': float(offset),
            'offset_frames': float(frames),
            'fps': fps,
            'qr_count': cam_data['qr_count']
        }

        print(f"  {cam_name}: {offset:+.3f}s ({frames:+.1f} 帧 @ {fps:.0f}fps)")

        # 分类
        if frames <= fps_threshold:
            verified_cameras.append(cam_name)
        else:
            failed_cameras.append(cam_name)

    # 未验证的相机（没有QR码）
    unverified_cameras = [cam for cam in camera_results.keys() if cam not in camera_offsets]

    # 判断整体质量
    max_offset_frames = max(v['offset_frames'] for v in relative_offsets.values()) if relative_offsets else 0

    if max_offset_frames <= 1.0:
        sync_quality = 'excellent'
    elif max_offset_frames <= 2.0:
        sync_quality = 'good'
    else:
        sync_quality = 'poor'

    # 打印分类统计
    print(f"\n同步质量: {sync_quality}")
    print(f"最大偏移: {max_offset_frames:.2f} 帧")
    print(f"\n分类统计:")
    print(f"  ✅ 验证成功: {len(verified_cameras)} 个相机 - {', '.join(verified_cameras)}")
    if failed_cameras:
        print(f"  ❌ 验证失败: {len(failed_cameras)} 个相机 - {', '.join(failed_cameras)}")
    print(f"  ⊘  未验证: {len(unverified_cameras)} 个相机 - {', '.join(unverified_cameras) if unverified_cameras else '无'}")

    # 生成报告
    report_lines = [
        f"最终同步验证（视频结尾段）",
        f"同步质量: {sync_quality}",
        f"最大偏移: {max_offset_frames:.2f} 帧",
        f"参考相机: {ref_cam}",
        "",
        f"✅ 验证成功: {len(verified_cameras)} 个相机"
    ]

    for cam_name in verified_cameras:
        data = relative_offsets[cam_name]
        report_lines.append(f"  {cam_name}: {data['offset_seconds']:+.3f}s ({data['offset_frames']:+.1f} 帧) - {data['qr_count']} QR码")

    if failed_cameras:
        report_lines.append("")
        report_lines.append(f"❌ 验证失败: {len(failed_cameras)} 个相机（偏移>{fps_threshold}帧）")
        for cam_name in failed_cameras:
            data = relative_offsets[cam_name]
            report_lines.append(f"  {cam_name}: {data['offset_seconds']:+.3f}s ({data['offset_frames']:+.1f} 帧) - {data['qr_count']} QR码")
        report_lines.append("警告: 这些相机与参考相机的同步偏移过大")

    if unverified_cameras:
        report_lines.append("")
        report_lines.append(f"⊘  未验证: {len(unverified_cameras)} 个相机（无QR码）")
        for cam_name in unverified_cameras:
            report_lines.append(f"  {cam_name}")

    return {
        'sync_quality': sync_quality,
        'max_offset_frames': float(max_offset_frames),
        'verified_cameras': verified_cameras,
        'failed_cameras': failed_cameras,
        'unverified_cameras': unverified_cameras,
        'relative_offsets': relative_offsets,
        'report': '\n'.join(report_lines)
    }


def main():
    parser = argparse.ArgumentParser(
        description='最终同步验证：使用视频结尾的QR码验证所有相机',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--gopro_dir', required=True,
                       help='同步后的GoPro视频目录')
    parser.add_argument('--primecolor_video', type=str, default=None,
                       help='同步后的PrimeColor视频路径（可选）')
    parser.add_argument('--anchor_video', required=True,
                       help='QR anchor视频路径')
    parser.add_argument('--end_duration', type=float, default=60.0,
                       help='从结尾往前扫描的时长（秒），默认60')
    parser.add_argument('--frame_step', type=int, default=5,
                       help='帧步长，默认5')
    parser.add_argument('--fps_threshold', type=float, default=2.0,
                       help='帧偏移阈值（≤此值为成功），默认2.0')
    parser.add_argument('--qr_prefix', type=str, default='',
                       help='QR码前缀，默认无')
    parser.add_argument('--save_json', type=str, default=None,
                       help='保存验证结果到JSON文件')

    args = parser.parse_args()

    # 1. 提取anchor metadata
    print("=" * 80)
    print("步骤1: 提取QR Anchor Metadata")
    print("=" * 80)

    anchor_map, anchor_fps = extract_anchor_metadata_from_video(
        args.anchor_video,
        prefix=args.qr_prefix
    )

    if not anchor_map:
        print("❌ 错误: Anchor视频中没有检测到QR码")
        return 1

    # 2. 扫描所有相机视频
    print("\n" + "=" * 80)
    print("步骤2: 扫描所有相机视频的结尾段")
    print("=" * 80)

    camera_results = {}

    # 扫描GoPro视频
    gopro_dir = Path(args.gopro_dir)
    gopro_videos = sorted(gopro_dir.glob('*/Video.MP4'))
    if not gopro_videos:
        gopro_videos = sorted(gopro_dir.glob('*.MP4'))

    for video_path in gopro_videos:
        cam_name = video_path.parent.name if video_path.parent != gopro_dir else video_path.stem
        print(f"\n扫描GoPro {cam_name}:")

        video_info = get_video_info(str(video_path))
        detections = scan_video_end_segment(
            str(video_path),
            args.end_duration,
            args.frame_step,
            args.qr_prefix
        )

        camera_results[cam_name] = {
            'type': 'gopro',
            'detections': detections,
            'fps': video_info['fps'],
            'duration': video_info['duration']
        }

    # 扫描PrimeColor视频（如果有）
    if args.primecolor_video and os.path.exists(args.primecolor_video):
        print(f"\n扫描PrimeColor:")

        video_info = get_video_info(args.primecolor_video)
        detections = scan_video_end_segment(
            args.primecolor_video,
            args.end_duration,
            args.frame_step,
            args.qr_prefix
        )

        camera_results['primecolor'] = {
            'type': 'primecolor',
            'detections': detections,
            'fps': video_info['fps'],
            'duration': video_info['duration']
        }

    # 3. 验证同步质量
    try:
        verification = verify_all_cameras_sync(
            camera_results,
            anchor_map,
            anchor_fps,
            args.fps_threshold
        )
    except Exception as e:
        print(f"\n❌ 错误: 验证同步质量时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    # 4. 保存结果
    if args.save_json:
        result = {
            'gopro_dir': args.gopro_dir,
            'primecolor_video': args.primecolor_video,
            'anchor_video': args.anchor_video,
            'end_duration': args.end_duration,
            'verification': verification,
            'camera_results': {
                cam_name: {
                    'type': data['type'],
                    'fps': data['fps'],
                    'duration': data['duration'],
                    'qr_count': len(data['detections']),
                    'qr_range': [data['detections'][0][1], data['detections'][-1][1]] if data['detections'] else None
                }
                for cam_name, data in camera_results.items()
            }
        }

        with open(args.save_json, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n✅ 验证结果已保存: {args.save_json}")

    # 打印最终报告
    print("\n" + "=" * 80)
    print("最终报告")
    print("=" * 80)
    print(verification['report'])

    return 0 if verification['sync_quality'] in ['excellent', 'good'] else 1


if __name__ == '__main__':
    sys.exit(main())
