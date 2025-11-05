#!/usr/bin/env python3
"""
基于Anchor QR码视频的相机同步工具 (增强版)

核心理念:
- 使用已知时间序列的QR码视频作为anchor（参考基准）
- 两个相机录制该QR码视频，即使看到的QR码序列不同
- 通过anchor timecode映射，计算两个相机的相对时间偏移

工作原理:
1. Camera1 在 t1 时刻看到 QR码 #100
2. Camera2 在 t2 时刻看到 QR码 #150
3. 从anchor metadata得知: QR#100 对应 anchor时间 T1, QR#150 对应 anchor时间 T2
4. 计算偏移: offset = (t1 - T1) - (t2 - T2)
"""

import cv2
import numpy as np
import os
import json
import csv
import argparse
import subprocess
import shutil
import sys
from typing import List, Tuple, Optional, Dict
from pathlib import Path

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False


def get_ffmpeg_path() -> str:
    """Get ffmpeg path from system PATH or conda environment"""
    # Try system PATH first
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg

    # Try conda environment
    if hasattr(sys, 'base_prefix'):
        conda_ffmpeg = os.path.join(os.path.dirname(sys.executable), 'ffmpeg')
        if os.path.exists(conda_ffmpeg):
            return conda_ffmpeg

    # Fallback to 'ffmpeg' and let it fail with clear error
    return 'ffmpeg'


# Get ffmpeg path once at module load
FFMPEG = get_ffmpeg_path()


def detect_qr_fast(frame: np.ndarray) -> List[str]:
    """
    快速QR检测（支持pyzbar和OpenCV双引擎）
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # 降采样加速
    if gray.shape[0] > 1080:
        scale = 1080.0 / gray.shape[0]
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)

    results = []

    # 优先使用pyzbar
    if HAS_PYZBAR:
        try:
            detected = pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE])
            if detected:
                for obj in detected:
                    results.append(obj.data.decode('utf-8'))
        except:
            pass

    # 备用：OpenCV
    if not results:
        try:
            detector = cv2.QRCodeDetector()
            data, vertices, _ = detector.detectAndDecode(gray)
            if data:
                results.append(data)
        except:
            pass

    return results


def parse_qr_frame_number(qr_data: str, prefix: str = "") -> Optional[int]:
    """
    解析QR码，提取帧编号

    Args:
        qr_data: QR码数据（如"000042"或"SYNC-000042"）
        prefix: 预期的前缀（如"SYNC-"）

    Returns:
        帧编号（整数），如果解析失败返回None
    """
    try:
        if prefix and qr_data.startswith(prefix):
            qr_data = qr_data[len(prefix):]
        frame_num = int(qr_data)
        return frame_num
    except:
        return None


def extract_anchor_metadata_from_video(video_path: str,
                                       prefix: str = "",
                                       sample_frames: int = 100,
                                       frame_step: int = 10) -> Tuple[Dict[int, float], float]:
    """
    从anchor QR码视频中提取metadata（自动检测QR码序列）

    Args:
        video_path: anchor视频路径
        prefix: QR码前缀
        sample_frames: 最多采样的帧数
        frame_step: 采样步长

    Returns:
        (anchor_map, detected_fps): anchor映射字典和检测到的FPS
    """
    print(f"从anchor视频提取metadata: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开anchor视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"  视频信息: {fps:.2f}fps, {duration:.2f}s, {total_frames}帧")
    print(f"  采样策略: 每{frame_step}帧采样一次，最多{sample_frames}帧")

    anchor_map = {}
    frame_idx = 0
    sampled_count = 0

    while frame_idx < total_frames and sampled_count < sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        # 检测QR码
        qr_codes = detect_qr_fast(frame)
        for qr_data in qr_codes:
            qr_frame_num = parse_qr_frame_number(qr_data, prefix)
            if qr_frame_num is not None:
                anchor_time = frame_idx / fps
                anchor_map[qr_frame_num] = anchor_time

                sampled_count += 1
                if sampled_count % 20 == 0:
                    print(f"    已采样 {sampled_count} 个QR码...", end='\r')
                break  # 每帧只取第一个QR码

        frame_idx += frame_step

    cap.release()

    print(f"\n  ✅ 提取了 {len(anchor_map)} 个QR码映射")

    if not anchor_map:
        raise ValueError(f"❌ 无法从anchor视频中提取QR码，请检查视频质量")

    # 验证QR码序列的连续性
    qr_numbers = sorted(anchor_map.keys())
    print(f"  QR码范围: {qr_numbers[0]} - {qr_numbers[-1]}")

    # 检测FPS（通过QR码序列推断）
    if len(qr_numbers) >= 2:
        # 计算QR码编号增长速度
        qr_diffs = np.diff(qr_numbers)
        time_diffs = np.diff([anchor_map[qr] for qr in qr_numbers])

        # QR码每秒增长速度 = QR码增量 / 时间增量
        qr_rates = qr_diffs / time_diffs
        detected_qr_fps = np.median(qr_rates)

        print(f"  检测到的QR码帧率: {detected_qr_fps:.2f} fps")

        # 如果QR码帧率接近视频FPS，说明是标准的逐帧QR码视频
        if abs(detected_qr_fps - fps) < 2.0:
            print(f"  ✓ QR码序列与视频FPS一致（逐帧QR码）")
        else:
            print(f"  ⚠️  QR码序列FPS ({detected_qr_fps:.2f}) 与视频FPS ({fps:.2f}) 不同")
            print(f"      可能是循环播放或非标准生成方式")

    return anchor_map, fps


def load_anchor_metadata(csv_path: Optional[str],
                         video_path: Optional[str],
                         fps: float = 30.0,
                         prefix: str = "") -> Tuple[Optional[Dict[int, float]], float]:
    """
    加载anchor QR码视频的metadata（QR帧编号 -> anchor时间）

    优先级: CSV > 视频提取 > 默认假设

    Args:
        csv_path: CSV文件路径（可选）
        video_path: anchor视频路径（可选）
        fps: 如果都没有，假设anchor视频的FPS
        prefix: QR码前缀

    Returns:
        (anchor_map, effective_fps): anchor映射字典和有效FPS
    """
    # 优先级1: CSV文件
    if csv_path and Path(csv_path).exists():
        print(f"加载anchor metadata CSV: {csv_path}")
        anchor_map = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame_num = int(row.get('frame_number', row.get('frame', 0)))
                    anchor_time = float(row.get('anchor_time', row.get('time', 0)))
                    anchor_map[frame_num] = anchor_time
                except:
                    continue
        print(f"  ✅ 加载了 {len(anchor_map)} 条anchor时间映射")
        return anchor_map, fps

    # 优先级2: 从anchor视频提取
    if video_path and Path(video_path).exists():
        try:
            anchor_map, detected_fps = extract_anchor_metadata_from_video(
                video_path, prefix, sample_frames=200, frame_step=5
            )
            return anchor_map, detected_fps
        except Exception as e:
            print(f"  ⚠️  从视频提取失败: {e}")
            print(f"  回退到默认映射")

    # 优先级3: 默认假设
    print(f"使用默认anchor映射: frame_number / {fps}")
    return None, fps


def get_anchor_time(qr_frame_num: int,
                    anchor_map: Optional[Dict[int, float]],
                    fps: float = 30.0) -> float:
    """
    获取QR帧编号对应的anchor时间

    Args:
        qr_frame_num: QR码帧编号
        anchor_map: anchor时间映射字典（可为None）
        fps: 默认FPS（当anchor_map为None时使用）

    Returns:
        anchor时间（秒）
    """
    if anchor_map is not None:
        return anchor_map.get(qr_frame_num, qr_frame_num / fps)
    else:
        return qr_frame_num / fps


def scan_video_qr_segment(video_path: str,
                          start_time: float = 0.0,
                          duration: float = 60.0,
                          frame_step: int = 5,
                          prefix: str = "") -> List[Tuple[float, int]]:
    """
    扫描视频片段中的QR码

    Args:
        video_path: 视频路径
        start_time: 开始时间（秒）
        duration: 扫描时长（秒）
        frame_step: 帧间隔
        prefix: QR码前缀

    Returns:
        [(video_time, qr_frame_number), ...] 列表
    """
    print(f"扫描: {os.path.basename(video_path)}")
    print(f"  时间段: {start_time:.1f}s - {start_time + duration:.1f}s")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ 无法打开视频")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    print(f"  视频信息: {fps:.2f}fps, {video_duration:.2f}s, {total_frames}帧")

    start_frame = int(start_time * fps)
    end_frame = min(int((start_time + duration) * fps), total_frames)

    if start_frame >= total_frames:
        print(f"  ⚠️ 起始时间超出视频范围")
        cap.release()
        return []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detections = []
    seen_qr_frames = {}

    frame_idx = start_frame
    scan_count = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_step == 0:
            scan_count += 1
            qr_codes = detect_qr_fast(frame)

            for qr_data in qr_codes:
                qr_frame_num = parse_qr_frame_number(qr_data, prefix)
                if qr_frame_num is not None:
                    video_time = frame_idx / fps
                    if qr_frame_num not in seen_qr_frames:
                        seen_qr_frames[qr_frame_num] = []
                    seen_qr_frames[qr_frame_num].append(video_time)

        frame_idx += 1

        if scan_count % 200 == 0:
            print(f"    进度: {frame_idx - start_frame}/{end_frame - start_frame}",
                  end='\r')

    cap.release()

    # 取中位数
    for qr_frame_num, times in seen_qr_frames.items():
        median_time = np.median(times)
        detections.append((median_time, qr_frame_num))

    detections.sort()

    print(f"\n  ✅ 检测到 {len(detections)} 个唯一QR码 (扫描了{scan_count}帧)")
    if detections:
        qr_range = f"{detections[0][1]} - {detections[-1][1]}"
        print(f"  QR码范围: {qr_range}")

    return detections


def calculate_sync_offset_with_anchor(
        video1_detections: List[Tuple[float, int]],
        video2_detections: List[Tuple[float, int]],
        anchor_map: Optional[Dict[int, float]],
        anchor_fps: float = 30.0) -> Optional[Dict]:
    """
    通过anchor timecode映射计算同步偏移

    原理:
    - Video1在时刻t1看到QR#N1，对应anchor时间T1
    - Video2在时刻t2看到QR#N2，对应anchor时间T2
    - Video1相对anchor的偏移: offset1 = t1 - T1
    - Video2相对anchor的偏移: offset2 = t2 - T2
    - 两个视频的相对偏移: offset = offset1 - offset2

    Args:
        video1_detections: 视频1的检测结果 [(video_time, qr_frame_num), ...]
        video2_detections: 视频2的检测结果
        anchor_map: anchor时间映射
        anchor_fps: anchor FPS

    Returns:
        同步结果字典，包含offset、统计信息等
    """
    if not video1_detections or not video2_detections:
        print("❌ 至少一个视频没有检测到QR码")
        return None

    print("\n计算同步偏移（基于anchor timecode）...")

    # 1. 将每个检测映射到anchor时间
    video1_pairs = []  # [(video_time, anchor_time), ...]
    video2_pairs = []

    for video_time, qr_frame_num in video1_detections:
        anchor_time = get_anchor_time(qr_frame_num, anchor_map, anchor_fps)
        video1_pairs.append((video_time, anchor_time, qr_frame_num))

    for video_time, qr_frame_num in video2_detections:
        anchor_time = get_anchor_time(qr_frame_num, anchor_map, anchor_fps)
        video2_pairs.append((video_time, anchor_time, qr_frame_num))

    print(f"  Video1: {len(video1_pairs)} 个QR码映射")
    print(f"  Video2: {len(video2_pairs)} 个QR码映射")

    # 2. 计算每个视频相对anchor的偏移
    video1_offsets = [(vt - at, qr) for vt, at, qr in video1_pairs]
    video2_offsets = [(vt - at, qr) for vt, at, qr in video2_pairs]

    # 3. 使用中位数估计每个视频的anchor偏移（鲁棒性）
    video1_offset_median = np.median([off for off, _ in video1_offsets])
    video2_offset_median = np.median([off for off, _ in video2_offsets])

    print(f"  Video1相对anchor偏移: {video1_offset_median:.3f}s")
    print(f"  Video2相对anchor偏移: {video2_offset_median:.3f}s")

    # 4. 计算相对偏移
    relative_offset = video1_offset_median - video2_offset_median

    print(f"  相对偏移 (Video1 - Video2): {relative_offset:.3f}s")

    # 5. 验证一致性（检查离群点）
    video1_std = np.std([off for off, _ in video1_offsets])
    video2_std = np.std([off for off, _ in video2_offsets])

    print(f"  Video1偏移标准差: {video1_std:.3f}s")
    print(f"  Video2偏移标准差: {video2_std:.3f}s")

    if video1_std > 0.5 or video2_std > 0.5:
        print(f"  ⚠️ 警告: 偏移标准差较大，可能存在时间漂移或检测错误")

    # 6. 可视化QR码映射（前10个）
    print("\n  QR码映射示例（前10个）:")
    print("  Video1:")
    for i, (vt, at, qr) in enumerate(video1_pairs[:10]):
        print(f"    [{i+1}] QR#{qr:06d}: video_t={vt:.2f}s, anchor_t={at:.2f}s, offset={vt-at:.3f}s")

    print("  Video2:")
    for i, (vt, at, qr) in enumerate(video2_pairs[:10]):
        print(f"    [{i+1}] QR#{qr:06d}: video_t={vt:.2f}s, anchor_t={at:.2f}s, offset={vt-at:.3f}s")

    result = {
        "offset_seconds": float(relative_offset),
        "video1_anchor_offset": float(video1_offset_median),
        "video2_anchor_offset": float(video2_offset_median),
        "video1_offset_std": float(video1_std),
        "video2_offset_std": float(video2_std),
        "video1_qr_count": len(video1_pairs),
        "video2_qr_count": len(video2_pairs),
        "video1_qr_range": [int(video1_pairs[0][2]), int(video1_pairs[-1][2])],
        "video2_qr_range": [int(video2_pairs[0][2]), int(video2_pairs[-1][2])],
    }

    return result


def get_video_info(video_path: str) -> Dict:
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


def create_stacked_video(video1_path: str, video2_path: str, output_path: str,
                         layout: str = "hstack", duration: float = 10.0) -> bool:
    """
    创建stacked对比视频（用于验证同步效果）

    Args:
        video1_path: 视频1路径
        video2_path: 视频2路径（已同步）
        output_path: 输出路径
        layout: 布局方式 ("hstack"=左右, "vstack"=上下)
        duration: 输出时长（秒）

    Returns:
        是否成功
    """
    print(f"\n创建Stacked对比视频...")
    print(f"  布局: {layout}")
    print(f"  时长: {duration:.1f}s")

    # 获取视频信息
    video1_info = get_video_info(video1_path)
    video2_info = get_video_info(video2_path)

    # 使用较低的分辨率以加快处理
    scale_width = 960  # 缩放到960宽度

    if layout == "hstack":
        # 左右拼接
        filter_complex = (
            f"[0:v]scale={scale_width}:-1[v0];"
            f"[1:v]scale={scale_width}:-1[v1];"
            f"[v0][v1]hstack=inputs=2"
        )
    else:
        # 上下拼接
        filter_complex = (
            f"[0:v]scale={scale_width}:-1[v0];"
            f"[1:v]scale={scale_width}:-1[v1];"
            f"[v0][v1]vstack=inputs=2"
        )

    cmd = [
        FFMPEG, '-y',
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
        print(f"  ✅ Stacked视频创建完成: {output_path}")
        output_info = get_video_info(output_path)
        print(f"  输出: {output_info['width']}x{output_info['height']}, {output_info['duration']:.2f}s")
        return True
    else:
        print(f"  ❌ 创建失败")
        if result.stderr:
            print(f"  错误: {result.stderr[:200]}")
        return False


def create_synced_video(video1_path: str, video2_path: str, output_path: str,
                        offset_seconds: float, target_fps: float) -> bool:
    """
    创建同步后的video2

    Args:
        video1_path: 参考视频（Video1）
        video2_path: 需要同步的视频（Video2）
        output_path: 输出路径
        offset_seconds: 时间偏移（正值表示Video2需要延迟）
        target_fps: 目标帧率
    """
    print(f"\n创建同步视频...")
    print(f"  参考: {os.path.basename(video1_path)}")
    print(f"  同步: {os.path.basename(video2_path)} -> {os.path.basename(output_path)}")
    print(f"  偏移: {offset_seconds:.3f}s")

    video1_info = get_video_info(video1_path)
    video2_info = get_video_info(video2_path)

    target_duration = video1_info['duration']

    print(f"  目标: {target_duration:.2f}s @ {target_fps:.2f}fps")

    # 根据offset决定策略
    if offset_seconds > 0:
        # Video2需要延迟 -> 前面填充黑帧
        black_duration = offset_seconds
        content_duration = target_duration - black_duration

        if content_duration <= 0:
            print(f"  ❌ 错误: offset太大，无法创建同步视频")
            return False

        print(f"  方案: 前面填充 {black_duration:.3f}s 黑帧")

        # 创建黑帧视频
        black_video = output_path.replace('.mp4', '_black.mp4')
        cmd_black = [
            FFMPEG, '-y',
            '-f', 'lavfi',
            '-i', f'color=c=black:s={video2_info["width"]}x{video2_info["height"]}:r={target_fps}',
            '-t', str(black_duration),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0',
            '-pix_fmt', 'yuv420p',
            black_video
        ]

        print(f"  创建黑帧视频...")
        subprocess.run(cmd_black, capture_output=True)

        # 调整Video2的帧率和时长
        adjusted_video = output_path.replace('.mp4', '_adjusted.mp4')
        vf_str = f'fps={target_fps}'

        cmd_adjust = [
            FFMPEG, '-y',
            '-i', video2_path,
            '-vf', vf_str,
            '-t', str(content_duration),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0',
            '-pix_fmt', 'yuv420p',
            adjusted_video
        ]

        print(f"  调整Video2: {video2_info['duration']:.2f}s @ {video2_info['fps']:.2f}fps -> {content_duration:.2f}s @ {target_fps:.2f}fps")
        subprocess.run(cmd_adjust, capture_output=True)

        # 拼接
        concat_list = output_path.replace('.mp4', '_concat.txt')
        with open(concat_list, 'w') as f:
            f.write(f"file '{os.path.abspath(black_video)}'\n")
            f.write(f"file '{os.path.abspath(adjusted_video)}'\n")

        cmd_concat = [
            FFMPEG, '-y',
            '-f', 'concat', '-safe', '0', '-i', concat_list,
            '-r', str(target_fps),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path
        ]

        print(f"  拼接最终视频...")
        subprocess.run(cmd_concat, capture_output=True)

        # 清理临时文件
        for temp in [black_video, adjusted_video, concat_list]:
            if os.path.exists(temp):
                os.remove(temp)

    else:
        # Video2需要提前 -> 裁剪开头
        trim_duration = abs(offset_seconds)
        content_duration = min(target_duration, video2_info['duration'] - trim_duration)

        print(f"  方案: 裁剪开头 {trim_duration:.3f}s")

        vf_str = f'fps={target_fps}'

        cmd = [
            FFMPEG, '-y',
            '-ss', str(trim_duration),
            '-i', video2_path,
            '-vf', vf_str,
            '-t', str(content_duration),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path
        ]

        subprocess.run(cmd, capture_output=True)

    # 验证输出
    if os.path.exists(output_path):
        output_info = get_video_info(output_path)
        print(f"  ✅ 创建完成")
        print(f"  验证输出: {output_info['duration']:.2f}s @ {output_info['fps']:.2f}fps")

        if abs(output_info['duration'] - target_duration) > 0.5:
            print(f"  ⚠️ 警告: 输出时长 ({output_info['duration']:.2f}s) 与目标 ({target_duration:.2f}s) 不匹配")

        return True
    else:
        print(f"  ❌ 创建失败")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='基于anchor QR码视频的相机同步工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
工作原理:
  1. 生成anchor QR码视频（使用generate_qr_sync_video.py）
  2. 两个相机录制该QR码视频（可以不同时开始/结束）
  3. 本工具检测QR码，通过anchor timecode映射计算偏移

使用示例:
  # 方法1: 直接使用anchor视频（最简单，推荐）
  python sync_with_qr_anchor.py \\
    --video1 camera1.mp4 \\
    --video2 camera2.mp4 \\
    --output camera2_synced.mp4 \\
    --anchor-video qr_anchor.mp4

  # 方法2: 提供anchor metadata CSV
  python sync_with_qr_anchor.py \\
    --video1 camera1.mp4 \\
    --video2 camera2.mp4 \\
    --output camera2_synced.mp4 \\
    --anchor-csv qr_metadata.csv

  # 方法3: 使用默认映射（需要知道anchor FPS）
  python sync_with_qr_anchor.py \\
    --video1 camera1.mp4 \\
    --video2 camera2.mp4 \\
    --output camera2_synced.mp4 \\
    --anchor-fps 30

  # 指定扫描范围和前缀
  python sync_with_qr_anchor.py \\
    --video1 camera1.mp4 \\
    --video2 camera2.mp4 \\
    --output camera2_synced.mp4 \\
    --anchor-video qr_anchor.mp4 \\
    --prefix "SYNC-" \\
    --scan-start 5 \\
    --scan-duration 30 \\
    --save-json sync_result.json

  # 生成stacked对比视频（验证同步效果）
  python sync_with_qr_anchor.py \\
    --video1 camera1.mp4 \\
    --video2 camera2.mp4 \\
    --output camera2_synced.mp4 \\
    --anchor-video qr_anchor.mp4 \\
    --stacked verify_sync.mp4 \\
    --stacked-layout hstack \\
    --stacked-duration 15

Anchor CSV格式（可选）:
  frame_number,anchor_time
  0,0.0
  1,0.033333
  2,0.066667
  ...
        """
    )

    parser.add_argument('--video1', required=True,
                       help='视频1路径（作为参考）')
    parser.add_argument('--video2', required=True,
                       help='视频2路径（需要同步）')
    parser.add_argument('--output', required=True,
                       help='输出同步后的视频路径')

    parser.add_argument('--anchor-video', default=None,
                       help='Anchor QR码视频路径（推荐，自动提取metadata）')
    parser.add_argument('--anchor-csv', default=None,
                       help='Anchor metadata CSV文件路径（可选，优先级高于视频）')
    parser.add_argument('--anchor-fps', type=float, default=30.0,
                       help='Anchor视频FPS（默认30，仅在没有视频/CSV时使用）')

    parser.add_argument('--scan-start', type=float, default=0.0,
                       help='开始扫描时间（秒），默认0')
    parser.add_argument('--scan-duration', type=float, default=30.0,
                       help='扫描时长（秒），默认30')
    parser.add_argument('--step', type=int, default=5,
                       help='帧步长（每N帧检测一次），默认5')
    parser.add_argument('--prefix', type=str, default='',
                       help='QR码前缀（如"SYNC-"），默认无')

    parser.add_argument('--target-fps', type=float, default=None,
                       help='输出视频FPS（默认使用video1的FPS）')
    parser.add_argument('--save-json', type=str, default=None,
                       help='保存同步结果到JSON文件')

    parser.add_argument('--stacked', type=str, default=None,
                       help='生成stacked对比视频路径（可选，用于验证同步效果）')
    parser.add_argument('--stacked-layout', type=str, default='hstack',
                       choices=['hstack', 'vstack'],
                       help='Stacked视频布局: hstack=左右, vstack=上下，默认hstack')
    parser.add_argument('--stacked-duration', type=float, default=10.0,
                       help='Stacked视频时长（秒），默认10秒')

    args = parser.parse_args()

    # 检查依赖
    if not HAS_PYZBAR:
        print("⚠️ 警告: pyzbar未安装，将使用OpenCV检测（较慢）")
        print("   推荐安装: pip install pyzbar")

    # 加载anchor metadata（优先级: CSV > 视频 > 默认）
    print("\n" + "=" * 80)
    print("步骤0: 加载Anchor Metadata")
    print("=" * 80)
    anchor_map, effective_fps = load_anchor_metadata(
        args.anchor_csv, args.anchor_video, args.anchor_fps, args.prefix
    )

    # 如果从视频提取了FPS，更新anchor_fps
    if args.anchor_video and effective_fps != args.anchor_fps:
        print(f"  使用检测到的FPS: {effective_fps:.2f} (覆盖命令行参数 {args.anchor_fps})")
        args.anchor_fps = effective_fps

    # 扫描两个视频
    print("\n" + "=" * 80)
    print("步骤1: 扫描Video1")
    print("=" * 80)
    video1_detections = scan_video_qr_segment(
        args.video1, args.scan_start, args.scan_duration, args.step, args.prefix
    )

    print("\n" + "=" * 80)
    print("步骤2: 扫描Video2")
    print("=" * 80)
    video2_detections = scan_video_qr_segment(
        args.video2, args.scan_start, args.scan_duration, args.step, args.prefix
    )

    # 计算偏移
    print("\n" + "=" * 80)
    print("步骤3: 计算同步偏移")
    print("=" * 80)

    sync_result = calculate_sync_offset_with_anchor(
        video1_detections, video2_detections, anchor_map, args.anchor_fps
    )

    if not sync_result:
        print("\n❌ 同步失败")
        return 1

    # 创建同步视频
    print("\n" + "=" * 80)
    print("步骤4: 创建同步视频")
    print("=" * 80)

    video1_info = get_video_info(args.video1)
    target_fps = args.target_fps if args.target_fps else video1_info['fps']

    success = create_synced_video(
        args.video1, args.video2, args.output,
        sync_result['offset_seconds'], target_fps
    )

    # 保存JSON结果
    if args.save_json:
        result_data = {
            "video1": {
                "path": args.video1,
                "info": get_video_info(args.video1),
                "detections": [[float(t), int(qr)] for t, qr in video1_detections],
            },
            "video2": {
                "path": args.video2,
                "info": get_video_info(args.video2),
                "detections": [[float(t), int(qr)] for t, qr in video2_detections],
            },
            "sync_result": sync_result,
            "output": args.output,
        }

        with open(args.save_json, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\n💾 同步结果已保存: {args.save_json}")

    # 生成stacked对比视频（可选）
    if success and args.stacked:
        print("\n" + "=" * 80)
        print("步骤5: 创建Stacked对比视频")
        print("=" * 80)

        stacked_success = create_stacked_video(
            args.video1,
            args.output,  # 使用同步后的视频
            args.stacked,
            layout=args.stacked_layout,
            duration=args.stacked_duration
        )

        if stacked_success:
            print(f"  💡 提示: 播放 {args.stacked} 来验证同步效果")

    if success:
        print("\n" + "=" * 80)
        print("✅ 同步完成！")
        print("=" * 80)
        print(f"输出: {args.output}")
        print(f"偏移: {sync_result['offset_seconds']:.3f}秒")
        if args.stacked:
            print(f"对比视频: {args.stacked}")
        return 0
    else:
        print("\n❌ 同步视频创建失败")
        return 1


if __name__ == '__main__':
    exit(main())
