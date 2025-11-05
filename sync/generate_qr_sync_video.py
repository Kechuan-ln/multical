#!/usr/bin/env python3
"""
QR码同步视频生成器
生成一个视频，每一帧都有唯一的帧编号作为QR码
"""

import cv2
import qrcode
import numpy as np
import argparse
import os
from typing import Tuple

def generate_qr_frame(frame_number: int, resolution: Tuple[int, int],
                     qr_size: int = 800, prefix: str = "") -> np.ndarray:
    """
    生成单帧QR码图像

    Args:
        frame_number: 帧编号
        resolution: 输出分辨率 (width, height)
        qr_size: QR码尺寸（像素）
        prefix: QR码数据前缀（可选，如"SYNC-"）

    Returns:
        RGB图像 (height, width, 3)
    """
    # 生成QR码数据：6位数字足够表示99分钟@60fps（约360,000帧）
    qr_data = f"{prefix}{frame_number:06d}"

    # 创建QR码
    qr = qrcode.QRCode(
        version=1,  # 自动选择最小版本
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # 高纠错能力
        box_size=10,
        border=4,
    )
    qr.add_data(qr_data)
    qr.make(fit=True)

    # 生成PIL图像并转换为numpy数组
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_array = np.array(qr_img.convert('RGB'))

    # 调整QR码尺寸
    qr_resized = cv2.resize(qr_array, (qr_size, qr_size),
                           interpolation=cv2.INTER_NEAREST)

    # 创建黑色背景
    frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    # 居中放置QR码
    y_offset = (resolution[1] - qr_size) // 2
    x_offset = (resolution[0] - qr_size) // 2

    # 确保不越界
    if y_offset >= 0 and x_offset >= 0:
        frame[y_offset:y_offset+qr_size, x_offset:x_offset+qr_size] = qr_resized
    else:
        print(f"警告: QR码尺寸({qr_size})大于视频分辨率{resolution}")
        # 缩放QR码以适应
        scale = min(resolution[0], resolution[1]) * 0.8
        qr_resized = cv2.resize(qr_array, (int(scale), int(scale)),
                               interpolation=cv2.INTER_NEAREST)
        y_offset = (resolution[1] - int(scale)) // 2
        x_offset = (resolution[0] - int(scale)) // 2
        frame[y_offset:y_offset+int(scale), x_offset:x_offset+int(scale)] = qr_resized

    # 添加文字标注（帧编号和时间戳）
    font = cv2.FONT_HERSHEY_SIMPLEX
    time_sec = frame_number / 30.0  # 假设30fps

    # 顶部：帧编号
    text_frame = f"Frame: {frame_number:06d}"
    cv2.putText(frame, text_frame, (50, 80),
               font, 2.5, (255, 255, 255), 4, cv2.LINE_AA)

    # 底部：时间戳
    text_time = f"Time: {time_sec:.2f}s"
    cv2.putText(frame, text_time, (50, resolution[1] - 50),
               font, 2.0, (255, 255, 255), 3, cv2.LINE_AA)

    return frame


def generate_qr_sync_video(output_path: str,
                           duration_seconds: int = 60,
                           fps: int = 30,
                           resolution: Tuple[int, int] = (1920, 1080),
                           qr_size: int = 800,
                           prefix: str = "",
                           codec: str = 'mp4v') -> bool:
    """
    生成同步用QR码视频

    Args:
        output_path: 输出视频路径（支持.mp4, .avi等）
        duration_seconds: 视频时长（秒）
        fps: 帧率
        resolution: 分辨率 (width, height)
        qr_size: QR码尺寸（像素）
        prefix: QR码数据前缀
        codec: 视频编码器（'mp4v'适用于mp4，'XVID'适用于avi）

    Returns:
        是否成功
    """
    print("=" * 80)
    print(f"生成QR码同步视频")
    print("=" * 80)
    print(f"输出路径: {output_path}")
    print(f"时长: {duration_seconds}秒")
    print(f"帧率: {fps} fps")
    print(f"分辨率: {resolution[0]}x{resolution[1]}")
    print(f"QR码尺寸: {qr_size}px")
    print(f"QR码前缀: '{prefix}' (留空则为纯数字)")
    print("=" * 80)

    total_frames = duration_seconds * fps

    # 根据文件扩展名选择编码器
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.avi' and codec == 'mp4v':
        codec = 'XVID'
        print(f"检测到.avi格式，自动切换编码器为: {codec}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    temp_path = output_path.replace(ext, f'_temp{ext}')

    out = cv2.VideoWriter(temp_path, fourcc, fps, resolution)

    if not out.isOpened():
        print(f"❌ 无法创建视频写入器，请检查编码器: {codec}")
        return False

    print(f"\n开始生成 {total_frames} 帧...")

    for frame_num in range(total_frames):
        frame = generate_qr_frame(frame_num, resolution, qr_size, prefix)
        out.write(frame)

        if (frame_num + 1) % (fps * 10) == 0:  # 每10秒报告一次
            progress = (frame_num + 1) / total_frames * 100
            print(f"  进度: {frame_num + 1}/{total_frames} ({progress:.1f}%)")

    out.release()
    print(f"\n✅ 初始视频生成完成")

    # 使用ffmpeg重新编码，确保兼容性
    print(f"\n重新编码为高质量视频...")
    import subprocess

    cmd = [
        'ffmpeg', '-y',
        '-i', temp_path,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',  # 高质量
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if result.returncode == 0:
        print(f"✅ 最终视频生成完成: {output_path}")

        # 显示文件大小
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"文件大小: {file_size_mb:.2f} MB")

        return True
    else:
        print(f"❌ ffmpeg重新编码失败")
        print(f"错误信息: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='生成QR码同步视频，每帧包含唯一的帧编号QR码',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成60秒、30fps的QR码视频（1920x1080）
  python generate_qr_sync_video.py --output qr_sync.mp4 --duration 60 --fps 30

  # 生成120秒、60fps的高分辨率QR码视频
  python generate_qr_sync_video.py --output qr_sync_60fps.mp4 --duration 120 --fps 60 --resolution 3840x2160

  # 生成带前缀的QR码（如"SYNC-000001"）
  python generate_qr_sync_video.py --output qr_sync.mp4 --duration 60 --prefix "SYNC-"

  # 生成AVI格式
  python generate_qr_sync_video.py --output qr_sync.avi --duration 60
        """
    )

    parser.add_argument('--output', type=str, required=True,
                       help='输出视频路径（.mp4或.avi）')
    parser.add_argument('--duration', type=int, default=60,
                       help='视频时长（秒），默认60秒')
    parser.add_argument('--fps', type=int, default=30,
                       help='帧率，默认30fps')
    parser.add_argument('--resolution', type=str, default='1920x1080',
                       help='分辨率，格式: WIDTHxHEIGHT，默认1920x1080')
    parser.add_argument('--qr-size', type=int, default=800,
                       help='QR码尺寸（像素），默认800')
    parser.add_argument('--prefix', type=str, default='',
                       help='QR码数据前缀（可选），如"SYNC-"')
    parser.add_argument('--codec', type=str, default='mp4v',
                       help='临时视频编码器，默认mp4v（AVI自动使用XVID）')

    args = parser.parse_args()

    # 解析分辨率
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"❌ 无效的分辨率格式: {args.resolution}")
        print("   请使用格式: WIDTHxHEIGHT (例如: 1920x1080)")
        return 1

    # 检查依赖
    try:
        import qrcode
    except ImportError:
        print("❌ 缺少依赖: qrcode")
        print("   请安装: pip install qrcode[pil]")
        return 1

    # 生成视频
    success = generate_qr_sync_video(
        output_path=args.output,
        duration_seconds=args.duration,
        fps=args.fps,
        resolution=resolution,
        qr_size=args.qr_size,
        prefix=args.prefix,
        codec=args.codec
    )

    if success:
        print("\n" + "=" * 80)
        print("✅ 完成！现在可以播放此视频并用相机录制")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ 生成失败")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
