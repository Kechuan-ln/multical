#!/usr/bin/env python3
"""
生成QR码视频的metadata CSV文件

配合 generate_qr_sync_video.py 和 sync_with_qr_anchor.py 使用
"""

import argparse
import csv
from pathlib import Path


def generate_qr_metadata_csv(output_csv: str,
                              duration_seconds: int = 60,
                              fps: int = 30,
                              prefix: str = "") -> bool:
    """
    生成QR码metadata CSV文件

    Args:
        output_csv: 输出CSV路径
        duration_seconds: 视频时长（秒）
        fps: 帧率
        prefix: QR码前缀

    Returns:
        是否成功
    """
    total_frames = duration_seconds * fps

    print("=" * 80)
    print("生成QR码Anchor Metadata")
    print("=" * 80)
    print(f"输出文件: {output_csv}")
    print(f"时长: {duration_seconds}秒")
    print(f"帧率: {fps} fps")
    print(f"总帧数: {total_frames}")
    print(f"QR码前缀: '{prefix}' (如果为空则为纯数字)")
    print("=" * 80)

    # 写入CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_number', 'anchor_time', 'qr_data'])

        for frame_num in range(total_frames):
            anchor_time = frame_num / fps
            qr_data = f"{prefix}{frame_num:06d}"
            writer.writerow([frame_num, f"{anchor_time:.6f}", qr_data])

            if (frame_num + 1) % (fps * 10) == 0:  # 每10秒报告
                progress = (frame_num + 1) / total_frames * 100
                print(f"  进度: {frame_num + 1}/{total_frames} ({progress:.1f}%)")

    print(f"\n✅ Metadata生成完成")

    # 显示文件大小
    file_size_kb = Path(output_csv).stat().st_size / 1024
    print(f"文件大小: {file_size_kb:.2f} KB")

    # 显示前5行预览
    print(f"\n前5行预览:")
    with open(output_csv, 'r') as f:
        for i, line in enumerate(f):
            if i < 6:  # 包括标题行
                print(f"  {line.strip()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='生成QR码视频的anchor metadata CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用流程:
  1. 使用 generate_qr_sync_video.py 生成QR码视频
  2. 使用本工具生成对应的metadata CSV
  3. 使用 sync_with_qr_anchor.py 同步相机视频

示例:
  # 生成60秒、30fps的metadata
  python generate_qr_metadata.py --output qr_metadata.csv --duration 60 --fps 30

  # 生成带前缀的metadata
  python generate_qr_metadata.py --output qr_metadata.csv --duration 120 --fps 30 --prefix "SYNC-"

注意:
  - 参数必须与 generate_qr_sync_video.py 使用的参数一致
  - CSV格式: frame_number,anchor_time,qr_data
  - anchor_time = frame_number / fps
        """
    )

    parser.add_argument('--output', type=str, required=True,
                       help='输出CSV文件路径')
    parser.add_argument('--duration', type=int, default=60,
                       help='视频时长（秒），默认60秒')
    parser.add_argument('--fps', type=int, default=30,
                       help='帧率，默认30fps')
    parser.add_argument('--prefix', type=str, default='',
                       help='QR码数据前缀（可选），如"SYNC-"')

    args = parser.parse_args()

    # 生成CSV
    success = generate_qr_metadata_csv(
        output_csv=args.output,
        duration_seconds=args.duration,
        fps=args.fps,
        prefix=args.prefix
    )

    if success:
        print("\n" + "=" * 80)
        print("✅ 完成！现在可以使用此CSV进行视频同步")
        print("=" * 80)
        print(f"\n使用方法:")
        print(f"  python sync_with_qr_anchor.py \\")
        print(f"    --video1 camera1.mp4 \\")
        print(f"    --video2 camera2.mp4 \\")
        print(f"    --output camera2_synced.mp4 \\")
        print(f"    --anchor-csv {args.output} \\")
        print(f"    --anchor-fps {args.fps}")
        if args.prefix:
            print(f"    --prefix \"{args.prefix}\"")
        return 0
    else:
        print("\n❌ 生成失败")
        return 1


if __name__ == "__main__":
    exit(main())
