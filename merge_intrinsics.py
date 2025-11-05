#!/usr/bin/env python3
"""
合并多个内参JSON文件，用于联合外参标定

用法示例:
  # 方法1: 指定源文件和相机名称
  python merge_intrinsics.py \
    --input1 /path/to/intrinsic1.json \
    --cameras1 cam1 \
    --input2 /path/to/intrinsic2.json \
    --cameras2 primecolor \
    --output intrinsic_merged.json

  # 方法2: 自动合并所有相机
  python merge_intrinsics.py \
    --input1 intrinsic_gopro.json \
    --input2 intrinsic_primecolor.json \
    --output intrinsic_merged.json
"""

import json
import argparse
from pathlib import Path


def merge_intrinsics(input_files, camera_selections, output_json):
    """
    合并多个内参JSON文件

    Args:
        input_files: 输入JSON文件路径列表
        camera_selections: 每个文件要选择的相机列表，如果为None则选择所有相机
        output_json: 输出JSON文件路径
    """
    merged_data = {"cameras": {}}

    print("=" * 60)
    print("合并内参JSON文件")
    print("=" * 60)

    for idx, input_file in enumerate(input_files):
        print(f"\n[{idx + 1}/{len(input_files)}] 读取: {input_file}")

        if not Path(input_file).exists():
            print(f"  ⚠️  错误: 文件不存在")
            continue

        with open(input_file, 'r') as f:
            data = json.load(f)

        if 'cameras' not in data:
            print(f"  ⚠️  警告: 文件中没有 'cameras' 字段")
            continue

        # 确定要提取的相机
        if camera_selections and camera_selections[idx]:
            selected_cameras = camera_selections[idx]
        else:
            selected_cameras = list(data['cameras'].keys())

        print(f"  可用相机: {', '.join(data['cameras'].keys())}")
        print(f"  选择相机: {', '.join(selected_cameras)}")

        # 提取相机
        for cam in selected_cameras:
            if cam in data['cameras']:
                if cam in merged_data['cameras']:
                    print(f"  ⚠️  警告: {cam} 已存在，将被覆盖")
                merged_data['cameras'][cam] = data['cameras'][cam]

                # 显示相机信息
                cam_data = data['cameras'][cam]
                image_size = cam_data.get('image_size', [0, 0])
                K = cam_data.get('K', [[0, 0, 0], [0, 0, 0], [0, 0, 1]])
                fx, fy = K[0][0], K[1][1]
                print(f"    ✓ {cam}: {image_size[0]}x{image_size[1]}, "
                      f"fx={fx:.2f}, fy={fy:.2f}")

                # 如果有FOV信息，也显示
                if 'fov' in cam_data:
                    fov = cam_data['fov']
                    print(f"      FOV: {fov['horizontal']:.2f}° × {fov['vertical']:.2f}°")
            else:
                print(f"  ⚠️  错误: {cam} 不在文件中")

    # 保存合并结果
    print(f"\n保存到: {output_json}")
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ 合并完成！")
    print("=" * 60)
    print(f"输出文件: {output_json}")
    print(f"包含相机: {', '.join(sorted(merged_data['cameras'].keys()))}")
    print(f"总计: {len(merged_data['cameras'])} 个相机\n")


def main():
    parser = argparse.ArgumentParser(
        description='合并多个内参JSON文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从GoPro内参中提取cam1，从PrimeColor内参中提取primecolor
  python merge_intrinsics.py \\
    --input1 intrinsic_hyperoff_linear_60fps.json \\
    --cameras1 cam1 \\
    --input2 /Volumes/FastACIS/csldata/video/exandin/original/intrinsic.json \\
    --cameras2 primecolor \\
    --output intrinsic_merged.json

  # 合并所有相机（不指定cameras参数）
  python merge_intrinsics.py \\
    --input1 intrinsic_gopro.json \\
    --input2 intrinsic_primecolor.json \\
    --output intrinsic_merged.json
        """
    )

    # 输入文件1
    parser.add_argument('--input1', required=True,
                        help='第一个输入JSON文件路径')
    parser.add_argument('--cameras1', nargs='+',
                        help='从input1中选择的相机（可选，不指定则选择所有）')

    # 输入文件2
    parser.add_argument('--input2', required=True,
                        help='第二个输入JSON文件路径')
    parser.add_argument('--cameras2', nargs='+',
                        help='从input2中选择的相机（可选，不指定则选择所有）')

    # 可选的第三个文件
    parser.add_argument('--input3',
                        help='第三个输入JSON文件路径（可选）')
    parser.add_argument('--cameras3', nargs='+',
                        help='从input3中选择的相机（可选）')

    # 输出文件
    parser.add_argument('--output', required=True,
                        help='输出JSON文件路径')

    args = parser.parse_args()

    # 收集输入文件和相机选择
    input_files = [args.input1, args.input2]
    camera_selections = [args.cameras1, args.cameras2]

    if args.input3:
        input_files.append(args.input3)
        camera_selections.append(args.cameras3)

    # 执行合并
    merge_intrinsics(input_files, camera_selections, args.output)


if __name__ == '__main__':
    main()
