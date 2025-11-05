#!/usr/bin/env python3
"""
直接运行外参标定（跳过前面的步骤）
适用场景：已经完成了帧提取和内参准备，只需要运行calibration
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# ==============================================================================
# 配置参数
# ==============================================================================

# 工作目录（包含frames/和intrinsic_merged.json）
EXTRINSICS_DIR = "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics"

# 图像目录（包含cam4/和primecolor/子目录）
FRAMES_DIR = os.path.join(EXTRINSICS_DIR, "frames")

# 合并后的内参文件
INTRINSIC_FILE = os.path.join(EXTRINSICS_DIR, "intrinsic_merged.json")

# 标定板配置（使用优化版本）
BOARD_CONFIG = "./asset/charuco_b1_2_dark.yaml"

# Multical目录
MULTICAL_DIR = "/Volumes/FastACIS/annotation_pipeline/multical"

# 限制使用的图像数量（设为1000或更大值以使用所有图像）
LIMIT_IMAGES = 1000

# ==============================================================================
# 主程序
# ==============================================================================

def main():
    print("=" * 80)
    print("直接运行外参标定")
    print("=" * 80)
    print()

    # 检查文件和目录
    print("检查输入文件和目录...")

    if not os.path.exists(FRAMES_DIR):
        print(f"❌ 图像目录不存在: {FRAMES_DIR}")
        return 1

    if not os.path.exists(INTRINSIC_FILE):
        print(f"❌ 内参文件不存在: {INTRINSIC_FILE}")
        return 1

    if not os.path.exists(MULTICAL_DIR):
        print(f"❌ Multical目录不存在: {MULTICAL_DIR}")
        return 1

    board_config_path = os.path.join(MULTICAL_DIR, BOARD_CONFIG)
    if not os.path.exists(board_config_path):
        print(f"❌ 标定板配置不存在: {board_config_path}")
        return 1

    print(f"✅ 图像目录: {FRAMES_DIR}")
    print(f"✅ 内参文件: {INTRINSIC_FILE}")
    print(f"✅ 标定板配置: {BOARD_CONFIG}")
    print()

    # 检查子目录
    cam_dirs = [d for d in os.listdir(FRAMES_DIR)
                if os.path.isdir(os.path.join(FRAMES_DIR, d)) and not d.startswith('.')]

    if not cam_dirs:
        print(f"❌ 在 {FRAMES_DIR} 中没有找到相机目录")
        return 1

    print(f"找到相机目录: {cam_dirs}")

    # 统计图像数量
    for cam_dir in cam_dirs:
        cam_path = os.path.join(FRAMES_DIR, cam_dir)
        images = [f for f in os.listdir(cam_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  {cam_dir}: {len(images)} 张图像")

    print()

    # 构建calibrate.py命令
    cmd = [
        sys.executable, 'calibrate.py',
        '--boards', BOARD_CONFIG,
        '--image_path', FRAMES_DIR,
        '--calibration', INTRINSIC_FILE,
        '--fix_intrinsic',
        '--limit_images', str(LIMIT_IMAGES),
        '--vis'
    ]

    print("运行外参标定...")
    print(f"工作目录: {MULTICAL_DIR}")
    print(f"命令: {' '.join(cmd)}")
    print()
    print("=" * 80)
    print()

    # 运行
    result = subprocess.run(
        cmd,
        cwd=MULTICAL_DIR,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print()
        print("=" * 80)
        print("❌ 外参标定失败")
        print("=" * 80)
        return 1

    # 检查输出
    calib_json = os.path.join(FRAMES_DIR, 'calibration.json')

    print()
    print("=" * 80)

    if os.path.exists(calib_json):
        print("✅ 外参标定完成！")
        print("=" * 80)
        print()

        # 读取并显示结果
        with open(calib_json, 'r') as f:
            data = json.load(f)

        rms = data.get('rms', 999)
        print(f"🎯 标定结果:")
        print(f"   RMS误差: {rms:.4f} 像素")

        if rms < 1.0:
            print(f"   ✅ 标定质量: 优秀 (RMS < 1.0)")
        elif rms < 1.5:
            print(f"   ✅ 标定质量: 良好 (RMS < 1.5)")
        else:
            print(f"   ⚠️  标定质量: 一般 (建议RMS < 1.0)")

        print()
        print(f"📁 输出文件:")
        print(f"   外参标定: {calib_json}")
        print(f"   可视化:   {os.path.join(FRAMES_DIR, 'vis')}")

        print()
        print(f"📊 下一步:")
        print(f"   1. 查看可视化验证检测质量:")
        print(f"      open {os.path.join(FRAMES_DIR, 'vis')}")
        print(f"   2. 查看详细标定结果:")
        print(f"      cat {calib_json} | python -m json.tool")
        print(f"   3. 如果满意，复制到你的项目中使用")

        # 显示检测统计
        print()
        print(f"📈 检测统计:")
        base2cam = data.get('camera_base2cam', {})
        for cam_name in base2cam.keys():
            print(f"   {cam_name}: 标定成功")

        return 0
    else:
        print("❌ 未找到标定结果文件")
        print("=" * 80)
        print(f"   预期文件: {calib_json}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
