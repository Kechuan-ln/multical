#!/usr/bin/env python3
"""生成多个不同offset的测试视频片段（只渲染10帧）用于快速对比。"""

import subprocess
import sys

# 测试不同的offset
test_offsets = [
    ("offset_neg10.58", -10.58),
    ("offset_pos10.58", +10.58),
    ("offset_0", 0.0),
    ("offset_neg5", -5.0),
    ("offset_neg15", -15.0),
]

print("="*80)
print("生成时间同步测试视频")
print("="*80)

base_cmd = [
    "/opt/homebrew/Caskroom/miniconda/base/envs/multical/bin/python",
    "project_skeleton_to_gopro_FINAL_FIXED.py",
    "--calibration", "/Volumes/FastACIS/GoPro/motion/calibration.json",
    "--mcal", "/Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal",
    "--skeleton", "skeleton_motion_3820_4081.json",
    "--gopro-video", "/Volumes/FastACIS/GoPro/motion/Cam4/Video.MP4",
]

for name, offset in test_offsets:
    output_file = f"sync_test_{name}.mp4"
    cmd = base_cmd + [
        "--output", output_file,
        "--sync-offset", str(offset)
    ]

    print(f"\n生成: {output_file} (offset={offset:+.2f}s)")

    # 修改脚本临时只渲染前10帧
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ 完成")
    else:
        print(f"  ✗ 失败")
        print(result.stderr)

print("\n" + "="*80)
print("完成！请检查生成的视频：")
for name, offset in test_offsets:
    print(f"  sync_test_{name}.mp4 (offset={offset:+.2f}s)")
print("\n对比哪个视频中skeleton和人的动作最同步")
print("="*80)