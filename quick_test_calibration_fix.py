#!/usr/bin/env python3
"""
快速验证PrimeColor标定改进效果
在5分钟内对比原始配置 vs 优化配置 vs 图像增强
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# 添加multical路径
sys.path.insert(0, '/Volumes/FastACIS/annotation_pipeline/multical')


def quick_detect_test(image_path: str):
    """快速测试单张图像的检测效果"""
    print(f"\n{'='*80}")
    print(f"快速检测测试: {Path(image_path).name}")
    print(f"{'='*80}\n")

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 配置1: 原始配置
    print("1. 原始配置（charuco_b1_2.yaml）:")
    result1 = detect_with_params(
        gray,
        adaptiveThreshWinSizeMax=23,
        adaptiveThreshConstant=7,
        minMarkerPerimeterRate=0.03
    )
    print(f"   Markers: {result1['markers']}, Corners: {result1['corners']}")

    # 配置2: 优化配置
    print("\n2. 优化配置（charuco_b1_2_dark.yaml）:")
    result2 = detect_with_params(
        gray,
        adaptiveThreshWinSizeMax=35,
        adaptiveThreshConstant=10,
        minMarkerPerimeterRate=0.01,
        errorCorrectionRate=0.8,
        cornerRefinementMethod=2
    )
    print(f"   Markers: {result2['markers']}, Corners: {result2['corners']}")
    improvement = (result2['corners'] - result1['corners']) / max(result1['corners'], 1) * 100
    print(f"   改进: {improvement:+.1f}%")

    # 配置3: CLAHE增强 + 优化配置
    print("\n3. CLAHE增强 + 优化配置:")
    enhanced = enhance_clahe(gray)
    result3 = detect_with_params(
        enhanced,
        adaptiveThreshWinSizeMax=35,
        adaptiveThreshConstant=10,
        minMarkerPerimeterRate=0.01,
        errorCorrectionRate=0.8,
        cornerRefinementMethod=2
    )
    print(f"   Markers: {result3['markers']}, Corners: {result3['corners']}")
    improvement_total = (result3['corners'] - result1['corners']) / max(result1['corners'], 1) * 100
    print(f"   总改进: {improvement_total:+.1f}%")

    # 可视化
    print("\n生成可视化对比...")
    vis_compare = create_comparison_vis(image, [result1, result2, result3],
                                       ['Original', 'Optimized Params', 'Params + CLAHE'])

    output_path = f"comparison_{Path(image_path).stem}.png"
    cv2.imwrite(output_path, vis_compare)
    print(f"✅ 对比图已保存: {output_path}")

    # 总结
    print(f"\n{'='*80}")
    print("测试总结:")
    print(f"  原始检测:       {result1['corners']} 角点")
    print(f"  优化参数:       {result2['corners']} 角点 ({improvement:+.1f}%)")
    print(f"  参数+增强:      {result3['corners']} 角点 ({improvement_total:+.1f}%)")
    print(f"{'='*80}\n")


def detect_with_params(gray, **params):
    """使用指定参数检测ChArUco"""
    # 创建ArUco检测器配置
    aruco_params = cv2.aruco.DetectorParameters_create()

    # 设置参数
    for key, value in params.items():
        if hasattr(aruco_params, key):
            setattr(aruco_params, key, value)

    # ChArUco板配置（B1板）
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
    board = cv2.aruco.CharucoBoard_create(7, 9, 0.095, 0.075, aruco_dict)

    # 检测
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    num_markers = len(marker_ids) if marker_ids is not None else 0
    num_corners = 0

    if marker_ids is not None and len(marker_ids) > 0:
        _, corners, ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
        num_corners = len(corners) if corners is not None else 0

    return {
        'markers': num_markers,
        'corners': num_corners,
        'marker_corners': marker_corners,
        'marker_ids': marker_ids
    }


def enhance_clahe(gray):
    """CLAHE增强"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def create_comparison_vis(image, results, titles):
    """创建对比可视化"""
    vis_images = []

    for result, title in zip(results, titles):
        vis = image.copy()

        # 绘制marker
        if result['marker_ids'] is not None and len(result['marker_ids']) > 0:
            cv2.aruco.drawDetectedMarkers(vis, result['marker_corners'], result['marker_ids'])

        # 添加标题和统计
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, f"Markers: {result['markers']}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"Corners: {result['corners']}", (10, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 缩小以便并排显示
        h, w = vis.shape[:2]
        scale = 0.4
        vis_small = cv2.resize(vis, (int(w * scale), int(h * scale)))
        vis_images.append(vis_small)

    # 拼接
    comparison = np.hstack(vis_images)
    return comparison


def main():
    import argparse

    parser = argparse.ArgumentParser(description='快速验证PrimeColor标定改进效果')
    parser.add_argument('--image', '-i',
                       default="/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor/frame_000000.png",
                       help='测试图像路径')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ 图像不存在: {args.image}")
        print("\n请指定正确的primecolor图像路径，例如:")
        print('  python quick_test_calibration_fix.py --image "path/to/primecolor/frame.png"')
        return

    quick_detect_test(args.image)

    print("💡 提示:")
    print("  1. 如果改进明显，建议修改 run_gopro_primecolor_calibration.py")
    print("     使用 charuco_b1_2_dark.yaml 配置")
    print("  2. 如果需要更大改进，可以添加图像预处理（CLAHE）")
    print("  3. 详细使用指南参见: PRIMECOLOR_CALIBRATION_FIX.md")


if __name__ == "__main__":
    main()
