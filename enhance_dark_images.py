#!/usr/bin/env python3
"""
图像增强工具：针对暗环境primecolor图像进行预处理
提升亮度、对比度和降噪，改善ChArUco检测效果
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def enhance_dark_image(image, method='clahe'):
    """
    增强暗图像

    Args:
        image: 输入图像（BGR格式）
        method: 增强方法
            - 'clahe': 自适应直方图均衡化（推荐）
            - 'gamma': Gamma校正
            - 'hybrid': 混合方法

    Returns:
        增强后的图像
    """

    if method == 'clahe':
        # 方法1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # 最适合暗图像，局部对比度增强

        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 对L通道应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # 合并通道
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced

    elif method == 'gamma':
        # 方法2: Gamma校正
        # 简单有效，提升整体亮度
        gamma = 1.5  # >1 提亮，<1 变暗

        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")

        enhanced = cv2.LUT(image, table)
        return enhanced

    elif method == 'hybrid':
        # 方法3: 混合方法（最强但可能引入噪声）

        # 步骤1: Gamma校正提亮
        gamma = 1.3
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(image, table)

        # 步骤2: 降噪（可选）
        denoised = cv2.fastNlMeansDenoisingColored(gamma_corrected, None, 5, 5, 7, 21)

        # 步骤3: CLAHE增强对比度
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 步骤4: 锐化（可选，增强边缘）
        kernel = np.array([[-0.5, -0.5, -0.5],
                           [-0.5,  5.0, -0.5],
                           [-0.5, -0.5, -0.5]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # 混合锐化和增强图像
        final = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

        return final

    else:
        raise ValueError(f"Unknown method: {method}")


def enhance_directory(input_dir, output_dir, method='clahe', pattern='*.png'):
    """
    批量增强目录中的图像

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        method: 增强方法
        pattern: 文件匹配模式
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有匹配的图像
    image_files = list(input_path.glob(pattern))

    if not image_files:
        print(f"⚠️  没有找到匹配 {pattern} 的图像文件")
        return

    print(f"找到 {len(image_files)} 张图像")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"增强方法: {method}")
    print()

    # 批量处理
    for img_file in tqdm(image_files, desc="处理图像"):
        # 读取图像
        image = cv2.imread(str(img_file))

        if image is None:
            print(f"⚠️  无法读取: {img_file.name}")
            continue

        # 增强
        enhanced = enhance_dark_image(image, method=method)

        # 保存
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), enhanced)

    print(f"\n✅ 完成！增强后的图像已保存到: {output_dir}")


def compare_enhancement(image_path, output_path=None):
    """
    对比不同增强方法的效果

    Args:
        image_path: 输入图像路径
        output_path: 输出对比图路径（可选）
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return

    # 应用不同方法
    clahe = enhance_dark_image(image, method='clahe')
    gamma = enhance_dark_image(image, method='gamma')
    hybrid = enhance_dark_image(image, method='hybrid')

    # 调整大小以便并排显示
    h, w = image.shape[:2]
    scale = 0.4  # 缩放比例
    new_h, new_w = int(h * scale), int(w * scale)

    original_small = cv2.resize(image, (new_w, new_h))
    clahe_small = cv2.resize(clahe, (new_w, new_h))
    gamma_small = cv2.resize(gamma, (new_w, new_h))
    hybrid_small = cv2.resize(hybrid, (new_w, new_h))

    # 添加标签
    def add_label(img, text):
        labeled = img.copy()
        cv2.putText(labeled, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return labeled

    original_labeled = add_label(original_small, "Original")
    clahe_labeled = add_label(clahe_small, "CLAHE")
    gamma_labeled = add_label(gamma_small, "Gamma")
    hybrid_labeled = add_label(hybrid_small, "Hybrid")

    # 拼接图像（2x2布局）
    top_row = np.hstack([original_labeled, clahe_labeled])
    bottom_row = np.hstack([gamma_labeled, hybrid_labeled])
    comparison = np.vstack([top_row, bottom_row])

    # 显示
    cv2.imshow("Enhancement Comparison", comparison)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存
    if output_path:
        cv2.imwrite(output_path, comparison)
        print(f"✅ 对比图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='增强暗环境图像以改善ChArUco检测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 批量增强目录:
   python enhance_dark_images.py --input primecolor/ --output primecolor_enhanced/ --method clahe

2. 对比不同方法:
   python enhance_dark_images.py --compare frame_000000.png --output comparison.png

3. 快速测试:
   python enhance_dark_images.py --input primecolor/ --output test_output/ --method hybrid --pattern "frame_0000*.png"
        """
    )

    parser.add_argument('--input', '-i', help='输入目录或图像文件')
    parser.add_argument('--output', '-o', help='输出目录或图像文件')
    parser.add_argument('--method', '-m',
                       choices=['clahe', 'gamma', 'hybrid'],
                       default='clahe',
                       help='增强方法（默认: clahe）')
    parser.add_argument('--pattern', '-p',
                       default='*.png',
                       help='文件匹配模式（默认: *.png）')
    parser.add_argument('--compare', '-c',
                       help='对比模式：显示不同增强方法的效果')

    args = parser.parse_args()

    if args.compare:
        # 对比模式
        compare_enhancement(args.compare, args.output)

    elif args.input and args.output:
        # 批量处理模式
        if os.path.isdir(args.input):
            enhance_directory(args.input, args.output, args.method, args.pattern)
        else:
            # 单文件处理
            image = cv2.imread(args.input)
            enhanced = enhance_dark_image(image, method=args.method)
            cv2.imwrite(args.output, enhanced)
            print(f"✅ 已保存增强图像: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
