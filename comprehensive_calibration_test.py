#!/usr/bin/env python3
"""
PrimeColor标定改进 - 综合批量测试
测试所有primecolor图像，对比不同配置和增强方法的效果
"""

import cv2
import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加multical路径
sys.path.insert(0, '/Volumes/FastACIS/annotation_pipeline/multical')


@dataclass
class TestConfig:
    """测试配置"""
    name: str
    yaml_config: str
    enhance_method: str = None  # None, 'clahe', 'gamma', 'hybrid'

    def __str__(self):
        enhance = f"_{self.enhance_method}" if self.enhance_method else ""
        return f"{self.name}{enhance}"


@dataclass
class DetectionResult:
    """单张图像的检测结果"""
    image_name: str
    config_name: str
    num_markers: int
    num_corners: int
    success: bool
    min_points: int
    detection_time_ms: float

    def to_dict(self):
        return asdict(self)


class ComprehensiveCalibrationTest:
    """综合标定测试"""

    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        self.results = []

        # 定义测试配置
        self.test_configs = [
            TestConfig("original",
                      "/Volumes/FastACIS/annotation_pipeline/multical/asset/charuco_b1_2.yaml"),
            TestConfig("original",
                      "/Volumes/FastACIS/annotation_pipeline/multical/asset/charuco_b1_2.yaml",
                      "clahe"),
            TestConfig("optimized",
                      "/Volumes/FastACIS/annotation_pipeline/multical/asset/charuco_b1_2_dark.yaml"),
            TestConfig("optimized",
                      "/Volumes/FastACIS/annotation_pipeline/multical/asset/charuco_b1_2_dark.yaml",
                      "clahe"),
            TestConfig("optimized",
                      "/Volumes/FastACIS/annotation_pipeline/multical/asset/charuco_b1_2_dark.yaml",
                      "gamma"),
            TestConfig("optimized",
                      "/Volumes/FastACIS/annotation_pipeline/multical/asset/charuco_b1_2_dark.yaml",
                      "hybrid"),
        ]

        # ChArUco板配置（B1板）
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
        self.board = cv2.aruco.CharucoBoard_create(7, 9, 0.095, 0.075, self.aruco_dict)
        self.theoretical_max_corners = 48  # B1板理论最大角点数

    def load_aruco_params(self, yaml_path: str) -> cv2.aruco.DetectorParameters:
        """从YAML加载ArUco参数"""
        import yaml

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        aruco_params_dict = config.get('aruco_params', {})
        params = cv2.aruco.DetectorParameters_create()

        for key, value in aruco_params_dict.items():
            if hasattr(params, key):
                setattr(params, key, value)

        return params, config.get('common', {}).get('min_points', 20)

    def enhance_image(self, image, method: str):
        """图像增强"""
        if method is None:
            return image

        if method == 'clahe':
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        elif method == 'gamma':
            gamma = 1.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)

        elif method == 'hybrid':
            # Gamma校正
            gamma = 1.3
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(image, table)

            # 降噪
            denoised = cv2.fastNlMeansDenoisingColored(gamma_corrected, None, 5, 5, 7, 21)

            # CLAHE
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return image

    def detect_charuco(self, image, config: TestConfig) -> DetectionResult:
        """检测ChArUco板"""
        start_time = time.time()

        # 加载参数
        aruco_params, min_points = self.load_aruco_params(config.yaml_config)

        # 图像增强
        processed = self.enhance_image(image, config.enhance_method)

        # 转灰度
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed

        # 检测markers
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=aruco_params)

        num_markers = len(marker_ids) if marker_ids is not None else 0
        num_corners = 0

        # 插值角点
        if marker_ids is not None and len(marker_ids) > 0:
            _, corners, ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board)
            num_corners = len(corners) if corners is not None else 0

        detection_time = (time.time() - start_time) * 1000  # ms

        return DetectionResult(
            image_name="",  # 后面填充
            config_name=str(config),
            num_markers=num_markers,
            num_corners=num_corners,
            success=(num_corners >= min_points),
            min_points=min_points,
            detection_time_ms=detection_time
        )

    def test_single_image(self, image_path: Path) -> List[DetectionResult]:
        """测试单张图像的所有配置"""
        image = cv2.imread(str(image_path))
        if image is None:
            return []

        results = []
        for config in self.test_configs:
            result = self.detect_charuco(image, config)
            result.image_name = image_path.name
            results.append(result)

        return results

    def run_batch_test(self, pattern: str = "*.png", limit: int = None):
        """批量测试"""
        image_files = sorted(self.image_dir.glob(pattern))

        if limit:
            image_files = image_files[:limit]

        total_images = len(image_files)

        if total_images == 0:
            print(f"❌ 没有找到匹配 {pattern} 的图像")
            return

        print(f"\n{'='*80}")
        print(f"综合标定测试")
        print(f"{'='*80}")
        print(f"图像目录: {self.image_dir}")
        print(f"测试图像数: {total_images}")
        print(f"测试配置数: {len(self.test_configs)}")
        print(f"总测试数: {total_images * len(self.test_configs)}")
        print(f"{'='*80}\n")

        # 显示测试配置
        print("测试配置:")
        for i, config in enumerate(self.test_configs, 1):
            enhance_str = f" + {config.enhance_method.upper()}" if config.enhance_method else ""
            print(f"  {i}. {config.name}{enhance_str}")
        print()

        # 批量测试（显示进度条）
        for image_file in tqdm(image_files, desc="测试进度", unit="张"):
            results = self.test_single_image(image_file)
            self.results.extend(results)

        print("\n✅ 测试完成！\n")

    def generate_statistics(self) -> Dict:
        """生成统计数据"""
        stats = defaultdict(lambda: {
            'total': 0,
            'success': 0,
            'total_markers': 0,
            'total_corners': 0,
            'detection_times': []
        })

        for result in self.results:
            config = result.config_name
            stats[config]['total'] += 1
            if result.success:
                stats[config]['success'] += 1
            stats[config]['total_markers'] += result.num_markers
            stats[config]['total_corners'] += result.num_corners
            stats[config]['detection_times'].append(result.detection_time_ms)

        # 计算平均值和百分比
        summary = {}
        for config, data in stats.items():
            total = data['total']
            if total == 0:
                continue

            summary[config] = {
                'total_images': total,
                'success_count': data['success'],
                'success_rate': data['success'] / total * 100,
                'avg_markers': data['total_markers'] / total,
                'avg_corners': data['total_corners'] / total,
                'corner_detection_rate': (data['total_corners'] / total) / self.theoretical_max_corners * 100,
                'avg_detection_time_ms': np.mean(data['detection_times']),
                'std_detection_time_ms': np.std(data['detection_times'])
            }

        return summary

    def print_summary(self, stats: Dict):
        """打印统计摘要"""
        print(f"\n{'='*100}")
        print("测试结果汇总")
        print(f"{'='*100}\n")

        # 表头
        print(f"{'配置':<30} {'成功率':<12} {'平均Marker':<12} {'平均角点':<12} {'检测率%':<12} {'平均耗时(ms)'}")
        print("-" * 100)

        # 按配置排序输出
        for config in sorted(stats.keys()):
            data = stats[config]
            print(f"{config:<30} "
                  f"{data['success_rate']:>6.1f}% ({data['success_count']}/{data['total_images']})   "
                  f"{data['avg_markers']:>6.1f}        "
                  f"{data['avg_corners']:>6.1f}        "
                  f"{data['corner_detection_rate']:>6.1f}%      "
                  f"{data['avg_detection_time_ms']:>6.1f}")

        print("\n" + "="*100)

        # 改进对比
        self.print_improvement_analysis(stats)

    def print_improvement_analysis(self, stats: Dict):
        """打印改进分析"""
        print("\n改进效果分析:")
        print("-" * 100)

        # 找到baseline（original无增强）
        baseline_key = None
        for key in stats.keys():
            if 'original' in key and 'clahe' not in key and 'gamma' not in key and 'hybrid' not in key:
                baseline_key = key
                break

        if not baseline_key:
            print("未找到baseline配置")
            return

        baseline = stats[baseline_key]

        print(f"\n基准配置: {baseline_key}")
        print(f"  - 成功率: {baseline['success_rate']:.1f}%")
        print(f"  - 平均角点: {baseline['avg_corners']:.1f}")
        print(f"  - 检测率: {baseline['corner_detection_rate']:.1f}%")

        print(f"\n与基准对比:")
        print(f"{'配置':<35} {'成功率变化':<15} {'角点数变化':<15} {'检测率变化'}")
        print("-" * 100)

        for config, data in sorted(stats.items()):
            if config == baseline_key:
                continue

            success_diff = data['success_rate'] - baseline['success_rate']
            corners_diff = data['avg_corners'] - baseline['avg_corners']
            detection_diff = data['corner_detection_rate'] - baseline['corner_detection_rate']

            print(f"{config:<35} "
                  f"{success_diff:>+6.1f}%         "
                  f"{corners_diff:>+6.1f}         "
                  f"{detection_diff:>+6.1f}%")

        # 找出最佳配置
        best_config = max(stats.items(), key=lambda x: x[1]['success_rate'])
        print(f"\n🏆 最佳配置: {best_config[0]}")
        print(f"   成功率: {best_config[1]['success_rate']:.1f}%")
        print(f"   比基准提升: +{best_config[1]['success_rate'] - baseline['success_rate']:.1f}%")
        print(f"   平均角点: {best_config[1]['avg_corners']:.1f} (基准: {baseline['avg_corners']:.1f})")

        print("="*100)

    def save_results(self, output_file: str):
        """保存详细结果到JSON"""
        data = {
            'test_info': {
                'image_dir': str(self.image_dir),
                'total_images': len(set(r.image_name for r in self.results)),
                'configs_tested': len(self.test_configs),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': [r.to_dict() for r in self.results],
            'statistics': self.generate_statistics()
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ 详细结果已保存: {output_file}")

    def plot_comparison(self, output_file: str):
        """生成可视化对比图"""
        stats = self.generate_statistics()

        configs = list(stats.keys())
        success_rates = [stats[c]['success_rate'] for c in configs]
        avg_corners = [stats[c]['avg_corners'] for c in configs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 子图1: 成功率对比
        colors = ['#FF6B6B' if 'original' in c and not any(x in c for x in ['clahe', 'gamma', 'hybrid'])
                 else '#4ECDC4' if 'optimized' in c
                 else '#95E1D3' for c in configs]

        bars1 = ax1.bar(range(len(configs)), success_rates, color=colors, alpha=0.8)
        ax1.set_xlabel('配置', fontsize=12)
        ax1.set_ylabel('成功率 (%)', fontsize=12)
        ax1.set_title('ChArUco检测成功率对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # 在柱子上添加数值
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=9)

        # 子图2: 平均角点数对比
        bars2 = ax2.bar(range(len(configs)), avg_corners, color=colors, alpha=0.8)
        ax2.set_xlabel('配置', fontsize=12)
        ax2.set_ylabel('平均角点数', fontsize=12)
        ax2.set_title('平均检测角点数对比 (理论最大: 48)', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        ax2.axhline(y=48, color='r', linestyle='--', alpha=0.5, label='理论最大值')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend()

        # 在柱子上添加数值
        for bar, corners in zip(bars2, avg_corners):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{corners:.1f}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化图表已保存: {output_file}")

        # 关闭图表以释放内存
        plt.close()

    def generate_report(self, report_file: str):
        """生成Markdown格式的测试报告"""
        stats = self.generate_statistics()

        # 找baseline
        baseline_key = None
        for key in stats.keys():
            if 'original' in key and not any(x in key for x in ['clahe', 'gamma', 'hybrid']):
                baseline_key = key
                break

        baseline = stats.get(baseline_key, {})

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# PrimeColor标定改进测试报告\n\n")
            f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**图像目录**: `{self.image_dir}`\n\n")
            f.write(f"**测试图像数**: {len(set(r.image_name for r in self.results))}\n\n")
            f.write(f"**测试配置数**: {len(self.test_configs)}\n\n")

            f.write("---\n\n")
            f.write("## 测试结果汇总\n\n")

            # 表格
            f.write("| 配置 | 成功率 | 平均Marker | 平均角点 | 检测率% | 平均耗时(ms) |\n")
            f.write("|------|--------|------------|----------|---------|-------------|\n")

            for config in sorted(stats.keys()):
                data = stats[config]
                f.write(f"| {config} | "
                       f"{data['success_rate']:.1f}% ({data['success_count']}/{data['total_images']}) | "
                       f"{data['avg_markers']:.1f} | "
                       f"{data['avg_corners']:.1f} | "
                       f"{data['corner_detection_rate']:.1f}% | "
                       f"{data['avg_detection_time_ms']:.1f} |\n")

            f.write("\n---\n\n")
            f.write("## 改进效果分析\n\n")

            if baseline_key:
                f.write(f"### 基准配置: `{baseline_key}`\n\n")
                f.write(f"- 成功率: **{baseline['success_rate']:.1f}%**\n")
                f.write(f"- 平均角点: **{baseline['avg_corners']:.1f}**\n")
                f.write(f"- 检测率: **{baseline['corner_detection_rate']:.1f}%**\n\n")

                f.write("### 各配置与基准对比\n\n")
                f.write("| 配置 | 成功率变化 | 角点数变化 | 检测率变化 |\n")
                f.write("|------|-----------|-----------|----------|\n")

                for config, data in sorted(stats.items()):
                    if config == baseline_key:
                        continue

                    success_diff = data['success_rate'] - baseline['success_rate']
                    corners_diff = data['avg_corners'] - baseline['avg_corners']
                    detection_diff = data['corner_detection_rate'] - baseline['corner_detection_rate']

                    f.write(f"| {config} | "
                           f"{success_diff:+.1f}% | "
                           f"{corners_diff:+.1f} | "
                           f"{detection_diff:+.1f}% |\n")

                # 最佳配置
                best_config = max(stats.items(), key=lambda x: x[1]['success_rate'])
                f.write(f"\n### 🏆 推荐配置\n\n")
                f.write(f"**最佳配置**: `{best_config[0]}`\n\n")
                f.write(f"- 成功率: **{best_config[1]['success_rate']:.1f}%**\n")
                f.write(f"- 比基准提升: **+{best_config[1]['success_rate'] - baseline['success_rate']:.1f}%**\n")
                f.write(f"- 平均角点: **{best_config[1]['avg_corners']:.1f}** (基准: {baseline['avg_corners']:.1f})\n")
                f.write(f"- 检测率: **{best_config[1]['corner_detection_rate']:.1f}%** (基准: {baseline['corner_detection_rate']:.1f}%)\n\n")

            f.write("---\n\n")
            f.write("## 使用建议\n\n")

            best = max(stats.items(), key=lambda x: x[1]['success_rate'])

            if 'optimized' in best[0]:
                f.write("✅ **建议使用优化配置**\n\n")
                f.write("在 `run_gopro_primecolor_calibration.py` 中修改:\n")
                f.write("```python\n")
                f.write('BOARD_CONFIG = "./asset/charuco_b1_2_dark.yaml"\n')
                f.write("```\n\n")

            if best[1].get('success_rate', 0) - baseline.get('success_rate', 0) > 30:
                f.write("📈 **改进显著！**建议立即应用最佳配置重新标定\n\n")
            elif best[1].get('success_rate', 0) - baseline.get('success_rate', 0) > 10:
                f.write("📊 **有一定改进**，建议尝试应用优化配置\n\n")
            else:
                f.write("⚠️ **改进有限**，可能需要从硬件层面改善（增加光照、更换标定板等）\n\n")

        print(f"✅ 测试报告已保存: {report_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='PrimeColor标定改进 - 综合批量测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 测试所有图像（推荐）:
   python comprehensive_calibration_test.py

2. 测试前50张图像（快速验证）:
   python comprehensive_calibration_test.py --limit 50

3. 指定输出目录:
   python comprehensive_calibration_test.py --output test_results/

4. 自定义图像目录:
   python comprehensive_calibration_test.py --dir /path/to/primecolor/frames
        """
    )

    parser.add_argument('--dir', '-d',
                       default="/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor",
                       help='primecolor图像目录')
    parser.add_argument('--limit', '-l', type=int,
                       help='限制测试图像数量（用于快速验证）')
    parser.add_argument('--pattern', '-p', default='*.png',
                       help='文件匹配模式')
    parser.add_argument('--output', '-o', default='.',
                       help='输出目录')

    args = parser.parse_args()

    # 检查图像目录
    if not os.path.exists(args.dir):
        print(f"❌ 图像目录不存在: {args.dir}")
        return 1

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行测试
    tester = ComprehensiveCalibrationTest(args.dir)
    tester.run_batch_test(pattern=args.pattern, limit=args.limit)

    # 生成统计
    stats = tester.generate_statistics()
    tester.print_summary(stats)

    # 保存结果
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    json_file = output_dir / f"calibration_test_results_{timestamp}.json"
    plot_file = output_dir / f"calibration_test_comparison_{timestamp}.png"
    report_file = output_dir / f"calibration_test_report_{timestamp}.md"

    tester.save_results(str(json_file))
    tester.plot_comparison(str(plot_file))
    tester.generate_report(str(report_file))

    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"{'='*80}")
    print(f"\n📁 输出文件:")
    print(f"   - JSON结果: {json_file}")
    print(f"   - 可视化图: {plot_file}")
    print(f"   - 测试报告: {report_file}")
    print(f"\n💡 下一步:")
    print(f"   1. 查看测试报告: cat {report_file}")
    print(f"   2. 查看可视化图: open {plot_file}")
    print(f"   3. 如果改进显著，应用推荐配置重新标定")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
