# PrimeColor暗图像标定改进 - 快速开始

## 🎯 问题概述

你的PrimeColor相机标定检测率远低于GoPro：
- **GoPro cam4**: 23.4点/帧 ✅
- **PrimeColor**: 5.6点/帧 ❌ (仅24%的检测率)
- **成功配对**: 574张图像中只有100对成功

**根本原因**: 图像偏暗 + 默认检测参数不适合低对比度场景

---

## ⚡ 5分钟快速验证

运行快速测试看看改进效果：

```bash
cd /Volumes/FastACIS/annotation_pipeline

# 测试单张图像
python quick_test_calibration_fix.py \
  --image "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor/frame_000000.png"

# 查看生成的对比图
open comparison_frame_000000.png
```

**期望看到**: 优化配置 vs 原始配置的检测角点数对比

---

## 🚀 推荐解决方案

### 方案A：仅优化参数（最简单，效果30-50%）

**1. 修改配置文件**

编辑 `run_gopro_primecolor_calibration.py` 第74行：

```python
# 原来
BOARD_CONFIG = "./asset/charuco_b1_2.yaml"

# 改为
BOARD_CONFIG = "./asset/charuco_b1_2_dark.yaml"
```

**2. 重新运行标定**

```bash
conda activate multical
python run_gopro_primecolor_calibration.py
```

**3. 验证改进**

查看新的标定输出，对比：
- 成功配对帧数（目标：从100 → 150+）
- RMS误差（目标：从1.402 → <1.2）

---

### 方案B：参数 + 图像增强（更好，效果80-120%）

**1. 修改配置**（同方案A）

**2. 添加图像增强**

编辑 `calibrate_gopro_primecolor_extrinsics.py`，在 `extract_sync_frames` 函数中第52行之前添加：

```python
# 在 cv2.imwrite(output_path, frame) 之前
if cam_name == 'primecolor':
    # 导入增强函数
    import sys
    sys.path.append('/Volumes/FastACIS/annotation_pipeline')
    from enhance_dark_images import enhance_dark_image

    # 增强图像
    frame = enhance_dark_image(frame, method='clahe')
```

**3. 重新运行标定**

```bash
python run_gopro_primecolor_calibration.py
```

**预期改进**：
- 成功配对帧数：100 → **180-220** (+80-120%)
- RMS误差：1.402 → **0.8-1.0** (-30-40%)

---

## 📊 完整批量测试（可选）

如果想看到所有574张图像的详细对比：

```bash
# 测试原始配置
python test_charuco_detection.py \
  --dir "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor" \
  --output baseline.json

# 测试优化配置 + CLAHE增强
python test_charuco_detection.py \
  --dir "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor" \
  --enhance clahe \
  --output optimized.json

# 对比统计
python -c "
import json
with open('baseline.json') as f: b = json.load(f)
with open('optimized.json') as f: o = json.load(f)
print(f'原始成功率: {sum(r[\"success\"] for r in b)}/{len(b)} = {sum(r[\"success\"] for r in b)/len(b)*100:.1f}%')
print(f'优化成功率: {sum(r[\"success\"] for r in o)}/{len(o)} = {sum(r[\"success\"] for r in o)/len(o)*100:.1f}%')
"
```

---

## 📁 文件说明

```
/Volumes/FastACIS/annotation_pipeline/
├── multical/asset/
│   └── charuco_b1_2_dark.yaml          # 🔧 优化的检测配置
├── enhance_dark_images.py              # 🖼️  图像增强工具
├── test_charuco_detection.py           # 🧪 批量检测测试
├── quick_test_calibration_fix.py       # ⚡ 快速验证脚本
├── PRIMECOLOR_CALIBRATION_FIX.md       # 📖 详细技术文档
├── CALIBRATION_ANALYSIS_SUMMARY.md     # 📊 深度分析报告
└── README_CALIBRATION_FIX.md           # 📘 本文件
```

---

## 🔧 参数调优（高级）

如果标准方案效果不佳，可以手动微调：

### ArUco检测参数（charuco_b1_2_dark.yaml）

```yaml
aruco_params:
  # 检测到的marker太少 → 增大这些值
  adaptiveThreshWinSizeMax: 40        # 默认35，尝试40-50
  adaptiveThreshConstant: 12          # 默认10，尝试12-15

  # 检测到很多错误marker → 降低容错率
  errorCorrectionRate: 0.6            # 默认0.8，尝试0.5-0.7

  # 角点位置不准确 → 增强细化
  cornerRefinementWinSize: 7          # 默认5，尝试7-9
  cornerRefinementMaxIterations: 100  # 默认50，尝试100-150
```

### 图像增强参数（enhance_dark_images.py）

```python
# CLAHE增强强度（第30行）
clahe = cv2.createCLAHE(
    clipLimit=4.0,        # 默认3.0，尝试2.5-5.0
    tileGridSize=(8, 8)   # 默认(8,8)，尝试(4,4)或(16,16)
)

# Gamma校正亮度（第56行）
gamma = 1.8              # 默认1.5，尝试1.3-2.0
```

---

## 💡 其他建议

### 短期改进
1. ✅ 使用 `charuco_b1_2_dark.yaml` 配置
2. ✅ 添加CLAHE图像增强
3. 增加提取帧率：`EXTRINSIC_FPS = 3`（原1）
4. 增加最大帧数：`EXTRINSIC_MAX_FRAMES = 200`（原100）

### 长期根本解决
1. 🔦 **增加拍摄光照**：使用补光灯或提高环境亮度
2. 📷 **调整相机设置**：
   - 提高ISO（如果噪声可接受）
   - 降低快门速度（确保无运动模糊）
   - 开大光圈
3. 📐 **更换标定板**：
   - 使用更大尺寸（当前B1，1.2m x 0.9m）
   - 高对比度打印（激光打印 + 磨砂表面）
4. 🔧 **检查硬件**：
   - 清洁镜头
   - 确认sensor正常工作

---

## ❓ 常见问题

### Q: 为什么我测试的图像改进不明显？

**A**: 你可能测试了已经检测成功的"优秀样本"。574张图像中只有约100张能成功配对，说明大量图像检测失败。需要批量测试所有图像才能看到真实改进。

### Q: 改进后RMS误差反而增加了？

**A**: 可能因为：
1. 增强过度引入噪声 → 降低CLAHE的clipLimit
2. 检测到了更多低质量角点 → 提高min_points阈值
3. 内参不准确 → 考虑重新标定primecolor内参

### Q: 成功帧数没有明显增加？

**A**: 可能原因：
1. 很多帧标定板角度太大或部分遮挡 → 无法通过算法改进
2. 运动模糊严重 → 需要重新拍摄
3. 同步误差 → 检查QR同步offset是否准确

---

## 📞 获取帮助

1. 查看详细技术文档：`PRIMECOLOR_CALIBRATION_FIX.md`
2. 查看深度分析报告：`CALIBRATION_ANALYSIS_SUMMARY.md`
3. 运行测试脚本收集数据，分析具体问题

---

## ✅ 验收标准

优化成功的标志：
- ✅ 成功配对帧数 >150（原100）
- ✅ RMS误差 <1.0像素（原1.402）
- ✅ primecolor平均角点数 >15点/帧（原5.6）
- ✅ 可视化结果中3D轴对齐良好

---

**最后提醒**：优先尝试方案A（仅修改配置），如果改进不足再尝试方案B（添加图像增强）。记录优化前后的具体数值以便评估效果。
