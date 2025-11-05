# PrimeColor Intrinsics Verification Guide

## 目的

验证`.mcal`文件中的内参是否准确，通过对比：
- **A.** `.mcal`中的内参（OptiTrack标定）
- **B.** ChArUco板标定的内参（multical标定）

如果两者差异显著（>5%），说明`.mcal`内参不准确，这是导致marker投影偏差的根本原因。

---

## 快速开始

### 前提条件

1. ✅ 有PrimeColor相机拍摄ChArUco标定板的视频
2. ✅ 视频路径：`/Users/dongkechuan/Downloads/GoPro/gopro_primecolor_extrinsic /Primecolor/Video.avi`
3. ✅ ChArUco板配置文件：`multical/asset/charuco_b3.yaml` (根据你的板子调整)
4. ✅ conda环境已激活：`conda activate multical`

### 一键运行

```bash
cd /Volumes/FastACIS/annotation_pipeline
./verify_primecolor_intrinsics.sh
```

这个脚本会自动完成：
1. ✅ 从视频提取帧（5fps）
2. ✅ 用multical标定内参
3. ✅ 对比新标定的内参和.mcal中的内参
4. ✅ 生成详细对比报告

---

## 输出文件

所有输出在 `/Volumes/FastACIS/annotation_pipeline/primecolor_intrinsic_test/`:

```
primecolor_intrinsic_test/
├── frames/                          # 提取的视频帧
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
├── intrinsic_calibrated.json        # 新标定的内参
├── intrinsics_comparison.txt        # ⭐ 对比报告
├── calibration.log                  # 标定日志
└── vis/                             # 可视化图像（角点检测）
    ├── frame_0001_detected.png
    └── ...
```

---

## 查看结果

### 步骤1：查看对比报告

```bash
cat primecolor_intrinsic_test/intrinsics_comparison.txt
```

### 步骤2：理解报告

报告会告诉你三种可能的结果：

#### ✅ 结果A：内参匹配良好（差异<1%）

```
✅ INTRINSICS MATCH VERY WELL
   • All parameters differ by less than 1%
   • .mcal intrinsics are accurate
   • No need to update calibration
```

**说明**：.mcal的内参是准确的，投影偏差不是内参导致的。

**下一步**：
- 检查外参是否正确
- 检查时间同步
- 检查mocap数据质量

---

#### ⚠️ 结果B：有小差异（1-5%）

```
⚠️  MINOR DIFFERENCES DETECTED
   • Maximum difference: 3.2%
   • Differences detected:
     - fx: +15.234 (+1.22%)
     - cy: -5.678 (-1.05%)

   Recommendation:
   • If RMS is good (<1.0), either intrinsics can be used
   • For critical applications, use the one with lower RMS
```

**说明**：内参有轻微差异，可能是正常的标定变化。

**下一步**：
- 如果新标定的RMS < 1.0，可以尝试使用新内参
- 对比使用两组内参的投影效果
- 选择投影误差更小的一组

---

#### ❌ 结果C：显著差异（>5%）

```
❌ SIGNIFICANT DIFFERENCES DETECTED
   • Maximum difference: 12.5% (>5%)
   • Differences detected:
     - fx: +156.234 (+12.51%)
     - cx: -45.678 (-4.76%)

   THIS IS THE ROOT CAUSE OF YOUR PROJECTION ERRORS!

   RECOMMENDED ACTION:
   ✅ Use the newly calibrated intrinsics!
```

**说明**：🎯 **这就是投影偏差的根本原因！** .mcal的内参不准确。

**下一步**：
1. ✅ 使用新标定的内参
2. ✅ 更新你的投影代码
3. ✅ 重新运行投影测试

---

## 如何使用新标定的内参

### 方法1：修改annotate_mocap_markers_2d3d.py

在 `load_optitrack_calibration()` 函数中，替换内参：

```python
# 旧代码（从.mcal读取）：
fx = float(intrinsic.get('HorizontalFocalLength'))
fy = float(intrinsic.get('VerticalFocalLength'))
cx = float(intrinsic.get('LensCenterX'))
cy = float(intrinsic.get('LensCenterY'))

# 新代码（使用标定值）：
# 从 intrinsic_calibrated.json 复制这些值
fx = 1250.123456  # 替换为实际标定值
fy = 1248.654321
cx = 962.345678
cy = 540.123456
```

### 方法2：创建新的.mcal文件

如果你需要保留原始.mcal，可以创建一个修正版本：

```bash
# 手动编辑.mcal XML文件，更新以下字段：
<IntrinsicStandardCameraModel
    HorizontalFocalLength="新的fx值"
    VerticalFocalLength="新的fy值"
    LensCenterX="新的cx值"
    LensCenterY="新的cy值"
    ...
/>
```

### 方法3：直接使用标定JSON

修改代码，让它可以从multical的JSON直接加载内参，而不是从.mcal：

```python
def load_calibration_from_json(json_path, camera_name):
    """Load calibration from multical JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    cam = data['cameras'][camera_name]
    K = np.array(cam['K'])
    dist = np.array(cam['dist']).flatten()

    # 仍然使用 negative fx
    K[0, 0] = -K[0, 0]

    return K, dist, ...
```

---

## 验证新内参是否有效

使用新内参后，重新运行投影测试：

```bash
# 使用新内参投影
python project_markers_final.py \
  --mcal primecolor_intrinsic_test/intrinsic_calibrated.json \
  --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \
  --output test_with_new_intrinsics.mp4 \
  --start-frame 100 \
  --num-frames 10
```

**期望结果**：
- ✅ marker投影偏差从"几十到上百像素"降低到"<10像素"
- ✅ marker对齐人体部位
- ✅ 投影一致性好

---

## 故障排除

### 问题1：标定失败 "No intrinsic.json created"

**原因**：ChArUco板未被检测到

**解决方案**：
1. 检查ChArUco板配置 `charuco_b3.yaml` 是否匹配实际板子
2. 查看可视化图像 `vis/` 中是否检测到角点
3. 确保视频中板子清晰、无模糊
4. 增加FPS提取更多帧：编辑脚本中的 `FPS_EXTRACT=10`

### 问题2：标定RMS很高 (>2.0)

**原因**：标定质量差

**解决方案**：
1. 重新拍摄标定视频：
   - 保持板子静止
   - 多角度拍摄（正面、侧面、倾斜）
   - 确保对焦清晰
   - 覆盖整个视野
2. 增加标定帧数：编辑脚本中的 `LIMIT_IMAGES=500`

### 问题3：视频路径有空格导致错误

**解决方案**：
脚本已经正确处理空格（使用引号），但如果仍有问题：

```bash
# 方法1：重命名视频去掉空格
mv "/Users/.../gopro_primecolor_extrinsic /Primecolor/Video.avi" \
   "/Users/.../gopro_primecolor_extrinsic/Primecolor/Video.avi"

# 方法2：创建符号链接
ln -s "/Users/.../gopro_primecolor_extrinsic /Primecolor/Video.avi" \
      /tmp/primecolor_calib_video.avi
```

---

## ChArUco板配置

如果你的板子不是默认配置，需要修改 `BOARD_CONFIG`:

### 查看可用配置

```bash
ls -1 multical/asset/charuco*.yaml
```

### 常见配置

| 文件 | 说明 | 尺寸 |
|------|------|------|
| `charuco_b3.yaml` | B3尺寸板 | 5x9 格子 |
| `charuco_b1_2.yaml` | B1尺寸板 | 10x14 格子 |

### 自定义板

如果你的板子是自定义的，需要创建新的YAML配置文件：

```yaml
# multical/asset/my_charuco.yaml
---
type: charuco
aruco_dict: DICT_7X7_250
rows: 9          # ChArUco行数
cols: 5          # ChArUco列数
square_size: 50  # 方格大小(mm)
marker_size: 40  # marker大小(mm)
```

---

## 预期结果

根据我们之前的诊断：

### 如果内参差异显著

- 🎯 **投影偏差会从100像素降低到<10像素**
- ✅ marker会正确对齐人体
- ✅ 这证实了内参不准确是问题根源

### 如果内参差异很小

- ⚠️ 投影偏差可能不会明显改善
- 🔍 需要检查其他原因：
  - 外参问题（相机位置/姿态）
  - 时间同步问题
  - Mocap数据质量

---

## 完整工作流总结

```bash
# 1. 验证内参
./verify_primecolor_intrinsics.sh

# 2. 查看对比报告
cat primecolor_intrinsic_test/intrinsics_comparison.txt

# 3. 如果内参有显著差异，更新代码使用新内参

# 4. 重新测试投影
python visual_alignment_check.py

# 5. 如果投影现在准确了，更新所有工具使用新内参
```

---

## 相关文档

- [MULTI_CAMERA_WORKFLOW.md](MULTI_CAMERA_WORKFLOW.md) - 完整标定流程
- [MARKER_PROJECTION_GUIDE.md](MARKER_PROJECTION_GUIDE.md) - 投影技术细节
- [CLAUDE.md](CLAUDE.md) - Pipeline概述

---

**创建日期**：2025-10-28
**目的**：诊断marker投影偏差的根本原因
**预期结果**：确定内参是否准确，并提供修复方案
