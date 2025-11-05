# 更新后的使用指南

## 📌 **重要更新**

### ✅ **已修复的问题**

1. **外参标注工具保存 bug** - 点击 "Save" 时保存旧值的问题已修复
2. **内参来源支持** - 标注工具现在支持使用 .mcal 内参（`--use-mcal-intrinsics`）
3. **自动内参读取** - 标注查看工具现在能从 extrinsics JSON 自动读取内参

---

## 🎯 **完整工作流程**

### **步骤 1：外参标注**

使用 .mcal 内参标注外参（推荐）：

```bash
python annotate_extrinsics_interactive.py \
  --csv "/Volumes/FastACIS/GoPro/motion/mocap/mocap.csv" \
  --video "/Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi" \
  --mcal "/Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal" \
  --camera_serial "C11764" \
  --use-mcal-intrinsics \
  --start_frame 8702 \
  --port 8050
```

**关键步骤**：
1. 标注 6-8 个高质量对应点
2. ⚠️ **必须点击 "Recompute Extrinsics" 按钮**
3. 确认状态显示成功（绿色）
4. 然后点击 "Save Extrinsics"

**输出文件**：`extrinsics_calibrated.json`
- 包含优化后的 rvec/tvec
- 包含使用的内参（K, dist）
- 记录内参来源（`intrinsics_source: "mcal"`）

---

### **步骤 2：Marker 标注（2D+3D）**

**最简单的方式**（自动使用优化后的内参+外参）：

```bash
python annotate_mocap_markers_2d3d.py \
  --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --extrinsics extrinsics_calibrated.json \
  --start_frame 8702 \
  --num_frames 100 \
  --port 8050
```

**工具会自动**：
- ✅ 从 `extrinsics_calibrated.json` 读取内参
- ✅ 从 `extrinsics_calibrated.json` 读取外参
- ✅ 确保内参和外参来自同一标定

**输出文件**：`marker_labels.csv`

---

### **步骤 3：Marker 投影验证**

使用优化后的外参投影：

```bash
python project_markers_final.py \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \
  --output mocap_with_markers_optimized.mp4 \
  --extrinsics extrinsics_calibrated.json \
  --start-frame 8702 \
  --num-frames 100
```

**注意**：
- 如果 `extrinsics_calibrated.json` 包含内参，会自动使用
- 也可以用 `--intrinsics` 明确指定不同的内参

---

## 🔧 **验证工具**

### **1. 验证外参是否优化**

```bash
python3 diagnose_projection_mismatch.py \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --extrinsics extrinsics_calibrated.json \
  --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --frame 8702
```

**检查**：`Extrinsics changed from initial: True` ✅

### **2. 验证标注质量**

```bash
python3 verify_correspondences.py \
  --extrinsics extrinsics_calibrated.json \
  --intrinsics /Volumes/FastACIS/annotation_pipeline/primecolor_intrinsic_test/frames/intrinsic.json \
  --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv
```

**期望**：Mean error < 10 pixels

### **3. 比较内参**

```bash
python3 compare_intrinsics_mcal_vs_multical.py \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --intrinsics /Volumes/FastACIS/annotation_pipeline/primecolor_intrinsic_test/frames/intrinsic.json
```

---

## 📊 **内参来源选择指南**

### **使用 .mcal 内参（推荐）**

✅ **适用场景**：
- 需要与 OptiTrack Motive 软件对比
- 使用 OptiTrack 系统标定的相机
- 已有可靠的 .mcal 标定文件

```bash
--use-mcal-intrinsics
```

### **使用 multical JSON 内参**

✅ **适用场景**：
- 完全自主标定流程
- multical 标定质量更好（RMS < 0.5）
- 不需要与 Motive 对比

```bash
--intrinsics /path/to/intrinsic.json
```

⚠️ **关键**：标注和投影必须使用**相同的内参**！

---

## 📁 **文件格式说明**

### **extrinsics_calibrated.json**

```json
{
  "camera_serial": "C11764",
  "rvec": [0.506, 0.037, -0.044],        // 优化后的旋转向量
  "tvec": [-0.874, -0.480, -4.607],      // 优化后的平移向量
  "camera_position_world": [0.64, 2.69, 3.81],
  "intrinsics_source": "mcal",            // 内参来源 ✅ 新增
  "intrinsics": {                         // 使用的内参 ✅ 新增
    "K": [[1247.84, 0, 960.60], [0, 1247.75, 538.61], [0, 0, 1]],
    "dist": [0.136, -0.126, 0.0003, -0.0003, 0.00003],
    "fx": 1247.84,
    "fy": 1247.75,
    "cx": 960.60,
    "cy": 538.61
  },
  "correspondences": [...],
  "num_correspondences": 7
}
```

**新增字段说明**：
- `intrinsics_source`: 记录内参来源（`"mcal"` 或 `"multical_json"`）
- `intrinsics`: 保存完整的内参，确保能重现投影结果

---

## 🐛 **常见问题**

### Q1: 保存的外参与计算的不一致？

**原因**：之前的 bug（已修复）
**解决**：更新到最新版本，确保点击 "Recompute" 后再 "Save"

### Q2: 投影结果与标注工具不一致？

**检查清单**：
1. 确认外参是否真的优化过（运行诊断脚本）
2. 确认标注和投影使用相同的内参
3. 检查 `extrinsics_calibrated.json` 中的 `intrinsics_source`

### Q3: 应该使用哪种内参？

**判断标准**：
- 如果需要与 OptiTrack Motive 对比 → 用 .mcal
- 如果完全自主标定和投影 → 用 multical
- **最重要**：标注和投影必须用**相同的内参**

---

## ✅ **最佳实践**

### **1. 标注质量**
- 选择清晰可见的 marker
- 空间分布均匀（不要都在画面一角）
- 至少 6 个点，推荐 8-10 个
- 跨帧标注同一个 marker 可以提高精度

### **2. 验证流程**
1. 标注后立即运行诊断脚本
2. 检查重投影误差
3. 生成短视频验证投影质量
4. 确认后再进行大规模标注

### **3. 文件管理**
- 为每次标定保存不同的 extrinsics 文件
- 命名示例：`extrinsics_calibrated_20251028.json`
- 记录标注参数和质量指标

---

## 📚 **相关文档**

- [EXTRINSICS_CALIBRATION_GUIDE.md](EXTRINSICS_CALIBRATION_GUIDE.md) - 详细标定指南
- [CLAUDE.md](CLAUDE.md) - 完整 pipeline 说明
- 诊断工具脚本：
  - `diagnose_projection_mismatch.py`
  - `verify_correspondences.py`
  - `compare_intrinsics_mcal_vs_multical.py`

---

## 🎉 **更新历史**

**2025-10-28**:
- ✅ 修复外参保存 bug
- ✅ 添加 `--use-mcal-intrinsics` 选项
- ✅ 保存内参到 extrinsics JSON
- ✅ 自动从 extrinsics JSON 读取内参
- ✅ 创建诊断和验证工具
