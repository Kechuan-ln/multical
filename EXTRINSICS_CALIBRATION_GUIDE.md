# Extrinsics Calibration Guide

## 问题诊断总结

### 发现的问题

1. **外参没有真正优化** - 诊断脚本显示保存的 `rvec` 和 `tvec` 与 .mcal 初始值完全相同
2. **内参不匹配的风险** - 标注工具使用 multical JSON 内参，但实际投影可能需要 .mcal 内参

### 解决方案

已更新 `annotate_extrinsics_interactive.py` 支持使用 .mcal 内参。

---

## 使用方法

### 选项 1：使用 .mcal 内参（推荐用于 OptiTrack 系统）

```bash
python annotate_extrinsics_interactive.py \
  --csv "/Volumes/FastACIS/GoPro/motion/mocap/mocap.csv" \
  --video "/Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi" \
  --mcal "/Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal" \
  --camera_serial "C11764" \
  --use-mcal-intrinsics \
  --start_frame 8465 \
  --port 8050
```

**优点**：
- 与 OptiTrack 系统的内参完全一致
- 适合需要与 Motive 软件对比的场景
- 保证投影一致性

### 选项 2：使用 multical JSON 内参

```bash
python annotate_extrinsics_interactive.py \
  --csv "/Volumes/FastACIS/GoPro/motion/mocap/mocap.csv" \
  --video "/Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi" \
  --mcal "/Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal" \
  --intrinsics "/Volumes/FastACIS/annotation_pipeline/primecolor_intrinsic_test/frames/intrinsic.json" \
  --camera_serial "C11764" \
  --start_frame 8465 \
  --port 8050
```

**优点**：
- 使用自己标定的高精度内参
- 可能比 .mcal 内参更准确（如果标定质量好）
- 适合完全自主的标定流程

---

## 标注流程（重要！）

### 1. 选择好的标注点

- ✅ 选择**清晰可见**的 marker（避免模糊或遮挡）
- ✅ 选择**空间分布均匀**的点（不要都在画面一角）
- ✅ 建议在**同一帧**先标注 6-8 个点
- ✅ 精确点击 2D 位置（尽量点在 marker 中心）

### 2. 标注步骤

1. 在左侧 3D 视图选择一个 marker（点击或输入编号）
   - marker 会变成**红色**
2. 在右侧 2D 视图点击对应的位置
   - 添加成功后会显示**红色 X**
3. 重复步骤 1-2，至少标注 **6 个点**
4. **点击绿色的 "Recompute Extrinsics" 按钮** ← 这是关键！
5. 检查状态消息：应显示 `"✓ Extrinsics updated! X/X points used"`
6. 如果投影看起来好了，点击 "Save Extrinsics"

### 3. 常见错误

❌ **标注后直接点 "Save"，没有点 "Recompute"**
   - 结果：保存的是初始外参，没有优化
   - 解决：必须先点 "Recompute"，看到成功消息后再 "Save"

❌ **标注点太少（<6 个）**
   - 结果：solvePnP 失败
   - 解决：至少标注 6 个高质量的点

❌ **标注点位置不准确**
   - 结果：重投影误差大
   - 解决：放大视频，精确点击 marker 中心

---

## 验证标注质量

### 方法 1：诊断脚本

```bash
python3 diagnose_projection_mismatch.py \
  --intrinsics /Volumes/FastACIS/annotation_pipeline/primecolor_intrinsic_test/frames/intrinsic.json \
  --extrinsics extrinsics_calibrated.json \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --frame 8465
```

**检查点**：
- `Extrinsics changed from initial: True` ✅（应该是 True）
- `Projections are IDENTICAL` ✅

### 方法 2：验证重投影误差

```bash
python3 verify_correspondences.py \
  --extrinsics extrinsics_calibrated.json \
  --intrinsics /Volumes/FastACIS/annotation_pipeline/primecolor_intrinsic_test/frames/intrinsic.json \
  --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv
```

**期望结果**：
- Mean error < 10 pixels ✅
- 大部分点 error < 20 pixels ✅

---

## 投影使用

### 使用优化后的外参投影

```bash
python project_markers_final.py \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \
  --output mocap_with_markers_calibrated.mp4 \
  --extrinsics extrinsics_calibrated.json
```

**注意**：
- 如果标注时用了 `--use-mcal-intrinsics`，投影会自动使用 .mcal 内参
- 如果标注时用了 multical JSON，投影需要指定 `--intrinsics`：
  ```bash
  --intrinsics /Volumes/FastACIS/annotation_pipeline/primecolor_intrinsic_test/frames/intrinsic.json
  ```

---

## 输出文件格式

### extrinsics_calibrated.json

```json
{
  "camera_serial": "C11764",
  "rvec": [r1, r2, r3],                    // 优化后的旋转向量
  "tvec": [t1, t2, t3],                    // 优化后的平移向量
  "rotation_matrix": [[...], ...],         // 3x3 旋转矩阵
  "camera_position_world": [x, y, z],      // 相机在世界坐标系的位置
  "intrinsics_source": "mcal",             // 使用的内参来源（新增）
  "intrinsics": {                          // 使用的内参（新增）
    "K": [[...], ...],
    "dist": [...],
    "fx": ..., "fy": ..., "cx": ..., "cy": ...
  },
  "correspondences": [                     // 标注的对应点
    {"marker_idx": 170, "frame_idx": 8465, "point_2d": [660, 268]},
    ...
  ],
  "num_correspondences": 13
}
```

**新增字段说明**：
- `intrinsics_source`: 记录使用的内参来源（`"mcal"` 或 `"multical_json"`）
- `intrinsics`: 保存使用的内参，确保以后能重现投影结果

---

## 常见问题

### Q: 投影结果与标注系统不一致？

**可能原因**：
1. 没有点击 "Recompute Extrinsics"
2. 标注时用的内参 ≠ 投影时用的内参
3. mocap 数据帧号不匹配

**排查方法**：
1. 运行 `diagnose_projection_mismatch.py` 检查外参是否改变
2. 检查 `extrinsics_calibrated.json` 中的 `intrinsics_source`
3. 确保投影时使用相同的内参

### Q: 应该用 .mcal 内参还是 multical 内参？

**判断标准**：
1. 如果需要与 OptiTrack Motive 软件对比 → 用 .mcal
2. 如果完全自主标定和投影 → 用 multical
3. **最重要**：标注和投影必须用**相同的内参**

**比较两种内参差异**：
```bash
python3 compare_intrinsics_mcal_vs_multical.py \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --intrinsics /Volumes/FastACIS/annotation_pipeline/primecolor_intrinsic_test/frames/intrinsic.json
```

### Q: 浏览器控制台显示什么？

打开浏览器开发者工具 (F12)，点击 "Recompute Extrinsics" 后应该看到：

```
=== DEBUG: Recomputing extrinsics ===
Total correspondences: 6
...
solvePnPRansac result: success=True
  rvec=[...]
  tvec=[...]
  inliers=[0 1 2 3 4 5]
✓ Extrinsics updated! 6/6 points used
```

如果看到 `success=False`，说明优化失败，需要：
- 检查标注点是否正确
- 增加标注点数量
- 重新选择更清晰的 marker

---

## 更新历史

- 2025-10-28: 添加 `--use-mcal-intrinsics` 选项
- 2025-10-28: 在保存的 JSON 中记录内参来源和值
- 2025-10-28: 添加诊断和验证工具
