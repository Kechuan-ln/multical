# Interactive Extrinsics Calibration Tool

交互式外参标定工具，通过手动标注2D-3D对应点来优化相机外参（位置和姿态）。

## 快速开始

```bash
chmod +x run_extrinsic_annotation.sh
./run_extrinsic_annotation.sh
```

浏览器打开：`http://localhost:8050`

## 使用流程

### 1. 选择3D标记点
- 在**左侧3D视图**中点击一个marker（会变成**红色**）
- 此时右侧2D视图中对应marker的投影会高亮

### 2. 标注2D对应点
- 在**右侧2D视图**中点击marker在视频中的**实际位置**
- 这会创建一个2D-3D对应关系
- 标注的点会显示为**红色X标记**

### 3. 重复标注
- 重复步骤1-2，至少标注**6个对应点**
- 建议标注**10-15个点**以获得更好的结果
- 可以使用**Frame滑块**切换到不同帧来增加标注点的空间分布

### 4. 重新计算外参
- 点击**"Recompute Extrinsics"**按钮
- 使用cv2.solvePnP + RANSAC计算新的外参
- 所有marker的投影会立即更新

### 5. 迭代优化
- 查看更新后的投影是否准确
- 如果还有偏差，继续添加更多对应点
- 再次点击"Recompute Extrinsics"

### 6. 保存结果
- 满意后点击**"Save Extrinsics"**按钮
- 外参保存到 `extrinsics_calibrated.json`

## 界面说明

### 左侧：3D Markers视图
- **蓝色点**：未标注的markers
- **绿色点**：已标注的markers
- **红色点**：当前选中的marker
- 每个点旁边显示编号

### 右侧：2D Video Frame视图
- 显示当前帧的视频图像
- **黄色圆点**：未标注marker的投影
- **绿色圆点**：已标注marker的投影
- **红色X**：手动标注的对应点位置

### 控制按钮
- **Recompute Extrinsics**：用当前对应点重新计算外参（需要6+点）
- **Undo Last**：撤销最后一个标注
- **Clear All**：清除所有标注
- **Save Extrinsics**：保存当前外参到JSON文件

### 状态显示
- **Status**：显示当前操作状态和提示
- **Correspondences**：显示已标注的对应点数量

## 标注技巧

### 1. 选择好的标注点
- ✅ 选择清晰可见、位置明确的markers
- ✅ 选择分布在不同深度和位置的markers
- ✅ 优先标注图像边缘和角落的markers
- ❌ 避免标注模糊或部分遮挡的markers

### 2. 多帧标注
- 在**不同帧**标注可以提高鲁棒性
- 特别是当markers在运动时
- 使用Frame滑块跳转到markers分布较好的帧

### 3. 验证结果
- 重新计算外参后，检查**所有markers**的投影
- 如果大部分markers对齐良好，外参可能已经准确
- 如果仍有系统性偏移，继续添加更多对应点

### 4. RANSAC outliers
- solvePnP使用RANSAC，会自动剔除outliers
- 如果某些标注点是outliers，不会影响最终结果
- 状态栏会显示 "X/Y inliers"（X个inliers，Y个总点）

## 输出文件

### extrinsics_calibrated.json
```json
{
  "camera_serial": "C11764",
  "rvec": [r1, r2, r3],          // Rodrigues旋转向量
  "tvec": [t1, t2, t3],          // 平移向量（meters）
  "rotation_matrix": [[...], ...], // 3x3旋转矩阵
  "camera_position_world": [x, y, z], // 相机在世界坐标系的位置
  "correspondences": [
    {"marker_idx": 5, "point_2d": [x, y]},
    ...
  ],
  "num_correspondences": 12
}
```

## 技术细节

### 坐标系
- **Mocap坐标系**：OptiTrack世界坐标（Y轴向上）
- **相机坐标系**：OpenCV标准（Z轴向前，需要使用negative fx）

### 投影公式
```
点3D（mm）→ 转为米 → cv2.projectPoints（K_neg, dist, rvec, tvec）→ 点2D（pixels）
```

### solvePnP参数
- 方法：`SOLVEPNP_ITERATIVE`
- RANSAC重投影误差阈值：10 pixels
- 置信度：0.99
- 最少需要：6个对应点

## 故障排除

### "Need at least 6 correspondences"
- 至少需要标注6个2D-3D对应点才能计算外参
- 继续标注更多点

### "solvePnP failed"
- 可能标注的对应点不正确
- 检查是否点击了错误的位置
- 使用"Clear All"重新开始

### 投影仍然不准确
- 增加更多对应点（建议10-15个）
- 检查内参是否准确（cy偏移143像素的问题）
- 确保标注点分布均匀（不要只标注图像一个区域）

### 所有markers投影都偏移
- 可能是**内参**不准确，而非外参
- 建议先用正确的内参，再标定外参
- 参考：`compare_intrinsics_detailed.py` 检查内参

## 命令行选项

```bash
python annotate_extrinsics_interactive.py \
  --csv mocap.csv \
  --video primecolor.avi \
  --mcal Primecolor.mcal \
  --intrinsics intrinsic.json \
  --camera_serial C11764 \
  --start_frame 0 \
  --port 8050
```

## 相关工具

- `run_marker_annotation_2d3d.sh`：Marker到skeleton映射标注工具
- `compare_intrinsics_detailed.py`：对比内参差异
- `compare_projection_old_vs_new.py`：对比新旧内参投影效果

## 工作流程建议

1. **先检查内参**：运行 `compare_intrinsics_detailed.py` 确认内参准确
2. **标定外参**：使用本工具手动标注6+对应点
3. **验证效果**：在不同帧检查投影精度
4. **保存结果**：保存优化后的外参JSON
5. **后续使用**：在其他脚本中加载 `extrinsics_calibrated.json`

## 注意事项

⚠️ **内参优先**：如果内参不准（如cy偏移143像素），先修正内参再标定外参

⚠️ **分辨率匹配**：标定视频和mocap视频必须是同一分辨率

⚠️ **时间同步**：确保视频帧和mocap数据在时间上对齐

⚠️ **标注质量**：标注质量直接影响外参精度，请仔细标注
