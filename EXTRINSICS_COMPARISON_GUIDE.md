# Mocap到PrimeColor外参对比指南

## 概述

`project_skeleton_to_gopro_FINAL_FIXED.py` 现在支持两种方式加载 Mocap→PrimeColor 外参:

1. **OptiTrack .mcal 文件** (默认): OptiTrack厂商标定
2. **自定义 JSON 文件** (可选): 用户用ChArUco板自行标定

## 两种外参的差异

### 测试结果

使用 `test_custom_vs_mcal_extrinsics.py` 进行对比:

```
R矩阵差异:
  最大值: 0.0645 (约3.7°旋转误差)
  平均值: 0.0244

T向量差异:
  最大值: 0.371 meters = 371 mm
  平均值: 0.138 meters = 138 mm

投影差异 (Pelvis关节):
  位置差异: 281 mm
```

### 差异原因

这两个标定来自**不同的标定过程**:

| 特征 | .mcal (OptiTrack) | extrinsics_calibrated.json (自定义) |
|------|-------------------|-------------------------------------|
| **标定方法** | OptiTrack系统内部标定 | ChArUco棋盘格板 + multical |
| **标定对象** | OptiTrack相机与PrimeColor | PrimeColor内参+外参联合优化 |
| **坐标系** | OptiTrack约定 (-Z前向) | 标准OpenCV约定 (+Z前向) |
| **优点** | 与mocap数据天然对齐 | 可以独立验证和调整 |
| **缺点** | 依赖OptiTrack黑盒标定 | 需要额外标定工作 |

### 哪个更准确？

**取决于你的用例**:

1. **如果你信任OptiTrack标定**: 使用 `.mcal`
   - OptiTrack是专业动捕系统
   - 标定过程经过厂商验证
   - 与mocap数据直接对齐

2. **如果你想独立验证**: 使用 `extrinsics_calibrated.json`
   - 可以重新标定和调整
   - 基于可见的ChArUco标定板
   - 可以检查重投影误差

3. **建议**:
   - 先用 `.mcal` 投影，检查结果
   - 如果投影位置明显偏移，尝试自定义外参
   - 对比两种方法的视觉效果，选择更准确的

## 使用方法

### 方法1: 使用 .mcal (默认)

```bash
python project_skeleton_to_gopro_FINAL_FIXED.py \
  --calibration /path/to/calibration.json \
  --mcal /path/to/Primecolor.mcal \
  --skeleton skeleton.json \
  --gopro-video video.MP4 \
  --output output.mp4 \
  --sync-offset 10.5799
```

输出日志:
```
Loading calibration...
  Using camera: cam1
  Using transform: primecolor_to_cam1
  Loading Mocap→PrimeColor from .mcal: /path/to/Primecolor.mcal
  ✓ Loaded OptiTrack extrinsics
```

### 方法2: 使用自定义 JSON

```bash
python project_skeleton_to_gopro_FINAL_FIXED.py \
  --calibration /path/to/calibration.json \
  --mcal /path/to/Primecolor.mcal \
  --extrinsics-json /path/to/extrinsics_calibrated.json \
  --skeleton skeleton.json \
  --gopro-video video.MP4 \
  --output output.mp4 \
  --sync-offset 10.5799
```

输出日志:
```
Loading calibration...
  Using camera: cam1
  Using transform: primecolor_to_cam1
  Loading Mocap→PrimeColor from custom JSON: /path/to/extrinsics_calibrated.json
  ✓ Loaded custom extrinsics
```

**注意**: `--mcal` 参数仍然是必需的（历史原因），但如果提供了 `--extrinsics-json`，将使用JSON文件而忽略.mcal。

## 自定义外参JSON格式

`extrinsics_calibrated.json` 格式:

```json
{
  "rotation_matrix": [
    [r00, r01, r02],
    [r10, r11, r12],
    [r20, r21, r22]
  ],
  "tvec": [tx, ty, tz],
  "camera_position_world": [x, y, z],
  "intrinsics": {
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist": [k1, k2, p1, p2, k3]
  }
}
```

**关键点**:
- `rotation_matrix`: 世界坐标→相机坐标的旋转矩阵 (R_w2c)
- `tvec`: 平移向量 (世界→相机)
- 坐标系: 标准OpenCV约定 (+Z前向, Y向下)

## 验证和调试

### 1. 对比两种外参

```bash
python test_custom_vs_mcal_extrinsics.py
```

输出:
- R矩阵和T向量的数值对比
- 差异统计
- 测试点投影差异

### 2. 生成对比视频

```bash
bash run_projection_with_custom_extrinsics.sh
```

生成两个视频:
- `output_using_mcal.mp4`: 使用OptiTrack标定
- `output_using_custom.mp4`: 使用自定义标定

### 3. 检查投影准确性

观察视频中骨架投影:
- ✓ 骨架是否在人体上
- ✓ 关节位置是否准确
- ✓ 左右关系是否正确
- ✓ 深度（大小）是否合理

## 坐标系转换

脚本内部的转换流程:

```
Mocap世界坐标 (OptiTrack约定, -Z前向)
    ↓ (使用.mcal或custom JSON的R_w2p, T_w2p)
PrimeColor相机坐标 (OptiTrack约定, -Z前向)
    ↓ (R_OPTI_TO_STD = diag[1,-1,-1], YZ翻转)
PrimeColor相机坐标 (标准约定, +Z前向)
    ↓ (使用calibration.json的R_p2g, T_p2g的逆变换)
GoPro相机坐标 (标准约定, +Z前向)
    ↓ (cv2.projectPoints, K_gopro, dist_gopro)
GoPro图像2D坐标
```

**重要**:
- .mcal的R和T已经是OptiTrack约定
- custom JSON的R和T应该也是OptiTrack约定（如果是从.mcal派生的）
- 如果custom JSON是标准约定，需要额外转换！

## 常见问题

### Q1: 为什么两个标定结果差这么多？

**A**: 可能原因:
1. 标定时间不同（相机位置可能移动了）
2. 标定方法不同（OptiTrack vs ChArUco板）
3. 优化目标不同（不同的误差函数）
4. 坐标系定义不同（需要确认custom JSON的坐标系约定）

### Q2: 如何判断哪个标定更准？

**A**:
1. 生成两个投影视频，人工对比
2. 检查骨架是否对齐人体
3. 测量几个关键帧的重投影误差
4. 如果都不准，考虑重新标定

### Q3: 我的custom JSON是用multical生成的，但投影位置不对？

**A**: 检查:
1. multical输出的R是世界→相机还是相机→世界？
2. 坐标系约定是+Z前向还是-Z前向？
3. 是否需要在JSON中添加坐标系翻转？

### Q4: 能否同时混合使用两种标定？

**A**: 不建议。整个变换链必须一致:
- 要么全用OptiTrack标定
- 要么全用自定义标定
- 混合使用会导致坐标系不匹配

## 相关文件

- `project_skeleton_to_gopro_FINAL_FIXED.py`: 主投影脚本
- `test_custom_vs_mcal_extrinsics.py`: 外参对比测试
- `run_projection_with_custom_extrinsics.sh`: 批量对比脚本
- `extrinsics_calibrated.json`: 自定义外参文件
- `/Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal`: OptiTrack标定

## 总结

- ✓ 脚本现在支持两种外参源
- ✓ 可以灵活选择使用OptiTrack标定或自定义标定
- ✓ 两种标定有明显差异（~280mm投影误差）
- ✓ 建议先用.mcal测试，如不准再尝试自定义外参
- ✓ 通过视觉对比选择更准确的标定

---

**更新日期**: 2025年
**作者**: Claude Code
**版本**: 1.0
