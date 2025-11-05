# Marker投影测试工具

## 概述

`project_markers_to_gopro.py` 是一个简化的测试工具，用于验证投影算法的正确性。与完整的skeleton投影不同，这个脚本只投影mocap系统捕获的原始marker点（3D→2D），不涉及skeleton建模和mesh渲染。

## 为什么先测试Marker？

1. **简单直接**: Marker是mocap系统直接捕获的原始3D点，没有中间处理
2. **容易验证**: 可以直接看marker点是否投影到视频中人体marker的物理位置
3. **快速调试**: 如果marker投影不准，说明投影管线有问题；如果准确，skeleton问题在其他地方
4. **独立测试**: 分离投影算法测试和skeleton建模测试

## 使用方法

### 快速测试

```bash
# 给脚本执行权限
chmod +x run_marker_projection_test.sh

# 运行测试（生成300帧测试视频）
bash run_marker_projection_test.sh
```

这会生成两个视频：
- `marker_projection_mcal.mp4`: 使用OptiTrack标定
- `marker_projection_custom.mp4`: 使用自定义标定

### 手动运行

```bash
python project_markers_to_gopro.py \
  --calibration /path/to/calibration.json \
  --mcal /path/to/Primecolor.mcal \
  --mocap-csv /path/to/mocap.csv \
  --marker-labels marker_labels.csv \
  --gopro-video /path/to/Video.MP4 \
  --output output.mp4 \
  --mocap-fps 120.0 \
  --sync-offset 10.5799 \
  --start-frame 0 \
  --num-frames 300
```

### 使用自定义外参

添加 `--extrinsics-json` 参数：

```bash
python project_markers_to_gopro.py \
  --calibration /path/to/calibration.json \
  --mcal /path/to/Primecolor.mcal \
  --extrinsics-json extrinsics_calibrated.json \
  --mocap-csv /path/to/mocap.csv \
  ...
```

## 参数说明

### 必需参数

- `--calibration`: calibration.json路径（包含PrimeColor→GoPro外参和GoPro内参）
- `--mcal`: .mcal文件路径（包含Mocap→PrimeColor外参，如果没有--extrinsics-json）
- `--mocap-csv`: mocap.csv文件路径（包含每一帧的marker 3D坐标）
- `--gopro-video`: GoPro视频文件路径
- `--output`: 输出视频路径

### 可选参数

- `--extrinsics-json`: 自定义外参JSON路径（替代.mcal）
- `--marker-labels`: marker标签CSV路径（用于在视频上显示marker名称）
- `--mocap-fps`: Mocap帧率（默认120）
- `--sync-offset`: 同步偏移秒数（默认0.0）
- `--start-frame`: GoPro起始帧号（默认0）
- `--num-frames`: 处理帧数（默认处理全部）

## 输入文件格式

### mocap.csv

OptiTrack导出的标准CSV格式：

```
Format Version,1.25,Take Name,...
,Type,Marker,Marker,Marker,...
,Name,Unlabeled 1000,Unlabeled 1001,...
,ID,9F:...,B7:...,...
,Parent,,,,...
,,Position,Position,Position,...
Frame,Time (Seconds),X,Y,Z,X,Y,Z,...
0,0.000000,496.063,516.091,2854.694,...
1,0.008333,496.231,516.068,2854.556,...
```

- 单位：毫米
- 坐标系：OptiTrack世界坐标系（-Z前向，Y向上）
- 帧率：通常120fps

### marker_labels.csv

Marker ID到人体部位的映射（可选，用于显示标签）：

```csv
original_name,marker_id,label
Unlabeled 1218,788:73CB7475AD7A11F0,Lanklel1
Unlabeled 1231,8BA:73CB7475AD7A11F0,Lanklel2
...
```

### extrinsics_calibrated.json

自定义外参格式（可选，替代.mcal）：

```json
{
  "rotation_matrix": [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
  "tvec": [tx, ty, tz],
  "camera_position_world": [x, y, z],
  "intrinsics": {...}
}
```

## 投影管线

```
Mocap世界坐标 (OptiTrack, -Z前向, 单位mm)
    ↓ 1. 转换为米
Mocap世界坐标 (OptiTrack, -Z前向, 单位m)
    ↓ 2. 使用R_w2p, T_w2p (.mcal或custom JSON)
PrimeColor相机坐标 (OptiTrack约定, -Z前向)
    ↓ 3. YZ翻转 [1,-1,-1]
PrimeColor相机坐标 (标准约定, +Z前向)
    ↓ 4. 使用R_p2g, T_p2g的逆变换 (calibration.json)
GoPro相机坐标 (标准约定, +Z前向)
    ↓ 5. cv2.projectPoints (K_gopro, dist_gopro)
GoPro图像2D坐标 (像素)
```

## 如何验证结果

### 1. 打开输出视频

用视频播放器打开 `marker_projection_mcal.mp4` 或 `marker_projection_custom.mp4`

### 2. 检查marker位置

观察绿色圆点是否准确投影到：
- ✓ 人体上的marker物理位置（如果能看到真实marker）
- ✓ 正确的身体部位（根据marker标签）
- ✓ 合理的深度顺序（前面的marker遮挡后面的）

### 3. 常见问题

| 现象 | 可能原因 |
|------|---------|
| Marker完全不在人体上 | 外参错误、同步偏移错误 |
| Marker在人体附近但偏移 | 外参精度问题、坐标系转换错误 |
| Marker左右颠倒 | 坐标系翻转错误（应该用YZ翻转） |
| Marker上下颠倒 | 坐标系翻转错误 |
| Marker深度错误（前后关系） | Z坐标符号错误 |
| 部分marker准确，部分不准 | Marker遮挡、mocap捕获丢失 |

### 4. 对比两种外参

如果同时生成了两个视频：
- `marker_projection_mcal.mp4`: OptiTrack标定
- `marker_projection_custom.mp4`: 自定义标定

对比哪个更准确，可以判断使用哪种外参更好。

## 示例：理想的投影结果

```
✓ 头部markers → 投影到头部区域
✓ 肩部markers → 投影到肩膀
✓ 手腕markers → 跟随手部运动
✓ 膝盖markers → 投影到膝盖位置
✓ 脚踝markers → 投影到脚踝
```

## 故障排除

### 问题1: 没有marker投影

**检查**:
- mocap.csv文件路径是否正确
- 同步偏移是否正确（`--sync-offset`）
- mocap帧号范围是否覆盖GoPro视频范围

**调试**:
```bash
# 打印第一帧的mocap数据
head -10 /path/to/mocap.csv
```

### 问题2: Marker标签不显示

**原因**: 没有提供 `--marker-labels` 参数

**解决**: 添加参数或使用marker ID作为标签（默认行为）

### 问题3: 投影位置完全错误

**检查**:
1. 外参文件是否正确（.mcal或custom JSON）
2. calibration.json是否对应Cam4
3. 同步偏移方向是否正确

**调试**:
```bash
# 测试不同的同步偏移
python project_markers_to_gopro.py ... --sync-offset 10.5799
python project_markers_to_gopro.py ... --sync-offset 11.0
python project_markers_to_gopro.py ... --sync-offset 10.0
```

### 问题4: 解析mocap.csv失败

**检查**: CSV格式是否是OptiTrack标准格式

**调试**: 手动检查CSV前10行
```bash
head -10 /path/to/mocap.csv
```

## 性能优化

### 只处理部分帧

```bash
# 只处理前300帧（120fps下约2.5秒）
python project_markers_to_gopro.py ... --start-frame 0 --num-frames 300
```

### 跳到特定时间

```bash
# 从第1000帧开始（120fps下约8.3秒）
python project_markers_to_gopro.py ... --start-frame 1000 --num-frames 300
```

## 下一步

如果marker投影准确：
1. ✓ 投影算法正确
2. ✓ 外参和内参正确
3. ✓ 同步偏移正确
4. → 可以继续测试skeleton投影

如果marker投影不准确：
1. ✗ 需要先修正外参或内参
2. ✗ 或者调整同步偏移
3. → 不要继续skeleton投影，先解决marker问题

## 相关文件

- `project_markers_to_gopro.py`: 主脚本
- `run_marker_projection_test.sh`: 快速测试脚本
- `project_skeleton_to_gopro_FINAL_FIXED.py`: 完整的skeleton投影脚本
- `test_custom_vs_mcal_extrinsics.py`: 外参对比工具
- `EXTRINSICS_COMPARISON_GUIDE.md`: 外参对比指南

---

**更新日期**: 2025年
**作者**: Claude Code
**版本**: 1.0
