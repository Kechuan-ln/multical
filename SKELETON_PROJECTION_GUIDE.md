# Skeleton Projection to Video Guide

## 概述

将3D skeleton（17个关节）投影到PrimeColor相机视频上，使用OptiTrack标定参数。

**关键技术**：
- 使用**negative focal length (fx)**处理OptiTrack的-Z前向坐标系
- 不检查Z>0（因为negative fx会导致Z<0）
- 只检查2D坐标是否在图像边界内

## 快速开始

### 基本用法

```bash
python project_skeleton_to_video.py \
  --mcal optitrack.mcal \
  --skeleton skeleton_joints.json \
  --video video.avi \
  --output skeleton_video.mp4
```

### 处理指定帧范围

```bash
python project_skeleton_to_video.py \
  --skeleton skeleton_joints.json \
  --video video.avi \
  --start-frame 1000 \
  --num-frames 500 \
  --output skeleton_1000-1500.mp4
```

### 自定义样式

```bash
python project_skeleton_to_video.py \
  --line-thickness 3 \
  --point-radius 5 \
  --output thick_skeleton.mp4
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mcal` | `optitrack.mcal` | OptiTrack标定文件 |
| `--skeleton` | `skeleton_joints.json` | Skeleton JSON文件 |
| `--video` | `video.avi` | 输入视频 |
| `--output` | `skeleton_video.mp4` | 输出视频 |
| `--camera-serial` | `C11764` | 相机序列号 |
| `--start-frame` | `0` | 起始帧 |
| `--num-frames` | `-1` | 处理帧数（-1=全部） |
| `--line-thickness` | `2` | 骨架线条粗细（像素） |
| `--point-radius` | `4` | 关节点半径（像素） |
| `--no-frame-info` | `False` | 不显示帧信息叠加 |

## 输出效果

### 骨架颜色编码

- 🔵 **蓝色** (255,0,0 BGR): 脊柱/躯干 (Spine, Neck)
- 🟣 **洋红** (255,0,255 BGR): 头部/下颌 (Head, Jaw)
- 🟢 **绿色** (0,255,0 BGR): 左臂 (LShoulder, LElbow, LWrist)
- 🔴 **红色** (0,0,255 BGR): 右臂 (RShoulder, RElbow, RWrist)
- 🔵 **青色** (255,255,0 BGR): 左腿 (LHip, LKnee, LAnkle)
- 🟠 **橙色** (0,165,255 BGR): 右腿 (RHip, RKnee, RAnkle)

### 关节点样式

- **填充**: 白色
- **轮廓**: 黑色
- **半径**: 默认4像素（可调整）

### 帧信息叠加

左上角显示：
```
Frame 1234 | Joints: 15/17
```

- `Frame 1234`: 当前帧号
- `Joints: 15/17`: 成功投影的关节数 / 总关节数

## 完整工作流程

### 1. 准备数据

```bash
# 1a. 标注markers（如果还没有）
python annotate_mocap_markers.py --start-frame 2 --num_frames 200

# 1b. 转换为skeleton
python markers_to_skeleton.py \
  --mocap_csv /path/to/mocap.csv \
  --labels_csv marker_labels.csv \
  --start_frame 2 \
  --end_frame 10000
```

输出：`skeleton_joints.json`

### 2. 投影到视频

```bash
python project_skeleton_to_video.py \
  --mcal /path/to/optitrack.mcal \
  --skeleton skeleton_joints.json \
  --video /path/to/video.avi \
  --output skeleton_video.mp4
```

### 3. 验证结果

```bash
# 用视频播放器查看
open skeleton_video.mp4

# 或用ffplay
ffplay skeleton_video.mp4
```

## 技术细节

### 坐标系转换

**OptiTrack → OpenCV 投影**：

1. **OptiTrack提供** (来自.mcal):
   - `R_c2w`: Camera-to-World 旋转矩阵
   - `T_world`: 相机在世界坐标系的位置 (m)

2. **转换为OpenCV格式**:
   ```python
   R_w2c = R_c2w.T  # World-to-Camera
   tvec = -R_w2c @ T_world
   rvec = cv2.Rodrigues(R_w2c)[0]
   ```

3. **内参矩阵**（关键！）:
   ```python
   K = [[-fx,  0,  cx],  # 注意：negative fx!
        [ 0,  fy,  cy],
        [ 0,   0,   1]]
   ```

4. **投影**:
   ```python
   points_2d, _ = cv2.projectPoints(
       joints_3d_m,  # 3D joints in meters
       rvec, tvec, K, dist
   )
   ```

### 为什么使用Negative fx？

OptiTrack和OpenCV的Z轴方向相反：

```
OptiTrack:           OpenCV:
    Y                   Y
    |                   |
    |                   |
    +---X               +---X
   /                   /
  Z (backward)       Z (forward)
```

**Negative fx** 补偿这个差异，使投影结果正确。

### 为什么不检查Z>0？

使用negative fx后，相机坐标中的Z值会是负数（几何上"在相机后方"），但投影仍然正确。因此：

✅ **只检查2D边界**:
```python
in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
```

❌ **不检查Z**:
```python
# DO NOT DO THIS:
# in_front = Z > 0  # 会过滤掉所有点！
```

## 示例输出

```
======================================================================
Skeleton Projection to Video
======================================================================

Loading calibration...
  Camera intrinsics: fx=-1247.84 (negative for coord conversion), fy=1247.75
  Image size: 1920x1080
  Camera position (world): [-0.26864  2.655145 -3.509723]

Loading skeleton data...
  Skeleton: 17 joints
  Frames: 998
  FPS: 120.0

Opening video...
  Video frames: 23375
  FPS: 120.0
  Resolution: 1920x1080

Processing frames 0 to 998
Projecting skeleton: 100%|████████████████| 998/998 [01:23<00:00, 11.95it/s]

✓ Done!
  Frames with skeleton: 956/998
  Output saved to: skeleton_video.mp4
```

## 故障排除

### Q: 投影结果看不到skeleton

**A**: 检查：
1. `skeleton_joints.json` 是否包含有效的关节数据
2. `.mcal` 和 skeleton 数据来自同一次session
3. 视频分辨率是否与 `.mcal` 一致（1920x1080）
4. 相机Serial是否正确（默认C11764）

### Q: Frames with skeleton: 0/N

**A**: 可能原因：
1. Skeleton数据全是NaN/None（检查`skeleton_joints.json`）
2. 帧索引不匹配（skeleton从frame 2开始，视频从frame 0）
3. 视频无法读取（检查路径）

### Q: 只有部分关节显示

**A**: 这是正常的！原因：
1. 某些关节在标注时markers缺失（如Head, Neck等）
2. 某些关节投影在图像外（相机视角限制）
3. 显示 `Joints: X/17` 表示X个关节在视野内

### Q: 骨架位置偏移

**A**: 检查：
1. 是否使用了 **negative fx** (脚本已内置)
2. `.mcal` 文件是否正确
3. Skeleton数据单位是否为毫米（应该是）

### Q: 视频分辨率不匹配

**A**:
```
WARNING: Video resolution doesn't match calibration!
  Video: 1280x720
  Calibration: 1920x1080
```

解决方案：
- 使用正确分辨率的视频
- 或重新标定相机

## 性能优化

### 处理大视频

**选项1**: 分段处理
```bash
# 前5000帧
python project_skeleton_to_video.py \
  --start-frame 0 --num-frames 5000 \
  --output part1.mp4

# 后5000帧
python project_skeleton_to_video.py \
  --start-frame 5000 --num-frames 5000 \
  --output part2.mp4

# 合并视频
ffmpeg -f concat -i filelist.txt -c copy full_video.mp4
```

**选项2**: 降低输出质量
```python
# 修改脚本中的fourcc
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 更快，较低质量
```

### 处理速度

典型性能（MacBook Pro M1）：
- ~10-15 fps 处理速度
- 1000帧 ≈ 1-2分钟
- 主要瓶颈：视频I/O + cv2.projectPoints

## 高级用法

### 只显示特定身体部位

修改 `draw_skeleton_on_frame()` 函数：

```python
# 只显示上半身（躯干+手臂+头部）
upper_body_joints = ['Pelvis', 'Spine1', 'Neck', 'Head', 'Jaw',
                     'LShoulder', 'LElbow', 'LWrist',
                     'RShoulder', 'RElbow', 'RWrist']

# 过滤joints_2d
joints_2d_filtered = {k: v for k, v in joints_2d.items()
                      if k in upper_body_joints}
```

### 自定义颜色方案

修改 `get_bone_color()` 函数：

```python
def get_bone_color(parent_idx, child_idx, joint_names):
    # 全部使用单一颜色
    return (0, 255, 0)  # 绿色

    # 或者按高度着色
    # return color_by_height(joints_3d[child_idx][1])
```

### 添加关节标签

在 `draw_skeleton_on_frame()` 中添加：

```python
# 在关节旁显示名称
for joint_name, pos in joints_2d.items():
    center = tuple(pos.astype(int))
    cv2.putText(frame, joint_name, (center[0]+5, center[1]-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
```

## 依赖环境

```bash
conda activate multical
pip install numpy opencv-python tqdm
```

## 相关文件

- `project_skeleton_to_video.py` - 本脚本
- `markers_to_skeleton.py` - Skeleton生成脚本
- `skeleton_joints.json` - Skeleton数据（输入）
- `optitrack.mcal` - 相机标定（输入）
- `video.avi` - 原始视频（输入）
- `skeleton_video.mp4` - 结果视频（输出）

## 与Marker投影的区别

| 特性 | Marker投影 | Skeleton投影 |
|------|-----------|-------------|
| 输入数据 | mocap.csv (228 markers) | skeleton_joints.json (17 joints) |
| 可视化 | 散点（绿色圆点） | 骨架（彩色线+点） |
| 数据量 | 大（~70MB CSV） | 小（~2MB JSON） |
| 语义信息 | 无（Unlabeled） | 有（关节名称） |
| 处理速度 | 稍慢（更多点） | 稍快（只17个点） |
| 适用场景 | 验证标定、原始数据 | 人体姿态可视化 |

## 参考文档

- [MARKER_PROJECTION_GUIDE.md](MARKER_PROJECTION_GUIDE.md) - Marker投影方法
- [SKELETON_CONVERSION_README.md](SKELETON_CONVERSION_README.md) - Skeleton生成指南
- [OptiTrack .mcal XML Format](https://docs.optitrack.com/motive/calibration/.mcal-xml-calibration-files)

---

**创建日期**: 2025-10-23
**维护者**: Annotation Pipeline Team
