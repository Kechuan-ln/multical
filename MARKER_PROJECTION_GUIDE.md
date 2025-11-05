# OptiTrack Marker 3D-to-2D Projection Guide

## 概述

本文档说明如何将 OptiTrack 动捕系统的 3D marker 坐标投影到 PrimeColor 相机的 2D 视频帧上。

**核心发现**：
- OptiTrack 使用 `-Z` 轴指向相机前方（与 OpenCV 的 `+Z` 约定相反）
- 需要使用 **negative focal length (fx)** 来补偿坐标系差异
- 最终方法产生正确的 2D 投影（尽管几何上 Z < 0）

---

## 1. OptiTrack .mcal 文件解析

### 1.1 文件格式

`.mcal` 文件是 UTF-16LE 编码的 XML 文件，包含 OptiTrack 系统所有相机的标定参数。

### 1.2 关键数据结构

每个相机包含以下标签：

```xml
<Camera Serial="C11764">
    <Attributes ImagerPixelWidth="1920" ImagerPixelHeight="1080"/>

    <!-- 内参：标准相机模型 -->
    <IntrinsicStandardCameraModel
        LensCenterX="960.596"
        LensCenterY="538.614"
        HorizontalFocalLength="1247.841"
        VerticalFocalLength="1247.755"
        k1="0.136385"
        k2="-0.125546"
        k3="0.000035"
        TangentialX="0.000259"
        TangentialY="-0.000259"/>

    <!-- 外参：相机在世界坐标系的位置和朝向 -->
    <Extrinsic
        X="-0.268640"
        Y="2.655145"
        Z="-3.509723"
        OrientMatrix0="-0.995356"
        OrientMatrix1="-0.046982"
        OrientMatrix2="0.084021"
        OrientMatrix3="0.000355"
        OrientMatrix4="0.871016"
        OrientMatrix5="0.491255"
        OrientMatrix6="-0.096264"
        OrientMatrix7="0.489003"
        OrientMatrix8="-0.866954"/>
</Camera>
```

### 1.3 参数解释

#### 内参 (Intrinsics)

| 参数 | 含义 | OpenCV 对应 |
|------|------|-------------|
| `HorizontalFocalLength` | 水平焦距（像素） | `fx` |
| `VerticalFocalLength` | 垂直焦距（像素） | `fy` |
| `LensCenterX` | 主点 X 坐标 | `cx` |
| `LensCenterY` | 主点 Y 坐标 | `cy` |
| `k1`, `k2`, `k3` | 径向畸变系数 | `dist[0]`, `dist[1]`, `dist[4]` |
| `TangentialX`, `TangentialY` | 切向畸变系数 | `dist[2]` (p1), `dist[3]` (p2) |

**相机内参矩阵 K**：
```
K = [fx   0   cx]
    [0   fy   cy]
    [0    0    1]
```

**畸变系数向量**：
```
dist = [k1, k2, p1, p2, k3]
```

#### 外参 (Extrinsics)

| 参数 | 含义 |
|------|------|
| `X`, `Y`, `Z` | **相机在世界坐标系中的位置**（米） |
| `OrientMatrix0-8` | **Camera-to-World 旋转矩阵** R_c2w（row-major，3×3） |

**旋转矩阵构造**：
```python
R_c2w = [[OrientMatrix0, OrientMatrix1, OrientMatrix2],
         [OrientMatrix3, OrientMatrix4, OrientMatrix5],
         [OrientMatrix6, OrientMatrix7, OrientMatrix8]]
```

**关键理解**：
- `T_world = [X, Y, Z]`：相机光心在世界坐标系的位置
- `R_c2w`：将相机坐标系的向量转换到世界坐标系（**不是** World-to-Camera）

---

## 2. 坐标系转换公式

### 2.1 标准计算机视觉公式

标准的 3D-to-2D 投影公式（OpenCV）：

```
p_2d = K @ [R | t] @ P_world

其中：
- P_world: 3D 点的世界坐标 [X, Y, Z, 1]ᵀ
- R: World-to-Camera 旋转矩阵 (3×3)
- t: World-to-Camera 平移向量 (3×1)
- K: 相机内参矩阵 (3×3)
- p_2d: 2D 图像坐标（齐次坐标）
```

### 2.2 从 OptiTrack 参数计算 World-to-Camera

由于 OptiTrack 提供的是 **Camera-to-World** 参数，需要转换：

```python
# OptiTrack 提供的参数
R_c2w = 3×3 rotation matrix from OrientMatrix0-8
T_world = [X, Y, Z]  # 相机在世界坐标系的位置

# 转换为 World-to-Camera
R_w2c = R_c2w.T  # 旋转矩阵的转置 = 逆矩阵（对于正交矩阵）
t_w2c = -R_w2c @ T_world  # 平移向量

# OpenCV 格式
rvec = cv2.Rodrigues(R_w2c)[0]  # 旋转向量（3×1）
tvec = t_w2c  # 平移向量（3×1）
```

### 2.3 OptiTrack 坐标系差异处理

**关键问题**：
- **OptiTrack 相机坐标系**：`-Z` 轴指向前方
- **OpenCV 相机坐标系**：`+Z` 轴指向前方

**解决方案（Method 4）**：
使用 **negative focal length** `fx` 来补偿：

```python
K = [[-fx,  0,  cx],
     [ 0,  fy,  cy],
     [ 0,   0,   1]]
```

这个方法虽然产生 Z < 0 的相机坐标（几何上"在相机后方"），但 2D 投影结果**完全正确**。

### 2.4 完整投影流程

```python
import numpy as np
import cv2

# 1. 读取 OptiTrack 参数
fx, fy, cx, cy = ...  # 内参
k1, k2, k3, p1, p2 = ...  # 畸变
R_c2w = ...  # Camera-to-World 旋转
T_world = ...  # 相机世界位置

# 2. 构造 OpenCV 参数
K = np.array([[-fx, 0, cx],      # 注意：负 fx！
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float64)

dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

R_w2c = R_c2w.T
rvec, _ = cv2.Rodrigues(R_w2c)
tvec = -R_w2c @ T_world

# 3. 投影 3D 点（世界坐标，单位：米）
P_world_m = markers_mm / 1000.0  # 转换 mm → m
points_2d, _ = cv2.projectPoints(
    P_world_m.reshape(-1, 1, 3),
    rvec, tvec, K, dist
)
points_2d = points_2d.reshape(-1, 2)

# 4. 过滤：只保留图像范围内的点
in_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
valid_points = points_2d[in_bounds]
```

**注意**：
- ❌ **不要**检查 `Z > 0`（因为 negative fx 会导致 Z < 0）
- ✅ **只需**检查 2D 坐标是否在图像范围内

---

## 3. 生产环境代码

### 3.1 主要脚本

**文件**：`project_markers_final.py`

**功能**：将 OptiTrack mocap CSV 中的 3D markers 投影到 PrimeColor 视频上

**依赖**：
```bash
conda activate multical
pip install numpy opencv-python pandas tqdm
```

### 3.2 使用方法

```bash
# 基本用法：处理整个视频
python project_markers_final.py \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \
  --output /Volumes/FastACIS/GoPro/motion/mocap/mocap_with_markers.mp4

# 处理指定帧范围
python project_markers_final.py \
  --start-frame 1000 \
  --num-frames 500 \
  --output output_1000-1500.mp4

# 自定义 marker 样式
python project_markers_final.py \
  --marker-size 5 \
  --marker-color 0,0,255 \
  --output output_red_markers.mp4
```

### 3.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mcal` | `optitrack.mcal` | OptiTrack 标定文件路径 |
| `--csv` | `mocap.csv` | Mocap 数据 CSV 文件 |
| `--video` | `mocap.avi` | 输入视频文件 |
| `--output` | `mocap_with_markers.mp4` | 输出视频路径 |
| `--start-frame` | `0` | 起始帧编号 |
| `--num-frames` | `-1` | 处理帧数（-1=全部） |
| `--marker-size` | `3` | Marker 圆点半径（像素） |
| `--marker-color` | `0,255,0` | Marker 颜色（BGR 格式，绿色） |

### 3.4 示例输出

```
======================================================================
OptiTrack Marker Projection to Video
======================================================================

Loading calibration...
  Camera intrinsics: fx=1247.84 (negative for coord conversion), fy=1247.75
  Image size: 1920x1080

Loading mocap data...
  Total frames: 23376
  Total markers: 228

Opening video...
  Video frames: 23375
  FPS: 120.0

Processing frames 0 to 23374
Projecting markers: 100%|████████████████| 23375/23375 [05:45<00:00, 67.56it/s]

✓ Done!
  Frames with projections: 22777/23375
  Output saved to: /Volumes/FastACIS/csldata/video/mocap_with_markers_full.mp4
```

---

## 4. 核心代码片段

### 4.1 加载标定参数

```python
def load_optitrack_calibration(mcal_path, camera_serial='C11764'):
    """从 OptiTrack .mcal 文件加载相机标定参数。"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(mcal_path)
    root = tree.getroot()

    for cam in root.findall('.//Camera'):
        if cam.get('Serial') == camera_serial:
            # 图像尺寸
            attributes = cam.find('Attributes')
            width = int(attributes.get('ImagerPixelWidth'))
            height = int(attributes.get('ImagerPixelHeight'))

            # 内参
            intrinsic = cam.find('.//IntrinsicStandardCameraModel')
            fx = float(intrinsic.get('HorizontalFocalLength'))
            fy = float(intrinsic.get('VerticalFocalLength'))
            cx = float(intrinsic.get('LensCenterX'))
            cy = float(intrinsic.get('LensCenterY'))
            k1 = float(intrinsic.get('k1'))
            k2 = float(intrinsic.get('k2'))
            k3 = float(intrinsic.get('k3'))
            p1 = float(intrinsic.get('TangentialX'))
            p2 = float(intrinsic.get('TangentialY'))

            # 关键：使用 negative fx
            K = np.array([[-fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float64)
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

            # 外参
            extrinsic = cam.find('Extrinsic')
            T_world = np.array([
                float(extrinsic.get('X')),
                float(extrinsic.get('Y')),
                float(extrinsic.get('Z'))
            ])

            R_c2w = np.array([
                [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(3)],
                [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(3,6)],
                [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(6,9)]
            ])

            # 转换为 World-to-Camera
            R_w2c = R_c2w.T
            rvec, _ = cv2.Rodrigues(R_w2c)
            tvec = -R_w2c @ T_world

            return K, dist, rvec, tvec, [width, height]
```

### 4.2 投影 Markers

```python
def project_markers_to_frame(markers_mm, K, dist, rvec, tvec, img_size):
    """将 3D markers 投影到 2D 图像坐标。"""
    # 过滤 NaN
    valid_mask = ~np.isnan(markers_mm[:, 0])
    if not valid_mask.any():
        return None

    valid_markers = markers_mm[valid_mask]

    # 转换单位：mm → m
    markers_m = valid_markers / 1000.0

    # OpenCV 投影
    points_2d, _ = cv2.projectPoints(
        markers_m.reshape(-1, 1, 3),
        rvec, tvec, K, dist
    )
    points_2d = points_2d.reshape(-1, 2)

    # 只检查图像边界（不检查 Z！）
    in_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_size[0]) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_size[1])

    if not in_bounds.any():
        return None

    return points_2d[in_bounds]
```

---

## 5. 故障排除

### 5.1 常见问题

#### Q: 投影结果看不到 markers
**A**: 检查：
1. `.mcal` 文件和 `mocap.csv` 是否来自同一次标定会话
2. 视频分辨率是否与 `.mcal` 中的 `ImagerPixelWidth/Height` 一致
3. 相机 Serial 是否正确（默认 `C11764`）

#### Q: Frames with projections: 0/N
**A**: 可能原因：
1. **代码检查了 Z > 0**：确保使用 `project_markers_final.py`（已移除 Z 检查）
2. Marker 数据全是 NaN：检查 CSV 文件格式
3. 视频无法读取：检查文件路径

#### Q: 投影位置偏移很大
**A**:
1. **未使用 negative fx**：必须使用 `K[0,0] = -fx`
2. 内参不匹配：确保使用 `.mcal` 中的内参
3. 畸变参数错误：检查 `k1, k2, p1, p2, k3` 的顺序

### 5.2 验证方法

```bash
# 生成单帧测试图像
python project_markers_final.py \
  --start-frame 100 \
  --num-frames 1 \
  --output test_frame100.mp4

# 提取帧查看
ffmpeg -i test_frame100.mp4 -vframes 1 test_frame100.jpg
open test_frame100.jpg
```

---

## 6. 技术细节说明

### 6.1 为什么使用 Negative fx？

OptiTrack 和 OpenCV 的相机坐标系 Z 轴方向相反：

```
OptiTrack:           OpenCV:
    Y                   Y
    |                   |
    |                   |
    +---X               +---X
   /                   /
  Z (backward)       Z (forward)
```

Negative fx 实现了 X 轴的镜像翻转，配合标准 W2C 变换恰好补偿这个差异：

```
标准投影：x' = fx * (X_cam / Z_cam) + cx
Negative fx: x' = -fx * (X_cam / Z_cam) + cx = fx * (-X_cam / Z_cam) + cx
```

这等效于先对相机坐标系做 `X → -X` 的翻转。

### 6.2 为什么不检查 Z > 0？

使用 negative fx 后，即使几何上 markers 在相机"后方"（Z < 0），投影公式仍然产生正确的 2D 坐标。这是因为：

1. Negative fx 已经隐含了坐标系翻转
2. OpenCV 的 `projectPoints` 在 Z < 0 时仍会计算（数学上合法）
3. 最终的 2D 坐标**在图像范围内**且**与实际视频对齐**

因此只需检查 `in_bounds`，不需要检查 `Z > 0`。

### 6.3 其他尝试过的方法

| Method | 变换方式 | Z > 0 | In Bounds | 结果 |
|--------|---------|-------|-----------|------|
| 1. Standard W2C | 正常 fx, R_w2c | ❌ 0/16 | ✓ 10/16 | ❌ 几何错误 |
| 2. Flip Z | 正常 fx, flip_z @ R_w2c | ✓ 16/16 | ❌ 0/16 | ❌ 投影错误 |
| 3. Flip world Z | 正常 fx, R_w2c @ flip_z | ✓ 16/16 | ❌ 0/16 | ❌ 投影错误 |
| **4. Negative fx** | **-fx, R_w2c** | **0/16** | **✓ 10/16** | **✅ 完全正确** |
| 5. Flip XZ | 正常 fx, flip_xz @ R_w2c | ✓ 16/16 | ✓ 10/16 | ❌ 投影错误 |
| 6. Negative fx + flip Z | -fx, flip_z @ R_w2c | ✓ 16/16 | ❌ 0/16 | ❌ 投影错误 |

**Method 4 (Negative fx)** 是唯一产生正确结果的方法。

---

## 7. 输入文件要求

### 7.1 OptiTrack .mcal 文件
- UTF-16LE 编码的 XML
- 包含 PrimeColor 相机（Serial="C11764"）的内参和外参
- 与 mocap 数据来自**同一次标定会话**

### 7.2 Mocap CSV 文件
- OptiTrack Motive 导出格式
- 前 7 行为元数据，第 8 行开始为数据
- 每个 marker 占 3 列（X, Y, Z），单位：毫米
- 缺失数据用空字符串或 NaN 表示

### 7.3 视频文件
- 格式：AVI, MP4 等 OpenCV 支持的格式
- 分辨率：必须与 .mcal 中的 `ImagerPixelWidth × ImagerPixelHeight` 一致
- 帧数：应与 mocap CSV 的行数匹配

---

## 8. 参考资料

### 8.1 OptiTrack 文档
- [.mcal XML Calibration Files](https://docs.optitrack.com/motive/calibration/.mcal-xml-calibration-files)
- [Motive Coordinate Systems](https://forums.naturalpoint.com/viewtopic.php?t=19116)

### 8.2 OpenCV 文档
- [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [cv::projectPoints](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c)

### 8.3 相关文件
- `project_markers_final.py`：生产环境投影脚本
- `test_coordinate_flip.py`：坐标系转换测试脚本
- `optitrack.mcal`：OptiTrack 标定文件
- `mocap.csv`：Mocap 数据文件

---

**创建日期**：2025-10-23  
**最后更新**：2025-10-23  
**维护者**：Annotation Pipeline Team
