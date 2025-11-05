# project_markers_to_video_v2.py 使用指南

## 新功能

相比 V1 版本，V2 添加了以下功能：

✅ **帧范围控制**: 可以指定起始帧和处理帧数
✅ **配置嵌入**: 所有参数在代码中配置，无需命令行参数
✅ **清晰配置区**: 在 `main()` 函数顶部有明确的配置区域

---

## 快速开始

### 1. 打开文件

编辑 `project_markers_to_video_v2.py`

### 2. 修改配置区域

在 `main()` 函数中找到配置区域（第 386-413 行）：

```python
# ============================================================
# 配置区域 - 在此修改所有参数
# ============================================================

# 输入文件路径
mcal_path = Path("/Volumes/FastACIS/annotation_pipeline/optitrack.mcal")
mocap_csv = Path("/Volumes/FastACIS/csldata/csl/mocap.csv")
video_path = Path("/Volumes/FastACIS/csldata/video/mocap.avi")

# 标定文件路径（可选，如不指定则使用 mcal）
INTRINSICS_JSON = Path("/Volumes/FastACIS/gopro/prime_gopro_sync/intrinsic_merged.json")  # 用户标定的内参
EXTRINSICS_JSON = Path("/Volumes/FastACIS/annotation_pipeline/extrinsics_calibrated.json")  # 用户标定的外参

# 输出文件路径
output_path = Path("/Volumes/FastACIS/csldata/video/mocap_with_markers_v2.mp4")

# 帧范围设置
START_FRAME = 5747       # 起始帧号（从0开始）
NUM_FRAMES = 100         # 处理帧数（None = 处理到视频结束）

# 可视化设置
MARKER_COLOR = (0, 255, 0)    # BGR格式：绿色
MARKER_SIZE = 3               # Marker半径（像素）
```

### 3. 运行脚本

```bash
python project_markers_to_video_v2.py
```

---

## 参数说明

### 输入文件

| 参数 | 类型 | 说明 |
|------|------|------|
| `mcal_path` | Path | OptiTrack .mcal 文件（用于图像尺寸和fallback标定） |
| `mocap_csv` | Path | OptiTrack 导出的 mocap CSV 文件（包含 marker 3D 坐标） |
| `video_path` | Path | PrimeColor 输入视频 |
| `INTRINSICS_JSON` | Path (可选) | 用户标定的内参 JSON（multical格式），如不指定则使用 .mcal 内参 |
| `EXTRINSICS_JSON` | Path (可选) | 用户标定的外参 JSON（rvec/tvec格式），如不指定则使用 .mcal 外参 |

### 输出文件

| 参数 | 类型 | 说明 |
|------|------|------|
| `output_path` | Path | 带 marker 投影的输出视频路径 |

### 帧范围设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `START_FRAME` | int | 0 | 起始帧号（从 0 开始） |
| `NUM_FRAMES` | int or None | None | 处理帧数（None = 处理到视频结束） |

### 可视化设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `MARKER_COLOR` | tuple | (0, 255, 0) | Marker 颜色，BGR 格式<br>绿色: (0, 255, 0)<br>红色: (0, 0, 255)<br>蓝色: (255, 0, 0)<br>黄色: (0, 255, 255) |
| `MARKER_SIZE` | int | 3 | Marker 圆点半径（像素） |

---

## 使用示例

### 示例 1：处理整个视频

```python
START_FRAME = 0
NUM_FRAMES = None
```

这将从第 0 帧开始，处理到视频结束。

### 示例 2：处理前 1000 帧

```python
START_FRAME = 0
NUM_FRAMES = 1000
```

这将处理帧 0-999（共 1000 帧）。

### 示例 3：处理中间的 500 帧

```python
START_FRAME = 1000
NUM_FRAMES = 500
```

这将处理帧 1000-1499（共 500 帧）。

### 示例 4：从第 500 帧处理到视频结束

```python
START_FRAME = 500
NUM_FRAMES = None
```

这将从第 500 帧开始，处理到视频结束。

### 示例 5：修改 marker 颜色和大小

```python
MARKER_COLOR = (0, 0, 255)   # 红色
MARKER_SIZE = 5              # 更大的圆点
```

---

## 输出信息

运行时会显示：

```
============================================================
Loading user-calibrated intrinsics from: ...
  Intrinsics: fx=..., fy=..., cx=..., cy=...
  Image size: 1920x1080

Loading OptiTrack extrinsics from: ...
  Camera position (world frame): [...] meters
  Using Method 4 with negative fx for correct projection
============================================================
Loading mocap data from ...
Found XX markers

Video properties:
  Resolution: 1920x1080
  FPS: 30.0
  Total frames: 10000
  Mocap frames: 10000

Processing range:
  Start frame: 0
  End frame: 1000
  Total frames to process: 1000

Processing frames: 100%|████████████████| 1000/1000 [00:30<00:00, 33.33it/s]

Projection statistics:
  Total markers detected: 50000
  Markers in front of camera: 48000
  Markers projected in image: 45000
  Success rate: 90.0%

Output saved to: .../mocap_with_markers_v2.mp4
============================================================
Done!
```

---

## 常见问题

### Q1: 视频处理很慢怎么办？

**A**: 使用帧范围功能，先处理一小段测试：

```python
START_FRAME = 0
NUM_FRAMES = 100  # 只处理前 100 帧测试
```

确认效果后再处理完整视频。

### Q2: Marker 投影位置不对？

**A**: 检查以下几点：

1. 确认 `intrinsic_json` 路径正确，对应 PrimeColor 相机
2. 确认 `mcal_path` 包含正确的相机外参
3. 确认 mocap 和 video 的时间同步正确

### Q3: 有些 marker 没有显示？

**A**: 可能的原因：

1. Marker 在相机后方（不可见）
2. Marker 在图像边界外
3. Marker 数据缺失（mocap CSV 中为空）

检查输出的统计信息：
```
Total markers detected: 50000      # mocap 中检测到的 marker
Markers in front of camera: 48000  # 在相机前方的 marker
Markers projected in image: 45000  # 成功投影在图像内的 marker
```

### Q4: 想要不同颜色的 marker？

**A**: 修改 `MARKER_COLOR`：

```python
# BGR 格式
MARKER_COLOR = (0, 255, 0)    # 绿色
MARKER_COLOR = (0, 0, 255)    # 红色
MARKER_COLOR = (255, 0, 0)    # 蓝色
MARKER_COLOR = (0, 255, 255)  # 黄色
MARKER_COLOR = (255, 0, 255)  # 品红色
MARKER_COLOR = (255, 255, 0)  # 青色
```

### Q5: marker 太小/太大？

**A**: 修改 `MARKER_SIZE`：

```python
MARKER_SIZE = 1   # 很小
MARKER_SIZE = 3   # 默认
MARKER_SIZE = 5   # 较大
MARKER_SIZE = 10  # 很大
```

### Q6: "Total markers detected: 0"，但 mocap 文件有数据？

**A**: 这可能是 CSV header 解析问题（V2.1 已修复）。如果遇到此问题：

1. 检查是否使用最新版本的 `project_markers_to_video_v2.py`
2. 使用 `debug_mocap_frames.py` 诊断哪些帧有数据：
   ```bash
   python debug_mocap_frames.py
   ```
3. 确认 mocap CSV 文件格式正确（OptiTrack 标准导出格式）
4. 检查 `START_FRAME` 是否设置在有数据的帧范围内

**技术细节**: OptiTrack CSV 格式中，marker names 在第4行（前3行是metadata）。如果脚本读取错误的行，会导致无法解析 marker 数据。

---

## 技术细节

### 坐标系转换

脚本执行以下转换链：

```
OptiTrack 3D (mm)
    ↓ (除以 1000 转为米)
OptiTrack 3D (m)
    ↓ (使用 rvec, tvec 从 .mcal)
PrimeColor Camera 3D
    ↓ (使用 K 和 dist 从 intrinsic_merged.json)
PrimeColor Image 2D (pixels)
```

### 使用的标定参数

1. **内参** (来自 `INTRINSICS_JSON` 或 fallback 到 .mcal):
   - K 矩阵: 焦距 fx, fy 和主点 cx, cy
   - 畸变系数: k1, k2, p1, p2, k3
   - 推荐使用用户标定的内参 JSON 文件

2. **外参** (来自 `EXTRINSICS_JSON` 或 fallback 到 .mcal):
   - rvec: 旋转向量（3x1）
   - tvec: 平移向量（3x1）
   - 推荐使用用户标定的外参（Mocap → PrimeColor）

3. **Method 4 with negative fx**: 用于修正 OptiTrack 坐标系的 X 轴镜像问题

---

## 与其他脚本的区别

| 脚本 | 内参来源 | 外参来源 | 目标相机 | 帧范围控制 | 灵活配置 |
|------|----------|----------|----------|------------|----------|
| `project_markers_to_video.py` | .mcal | .mcal | PrimeColor | ❌ | ❌ |
| `project_markers_to_video_v2.py` | JSON (可选) / .mcal | JSON (可选) / .mcal | PrimeColor | ✅ | ✅ |
| `project_markers_final.py` | JSON (可选) / .mcal | JSON (可选) / .mcal | PrimeColor | ✅ (命令行) | ❌ (命令行参数) |
| `project_markers_to_gopro.py` | calibration.json | calibration.json + .mcal | GoPro | ✅ | ❌ |

---

## 更新日志

**V2.1** (2025-10-29):
- 🐛 **修复**: CSV header 解析错误（现在正确读取第4行marker names）
- ✅ 添加灵活的标定源支持（支持用户标定的内参/外参 JSON）
- ✅ 添加 `start_frame` 和 `num_frames` 参数
- ✅ 配置区域移到 `main()` 函数顶部
- ✅ 改进文档和注释
- ✅ 测试验证：100%投影成功率（3800/3800 markers）

**V1** (2024-10-23):
- 初始版本
- 使用 user-calibrated intrinsics

---

如有问题，请检查：
1. 文件路径是否正确
2. 标定文件格式是否正确
3. mocap 和 video 是否时间同步
4. 输出统计信息中的 success rate
