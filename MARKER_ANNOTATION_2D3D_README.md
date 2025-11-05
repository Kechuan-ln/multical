# Mocap Marker 2D+3D Annotation Tool

## 概述

这是一个增强版的动捕marker标注工具，同时支持**3D和2D视图**的可视化和交互标注。

### 主要特性

✅ **3D scatter plot** - 显示marker的3D空间位置
✅ **2D video view** - 显示视频帧和投影的marker点
✅ **双向交互** - 可以在3D或2D视图中点击选择marker
✅ **同步高亮** - 选中的marker在两个视图中同时高亮显示
✅ **实时投影** - 使用OptiTrack .mcal标定参数将3D点投影到2D
✅ **自动保存** - 标签自动保存到CSV文件

---

## 使用方法

### 1. 基本命令

```bash
conda activate multical  # 或你的环境名

python annotate_mocap_markers_2d3d.py \
  --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \
  --mcal /Volumes/FastACIS/annotation_pipeline/optitrack.mcal \
  --start_frame 100 \
  --num_frames 500 \
  --port 8050
```

### 2. 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--csv` | ✅ | - | Motive导出的mocap CSV文件 |
| `--video` | ✅ | - | 视频文件（.avi, .mp4等） |
| `--mcal` | ✅ | - | OptiTrack标定文件 |
| `--camera_serial` | ❌ | `C11764` | .mcal文件中的相机序列号 |
| `--start_frame` | ❌ | `0` | 起始帧编号 |
| `--num_frames` | ❌ | `500` | 加载帧数 |
| `--labels` | ❌ | `marker_labels.csv` | 标签保存文件 |
| `--port` | ❌ | `8050` | Web服务器端口 |

### 3. 启动后操作

1. **打开浏览器**：访问 `http://localhost:8050`
2. **查看界面**：
   - 左侧：3D scatter plot（marker的3D位置）
   - 右侧：2D视频帧（带投影的marker）
   - 下方：帧滑动条
   - 上方：控制面板

3. **标注流程**：
   - **选择marker**：在3D或2D视图中点击任意marker点
   - **查看信息**：右上角显示选中marker的原始名称和当前标签
   - **输入标签**：在"Label Name"输入框输入标签（如 `Laxisl1`）
   - **保存标签**：点击"Set Label"按钮
   - **自动保存**：标签立即保存到CSV文件

4. **导航**：
   - 使用底部滑动条切换帧
   - 已标注的marker显示为绿色并带标签文字
   - 选中的marker在两个视图中都显示红色高亮

---

## 界面说明

### 视图元素

#### 3D视图（左侧）
- **灰色点**：未标注的marker
- **绿色点+文字**：已标注的marker（按Y坐标高度着色）
- **红色钻石+黄色边框**：当前选中的marker

#### 2D视图（右侧）
- **灰色小圆点**：未标注的marker
- **绿色圆点+文字**：已标注的marker
- **红色大圆+黄色环**：当前选中的marker
- **背景图像**：当前帧的视频画面

### 标注信息面板

**Selected Marker（选中的marker）**：
- 📍 Original Name：原始marker名称（来自CSV）
- 🏷️ Current Label：当前标签（如果已标注）
- 🎯 Source：点击来源（3D View 或 2D View）

**Current Labels（当前标签列表）**：
- 显示所有已标注的marker
- 格式：`🏷️ 标签名 ← 原始名称`

---

## 技术细节

### 坐标投影原理

根据 [MARKER_PROJECTION_GUIDE.md](MARKER_PROJECTION_GUIDE.md)，投影使用以下流程：

1. **加载.mcal标定参数**：
   - 内参：K矩阵（**使用negative fx**补偿OptiTrack坐标系）
   - 畸变系数：`[k1, k2, p1, p2, k3]`
   - 外参：R和T矩阵（Camera-to-World转换为World-to-Camera）

2. **投影计算**：
   ```python
   # 3D (mm) → 2D (pixel)
   markers_m = markers_mm / 1000.0  # mm转米
   points_2d, _ = cv2.projectPoints(markers_m, rvec, tvec, K, dist)
   ```

3. **过滤条件**：
   - ❌ 不检查 `Z > 0`（因为negative fx会产生负Z）
   - ✅ 只检查2D坐标是否在图像范围内

### 数据结构

**markers_xyz（3D数据）**：
```python
{
  'marker_001': np.array([[x, y, z], ...]),  # shape: (num_frames, 3)
  'marker_002': np.array([[x, y, z], ...]),
  ...
}
```

**marker_2d_positions（投影结果）**：
```python
{
  'marker_001': {'x': 1024.5, 'y': 768.2},
  'marker_002': {'x': 512.3, 'y': 400.1},
  ...
}
```

**labels_data（标签）**：
```python
{
  'marker_001': {'label': 'Laxisl1', 'marker_id': '1'},
  'marker_002': {'label': 'Laxisl2', 'marker_id': '2'},
  ...
}
```

### 帧缓存机制

- **VideoFrameReader**类使用LRU缓存
- 默认缓存最近50帧
- 自动管理内存，防止溢出
- BGR自动转换为RGB

---

## 常见问题

### Q1: 启动时报错 "Camera with serial C11764 not found"
**A**: 检查.mcal文件中的相机序列号，使用 `--camera_serial` 参数指定正确的序列号：
```bash
# 查看.mcal中的相机序列号
grep "Camera Serial" /path/to/optitrack.mcal

# 使用正确的序列号
python annotate_mocap_markers_2d3d.py ... --camera_serial C12345
```

### Q2: 2D视图中看不到marker投影
**A**: 可能原因：
1. **分辨率不匹配**：视频分辨率必须与.mcal中的 `ImagerPixelWidth × ImagerPixelHeight` 一致
2. **标定参数错误**：.mcal文件与视频不对应
3. **marker在视野外**：尝试不同的帧（使用滑动条）

启动时会显示警告：
```
⚠️  Warning: Video resolution doesn't match calibration!
   Video: 1920x1080
   Calibration: 2048x1080
```

### Q3: 点击marker没有反应
**A**:
- 确保点击的是marker圆点，而不是背景
- 尝试调大浏览器窗口
- 刷新页面（F5）

### Q4: 标签没有保存
**A**:
- 检查终端输出是否有 `✓ Labels saved to marker_labels.csv`
- 确保当前目录有写入权限
- 使用绝对路径 `--labels /path/to/labels.csv`

### Q5: 视频加载很慢
**A**:
- 使用 `--num_frames` 限制加载的帧数（默认500）
- 缓存会在第一次访问时加载帧，之后会变快
- 考虑使用ffmpeg转换视频为更快的编码格式

---

## 文件依赖

必需文件：
- **mocap.csv** - Motive导出的3D marker数据
- **video.avi** - 视频文件（与mocap同步）
- **calibration.mcal** - OptiTrack标定文件（UTF-16LE XML格式）

输出文件：
- **marker_labels.csv** - 标签保存文件（自动创建）

格式：
```csv
original_name,marker_id,label
marker_001,1,Laxisl1
marker_002,2,Laxisl2
```

---

## 环境要求

```bash
conda activate multical  # 或你的环境
pip install pandas numpy opencv-python plotly dash pillow
```

依赖版本：
- Python >= 3.8
- opencv-python >= 4.5
- plotly >= 5.0
- dash >= 2.0
- pandas >= 1.3
- numpy >= 1.20

---

## 与原版工具的对比

| 特性 | 原版 (`annotate_mocap_markers.py`) | 增强版 (本工具) |
|------|-----------------------------------|----------------|
| 3D视图 | ✅ | ✅ |
| 2D视图 | ❌ | ✅ 显示视频+投影 |
| 点击选择 | ✅ 仅3D | ✅ 2D和3D都支持 |
| 同步高亮 | - | ✅ 两个视图同步 |
| 实时投影 | ❌ | ✅ 基于.mcal标定 |
| 视频帧显示 | ❌ | ✅ 直观的2D可视化 |

---

## 工作流程建议

1. **初步浏览**（快速）：
   ```bash
   # 每隔10帧采样
   python annotate_mocap_markers_2d3d.py \
     --csv mocap.csv --video video.avi --mcal calib.mcal \
     --start_frame 0 --num_frames 100
   ```

2. **精细标注**（详细）：
   - 在3D视图中识别marker的空间关系
   - 在2D视图中验证投影是否正确
   - 利用视频背景辅助识别身体部位

3. **验证标注**：
   - 切换不同帧查看标签一致性
   - 检查绿色marker（已标注）是否覆盖所有目标

4. **导出标签**：
   - 标签自动保存在 `marker_labels.csv`
   - 可用于后续的骨架转换或分析

---

## 扩展功能建议

未来可以添加：
- [ ] 批量标注（一次标注多个相似marker）
- [ ] 撤销/重做功能
- [ ] 标签导入/导出（JSON格式）
- [ ] 2D视图放大功能（zoom）
- [ ] 播放模式（自动播放帧序列）
- [ ] 标签搜索和过滤

---

## 参考资料

- [MARKER_PROJECTION_GUIDE.md](MARKER_PROJECTION_GUIDE.md) - 3D到2D投影技术文档
- [project_markers_final.py](project_markers_final.py) - 批量投影脚本
- [OptiTrack .mcal XML Calibration Files](https://docs.optitrack.com/motive/calibration/.mcal-xml-calibration-files)
- [OpenCV projectPoints Documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c)

---

**创建日期**：2025-10-28
**作者**：Claude Code
**版本**：1.0
