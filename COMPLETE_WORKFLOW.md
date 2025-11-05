# 完整工作流程：从Mocap Markers到视频Skeleton投影

## 概述

这是一个完整的3D人体姿态标注和可视化流程：

```
Mocap CSV (228 markers)
    ↓
[标注工具] → Marker Labels (38 markers)
    ↓
[转换] → Skeleton (17 joints)
    ↓
[投影] → Video with Skeleton
```

## 🚀 快速开始（5步完成）

### 步骤1: 标注Markers

```bash
python annotate_mocap_markers.py --start_frame 2 --num_frames 200
```

- 打开 `http://localhost:8050`
- 点击并标注38个markers
- 标签自动保存到 `marker_labels.csv`

**需要标注的markers**：见 [MARKER_ANNOTATION_README.md](MARKER_ANNOTATION_README.md)

### 步骤2: 转换为Skeleton

```bash
python markers_to_skeleton.py \
  --mocap_csv /Volumes/FastACIS/csldata/csl/mocap.csv \
  --labels_csv marker_labels.csv \
  --start_frame 2 \
  --end_frame 10000
```

**输出**：
- `skeleton_joints.csv` - CSV格式
- `skeleton_joints.json` - JSON格式（17关节）

### 步骤3: 可视化Skeleton（离线）

```bash
# 生成MP4视频
python visualize_skeleton_video.py \
  --skeleton_json skeleton_joints.json \
  --output skeleton_3d.mp4 \
  --num_frames 200 \
  --fps 30

# 或生成GIF
python visualize_skeleton_video.py \
  --skeleton_json skeleton_joints.json \
  --output skeleton_3d.gif \
  --num_frames 100 \
  --fps 20
```

**输出**：3D骨架动画（可旋转视图的离线渲染）

### 步骤4: 投影到真实视频

```bash
python project_skeleton_to_video.py \
  --mcal /path/to/optitrack.mcal \
  --skeleton skeleton_joints.json \
  --video /path/to/video.avi \
  --output skeleton_video.mp4
```

**输出**：真实视频上叠加骨架

### 步骤5: 查看结果

```bash
# 查看3D动画
open skeleton_3d.mp4

# 查看投影视频
open skeleton_video.mp4
```

## 📁 文件结构

```
annotation_pipeline/
├── 数据文件
│   ├── marker_labels.csv              # 步骤1输出：Marker标注
│   ├── skeleton_joints.csv            # 步骤2输出：Skeleton CSV
│   ├── skeleton_joints.json           # 步骤2输出：Skeleton JSON
│   ├── skeleton_3d.mp4                # 步骤3输出：3D动画
│   └── skeleton_video.mp4             # 步骤4输出：投影视频
│
├── 配置文件
│   └── skeleton_config.json           # Skeleton定义（17关节）
│
├── 核心脚本
│   ├── annotate_mocap_markers.py      # 步骤1：交互式标注
│   ├── markers_to_skeleton.py         # 步骤2：Markers→Skeleton
│   ├── visualize_skeleton_video.py    # 步骤3：3D可视化
│   └── project_skeleton_to_video.py   # 步骤4：投影到视频
│
├── 辅助工具
│   ├── check_mocap_data.py            # 检查mocap数据范围
│   ├── visualize_mocap.py             # Markers 3D可视化
│   ├── visualize_skeleton.py          # Skeleton交互式可视化
│   └── run_skeleton_projection.sh     # 快速启动脚本
│
└── 文档
    ├── COMPLETE_WORKFLOW.md           # 本文档
    ├── MARKER_ANNOTATION_README.md    # 标注工具说明
    ├── SKELETON_CONVERSION_README.md  # 转换工具说明
    ├── SKELETON_PROJECTION_GUIDE.md   # 投影工具说明
    ├── MARKER_PROJECTION_GUIDE.md     # Marker投影方法
    └── QUICK_START_SKELETON.md        # 快速入门
```

## 🎯 两种可视化方式

### 方式1: 3D动画（离线渲染）

**脚本**: `visualize_skeleton_video.py`

**优点**：
- ✅ 可以从任意角度观察
- ✅ 清晰展示骨架结构
- ✅ 适合理解姿态

**缺点**：
- ❌ 不是真实相机视角
- ❌ 没有场景上下文

**适用场景**：
- 验证skeleton计算是否正确
- 分析人体动作
- 制作演示动画

### 方式2: 视频投影（真实视角）

**脚本**: `project_skeleton_to_video.py`

**优点**：
- ✅ 真实相机视角
- ✅ 结合场景上下文
- ✅ 验证标定准确性

**缺点**：
- ❌ 固定视角
- ❌ 可能有遮挡

**适用场景**：
- 验证marker-相机对齐
- 制作真实场景演示
- 姿态估计对比

## 🔧 常用命令组合

### 场景1: 快速测试（100帧）

```bash
# 1. 标注
python annotate_mocap_markers.py --num_frames 100

# 2. 转换
python markers_to_skeleton.py --start_frame 2 --end_frame 102

# 3. 3D预览
python visualize_skeleton_video.py --num_frames 100 --output preview.mp4

# 4. 投影（如果满意）
python project_skeleton_to_video.py --num_frames 100 --output test.mp4
```

### 场景2: 处理整个序列

```bash
# 1. 标注（加载200帧用于标注，足够覆盖所有markers）
python annotate_mocap_markers.py --num_frames 200

# 2. 转换全部帧
python markers_to_skeleton.py --start_frame 2 --end_frame 23374

# 3. 生成高质量3D动画（前1000帧）
python visualize_skeleton_video.py \
  --num_frames 1000 \
  --fps 60 \
  --dpi 150 \
  --output skeleton_hq.mp4

# 4. 投影到视频（前1000帧）
python project_skeleton_to_video.py \
  --num_frames 1000 \
  --output skeleton_projected.mp4
```

### 场景3: 分段处理大视频

```bash
# 每次处理5000帧
for i in {0..4}; do
  START=$((i * 5000))
  python project_skeleton_to_video.py \
    --start-frame $START \
    --num-frames 5000 \
    --output part_${i}.mp4
done

# 合并视频
ffmpeg -f concat -safe 0 -i <(for i in {0..4}; do echo "file 'part_${i}.mp4'"; done) -c copy full_video.mp4
```

## 📊 数据流和格式

### Mocap CSV → Marker Labels
```
Input:  mocap.csv (23375 frames × 228 markers × 3 coords = 160MB)
        ↓ [annotate_mocap_markers.py]
Output: marker_labels.csv (38 rows)
```

### Marker Labels → Skeleton JSON
```
Input:  marker_labels.csv (38 labeled markers)
        mocap.csv (raw 3D positions)
        ↓ [markers_to_skeleton.py]
Output: skeleton_joints.json (17 joints × N frames)
        skeleton_joints.csv (17 joints × N frames)
```

### Skeleton → Video
```
Input:  skeleton_joints.json (17 joints, mm)
        ↓ [visualize_skeleton_video.py]
Output: skeleton_3d.mp4 (3D animation)

Input:  skeleton_joints.json (17 joints, mm)
        optitrack.mcal (calibration)
        video.avi (raw footage)
        ↓ [project_skeleton_to_video.py]
Output: skeleton_video.mp4 (projected overlay)
```

## 🎨 可视化对比

| 特性 | 3D动画 | 视频投影 |
|------|--------|----------|
| 脚本 | `visualize_skeleton_video.py` | `project_skeleton_to_video.py` |
| 输入 | `skeleton_joints.json` | `skeleton_joints.json` + `.mcal` + `video` |
| 视角 | 可旋转（离线渲染） | 固定（相机视角） |
| 背景 | 纯色 | 真实场景 |
| 骨架颜色 | 按部位着色 | 按部位着色 |
| 文件大小 | ~10-50MB (100帧) | ~50-200MB (100帧) |
| 处理速度 | ~30s/100帧 | ~60s/100帧 |
| 适用场景 | 动作分析、演示 | 验证对齐、真实场景 |

## ⚙️ 技术要点

### 坐标系统

**Mocap数据**：
- Y轴向上（垂直）
- XZ为水平面
- 单位：毫米 (mm)

**投影到视频**：
- OptiTrack: -Z轴向前
- OpenCV: +Z轴向前
- **解决方案**: 使用negative fx

### 关键转换

```python
# 1. Markers (mm) → Skeleton (mm)
skeleton = compute_joints_from_markers(markers)

# 2. Skeleton (mm) → Skeleton (m)
skeleton_m = skeleton / 1000.0

# 3. World coords (m) → Camera coords (m)
cam_coords = R_w2c @ (world_coords - T_world)

# 4. Camera coords (m) → Image coords (pixels)
image_coords = project_with_negative_fx(cam_coords, K, dist)
```

## 🔍 质量检查清单

### ✅ 标注质量
- [ ] 所有38个markers已标注
- [ ] 标签命名正确（大小写匹配）
- [ ] marker_labels.csv文件存在

### ✅ Skeleton质量
- [ ] 17个关节中至少12个成功计算
- [ ] Pelvis, LHip, RHip 存在（必需）
- [ ] skeleton_joints.json文件正确生成

### ✅ 3D动画质量
- [ ] 骨架看起来像人形
- [ ] 运动流畅（无跳跃）
- [ ] 身体比例合理

### ✅ 投影质量
- [ ] 骨架在视频画面内
- [ ] 骨架位置与人体对齐
- [ ] 没有大的偏移或抖动

## 📚 完整文档索引

### 标注阶段
- [MARKER_ANNOTATION_README.md](MARKER_ANNOTATION_README.md) - 交互式标注工具完整说明

### 转换阶段
- [SKELETON_CONVERSION_README.md](SKELETON_CONVERSION_README.md) - Skeleton转换详细指南
- [skeleton_config.json](skeleton_config.json) - 17关节定义

### 可视化阶段
- [QUICK_START_SKELETON.md](QUICK_START_SKELETON.md) - 快速入门
- [SKELETON_PROJECTION_GUIDE.md](SKELETON_PROJECTION_GUIDE.md) - 投影技术文档
- [MARKER_PROJECTION_GUIDE.md](MARKER_PROJECTION_GUIDE.md) - 投影原理说明

## 🛠️ 环境要求

### Python环境
```bash
conda activate multical
python --version  # 3.10+
```

### 必需库
```bash
pip install pandas numpy plotly dash opencv-python tqdm matplotlib
```

### 可选工具
```bash
# FFmpeg（生成视频）
conda install -c conda-forge ffmpeg

# 视频播放
brew install ffmpeg  # macOS
```

## 🐛 常见问题

### Q: 标注工具无法启动
**A**: 检查是否安装了dash：`pip install dash`

### Q: Skeleton转换显示missing markers
**A**: 这是正常的！脚本会跳过缺失的关节继续处理。

### Q: 3D动画生成失败
**A**: 确保安装了matplotlib：`pip install matplotlib`

### Q: 视频投影看不到骨架
**A**: 检查：
1. `.mcal` 文件路径正确
2. 视频分辨率与标定匹配（1920x1080）
3. skeleton_joints.json包含有效数据

### Q: 投影位置偏移
**A**: 确保使用了negative fx（脚本已内置，无需手动修改）

## 🎓 学习路径

### 初学者
1. 阅读 [QUICK_START_SKELETON.md](QUICK_START_SKELETON.md)
2. 运行快速测试（100帧）
3. 查看 [MARKER_ANNOTATION_README.md](MARKER_ANNOTATION_README.md)

### 进阶用户
1. 理解 [SKELETON_CONVERSION_README.md](SKELETON_CONVERSION_README.md)
2. 自定义 [skeleton_config.json](skeleton_config.json)
3. 阅读 [SKELETON_PROJECTION_GUIDE.md](SKELETON_PROJECTION_GUIDE.md)

### 专家用户
1. 研究 [MARKER_PROJECTION_GUIDE.md](MARKER_PROJECTION_GUIDE.md)
2. 修改投影参数和可视化样式
3. 集成到自己的pipeline

## 📝 版本历史

- **v1.0** (2025-10-23): 初始版本
  - 交互式marker标注工具
  - Markers到Skeleton转换
  - 3D动画生成
  - 视频投影功能

---

**维护者**: Annotation Pipeline Team
**最后更新**: 2025-10-23
