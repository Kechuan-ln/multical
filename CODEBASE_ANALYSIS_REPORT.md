# 代码库详细分析报告

**生成日期**: 2025-10-31
**分析范围**: /Volumes/FastACIS/annotation_pipeline/

---

## 📋 执行摘要

本代码库是一个**多功能的多相机3D人体姿态标注系统**，经过多次迭代开发，目前包含三个主要功能域：

1. **核心Pipeline**: 基于ChArUco的多相机标定、人体姿态检测与标注（原始功能）
2. **GoPro-PrimeColor混合系统**: QR码同步 + 混合相机系统外参标定（新增功能）
3. **动捕数据处理**: OptiTrack marker标注、骨架转换、3D-2D投影、假肢建模（实验性功能）

**主要问题**：
- ✅ 功能丰富但组织混乱
- ⚠️ 存在大量重复和迭代版本的文件
- ⚠️ 缺乏清晰的版本管理和废弃标记
- ⚠️ 文档分散，缺乏统一的入口文档

---

## 🗂️ 代码库结构概览

### 核心目录结构

```
annotation_pipeline/
├── scripts/                    # 核心pipeline脚本（人体姿态标注）
│   ├── sync_timecode.py       # GoPro timecode同步
│   ├── convert_video_to_images.py
│   ├── run_yolo_tracking.py   # 人体检测
│   ├── run_vitpose.py         # 2D姿态估计
│   ├── run_triangulation.py   # 3D三角化
│   └── run_refinement.py      # 姿态精化
│
├── multical/                   # ChArUco标定子模块
│   ├── intrinsic.py           # 内参标定
│   ├── calibrate.py           # 外参标定
│   └── asset/                 # 标定板配置
│
├── utils/                      # 工具函数
│   ├── calib_utils.py         # 相机标定工具（timecode同步）
│   ├── io_utils.py            # 视频/图像IO
│   └── constants.py           # 全局配置
│
├── [根目录]/                   # 混合系统和实验性功能（45个Python文件）
│   ├── **GoPro-PrimeColor标定** (5个文件)
│   ├── **QR码同步** (6个文件)
│   ├── **Marker/Skeleton处理** (18个文件)
│   ├── **投影工具** (13个版本)
│   └── **辅助工具** (多个)
│
└── *.md                        # 分散的文档（20+个MD文件）
```

---

## 🔄 功能域分析

### 1️⃣ 核心Pipeline - 人体姿态标注系统

**目的**: 从多相机视频生成高质量3D人体关节标注（COCO 17关节）

**工作流程**:
```
视频 → Timecode同步 → 帧提取 → YOLO检测 → ViTPose 2D → 三角化 → EgoHumans精化 → 人工审核
```

**关键文件** (位于 `scripts/`):
- ✅ **活跃使用**
- `sync_timecode.py` - GoPro硬件timecode同步
- `convert_video_to_images.py` - 视频转图像
- `run_yolo_tracking.py` - ByteTrack人体跟踪
- `run_vitpose.py` - ViTPose 2D关节检测
- `run_triangulation.py` - DLT三角化
- `run_refinement.py` - 时空一致性精化
- `tool_*.py` - Gradio标注工具（bbox, 2D keypoints, 3D approval）

**输入要求**:
- GoPro相机必须支持timecode嵌入
- 预先标定的内参（`intrinsic_hyperoff_linear_60fps.json`）
- ChArUco标定板视频用于外参标定

**局限性**:
- ❌ **不支持QR码同步**（仅硬件timecode）
- ✅ 仅支持单人跟踪
- ✅ 需要手动标注修正

---

### 2️⃣ GoPro-PrimeColor混合系统

**目的**: 将消费级GoPro与专业级PrimeColor相机联合标定，用于混合多相机捕捉

**核心创新**: QR码时间同步 + 异构相机外参标定

#### 📂 关键文件分析

##### **QR码同步模块** (6个文件)

| 文件名 | 状态 | 功能 |
|--------|------|------|
| `generate_qr_sync_video.py` | ✅ 活跃 | 生成QR码anchor视频（每帧唯一编号） |
| `inspect_qr_video.py` | ✅ 活跃 | 检测QR码内容和时间戳 |
| `comprehensive_qr_matrix_test.py` | ✅ 活跃 | 测试QR检测稳定性（支持zbarlight/pyzbar） |
| `generate_qr_metadata.py` | ⚠️ 辅助 | 生成QR码元数据JSON |
| `qr_data_v4.json` | 📄 数据 | QR同步数据（已过时？） |
| `qr_sync_data.json` | 📄 数据 | QR同步结果 |

**工作原理**:
```
1. Anchor视频生成: 电脑播放QR码视频（每帧编号0-9999）
2. 多相机录制: GoPro + PrimeColor同时录制屏幕QR码
3. QR检测: 提取各相机看到的QR码序列
4. 时间对齐: 找到公共QR码区间，计算offset
```

**检测算法**: 支持多种检测器（OpenCV, pyzbar, zbarlight）

##### **GoPro-PrimeColor标定** (5个文件)

| 文件名 | 状态 | 功能 | 最后修改 |
|--------|------|------|----------|
| `run_gopro_primecolor_calibration.py` | ✅ **主控脚本** | 完整标定工作流程 | 2025-10-29 |
| `calibrate_gopro_primecolor_extrinsics.py` | ✅ 核心 | 外参标定（含CLAHE增强） | 2025-10-29 |
| `calibrate_primecolor_intrinsics.py` | ✅ 活跃 | PrimeColor单独内参标定 | 2025-10-07 |
| `run_gopro_calibration.py` | ⚠️ 仅GoPro | GoPro单独标定（可能废弃） | 2025-10-28 |
| `extract_tak_calibration.py` | ⚠️ 辅助 | TAK文件解析（OptiTrack） | 2025-10-22 |

**主工作流程** (`run_gopro_primecolor_calibration.py`):
```python
# 配置文件驱动的完整流程
WORKING_DIR = "/path/to/data/"
GOPRO_VIDEO = "Cam4/Video.MP4"      # 包含QR码+标定板
PRIMECOLOR_VIDEO = "Primecolor/Video.avi"
QR_ANCHOR = "Anchor.mp4"

# 时间分段
0-30s:  QR码同步
30-180s: ChArUco标定板外参标定
```

**关键技术**:
1. **CLAHE增强**: PrimeColor暗图像增强（检测率从25%提升到91%）
2. **内参合并**: 自动合并GoPro JSON + PrimeColor .mcal
3. **QR同步**: 替代硬件timecode，支持异构相机
4. **Multical集成**: 使用`--fix_intrinsic`锁定内参，仅优化外参

**输入文件**:
- `Intrinsic-16.json` - GoPro预标定内参
- `Primecolor.mcal` - PrimeColor Motive导出文件
- `Anchor.mp4` - QR码参考视频
- 录制视频 - 包含QR码+标定板

**输出**:
- `merged_intrinsics.json` - 合并的内参
- `calibration.json` - Multical外参结果
- `extrinsics_calibrated.json` - 最终外参

##### **辅助标定工具**

| 文件名 | 状态 | 功能 |
|--------|------|------|
| `comprehensive_calibration_test.py` | ✅ 测试 | 标定质量验证（重投影误差、FOV） |
| `quick_test_calibration_fix.py` | ⚠️ 调试 | 快速测试标定修复 |
| `merge_calibrations_for_cam4.py` | ⚠️ 一次性 | 合并Cam4标定（特定任务） |
| `merge_intrinsics.py` | ✅ 工具 | 通用内参合并 |
| `filter_intrinsics.py` | ✅ 工具 | 内参清理和筛选 |
| `inspect_mcal_c11764.py` | ⚠️ 调试 | 检查特定.mcal文件 |
| `enhance_dark_images.py` | ✅ 工具 | 批量CLAHE图像增强 |

---

### 3️⃣ 动捕数据处理系统

**目的**: 处理OptiTrack动捕数据，转换为骨架，投影到视频，支持假肢建模

#### 📂 Marker标注与骨架转换 (6个文件)

| 文件名 | 状态 | 功能 | 最后修改 |
|--------|------|------|----------|
| `annotate_mocap_markers.py` | ✅ 活跃 | 基础marker标注（Gradio 3D+表格） | 2025-10-23 |
| `annotate_mocap_markers_2d3d.py` | ✅ **推荐** | 2D+3D同步标注（Dash）| 2025-10-28 |
| `markers_to_skeleton.py` | ✅ 活跃 | Markers→H36M骨架（17关节） | - |
| `markers_to_skeleton_with_prosthesis.py` | ✅ 活跃 | 骨架+假肢刚体变换 | 2025-10-28 |
| `annotate_prosthesis_points.py` | ✅ 活跃 | 假肢anchor点标注 | 2025-10-28 |
| `annotate_extrinsics_interactive.py` | ⚠️ 实验 | 交互式外参标定 | 2025-10-28 |

**工作流程**:
```
1. OptiTrack导出 mocap.csv
2. annotate_mocap_markers_2d3d.py → 标注marker名称 → marker_labels.csv
3. markers_to_skeleton.py → 计算17关节 → skeleton_joints.json
4. [可选] markers_to_skeleton_with_prosthesis.py → 添加假肢 → skeleton_with_prosthesis.json
```

**配置文件**:
- `skeleton_config.json` - 定义17关节计算公式（例如：Hip = (LHip + RHip) / 2）
- `prosthesis_config.json` - 假肢anchor点和CAD模型
- `marker_labels.csv` - marker ID到名称映射
- `Genesis.STL` - 假肢3D模型（37KB）

**关节定义** (H36M格式):
```
0=Pelvis, 1=RHip, 2=RKnee, 3=RAnkle, 4=LHip, 5=LKnee, 6=LAnkle,
7=Spine, 8=Thorax, 9=Neck, 10=Head, 11=LShoulder, 12=LElbow,
13=LWrist, 14=RShoulder, 15=RElbow, 16=RWrist
```

#### 📂 投影工具 - **问题严重区域** ⚠️

**发现**: 存在**13个版本的投影脚本**，功能重叠严重

##### **Skeleton投影** (5个版本)

| 文件名 | 状态 | 推荐 | 最后修改 | 说明 |
|--------|------|------|----------|------|
| `project_skeleton_to_video.py` | ✅ 通用 | ⭐⭐⭐ | 2025-10-28 | PrimeColor→PrimeColor投影 |
| `project_skeleton_with_prosthesis.py` | ✅ 假肢 | ⭐⭐⭐ | 2025-10-28 | 骨架+假肢CAD模型投影 |
| `project_skeleton_to_gopro.py` | ⚠️ 中间版本 | - | 2025-10-29 | OptiTrack→GoPro（旧版） |
| `project_skeleton_to_gopro_continuous.py` | ⚠️ 中间版本 | - | 2025-10-29 | 连续帧处理 |
| `project_skeleton_to_gopro_direct.py` | ⚠️ 中间版本 | - | 2025-10-29 | 直接投影方法 |
| `project_skeleton_to_gopro_FINAL_FIXED.py` | ✅ **最终版** | ⭐⭐⭐⭐ | 2025-10-29 | 修复坐标系问题 |

**推荐使用**:
- **PrimeColor投影**: `project_skeleton_to_video.py` 或 `project_skeleton_with_prosthesis.py`
- **GoPro投影**: `project_skeleton_to_gopro_FINAL_FIXED.py`

##### **Marker投影** (8个版本) - **混乱严重**

| 文件名 | 状态 | 推荐 | 最后修改 | 说明 |
|--------|------|------|----------|------|
| `project_markers_final.py` | ✅ **推荐** | ⭐⭐⭐⭐ | 2025-10-28 | 最终版本 |
| `project_markers_dual_video.py` | ✅ 双视频 | ⭐⭐⭐ | 2025-10-29 | 并排对比 |
| `project_markers_to_video_v2.py` | ✅ V2 | ⭐⭐⭐ | 2025-10-29 | 改进版 |
| `project_markers_new_extrinsics.py` | ⚠️ 测试 | - | 2025-10-28 | 测试新外参 |
| `project_markers_to_gopro.py` | ⚠️ GoPro | - | 2025-10-29 | GoPro特定 |
| `project_markers_to_video.py` | ⚠️ 旧版 | ❌ | 2025-10-23 | 已被v2替代 |
| `sync_and_project_markers.py` | ⚠️ 实验 | - | - | 同步+投影 |
| `correct_projection.py` | ⚠️ 调试 | - | 2025-10-23 | 投影修正 |

**关键差异**:
1. **坐标系处理**: OptiTrack使用`-Z`轴朝前，需要negative fx补偿
2. **畸变处理**: 某些版本使用undistorted points
3. **视频源**: GoPro vs PrimeColor有不同的投影逻辑

#### 📂 可视化工具

| 文件名 | 功能 |
|--------|------|
| `create_stacked_video.py` | 多相机网格/水平堆叠视频 |
| `generate_skeleton_gif.py` | 骨架运动GIF动画 |
| `process_and_animate.py` | 骨架处理和动画 |
| `mocap_visualization.html` | HTML 3D可视化 |

---

## 📊 文件状态总结

### ✅ 活跃使用 (推荐保留)

**核心Pipeline** (scripts/):
- `sync_timecode.py`
- `convert_video_to_images.py`
- `run_yolo_tracking.py`
- `run_vitpose.py`
- `run_triangulation.py`
- `run_refinement.py`
- `tool_*.py` (所有Gradio工具)

**GoPro-PrimeColor系统**:
- `run_gopro_primecolor_calibration.py` ⭐
- `calibrate_gopro_primecolor_extrinsics.py`
- `generate_qr_sync_video.py`
- `inspect_qr_video.py`
- `comprehensive_qr_matrix_test.py`

**Marker/Skeleton系统**:
- `annotate_mocap_markers_2d3d.py` ⭐
- `markers_to_skeleton.py`
- `markers_to_skeleton_with_prosthesis.py`
- `project_markers_final.py` ⭐
- `project_skeleton_to_gopro_FINAL_FIXED.py` ⭐
- `project_skeleton_with_prosthesis.py`

**工具**:
- `merge_intrinsics.py`
- `filter_intrinsics.py`
- `enhance_dark_images.py`
- `create_stacked_video.py`

### ⚠️ 中间版本 (建议归档)

**Projection迭代版本** (保留1-2个最终版，归档其余):
- `project_skeleton_to_gopro.py` → 已被FINAL_FIXED替代
- `project_skeleton_to_gopro_continuous.py` → 中间版本
- `project_skeleton_to_gopro_direct.py` → 中间版本
- `project_markers_to_video.py` → 已被v2替代
- `project_markers_new_extrinsics.py` → 测试脚本

**单次使用脚本**:
- `merge_calibrations_for_cam4.py` - 特定任务
- `quick_test_calibration_fix.py` - 调试
- `inspect_mcal_c11764.py` - 调试特定相机
- `fix_mirror.py` - 一次性修复

### ❓ 状态不明 (需要用户确认)

| 文件名 | 问题 |
|--------|------|
| `run_gopro_calibration.py` | 是否还需要GoPro单独标定？ |
| `run_calibration.py` | 功能不明确 |
| `run_calibration_directly.py` | 与上面有何区别？ |
| `sync_and_project_markers.py` | 是否完成？ |
| `correct_projection.py` | 是否解决了问题？ |
| `generate_sync_tests.py` | 测试完成了吗？ |

---

## 📚 文档状态分析

### 现有文档 (20+ MD文件)

**核心文档**:
- ✅ `CLAUDE.md` - 主文档（已更新，内容完善）
- ✅ `README.md` - 原始README（核心pipeline）

**功能域文档**:
- ✅ `README_GOPRO_PRIMECOLOR.md` - GoPro-PrimeColor完整指南
- ✅ `SKELETON_CONVERSION_README.md` - Marker→骨架转换
- ✅ `MARKER_PROJECTION_GUIDE.md` - 投影技术细节
- ✅ `MARKER_ANNOTATION_2D3D_README.md` - 标注工具使用
- ✅ `GOPRO_CALIBRATION_README.md` - GoPro标定

**状态报告**:
- `CALIBRATION_SUCCESS_REPORT.md` - 标定成功案例
- `CALIBRATION_ANALYSIS_SUMMARY.md` - 标定分析
- `BINARY_PARSING_SUMMARY.md` - 二进制文件解析
- `TAK_FILE_PROCESSING_SUMMARY.md` - TAK文件处理

**待整理**:
- `MODIFICATIONS_APPLIED.md` - 修改记录
- `MULTI_CAMERA_WORKFLOW.md` - 多相机工作流
- `VIDEO_PROJECTION_GUIDE.md` - 视频投影
- `MOTIVE_API_SOLUTION.md` - Motive API
- `投影流程说明.md` - 中文投影说明

### 📌 文档问题

1. **缺乏入口文档**: 用户不知道从哪里开始
2. **功能域分离**: 三个功能域文档分散
3. **重复内容**: 多个文档描述类似流程
4. **版本不一致**: 某些文档可能过时

---

## 🚨 主要问题与风险

### 1. 文件版本管理混乱 ⚠️⚠️⚠️

**问题**:
- 13个投影脚本版本，无明确标记哪个是最终版
- 文件名使用`_FINAL_FIXED`、`_v2`等临时命名
- 缺乏Git tag或release管理

**风险**:
- 用户不知道使用哪个版本
- 可能使用过时的脚本产生错误结果
- 代码难以维护

**建议**: 见下方"建议措施"

### 2. 功能域职责不清 ⚠️⚠️

**问题**:
- 根目录混合了三个功能域（45个Python文件）
- `scripts/`目录仅包含原始pipeline
- QR同步功能分散在多个文件

**建议**: 重新组织目录结构

### 3. 配置管理分散 ⚠️

**问题**:
- 配置在多个地方：`constants.py`、各脚本头部、JSON文件
- 路径硬编码（如`/Volumes/FastACIS/...`）
- 缺乏统一的配置管理

### 4. 废弃代码未清理 ⚠️

**问题**:
- 中间版本文件保留在主目录
- 测试脚本未标记
- 一次性脚本未移除

---

## 💡 建议措施

### 🔧 立即行动 (高优先级)

#### 1. 清理投影脚本版本

**建议目录结构**:
```
scripts/
└── projection/
    ├── project_skeleton_to_primecolor.py  # 重命名自 project_skeleton_to_video.py
    ├── project_skeleton_to_gopro.py       # 重命名自 project_skeleton_to_gopro_FINAL_FIXED.py
    ├── project_skeleton_with_prosthesis.py
    ├── project_markers.py                 # 重命名自 project_markers_final.py
    ├── project_markers_dual_video.py
    └── legacy/                            # 归档旧版本
        ├── project_skeleton_to_gopro_v1.py
        ├── project_skeleton_to_gopro_continuous.py
        ├── project_markers_to_video_v1.py
        └── ...
```

**行动**:
```bash
# 1. 创建legacy目录
mkdir -p scripts/projection/legacy

# 2. 移动最终版本
mv project_skeleton_to_gopro_FINAL_FIXED.py scripts/projection/project_skeleton_to_gopro.py
mv project_markers_final.py scripts/projection/project_markers.py

# 3. 归档旧版本
mv project_skeleton_to_gopro.py scripts/projection/legacy/
mv project_skeleton_to_gopro_continuous.py scripts/projection/legacy/
mv project_skeleton_to_gopro_direct.py scripts/projection/legacy/
mv project_markers_to_video.py scripts/projection/legacy/
```

#### 2. 重组根目录

**建议结构**:
```
annotation_pipeline/
├── scripts/                    # 核心pipeline（保持不变）
├── multical/                   # 标定子模块（保持不变）
├── utils/                      # 工具函数（保持不变）
│
├── gopro_primecolor/           # 【新建】混合系统
│   ├── calibration/
│   │   ├── run_gopro_primecolor_calibration.py  # 主脚本
│   │   ├── calibrate_extrinsics.py
│   │   ├── calibrate_primecolor_intrinsics.py
│   │   └── merge_intrinsics.py
│   ├── synchronization/
│   │   ├── generate_qr_sync_video.py
│   │   ├── inspect_qr_video.py
│   │   └── comprehensive_qr_matrix_test.py
│   └── docs/
│       └── README_GOPRO_PRIMECOLOR.md
│
├── mocap/                      # 【新建】动捕处理
│   ├── annotation/
│   │   ├── annotate_mocap_markers_2d3d.py
│   │   ├── annotate_prosthesis_points.py
│   │   └── configs/
│   │       ├── skeleton_config.json
│   │       └── prosthesis_config.json
│   ├── conversion/
│   │   ├── markers_to_skeleton.py
│   │   └── markers_to_skeleton_with_prosthesis.py
│   ├── projection/
│   │   ├── project_skeleton_to_primecolor.py
│   │   ├── project_skeleton_to_gopro.py
│   │   ├── project_skeleton_with_prosthesis.py
│   │   ├── project_markers.py
│   │   └── project_markers_dual_video.py
│   ├── assets/
│   │   └── Genesis.STL
│   └── docs/
│       ├── SKELETON_CONVERSION_README.md
│       └── MARKER_PROJECTION_GUIDE.md
│
├── tools/                      # 【新建】通用工具
│   ├── create_stacked_video.py
│   ├── enhance_dark_images.py
│   ├── filter_intrinsics.py
│   └── generate_skeleton_gif.py
│
├── legacy/                     # 【新建】废弃代码
│   ├── calibration/
│   │   ├── run_gopro_calibration.py
│   │   ├── quick_test_calibration_fix.py
│   │   └── merge_calibrations_for_cam4.py
│   ├── projection/
│   │   └── [旧版本投影脚本]
│   └── debug/
│       ├── inspect_mcal_c11764.py
│       ├── correct_projection.py
│       └── fix_mirror.py
│
└── docs/                       # 【新建】统一文档
    ├── README.md               # 主入口文档
    ├── GETTING_STARTED.md      # 快速开始
    ├── CLAUDE.md               # AI助手文档
    ├── core_pipeline/
    ├── gopro_primecolor/
    ├── mocap_processing/
    └── archive/                # 归档旧文档
```

#### 3. 创建统一入口文档

**新建 `docs/README.md`**:
```markdown
# Annotation Pipeline - 统一文档入口

## 🎯 选择你的工作流程

### 1️⃣ 人体姿态标注 (核心Pipeline)
**用途**: 从多相机GoPro视频生成3D人体关节标注

→ [核心Pipeline文档](core_pipeline/README.md)
→ [快速开始](core_pipeline/QUICK_START.md)

**特点**:
- ✅ GoPro timecode硬件同步
- ✅ ChArUco标定板
- ✅ YOLO + ViTPose + 三角化
- ❌ 不支持QR码同步

---

### 2️⃣ GoPro + PrimeColor 混合系统
**用途**: 消费级GoPro + 专业PrimeColor联合标定

→ [GoPro-PrimeColor文档](gopro_primecolor/README_GOPRO_PRIMECOLOR.md)
→ [QR码同步指南](gopro_primecolor/QR_SYNC_GUIDE.md)

**特点**:
- ✅ QR码时间同步（无需硬件timecode）
- ✅ 异构相机外参标定
- ✅ CLAHE暗图像增强
- ✅ 一站式标定脚本

---

### 3️⃣ OptiTrack动捕处理
**用途**: OptiTrack marker标注、骨架转换、视频投影

→ [动捕处理文档](mocap/README.md)
→ [骨架转换指南](mocap/SKELETON_CONVERSION_README.md)
→ [投影技术](mocap/MARKER_PROJECTION_GUIDE.md)

**特点**:
- ✅ 2D+3D同步标注
- ✅ H36M 17关节骨架
- ✅ 假肢建模支持
- ✅ OptiTrack → GoPro/PrimeColor投影

---

## 📖 其他资源

- [CLAUDE.md](CLAUDE.md) - Claude AI助手专用文档
- [API文档](api/) - Python API参考
- [FAQ](FAQ.md) - 常见问题
```

#### 4. 添加文件状态标记

**在每个脚本头部添加**:
```python
"""
Script: project_skeleton_to_gopro.py
Status: ACTIVE | STABLE | RECOMMENDED
Last Updated: 2025-10-29
Replaces: project_skeleton_to_gopro_FINAL_FIXED.py

Purpose:
    Project 3D skeleton from OptiTrack coordinate system to GoPro video frames.

Usage:
    python project_skeleton_to_gopro.py --skeleton skeleton.json --video video.mp4 ...

See Also:
    - docs/mocap/SKELETON_PROJECTION.md
    - project_skeleton_with_prosthesis.py (if using prosthesis)
"""
```

**为废弃文件添加**:
```python
"""
⚠️ DEPRECATED - DO NOT USE

This script has been replaced by: project_skeleton_to_gopro.py

Reason: Coordinate system fix, better error handling
Date Deprecated: 2025-10-29
Will Be Removed: 2025-12-01

For migration guide, see: docs/migration/PROJECTION_V2.md
"""
```

### 📝 短期改进 (中优先级)

#### 5. 配置文件统一

**创建 `config/` 目录**:
```
config/
├── paths.yaml              # 路径配置
├── cameras/
│   ├── gopro_intrinsics.json
│   ├── primecolor_intrinsics.json
│   └── mixed_system.json
├── calibration/
│   ├── charuco_boards.yaml
│   └── calibration_params.yaml
├── mocap/
│   ├── skeleton_config.json
│   └── prosthesis_config.json
└── defaults.yaml           # 默认参数
```

**好处**:
- 集中管理配置
- 方便切换数据集
- 避免硬编码路径

#### 6. Git标签管理

```bash
# 为当前稳定版本打标签
git tag -a v1.0.0 -m "Stable release: Core pipeline + GoPro-PrimeColor + Mocap"

# 为功能模块打标签
git tag -a gopro-primecolor-v1.0 -m "GoPro-PrimeColor calibration stable"
git tag -a mocap-processing-v1.0 -m "Mocap processing stable"
```

#### 7. 添加单元测试

```
tests/
├── test_calibration.py
├── test_qr_sync.py
├── test_skeleton_conversion.py
└── test_projection.py
```

### 📚 长期规划 (低优先级)

#### 8. 代码模块化

- 提取公共函数到`utils/`
- 创建Python包结构
- 添加`setup.py`/`pyproject.toml`

#### 9. Web界面

- 统一的Gradio/Streamlit界面
- 工作流程可视化
- 进度跟踪

#### 10. Docker化

```dockerfile
# 支持一键部署
docker-compose up
```

---

## 📋 文件清单与建议操作

### 保留（移动到新位置）

**GoPro-PrimeColor** → `gopro_primecolor/`:
```
✅ run_gopro_primecolor_calibration.py  → calibration/
✅ calibrate_gopro_primecolor_extrinsics.py → calibration/
✅ calibrate_primecolor_intrinsics.py → calibration/
✅ generate_qr_sync_video.py → synchronization/
✅ inspect_qr_video.py → synchronization/
✅ comprehensive_qr_matrix_test.py → synchronization/
✅ merge_intrinsics.py → calibration/
✅ filter_intrinsics.py → calibration/
✅ enhance_dark_images.py → tools/
```

**Mocap处理** → `mocap/`:
```
✅ annotate_mocap_markers_2d3d.py → annotation/
✅ annotate_prosthesis_points.py → annotation/
✅ markers_to_skeleton.py → conversion/
✅ markers_to_skeleton_with_prosthesis.py → conversion/
✅ project_skeleton_to_gopro_FINAL_FIXED.py → projection/ (重命名)
✅ project_skeleton_to_video.py → projection/
✅ project_skeleton_with_prosthesis.py → projection/
✅ project_markers_final.py → projection/ (重命名)
✅ project_markers_dual_video.py → projection/
✅ project_markers_to_video_v2.py → projection/
```

**工具** → `tools/`:
```
✅ create_stacked_video.py
✅ generate_skeleton_gif.py
✅ process_and_animate.py
✅ comprehensive_calibration_test.py
```

### 归档到 `legacy/`

**调试脚本**:
```
⚠️ inspect_mcal_c11764.py → legacy/debug/
⚠️ correct_projection.py → legacy/debug/
⚠️ fix_mirror.py → legacy/debug/
⚠️ quick_test_calibration_fix.py → legacy/debug/
```

**一次性任务**:
```
⚠️ merge_calibrations_for_cam4.py → legacy/tasks/
⚠️ generate_sync_tests.py → legacy/tasks/
```

**旧版本投影**:
```
⚠️ project_skeleton_to_gopro.py → legacy/projection/
⚠️ project_skeleton_to_gopro_continuous.py → legacy/projection/
⚠️ project_skeleton_to_gopro_direct.py → legacy/projection/
⚠️ project_markers_to_video.py → legacy/projection/
⚠️ project_markers_new_extrinsics.py → legacy/projection/
⚠️ project_markers_to_gopro.py → legacy/projection/
⚠️ sync_and_project_markers.py → legacy/projection/
```

**过时标定**:
```
⚠️ run_gopro_calibration.py → legacy/calibration/
⚠️ run_calibration.py → legacy/calibration/
⚠️ run_calibration_directly.py → legacy/calibration/
```

**实验性**:
```
⚠️ annotate_extrinsics_interactive.py → legacy/experimental/
⚠️ annotate_mocap_markers.py → legacy/experimental/ (被2d3d版本替代)
```

### 删除（或确认后删除）

**TAK文件** (用户确认):
```
❓ explore_tak_file.py
❓ extract_tak_calibration.py
→ 如果TAK格式不再使用，可删除
```

**QR测试数据**:
```
❓ qr_data_v4.json (过时？)
❓ qr_detections.json (测试数据？)
→ 确认是否是临时测试文件
```

**其他**:
```
❓ generate_qr_metadata.py (是否已完成功能？)
```

---

## 🎯 总结与优先级

### 核心问题 (按严重程度排序)

1. **🔴 高**: 投影脚本版本混乱（13个版本，用户难以选择）
2. **🟡 中**: 根目录文件过多（45个Python文件）
3. **🟡 中**: 文档分散，缺乏入口
4. **🟢 低**: 配置管理分散
5. **🟢 低**: 缺乏自动化测试

### 推荐行动优先级

#### 第1周：立即清理
- ✅ 标记所有文件状态（ACTIVE/DEPRECATED）
- ✅ 重命名最终版本（去掉_FINAL_FIXED后缀）
- ✅ 移动旧版本到`legacy/`
- ✅ 创建`docs/README.md`入口文档

#### 第2-3周：重组结构
- ✅ 创建`gopro_primecolor/`、`mocap/`、`tools/`目录
- ✅ 移动文件到新位置
- ✅ 更新所有import路径
- ✅ 测试各工作流程

#### 第4周：文档完善
- ✅ 整合分散的MD文档
- ✅ 编写GETTING_STARTED.md
- ✅ 更新CLAUDE.md
- ✅ 归档过时文档

#### 后续：持续改进
- 统一配置文件
- 添加单元测试
- Git标签管理
- 代码模块化

---

## 📞 需要用户确认的问题

1. **TAK文件处理**: `extract_tak_calibration.py`等TAK相关文件是否还需要？OptiTrack现在使用.mcal格式。

2. **GoPro单独标定**: `run_gopro_calibration.py`是否还需要？现在主要使用混合标定。

3. **QR同步数据**: `qr_data_v4.json`、`qr_detections.json`是测试数据还是需要保留的配置？

4. **标定测试结果**: `calibration_test_results_20251029_165958.json`等JSON文件是否需要保留？

5. **中间投影版本**: 是否需要保留投影脚本的演化历史（用于教学/参考），还是直接删除？

6. **Marker标注工具**: `annotate_mocap_markers.py`（旧版）是否可以完全被`annotate_mocap_markers_2d3d.py`替代？

7. **同步debug数据**: `sync_debug.json`、`sync_result.json`、`qr_sync_data.json`是否是临时文件？

---

## 附录：完整文件列表

### 根目录Python文件（45个）

按功能分类：

**GoPro-PrimeColor标定 (5)**:
- run_gopro_primecolor_calibration.py ✅
- calibrate_gopro_primecolor_extrinsics.py ✅
- calibrate_primecolor_intrinsics.py ✅
- run_gopro_calibration.py ⚠️
- extract_tak_calibration.py ⚠️

**QR同步 (6)**:
- generate_qr_sync_video.py ✅
- inspect_qr_video.py ✅
- comprehensive_qr_matrix_test.py ✅
- generate_qr_metadata.py ❓
- generate_sync_tests.py ⚠️
- sync_and_project_markers.py ⚠️

**Marker/Skeleton标注 (6)**:
- annotate_mocap_markers.py ⚠️
- annotate_mocap_markers_2d3d.py ✅
- annotate_prosthesis_points.py ✅
- annotate_extrinsics_interactive.py ⚠️
- markers_to_skeleton.py ✅
- markers_to_skeleton_with_prosthesis.py ✅

**Skeleton投影 (5)**:
- project_skeleton_to_video.py ✅
- project_skeleton_with_prosthesis.py ✅
- project_skeleton_to_gopro.py ⚠️
- project_skeleton_to_gopro_continuous.py ⚠️
- project_skeleton_to_gopro_direct.py ⚠️
- project_skeleton_to_gopro_FINAL_FIXED.py ✅

**Marker投影 (8)**:
- project_markers_final.py ✅
- project_markers_dual_video.py ✅
- project_markers_to_video_v2.py ✅
- project_markers_to_video.py ⚠️
- project_markers_new_extrinsics.py ⚠️
- project_markers_to_gopro.py ⚠️
- correct_projection.py ⚠️

**辅助工具 (9)**:
- comprehensive_calibration_test.py ✅
- quick_test_calibration_fix.py ⚠️
- merge_calibrations_for_cam4.py ⚠️
- merge_intrinsics.py ✅
- filter_intrinsics.py ✅
- enhance_dark_images.py ✅
- create_stacked_video.py ✅
- generate_skeleton_gif.py ✅
- process_and_animate.py ✅

**调试 (5)**:
- inspect_mcal_c11764.py ⚠️
- explore_tak_file.py ❓
- fix_mirror.py ⚠️
- run_calibration.py ❓
- run_calibration_directly.py ❓

**符号说明**:
- ✅ 活跃使用，推荐保留
- ⚠️ 中间版本/调试工具，建议归档
- ❓ 状态不明，需要用户确认

---

**报告结束**

如有疑问或需要详细分析某个具体模块，请告知。
