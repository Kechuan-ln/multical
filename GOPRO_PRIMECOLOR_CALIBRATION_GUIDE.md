# GoPro + PrimeColor 同步和外参标定完整指南

本指南详细说明如何完成 GoPro 和 PrimeColor 相机的时间同步和外参标定流程。

## 概述

**目标**：
1. 使用 QR 码实现 GoPro 和 PrimeColor 相机的时间同步
2. 计算两个相机系统的外参（相对位置和姿态）

**前提条件**：
- 已有 GoPro 相机的内参文件（如 `intrinsic_hyperoff_linear_60fps.json`）
- 已有 PrimeColor 相机的 .mcal 标定文件
- 两个相机均录制了 QR 码同步视频（前30秒）
- 两个相机均录制了 ChArUco 标定板视频（用于外参计算）

## 工作流程

### 步骤 0: 准备工作

#### 0.1 确认文件位置

需要准备以下文件：

```bash
# GoPro 内参（已有）
/Volumes/FastACIS/annotation_pipeline/intrinsic_hyperoff_linear_60fps.json

# PrimeColor .mcal 文件（需要提供路径）
/path/to/primecolor/calibration.mcal

# QR 码 anchor 视频（或生成）
/path/to/qr_anchor.mp4

# GoPro 录制的 QR 码视频
/path/to/gopro_qr.mp4

# PrimeColor 录制的 QR 码视频
/path/to/primecolor_qr.mp4

# GoPro 录制的标定板视频
/path/to/gopro_charuco.mp4

# PrimeColor 录制的标定板视频
/path/to/primecolor_charuco.mp4
```

#### 0.2 创建工作目录

```bash
cd /Volumes/FastACIS/annotation_pipeline

# 创建输出目录
mkdir -p calibration_output/gopro_primecolor/{sync,intrinsics,extrinsics}
```

---

### 步骤 1: 提取 PrimeColor 内参（从 .mcal 文件）

**说明**：使用 `parse_optitrack_cal.py` 从 OptiTrack/Motive 导出的 .mcal 文件中提取内参。

#### 1.1 解析 .mcal 文件

```bash
# 解析 .mcal 文件，提取所有相机信息
python parse_optitrack_cal.py \
  /path/to/calibration.mcal \
  --output calibration_output/gopro_primecolor/intrinsics/primecolor_intrinsic.json
```

**输出**：
- `optitrack_cameras.json` - 所有相机的原始信息
- `primecolor_intrinsic.json` - Pipeline 兼容格式

#### 1.2 验证提取的内参

```bash
# 查看提取的内参
cat calibration_output/gopro_primecolor/intrinsics/primecolor_intrinsic.json | python -m json.tool
```

**检查内容**：
- `cameras` 字段：包含 K 矩阵（3x3 内参矩阵）和 dist（畸变系数）
- `camera_base2cam` 字段：包含外参（如果需要多个 PrimeColor 相机的外参）

**注意**：
- 如果 .mcal 文件中有多个相机，使用 `--camera <camera_id>` 参数选择特定相机
- 例如：`--camera Camera_13`

#### 1.3 重命名相机（如果需要）

如果提取的相机名称不是 `primecolor`，需要手动编辑 JSON 文件：

```bash
# 编辑文件，将相机名称改为 "primecolor"
# 例如：将 "Camera_13" 改为 "primecolor"
nano calibration_output/gopro_primecolor/intrinsics/primecolor_intrinsic.json
```

**预期格式**：
```json
{
  "cameras": {
    "primecolor": {
      "K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
      "dist": [k1, k2, p1, p2, k3]
    }
  }
}
```

---

### 步骤 2: 时间同步（基于 QR 码 + Anchor Video）

**说明**：使用 QR 码视频和 anchor timecode 映射来同步 GoPro 和 PrimeColor。

#### 2.1 同步 GoPro 到 Anchor

```bash
python sync_with_qr_anchor.py \
  --video1 /path/to/qr_anchor.mp4 \
  --video2 /path/to/gopro_qr.mp4 \
  --output calibration_output/gopro_primecolor/sync/gopro_synced.mp4 \
  --anchor-video /path/to/qr_anchor.mp4 \
  --scan-start 0 \
  --scan-duration 30 \
  --step 5 \
  --save-json calibration_output/gopro_primecolor/sync/gopro_sync_result.json \
  --stacked calibration_output/gopro_primecolor/sync/gopro_verify.mp4 \
  --stacked-duration 15
```

**参数说明**：
- `--video1`: Anchor 视频（参考基准）
- `--video2`: GoPro 录制的 QR 码视频
- `--output`: 同步后的 GoPro 视频输出路径
- `--anchor-video`: Anchor 视频路径（自动提取 QR 码映射）
- `--scan-start 0`: 从第 0 秒开始扫描
- `--scan-duration 30`: 扫描前 30 秒
- `--step 5`: 每 5 帧检测一次 QR 码
- `--save-json`: 保存同步结果（偏移量、统计信息）
- `--stacked`: 生成并排对比视频（用于验证同步效果）

#### 2.2 同步 PrimeColor 到 Anchor

```bash
python sync_with_qr_anchor.py \
  --video1 /path/to/qr_anchor.mp4 \
  --video2 /path/to/primecolor_qr.mp4 \
  --output calibration_output/gopro_primecolor/sync/primecolor_synced.mp4 \
  --anchor-video /path/to/qr_anchor.mp4 \
  --scan-start 0 \
  --scan-duration 30 \
  --step 5 \
  --save-json calibration_output/gopro_primecolor/sync/primecolor_sync_result.json \
  --stacked calibration_output/gopro_primecolor/sync/primecolor_verify.mp4 \
  --stacked-duration 15
```

#### 2.3 验证同步效果

```bash
# 播放并排对比视频，目视检查 QR 码是否对齐
open calibration_output/gopro_primecolor/sync/gopro_verify.mp4
open calibration_output/gopro_primecolor/sync/primecolor_verify.mp4

# 查看同步结果（偏移量）
cat calibration_output/gopro_primecolor/sync/gopro_sync_result.json | jq '.sync_result.offset_seconds'
cat calibration_output/gopro_primecolor/sync/primecolor_sync_result.json | jq '.sync_result.offset_seconds'
```

**验证标准**：
- ✅ 并排视频中 QR 码对齐
- ✅ 偏移量标准差 < 0.5 秒（`video1_offset_std`, `video2_offset_std`）
- ✅ 检测到足够的 QR 码（> 20 个）

---

### 步骤 3: 准备 GoPro 内参文件

#### 3.1 确认 GoPro 内参

```bash
# 查看 GoPro 内参文件
cat /Volumes/FastACIS/annotation_pipeline/intrinsic_hyperoff_linear_60fps.json | python -m json.tool
```

**检查**：
- 确认文件包含 `cam1`（或你使用的 GoPro 相机名称）
- 确认 K 矩阵和 dist 系数存在

#### 3.2 提取单个 GoPro 相机的内参（如果需要）

如果 GoPro 内参文件包含多个相机（cam2, cam3, cam4 等），需要选择一个相机并重命名为 `cam1`：

```bash
# 方法1: 使用 jq 提取并重命名
cat intrinsic_hyperoff_linear_60fps.json | \
  jq '{cameras: {cam1: .cameras.cam2}}' > \
  calibration_output/gopro_primecolor/intrinsics/gopro_intrinsic.json

# 方法2: 手动编辑
nano calibration_output/gopro_primecolor/intrinsics/gopro_intrinsic.json
```

**预期格式**：
```json
{
  "cameras": {
    "cam1": {
      "K": [...],
      "dist": [...]
    }
  }
}
```

---

### 步骤 4: 外参标定（GoPro + PrimeColor）

**说明**：使用同步后的标定板视频计算两个相机的外参。

#### 4.1 准备标定板视频

**选项 A**：如果标定板视频已经同步（与 QR 码视频在同一时间录制）
```bash
# 直接使用同步后的视频
# gopro_synced.mp4 和 primecolor_synced.mp4
```

**选项 B**：如果标定板视频是单独录制的（推荐）
```bash
# 使用专门录制的标定板视频（未同步）
# 将在下一步中从这些视频中提取帧
```

#### 4.2 运行外参标定

```bash
python calibrate_gopro_primecolor_extrinsics.py \
  --gopro-video /path/to/gopro_charuco.mp4 \
  --prime-video /path/to/primecolor_charuco.mp4 \
  --gopro-intrinsic calibration_output/gopro_primecolor/intrinsics/gopro_intrinsic.json \
  --prime-intrinsic calibration_output/gopro_primecolor/intrinsics/primecolor_intrinsic.json \
  --output-dir calibration_output/gopro_primecolor/extrinsics \
  --board ./multical/asset/charuco_b3.yaml \
  --fps 1.0 \
  --max-frames 100
```

**参数说明**：
- `--gopro-video`: GoPro 录制的 ChArUco 标定板视频
- `--prime-video`: PrimeColor 录制的 ChArUco 标定板视频
- `--gopro-intrinsic`: GoPro 内参文件（步骤 3 准备）
- `--prime-intrinsic`: PrimeColor 内参文件（步骤 1 提取）
- `--output-dir`: 输出目录
- `--board`: ChArUco 标定板配置文件
- `--fps 1.0`: 每秒提取 1 帧
- `--max-frames 100`: 最多提取 100 帧

**标定板配置**：
- `charuco_b3.yaml`: B3 尺寸标定板（5x9 网格，50mm 方格）
- `charuco_b1_2.yaml`: B1 尺寸标定板（10x14 网格，70mm 方格）

#### 4.3 脚本执行步骤

脚本会自动执行以下操作：

1. **提取同步帧**：
   - 从两个视频中提取对应的帧
   - 保存到 `output-dir/frames/cam1/` 和 `output-dir/frames/primecolor/`

2. **合并内参**：
   - 将 GoPro 和 PrimeColor 的内参合并到一个文件
   - 保存为 `output-dir/intrinsic_merged.json`

3. **运行 multical 外参标定**：
   - 使用 `multical/calibrate.py` 计算外参
   - 使用 `--fix_intrinsic` 锁定内参，仅优化外参
   - 生成可视化结果

#### 4.4 验证标定结果

```bash
# 查看外参标定结果
cat calibration_output/gopro_primecolor/extrinsics/frames/calibration.json | python -m json.tool

# 查看可视化结果
open calibration_output/gopro_primecolor/extrinsics/frames/vis/
```

**检查标准**：
- ✅ RMS 误差 < 1.0 像素（越小越好）
- ✅ `camera_base2cam` 包含 `cam1` 和 `primecolor` 的 R 和 T 矩阵
- ✅ 可视化图像显示 ChArUco 角点检测正确

**输出文件**：
```
calibration_output/gopro_primecolor/extrinsics/
├── frames/
│   ├── cam1/                          # GoPro 提取的帧
│   ├── primecolor/                    # PrimeColor 提取的帧
│   ├── calibration.json               # 外参标定结果（主输出）
│   └── vis/                           # 可视化结果
├── intrinsic_merged.json              # 合并的内参文件
```

---

## 最终输出文件

完成所有步骤后，你将得到以下文件：

### 1. 内参文件

```bash
# GoPro 内参
calibration_output/gopro_primecolor/intrinsics/gopro_intrinsic.json

# PrimeColor 内参
calibration_output/gopro_primecolor/intrinsics/primecolor_intrinsic.json

# 合并的内参
calibration_output/gopro_primecolor/extrinsics/intrinsic_merged.json
```

### 2. 同步结果

```bash
# GoPro 同步视频
calibration_output/gopro_primecolor/sync/gopro_synced.mp4

# PrimeColor 同步视频
calibration_output/gopro_primecolor/sync/primecolor_synced.mp4

# 同步偏移量
calibration_output/gopro_primecolor/sync/gopro_sync_result.json
calibration_output/gopro_primecolor/sync/primecolor_sync_result.json
```

### 3. 外参标定（最重要）

```bash
# 外参标定文件（包含 camera_base2cam 和完整内参）
calibration_output/gopro_primecolor/extrinsics/frames/calibration.json
```

**`calibration.json` 格式**：
```json
{
  "cameras": {
    "cam1": {
      "K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
      "dist": [k1, k2, p1, p2, k3]
    },
    "primecolor": {
      "K": [...],
      "dist": [...]
    }
  },
  "camera_base2cam": {
    "cam1": {
      "R": [r00, r01, r02, r10, r11, r12, r20, r21, r22],
      "T": [tx, ty, tz]
    },
    "primecolor": {
      "R": [...],
      "T": [...]
    }
  },
  "rms": 0.5
}
```

---

## 使用标定结果

### 在 Pipeline 中使用

将 `calibration.json` 复制到你的录制目录：

```bash
# 复制到录制目录
cp calibration_output/gopro_primecolor/extrinsics/frames/calibration.json \
   /Volumes/FastACIS/csldata/<recording_name>/original/calibration.json
```

### 进行 3D 重建

```bash
# 使用标定结果进行三角化
python scripts/run_triangulation.py \
  --recording_tag <recording_name>/original \
  --verbose
```

---

## 故障排查

### 问题 1: QR 码检测不到

**症状**：`❌ 检测到 0 个唯一QR码`

**解决方案**：
1. 检查 QR 码视频质量（是否清晰、对焦）
2. 降低扫描步长：`--step 2`（更密集检测）
3. 安装 pyzbar：`pip install pyzbar`（更快更准确）
4. 延长扫描时长：`--scan-duration 60`

### 问题 2: 偏移量标准差过大

**症状**：`Video1偏移标准差: 2.5s` 或 `Video2偏移标准差: 3.0s`

**原因**：
- QR 码视频播放速度不稳定
- 相机录制帧率不稳定
- QR 码序列不连续

**解决方案**：
1. 使用稳定的 QR 码播放设备（电脑屏幕，避免手机）
2. 增加扫描时长，获取更多样本
3. 检查 QR 码序列是否连续（查看 JSON 输出）

### 问题 3: ChArUco 角点检测失败

**症状**：`❌ 外参标定失败`，可视化图像中没有检测到角点

**原因**：
- 标定板不在视野中
- 标定板模糊或光照不足
- 标定板配置文件不匹配

**解决方案**：
1. 检查提取的帧图像（`frames/cam1/`, `frames/primecolor/`）
2. 确保标定板占据视野的 30-70%
3. 确认标定板配置与实际尺寸匹配：
   ```bash
   cat multical/asset/charuco_b3.yaml
   ```
4. 增加提取帧数：`--max-frames 200`

### 问题 4: RMS 误差过大

**症状**：`RMS误差: 5.2 像素` (> 1.0)

**原因**：
- 内参不准确
- 标定板检测质量差
- 相机移动或抖动

**解决方案**：
1. 重新标定内参（确保 RMS < 0.5）
2. 使用稳定的三脚架拍摄标定板
3. 确保标定板平整（没有弯曲）
4. 增加标定图像数量和多样性

### 问题 5: 内参文件格式错误

**症状**：`❌ GoPro内参中没有找到cam1`

**解决方案**：
1. 检查 JSON 文件格式：
   ```bash
   cat gopro_intrinsic.json | jq .cameras
   ```
2. 确保相机名称为 `cam1` 和 `primecolor`
3. 手动编辑 JSON 文件重命名相机

---

## 高级选项

### 使用不同的标定板

如果你使用的是 B1 尺寸标定板（更大）：

```bash
# 使用 B1 标定板配置
python calibrate_gopro_primecolor_extrinsics.py \
  ... \
  --board ./multical/asset/charuco_b1_2.yaml
```

### 生成自定义 QR 码 Anchor 视频

如果你需要生成 QR 码视频：

```bash
# 使用 generate_qr_sync_video.py（如果存在）
python generate_qr_sync_video.py \
  --output qr_anchor.mp4 \
  --fps 30 \
  --duration 60 \
  --prefix "SYNC-"
```

### 跳过 QR 码同步（使用手动偏移）

如果你已经知道时间偏移量：

```bash
# 直接使用 ffmpeg 裁剪视频
ffmpeg -ss 5.5 -i primecolor.mp4 -t 30 primecolor_synced.mp4
```

---

## 参考文档

- [SYNC_FIX_SUMMARY.md](SYNC_FIX_SUMMARY.md) - QR 码同步脚本修复总结
- [CLAUDE.md](CLAUDE.md) - Pipeline 完整说明
- [EXTRINSIC_CALIBRATION_GUIDE.md](EXTRINSIC_CALIBRATION_GUIDE.md) - 外参标定详细说明

---

## 快速参考：完整命令序列

```bash
# 0. 设置环境
cd /Volumes/FastACIS/annotation_pipeline
conda activate multical

# 1. 提取 PrimeColor 内参
python parse_optitrack_cal.py \
  /path/to/calibration.mcal \
  --output calibration_output/gopro_primecolor/intrinsics/primecolor_intrinsic.json

# 2. 同步 GoPro
python sync_with_qr_anchor.py \
  --video1 /path/to/qr_anchor.mp4 \
  --video2 /path/to/gopro_qr.mp4 \
  --output calibration_output/gopro_primecolor/sync/gopro_synced.mp4 \
  --anchor-video /path/to/qr_anchor.mp4 \
  --scan-duration 30 \
  --save-json calibration_output/gopro_primecolor/sync/gopro_sync_result.json

# 3. 同步 PrimeColor
python sync_with_qr_anchor.py \
  --video1 /path/to/qr_anchor.mp4 \
  --video2 /path/to/primecolor_qr.mp4 \
  --output calibration_output/gopro_primecolor/sync/primecolor_synced.mp4 \
  --anchor-video /path/to/qr_anchor.mp4 \
  --scan-duration 30 \
  --save-json calibration_output/gopro_primecolor/sync/primecolor_sync_result.json

# 4. 外参标定
python calibrate_gopro_primecolor_extrinsics.py \
  --gopro-video /path/to/gopro_charuco.mp4 \
  --prime-video /path/to/primecolor_charuco.mp4 \
  --gopro-intrinsic calibration_output/gopro_primecolor/intrinsics/gopro_intrinsic.json \
  --prime-intrinsic calibration_output/gopro_primecolor/intrinsics/primecolor_intrinsic.json \
  --output-dir calibration_output/gopro_primecolor/extrinsics \
  --board ./multical/asset/charuco_b3.yaml

# 5. 验证结果
cat calibration_output/gopro_primecolor/extrinsics/frames/calibration.json | jq .rms
```

---

**完成！** 你现在拥有了 GoPro 和 PrimeColor 相机的完整标定文件（内参 + 外参），可以用于 3D 重建和姿态估计。
