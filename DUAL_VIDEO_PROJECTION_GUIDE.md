# Dual Video Marker Projection Guide

## 概述

`project_markers_dual_video.py` 可以**同时**在 GoPro 和 PrimeColor 视频上投影 mocap markers，并生成 stacked video（堆叠视频）。

### 解决的问题

GoPro 和 PrimeColor 视频之间存在**时间偏移**（sync offset），导致使用相同 frame index 无法对齐。此工具：
- ✅ GoPro 使用 `sync-offset` 进行时间同步
- ✅ PrimeColor 直接使用 frame index 对齐
- ✅ 生成 stacked video 进行可视化对比

### 技术细节

不同的投影方法：
- **GoPro**: 坐标系转换方法 (YZ flip) + positive fx
- **PrimeColor**: Method 4 (negative fx) + 直接投影

---

## 快速开始

### 1. 基本用法

```bash
bash run_dual_video_projection.sh
```

或手动运行：

```bash
python project_markers_dual_video.py \
  --calibration /path/to/calibration.json \
  --mcal /path/to/Primecolor.mcal \
  --extrinsics-json extrinsics_calibrated.json \
  --intrinsics-json intrinsic_merged.json \
  --mocap-csv /path/to/mocap.csv \
  --gopro-video /path/to/Video.MP4 \
  --primecolor-video /path/to/primecolor.avi \
  --output dual_markers.mp4 \
  --mocap-fps 120.0 \
  --sync-offset 10.5799 \
  --start-frame 3800 \
  --num-frames 600 \
  --stack-mode horizontal
```

---

## 参数说明

### 必需参数

| 参数 | 说明 |
|------|------|
| `--calibration` | calibration.json 路径（包含 GoPro 内参和 PrimeColor→GoPro 外参） |
| `--mcal` | .mcal 文件路径（PrimeColor 相机参数） |
| `--mocap-csv` | mocap.csv 路径（marker 3D 坐标） |
| `--gopro-video` | GoPro 视频路径 |
| `--primecolor-video` | PrimeColor 视频路径 |
| `--output` | 输出 stacked 视频路径 |
| `--start-frame` | GoPro 起始帧号 |
| `--num-frames` | 处理帧数 |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--extrinsics-json` | None | 自定义 Mocap→PrimeColor 外参 JSON |
| `--intrinsics-json` | None | PrimeColor 内参 JSON（multical 格式） |
| `--mocap-fps` | 120.0 | Mocap 帧率 |
| `--sync-offset` | 0.0 | 同步偏移（秒），`mocap_time = gopro_time - offset` |
| `--stack-mode` | horizontal | 堆叠模式：`horizontal`（左右）或 `vertical`（上下） |
| `--marker-color` | 0,255,0 | Marker 颜色（BGR 格式，默认绿色） |
| `--marker-size` | 3 | Marker 半径（像素） |

---

## 时间同步说明

### Sync Offset 计算

`sync-offset` 定义了 GoPro 和 Mocap 的时间关系：

```
mocap_time = gopro_time - sync_offset
```

**示例**：
- `sync-offset = 10.5799` 秒
- GoPro 帧 3800（假设 60fps）= 63.33 秒
- Mocap 时间 = 63.33 - 10.5799 = 52.75 秒
- Mocap 帧 = 52.75 × 120 fps = 6330

### PrimeColor 同步

PrimeColor 视频**直接使用 frame index** 与 mocap 对齐（不使用 sync-offset），假设：
- PrimeColor 帧 N → Mocap 帧 N
- 这要求 PrimeColor 和 Mocap 已经预先同步

---

## 输出视频格式

### Stacked Video

生成的视频包含两个画面：

**Horizontal（左右堆叠）**：
```
┌─────────────┬─────────────┐
│   GoPro     │ PrimeColor  │
│   (红色)    │   (绿色)    │
└─────────────┴─────────────┘
```

**Vertical（上下堆叠）**：
```
┌─────────────┐
│   GoPro     │
│   (红色)    │
├─────────────┤
│ PrimeColor  │
│   (绿色)    │
└─────────────┘
```

### 画面标识

- GoPro 画面左上角显示 "GoPro"（红色）
- PrimeColor 画面左上角显示 "PrimeColor"（绿色）

---

## 使用场景

### 场景 1：验证时间同步

检查 GoPro 和 PrimeColor 的 markers 是否对齐：

```bash
python project_markers_dual_video.py \
  ... (标准参数) \
  --start-frame 3800 \
  --num-frames 100 \
  --stack-mode horizontal
```

观察输出视频：
- ✅ 如果 markers **在同一位置**，说明同步正确
- ❌ 如果 markers **位置不一致**，需要调整 `sync-offset`

### 场景 2：调整 Sync Offset

如果发现不对齐，尝试调整偏移：

```bash
# 测试不同的 offset
python project_markers_dual_video.py ... --sync-offset 10.5
python project_markers_dual_video.py ... --sync-offset 10.6
python project_markers_dual_video.py ... --sync-offset 10.7
```

### 场景 3：长视频可视化

生成完整对比视频：

```bash
python project_markers_dual_video.py \
  ... (标准参数) \
  --start-frame 3800 \
  --num-frames 3000 \
  --stack-mode vertical
```

---

## 故障排除

### Q1: 两个画面 markers 不对齐？

**A**: 调整 `--sync-offset` 参数：
1. 先用短视频测试（`--num-frames 100`）
2. 观察 markers 位置差异
3. 根据差异调整 offset
4. 重复测试直到对齐

### Q2: 某个画面没有 markers？

**A**: 检查以下几点：
- GoPro 投影：检查 `calibration.json` 和 `extrinsics-json`
- PrimeColor 投影：检查 `intrinsics-json` 和 `extrinsics-json`
- Mocap 数据：确认指定帧范围有 marker 数据

### Q3: 视频分辨率不匹配？

**A**: 脚本会自动调整分辨率进行堆叠：
- Horizontal 模式：调整为相同高度
- Vertical 模式：调整为相同宽度

### Q4: 输出视频太大？

**A**: 使用更小的帧范围：
```bash
--start-frame 3800 --num-frames 300  # 只处理 300 帧
```

---

## 与其他脚本的对比

| 脚本 | 目标视频 | 时间同步 | 输出格式 |
|------|----------|----------|----------|
| `project_markers_to_gopro.py` | GoPro | ✅ sync-offset | 单视频 |
| `project_markers_to_video_v2.py` | PrimeColor | ❌ 直接 frame index | 单视频 |
| `project_markers_dual_video.py` | **GoPro + PrimeColor** | ✅ GoPro 用 offset | **Stacked 视频** |

---

## 示例命令

### 示例 1：标准使用（横向堆叠）

```bash
python project_markers_dual_video.py \
  --calibration /Volumes/FastACIS/GoPro/motion/calibration.json \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --extrinsics-json extrinsics_calibrated.json \
  --intrinsics-json /Volumes/FastACIS/gopro/prime_gopro_sync/intrinsic_merged.json \
  --mocap-csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --gopro-video /Volumes/FastACIS/GoPro/motion/Cam4/Video.MP4 \
  --primecolor-video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \
  --output dual_horizontal.mp4 \
  --mocap-fps 120.0 \
  --sync-offset 10.5799 \
  --start-frame 3800 \
  --num-frames 600 \
  --stack-mode horizontal
```

### 示例 2：纵向堆叠 + 自定义颜色

```bash
python project_markers_dual_video.py \
  ... (同上) \
  --output dual_vertical.mp4 \
  --stack-mode vertical \
  --marker-color 0,0,255 \
  --marker-size 5
```

### 示例 3：快速测试（100帧）

```bash
python project_markers_dual_video.py \
  ... (同上) \
  --output quick_test.mp4 \
  --start-frame 3800 \
  --num-frames 100
```

---

## 技术细节

### 投影流程

**GoPro 投影**：
1. OptiTrack world → PrimeColor OptiTrack cam
2. 坐标系翻转 (YZ flip: [1,-1,-1])
3. PrimeColor standard → GoPro standard (逆变换)
4. 标准投影 (positive fx)

**PrimeColor 投影**：
1. OptiTrack world → PrimeColor cam (直接)
2. Method 4 投影 (negative fx)

### 时间轴计算

```python
# GoPro
gopro_time = gopro_frame / gopro_fps
mocap_time = gopro_time - sync_offset
mocap_frame = int(mocap_time * mocap_fps)

# PrimeColor
mocap_frame = primecolor_frame  # 直接映射
```

---

## 输出统计

运行后会显示：

```
✓ 完成!
  总帧数: 600
  GoPro投影帧数: 598
  PrimeColor投影帧数: 600
  输出: dual_marker_projection.mp4
```

- **总帧数**: 处理的视频帧数
- **GoPro投影帧数**: 成功投影 markers 的帧数
- **PrimeColor投影帧数**: 成功投影 markers 的帧数

---

## 注意事项

1. **内存使用**: 处理长视频可能消耗大量内存，建议分段处理
2. **FPS 匹配**: GoPro 和 PrimeColor 视频应该有相同或兼容的 FPS
3. **帧范围**: 确保指定的 `start-frame` 和 `num-frames` 在两个视频的有效范围内
4. **Sync Offset**: 需要提前通过其他方法（如 QR 码同步）确定正确的偏移值

---

如有问题，请检查：
1. 所有文件路径是否正确
2. 标定文件格式是否匹配
3. Sync offset 是否准确
4. Mocap 数据在指定帧范围内是否存在
