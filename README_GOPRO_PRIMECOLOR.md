# GoPro + PrimeColor 标定脚本使用说明

## 数据组织要求

你的工作目录应该包含以下文件：

```
/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /
├── Anchor.mp4              # QR 码 anchor 视频（参考基准）
├── Intrinsic-16.json       # GoPro 内参文件（包含 cam2, cam3, cam4 等）
├── Primecolor.mcal         # PrimeColor 标定文件（从 Motive/OptiTrack 导出）
├── Cam4/
│   └── Video.MP4           # Cam4 录制的视频（包含 QR 码 + ChArUco 标定板）
└── Primecolor/
    └── Video.avi           # PrimeColor 录制的视频（包含 QR 码 + ChArUco 标定板）
```

**注意**：工作目录路径末尾有一个空格！

## 视频录制要求

### 推荐录制方式（QR 码 + 标定板在同一视频）

1. **准备 QR 码视频**：
   - 在电脑屏幕上播放 QR 码 anchor 视频
   - 同时启动 GoPro 和 PrimeColor 录制

2. **录制时间线**：
   ```
   0-30秒：  录制屏幕上的 QR 码（用于时间同步）
   30-120秒：录制 ChArUco 标定板（用于外参标定）
   ```

3. **标定板录制技巧**：
   - 标定板占画面 30-70%
   - 从不同角度、距离录制
   - 确保标定板清晰、无模糊

## 配置文件

打开 `run_gopro_primecolor_calibration.py`，查看 **CONFIGURATION** 部分（第 25-76 行）：

```python
# ---------- Working Directory ----------
WORKING_DIR = "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic "  # 注意末尾空格！

# ---------- Input Files ----------
PRIMECOLOR_MCAL = "Primecolor.mcal"
GOPRO_INTRINSIC = "Intrinsic-16.json"
QR_ANCHOR = "Anchor.mp4"
GOPRO_VIDEO = "Cam4/Video.MP4"
PRIMECOLOR_VIDEO = "Primecolor/Video.avi"

# ---------- Video Segmentation ----------
USE_SAME_VIDEO = True       # True: QR 和标定板在同一视频
QR_START_TIME = 0           # QR 码开始时间（秒）
QR_END_TIME = 30            # QR 码结束时间（秒）
CHARUCO_START_TIME = 30     # 标定板开始时间（秒）
CHARUCO_DURATION = 90       # 标定板时长（秒）

# ---------- Camera Configuration ----------
GOPRO_CAMERA_NAME = "cam4"  # 必须与 Intrinsic-16.json 中的名称匹配（小写）
PRIMECOLOR_CAMERA_ID = None # None = 自动使用第一个相机
```

### 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `USE_SAME_VIDEO` | 是否 QR 和标定板在同一视频 | True |
| `QR_START_TIME` | QR 码开始时间（秒） | 0 |
| `QR_END_TIME` | QR 码结束时间（秒） | 30 |
| `CHARUCO_START_TIME` | ChArUco 标定板开始时间（秒） | 30 |
| `CHARUCO_DURATION` | 标定板录制时长（秒） | 90 |
| `GOPRO_CAMERA_NAME` | GoPro 相机名称（必须与 JSON 匹配） | "cam4" |
| `SCAN_DURATION` | QR 码扫描时长（秒） | 30.0 |
| `QR_STEP` | QR 码检测步长（帧） | 5 |
| `EXTRINSIC_FPS` | 外参标定提取帧率 | 1.0 |
| `BOARD_CONFIG` | 标定板配置文件 | charuco_b3.yaml |

## 运行脚本

### 1. 激活环境

```bash
conda activate multical
cd /Volumes/FastACIS/annotation_pipeline
```

### 2. 检查配置

打开脚本确认配置正确：

```bash
nano run_gopro_primecolor_calibration.py
```

重点检查：
- ✅ `WORKING_DIR` 路径正确（注意末尾空格）
- ✅ `GOPRO_CAMERA_NAME` 与 `Intrinsic-16.json` 匹配（小写）
- ✅ 时间分段设置合理（QR + ChArUco）

### 3. 运行

```bash
python run_gopro_primecolor_calibration.py
```

## 输出文件

脚本会在工作目录下创建 `calibration_output/` 文件夹：

```
calibration_output/
├── intrinsics/
│   ├── primecolor_intrinsic.json    # PrimeColor 内参
│   └── gopro_intrinsic.json         # GoPro 内参（cam1）
├── sync/
│   ├── gopro_qr_segment.mp4         # GoPro QR 片段（0-30s）
│   ├── primecolor_qr_segment.mp4    # PrimeColor QR 片段（0-30s）
│   ├── gopro_charuco_segment.mp4    # GoPro 标定板片段（30-120s）
│   ├── primecolor_charuco_segment.avi # PrimeColor 标定板片段（30-120s）
│   ├── gopro_synced.mp4             # GoPro 同步后的视频
│   ├── primecolor_synced.mp4        # PrimeColor 同步后的视频
│   ├── gopro_verify.mp4             # 验证视频（并排对比）
│   └── primecolor_verify.mp4
└── extrinsics/
    ├── frames/
    │   ├── calibration.json         # 🎯 最终标定文件（最重要！）
    │   ├── cam1/                    # GoPro 提取的帧
    │   ├── primecolor/              # PrimeColor 提取的帧
    │   └── vis/                     # 可视化结果
    └── intrinsic_merged.json        # 合并的内参
```

## 验证结果

### 1. 检查视频分段

```bash
# 查看分段视频时长
ls -lh "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/sync/"

# 应该看到：
# gopro_qr_segment.mp4 (~30秒)
# gopro_charuco_segment.mp4 (~90秒)
```

### 2. 检查同步效果

```bash
# 播放并排验证视频
open "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/sync/gopro_verify.mp4"
open "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/sync/primecolor_verify.mp4"
```

**验证标准**：
- ✅ QR 码在两个画面中完全对齐
- ✅ QR 码内容同时变化

### 3. 检查外参标定质量

```bash
# 查看 RMS 误差
cd "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic "
cat calibration_output/extrinsics/frames/calibration.json | grep rms

# 查看可视化结果
open calibration_output/extrinsics/frames/vis/
```

**验证标准**：
- ✅ RMS 误差 < 1.0 像素
- ✅ 可视化图像中绿色角点检测正确

## 常见问题

### Q1: 提示 "Working directory does not exist"

**原因**：路径末尾的空格没有保留

**解决**：
```python
# 确保路径末尾有空格（注意引号内的空格）
WORKING_DIR = "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic "
                                                              # ↑ 这里有空格
```

### Q2: 提示 "GoPro intrinsic does not contain 'cam4'"

**原因**：相机名称不匹配

**解决**：
```bash
# 查看 Intrinsic-16.json 中的相机名称
head -50 "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /Intrinsic-16.json"

# 使用正确的名称（小写）
GOPRO_CAMERA_NAME = "cam4"  # 或 cam2, cam3 等
```

### Q3: QR 码检测不到

**原因**：时间分段不对或 QR 码不清晰

**解决**：
1. 检查视频分段时间是否正确
2. 播放 `gopro_qr_segment.mp4`，确认包含 QR 码
3. 调整参数：
   ```python
   QR_STEP = 2         # 更密集检测
   SCAN_DURATION = 30.0  # 扫描整个 30 秒
   ```

### Q4: 标定板角点检测失败

**原因**：标定板不清晰或配置不匹配

**解决**：
1. 播放 `gopro_charuco_segment.mp4`，确认包含标定板
2. 检查标定板配置：
   ```python
   # B3 标定板（5x9 网格，50mm 方格）
   BOARD_CONFIG = "./asset/charuco_b3.yaml"

   # B1 标定板（10x14 网格，70mm 方格）
   BOARD_CONFIG = "./asset/charuco_b1_2.yaml"
   ```
3. 增加提取帧数：
   ```python
   EXTRINSIC_MAX_FRAMES = 200  # 默认 100
   ```

### Q5: 视频分段失败

**症状**：找不到 `gopro_qr_segment.mp4` 或时长不对

**解决**：
```bash
# 手动测试 ffmpeg 分段
ffmpeg -ss 0 -t 30 -i "Cam4/Video.MP4" -c copy test_qr.mp4
ffmpeg -ss 30 -t 90 -i "Cam4/Video.MP4" -c copy test_charuco.mp4

# 检查时长
ffprobe test_qr.mp4
ffprobe test_charuco.mp4
```

## 调试技巧

### 跳过已完成的步骤

如果某个步骤已经完成，可以跳过：

```python
SKIP_INTRINSIC = True   # 跳过内参提取
SKIP_SYNC = True        # 跳过 QR 码同步
SKIP_EXTRINSIC = True   # 跳过外参标定
```

### 查看中间文件

```bash
cd "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output"

# 查看分段视频
ls -lh sync/*.mp4 sync/*.avi

# 查看内参文件
cat intrinsics/primecolor_intrinsic.json | python -m json.tool
cat intrinsics/gopro_intrinsic.json | python -m json.tool

# 查看同步结果
cat sync/gopro_sync_result.json | python -m json.tool
cat sync/primecolor_sync_result.json | python -m json.tool
```

## 下一步

标定完成后：

1. **复制标定文件到录制目录**：
   ```bash
   cp calibration_output/extrinsics/frames/calibration.json \
      /path/to/your/recording/original/calibration.json
   ```

2. **运行 3D 重建**：
   ```bash
   cd /Volumes/FastACIS/annotation_pipeline
   python scripts/run_triangulation.py \
     --recording_tag <recording_name>/original \
     --verbose
   ```

## 参考文档

- [完整指南](GOPRO_PRIMECOLOR_CALIBRATION_GUIDE.md) - 详细的分步说明
- [快速入门](QUICK_START_GOPRO_PRIMECOLOR.md) - 简化的使用指南
- [同步修复总结](SYNC_FIX_SUMMARY.md) - QR 码同步原理
- [Pipeline 总览](CLAUDE.md) - 完整 Pipeline 说明

---

**祝你标定成功！** 🎉

如果遇到问题，请参考上面的"常见问题"部分或查看完整文档。
