# 快速入门：GoPro + PrimeColor 标定

只需 3 步完成 GoPro 和 PrimeColor 相机的时间同步和外参标定。

## 前提条件

### 1. 环境准备

```bash
# 激活 multical 环境
conda activate multical

# 确认依赖已安装
pip install opencv-python numpy pyzbar
```

### 2. 准备视频文件

你需要录制以下视频：

| 视频类型 | GoPro | PrimeColor | 说明 |
|---------|-------|-----------|------|
| **QR 码视频** | ✅ | ✅ | 两个相机同时录制屏幕上播放的 QR 码视频（前 30 秒） |
| **ChArUco 标定板** | ✅ | ✅ | 两个相机同时录制 ChArUco 标定板（各个角度，100+ 帧） |

**QR 码视频录制方法**：
1. 在电脑屏幕上播放 QR 码视频（anchor video）
2. 同时启动 GoPro 和 PrimeColor 录制
3. 录制前 30 秒即可

**ChArUco 标定板录制方法**：
1. 打印或显示 ChArUco 标定板（B3 或 B1 尺寸）
2. 两个相机同时录制标定板
3. 从不同角度、距离录制（确保标定板占画面 30-70%）
4. 录制 30-60 秒即可

### 3. 准备内参文件

| 文件 | 路径 | 说明 |
|------|------|------|
| **GoPro 内参** | `intrinsic_hyperoff_linear_60fps.json` | 已有预计算内参（或自行标定） |
| **PrimeColor 内参** | `calibration.mcal` | 从 Motive/OptiTrack 导出的标定文件 |

---

## 方法一：使用 Shell 脚本（最简单）

### 步骤 1: 编辑配置

打开 `run_gopro_primecolor_calibration.sh`，修改以下路径：

```bash
nano run_gopro_primecolor_calibration.sh
```

修改这些变量：
```bash
# 输入文件路径（修改为你的实际路径）
PRIMECOLOR_MCAL="/path/to/your/calibration.mcal"
GOPRO_INTRINSIC="/path/to/your/intrinsic_hyperoff_linear_60fps.json"

QR_ANCHOR="/path/to/qr_anchor.mp4"
GOPRO_QR="/path/to/gopro_qr.mp4"
PRIMECOLOR_QR="/path/to/primecolor_qr.mp4"

GOPRO_CHARUCO="/path/to/gopro_charuco.mp4"
PRIMECOLOR_CHARUCO="/path/to/primecolor_charuco.mp4"

# GoPro 相机名称（如果内参文件包含多个相机）
GOPRO_CAMERA_NAME="cam2"  # 改为你的相机名称（cam2, cam3, 等）
```

### 步骤 2: 运行脚本

```bash
cd /Volumes/FastACIS/annotation_pipeline
./run_gopro_primecolor_calibration.sh
```

### 步骤 3: 等待完成

脚本会自动执行所有步骤（5-15 分钟，取决于视频长度）：
- ✅ 提取 PrimeColor 内参
- ✅ 同步 GoPro 和 PrimeColor
- ✅ 计算外参标定

---

## 方法二：使用 Python 脚本（灵活）

### 单条命令完成所有步骤

```bash
python run_gopro_primecolor_calibration.py \
  --primecolor-mcal /path/to/calibration.mcal \
  --gopro-intrinsic intrinsic_hyperoff_linear_60fps.json \
  --qr-anchor /path/to/qr_anchor.mp4 \
  --gopro-qr /path/to/gopro_qr.mp4 \
  --primecolor-qr /path/to/primecolor_qr.mp4 \
  --gopro-charuco /path/to/gopro_charuco.mp4 \
  --primecolor-charuco /path/to/primecolor_charuco.mp4 \
  --output-dir calibration_output/gopro_primecolor \
  --gopro-camera-name cam2
```

### 可选参数

```bash
# 跳过某些步骤（如果已完成）
--skip-intrinsic      # 跳过内参提取
--skip-sync           # 跳过 QR 码同步
--skip-extrinsic      # 跳过外参标定

# 调整参数
--scan-duration 45    # 扫描更长时间的 QR 码（默认 30 秒）
--qr-step 2           # 更密集地检测 QR 码（默认每 5 帧）
--extrinsic-fps 2.0   # 提取更多帧用于外参标定（默认 1.0）
--board ./multical/asset/charuco_b1_2.yaml  # 使用 B1 标定板
```

---

## 方法三：分步执行（调试用）

如果需要逐步执行或调试，参考 [完整指南](GOPRO_PRIMECOLOR_CALIBRATION_GUIDE.md)。

---

## 输出文件

完成后，你会得到以下文件：

```
calibration_output/gopro_primecolor/
├── intrinsics/
│   ├── primecolor_intrinsic.json    # PrimeColor 内参
│   └── gopro_intrinsic.json         # GoPro 内参
├── sync/
│   ├── gopro_synced.mp4             # GoPro 同步视频
│   ├── primecolor_synced.mp4        # PrimeColor 同步视频
│   ├── gopro_verify.mp4             # 验证视频（并排对比）
│   ├── primecolor_verify.mp4        # 验证视频
│   ├── gopro_sync_result.json       # 同步偏移量
│   └── primecolor_sync_result.json
└── extrinsics/
    ├── frames/
    │   ├── calibration.json         # 🎯 最终标定文件（最重要！）
    │   ├── cam1/                    # GoPro 提取的帧
    │   ├── primecolor/              # PrimeColor 提取的帧
    │   └── vis/                     # 可视化结果
    └── intrinsic_merged.json        # 合并的内参
```

---

## 验证结果

### 1. 检查同步效果

```bash
# 播放并排对比视频
open calibration_output/gopro_primecolor/sync/gopro_verify.mp4
open calibration_output/gopro_primecolor/sync/primecolor_verify.mp4
```

**验证标准**：
- ✅ QR 码在两个画面中完全对齐
- ✅ QR 码内容同时变化

### 2. 检查外参标定质量

```bash
# 查看 RMS 误差
cat calibration_output/gopro_primecolor/extrinsics/frames/calibration.json | \
  python -c "import sys, json; print('RMS:', json.load(sys.stdin)['rms'])"

# 查看可视化结果
open calibration_output/gopro_primecolor/extrinsics/frames/vis/
```

**验证标准**：
- ✅ RMS 误差 < 1.0 像素（越小越好）
- ✅ 可视化图像中角点检测正确（绿色标记）

### 3. 检查标定文件内容

```bash
cat calibration_output/gopro_primecolor/extrinsics/frames/calibration.json | \
  python -m json.tool
```

**应包含**：
- ✅ `cameras.cam1`: GoPro 内参（K 矩阵、dist）
- ✅ `cameras.primecolor`: PrimeColor 内参
- ✅ `camera_base2cam.cam1`: GoPro 外参（R、T）
- ✅ `camera_base2cam.primecolor`: PrimeColor 外参
- ✅ `rms`: 重投影误差

---

## 使用标定结果

### 复制到录制目录

```bash
# 将 calibration.json 复制到你的录制目录
cp calibration_output/gopro_primecolor/extrinsics/frames/calibration.json \
   /Volumes/FastACIS/csldata/<recording_name>/original/calibration.json
```

### 运行 3D 三角化

```bash
# 使用标定结果进行 3D 重建
python scripts/run_triangulation.py \
  --recording_tag <recording_name>/original \
  --verbose
```

---

## 常见问题

### Q1: QR 码检测不到怎么办？

**原因**：QR 码视频质量差、对焦不准、步长太大

**解决**：
```bash
# 降低检测步长（更密集）
--qr-step 2

# 延长扫描时间
--scan-duration 60

# 安装更好的 QR 码检测库
pip install pyzbar
```

### Q2: 同步偏移量不稳定（标准差大）？

**原因**：QR 码播放不稳定、相机帧率不稳定

**解决**：
1. 使用稳定的屏幕播放 QR 码（避免手机）
2. 确保相机帧率设置正确
3. 增加扫描时长以获取更多样本

### Q3: ChArUco 角点检测失败？

**原因**：标定板不清晰、配置文件错误、视野范围问题

**解决**：
1. 检查提取的帧图像是否清晰
2. 确保标定板占视野 30-70%
3. 使用正确的标定板配置文件：
   - B3 标定板: `--board ./multical/asset/charuco_b3.yaml`
   - B1 标定板: `--board ./multical/asset/charuco_b1_2.yaml`

### Q4: RMS 误差 > 1.0 像素？

**原因**：内参不准确、标定板质量差、相机抖动

**解决**：
1. 重新标定内参（确保 RMS < 0.5）
2. 使用三脚架固定相机
3. 确保标定板平整无变形
4. 增加标定图像数量和角度多样性

### Q5: 相机名称不匹配？

**症状**：`❌ GoPro内参中没有找到cam1`

**解决**：
```bash
# 查看内参文件包含哪些相机
cat intrinsic_hyperoff_linear_60fps.json | jq '.cameras | keys'

# 使用正确的相机名称
--gopro-camera-name cam2  # 或 cam3, cam4 等
```

---

## 下一步

完成标定后，你可以：

1. **进行视频同步**：使用 `scripts/sync_timecode.py` 同步多机位录制
2. **运行 2D 检测**：使用 ViTPose 检测关键点
3. **3D 三角化**：将 2D 检测三角化为 3D 坐标
4. **3D 姿态估计**：使用标定结果进行人体姿态估计

详见 [CLAUDE.md](CLAUDE.md) 了解完整的 Pipeline 流程。

---

## 获取帮助

- **完整文档**：[GOPRO_PRIMECOLOR_CALIBRATION_GUIDE.md](GOPRO_PRIMECOLOR_CALIBRATION_GUIDE.md)
- **同步问题**：[SYNC_FIX_SUMMARY.md](SYNC_FIX_SUMMARY.md)
- **Pipeline 说明**：[CLAUDE.md](CLAUDE.md)

---

**祝你标定成功！** 🎉
