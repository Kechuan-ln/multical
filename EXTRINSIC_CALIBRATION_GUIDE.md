# 外参标定快速指南

## 🎯 你的情况

- **视频**：15个相机，10分钟，60fps，已同步
- **标定板出现时间**：1分50秒 到 6分10秒（110秒 - 370秒）
- **内参文件**：`intrinsic_hyperoff_linear_60fps.json`（包含所有相机）
- **Board**：`charuco_b1_2.yaml`（7x9 ChArUco板）

## ⚡ 快速运行（推荐）

```bash
cd /Volumes/FastACIS/annotation_pipeline
bash run_extrinsic_calibration_custom.sh
```

这个脚本会：
1. ✅ 自动过滤内参（只保留实际使用的15个相机）
2. ✅ 提取标定板时间段的图像（110秒-370秒，2fps采样）
3. ✅ 外参标定 + 可视化

**预计时间**：15-20分钟

## 🔧 关键参数（在脚本中修改）

打开 `run_extrinsic_calibration_custom.sh`，修改这些参数：

```bash
START_TIME=110         # 标定板开始时间（秒）
DURATION=260           # 持续时间（秒），260秒 = 4分20秒
SAMPLE_FPS=2           # 采样帧率（推荐1-3fps）
```

### 采样帧率建议

| FPS | 帧数/相机 | 总帧数(15相机) | 用途 |
|-----|----------|--------------|------|
| 1 fps | 260帧 | 3,900帧 | 快速测试 |
| **2 fps** | **520帧** | **7,800帧** | **推荐（平衡速度和质量）** |
| 3 fps | 780帧 | 11,700帧 | 高精度（较慢） |
| 5 fps | 1,300帧 | 19,500帧 | 最高精度（很慢，可能过拟合） |

## 📊 使用稳定帧选择（高级）

如果你想获得最佳标定质量，使用稳定帧选择：

### 步骤1: 提取所有帧（高FPS）
```bash
cd /Volumes/FastACIS/annotation_pipeline

export PATH_ASSETS_VIDEOS="/Volumes/FastACIS/gorpos-2-calibration"

# 提取标定板时间段的所有帧（5fps）
python scripts/convert_video_to_images.py \
  --src_tag "videos" \
  --cam_tags "cam1,cam10,cam11,cam12,cam15,cam16,cam17,cam18,cam2,cam3,cam5,cam6,cam7,cam8,cam9" \
  --fps 5 \
  --ss 110 \
  --duration 260
```
python scripts/convert_video_to_images.py \
  --src_tag "videos" \
  --cam_tags "cam0" \
  --fps 5 \
  --ss 0 \
  --duration 90

### 步骤2: 检测稳定帧
```bash
cd scripts

python find_stable_boards.py \
  --recording_tag videos/original \
  --boards ../multical/asset/charuco_b1_2.yaml \
  --movement_threshold 10.0 \
  --min_detection_quality 40 \
  --downsample_rate 5
```

**输出示例**：
```
Stable frame indices: [15, 25, 42, 58, 73, 89, ...]
```

### 步骤3: 复制稳定帧
```bash
python copy_image_subset.py \
  --image_path ../gorpos-2-calibration/videos/original \
  --dest_path ../gorpos-2-calibration/videos_stable \
  --frames 15,25,42,58,73,89  # 使用步骤2的输出
```

### 步骤4: 用稳定帧标定
```bash
cd ../multical

python calibrate.py \
  --boards ./asset/charuco_b1_2.yaml \
  --image_path /Volumes/FastACIS/gorpos-2-calibration/videos_stable \
  --calibration /Volumes/FastACIS/gorpos-2-calibration/intrinsic_filtered.json \
  --fix_intrinsic \
  --vis
```

## ⚠️ 常见问题

### 问题1: Total: 0（没有检测到标定板）

**原因**：时间段设置错误，没有覆盖标定板出现的时间

**解决**：
```bash
# 修改脚本中的参数
START_TIME=110    # 1分50秒
DURATION=260      # 4分20秒
```

### 问题2: 检测到的帧太少（< 50帧）

**原因**：
- SAMPLE_FPS太低
- 标定板移动太快
- 光照条件差

**解决**：
```bash
# 方案1: 提高采样率
SAMPLE_FPS=3  # 或 5

# 方案2: 扩大时间范围
START_TIME=100
DURATION=300

# 方案3: 使用稳定帧选择（见上面高级方法）
```

### 问题3: RMS误差太大（> 1.0像素）

**原因**：
- 标定板移动模糊
- 相机之间重叠视野不足
- 标定板检测质量差

**解决**：
```bash
# 1. 使用稳定帧选择
python find_stable_boards.py --movement_threshold 5.0  # 更严格

# 2. 降低采样率，减少模糊帧
SAMPLE_FPS=1

# 3. 检查可视化图像
open /Volumes/FastACIS/gorpos-2-calibration/videos/vis/
# 确保标定板清晰可见，3D坐标轴投影正确
```

### 问题4: 某些相机没有 camera_base2cam

**原因**：该相机没有检测到足够的标定板

**解决**：
- 查看该相机的图像，确认能看到标定板
- 检查该相机的内参是否正确
- 尝试调整标定板在视野中的位置

## 📈 标定质量指标

### 优秀标定
- ✅ RMS < 0.5 像素
- ✅ 每个相机检测到 > 50 帧标定板
- ✅ 所有相机对之间有重叠视野

### 可接受标定
- ⚠️ RMS < 1.0 像素
- ⚠️ 每个相机检测到 > 30 帧
- ⚠️ 大部分相机对有重叠

### 需要重新标定
- ❌ RMS > 1.0 像素
- ❌ 某些相机检测帧数 < 20
- ❌ 可视化结果中坐标轴投影明显偏移

## 🎓 最佳实践

### 1. 录制标定视频时
- ✅ 在不同位置停顿1-2秒（方便稳定帧检测）
- ✅ 覆盖所有相机视野
- ✅ 多角度、不同距离
- ✅ 避免快速移动（减少模糊）
- ✅ 充足光照

### 2. 选择时间段时
- ✅ 确保标定板在所有相机中可见
- ✅ 避开过渡阶段（进场/退场）
- ✅ 优先选择标定板静止的时段

### 3. 采样策略
- **快速测试**：1-2fps
- **生产使用**：2-3fps + 稳定帧选择
- **高精度**：5fps + 稳定帧选择

## 📝 完整工作流示例

```bash
# 1. 视频同步（已完成）
# 输出: /Volumes/FastACIS/gorpos-2-sync/gorpos-2/*.MP4

# 2. 运行外参标定（一键）
cd /Volumes/FastACIS/annotation_pipeline
bash run_extrinsic_calibration_custom.sh

# 3. 验证结果
# 查看可视化
open /Volumes/FastACIS/gorpos-2-calibration/videos/vis/cam1/

# 检查RMS
cat /Volumes/FastACIS/gorpos-2-calibration/videos/original/calibration.json | grep rms

# 4. 如果满意，复制标定结果
cp /Volumes/FastACIS/gorpos-2-calibration/videos/original/calibration.json \
   /your/project/directory/
```

## 🔍 调试技巧

### 检查提取的图像
```bash
# 随机查看几张图像
open /Volumes/FastACIS/gorpos-2-calibration/videos/original/cam1/frame_0050.png
open /Volumes/FastACIS/gorpos-2-calibration/videos/original/cam1/frame_0150.png

# 应该能清晰看到ChArUco标定板
```

### 查看标定日志
```bash
cat /Volumes/FastACIS/gorpos-2-calibration/videos/original/calibration.txt
```

### 单独测试一个相机
```bash
# 只处理一个相机进行快速测试
python scripts/convert_video_to_images.py \
  --src_tag "videos" \
  --cam_tags "cam1" \
  --fps 2 \
  --ss 110 \
  --duration 260
```

## ✅ 成功标志

标定成功后，你会得到：
- 📄 `calibration.json` - 包含所有相机的外参（camera_base2cam）
- 🖼️ `videos/original/cam*/` - 提取的图像帧
- 📊 `videos/vis/` - 可视化结果（带3D坐标轴投影）

**calibration.json 结构**：
```json
{
  "cameras": {...},  // 内参（从输入复制）
  "camera_base2cam": {
    "cam1": {"R": [...], "T": [...]},
    "cam2": {"R": [...], "T": [...]},
    ...
  },
  "rms": 0.45  // 重投影误差
}
```

现在可以用这个文件进行3D重建、多视角姿态估计等任务！
