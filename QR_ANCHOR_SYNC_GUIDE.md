# QR Anchor同步方法完整指南

## 🎉 v2.0 新特性

**最大改进：无需手动生成CSV！**

```bash
# 以前（v1.0）：需要3步
python generate_qr_sync_video.py --output qr.mp4 --duration 60 --fps 30
python generate_qr_metadata.py --output qr.csv --duration 60 --fps 30  # 需要手动生成
python sync_with_qr_anchor.py --video1 cam1.mp4 --video2 cam2.mp4 --output out.mp4 --anchor-csv qr.csv

# 现在（v2.0）：只需2步
python generate_qr_sync_video.py --output qr.mp4 --duration 60 --fps 30
python sync_with_qr_anchor.py --video1 cam1.mp4 --video2 cam2.mp4 --output out.mp4 --anchor-video qr.mp4
```

**新增功能**：
- ✨ `--anchor-video` 参数：直接指定anchor视频
- ✨ 自动提取QR码序列和时间映射
- ✨ 自动检测anchor视频FPS
- ✨ 自动验证QR码连续性

---

## 🎯 核心概念

### 问题：传统QR同步的局限
传统的 `sync_with_qr_frames.py` 要求两个相机**必须同时看到相同的QR码序列**才能同步。如果：
- Camera1 看到 QR码 #100-#200
- Camera2 看到 QR码 #300-#400
- **没有重叠** → 传统方法失败 ❌

### 解决方案：Anchor Timecode映射
使用已知时间序列的**Anchor QR码视频**作为参考基准：

1. **生成Anchor视频**：每帧QR码 = 帧编号（如 `000001`, `000002`, ...）
2. **相机录制**：两个相机分别录制该Anchor视频（可以不同时开始）
3. **Anchor映射**：
   - Camera1 在 `t1=5.0s` 看到 QR#100 → Anchor时间 `T1=3.33s` (100/30fps)
   - Camera2 在 `t2=8.0s` 看到 QR#150 → Anchor时间 `T2=5.0s` (150/30fps)
4. **计算偏移**：
   - Camera1相对Anchor: `offset1 = t1 - T1 = 5.0 - 3.33 = 1.67s`
   - Camera2相对Anchor: `offset2 = t2 - T2 = 8.0 - 5.0 = 3.0s`
   - **相对偏移**: `offset = offset1 - offset2 = 1.67 - 3.0 = -1.33s`

✅ **即使两个相机看到完全不同的QR码序列，也能同步！**

---

## 📦 工具集

### 1. `generate_qr_sync_video.py`
生成Anchor QR码视频

### 2. `generate_qr_metadata.py` (新)
生成Anchor metadata CSV（可选，用于非标准FPS）

### 3. `sync_with_qr_anchor.py` (v2.0 ⭐)
基于Anchor映射的视频同步工具

**v2.0 新功能**：
- 直接从anchor视频提取metadata（无需CSV）
- 自动检测FPS和验证QR码序列
- 三种模式：视频提取 > CSV > 默认映射

---

## 🚀 完整工作流程

### **步骤1：生成Anchor QR码视频**

```bash
cd /Volumes/FastACIS/annotation_pipeline

# 生成60秒、30fps的QR码视频
python generate_qr_sync_video.py \
  --output qr_anchor_30fps_60s.mp4 \
  --duration 60 \
  --fps 30 \
  --resolution 1920x1080 \
  --qr-size 800
```

**输出**：
- `qr_anchor_30fps_60s.mp4` - Anchor视频
- 每帧包含唯一的QR码（000000, 000001, 000002, ...）

**重要提示**：
- 记住使用的 `fps` 和 `duration`，后续需要匹配
- 如果使用了 `--prefix`（如 `SYNC-`），后续也需要指定

---

### **步骤2：（可选）生成Metadata CSV**

⚠️ **注意：现在可以跳过此步骤！**

`sync_with_qr_anchor.py` v2.0 支持**直接从anchor视频提取metadata**，无需手动生成CSV：

```bash
# 新方法：直接使用anchor视频（推荐）
python sync_with_qr_anchor.py \
  --video1 camera1.mp4 \
  --video2 camera2.mp4 \
  --output camera2_synced.mp4 \
  --anchor-video qr_anchor_30fps_60s.mp4  # 直接指定anchor视频
```

如果仍想使用CSV（可选）：

```bash
# 生成对应的metadata CSV
python generate_qr_metadata.py \
  --output qr_anchor_metadata.csv \
  --duration 60 \
  --fps 30
```

**输出CSV格式**：
```csv
frame_number,anchor_time,qr_data
0,0.000000,000000
1,0.033333,000001
2,0.066667,000002
...
```

**何时需要CSV？**
- ❌ 大多数情况**不需要**（直接用 `--anchor-video`）
- ✅ 如果anchor视频质量太差无法自动提取
- ✅ 如果想要预先缓存metadata加速重复运行

---

### **步骤3：相机录制Anchor视频**

1. **播放** `qr_anchor_30fps_60s.mp4`（在显示器/电视上）
2. **同时录制**：使用GoPro和PrimeColor相机录制该视频
   - 相机可以不同时开始/停止
   - 相机可以使用不同的FPS（如60fps vs 30fps）
   - 只要能看清QR码即可

**示例录制文件**：
- `gopro1_recording.MP4` (60fps)
- `primecolor_recording.mp4` (30fps)

---

### **步骤4：同步相机视频**

#### 方法A：直接使用Anchor视频（🌟 推荐，最简单）

```bash
cd /Volumes/FastACIS/annotation_pipeline

python sync_with_qr_anchor.py \
  --video1 /path/to/gopro1_recording.MP4 \
  --video2 /path/to/primecolor_recording.mp4 \
  --output /path/to/primecolor_synced.mp4 \
  --anchor-video qr_anchor_30fps_60s.mp4 \
  --scan-start 0 \
  --scan-duration 30 \
  --save-json sync_result.json
```

**优势**：
- ✅ 无需手动生成CSV
- ✅ 自动检测anchor视频的FPS
- ✅ 自动验证QR码序列
- ✅ 最简单的使用方式

#### 方法B：使用CSV（可选）

```bash
python sync_with_qr_anchor.py \
  --video1 /path/to/gopro1_recording.MP4 \
  --video2 /path/to/primecolor_recording.mp4 \
  --output /path/to/primecolor_synced.mp4 \
  --anchor-csv qr_anchor_metadata.csv \
  --scan-start 0 \
  --scan-duration 30 \
  --save-json sync_result.json
```

#### 方法C：使用默认映射（需要知道anchor FPS）

```bash
python sync_with_qr_anchor.py \
  --video1 /path/to/gopro1_recording.MP4 \
  --video2 /path/to/primecolor_recording.mp4 \
  --output /path/to/primecolor_synced.mp4 \
  --anchor-fps 30 \
  --scan-start 0 \
  --scan-duration 30 \
  --save-json sync_result.json
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video1` | 参考视频（如GoPro） | 必需 |
| `--video2` | 需要同步的视频（如PrimeColor） | 必需 |
| `--output` | 输出同步后的视频 | 必需 |
| `--anchor-video` | 🌟 Anchor视频路径（推荐） | 可选 |
| `--anchor-csv` | Anchor metadata CSV（优先级高于视频） | 可选 |
| `--anchor-fps` | Anchor视频FPS（仅在无视频/CSV时使用） | 30 |
| `--scan-start` | 开始扫描时间（秒） | 0 |
| `--scan-duration` | 扫描时长（秒） | 30 |
| `--step` | 帧步长（每N帧检测一次） | 5 |
| `--prefix` | QR码前缀（如 `SYNC-`） | 无 |
| `--target-fps` | 输出视频FPS | video1的FPS |
| `--save-json` | 保存同步结果JSON | 可选 |

**优先级**：`--anchor-csv` > `--anchor-video` > `--anchor-fps`（默认映射）

---

### **步骤5：验证同步结果**

#### 5.1 查看终端输出

```
============================================================
步骤3: 计算同步偏移
============================================================
计算同步偏移（基于anchor timecode）...
  Video1: 25 个QR码映射
  Video2: 32 个QR码映射
  Video1相对anchor偏移: 2.134s
  Video2相对anchor偏移: 5.678s
  相对偏移 (Video1 - Video2): -3.544s
  Video1偏移标准差: 0.012s
  Video2偏移标准差: 0.018s

  QR码映射示例（前10个）:
  Video1:
    [1] QR#000123: video_t=5.12s, anchor_t=4.10s, offset=1.020s
    [2] QR#000156: video_t=7.45s, anchor_t=5.20s, offset=2.250s
    ...
  Video2:
    [1] QR#000234: video_t=12.34s, anchor_t=7.80s, offset=4.540s
    [2] QR#000267: video_t=15.67s, anchor_t=8.90s, offset=6.770s
    ...
```

**关键指标**：
- **相对偏移**：`-3.544s` 表示Video2需要延迟3.544秒
- **标准差**：`< 0.05s` 说明同步稳定
- **QR码映射**：检查是否合理

#### 5.2 查看JSON结果

```bash
cat sync_result.json | jq '.sync_result'
```

```json
{
  "offset_seconds": -3.544,
  "video1_anchor_offset": 2.134,
  "video2_anchor_offset": 5.678,
  "video1_offset_std": 0.012,
  "video2_offset_std": 0.018,
  "video1_qr_count": 25,
  "video2_qr_count": 32,
  "video1_qr_range": [123, 456],
  "video2_qr_range": [234, 567]
}
```

#### 5.3 手动验证

```bash
# 检查输出视频时长和FPS
ffprobe -v error -show_entries format=duration,stream=r_frame_rate \
  /path/to/primecolor_synced.mp4

# 播放对比（目视检查QR码是否对齐）
ffplay /path/to/gopro1_recording.MP4
ffplay /path/to/primecolor_synced.mp4
```

---

## 📊 示例场景

### 场景1：GoPro 60fps + PrimeColor 30fps

```bash
# 1. 生成30fps的Anchor视频
python generate_qr_sync_video.py \
  --output qr_anchor.mp4 \
  --duration 60 \
  --fps 30

# 2. 播放并录制
#    - GoPro: 录制60秒（60fps）
#    - PrimeColor: 录制45秒（30fps，晚开始5秒）

# 3. 同步
python sync_with_qr_anchor.py \
  --video1 gopro_recording.MP4 \
  --video2 primecolor_recording.mp4 \
  --output primecolor_synced.mp4 \
  --anchor-fps 30 \
  --target-fps 60 \
  --scan-duration 30

# 4. 结果：
#    - primecolor_synced.mp4: 60秒 @ 60fps（与GoPro对齐）
#    - 前5秒为黑帧（因为PrimeColor晚开始）
```

---

### 场景2：使用前缀的QR码

```bash
# 1. 生成带前缀的Anchor视频
python generate_qr_sync_video.py \
  --output qr_anchor_prefix.mp4 \
  --duration 60 \
  --fps 30 \
  --prefix "SYNC-"

# QR码内容: SYNC-000000, SYNC-000001, ...

# 2. 生成对应的metadata
python generate_qr_metadata.py \
  --output qr_metadata.csv \
  --duration 60 \
  --fps 30 \
  --prefix "SYNC-"

# 3. 同步（需要指定prefix）
python sync_with_qr_anchor.py \
  --video1 camera1.mp4 \
  --video2 camera2.mp4 \
  --output camera2_synced.mp4 \
  --anchor-csv qr_metadata.csv \
  --prefix "SYNC-" \
  --anchor-fps 30
```

---

### 场景3：完全不重叠的QR码序列

```bash
# Camera1 看到 QR码 #000-#300
# Camera2 看到 QR码 #500-#800
# 传统方法: ❌ 无重叠，失败
# Anchor方法: ✅ 通过anchor映射，成功同步

python sync_with_qr_anchor.py \
  --video1 camera1.mp4 \
  --video2 camera2.mp4 \
  --output camera2_synced.mp4 \
  --anchor-fps 30 \
  --scan-start 0 \
  --scan-duration 60  # 扫描更长时间以覆盖不同QR码范围
```

---

## 🔍 故障排查

### 问题1：检测不到QR码

**症状**：
```
❌ 至少一个视频没有检测到QR码
```

**解决方法**：
1. **检查QR码可见性**：
   ```bash
   # 手动检查某一帧
   ffmpeg -i camera1.mp4 -ss 10 -vframes 1 frame_10s.png
   # 打开frame_10s.png，查看QR码是否清晰可见
   ```

2. **调整扫描参数**：
   ```bash
   # 增加扫描时长
   --scan-duration 60

   # 减小步长（更密集检测）
   --step 2

   # 改变扫描起始位置
   --scan-start 5
   ```

3. **安装pyzbar**（更准确的检测）：
   ```bash
   pip install pyzbar
   ```

---

### 问题2：偏移标准差很大（> 0.5s）

**症状**：
```
⚠️ 警告: 偏移标准差较大，可能存在时间漂移或检测错误
Video1偏移标准差: 1.234s
```

**可能原因**：
1. **QR码检测错误**（误检或跳帧）
2. **相机时间漂移**（不同FPS导致累积误差）
3. **播放不稳定**（Anchor视频播放卡顿）

**解决方法**：
1. **查看JSON中的检测详情**：
   ```bash
   cat sync_result.json | jq '.video1.detections[0:20]'
   # 检查QR码序列是否连续
   ```

2. **只使用开始部分**（避免累积漂移）：
   ```bash
   --scan-start 5
   --scan-duration 15  # 只用15秒
   ```

3. **增加检测频率**：
   ```bash
   --step 1  # 每帧都检测（慢但准确）
   ```

---

### 问题3：输出视频时长不对

**症状**：
```
⚠️ 警告: 输出时长 (45.23s) 与目标 (60.00s) 不匹配
```

**解决方法**：
1. **检查输入视频时长**：
   ```bash
   ffprobe -v error -show_entries format=duration camera1.mp4
   ffprobe -v error -show_entries format=duration camera2.mp4
   ```

2. **确认offset计算正确**：
   ```bash
   cat sync_result.json | jq '.sync_result.offset_seconds'
   ```

3. **手动调整**：
   如果自动同步结果不理想，可以直接使用ffmpeg：
   ```bash
   # 假设offset=-5.0（video2需要延迟5秒）
   ffmpeg -f lavfi -i color=black:s=1920x1080:r=30 -t 5.0 black.mp4
   ffmpeg -i camera2.mp4 -vf "fps=30" -t 55.0 content.mp4
   echo "file 'black.mp4'" > concat.txt
   echo "file 'content.mp4'" >> concat.txt
   ffmpeg -f concat -i concat.txt -c:v libx264 output.mp4
   ```

---

## 💡 最佳实践

### 1. Anchor视频设置

✅ **推荐**：
- FPS: 30（标准，易检测）
- 分辨率: 1920x1080（兼容性好）
- 时长: 60-120秒（足够长）
- QR码尺寸: 800px（大而清晰）

❌ **避免**：
- 过高FPS（如120fps）→ QR码变化太快，难检测
- 过小QR码（<400px）→ 相机可能无法识别
- 过短时长（<30秒）→ 同步窗口太小

### 2. 录制设置

✅ **推荐**：
- 相机稳定放置（减少模糊）
- 显示器亮度调到最高
- 避免反光和遮挡
- 录制时长 > Anchor视频时长（避免截断）

### 3. 同步参数

✅ **推荐**：
```bash
--scan-start 5          # 跳过开始5秒（避免播放启动延迟）
--scan-duration 20-30   # 20-30秒足够
--step 5                # 平衡速度和准确度
```

### 4. 验证流程

1. **检查检测数量**：每个视频应至少检测到20+个QR码
2. **检查标准差**：应 < 0.1s（< 0.05s为理想）
3. **检查QR码范围**：是否合理（如Video1: 100-400, Video2: 200-500）
4. **目视验证**：播放同步后的视频，检查QR码是否对齐

---

## 🆚 与传统方法对比

| 特性 | 传统 `sync_with_qr_frames.py` | 新 `sync_with_qr_anchor.py` |
|------|------------------------------|----------------------------|
| **需要重叠QR码** | ✅ 必须 | ❌ 不需要 |
| **支持不同开始时间** | ❌ 限制 | ✅ 支持 |
| **支持不同FPS** | ✅ 支持 | ✅ 支持 |
| **需要anchor视频** | ❌ 不需要 | ✅ 需要 |
| **精度** | 高（直接匹配） | 高（anchor映射） |
| **鲁棒性** | 低（需要重叠） | 高（无重叠要求） |
| **适用场景** | 两相机同时录制同一QR码视频 | 任意时间录制anchor视频 |

---

## 📚 完整命令参考

### 生成Anchor视频

```bash
python generate_qr_sync_video.py \
  --output <path> \
  [--duration <seconds>] \
  [--fps <fps>] \
  [--resolution WxH] \
  [--qr-size <pixels>] \
  [--prefix <string>]
```

### 生成Metadata CSV

```bash
python generate_qr_metadata.py \
  --output <path> \
  [--duration <seconds>] \
  [--fps <fps>] \
  [--prefix <string>]
```

### 同步视频

```bash
python sync_with_qr_anchor.py \
  --video1 <path> \
  --video2 <path> \
  --output <path> \
  [--anchor-csv <path>] \
  [--anchor-fps <fps>] \
  [--scan-start <seconds>] \
  [--scan-duration <seconds>] \
  [--step <frames>] \
  [--prefix <string>] \
  [--target-fps <fps>] \
  [--save-json <path>]
```

---

## ✅ 总结

**Anchor QR同步方法的优势**：
1. ✅ **无需重叠**：两相机可以看到完全不同的QR码序列
2. ✅ **灵活录制**：相机可以不同时开始/停止
3. ✅ **高鲁棒性**：通过anchor映射，避免直接匹配的限制
4. ✅ **易于验证**：CSV metadata清晰可查

**何时使用**：
- 两相机无法同时看到相同的QR码
- 录制时间窗口不同
- 需要更灵活的同步方案

**前提条件**：
- 需要预先生成anchor QR码视频
- 相机能够清晰拍摄QR码
- 安装了pyzbar或OpenCV QR检测

---

**文档版本**: 1.0
**创建日期**: 2025-10-22
**工具版本**: sync_with_qr_anchor.py v1.0
