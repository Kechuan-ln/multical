# GoPro外参标定指南

## 当前情况

### 视频文件位置
```
/Volumes/FastACIS/csltest1/gopros/
├── cam1/calibration.MP4  (1.3GB)
├── cam2/calibration.MP4  (1.3GB)
├── cam3/calibration.MP4  (1.2GB)
└── cam5/calibration.MP4  (1.3GB)
```

**注意**: 你的相机编号是 cam1, cam2, cam3, cam5（没有cam4）

### 预存内参文件
- 位置: `/Volumes/FastACIS/annotation_pipeline/intrinsic_hyperoff_linear_60fps.json`
- 包含: cam2, cam3, cam4, cam5, cam6, cam7, cam8, cam9的内参

**⚠️ 重要**: 你的cam1在预存内参中**没有对应项**，有以下选择：
1. **推荐**: 先用ChArUco板单独标定cam1的内参
2. 如果cam1相机设置与其他相机相同，可以暂时借用cam2的内参
3. 只用cam2, cam3, cam5进行外参标定（跳过cam1）

## 快速开始

### 方式A: 自动化脚本（推荐）

```bash
cd /Volumes/FastACIS/annotation_pipeline
./calibration_workflow.sh
```

这个脚本会自动完成：
1. ✓ 检查视频文件
2. ⚠️ 尝试timecode同步（如果失败会跳过）
3. ✓ 提取视频帧（5fps，60秒）
4. ✓ 运行外参标定
5. ✓ 生成可视化

**输出位置**: `/Volumes/FastACIS/csltest1/output/`

---

### 方式B: 手动分步执行

如果自动脚本出现问题，可以手动执行以下步骤：

#### 步骤1: 准备目录结构

```bash
# 创建工作目录
mkdir -p /Volumes/FastACIS/csltest1/output/calibration_videos

# 复制或链接视频文件
cd /Volumes/FastACIS/csltest1/output/calibration_videos
for cam in cam1 cam2 cam3 cam5; do
    mkdir -p $cam
    ln -s ../../gopros/$cam/calibration.MP4 $cam/calibration.MP4
done
```

#### 步骤2: 检查视频是否有timecode

```bash
cd /Volumes/FastACIS/annotation_pipeline

# 尝试同步（如果有timecode）
python3 scripts/sync_timecode.py \
  --src_tag "../../csltest1/output/calibration_videos" \
  --out_tag "../../csltest1/output/calibration_synced"
```

**如果失败**: 说明视频没有timecode，跳过同步，使用原始视频

#### 步骤3: 提取视频帧

```bash
cd /Volumes/FastACIS/annotation_pipeline

# 如果步骤2同步成功，使用：
python3 scripts/convert_video_to_images.py \
  --src_tag "../../csltest1/output/calibration_synced" \
  --cam_tags cam1,cam2,cam3,cam5 \
  --fps 5 \
  --ss 5 \
  --duration 60

# 如果步骤2失败，使用原始视频：
python3 scripts/convert_video_to_images.py \
  --src_tag "../../csltest1/output/calibration_videos" \
  --cam_tags cam1,cam2,cam3,cam5 \
  --fps 5 \
  --ss 5 \
  --duration 60
```

**参数说明**:
- `--fps 5`: 每秒提取5帧（共300帧/分钟）
- `--ss 5`: 跳过前5秒
- `--duration 60`: 提取60秒

**输出**: `/Volumes/FastACIS/csltest1/output/calibration_*/original/cam*/frame_*.png`

#### 步骤4: 外参标定

```bash
cd /Volumes/FastACIS/annotation_pipeline/multical

# 使用预存内参标定外参
python3 calibrate.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path "../../csltest1/output/calibration_synced/original" \
  --calibration ../intrinsic_hyperoff_linear_60fps.json \
  --fix_intrinsic \
  --limit_images 300 \
  --vis
```

**如果cam1没有预存内参**，有两种处理方式：

##### 选项A: 只标定cam2, cam3, cam5
```bash
# 修改命令，只使用有内参的相机
# 需要手动从calibration_*/original/目录中删除cam1文件夹
rm -rf /Volumes/FastACIS/csltest1/output/calibration_*/original/cam1

# 然后运行标定
python3 calibrate.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path "../../csltest1/output/calibration_synced/original" \
  --calibration ../intrinsic_hyperoff_linear_60fps.json \
  --fix_intrinsic \
  --limit_images 300 \
  --vis
```

##### 选项B: 先标定cam1内参
```bash
# 1. 先单独标定cam1内参
cd /Volumes/FastACIS/annotation_pipeline/multical

python3 intrinsic.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path "../../csltest1/output/calibration_synced/original" \
  --cameras cam1 \
  --limit_images 300 \
  --vis

# 2. 手动合并cam1内参到intrinsic_hyperoff_linear_60fps.json
#    复制输出的JSON中cam1的K和dist字段

# 3. 再运行外参标定（使用完整的内参文件）
```

---

## 输出文件说明

标定完成后，会生成以下文件：

### 主要输出
```
/Volumes/FastACIS/csltest1/output/
├── calibration_synced/original/
│   ├── calibration.json          ← 🎯 最重要！外参标定结果
│   ├── cam1/, cam2/, cam3/, cam5/  (提取的图像帧)
│   └── vis/                       ← 可视化结果
│       ├── cam1/, cam2/, cam3/, cam5/
│       └── (检测到的ChArUco角点 + 3D坐标轴投影)
├── sync_log.txt                  (同步日志)
├── convert_log.txt               (转换日志)
└── calibration_log.txt           (标定日志)
```

### calibration.json格式
```json
{
  "cameras": {
    "cam1": {"K": [...], "dist": [...]},  // 从输入复制
    "cam2": {"K": [...], "dist": [...]},
    ...
  },
  "camera_base2cam": {
    "cam1": {"R": [3x3矩阵], "T": [3向量]},  // 🔑 外参结果
    "cam2": {"R": [...], "T": [...]},
    ...
  }
}
```

---

## 验证标定质量

### 1. 查看终端输出
```
Final reprojection RMS=0.45 (0.48)  ← 应该 < 1.0像素
```

### 2. 检查可视化结果
打开 `output/calibration_synced/vis/cam*/` 中的图像：
- ✓ ChArUco角点应被正确检测（黄色圆圈）
- ✓ 3D坐标轴投影应正确（红=X，绿=Y，蓝=Z）
- ✓ 轴长度应与标定板尺寸一致
- ✓ Z轴应指向标定板内部

### 3. 查看日志文件
```bash
# 检查是否有错误或警告
cat /Volumes/FastACIS/csltest1/output/calibration_log.txt
```

---

## 常见问题

### Q1: 视频没有timecode怎么办？
**A**: 同步步骤会失败，但可以继续。直接使用原始视频进行标定。只要4个相机视频内容是同步的（手动拍板等），标定板在各视频中同时可见即可。

### Q2: cam1没有预存内参怎么办？
**A**: 三个选择：
1. 只用cam2, cam3, cam5标定（删除cam1图像文件夹）
2. 先单独标定cam1内参，然后合并JSON
3. 如果cam1设置与cam2完全相同，临时借用cam2的内参测试

### Q3: RMS误差很大（>2像素）怎么办？
**A**: 可能原因：
- 标定板检测不准确（光照不好、运动模糊）
- 相机设置与预存内参不匹配（检查HyperSmooth、镜头模式、分辨率）
- 标定板在某些帧中不够清晰
- 建议：重新拍摄标定视频，确保标定板静止且清晰

### Q4: 某个相机没有检测到标定板？
**A**: 检查：
- 该相机视频中标定板是否清晰可见
- 是否有足够的帧数包含标定板
- ChArUco板配置是否匹配（检查 `multical/asset/charuco_b3.yaml`）

### Q5: 标定后如何使用？
**A**: 将 `calibration.json` 复制到你的项目目录，运行后续的3D重建/姿态估计pipeline时指定 `--path_camera` 参数指向这个文件。

---

## 下一步

标定完成后，可以：

1. **3D人体姿态估计**：
   ```bash
   python3 scripts/run_yolo_tracking.py --recording_tag your_video/original
   python3 scripts/run_vitpose.py --recording_tag your_video/original
   python3 scripts/run_triangulation.py --recording_tag your_video/original
   ```

2. **使用标定参数**：
   在所有需要相机参数的脚本中，指定：
   ```bash
   --path_camera /Volumes/FastACIS/csltest1/output/calibration_synced/original/calibration.json
   ```

---

## 技术支持

如遇到问题：
1. 查看日志文件：`output/*_log.txt`
2. 查看可视化结果：`output/*/vis/`
3. 检查README.md和CLAUDE.md中的详细说明
4. 确认GoPro设置与预存内参匹配
