# 快速开始：Markers → Skeleton → Video

## 🚀 三步完成可视化

### 1️⃣ 转换 Markers 到 Skeleton（允许部分缺失）

```bash
python markers_to_skeleton.py \
  --mocap_csv /Volumes/FastACIS/csldata/csl/mocap.csv \
  --labels_csv marker_labels.csv \
  --start_frame 2 \
  --end_frame 23374
```

**注意**：Missing markers 是正常的！脚本会：
- ✅ 计算所有有足够markers的关节
- ⚠️  跳过markers缺失的关节（只显示警告）
- ✅ 继续处理，不会中止

**输出**：
- `skeleton_joints.csv` - CSV格式
- `skeleton_joints.json` - JSON格式

### 2️⃣ 生成视频 (MP4)

```bash
python visualize_skeleton_video.py \
  --skeleton_json skeleton_joints.json \
  --output skeleton.mp4 \
  --num_frames 200 \
  --fps 30 \
  --dpi 100
```

**参数说明**：
- `--num_frames 200`: 渲染200帧（约6.7秒 @30fps）
- `--fps 30`: 输出30帧/秒
- `--dpi 100`: 分辨率（更高=更清晰但文件更大）

**推荐设置**：
- 快速预览：`--num_frames 100 --fps 30 --dpi 80`
- 标准质量：`--num_frames 200 --fps 30 --dpi 100`
- 高质量：`--num_frames 500 --fps 60 --dpi 150`

### 3️⃣ 或生成 GIF

```bash
python visualize_skeleton_video.py \
  --skeleton_json skeleton_joints.json \
  --output skeleton.gif \
  --num_frames 100 \
  --fps 20 \
  --dpi 80
```

**GIF建议**：
- 帧数少一点（100-150帧）
- FPS低一点（15-20 fps）
- DPI低一点（80-100）
- 否则GIF文件会很大

## 📊 输出示例

### MP4 视频
- **文件大小**: ~5-20 MB (200帧, dpi=100)
- **质量**: 高，支持流畅播放
- **适合**: 分享、演示、详细分析

### GIF 动画
- **文件大小**: ~10-50 MB (100帧, dpi=80)
- **质量**: 中等
- **适合**: 网页嵌入、快速预览

## 🎨 骨架颜色

- 🔵 **蓝色**: 脊柱/躯干
- 🟣 **紫色**: 头部/下颌
- 🟢 **绿色**: 左臂
- 🔴 **红色**: 右臂
- 🔵 **青色**: 左腿
- 🟠 **橙色**: 右腿

关节点颜色：按高度（Y值）着色

## ⚙️ 依赖检查

### FFmpeg（生成MP4需要）

检查是否已安装：
```bash
ffmpeg -version
```

如果未安装：
```bash
conda install -c conda-forge ffmpeg
```

### Matplotlib

应该已经在 multical 环境中安装了。如果没有：
```bash
pip install matplotlib
```

## 🔧 常见问题

### Q: Missing markers 警告很多？
**A**: 这是正常的！只要有足够的markers能计算出部分关节就可以。脚本会跳过缺失的关节继续处理。

### Q: MP4 生成失败？
**A**:
1. 检查 ffmpeg 是否安装：`ffmpeg -version`
2. 如果没有，安装：`conda install -c conda-forge ffmpeg`
3. 或者改用 GIF：`--output skeleton.gif`

### Q: 视频/GIF 文件太大？
**A**: 降低参数：
- 减少帧数：`--num_frames 100`
- 降低 FPS：`--fps 20`
- 降低分辨率：`--dpi 80`

### Q: 视频生成太慢？
**A**:
- 减少帧数是最有效的方法
- 降低 DPI 也能加速
- 200帧 @ dpi=100 大约需要 1-3 分钟

### Q: 骨架看起来不对？
**A**:
1. 检查 skeleton_joints.json 中的关节数量
2. 可能是某些关键关节缺失（如 Pelvis, LHip, RHip）
3. 尝试标注更多的 markers

## 📝 完整流程示例

```bash
# 1. 标注 markers（交互式）
python annotate_mocap_markers.py --start_frame 2 --num_frames 200

# 2. 转换为 skeleton
python markers_to_skeleton.py \
  --mocap_csv /Volumes/FastACIS/csldata/csl/mocap.csv \
  --labels_csv marker_labels.csv \
  --start_frame 2 \
  --end_frame 1000

# 3. 生成预览视频（快速）
python visualize_skeleton_video.py \
  --skeleton_json skeleton_joints.json \
  --output preview.mp4 \
  --num_frames 100 \
  --fps 30

# 4. 如果满意，生成完整视频
python visualize_skeleton_video.py \
  --skeleton_json skeleton_joints.json \
  --output full_skeleton.mp4 \
  --num_frames 1000 \
  --fps 30 \
  --dpi 120
```

## 🎯 性能优化

### 对于长序列（>1000帧）

**选项1**: 分段处理
```bash
# 前500帧
python markers_to_skeleton.py --start_frame 2 --end_frame 502
python visualize_skeleton_video.py --output part1.mp4 --num_frames 500

# 后500帧
python markers_to_skeleton.py --start_frame 502 --end_frame 1002
python visualize_skeleton_video.py --output part2.mp4 --num_frames 500
```

**选项2**: 降采样
```bash
# 只处理每N帧
# 修改 markers_to_skeleton.py 添加 --frame_step 参数
```

**选项3**: 降低输出质量
```bash
python visualize_skeleton_video.py \
  --num_frames 2000 \
  --fps 20 \
  --dpi 80
```

## 📂 输出文件清单

运行完整流程后，你会得到：

```
marker_labels.csv          # Marker 标注结果
skeleton_joints.csv        # Skeleton CSV格式
skeleton_joints.json       # Skeleton JSON格式
skeleton.mp4              # Skeleton 视频
skeleton.gif              # Skeleton GIF（可选）
```
