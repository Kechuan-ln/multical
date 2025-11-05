#!/bin/bash
# GoPro外参标定完整流程
# 使用场景：4个GoPro相机的calibration.MP4视频
# 使用GoPro timecode进行同步

set -e  # 遇到错误立即退出

echo "=========================================="
echo "GoPro外参标定工作流"
echo "=========================================="
echo "使用multical conda环境"
echo "使用GoPro timecode同步"
echo ""

# 激活conda环境
echo "激活multical conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate multical

# 验证环境
echo "验证工具..."
which python
which ffprobe
python --version
echo ""

# 配置
SOURCE_DIR="/Volumes/FastACIS/csltest1/gopros"
OUTPUT_DIR="/Volumes/FastACIS/csltest1/output"
INTRINSIC_JSON="/Volumes/FastACIS/annotation_pipeline/intrinsic_hyperoff_linear_60fps.json"
CAMERAS="cam1,cam2,cam3,cam5"
WORK_DIR="/Volumes/FastACIS/annotation_pipeline"

# 检查源目录
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源目录不存在: $SOURCE_DIR"
    exit 1
fi

# 检查内参文件
if [ ! -f "$INTRINSIC_JSON" ]; then
    echo "错误: 内参文件不存在: $INTRINSIC_JSON"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "✓ 输出目录: $OUTPUT_DIR"

# 检查calibration.MP4文件
echo ""
echo "检查视频文件..."
for cam in cam1 cam2 cam3 cam5; do
    video="$SOURCE_DIR/$cam/calibration.MP4"
    if [ -f "$video" ]; then
        size=$(ls -lh "$video" | awk '{print $5}')
        echo "  ✓ $cam/calibration.MP4 ($size)"
    else
        echo "  ❌ $cam/calibration.MP4 不存在"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "步骤1: 准备视频目录结构"
echo "=========================================="

# 创建符合pipeline要求的目录结构
VIDEO_DIR="$OUTPUT_DIR/calibration_videos"
mkdir -p "$VIDEO_DIR"

for cam in cam1 cam2 cam3 cam5; do
    mkdir -p "$VIDEO_DIR/$cam"
    # 创建符号链接而不是复制（节省空间）
    ln -sf "$SOURCE_DIR/$cam/calibration.MP4" "$VIDEO_DIR/$cam/calibration.MP4"
    echo "  ✓ 链接 $cam/calibration.MP4"
done

echo ""
echo "检查视频timecode..."
for cam in cam1 cam2 cam3 cam5; do
    video="$VIDEO_DIR/$cam/calibration.MP4"
    timecode=$(ffprobe -v error -select_streams v:0 -show_entries stream_tags=timecode -of default=noprint_wrappers=1:nokey=1 "$video" 2>&1)
    if [ -z "$timecode" ]; then
        echo "  ❌ $cam: 无timecode"
        echo ""
        echo "错误: GoPro视频必须有timecode才能同步！"
        echo "请检查："
        echo "  1. GoPro是否开启了timecode功能"
        echo "  2. 视频是否是原始GoPro录制文件"
        exit 1
    else
        echo "  ✓ $cam: $timecode"
    fi
done

echo ""
echo "=========================================="
echo "步骤2: GoPro Timecode同步"
echo "=========================================="
echo "使用ffprobe提取timecode并同步视频..."

cd "$WORK_DIR"

# 修改PATH_ASSETS_VIDEOS临时指向输出目录
export PATH_ASSETS_VIDEOS="$OUTPUT_DIR"

# 同步视频（使用相对路径）
python scripts/sync_timecode.py \
    --src_tag "calibration_videos" \
    --out_tag "calibration_synced" \
    2>&1 | tee "$OUTPUT_DIR/sync_log.txt"

echo "✓ 视频同步成功"
VIDEO_SOURCE="calibration_synced"

# 显示同步信息
if [ -f "$OUTPUT_DIR/calibration_synced/meta_info.json" ]; then
    echo ""
    echo "同步信息:"
    cat "$OUTPUT_DIR/calibration_synced/meta_info.json" | python -m json.tool | grep -A 5 "offset\|duration" | head -20
fi

echo ""
echo "=========================================="
echo "步骤3: 转换视频为图像帧"
echo "=========================================="
echo "提取关键帧用于标定（5fps，持续60秒）"

cd "$WORK_DIR"

# 转换为图像（使用相对路径）
python scripts/convert_video_to_images.py \
    --src_tag "$VIDEO_SOURCE" \
    --cam_tags "$CAMERAS" \
    --fps 5 \
    --ss 5 \
    --duration 60 \
    2>&1 | tee "$OUTPUT_DIR/convert_log.txt"

echo "✓ 图像提取完成"
echo "  位置: $OUTPUT_DIR/${VIDEO_SOURCE}/original/"

# 统计提取的图像数量
echo ""
echo "图像统计:"
for cam in cam1 cam2 cam3 cam5; do
    if [ -d "$OUTPUT_DIR/${VIDEO_SOURCE}/original/$cam" ]; then
        count=$(ls "$OUTPUT_DIR/${VIDEO_SOURCE}/original/$cam"/*.png 2>/dev/null | wc -l)
        echo "  $cam: $count 张图像"
    fi
done

echo ""
echo "=========================================="
echo "步骤4: 外参标定"
echo "=========================================="
echo "使用预存的GoPro内参 + ChArUco板检测"
echo ""
echo "⚠️  注意: cam1不在预存内参文件中"
echo "   选项1: 手动从图像目录删除cam1（只用cam2,3,5标定）"
echo "   选项2: 先标定cam1内参再合并"
echo "   选项3: 暂时使用cam2内参替代cam1（如果设置相同）"
echo ""
read -p "是否删除cam1继续？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "删除cam1图像..."
    rm -rf "$OUTPUT_DIR/${VIDEO_SOURCE}/original/cam1"
    echo "✓ 已删除cam1，将只用cam2, cam3, cam5进行标定"
else
    echo "保留cam1，标定可能失败（cam1无内参）"
fi

echo ""
echo "开始外参标定..."

# 进入multical目录
cd "$WORK_DIR/multical"

# 临时设置PATH_ASSETS_VIDEOS
export PATH_ASSETS_VIDEOS="$OUTPUT_DIR"

# 运行外参标定
python calibrate.py \
    --boards ./asset/charuco_b3.yaml \
    --image_path "${VIDEO_SOURCE}/original" \
    --calibration "$INTRINSIC_JSON" \
    --fix_intrinsic \
    --limit_images 300 \
    --vis \
    2>&1 | tee "$OUTPUT_DIR/calibration_log.txt"

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="

CALIB_OUTPUT="$OUTPUT_DIR/${VIDEO_SOURCE}/original/calibration.json"
if [ -f "$CALIB_OUTPUT" ]; then
    echo "✅ 外参标定成功！"
    echo ""
    echo "输出文件:"
    echo "  - 标定结果: $CALIB_OUTPUT"
    echo "  - 图像帧: $OUTPUT_DIR/${VIDEO_SOURCE}/original/"
    echo "  - 可视化: $OUTPUT_DIR/${VIDEO_SOURCE}/vis/"
    echo "  - 日志: $OUTPUT_DIR/*.txt"
    echo ""
    echo "下一步:"
    echo "  1. 查看可视化结果验证标定质量"
    echo "  2. 检查calibration.json中的RMS误差（应该<1像素）"
    echo "  3. 如果满意，将calibration.json用于后续的3D重建"
else
    echo "❌ 标定失败，请查看日志文件"
    echo "  - $OUTPUT_DIR/calibration_log.txt"
    exit 1
fi
