#!/bin/bash
# 外参标定脚本（可自定义时间段）
# 使用场景：已同步的GoPro视频 -> 外参标定

set -e  # 遇到错误立即退出

echo "=========================================="
echo "GoPro外参标定（自定义时间段）"
echo "=========================================="

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate multical

# 配置（根据你的实际情况修改）
SYNCED_VIDEO_DIR="/Volumes/FastACIS/gorpos-2-sync/gorpos-2"
OUTPUT_DIR="/Volumes/FastACIS/gorpos-2-calibration"
INTRINSIC_JSON="/Volumes/FastACIS/annotation_pipeline/intrinsic_hyperoff_linear_60fps.json"
BOARD_YAML="/Volumes/FastACIS/annotation_pipeline/multical/asset/charuco_b1_2.yaml"
WORK_DIR="/Volumes/FastACIS/annotation_pipeline"

# ⚠️ 重要：标定板出现的时间段
START_TIME=110         # 1分50秒 = 110秒
DURATION=260           # 6分10秒 - 1分50秒 = 4分20秒 = 260秒
SAMPLE_FPS=2           # 采样帧率（2fps = 每0.5秒一帧，共520帧）

echo "标定板时间段: ${START_TIME}秒 - $((START_TIME + DURATION))秒"
echo "采样帧率: ${SAMPLE_FPS} fps"
echo "预计提取帧数: $((DURATION * SAMPLE_FPS)) 帧/相机"

# 检查输入
if [ ! -d "$SYNCED_VIDEO_DIR" ]; then
    echo "❌ 错误: 同步视频目录不存在: $SYNCED_VIDEO_DIR"
    exit 1
fi

if [ ! -f "$INTRINSIC_JSON" ]; then
    echo "❌ 错误: 内参文件不存在: $INTRINSIC_JSON"
    exit 1
fi

if [ ! -f "$BOARD_YAML" ]; then
    echo "❌ 错误: Board配置文件不存在: $BOARD_YAML"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "✓ 输出目录: $OUTPUT_DIR"

echo ""
echo "=========================================="
echo "步骤0: 过滤内参文件"
echo "=========================================="
cd "$WORK_DIR"

FILTERED_INTRINSIC="$OUTPUT_DIR/intrinsic_filtered.json"
if [ ! -f "$FILTERED_INTRINSIC" ]; then
    echo "生成过滤后的内参..."
    python filter_intrinsics.py \
        --input "$INTRINSIC_JSON" \
        --output "$FILTERED_INTRINSIC" \
        --auto-detect "$SYNCED_VIDEO_DIR"
else
    echo "✓ 使用已有的过滤内参: $FILTERED_INTRINSIC"
fi

INTRINSIC_JSON="$FILTERED_INTRINSIC"

# 统计视频
echo ""
echo "检测到的视频文件:"
video_count=$(ls "$SYNCED_VIDEO_DIR"/*.MP4 2>/dev/null | wc -l)
echo "  共 $video_count 个视频"

echo ""
echo "=========================================="
echo "步骤1: 提取标定板时间段的图像"
echo "=========================================="

cd "$WORK_DIR"

# 构建目录结构
VIDEO_STAGING="$OUTPUT_DIR/videos"
mkdir -p "$VIDEO_STAGING"

# 创建符号链接
for video in "$SYNCED_VIDEO_DIR"/*.MP4; do
    if [ -f "$video" ]; then
        filename=$(basename "$video")
        camname="${filename%.MP4}"
        mkdir -p "$VIDEO_STAGING/$camname"
        ln -sf "$video" "$VIDEO_STAGING/$camname/$filename" 2>/dev/null || true
    fi
done

export PATH_ASSETS_VIDEOS="$OUTPUT_DIR"

# 构建相机列表
CAM_LIST=""
for video in "$SYNCED_VIDEO_DIR"/*.MP4; do
    if [ -f "$video" ]; then
        camname=$(basename "$video" .MP4)
        if [ -z "$CAM_LIST" ]; then
            CAM_LIST="$camname"
        else
            CAM_LIST="$CAM_LIST,$camname"
        fi
    fi
done

echo "相机列表: $CAM_LIST"
echo "提取时间段: ${START_TIME}秒 - $((START_TIME + DURATION))秒"
echo "采样帧率: ${SAMPLE_FPS} fps"

# 转换为图像
python scripts/convert_video_to_images.py \
    --src_tag "videos" \
    --cam_tags "$CAM_LIST" \
    --fps $SAMPLE_FPS \
    --ss $START_TIME \
    --duration $DURATION

echo "✓ 图像提取完成"
IMAGE_DIR="$OUTPUT_DIR/videos/original"

# 统计图像数量
echo ""
echo "图像统计:"
total_images=0
for cam_dir in "$IMAGE_DIR"/cam*/; do
    if [ -d "$cam_dir" ]; then
        camname=$(basename "$cam_dir")
        count=$(ls "$cam_dir"/*.png 2>/dev/null | wc -l)
        echo "  $camname: $count 张"
        total_images=$((total_images + count))
    fi
done
echo "  总计: $total_images 张图像"

if [ $total_images -eq 0 ]; then
    echo ""
    echo "❌ 错误: 没有提取到任何图像！"
    echo "可能原因:"
    echo "  1. 视频路径不正确"
    echo "  2. 时间参数设置错误"
    echo "  3. ffmpeg命令失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "步骤2: 外参标定"
echo "=========================================="
echo "使用ChArUco板检测和预存GoPro内参..."

mkdir -p "$IMAGE_DIR"

cd "$WORK_DIR/multical"

# 运行标定
python calibrate.py \
    --boards "$BOARD_YAML" \
    --image_path "$IMAGE_DIR" \
    --calibration "$INTRINSIC_JSON" \
    --fix_intrinsic \
    --limit_images 300 \
    --vis

CALIB_OUTPUT="$IMAGE_DIR/calibration.json"

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="

if [ -f "$CALIB_OUTPUT" ]; then
    echo "✅ 外参标定成功！"
    echo ""
    echo "输出文件:"
    echo "  📄 标定结果: $CALIB_OUTPUT"
    echo "  🖼️  图像帧: $IMAGE_DIR/"
    echo "  📊 可视化: $OUTPUT_DIR/videos/vis/"
    echo ""

    # 提取RMS信息
    if command -v jq &> /dev/null; then
        echo "标定质量检查:"
        # 使用jq提取RMS（如果安装了）
        rms=$(jq -r '.rms // "N/A"' "$CALIB_OUTPUT" 2>/dev/null || echo "无法读取")
        echo "  RMS: $rms 像素（应该 < 1.0）"
    fi

    echo ""
    echo "验证标定:"
    echo "  1. 查看可视化图像（vis/目录），检查3D坐标轴是否正确投影"
    echo "  2. 打开 calibration.json，检查 RMS 误差"
    echo "  3. 确认所有相机都有 camera_base2cam 数据"
    echo ""
    echo "使用标定结果:"
    echo "  cp $CALIB_OUTPUT <你的项目目录>/"
else
    echo "❌ 标定失败"
    echo ""
    echo "可能原因:"
    echo "  1. 检测到的ChArUco板太少（< 10帧）"
    echo "  2. 标定板角点检测质量差"
    echo "  3. 多相机之间没有足够的重叠视野"
    echo ""
    echo "建议:"
    echo "  1. 查看 $IMAGE_DIR/*.png，确认能看到清晰的ChArUco板"
    echo "  2. 调整 START_TIME 和 DURATION 参数"
    echo "  3. 降低 SAMPLE_FPS（如改为1fps）减少模糊帧"
    echo "  4. 查看 $IMAGE_DIR/calibration.txt 日志"
    exit 1
fi
