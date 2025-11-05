#!/bin/bash
# 外参标定快速脚本
# 使用场景：已同步的GoPro视频 -> 外参标定

set -e  # 遇到错误立即退出

echo "=========================================="
echo "GoPro外参标定（简化版）"
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
echo "从完整内参中提取实际使用的相机..."

cd "$WORK_DIR"

# 过滤内参JSON，只保留实际使用的相机
FILTERED_INTRINSIC="$OUTPUT_DIR/intrinsic_filtered.json"
python filter_intrinsics.py \
    --input "$INTRINSIC_JSON" \
    --output "$FILTERED_INTRINSIC" \
    --auto-detect "$SYNCED_VIDEO_DIR"

if [ ! -f "$FILTERED_INTRINSIC" ]; then
    echo "❌ 错误: 内参过滤失败"
    exit 1
fi

# 使用过滤后的内参
INTRINSIC_JSON="$FILTERED_INTRINSIC"
echo "✓ 使用过滤后的内参: $INTRINSIC_JSON"

# 统计视频
echo ""
echo "检测到的视频文件:"
video_count=$(ls "$SYNCED_VIDEO_DIR"/*.MP4 2>/dev/null | wc -l)
echo "  共 $video_count 个视频"
ls "$SYNCED_VIDEO_DIR"/*.MP4 | xargs -n 1 basename

echo ""
echo "=========================================="
echo "步骤1: 将视频转换为图像帧"
echo "=========================================="
echo "提取关键帧（5fps，前60秒）..."

cd "$WORK_DIR"

# 构建目录结构: OUTPUT_DIR/videos/camX/video.MP4
VIDEO_STAGING="$OUTPUT_DIR/videos"
mkdir -p "$VIDEO_STAGING"

for video in "$SYNCED_VIDEO_DIR"/*.MP4; do
    if [ -f "$video" ]; then
        # 获取文件名（如 cam1.MP4）
        filename=$(basename "$video")
        # 提取相机名（如 cam1）
        camname="${filename%.MP4}"

        # 创建相机目录并链接视频
        mkdir -p "$VIDEO_STAGING/$camname"
        ln -sf "$video" "$VIDEO_STAGING/$camname/$filename"
        echo "  ✓ 链接 $camname"
    fi
done

# 设置环境变量让脚本找到视频
export PATH_ASSETS_VIDEOS="$OUTPUT_DIR"

# 构建相机列表（从视频文件名提取）
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

# 转换为图像
python scripts/convert_video_to_images.py \
    --src_tag "videos" \
    --cam_tags "$CAM_LIST" \
    --fps 5 \
    --ss 5 \
    --duration 60

echo "✓ 图像提取完成"
IMAGE_DIR="$OUTPUT_DIR/videos/original"

# 统计图像数量
echo ""
echo "图像统计:"
for cam_dir in "$IMAGE_DIR"/cam*/; do
    if [ -d "$cam_dir" ]; then
        camname=$(basename "$cam_dir")
        count=$(ls "$cam_dir"/*.png 2>/dev/null | wc -l)
        echo "  $camname: $count 张"
    fi
done

echo ""
echo "=========================================="
echo "步骤2: 外参标定"
echo "=========================================="
echo "使用ChArUco板检测和预存GoPro内参..."

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
    echo "验证标定质量:"
    echo "  1. 查看 calibration.json 中的 RMS 误差（应该 < 1.0 像素）"
    echo "  2. 检查 vis/ 目录中的可视化结果"
    echo "  3. 确认所有相机都有 camera_base2cam 数据"
    echo ""
    echo "使用标定结果:"
    echo "  将 $CALIB_OUTPUT 复制到你的项目中使用"
else
    echo "❌ 标定失败"
    echo "请检查:"
    echo "  1. 是否所有相机都在内参文件中"
    echo "  2. 图像中是否能检测到ChArUco板"
    echo "  3. 查看上面的错误信息"
    exit 1
fi
