#!/bin/bash
# GoPro + PrimeColor 标定配置示例
# 复制此文件并根据你的实际路径修改

# ============================================================================
# 步骤 1: 修改下面的路径为你的实际文件位置
# ============================================================================

# PrimeColor 内参源文件（从 Motive/OptiTrack 导出的 .mcal 文件）
export PRIMECOLOR_MCAL="/Volumes/FastACIS/primecolor/calibration.mcal"

# GoPro 内参文件（预计算的内参 JSON）
export GOPRO_INTRINSIC="/Volumes/FastACIS/annotation_pipeline/intrinsic_hyperoff_linear_60fps.json"

# QR 码视频（用于时间同步）
export QR_ANCHOR="/Volumes/FastACIS/gopro/qr_anchor.mp4"           # QR 码 anchor 视频（参考基准）
export GOPRO_QR="/Volumes/FastACIS/gopro/cam1_qr.mp4"             # GoPro 录制的 QR 码视频
export PRIMECOLOR_QR="/Volumes/FastACIS/primecolor/primecolor_qr.mp4"  # PrimeColor 录制的 QR 码视频

# ChArUco 标定板视频（用于外参标定）
export GOPRO_CHARUCO="/Volumes/FastACIS/gopro/cam1_charuco.mp4"        # GoPro 录制的标定板视频
export PRIMECOLOR_CHARUCO="/Volumes/FastACIS/primecolor/primecolor_charuco.mp4"  # PrimeColor 录制的标定板视频

# 输出目录
export OUTPUT_DIR="calibration_output/gopro_primecolor"

# ============================================================================
# 步骤 2: 根据需要调整这些参数
# ============================================================================

# GoPro 相机名称（如果内参文件包含多个相机，指定要使用的相机）
# 常见值: cam1, cam2, cam3, cam4
export GOPRO_CAMERA_NAME="cam2"

# 标定板配置文件
# 选项: ./multical/asset/charuco_b3.yaml (B3, 5x9 网格, 50mm 方格)
#       ./multical/asset/charuco_b1_2.yaml (B1, 10x14 网格, 70mm 方格)
export BOARD="./multical/asset/charuco_b3.yaml"

# QR 码同步参数
export SCAN_DURATION=30      # 扫描前 N 秒的 QR 码（默认 30）
export QR_STEP=5             # 每 N 帧检测一次 QR 码（默认 5，降低可提高精度但变慢）

# 外参标定参数
export EXTRINSIC_FPS=1.0     # 每秒提取 N 帧用于外参标定（默认 1.0）
export EXTRINSIC_MAX_FRAMES=100  # 最多提取 N 帧（默认 100）

# ============================================================================
# 步骤 3: 运行标定脚本
# ============================================================================

echo "配置已加载，准备运行标定..."
echo ""
echo "输入文件:"
echo "  PrimeColor .mcal: $PRIMECOLOR_MCAL"
echo "  GoPro 内参:       $GOPRO_INTRINSIC"
echo "  QR Anchor:        $QR_ANCHOR"
echo "  GoPro QR:         $GOPRO_QR"
echo "  PrimeColor QR:    $PRIMECOLOR_QR"
echo "  GoPro ChArUco:    $GOPRO_CHARUCO"
echo "  PrimeColor ChArUco: $PRIMECOLOR_CHARUCO"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "按 Enter 继续，或 Ctrl+C 取消..."
read

# 运行 Python 标定脚本
python run_gopro_primecolor_calibration.py \
  --primecolor-mcal "$PRIMECOLOR_MCAL" \
  --gopro-intrinsic "$GOPRO_INTRINSIC" \
  --qr-anchor "$QR_ANCHOR" \
  --gopro-qr "$GOPRO_QR" \
  --primecolor-qr "$PRIMECOLOR_QR" \
  --gopro-charuco "$GOPRO_CHARUCO" \
  --primecolor-charuco "$PRIMECOLOR_CHARUCO" \
  --output-dir "$OUTPUT_DIR" \
  --board "$BOARD" \
  --gopro-camera-name "$GOPRO_CAMERA_NAME" \
  --scan-duration "$SCAN_DURATION" \
  --qr-step "$QR_STEP" \
  --extrinsic-fps "$EXTRINSIC_FPS" \
  --extrinsic-max-frames "$EXTRINSIC_MAX_FRAMES"

# ============================================================================
# 完成后的验证步骤
# ============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 标定完成！"
    echo ""
    echo "下一步验证:"
    echo "  1. 查看同步验证视频:"
    echo "     open ${OUTPUT_DIR}/sync/gopro_verify.mp4"
    echo "     open ${OUTPUT_DIR}/sync/primecolor_verify.mp4"
    echo ""
    echo "  2. 查看标定可视化:"
    echo "     open ${OUTPUT_DIR}/extrinsics/frames/vis/"
    echo ""
    echo "  3. 检查 RMS 误差:"
    echo "     cat ${OUTPUT_DIR}/extrinsics/frames/calibration.json | python -c \"import sys, json; print('RMS:', json.load(sys.stdin)['rms'])\""
    echo ""
    echo "  4. 将 calibration.json 复制到录制目录:"
    echo "     cp ${OUTPUT_DIR}/extrinsics/frames/calibration.json /path/to/your/recording/original/"
fi
