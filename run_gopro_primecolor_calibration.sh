#!/bin/bash
# GoPro + PrimeColor 完整标定工作流程（快速启动脚本）

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         GoPro + PrimeColor 完整标定工作流程（快速启动脚本）               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"

# ============================================================================
# 配置区域 - 请根据实际情况修改这些路径
# ============================================================================

# 输入文件路径（请修改为你的实际路径）
PRIMECOLOR_MCAL="/Volumes/FastACIS/primecolor/calibration.mcal"
GOPRO_INTRINSIC="/Volumes/FastACIS/annotation_pipeline/intrinsic_hyperoff_linear_60fps.json"

# QR 码视频
QR_ANCHOR="/Volumes/FastACIS/gopro/qr_anchor.mp4"
GOPRO_QR="/Volumes/FastACIS/gopro/cam1_qr.mp4"
PRIMECOLOR_QR="/Volumes/FastACIS/primecolor/primecolor_qr.mp4"

# ChArUco 标定板视频
GOPRO_CHARUCO="/Volumes/FastACIS/gopro/cam1_charuco.mp4"
PRIMECOLOR_CHARUCO="/Volumes/FastACIS/primecolor/primecolor_charuco.mp4"

# 输出目录
OUTPUT_DIR="calibration_output/gopro_primecolor"

# 可选参数
BOARD="./multical/asset/charuco_b3.yaml"  # 标定板配置
GOPRO_CAMERA_NAME="cam2"                  # GoPro 相机名称（如果内参文件包含多个相机）
SCAN_DURATION=30                          # QR 码扫描时长（秒）
QR_STEP=5                                 # QR 码检测步长
EXTRINSIC_FPS=1.0                         # 外参标定提取帧率
EXTRINSIC_MAX_FRAMES=100                  # 外参标定最大帧数

# ============================================================================
# 检查输入文件
# ============================================================================

echo -e "\n${YELLOW}检查输入文件...${NC}"

check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}❌ 文件不存在: $2${NC}"
        echo -e "${RED}   路径: $1${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ $2${NC}"
}

check_file "$PRIMECOLOR_MCAL" "PrimeColor .mcal 文件"
check_file "$GOPRO_INTRINSIC" "GoPro 内参文件"
check_file "$QR_ANCHOR" "QR Anchor 视频"
check_file "$GOPRO_QR" "GoPro QR 视频"
check_file "$PRIMECOLOR_QR" "PrimeColor QR 视频"
check_file "$GOPRO_CHARUCO" "GoPro ChArUco 视频"
check_file "$PRIMECOLOR_CHARUCO" "PrimeColor ChArUco 视频"

# ============================================================================
# 激活 conda 环境
# ============================================================================

echo -e "\n${YELLOW}激活 conda 环境 (multical)...${NC}"

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate multical || {
        echo -e "${RED}❌ 无法激活 multical 环境${NC}"
        echo -e "${YELLOW}请先创建环境: conda create -n multical python==3.10${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ conda 环境已激活${NC}"
else
    echo -e "${YELLOW}⚠️  conda 未找到，跳过环境激活${NC}"
fi

# ============================================================================
# 运行 Python 脚本
# ============================================================================

echo -e "\n${BLUE}═════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}开始运行标定流程...${NC}"
echo -e "${BLUE}═════════════════════════════════════════════════════════════════════════${NC}"

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

# 检查是否成功
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                            ✅ 标定完成！                                    ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"

    echo -e "\n${BLUE}📁 输出文件位置:${NC}"
    echo -e "  ${OUTPUT_DIR}/"

    echo -e "\n${BLUE}🎯 最终标定文件:${NC}"
    CALIBRATION_JSON="${OUTPUT_DIR}/extrinsics/frames/calibration.json"
    if [ -f "$CALIBRATION_JSON" ]; then
        echo -e "  ${GREEN}${CALIBRATION_JSON}${NC}"

        # 显示 RMS 误差
        RMS=$(cat "$CALIBRATION_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('rms', 'N/A'))" 2>/dev/null || echo "N/A")
        echo -e "\n  RMS 误差: ${GREEN}${RMS}${NC} 像素"
    else
        echo -e "  ${RED}未找到 calibration.json${NC}"
    fi

    echo -e "\n${BLUE}📊 验证步骤:${NC}"
    echo -e "  1. 播放同步验证视频:"
    echo -e "     ${YELLOW}open ${OUTPUT_DIR}/sync/gopro_verify.mp4${NC}"
    echo -e "     ${YELLOW}open ${OUTPUT_DIR}/sync/primecolor_verify.mp4${NC}"
    echo -e "  2. 查看标定可视化结果:"
    echo -e "     ${YELLOW}open ${OUTPUT_DIR}/extrinsics/frames/vis/${NC}"
    echo -e "  3. 检查 calibration.json:"
    echo -e "     ${YELLOW}cat ${CALIBRATION_JSON} | python -m json.tool${NC}"

    echo -e "\n${BLUE}下一步:${NC}"
    echo -e "  将 calibration.json 复制到你的录制目录，然后运行 3D 重建。"

else
    echo -e "\n${RED}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                            ❌ 标定失败！                                    ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "\n请查看上面的错误信息并修复问题。"
    exit 1
fi
