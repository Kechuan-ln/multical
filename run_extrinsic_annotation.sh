#!/bin/bash
#
# Interactive extrinsics calibration tool launcher
#
# This tool allows manual refinement of camera extrinsics by annotating
# 2D-3D correspondences between video frames and mocap markers.
#
# Usage:
#   ./run_extrinsic_annotation.sh [start_frame]
#
# Example:
#   ./run_extrinsic_annotation.sh 100
#

set -e

# ======== Configuration ========

# Data paths
CSV_FILE="/Volumes/FastACIS/GoPro/motion/mocap/mocap.csv"
VIDEO_FILE="/Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi"
MCAL_FILE="/Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal"

# Intrinsics - Choose which one to use:
# Option 1: Use .mcal intrinsics (OptiTrack calibration)
# INTRINSICS_FILE="$MCAL_FILE"

# Option 2: Use newly calibrated intrinsics (recommended if available)
INTRINSICS_FILE="/Volumes/FastACIS/annotation_pipeline/primecolor_intrinsic_test/frames/intrinsic.json"

# Camera settings
CAMERA_SERIAL="C11764"
PORT=8050

# Frame range
START_FRAME=${1:-0}

# ======== Validation ========

echo "========================================"
echo "Interactive Extrinsics Calibration Tool"
echo "========================================"
echo ""

if [ ! -f "$CSV_FILE" ]; then
    echo "❌ Error: CSV file not found: $CSV_FILE"
    exit 1
fi

if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Error: Video file not found: $VIDEO_FILE"
    exit 1
fi

if [ ! -f "$MCAL_FILE" ]; then
    echo "❌ Error: .mcal file not found: $MCAL_FILE"
    exit 1
fi

if [ ! -f "$INTRINSICS_FILE" ]; then
    echo "❌ Error: Intrinsics file not found: $INTRINSICS_FILE"
    exit 1
fi

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/annotate_extrinsics_interactive.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Script not found: $SCRIPT_PATH"
    exit 1
fi

# ======== Display Configuration ========

echo "📋 Configuration:"
echo "  CSV file:       $CSV_FILE"
echo "  Video file:     $VIDEO_FILE"
echo "  .mcal file:     $MCAL_FILE"
echo "  Intrinsics:     $INTRINSICS_FILE"
echo "  Camera:         $CAMERA_SERIAL"
echo "  Start frame:    $START_FRAME"
echo "  Web port:       $PORT"
echo ""

# Check conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda environment is active"
    echo "   Consider activating: conda activate multical"
    echo ""
fi

# ======== Run Tool ========

echo "🚀 Starting extrinsics calibration tool..."
echo ""
echo "📖 How to use:"
echo "   1. Click a marker in 3D view (left) to select it - turns RED"
echo "   2. Click the corresponding location in 2D view (right)"
echo "   3. Repeat for 6+ points"
echo "   4. Click 'Recompute Extrinsics' to optimize camera pose"
echo "   5. Continue adding points or switch frames as needed"
echo "   6. Click 'Save Extrinsics' when satisfied"
echo ""
echo "   Open browser: http://localhost:$PORT"
echo ""
echo "   Press Ctrl+C to stop"
echo "========================================"
echo ""

python "$SCRIPT_PATH" \
    --csv "$CSV_FILE" \
    --video "$VIDEO_FILE" \
    --mcal "$MCAL_FILE" \
    --intrinsics "$INTRINSICS_FILE" \
    --camera_serial "$CAMERA_SERIAL" \
    --start_frame "$START_FRAME" \
    --port "$PORT"
