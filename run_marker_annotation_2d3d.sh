#!/bin/bash

# Quick start script for 2D+3D marker annotation tool
#
# Usage:
#   ./run_marker_annotation_2d3d.sh [start_frame] [num_frames]
#
# Examples:
#   ./run_marker_annotation_2d3d.sh           # Start from frame 0, load 500 frames
#   ./run_marker_annotation_2d3d.sh 100       # Start from frame 100, load 500 frames
#   ./run_marker_annotation_2d3d.sh 100 200   # Start from frame 100, load 200 frames

set -e  # Exit on error

# ======== Configuration ========

# Data paths (GoPro motion capture data)
CSV_FILE="/Volumes/FastACIS/GoPro/motion/mocap/mocap.csv"
VIDEO_FILE="/Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi"
MCAL_FILE="/Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal"  # For default intrinsics

# Optional: Custom calibration files (leave empty to use .mcal)
# To use custom calibrations, uncomment and set paths:
INTRINSICS_FILE=""  # e.g., "primecolor_intrinsic_test/frames/intrinsic.json"
EXTRINSICS_FILE=""  # e.g., "extrinsics_calibrated.json"

# Example with custom calibrations:
# INTRINSICS_FILE="primecolor_intrinsic_test/frames/intrinsic.json"
# EXTRINSICS_FILE="extrinsics_calibrated.json"

# Parameters
CAMERA_SERIAL="C11764"  # PrimeColor camera serial
LABELS_FILE="marker_labels.csv"
PORT=8050

# Frame range (can be overridden by command line args)
START_FRAME=${1:-0}
NUM_FRAMES=${2:-500}

# ======== Validation ========

echo "========================================"
echo "Mocap Marker 2D+3D Annotation Tool"
echo "========================================"
echo ""

# Check if files exist
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ Error: CSV file not found: $CSV_FILE"
    exit 1
fi

if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Error: Video file not found: $VIDEO_FILE"
    exit 1
fi

if [ ! -f "$MCAL_FILE" ]; then
    echo "❌ Error: Calibration file not found: $MCAL_FILE"
    exit 1
fi

# Check if script exists
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/annotate_mocap_markers_2d3d.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Script not found: $SCRIPT_PATH"
    exit 1
fi

# ======== Display Configuration ========

echo "📋 Configuration:"
echo "  CSV file:       $CSV_FILE"
echo "  Video file:     $VIDEO_FILE"
echo "  .mcal file:     $MCAL_FILE"
if [ -n "$INTRINSICS_FILE" ]; then
    echo "  Intrinsics:     $INTRINSICS_FILE (custom)"
else
    echo "  Intrinsics:     (from .mcal)"
fi
if [ -n "$EXTRINSICS_FILE" ]; then
    echo "  Extrinsics:     $EXTRINSICS_FILE (custom)"
else
    echo "  Extrinsics:     (from .mcal)"
fi
echo "  Camera:         $CAMERA_SERIAL"
echo "  Start frame:    $START_FRAME"
echo "  Num frames:     $NUM_FRAMES"
echo "  Labels file:    $LABELS_FILE"
echo "  Web port:       $PORT"
echo ""

# Check if conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda environment is active"
    echo "   Consider activating: conda activate multical"
    echo ""
fi

# ======== Run Tool ========

echo "🚀 Starting annotation tool..."
echo "   Open browser: http://localhost:$PORT"
echo ""
echo "   Press Ctrl+C to stop"
echo "========================================"
echo ""

# Build command with optional parameters
CMD="python \"$SCRIPT_PATH\" \
    --csv \"$CSV_FILE\" \
    --video \"$VIDEO_FILE\" \
    --mcal \"$MCAL_FILE\" \
    --camera_serial \"$CAMERA_SERIAL\" \
    --start_frame \"$START_FRAME\" \
    --num_frames \"$NUM_FRAMES\" \
    --labels \"$LABELS_FILE\" \
    --port \"$PORT\""

# Add optional intrinsics if specified
if [ -n "$INTRINSICS_FILE" ]; then
    CMD="$CMD --intrinsics \"$INTRINSICS_FILE\""
fi

# Add optional extrinsics if specified
if [ -n "$EXTRINSICS_FILE" ]; then
    CMD="$CMD --extrinsics \"$EXTRINSICS_FILE\""
fi

# Execute the command
eval $CMD
