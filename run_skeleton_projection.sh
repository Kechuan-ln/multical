#!/bin/bash
# Quick start script for skeleton projection to video

echo "=========================================="
echo "Skeleton Projection - Quick Start"
echo "=========================================="

# Default paths (modify these)
MCAL="/Volumes/FastACIS/csldata/csl/optitrack.mcal"
SKELETON="skeleton_joints.json"
VIDEO="/Volumes/FastACIS/csldata/video/mocap.avi"
OUTPUT="skeleton_video.mp4"

# Check if files exist
if [ ! -f "$MCAL" ]; then
    echo "ERROR: Calibration file not found: $MCAL"
    echo "Please edit this script and set the correct path."
    exit 1
fi

if [ ! -f "$SKELETON" ]; then
    echo "ERROR: Skeleton file not found: $SKELETON"
    echo "Please run markers_to_skeleton.py first."
    exit 1
fi

if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video file not found: $VIDEO"
    echo "Please edit this script and set the correct path."
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Calibration: $MCAL"
echo "  Skeleton:    $SKELETON"
echo "  Video:       $VIDEO"
echo "  Output:      $OUTPUT"
echo ""

# Parse command line arguments
START_FRAME=0
NUM_FRAMES=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-frame)
            START_FRAME="$2"
            shift 2
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--start-frame N] [--num-frames N] [--output path]"
            exit 1
            ;;
    esac
done

echo "Running projection..."
echo "  Start frame: $START_FRAME"
echo "  Num frames:  $NUM_FRAMES"
echo ""

python project_skeleton_to_video.py \
    --mcal "$MCAL" \
    --skeleton "$SKELETON" \
    --video "$VIDEO" \
    --output "$OUTPUT" \
    --start-frame $START_FRAME \
    --num-frames $NUM_FRAMES

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Success!"
    echo "=========================================="
    echo "Output saved to: $OUTPUT"
    echo ""
    echo "To view:"
    echo "  open $OUTPUT"
else
    echo ""
    echo "=========================================="
    echo "✗ Failed!"
    echo "=========================================="
    exit 1
fi
