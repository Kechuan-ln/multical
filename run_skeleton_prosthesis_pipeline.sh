#!/bin/bash
# Complete pipeline: Markers -> Skeleton + Prosthesis -> Visualization

set -e  # Exit on error

echo "========================================================================"
echo "Skeleton + Prosthesis Pipeline"
echo "========================================================================"
echo ""

# Configuration
MOCAP_CSV="/Volumes/FastACIS/csltest1/gopros/Take 2025-10-20 03.12.51 PM.csv"
MARKER_LABELS="marker_labels_final.csv"
SKELETON_CONFIG="skeleton_config.json"
PROSTHESIS_CONFIG="prosthesis_config.json"
OUTPUT_JSON="skeleton_with_prosthesis.json"

# Frame range (adjust as needed)
FRAME_RANGE="3832"  # Single frame or range like "3830-3840"

echo "Step 1: Converting markers to skeleton + computing prosthesis transform..."
echo "------------------------------------------------------------------------"
python3 markers_to_skeleton_with_prosthesis.py \
    --mocap "$MOCAP_CSV" \
    --marker_labels "$MARKER_LABELS" \
    --skeleton_config "$SKELETON_CONFIG" \
    --prosthesis_config "$PROSTHESIS_CONFIG" \
    --output "$OUTPUT_JSON" \
    --frames "$FRAME_RANGE"

echo ""
echo "✓ Conversion complete!"
echo ""

echo "Step 2: Launching 3D visualization..."
echo "------------------------------------------------------------------------"
echo "Opening visualization at http://localhost:8061"
echo "Press Ctrl+C to stop the viewer"
echo ""

python3 visualize_skeleton_prosthesis.py \
    --input "$OUTPUT_JSON" \
    --port 8061

echo ""
echo "========================================================================"
echo "Pipeline complete!"
echo "========================================================================"
