#!/bin/bash
# Convenience script to run marker projection for primecolor data

MOCAP_CSV="/Volumes/FastACIS/primecolor/motions with markers.csv"
VIDEO="/Volumes/FastACIS/primecolor/motions with markers-Camera 13 (C11764).avi"
INTRINSIC="/Volumes/FastACIS/primecolor/intrinsic.json"
OUTPUT="/Volumes/FastACIS/primecolor/output_with_markers.mp4"

# Test run with first 300 frames
echo "Running marker projection (first 300 frames for testing)..."
python sync_and_project_markers.py \
  --mocap_csv "$MOCAP_CSV" \
  --video "$VIDEO" \
  --intrinsic "$INTRINSIC" \
  --output "$OUTPUT" \
  --camera_name cam0 \
  --offset_frames 0 \
  --max_frames 300

echo ""
echo "To run on full video, remove the --max_frames 300 option"
echo "To adjust synchronization, use --offset_frames N (positive if mocap starts later)"
