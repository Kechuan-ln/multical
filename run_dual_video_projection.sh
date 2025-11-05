#!/bin/bash
# 同时在GoPro和PrimeColor上投影markers，生成stacked video
# PrimeColor使用.mcal内参（和project_markers_final.py一致）

/opt/homebrew/Caskroom/miniconda/base/envs/multical/bin/python project_markers_dual_video.py \
  --calibration /Volumes/FastACIS/GoPro/motion/calibration.json \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --extrinsics-json extrinsics_calibrated.json \
  --mocap-csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --gopro-video /Volumes/FastACIS/GoPro/motion/Cam4/Video.MP4 \
  --primecolor-video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \
  --output dual_marker_projection.mp4 \
  --mocap-fps 120.0 \
  --gopro-sync-offset 10.5799 \
  --primecolor-sync-offset 0 \
  --start-frame 3800 \
  --num-frames 600 \
  --stack-mode horizontal \
  --marker-color 0,255,0 \
  --marker-size 3
