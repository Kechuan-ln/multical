#!/bin/bash
# 使用自定义外参投影骨架到GoPro视频

# 方法1: 使用.mcal (OptiTrack标定)
echo "方法1: 使用.mcal文件"
python project_skeleton_to_gopro_FINAL_FIXED.py \
  --calibration /Volumes/FastACIS/GoPro/motion/calibration.json \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --skeleton skeleton_motion_3820_4081.json \
  --gopro-video /Volumes/FastACIS/GoPro/motion/Cam4/Video.MP4 \
  --output output_using_mcal.mp4 \
  --mocap-fps 120.0 \
  --sync-offset 10.5799 \
  --no-prosthesis

echo ""
echo "================================"
echo ""

# 方法2: 使用extrinsics_calibrated.json (你自己计算的标定)
echo "方法2: 使用自定义JSON文件"
python project_skeleton_to_gopro_FINAL_FIXED.py \
  --calibration /Volumes/FastACIS/GoPro/motion/calibration.json \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --extrinsics-json extrinsics_calibrated.json \
  --skeleton skeleton_motion_3820_4081.json \
  --gopro-video /Volumes/FastACIS/GoPro/motion/Cam4/Video.MP4 \
  --output output_using_custom.mp4 \
  --mocap-fps 120.0 \
  --sync-offset 10.5799 \
  --no-prosthesis

echo ""
echo "完成！请对比两个输出视频："
echo "  output_using_mcal.mp4    - 使用OptiTrack标定"
echo "  output_using_custom.mp4  - 使用自定义标定"
