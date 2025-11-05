#!/bin/bash
# 测试marker投影到GoPro视频

echo "================================"
echo "测试Marker投影算法"
echo "================================"
echo ""

# 方法1: 使用OptiTrack .mcal


# 方法2: 使用自定义JSON
echo "方法2: 使用自定义外参JSON"
echo "--------------------------------"
/opt/homebrew/Caskroom/miniconda/base/envs/multical/bin/python project_markers_to_gopro.py \
  --calibration /Volumes/FastACIS/GoPro/motion/calibration.json \
  --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
  --extrinsics-json extrinsics_calibrated.json \
  --mocap-csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
  --marker-labels marker_labels.csv \
  --gopro-video /Volumes/FastACIS/GoPro/motion/Cam4/Video.MP4 \
  --output marker_projection_custom-begin.mp4 \
  --mocap-fps 120.0 \
  --sync-offset 10.5799 \
  --start-frame 0 \
  --num-frames 20000

echo ""
echo "================================"
echo "完成！"
echo "================================"
echo ""
echo "请对比两个输出视频："
echo "  marker_projection_mcal.mp4   - 使用OptiTrack标定"
echo "  marker_projection_custom.mp4 - 使用自定义标定"
echo ""
echo "如果marker点准确投影到人体上的marker位置，说明投影算法正确。"
