# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a multi-camera 3D human pose annotation pipeline that performs camera calibration, synchronization, 2D detection/tracking, 3D triangulation, and manual annotation refinement. The pipeline processes multi-view video recordings to produce high-quality 3D joint position annotations in COCO format.

## Environment Setup

```bash
conda create -n multical python==3.10
conda activate multical
conda install -c conda-forge ffmpeg -n multical
pip install -r readme/requirements_multical.txt
pip install -r requirements_annotation.txt  # For annotation tools (pandas, plotly, dash)
```

Key dependencies: numpy==1.23, opencv-python==4.6.0.66, opencv-contrib-python==4.6.0.66, matplotlib, scipy, easydict, multical, torch, gradio

**Environment Variables** (optional):
- `PATH_ASSETS`: Base path for assets (default: `../assets/`)
- `PATH_ASSETS_VIDEOS`: Path for video files (default: `../assets/videos/`)

Set these if your data is stored in non-default locations.

## Directory Structure

The pipeline expects this structure:
```
assets/
├── videos/<recording_name>/
│   └── original/
│       ├── calibration.json          # Camera intrinsics/extrinsics
│       ├── cam01/, cam02/, etc.      # Frame images
├── results/
│   ├── bbox/auto/                    # Auto bounding boxes
│   ├── bbox/manual/                  # Manual bbox corrections
│   ├── vitpose/                      # 2D pose detections
│   ├── manual2d/                     # Manual 2D corrections
│   ├── triangulation/                # 3D triangulation results
│   └── refined3d/                    # Refined 3D poses
```

All paths are configured in [utils/constants.py](utils/constants.py) via `PATH_ASSETS_*` variables.

## Core Pipeline Commands

### 1. Camera Calibration

#### ChArUco Board Generation (Optional)
```bash
cd gen_charuco
# Generate B3 size board (5x9 grid)
python gen_pattern.py -o charuco_board.svg --rows 9 --columns 5 --type charuco_board \
  --square_size 50 --marker_size 40 -f DICT_7X7_250.json.gz --page_size B3

# Generate B1 size board (7x14 grid)
python gen_pattern.py -o charuco_board.svg --rows 14 --columns 10 --type charuco_board \
  --square_size 70 --marker_size 50 -f DICT_7X7_250.json.gz --page_size B1
```

#### Intrinsic Calibration
```bash
cd multical
# Calibrate each camera separately using ChArUco board images
python intrinsic.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path intr_0912 \
  --limit_images 1000 \
  --limit_intrinsic 1000 \
  --vis

# Output: intrinsic.json with K matrix, dist coefficients, FOV, RMS per camera
# Verification: RMS should be < 0.5 pixels
```

**Using Pre-calibrated GoPro Intrinsics**: The repository includes [intrinsic_hyperoff_linear_60fps.json](intrinsic_hyperoff_linear_60fps.json) with pre-computed intrinsics for GoPro cameras (HyperSmooth OFF, Linear lens, 60fps, 4K resolution). Each camera entry (cam2, cam3, etc.) contains K matrix, distortion coefficients, and FOV values.

#### Extrinsic Calibration
```bash
cd multical
# Option A: Calibrate extrinsics using pre-computed intrinsics (RECOMMENDED for GoPro)
python calibrate.py \
  --boards ./asset/charuco.yaml \
  --image_path extr_91122/original \
  --calibration ../intrinsic_hyperoff_linear_60fps.json \
  --fix_intrinsic \
  --limit_images 1000 \
  --vis

# Option B: Joint intrinsic+extrinsic calibration (from scratch)
python calibrate.py \
  --boards ./asset/charuco.yaml \
  --image_path extr_91122/original \
  --calibration extr_91122/intrinsic.json \
  --limit_images 1000 \
  --vis

# Output: calibration.json with camera_base2cam (R, T per camera)
# Verification: RMS should be sub-pixel for inliers
```

**Key Parameters**:
- `--fix_intrinsic`: Locks intrinsic parameters, only optimizes extrinsics (use with pre-calibrated cameras)
- `--calibration`: Path to intrinsic calibration JSON (relative to PATH_ASSETS_VIDEOS)
- `--camera_pattern '{camera}'`: Pattern to match camera folder names
- `--vis`: Generate visualization with detected corners and projected 3D axes

#### Verify Calibration
```bash
# Compute and compare FOV with GoPro specifications
python tool_scripts/intrinsics_to_fov.py

# Compare two calibration files
python tool_scripts/compare_calibrations.py
```

### 2. Video Synchronization

#### GoPro Timecode Synchronization (Hardware-Based)
```bash
# Synchronize videos using embedded timecodes (requires GoPro with timecode support)
python scripts/sync_timecode.py \
  --src_tag recording \
  --out_tag sync \
  --stacked \
  --fast_copy  # Use for faster processing (1-2 frame accuracy at 60fps)

# Without --fast_copy: Re-encodes video with libx264 (frame-accurate but slower)
```

**How it works** ([utils/calib_utils.py:71-102](utils/calib_utils.py)):
1. Extracts embedded timecode from each video using ffprobe
2. Converts timecodes to seconds: `HH:MM:SS:FF` → seconds
3. Finds common time window: `max(start_times)` to `min(end_times)`
4. Trims each video with ffmpeg `-ss` (offset) and `-t` (duration)

**Limitations**:
- ❌ **NO QR code support** - This pipeline does NOT support QR code-based synchronization
- ✅ Only works with cameras that embed hardware timecode (GoPro with timecode, professional cameras)
- ✅ All videos must have the same FPS
- ✅ Videos can have different start times and durations

**Output**: Synchronized videos in `<out_tag>/` with `meta_info.json` containing offset/duration per camera

#### Verify Synchronization
```bash
# Extract timecode frames as images to manually verify alignment
python scripts/convert_video_to_images.py \
  --src_tag sync \
  --cam_tags cam1,cam2,cam3 \
  --fps 15
# Check that timecode displays are aligned across cameras
```

### 3. Video to Images
```bash
python scripts/convert_video_to_images.py \
  --src_tag <recording_name> \
  --cam_tags cam1,cam2,cam3 \
  --fps 60 \
  --ss 180 \
  --duration 60

# Output: <recording_name>/original/ with frame images per camera
# Optionally add --path_intr to generate undistorted images
```

### 4. 2D Detection Pipeline
```bash
# 1. Auto detection with YOLO + ByteTrack
python scripts/run_yolo_tracking.py --recording_tag <recording_name>/original --verbose

# 2. Manual bbox correction (Gradio GUI)
python scripts/tool_bbox_annotation.py --recording_tag <recording_name>/original --cam_key cam1

# 3. ViTPose 2D keypoint detection
python scripts/run_vitpose.py --recording_tag <recording_name>/original --verbose --cam_keys cam1,cam2
```

### 5. 3D Triangulation Pipeline
```bash
# 1. Initial triangulation from auto detections
python scripts/run_triangulation.py --recording_tag <recording_name>/original --verbose

# 2. Refine with EgoHumans method
python scripts/run_refinement.py --recording_tag <recording_name>/original --verbose

# 3. Check for false detections
python scripts/check_kpt_autodet.py --recording_tag <recording_name>/original

# 4. Manual 2D keypoint correction (Gradio GUI)
python scripts/tool_kpt_annotation.py --recording_tag <recording_name>/original

# 5. Re-triangulate with manual corrections
python scripts/run_triangulation.py --recording_tag <recording_name>/original --verbose --use_manual_annotation

# 6. Refine again
python scripts/run_refinement.py --recording_tag <recording_name>/original --verbose --use_manual_annotation

# 7. Final 3D approval (Gradio GUI)
python scripts/tool_3d_approval.py --recording_tag <recording_name>/original --use_manual_annotation
```

## Architecture

### Core Modules

**[utils/triangulation.py](utils/triangulation.py)**: `Triangulator` class performs DLT-based triangulation with RANSAC outlier rejection. Supports aniposelib's `CameraGroup` or custom implementation.

**[dataset/recording.py](dataset/recording.py)**: `Recording` class manages multi-camera data loading, undistortion, and camera parameters. Loads from JSON calibration files with K, dist, rvec, tvec per camera.

**[utils/refine_pose3d.py](utils/refine_pose3d.py)**: Temporal and spatial refinement using EgoHumans method. Fills missing keypoints using motion priors and limb length constraints.

**[utils/fit_pose3d.py](utils/fit_pose3d.py)**: Optimization-based pose fitting with symmetry, temporal consistency, and limb length losses.

**[utils/calib_utils.py](utils/calib_utils.py)**: Camera undistortion utilities using OpenCV's `getOptimalNewCameraMatrix` and `undistort`.

**bytetrack/**: Integration of ByteTrack MOT algorithm for multi-person tracking across frames. Used before pose detection to maintain person IDs.

**multical/**: Submodule for ChArUco board-based camera calibration (both intrinsic and extrinsic).

### Data Flow

1. Multi-view videos → Timecode sync → Frame extraction
2. Frames → YOLO detection + ByteTrack → Bounding boxes (auto)
3. Manual bbox correction → Bounding boxes (manual)
4. Bboxes → ViTPose → 2D keypoints (auto, 17 COCO joints)
5. 2D keypoints (multi-view) → Triangulation → 3D keypoints (auto)
6. 3D keypoints → EgoHumans refinement → Refined 3D (auto)
7. Check reprojection errors → Flag frames needing manual annotation
8. Manual 2D correction → 2D keypoints (manual)
9. 2D keypoints (manual, multi-view) → Triangulation → 3D keypoints (manual)
10. 3D keypoints → EgoHumans refinement → Refined 3D (manual)
11. Manual 3D approval → Final approved 3D annotations

### Configuration

All annotation parameters are in [utils/constants.py](utils/constants.py) via `cfg_annotation`:
- `POSE3D.KEYPOINTS_THRES`: Detection confidence threshold (0.5)
- `POSE3D.MIN_VIEWS`: Minimum cameras for triangulation (2)
- `POSE3D.REPROJECTION_ERROR_EPSILON`: Optimization tolerance (0.01)
- `REFINE_POSE3D.*`: Temporal smoothing and outlier detection params
- `FIT_POSE3D.*`: Optimization hyperparameters

### Keypoint Format

COCO 17 keypoints (see `VIT_JOINTS_NAME` in constants.py):
0=Nose, 1=L_Eye, 2=R_Eye, 3=L_Ear, 4=R_Ear, 5=L_Shoulder, 6=R_Shoulder, 7=L_Elbow, 8=R_Elbow, 9=L_Wrist, 10=R_Wrist, 11=L_Hip, 12=R_Hip, 13=L_Knee, 14=R_Knee, 15=L_Ankle, 16=R_Ankle

2D format: `[x, y, confidence]` per joint
3D format: `[x, y, z, valid]` per joint (valid=1 if triangulated successfully)

### JSON Structures

**calibration.json**: Camera parameters
```json
{
  "cameras": {"cam01": {"K": [9 values], "dist": [5 values]}},
  "camera_base2cam": {"cam01": {"R": [9 values], "T": [3 values]}}
}
```

**triangulation results** (`<recording>_auto.json` or `<recording>_manual.json`):
```json
{
  "frame_0001": {
    "bbox_xyxy": {"cam01": [x1,y1,x2,y2]},
    "vitpose_2d": {"cam01": [[x,y,conf], ...17 joints]},
    "triangulated_3d": [[x,y,z,valid], ...17 joints],
    "reproj_err": {"cam01": [err0, err1, ...17 values]},
    "need_annot_flag": {"cam01": [bool, ...17 flags]}
  }
}
```

## Utility Scripts

### In tool_scripts/
- `intrinsics_to_fov.py`: Verify calibration by computing FOV
- `fov_to_intrinsics.py`: Generate intrinsics from known FOV
- `compare_calibrations.py`: Diff two calibration files
- `check_bone_lengths.py`: Validate 3D poses via skeleton proportions
- `convert_images_to_video.py`: Create videos from frame sequences
- `combine_intrinsic_json.py`: Merge multiple intrinsic calibration files
- `replace_image_with_placeholder.py`: Replace images with placeholders
- `compare_image_directories.py`: Compare two image directories

### In scripts/
- `find_stable_boards.py`: Detect frames with static calibration boards
- `copy_image_subset.py`: Select and copy image subsets for calibration
- `tool_pnp_pairing.py`: Gradio GUI for 2D-3D point pairing (gravity alignment)
- `calculate_world2cam.py`: Compute gravity-aligned world coordinate system

### In Root Directory (Advanced/Experimental Features)
- `annotate_mocap_markers.py`: Annotate motion capture marker positions
- `annotate_mocap_markers_2d3d.py`: Interactive 2D-3D marker annotation
- `annotate_extrinsics_interactive.py`: Interactive extrinsic calibration refinement
- `annotate_prosthesis_points.py`: Annotate prosthesis attachment points
- `project_skeleton_to_gopro*.py`: Project 3D skeleton onto GoPro videos
- `project_markers_*.py`: Project mocap markers onto videos
- `calibrate_gopro_primecolor_extrinsics.py`: GoPro-PrimeColor camera calibration
- `markers_to_skeleton*.py`: Convert marker data to skeleton representation
- `create_stacked_video.py`: Create multi-camera stacked video visualizations
- `enhance_dark_images.py`: Enhance low-light calibration images
- `filter_intrinsics.py`: Filter and clean intrinsic calibration data

## Camera Calibration Q&A

### Q1: 这个pipeline有内参和外参标定功能吗？
**是的，有完整的内外参标定功能**：
- **内参标定**: 使用 `multical/intrinsic.py`，基于ChArUco标定板，为每个相机单独标定K矩阵和畸变系数
- **外参标定**: 使用 `multical/calibrate.py`，标定多相机系统的相对位置关系（R, T矩阵）

### Q2: 对于GoPro，是否有预计算的内参？
**是的**，仓库包含预计算的GoPro内参文件：
- **文件**: [intrinsic_hyperoff_linear_60fps.json](intrinsic_hyperoff_linear_60fps.json)
- **适用**: GoPro Hero系列（HyperSmooth关闭，Linear镜头模式，60fps，4K分辨率）
- **包含**: 多个相机编号（cam2, cam3, cam4等），每个包含K矩阵、畸变系数、FOV、RMS误差
- **格式**: 与multical输出格式兼容

### Q3: 能否使用预计算内参+已有GoPro视频计算外参？
**完全可以**，这是推荐的工作流程：

1. **准备**: 使用预存的内参JSON文件（如 `intrinsic_hyperoff_linear_60fps.json`）
2. **外参标定**:
   ```bash
   cd multical
   python calibrate.py \
     --boards ./asset/charuco.yaml \
     --image_path your_gopro_videos/original \
     --calibration ../intrinsic_hyperoff_linear_60fps.json \
     --fix_intrinsic \
     --vis
   ```
3. **关键参数**: `--fix_intrinsic` 锁定内参，只优化外参
4. **输出**: `calibration.json` 包含 `camera_base2cam` 的R和T矩阵

**前提条件**:
- GoPro相机设置必须与内参文件匹配（HyperSmooth OFF, Linear, 60fps, 4K）
- 需要拍摄ChArUco标定板的同步视频
- 所有相机视频已经时间同步

### Q4: 是否支持GoPro时间码同步？
**是的**，通过硬件嵌入的timecode：
- **功能**: `scripts/sync_timecode.py` 使用ffprobe提取嵌入的timecode
- **原理**:
  1. 提取每个视频的 `HH:MM:SS:FF` timecode
  2. 找到所有视频的公共时间窗口
  3. 用ffmpeg裁剪到同步的时间段
- **要求**: GoPro必须支持timecode功能（如通过Timecode Systems同步器）
- **精度**:
  - 使用 `--fast_copy`: 快速但可能有1-2帧误差（60fps）
  - 不使用: 重新编码，帧精确但慢

### Q5: 是否支持QR码同步？
**❌ 不支持**：
- 此pipeline **没有** QR码识别和解析功能
- 仅支持硬件嵌入的timecode（如GoPro的timecode功能）
- 如需QR码同步，需要：
  1. 自行开发QR码检测和时间戳提取模块
  2. 替换 `utils/calib_utils.py` 中的 `synchronize_cameras()` 函数
  3. 或使用其他QR码同步工具预处理视频

### Q6: 非GoPro相机能用这个pipeline吗？
**可以，但需要满足条件**：
- **内参标定**: 任何相机都可以用ChArUco板标定内参
- **外参标定**: 同样适用于任何相机
- **同步**:
  - ✅ 如果相机支持timecode嵌入（专业摄像机），可直接使用
  - ❌ 如果用QR码/闪光灯同步，需自行实现同步逻辑
  - 💡 替代方案: 手动确定各视频的offset，修改 `sync_timecode.py` 或跳过自动同步

## Shell Script Workflows

Several shell scripts provide end-to-end workflows for common tasks:

```bash
# Example calibration workflow
./calibration_workflow.sh

# GoPro-PrimeColor calibration
./run_gopro_primecolor_calibration.sh

# Extrinsic calibration
./run_extrinsic_calibration.sh
./run_extrinsic_calibration_custom.sh

# Marker projection workflows
./run_marker_projection.sh
./run_marker_annotation_2d3d.sh
./run_marker_projection_test.sh

# Skeleton projection
./run_skeleton_projection.sh
./run_skeleton_prosthesis_pipeline.sh

# Dual video projection
./run_dual_video_projection.sh
```

Review these scripts to understand parameter configurations and adapt them to your specific setup.

## Advanced Features (Experimental)

### Motion Capture Integration
The repository includes experimental tools for integrating with motion capture systems:
- **Marker Annotation**: Tools for annotating mocap marker positions in videos
- **Skeleton Projection**: Project 3D skeleton data onto multi-camera videos
- **2D-3D Correspondence**: Interactive tools for establishing 2D-3D marker correspondences
- **Prosthesis Support**: Specialized annotation for prosthetic limb tracking

**Note**: These features are experimental and may require additional configuration. See individual script files for usage examples.

### Multi-Camera Visualization
- `create_stacked_video.py`: Create synchronized multi-camera grid or horizontal stacked videos
- Dual video projection scripts for side-by-side comparison
- HTML-based 3D visualization (`mocap_visualization.html`)

### Cross-Camera-System Calibration
- `calibrate_gopro_primecolor_extrinsics.py`: Calibrate between different camera systems (e.g., GoPro consumer cameras and PrimeColor professional cameras)
- `merge_intrinsics.py`, `filter_intrinsics.py`: Tools for managing multi-system calibrations

## Implementation Notes

- **Working Directory**: Always work from the `scripts/` directory when running core pipeline commands (the scripts in `scripts/` directory). Root-level scripts can be run from the project root.
- **Image Undistortion**: Images are undistorted on-the-fly using calibration params unless `load_undistort_images=True`
- **Manual Annotation Tools**: Use Gradio web interfaces (default port 7860). Access at `http://localhost:7860` after launching.
- **Reprojection Error Thresholds**: Typical values are `threshold_det_conf=0.5`, `threshold_reproj_err=20.0` pixels
- **Coordinate Systems**: The pipeline supports both `camera_world2cam` (gravity-aligned) and `camera_base2cam` (rig-relative) extrinsics
- **ChArUco Board Configuration**: Board geometry and detection parameters are defined in `multical/asset/*.yaml`
- **Camera Naming**: Camera folders must follow the pattern `cam{id}` or `Cam{id}` (e.g., `cam01`, `cam2`, `Cam03`)
- **Frame Naming**: Frame images should be named `frame_XXXX.png` with zero-padded frame numbers for proper sorting
