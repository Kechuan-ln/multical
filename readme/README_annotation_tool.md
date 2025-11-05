# Human Pose Annotation Tool

A Gradio-based web interface for annotating, editing, and validating human body poses using pre-computed automatic detections and triangulated 3D poses.

## Overview

This tool provides an interactive interface for manual pose annotation and validation, designed to improve pose estimation quality by allowing human experts to:
- Review and validate automatic 2D detections and its 3D triangulation results
- Manually annotate incorrect or missing joints. 


## Prerequisites

1. **Data Requirements:**
   - Results from `run_vitpose_triangulation.py` saved as JSON files
   - LMDB storage of undistorted images
   - Camera calibration parameters

2. **Dependencies:**
   - Python 3.7+
   - Gradio
   - OpenCV
   - NumPy
   - PIL

## Quick Start

### Launch the Tool
```bash
python tool_annotation.py --seq_name your_sequence_name --cam_keys cam01,cam02,cam03,cam04
```

### Command Line Arguments
- `--participant_uid`: Participant UID for EgoExo dataset (default: 330)
- `--cam_keys`: Camera keys, comma-separated (default: cam01,cam02,cam03,cam04)
- `--seq_name`: Sequence name to load (default: uniandes_basketball_003_49)
- `--threshold_det_conf`: Threshold for 2D joint confidence scores (default: 0.5)
- `--threshold_reproj_err`: Threshold for 2D reprojection errors (default: 20.0)
- `--min_cam`: Minimum number of cameras for valid triangulation (default: 3)
- `--port`: Port for Gradio interface (default: 7860)
- `--share`: Create public sharing link

## Interface Overview

### Left Panel Controls

#### Frame Navigation
- **Frame Slider**: Navigate through available frames in the sequence

#### Display Modes
Choose from four visualization modes:

1. **Annotation Mode** 🎯
   - Primary mode for manual annotation
   - Shows annotation status (red circles = needs annotation, green = manually annotated, white = auto-detected)
   - Enables all annotation tools

2. **Original Image** 📷
   - Shows raw images without pose overlays
   - Displays annotation status in camera metadata

3. **2D Detection** 🤖
   - Visualizes automatic 2D detections

4. **2D Detection (Reproj.)** 🔄
   - Shows 2D reprojection of 3D triangulation results from automatic 2D detections

#### Annotation Tools (Annotation Mode Only)
- **Camera Selection**: Choose which camera view to annotate
- **Joint Selection**: Select which body joint to annotate (17 COCO joints)
- **Confidence Slider**: Set confidence score for manual annotations (0.0-1.0)
- **Approve Auto Annot.**: Button to approve high-confidence automatic detections
- **Pass Joint**: Mark invisible or occluded joints

### Right Panel - Camera Views

#### Multi-Camera Display
- Simultaneous view of all camera angles
- Each camera shows selected visualization mode
- Click-to-annotate functionality in Annotation Mode

#### Camera Metadata
Color-coded annotation status for each joint:
- **🔴 Red Text**: "Annot. Needed" - Action required
- **🟢 Green Text**: "Annot. Done" - Manual annotation completed
- **🔵 Blue Text**: "Annot. Pass" - Approved or passed


## Annotation Workflow

### 1. Review Mode
```
1. Select "2D Detection" or "2D Detection (Reproj.)" mode
2. Navigate through frames to identify problematic poses
3. Note joints highlighted in RED (need annotation)
```

### 2. Manual Annotation
```
1. Switch to "Annotation Mode"
2. Select target camera and joint
3. Set confidence score (0.0-1.0)
4. Click precise joint location on image
5. Joint automatically saves and updates display
```

### 3. Approve Good Detections
```
1. In "Annotation Mode", select camera and joint
2. If automatic detection confidence ≥ threshold, "Approve Auto Annot." button appears
3. Click to approve without manual clicking
4. Joint marked as "Annot. Pass" (blue) in meta data
```

### 4. Handle Occluded and Invisible Joints
```
1. Select camera and joint that's occluded/invisible
2. Click "Pass Joint (Occluded/Invisible)"
3. Joint marked with invalid coordinates (-1, -1, 0), shown as "Annot. Pass" (blue) in metadata
```




### Troubleshooting with Common Issues


**Button not appearing:**
- Check automatic detection confidence ≥ threshold_det_conf
- Ensure you're in "Annotation Mode"
- Verify automatic detection exists for selected camera/joint

**Click not registering:**
- Ensure correct camera is selected
- Verify you're in "Annotation Mode"
- Check that you're clicking on the selected camera's image

**Empty frame data:**
- Verify triangulation results JSON file exists
- Check sequence name matches available data
- Ensure camera keys match dataset structure



## Data Format

### Input Data Structure
```json
{
  "frame_id": {
    "bbox_xyxy": {"cam01": [x1, y1, x2, y2], ...},
    "vitpose_2d": {"cam01": [[x, y, conf], ...], ...},
    "triangulated_3d": [[x, y, z, valid], ...],
    "reproj_err": {"cam01": [error1, error2, ...], ...},
    "need_annot_flag": {"cam01": [true, false, ...], ...}
  }
}
```

### Output Annotations
```json
{
  "manual_2d": {"cam01": [[x, y, conf], ...], ...},
  "manual_flags": {"cam01": [0, 1, 2, 3, ...], ...}
}
```

For questions or issues, check the troubleshooting section or refer to the source code documentation.