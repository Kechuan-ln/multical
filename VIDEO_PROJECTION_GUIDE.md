# Skeleton + Prosthesis Video Projection Guide

Project 3D skeleton and prosthesis onto video frames.

## Step 1: Analyze Skeleton Completeness

First, check why upper body joints are missing:

```bash
python3 analyze_skeleton_completeness.py skeleton_with_prosthesis.json
```

This will show:
- Which joints are valid/invalid
- Required markers for each missing joint
- Which markers are missing from your label file
- Suggestions for fixing the issue

## Step 2: Project to Video

Project skeleton + prosthesis onto PrimeColor video:

```bash
python3 project_skeleton_to_video.py \
    --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
    --skeleton skeleton_with_prosthesis.json \
    --video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \
    --output projected_video.mp4 \
    --start-frame 3800 \
    --num-frames 200 \
    --camera-serial C11764
```

**Parameters:**
- `--mcal`: OptiTrack calibration file (.mcal)
- `--skeleton`: Skeleton JSON (from markers_to_skeleton_with_prosthesis.py)
- `--video`: Input video file
- `--output`: Output video file
- `--start-frame`: Start frame number
- `--num-frames`: Number of frames to process (-1 for all)
- `--camera-serial`: PrimeColor camera serial (default: C11764)
- `--line-thickness`: Skeleton bone line thickness (default: 2)
- `--point-radius`: Joint point radius (default: 4)
- `--no-frame-info`: Disable frame info overlay

## Visualization

The video will show:
- **Colored lines**: Skeleton bones
  - Blue: Spine/Neck
  - Magenta: Head/Jaw
  - Green: Left arm
  - Red: Right arm
  - Cyan: Left leg
  - Orange: Right leg
- **White circles**: Joint positions (with black outline)
- **Green overlay**: Prosthesis mesh (semi-transparent)
- **Green text**: Frame info (frame number, joint count, prosthesis status)

## Complete Pipeline Example

Process mocap → generate skeleton → project to video:

```bash
# 1. Process mocap data (if not done already)
python3 markers_to_skeleton_with_prosthesis.py \
    --mocap /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
    --marker_labels marker_labels_final.csv \
    --skeleton_config skeleton_config.json \
    --prosthesis_config prosthesis_config.json \
    --output skeleton_with_prosthesis.json \
    --frames "3800-4000"

# 2. Project to video
python3 project_skeleton_to_video.py \
    --mcal /Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal \
    --skeleton skeleton_with_prosthesis.json \
    --video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \
    --output projected_video.mp4 \
    --start-frame 3800 \
    --num-frames 200
```

## Fixing Missing Joints

If you see incomplete skeleton (e.g., "13/16 valid joints"), it means some markers are missing.

**Common issues:**

1. **Markers not labeled**: Check `marker_labels_final.csv`
   - Solution: Re-label markers using annotation tool

2. **Markers occluded in frame**: Mocap tracking loss
   - Solution: Use frames where subject is fully visible
   - Or: Manually annotate missing markers

3. **Wrong marker names in skeleton_config.json**:
   - Solution: Update skeleton_config.json to match your marker labels

**To debug:**
```bash
python3 analyze_skeleton_completeness.py skeleton_with_prosthesis.json
```

## Tips

**For best results:**
- Process frames where subject is fully visible
- Ensure all 40 markers are labeled correctly
- Use frames with good mocap tracking (no occlusions)
- Process 100-300 frames for short clips

**Performance:**
- Processing speed: ~5-15 fps (depends on resolution)
- 200 frames @ 1080p ≈ 15-40 seconds

**File sizes:**
- 200 frames @ 1080p ≈ 20-40 MB
- 500 frames @ 1080p ≈ 50-100 MB

## Troubleshooting

### "No frames with skeleton overlay"
- Check frame range matches skeleton data
- Verify `--start-frame` and skeleton frame range overlap
- Run `python3 analyze_skeleton_completeness.py` to check data

### "Camera with Serial='C11764' not found"
- Check camera serial in .mcal file
- Use `--camera-serial` to specify correct serial

### "Video resolution doesn't match calibration"
- Video and calibration must have same resolution
- Re-calibrate camera if needed

### Prosthesis not visible
- Check prosthesis_transform.valid in JSON
- Ensure all 4 prosthesis markers (RPBR, RPBL, RPUL, RPUR) are present
- Try different frames

### Skeleton appears offset
- Check camera extrinsics calibration
- Verify mocap and video are synchronized
- Check frame number correspondence

## Advanced Usage

### Custom styling
```bash
python3 project_skeleton_to_video.py \
    --skeleton skeleton_with_prosthesis.json \
    --video primecolor.avi \
    --output styled_video.mp4 \
    --line-thickness 4 \
    --point-radius 6 \
    --start-frame 3800 \
    --num-frames 100
```

### Process entire video
```bash
python3 project_skeleton_to_video.py \
    --skeleton skeleton_with_prosthesis.json \
    --video primecolor.avi \
    --output full_video.mp4 \
    --num-frames -1
```

### No frame info overlay
```bash
python3 project_skeleton_to_video.py \
    --skeleton skeleton_with_prosthesis.json \
    --video primecolor.avi \
    --output clean_video.mp4 \
    --no-frame-info \
    --start-frame 3800 \
    --num-frames 200
```

## Output Format

Output video:
- Format: MP4 (H.264)
- Frame rate: Same as input video
- Resolution: Same as input video
- Codec: MPEG-4

You can convert to other formats:
```bash
# Convert to high-quality MP4
ffmpeg -i projected_video.mp4 -c:v libx264 -crf 18 output_hq.mp4

# Convert to GIF (for sharing)
ffmpeg -i projected_video.mp4 -vf "fps=10,scale=640:-1" output.gif
```
