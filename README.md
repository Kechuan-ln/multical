# Requirements

Take conda as an example.
```
conda create -n multical python==3.10
conda activate multical
conda install -c conda-forge ffmpeg -n multical
pip install -r requirements_multical.txt
```



# Camera Calibration and Synchronisation
## Gen ChArUco Board
Goto ```gen_charuco``` and refer to the readme there.


## Intrinsic calibration

**Step 1:** Convert videos to images by running ```convert_video_to_images.py```.

**Step 2:** Refer to ```find_stable_boards.py``` to detect frames where boards are almost static, it will output the frame index for static boards.
- I did not do this step when I calibrate for gopro cameras.

**Step 3:** Refer to ```copy_image_subset.py``` to mannually select and copy images for calibration. 


**Step 4:** Goto ```multical``` for intrinsic calibration and refer to the readme there. 


**Step 5:** One should check the following things to verify whether the intrinsic calibration is acceptable
- Check the visualization, whether enough images under various board pose has good corner detections.
- Check RMS, should be below 0.5
- Check FOV by referring to ```tool_scripts/intrinsics_to_fov.py```, and compare to the value defined in [GoPro official info](https://community.gopro.com/s/article/HERO12-Black-Digital-Lenses-FOV-Information?language=en_US)





## Timecode synchronization

### Synchrnoize and output videos
Refer to ```sync_timecode.py```, organize the source videos under ```<src_tag>```. By running the code, it will output trimmed and syncrhonised videos in the ```<out_tag>``` folder.
- Furthermore, one can decide whether use `-c:v copy` or `-c:v libx264` when writing synchornised MP4. `-c:v copy` is faster but not that accurately synchronised, with 1 or 2 frame difference at 60FPS.

For example, run 

```bash
python sync_timecode.py --src_tag recording --out_tag sync --stacked
```

And below is the expected directory-tree

```
├── <PATH_ASSETS_VIDEOS>/
|   ├── <src_tag>/
|   |   ├── <cam01>/
|   |   |   └── <video1>.MP4
|   |   ├── <cam02>/
|   |   |   └── <video2>.MP4
|   |   └── .../
|   ├── <out_tag>
|   |   ├── <cam01>/
|   |   |   └── <video1>.MP4
|   |   ├── <cam02>/
|   |   |   └── <video2>.MP4
|   |   └── .../
```




### Verify camera synchronisation

Refer to ```convert_video_to_images.py``` to decompose timecode related video segments into images, one can use a fps as 15. Then check whether time code are aligned across cameras.


## Multi-Camera QR Code Synchronization (GoPro + PrimeColor + Mocap)

### Overview

This workflow synchronizes multiple camera systems (GoPro, PrimeColor) and motion capture data using QR code anchor videos. It provides frame-accurate synchronization across different FPS cameras without requiring common QR codes.

**Key Features:**
- ✅ Automatic detection of data structure
- ✅ Hardware timecode-based GoPro synchronization
- ✅ QR anchor-based PrimeColor alignment (no common QR codes needed!)
- ✅ Automatic Mocap CSV synchronization
- ✅ Quality verification with QR code validation
- ✅ Stacked video generation for visual verification
- ✅ Support for different sync modes (fast/accurate)

### Prerequisites

```bash
# Install required dependencies
pip install -r requirements_annotation.txt

# Dependencies include:
# - opencv-python
# - numpy
# - pandas
# - scipy
# - pyzbar (for QR code detection)
```

### Directory Structure

```
input_dir/
├── gopro_raw/                    # Raw GoPro videos
│   ├── cam01/Video.MP4
│   ├── cam02/Video.MP4
│   └── ...
├── primecolor_raw/               # Raw PrimeColor video
│   └── Video.avi
├── qr_sync.mp4                   # QR anchor video (reference)
└── mocap/                        # Motion capture data
    └── video.csv
```

### Quick Start: One-Command Workflow

```bash
python sync/sync_multi_camera_workflow.py \
  --input_dir /path/to/data \
  --output_dir /path/to/output \
  --sync_mode ultrafast \
  --stacked
```

**Parameters:**
- `--input_dir`: Input directory containing all data
- `--output_dir`: Output directory for synchronized results
- `--sync_mode`: Sync mode - `fast_copy` / `ultrafast` (default) / `accurate`
- `--stacked`: Generate stacked video for verification (optional)
- `--videos_per_row`: Videos per row in stacked video (default: 3)

**Sync Modes:**
- `fast_copy`: Fastest (~1 min), uses copy codec, keyframe accuracy (0-2s possible error @ 60fps)
- `ultrafast`: Fast + frame-accurate (~5-10 min), libx264 ultrafast preset **[Recommended]**
- `accurate`: Slowest but highest quality (~60 min), libx264 medium preset

### Workflow Steps

The workflow automatically executes 5 steps:

#### Step 1: GoPro Timecode Synchronization
Synchronizes GoPro cameras using embedded hardware timecode.

**How it works:**
1. Extracts timecode from each video using ffprobe (`HH:MM:SS:FF`)
2. Finds common time window: `max(start_times)` to `min(end_times)`
3. Trims videos with ffmpeg to synchronized segment

**Requirements:**
- All GoPro videos must have embedded timecode
- All videos must have the same FPS
- Videos can have different start times and durations

#### Step 2: GoPro QR Verification
Validates GoPro synchronization quality using QR codes.

**Process:**
1. Scans first 30 seconds of each synced GoPro video
2. Detects QR codes and calculates frame time differences
3. Evaluates sync quality: excellent / good / poor

#### Step 3: PrimeColor & Mocap Synchronization ⭐
Synchronizes PrimeColor and Mocap data using QR anchor alignment.

**Core Algorithm (Anchor Alignment Method):**
```
1. Scan GoPro video → detect QR codes → [(gopro_time, qr_num), ...]
2. Scan PrimeColor video → detect QR codes → [(primecolor_time, qr_num), ...]
3. Map each detection to anchor time:
   gopro_offset = gopro_time - anchor_time(qr_num)
   primecolor_offset = primecolor_time - anchor_time(qr_num)
4. Calculate relative offset:
   offset = gopro_offset_median - primecolor_offset_median
5. Align videos:
   if offset > 0: Add black frames to PrimeColor start
   if offset < 0: Trim PrimeColor start
```

**Key Points:**
- ✅ **No common QR codes needed** between GoPro and PrimeColor
- ✅ Each video aligns independently with anchor
- ✅ Supports different FPS (GoPro 60fps, PrimeColor 120fps)
- ✅ Mocap CSV synchronized automatically (same FPS as PrimeColor)

**Reference:** See `sync/sync_with_qr_anchor.py` for the reference implementation.

#### Step 4: Final Verification
Validates synchronization by scanning video endings.

#### Step 5: Stacked Video Generation
Creates multi-camera stacked video (GoPro + PrimeColor) for visual verification.

**Layout:**
- Currently: Horizontal layout (all cameras in one row)
- Future: Grid layout with custom videos per row

### Output Structure

```
output_dir/
├── gopro_synced/                  # Synchronized GoPro videos
│   ├── cam01/Video.MP4
│   ├── cam02/Video.MP4
│   ├── ...
│   └── meta_info.json            # Sync metadata (offset, duration)
├── gopro_qr_verification.json    # GoPro sync quality report
├── primecolor_mocap_synced/       # PrimeColor & Mocap output
│   ├── primecolor_synced.mp4     # Synchronized PrimeColor video
│   ├── mocap_synced.csv          # Synchronized Mocap data
│   └── sync_mapping.json         # Time mapping parameters
├── final_verification.json        # Final verification report
├── stacked_all_cameras.mp4        # Stacked video (optional)
└── sync_workflow_report.json     # Complete workflow report
```

### Advanced Usage: Step-by-Step Execution

If you prefer manual control, run each step separately:

```bash
# Step 1: GoPro Synchronization
python scripts/sync_timecode.py \
  --src_tag gopro_raw \
  --out_tag gopro_synced \
  --sync_mode ultrafast

# Step 2: GoPro QR Verification
python sync/verify_gopro_sync_with_qr.py \
  --gopro_dir gopro_synced \
  --anchor_video qr_sync.mp4 \
  --save_json gopro_qr_verification.json

# Step 3: PrimeColor Synchronization
python sync/sync_primecolor_gopro.py \
  --gopro_video gopro_synced/cam04/Video.MP4 \
  --primecolor_video primecolor_raw/Video.avi \
  --anchor_video qr_sync.mp4 \
  --mocap_csv mocap/video.csv \
  --output_dir primecolor_synced \
  --scan_duration 30

# Step 4: Final Verification
python sync/verify_final_sync_all_cameras.py \
  --gopro_dir gopro_synced \
  --primecolor_video primecolor_synced/primecolor_synced.mp4 \
  --anchor_video qr_sync.mp4

# Step 5: Create Stacked Video
python sync/create_stacked_video.py \
  --gopro_dir gopro_synced \
  --primecolor_video primecolor_synced/primecolor_synced.mp4 \
  --output stacked_output.mp4 \
  --layout horizontal
```

### Quality Metrics

**Synchronization Quality Indicators:**

1. **GoPro Sync Quality:**
   - excellent: max_time_diff < 0.1s between cameras
   - good: max_time_diff < 0.5s
   - poor: max_time_diff >= 0.5s

2. **PrimeColor Alignment:**
   - Offset standard deviation: < 0.01s (excellent)
   - FPS ratio error: < 0.1%
   - Typical offset std: ~0.008s

3. **Mocap CSV Alignment:**
   - Frame offset = offset_seconds × mocap_fps
   - All frame numbers shifted by the same offset

### Troubleshooting

**Q: "QR码检测不足" (Insufficient QR codes detected)**
- Increase `--scan_duration` (default 30s, try 60s)
- Check QR code visibility in videos
- Ensure QR codes are in focus and well-lit

**Q: PrimeColor视频比GoPro短很多**
- This is expected when offset requires trimming
- Output duration = overlap between GoPro and PrimeColor
- Example: GoPro 90s, PrimeColor needs +9s padding → output 75s

**Q: Stacked video shows cameras not aligned**
- Check sync_mapping.json for offset sign (should be positive for delay)
- Verify all videos use synced GoPro (not raw)
- Re-run workflow with latest code

**Q: 不支持QR码同步吗？**
- ✅ 支持！必须使用QR anchor视频作为时间基准
- ❌ 不支持传统的"在视频中出现QR闪光"方法
- 必须有独立的QR anchor视频（逐帧QR码序列）

### Implementation Notes

**Critical Bug Fixes (2025-11-04):**

1. **Removed Common QR Code Logic** ✅
   - Old: Tried to find common QR codes between GoPro and PrimeColor
   - New: Pure anchor alignment (each video aligns independently)

2. **Fixed Offset Sign Error** ✅
   - Old: `offset = primecolor - gopro` (wrong!)
   - New: `offset = gopro - primecolor` (correct!)
   - Impact: Determines whether to add black frames or trim

3. **CSV Alignment Verified** ✅
   - Mocap frames shifted by: `offset_frames = round(offset_seconds × fps)`
   - Maintains data integrity and frame correspondence

**Reference Files:**
- Main workflow: `sync/sync_multi_camera_workflow.py`
- PrimeColor sync: `sync/sync_primecolor_gopro.py`
- Anchor reference: `sync/sync_with_qr_anchor.py`
- Detailed documentation: `同步流程完整说明.md`

### Example Results

**Typical Synchronization:**
```
GoPro offset relative to anchor: +7.712s (std: 0.008s)
PrimeColor offset relative to anchor: -1.667s (std: 0.008s)
Relative offset: 9.379s
→ PrimeColor delayed 9.38s (add 1125 black frames @ 120fps)
→ Output: 84.93s synchronized video
```

**Sync Quality:**
- GoPro sync quality: excellent (< 0.1s difference)
- Offset consistency: 0.008s std (< 1 frame @ 120fps)
- FPS ratio error: 0.0% (2.002 actual vs 2.002 expected)


## Extrinsic calibration

**Step 1:** Refer to ```convert_video_to_images.py``` to decompose calibrated related video segments into images, one can use a fps as 5~15.

**Step 2:** Refer to ```find_stable_boards.py``` to detect frames where boards are almost static, it will output the frame index for static boards.
- This can be the key for a good calibration.

**Step 3:** Refer to ```copy_image_subset.py``` to copy distorted/undistorted images used for extrinsic calibration. 

**Step 4:** Goto ```multical``` for extrinsic calibration and refer to the readme there. 
- Make sure use the corresponding intrinsics json.


**Step 5:** One should check the following things to verify whether the extrinsic calibration is acceptable
- Check RMS, should be sub-pixel for inliers.
- Check the visualization, whether board could be simulatenously detected by all possible camera paris, with various pose and positions.

### Get gravity-aligned world coordinate system

**(I did not check and brush-up paths for this function)**

This is to get the world system that has xOy as ground plane, z-axis aligned with gravity. As achieved by putting the calibration board on the floor, assign 2D-3D for corner points, and calculate PnP to get the board2cam. 

**Step 1:** Refer to `convert_video_to_images.py`to decompose related videos and get undistorted images.

**Step 2:** Refer to `tool_pnp_pairing.py`, to assign 2D-3D corner points with Gradio. 
- The input are 
    - `--path_image`: the undistorted image with board on the floor 
    - `--boards`: the path of board info json
    - `--path_points`: output json.
- In the GUI, user input row and column index of corner points. The 3D position will be calculated automatically (with z=0), where the user then either assign automatic 2D detection or click to manually annotate.
- Note that the computed world2cam may lead to upside-down 3D triangulation. In this case, just swap the row and column and annotate again.

An example is 

``` bash
python tool_pnp_pairing.py --path_image ./undistorted/cam2/frame_0216.png --path_points ./board2base_points.json
```

**Step 3:** Refer to `calculate_world2cam.py`, it will compute transformation board coordinate system to this pnp camera, and combine with base2cam obtained from extrinsic calibration, thus obtaining the `camera_world2cam`.
- The input are
    - `--path_points`: Path of json file of 2D-3D point pairs, obtained from the previous Gradio tool.
    - `--path_extr`: Path of json file of camera extrinsic calibration results, obtained from multical.
    - `--basecam_pnp`: The camera used in assigning 2D-3D point pairs.
    - `--path_output`: The output json path to save the updated extrinsic results,
- It will first compute the world (i.e., board) to the camera used 2D-3D pair assignments (i.e., pnp camera), and get the transformation between this pnp camera and base camera in json of path_extr (i.e., base json camera). And then, by further combining with the transformation between base json camera and each camera, we could get world2cam, with world has xOy as ground plane and z-axis roughly aligned to gravity.
- The results will be `camera_world2cam`, appended to the original extrinsic info and saved.

An example is

```bash
python calculate_world2cam.py --path_points ./board2base_points.json --path_extr ./calibration.json --basecam_pnp cam2 --path_output ./calibration2.json
```




# Get 3D Joint Annotations

## Default directory tree

It would facilitate if one put the camera calibration information under original subfolder, and use original images. Therefore, all programs will refer to original images and undistort with `calibration.json`

```
├── <PATH_ASSETS_VIDEOS>/<recording_name>/
|   ├── original
|   |   ├── calibration.json
|   |   ├── <cam01>/
|   |   |   ├── frame_0001.png
|   |   |   └── ...
|   |   ├── <cam02>/
|   |   |   ├── frame_0001.png
|   |   |   └── ...
|   |   └── .../
```



## 2D bounding box 

### Automatic detection and tracking

Refer to `run_yolo_tracking.py`. It will first undistort images, and then leverage YOLO detection and ByteTrack to detect and track humans from images.


The output json results for bounding boxes are saved in `<PATH_ASSETS_BBOX_AUTO>/<recording_name>_<cam_key>.json`

For parameters
- `--recording_tag` is `<recording_name>/original` or `<recording_name>/undistort`, contains images, and under `PATH_ASSETS_VIDEOS`
- `--path_camera` contains camera intrinsics and extrinsics. This is required to undistort images. if it is empty, then we will use `calibration.json` under `--recording_tag`.
- `--verbose` indicates whether show visualisation



For example, run
```bash
python run_yolo_tracking.py --recording_tag <recording_name>/original --verbose
```


### Manual correction

Based on the automatic detection and tracking results, one then refer to `tool_bbox_annotation.py` to manually check, approve, annotate and modify tracking results.


The input auto bbox detection is `<PATH_ASSETS_BBOX_AUTO>/<recording_name>_<cam_key>.json`.

The output json is saved in `<PATH_ASSETS_BBOX_MANUAL>/<recording_name>_<cam_key>.json`


For example, run
```bash
python tool_bbox_annotation.py --recording_tag <recording_name>/original --cam_key cam3
```


## 2D joint positions


### Automatic detection with ViTPose

Refer to `run_vitpose.py`. It will first undistort images, then based on the provided 2D bounding boxes (i.e., the one after mannual annotation), detect 2D joint positions in COCO format.


The input manual bounding box detection is `<PATH_ASSETS_BBOX_MANUAL>/<recording_name>_*.json`, which includes all cameras of the given recording. 

The output json results for 2D joint detections are saved in `<PATH_ASSETS_KPT2D_AUTO>/<recording_name>_*.json`
- One can further run only a subset of cameras by indicating `--cam_keys`

For example, run
```bash
python run_vitpose.py --recording_tag <recording_name>/original --verbose --cam_keys cam1,cam2
```

### Triangulation to detect false detection

#### Triangulation
Refer to `run_triangulation.py`. It will do triangulation based on undistorted camera parameters and 2D joint positions. Here, just use vitpose detections, and do not refer to any manual annotations.

The input 2D joint position json is  `<PATH_ASSETS_KPT2D_AUTO>/<recording_names>*.json`. 

The output json results for this automatic triangulation are saved in `<PATH_ASSETS_KPT3D>/<recording_name>_auto.json`

For example, run
```bash
python run_triangulation.py --recording_tag <recording_name>/original  --verbose
```

#### Fill missing keypoints
Refer to `run_refinement.py`, to fill missing 3D annotations. 

The input triangulation results from `<PATH_ASSETS_KPT3D>`, and decide whether refer to `_auto.json` or `_manual.json` based on `--use_manual_annotation`. We here use the auto detection.

The corresponding output json is saved to `<PATH_ASSETS_REFINED_KPT3D>`

The parameter is similar as that defined in `run_triangulation.py`. For example, run
```bash
python run_refinement.py --recording_tag <recording_name>/original  --verbose
```

#### Detect false detection

Refer to `check_kpt_autodet.py`, it will judge whether needs annotation, based on the reprojection errors, as indicated by `--threshold_det_conf` and `--threshold_reproj_err`.

The input refined triangulation json is `<PATH_ASSETS_REFINED_KPT3D>/<recording_name>_auto.json`


For example, run
```bash
python check_kpt_autodet.py --recording_tag <recording_name>/original
```

### Manual correction

Refer to `tool_kpt_annotation.py`. It enables approving and denying automatic 2D joint detections, as well as attaching manual annotations. Note that
- For each joint, it will judge whether needs manual annotation, based on the number of confident cameras and reprojection errors, as indicated by `--threshold_det_conf`, `--threshold_reproj_err`, and `--min_cam`.
- It would be more efficient if only approve and deny.


The input automatic triangulation json is `<PATH_ASSETS_KPT3D>/<recording_name>_auto.json`. 

The output json results are saved in `<PATH_ASSETS_KPT2D_MANUAL>/<recording_name>_*.json`.


For example, run
```bash
python tool_kpt_annotation.py --recording_tag <recording_name>/original
```


## Triangulation for 3D joint positions

### Triangulation

Again, refer to `run_triangulation.py`, and it will do triangulation based on undistorted camera parameters and 2D joint positions. Note that 
- Here, further refer to `--use_manual_annotation` to leverage manually-checked detections.
- Use `--only_frames_with_manual` to decide whether ignore frames that do not have manually-checked annotations. 

The input 2D joint positions are from `<PATH_ASSETS_KPT2D_MANUAL>/<recording_name>*.json`.

The output json results for this triangulation are saved in `<PATH_ASSETS_KPT3D>/<recording_name>_manual.json`


### Refinement

Refer to `run_refinement.py`. This refers to EgoHumans to refine the triangulation results. 

Here, refer to `--use manual_annotation` to leverage the input 3D joints from `<PATH_ASSETS_KPT3D>/<recording_name>_manual.json`. 

The corresponding output json is saved to `<PATH_ASSETS_REFINED_KPT3D>`.

For example, run
```bash
python run_refinement.py --recording_tag <recording_name>/original --verbose --use_manual_annotation
```

### Manual approval

Refer to `tool_3d_approval.py`. It enables approving and denying 3D joint annotations, by referring to 3D visualisation and 2D projections.

The input 3D annotations are from `<PATH_ASSETS_KPT3D_REFINED>/<recording_name>_<manual>.json`. One can further use `--use_manual_annotations` defined whether the 3D results are from manual 2D or pure automatic 2D detection.

The output json is saved to `<recording_name>_<manual>_approved.json` under the same directory.



For example, run
```bash
python tool_3d_approval.py --recording_tag <recording_name>/original --use_manual_annotation
```

# Miscs

## Decompose videos

Refer to ```convert_video_to_images.py```. It will convert videos from ```<PATH_ASSETS_VIDEOS>/<src_tag>/``` and save under to ```<PATH_ASSETS_VIDEOS>/<src_tag>/original```
- if `original` or `undistort` is non-empty, the program will first remove all files under it to avoid conflicts.
- One can further designate ```--fps```, ```--ss```, and ```--duration```, they will be used in ```ffmpeg``` command as ```-ss <ss>, -t <duration>, -vf fps=<fps>``` 
- One can optionally input ```--path_intr```, to output undistorted images and undistorted intrinsics.

For example, run

```bash
python convert_video_to_images.py --src_tag  extr_86_sync --cam_tags cam2,cam3,cam6,cam7,cam8,cam9 --fps 5 --ss 180 --duration 60 
```

And below is the expected directory-tree

```
├── <src_tag>/
|   ├── original
|   |   ├── <cam01>/
|   |   |   ├── frame_0001.png
|   |   |   └── ...
|   |   ├── <cam02>/
|   |   |   ├── frame_0001.png
|   |   |   └── ...
|   |   └── .../
|   ├── undistort
|   |   ├── <cam01>/
|   |   |   ├── frame_0001.png
|   |   |   └── ...
|   |   ├── <cam02>/
|   |   |   ├── frame_0001.png
|   |   |   └── ...
|   |   └── .../
|   ├── <cam01>/
|   |   └── <video1>.MP4
|   ├── <cam02>/
|   |   └── <video2>.MP4
|   └── .../
```

The output intrinsics json file for the undistorted images should locate in the same directory of ```--path_intr```, but has name with ```_undistorted.json``` suffix. It should with
- Updated K matrix: Optimized camera matrix from `getOptimalNewCameraMatrix()`
- Zero distortion: `[0.0, 0.0, 0.0, 0.0, 0.0]`
- Same image size: Preserves original dimensions


## Select Subset Images for Calibration

This applies to both intrinsic and extrinsic calibration, where after decomposing the MP4 into images, one may detect relatively static board frames, manually select a subset of good images and use only them for calibration. 

To achieve this, one may refer to `find_stable_boards.py` and `copy_image_subset.py`. Note that in both scripts, only camera images (i.e., under folders defined by `<cam{id}>`) will be considered. Images of other folders, such as those saving visualisation results, will be ignored and passed.# multical
