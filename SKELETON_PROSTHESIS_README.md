# Skeleton + Prosthesis Visualization Pipeline

This pipeline converts motion capture markers to H36M skeleton format and computes rigid body transformation for prosthesis mesh alignment.

## Overview

1. **Annotate prosthesis reference points** - Select 4 points on the prosthesis STL mesh
2. **Convert markers to skeleton** - Map mocap markers to 17 H36M joints + compute prosthesis transform
3. **Visualize results** - Interactive 3D viewer showing skeleton + aligned prosthesis mesh

## Files

- `annotate_prosthesis_points.py` - Interactive tool to select 4 reference points on prosthesis STL
- `markers_to_skeleton_with_prosthesis.py` - Convert markers to skeleton with prosthesis transformation
- `visualize_skeleton_prosthesis.py` - 3D visualization of skeleton + prosthesis
- `run_skeleton_prosthesis_pipeline.sh` - Complete pipeline script

## Step 1: Annotate Prosthesis Reference Points

Select 4 corner points on the prosthesis mesh (RPBR, RPBL, RPUL, RPUR):

```bash
python3 annotate_prosthesis_points.py \
    --stl Genesis.STL \
    --output prosthesis_config.json \
    --port 8060
```

**Usage:**
1. Open browser: http://localhost:8060
2. Click on mesh surface to select points (points snap to nearest vertex)
3. Use coordinate input boxes for fine-tuning
4. Use view buttons to change camera angle (Top/Front/Side/Reset)
5. Click "Confirm Point" to save each point
6. Click "Save Configuration" when all 4 points are selected

**Output:** `prosthesis_config.json`
```json
{
  "prosthesis_name": "Genesis Running Blade",
  "stl_file": "/Volumes/FastACIS/annotation_pipeline/Genesis.STL",
  "mesh_center": [x, y, z],
  "points": {
    "RPBR": {"position": [x, y, z], "description": "..."},
    "RPBL": {"position": [x, y, z], "description": "..."},
    "RPUL": {"position": [x, y, z], "description": "..."},
    "RPUR": {"position": [x, y, z], "description": "..."}
  },
  "marker_order": ["RPBR", "RPBL", "RPUL", "RPUR"]
}
```

## Step 2: Convert Markers to Skeleton

Process mocap data to generate skeleton joints and prosthesis transformation:

```bash
python3 markers_to_skeleton_with_prosthesis.py \
    --mocap /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \
    --marker_labels marker_labels_final.csv \
    --skeleton_config skeleton_config.json \
    --prosthesis_config prosthesis_config.json \
    --output skeleton_with_prosthesis.json \
    --frames "3832"  # or "3830-3840" for range
```

**Parameters:**
- `--mocap`: Motive CSV export file
- `--marker_labels`: CSV mapping marker IDs to labels (from annotation tool)
- `--skeleton_config`: H36M skeleton configuration JSON
- `--prosthesis_config`: Prosthesis reference points JSON (from Step 1)
- `--output`: Output JSON file
- `--frames`: Frame number or range (e.g., "3832" or "3830-3840")

**What it does:**
1. Loads mocap markers for each frame
2. Computes 16 H36M joints using marker-to-joint mapping
3. Computes 4-point rigid body transformation (rotation + translation + scale) for prosthesis
4. Calculates RAnkle position as center of 4 prosthesis markers
5. Transforms prosthesis mesh vertices using computed transformation
6. Saves all data to JSON

**Output:** `skeleton_with_prosthesis.json`
```json
{
  "metadata": {
    "mocap_file": "...",
    "num_frames": 1,
    "frame_range": [3832, 3832]
  },
  "frames": [
    {
      "frame": 3832,
      "skeleton": {
        "Pelvis": {"joint_id": 0, "position": [x,y,z], "valid": true},
        ...
        "RAnkle": {"joint_id": 16, "position": [x,y,z], "valid": true}
      },
      "prosthesis_transform": {
        "R": [[...], [...], [...]],  // 3x3 rotation matrix
        "t": [x, y, z],               // translation vector
        "s": 1.05,                    // scale factor
        "valid": true,
        "ankle_position": [x, y, z],
        "marker_positions": {
          "RPBR": [x, y, z],
          ...
        }
      },
      "prosthesis_mesh": [[x,y,z], ...]  // transformed mesh vertices
    }
  ]
}
```

## Step 3: Visualize Results

Interactive 3D visualization:

```bash
python3 visualize_skeleton_prosthesis.py \
    --input skeleton_with_prosthesis.json \
    --port 8061
```

**Usage:**
1. Open browser: http://localhost:8061
2. Use dropdown to select frame
3. Use "Previous Frame" / "Next Frame" buttons to navigate
4. Rotate/zoom/pan the 3D view with mouse
5. View frame information (valid joints, prosthesis status, etc.)

**Visualization Elements:**
- **Blue lines**: Skeleton bones (17 H36M connections)
- **Red circles**: Skeleton joints (with labels)
- **Light green mesh**: Prosthesis (transformed to align with markers)
- **Orange diamonds**: 4 prosthesis markers (RPBR, RPBL, RPUL, RPUR)

**Single frame static view:**
```bash
python3 visualize_skeleton_prosthesis.py \
    --input skeleton_with_prosthesis.json \
    --frame 3832
```

## Complete Pipeline Script

Run all steps automatically:

```bash
./run_skeleton_prosthesis_pipeline.sh
```

Edit the script to configure:
- `MOCAP_CSV`: Path to mocap CSV file
- `FRAME_RANGE`: Frame number or range (e.g., "3832" or "3830-3840")

## H36M Skeleton Format

17 joints with parent-child relationships:

| ID | Joint Name | Parent | Computed From |
|----|------------|--------|---------------|
| 0  | Pelvis     | -      | mean(LHip_computed, RHip_computed) |
| 1  | LHip       | 0      | mean(LPelvisf, Lpelvisb) |
| 2  | RHip       | 0      | mean(Rpelvisf, Rpelvisb) |
| 3  | Spine1     | 0      | mean(Spinef1, Spineb2) |
| 4  | Neck       | 3      | mean(Neckb1, Neckf2) |
| 5  | Head       | 4      | mean(head1, head2, head3, head4) |
| 6  | Jaw        | 5      | mean(face1, face2) |
| 7  | LShoulder  | 4      | mean(Lshoulder1, Lshoulder2, Lshoulder3) |
| 8  | LElbow     | 7      | mean(Lelbowl1, Lelbowl2, Lelbowl3) |
| 9  | LWrist     | 8      | mean(Lwristl1, Lwristl2) |
| 10 | RShoulder  | 4      | mean(Rshoulder1, Rshoulder2, Rshoulder3) |
| 11 | RElbow     | 10     | mean(Relbowr1, Relbowr2, Relbowr3) |
| 12 | RWrist     | 11     | mean(Rwristr1, Rwristr2) |
| 13 | LKnee      | 1      | mean(Lkneel1, Lkneel2) |
| 14 | LAnkle     | 13     | mean(Lanklel1, Lanklel2) |
| 15 | RKnee      | 2      | mean(Rkneer1, Rkneer2) |
| 16 | RAnkle     | 15     | **Prosthesis: mean(RPBR, RPBL, RPUL, RPUR)** |

## Prosthesis Transformation Algorithm

Uses **Kabsch algorithm** (optimal rigid body transformation):

1. **Input:**
   - Source points P: 4 reference points on mesh (from annotation)
   - Target points Q: 4 marker positions from mocap

2. **Steps:**
   - Center both point sets
   - Compute scale factor: `s = ||Q|| / ||P||`
   - Normalize point sets
   - Compute covariance matrix: `H = P^T @ Q`
   - SVD: `H = U @ S @ V^T`
   - Rotation matrix: `R = V @ U^T`
   - Translation: `t = centroid_Q - s * R @ centroid_P`

3. **Apply to mesh:**
   - For each vertex v: `v' = s * (v @ R^T) + t`

4. **RAnkle position:**
   - `RAnkle = mean(RPBR, RPBL, RPUL, RPUR)`

## Marker Label File Format

CSV file mapping marker IDs to human-readable labels:

```csv
original_name,marker_id,label
Unlabeled 1000,9F:D842698FB0A611F0,Lkneel1
Unlabeled 1004,D1:D842698FB0A611F0,RPBR
Unlabeled 1005,EA:D842698FB0A611F0,Spine2
...
```

Created using the annotation tool (annotate_mocap_markers_2d3d.py).

## Troubleshooting

### Issue: Prosthesis not visible in visualization
- Check that all 4 markers (RPBR, RPBL, RPUL, RPUR) are present in the frame
- Verify `prosthesis_transform.valid` is `true` in output JSON

### Issue: Skeleton joints missing
- Check marker labels in CSV match the skeleton_config requirements
- Some joints may be invalid if required markers are missing in the frame

### Issue: Prosthesis misaligned
- Re-annotate reference points on prosthesis mesh
- Ensure reference points correspond to the actual marker locations

### Issue: CSV loading error
- Verify mocap CSV is in Motive export format
- Check that frame number exists in the CSV file

## Requirements

```bash
conda create -n multical python==3.10
conda activate multical
pip install numpy pandas scipy plotly dash stl
```

## References

- **H36M Format**: Human3.6M dataset skeleton structure
- **Kabsch Algorithm**: Optimal rigid body transformation
- **Motive**: OptiTrack motion capture software
