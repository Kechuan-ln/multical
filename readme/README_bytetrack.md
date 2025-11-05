# ByteTrack Human Tracking with VitPose Integration

This document explains the tracking behaviors of the ByteTrack integration in `run_yolo_vitpose.py`, specifically how the system handles new people appearing and people disappearing/reappearing in video sequences.

## Overview

Our implementation combines:
- **YOLO Detection**: Detects human bounding boxes in each frame
- **ByteTracker**: Assigns persistent IDs and tracks people across frames
- **VitPose**: Estimates pose for each tracked person

## Tracking Behaviors

### 🆕 New Person Appears

#### What Happens:
1. **Detection**: YOLO detects a new human bounding box
2. **Association Attempt**: ByteTracker tries to match with existing tracks
3. **New Track Creation**: Since no good match exists, creates a new track
4. **ID Assignment**: Assigns a unique, sequential track ID
5. **Pose Estimation**: Runs pose estimation with the new track ID

#### Example Timeline:
```
Frame 1:  Person A (ID:1), Person B (ID:2)
Frame 5:  Person A (ID:1), Person B (ID:2), Person C (ID:3) ← NEW PERSON
Frame 10: Person A (ID:1), Person B (ID:2), Person C (ID:3) ← ID MAINTAINED
```

#### JSON Output:
```json
{
  "sequence_name": {
    "1_track_1": {"track_id": 1, "bbox_xyxy": [...], "joints_2d": {...}},
    "1_track_2": {"track_id": 2, "bbox_xyxy": [...], "joints_2d": {...}},
    "5_track_1": {"track_id": 1, "bbox_xyxy": [...], "joints_2d": {...}},
    "5_track_2": {"track_id": 2, "bbox_xyxy": [...], "joints_2d": {...}},
    "5_track_3": {"track_id": 3, "bbox_xyxy": [...], "joints_2d": {...}}
  }
}
```

### 👻 Person Disappears and Reappears

The behavior depends on how long the person is missing:

#### Scenario 1: Short Disappearance (≤ 30 frames)

**Configuration**: `track_buffer = 30` frames

**What Happens:**
1. **Disappears**: Person not detected by YOLO (occlusion, exits frame, etc.)
2. **Track Maintained**: ByteTracker keeps track in "lost" state for up to 30 frames
3. **Prediction**: Uses Kalman filter to predict where person might be
4. **Reappears**: When detected again, **SAME track ID is restored**
5. **Seamless Continuation**: Person maintains their original identity

**Example Timeline:**
```
Frame 10: Person A (ID:1), Person B (ID:2)
Frame 15: Person A (ID:1) ← Person B disappears (behind object)
Frame 16: Person A (ID:1) ← Person B still missing
Frame 20: Person A (ID:1), Person B (ID:2) ← Person B reappears with SAME ID
```

**JSON Output:**
```json
{
  "sequence_name": {
    "10_track_1": {"track_id": 1, ...},
    "10_track_2": {"track_id": 2, ...},
    "15_track_1": {"track_id": 1, ...},
    "20_track_1": {"track_id": 1, ...},
    "20_track_2": {"track_id": 2, ...}  ← SAME ID RESTORED
  }
}
```

#### Scenario 2: Long Disappearance (> 30 frames)

**What Happens:**
1. **Track Removal**: After 30 frames, track is permanently deleted
2. **Memory Cleanup**: Tracker removes all trace of the person
3. **Reappears as New**: When person reappears, treated as completely new
4. **New Track ID**: Gets a fresh, unique ID

**Example Timeline:**
```
Frame 10: Person A (ID:1), Person B (ID:2)
Frame 15: Person A (ID:1) ← Person B disappears
Frame 50: Person A (ID:1), Person B (ID:5) ← Person B gets NEW ID (was missing 35 frames)
```

**JSON Output:**
```json
{
  "sequence_name": {
    "10_track_1": {"track_id": 1, ...},
    "10_track_2": {"track_id": 2, ...},
    "50_track_1": {"track_id": 1, ...},
    "50_track_5": {"track_id": 5, ...}  ← NEW ID (not 2)
  }
}
```

## Configuration Parameters

### TrackingArgs Class
```python
class TrackingArgs:
    def __init__(self):
        self.track_thresh = 0.5      # Minimum confidence for new tracks
        self.track_buffer = 30       # Frames to keep lost tracks
        self.match_thresh = 0.8      # Similarity threshold for re-association
        self.aspect_ratio_thresh = 1.6  # Filter vertical boxes
        self.min_box_area = 10       # Filter tiny boxes
        self.mot20 = False           # MOT20 dataset specific settings
```

### Key Parameters Explained:

- **`track_buffer = 30`**: How long to remember disappeared people
  - Increase for longer memory (more robust to occlusions)
  - Decrease for faster cleanup (less memory usage)

- **`match_thresh = 0.8`**: How similar detections must be to existing tracks
  - Higher = stricter matching (fewer false associations)
  - Lower = more lenient matching (may cause ID switches)

- **`track_thresh = 0.5`**: Minimum detection confidence to start new track
  - Higher = only track high-confidence detections
  - Lower = track more detections (may include false positives)

## Real-World Scenarios

### 🏃‍♂️ Sports/Activity Tracking
```
Frame 1-10:  Player A (ID:1), Player B (ID:2) running
Frame 11-15: Player A (ID:1) ← Player B behind Player A (occluded)
Frame 16-20: Player A (ID:1), Player B (ID:2) ← Player B emerges, same ID
```

### 🚶‍♀️ Surveillance/Monitoring
```
Frame 1-20:  Person A (ID:1), Person B (ID:2) walking
Frame 21-25: Person A (ID:1) ← Person B exits camera view
Frame 60:    Person A (ID:1), Person C (ID:3) ← New person enters
Frame 100:   Person A (ID:1), Person C (ID:3), Person D (ID:4) ← Person B returns as new ID
```

### 🎬 Video Analysis
```
Frame 1-50:  Actor A (ID:1), Actor B (ID:2) in scene
Frame 51-55: Actor A (ID:1) ← Actor B steps behind prop
Frame 56-60: Actor A (ID:1), Actor B (ID:2) ← Actor B reappears, same ID
```

## Visualization Files

The system generates several visualization files:

- **`temp.png`**: Shows YOLO detections with confidence scores
- **`temp_tracking.png`**: Shows tracked bounding boxes with unique colors and track IDs
- **`test_{sample_idx}_track_{track_id}.jpg`**: Pose visualization for specific tracks

## Output Format

Each tracked person generates entries in the JSON output:

```json
{
  "sequence_name": {
    "frame_track_ID": {
      "bbox_xyxy": [x1, y1, x2, y2],
      "track_id": 123,
      "confidence": 0.85,
      "joints_2d": {
        "joint_name": [x, y, confidence],
        ...
      }
    }
  }
}
```

## Best Practices

### For Robust Tracking:
1. **Adjust `track_buffer`** based on your use case:
   - Surveillance: 30-60 frames (1-2 seconds at 30fps)
   - Sports: 15-30 frames (fast movement)
   - Interviews: 60-90 frames (people may be stationary)

2. **Monitor track ID consistency** in your output
3. **Use visualization files** to debug tracking issues

### For Performance:
1. **Lower `track_buffer`** for faster processing
2. **Higher `track_thresh`** to reduce false tracks
3. **Adjust `min_box_area`** to filter noise

## Troubleshooting

### Problem: Too many new IDs for same person
**Solution**: Decrease `match_thresh` or increase `track_buffer`

### Problem: Wrong person gets same ID
**Solution**: Increase `match_thresh` or decrease `track_buffer`

### Problem: People disappear too quickly
**Solution**: Increase `track_buffer` or decrease `track_thresh`

### Problem: Too many false tracks
**Solution**: Increase `track_thresh` or `min_box_area`

## Technical Details

The tracking system uses:
- **Kalman Filter**: Predicts person movement during occlusions
- **IoU Matching**: Associates detections with existing tracks
- **Two-stage Association**: High and low confidence detection matching
- **Track State Management**: Tracks, Lost, Removed states

This ensures robust, real-time human tracking with persistent identity across video sequences.
