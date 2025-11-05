#!/usr/bin/env python3
"""
Convert mocap markers to skeleton joints (17 joints in H36M format).

Usage:
    python markers_to_skeleton.py \
        --mocap_csv /path/to/mocap.csv \
        --labels_csv marker_labels.csv \
        --config skeleton_config.json \
        --output skeleton_joints.csv \
        --start_frame 2 \
        --end_frame 23374
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path


def load_marker_labels(labels_csv):
    """
    Load marker label mappings.

    Returns:
        dict: {label -> original_name}
    """
    if not Path(labels_csv).exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")

    df = pd.read_csv(labels_csv)

    # Create reverse mapping: label -> original_name
    label_to_original = {}
    for _, row in df.iterrows():
        label_to_original[row['label']] = row['original_name']

    print(f"Loaded {len(label_to_original)} marker labels")
    return label_to_original


def load_mocap_data(mocap_csv):
    """
    Load mocap CSV data.

    Returns:
        metadata: dict with capture info
        marker_names: list of original marker names
        data_df: DataFrame with marker positions
    """
    # Read metadata from first row
    with open(mocap_csv, 'r') as f:
        first_line = f.readline().strip()

    metadata_parts = first_line.split(',')
    metadata = {}
    for i in range(0, len(metadata_parts), 2):
        if i+1 < len(metadata_parts):
            metadata[metadata_parts[i]] = metadata_parts[i+1]

    # Read marker names from header
    marker_info = pd.read_csv(mocap_csv, skiprows=1, nrows=4, header=None)
    marker_names_row = marker_info.iloc[1, 2:]  # Name row

    marker_names = []
    for i in range(0, len(marker_names_row), 3):
        if pd.notna(marker_names_row.iloc[i]):
            marker_names.append(marker_names_row.iloc[i])

    # Read actual data
    data_df = pd.read_csv(mocap_csv, skiprows=7)
    data_df.columns = data_df.columns.str.strip()

    print(f"Loaded mocap data: {len(data_df)} frames, {len(marker_names)} markers")

    return metadata, marker_names, data_df


def extract_marker_positions(data_df, marker_names, label_to_original, start_frame, end_frame):
    """
    Extract marker positions and organize by label.

    Returns:
        dict: {label -> (num_frames, 3) array of [x, y, z]}
    """
    # Create mapping: original_name -> column_index
    original_to_col_idx = {}
    first_data_col = 2  # After Frame and Time

    for i, marker_name in enumerate(marker_names):
        col_idx = first_data_col + i * 3  # X column index
        original_to_col_idx[marker_name] = col_idx

    # Extract positions for each labeled marker
    marker_positions = {}

    for label, original_name in label_to_original.items():
        if original_name not in original_to_col_idx:
            print(f"Warning: Marker '{original_name}' (label: '{label}') not found in mocap data")
            continue

        col_idx = original_to_col_idx[original_name]

        # Extract X, Y, Z for the frame range
        x = data_df.iloc[start_frame:end_frame, col_idx].values
        y = data_df.iloc[start_frame:end_frame, col_idx + 1].values
        z = data_df.iloc[start_frame:end_frame, col_idx + 2].values

        # Stack into (num_frames, 3)
        xyz = np.stack([x, y, z], axis=1)
        marker_positions[label] = xyz

    print(f"Extracted positions for {len(marker_positions)} labeled markers")
    return marker_positions


def compute_joint(joint_def, marker_positions, computed_joints):
    """
    Compute joint position based on formula.

    Args:
        joint_def: Joint definition from config
        marker_positions: dict of marker positions
        computed_joints: dict of already computed joints (for dependencies)

    Returns:
        (num_frames, 3) array or None if markers missing
    """
    formula = joint_def['formula']
    required_markers = joint_def['markers']

    # Collect marker/joint positions
    positions = []
    missing_markers = []

    for marker_label in required_markers:
        # Check if it's a computed joint (ends with _computed)
        if marker_label.endswith('_computed'):
            base_name = marker_label.replace('_computed', '')
            if base_name in computed_joints:
                positions.append(computed_joints[base_name])
            else:
                missing_markers.append(marker_label)
        # Check if it's a labeled marker
        elif marker_label in marker_positions:
            positions.append(marker_positions[marker_label])
        else:
            missing_markers.append(marker_label)

    if missing_markers:
        print(f"  Warning: Joint '{joint_def['joint_name']}' missing markers: {missing_markers}")
        return None

    if len(positions) == 0:
        return None

    # Apply formula
    if formula == 'mean':
        # Stack all positions and compute mean
        # positions is a list of (num_frames, 3) arrays
        stacked = np.stack(positions, axis=0)  # (num_markers, num_frames, 3)

        # For each frame, compute mean across markers (ignoring NaN)
        joint_pos = np.nanmean(stacked, axis=0)  # (num_frames, 3)

        return joint_pos
    else:
        raise ValueError(f"Unknown formula: {formula}")


def markers_to_skeleton(marker_positions, skeleton_config):
    """
    Convert marker positions to skeleton joints.

    Args:
        marker_positions: dict {label -> (num_frames, 3) array}
        skeleton_config: skeleton configuration dict

    Returns:
        skeleton_joints: dict {joint_name -> (num_frames, 3) array}
        joint_order: list of joint names in order
    """
    joints_def = skeleton_config['skeleton_joints']
    skeleton_joints = {}

    print("\nComputing skeleton joints...")

    # First pass: compute joints that don't depend on other computed joints
    # Second pass: compute joints that depend on computed joints (like Pelvis)

    for pass_num in range(2):
        for joint_def in joints_def:
            joint_name = joint_def['joint_name']

            # Skip if already computed
            if joint_name in skeleton_joints:
                continue

            # Check if this joint depends on computed joints
            requires_computed = any(m.endswith('_computed') for m in joint_def['markers'])

            # First pass: only compute non-dependent joints
            # Second pass: compute dependent joints
            if (pass_num == 0 and requires_computed) or (pass_num == 1 and not requires_computed):
                continue

            print(f"  Computing {joint_name}...")
            joint_pos = compute_joint(joint_def, marker_positions, skeleton_joints)

            if joint_pos is not None:
                skeleton_joints[joint_name] = joint_pos
                print(f"    ✓ {joint_name} computed successfully")
            else:
                print(f"    ✗ {joint_name} failed (missing markers)")

    # Get joint order from config
    joint_order = [j['joint_name'] for j in joints_def]

    return skeleton_joints, joint_order


def save_skeleton_csv(skeleton_joints, joint_order, output_csv, start_frame, metadata):
    """
    Save skeleton joints to CSV file.

    Format:
    Frame, Time, Joint1_X, Joint1_Y, Joint1_Z, Joint2_X, ...
    """
    num_frames = len(list(skeleton_joints.values())[0])

    # Create column names
    columns = ['Frame', 'Time']
    for joint_name in joint_order:
        columns.extend([f'{joint_name}_X', f'{joint_name}_Y', f'{joint_name}_Z'])

    # Prepare data
    data_rows = []
    fps = float(metadata.get('Capture Frame Rate', 120.0))

    for frame_idx in range(num_frames):
        frame_num = start_frame + frame_idx
        time = frame_num / fps

        row = [frame_num, time]

        for joint_name in joint_order:
            if joint_name in skeleton_joints:
                pos = skeleton_joints[joint_name][frame_idx]
                row.extend([pos[0], pos[1], pos[2]])
            else:
                row.extend([np.nan, np.nan, np.nan])

        data_rows.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv(output_csv, index=False)

    print(f"\nSkeleton joints saved to: {output_csv}")
    print(f"  Frames: {num_frames}")
    print(f"  Joints: {len([j for j in joint_order if j in skeleton_joints])}/{len(joint_order)}")


def save_skeleton_json(skeleton_joints, joint_order, output_json, start_frame, metadata):
    """
    Save skeleton joints to JSON file (frame-by-frame format).

    Format:
    {
        "metadata": {...},
        "frames": {
            "0": {
                "frame_num": 2,
                "time": 0.0167,
                "joints": {
                    "Pelvis": [x, y, z],
                    "LHip": [x, y, z],
                    ...
                }
            },
            ...
        }
    }
    """
    num_frames = len(list(skeleton_joints.values())[0])
    fps = float(metadata.get('Capture Frame Rate', 120.0))

    output_data = {
        'metadata': {
            'fps': fps,
            'start_frame': start_frame,
            'num_frames': num_frames,
            'num_joints': len(joint_order),
            'joint_names': joint_order,
            'coordinate_system': 'Y-up (vertical), XZ-horizontal'
        },
        'frames': {}
    }

    for frame_idx in range(num_frames):
        frame_num = start_frame + frame_idx
        time = frame_num / fps

        frame_data = {
            'frame_num': frame_num,
            'time': time,
            'joints': {}
        }

        for joint_name in joint_order:
            if joint_name in skeleton_joints:
                pos = skeleton_joints[joint_name][frame_idx]
                # Convert to list, handling NaN
                pos_list = [float(p) if not np.isnan(p) else None for p in pos]
                frame_data['joints'][joint_name] = pos_list
            else:
                frame_data['joints'][joint_name] = [None, None, None]

        output_data['frames'][str(frame_idx)] = frame_data

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Skeleton joints saved to: {output_json}")


def validate_markers(label_to_original, skeleton_config):
    """
    Validate that all required markers are labeled.
    """
    required_markers = set()

    for joint_def in skeleton_config['skeleton_joints']:
        for marker in joint_def['markers']:
            # Skip computed joints
            if not marker.endswith('_computed'):
                required_markers.add(marker)

    labeled_markers = set(label_to_original.keys())

    missing_markers = required_markers - labeled_markers

    if missing_markers:
        print("\n" + "="*60)
        print("WARNING: Missing marker labels!")
        print("="*60)
        print("The following markers are required but not labeled:")
        for marker in sorted(missing_markers):
            print(f"  - {marker}")
        print("\nPlease label these markers using annotate_mocap_markers.py")
        print("="*60 + "\n")
        return False
    else:
        print("\n✓ All required markers are labeled!")
        return True


def main():
    parser = argparse.ArgumentParser(description='Convert mocap markers to skeleton joints')
    parser.add_argument('--mocap_csv', type=str,
                        default='/Volumes/FastACIS/csldata/csl/mocap.csv',
                        help='Path to mocap CSV file')
    parser.add_argument('--labels_csv', type=str,
                        default='marker_labels.csv',
                        help='Path to marker labels CSV file')
    parser.add_argument('--config', type=str,
                        default='skeleton_config.json',
                        help='Path to skeleton configuration JSON file')
    parser.add_argument('--output_csv', type=str,
                        default='skeleton_joints.csv',
                        help='Output CSV file for skeleton joints')
    parser.add_argument('--output_json', type=str,
                        default='skeleton_joints.json',
                        help='Output JSON file for skeleton joints')
    parser.add_argument('--start_frame', type=int, default=2,
                        help='Start frame')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='End frame (None = all frames)')

    args = parser.parse_args()

    print("="*60)
    print("Markers to Skeleton Converter")
    print("="*60)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        skeleton_config = json.load(f)
    print(f"  Skeleton has {len(skeleton_config['skeleton_joints'])} joints")

    # Load marker labels
    print(f"\nLoading marker labels from: {args.labels_csv}")
    label_to_original = load_marker_labels(args.labels_csv)

    # Validate markers (just warning, don't abort)
    validate_markers(label_to_original, skeleton_config)

    # Load mocap data
    print(f"\nLoading mocap data from: {args.mocap_csv}")
    metadata, marker_names, data_df = load_mocap_data(args.mocap_csv)

    # Determine frame range
    if args.end_frame is None:
        args.end_frame = len(data_df)

    print(f"\nProcessing frames {args.start_frame} to {args.end_frame}")
    num_frames = args.end_frame - args.start_frame
    print(f"Total frames to process: {num_frames}")

    # Extract marker positions
    print("\nExtracting marker positions...")
    marker_positions = extract_marker_positions(
        data_df, marker_names, label_to_original,
        args.start_frame, args.end_frame
    )

    # Convert to skeleton
    skeleton_joints, joint_order = markers_to_skeleton(marker_positions, skeleton_config)

    # Save outputs
    print("\nSaving results...")
    save_skeleton_csv(skeleton_joints, joint_order, args.output_csv, args.start_frame, metadata)
    save_skeleton_json(skeleton_joints, joint_order, args.output_json, args.start_frame, metadata)

    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  CSV: {args.output_csv}")
    print(f"  JSON: {args.output_json}")
    print("\nYou can now use these skeleton joints for further analysis.")
    print("="*60)


if __name__ == '__main__':
    main()
