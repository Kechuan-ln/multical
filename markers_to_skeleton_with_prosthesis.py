#!/usr/bin/env python3
"""
Convert mocap markers to H36M skeleton + compute prosthesis rigid transformation.

Usage:
    python markers_to_skeleton_with_prosthesis.py \
        --mocap /path/to/mocap.csv \
        --marker_labels marker_labels_final.csv \
        --skeleton_config skeleton_config.json \
        --prosthesis_config prosthesis_config.json \
        --output skeleton_with_prosthesis.json
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_marker_labels(label_file):
    """Load marker ID to label mapping."""
    df = pd.read_csv(label_file)
    # Create mapping: marker_id -> label
    mapping = {}
    for _, row in df.iterrows():
        marker_id = row['marker_id']
        label = row['label']
        mapping[marker_id] = label
    return mapping


def load_mocap_csv_metadata(mocap_file):
    """Load Motive CSV metadata (marker IDs from line 5)."""
    with open(mocap_file, 'r') as f:
        lines = [f.readline() for _ in range(8)]

    # Line 5 (index 4) contains marker IDs
    # Line structure: Line 1=Format, 2=empty, 3=Type, 4=Name, 5=ID, 6=Parent, 7=Position, 8=Frame...
    id_line = lines[4].strip().split(',')
    # Skip first two columns (empty, "ID")
    all_ids = [mid.strip('"') for mid in id_line[2:] if mid]  # Remove quotes and skip empty

    # Each marker appears 3 times (X, Y, Z), so deduplicate
    # Keep order by using dict.fromkeys()
    unique_marker_ids = list(dict.fromkeys(all_ids))

    return unique_marker_ids


def load_mocap_frame(mocap_file, frame_num, marker_labels, marker_ids_cached=None):
    """Load mocap markers for a specific frame."""
    # Load data with header at row 8 (skiprows=7 to skip rows 1-7)
    df = pd.read_csv(mocap_file, skiprows=7, low_memory=False)

    # Clean column names (remove leading/trailing whitespace)
    df.columns = df.columns.str.strip()

    # Debug: print column names if Frame not found
    if 'Frame' not in df.columns:
        print(f"ERROR: 'Frame' column not found!")
        print(f"Available columns (first 10): {df.columns.tolist()[:10]}")
        raise KeyError("Frame column not found in mocap CSV")

    # Filter to specific frame
    frame_data = df[df['Frame'] == frame_num]

    if len(frame_data) == 0:
        print(f"WARNING: Frame {frame_num} not found in mocap data")
        print(f"Available frame range: {df['Frame'].min()} - {df['Frame'].max()}")
        return None

    # Get marker IDs if not cached
    if marker_ids_cached is None:
        marker_ids_cached = load_mocap_csv_metadata(mocap_file)

    # Extract markers
    markers = {}
    row = frame_data.iloc[0]

    # Each marker has 3 columns (X, Y, Z)
    # Column indices: Frame=0, Time=1, then markers start at index 2
    col_idx = 2  # Start of marker data columns

    matched_markers = 0
    for marker_id in marker_ids_cached:
        if marker_id in marker_labels:
            label = marker_labels[marker_id]

            # Try to get X, Y, Z for this marker
            try:
                x = row.iloc[col_idx]
                y = row.iloc[col_idx + 1]
                z = row.iloc[col_idx + 2]

                # Check if values are valid (not NaN)
                if pd.notna(x) and pd.notna(y) and pd.notna(z):
                    markers[label] = np.array([float(x), float(y), float(z)])
                    matched_markers += 1
            except (IndexError, ValueError):
                pass

        col_idx += 3  # Move to next marker (skip X, Y, Z)

    print(f"      Loaded {matched_markers} markers for frame {frame_num}")
    return markers


def compute_joint_from_markers(markers, joint_config):
    """Compute a single joint position from markers using the formula."""
    formula = joint_config['formula']
    marker_names = joint_config['markers']

    # Check if all required markers are available
    available_markers = []
    for marker_name in marker_names:
        if marker_name in markers:
            available_markers.append(markers[marker_name])

    if len(available_markers) == 0:
        return None

    # Apply formula
    if formula == 'mean':
        return np.mean(available_markers, axis=0)
    else:
        # Default to mean
        return np.mean(available_markers, axis=0)


def markers_to_skeleton(markers, skeleton_config):
    """Convert markers to skeleton joints."""
    joints = {}

    # Compute LHip and RHip first (needed for Pelvis)
    for joint_def in skeleton_config['skeleton_joints']:
        joint_name = joint_def['joint_name']
        if joint_name in ['LHip', 'RHip']:
            joint_pos = compute_joint_from_markers(markers, joint_def)
            if joint_pos is not None:
                markers[f"{joint_name}_computed"] = joint_pos

    # Now compute all joints including Pelvis
    for joint_def in skeleton_config['skeleton_joints']:
        joint_name = joint_def['joint_name']
        joint_id = joint_def['joint_id']

        # Skip RAnkle (will be computed from prosthesis)
        if joint_name == 'RAnkle':
            continue

        joint_pos = compute_joint_from_markers(markers, joint_def)

        if joint_pos is not None:
            joints[joint_name] = {
                'joint_id': joint_id,
                'position': joint_pos.tolist(),
                'valid': True
            }
        else:
            joints[joint_name] = {
                'joint_id': joint_id,
                'position': [0.0, 0.0, 0.0],
                'valid': False
            }

    return joints


def kabsch_rigid_transform(P, Q):
    """
    Compute optimal rotation and translation using Kabsch algorithm.

    Args:
        P: Source points (N x 3)
        Q: Target points (N x 3)

    Returns:
        R: Rotation matrix (3 x 3)
        t: Translation vector (3,)
        s: Scale factor (scalar)
    """
    assert P.shape == Q.shape, "Point sets must have the same shape"

    # Center the points
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute scale (optional, set to 1.0 for rigid-body only)
    scale_P = np.sqrt(np.sum(P_centered**2) / len(P))
    scale_Q = np.sqrt(np.sum(Q_centered**2) / len(Q))
    s = scale_Q / scale_P if scale_P > 1e-8 else 1.0

    # Normalize for rotation computation
    P_normalized = P_centered / (scale_P + 1e-8)
    Q_normalized = Q_centered / (scale_Q + 1e-8)

    # Compute covariance matrix
    H = P_normalized.T @ Q_normalized

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_Q - s * R @ centroid_P

    return R, t, s


def compute_prosthesis_transform(markers, prosthesis_config):
    """
    Compute rigid transformation for prosthesis mesh.

    Args:
        markers: Dict of marker_name -> position
        prosthesis_config: Prosthesis configuration with reference points

    Returns:
        transform_dict: {R, t, s, valid, ankle_position}
    """
    # Get the 4 prosthesis markers
    marker_names = prosthesis_config['marker_order']

    # Check if all 4 markers are available
    marker_positions = []
    for marker_name in marker_names:
        if marker_name in markers:
            marker_positions.append(markers[marker_name])
        else:
            # Missing marker
            return {
                'R': np.eye(3).tolist(),
                't': [0.0, 0.0, 0.0],
                's': 1.0,
                'valid': False,
                'ankle_position': [0.0, 0.0, 0.0]
            }

    # Source points: reference points on mesh (from annotation tool)
    P_mesh = []
    for marker_name in marker_names:
        pt = prosthesis_config['points'][marker_name]['position']
        P_mesh.append(pt)
    P_mesh = np.array(P_mesh)

    # Target points: actual marker positions
    Q_markers = np.array(marker_positions)

    # Compute transformation
    R, t, s = kabsch_rigid_transform(P_mesh, Q_markers)

    # Compute RAnkle position (center of 4 markers)
    ankle_position = np.mean(Q_markers, axis=0)

    return {
        'R': R.tolist(),
        't': t.tolist(),
        's': float(s),
        'valid': True,
        'ankle_position': ankle_position.tolist(),
        'marker_positions': {
            marker_names[i]: Q_markers[i].tolist()
            for i in range(len(marker_names))
        }
    }


def apply_transform_to_mesh(vertices, R, t, s):
    """Apply rigid transformation to mesh vertices."""
    vertices_transformed = s * (vertices @ R.T) + t
    return vertices_transformed


def load_prosthesis_mesh(stl_path):
    """Load prosthesis STL mesh."""
    import stl
    mesh = stl.mesh.Mesh.from_file(stl_path)
    vertices = mesh.vectors.reshape(-1, 3)
    return vertices


def process_frame(frame_num, mocap_file, marker_labels, skeleton_config,
                  prosthesis_config, prosthesis_vertices, marker_ids_cached=None):
    """Process a single frame."""
    # Load markers
    markers = load_mocap_frame(mocap_file, frame_num, marker_labels, marker_ids_cached)

    if markers is None or len(markers) == 0:
        print(f"      ERROR: No markers loaded for frame {frame_num}")
        return None

    print(f"      Converting {len(markers)} markers to skeleton...")

    # Convert to skeleton
    skeleton = markers_to_skeleton(markers, skeleton_config)

    # Count valid joints
    valid_joints = sum(1 for j in skeleton.values() if j['valid'])
    print(f"      Skeleton has {valid_joints}/{len(skeleton)} valid joints")

    # Compute prosthesis transformation
    prosthesis_transform = compute_prosthesis_transform(markers, prosthesis_config)
    print(f"      Prosthesis transform valid: {prosthesis_transform['valid']}")

    # Add RAnkle joint (set as INVALID - replaced by prosthesis mesh)
    skeleton['RAnkle'] = {
        'joint_id': 16,
        'position': [0.0, 0.0, 0.0],  # Not computed
        'valid': False  # Always invalid - use prosthesis mesh instead
    }

    # Transform mesh if valid
    transformed_mesh = None
    if prosthesis_transform['valid']:
        R = np.array(prosthesis_transform['R'])
        t = np.array(prosthesis_transform['t'])
        s = prosthesis_transform['s']

        # Apply transformation to mesh vertices
        transformed_vertices = apply_transform_to_mesh(prosthesis_vertices, R, t, s)
        transformed_mesh = transformed_vertices.tolist()
        print(f"      Transformed mesh: {len(transformed_mesh)} vertices")

    return {
        'frame': frame_num,
        'skeleton': skeleton,
        'prosthesis_transform': prosthesis_transform,
        'prosthesis_mesh': transformed_mesh
    }


def main():
    parser = argparse.ArgumentParser(description='Convert markers to skeleton with prosthesis')
    parser.add_argument('--mocap', required=True, help='Mocap CSV file')
    parser.add_argument('--marker_labels', required=True, help='Marker labels CSV')
    parser.add_argument('--skeleton_config', required=True, help='Skeleton configuration JSON')
    parser.add_argument('--prosthesis_config', required=True, help='Prosthesis configuration JSON')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--frames', type=str, default=None,
                       help='Frame range (e.g., "3830-3840" or "3832")')

    args = parser.parse_args()

    # Load configurations
    print("Loading configurations...")
    marker_labels = load_marker_labels(args.marker_labels)

    with open(args.skeleton_config, 'r') as f:
        skeleton_config = json.load(f)

    with open(args.prosthesis_config, 'r') as f:
        prosthesis_config = json.load(f)

    # Load mocap metadata (marker IDs)
    print(f"Loading mocap metadata from: {args.mocap}")
    marker_ids_cached = load_mocap_csv_metadata(args.mocap)
    print(f"  Found {len(marker_ids_cached)} markers in mocap file")
    print(f"  Sample marker IDs from mocap: {marker_ids_cached[:3]}")
    print(f"  Sample marker IDs from labels: {list(marker_labels.keys())[:3]}")

    # Load prosthesis mesh
    print(f"Loading prosthesis mesh: {prosthesis_config['stl_file']}")
    prosthesis_vertices = load_prosthesis_mesh(prosthesis_config['stl_file'])

    # Center the mesh (same as in annotation tool)
    mesh_center = np.array(prosthesis_config['mesh_center'])
    prosthesis_vertices = prosthesis_vertices - mesh_center

    # Determine frames to process
    if args.frames:
        if '-' in args.frames:
            start, end = map(int, args.frames.split('-'))
            frames = list(range(start, end + 1))
        else:
            frames = [int(args.frames)]
    else:
        # Read all frames from mocap file
        df = pd.read_csv(args.mocap, skiprows=7, low_memory=False)
        df.columns = df.columns.str.strip()  # Clean column names
        frames = sorted(df['Frame'].unique())

    print(f"Processing {len(frames)} frames...")

    # Process frames
    results = []
    for frame_num in frames:
        print(f"  Processing frame {frame_num}...")

        frame_data = process_frame(
            frame_num, args.mocap, marker_labels, skeleton_config,
            prosthesis_config, prosthesis_vertices, marker_ids_cached
        )

        if frame_data is not None:
            results.append(frame_data)
            print(f"    ✓ Frame {frame_num} processed successfully")
        else:
            print(f"    ✗ Frame {frame_num} returned None (skipped)")

    # Save results
    output_data = {
        'metadata': {
            'mocap_file': args.mocap,
            'marker_labels_file': args.marker_labels,
            'skeleton_config_file': args.skeleton_config,
            'prosthesis_config_file': args.prosthesis_config,
            'num_frames': len(results),
            'frame_range': [frames[0], frames[-1]] if frames else [0, 0]
        },
        'frames': results
    }

    print(f"\nSaving results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Successfully processed {len(results)} frames")
    print(f"✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
