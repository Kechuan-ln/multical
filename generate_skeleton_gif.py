#!/usr/bin/env python3
"""
Generate animated GIF of skeleton + prosthesis from processed data.

Usage:
    python generate_skeleton_gif.py \
        --input skeleton_with_prosthesis.json \
        --output skeleton_animation.gif \
        --fps 10 \
        --duration 5.0
"""

import numpy as np
import plotly.graph_objects as go
import json
import argparse
from PIL import Image
import io
import tempfile
from pathlib import Path


# H36M skeleton connections
H36M_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (5, 6),
    (4, 7), (7, 8), (8, 9), (4, 10), (10, 11), (11, 12),
    (1, 13), (13, 14), (2, 15), (15, 16)
]

JOINT_NAMES = [
    "Pelvis", "LHip", "RHip", "Spine1", "Neck", "Head", "Jaw",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
    "LKnee", "LAnkle", "RKnee", "RAnkle"
]


def create_frame_figure(frame_data, view_angle='default'):
    """Create plotly figure for a single frame."""
    fig = go.Figure()

    # Extract skeleton
    skeleton = frame_data['skeleton']
    joint_positions = {}

    for joint_name, joint_info in skeleton.items():
        joint_id = joint_info['joint_id']
        position = np.array(joint_info['position'])
        valid = joint_info['valid']
        joint_positions[joint_id] = {
            'name': joint_name,
            'position': position,
            'valid': valid
        }

    # Get all joint positions
    joint_pos_array = []
    joint_valid_array = []
    for joint_id in range(17):
        if joint_id in joint_positions and joint_positions[joint_id]['valid']:
            joint_pos_array.append(joint_positions[joint_id]['position'])
            joint_valid_array.append(True)
        else:
            joint_pos_array.append([0, 0, 0])
            joint_valid_array.append(False)

    joint_pos_array = np.array(joint_pos_array)

    # Create bones
    bone_x, bone_y, bone_z = [], [], []
    for (joint1_id, joint2_id) in H36M_CONNECTIONS:
        if joint_valid_array[joint1_id] and joint_valid_array[joint2_id]:
            pos1 = joint_pos_array[joint1_id]
            pos2 = joint_pos_array[joint2_id]
            bone_x.extend([pos1[0], pos2[0], None])
            bone_y.extend([pos1[1], pos2[1], None])
            bone_z.extend([pos1[2], pos2[2], None])

    # Add bones
    if bone_x:
        fig.add_trace(go.Scatter3d(
            x=bone_x, y=bone_y, z=bone_z,
            mode='lines',
            line=dict(color='blue', width=6),
            name='Skeleton',
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add joints
    valid_joints_x, valid_joints_y, valid_joints_z = [], [], []
    for joint_id in range(17):
        if joint_valid_array[joint_id]:
            pos = joint_pos_array[joint_id]
            valid_joints_x.append(pos[0])
            valid_joints_y.append(pos[1])
            valid_joints_z.append(pos[2])

    if valid_joints_x:
        fig.add_trace(go.Scatter3d(
            x=valid_joints_x,
            y=valid_joints_y,
            z=valid_joints_z,
            mode='markers',
            marker=dict(size=6, color='red'),
            name='Joints',
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add prosthesis mesh
    mesh_vertices = frame_data.get('prosthesis_mesh', None)
    if mesh_vertices:
        vertices = np.array(mesh_vertices)
        n_vertices = len(vertices)
        faces = np.arange(n_vertices).reshape(-1, 3)

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightgreen',
            opacity=0.7,
            name='Prosthesis',
            showlegend=False
        ))

    # Set camera view
    camera_dict = None
    if view_angle == 'side':
        camera_dict = dict(
            eye=dict(x=2.0, y=0, z=0.3),
            up=dict(x=0, y=0, z=1)
        )
    elif view_angle == 'front':
        camera_dict = dict(
            eye=dict(x=0, y=-2.0, z=0.3),
            up=dict(x=0, y=0, z=1)
        )
    elif view_angle == 'top':
        camera_dict = dict(
            eye=dict(x=0, y=0, z=2.5),
            up=dict(x=0, y=1, z=0)
        )
    else:  # default isometric
        camera_dict = dict(
            eye=dict(x=1.5, y=-1.5, z=1.0),
            up=dict(x=0, y=0, z=1)
        )

    # Compute axis limits based on all valid joint positions
    if valid_joints_x:
        x_coords = valid_joints_x
        y_coords = valid_joints_y
        z_coords = valid_joints_z

        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        z_center = np.mean(z_coords)

        # Set consistent range
        range_size = 1500  # mm

        x_range = [x_center - range_size, x_center + range_size]
        y_range = [y_center - range_size, y_center + range_size]
        z_range = [z_center - range_size, z_center + range_size]
    else:
        x_range = [-1500, 1500]
        y_range = [-1500, 1500]
        z_range = [0, 3000]

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm)', range=x_range, showgrid=True),
            yaxis=dict(title='Y (mm)', range=y_range, showgrid=True),
            zaxis=dict(title='Z (mm)', range=z_range, showgrid=True),
            aspectmode='cube',
            camera=camera_dict
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='white',
        width=800,
        height=600,
        title=dict(
            text=f"Frame {frame_data['frame']}",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        showlegend=False
    )

    return fig


def generate_gif(input_file, output_file, fps=10, duration=None,
                view_angle='default', frame_range=None):
    """Generate animated GIF from skeleton data."""

    print(f"Loading data from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    frames = data['frames']

    if len(frames) == 0:
        print("ERROR: No frames in data!")
        return

    # Filter frame range if specified
    if frame_range:
        start_frame, end_frame = frame_range
        frames = [f for f in frames if start_frame <= f['frame'] <= end_frame]
        print(f"Filtered to {len(frames)} frames in range [{start_frame}, {end_frame}]")

    # Limit duration if specified
    if duration:
        max_frames = int(duration * fps)
        if len(frames) > max_frames:
            # Sample frames evenly
            indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            print(f"Sampled to {len(frames)} frames for {duration}s duration at {fps} fps")

    print(f"Generating GIF with {len(frames)} frames at {fps} fps...")
    print(f"View angle: {view_angle}")

    # Generate images
    images = []
    temp_dir = tempfile.mkdtemp()

    for i, frame_data in enumerate(frames):
        if i % 10 == 0:
            print(f"  Rendering frame {i+1}/{len(frames)}...")

        fig = create_frame_figure(frame_data, view_angle)

        # Convert plotly figure to image
        img_bytes = fig.to_image(format="png", width=800, height=600)
        img = Image.open(io.BytesIO(img_bytes))
        images.append(img)

    print(f"Saving GIF to: {output_file}")

    # Calculate frame duration in milliseconds
    frame_duration = int(1000 / fps)

    # Save as GIF
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration,
        loop=0,  # Loop forever
        optimize=False
    )

    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"✓ GIF saved: {output_file} ({file_size_mb:.1f} MB)")
    print(f"  Total frames: {len(images)}")
    print(f"  Frame rate: {fps} fps")
    print(f"  Duration: {len(images)/fps:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Generate animated GIF of skeleton + prosthesis')
    parser.add_argument('--input', required=True, help='Input JSON file with skeleton data')
    parser.add_argument('--output', default='skeleton_animation.gif', help='Output GIF file')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second (default: 10)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Maximum duration in seconds (default: all frames)')
    parser.add_argument('--view', choices=['default', 'side', 'front', 'top'],
                       default='default', help='Camera view angle')
    parser.add_argument('--frame_range', type=str, default=None,
                       help='Frame range to process (e.g., "3800-3900")')

    args = parser.parse_args()

    # Parse frame range
    frame_range = None
    if args.frame_range:
        start, end = map(int, args.frame_range.split('-'))
        frame_range = (start, end)

    generate_gif(
        args.input,
        args.output,
        fps=args.fps,
        duration=args.duration,
        view_angle=args.view,
        frame_range=frame_range
    )


if __name__ == "__main__":
    main()
