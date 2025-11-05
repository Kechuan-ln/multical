#!/usr/bin/env python3
"""
Interactive mocap marker annotation tool with 2D+3D views using Dash.

Features:
- 3D scatter plot of markers
- 2D video view with projected markers
- Click on markers in either view to select
- Synchronized selection between 2D and 3D
- Interactive labeling with auto-save

Usage:
    python annotate_mocap_markers_2d3d.py \
        --csv /path/to/mocap.csv \
        --video /path/to/video.avi \
        --mcal /path/to/calibration.mcal \
        --start_frame 2 \
        --num_frames 200
"""

import pandas as pd
import numpy as np
import cv2
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image


def load_optitrack_calibration(mcal_path, camera_serial='C11764',
                               intrinsics_json=None, extrinsics_json=None):
    """
    Load OptiTrack calibration from .mcal file or custom JSON files.

    Args:
        mcal_path: Path to .mcal file (for image size and default calibration)
        camera_serial: Camera serial number
        intrinsics_json: Optional path to custom intrinsics JSON (multical format)
        extrinsics_json: Optional path to custom extrinsics JSON

    Returns:
        K: Camera intrinsic matrix (3x3) with NEGATIVE fx for coordinate conversion
        dist: Distortion coefficients [k1, k2, p1, p2, k3]
        rvec: Rotation vector (3x1)
        tvec: Translation vector (3x1)
        img_size: [width, height]
    """
    # Parse XML (UTF-16LE encoding)
    tree = ET.parse(mcal_path)
    root = tree.getroot()

    for cam in root.findall('.//Camera'):
        if cam.get('Serial') == camera_serial:
            # Image size (always from .mcal)
            attributes = cam.find('Attributes')
            width = int(attributes.get('ImagerPixelWidth'))
            height = int(attributes.get('ImagerPixelHeight'))

            # Load intrinsics and extrinsics
            # Priority: intrinsics_json > extrinsics_json (if it contains intrinsics) > .mcal
            extr_data_cached = None  # Cache for extrinsics data if loaded early
            intrinsics_loaded = False
            intrinsics_source = None

            if intrinsics_json:
                print(f"  Loading custom intrinsics from: {intrinsics_json}")
                with open(intrinsics_json, 'r') as f:
                    intr_data = json.load(f)

                # Find camera data (try multiple names)
                cam_data = None
                for cam_name in ['primecolor', 'C11764', camera_serial]:
                    if cam_name in intr_data.get('cameras', {}):
                        cam_data = intr_data['cameras'][cam_name]
                        print(f"    Found camera: {cam_name}")
                        break

                if cam_data is None:
                    raise ValueError(f"Camera not found in intrinsics JSON. Available: {list(intr_data.get('cameras', {}).keys())}")

                # Multical format: K is 3x3 matrix
                K_multical = np.array(cam_data['K'], dtype=np.float64)
                fx = K_multical[0, 0]
                fy = K_multical[1, 1]
                cx = K_multical[0, 2]
                cy = K_multical[1, 2]

                # Distortion coefficients
                dist = np.array(cam_data['dist'], dtype=np.float64).flatten()

                print(f"    fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                intrinsics_loaded = True
                intrinsics_source = 'custom_intrinsics_json'
            elif extrinsics_json:
                # Load extrinsics JSON and check if it contains intrinsics
                with open(extrinsics_json, 'r') as f:
                    extr_data_cached = json.load(f)

                if 'intrinsics' in extr_data_cached:
                    print(f"  Loading intrinsics from extrinsics JSON: {extrinsics_json}")
                    intr = extr_data_cached['intrinsics']
                    fx = float(intr['fx'])
                    fy = float(intr['fy'])
                    cx = float(intr['cx'])
                    cy = float(intr['cy'])
                    dist = np.array(intr['dist'], dtype=np.float64).flatten()

                    source = extr_data_cached.get('intrinsics_source', 'unknown')
                    print(f"    Using {source} intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                    intrinsics_loaded = True
                    intrinsics_source = f'extrinsics_json({source})'

            if not intrinsics_loaded:
                # Load from .mcal
                print(f"  Loading intrinsics from .mcal")
                intrinsic = cam.find('.//IntrinsicStandardCameraModel')
                fx = float(intrinsic.get('HorizontalFocalLength'))
                fy = float(intrinsic.get('VerticalFocalLength'))
                cx = float(intrinsic.get('LensCenterX'))
                cy = float(intrinsic.get('LensCenterY'))
                k1 = float(intrinsic.get('k1'))
                k2 = float(intrinsic.get('k2'))
                k3 = float(intrinsic.get('k3'))
                p1 = float(intrinsic.get('TangentialX'))
                p2 = float(intrinsic.get('TangentialY'))

                dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
                intrinsics_source = 'mcal'

            # KEY: Use NEGATIVE fx for OptiTrack coordinate system
            K = np.array([[-fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float64)

            # Load extrinsics
            if extrinsics_json:
                print(f"  Loading custom extrinsics from: {extrinsics_json}")
                # Use cached data if already loaded
                if extr_data_cached is None:
                    with open(extrinsics_json, 'r') as f:
                        extr_data_cached = json.load(f)

                rvec = np.array(extr_data_cached['rvec'], dtype=np.float64).reshape(3, 1)
                tvec = np.array(extr_data_cached['tvec'], dtype=np.float64).reshape(3, 1)

                print(f"    Camera position (world): {extr_data_cached.get('camera_position_world', 'N/A')}")
                if intrinsics_source.startswith('extrinsics_json'):
                    print(f"    ✓ Using calibrated intrinsics+extrinsics from same source")
            else:
                # Load from .mcal
                extrinsic = cam.find('Extrinsic')
                T_world = np.array([
                    float(extrinsic.get('X')),
                    float(extrinsic.get('Y')),
                    float(extrinsic.get('Z'))
                ])

                R_c2w = np.array([
                    [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(3)],
                    [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(3, 6)],
                    [float(extrinsic.get(f'OrientMatrix{i}')) for i in range(6, 9)]
                ])

                # Convert Camera-to-World to World-to-Camera
                R_w2c = R_c2w.T
                rvec, _ = cv2.Rodrigues(R_w2c)
                tvec = -R_w2c @ T_world

            return K, dist, rvec, tvec, [width, height]

    raise ValueError(f"Camera with serial {camera_serial} not found in {mcal_path}")


def project_markers_to_2d(markers_3d, K, dist, rvec, tvec, img_size):
    """
    Project 3D markers (in mm) to 2D image coordinates.

    Args:
        markers_3d: Nx3 array of 3D positions in mm (world coordinates)
        K: Camera intrinsic matrix (3x3)
        dist: Distortion coefficients (5,)
        rvec: Rotation vector (3x1)
        tvec: Translation vector (3x1)
        img_size: [width, height]

    Returns:
        points_2d: Nx2 array of 2D pixel coordinates
        valid_mask: N boolean array indicating points within image bounds
    """
    # Filter NaN values
    valid_mask = ~np.isnan(markers_3d).any(axis=1)
    if not valid_mask.any():
        return None, valid_mask

    # Convert mm to meters
    markers_m = markers_3d[valid_mask] / 1000.0

    # Project to 2D
    points_2d, _ = cv2.projectPoints(
        markers_m.reshape(-1, 1, 3),
        rvec, tvec, K, dist
    )
    points_2d = points_2d.reshape(-1, 2)

    # Check image bounds (don't check Z > 0 due to negative fx)
    in_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_size[0]) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_size[1])

    # Create full-size result with NaN for invalid points
    result_2d = np.full((len(markers_3d), 2), np.nan)
    valid_indices = np.where(valid_mask)[0]
    result_2d[valid_indices[in_bounds]] = points_2d[in_bounds]

    final_valid = ~np.isnan(result_2d[:, 0])

    return result_2d, final_valid


def parse_motive_csv(csv_path):
    """Parse Motive CSV export file."""
    # Read metadata from first row
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()

    metadata_parts = first_line.split(',')
    metadata = {}
    for i in range(0, len(metadata_parts), 2):
        if i+1 < len(metadata_parts):
            metadata[metadata_parts[i]] = metadata_parts[i+1]

    # Read the Name row (row 3, which is index 1 after skiprows=1)
    marker_info = pd.read_csv(csv_path, skiprows=1, nrows=4, header=None)
    marker_names_row = marker_info.iloc[1, 2:]  # Name row
    marker_ids_row = marker_info.iloc[2, 2:]     # ID row

    # Extract unique marker names (every 3rd column since X,Y,Z)
    marker_names = []
    marker_ids = []
    for i in range(0, len(marker_names_row), 3):
        if pd.notna(marker_names_row.iloc[i]):
            marker_names.append(marker_names_row.iloc[i])
            marker_ids.append(marker_ids_row.iloc[i])

    # Read actual data (skip first 7 rows)
    data_df = pd.read_csv(csv_path, skiprows=7)
    data_df.columns = data_df.columns.str.strip()

    return metadata, marker_names, marker_ids, data_df


def extract_marker_data(data_df, marker_names):
    """Extract marker positions from data DataFrame."""
    markers_xyz = {}
    first_data_col = 2  # After Frame and Time

    for i, marker_name in enumerate(marker_names):
        x_col_idx = first_data_col + i * 3
        y_col_idx = first_data_col + i * 3 + 1
        z_col_idx = first_data_col + i * 3 + 2

        if z_col_idx < len(data_df.columns):
            x = data_df.iloc[:, x_col_idx].values
            y = data_df.iloc[:, y_col_idx].values
            z = data_df.iloc[:, z_col_idx].values
            xyz = np.stack([x, y, z], axis=1)
            markers_xyz[marker_name] = xyz

    return markers_xyz


def load_labels(labels_file):
    """Load existing labels from CSV file."""
    if os.path.exists(labels_file):
        df = pd.read_csv(labels_file)
        labels = {}
        for _, row in df.iterrows():
            labels[row['original_name']] = {
                'label': row['label'],
                'marker_id': row['marker_id']
            }
        return labels
    return {}


def save_labels(labels, labels_file):
    """Save labels to CSV file."""
    rows = []
    for original_name, info in labels.items():
        rows.append({
            'original_name': original_name,
            'marker_id': info['marker_id'],
            'label': info['label']
        })
    df = pd.DataFrame(rows)
    df.to_csv(labels_file, index=False)
    print(f"✓ Labels saved to {labels_file}")


class VideoFrameReader:
    """Efficient video frame reader with caching."""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cache = {}
        self.cache_size = 50  # Cache last 50 frames

    def get_frame(self, frame_idx):
        """Get frame by index (with caching)."""
        if frame_idx in self.cache:
            return self.cache[frame_idx]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys())
                del self.cache[oldest_key]

            self.cache[frame_idx] = frame_rgb
            return frame_rgb

        return None

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


def numpy_to_base64(img_array):
    """Convert numpy image array to base64 string for Dash."""
    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_str}"


def create_dash_app(markers_xyz, marker_ids, frame_range, metadata, labels_file,
                   video_reader, K, dist, rvec, tvec, img_size):
    """Create Dash app for interactive annotation with 2D+3D views."""
    start_frame, end_frame = frame_range
    num_frames = end_frame - start_frame

    # Initialize labels
    labels = load_labels(labels_file)

    # Create marker name to ID mapping
    marker_name_to_id = {}
    marker_names_list = list(markers_xyz.keys())
    for marker_name, marker_id in zip(marker_names_list, marker_ids):
        marker_name_to_id[marker_name] = marker_id

    # Calculate data bounds for 3D view
    all_markers = []
    for marker_name, xyz in markers_xyz.items():
        all_markers.append(xyz[start_frame:end_frame])

    all_data = np.concatenate(all_markers, axis=0)
    valid_data = all_data[~np.isnan(all_data).any(axis=1)]

    x_range = [valid_data[:, 0].min() - 100, valid_data[:, 0].max() + 100]
    y_range = [valid_data[:, 1].min() - 100, valid_data[:, 1].max() + 100]
    z_range = [valid_data[:, 2].min() - 100, valid_data[:, 2].max() + 100]

    # Create Dash app
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Mocap Marker Annotation Tool (2D + 3D)",
                style={'textAlign': 'center', 'marginBottom': '10px'}),

        # Top control panel
        html.Div([
            html.Div([
                html.H3("Instructions:", style={'marginTop': '0'}),
                html.Ul([
                    html.Li("Click on a marker in 2D or 3D view to select"),
                    html.Li("Enter a label name (e.g., 'Laxisl1')"),
                    html.Li("Click 'Set Label' to save"),
                    html.Li("Use slider to navigate frames"),
                    html.Li("Selected marker is highlighted in both views")
                ], style={'fontSize': '14px'})
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'padding': '10px', 'boxSizing': 'border-box'}),

            html.Div([
                html.H3("Selected Marker:", style={'marginTop': '0'}),
                html.Div(id='selected-marker-info', children='None selected'),
                html.Br(),
                html.Div([
                    html.Label("Label Name:", style={'fontWeight': 'bold'}),
                    dcc.Input(
                        id='label-input',
                        type='text',
                        placeholder='e.g., Laxisl1',
                        style={'width': '150px', 'marginLeft': '10px', 'marginRight': '10px'}
                    ),
                    html.Button('Set Label', id='set-label-btn', n_clicks=0,
                               style={'backgroundColor': '#4CAF50', 'color': 'white',
                                     'border': 'none', 'padding': '8px 16px', 'cursor': 'pointer'})
                ]),
                html.Div(id='label-status', style={'marginTop': '10px', 'color': 'green',
                                                   'fontWeight': 'bold'}),
            ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'padding': '10px', 'boxSizing': 'border-box'}),

            html.Div([
                html.H3("Current Labels:", style={'marginTop': '0'}),
                html.Div(id='labels-list',
                        style={'maxHeight': '150px', 'overflowY': 'scroll',
                              'border': '1px solid #ddd', 'padding': '5px',
                              'fontSize': '12px', 'backgroundColor': '#f9f9f9'})
            ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'padding': '10px', 'boxSizing': 'border-box'})
        ], style={'borderBottom': '2px solid #ddd', 'marginBottom': '10px'}),

        # Main content area: 3D on left, 2D on right
        html.Div([
            # 3D scatter plot
            html.Div([
                html.H3("3D View", style={'textAlign': 'center', 'marginTop': '0'}),
                dcc.Graph(id='3d-scatter', style={'height': '600px'})
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            # 2D image view
            html.Div([
                html.H3("2D View (Video + Projected Markers)",
                       style={'textAlign': 'center', 'marginTop': '0'}),
                dcc.Graph(id='2d-image', style={'height': '600px'})
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ]),

        # Frame slider
        html.Div([
            html.Label('Frame:', id='frame-label', style={'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Slider(
                id='frame-slider',
                min=0,
                max=num_frames - 1,
                value=0,
                marks={i: str(start_frame + i) for i in range(0, num_frames, max(1, num_frames // 20))},
                step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'padding': '20px'}),

        # Store for labels, selected marker, and marker positions
        dcc.Store(id='labels-store', data=labels),
        dcc.Store(id='selected-marker', data=None),
        dcc.Store(id='marker-2d-positions', data={})  # Store 2D positions for click detection
    ])

    @app.callback(
        Output('3d-scatter', 'figure'),
        Output('2d-image', 'figure'),
        Output('frame-label', 'children'),
        Output('marker-2d-positions', 'data'),
        Input('frame-slider', 'value'),
        Input('labels-store', 'data'),
        Input('selected-marker', 'data')
    )
    def update_graphs(frame_idx, labels_data, selected_marker):
        current_frame = start_frame + frame_idx

        # Get video frame
        video_frame = video_reader.get_frame(current_frame)
        if video_frame is None:
            video_frame = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

        # Collect marker data for current frame
        marker_points_3d = []
        markers_3d_array = []
        marker_names_ordered = []

        for marker_name, xyz in markers_xyz.items():
            frame_xyz = xyz[current_frame]
            marker_names_ordered.append(marker_name)
            markers_3d_array.append(frame_xyz)

            # For 3D plot
            if not np.isnan(frame_xyz).any():
                label = labels_data.get(marker_name, {}).get('label', marker_name)
                marker_points_3d.append({
                    'original_name': marker_name,
                    'display_name': label,
                    'x': frame_xyz[0],
                    'y': frame_xyz[1],
                    'z': frame_xyz[2],
                    'is_labeled': marker_name in labels_data,
                    'is_selected': marker_name == selected_marker
                })

        # Project markers to 2D
        markers_3d_array = np.array(markers_3d_array)
        points_2d, valid_mask = project_markers_to_2d(markers_3d_array, K, dist, rvec, tvec, img_size)

        # Create mapping of marker names to 2D positions for click detection
        marker_2d_positions = {}
        for i, marker_name in enumerate(marker_names_ordered):
            if valid_mask[i]:
                marker_2d_positions[marker_name] = {
                    'x': float(points_2d[i, 0]),
                    'y': float(points_2d[i, 1])
                }

        # ========== CREATE 3D FIGURE ==========
        labeled_3d = [p for p in marker_points_3d if p['is_labeled'] and not p['is_selected']]
        unlabeled_3d = [p for p in marker_points_3d if not p['is_labeled'] and not p['is_selected']]
        selected_3d = [p for p in marker_points_3d if p['is_selected']]

        fig_3d = go.Figure()

        # Unlabeled markers (gray)
        if unlabeled_3d:
            fig_3d.add_trace(go.Scatter3d(
                x=[p['x'] for p in unlabeled_3d],
                y=[p['y'] for p in unlabeled_3d],
                z=[p['z'] for p in unlabeled_3d],
                mode='markers',
                name='Unlabeled',
                marker=dict(size=4, color='lightgray', opacity=0.5),
                text=[p['original_name'] for p in unlabeled_3d],
                customdata=[p['original_name'] for p in unlabeled_3d],
                hovertemplate='<b>%{text}</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
            ))

        # Labeled markers (colored)
        if labeled_3d:
            fig_3d.add_trace(go.Scatter3d(
                x=[p['x'] for p in labeled_3d],
                y=[p['y'] for p in labeled_3d],
                z=[p['z'] for p in labeled_3d],
                mode='markers+text',
                name='Labeled',
                marker=dict(size=6, color=[p['y'] for p in labeled_3d],
                           colorscale='Viridis', showscale=True),
                text=[p['display_name'] for p in labeled_3d],
                textposition='top center',
                textfont=dict(size=7),
                customdata=[p['original_name'] for p in labeled_3d],
                hovertemplate='<b>%{text}</b><br>Original: %{customdata}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
            ))

        # Selected marker (highlighted)
        if selected_3d:
            fig_3d.add_trace(go.Scatter3d(
                x=[p['x'] for p in selected_3d],
                y=[p['y'] for p in selected_3d],
                z=[p['z'] for p in selected_3d],
                mode='markers',
                name='Selected',
                marker=dict(size=12, color='red', symbol='diamond',
                           line=dict(color='yellow', width=2)),
                text=[p['display_name'] for p in selected_3d],
                customdata=[p['original_name'] for p in selected_3d],
                hovertemplate='<b>SELECTED: %{text}</b><br>%{customdata}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
            ))

        camera = dict(eye=dict(x=1.5, y=2.0, z=1.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0))
        fig_3d.update_layout(
            title=f'Frame {current_frame} | Labeled: {len(labeled_3d)} / {len(marker_points_3d)}',
            scene=dict(
                xaxis=dict(title='X (mm)', range=x_range),
                yaxis=dict(title='Y (mm) - Up', range=y_range),
                zaxis=dict(title='Z (mm)', range=z_range),
                aspectmode='data',
                camera=camera
            ),
            showlegend=True,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # ========== CREATE 2D FIGURE ==========
        # Draw markers on image
        # Note: video_frame is already RGB from VideoFrameReader
        img_with_markers = video_frame.copy()

        for marker_name in marker_names_ordered:
            if marker_name in marker_2d_positions:
                x, y = marker_2d_positions[marker_name]['x'], marker_2d_positions[marker_name]['y']
                is_labeled = marker_name in labels_data
                is_selected = marker_name == selected_marker

                # Choose color (RGB since img_with_markers is RGB)
                if is_selected:
                    color = (255, 0, 0)  # Red (RGB) for selected
                    radius = 8
                    thickness = 2
                elif is_labeled:
                    color = (0, 255, 0)  # Green (RGB) for labeled
                    radius = 5
                    thickness = -1
                else:
                    color = (150, 150, 150)  # Gray (RGB) for unlabeled
                    radius = 3
                    thickness = -1

                cv2.circle(img_with_markers, (int(x), int(y)), radius, color, thickness)

                # Draw label text for labeled markers
                if is_labeled:
                    label = labels_data[marker_name]['label']
                    cv2.putText(img_with_markers, label, (int(x)+10, int(y)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw highlight ring for selected
                if is_selected:
                    cv2.circle(img_with_markers, (int(x), int(y)), 12, (255, 255, 0), 2)  # Yellow (RGB)

        # Create plotly figure with image
        # Important: OpenCV images are (H, W, 3) and origin is top-left
        # Plotly displays images with origin at bottom-left by default
        fig_2d = go.Figure()

        # Convert image for display
        # Plotly expects RGB, video_frame is already RGB from VideoFrameReader
        fig_2d.add_trace(go.Image(z=img_with_markers))

        # Add invisible scatter points for click detection
        # Note: Image origin is (0,0) at top-left in array indexing
        # But plotly Image trace uses pixel coordinates naturally
        scatter_x = []
        scatter_y = []
        scatter_text = []
        scatter_customdata = []

        for marker_name in marker_names_ordered:
            if marker_name in marker_2d_positions:
                pos = marker_2d_positions[marker_name]
                scatter_x.append(pos['x'])
                scatter_y.append(pos['y'])  # Y coordinate as-is from projection
                label = labels_data.get(marker_name, {}).get('label', marker_name)
                scatter_text.append(label)
                scatter_customdata.append(marker_name)

        if scatter_x:
            fig_2d.add_trace(go.Scatter(
                x=scatter_x,
                y=scatter_y,
                mode='markers',
                marker=dict(size=15, color='rgba(0,0,0,0)'),  # Invisible but larger for easier clicking
                text=scatter_text,
                customdata=scatter_customdata,
                hovertemplate='<b>%{text}</b><br>Original: %{customdata}<br>X: %{x:.0f} px<br>Y: %{y:.0f} px<extra></extra>',
                showlegend=False
            ))

        fig_2d.update_layout(
            title=f'Video Frame {current_frame} | Projected Markers: {len(marker_2d_positions)}',
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False, scaleanchor="x"),
            showlegend=False,
            hovermode='closest'
        )

        return fig_3d, fig_2d, f'Frame: {current_frame}', marker_2d_positions

    @app.callback(
        Output('selected-marker', 'data'),
        Output('selected-marker-info', 'children'),
        Output('label-input', 'value'),
        Input('3d-scatter', 'clickData'),
        Input('2d-image', 'clickData'),
        State('labels-store', 'data'),
        prevent_initial_call=True
    )
    def select_marker(click_3d, click_2d, labels_data):
        ctx = callback_context
        if not ctx.triggered:
            return None, 'None selected', ''

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        try:
            if trigger_id == '3d-scatter' and click_3d:
                point_data = click_3d['points'][0]
                original_name = point_data['customdata']
            elif trigger_id == '2d-image' and click_2d:
                point_data = click_2d['points'][0]
                original_name = point_data['customdata']
            else:
                return None, 'None selected', ''

            # Get current label
            current_label = labels_data.get(original_name, {}).get('label', '')

            info = html.Div([
                html.P(f"📍 Original Name: {original_name}", style={'fontWeight': 'bold'}),
                html.P(f"🏷️  Current Label: {current_label if current_label else 'Not labeled'}"),
                html.P(f"🎯 Source: {'3D View' if trigger_id == '3d-scatter' else '2D View'}")
            ])

            return original_name, info, current_label

        except Exception as e:
            return None, f'Error: {str(e)}', ''

    @app.callback(
        Output('labels-store', 'data'),
        Output('label-status', 'children'),
        Output('labels-list', 'children'),
        Input('set-label-btn', 'n_clicks'),
        State('selected-marker', 'data'),
        State('label-input', 'value'),
        State('labels-store', 'data'),
        prevent_initial_call=True
    )
    def set_label(n_clicks, selected_marker, label_value, labels_data):
        if selected_marker is None:
            return labels_data, '⚠️  Please select a marker first', generate_labels_list(labels_data)

        if not label_value or label_value.strip() == '':
            return labels_data, '⚠️  Please enter a label name', generate_labels_list(labels_data)

        # Update labels
        labels_data[selected_marker] = {
            'label': label_value.strip(),
            'marker_id': marker_name_to_id.get(selected_marker, '')
        }

        # Save to file
        save_labels(labels_data, labels_file)

        status = f'✅ Label "{label_value}" set for {selected_marker}'

        return labels_data, status, generate_labels_list(labels_data)

    def generate_labels_list(labels_data):
        """Generate HTML list of current labels."""
        if not labels_data:
            return html.P("No labels yet", style={'fontStyle': 'italic', 'color': '#999'})

        items = []
        for original_name, info in sorted(labels_data.items(), key=lambda x: x[1]['label']):
            items.append(
                html.Div(f"🏷️  {info['label']} ← {original_name}",
                        style={'marginBottom': '3px'})
            )

        return html.Div(items)

    return app


def main():
    parser = argparse.ArgumentParser(
        description='Interactive mocap marker annotation tool with 2D+3D views',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python annotate_mocap_markers_2d3d.py \\
      --csv /Volumes/FastACIS/GoPro/motion/mocap/mocap.csv \\
      --video /Volumes/FastACIS/GoPro/motion/mocap/primecolor.avi \\
      --mcal /Volumes/FastACIS/annotation_pipeline/optitrack.mcal \\
      --start_frame 100 --num_frames 500
        """
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to mocap CSV file')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file (.avi)')
    parser.add_argument('--mcal', type=str, required=True,
                        help='Path to OptiTrack calibration file (.mcal)')
    parser.add_argument('--camera_serial', type=str, default='C11764',
                        help='Camera serial number in .mcal file (default: C11764)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Start frame for visualization')
    parser.add_argument('--num_frames', type=int, default=500,
                        help='Number of frames to load')
    parser.add_argument('--labels', type=str, default='marker_labels.csv',
                        help='Path to labels CSV file (default: marker_labels.csv)')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port for web server (default: 8050)')
    parser.add_argument('--intrinsics', type=str, default=None,
                        help='Path to custom intrinsics JSON file (multical format, optional)')
    parser.add_argument('--extrinsics', type=str, default=None,
                        help='Path to custom extrinsics JSON file (optional)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Mocap Marker Annotation Tool (2D + 3D)")
    print("="*70)

    # Load calibration
    print("\n[1/4] Loading calibration...")
    try:
        K, dist, rvec, tvec, img_size = load_optitrack_calibration(
            args.mcal, args.camera_serial,
            intrinsics_json=args.intrinsics,
            extrinsics_json=args.extrinsics
        )
        print(f"  ✓ Camera intrinsics: fx={K[0,0]:.2f} (negative), fy={K[1,1]:.2f}")
        print(f"  ✓ Image size: {img_size[0]}x{img_size[1]}")
        if args.intrinsics:
            print(f"  ✓ Using custom intrinsics from: {args.intrinsics}")
        if args.extrinsics:
            print(f"  ✓ Using custom extrinsics from: {args.extrinsics}")
    except Exception as e:
        print(f"  ✗ Error loading calibration: {e}")
        return

    # Load mocap data
    print("\n[2/4] Loading mocap data...")
    try:
        metadata, marker_names, marker_ids, data_df = parse_motive_csv(args.csv)
        print(f"  ✓ Found {len(marker_names)} markers")
        print(f"  ✓ Total frames: {len(data_df)}")
        markers_xyz = extract_marker_data(data_df, marker_names)
    except Exception as e:
        print(f"  ✗ Error loading mocap data: {e}")
        return

    # Open video
    print("\n[3/4] Opening video...")
    try:
        video_reader = VideoFrameReader(args.video)
        print(f"  ✓ Video frames: {video_reader.total_frames}")
        print(f"  ✓ FPS: {video_reader.fps:.1f}")
        print(f"  ✓ Resolution: {video_reader.width}x{video_reader.height}")

        # Verify resolution matches calibration
        if video_reader.width != img_size[0] or video_reader.height != img_size[1]:
            print(f"  ⚠️  Warning: Video resolution doesn't match calibration!")
            print(f"     Video: {video_reader.width}x{video_reader.height}")
            print(f"     Calibration: {img_size[0]}x{img_size[1]}")
    except Exception as e:
        print(f"  ✗ Error opening video: {e}")
        return

    # Prepare frame range
    end_frame = min(args.start_frame + args.num_frames, len(data_df))
    print(f"\n[4/4] Setting up visualization...")
    print(f"  ✓ Frame range: {args.start_frame} to {end_frame}")
    print(f"  ✓ Labels file: {args.labels}")

    # Create app
    try:
        app = create_dash_app(
            markers_xyz,
            marker_ids,
            (args.start_frame, end_frame),
            metadata,
            args.labels,
            video_reader,
            K, dist, rvec, tvec,
            img_size
        )
    except Exception as e:
        print(f"  ✗ Error creating app: {e}")
        return

    print("\n" + "="*70)
    print("🚀 Starting web server...")
    print("="*70)
    print(f"\n  👉 Open your browser: http://localhost:{args.port}")
    print(f"\n  Features:")
    print(f"    • Click markers in 3D or 2D view to select")
    print(f"    • Type label name and click 'Set Label'")
    print(f"    • Selected marker is highlighted in both views")
    print(f"    • Navigate frames with slider")
    print(f"\n  Press Ctrl+C to stop")
    print("="*70 + "\n")

    app.run(debug=False, port=args.port)


if __name__ == '__main__':
    main()
