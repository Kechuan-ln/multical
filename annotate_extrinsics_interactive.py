#!/usr/bin/env python3
"""
Interactive extrinsics calibration tool.

Allows manual annotation of 2D-3D correspondences to refine camera extrinsics.

Workflow:
1. View 3D markers (left) and 2D projection (right)
2. Click on a 3D marker to select it (turns RED)
3. Click on the corresponding 2D location in the video frame
4. Repeat for multiple points (recommended: 6+ points)
5. Click "Recompute Extrinsics" to optimize camera pose
6. All projections update with new extrinsics
7. Continue adding more points or switch frames as needed
"""

import cv2
import numpy as np
import pandas as pd
import json
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


class VideoFrameReader:
    """Efficient video frame reader with caching."""

    def __init__(self, video_path, cache_size=50):
        self.video_path = str(video_path)
        self.cache = {}
        self.cache_size = cache_size
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video initialized: {self.total_frames} frames available")

    def get_frame(self, frame_idx):
        """Get frame by index, using cache."""
        if frame_idx in self.cache:
            return self.cache[frame_idx]

        # Clamp frame index to valid range
        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= self.total_frames:
            frame_idx = self.total_frames - 1
            print(f"Warning: Frame index clamped to {frame_idx} (max)")

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Cache management
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))

                self.cache[frame_idx] = frame_rgb
                return frame_rgb
            else:
                print(f"Error: Failed to read frame {frame_idx}")
                return None
        except Exception as e:
            print(f"Error reading frame {frame_idx}: {e}")
            return None

    def __del__(self):
        if self.cap is not None:
            self.cap.release()


class ExtrinsicsAnnotationTool:
    """Interactive tool for extrinsics calibration via 2D-3D correspondences."""

    def __init__(self, csv_path, video_path, mcal_path, intrinsics_path,
                 camera_serial='C11764', start_frame=0, port=8050, use_mcal_intrinsics=False):
        self.csv_path = csv_path
        self.video_path = video_path
        self.mcal_path = mcal_path
        self.intrinsics_path = intrinsics_path
        self.camera_serial = camera_serial
        self.current_frame = start_frame
        self.port = port
        self.use_mcal_intrinsics = use_mcal_intrinsics

        # Load data
        print("Loading data...")
        self.load_intrinsics()
        self.load_extrinsics()
        self.load_mocap_data()
        self.video_reader = VideoFrameReader(video_path)

        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Annotation state
        self.correspondences = []  # List of (marker_idx, frame_idx, point_2d) tuples
        self.selected_marker_idx = None

        # Initial extrinsics (will be updated)
        self.current_rvec = self.initial_rvec.copy()
        self.current_tvec = self.initial_tvec.copy()

        print(f"✓ Loaded {len(self.marker_names)} markers")
        print(f"✓ Video: {self.total_frames} frames, {self.img_width}x{self.img_height}")
        print(f"✓ Ready to start annotation")

    def load_intrinsics(self):
        """Load intrinsics from multical JSON or .mcal file."""
        if self.use_mcal_intrinsics:
            # Load from .mcal file
            tree = ET.parse(self.mcal_path)
            root = tree.getroot()

            for cam in root.findall('.//Camera'):
                if cam.get('Serial') == self.camera_serial:
                    intrinsic = cam.find('.//IntrinsicStandardCameraModel')
                    if intrinsic is None:
                        intrinsic = cam.find('.//Intrinsic')

                    fx = float(intrinsic.get('HorizontalFocalLength'))
                    fy = float(intrinsic.get('VerticalFocalLength'))
                    cx = float(intrinsic.get('LensCenterX'))
                    cy = float(intrinsic.get('LensCenterY'))
                    k1 = float(intrinsic.get('k1', 0.0))
                    k2 = float(intrinsic.get('k2', 0.0))
                    k3 = float(intrinsic.get('k3', 0.0))
                    p1 = float(intrinsic.get('TangentialX', 0.0))
                    p2 = float(intrinsic.get('TangentialY', 0.0))

                    self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                    self.dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

                    print(f"✓ Intrinsics from .mcal: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
                    return

            raise ValueError(f"Camera {self.camera_serial} not found in {self.mcal_path}")
        else:
            # Load from multical JSON
            with open(self.intrinsics_path, 'r') as f:
                data = json.load(f)

            camera_data = data['cameras']['primecolor']
            self.K = np.array(camera_data['K'])
            self.dist = np.array(camera_data['dist']).flatten()

            print(f"✓ Intrinsics from multical JSON: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}, cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}")

    def load_extrinsics(self):
        """Load initial extrinsics from .mcal."""
        tree = ET.parse(self.mcal_path)
        root = tree.getroot()

        for camera in root.findall('.//Camera'):
            serial = camera.get('Serial')
            props = camera.find('.//Properties')

            if serial == self.camera_serial or (props is not None and props.get('CameraID') == '13'):
                extrinsic_elem = camera.find('.//Extrinsic')

                T_world = np.array([
                    float(extrinsic_elem.get('X')),
                    float(extrinsic_elem.get('Y')),
                    float(extrinsic_elem.get('Z'))
                ])

                R_c2w = np.array([
                    [float(extrinsic_elem.get('OrientMatrix0')), float(extrinsic_elem.get('OrientMatrix1')), float(extrinsic_elem.get('OrientMatrix2'))],
                    [float(extrinsic_elem.get('OrientMatrix3')), float(extrinsic_elem.get('OrientMatrix4')), float(extrinsic_elem.get('OrientMatrix5'))],
                    [float(extrinsic_elem.get('OrientMatrix6')), float(extrinsic_elem.get('OrientMatrix7')), float(extrinsic_elem.get('OrientMatrix8'))]
                ])

                R_w2c = R_c2w.T
                tvec = -R_w2c @ T_world
                rvec, _ = cv2.Rodrigues(R_w2c)

                self.initial_rvec = rvec
                self.initial_tvec = tvec

                print(f"Initial extrinsics: T={T_world}")
                return

        raise ValueError(f"Camera {self.camera_serial} not found in {self.mcal_path}")

    def load_mocap_data(self):
        """Load mocap marker data."""
        with open(self.csv_path, 'r') as f:
            lines = [next(f) for _ in range(3)]

        marker_names_row = lines[2].strip().split(',')
        self.marker_names = []
        for i in range(2, len(marker_names_row), 3):
            name = marker_names_row[i].strip()
            if name and name != 'Name':
                self.marker_names.append(name)

        self.mocap_df = pd.read_csv(self.csv_path, skiprows=7, header=None, low_memory=False)

    def get_markers_3d(self, frame_idx):
        """Get 3D markers for a specific frame."""
        if frame_idx >= len(self.mocap_df):
            return None

        row = self.mocap_df.iloc[frame_idx]
        markers = []

        for i in range(len(self.marker_names)):
            col_idx = 2 + i * 3

            if col_idx + 2 < len(row):
                x = row.iloc[col_idx]
                y = row.iloc[col_idx + 1]
                z = row.iloc[col_idx + 2]

                try:
                    x = float(x) if x != '' else np.nan
                    y = float(y) if y != '' else np.nan
                    z = float(z) if z != '' else np.nan
                except (ValueError, TypeError):
                    x, y, z = np.nan, np.nan, np.nan

                markers.append([x, y, z])
            else:
                markers.append([np.nan, np.nan, np.nan])

        return np.array(markers)

    def project_markers(self, markers_3d, rvec, tvec):
        """Project 3D markers to 2D using current extrinsics."""
        valid_mask = ~np.isnan(markers_3d).any(axis=1) & (np.abs(markers_3d).sum(axis=1) > 0.1)

        if not valid_mask.any():
            return np.array([]), np.array([])

        valid_markers = markers_3d[valid_mask]
        markers_m = valid_markers / 1000.0

        K_neg = self.K.copy()
        K_neg[0, 0] = -K_neg[0, 0]

        points_2d, _ = cv2.projectPoints(
            markers_m.reshape(-1, 1, 3),
            rvec, tvec, K_neg, self.dist
        )

        points_2d = points_2d.reshape(-1, 2)

        in_bounds = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < self.img_width) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < self.img_height)
        )

        valid_indices = np.where(valid_mask)[0][in_bounds]
        points_2d_valid = points_2d[in_bounds]

        return points_2d_valid, valid_indices

    def recompute_extrinsics(self):
        """Recompute extrinsics using solvePnP with annotated correspondences."""
        if len(self.correspondences) < 6:
            return False, "Need at least 6 correspondences (currently have {})".format(len(self.correspondences))

        # Extract 3D and 2D points from correspondences (from their respective frames)
        points_3d = []
        points_2d = []

        print(f"\n=== DEBUG: Recomputing extrinsics ===")
        print(f"Total correspondences: {len(self.correspondences)}")

        for i, (marker_idx, frame_idx, pt_2d) in enumerate(self.correspondences):
            # Get markers from the frame where this correspondence was annotated
            markers_3d = self.get_markers_3d(frame_idx)

            if marker_idx < len(markers_3d):
                pt_3d = markers_3d[marker_idx]
                if not np.isnan(pt_3d).any():
                    pt_3d_m = pt_3d / 1000.0  # mm to meters
                    points_3d.append(pt_3d_m)
                    points_2d.append(pt_2d)
                    print(f"  [{i}] Marker {marker_idx} @ frame {frame_idx}: 3D={pt_3d_m} (m), 2D={pt_2d}")
                else:
                    print(f"  [{i}] Marker {marker_idx} @ frame {frame_idx}: INVALID (NaN)")
            else:
                print(f"  [{i}] Marker {marker_idx} @ frame {frame_idx}: OUT OF RANGE (max={len(markers_3d)})")

        if len(points_3d) < 6:
            return False, f"Only {len(points_3d)} valid correspondences (need 6+)"

        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)

        print(f"\n3D points shape: {points_3d.shape}")
        print(f"3D range: X=[{points_3d[:,0].min():.3f}, {points_3d[:,0].max():.3f}], "
              f"Y=[{points_3d[:,1].min():.3f}, {points_3d[:,1].max():.3f}], "
              f"Z=[{points_3d[:,2].min():.3f}, {points_3d[:,2].max():.3f}] (meters)")
        print(f"2D points shape: {points_2d.shape}")
        print(f"2D range: X=[{points_2d[:,0].min():.1f}, {points_2d[:,0].max():.1f}], "
              f"Y=[{points_2d[:,1].min():.1f}, {points_2d[:,1].max():.1f}] (pixels)")

        # Use negative fx for OptiTrack coordinate system
        K_neg = self.K.copy()
        K_neg[0, 0] = -K_neg[0, 0]

        print(f"\nIntrinsics K_neg:")
        print(f"  fx={K_neg[0,0]:.1f}, fy={K_neg[1,1]:.1f}, cx={K_neg[0,2]:.1f}, cy={K_neg[1,2]:.1f}")
        print(f"Distortion: {self.dist}")

        # Solve PnP with RANSAC
        print(f"\nCalling solvePnPRansac...")
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K_neg, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=10.0,
            confidence=0.99
        )

        print(f"solvePnPRansac result: success={success}")

        if not success:
            # Try without RANSAC as fallback
            print("\nRANSAC failed (likely too few points). Trying solvePnP without RANSAC...")
            try:
                success2, rvec2, tvec2 = cv2.solvePnP(
                    points_3d, points_2d, K_neg, self.dist,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                print(f"solvePnP (no RANSAC) result: success={success2}")
                if success2:
                    rvec = rvec2
                    tvec = tvec2
                    inliers = None  # No inlier detection without RANSAC
                    success = True
                    print(f"  rvec={rvec.flatten()}")
                    print(f"  tvec={tvec.flatten()}")
                    print(f"  ⚠️ Using all {len(points_3d)} points (no outlier rejection)")
                else:
                    return False, "Both solvePnPRansac and solvePnP failed"
            except Exception as e:
                print(f"solvePnP (no RANSAC) error: {e}")
                return False, f"solvePnP error: {e}"
        else:
            print(f"  rvec={rvec.flatten()}")
            print(f"  tvec={tvec.flatten()}")
            print(f"  inliers={inliers.flatten() if inliers is not None else 'None'}")

        # Update current extrinsics
        self.current_rvec = rvec
        self.current_tvec = tvec

        inlier_count = len(inliers) if inliers is not None else len(points_3d)

        print(f"\n✓ Extrinsics updated! {inlier_count}/{len(points_3d)} points used")
        print("=" * 50)

        return True, f"✓ Extrinsics updated! {inlier_count}/{len(points_3d)} points used"

    def save_extrinsics(self, output_path):
        """Save current extrinsics to JSON."""
        R, _ = cv2.Rodrigues(self.current_rvec)
        R_c2w = R.T
        T_world = -R_c2w @ self.current_tvec.flatten()

        data = {
            'camera_serial': self.camera_serial,
            'rvec': self.current_rvec.flatten().tolist(),
            'tvec': self.current_tvec.flatten().tolist(),
            'rotation_matrix': R.tolist(),
            'camera_position_world': T_world.tolist(),
            'intrinsics_source': 'mcal' if self.use_mcal_intrinsics else 'multical_json',
            'intrinsics': {
                'K': self.K.tolist(),
                'dist': self.dist.tolist(),
                'fx': float(self.K[0, 0]),
                'fy': float(self.K[1, 1]),
                'cx': float(self.K[0, 2]),
                'cy': float(self.K[1, 2])
            },
            'correspondences': [
                {
                    'marker_idx': int(m_idx),
                    'frame_idx': int(f_idx),
                    'point_2d': [float(pt[0]), float(pt[1])]
                }
                for m_idx, f_idx, pt in self.correspondences
            ],
            'num_correspondences': len(self.correspondences)
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved extrinsics to {output_path}")
        print(f"  Intrinsics source: {data['intrinsics_source']}")

    def create_app(self):
        """Create Dash application."""
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H1("Interactive Extrinsics Calibration Tool"),

            html.Div([
                html.Div([
                    html.H3("Instructions:"),
                    html.Ol([
                        html.Li("Enter marker number below OR click a marker in 3D view"),
                        html.Li("Click 'Select Marker' or click the marker in 3D"),
                        html.Li("Click the corresponding location in 2D view (right)"),
                        html.Li("Repeat for 6+ points"),
                        html.Li("Click 'Recompute Extrinsics' to optimize"),
                        html.Li("Continue refining or save results")
                    ]),
                ], style={'width': '100%', 'padding': '10px', 'backgroundColor': '#f0f0f0'}),
            ]),

            html.Div([
                html.Label("Select Marker by Number:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Input(
                    id='marker-input',
                    type='number',
                    placeholder='Enter marker number (e.g., 116)',
                    min=0,
                    max=199,
                    step=1,
                    style={'width': '200px', 'marginRight': '10px', 'padding': '5px'}
                ),
                html.Button('Select Marker', id='select-marker-btn', n_clicks=0,
                           style={'fontSize': '14px', 'padding': '5px 15px', 'backgroundColor': '#FF9800', 'color': 'white'}),
            ], style={'padding': '10px 20px', 'textAlign': 'left', 'backgroundColor': '#fff3cd'}),

            html.Div([
                html.Label(f"Frame: {self.current_frame} / {self.total_frames - 1}"),
                dcc.Slider(
                    id='frame-slider',
                    min=0,
                    max=self.total_frames - 1,
                    value=self.current_frame,
                    step=1,
                    marks={i: str(i) for i in range(0, self.total_frames, max(1, self.total_frames // 10))},
                ),
            ], style={'width': '100%', 'padding': '20px'}),

            html.Div([
                html.Div([
                    html.H3("3D Markers (Click to select)"),
                    dcc.Graph(id='3d-scatter', style={'height': '600px'}),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.H3("2D Video Frame (Click to annotate)"),
                    dcc.Graph(id='2d-image', style={'height': '600px'}),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
            ]),

            html.Div([
                html.Div([
                    html.H4(id='status-text', children='Status: Ready', style={'color': 'blue'}),
                    html.H4(id='correspondence-count', children=f'Correspondences: 0 (need 6+)'),
                    html.Button('Recompute Extrinsics', id='recompute-btn', n_clicks=0,
                               style={'fontSize': '16px', 'padding': '10px 20px', 'margin': '10px', 'backgroundColor': '#4CAF50', 'color': 'white'}),
                    html.Button('Undo Last', id='undo-btn', n_clicks=0,
                               style={'fontSize': '16px', 'padding': '10px 20px', 'margin': '10px'}),
                    html.Button('Clear All', id='clear-btn', n_clicks=0,
                               style={'fontSize': '16px', 'padding': '10px 20px', 'margin': '10px', 'backgroundColor': '#f44336', 'color': 'white'}),
                    html.Button('Save Extrinsics', id='save-btn', n_clicks=0,
                               style={'fontSize': '16px', 'padding': '10px 20px', 'margin': '10px', 'backgroundColor': '#2196F3', 'color': 'white'}),
                ], style={'padding': '20px', 'textAlign': 'center'}),
            ]),

            # Hidden stores for state
            dcc.Store(id='selected-marker-store', data=None),
            dcc.Store(id='correspondences-store', data=[]),
            dcc.Store(id='extrinsics-store', data={
                'rvec': self.current_rvec.flatten().tolist(),
                'tvec': self.current_tvec.flatten().tolist()
            }),
            dcc.Store(id='hover-marker-store', data=None),
        ])

        self.setup_callbacks(app)

        return app

    def setup_callbacks(self, app):
        """Setup Dash callbacks."""

        @app.callback(
            [Output('3d-scatter', 'figure'),
             Output('2d-image', 'figure'),
             Output('correspondence-count', 'children'),
             Output('status-text', 'children'),
             Output('status-text', 'style')],
            [Input('frame-slider', 'value'),
             Input('select-marker-btn', 'n_clicks'),
             Input('3d-scatter', 'clickData'),
             Input('2d-image', 'clickData'),
             Input('recompute-btn', 'n_clicks'),
             Input('undo-btn', 'n_clicks'),
             Input('clear-btn', 'n_clicks'),
             Input('save-btn', 'n_clicks'),
             Input('hover-marker-store', 'data')],
            [State('marker-input', 'value'),
             State('selected-marker-store', 'data'),
             State('correspondences-store', 'data'),
             State('extrinsics-store', 'data')]
        )
        def update_all(frame_idx, select_marker_clicks, click_3d, click_2d, recompute_clicks, undo_clicks,
                      clear_clicks, save_clicks, hover_marker_idx, marker_input_value, selected_marker, correspondences_data, extrinsics_data):

            # Get triggered ID first (needed for state restoration logic)
            triggered_id = ctx.triggered_id if ctx.triggered_id else None

            # Update current frame
            self.current_frame = frame_idx

            # Restore state from stores
            if correspondences_data:
                # Handle both old format (no frame_idx) and new format (with frame_idx)
                self.correspondences = [
                    (c['marker_idx'], c.get('frame_idx', self.current_frame), tuple(c['point_2d']))
                    for c in correspondences_data
                ]
            else:
                self.correspondences = []

            # Only restore extrinsics if NOT recomputing or saving
            # (otherwise we'd overwrite the newly computed values)
            if extrinsics_data and triggered_id not in ['recompute-btn', 'save-btn']:
                self.current_rvec = np.array(extrinsics_data['rvec']).reshape(3, 1)
                self.current_tvec = np.array(extrinsics_data['tvec']).reshape(3, 1)

            # Restore selected marker from store
            if selected_marker is not None:
                self.selected_marker_idx = selected_marker
            # Important: Don't reset to None if store is None, keep current value

            status_msg = 'Ready'
            status_style = {'color': 'blue'}

            # Debug: print current state
            print(f"Triggered: {triggered_id}, Selected marker: {self.selected_marker_idx} (from store: {selected_marker}), Correspondences: {len(self.correspondences)}")

            # Handle interactions based on trigger
            try:
                # Handle marker selection by input
                if triggered_id == 'select-marker-btn':
                    if marker_input_value is not None:
                        self.selected_marker_idx = int(marker_input_value)
                        status_msg = f'Selected marker #{self.selected_marker_idx} - now click on 2D image'
                        status_style = {'color': 'orange'}
                        print(f"Selected marker via input: {self.selected_marker_idx}")
                    else:
                        status_msg = 'Please enter a marker number first'
                        status_style = {'color': 'red'}

                # Handle button clicks
                elif triggered_id == 'recompute-btn':
                    success, msg = self.recompute_extrinsics()
                    status_msg = msg
                    status_style = {'color': 'green' if success else 'red'}

                elif triggered_id == 'undo-btn':
                    if self.correspondences:
                        self.correspondences.pop()
                        status_msg = 'Removed last correspondence'
                    else:
                        status_msg = 'No correspondences to undo'

                elif triggered_id == 'clear-btn':
                    self.correspondences = []
                    self.selected_marker_idx = None
                    status_msg = 'Cleared all correspondences'

                elif triggered_id == 'save-btn':
                    output_path = Path('./extrinsics_calibrated.json')
                    self.save_extrinsics(output_path)
                    status_msg = f'✓ Saved to {output_path}'
                    status_style = {'color': 'green'}

                # Handle 3D marker selection
                elif triggered_id == '3d-scatter':
                    if click_3d and 'points' in click_3d and len(click_3d['points']) > 0:
                        point_data = click_3d['points'][0]
                        print(f"3D click data: {point_data}")

                        # Try different possible keys for point index
                        point_idx = None
                        if 'pointIndex' in point_data:
                            point_idx = point_data['pointIndex']
                        elif 'pointNumber' in point_data:
                            point_idx = point_data['pointNumber']
                        elif 'text' in point_data:
                            # Fallback: use the text label to get marker index
                            point_idx = int(point_data.get('text', '0'))
                        elif 'customdata' in point_data:
                            point_idx = int(point_data['customdata'])

                        if point_idx is not None:
                            self.selected_marker_idx = point_idx
                            status_msg = f'Selected marker #{point_idx} - now click on 2D image'
                            status_style = {'color': 'orange'}
                            print(f"Set selected_marker_idx to: {self.selected_marker_idx}")
                        else:
                            status_msg = 'Could not determine marker index from click'
                            status_style = {'color': 'red'}
                            print(f"Failed to get marker index from: {point_data}")

                # Handle 2D annotation
                elif triggered_id == '2d-image':
                    if click_2d:
                        # Debug: print click data structure
                        print(f"2D click data: {click_2d}")

                        # Try to extract coordinates from various possible formats
                        x_2d, y_2d = None, None

                        if 'points' in click_2d and len(click_2d['points']) > 0:
                            point_data = click_2d['points'][0]
                            x_2d = point_data.get('x')
                            y_2d = point_data.get('y')

                        if x_2d is not None and y_2d is not None:
                            if self.selected_marker_idx is not None:
                                # Check if this marker+frame combination already exists
                                current_frame = frame_idx  # Use current frame from slider
                                existing_idx = None
                                for i, (m_idx, f_idx, _) in enumerate(self.correspondences):
                                    if m_idx == self.selected_marker_idx and f_idx == current_frame:
                                        existing_idx = i
                                        break

                                if existing_idx is not None:
                                    # Replace existing correspondence for same marker+frame
                                    old_pt = self.correspondences[existing_idx][2]
                                    self.correspondences[existing_idx] = (self.selected_marker_idx, current_frame, (x_2d, y_2d))
                                    status_msg = f'⚠️ Updated marker #{self.selected_marker_idx} (frame {current_frame}): ({old_pt[0]:.1f}, {old_pt[1]:.1f}) → ({x_2d:.1f}, {y_2d:.1f})'
                                    status_style = {'color': 'orange'}
                                    print(f"WARNING: Marker {self.selected_marker_idx} at frame {current_frame} already annotated. Replacing.")
                                else:
                                    # Add new correspondence with frame info
                                    self.correspondences.append((self.selected_marker_idx, current_frame, (x_2d, y_2d)))
                                    status_msg = f'✓ Added marker #{self.selected_marker_idx} (frame {current_frame}) at ({x_2d:.1f}, {y_2d:.1f})'
                                    status_style = {'color': 'green'}
                                    print(f"Added correspondence: marker {self.selected_marker_idx}, frame {current_frame}, 2D=({x_2d:.1f}, {y_2d:.1f})")

                                self.selected_marker_idx = None
                            else:
                                status_msg = 'Please select a 3D marker first'
                                status_style = {'color': 'red'}
                        else:
                            status_msg = 'Could not get coordinates from click'
                            status_style = {'color': 'red'}
                            print(f"Failed to extract coordinates from: {click_2d}")

                # Default case - show current selected marker
                elif triggered_id is None or triggered_id == 'hover-marker-store':
                    if self.selected_marker_idx is not None:
                        status_msg = f'Marker #{self.selected_marker_idx} selected - click on 2D image to annotate'
                        status_style = {'color': 'orange'}

            except Exception as e:
                status_msg = f'Error: {str(e)}'
                status_style = {'color': 'red'}
                print(f"Callback error: {e}")
                import traceback
                traceback.print_exc()

            # Generate figures with hover highlighting
            try:
                fig_3d = self.create_3d_figure(hover_marker_idx=hover_marker_idx)
                fig_2d = self.create_2d_figure(hover_marker_idx=hover_marker_idx)
            except Exception as e:
                print(f"Figure generation error: {e}")
                import traceback
                traceback.print_exc()
                fig_3d = go.Figure()
                fig_2d = go.Figure()

            # Update correspondence count
            corr_text = f'Correspondences: {len(self.correspondences)}'
            if len(self.correspondences) < 6:
                corr_text += f' (need {6 - len(self.correspondences)} more)'
            else:
                corr_text += ' ✓'

            return fig_3d, fig_2d, corr_text, f'Status: {status_msg}', status_style

        @app.callback(
            Output('hover-marker-store', 'data'),
            [Input('3d-scatter', 'hoverData'),
             Input('2d-image', 'hoverData')]
        )
        def update_hover(hover_3d, hover_2d):
            """Update hover marker when hovering over either view."""
            try:
                # Check which graph triggered the hover
                triggered_id = ctx.triggered_id if ctx.triggered_id else None

                if triggered_id == '3d-scatter' and hover_3d:
                    if 'points' in hover_3d and len(hover_3d['points']) > 0:
                        # Get marker index from text or customdata
                        point_data = hover_3d['points'][0]
                        if 'text' in point_data:
                            return int(point_data['text'])
                        elif 'customdata' in point_data:
                            return int(point_data['customdata'])

                elif triggered_id == '2d-image' and hover_2d:
                    if 'points' in hover_2d and len(hover_2d['points']) > 0:
                        # Get marker index from text or customdata
                        point_data = hover_2d['points'][0]
                        if 'text' in point_data:
                            text = point_data['text']
                            # Handle "marker@frame" format (e.g., "170@f8423")
                            if '@' in str(text):
                                marker_idx = str(text).split('@')[0]
                                return int(marker_idx)
                            else:
                                return int(text)
                        elif 'customdata' in point_data:
                            return int(point_data['customdata'])

                return None

            except Exception as e:
                print(f"Hover update error: {e}")
                return None

        @app.callback(
            [Output('selected-marker-store', 'data'),
             Output('correspondences-store', 'data'),
             Output('extrinsics-store', 'data')],
            [Input('select-marker-btn', 'n_clicks'),
             Input('3d-scatter', 'clickData'),
             Input('2d-image', 'clickData'),
             Input('recompute-btn', 'n_clicks'),
             Input('undo-btn', 'n_clicks'),
             Input('clear-btn', 'n_clicks'),
             Input('save-btn', 'n_clicks'),
             Input('frame-slider', 'value')],
            [State('selected-marker-store', 'data'),
             State('correspondences-store', 'data'),
             State('extrinsics-store', 'data')],
            prevent_initial_call=True
        )
        def update_stores(select_btn, click_3d, click_2d, recompute, undo, clear, save_btn, frame_val,
                         selected_marker, correspondences_data, extrinsics_data):

            # Update stores based on current state
            try:
                selected_store = self.selected_marker_idx

                correspondences_store = [
                    {'marker_idx': int(idx), 'frame_idx': int(f_idx), 'point_2d': [float(pt[0]), float(pt[1])]}
                    for idx, f_idx, pt in self.correspondences
                ]

                extrinsics_store = {
                    'rvec': self.current_rvec.flatten().tolist(),
                    'tvec': self.current_tvec.flatten().tolist()
                }

                return selected_store, correspondences_store, extrinsics_store

            except Exception as e:
                print(f"Store update error: {e}")
                import traceback
                traceback.print_exc()
                # Return previous state on error
                return selected_marker, correspondences_data, extrinsics_data

    def create_3d_figure(self, hover_marker_idx=None):
        """Create 3D scatter plot of markers."""
        markers_3d = self.get_markers_3d(self.current_frame)

        if markers_3d is None:
            return go.Figure()

        valid_mask = ~np.isnan(markers_3d).any(axis=1) & (np.abs(markers_3d).sum(axis=1) > 0.1)
        valid_indices = np.where(valid_mask)[0]
        valid_markers = markers_3d[valid_mask]

        # Annotated markers (from any frame)
        annotated_indices = set(idx for idx, _, _ in self.correspondences)

        # Colors with hover highlighting
        colors = []
        sizes = []
        for idx in valid_indices:
            if idx == self.selected_marker_idx:
                colors.append('red')  # Selected (clicked)
                sizes.append(8)
            elif idx == hover_marker_idx:
                colors.append('orange')  # Hovered
                sizes.append(10)
            elif idx in annotated_indices:
                colors.append('green')  # Annotated
                sizes.append(6)
            else:
                colors.append('lightblue')  # Default
                sizes.append(5)

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=valid_markers[:, 0],
            y=valid_markers[:, 1],
            z=valid_markers[:, 2],
            mode='markers+text',
            marker=dict(size=sizes, color=colors),
            text=[f"{idx}" for idx in valid_indices],
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate='Marker %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>',
            customdata=valid_indices  # Store marker indices for hover detection
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='closest'
        )

        return fig

    def create_2d_figure(self, hover_marker_idx=None):
        """Create 2D image with projected markers."""
        frame = self.video_reader.get_frame(self.current_frame)

        if frame is None:
            return go.Figure()

        # Project all markers
        markers_3d = self.get_markers_3d(self.current_frame)
        points_2d, valid_indices = self.project_markers(markers_3d, self.current_rvec, self.current_tvec)

        # Annotated markers (from any frame)
        annotated_indices = set(idx for idx, _, _ in self.correspondences)

        fig = go.Figure()

        # Display image
        fig.add_trace(go.Image(z=frame))

        # Add transparent clickable layer covering the entire image
        # This ensures clicks anywhere on the image are captured
        fig.add_trace(go.Scatter(
            x=[0, self.img_width, self.img_width, 0, 0],
            y=[0, 0, self.img_height, self.img_height, 0],
            mode='lines',
            line=dict(width=0),
            fill='toself',
            fillcolor='rgba(0,0,0,0)',  # Transparent
            hoverinfo='skip',
            showlegend=False,
            name='clickable_layer'
        ))

        # Plot projected markers with hover highlighting
        if len(points_2d) > 0:
            colors_proj = []
            sizes_proj = []
            for idx in valid_indices:
                if idx == hover_marker_idx:
                    colors_proj.append('orange')  # Hovered
                    sizes_proj.append(12)
                elif idx in annotated_indices:
                    colors_proj.append('green')  # Annotated
                    sizes_proj.append(8)
                else:
                    colors_proj.append('yellow')  # Default
                    sizes_proj.append(8)

            fig.add_trace(go.Scatter(
                x=points_2d[:, 0],
                y=points_2d[:, 1],
                mode='markers+text',
                marker=dict(size=sizes_proj, color=colors_proj, line=dict(width=1, color='white')),
                text=[f"{idx}" for idx in valid_indices],
                textposition='top center',
                textfont=dict(size=10, color='white'),
                hovertemplate='Marker %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>',
                name='projected',
                customdata=valid_indices  # Store marker indices for hover detection
            ))

        # Plot annotated correspondences
        if self.correspondences:
            corr_x = [pt[0] for _, _, pt in self.correspondences]
            corr_y = [pt[1] for _, _, pt in self.correspondences]
            corr_text = [f"{idx}@f{f_idx}" for idx, f_idx, _ in self.correspondences]

            fig.add_trace(go.Scatter(
                x=corr_x,
                y=corr_y,
                mode='markers',
                marker=dict(size=12, color='red', symbol='x', line=dict(width=2, color='white')),
                text=corr_text,
                hovertemplate='Annotated #%{text}<extra></extra>'
            ))

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, self.img_width]),
            yaxis=dict(visible=False, range=[self.img_height, 0], scaleanchor='x'),
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='closest',
            clickmode='event+select'  # Ensure all clicks are captured
        )

        return fig

    def run(self):
        """Run the Dash application."""
        app = self.create_app()
        print(f"\n{'='*70}")
        print(f"Starting Interactive Extrinsics Calibration Tool")
        print(f"{'='*70}")
        print(f"\nOpen browser: http://localhost:{self.port}")
        print(f"\nPress Ctrl+C to stop")
        print(f"{'='*70}\n")

        app.run(debug=False, host='0.0.0.0', port=self.port)


def main():
    parser = argparse.ArgumentParser(description='Interactive extrinsics calibration tool')
    parser.add_argument('--csv', required=True, help='Path to mocap CSV file')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--mcal', required=True, help='Path to .mcal file')
    parser.add_argument('--intrinsics', help='Path to intrinsics JSON file (ignored if --use-mcal-intrinsics is set)')
    parser.add_argument('--camera_serial', default='C11764', help='Camera serial number')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame')
    parser.add_argument('--port', type=int, default=8050, help='Web server port')
    parser.add_argument('--use-mcal-intrinsics', action='store_true',
                       help='Use intrinsics from .mcal file instead of JSON file (RECOMMENDED for OptiTrack)')

    args = parser.parse_args()

    # Validate arguments
    if not args.use_mcal_intrinsics and not args.intrinsics:
        parser.error("Either --intrinsics or --use-mcal-intrinsics must be specified")

    tool = ExtrinsicsAnnotationTool(
        csv_path=args.csv,
        video_path=args.video,
        mcal_path=args.mcal,
        intrinsics_path=args.intrinsics,
        camera_serial=args.camera_serial,
        start_frame=args.start_frame,
        port=args.port,
        use_mcal_intrinsics=args.use_mcal_intrinsics
    )

    tool.run()


if __name__ == "__main__":
    main()
