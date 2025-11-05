#!/usr/bin/env python3
"""
Interactive mocap marker annotation tool using Dash.

Usage:
    python annotate_mocap_markers.py --csv /path/to/mocap.csv --start_frame 2 --num_frames 200

This will open a web interface where you can:
1. Click on markers to select them
2. Type a label name and press "Set Label"
3. Labels are automatically saved to a CSV file
4. Navigate through frames with the slider
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
import argparse
import json
import os


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
    print(f"Labels saved to {labels_file}")


def create_dash_app(markers_xyz, marker_ids, frame_range, metadata, labels_file):
    """Create Dash app for interactive annotation."""
    start_frame, end_frame = frame_range
    num_frames = end_frame - start_frame

    # Initialize labels
    labels = load_labels(labels_file)

    # Create marker name to ID mapping
    marker_name_to_id = {}
    for marker_name, marker_id in zip(markers_xyz.keys(), marker_ids):
        marker_name_to_id[marker_name] = marker_id

    # Calculate data bounds
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
        html.H1("Mocap Marker Annotation Tool", style={'textAlign': 'center'}),

        html.Div([
            html.Div([
                html.H3("Instructions:"),
                html.Ul([
                    html.Li("Click on a marker point to select it"),
                    html.Li("Enter a label name below (e.g., 'Laxisl1')"),
                    html.Li("Click 'Set Label' to save"),
                    html.Li("Labels are automatically saved to CSV"),
                    html.Li("Use slider to navigate frames")
                ])
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

            html.Div([
                html.H3("Selected Marker:"),
                html.Div(id='selected-marker-info', children='None selected'),
                html.Br(),
                html.Label("Label Name:"),
                dcc.Input(
                    id='label-input',
                    type='text',
                    placeholder='e.g., Laxisl1',
                    style={'width': '200px', 'marginLeft': '10px'}
                ),
                html.Button('Set Label', id='set-label-btn', n_clicks=0,
                           style={'marginLeft': '10px'}),
                html.Div(id='label-status', style={'marginTop': '10px', 'color': 'green'}),
                html.Br(),
                html.H4("Current Labels:"),
                html.Div(id='labels-list', style={'maxHeight': '300px', 'overflowY': 'scroll'})
            ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'})
        ]),

        dcc.Graph(id='3d-scatter', style={'height': '700px'}),

        html.Div([
            html.Label(f'Frame: {start_frame}', id='frame-label'),
            dcc.Slider(
                id='frame-slider',
                min=0,
                max=num_frames - 1,
                value=0,
                marks={i: str(start_frame + i) for i in range(0, num_frames, max(1, num_frames // 10))},
                step=1
            )
        ], style={'padding': '20px'}),

        # Store for labels and selected marker
        dcc.Store(id='labels-store', data=labels),
        dcc.Store(id='selected-marker', data=None)
    ])

    @app.callback(
        Output('3d-scatter', 'figure'),
        Output('frame-label', 'children'),
        Input('frame-slider', 'value'),
        Input('labels-store', 'data')
    )
    def update_graph(frame_idx, labels_data):
        current_frame = start_frame + frame_idx

        # Collect marker data for current frame
        marker_points = []
        for marker_name, xyz in markers_xyz.items():
            frame_xyz = xyz[current_frame]

            # Skip if NaN
            if not np.isnan(frame_xyz).any():
                # Get label if exists
                label = labels_data.get(marker_name, {}).get('label', marker_name)

                marker_points.append({
                    'original_name': marker_name,
                    'display_name': label,
                    'x': frame_xyz[0],
                    'y': frame_xyz[1],
                    'z': frame_xyz[2],
                    'is_labeled': marker_name in labels_data
                })

        # Separate labeled and unlabeled markers
        labeled_points = [p for p in marker_points if p['is_labeled']]
        unlabeled_points = [p for p in marker_points if not p['is_labeled']]

        # Create figure
        fig = go.Figure()

        # Add unlabeled markers (gray)
        if unlabeled_points:
            fig.add_trace(go.Scatter3d(
                x=[p['x'] for p in unlabeled_points],
                y=[p['y'] for p in unlabeled_points],
                z=[p['z'] for p in unlabeled_points],
                mode='markers',
                name='Unlabeled',
                marker=dict(
                    size=5,
                    color='lightgray',
                    opacity=0.6
                ),
                text=[p['original_name'] for p in unlabeled_points],
                customdata=[p['original_name'] for p in unlabeled_points],
                hovertemplate='<b>%{text}</b><br>X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<extra></extra>'
            ))

        # Add labeled markers (colored by height)
        if labeled_points:
            fig.add_trace(go.Scatter3d(
                x=[p['x'] for p in labeled_points],
                y=[p['y'] for p in labeled_points],
                z=[p['z'] for p in labeled_points],
                mode='markers+text',
                name='Labeled',
                marker=dict(
                    size=7,
                    color=[p['y'] for p in labeled_points],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Height (mm)")
                ),
                text=[p['display_name'] for p in labeled_points],
                textposition='top center',
                textfont=dict(size=8),
                customdata=[p['original_name'] for p in labeled_points],
                hovertemplate='<b>%{text}</b><br>Original: %{customdata}<br>X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<extra></extra>'
            ))

        # Update layout
        camera = dict(
            eye=dict(x=1.5, y=2.0, z=1.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        )

        fig.update_layout(
            title=f'Frame {current_frame} | Labeled: {len(labeled_points)} / {len(marker_points)}',
            scene=dict(
                xaxis=dict(title='X (mm) - Horizontal', range=x_range),
                yaxis=dict(title='Y (mm) - Vertical (Up)', range=y_range),
                zaxis=dict(title='Z (mm) - Horizontal', range=z_range),
                aspectmode='data',
                camera=camera
            ),
            showlegend=True,
            height=700
        )

        return fig, f'Frame: {current_frame}'

    @app.callback(
        Output('selected-marker', 'data'),
        Output('selected-marker-info', 'children'),
        Output('label-input', 'value'),
        Input('3d-scatter', 'clickData'),
        State('labels-store', 'data')
    )
    def select_marker(clickData, labels_data):
        if clickData is None:
            return None, 'None selected', ''

        try:
            # Get the original marker name from customdata
            point_data = clickData['points'][0]
            original_name = point_data['customdata']

            # Get current label if exists
            current_label = labels_data.get(original_name, {}).get('label', '')

            info = html.Div([
                html.P(f"Original Name: {original_name}"),
                html.P(f"Current Label: {current_label if current_label else 'Not labeled'}"),
                html.P(f"Position: X={point_data['x']:.1f}, Y={point_data['y']:.1f}, Z={point_data['z']:.1f}")
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
            return labels_data, 'Please select a marker first', generate_labels_list(labels_data)

        if not label_value or label_value.strip() == '':
            return labels_data, 'Please enter a label name', generate_labels_list(labels_data)

        # Update labels
        labels_data[selected_marker] = {
            'label': label_value.strip(),
            'marker_id': marker_name_to_id.get(selected_marker, '')
        }

        # Save to file
        save_labels(labels_data, labels_file)

        status = f'✓ Label "{label_value}" set for {selected_marker}'

        return labels_data, status, generate_labels_list(labels_data)

    def generate_labels_list(labels_data):
        """Generate HTML list of current labels."""
        if not labels_data:
            return html.P("No labels yet", style={'fontStyle': 'italic'})

        items = []
        for original_name, info in sorted(labels_data.items(), key=lambda x: x[1]['label']):
            items.append(
                html.Li(f"{info['label']} ← {original_name}")
            )

        return html.Ul(items)

    return app


def main():
    parser = argparse.ArgumentParser(description='Interactive mocap marker annotation tool')
    parser.add_argument('--csv', type=str,
                        default='/Volumes/FastACIS/csldata/csl/mocap.csv',
                        help='Path to mocap CSV file')
    parser.add_argument('--start_frame', type=int, default=2,
                        help='Start frame for visualization')
    parser.add_argument('--num_frames', type=int, default=200,
                        help='Number of frames to load')
    parser.add_argument('--labels', type=str, default='marker_labels.csv',
                        help='Path to labels CSV file')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port for web server (default: 8050)')

    args = parser.parse_args()

    print("="*60)
    print("Loading mocap data...")
    print("="*60)

    metadata, marker_names, marker_ids, data_df = parse_motive_csv(args.csv)

    print(f"Found {len(marker_names)} markers")
    print(f"Total frames: {len(data_df)}")

    markers_xyz = extract_marker_data(data_df, marker_names)

    end_frame = min(args.start_frame + args.num_frames, len(data_df))

    print(f"Loaded frames {args.start_frame} to {end_frame}")
    print(f"Labels will be saved to: {args.labels}")
    print("="*60)

    app = create_dash_app(
        markers_xyz,
        marker_ids,
        (args.start_frame, end_frame),
        metadata,
        args.labels
    )

    print("\n" + "="*60)
    print("Starting web server...")
    print("="*60)
    print(f"\nOpen your browser and go to: http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")

    app.run(debug=True, port=args.port)


if __name__ == '__main__':
    main()
