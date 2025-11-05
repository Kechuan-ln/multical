#!/usr/bin/env python3
"""
Interactive tool to select 4 points on prosthesis STL mesh.

Usage:
    python annotate_prosthesis_points.py \
        --stl /path/to/Genesis.STL \
        --output prosthesis_config.json
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL
import json
import argparse
from pathlib import Path
from scipy.spatial import KDTree


def load_stl(stl_path):
    """Load STL file and return vertices and faces."""
    try:
        import stl
        mesh = stl.mesh.Mesh.from_file(stl_path)

        # Extract unique vertices
        vertices = mesh.vectors.reshape(-1, 3)
        n_vertices = len(vertices)

        # Create faces (triangles)
        faces = np.arange(n_vertices).reshape(-1, 3)

        print(f"✓ Loaded STL: {n_vertices} vertices, {len(faces)} faces")

        # Center the mesh
        center = vertices.mean(axis=0)
        vertices_centered = vertices - center

        # Build KDTree for nearest neighbor search
        kdtree = KDTree(vertices_centered)

        return vertices_centered, faces, center, kdtree

    except ImportError:
        print("Installing numpy-stl...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'numpy-stl'])
        return load_stl(stl_path)


class ProsthesisAnnotationTool:
    """Interactive tool for selecting 4 points on prosthesis mesh."""

    def __init__(self, stl_path, output_path, port=8060):
        self.stl_path = stl_path
        self.output_path = output_path
        self.port = port

        # Load mesh
        print(f"\nLoading mesh from: {stl_path}")
        self.vertices, self.faces, self.mesh_center, self.kdtree = load_stl(stl_path)

        # Selected points
        self.selected_points = {}  # {marker_name: [x, y, z]}
        self.marker_names = ['RPBR', 'RPBL', 'RPUL', 'RPUR']
        self.marker_descriptions = {
            'RPBR': 'Right Prosthesis Back Right',
            'RPBL': 'Right Prosthesis Back Left',
            'RPUL': 'Right Prosthesis Upper Left',
            'RPUR': 'Right Prosthesis Upper Right'
        }
        self.current_marker_idx = 0

        print(f"\n✓ Ready to annotate 4 points")
        print(f"  Points to annotate: {', '.join(self.marker_names)}")

    def snap_to_mesh(self, point):
        """Snap a clicked point to the nearest mesh vertex."""
        point_arr = np.array(point)
        dist, idx = self.kdtree.query(point_arr)
        nearest_point = self.vertices[idx]
        return nearest_point.tolist()

    def create_app(self):
        """Create Dash application."""
        app = Dash(__name__)

        app.layout = html.Div([
            html.H1("Prosthesis Point Annotation Tool", style={'textAlign': 'center'}),

            # Instructions panel
            html.Div([
                html.H3("Instructions:"),
                html.Ol([
                    html.Li("Click on the mesh surface to select a point (point will snap to nearest vertex)"),
                    html.Li("Fine-tune coordinates using the input boxes below if needed"),
                    html.Li("Click 'Confirm Point' to save and move to next marker"),
                    html.Li("Use view buttons to change camera angle"),
                    html.Li("Repeat for all 4 markers, then click 'Save Configuration'")
                ]),
            ], style={'padding': '10px', 'backgroundColor': '#f0f0f0', 'margin': '10px'}),

            # Current marker info
            html.Div([
                html.H3(id='current-marker-text',
                       children=f'Current marker: {self.marker_names[0]}',
                       style={'color': 'blue', 'marginBottom': '5px'}),
                html.P(id='marker-description',
                      children=self.marker_descriptions[self.marker_names[0]],
                      style={'fontStyle': 'italic', 'color': '#666'}),
            ], style={'padding': '10px', 'backgroundColor': '#fff3cd', 'margin': '10px'}),

            # View control buttons
            html.Div([
                html.H4("View Controls:"),
                html.Button('Top View', id='view-top', n_clicks=0,
                           style={'margin': '5px', 'padding': '8px 15px'}),
                html.Button('Front View', id='view-front', n_clicks=0,
                           style={'margin': '5px', 'padding': '8px 15px'}),
                html.Button('Side View', id='view-side', n_clicks=0,
                           style={'margin': '5px', 'padding': '8px 15px'}),
                html.Button('Reset View', id='view-reset', n_clicks=0,
                           style={'margin': '5px', 'padding': '8px 15px'}),
            ], style={'padding': '10px', 'margin': '10px', 'backgroundColor': '#e8f4f8'}),

            # 3D plot
            dcc.Graph(id='mesh-plot', style={'height': '600px', 'margin': '10px'}),

            # Coordinate adjustment panel
            html.Div([
                html.H4("Fine-tune Coordinates:"),
                html.Div([
                    html.Label("X: "),
                    dcc.Input(id='coord-x', type='number', step=0.1,
                             style={'width': '100px', 'margin': '5px'}),
                    html.Label(" Y: ", style={'marginLeft': '20px'}),
                    dcc.Input(id='coord-y', type='number', step=0.1,
                             style={'width': '100px', 'margin': '5px'}),
                    html.Label(" Z: ", style={'marginLeft': '20px'}),
                    dcc.Input(id='coord-z', type='number', step=0.1,
                             style={'width': '100px', 'margin': '5px'}),
                    html.Button('Update Point', id='update-coords-btn', n_clicks=0,
                               style={'marginLeft': '20px', 'padding': '5px 15px'}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.P(id='coord-hint',
                      children='Click on the mesh or enter coordinates manually',
                      style={'color': '#666', 'fontSize': '12px', 'marginTop': '10px'}),
            ], style={'padding': '10px', 'margin': '10px', 'backgroundColor': '#f9f9f9'}),

            # Action buttons
            html.Div([
                html.H4(id='status-text', children='Status: Ready',
                       style={'marginBottom': '15px'}),
                html.Button('Confirm Point', id='confirm-btn', n_clicks=0,
                           style={'fontSize': '18px', 'padding': '12px 30px',
                                  'margin': '10px', 'backgroundColor': '#4CAF50',
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px',
                                  'cursor': 'pointer'}),
                html.Button('Undo Last', id='undo-btn', n_clicks=0,
                           style={'fontSize': '18px', 'padding': '12px 30px',
                                  'margin': '10px', 'backgroundColor': '#ff9800',
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px',
                                  'cursor': 'pointer'}),
                html.Button('Save Configuration', id='save-btn', n_clicks=0,
                           style={'fontSize': '18px', 'padding': '12px 30px',
                                  'margin': '10px', 'backgroundColor': '#2196F3',
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px',
                                  'cursor': 'pointer'}),
            ], style={'padding': '20px', 'textAlign': 'center', 'backgroundColor': '#f0f0f0',
                     'margin': '10px'}),

            # Selected points list
            html.Div([
                html.H4("Selected Points:"),
                html.Div(id='points-list', children=self.format_points_list())
            ], style={'padding': '20px', 'margin': '10px', 'backgroundColor': '#e8f5e9'}),

            # Hidden stores
            dcc.Store(id='temp-point', data=None),
            dcc.Store(id='selected-points-store', data={}),
            dcc.Store(id='camera-view', data=None),
        ])

        self.setup_callbacks(app)
        return app

    def format_points_list(self):
        """Format selected points as HTML."""
        if not self.selected_points:
            return html.P("No points selected yet")

        items = []
        for marker_name in self.marker_names:
            if marker_name in self.selected_points:
                pt = self.selected_points[marker_name]
                items.append(html.Li(f"{marker_name}: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}) ✓"))
            else:
                items.append(html.Li(f"{marker_name}: (not selected yet)"))

        return html.Ul(items)

    def setup_callbacks(self, app):
        """Setup Dash callbacks."""

        @app.callback(
            [Output('mesh-plot', 'figure'),
             Output('status-text', 'children'),
             Output('current-marker-text', 'children'),
             Output('marker-description', 'children'),
             Output('points-list', 'children'),
             Output('temp-point', 'data'),
             Output('coord-x', 'value'),
             Output('coord-y', 'value'),
             Output('coord-z', 'value'),
             Output('camera-view', 'data')],
            [Input('mesh-plot', 'clickData'),
             Input('confirm-btn', 'n_clicks'),
             Input('undo-btn', 'n_clicks'),
             Input('save-btn', 'n_clicks'),
             Input('update-coords-btn', 'n_clicks'),
             Input('view-top', 'n_clicks'),
             Input('view-front', 'n_clicks'),
             Input('view-side', 'n_clicks'),
             Input('view-reset', 'n_clicks')],
            [State('temp-point', 'data'),
             State('selected-points-store', 'data'),
             State('coord-x', 'value'),
             State('coord-y', 'value'),
             State('coord-z', 'value'),
             State('camera-view', 'data')]
        )
        def update_all(click_data, confirm_clicks, undo_clicks, save_clicks,
                      update_coords_clicks, view_top, view_front, view_side, view_reset,
                      temp_point, selected_points_data,
                      coord_x, coord_y, coord_z, camera_view):

            # Restore state
            if selected_points_data:
                self.selected_points = {k: v for k, v in selected_points_data.items()}

            triggered_id = ctx.triggered_id if ctx.triggered_id else None

            status_msg = "Ready"
            temp_point_new = temp_point
            camera_view_new = camera_view

            # Handle view button clicks
            if triggered_id == 'view-top':
                camera_view_new = 'top'
                status_msg = "View changed to: Top"
            elif triggered_id == 'view-front':
                camera_view_new = 'front'
                status_msg = "View changed to: Front"
            elif triggered_id == 'view-side':
                camera_view_new = 'side'
                status_msg = "View changed to: Side"
            elif triggered_id == 'view-reset':
                camera_view_new = None
                status_msg = "View reset to default"

            # Handle coordinate update
            elif triggered_id == 'update-coords-btn':
                if coord_x is not None and coord_y is not None and coord_z is not None:
                    temp_point_new = [float(coord_x), float(coord_y), float(coord_z)]
                    # Snap to nearest mesh vertex
                    temp_point_new = self.snap_to_mesh(temp_point_new)
                    status_msg = "Coordinates updated and snapped to mesh"
                else:
                    status_msg = "⚠️  Please enter all X, Y, Z coordinates"

            # Handle button clicks
            elif triggered_id == 'confirm-btn':
                if temp_point:
                    # Confirm the temporary point
                    if self.current_marker_idx < len(self.marker_names):
                        marker_name = self.marker_names[self.current_marker_idx]
                        self.selected_points[marker_name] = temp_point
                        self.current_marker_idx += 1
                        temp_point_new = None

                        if self.current_marker_idx < len(self.marker_names):
                            status_msg = f"✓ Confirmed {marker_name}. Select next point."
                        else:
                            status_msg = "✓ All 4 points selected! Click 'Save Configuration'."
                else:
                    status_msg = "⚠️  Please click on the mesh first or enter coordinates"

            elif triggered_id == 'undo-btn':
                if self.current_marker_idx > 0:
                    self.current_marker_idx -= 1
                    marker_name = self.marker_names[self.current_marker_idx]
                    if marker_name in self.selected_points:
                        del self.selected_points[marker_name]
                    temp_point_new = None
                    status_msg = f"Undone. Select point for {marker_name}"

            elif triggered_id == 'save-btn':
                if len(self.selected_points) == 4:
                    self.save_configuration()
                    status_msg = f"✓ Configuration saved to {self.output_path}"
                else:
                    status_msg = f"⚠️  Need 4 points, currently have {len(self.selected_points)}"

            elif triggered_id == 'mesh-plot' and click_data:
                # Extract clicked point and snap to mesh
                if 'points' in click_data and len(click_data['points']) > 0:
                    point_data = click_data['points'][0]
                    if 'x' in point_data and 'y' in point_data and 'z' in point_data:
                        clicked_point = [point_data['x'], point_data['y'], point_data['z']]
                        # Snap to nearest mesh vertex
                        temp_point_new = self.snap_to_mesh(clicked_point)
                        status_msg = f"Point selected and snapped to mesh. Click 'Confirm' to save."

            # Update UI text
            if self.current_marker_idx < len(self.marker_names):
                current_marker = self.marker_names[self.current_marker_idx]
                current_marker_text = f"Current marker: {current_marker} ({self.current_marker_idx+1}/4)"
                marker_description = self.marker_descriptions[current_marker]
            else:
                current_marker_text = "All markers selected ✓"
                marker_description = "Ready to save configuration"

            # Update coordinate input boxes
            if temp_point_new:
                coord_x_val, coord_y_val, coord_z_val = temp_point_new
            else:
                coord_x_val = coord_y_val = coord_z_val = None

            # Create figure
            fig = self.create_figure(temp_point_new, camera_view_new)
            points_list = self.format_points_list()

            return (fig, status_msg, current_marker_text, marker_description,
                   points_list, temp_point_new, coord_x_val, coord_y_val, coord_z_val,
                   camera_view_new)

        @app.callback(
            Output('selected-points-store', 'data'),
            [Input('confirm-btn', 'n_clicks'),
             Input('undo-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def update_store(confirm, undo):
            return {k: v for k, v in self.selected_points.items()}

    def create_figure(self, temp_point=None, camera_view=None):
        """Create 3D mesh figure with selected points."""
        fig = go.Figure()

        # Add mesh
        fig.add_trace(go.Mesh3d(
            x=self.vertices[:, 0],
            y=self.vertices[:, 1],
            z=self.vertices[:, 2],
            i=self.faces[:, 0],
            j=self.faces[:, 1],
            k=self.faces[:, 2],
            color='lightblue',
            opacity=0.7,
            name='Prosthesis'
        ))

        # Add confirmed selected points (larger markers)
        for marker_name in self.marker_names:
            if marker_name in self.selected_points:
                pt = self.selected_points[marker_name]
                fig.add_trace(go.Scatter3d(
                    x=[pt[0]],
                    y=[pt[1]],
                    z=[pt[2]],
                    mode='markers+text',
                    marker=dict(size=15, color='red', symbol='circle',
                               line=dict(color='darkred', width=2)),
                    text=[marker_name],
                    textposition='top center',
                    textfont=dict(size=14, color='darkred'),
                    name=marker_name
                ))

        # Add temporary point (not yet confirmed)
        if temp_point:
            fig.add_trace(go.Scatter3d(
                x=[temp_point[0]],
                y=[temp_point[1]],
                z=[temp_point[2]],
                mode='markers+text',
                marker=dict(size=18, color='orange', symbol='diamond',
                           line=dict(color='darkorange', width=2)),
                text=['PENDING'],
                textposition='top center',
                textfont=dict(size=14, color='darkorange'),
                name='Temp (click Confirm)'
            ))

        # Set camera view
        camera_dict = None
        if camera_view == 'top':
            camera_dict = dict(
                eye=dict(x=0, y=0, z=2.5),
                up=dict(x=0, y=1, z=0)
            )
        elif camera_view == 'front':
            camera_dict = dict(
                eye=dict(x=0, y=-2.5, z=0),
                up=dict(x=0, y=0, z=1)
            )
        elif camera_view == 'side':
            camera_dict = dict(
                eye=dict(x=2.5, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )

        # Update layout
        scene_config = dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )

        if camera_dict:
            scene_config['camera'] = camera_dict

        fig.update_layout(
            scene=scene_config,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            hovermode='closest'
        )

        return fig

    def save_configuration(self):
        """Save selected points to JSON file."""
        config = {
            'prosthesis_name': 'Genesis Running Blade',
            'stl_file': str(Path(self.stl_path).absolute()),
            'mesh_center': self.mesh_center.tolist(),
            'points': {
                marker_name: {
                    'position': self.selected_points[marker_name],
                    'description': f'Point on prosthesis mesh corresponding to marker {marker_name}'
                }
                for marker_name in self.marker_names
                if marker_name in self.selected_points
            },
            'marker_order': self.marker_names,
            'note': 'Positions are in mesh-centered coordinates'
        }

        with open(self.output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ Configuration saved to: {self.output_path}")
        print(f"{'='*70}")
        print(f"\nSelected points:")
        for marker_name in self.marker_names:
            if marker_name in self.selected_points:
                pt = self.selected_points[marker_name]
                print(f"  {marker_name}: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")

    def run(self):
        """Run the Dash application."""
        app = self.create_app()
        print(f"\n{'='*70}")
        print(f"Starting Prosthesis Annotation Tool")
        print(f"{'='*70}")
        print(f"\nOpen browser: http://localhost:{self.port}")
        print(f"\nPress Ctrl+C to stop")
        print(f"{'='*70}\n")

        app.run(debug=False, host='0.0.0.0', port=self.port)


def main():
    parser = argparse.ArgumentParser(description='Annotate 4 points on prosthesis STL mesh')
    parser.add_argument('--stl', required=True, help='Path to STL file')
    parser.add_argument('--output', default='prosthesis_config.json',
                       help='Output configuration file')
    parser.add_argument('--port', type=int, default=8060, help='Web server port')

    args = parser.parse_args()

    tool = ProsthesisAnnotationTool(
        stl_path=args.stl,
        output_path=args.output,
        port=args.port
    )

    tool.run()


if __name__ == "__main__":
    main()
