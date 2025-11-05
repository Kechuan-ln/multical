#!/usr/bin/env python3
"""
Gradio-based 3D Joint Annotation Approval Tool

This tool provides a web interface for reviewing and approving automatically 
generated 3D joint triangulation results by visualizing 2D projections 
overlaid on images from multiple cameras.

Features:
- Load triangulated 3D poses from JSON files
- Display 2D projections of 3D poses on multi-view images
- Interactive approval/discard functionality for each frame
- Export approval decisions

Prerequisites:
- Triangulation results from automatic 3D pose estimation saved as JSON files
- EgoExo dataset with LMDB image storage
- Camera calibration parameters
"""

import copy
import os
import sys
import json
import argparse
import numpy as np
import cv2
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.append('..')
from dataset.recording import Recording
from utils.logger import ColorLogger
from utils.constants import PATH_ASSETS, PATH_ASSETS_KPT3D, VIT_JOINTS_NAME, VIT_SKELETON, VIT_KEYPOINT_COLORS, PATH_ASSETS_VIDEOS, PATH_ASSETS_REFINED_KPT3D
from utils.io_utils import load_3d_keypoint_json, NumpyEncoder

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gradio-based 3D Joint Annotation Approval Tool', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--recording_tag', type=str, default='sync_9122', help='recording name')
    parser.add_argument('--path_camera',type=str, default='', help='Path to the camera meta json file, if empty, use calibration.json under --recording_tag')
    parser.add_argument('--use_manual_annotation', action='store_true', help='use manual annotation for triangulation')

    
    parser.add_argument('--port', type=int, default=7862, help='Port for Gradio interface')
    parser.add_argument('--share', action='store_true', help='Create public sharing link')

    return parser.parse_args()

class Pose3DApprovalTool:
    def __init__(self, args):
        """Initialize the 3D pose approval tool with required data."""
        self.dir_annotation = PATH_ASSETS_REFINED_KPT3D
        self.setup_dataset(args)
        self.load_existing_json(args)


        # Set up output path for saving approvals
        auto_tag = 'manual' if args.use_manual_annotation else 'auto'
        self.output_path = os.path.join(self.dir_annotation, f"{self.recording_name}_{auto_tag}_approved.json")

        self.vit_joints_name = VIT_JOINTS_NAME
        self.vit_skeleton = VIT_SKELETON
        self.vit_keypoint_colors = VIT_KEYPOINT_COLORS
        
        # Approval status constants
        self.FLAG_PENDING = 0
        self.FLAG_APPROVED = 1
        self.FLAG_DISCARDED = 2
        
    def setup_dataset(self,args):
        load_undistort_images  = 'undistort' in args.recording_tag
        self.recording_name = args.recording_tag.split('/')[-2]

        self.log_dir = os.path.join(self.dir_annotation, 'logs')
        # Set up logger
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = ColorLogger(self.log_dir, log_name=f'{self.recording_name}_approved.txt')

        path_root_dir = os.path.join(PATH_ASSETS_VIDEOS, args.recording_tag)
        path_camera = args.path_camera if args.path_camera!='' else os.path.join(path_root_dir, 'calibration.json')
        assert os.path.exists(path_camera), f"Camera meta file does not exist: {path_camera}"

        self.dataset =  Recording(root_dir = path_root_dir,
                        path_cam_meta = path_camera, 
                        logger = self.logger, 
                        recording_name = self.recording_name,
                        load_undistort_images = load_undistort_images)
        
        self.logger.info("Dataset initialized successfully")
        self.image_hw = self.dataset.image_hw
        self.cam_params = self.dataset.get_camera_params()
        self.cam_keys = list(self.cam_params.keys())

        self.input_zup = self.dataset.input_zup

    def load_existing_json(self, args):
        """Load triangulation results from JSON files."""
        auto_tag = 'manual' if args.use_manual_annotation else 'auto'
        triangulation_path = os.path.join(self.dir_annotation, f"{self.recording_name}_{auto_tag}")

        self.json_data = load_3d_keypoint_json(triangulation_path)

        # Get available frames from dataset
        self.available_frames = sorted([int(img_id) for img_id in self.json_data.keys()])
        

    
    def save_annotations(self):
        """Save approved 3D annotations to the output file."""
        import datetime
        
        box_style = "border: 1px solid #ddd; border-radius: 8px; padding: 12px; background-color: #f9f9f9; font-family: Arial, sans-serif;"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            
            # Create annotation data with refined_3d and updated validity flags
            annotation_data = {}
            for frame_key, frame_data in self.json_data.items(): 
                if 'approval_status' not in frame_data:
                    continue
                # Get original refined_3d and approval status
                refined_3d = np.array(frame_data['refined_3d'])
                approval_status = frame_data['approval_status']
                
                
                new_refined_3d = refined_3d.copy()
                
                for joint_idx in range(len(self.vit_joints_name)):
                    # Original validity check
                    originally_valid = refined_3d[joint_idx, 3] > 1e-4
                    
                    # Joint is valid only if originally valid AND approved
                    new_refined_3d[joint_idx, 3] = 1.0 if  originally_valid and approval_status[joint_idx] != self.FLAG_DISCARDED else 0.0
                # Save frame data with updated refined_3d
                annotation_data[frame_key] = {'refined_3d': new_refined_3d.tolist()}
                if 'vitpose_2d' in frame_data:
                    annotation_data[frame_key]['vitpose_2d'] = frame_data['vitpose_2d']
                if 'annotation_2d' in frame_data:
                    annotation_data[frame_key]['annotation_2d'] = frame_data['annotation_2d']
            
            # Save annotation data with proper formatting
            with open(self.output_path, 'w') as f:
                json.dump(annotation_data, f, separators=(',', ':'), cls=NumpyEncoder)
            
            frame_count = len(annotation_data)
            content = f"<h3 style='color: #28A745; margin: 0; margin-bottom: 8px;'>✅ SUCCESS</h3><p>3D annotations saved successfully<br><strong>Frames:</strong> {frame_count}<br><strong>Output:</strong> {self.output_path}<br><strong>Time:</strong> {timestamp}</p>"
            return f"<div style='{box_style}'><strong>Save Status</strong><br><br>{content}</div>"
        except Exception as e:
            content = f"<h3 style='color: #DC3545; margin: 0; margin-bottom: 8px;'>❌ ERROR</h3><p>Failed to save annotations<br><strong>Error:</strong> {str(e)}<br><strong>Time:</strong> {timestamp}</p>"
            return f"<div style='{box_style}'><strong>Save Status</strong><br><br>{content}</div>"




    def get_frame_data(self, frame_id):
        """Get all data for a specific frame."""
        if frame_id not in self.available_frames:
            return None
            
        frame_key = str(frame_id)            
        frame_data = self.json_data[frame_key]

        if 'approval_status' not in frame_data:
            frame_data['approval_status'] = np.zeros(len(self.vit_joints_name), dtype=np.float32) + self.FLAG_PENDING


        frame_data = copy.deepcopy(self.json_data[frame_key])

        # Load images for all cameras
        images = {}
        for cam_key in self.cam_keys:
            img_bgr = self.dataset.load_bgr_image_and_undistort(f"{cam_key}/frame_{frame_id:04d}.png")
            images[cam_key] = img_bgr
            
        frame_data['images'] = images
        frame_data['frame_id'] = frame_id
        
        return frame_data
    
    
    
    def visualize_2d_projection(self, image, pose_2d, approval_status):
        """Visualize 2D projection of 3D pose on image as a cropped patch."""
        # Find valid points for bounding box calculation
        valid_mask = (pose_2d[:, 0] >= 0) & (pose_2d[:, 1] >= 0)
        if np.sum(valid_mask) ==0:
            canvas = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)
            img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            return img_pil
        

        valid_points = pose_2d[valid_mask]
        min_x, min_y = np.min(valid_points, axis=0).astype(int)
        max_x, max_y = np.max(valid_points, axis=0).astype(int)
        img_height, img_width = image.shape[:2]

        if min_x>=img_width or min_y>=img_height or max_x<0 or max_y<0:
            canvas = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)
            img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            return img_pil

        
        
        # Add padding (20% of bbox size, minimum 100 pixels)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        padding_x = max(100, int(bbox_width * 0.2))
        padding_y = max(100, int(bbox_height * 0.2))
        
        crop_x1 = max(0, min_x - padding_x)
        crop_y1 = max(0, min_y - padding_y)
        crop_x2 = min(img_width, max_x + padding_x)
        crop_y2 = min(img_height, max_y + padding_y)
        
        
        # Resize to fixed patch size (500, 500) while preserving aspect ratio
        crop_height, crop_width = crop_y2 - crop_y1, crop_x2 - crop_x1
        scale_factor = min(500 / crop_width, 500 / crop_height)
        new_width = int(crop_width * scale_factor)
        new_height = int(crop_height * scale_factor)

        point_size =  max(2, int(8/scale_factor))

        # Draw skeleton connections
        # First, draw pose on the full image
        img_vis = image.copy()
        for connection in self.vit_skeleton:
            pt1_idx, pt2_idx = connection
            
            pt1 = tuple(pose_2d[pt1_idx].astype(int))
            pt2 = tuple(pose_2d[pt2_idx].astype(int))

            if approval_status[pt1_idx] == self.FLAG_DISCARDED or approval_status[pt2_idx] == self.FLAG_DISCARDED:
                continue
            
            if pt1[0] > -1e-4 and pt1[1] > -1e-4 and pt2[0] > -1e-4 and pt2[1] > -1e-4:
                cv2.line(img_vis, pt1, pt2, (128, 128, 128), min(5, point_size))
        
        # Draw keypoints using designated colors from vit_keypoint_colors
        for i, point in enumerate(pose_2d):
            center = tuple(point.astype(int))
            if center[0] < 0 or center[1] < 0:
                continue
                
            if approval_status[i] == self.FLAG_DISCARDED:
                continue
            
            
            if approval_status[i] == self.FLAG_APPROVED:
                color = (0, 255, 0)  # Green for approved joints
            else:
                color = self.vit_keypoint_colors[i % len(self.vit_keypoint_colors)]
            
            cv2.circle(img_vis, center, point_size, color, -1)
        
        # Resize with preserved aspect ratio
        img_cropped = img_vis[crop_y1:crop_y2, crop_x1:crop_x2]
        img_resized = cv2.resize(img_cropped, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create a 500x500 canvas and center the resized image
        canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255  # White background
        
        # Calculate position to center the image
        start_x = (500 - new_width) // 2
        start_y = (500 - new_height) // 2
        
        # Place the resized image on the canvas
        canvas[start_y:start_y + new_height, start_x:start_x + new_width] = img_resized
        
        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        return img_pil
        
    
    def create_frame_visualization(self, frame_data):
        """Create visualization for all cameras in a frame."""
            
        camera_images = []
        approval_status = frame_data['approval_status']
        
        for cam_key in self.cam_keys:
            #pose_2d = frame_data['projection_2d'][cam_key]

            triangulated_3d = frame_data['refined_3d'][...,:3]
            valid_3d = frame_data['refined_3d'][..., 3] > 1e-4
                
            cam_param = self.cam_params[cam_key]
            K = cam_param['K']
            rvec = cam_param['rvec']
            tvec = cam_param['tvec']
            
            # Use OpenCV's projectPoints function
            pose_2d, _ = cv2.projectPoints(triangulated_3d.reshape(-1, 1, 3), rvec, tvec, K, None)
            pose_2d = pose_2d.reshape(-1, 2)  # Shape: (N, 2)
            pose_2d[~valid_3d] = -1


            img_pil = self.visualize_2d_projection(frame_data['images'][cam_key], pose_2d, approval_status)
            camera_images.append(img_pil)
        
        return camera_images



    def visualize_3d_pose(self, frame_data, elev=20, azim=45, input_zup=True):
        """Create 3D visualization of the pose using matplotlib."""
        if frame_data is None:
            return None
            
        triangulated_3d = frame_data['refined_3d'].copy()
        approval_status = frame_data['approval_status'].copy()
        
        # Extract 3D coordinates and validity
        points_3d = triangulated_3d[..., :3]
        if not input_zup:
            points_3d = points_3d[..., [0, 2, 1]]  # Convert to Z-up if needed
            points_3d[..., 2] = -points_3d[..., 2]
            

        valid_3d = (triangulated_3d[..., 3] > 1e-4) & (approval_status != self.FLAG_DISCARDED)


        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw skeleton connections
        for connection in self.vit_skeleton:
            pt1_idx, pt2_idx = connection
            
            if valid_3d[pt1_idx] and valid_3d[pt2_idx]:
                pt1 = points_3d[pt1_idx]
                pt2 = points_3d[pt2_idx]
                
                ax.plot3D([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                         'gray', linewidth=4, alpha=0.6)
        
        # Draw keypoints
        for i, point in enumerate(points_3d):
            if not valid_3d[i]:
                continue
                
            # Use green for approved joints, otherwise use original keypoint colors
            if approval_status[i] == self.FLAG_APPROVED:
                color = '#00ff00'
                marker_size = 80
            else:
                # Convert BGR to RGB and normalize
                bgr_color = self.vit_keypoint_colors[i % len(self.vit_keypoint_colors)]
                rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)
                color = rgb_color
                marker_size = 80
            
            ax.scatter(point[0], point[1], point[2], 
                      c=[color], s=marker_size, alpha=0.8)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Pose - Frame {frame_data["frame_id"]} (elev={elev}°, azim={azim}°)')
        
        # Set equal aspect ratio
        max_range = np.array([points_3d[valid_3d, 0].max() - points_3d[valid_3d, 0].min(),
                             points_3d[valid_3d, 1].max() - points_3d[valid_3d, 1].min(),
                             points_3d[valid_3d, 2].max() - points_3d[valid_3d, 2].min()]).max() / 2.0
        
        mid_x = (points_3d[valid_3d, 0].max() + points_3d[valid_3d, 0].min()) * 0.5
        mid_y = (points_3d[valid_3d, 1].max() + points_3d[valid_3d, 1].min()) * 0.5
        mid_z = (points_3d[valid_3d, 2].max() + points_3d[valid_3d, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Convert to PIL Image
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        
        plt.close(fig)
        
        return Image.fromarray(img_array)
    


    
    def update_frame_display(self, frame_data):
        """Update display with given frame data."""
        camera_images = self.create_frame_visualization(frame_data)
        pose_3d_img = self.visualize_3d_pose(frame_data, input_zup=self.input_zup)
        
        return [pose_3d_img] + camera_images
    
    def approve_joint(self, frame_data, selected_joint):
        """Approve a specific joint."""
        joint_idx = list(self.vit_joints_name).index(selected_joint)
        
        
        refined_3d = frame_data['refined_3d']
        if refined_3d[joint_idx, 3] <= 1e-4:
            msg = f"<div style='color: red; font-weight: bold;'>❌ WARNING: Cannot approve joint '{selected_joint}' with unavailable 3D</div>"
            updated_displays = self.update_frame_display(frame_data)
            return updated_displays + [frame_data, msg, gr.update()]  # No joint update on error
        
        # If valid, proceed with approval
        frame_data['approval_status'][joint_idx] = self.FLAG_APPROVED
        
        # Update the stored data
        frame_key = str(frame_data['frame_id'])
        self.json_data[frame_key]['approval_status'][joint_idx] = self.FLAG_APPROVED
        
        # Get next joint
        next_joint_idx = (joint_idx + 1) % len(self.vit_joints_name)
        next_joint = self.vit_joints_name[next_joint_idx]
        
        # Update displays
        updated_displays = self.update_frame_display(frame_data)
        msg = f"<div style='color: green; font-weight: bold;'>✅ Joint '{selected_joint}' approved</div>"
        return updated_displays + [frame_data, msg, gr.update(value=next_joint)]
    
    def discard_joint(self, frame_data, selected_joint):
        """Discard a specific joint."""
        joint_idx = list(self.vit_joints_name).index(selected_joint)
        frame_data['approval_status'][joint_idx] = self.FLAG_DISCARDED
        
        # Update the stored data
        frame_key = str(frame_data['frame_id'])
        self.json_data[frame_key]['approval_status'][joint_idx] = self.FLAG_DISCARDED
        
        # Get next joint
        next_joint_idx = (joint_idx + 1) % len(self.vit_joints_name)
        next_joint = self.vit_joints_name[next_joint_idx]
        
        # Update displays
        updated_displays = self.update_frame_display(frame_data)
        msg = f"<div style='color: orange; font-weight: bold;'>🗑️ Joint '{selected_joint}' discarded</div>"
        return updated_displays + [frame_data, msg, gr.update(value=next_joint)]
    
    

    def create_approval_interface(self):
        """Create Gradio interface for 3D pose approval."""
        
        with gr.Blocks(title="3D Pose Approval Tool", css="""
            .camera-container {
                max-height: 70vh;
                overflow-y: auto;
            }
            .status-container {
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
            }
        """) as interface:
            
            gr.Markdown("# 3D Joint Annotation Approval Tool")
            gr.Markdown("Review and approve automatically generated 3D pose triangulation results.")
            
            with gr.Row():
                # Left column for controls
                with gr.Column(scale=1):
                    # Instructions
                    gr.Markdown("""
                    ### Instructions
                                
                    **Actions:**
                    - **Approve**: Accept the 3D pose of joint being selected
                    - **Discard**: Reject the 3D pose of joint being selected
                    - Navigate between frames to review all poses
                    - Save results when finished reviewing
                    """)

                    # Navigation buttons
                    gr.Markdown("### Frame Navigation")

                    frame_slider = gr.Slider(
                        minimum=min(self.available_frames),
                        maximum=max(self.available_frames),
                        step=1,
                        value=self.available_frames[0],
                        label="Frame ID",
                        interactive=True
                    )

                    
                    # Approval controls
                    gr.Markdown("### Approval Controls")
                    
                    # Joint selection dropdown
                    joint_dropdown = gr.Dropdown(
                        choices=list(self.vit_joints_name),
                        value=self.vit_joints_name[0],
                        label="Select Joint to Approve/Discard",
                        interactive=True
                    )
                    
                    # Joint navigation buttons
                    with gr.Row():
                        prev_joint_btn = gr.Button("⬅️ Previous Joint", variant="secondary", scale=1)
                        next_joint_btn = gr.Button("➡️ Next Joint", variant="secondary", scale=1)
                    
                    with gr.Row():
                        approve_btn = gr.Button("✅ Approve Joint", variant="primary", scale=1)
                        discard_btn = gr.Button("❌ Discard Joint", variant="secondary", scale=1)
                    
                    # Save results
                    save_btn = gr.Button("💾 Save Results", variant="primary", size="lg")
                    msg_status = gr.HTML(value="", label="Save Status")
                    
                # Hidden state to store current frame data
                frame_data_state = gr.State(self.get_frame_data(self.available_frames[0]))
                
                # Right column for camera views
                with gr.Column(scale=3):
                    # 3D visualization
                    gr.Markdown("### 3D Pose Visualization")
                    
                    # Viewpoint controls
                    with gr.Row():
                        viewpoint_front = gr.Button("Front View", variant="secondary", scale=1)
                        viewpoint_side = gr.Button("Side View", variant="secondary", scale=1)
                        viewpoint_top = gr.Button("Top View", variant="secondary", scale=1)
                        viewpoint_iso = gr.Button("Isometric", variant="secondary", scale=1)
                    
                    pose_3d_display = gr.Image(
                        label="3D Pose",
                        type="pil",
                        interactive=False,
                        height=400,
                        width=600
                    )
                    
                    gr.Markdown("### Camera Views with Projections")
                    
                    # Create camera image displays in a 2x2 grid (or Nx2 for more cameras)
                    camera_displays = []
                    num_cameras = len(self.cam_keys)
                    
                    # Create rows with 2 cameras each
                    n_col = 3
                    for crow in range(0, num_cameras, n_col):
                        with gr.Row():
                            for ccol in range(0, n_col):
                                if crow + ccol >= num_cameras:
                                    with gr.Column(scale=1):
                                        pass
                                    continue

                                with gr.Column(scale=1):
                                    img_display = gr.Image(label=f"{self.cam_keys[crow+ccol].upper()}",
                                            type="pil",
                                            interactive=False,
                                            height=400,
                                            width=400)
                                    camera_displays.append(img_display)
                            
                            
            
            
            # Frame update function that fetches data and updates state
            def update_frame_and_display(frame_id):
                frame_data = self.get_frame_data(frame_id)
                display_result = self.update_frame_display(frame_data)
                return display_result + [frame_data]
            
            frame_slider.change(
                fn=update_frame_and_display,
                inputs=[frame_slider],
                outputs=[pose_3d_display] + camera_displays + [frame_data_state]
            )
            
            
            # Viewpoint controls
            def change_viewpoint_front(frame_data):
                return self.visualize_3d_pose(frame_data, elev=10, azim=0, input_zup=self.input_zup)  # Front view
            
            def change_viewpoint_side(frame_data):
                return self.visualize_3d_pose(frame_data, elev=10, azim=90, input_zup=self.input_zup)  # Side view
            
            def change_viewpoint_top(frame_data):
                return self.visualize_3d_pose(frame_data, elev=60, azim=45, input_zup=self.input_zup)  # Top view
            
            def change_viewpoint_iso(frame_data):
                return self.visualize_3d_pose(frame_data, elev=20, azim=45, input_zup=self.input_zup)  # Isometric view
            
            viewpoint_front.click(
                fn=change_viewpoint_front,
                inputs=[frame_data_state],
                outputs=[pose_3d_display]
            )
            
            viewpoint_side.click(
                fn=change_viewpoint_side,
                inputs=[frame_data_state],
                outputs=[pose_3d_display]
            )
            
            viewpoint_top.click(
                fn=change_viewpoint_top,
                inputs=[frame_data_state],
                outputs=[pose_3d_display]
            )
            
            viewpoint_iso.click(
                fn=change_viewpoint_iso,
                inputs=[frame_data_state],
                outputs=[pose_3d_display]
            )
            
            # Joint approval/discard handlers
            approve_btn.click(
                fn=self.approve_joint,
                inputs=[frame_data_state, joint_dropdown],
                outputs=[pose_3d_display] + camera_displays + [frame_data_state, msg_status, joint_dropdown]
            )
            
            discard_btn.click(
                fn=self.discard_joint,
                inputs=[frame_data_state, joint_dropdown],
                outputs=[pose_3d_display] + camera_displays + [frame_data_state, msg_status, joint_dropdown]
            )
            
            # Joint navigation handlers
            def get_previous_joint(current_joint):
                current_idx = list(self.vit_joints_name).index(current_joint)
                prev_idx = (current_idx - 1) % len(self.vit_joints_name)
                return self.vit_joints_name[prev_idx]
            
            def get_next_joint(current_joint):
                current_idx = list(self.vit_joints_name).index(current_joint)
                next_idx = (current_idx + 1) % len(self.vit_joints_name)
                return self.vit_joints_name[next_idx]
            
            prev_joint_btn.click(
                fn=get_previous_joint,
                inputs=[joint_dropdown],
                outputs=[joint_dropdown]
            )
            
            next_joint_btn.click(
                fn=get_next_joint,
                inputs=[joint_dropdown],
                outputs=[joint_dropdown]
            )
            
            # Save button handler
            save_btn.click(
                fn=self.save_annotations,
                inputs=[],
                outputs=[msg_status]
            )
            
            # Initialize display
            interface.load(
                fn=update_frame_and_display,
                inputs=[frame_slider],
                outputs=[pose_3d_display] + camera_displays + [frame_data_state]
            )
            
        return interface

def main():
    """Main function to launch the 3D pose approval tool."""
    try:
        # Parse command line arguments
        args = get_args_parser()
        
        # Initialize approval tool
        approval_tool = Pose3DApprovalTool(args)
        
        # Create and launch interface
        interface = approval_tool.create_approval_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            debug=True
        )
        
    except KeyboardInterrupt:
        print("\n⚠️  Received interrupt signal, saving results...")
        if 'approval_tool' in locals():
            save_result = approval_tool.save_annotations()
            print(f"Auto-save result: {save_result}")
        print("✅ 3D pose approval tool closed safely.")
        
    except Exception as e:
        print(f"Error launching 3D pose approval tool: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()