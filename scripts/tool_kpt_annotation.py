#!/usr/bin/env python3
"""
Gradio-based Human Body Pose Annotator

This tool provides a web interface for annotating, editing, and validating 
human body poses using pre-computed VitPose detections and triangulated 3D poses.

Features:
- Load and visualize images from LMDB dataset
- Display pre-computed VitPose 2D detections
- Interactive pose editing and annotation
- Export corrected annotations

Prerequisites:
- Results from run_vitpose_triangulation.py saved as JSON files
- EgoExo dataset with LMDB image storage
- Camera calibration parameters
"""
import glob
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

# Add parent directory to path for imports
sys.path.append('..')
from dataset.recording import Recording
from utils.logger import ColorLogger
from utils.constants import PATH_ASSETS, VIT_JOINTS_NAME, VIT_SKELETON, VIT_KEYPOINT_COLORS, PATH_ASSETS_VIDEOS, PATH_ASSETS_KPT2D_AUTO, PATH_ASSETS_KPT2D_MANUAL, PATH_ASSETS_KPT3D
from utils.io_utils import load_vitpose_json, NumpyEncoder

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gradio-based Human Body Pose Annotator', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--recording_tag', type=str, default='sync_9122', help='recording name')
    parser.add_argument('--path_camera',type=str, default='', help='Path to the camera meta json file, if empty, use calibration.json under --recording_tag')

    parser.add_argument('--threshold_det_conf', type=float, default=0.5, help='threshold for 2D joint scores')
    parser.add_argument('--threshold_reproj_err', type=float, default=50., help='threshold for 2D reprojection scores')
    parser.add_argument('--min_cam', type=int, default=4, help='threshold for qualified camera detections')
    
    parser.add_argument('--port', type=int, default=7860, help='Port for Gradio interface')
    parser.add_argument('--share', action='store_true', help='Create public sharing link')

    return parser.parse_args()

  

class PoseAnnotator:
    def __init__(self, args):
        """Initialize the pose annotator with required data and models."""
        self.setup_dataset(args)
        self.load_existing_json()

        self.min_cam = args.min_cam
        self.threshold_det_conf = args.threshold_det_conf
        self.threshold_reproj_err = args.threshold_reproj_err


        # Set up output path for saving annotations
        list_paths = glob.glob(os.path.join(PATH_ASSETS_KPT2D_MANUAL, f"{self.recording_name}*.json"))
        cnt = len(list_paths)
        self.output_path = os.path.join(PATH_ASSETS_KPT2D_MANUAL, f"{self.recording_name}_{cnt}.json")
        self.logger.info(f"Output annotation path: {self.output_path}")

        self.vit_joints_name = VIT_JOINTS_NAME
        self.vit_skeleton = VIT_SKELETON
        self.vit_keypoint_colors = VIT_KEYPOINT_COLORS

        self.NEED_ANNOTATION = 'NEED_ANNOTATION'
        self.MANUALLY_ANNOTATED = 'MANUALLY_ANNOTATED'
        self.AUTO_ANNOT_APPROVED = 'AUTO_ANNOT_APPROVED'
        self.MANUALLY_PASSED = 'MANUALLY_PASSED'
        self.NO_ACTION_REQUIRED = 'NO_ACTION_REQUIRED'
        
        # Manual flags integer values
        self.FLAG_AUTO = 0      # Original VitPose detection (no manual intervention)
        self.FLAG_MANUAL = 1    # Manually clicked/annotated by user
        self.FLAG_APPROVED = 2  # User approved VitPose detection as good
        self.FLAG_PASSED = 3    # User manually passed joint (e.g., occlusion)

        self.MODE_ANNOTATION_MODE = "Annotation Mode"
        self.MODE_2D_DETECTION = "2D Detection"
        self.MODE_2D_DETECTION_REPROJ = "2D Reprojection"
        self.MODE_ORIGINAL_IMAGE = "Original Image"


        
        self.in_zoom = True #State whether in zoom mode
        self.zoom_regions = {cam_key: None for cam_key in self.cam_keys}  # Store zoom region for each camera
        self.zoom_corner1 = {cam_key: None for cam_key in self.cam_keys}  # Temprorarily store first corner (top-left), should be reset when zoom_regions is set

        
    def setup_dataset(self, args):
        load_undistort_images  = 'undistort' in args.recording_tag
        self.recording_name = args.recording_tag.split('/')[-2]

        path_root_dir = os.path.join(PATH_ASSETS_VIDEOS, args.recording_tag)
        path_camera = args.path_camera if args.path_camera!='' else os.path.join(path_root_dir, 'calibration.json')
        assert os.path.exists(path_camera), f"Camera meta file does not exist: {path_camera}"

        self.log_dir = os.path.join(PATH_ASSETS_KPT2D_MANUAL, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = ColorLogger(self.log_dir, log_name=f'{self.recording_name}.txt')


        # Load EgoExo dataset
        self.dataset =  Recording(root_dir = path_root_dir, 
                        path_cam_meta = path_camera, 
                        logger = self.logger, 
                        recording_name = self.recording_name,
                        load_undistort_images = load_undistort_images)
        
        self.logger.info("Dataset initialized successfully")
        self.image_hw = self.dataset.image_hw
        self.cam_params = self.dataset.get_camera_params()
        self.cam_keys = list(self.cam_params.keys())
        

    # load existing annotation
    def load_existing_json(self):
        """Load pre-computed triangulation results from JSON files."""
        results_path =os.path.join(PATH_ASSETS_KPT3D, f"{self.recording_name}_auto")

        self.json_data = load_vitpose_json(results_path)
        self.available_frames = sorted([int(key) for key in self.json_data.keys()])

    
    def save_annotations(self):
        """Save current annotations to the output file."""
        import datetime
        
        box_style = "border: 1px solid #ddd; border-radius: 8px; padding: 12px; background-color: #f9f9f9; font-family: Arial, sans-serif;"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            # Create clean annotation data with only manual annotations
            annotation_data = {}
            for frame_key, frame_data in self.json_data.items():                
                # Save manual_2d if it exists and is not empty
                if 'manual_2d' in frame_data and len(frame_data['manual_2d']) > 0:
                    # check sum of manual_flags
                    to_save = False
                    for cam_key, flags in frame_data['manual_flags'].items():
                        if np.sum(flags) > 0:
                            to_save = True
                            break
                    if not to_save:
                        continue

                    annotation_data[frame_key]={}
                    for key in ['manual_2d']:
                        annotation_data[frame_key][key] = frame_data[key]
                    for key in ['manual_flags', 'need_annot_flag']:
                        to_save = {}
                        for cam_key, flags in frame_data[key].items():
                            to_save[cam_key] = np.array(flags, dtype=int).flatten().tolist()
                        annotation_data[frame_key][key] = to_save
            
            # Save only annotation data with proper formatting
            with open(self.output_path, 'w') as f:
                json.dump(annotation_data, f, separators=(',', ':'), cls=NumpyEncoder)
            
            frame_count = len(annotation_data)
            content = f"<h3 style='color: #28A745; margin: 0; margin-bottom: 8px;'>✅ SUCCESS</h3><p>Annotations saved successfully<br><strong>Frames:</strong> {frame_count}<br><strong>Time:</strong> {timestamp}</p>"
            return f"<div style='{box_style}'><strong>Save Status</strong><br><br>{content}</div>"
        except Exception as e:
            content = f"<h3 style='color: #DC3545; margin: 0; margin-bottom: 8px;'>❌ ERROR</h3><p>Failed to save annotations<br><strong>Error:</strong> {str(e)}<br><strong>Time:</strong> {timestamp}</p>"
            return f"<div style='{box_style}'><strong>Save Status</strong><br><br>{content}</div>"

    ###############################
    # start processing frames
    ###############################

    def get_frame_data(self, frame_id):
        """
        Get all data for a specific frame.
        """
        if frame_id not in self.available_frames:
            return None
            
        frame_key = str(frame_id)

        if frame_key not in self.json_data:
            frame_data = {'vitpose_2d': {}, 'reproj_err': {}, 
                          'triangulated_3d': np.zeros((len(self.vit_joints_name), 4), dtype=np.float32)}
        else:
            frame_data = self.json_data[frame_key]

        if 'manual_2d' not in frame_data:
            frame_data['manual_2d'] = {cam_key: frame_data['vitpose_2d'][cam_key].copy() if 'vitpose_2d' in frame_data and cam_key in frame_data['vitpose_2d'] 
                                       else np.zeros((len(self.vit_joints_name), 3), dtype=np.float32) - 1.0 for cam_key in self.cam_keys}
            
        
        if 'manual_flags' not in frame_data:
            frame_data['manual_flags'] = {cam_key: np.zeros(len(self.vit_joints_name), dtype=int) for cam_key in self.cam_keys}
        
        if 'need_annot_flag' not in frame_data:
            frame_data['need_annot_flag'] = {cam_key: np.zeros(len(self.vit_joints_name), dtype=bool) for cam_key in self.cam_keys}

            # Check each joint across all cameras
            for joint_idx in range(len(self.vit_joints_name)):
                need_annotation_per_camera = np.zeros(len(self.cam_keys), dtype=bool)
                safe_reproj_error = True
                
                # First pass: count confident detections for this joint
                for cam_idx, cam_key in enumerate(self.cam_keys):
                    has_pose = cam_key in frame_data['vitpose_2d']
                    
                    if has_pose:
                        cjoint_conf = frame_data['vitpose_2d'][cam_key][joint_idx,2]
                        creproj_err = frame_data['reproj_err'][cam_key][joint_idx]

                        if cjoint_conf >= self.threshold_det_conf and creproj_err > self.threshold_reproj_err:
                            safe_reproj_error = False

                        if cjoint_conf >= self.threshold_det_conf and creproj_err >0 and creproj_err <= self.threshold_reproj_err:
                            continue
                    
                    # cannot pass test, so we need annotation for this joint in this camera
                    need_annotation_per_camera[cam_idx] = True
                        
                
                # if not safe_reproj_error, then we need annotation for all cameras
                if not safe_reproj_error:
                    need_annotation_per_camera.fill(True)
                
                # Check if we have sufficient safe cameras and safe reprojection errors
                if np.sum(~need_annotation_per_camera) >= self.min_cam:
                    continue
                
                # Set the need_annot_flag flags for this joint
                for cam_idx, need_annotation in enumerate(need_annotation_per_camera):
                    cam_key = self.cam_keys[cam_idx]
                    self.json_data[frame_key]['need_annot_flag'][cam_key][joint_idx] = need_annotation

        
        frame_data = copy.deepcopy(self.json_data[frame_key]) 
        # Load images for all cameras
        images = {}
        for cam_key in self.cam_keys:
            img_bgr = self.dataset.load_bgr_image_and_undistort(f"{cam_key}/frame_{frame_id:04d}.png")
            images[cam_key] = img_bgr

        frame_data['images'] = images
        frame_data['frame_id'] = frame_id

        return frame_data
        
    # start to handle visualisation
    def visualize_2d_pose(self, image, points2d, frame_data=None, cam_key=None, feedback_annotated=True):
        """
        Visualize 2D pose on image.
        """

        img_vis = image.copy()
        
            
        # Draw pose keypoints and skeleton
        if len(points2d) == 0:
            return img_vis


        # Draw skeleton connections in green
        color_auto = (255,255,255)
        color_manual = (0, 255, 0)
        color_need_annot= (0, 0, 255)  

        points2d_vis = points2d[:,:2] if points2d.shape[-1] == 3 else points2d.copy()
        
        # get zoom info 
        point_scale = self.zoom_regions[cam_key][4] if self.zoom_regions[cam_key] is not None else 1.
        point_size = max(2, int(16/point_scale))


        for connection in self.vit_skeleton:
            pt1_idx, pt2_idx = connection
            
            pt1 = tuple(points2d_vis[pt1_idx].astype(int))
            pt2 = tuple(points2d_vis[pt2_idx].astype(int))
            
            if pt1[0]> -1e-4 and pt1[1]> -1e-4 and pt2[0]> -1e-4 and pt2[1]> -1e-4:           
                cv2.line(img_vis, pt1, pt2, (128,128,128), min(5,point_size))
        

        # Draw keypoints
        for i, point in enumerate(points2d_vis):
            center = tuple(point.astype(int))
            if center[0] < 0 or center[1] < 0:
                continue
            annotation_info = self.get_joint_annotation_status(frame_data, cam_key, i, feedback_annotated)
            if annotation_info == self.MANUALLY_ANNOTATED:
                cv2.circle(img_vis, center, point_size, color_manual, -1)
            else:
                # Use keypoint-specific color for non-manually annotated points
                keypoint_color = self.vit_keypoint_colors[i % len(self.vit_keypoint_colors)]
                cv2.circle(img_vis, center, point_size, keypoint_color, -1)
        
        return img_vis
    

    def create_image_pil_for_vis(self, image, frame_data, cam_key, display_mode):                
        # Create visualization based on display mode
        pose_2d = None
        if display_mode == self.MODE_2D_DETECTION and cam_key in frame_data['vitpose_2d']:
            pose_2d = frame_data['vitpose_2d'][cam_key]

        elif display_mode == self.MODE_2D_DETECTION_REPROJ:
            # 3D Reprojection (Auto Triangulation) visualization                    
            triangulated_3d = frame_data['triangulated_3d'][...,:3]
            valid_3d = frame_data['triangulated_3d'][..., 3] > 1e-4
                
            cam_param = self.cam_params[cam_key]
            K = cam_param['K']
            rvec = cam_param['rvec']
            tvec = cam_param['tvec']
            
            # Use OpenCV's projectPoints function
            pose_2d, _ = cv2.projectPoints(triangulated_3d.reshape(-1, 1, 3), rvec, tvec, K, None)
            pose_2d = pose_2d.reshape(-1, 2)  # Shape: (N, 2)
            pose_2d[~valid_3d] = -1

        elif display_mode == self.MODE_ANNOTATION_MODE:
            # Show current manual annotations or original detections with annotation status
            pose_2d = frame_data['manual_2d'][cam_key] 

        if pose_2d is not None:
            img_display = self.visualize_2d_pose(image.copy(), pose_2d, frame_data=frame_data, cam_key=cam_key, feedback_annotated=(display_mode == self.MODE_ANNOTATION_MODE))
        else:
            img_display = image.copy()
            
        img_pil = Image.fromarray(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        
        # Apply zoom if enabled        
        """Apply zoom region to image if zoom is enabled and region is set."""
        if self.zoom_regions[cam_key] is None:
            return img_pil
            
        # Get zoom region coordinates
        x1, y1, x2, y2, scale_factor, x_offset, y_offset = self.zoom_regions[cam_key]
                            
        # Crop the image to zoom region
        cropped = img_pil.crop((x1, y1, x2, y2))
        crop_width, crop_height = cropped.size
        
        # Scale the cropped region while preserving aspect ratio
        new_width = int(crop_width * scale_factor)
        new_height = int(crop_height * scale_factor)
        scaled = cropped.resize((new_width, new_height), Image.LANCZOS)
        
        target_height, target_width = self.image_hw
        result = Image.new('RGB', (target_width, target_height), (255, 255, 255))  # Gray background
        
        result.paste(scaled, (x_offset, y_offset))

        return result
       
    # handle joint annotation status
    def get_joint_annotation_status(self, frame_data, cam_key, joint_idx, feedback_annotated):
        #Get annotation status for a specific joint.
        
        if feedback_annotated and cam_key in frame_data['manual_flags']:
            flag_value = frame_data['manual_flags'][cam_key][joint_idx]
            if flag_value == self.FLAG_MANUAL:
                return self.MANUALLY_ANNOTATED
            elif flag_value == self.FLAG_APPROVED:
                return self.AUTO_ANNOT_APPROVED
            elif flag_value == self.FLAG_PASSED:
                return self.MANUALLY_PASSED
        
        if cam_key in frame_data['need_annot_flag'] and frame_data['need_annot_flag'][cam_key][joint_idx]:
            return self.NEED_ANNOTATION
        else:
            return self.NO_ACTION_REQUIRED
        
        
    def get_frame_camera_annotation_status_strings(self, frame_data, cam_key,display_mode):
        feedback_annotated = display_mode in [self.MODE_ANNOTATION_MODE, self.MODE_ORIGINAL_IMAGE]

        info_data = []
        info_data.append((f"=== {cam_key.upper()} ===\n", None))
        if feedback_annotated:
            info_data.append(("Joints \t (x, y, conf):\n", None))
        else:
            info_data.append(("Joints \t (x, y, conf, reproj_err):\n", None))


        for joint_idx, joint_name in enumerate(self.vit_joints_name):
            status = self.get_joint_annotation_status(frame_data, cam_key, joint_idx, feedback_annotated=feedback_annotated)

            
            if status in [self.MANUALLY_ANNOTATED, self.AUTO_ANNOT_APPROVED, self.MANUALLY_PASSED]:
                x, y, conf = frame_data['manual_2d'][cam_key][joint_idx]
                joint_line = f"{joint_name:12}: ({x:6.1f}, {y:6.1f}, {conf:.2f})"

                if status == self.MANUALLY_PASSED:
                    status_str = "Annot. Pass"
                elif status == self.AUTO_ANNOT_APPROVED:
                    status_str = "Auto Approv."
                else:  # MANUALLY_ANNOTATED
                    status_str = "Annot. Done"

                info_data.append((joint_line, status_str))
                                
            else:
                if cam_key in frame_data['vitpose_2d']:
                    pose_2d = frame_data['vitpose_2d'][cam_key]  # Shape: (17, 3)
                    reproj_err = frame_data['reproj_err'].get(cam_key, np.full(17, -1))
                    
                    x, y, conf = pose_2d[joint_idx]
                    if feedback_annotated:
                        joint_line = f"{joint_name:12}: ({x:6.1f}, {y:6.1f}, {conf:.2f})"
                    else:
                        err_val = reproj_err[joint_idx] #if reproj_err[joint_idx] != -1 else float('nan')
                        err_str = f"{err_val:.1f}" if err_val>-0.1 else 'N/A'
                        joint_line = f"{joint_name:12}: ({x:6.1f}, {y:6.1f}, {conf:.2f}, {err_str})"
                else:
                    joint_line = f"{joint_name:12}: No data"

                    
                if status == self.NEED_ANNOTATION:
                    info_data.append((joint_line, "Annot. Needed"))  # Highlight as error
                else:
                    info_data.append((joint_line, None))
                            
            info_data.append(("\n", None))
        
        return info_data
    

    # Update annotation status
    def update_annotation_status(self, selected_camera, selected_joint, confidence_score, display_mode, annotate_flag=None, interact_camera=None):      
        # Create styled box content
        box_style = "border: 1px solid #ddd; border-radius: 8px; padding: 12px; background-color: #f9f9f9; font-family: Arial, sans-serif;"
        mode_status = "<h2 style='color: #FF6B35; margin: 0; margin-bottom: 8px;'>🔍 ZOOM MODE</h2>" if self.in_zoom \
            else "<h2 style='color: #28A745; margin: 0; margin-bottom: 8px;'>📌 READY FOR ANNOTATION</h2>"

        if display_mode not in [self.MODE_ANNOTATION_MODE]:
            mode_status = ""
            content = "<h2 style='color: #000000; margin: 0; margin-bottom: 8px;'>🔒 Switch to 'Annotation Mode' to edit joint positions</h2>"

        # quick handle approove auto and passed joint events
        elif annotate_flag == self.FLAG_APPROVED:
            content = f"<p>✅ Approved Auto Annotation: {selected_joint.upper()} for {selected_camera.upper()}</p>"
        elif annotate_flag ==self.FLAG_PASSED:
            content = f"<p>✅ Passed Joint: {selected_joint.upper()} for {selected_camera.upper()}</p>"
        
        # If interact_camera is clicked, show a warning if it doesn't match selected_camera
        elif interact_camera is not None and interact_camera != selected_camera:
            content = f"<p>❌ Please click on {selected_camera.upper()} image (you clicked on {interact_camera.upper()})</p>"

        # If in zoom mode, show zoom instructions or current zoom status
        elif self.in_zoom:        
            if self.zoom_regions[selected_camera] is not None:
                content = f"<p>{selected_camera.upper()}: Click [Edit Annotation] to annotate; or click [RESET ZOOM] to reset zoom-in.</p>"
            elif self.zoom_corner1[selected_camera] is None:
                content = f"<p>Click on the {selected_camera.upper()} image to indicate top-left corner.<br><br> Or click [Edit Annotation] to annotate</p>"
            else:
                click_x, click_y = self.zoom_corner1[selected_camera]
                content = f"<p>{selected_camera.upper()}: Top-left corner set at ({click_x}, {click_y}).<br><br> Click again to set bottom-right corner. <br><br> Click [RESET ZOOM] to undo.</p>"
        
        # else is in annotation mode
        else:
            if annotate_flag is None:
                content = f"<p>Click on the <strong>{selected_camera.upper()}</strong> image to annotate the <strong>{selected_joint.upper()}</strong>, with conf=<strong>{confidence_score:.1f}</strong>)</p>"
            else: # annotate_flag == self.FLAG_MANUAL:
                content = f"<p>✅ Annotated: {selected_joint.upper()} for {selected_camera.upper()}, with conf={confidence_score:.1f}</p>"

        return f"<div style='{box_style}'><strong>Annotation Status</strong><br><br>{mode_status}{content}</div>"

    def transform_zoom_coords_to_original(self, cam_key, click_x, click_y):
        """Transform coordinates from zoomed image back to original image coordinates."""

        # No zoom applied, return original coordinates
        if self.zoom_regions[cam_key] is None:
            return click_x, click_y
            
        # Get zoom information: (x1, y1, x2, y2, scale_factor, x_offset, y_offset)
        x1, y1, x2, y2, scale_factor, x_offset, y_offset = self.zoom_regions[cam_key]
        
        # Convert click coordinates from zoomed display back to original image
        # 1. Subtract the centering offset
        original_x_in_scaled = click_x - x_offset
        original_y_in_scaled = click_y - y_offset
        
        # 2. Scale back down to crop size
        crop_x = original_x_in_scaled / scale_factor
        crop_y = original_y_in_scaled / scale_factor
        
        # 3. Add back the crop region offset to get original image coordinates
        original_x = crop_x + x1
        original_y = crop_y + y1
        
        return original_x, original_y


    # handle manual annotaiton
    def handle_annotation(self, cam_key, selected_joint, frame_data, annotate_flag, annotate_xyc):                            
        # Get joint index
        joint_idx = list(self.vit_joints_name).index(selected_joint)
                
        # Update the joint coordinates (x, y, user_confidence)
        confidence_score = 0.
        if annotate_flag == self.FLAG_MANUAL:
            x, y, confidence_score = annotate_xyc
            x, y = self.transform_zoom_coords_to_original(cam_key, x, y)
            frame_data['manual_2d'][cam_key][joint_idx] = [x, y, confidence_score]
        elif annotate_flag == self.FLAG_PASSED:
            # Mark as passed with invalid coordinates
            frame_data['manual_2d'][cam_key][joint_idx] = [-1, -1, 0]
        elif annotate_flag == self.FLAG_APPROVED:
            x, y, confidence_score = frame_data['vitpose_2d'][cam_key][joint_idx]
            frame_data['manual_2d'][cam_key][joint_idx] = [x, y, confidence_score]

        frame_data['manual_flags'][cam_key][joint_idx] = annotate_flag
        
        # Save back to triangulation_data to persist changes
        frame_key = str(frame_data['frame_id'])                    
        self.json_data[frame_key]['manual_2d'][cam_key] = frame_data['manual_2d'][cam_key].copy()
        self.json_data[frame_key]['manual_flags'][cam_key] = frame_data['manual_flags'][cam_key].copy()
        
        # Efficiently update only the annotated camera using instance variables
        camera_display_images = [gr.skip()] * len(self.cam_keys)
        camera_meta_text = [gr.skip()] * len(self.cam_keys)


        cam_idx = self.cam_keys.index(cam_key)
        image = frame_data['images'][cam_key]
        img_pil = self.create_image_pil_for_vis(image, frame_data, cam_key, self.MODE_ANNOTATION_MODE)
        camera_display_images[cam_idx] = img_pil
        
        
        # Generate updated info text for this camera only
        info_data = self.get_frame_camera_annotation_status_strings(frame_data, cam_key, self.MODE_ANNOTATION_MODE) 
        camera_meta_text[cam_idx] = info_data
        
        status_msg = self.update_annotation_status(cam_key, selected_joint, confidence_score, self.MODE_ANNOTATION_MODE, annotate_flag=annotate_flag)
        
        # Auto-advance to next joint for manual annotations
        next_joint = self.get_next_joint(selected_joint)
        if annotate_flag == self.FLAG_MANUAL:
            return camera_display_images + camera_meta_text + [status_msg, gr.update(value=next_joint)]
        else:
            # For approve/pass, return skip for joint selection (those functions handle it themselves)
            return camera_display_images + camera_meta_text + [status_msg, gr.update(value=next_joint)]
    

    # handle approving auto annotation
    def toggle_approve_auto(self, selected_camera, selected_joint, frame_data, display_mode):
        """Mark the selected joint's VitPose detection as approved."""
        try:
            if display_mode != self.MODE_ANNOTATION_MODE:
                return [gr.skip()] * (len(self.cam_keys) * 2 + 2)  # +2 for status and joint dropdown
            
            return self.handle_annotation(selected_camera, selected_joint, frame_data, self.FLAG_APPROVED, None)
            
        except Exception as e:
            status_msg = "Error"+str(e)
            return [gr.skip()] * (len(self.cam_keys) * 2) + [status_msg, gr.skip()]
    
    
    # pass joint annotation
    def toggle_pass_joint(self, selected_camera, selected_joint, frame_data, display_mode):
        """Mark the selected joint as manually passed (e.g., due to occlusion)."""
        try:
            if display_mode != self.MODE_ANNOTATION_MODE:
                return [gr.skip()] * (len(self.cam_keys) * 2 + 2)  # +2 for status and joint dropdown
            
            return self.handle_annotation(selected_camera, selected_joint, frame_data, self.FLAG_PASSED, None)
            
        except Exception as e:
            status_msg = str(e)
            return [gr.skip()] * (len(self.cam_keys) * 2) + [status_msg, gr.skip()]
                
    
    
    # switch zoom mode
    def toggle_zoom_enter(self, selected_camera, selected_joint, confidence_score):
        """Toggle zoom mode on/off."""
        self.in_zoom = not self.in_zoom

        button_text = "📌 Edit Annotation" if self.in_zoom else "🔍 Edit Zoom"
        status_msg = self.update_annotation_status(selected_camera, selected_joint, confidence_score, self.MODE_ANNOTATION_MODE)

        return gr.update(value=button_text), status_msg
    
    # handle editing for zoom in
    def handle_zoom_editing(self, cam_key, selected_joint, frame_data, annotate_xyc):
        """Handle zoom region selection with two clicks: top-left then bottom-right."""
        
        click_x, click_y = annotate_xyc[:2]
        
        # if already set zoom region, then need to warn users 
        if self.zoom_regions[cam_key] is not None:
            status_msg = self.update_annotation_status(cam_key, selected_joint, annotate_xyc[2], self.MODE_ANNOTATION_MODE)
            return [gr.skip()] * (len(self.cam_keys) * 2) + [status_msg]
        
        
        # First click: Set top-left corner
        if self.zoom_corner1[cam_key] is None:
            self.zoom_corner1[cam_key] = (click_x, click_y)
            status_msg = self.update_annotation_status(cam_key, selected_joint, annotate_xyc[2], self.MODE_ANNOTATION_MODE)
            return [gr.skip()] * (len(self.cam_keys) * 2) + [status_msg]
        
        # Second click: Set bottom-right corner and apply zoom
        else:            
            # Calculate zoom region from two corners
            x1_, y1_ = self.zoom_corner1[cam_key]
            x2_, y2_ = (click_x, click_y)
                                
            # Store zoom region for this camera
            x1, x2 = min(x1_, x2_), max(x1_, x2_)
            y1, y2 = min(y1_, y2_), max(y1_, y2_)
            
            # Ensure coordinates are within image bounds
            img_height, img_width = self.image_hw
            x1 = max(0, min(x1, img_width))
            x2 = max(0, min(x2, img_width))
            y1 = max(0, min(y1, img_height))
            y2 = max(0, min(y2, img_height))
                        
            
            # Calculate scaling factor to fit within original image size while preserving aspect ratio
            crop_width, crop_height = x2 - x1, y2 - y1
            target_height, target_width = self.image_hw
            
            # Calculate scale factor (use the smaller ratio to ensure it fits)
            scale_factor = min(target_width / crop_width, target_height / crop_height)
            
            # Scale the cropped region while preserving aspect ratio
            new_width = int(crop_width * scale_factor)
            new_height = int(crop_height * scale_factor)
            
            # Calculate position to center the scaled image
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            

            self.zoom_regions[cam_key] = (x1, y1, x2, y2, scale_factor, x_offset, y_offset)  # Update zoom region with scaled size
            self.zoom_corner1[cam_key] = None  # Reset for next zoom
            
            
            # Update only this camera's image with zoom applied (while zoom_enabled is still True)
            camera_display_images = [gr.skip()] * len(self.cam_keys)
            cam_idx = self.cam_keys.index(cam_key)
            image = frame_data['images'][cam_key]
            img_pil = self.create_image_pil_for_vis(image, frame_data, cam_key, self.MODE_ANNOTATION_MODE)
            camera_display_images[cam_idx] = img_pil
            
            
            status_msg = self.update_annotation_status(cam_key, selected_joint, annotate_xyc[2], self.MODE_ANNOTATION_MODE)
            return camera_display_images + [gr.skip()]*len(self.cam_keys) + [status_msg]
        

    # reset zoom for selected camera
    def toggle_zoom_reset(self, selected_camera, selected_joint, confidence_score, frame_data):
        """Reset zoom for the selected camera."""

        need_reset_image = self.zoom_regions[selected_camera] is not None
        
        self.zoom_regions[selected_camera] = None
        self.zoom_corner1[selected_camera] = None
            
        
        camera_display_images = [gr.skip()]*len(self.cam_keys)
        if need_reset_image:
            cam_idx = self.cam_keys.index(selected_camera)
            # Regenerate image without zoom for this camera
            image = frame_data['images'][selected_camera]
            
            img_pil = self.create_image_pil_for_vis(image, frame_data, selected_camera, self.MODE_ANNOTATION_MODE)
            camera_display_images[cam_idx] = img_pil
            

        self.in_zoom = False
        button_text, status_msg = self.toggle_zoom_enter(selected_camera, selected_joint, confidence_score)
        return camera_display_images + [button_text, status_msg]



    # Get next/previous joint and camera in the sequence
    def get_next_joint(self, current_joint):
        """Get the next joint in the sequence, wrapping around to the first joint if at the end."""
        current_joint_idx = list(self.vit_joints_name).index(current_joint)
        next_joint_idx = (current_joint_idx + 1) % len(self.vit_joints_name)
        return self.vit_joints_name[next_joint_idx]
    
    def get_previous_joint(self, current_joint):
        """Get the previous joint in the sequence, wrapping around to the last joint if at the beginning."""
        current_joint_idx = list(self.vit_joints_name).index(current_joint)
        prev_joint_idx = (current_joint_idx - 1) % len(self.vit_joints_name)
        return self.vit_joints_name[prev_joint_idx]
    
    def get_next_camera(self, current_camera):
        """Get the next camera in the sequence, wrapping around to the first camera if at the end."""
        current_camera_idx = self.cam_keys.index(current_camera)
        next_camera_idx = (current_camera_idx + 1) % len(self.cam_keys)
        return self.cam_keys[next_camera_idx]
    
    def get_previous_camera(self, current_camera):
        """Get the previous camera in the sequence, wrapping around to the last camera if at the beginning."""
        current_camera_idx = self.cam_keys.index(current_camera)
        prev_camera_idx = (current_camera_idx - 1) % len(self.cam_keys)
        return self.cam_keys[prev_camera_idx]

    ############################################################################
    # Gradio Interface Creation
    ############################################################################

    def create_annotation_interface(self):
        """Create Gradio interface for pose annotation."""
            
        # Toggle annotation tools visibility based on display mode
        def set_visibility_annotation_tools(display_mode):
            return gr.Group(visible=(display_mode == self.MODE_ANNOTATION_MODE))
        
        
        # Create individual click handlers for each camera
        def create_click_handler(cam_key):
            def handler(evt: gr.SelectData, selected_camera, selected_joint, confidence_score, frame_data, display_mode):
                return process_user_click_event(evt, cam_key, selected_camera, selected_joint, confidence_score, frame_data, display_mode)
            return handler
        
        def process_user_click_event(evt: gr.SelectData, cam_key, selected_camera, selected_joint, confidence_score, frame_data, display_mode):
            """Handle click events on images for both zoom selection and annotation."""
            try:
                # Only allow clicks in Annotation Mode - silently ignore otherwise
                if display_mode != self.MODE_ANNOTATION_MODE:
                    return [gr.skip()] * (len(self.cam_keys) * 2 + 2)  # Skip all outputs including zoom button
                
                # Only allow clicks on the selected camera
                if cam_key != selected_camera:    
                    status_msg = self.update_annotation_status(selected_camera, selected_joint, confidence_score, self.MODE_ANNOTATION_MODE, interact_camera=cam_key)
                    return [gr.skip()] * (len(self.cam_keys) * 2) + [status_msg, gr.skip()]  # Only update status, skip joint
                
                x, y = evt.index[0], evt.index[1]
                
                # If zoom mode is enabled, handle zoom selection
                if self.in_zoom:
                    result = self.handle_zoom_editing(selected_camera, selected_joint, frame_data, [x, y, confidence_score])
                    return result + [gr.skip()]  # Add skip for joint dropdown since zoom doesn't change joint
                else:  # Normal annotation mode
                    return self.handle_annotation(selected_camera, selected_joint, frame_data, self.FLAG_MANUAL, [x, y, confidence_score])
                    
            except Exception as e:
                status_msg = str(e)
                return [gr.skip()] * (len(self.cam_keys) * 2) + [status_msg, gr.skip()]  # Only update status, skip joint
        
        # Approve auto annotation 
        def set_visibility_approve_auto(selected_camera, selected_joint, frame_data, display_mode):
            """Update visibility of approve button based on VitPose confidence."""
            
            visible=False                
            # Check if VitPose detection exists and meets confidence threshold
            if (display_mode == self.MODE_ANNOTATION_MODE  and selected_camera in frame_data['vitpose_2d']):
                
                joint_idx = list(self.vit_joints_name).index(selected_joint)
                vitpose_conf = frame_data['vitpose_2d'][selected_camera][joint_idx, 2]
                
                # Show button only if confidence exceeds threshold
                if vitpose_conf >= self.threshold_det_conf:
                    visible=True
            return gr.update(visible=visible)
                
        def reset_joint_and_zoom_on_camera_change(selected_camera, confidence_score):
            """Reset to first joint when camera changes."""
            first_joint = self.vit_joints_name[0]

            self.in_zoom = False
            button_text, status_msg = self.toggle_zoom_enter(selected_camera, first_joint, confidence_score)
            return gr.update(value=first_joint), button_text, status_msg
        

        def update_all_cameras(selected_camera, selected_joint, confidence_score, frame_id, display_mode):
            """Update display for all cameras when frame changes."""
            try:
                frame_data = self.get_frame_data(int(frame_id))
            except Exception as e:
                # If there's an exception getting frame data, skip all outputs
                return [gr.skip()] * (len(self.cam_keys) * 2 + 3)  # cameras + meta + instruction + status + frame_data
                
            # Update instance variables for all cameras
            camera_display_images = []
            camera_meta_texts = []
            
            # Reset zoom regions and corners when changing frames
            self.zoom_regions = {cam_key: None for cam_key in self.cam_keys}
            self.zoom_corner1 = {cam_key: None for cam_key in self.cam_keys}
            
            for cam_key in self.cam_keys:
                assert cam_key in frame_data['images'], f"{cam_key} not found in frame data"
                    
                # Get the base image
                image = frame_data['images'][cam_key]
                img_pil = self.create_image_pil_for_vis(image, frame_data, cam_key, display_mode)
                camera_display_images.append(img_pil)
                
                # Generate info text for this camera - prepare data for HighlightedText
                info_data = self.get_frame_camera_annotation_status_strings(frame_data, cam_key, display_mode)
                camera_meta_texts.append(info_data)
            
            # Comprehensive annotation instructions with HTML styling
            box_style = "border: 1px solid #ddd; border-radius: 8px; padding: 12px; background-color: #f9f9f9; font-family: Arial, sans-serif;"
            
            instruction_texts = f"""
            <div style='{box_style}'>
                <strong>Instructions</strong><br><br>
                <h3 style='margin-top: 0; color: #333;'>Frame {frame_id}</h3>
                <p><span style='color: red; font-weight: bold;'>RED keypoints</span> need annotation.</p>
                
                <h4 style='color: #666; margin-bottom: 8px;'>💡 Quick Actions:</h4>
                <ul style='margin: 0; padding-left: 20px;'>
                    <li><span style='color: ##28A745; font-weight: bold;'> Edit Annotation and Click on Image</span> to manually annotate</li>
                    <li>Use <span style='color: #CC5500; font-weight: bold;'>'Edit Zoom'</span> to zoom-in by clicking to define corners</li>
                    <li>Use <span style='color: #FF8C00; font-weight: bold;'>'Reset Zoom'</span> to reset zoom-in for selected camera</li>
                    <li>Use <span style='color: #007BFF; font-weight: bold;'>'Approve Auto'</span> to approve good auto-detections</li>
                    <li>Use <span style='color: #8A2BE2; font-weight: bold;'>'Pass Joint'</span> to pass occluded/invisible joints</li>
                </ul>
                <h4 style='color: #666; margin-bottom: 8px;'>💡 Tips:</h4>
                <ul style='margin: 0; padding-left: 20px;'>
                    <li> Press <span style='color: #8A2BE2; font-weight: bold;'>'Pass Joint'</span> to temporally remove joint visualisation and annotate. </li>
                    <li> Press <span style='color: #007BFF; font-weight: bold;'>'Approve Auto'</span> to recover auto-detections. </li>
                </ul>
            </div>
            """
            
            status_msg = self.update_annotation_status(selected_camera, selected_joint, confidence_score, display_mode)
            

            # Return images and metadata from instance variables + reset to first camera and first joint
            first_camera = self.cam_keys[0]
            first_joint = self.vit_joints_name[0]

            # Reset to enter zoom mode
            self.in_zoom = False
            button_text, status_msg = self.toggle_zoom_enter(first_camera, first_joint, confidence_score)
            
            return camera_display_images + camera_meta_texts + [instruction_texts, status_msg, frame_data, gr.update(value=first_camera), gr.update(value=first_joint), button_text]


        # Gradio interface
        with gr.Blocks(title="Human Pose Annotator", css="""
            #camera_container {
                max-height: 60vh;
                overflow-y: auto;
                padding-right: 10px;
            }
            #camera_container::-webkit-scrollbar {
                width: 8px;
            }
            #camera_container::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 4px;
            }
            #camera_container::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 4px;
            }
            #camera_container::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
        """) as interface:
            gr.Markdown("# Human Body Pose Annotator")
            gr.Markdown("Load and annotate human poses from VitPose detections and triangulation results.")
            
            with gr.Row():
                # Left column for controls
                with gr.Column(scale=1):
                    instruction_display = gr.HTML(label="Instructions",
                                                    value="")
                    

                    frame_slider = gr.Slider(minimum=min(self.available_frames),
                                                maximum=max(self.available_frames),
                                                step=1,
                                                value=self.available_frames[0],
                                                label="Frame ID",
                                                interactive=True)
                    
                    
                    display_mode = gr.Radio(choices=[self.MODE_ANNOTATION_MODE,self.MODE_ORIGINAL_IMAGE, self.MODE_2D_DETECTION, self.MODE_2D_DETECTION_REPROJ],
                                            value= self.MODE_ANNOTATION_MODE,
                                            label="Display Mode")
                    
                    
                    annotation_status = gr.HTML(label="Annotation Status", value="")
                                       
                    # Annotation tools - only visible in Annotation Mode
                    with gr.Group(visible=True) as annotation_tools:
                        gr.Markdown("### Annotation Tools")

                        gr.Markdown('<div style="background-color: white; padding: 10px;"><br></div>')  # Add white space separator     
                        
                        with gr.Row():
                            zoom_toggle_btn = gr.Button("📌 Edit Annotation", variant="primary")
                            zoom_reset_btn = gr.Button("🔄 Reset Zoom", variant="secondary")
                        
                        gr.Markdown('<div style="background-color: white; padding: 10px;"><br></div>')  # Add white space separator
                        
                        selected_camera = gr.Dropdown(choices=self.cam_keys,
                                                        value=self.cam_keys[0],
                                                        label="Select Camera to Annotate")
                        
                        # Camera navigation buttons
                        with gr.Row():
                            prev_camera_btn = gr.Button("⬅️ Previous Camera", variant="secondary", scale=1)
                            next_camera_btn = gr.Button("➡️ Next Camera", variant="secondary", scale=1)
                        
                        selected_joint = gr.Dropdown(choices=list(self.vit_joints_name),
                                                        value=self.vit_joints_name[0],
                                                        label="Select Joint to Annotate")
                        
                        # Joint navigation buttons
                        with gr.Row():
                            prev_joint_btn = gr.Button("⬅️ Previous Joint", variant="secondary", scale=1)
                            next_joint_btn = gr.Button("➡️ Next Joint", variant="secondary", scale=1)
                        
                        confidence_score = gr.Slider(minimum=0.0,
                                                        maximum=1.0,
                                                        step=0.1,
                                                        value=1.0,
                                                        label="Annotation Confidence (0-1), higher is more confident")
                        
                        gr.Markdown('<div style="background-color: white; padding: 10px;"><br></div>')  # Add white space separator
                        with gr.Row():
                            approve_auto_btn = gr.Button("✅ Approve Auto Annot.", variant="primary")                        
                            pass_joint_btn = gr.Button("⏭️ Pass Joint", variant="secondary")
                    

                    save_btn = gr.Button("💾 Save Annotations", variant="primary", size="lg")
                    save_status = gr.HTML(label="Save Status", value= "")

                # Hidden state to store current frame data - initialize with first frame
                frame_data_state = gr.State(self.get_frame_data(self.available_frames[0]))
                    

                # Right column for camera views
                with gr.Column(scale=3):
                    gr.Markdown("### Camera Views")
                    
                    # Create scrollable container for camera views
                    with gr.Column(elem_id="camera_container") as camera_container:
                        # Create image displays and info text - one camera per row
                        camera_image_displays = []
                        camera_meta_displays = []
                        
                        for cam_key in self.cam_keys:
                            with gr.Row():
                                with gr.Column(scale=2):
                                    img_display = gr.Image(label=f"{cam_key.upper()} Image",
                                                            type="pil",
                                                            interactive=False)
                                    
                                    camera_image_displays.append(img_display)
                                    
                                with gr.Column(scale=1):
                                    meta_display = gr.HighlightedText(label=f"{cam_key.upper()} Data",
                                                                        interactive=False,
                                                                        color_map={"Annot. Needed": "red", "Annot. Done": "green", "Auto Approv.":"blue", "Annot. Pass": "purple"})
                                    
                                    camera_meta_displays.append(meta_display)
                    
                    # Add click event handlers after all displays are created
                    for i, cam_key in enumerate(self.cam_keys):
                        camera_image_displays[i].select(fn=create_click_handler(cam_key),
                                                        inputs=[selected_camera, selected_joint, confidence_score, frame_data_state, display_mode],
                                                        outputs=camera_image_displays + camera_meta_displays + [annotation_status, selected_joint])
                        
                
                approve_auto_btn.click(fn=self.toggle_approve_auto,
                            inputs=[selected_camera, selected_joint, frame_data_state, display_mode],
                            outputs=camera_image_displays + camera_meta_displays + [annotation_status, selected_joint] )
                
                pass_joint_btn.click(fn=self.toggle_pass_joint,
                            inputs=[selected_camera, selected_joint, frame_data_state, display_mode],
                            outputs=camera_image_displays + camera_meta_displays + [annotation_status, selected_joint] )
                
                zoom_toggle_btn.click(fn=self.toggle_zoom_enter,
                            inputs=[selected_camera, selected_joint, confidence_score],
                            outputs=[zoom_toggle_btn, annotation_status])
                
                zoom_reset_btn.click(fn=self.toggle_zoom_reset,
                            inputs=[selected_camera, selected_joint, confidence_score, frame_data_state],
                            outputs=camera_image_displays + [zoom_toggle_btn, annotation_status])
                
                save_btn.click(fn=self.save_annotations,
                            inputs=[],
                            outputs=[save_status])
                
                # Camera navigation button handlers
                prev_camera_btn.click(fn=self.get_previous_camera,
                            inputs=[selected_camera],
                            outputs=[selected_camera])
                
                next_camera_btn.click(fn=self.get_next_camera,
                            inputs=[selected_camera],
                            outputs=[selected_camera])
                
                # Joint navigation button handlers
                prev_joint_btn.click(fn=self.get_previous_joint,
                            inputs=[selected_joint],
                            outputs=[selected_joint])
                
                next_joint_btn.click(fn=self.get_next_joint,
                            inputs=[selected_joint],
                            outputs=[selected_joint])
                
            
            # Update displays when frame or display mode changes              
            frame_slider.change(fn=update_all_cameras, 
                                inputs=[selected_camera, selected_joint, confidence_score, frame_slider, display_mode], 
                                outputs=camera_image_displays + camera_meta_displays + [instruction_display, annotation_status, frame_data_state, selected_camera, selected_joint, zoom_toggle_btn])
            display_mode.change(fn=update_all_cameras,
                                inputs=[selected_camera, selected_joint, confidence_score, frame_slider, display_mode], 
                                outputs=camera_image_displays + camera_meta_displays + [instruction_display, annotation_status, frame_data_state])
            
            # Reset to first joint when camera changes
            selected_camera.change(fn=reset_joint_and_zoom_on_camera_change,
                                   inputs=[selected_camera, confidence_score],
                                   outputs=[selected_joint, zoom_toggle_btn, annotation_status])
            
            
            selected_camera.change(fn=self.update_annotation_status,
                                    inputs=[selected_camera, selected_joint, confidence_score, display_mode],
                                    outputs=[annotation_status])
            
            selected_joint.change(fn=self.update_annotation_status,
                                    inputs=[selected_camera, selected_joint, confidence_score, display_mode],
                                    outputs=[annotation_status])
            
            confidence_score.change(fn=self.update_annotation_status,
                                    inputs=[selected_camera, selected_joint, confidence_score, display_mode],
                                    outputs=[annotation_status])
            
            display_mode.change(fn=self.update_annotation_status,
                                inputs=[selected_camera, selected_joint, confidence_score, display_mode],
                                outputs=[annotation_status])
            

            
            
            
            # Update approve button visibility when selection changes
            selected_camera.change(fn=set_visibility_approve_auto,
                                    inputs=[selected_camera, selected_joint, frame_data_state, display_mode],
                                    outputs=[approve_auto_btn])
            
            selected_joint.change(fn=set_visibility_approve_auto,
                                    inputs=[selected_camera, selected_joint, frame_data_state, display_mode],
                                    outputs=[approve_auto_btn])
            
            display_mode.change(fn=set_visibility_approve_auto,
                                inputs=[selected_camera, selected_joint, frame_data_state, display_mode],
                                outputs=[approve_auto_btn])
            
            frame_slider.change(fn=set_visibility_approve_auto,
                                inputs=[selected_camera, selected_joint, frame_data_state, display_mode],
                                outputs=[approve_auto_btn])
            
            # Toggle annotation tools visibility when display mode changes
            display_mode.change(fn=set_visibility_annotation_tools,
                                inputs=[display_mode],
                                outputs=[annotation_tools])
            
                                



            # Initialize with first frame, and annotation status with current settings
            interface.load(fn=update_all_cameras, 
                           inputs=[selected_camera, selected_joint, confidence_score, frame_slider, display_mode], 
                            outputs=camera_image_displays + camera_meta_displays + [instruction_display, annotation_status, frame_data_state])
            
            interface.load(fn=set_visibility_approve_auto,
                            inputs=[selected_camera, selected_joint, frame_data_state, display_mode],
                            outputs=[approve_auto_btn])
            
        return interface

def main():
    """Main function to launch the annotation tool."""
    annotator = None
    try:
        # Parse command line arguments
        args = get_args_parser()
        
        
        # Initialize pose annotator with arguments
        annotator = PoseAnnotator(args)
        
        # Create and launch interface
        interface = annotator.create_annotation_interface()
        interface.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=args.port,  # Use port from arguments
            share=args.share,       # Use share flag from arguments
            debug=True
        )
        
    except KeyboardInterrupt:
        print("\n⚠️  Received interrupt signal, saving annotations...")
        if annotator:
            save_result = annotator.save_annotations()
            print(f"Auto-save result: {save_result}")
        print("✅ Annotation tool closed safely.")
    except Exception as e:
        print(f"Error launching annotation tool: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Auto-save on exit
        if annotator:
            print("💾 Auto-saving annotations before exit...")
            save_result = annotator.save_annotations()
            print(f"Auto-save result: {save_result}")

if __name__ == "__main__":
    main()