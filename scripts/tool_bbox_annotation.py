#!/usr/bin/env python3
"""
Gradio-based Bounding Box Annotation Tool

This tool provides a web interface for annotating, editing, and validating 
bounding boxes and track IDs using YOLO detection results.

Features:
- Load and visualize images from LMDB dataset
- Display YOLO 2D detections with track IDs
- Interactive bounding box editing and annotation
- Track ID management and correction
- Export corrected annotations

Prerequisites:
- Results from run_yolo_tracking.py saved as JSON files
- EgoExo dataset with LMDB image storage
- Camera parameters
"""

import copy
import os
import sys
import json
import argparse
import cv2
import gradio as gr
from PIL import Image

# Add parent directory to path for imports
sys.path.append('..')
#from dataset.EgoExo.dataset import EgoExo
from dataset.recording import Recording
from utils.constants import PATH_ASSETS, PATH_ASSETS_VIDEOS, PATH_ASSETS_BBOX_AUTO, PATH_ASSETS_BBOX_MANUAL 
from utils.logger import ColorLogger
from utils.plot_utils import draw_box_with_tracking
from utils.io_utils import load_yolo_track_json,  NumpyEncoder

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gradio-based Bounding Box Annotation Tool', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--port', type=int, default=7861, help='Port for Gradio interface')
    parser.add_argument('--share', action='store_true', help='Create public sharing link')

    parser.add_argument('--recording_tag', type=str, default='sync_9122/original', help='recording name')
    parser.add_argument('--path_camera',type=str, default='', help='Path to the camera meta json file, if empty, use calibration.json under --recording_tag')
    parser.add_argument('--cam_key', type=str, default='cam03', help='camera key to annotate')

    return parser.parse_args()

class BBoxAnnotator:
    def __init__(self, args):
        """Initialize the bounding box annotator with required data."""
        #self.args = args
        self.setup_dataset(args)
        self.load_existing_json()
        
        # Set up output path for saving annotations
        self.output_path = os.path.join(PATH_ASSETS_BBOX_MANUAL, f"{self.recording_name}_{self.cam_key}.json")
        
        # Initialize manual annotations storage
        self.recent_track_id = -1
        
        # Annotation modes
        self.in_edit_mode = False
        self.edit_corner1 = None  # Store first corner (top-left)
        self.selected_track_id = -1  # Currently selected track ID
        
    def setup_dataset(self,args):

        load_undistort_images  = 'undistort' in args.recording_tag
        self.recording_name = args.recording_tag.split('/')[-2]
        self.cam_key = args.cam_key
        
        self.log_dir = os.path.join(PATH_ASSETS_BBOX_MANUAL, 'logs')
        
        # Set up logger
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = ColorLogger(self.log_dir, log_name=f'{self.recording_name}_{self.cam_key}_manual.txt')

        path_root_dir = os.path.join(PATH_ASSETS_VIDEOS, args.recording_tag)
        path_cam_meta = args.path_camera if args.path_camera else os.path.join(path_root_dir, 'calibration.json')
        assert os.path.exists(path_cam_meta), f"Camera meta file does not exist: {path_cam_meta}"

        self.dataset =  Recording(root_dir = path_root_dir,
                        path_cam_meta = path_cam_meta,
                        logger = self.logger,
                        recording_name = self.recording_name,
                        cam_keys = [self.cam_key],
                        load_undistort_images = load_undistort_images)
        
        self.logger.info("Dataset initialized successfully")
        self.image_hw = self.dataset.image_hw
        
    def load_existing_json(self):
        """Load YOLO tracking results from JSON files."""
        results_path = os.path.join(PATH_ASSETS_BBOX_AUTO, f"{self.recording_name}_{self.cam_key}.json")
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"YOLO tracking results not found: {results_path}")
            
        self.json_data = load_yolo_track_json(results_path)
        # Get available frame IDs from dataset datalist (sorted from smallest to largest)
        self.available_frames = sorted([item['img_id'] for item in self.dataset.datalist])



    def save_annotations(self):
        """Save manual annotations to JSON file."""
        try:
            # Organize data by frame with both manual_xyxy and auto_id
            save_data = {}
            for frame_id in self.available_frames:
                frame_key = str(frame_id)
                if frame_key not in self.json_data:
                    continue
                if 'manual_xyxy' not in self.json_data[frame_key] and 'auto_id' not in self.json_data[frame_key]:
                    continue

                save_data[frame_key] = {}
                
                # Add manual annotations if they exist
                if 'manual_xyxy' in self.json_data[frame_key]:
                    save_data[frame_key]['manual_xyxy'] = self.json_data[frame_key]['manual_xyxy']
                
                # Add auto-approved track ID if it exists
                if 'auto_id' in self.json_data[frame_key]:
                    save_data[frame_key]['auto_id'] = self.json_data[frame_key]['auto_id']
                    
            
            with open(self.output_path, 'w') as f:
                json.dump(save_data, f, separators=(',', ':'), cls=NumpyEncoder)
                
            return f"✅ Saved {len(save_data)} frames."
            
        except Exception as e:
            return f"❌ Error saving annotations: {str(e)}"
    


    #######################################
    # Process each frame
    #######################################
    def get_frame_data(self, frame_id, load_image=True):
        """Get frame data including images and detection results."""
        frame_key = str(frame_id)
        if frame_key in self.json_data:
            frame_data = copy.deepcopy(self.json_data[frame_key])
        else:
            frame_data = {'auto_detect':[]}
        
        if load_image:
            #img_rgb = self.dataset.get_image_from_lmdb(f"{self.recording_name}/{self.cam_key}/frame_{frame_id:04d}.jpg")
            #img_bgr = img_rgb[:, :, ::-1].copy() if img_rgb is not None else None
            img_bgr = self.dataset.load_bgr_image_and_undistort(f"{self.cam_key}/frame_{frame_id:04d}.png")
            frame_data['image'] = img_bgr 
        frame_data['frame_id'] = frame_id
                
        return frame_data
        
    def create_image_pil_for_vis(self, frame_data):
        """Visualize bounding boxes and track IDs on image."""
        img_vis = frame_data['image'].copy()
        
        # Prepare data for visualization
        bboxes = []
        track_ids = []
        confidences = []
        
        for detection in frame_data['auto_detect']:
            bbox_xyxy = detection['bbox_xyxy']
            track_id = detection['track_id']
            if track_id>0:            
                bboxes.append(bbox_xyxy)
                track_ids.append(track_id)
                confidences.append(detection['confidence'])
            
        # Draw bounding boxes with track IDs
        if len(bboxes) > 0:
            img_vis = draw_box_with_tracking(img_vis, bboxes, track_ids, confidences)
        
        # Draw manual annotations in green
        if 'manual_xyxy' in frame_data:
            x1, y1, x2, y2 = frame_data['manual_xyxy']
            cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 8)
                    
        img_pil = Image.fromarray(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        return img_pil
        


    # Handle Display
    def update_display(self, frame_id):
        """Update display when frame changes."""
        try:
            frame_data = self.get_frame_data(int(frame_id))
            img_pil = self.create_image_pil_for_vis(frame_data)
            
            # Get available track IDs for dropdown
            available_track_ids = self.get_available_track_ids(frame_data)
            default_track_id = self.get_default_track_id(available_track_ids)
            
            # Create instruction text
            recent_track_display = "N/A" if self.recent_track_id == -1 else str(self.recent_track_id)
            instruction_text = f"""
            <div style='border: 1px solid #ddd; border-radius: 8px; padding: 12px; background-color: #f9f9f9;'>
                <h3>Frame {frame_id} - Bounding Box Annotation</h3>
                <p><strong>Available Track IDs:</strong> {available_track_ids}</p>
                <p><strong>Recent Track ID:</strong> {recent_track_display}</p>
                <p><strong>Instructions:</strong></p>
                <ul>
                    <li>Select a Track ID and click <strong>"Approve Auto Detection"</strong> to approve existing detection</li>
                    <li>Or click <strong>"Edit Mode"</strong> and then click on image twice to define manual bounding box</li>
                    <li>Approved auto annotations or manual annotations will be shown in <strong>Cyan</strong></li>
                    <li><strong>Fast Forward:</strong> Automatically advances to the next frame where the recent track ID is missing, approving detections along the way</li>
                </ul>
            </div>
            """
            
            status_msg = gr.skip()
            
            return img_pil, instruction_text, status_msg, frame_data, gr.update(choices=available_track_ids, value=default_track_id)
            
        except Exception as e:
            return gr.skip(), f"❌ Error loading frame {frame_id}: {str(e)}", None, None, gr.update(choices=[])
        
    def navigate_and_update_prev(self, current_frame):
        current_frame = int(current_frame)
        current_frame_idx = self.available_frames.index(current_frame)
        prev_frame_idx = (current_frame_idx - 1) % len(self.available_frames)
        new_frame_value = self.available_frames[prev_frame_idx]
        update_result = self.update_display(new_frame_value)
        return [gr.update(value=new_frame_value)] + list(update_result)
    
    def navigate_and_update_next(self, current_frame):
        current_frame = int(current_frame)
        current_frame_idx = self.available_frames.index(current_frame)
        next_frame_idx = (current_frame_idx + 1) % len(self.available_frames)
        new_frame_value = self.available_frames[next_frame_idx]
        update_result = self.update_display(new_frame_value)
        return [gr.update(value=new_frame_value)] + list(update_result)
    
    def navigate_and_update_fast_forward(self, current_frame):
        """Navigate to the first frame where the selected track ID doesn't exist."""
        if self.recent_track_id < 0:
            return [gr.update()] + [gr.skip(), gr.skip(), "❌ No track ID selected for fast forward", gr.skip(), gr.skip()]
        
        current_frame = int(current_frame)
        current_frame_idx = self.available_frames.index(current_frame)
        
        # Search for the first frame after current where selected_track_id doesn't exist
        for i in range(current_frame_idx, len(self.available_frames)):
            frame_id = self.available_frames[i]
            frame_data = self.get_frame_data(frame_id, load_image=False)  # No need to load image for this check

            # If current frame already has manual annotation, skip it
            if i==current_frame_idx and 'manual_xyxy' in frame_data:
                continue
            
            # Check if selected_track_id exists in this frame
            find_detection = False
            for detection in frame_data['auto_detect']:
                if detection['track_id'] == self.recent_track_id:
                    find_detection=True
                    # assign manual_xyxy to the frame
                    if 'manual_xyxy' not in frame_data:
                        self.json_data[str(frame_id)]['manual_xyxy'] = copy.copy(detection['bbox_xyxy'])
                        self.json_data[str(frame_id)]['auto_id'] = self.recent_track_id
                    break
            if not find_detection:
                update_result = self.update_display(frame_id)
                return [gr.update(value=frame_id)] + list(update_result)
        
        # If not found, navigate to the final frame with message
        final_frame_id = self.available_frames[-1]
        update_result = self.update_display(final_frame_id)
        return [gr.update(value=final_frame_id)] + list(update_result[:-1]) + [f"Track ID {self.recent_track_id} exists in all remaining frames - moved to final frame"]
        
    
    # handle auto detection with existing track ID
    
    def get_available_track_ids(self, frame_data):
        """Get list of available tracking IDs (>0) for current frame."""
        track_ids = []
        for detection in frame_data['auto_detect']:
            track_id = detection['track_id']
            # Only include tracked detections (track_id > 0) with sufficient confidence
            if track_id > 0:
                track_ids.append(track_id)
        return sorted(list(set(track_ids)))  # Remove duplicates and sort
        
    def get_default_track_id(self, available_track_ids):
        """Get default track ID (previous approved one or smallest available)."""
        if not available_track_ids:
            return None
        
        # If we have a recent track ID that's still available, use it
        if self.recent_track_id>=0 and self.recent_track_id in available_track_ids:
            return self.recent_track_id
        
        # Otherwise use the smallest available track ID
        return min(available_track_ids)
    

    def approve_auto_detection(self, selected_track_id, frame_data):
        """Approve the selected auto detection and add to manual annotations."""
        frame_key = str(frame_data['frame_id'])
        
        # Find the selected detection
        selected_detection = None
        for detection in frame_data['auto_detect']:
            if detection['track_id'] == selected_track_id:
                selected_detection = detection
                break
                
        if selected_detection is None:
            status_msg = "❌ No detection found with track ID {}".format(selected_track_id)
            return  gr.skip(), gr.skip(), status_msg, gr.skip(), gr.skip()
            
        frame_data['manual_xyxy'] = copy.copy(selected_detection['bbox_xyxy'])
        frame_data['auto_id'] = selected_track_id
        self.json_data[frame_key]['manual_xyxy'] = copy.copy(selected_detection['bbox_xyxy'])
        self.json_data[frame_key]['auto_id'] = selected_track_id
        
        
        # Update recent track ID
        self.recent_track_id = selected_track_id
        
        # Update display to refresh all UI elements including instruction text
        update_result = self.update_display(frame_data['frame_id'])
        
        # Modify the status message to include approval confirmation
        status_msg = f"✅ Approved track ID {selected_track_id}"
        
        return update_result[0], update_result[1], status_msg, update_result[3], update_result[4]

    # handle manual annotation by clicking on the image        
    def handle_image_click(self, frame_data, coord_xy):
        """Handle click events on the image."""
        if not self.in_edit_mode:
            return gr.skip(), "❌ Click 'Edit Mode' first to annotate"
            
        x, y = coord_xy
        
        if self.edit_corner1 is None:
            # First click - set top-left corner
            self.edit_corner1 = (x, y)
            return gr.skip(), f"📍 Top-left corner set at ({x}, {y}). Click again to set bottom-right corner."
        else:
            # Second click - set bottom-right corner and create bbox
            x1, y1 = self.edit_corner1
            x2, y2 = x, y
            
            # Ensure x1,y1 is top-left and x2,y2 is bottom-right
            bbox_xyxy = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            
            # Create manual annotation (just the bbox coordinates)
            frame_key = str(frame_data['frame_id'])
            
            # Reset edit mode
            self.edit_corner1 = None

            # Update frame_data to include the new manual annotation
            frame_data['manual_xyxy'] = bbox_xyxy
            if frame_key not in self.json_data:
                self.json_data[frame_key] = {}
            self.json_data[frame_key]['manual_xyxy'] = bbox_xyxy
            
            # Create updated visualization
            img_pil = self.create_image_pil_for_vis(frame_data)
            
            status_msg = f"✅ Manual bounding box at ({bbox_xyxy[0]}, {bbox_xyxy[1]}, {bbox_xyxy[2]}, {bbox_xyxy[3]})"
            
            return img_pil, status_msg
    
            
    def toggle_edit_mode(self):
        """Handle edit mode toggle button click."""

        self.in_edit_mode = not self.in_edit_mode
        self.edit_corner1 = None  # Reset corner when toggling

        button_text = "✏️ Exit Editing" if self.in_edit_mode else "✏️ Enter Editing Mode"
        status_msg = "Click on image to set top-left corner" if self.in_edit_mode else gr.skip()
        
            
        return gr.update(value=button_text), status_msg
    

    def check_progress(self):
        """Check progress by finding missing frames between min and max indices in manual_xyxy."""
        #if not self.manual_xyxy:
        #    return "❌ No manual annotations found. Please annotate some frames first."
        
        # Get frame indices from manual_xyxy (convert string keys to int)
        #annotated_frames = sorted([int(k) for k in self.manual_xyxy.keys()])
        
        annotated_frames = []
        for frame_key in self.json_data:
            if 'manual_xyxy' in self.json_data[frame_key]:
                annotated_frames.append(int(frame_key))


        if len(annotated_frames) == 0:
            return "❌ No manual annotations found."
        
        min_frame = min(annotated_frames)
        max_frame = max(annotated_frames)
        
        # Find missing frames in the range
        expected_frames = set(range(min_frame, max_frame + 1))
        missing_frames = sorted(expected_frames - set(annotated_frames))
        
        # Create progress report
        total_frames_in_range = max_frame - min_frame + 1
        annotated_count = len(annotated_frames)
        missing_count = len(missing_frames)
        
        progress_text = f"""
        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 12px; background-color: #f0f8ff;'>
            <h3>📊 Progress Report</h3>
            <p><strong>Frame Range:</strong> {min_frame} - {max_frame}</p>
            <p><strong>Total Frames in Range:</strong> {total_frames_in_range}</p>
            <p><strong>Annotated Frames:</strong> {annotated_count}</p>
            <p><strong>Missing Frames:</strong> {missing_count}</p>
            <p><strong>Progress:</strong> {annotated_count}/{total_frames_in_range} ({(annotated_count/total_frames_in_range)*100:.1f}%)</p>
        """
        
        if missing_frames:
            progress_text += f"<p><strong>Missing Frame IDs:</strong> {', '.join(map(str, missing_frames))}</p>"
        else:
            progress_text += "<p><strong>✅ All frames in range are annotated!</strong></p>"
            
        progress_text += "</div>"
        
        return progress_text
            
    def create_annotation_interface(self):
        """Create Gradio interface for bounding box annotation."""
        
        # Create click handler following the pattern from tool_annotation.py
        def create_click_handler():
            def handler(evt: gr.SelectData, frame_data_state):
                x, y = evt.index[0], evt.index[1]  # Get x, y coordinates from click event
                return self.handle_image_click(frame_data_state, (x, y))
            return handler
            
        # Gradio interface
        with gr.Blocks(title="Bounding Box Annotator") as interface:
            gr.Markdown("# Bounding Box Annotation Tool")
            gr.Markdown("Load and annotate bounding boxes from YOLO tracking results.")
            
            with gr.Row():
                # Left column for controls
                with gr.Column(scale=1):
                    instruction_display = gr.HTML(label="Instructions", value="")
                    
                    frame_slider = gr.Slider(minimum=min(self.available_frames),
                                           maximum=max(self.available_frames),
                                           step=1,
                                           value=self.available_frames[0],
                                           label="Frame ID",
                                           interactive=True)
                    
                    # Frame navigation buttons
                    with gr.Row():
                        prev_frame_btn = gr.Button("⬅️ Previous Frame", variant="secondary", scale=1)
                        next_frame_btn = gr.Button("➡️ Next Frame", variant="secondary", scale=1)
                        fast_forward_btn = gr.Button("⏩ Fast Forward", variant="secondary", scale=1)
                    
                    # Annotation tools
                    with gr.Group():
                        gr.Markdown("### Annotation Tools")

                        # Manual annotation
                        edit_mode_btn = gr.Button("✏️ Edit Mode", variant="primary")
                        
                        # Track ID selection for auto approval
                        track_id_dropdown = gr.Dropdown(label="Select Track ID to Approve", 
                                                       choices=[], 
                                                       value=None)
                        
                        approve_auto_btn = gr.Button("✅ Approve Auto Detection", variant="secondary")
                        
                        
                        
                    
                    annotation_status = gr.HTML(label="Status", value="")
                    
                    save_btn = gr.Button("💾 Save Annotations", variant="primary", size="lg")
                    save_status = gr.HTML(label="Save Status", value="")
                    
                    check_progress_btn = gr.Button("📊 Check Progress", variant="secondary", size="lg")
                    progress_status = gr.HTML(label="Progress Status", value="")
                    
                # Hidden state to store current frame data
                frame_data_state = gr.State(None)
                
                # Right column for image view
                with gr.Column(scale=2):
                    gr.Markdown("### Image View")
                    
                    image_display = gr.Image(label="Image",
                                           type="pil",
                                           interactive=False)
            
            # Frame navigation
            prev_frame_btn.click(fn=self.navigate_and_update_prev,
                               inputs=[frame_slider],
                               outputs=[frame_slider, image_display, instruction_display, annotation_status, frame_data_state, track_id_dropdown])
            
            next_frame_btn.click(fn=self.navigate_and_update_next,
                               inputs=[frame_slider],
                               outputs=[frame_slider, image_display, instruction_display, annotation_status, frame_data_state, track_id_dropdown])
            
            fast_forward_btn.click(fn=self.navigate_and_update_fast_forward,
                                 inputs=[frame_slider],
                                 outputs=[frame_slider, image_display, instruction_display, annotation_status, frame_data_state, track_id_dropdown])
            
            # Frame slider changes
            frame_slider.change(fn=self.update_display,
                              inputs=[frame_slider],
                              outputs=[image_display, instruction_display, annotation_status, frame_data_state, track_id_dropdown])
            
            # Approve auto detection
            approve_auto_btn.click(fn=self.approve_auto_detection,
                                 inputs=[track_id_dropdown, frame_data_state],
                                 outputs=[image_display, instruction_display, annotation_status, frame_data_state, track_id_dropdown])
            

            
            # Edit mode toggle
            edit_mode_btn.click(fn=self.toggle_edit_mode,
                              inputs=[],
                              outputs=[edit_mode_btn, annotation_status])
            
            # Image click handler
            image_display.select(fn=create_click_handler(),
                               inputs=[frame_data_state],
                               outputs=[image_display, annotation_status])
            
            # Save annotations
            save_btn.click(fn=self.save_annotations,
                         inputs=[],
                         outputs=[save_status])
            
            # Check progress
            check_progress_btn.click(fn=self.check_progress,
                                   inputs=[],
                                   outputs=[progress_status])
            
            # Initialize interface
            interface.load(fn=self.update_display,
                         inputs=[frame_slider],
                         outputs=[image_display, instruction_display, annotation_status, frame_data_state, track_id_dropdown])
                         
        return interface

def main():
    """Main function to run the bounding box annotation tool."""
    annotator = None
    try:
        # Parse command line arguments
        args = get_args_parser()
        
        # Initialize bbox annotator with arguments
        annotator = BBoxAnnotator(args)
        
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
        print("✅ Bounding box annotation tool closed safely.")
    except Exception as e:
        print(f"Error launching bounding box annotation tool: {e}")
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