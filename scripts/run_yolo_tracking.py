import sys
sys.path.append('..')
sys.path.append('../bytetrack')
import os
import argparse


from ultralytics import YOLO


import cv2
import numpy as np
from tqdm import tqdm
import torch
import json


from dataset.recording import Recording

from utils.logger import ColorLogger
from utils.plot_utils import draw_box, draw_box_with_tracking


from utils.constants import PATH_ASSETS,PATH_ASSETS_VIDEOS,PATH_ASSETS_BBOX_AUTO, get_args_parser
from utils.io_utils import NumpyEncoder




# ByteTrack imports
from bytetrack.yolox.tracker.byte_tracker import BYTETracker


class TrackingArgs:
    """Arguments for ByteTracker"""
    def __init__(self):
        self.track_thresh = 0.1      # tracking confidence threshold
        self.track_buffer = 60      # frames to keep lost tracks
        self.match_thresh = 0.9     # matching threshold for tracking
        self.aspect_ratio_thresh = 1.6  # filter boxes by aspect ratio
        self.min_box_area = 10       # filter out tiny boxes
        self.mot20 = False           # MOT20 dataset specific settings
        self.track_detection_iou = 0.5



if __name__ == '__main__':
    args = get_args_parser()


    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Load YOLOv8 model for human detection
    yolo_model = YOLO(os.path.join(PATH_ASSETS, 'vitpose', 'yolov8n-seg.pt'))
    yolo_model.verbose = False


    # Initialize ByteTracker
    tracking_args = TrackingArgs()
    ########################################################################

    # other paths 
    recording_name = args.recording_tag.split('/')[-2]
    load_undistort_images = 'undistort' in args.recording_tag

    log_dir = os.path.join(PATH_ASSETS_BBOX_AUTO, 'logs')


    # First load data of participant
    verbose = args.verbose

    # set logger
    os.makedirs(log_dir, exist_ok=True)
    logger = ColorLogger(log_dir, log_name=f'{recording_name}.txt')

    #set dataloader
    #dataset = EgoExo(root_dir=root_egoexo, participant_uid=participant_uid, cam_keys=cam_keys, logger=logger)
    #dataset.open_lmdb_img()
    path_root_dir = os.path.join(PATH_ASSETS_VIDEOS, args.recording_tag)
    path_cam_meta = args.path_camera if args.path_camera!='' else os.path.join(path_root_dir, 'calibration.json')
    assert os.path.exists(path_cam_meta), f"Camera meta file does not exist: {path_cam_meta}"


    dataset = Recording(root_dir = path_root_dir,
                        path_cam_meta = path_cam_meta,
                        logger = logger,
                        recording_name = recording_name,
                        load_undistort_images = load_undistort_images)


    dict_yolo = {}

    current_seq_name = None
    # iterate dataset
    for sample_idx, sample in tqdm(enumerate(dataset.datalist)): 
        img_bgr = dataset.load_bgr_image_and_undistort(sample['image_path'])
        seq_name = sample['seq_name']

        
        # Check if sequence has changed - reset tracker if so
        if current_seq_name != seq_name:
            if current_seq_name is not None:
                logger.info(f"Sequence changed from {current_seq_name} to {seq_name}, resetting tracker")
            else:
                logger.info(f"Starting new sequence: {seq_name}")
            
            # Reset the tracker for new sequence
            tracker = BYTETracker(tracking_args, frame_rate=30)
            current_seq_name = seq_name
        
        # Convert RGB to BGR for YOLO detection
        #img_bgr = img_rgb[:,:,::-1].copy()
        frame_idx = int(sample['img_id'])
        if seq_name not in dict_yolo:
            dict_yolo[seq_name] = {}

        # Detect bounding boxes of all 2D humans from the img_rgb (H, W, 3)
        yolo_results = yolo_model(img_bgr, classes=0, iou=0.3)  # Add NMS with IoU threshold
        det_bboxes = yolo_results[0].boxes.xyxy.detach().cpu().numpy()
        det_confidence = yolo_results[0].boxes.conf.detach().cpu().numpy()
        assert len(yolo_results) == 1, f"{len(yolo_results)}"


        tracked_bboxes = []
        track_ids = []
        tracked_scores = []
        
        if len(det_bboxes) > 0:
            # Create detection array in format [x1, y1, x2, y2, conf] for ByteTracker
            detections = np.column_stack([det_bboxes, det_confidence])
            
            # Get image dimensions
            img_h, img_w = img_bgr.shape[:2]
            img_info = [img_h, img_w]
            img_size = [img_h, img_w]
            
            # Update tracker with detections
            online_targets = tracker.update(detections, img_info, img_size)
            
            # Extract tracking results with filtering (like demo_track.py)
            for track in online_targets:
                if track.is_activated:
                    tlwh = track.tlwh  # [x, y, w, h]
                    bbox = track.tlbr  # [x1, y1, x2, y2]
                    track_id = track.track_id
                    score = track.score
                    
                    # Add filtering like demo_track.py
                    #vertical = tlwh[2] / tlwh[3] > tracking_args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > tracking_args.min_box_area:# and not vertical:
                        tracked_bboxes.append(bbox)
                        track_ids.append(track_id)
                        tracked_scores.append(score)
            
            logger.info(f"Frame {frame_idx}: {len(tracked_bboxes)} tracked humans, Track IDs: {track_ids}")
        else:
            logger.info(f"Frame {frame_idx}: No detections found")

        
        # Create a list to aggregate all people in this frame (tracked + untracked)
        frame_people_list = []
        
        # Process each detection for pose estimation
        for det_idx, (det_bbox, det_conf) in enumerate(zip(det_bboxes, det_confidence)):
            sx, sy, ex, ey = map(int, det_bbox)
            
            # Check if this detection has been tracked
            track_id = -1  # Default for untracked detections
            track_score = -1.0  # Use detection confidence as fallback
            
            # Find matching tracked bbox with highest IoU
            best_iou = 0
            best_track_id = -1
            best_track_score = -1.0
            
            for tracked_bbox, tid, tscore in zip(tracked_bboxes, track_ids, tracked_scores):
                # Calculate IoU between detection and tracked bbox
                tx1, ty1, tx2, ty2 = map(int, tracked_bbox)
                
                # Calculate intersection
                ix1 = max(sx, tx1)
                iy1 = max(sy, ty1)
                ix2 = min(ex, tx2)
                iy2 = min(ey, ty2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    det_area = (ex - sx) * (ey - sy)
                    track_area = (tx2 - tx1) * (ty2 - ty1)
                    union = det_area + track_area - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    # Keep track of the best IoU match
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = tid
                        best_track_score = tscore


            # Only assign track if IoU is above threshold
            if best_iou > tracking_args.track_detection_iou:
                track_id = best_track_id
                track_score = best_track_score
            
            # Convert to xywh format for VitPose
            bbox_xywh = [sx, sy, ex-sx, ey-sy]
            bboxes = [[bbox_xywh]]           
            
            
            cframe_to_save = { 'bbox_xyxy': [sx, sy, ex, ey],
                'track_id': int(track_id),
                'confidence': float(track_score),
                'detection_confidence': float(det_conf),
                'is_tracked': track_id != -1 }
            
                
            frame_people_list.append(cframe_to_save)
                
        # Save all people in this frame as a list
        if len(frame_people_list) > 0:
            dict_yolo[seq_name][str(frame_idx)] = frame_people_list
        
        # Create comprehensive visualization with detection, tracking
        if args.verbose:
            if frame_idx % 500 !=0:
                continue
            comprehensive_img = img_bgr.copy()
            
            # 1. Draw original YOLO detections (in green)
            if len(det_bboxes) > 0:
                comprehensive_img = draw_box(comprehensive_img, det_bboxes, det_confidence, color=(0, 255, 0))
            
            # 2. Draw tracking boxes with IDs using cframe_to_save info
            if len(frame_people_list) > 0:
                # Extract info from cframe_to_save for visualization
                viz_bboxes = []
                viz_track_ids = []
                viz_confidences = []
                
                for person_data in frame_people_list:
                    viz_bboxes.append(person_data['bbox_xyxy'])
                    viz_track_ids.append(person_data['track_id'])
                    viz_confidences.append(person_data['confidence'])
                
                comprehensive_img = draw_box_with_tracking(comprehensive_img, viz_bboxes, viz_track_ids, viz_confidences)
            
            # Single save with all information
            dir_vis = os.path.join(PATH_ASSETS, 'results', 'vis', 'bbox', seq_name)
            os.makedirs(dir_vis, exist_ok=True)
            cv2.imwrite(os.path.join(dir_vis, f"{frame_idx:04d}.png"), comprehensive_img)
            print(f"Saved comprehensive visualization: detections({len(det_bboxes)}), tracks({len(tracked_bboxes)})")#, poses({len(frame_poses_for_viz)})")

    # Save tracking results with pose estimation

    dir_json_output = PATH_ASSETS_BBOX_AUTO
    os.makedirs(dir_json_output, exist_ok=True)


    for k,v in dict_yolo.items():
        path_to_save = os.path.join(dir_json_output, f"{k}.json")
        with open(path_to_save, 'w') as f:
            json.dump(v, f, separators=(',', ':'), cls=NumpyEncoder)
        
    print(f"Tracking and pose estimation results saved to {dir_json_output}")
