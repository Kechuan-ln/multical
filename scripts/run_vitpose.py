import sys

from tqdm import tqdm
sys.path.append('..')
import os
import argparse


import cv2
import numpy as np
import torch
import glob



import json
from transformers import AutoProcessor,  VitPoseForPoseEstimation

from dataset.recording import Recording


from utils.constants import PATH_ASSETS, PATH_ASSETS_VIDEOS, PATH_ASSETS_BBOX_MANUAL, PATH_ASSETS_KPT2D_AUTO, VIT_JOINTS_NAME, get_args_parser
from utils.logger import ColorLogger
from utils.plot_utils import draw_points

from utils.io_utils import NumpyEncoder, load_manual_2d_bbox_json


if __name__ == '__main__':
    args = get_args_parser()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = VitPoseForPoseEstimation.from_pretrained(os.path.join(PATH_ASSETS,"vitpose","vitpose-plus-huge"), device_map=device)
    image_processor = AutoProcessor.from_pretrained(os.path.join(PATH_ASSETS,"vitpose","vitpose-plus-huge"))


    # other paths 
    verbose = args.verbose
    load_undistort_images = 'undistort' in args.recording_tag
    recording_name = args.recording_tag.split('/')[-2]


    log_dir = os.path.join(PATH_ASSETS_KPT2D_AUTO, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = ColorLogger(log_dir, log_name=f'{recording_name}.txt')

    path_root_dir = os.path.join(PATH_ASSETS_VIDEOS, args.recording_tag)
    path_cam_meta = args.path_camera if args.path_camera!='' else os.path.join(path_root_dir, 'calibration.json')
    assert os.path.exists(path_cam_meta), f"Camera meta file does not exist: {path_cam_meta}" 
    
    dataset = Recording(root_dir = path_root_dir, 
                        path_cam_meta = path_cam_meta,
                        logger = logger, 
                        recording_name = recording_name,
                        load_undistort_images = load_undistort_images)

    ########################################
    dict_det2ds, set_frame_ids = load_manual_2d_bbox_json(PATH_ASSETS_BBOX_MANUAL, recording_name)
    cam_keys = list(dict_det2ds.keys()) if args.cam_keys is None else args.cam_keys.split(',')
    print(f"Processing {len(set_frame_ids)} frames from {len(cam_keys)} cameras: {cam_keys}")
    ########################################

    list_existing_files = glob.glob(os.path.join(PATH_ASSETS_KPT2D_AUTO, f"{recording_name}*.json"))
    fid = len(list_existing_files)
    results_path = os.path.join(PATH_ASSETS_KPT2D_AUTO, f"{recording_name}_{fid}.json")
    logger.info(f"ViTPose results will be saved to: {results_path}")
    os.makedirs(PATH_ASSETS_KPT2D_AUTO, exist_ok=True)

    # Initialize results storages
    json_to_save = {}
    for _, cframe_id in tqdm(enumerate(set_frame_ids), total=len(set_frame_ids), desc="Processing frames"):
        if int(cframe_id)>2200:
            continue
        dict_img_bgr = {}
        dict_vitpose = {}
        dict_bbox_xyxy = {}


        # load images from lmdb
        for ccam_key in cam_keys:
            ccam_bgr = dataset.load_bgr_image_and_undistort(f"{ccam_key}/frame_{int(cframe_id):04d}.png")
            dict_img_bgr[ccam_key] = ccam_bgr.copy()
        
        # Process each camera to get ViTPose results
        for ccam_key in cam_keys:
            img_rgb = dict_img_bgr[ccam_key][..., ::-1].copy()  # Convert BGR to RGB for ViTPose
            img_bgr = dict_img_bgr[ccam_key]
                
            if cframe_id not in dict_det2ds[ccam_key]:
                logger.info(f" No detections for camera {ccam_key} frame {cframe_id}")
                continue
            
            # Get detections for this camera and frame
            frame_detection = dict_det2ds[ccam_key][cframe_id]
            target_bbox = frame_detection['manual_xyxy']
            ctrack_id = frame_detection.get('auto_id', -1)
            det_conf = frame_detection.get('detection_confidence', 1.0)

            
            if target_bbox is None:
                logger.info(f"No bbox detection: {ccam_key} frame {cframe_id}")
                continue
            

            # Convert bbox to xywh format for VitPose
            dict_bbox_xyxy[ccam_key] = target_bbox
            sx, sy, ex, ey = target_bbox
            bbox_xywh = [sx, sy, ex-sx, ey-sy]
            bboxes = [[bbox_xywh]]
            
            # Run pose estimation
            inputs = image_processor(img_rgb, boxes=bboxes, return_tensors="pt").to(device)
            data_index = torch.tensor([0], device=device)
            
            with torch.no_grad():
                outputs = model(**inputs, dataset_index=data_index)
            
            pose_results = image_processor.post_process_pose_estimation(outputs, boxes=bboxes)
            

            if len(pose_results) > 0 and len(pose_results[0]) > 0:
                cresult = pose_results[0][0]
                cscores = np.array(cresult["scores"])
                ckeypoints = np.array(cresult["keypoints"])
                
                vit_to_save = np.concatenate([ckeypoints, cscores[...,None]], axis=-1)
                dict_vitpose[ccam_key] = vit_to_save
                

                if args.verbose:
                    if int(cframe_id)%1000!=0:
                        continue
                    img_bgr = draw_points(img_bgr, vit_to_save[:,:2], scores=vit_to_save[:,2],
                                        pose_keypoint_color=[(0, 255, 255) for _ in range(len(VIT_JOINTS_NAME))], 
                                        keypoint_score_threshold=args.threshold_det_conf,
                                        radius=10, show_keypoint_weight=False)
                    
                    vis_dir = os.path.join(PATH_ASSETS, 'results', 'vis', 'vitpose', f"{recording_name}_{ccam_key}")
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_path = os.path.join(vis_dir, f"frame_{int(cframe_id):04d}.png")
                    cv2.imwrite(vis_path, img_bgr)
                    
                   

            else:
                logger.info(f"  No pose results for camera {ccam_key}, track_id {ctrack_id}")

            dict_img_bgr[ccam_key] = img_bgr.copy() 
        

        # Save frame results
        frame_result = {'bbox_xyxy': dict_bbox_xyxy.copy(), 
                        'vitpose_2d': dict_vitpose.copy(), }
        
        json_to_save[int(cframe_id)] = frame_result



        
    # Save all results to JSON file
    with open(results_path, 'w') as f:
        json.dump(json_to_save, f, separators=(',', ':'), cls=NumpyEncoder)
    
    print(f"\nTriangulation results saved to: {results_path}")
    print(f"Processed {len(json_to_save)} frames")
    
    
    logger.info(f"Total frames processed: {len(json_to_save)}")