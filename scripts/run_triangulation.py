import sys
sys.path.append('..')
import os
import argparse


import cv2
import numpy as np
import torch
import glob
import json
from tqdm import tqdm

from dataset.recording import Recording


from utils.constants import VIT_JOINTS_NAME, PATH_ASSETS, PATH_ASSETS_VIDEOS, PATH_ASSETS_KPT2D_MANUAL, PATH_ASSETS_KPT2D_AUTO, PATH_ASSETS_KPT3D, cfg_annotation, get_args_parser, update_cfg_annotation
from utils.logger import ColorLogger
from utils.plot_utils import  draw_points
from utils.triangulation import triangulate, create_camera_group
from utils.io_utils import NumpyEncoder, load_vitpose_json, load_manual_keypoint_json
from utils.triangulation import Triangulator , compute_frame_joint_reprojection_error, visulaize_triangulation_results




if __name__ == '__main__':
    args = get_args_parser()
    update_cfg_annotation(args)
    print(cfg_annotation)

    use_manual_annotation = args.use_manual_annotation
    only_frames_with_manual = args.only_frames_with_manual
    use_aniposelib = False

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # other paths 
    verbose = args.verbose
    load_undistort_images = 'undistort' in args.recording_tag
    recording_name = args.recording_tag.split('/')[-2]


    log_dir = os.path.join(PATH_ASSETS_KPT3D, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = ColorLogger(log_dir, log_name=f'{recording_name}.txt' if use_manual_annotation else f'{recording_name}_auto.txt')

    path_root_dir = os.path.join(PATH_ASSETS_VIDEOS, args.recording_tag)
    path_camera = args.path_camera if args.path_camera!='' else os.path.join(path_root_dir, 'calibration.json')

    dataset = Recording(root_dir = path_root_dir,
                        path_cam_meta = path_camera,
                        logger = logger,
                        recording_name = recording_name,
                        load_undistort_images = load_undistort_images)


    dict_cam_params = dataset.get_camera_params()
    cam_keys = list(dict_cam_params.keys()) if args.cam_keys=='' else args.cam_keys.split(',')
    logger.info(f"Using {len(cam_keys)} cameras for triangulation: {cam_keys}")


    # Save all results to JSON file
    os.makedirs(PATH_ASSETS_KPT3D, exist_ok=True)

    tag = 'manual' if use_manual_annotation else 'auto'
    results_path = os.path.join(PATH_ASSETS_KPT3D, f"{recording_name}_{tag}.json")
    logger.info(f"Triangulation results will be saved to: {results_path}")


    #########################################
    if use_aniposelib:
        cam_grp = create_camera_group(cam_keys, dict_cam_params)
    else:
        triangulator = Triangulator(cfg_annotation, cam_keys, dict_cam_params)

    prefix_vitpose_json = os.path.join(PATH_ASSETS_KPT2D_AUTO, f'{recording_name}')
    vitpose_data = load_vitpose_json(prefix_vitpose_json)

    prefix_manual_json = os.path.join(PATH_ASSETS_KPT2D_MANUAL, f'{recording_name}')
    manual_keypoint_data = load_manual_keypoint_json(prefix_manual_json)

    set_frame_ids = manual_keypoint_data.keys() if only_frames_with_manual else set(vitpose_data.keys()).union(set(manual_keypoint_data.keys()))
    set_frame_ids = sorted(set_frame_ids, key=lambda x: int(x))
    logger.info(f'Found {len(set_frame_ids)} frames in sequence {recording_name}.')


    # Initialize results storage
    json_to_save = {}
    summary_stats = {'reproj_err': []}

    num_joints_no_annotation = 0
    
    for _, cframe_id in tqdm(enumerate(set_frame_ids), total=len(set_frame_ids), desc="Processing frames"):
        verbose = args.verbose and int(cframe_id)%500==0

        dict_img_bgr = {}        
        dict_kpt2ds = manual_keypoint_data[cframe_id]['manual_2d'] if use_manual_annotation and cframe_id in manual_keypoint_data \
            else vitpose_data[cframe_id]['vitpose_2d']
        

        # traingulate 3D joints with RANSAC from 2D detections
        list_kpt_2ds = []
        list_kpt_scores = []

        for ccam_key in cam_keys:
            if verbose:
                ccam_bgr = dataset.load_bgr_image_and_undistort(f"{ccam_key}/frame_{int(cframe_id):04d}.png")
                dict_img_bgr[ccam_key] =  ccam_bgr

            if ccam_key in dict_kpt2ds:
                cdetection = dict_kpt2ds[ccam_key]

                list_kpt_2ds.append(cdetection[:,:2])
                list_kpt_scores.append(cdetection[:,2])
                
                if verbose:
                    # Draw 2D detections on the image
                    img_bgr = dict_img_bgr[ccam_key]
                    img_bgr = draw_points(img_bgr, cdetection[:,:2],  scores=cdetection[:,2],
                                        pose_keypoint_color=[(0, 255, 255) for _ in range(len(VIT_JOINTS_NAME))], 
                                        keypoint_score_threshold=args.threshold_det_conf,
                                        radius=10, show_keypoint_weight=False)
                    dict_img_bgr[ccam_key] = img_bgr
            else:
                list_kpt_2ds.append(np.zeros((17, 2), dtype=np.float32))
                list_kpt_scores.append(np.zeros((17,), dtype=np.float32))
                
        
        kpt_2ds = np.array(list_kpt_2ds, dtype=np.float32)
        kpt_scores = np.array(list_kpt_scores, dtype=np.float32)

        if use_aniposelib:
            assert False
            kpt_3d, _ = triangulate(cam_grp, kpt_2ds, kpt_scores, score_threshold=args.threshold_det_conf, use_ransac=True, min_cams=args.min_cam)
        else:
            kpt_3d, _ = triangulator.run(kpt_2ds, kpt_scores)
        

        #####################################
        dict_reproj_err, mean_proj_nonan = compute_frame_joint_reprojection_error(dict_kpt2ds, kpt_3d, dict_cam_params, cam_keys, args.threshold_det_conf)
        if not np.isnan(mean_proj_nonan):
            summary_stats['reproj_err'].append(mean_proj_nonan)



        # clean kpt_3d and to save
        is_valid_3d = ~np.isnan(kpt_3d)  # Shape: (17, 3) - True where not NaN 
        num_joints_no_annotation += np.sum(~is_valid_3d[...,0])

        is_valid_3d = is_valid_3d.all(axis=1).astype(float)  # Shape: (17,) - 1.0 if all xyz valid, 0.0 if any NaN

        
        kpt_3d[np.isnan(kpt_3d)] = 0.

        kpt_3d_valid = np.column_stack([kpt_3d, is_valid_3d])

        # Save frame results
        frame_result = {'annotation_2d' if use_manual_annotation else 'vitpose_2d': dict_kpt2ds.copy(), 
                        'triangulated_3d': kpt_3d_valid,  
                        'reproj_err': dict_reproj_err.copy()}
        
        json_to_save[int(cframe_id)] = frame_result


        # Compare with ground truth 3D joints
        mean_error = None

        if verbose:
            # Visualize results
            text_to_vis = {'mean_proj': mean_proj_nonan, 'mean_error': None}
            kpt_3d_gt_valid =  None

            dict_concat_imgs = visulaize_triangulation_results(dict_img_bgr, dict_cam_params, kpt_3d_valid, kpt_3d_gt_valid, text_to_vis, cam_keys, input_zup=False)
            vis_tag = 'triangulation' if use_manual_annotation else 'triangulation_auto'
            for ccam_key, concatenated_img in dict_concat_imgs.items():
                vis_dir = os.path.join(PATH_ASSETS,'results', 'vis', vis_tag, recording_name, ccam_key)
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"frame_{int(cframe_id):04d}.png")
                cv2.imwrite(vis_path, concatenated_img)

    
    with open(results_path, 'w') as f:
        json.dump(json_to_save, f, separators=(',', ':'), cls=NumpyEncoder)
    
    print(f"\nTriangulation results saved to: {results_path}")
    print(f"Processed {len(json_to_save)} frames")
    
    
    reproj_err = summary_stats['reproj_err']
    logger.info(f"Mean Reprojection Error: {np.mean(reproj_err):.2f} pixels; with {len(reproj_err)} valid frames")
    logger.info(f"Total joints without annotation: {num_joints_no_annotation} out of {len(json_to_save)*len(VIT_JOINTS_NAME)} ({num_joints_no_annotation/(len(json_to_save)*len(VIT_JOINTS_NAME))*100:.2f}%)")
    logger.info(f"Total frames processed: {len(json_to_save)}")