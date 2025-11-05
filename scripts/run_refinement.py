import sys
sys.path.append('..')
import os
import argparse


import cv2
import numpy as np
import torch
import glob
import json

from dataset.recording import Recording


from utils.constants import PATH_ASSETS, PATH_ASSETS_REFINED_KPT3D, PATH_ASSETS_VIDEOS, PATH_ASSETS_KPT3D, VIT_JOINTS_NAME,  cfg_annotation, get_args_parser, update_cfg_annotation
from utils.logger import ColorLogger
from utils.io_utils import NumpyEncoder, load_3d_keypoint_json
from utils.triangulation import  compute_frame_joint_reprojection_error, visulaize_triangulation_results
from utils.refine_pose3d import refine_pose3d, fill_missing_keypoints, fix_smoothing_mistakes, fix_limb_mistakes
from utils.fit_pose3d import fit_pose3d



if __name__ == '__main__':
    args = get_args_parser()
    cfg_annotation = update_cfg_annotation(args)

    use_manual_annotation = args.use_manual_annotation
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # other paths 
    verbose = args.verbose
    load_undistort_images = 'undistort' in args.recording_tag
    recording_name = args.recording_tag.split('/')[-2]


    log_dir = os.path.join(PATH_ASSETS_REFINED_KPT3D, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = ColorLogger(log_dir, log_name=f'{recording_name}_manual.txt' if use_manual_annotation else f'{recording_name}_auto.txt')

    path_root_dir = os.path.join(PATH_ASSETS_VIDEOS, args.recording_tag)
    path_camera = args.path_camera if args.path_camera!='' else os.path.join(path_root_dir, 'calibration.json')
    assert os.path.exists(path_camera), f"Camera meta file does not exist: {path_camera}"

    dataset = Recording(root_dir = path_root_dir,
                        path_cam_meta = path_camera,
                        logger = logger,
                        recording_name = recording_name,
                        load_undistort_images = load_undistort_images)


    dict_cam_params = dataset.get_camera_params()
    cam_keys = list(dict_cam_params.keys())
    cam_keys = cam_keys if args.cam_keys=='' else args.cam_keys.split(',')
    logger.info(f"Using {len(cam_keys)} cameras for triangulation: {cam_keys}")


    ########################################
    # load original 3D keypoints
    path_json = os.path.join(PATH_ASSETS_KPT3D, f"{recording_name}_manual" if use_manual_annotation else f"{recording_name}_auto")
    dict_ori_3d = load_3d_keypoint_json(path_json)


    # save to json
    os.makedirs(PATH_ASSETS_REFINED_KPT3D, exist_ok=True)
    output_path = os.path.join(PATH_ASSETS_REFINED_KPT3D, path_json.split('/')[-1]+".json")
    logger.info(f"Refined 3D keypoints will be saved to: {output_path}")
    ##########################################

    kpt_3ds = []
    list_frame_ids = []
    num_joints_no_annotation = 0
    summary_stats = {'reproj_err':[]}
    for frame_id, frame_data in dict_ori_3d.items():
        cframe_id = int(frame_id)

        list_frame_ids.append(cframe_id)
        kpt_3d = np.array(frame_data['triangulated_3d'], dtype=np.float32)
        kpt_3ds.append(kpt_3d)



    kpt_3ds = np.array(kpt_3ds, dtype=np.float32)  # Shape: (N, 17, 4)
    ref_kpt_3ds = kpt_3ds.copy()

    
    verbose = True
    ref_kpt_3ds = refine_pose3d(cfg_annotation, ref_kpt_3ds, num_keypoints=ref_kpt_3ds.shape[1], verbose=verbose)


    # then go optimisation
    #ref_kpt_3ds = fit_pose3d(cfg_annotation, logger, ref_kpt_3ds, verbose=verbose)
    
    for iid, frame_id in enumerate(list_frame_ids):
        kpt_3d_valid = ref_kpt_3ds[iid]
        dict_ori_3d[str(frame_id)]['refined_3d'] = kpt_3d_valid.copy()
        num_joints_no_annotation += np.sum(kpt_3d_valid[:,3]<1e-3)


        try:
            dict_kpt2ds = dict_ori_3d[str(frame_id)]['annotation_2d']
        except:
            dict_kpt2ds = dict_ori_3d[str(frame_id)]['vitpose_2d']

        is_3d_valid = kpt_3d_valid[:,3]>1e-3
        kpt_3d_valid[~is_3d_valid,:3] = np.nan

        dict_reproj_err, mean_proj_nonan = compute_frame_joint_reprojection_error(dict_kpt2ds, kpt_3d_valid[...,:3], dict_cam_params, cam_keys, args.threshold_det_conf)
        if not np.isnan(mean_proj_nonan):
            summary_stats['reproj_err'].append(mean_proj_nonan)
        



        if args.verbose:
            if int(frame_id)%500!=0:
                continue
            dict_img_bgr = {}

            for cam_key in cam_keys:
                ccam_bgr = dataset.load_bgr_image_and_undistort(f"{cam_key}/frame_{int(frame_id):04d}.png")
                dict_img_bgr[cam_key] = ccam_bgr
            # Visualize results
            text_to_vis = {'mean_error': None, 'mean_proj': mean_proj_nonan}

            gt_joints_world_valid = None
            dict_concat_imgs = visulaize_triangulation_results(dict_img_bgr, dict_cam_params, kpt_3d_valid, gt_joints_world_valid, text_to_vis, cam_keys, input_zup=False)
            vis_tag = 'refined_manual' if use_manual_annotation else 'refined_auto'
            for ccam_key, concatenated_img in dict_concat_imgs.items():
                vis_dir = os.path.join(PATH_ASSETS,'results', 'vis', vis_tag, recording_name, ccam_key)
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"frame_{(int(frame_id)):04d}.png")
                cv2.imwrite(vis_path, concatenated_img)
        
    
    
    reproj_err = summary_stats['reproj_err']
    logger.info(f"Mean Reprojection Error: {np.mean(reproj_err):.2f} pixels; with {len(reproj_err)} valid frames")
    logger.info(f"Total joints without annotation: {num_joints_no_annotation} out of {len(dict_ori_3d)*len(VIT_JOINTS_NAME)} ({num_joints_no_annotation/(len(dict_ori_3d)*len(VIT_JOINTS_NAME))*100:.2f}%)")
    logger.info(f"Total frames processed: {len(dict_ori_3d)}")

    with open(output_path, 'w') as f:
        json.dump(dict_ori_3d, f, separators=(',', ':'), cls=NumpyEncoder)
