import glob
import copy
import os
import sys
import json
import argparse
import shutil
import numpy as np
import cv2
from tqdm import tqdm
# Add parent directory to path for imports
sys.path.append('..')
from dataset.recording import Recording
from utils.logger import ColorLogger
from utils.constants import PATH_ASSETS, PATH_ASSETS_REFINED_KPT3D, VIT_JOINTS_NAME, VIT_SKELETON, VIT_KEYPOINT_COLORS, PATH_ASSETS_VIDEOS, PATH_ASSETS_KPT2D_AUTO, PATH_ASSETS_KPT3D, PATH_ASSETS_KPT2D_CHECK, get_args_parser
from utils.io_utils import load_vitpose_json, NumpyEncoder, convert_images_to_video
from utils.triangulation import compute_frame_joint_reprojection_error



def visualise_annotation(img_bgr, detection_2d, reprojection_2d, flags_need_annotation, threshold_det_conf, frame_id):
    """Visualize 2D keypoints and reprojections on the image."""
    vis_img = img_bgr.copy()

    str_info = [f"Frame #{frame_id:06d}","Need Annotation:"]
    pt_str_info = 30

    info_img = np.zeros((vis_img.shape[0], 450, 3), dtype=np.uint8)+255
    for i, line in enumerate(str_info):
        cv2.putText(info_img, line, (0, pt_str_info), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
        pt_str_info += 40


    for sk in []:#VIT_SKELETON:
        cpt1 = sk[0]
        cpt2 = sk[1]
        if reprojection_2d[cpt1,0]>0 and reprojection_2d[cpt2,0]>0:
            pt1 = (int(reprojection_2d[cpt1,0]), int(reprojection_2d[cpt1,1]))
            pt2 = (int(reprojection_2d[cpt2,0]), int(reprojection_2d[cpt2,1]))
            cv2.line(vis_img, pt1, pt2, (128,128,128), 4)

    for i in range(detection_2d.shape[0]):
        det_x, det_y, det_conf = detection_2d[i]
        reproj_x, reproj_y = reprojection_2d[i]

        #if reproj_x>0 and reproj_y>0:
            #cv2.circle(vis_img, (int(reproj_x), int(reproj_y)), 5, (128,128,128), -1)

        if det_conf>threshold_det_conf and flags_need_annotation[i]:
            color = (0,0,255)#VIT_KEYPOINT_COLORS[i]#(0,0,255) if not flags_need_annotation[i] else VIT_KEYPOINT_COLORS[i]
            cv2.circle(vis_img, (int(det_x), int(det_y)), 8, color, -1)
            cv2.putText(vis_img, f"{VIT_JOINTS_NAME[i]}", (int(det_x)-10, int(det_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            pt_str_info = 40*i + 70
            cv2.putText(info_img, f"#{i:02d}: {VIT_JOINTS_NAME[i]}", (0, pt_str_info), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
            

    #vis_img = np.hstack((vis_img, info_img))

    # resize to 720p height
    scale = 720/vis_img.shape[0]
    vis_img = cv2.resize(vis_img, (int(vis_img.shape[1]*scale), 720))

    return vis_img



def compress_frame_ranges(frames: np.ndarray) -> list[str]:
    split_points = np.where(np.diff(frames) > 1)[0] + 1
    segments = np.split(frames, split_points)
    result = []
    for segment in segments:
        start_frame = int(segment[0])
        end_frame = int(segment[-1])
        result.append(str(start_frame) if start_frame == end_frame else f"{start_frame}-{end_frame}")
    return result


def merge_and_filter_ranges(range_strings: list[str], max_gap: int) -> list[str]:
    # This is a helper function to merge close segments and filter out single-frame segments.
    if not range_strings:
        return []

    def _parse_range_to_tuple(range_string: str) -> tuple[int, int]:
        if '-' in range_string:
            start_str, end_str = range_string.split('-', 1)
            return int(start_str), int(end_str)
        frame_idx = int(range_string)
        return frame_idx, frame_idx

    segments = [_parse_range_to_tuple(range_string) for range_string in range_strings]

    merged_segments: list[tuple[int, int]] = []
    current_start, current_end = segments[0]

    for start, end in segments[1:]:
        if start - current_end <= max_gap:
            current_end = max(current_end, end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end

    merged_segments.append((current_start, current_end))

    filtered_segments = [(start, end) for start, end in merged_segments if start != end]

    return [f"{start}-{end}" for start, end in filtered_segments]

if __name__ == '__main__':
    args= get_args_parser()

    # set threshold
    threshold_det_conf = args.threshold_det_conf
    threshold_reproj_err = args.threshold_reproj_err

    load_undistort_images  = 'undistort' in args.recording_tag
    recording_name = args.recording_tag.split('/')[-2]

    path_root_dir = os.path.join(PATH_ASSETS_VIDEOS, args.recording_tag)
    path_camera = args.path_camera if args.path_camera!='' else os.path.join(path_root_dir, 'calibration.json')
    assert os.path.exists(path_camera), f"Camera meta file does not exist: {path_camera}"

    log_dir = os.path.join(PATH_ASSETS_KPT2D_AUTO, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = ColorLogger(log_dir, log_name=f'{recording_name}.txt')


    # Load EgoExo dataset
    dataset =  Recording(root_dir = path_root_dir, 
                    path_cam_meta = path_camera, 
                    logger = logger, 
                    recording_name = recording_name,
                    load_undistort_images = load_undistort_images)

    logger.info("Dataset initialized successfully")
    image_hw = dataset.image_hw
    cam_params = dataset.get_camera_params()
    cam_keys = list(cam_params.keys())
    cam_keys = cam_keys if args.cam_keys=='' else args.cam_keys.split(',')
    logger.info(f"Using {len(cam_keys)} cameras: {cam_keys}")

    # load_existing annotation
    results_path = os.path.join(PATH_ASSETS_REFINED_KPT3D, f"{recording_name}_auto")
    
    json_data = load_vitpose_json(results_path)
    available_frames = sorted([int(key) for key in json_data.keys()])
    print(f"Loaded {len(available_frames)} frames with 2D/3D keypoints")

    dict_need_annot_per_cam = {ccam_key: np.empty((0, len(VIT_JOINTS_NAME)), dtype=bool) for ccam_key in cam_keys}

    for iid, frame_id in tqdm(enumerate(available_frames), total=len(available_frames), desc="Processing frames"):
        frame_key = frame_id if frame_id in json_data else str(frame_id)
        v = json_data[frame_key]
        cframe_kpts3d_valid = v['refined_3d'].copy()
        cframe_kpts3d = cframe_kpts3d_valid[...,:3]
        cvalid_3d = cframe_kpts3d_valid[...,3]>1e-3

        cframe_detections = v['vitpose_2d'].copy()

        dict_reproj_err, _ = compute_frame_joint_reprojection_error(cframe_detections, cframe_kpts3d, cam_params, cam_keys, threshold_det_conf)

        # calculate joints error
        max_reproj_err = np.empty((0,cframe_kpts3d.shape[0]), dtype=np.float32)
        for ccam_key in dict_reproj_err.keys():
            creproj_err = dict_reproj_err[ccam_key]
            max_reproj_err = np.vstack((max_reproj_err, creproj_err[None,...]))
        max_reproj_err = np.max(max_reproj_err, axis=0)

        max_reproj_err[~cvalid_3d] = -1

        flags_need_annotation = max_reproj_err>threshold_reproj_err 

        for ccam_key in cam_keys:
            # consider only detected joints
            cvitpose_2d = cframe_detections[ccam_key]
            cdetected_flags = cvitpose_2d[:,2]>threshold_det_conf
            flags_need_annotation = flags_need_annotation & cdetected_flags

            dict_need_annot_per_cam[ccam_key] = np.vstack((dict_need_annot_per_cam[ccam_key], flags_need_annotation[None,...]))

    # extract frames that need annotation
    frame_indices = np.array(available_frames)
    dict_need_annot_range = {}

    for ccam_key in cam_keys:
        dict_need_annot_range[ccam_key] = {}
        logger.info(f"Camera {ccam_key}:")
        for joint_idx, joint_name in enumerate(VIT_JOINTS_NAME):
            joint_frames = frame_indices[dict_need_annot_per_cam[ccam_key][:, joint_idx]]
            segments, segments2 = [], []

            if joint_frames.size == 0:
                logger.info(f"Joint #{joint_idx:02d} ({joint_name}): None")
            else:
                segments = compress_frame_ranges(joint_frames)
                max_gap = 30 # 0.5 seconds for 60fps
                segments2 = merge_and_filter_ranges(segments, max_gap=max_gap)

                if not segments2:
                    logger.info(f"Joint #{joint_idx:02d} ({joint_name}) [Cleaned]: None")
                else:
                    logger.info(f"Joint #{joint_idx:02d} ({joint_name}) [Cleaned]: {', '.join(segments2)}")

            dict_need_annot_range[ccam_key][joint_name] = {"segments": segments,"segments_cleaned": segments2,}


    # Persist joint range metadata for downstream tooling/analysis.
    filename_safe_threshold_det = str(int(threshold_det_conf*100))
    filename_safe_threshold_reproj = str(int(threshold_reproj_err))
    output_filename = f"{recording_name}_conf{filename_safe_threshold_det}_reproj{filename_safe_threshold_reproj}.json"
    output_path = os.path.join(PATH_ASSETS_KPT2D_CHECK, output_filename)
    if not os.path.exists(PATH_ASSETS_KPT2D_CHECK):
        os.makedirs(PATH_ASSETS_KPT2D_CHECK, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dict_need_annot_range, f, separators=(',', ':'), cls=NumpyEncoder)

    print(f"Saved joint range summary to {output_path}")


    # Visualize segments
    vis_root_dir = os.path.join(PATH_ASSETS_KPT2D_CHECK, recording_name)
    os.makedirs(vis_root_dir, exist_ok=True)

    for joint_idx, joint_name in enumerate(VIT_JOINTS_NAME):
        for ccam_key in cam_keys:
            if ccam_key != 'cam2':
                continue
            if ccam_key not in dict_need_annot_range or joint_name not in dict_need_annot_range[ccam_key]:
                continue

            cleaned_segments = dict_need_annot_range[ccam_key][joint_name]['segments_cleaned']

            for segment_str in cleaned_segments:
                start_str, end_str = segment_str.split('-', 1)
                start_frame, end_frame = int(start_str), int(end_str)
                
                segment_dir = os.path.join(vis_root_dir, f"{start_frame}_{end_frame}_{ccam_key}_{joint_name}")
                if os.path.exists(segment_dir):
                    shutil.rmtree(segment_dir)
                os.makedirs(segment_dir, exist_ok=True)

                logger.info(f"Visualizing joint {joint_name} in frames {start_frame}-{end_frame} for camera {ccam_key}")
                for local_idx, frame_id in enumerate(range(start_frame,end_frame+1)):
                    frame_key = frame_id if frame_id in json_data else str(frame_id)
                    v = json_data[frame_key]
                    cframe_kpts3d_valid = v['refined_3d'].copy()
                    cframe_kpts3d = cframe_kpts3d_valid[...,:3]
                    cvalid_3d = cframe_kpts3d_valid[...,3]>1e-3

                    vitpose_2d = v['vitpose_2d'][ccam_key]

                    ccam_param = cam_params[ccam_key]
                    rvec = ccam_param['rvec']
                    tvec = ccam_param['tvec']
                    K = ccam_param['K']


                    projected_2d, _ = cv2.projectPoints(cframe_kpts3d.reshape(-1,1,3), rvec, tvec, K, None)
                    projected_2d = projected_2d.reshape(-1,2)

                    projected_2d[~cvalid_3d] = -1
                    vitpose_2d[~cvalid_3d] = -1

                    ccam_bgr = dataset.load_bgr_image_and_undistort(f"{ccam_key}/frame_{int(frame_id):04d}.png")
                    
                    highlight_flags = np.zeros(len(VIT_JOINTS_NAME), dtype=bool)
                    highlight_flags[joint_idx] = True

                    vis_img = visualise_annotation(ccam_bgr, vitpose_2d, projected_2d, highlight_flags, threshold_det_conf, frame_id)


                    vis_path = os.path.join(segment_dir, f"frame_{local_idx:04d}.png")
                    cv2.imwrite(vis_path, vis_img)


                video_filename = f"{recording_name}_{joint_name}_{start_frame}_{end_frame}_{ccam_key}.mp4"
                video_path = os.path.join(vis_root_dir, video_filename)

                convert_images_to_video(video_path, segment_dir, img_tag="frame", fps=30, use_yuv420p=True, rm_dir_images=True)

