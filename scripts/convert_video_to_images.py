import os
import argparse
import glob
import cv2
from tqdm import tqdm

import sys
import shutil
import numpy as np

# 动态添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

np.set_printoptions(precision=6, suppress=True)

from utils.io_utils import convert_video_to_images
from utils.calib_utils import undistort_cameras_from_json
from utils.constants import PATH_ASSETS_VIDEOS

def process_video_folder(src_dir, target_dir, list_cam_tags, fps, duration, ss, path_intr=None, image_format='png', quality=2):
    if path_intr is not None:
        json_intrinsics, json_intrinsics_nodist = undistort_cameras_from_json(path_intr)
    else:
        json_intrinsics, json_intrinsics_nodist = None, None

    for cam_tag in list_cam_tags:
        cam_path = os.path.join(src_dir, cam_tag)

        # Support multiple video formats: MP4, mp4, avi, AVI
        mp4_files = (glob.glob(os.path.join(cam_path, "*.MP4")) +
                     glob.glob(os.path.join(cam_path, "*.mp4")) +
                     glob.glob(os.path.join(cam_path, "*.avi")) +
                     glob.glob(os.path.join(cam_path, "*.AVI")))

        for path_video in mp4_files:
            video_name = os.path.splitext(os.path.basename(path_video))[0]
            if len(mp4_files)> 1:
                dir_images = os.path.join(target_dir, "original", cam_tag, video_name)
            else:
                dir_images = os.path.join(target_dir, "original", cam_tag)
            if os.path.exists(dir_images):
                # If the directory exists, remove it to avoid mixing old and new images
                shutil.rmtree(dir_images)

            os.makedirs(dir_images, exist_ok=True)

            print(f"Converting {path_video} to images in {dir_images} at {fps} fps (format: {image_format})")
            convert_video_to_images(path_video, dir_images, fps, duration, ss, image_format=image_format, quality=quality)

            if json_intrinsics is None:
                continue

            # Undistort images if intrinsics are provided
            if len(mp4_files) > 1:
                dir_output = os.path.join(target_dir, "undistorted", cam_tag, video_name)
            else:
                dir_output = os.path.join(target_dir, "undistorted", cam_tag)
            if os.path.exists(dir_output):
                shutil.rmtree(dir_output)
            os.makedirs(dir_output, exist_ok=True)

            # Support both PNG and JPG
            image_files = (glob.glob(os.path.join(dir_images, "*.png")) +
                          glob.glob(os.path.join(dir_images, "*.jpg")) +
                          glob.glob(os.path.join(dir_images, "*.jpeg")))
            image_files.sort() 
            ori_K = np.array(json_intrinsics['cameras'][cam_tag]['K'], dtype=np.float32)
            ori_dist = np.array(json_intrinsics['cameras'][cam_tag]['dist'], dtype=np.float32).flatten()
            
            new_K = np.array(json_intrinsics_nodist['cameras'][cam_tag]['K'], dtype=np.float32)

            for image_path in tqdm(image_files):
                img = cv2.imread(image_path)      
                undistorted_img = cv2.undistort(img, ori_K, ori_dist, None, new_K)

                path_output = os.path.join(dir_output, os.path.basename(image_path))
                cv2.imwrite(path_output, undistorted_img)


def main():
    parser = argparse.ArgumentParser(description="Convert MP4 videos to images")
    parser.add_argument("--src_tag", default='intr_09121',  help="Source directory containing camera folders, under PATH_ASSETS_VIDEOS")
    parser.add_argument("--cam_tags", default="cam01,cam02,cam03", help="Comma-separated list of camera tags to process")
    parser.add_argument("--fps", type=float, required=True, help="Frames per second for extraction")
    parser.add_argument("--duration", type=str, default=None, help="Duration in seconds to extract from the video (optional)")
    parser.add_argument("--ss", type=str, default=None, help="Offset to start extracting from the video (optional)")
    parser.add_argument("--path_intr", help="Path to intrinsics.json file for undistortion (optional), if provided, under PATH_ASSETS_VIDEOS")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "jpeg"], help="Output image format (default: png)")
    parser.add_argument("--quality", type=int, default=2, help="JPEG quality (2-31, lower is better, only for jpg), default: 2")

    args = parser.parse_args()

    src_dir = os.path.join(PATH_ASSETS_VIDEOS, args.src_tag)
    cam_tags = args.cam_tags.split(',')
    if args.path_intr is not None:
        args.path_intr = os.path.join(PATH_ASSETS_VIDEOS, args.path_intr)
        assert os.path.exists(args.path_intr), f"Intrinsics file does not exist: {args.path_intr}"
    process_video_folder(src_dir, src_dir, cam_tags, args.fps, args.duration, args.ss, args.path_intr,
                        image_format=args.format, quality=args.quality)


if __name__ == "__main__":
    main()
