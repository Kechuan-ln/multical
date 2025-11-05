import os
import glob
from tqdm import tqdm
import numpy as np
import cv2
import subprocess
import re
import json

np.set_printoptions(precision=6, suppress=True)



def extract_timecode(video_path):
    # Command to extract timecode using ffprobe
    command = [
        "ffprobe",
        "-v", "error",                  # Suppress unnecessary output
        "-select_streams", "v:0",       # Select the first video stream
        "-show_entries", "stream_tags=timecode",  # Show timecode from tags
        "-of", "default=noprint_wrappers=1:nokey=1",  # Clean output
        video_path
    ]
    
    # Run the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # The timecode should be in the standard output
    timecode = result.stdout.strip()
    if timecode:
        return timecode
    else:
        return "Timecode not found."


def get_video_length(video_path):
    # Run ffprobe and capture the output
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Extract the duration from the output
    duration = float(result.stdout.strip())
    return duration


def get_fps(video_path):
    # Run ffprobe command
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Extract and calculate FPS
    fps_fraction = result.stdout.strip()
    num, denom = map(int, fps_fraction.split('/'))
    fps = num / denom
    
    return fps

def timecode_to_seconds(tc, fps):
    hours, minutes, seconds, frames = map(int, re.split('[:;]', tc))
    return hours*3600 + minutes*60 + seconds + frames/fps  



def synchronize_cameras(list_src_videos):
    timecodes = [extract_timecode(vp) for vp in list_src_videos]
    assert not (None in timecodes), "Some videos do not have timecodes."
    durations = [get_video_length(vp) for vp in list_src_videos]


    fps = get_fps(list_src_videos[0])
    for i in range(1, len(list_src_videos)):
        assert fps == get_fps(list_src_videos[i]), "All videos should have the same FPS."
    
    fps = round(fps)

    # Convert timecodes to seconds
    start_times = [timecode_to_seconds(tc, fps=fps) for tc in timecodes]
    end_times = [start_times[i] + durations[i] for i in range(len(start_times))]

    # Find the maximum start time, as we'll sync all videos to this point
    max_start = max(start_times)
    min_end = min(end_times)
    duration = min_end - max_start
    
    print(f"Sync window: {duration:.2f} seconds starting at {max_start:.2f}s")

    meta_info = {}
    for i, path_video in enumerate(list_src_videos):
        offset = max_start - start_times[i]
        assert offset >= 0, "The offset should be positive."

        # 处理两种情况：
        # 1. path/cam01/video.MP4 -> video_tag = "cam01/video.MP4"
        # 2. path/Cam01/video.MP4 -> video_tag = "Cam01/video.MP4"
        # 3. path/cam01.MP4 -> video_tag = "cam01.MP4"
        video_basename = os.path.basename(path_video)
        parent_dir = os.path.basename(os.path.dirname(path_video))

        # 如果父目录看起来像相机名（cam或Cam开头），保留目录结构
        if parent_dir.lower().startswith('cam'):
            video_tag = os.path.join(parent_dir, video_basename)
        else:
            # 否则直接用文件名
            video_tag = video_basename

        meta_info[video_tag] = {"src_timecode": timecodes[i],"src_duration": durations[i],"offset": offset,"duration": duration,"fps": fps}

    return meta_info


def undistort_cameras_from_json(path_intr):
    with open(path_intr, 'r') as f:
        json_intrinsics = json.load(f)
    
    # Create undistorted intrinsics
    json_intrinsics_undist = json.loads(json.dumps(json_intrinsics))
    for cam_key in json_intrinsics['cameras']:
        w, h = json_intrinsics['cameras'][cam_key]['image_size']
        K = np.array(json_intrinsics['cameras'][cam_key]['K'], dtype=np.float32)
        dist = np.array(json_intrinsics['cameras'][cam_key]['dist'], dtype=np.float32).flatten()
        
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        
        json_intrinsics_undist['cameras'][cam_key]['K'] = new_K.tolist()
        json_intrinsics_undist['cameras'][cam_key]['dist'] = [[0.0, 0.0, 0.0, 0.0, 0.0]]
    
    
    # Save undistorted intrinsics
    path_intr_undist = path_intr.replace('.json', '_undistorted.json')
    with open(path_intr_undist, 'w') as f:
        json.dump(json_intrinsics_undist, f, indent=2)
        
    return json_intrinsics, json_intrinsics_undist