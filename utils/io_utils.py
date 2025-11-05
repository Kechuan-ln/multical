import os
import subprocess
import json
import glob
import tempfile
import numpy as np
import cv2


def convert_video_to_images(path_video, dir_images, fps=None, duration=None, ss=None, image_format='png', quality=2):
    """
    Convert video to image sequence.

    Args:
        path_video: Path to input video file
        dir_images: Directory to save extracted images
        fps: Frame rate for extraction (optional)
        duration: Duration in seconds to extract (optional)
        ss: Start offset in seconds (optional)
        image_format: Output image format ('png' or 'jpg'), default 'png'
        quality: JPEG quality (2-31, lower is better), only for jpg format, default 2
    """
    if not os.path.exists(dir_images):
        os.makedirs(dir_images)

    command = ["ffmpeg",  "-i", path_video]
    if ss is not None:
        command += ["-ss", str(ss)]

    if duration is not None:
        command += ["-t", str(duration)]

    if fps is not None:
        command += ["-vf", f"fps={fps}"]

    # Add quality settings for JPEG
    if image_format.lower() in ['jpg', 'jpeg']:
        command += ["-q:v", str(quality)]
        output_pattern = f"{dir_images}/frame_%04d.jpg"
    else:
        output_pattern = f"{dir_images}/frame_%04d.png"

    command += [output_pattern]
    print("COMMAND:", ' '.join(command))

    subprocess.run(command)


def convert_images_to_video(path_video, dir_images, img_tag, fps, use_yuv420p=False, rm_dir_images=False):
    image_pattern = os.path.join(dir_images, f"{img_tag}_%04d.png")

    if use_yuv420p:
        first_pass_cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", image_pattern,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-y", path_video,
        ]
    else:
        first_pass_cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", image_pattern,
            "-c:v", "libx264",
            "-y", path_video,
        ]
    subprocess.run(first_pass_cmd)

    if rm_dir_images:
        subprocess.run(["rm", "-rf", dir_images])
    

def stack_videos_grid(list_src_videos, path_output):
    # FFmpeg command initialization
    cmd = ["ffmpeg"]
    
    # Add all video inputs to the command
    for path in list_src_videos:
        cmd.extend(["-i", path])

    cmd.extend(['-filter_complex', (
            '[0:v]scale=1920:1080[v0]; '
            '[1:v]scale=1920:1080[v1]; '
            '[2:v]scale=1920:1080[v2]; '
            '[3:v]scale=1920:1080[v3]; '
            '[v0][v1]hstack=inputs=2[top]; '
            '[v2][v3]hstack=inputs=2[bottom]; '
            '[top][bottom]vstack=inputs=2'),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        path_output,
    ])

    # Run the command
    subprocess.run(cmd)


def stack_videos_horizontally(video_paths, output_path):
    # FFmpeg command initialization
    cmd = ["ffmpeg"]
    
    # Add all video inputs to the command
    for path in video_paths:
        cmd.extend(["-i", path])

    # Create the filter_complex argument
    num_videos = len(video_paths)
    filter_str = ';'.join([f"[{i}:v]scale=1920:1080[s{i}]" for i in range(num_videos)]) + ';'  # Scale each video (optional)
    filter_str += ''.join(f"[s{i}]" for i in range(num_videos))
    filter_str += f"hstack=inputs={num_videos}[v]"
    
    cmd.extend(["-filter_complex", filter_str, "-map", "[v]", "-y", output_path])

    # Execute the FFmpeg command
    subprocess.run(cmd)



def load_frames_from_video(video_path, start_frame_idx=0, end_frame_idx=-1, step=1):
    cap = cv2.VideoCapture(video_path)
    print("Loading video from", video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(0)

    list_frames=[]
    cnt=0
    while True:
        ret, frame = cap.read()
        cnt+=1
        if not ret or (end_frame_idx!=-1 and cnt>=end_frame_idx):
            break
        if cnt<start_frame_idx or (cnt-start_frame_idx)%step!=0:
            continue
            
        list_frames.append(frame)
        
    cap.release()
    return list_frames


def save_into_video(video_path, list_images, fps):
    height, width, _ = list_images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in list_images:
        video_writer.write(image)

    video_writer.release()


def extract_frames(root_dir, out_dir, input_tags, output_tags, start_idx, end_idx):
    for in_tag, out_tag in zip(input_tags, output_tags):
        print(in_tag, out_tag)
        os.makedirs(os.path.join(out_dir, out_tag), exist_ok=True)
        for frame_idx in range(start_idx, end_idx):
            path_cframe=os.path.join(root_dir, in_tag, "frame_{:04d}.png".format(frame_idx))
            path_oframe=os.path.join(out_dir, out_tag, "frame_{:04d}.png".format(frame_idx))
            print(path_cframe, path_oframe)
            subprocess.run(["cp", path_cframe, path_oframe])




class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)



def load_manual_2d_bbox_json(dir_detection_json, seq_name):
    # load 2D bounding box detections
    dict_det2ds = {}
    json_files = glob.glob(os.path.join(dir_detection_json, f"{seq_name}_*.json"))
    json_files.sort()  # Sort files to ensure consistent order
    set_frame_ids = set()
    for cpath_json in json_files:
        # Extract camera info from filename (assuming format like "seq_name_cam00.json")
        cam_key = cpath_json.split('/')[-1].replace('.json', '').split('_')[-1]  #
        # Find all JSON files that matc
        with open(cpath_json, 'r') as f:
            json_data = json.load(f)
                                       
            dict_det2ds[cam_key] = json_data
            set_frame_ids.update(json_data.keys())

    set_frame_ids = sorted(set_frame_ids, key=lambda x: int(x))
    
    return dict_det2ds, set_frame_ids


def load_yolo_track_json(path_json_seq):
    with open(path_json_seq, 'r') as f:
        json_data = json.load(f)
    
    return_data = {}
    for frame_key in json_data:
        return_data[frame_key] = {}
        return_data[frame_key]['auto_detect'] = json_data[frame_key]
    
    return return_data
            

def load_vitpose_json(prefix_json_seq):
    json_files = glob.glob(prefix_json_seq+'*.json')
    json_files.sort()  # Sort files to ensure consistent order
    all_json_data = {}
    for path_json in json_files:
        with open(path_json, 'r') as f:
            json_data = json.load(f)
        
        for frame_key in json_data:
            frame_data = json_data[frame_key]
            for kk in ['vitpose_2d', 'annotation_2d', 'reproj_err','triangulated_3d', 'refined_3d']:
                if kk not in frame_data:
                    continue
                if isinstance(frame_data[kk], dict):
                    for cam_key, v in frame_data[kk].items():
                        if isinstance(v, list):
                            frame_data[kk][cam_key] = np.array(v)
                elif isinstance(frame_data[kk], list):
                    frame_data[kk] = np.array(frame_data[kk])
            
            
            # Merge data to all_json_data
            if frame_key not in all_json_data:
                all_json_data[frame_key] = frame_data
            else:
                for k, v in frame_data.items():
                    # extend the dictionary
                    all_json_data[frame_key][k].update(v)
    return all_json_data




def load_manual_keypoint_json(prefix_json_seq):
    paths_json_seq = glob.glob(prefix_json_seq+'*.json')
    paths_json_seq.sort()  # Sort files to ensure consistent order
    
    json_data = {}
    for cpath_json in paths_json_seq:
        with open(cpath_json, 'r') as f:
            cjson_data = json.load(f)
            for frame_key in cjson_data:
                frame_data = cjson_data[frame_key]
                for kk in ['manual_2d','manual_flag','need_annot_flag']:
                    if kk not in frame_data:
                        continue
                    if isinstance(frame_data[kk], dict):
                        for cam_key, v in frame_data[kk].items():
                            if isinstance(v, list):
                                frame_data[kk][cam_key] = np.array(v)
                    elif isinstance(frame_data[kk], list):
                        frame_data[kk] = np.array(frame_data[kk])
                    
                json_data[frame_key] = frame_data

    return json_data


def load_3d_keypoint_json(prefix_json_seq):
    return load_vitpose_json(prefix_json_seq)
