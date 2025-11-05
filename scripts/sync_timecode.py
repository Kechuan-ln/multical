import os
import subprocess
import argparse
import glob
import json
import sys

# 动态添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils.io_utils import stack_videos_grid, stack_videos_horizontally
from utils.calib_utils import synchronize_cameras
from utils.constants import PATH_ASSETS_VIDEOS
    


def synchronize_videos(list_src_videos, out_dir, sync_mode='ultrafast'):
    """
    同步多个视频

    Args:
        sync_mode: 'fast_copy', 'ultrafast', 或 'accurate'
            - fast_copy: 最快，关键帧精度（可能有0-2秒误差）
            - ultrafast: 快速且帧精确（推荐，默认）
            - accurate: 最慢但最精确（medium preset）
    """
    list_output_videos=[]
    meta_info = synchronize_cameras(list_src_videos)

    print(meta_info)

    for i, path_video in enumerate(list_src_videos):
        # 处理两种情况：
        # 1. path/cam01/video.MP4 -> video_tag = "cam01/video.MP4"
        # 2. path/cam01.MP4 -> video_tag = "cam01.MP4"
        video_basename = os.path.basename(path_video)
        parent_dir = os.path.basename(os.path.dirname(path_video))

        # 如果父目录看起来像相机名（cam或Cam开头），保留目录结构
        if parent_dir.lower().startswith('cam'):
            video_tag = os.path.join(parent_dir, video_basename)
        else:
            # 否则直接用文件名
            video_tag = video_basename

        cmeta = meta_info[video_tag]
        offset = cmeta['offset']
        duration = cmeta['duration']

        path_output = os.path.join(out_dir, video_tag)
        if not os.path.exists(os.path.dirname(path_output)):
            os.makedirs(os.path.dirname(path_output))
        list_output_videos.append(path_output)


        # The video starts earlier than our sync point, so we'll trim the beginning
        if sync_mode == 'fast_copy':
            # 方法1: 纯copy（最快，但只有关键帧精度）
            cmd = ["ffmpeg", "-i", path_video, "-ss", str(offset), "-t", str(duration),
                   "-c:v", "copy", "-c:a", "copy", "-y", path_output]
        elif sync_mode == 'accurate':
            # 方法3: 最精确但慢（medium preset）
            # 使用 -ss 在 -i 之前 + medium preset
            cmd = ["ffmpeg", "-ss", str(offset), "-i", path_video, "-t", str(duration),
                   "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                   "-c:a", "copy", "-pix_fmt", "yuv420p", "-y", path_output]
        else:  # ultrafast (默认)
            # 方法2: 快速+帧精确（推荐）
            # 使用 -ss 在 -i 之前 + ultrafast preset
            cmd = ["ffmpeg", "-ss", str(offset), "-i", path_video, "-t", str(duration),
                   "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                   "-c:a", "copy", "-pix_fmt", "yuv420p", "-y", path_output]
        subprocess.run(cmd)

    return meta_info, list_output_videos



def args():
    parser = argparse.ArgumentParser(description='Synchronize GoPro videos using timecodes.')
    #parser.add_argument('--root_dir', type=str, default='./assets/', help='root dir of assets')
    parser.add_argument('--src_tag', type=str, default= 'ori', help='subdir containing videos from multiple cameras')
    parser.add_argument('--out_tag', type=str, default= 'sync',help='subdir to save results')
    parser.add_argument('--stacked', action='store_true', help='stacked videos visualization')

    # 同步模式选项（新参数）
    parser.add_argument('--sync_mode', type=str, default='ultrafast',
                       choices=['fast_copy', 'ultrafast', 'accurate'],
                       help='''视频同步模式:
                           fast_copy - 最快（关键帧精度，可能有0-2秒误差）
                           ultrafast - 快速且帧精确（推荐，默认）
                           accurate  - 最慢但最精确（medium preset）''')

    # 保留向后兼容（旧参数）
    parser.add_argument('--fast_copy', action='store_true',
                       help='（已弃用，使用 --sync_mode fast_copy 代替）')
    parser.add_argument('--accurate', action='store_true',
                       help='（已弃用，使用 --sync_mode accurate 代替）')

    args = parser.parse_args()

    # 向后兼容处理
    if args.fast_copy:
        args.sync_mode = 'fast_copy'
    elif args.accurate:
        args.sync_mode = 'accurate'

    return args

if __name__ == '__main__':
    args = args()

    # 支持绝对路径：如果src_tag是绝对路径，直接使用；否则相对于PATH_ASSETS_VIDEOS
    if os.path.isabs(args.src_tag):
        src_dir = args.src_tag
    else:
        root_dir = PATH_ASSETS_VIDEOS
        src_dir = os.path.join(root_dir, args.src_tag)

    # 支持绝对路径：如果out_tag是绝对路径，直接使用；否则相对于PATH_ASSETS_VIDEOS
    if os.path.isabs(args.out_tag):
        out_dir = args.out_tag
    else:
        root_dir = PATH_ASSETS_VIDEOS
        out_dir = os.path.join(root_dir, args.out_tag)

    os.makedirs(out_dir, exist_ok=True)

    # 支持两种目录结构：
    # 1. src_dir/cam*/video.MP4 (原始结构)
    # 2. src_dir/*.MP4 (平铺结构)
    list_src_videos = glob.glob(os.path.join(src_dir,'*/*.MP4'))
    if not list_src_videos:
        list_src_videos = glob.glob(os.path.join(src_dir,'*.MP4'))

    list_src_videos.sort()
    print("list_src_videos:",list_src_videos)

    meta_cam_info, list_output_videos = synchronize_videos(list_src_videos, out_dir, args.sync_mode)
    meta_info={"dir_src":src_dir,"dir_out":out_dir,"info_cam":meta_cam_info}

    print("meta_info:",meta_info)
    with open(os.path.join(out_dir,'meta_info.json'), 'w') as f:
        json.dump(meta_info, f, separators=(',', ':'))
    if args.stacked:
        #stack_videos_grid(list_output_videos, os.path.join(out_dir,"stacked_output_h.MP4"))
        stack_videos_horizontally(list_output_videos, os.path.join(out_dir,"stacked_output.mp4"))
   