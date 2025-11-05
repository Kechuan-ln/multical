#!/usr/bin/env python3
"""
GoPro外参标定自动化脚本
- 使用multical conda环境
- 使用GoPro timecode同步
- 自动处理cam1内参缺失问题
"""

import os
import sys
import subprocess
import json

# 配置
SOURCE_DIR = "/Volumes/FastACIS/csltest1/gopros"
OUTPUT_DIR = "/Volumes/FastACIS/csltest1/output"
INTRINSIC_JSON = "/Volumes/FastACIS/annotation_pipeline/intrinsic_hyperoff_linear_60fps.json"
WORK_DIR = "/Volumes/FastACIS/annotation_pipeline"
CAMERAS = ["cam1", "cam2", "cam3", "cam5"]

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def run_command(cmd, description=None, env=None):
    """在multical conda环境中运行命令"""
    if description:
        print(f"\n{description}")

    # 构建完整命令（激活conda环境）
    conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate multical"
    full_cmd = f"{conda_activate} && {cmd}"

    print(f"$ {cmd}")
    result = subprocess.run(full_cmd, shell=True, executable='/bin/bash',
                          capture_output=False, text=True, env=env)

    if result.returncode != 0:
        print(f"❌ 命令执行失败: {cmd}")
        return False
    return True

def check_environment():
    """检查conda环境"""
    print_section("检查环境")

    if not run_command("which python", "检查Python路径"):
        print("❌ multical conda环境未激活")
        sys.exit(1)

    if not run_command("which ffprobe", "检查ffprobe"):
        print("❌ ffprobe未安装（需要ffmpeg）")
        sys.exit(1)

    if not run_command("python --version", "检查Python版本"):
        sys.exit(1)

    print("✅ 环境检查通过")

def check_videos():
    """检查视频文件和timecode"""
    print_section("检查视频文件")

    # 检查文件存在
    for cam in CAMERAS:
        video_path = os.path.join(SOURCE_DIR, cam, "calibration.MP4")
        if not os.path.exists(video_path):
            print(f"❌ 视频不存在: {video_path}")
            sys.exit(1)

        size_mb = os.path.getsize(video_path) / 1024 / 1024
        print(f"  ✓ {cam}/calibration.MP4 ({size_mb:.1f} MB)")

    print("\n检查timecode...")

    # 创建临时目录并链接视频
    video_dir = os.path.join(OUTPUT_DIR, "calibration_videos")
    os.makedirs(video_dir, exist_ok=True)

    timecodes = {}
    for cam in CAMERAS:
        cam_dir = os.path.join(video_dir, cam)
        os.makedirs(cam_dir, exist_ok=True)

        src = os.path.join(SOURCE_DIR, cam, "calibration.MP4")
        dst = os.path.join(cam_dir, "calibration.MP4")

        # 创建符号链接
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)

        # 检查timecode
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream_tags=timecode -of default=noprint_wrappers=1:nokey=1 "{dst}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        timecode = result.stdout.strip()

        if not timecode:
            print(f"  ❌ {cam}: 无timecode")
            print("\n错误: GoPro视频必须有timecode才能同步！")
            print("请检查:")
            print("  1. GoPro是否开启了timecode功能")
            print("  2. 视频是否是原始GoPro录制文件")
            sys.exit(1)

        timecodes[cam] = timecode
        print(f"  ✓ {cam}: {timecode}")

    print("✅ 所有视频都有timecode")
    return video_dir

def sync_videos():
    """使用timecode同步视频"""
    print_section("步骤1: GoPro Timecode同步")

    # 切换到工作目录（重要！需要在这个目录才能导入utils模块）
    original_dir = os.getcwd()
    os.chdir(WORK_DIR)

    # 设置环境变量
    env = os.environ.copy()
    env['PATH_ASSETS_VIDEOS'] = OUTPUT_DIR
    env['PYTHONPATH'] = WORK_DIR  # 添加Python路径

    # 使用cd && 确保在正确目录执行
    # 添加--stacked参数来生成并排视频预览
    cmd = f'cd "{WORK_DIR}" && python scripts/sync_timecode.py --src_tag "calibration_videos" --out_tag "calibration_synced" --stacked'

    if not run_command(cmd, "同步视频中...", env=env):
        print("❌ 同步失败")
        os.chdir(original_dir)
        sys.exit(1)

    # 显示同步信息
    meta_file = os.path.join(OUTPUT_DIR, "calibration_synced", "meta_info.json")
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        print("\n同步信息:")
        for cam_path, info in meta.get('info_cam', {}).items():
            cam = cam_path.split('/')[-2]
            print(f"  {cam}: offset={info.get('offset', 0):.2f}s, duration={info.get('duration', 0):.2f}s")

    print("✅ 视频同步完成")

def extract_frames():
    """提取视频帧"""
    print_section("步骤2: 提取视频帧")

    env = os.environ.copy()
    env['PATH_ASSETS_VIDEOS'] = OUTPUT_DIR
    env['PYTHONPATH'] = WORK_DIR

    cam_tags = ",".join(CAMERAS)
    # 修改为15fps间隔（即每15帧取1帧，相当于1fps at 15fps视频或4fps at 60fps视频）
    # 不指定duration，提取整个视频
    # 这样可以获得更多样化的标定板姿态
    cmd = f'cd "{WORK_DIR}" && python scripts/convert_video_to_images.py --src_tag "calibration_synced" --cam_tags "{cam_tags}" --fps 4'

    if not run_command(cmd, "提取关键帧中（每15帧取1帧，约4fps）...", env=env):
        print("❌ 图像提取失败")
        sys.exit(1)

    # 统计图像数量
    print("\n图像统计:")
    for cam in CAMERAS:
        cam_dir = os.path.join(OUTPUT_DIR, "calibration_synced", "original", cam)
        if os.path.exists(cam_dir):
            images = [f for f in os.listdir(cam_dir) if f.endswith('.png')]
            print(f"  {cam}: {len(images)} 张图像")

    print("✅ 图像提取完成")

def check_intrinsics():
    """检查内参文件并创建过滤后的版本"""
    print_section("检查内参文件")

    with open(INTRINSIC_JSON, 'r') as f:
        intrinsics = json.load(f)

    cameras_in_json = intrinsics.get('cameras', {}).keys()
    print(f"预存内参包含相机: {', '.join(sorted(cameras_in_json))}")

    # 检查哪些相机存在于图像文件夹中
    image_dir = os.path.join(OUTPUT_DIR, "calibration_synced", "original")
    existing_cams = []
    for cam in CAMERAS:
        cam_dir = os.path.join(image_dir, cam)
        if os.path.exists(cam_dir) and os.listdir(cam_dir):
            existing_cams.append(cam)

    print(f"图像文件夹中的相机: {', '.join(existing_cams)}")

    # 创建只包含现有相机的过滤内参文件
    filtered_intrinsics = {'cameras': {}}
    missing_cams = []

    for cam in existing_cams:
        if cam in intrinsics['cameras']:
            filtered_intrinsics['cameras'][cam] = intrinsics['cameras'][cam]
        else:
            missing_cams.append(cam)

    if missing_cams:
        print(f"\n⚠️  警告: 以下相机没有预存内参: {', '.join(missing_cams)}")
        print("\n处理方案:")
        print("  1. [推荐] 删除这些相机的图像，只用其他相机标定")
        print("  2. 先单独标定这些相机的内参，然后合并JSON")

        response = input("\n是否删除缺失内参的相机图像？(y/n): ").strip().lower()

        if response == 'y':
            for cam in missing_cams:
                cam_dir = os.path.join(image_dir, cam)
                if os.path.exists(cam_dir):
                    import shutil
                    shutil.rmtree(cam_dir)
                    print(f"  ✓ 已删除 {cam}")
                    existing_cams.remove(cam)

            # 重新创建过滤文件
            filtered_intrinsics = {'cameras': {}}
            for cam in existing_cams:
                if cam in intrinsics['cameras']:
                    filtered_intrinsics['cameras'][cam] = intrinsics['cameras'][cam]
        else:
            print("⚠️  保留所有相机，标定可能失败")

    # 保存过滤后的内参文件
    filtered_path = os.path.join(OUTPUT_DIR, "intrinsic_filtered.json")
    with open(filtered_path, 'w') as f:
        json.dump(filtered_intrinsics, f, indent=2)

    print(f"\n✅ 已创建过滤后的内参文件: {filtered_path}")
    print(f"包含相机: {', '.join(sorted(filtered_intrinsics['cameras'].keys()))}")

    return existing_cams

def run_calibration():
    """运行外参标定"""
    print_section("步骤3: 外参标定")

    env = os.environ.copy()
    env['PATH_ASSETS_VIDEOS'] = OUTPUT_DIR
    env['PYTHONPATH'] = WORK_DIR

    multical_dir = os.path.join(WORK_DIR, "multical")

    # 使用过滤后的内参文件和更多的图像帧
    filtered_intrinsic = os.path.join(OUTPUT_DIR, "intrinsic_filtered.json")
    cmd = f'cd "{multical_dir}" && python calibrate.py --boards ./asset/charuco_b3.yaml --image_path "calibration_synced/original" --calibration "{filtered_intrinsic}" --fix_intrinsic --limit_images 1000 --vis'

    print("开始外参标定...")
    print("使用最多1000帧图像")
    print("这可能需要几分钟，请耐心等待...")

    if not run_command(cmd, env=env):
        print("❌ 标定失败，请查看日志")
        sys.exit(1)

    print("✅ 标定完成")

def check_results():
    """检查结果"""
    print_section("完成！")

    calib_file = os.path.join(OUTPUT_DIR, "calibration_synced", "original", "calibration.json")

    if not os.path.exists(calib_file):
        print("❌ 标定文件不存在")
        sys.exit(1)

    print("✅ 外参标定成功！")
    print("\n输出文件:")
    print(f"  - 标定结果: {calib_file}")
    print(f"  - 图像帧: {OUTPUT_DIR}/calibration_synced/original/")
    print(f"  - 可视化: {OUTPUT_DIR}/calibration_synced/vis/")

    print("\n下一步:")
    print("  1. 查看vis/目录中的可视化结果验证标定质量")
    print("  2. 检查calibration.json中的RMS误差（应该<1像素）")
    print("  3. 使用这个calibration.json进行3D姿态估计")

    # 尝试显示RMS
    try:
        with open(calib_file, 'r') as f:
            calib = json.load(f)
        if 'rms' in calib:
            print(f"\n📊 RMS误差: {calib['rms']:.3f} 像素")
    except:
        pass

def main():
    print("=" * 60)
    print("GoPro外参标定自动化脚本")
    print("=" * 60)
    print("使用: multical conda环境")
    print("同步: GoPro timecode")
    print("输出: /Volumes/FastACIS/csltest1/output/")

    try:
        check_environment()
        check_videos()
        sync_videos()
        extract_frames()
        check_intrinsics()
        run_calibration()
        check_results()

        print("\n" + "=" * 60)
        print("✅ 所有步骤完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
