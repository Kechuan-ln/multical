#!/usr/bin/env python3
"""
GoPro Extrinsic Calibration - Unified Script
=============================================

Features:
1. Video synchronization (timecode-based)
2. Video to images (JPG support)
3. Optional: Stable frame detection
4. Auto-filter intrinsics JSON
5. Extrinsic calibration
6. Visualization
7. Cleanup temporary files

Usage:
1. Modify the "CONFIGURATION" section below
2. Run: conda activate multical && python run_gopro_calibration.py
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import glob

# Add project path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from filter_intrinsics import filter_intrinsics, auto_detect_cameras

# ==============================================================================
# ============================= CONFIGURATION ==================================
# ==============================================================================

# ---------- Basic Path Configuration ----------
WORKING_DIR = "/Volumes/FastACIS/GoPro/gopro_extrinsic "  # Working directory (contains videos and intrinsics)
INTRINSIC_JSON = "Intrinsic-16.json"                       # Intrinsics filename
BOARD_CONFIG = "charuco_b1_2.yaml"                        # Board config

# ---------- Camera Selection ----------
# Option 1: Auto-detect cameras (recommended)
AUTO_DETECT_CAMERAS = True

# Option 2: Manual camera list (if AUTO_DETECT_CAMERAS=False)
MANUAL_CAMERA_LIST = ["Cam1", "Cam2", "Cam3", "Cam4"]

# ---------- Video Sync Parameters ----------
ENABLE_SYNC = True          # Enable video synchronization
START_TIME = 0              # Start time (seconds)
DURATION = 120              # Duration (seconds)
USE_FAST_COPY = True        # Fast copy (True: 1-2 frame error, faster)

# ---------- Image Extraction Parameters ----------
EXTRACT_FPS = 2             # Extraction frame rate
IMAGE_FORMAT = "jpg"        # Image format: jpg or png
JPEG_QUALITY = 2            # JPEG quality (2-31, lower is better)

# ---------- Stable Frame Detection (Optional) ----------
USE_STABLE_FRAMES = False        # Enable stable frame detection
MOVEMENT_THRESHOLD = 10.0        # Movement threshold (pixels)
MIN_DETECTION_QUALITY = 40       # Minimum detected corners
DOWNSAMPLE_RATE = 5              # Downsample interval

# ---------- Extrinsic Calibration Parameters ----------
FIX_INTRINSIC = True             # Fix intrinsics (recommended True)
LIMIT_IMAGES = 200               # Max images to use (None for all)
ENABLE_VIS = True                # Generate visualization

# ---------- Cleanup Options ----------
CLEANUP_TEMP = True              # Cleanup temporary files
KEEP_SYNC_VIDEOS = False         # Keep synchronized videos
KEEP_RAW_IMAGES = False          # Keep raw images (unfiltered)

# ==============================================================================
# ============================= MAIN PROGRAM ===================================
# ==============================================================================

def print_section(title):
    """Print section separator"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_command(cmd, description, cwd=None):
    """Run command and print output"""
    print(f">> {description}")
    print(f"   Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed: {description}")
        sys.exit(1)
    print(f"OK: {description} completed\n")


def main():
    print_section("GoPro Extrinsic Calibration - Starting")

    # Validate working directory
    working_dir = Path(WORKING_DIR).resolve()

    if not working_dir.exists():
        print(f"ERROR: Working directory does not exist: {WORKING_DIR}")
        print(f"       Resolved path: {working_dir}")
        sys.exit(1)

    print(f"Working directory: {working_dir}")

    # Create output directory structure
    output_dirs = {
        'sync': working_dir / 'sync',
        'images': working_dir / 'images_original',
        'images_stable': working_dir / 'images_stable',
        'calibration_output': working_dir / 'calibration_output'
    }

    for dir_path in output_dirs.values():
        dir_path.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Detect Cameras
    # -------------------------------------------------------------------------
    print_section("Step 1: Detect Cameras")

    if AUTO_DETECT_CAMERAS:
        print("Auto-detecting cameras...")
        camera_list = auto_detect_cameras(str(working_dir))
        if not camera_list:
            print(f"ERROR: No camera folders detected (Cam1/, Cam2/, etc.)")
            sys.exit(1)
    else:
        camera_list = MANUAL_CAMERA_LIST

    print(f"OK: Using cameras: {', '.join(camera_list)}")
    num_cameras = len(camera_list)

    # -------------------------------------------------------------------------
    # Step 2: Filter Intrinsics JSON
    # -------------------------------------------------------------------------
    print_section("Step 2: Filter Intrinsics")

    input_intrinsic = working_dir / INTRINSIC_JSON
    output_intrinsic = working_dir / "intrinsic_filtered.json"

    if not input_intrinsic.exists():
        print(f"ERROR: Intrinsics file does not exist: {input_intrinsic}")
        sys.exit(1)

    print(f"Input intrinsics: {input_intrinsic}")
    print(f"Output intrinsics: {output_intrinsic}")

    filter_intrinsics(
        str(input_intrinsic),
        str(output_intrinsic),
        camera_list
    )

    # -------------------------------------------------------------------------
    # Step 3: Video Synchronization (Optional)
    # -------------------------------------------------------------------------
    if ENABLE_SYNC:
        print_section("Step 3: Video Synchronization")

        # Check if sync is needed
        sync_dir = output_dirs['sync']
        needs_sync = True

        if sync_dir.exists() and len(list(sync_dir.glob('*/*.MP4'))) > 0:
            print("WARNING: Detected synchronized videos. Skip sync? (y/n)")
            print("   Enter 'y' to skip, other to re-sync")
            try:
                user_input = input("   Choice: ").strip().lower()
                if user_input == 'y':
                    needs_sync = False
                    print("OK: Skipping sync\n")
            except EOFError:
                print("INFO: No user input, proceeding with sync")

        if needs_sync:
            sync_cmd = [
                sys.executable, str(script_dir / "scripts" / "sync_timecode.py"),
                "--src_tag", str(working_dir.absolute()),
                "--out_tag", str(sync_dir.absolute())
            ]

            if USE_FAST_COPY:
                sync_cmd.append("--fast_copy")

            run_command(sync_cmd, "Synchronize videos")
            video_source_dir = sync_dir
        else:
            video_source_dir = working_dir
    else:
        print_section("Step 3: Skip Video Sync")
        video_source_dir = working_dir

    # -------------------------------------------------------------------------
    # Step 4: Video to Images
    # -------------------------------------------------------------------------
    print_section("Step 4: Video to Images")

    images_dir = output_dirs['images']

    # Build camera tag list (lowercase to match multical format)
    cam_tags_lower = [cam.lower() for cam in camera_list]

    # Create lowercase camera folder symlinks if needed
    print("Creating camera folder links...")
    for cam_upper, cam_lower in zip(camera_list, cam_tags_lower):
        src_cam_dir = video_source_dir / cam_upper
        link_cam_dir = video_source_dir / cam_lower

        if src_cam_dir.exists() and not link_cam_dir.exists():
            # Create symlink
            os.symlink(str(src_cam_dir), str(link_cam_dir))
            print(f"   {cam_upper} -> {cam_lower}")

    convert_cmd = [
        sys.executable, str(script_dir / "scripts" / "convert_video_to_images.py"),
        "--src_tag", str(video_source_dir.absolute()),
        "--cam_tags", ','.join(cam_tags_lower),
        "--fps", str(EXTRACT_FPS),
        "--format", IMAGE_FORMAT,
        "--quality", str(JPEG_QUALITY)
    ]

    if START_TIME > 0:
        convert_cmd.extend(["--ss", str(START_TIME)])

    if DURATION is not None:
        convert_cmd.extend(["--duration", str(DURATION)])

    run_command(convert_cmd, f"Extract images ({EXTRACT_FPS} fps, {IMAGE_FORMAT})")

    # Move images to correct location
    original_images = video_source_dir / "original"
    if original_images.exists():
        print("Moving images to output directory...")
        for cam_dir in original_images.iterdir():
            if cam_dir.is_dir():
                dest_dir = images_dir / cam_dir.name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.move(str(cam_dir), str(images_dir))
        if original_images.exists():
            original_images.rmdir()

    # -------------------------------------------------------------------------
    # Step 5: Stable Frame Detection (Optional)
    # -------------------------------------------------------------------------
    if USE_STABLE_FRAMES:
        print_section("Step 5: Stable Frame Detection")

        stable_cmd = [
            sys.executable, str(script_dir / "scripts" / "find_stable_boards.py"),
            "--recording_tag", str(images_dir.absolute()),
            "--boards", str(script_dir / "multical" / "asset" / BOARD_CONFIG),
            "--movement_threshold", str(MOVEMENT_THRESHOLD),
            "--min_detection_quality", str(MIN_DETECTION_QUALITY),
            "--downsample_rate", str(DOWNSAMPLE_RATE)
        ]

        # Capture output to extract stable frame indices
        print("Detecting stable frames...")
        result = subprocess.run(stable_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR: Stable frame detection failed")
            print(result.stderr)
            sys.exit(1)

        print(result.stdout)

        # Parse output to get stable frame indices
        stable_indices = []
        for line in result.stdout.split('\n'):
            if line.startswith("Stable frame indices:"):
                indices_str = line.split(":", 1)[1].strip()
                indices_str = indices_str.strip('[]')
                stable_indices = [int(x.strip()) for x in indices_str.split(',') if x.strip()]
                break

        if not stable_indices:
            print("WARNING: No stable frames detected, using all images")
        else:
            print(f"OK: Detected {len(stable_indices)} stable frames")

            # Save stable frames list
            stable_frames_file = working_dir / "stable_frames.txt"
            with open(stable_frames_file, 'w') as f:
                f.write(','.join(map(str, stable_indices)))
            print(f"Stable frames list saved: {stable_frames_file}")

            # Copy stable frames
            copy_cmd = [
                sys.executable, str(script_dir / "scripts" / "copy_image_subset.py"),
                "--image_path", str(images_dir),
                "--dest_path", str(output_dirs['images_stable']),
                "--frames", ','.join(map(str, stable_indices))
            ]

            run_command(copy_cmd, "Copy stable frames")

            # Use stable frames directory for calibration
            calibration_images_dir = output_dirs['images_stable']
    else:
        print_section("Step 5: Skip Stable Frame Detection")
        calibration_images_dir = images_dir

    # -------------------------------------------------------------------------
    # Step 6: Extrinsic Calibration
    # -------------------------------------------------------------------------
    print_section("Step 6: Extrinsic Calibration")

    calibrate_cmd = [
        sys.executable, str(script_dir / "multical" / "calibrate.py"),
        "--boards", str(script_dir / "multical" / "asset" / BOARD_CONFIG),
        "--image_path", str(calibration_images_dir.absolute()),
        "--calibration", str(output_intrinsic.absolute())
    ]

    if FIX_INTRINSIC:
        calibrate_cmd.append("--fix_intrinsic")

    if LIMIT_IMAGES is not None:
        calibrate_cmd.extend(["--limit_images", str(LIMIT_IMAGES)])

    if ENABLE_VIS:
        calibrate_cmd.append("--vis")

    run_command(calibrate_cmd, "Extrinsic calibration", cwd=str(script_dir / "multical"))

    # Copy calibration results
    calib_result = calibration_images_dir / "calibration.json"
    if calib_result.exists():
        final_calib = working_dir / "calibration.json"
        shutil.copy(str(calib_result), str(final_calib))
        print(f"OK: Calibration results saved: {final_calib}")

        # Display calibration quality
        with open(calib_result, 'r') as f:
            calib_data = json.load(f)
            if 'rms' in calib_data:
                print(f"Calibration RMS error: {calib_data['rms']:.3f} pixels")
    else:
        print(f"WARNING: Calibration result file not found: {calib_result}")

    # Copy visualization results
    if ENABLE_VIS:
        vis_src = calibration_images_dir.parent / "vis" / calibration_images_dir.name
        vis_dest = working_dir / "calibration_vis"

        if vis_src.exists():
            if vis_dest.exists():
                shutil.rmtree(vis_dest)
            shutil.copytree(str(vis_src), str(vis_dest))
            print(f"OK: Visualization results saved: {vis_dest}")

    # -------------------------------------------------------------------------
    # Step 7: Cleanup Temporary Files
    # -------------------------------------------------------------------------
    if CLEANUP_TEMP:
        print_section("Step 7: Cleanup Temporary Files")

        to_remove = []

        if not KEEP_SYNC_VIDEOS and ENABLE_SYNC:
            to_remove.append(output_dirs['sync'])

        if not KEEP_RAW_IMAGES:
            to_remove.append(images_dir)
            if USE_STABLE_FRAMES:
                to_remove.append(output_dirs['images_stable'])

        # Delete camera symlinks
        for cam_lower in cam_tags_lower:
            link_cam_dir = video_source_dir / cam_lower
            if link_cam_dir.is_symlink():
                to_remove.append(link_cam_dir)

        for path in to_remove:
            if path.exists():
                print(f"Deleting: {path}")
                if path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

        print("OK: Cleanup completed\n")
    else:
        print_section("Step 7: Skip Cleanup")

    # -------------------------------------------------------------------------
    # Completion
    # -------------------------------------------------------------------------
    print_section("SUCCESS: Calibration Complete!")

    print("Output files:")
    print(f"   - Calibration result: {working_dir / 'calibration.json'}")
    print(f"   - Filtered intrinsics: {output_intrinsic}")
    if ENABLE_VIS:
        print(f"   - Visualization:   {working_dir / 'calibration_vis'}")
    if USE_STABLE_FRAMES and 'stable_indices' in locals() and stable_indices:
        print(f"   - Stable frames:   {working_dir / 'stable_frames.txt'}")

    print("\nNext steps:")
    print("   1. Check visualization to verify calibration quality")
    print("   2. View RMS error in calibration.json")
    print("   3. Use calibration.json for 3D reconstruction or pose estimation")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWARNING: User interrupted, exiting")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
