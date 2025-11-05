#!/usr/bin/env python3
"""
GoPro + PrimeColor Complete Calibration Workflow
=================================================

Features:
1. Extract PrimeColor intrinsics from .mcal file
2. Prepare GoPro intrinsics
3. QR code synchronization (GoPro + PrimeColor)
4. Extrinsic calibration (compute relative pose)
5. Verification and validation

Usage:
1. Modify the "CONFIGURATION" section below with your file paths
2. Run: conda activate multical && python run_gopro_primecolor_calibration.py
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path


# ==============================================================================
# ============================= CONFIGURATION ==================================
# ==============================================================================

# ---------- Working Directory ----------
# Main working directory (contains all files)
WORKING_DIR = "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic "  # NOTE: Path has trailing space!

# ---------- Input Files ----------
# PrimeColor intrinsics source (.mcal file from Motive/OptiTrack)
PRIMECOLOR_MCAL = "Primecolor.mcal"  # Relative to WORKING_DIR

# GoPro intrinsics JSON file (pre-computed)
GOPRO_INTRINSIC = "Intrinsic-16.json"  # Relative to WORKING_DIR

# QR code anchor video (reference)
QR_ANCHOR = "Anchor.mp4"  # Relative to WORKING_DIR

# Camera videos (can be same file if QR code + board in one video)
GOPRO_VIDEO = "Cam4/Video.MP4"         # GoPro recording (QR + board)
PRIMECOLOR_VIDEO = "Primecolor/Video.avi"  # PrimeColor recording (QR + board)

# ---------- Video Time Ranges (for QR scanning and calibration extraction) ----------
# QR code sync range (only used for QR detection, does NOT crop video)
QR_START_TIME = 0        # QR code starts at 0s (for scanning only)
QR_DURATION = 30         # Scan first 30s for QR codes

# ChArUco board range (for calibration frame extraction AFTER sync)
CHARUCO_START_TIME = 0  # ChArUco board starts at 30s
CHARUCO_DURATION = 180    # ChArUco board duration (seconds)

# Output directory (relative to WORKING_DIR or absolute)
OUTPUT_DIR = "calibration_output"

# ---------- Camera Configuration ----------
# GoPro camera name in intrinsic JSON (if file contains multiple cameras)
# Common values: cam1, cam2, cam3, cam4
GOPRO_CAMERA_NAME = "cam4"  # NOTE: Lowercase to match Intrinsic-16.json

# PrimeColor camera ID (if .mcal contains multiple cameras, specify which one)
# Leave as None to use the first camera found
PRIMECOLOR_CAMERA_ID = "13"  # Camera ID from .mcal file (NOT "Camera_13", just "13")

# ---------- Calibration Board Configuration ----------
# Board config file (relative to multical directory)
# Options:
#   - charuco_b3.yaml        (B3 size, 5x9 grid, 50mm squares)
#   - charuco_b1_2.yaml      (B1 size, 10x14 grid, 70mm squares)
#   - charuco_b1_2_dark.yaml (B1 size, optimized for dark images) ⭐ RECOMMENDED
BOARD_CONFIG = "./asset/charuco_b1_2_dark.yaml"  # 优化暗图像检测（测试显示+66%成功率）

# ---------- QR Code Synchronization Parameters ----------
SCAN_DURATION = 30.0       # Scan first N seconds of QR video (default: 30)
QR_STEP = 5                # Detect QR every N frames (default: 5, lower=more accurate but slower)

# ---------- Extrinsic Calibration Parameters ----------
EXTRINSIC_FPS = 5      # Extract N frames per second from videos (default: 1.0)
EXTRINSIC_MAX_FRAMES = 800 # Maximum number of frames to extract (default: 100)

# ---------- Step Control ----------
SKIP_INTRINSIC = False     # Skip intrinsic extraction (if already done)
SKIP_SYNC = False          # Skip QR code synchronization (if already done)
SKIP_EXTRINSIC = False     # Skip extrinsic calibration (run only sync)

# ==============================================================================
# ============================= HELPER FUNCTIONS ================================
# ==============================================================================

def print_section(title):
    """Print section separator"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_command(cmd, description, cwd=None, check=True):
    """Run command and print output"""
    print(f">> {description}")
    print(f"   Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)

    if check and result.returncode != 0:
        print(f"\nERROR: {description} failed")
        sys.exit(1)

    print(f"OK: {description} completed\n")
    return result.returncode == 0


def check_file_exists(path, description):
    """Check if file exists"""
    if not Path(path).exists():
        print(f"ERROR: File does not exist: {description}")
        print(f"       Path: {path}")
        sys.exit(1)
    print(f"OK: {description}")


# ==============================================================================
# ============================= MAIN PROGRAM ===================================
# ==============================================================================

def main():
    print_section("GoPro + PrimeColor Complete Calibration Workflow")

    # Get script directory and working directory
    script_dir = Path(__file__).parent.absolute()
    working_dir = Path(WORKING_DIR).resolve()

    if not working_dir.exists():
        print(f"ERROR: Working directory does not exist: {WORKING_DIR}")
        print(f"       Resolved path: {working_dir}")
        sys.exit(1)

    print(f"Script directory:  {script_dir}")
    print(f"Working directory: {working_dir}")

    # Resolve all input file paths (relative to working directory)
    primecolor_mcal = working_dir / PRIMECOLOR_MCAL
    gopro_intrinsic_src = working_dir / GOPRO_INTRINSIC
    qr_anchor = working_dir / QR_ANCHOR
    gopro_video = working_dir / GOPRO_VIDEO
    primecolor_video = working_dir / PRIMECOLOR_VIDEO

    # Create output directories (in working directory)
    output_dir = working_dir / OUTPUT_DIR if not Path(OUTPUT_DIR).is_absolute() else Path(OUTPUT_DIR)
    intrinsics_dir = output_dir / 'intrinsics'
    sync_dir = output_dir / 'sync'
    extrinsics_dir = output_dir / 'extrinsics'

    intrinsics_dir.mkdir(parents=True, exist_ok=True)
    sync_dir.mkdir(parents=True, exist_ok=True)
    extrinsics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory:  {output_dir.absolute()}")

    # Output file paths
    primecolor_intrinsic = intrinsics_dir / 'primecolor_intrinsic.json'
    gopro_intrinsic = intrinsics_dir / 'gopro_intrinsic.json'
    gopro_synced = sync_dir / 'gopro_synced.mp4'           # Full synced GoPro video
    primecolor_synced = sync_dir / 'primecolor_synced.mp4'  # Full synced PrimeColor video
    gopro_charuco = extrinsics_dir / 'gopro_charuco.mp4'    # ChArUco segment extracted AFTER sync
    primecolor_charuco = extrinsics_dir / 'primecolor_charuco.mp4'  # ChArUco segment extracted AFTER sync
    calibration_json = extrinsics_dir / 'frames' / 'calibration.json'

    # ========================================================================
    # Step 0: Check Input Files & Segment Videos
    # ========================================================================
    print_section("Step 0: Check Input Files")

    check_file_exists(primecolor_mcal, "PrimeColor .mcal file")
    check_file_exists(gopro_intrinsic_src, "GoPro intrinsic JSON")
    check_file_exists(qr_anchor, "QR Anchor video")
    check_file_exists(gopro_video, "GoPro video")
    check_file_exists(primecolor_video, "PrimeColor video")

    print("\nℹ️  Video sync strategy:")
    print(f"   - QR scan range: {QR_START_TIME}s - {QR_START_TIME + QR_DURATION}s")
    print(f"   - ChArUco range: {CHARUCO_START_TIME}s - {CHARUCO_START_TIME + CHARUCO_DURATION}s")
    print(f"   - Synced videos will preserve ALL content except non-overlapping parts\n")

    # ========================================================================
    # Step 1: Extract PrimeColor Intrinsics
    # ========================================================================
    if not SKIP_INTRINSIC or not primecolor_intrinsic.exists():
        print_section("Step 1: Extract PrimeColor Intrinsics")

        cmd = [
            sys.executable, str(script_dir / 'parse_optitrack_cal.py'),
            str(primecolor_mcal),
            '--output', str(primecolor_intrinsic)
        ]

        if PRIMECOLOR_CAMERA_ID:
            cmd.extend(['--camera', PRIMECOLOR_CAMERA_ID])

        run_command(cmd, "Extract PrimeColor intrinsics from .mcal", cwd=str(script_dir))

        # Rename camera to "primecolor" if needed
        with open(primecolor_intrinsic, 'r') as f:
            data = json.load(f)

        cameras = data.get('cameras', {})
        if 'primecolor' not in cameras and cameras:
            # Rename first camera to primecolor
            first_cam = list(cameras.keys())[0]
            cameras['primecolor'] = cameras.pop(first_cam)

            with open(primecolor_intrinsic, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"   Renamed camera '{first_cam}' to 'primecolor'")
    else:
        print_section("Step 1: Skip PrimeColor Intrinsics (already exists)")

    # ========================================================================
    # Step 2: Prepare GoPro Intrinsics
    # ========================================================================
    print_section("Step 2: Prepare GoPro Intrinsics")

    # Read GoPro intrinsics
    with open(gopro_intrinsic_src, 'r') as f:
        gopro_data = json.load(f)

    cameras = gopro_data.get('cameras', {})

    # Check if specified camera exists
    if GOPRO_CAMERA_NAME in cameras:
        # Extract single camera - KEEP ORIGINAL NAME (e.g., cam4)
        gopro_single = {
            'cameras': {
                GOPRO_CAMERA_NAME: cameras[GOPRO_CAMERA_NAME]
            }
        }

        with open(gopro_intrinsic, 'w') as f:
            json.dump(gopro_single, f, indent=2)

        print(f"OK: Extracted GoPro camera '{GOPRO_CAMERA_NAME}' (keeping original name)")
    elif 'cam1' in cameras:
        # Already cam1, just copy
        with open(gopro_intrinsic, 'w') as f:
            json.dump({'cameras': {'cam1': cameras['cam1']}}, f, indent=2)

        print(f"OK: Using GoPro camera 'cam1'")
    else:
        print(f"ERROR: GoPro intrinsic does not contain '{GOPRO_CAMERA_NAME}' or 'cam1'")
        print(f"       Available cameras: {list(cameras.keys())}")
        sys.exit(1)

    # ========================================================================
    # Step 3: QR Code Synchronization (GoPro ↔ PrimeColor via Anchor)
    # ========================================================================
    if not SKIP_SYNC or not (gopro_synced.exists() and primecolor_synced.exists()):
        print_section("Step 3: QR Code Synchronization (GoPro ↔ PrimeColor)")

        print("Using Anchor-based QR sync:")
        print(f"  - Video1 (reference): GoPro (full video)")
        print(f"  - Video2 (to sync):   PrimeColor (full video)")
        print(f"  - Anchor QR video:    {qr_anchor.name}")
        print(f"  - QR scan range:      {QR_START_TIME}s - {QR_START_TIME + QR_DURATION}s")
        print(f"  - Output:             Full synchronized videos (preserving all content)")
        print()

        cmd = [
            sys.executable, str(script_dir / 'sync_with_qr_anchor.py'),
            '--video1', str(gopro_video),              # Full GoPro video
            '--video2', str(primecolor_video),         # Full PrimeColor video
            '--output', str(primecolor_synced),        # Output synced PrimeColor
            '--anchor-video', str(qr_anchor),          # Anchor as reference
            '--scan-start', str(QR_START_TIME),
            '--scan-duration', str(QR_DURATION),
            '--step', str(QR_STEP),
            '--save-json', str(sync_dir / 'sync_result.json'),
            '--stacked', str(sync_dir / 'verify_sync.mp4'),
            '--stacked-duration', '15'
        ]

        run_command(cmd, "Synchronize full GoPro and PrimeColor videos", cwd=str(script_dir))

        # Display sync result
        sync_result_file = sync_dir / 'sync_result.json'
        with open(sync_result_file, 'r') as f:
            sync_data = json.load(f)

        offset = sync_data.get('sync_result', {}).get('offset_seconds', 0)
        print(f"\n   ✅ Synchronization complete!")
        print(f"   Relative offset: {offset:.3f} seconds")
        print(f"   Output synced video: {primecolor_synced.name}")
        print(f"   (PrimeColor adjusted by {offset:.3f}s to align with GoPro)\n")

        # GoPro doesn't need syncing (it's the reference)
        # Just copy the original full video
        if not gopro_synced.exists():
            print("   Creating GoPro reference video (no sync adjustment needed)...")
            shutil.copy(str(gopro_video), str(gopro_synced))
            print(f"   ✅ GoPro reference: {gopro_synced.name}\n")
    else:
        print_section("Step 3: Skip Synchronization (already exists)")

    # ========================================================================
    # Step 4: Extract ChArUco Segments from Synced Videos
    # ========================================================================
    if not SKIP_EXTRINSIC:
        print_section("Step 4: Extract ChArUco Segments from Synced Videos")

        # Extract ChArUco segment from synced GoPro
        if not gopro_charuco.exists():
            print(">> Extracting ChArUco segment from synced GoPro...")
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(CHARUCO_START_TIME),
                '-t', str(CHARUCO_DURATION),
                '-i', str(gopro_synced),
                '-c', 'copy',
                str(gopro_charuco)
            ]
            subprocess.run(cmd, capture_output=True)
            print(f"   ✅ Created: {gopro_charuco.name}")
        else:
            print(f"   ✅ Already exists: {gopro_charuco.name}")

        # Extract ChArUco segment from synced PrimeColor
        if not primecolor_charuco.exists():
            print(">> Extracting ChArUco segment from synced PrimeColor...")
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(CHARUCO_START_TIME),
                '-t', str(CHARUCO_DURATION),
                '-i', str(primecolor_synced),
                '-c', 'copy',
                str(primecolor_charuco)
            ]
            subprocess.run(cmd, capture_output=True)
            print(f"   ✅ Created: {primecolor_charuco.name}")
        else:
            print(f"   ✅ Already exists: {primecolor_charuco.name}")

        print()

    # ========================================================================
    # Step 5: Extrinsic Calibration
    # ========================================================================
    if not SKIP_EXTRINSIC:
        print_section("Step 5: Extrinsic Calibration")

        cmd = [
            sys.executable, str(script_dir / 'calibrate_gopro_primecolor_extrinsics.py'),
            '--gopro-video', str(gopro_charuco),
            '--prime-video', str(primecolor_charuco),
            '--gopro-intrinsic', str(gopro_intrinsic),
            '--prime-intrinsic', str(primecolor_intrinsic),
            '--output-dir', str(extrinsics_dir),
            '--board', BOARD_CONFIG,
            '--fps', str(EXTRINSIC_FPS),
            '--max-frames', str(EXTRINSIC_MAX_FRAMES)
        ]

        run_command(cmd, "Compute extrinsic calibration", cwd=str(script_dir))

        # Check output
        if calibration_json.exists():
            with open(calibration_json, 'r') as f:
                calib_data = json.load(f)

            rms = calib_data.get('rms', 999)
            print(f"   Extrinsic calibration complete")
            print(f"   RMS error: {rms:.4f} pixels")

            if rms < 1.0:
                print(f"   ✅ Calibration quality: GOOD (RMS < 1.0)")
            else:
                print(f"   ⚠️  Calibration quality: FAIR (recommend RMS < 1.0)")
        else:
            print(f"   WARNING: Calibration result not found")
    else:
        print_section("Step 4-5: Skip Extrinsic Calibration")

    # ========================================================================
    # Completion Summary
    # ========================================================================
    print_section("SUCCESS: Calibration Complete!")

    print("📁 Output Files:")
    print(f"   1. PrimeColor intrinsics: {primecolor_intrinsic}")
    print(f"   2. GoPro intrinsics:      {gopro_intrinsic}")
    print(f"   3. GoPro synced video:    {gopro_synced}")
    print(f"   4. PrimeColor synced:     {primecolor_synced}")

    if not SKIP_EXTRINSIC and calibration_json.exists():
        print(f"\n🎯 Final Calibration File:")
        print(f"   {calibration_json}")
        print(f"\n   Use this file for 3D reconstruction and pose estimation.")

    print(f"\n📊 Verification Steps:")
    print(f"   1. Check sync video (stacked GoPro + PrimeColor):")
    print(f"      open {sync_dir}/verify_sync.mp4")
    print(f"      → QR codes should be aligned in both videos")

    if not SKIP_EXTRINSIC:
        print(f"   2. Check calibration visualization:")
        print(f"      open {extrinsics_dir}/frames/vis/")
        print(f"   3. View calibration JSON:")
        print(f"      cat {calibration_json} | python -m json.tool")

    print(f"\nNext Steps:")
    print(f"   1. Verify QR code alignment in stacked video")
    print(f"   2. Check corner detection in visualization")
    print(f"   3. Copy calibration.json to your recording directory")
    print(f"   4. Run 3D triangulation: python scripts/run_triangulation.py")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nWARNING: User interrupted, exiting")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
