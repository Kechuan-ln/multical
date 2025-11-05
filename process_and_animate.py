#!/usr/bin/env python3
"""
Complete pipeline: Process mocap CSV -> Generate skeleton + prosthesis -> Create GIF animation

Usage:
    python process_and_animate.py \
        --mocap /path/to/mocap.csv \
        --frame_range 3800-4000 \
        --output skeleton_animation.gif
"""

import subprocess
import argparse
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n✗ ERROR: {description} failed!")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline: Process mocap -> Generate skeleton + prosthesis -> Create GIF'
    )
    parser.add_argument('--mocap', required=True, help='Mocap CSV file path')
    parser.add_argument('--marker_labels', default='marker_labels_final.csv',
                       help='Marker labels CSV file')
    parser.add_argument('--skeleton_config', default='skeleton_config.json',
                       help='Skeleton configuration JSON')
    parser.add_argument('--prosthesis_config', default='prosthesis_config.json',
                       help='Prosthesis configuration JSON')
    parser.add_argument('--frame_range', required=True,
                       help='Frame range (e.g., "3800-4000" or "3832")')
    parser.add_argument('--output', default='skeleton_animation.gif',
                       help='Output GIF file name')
    parser.add_argument('--fps', type=int, default=15,
                       help='GIF frame rate (default: 15)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Maximum GIF duration in seconds (default: all frames)')
    parser.add_argument('--view', choices=['default', 'side', 'front', 'top'],
                       default='side', help='Camera view angle (default: side)')

    args = parser.parse_args()

    # Intermediate file
    json_output = 'skeleton_with_prosthesis.json'

    print("\n" + "="*70)
    print("SKELETON + PROSTHESIS ANIMATION PIPELINE")
    print("="*70)
    print(f"Input mocap: {args.mocap}")
    print(f"Frame range: {args.frame_range}")
    print(f"Output GIF: {args.output}")
    print(f"View angle: {args.view}")
    print(f"Frame rate: {args.fps} fps")
    if args.duration:
        print(f"Duration: {args.duration} seconds")
    else:
        print(f"Duration: Full sequence")

    # Step 1: Process markers to skeleton + prosthesis
    cmd1 = [
        'python3', 'markers_to_skeleton_with_prosthesis.py',
        '--mocap', args.mocap,
        '--marker_labels', args.marker_labels,
        '--skeleton_config', args.skeleton_config,
        '--prosthesis_config', args.prosthesis_config,
        '--output', json_output,
        '--frames', args.frame_range
    ]

    run_command(cmd1, "Step 1: Converting markers to skeleton + prosthesis")

    # Check if JSON was created and has frames
    import json
    with open(json_output, 'r') as f:
        data = json.load(f)

    num_frames = len(data['frames'])
    if num_frames == 0:
        print("\n✗ ERROR: No frames were processed!")
        print("Please check:")
        print("  1. Frame range exists in mocap CSV")
        print("  2. Marker labels match mocap marker IDs")
        print("  3. At least some markers are present in the frames")
        sys.exit(1)

    print(f"\n✓ Successfully processed {num_frames} frames")

    # Step 2: Generate GIF
    cmd2 = [
        'python3', 'generate_skeleton_gif.py',
        '--input', json_output,
        '--output', args.output,
        '--fps', str(args.fps),
        '--view', args.view
    ]

    if args.duration:
        cmd2.extend(['--duration', str(args.duration)])

    run_command(cmd2, "Step 2: Generating animated GIF")

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"✓ Processed {num_frames} frames")
    print(f"✓ Generated animation: {args.output}")
    print(f"\nYou can view the GIF with:")
    print(f"  open {args.output}  (macOS)")
    print(f"  xdg-open {args.output}  (Linux)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
