#!/usr/bin/env python3
"""
Multi-Camera Synchronization Data Analyzer

Analyzes PrimeColor video, GoPro sync metadata, and Mocap CSV structure
to provide detailed information for implementing the sync workflow.

Usage:
    conda activate multical
    python sync/analyze_sync_data.py
"""

import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path
from collections import OrderedDict


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_video_with_ffprobe(video_path):
    """Extract detailed video metadata using ffprobe"""
    print_section(f"Analyzing Video: {Path(video_path).name}")

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None

    # Get video stream info
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,width,height,duration,nb_frames,codec_name,pix_fmt,bit_rate',
        '-of', 'json',
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)

        if 'streams' not in video_info or len(video_info['streams']) == 0:
            print("❌ No video streams found")
            return None

        stream = video_info['streams'][0]

        # Parse frame rate
        fps_str = stream.get('r_frame_rate', '0/1')
        fps_num, fps_den = map(int, fps_str.split('/'))
        fps = fps_num / fps_den if fps_den != 0 else 0

        # Get format info (container, timecode, etc)
        cmd_format = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration,size,bit_rate,format_name',
            '-show_entries', 'format_tags=timecode',
            '-of', 'json',
            video_path
        ]
        result_format = subprocess.run(cmd_format, capture_output=True, text=True, check=True)
        format_info = json.loads(result_format.stdout)

        # Compile results
        info = {
            'file_path': video_path,
            'file_size_mb': os.path.getsize(video_path) / (1024**2),
            'codec': stream.get('codec_name', 'unknown'),
            'width': stream.get('width', 0),
            'height': stream.get('height', 0),
            'fps': fps,
            'fps_raw': fps_str,
            'duration_sec': float(stream.get('duration', 0)) if 'duration' in stream else float(format_info['format'].get('duration', 0)),
            'nb_frames': int(stream.get('nb_frames', 0)) if 'nb_frames' in stream else None,
            'pix_fmt': stream.get('pix_fmt', 'unknown'),
            'bit_rate_mbps': int(stream.get('bit_rate', 0)) / 1_000_000 if 'bit_rate' in stream else None,
            'container_format': format_info['format'].get('format_name', 'unknown'),
            'has_timecode': 'timecode' in format_info.get('format', {}).get('tags', {}),
            'timecode': format_info.get('format', {}).get('tags', {}).get('timecode', None)
        }

        # Calculate estimated frames if not available
        if info['nb_frames'] is None and info['duration_sec'] > 0:
            info['nb_frames_estimated'] = int(info['duration_sec'] * fps)

        # Print results
        print(f"✓ File Size: {info['file_size_mb']:.2f} MB")
        print(f"✓ Codec: {info['codec']}")
        print(f"✓ Resolution: {info['width']}x{info['height']}")
        print(f"✓ FPS: {fps:.3f} ({fps_str})")
        print(f"✓ Duration: {info['duration_sec']:.2f} seconds")
        print(f"✓ Total Frames: {info['nb_frames'] if info['nb_frames'] else info.get('nb_frames_estimated', 'unknown')}")
        print(f"✓ Pixel Format: {info['pix_fmt']}")
        print(f"✓ Bit Rate: {info['bit_rate_mbps']:.2f} Mbps" if info['bit_rate_mbps'] else "✓ Bit Rate: unknown")
        print(f"✓ Container: {info['container_format']}")
        print(f"✓ Has Embedded Timecode: {'Yes - ' + info['timecode'] if info['has_timecode'] else 'No'}")

        return info

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running ffprobe: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def analyze_mocap_csv(csv_path, max_rows_to_show=10):
    """Analyze Optitrack Motive mocap CSV structure"""
    print_section(f"Analyzing Mocap CSV: {Path(csv_path).name}")

    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return None

    try:
        # Read metadata from first line (it's not a standard header)
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        # Parse metadata
        metadata_items = first_line.split(',')
        metadata = {}
        for i in range(0, len(metadata_items), 2):
            if i + 1 < len(metadata_items):
                key = metadata_items[i].strip()
                value = metadata_items[i + 1].strip()
                if key:  # Skip empty keys
                    metadata[key] = value

        print("\n📋 CSV Metadata:")
        print(f"  Format Version: {metadata.get('Format Version', 'unknown')}")
        print(f"  Take Name: {metadata.get('Take Name', 'unknown')}")
        print(f"  Capture Frame Rate: {metadata.get('Capture Frame Rate', 'unknown')} fps")
        print(f"  Export Frame Rate: {metadata.get('Export Frame Rate', 'unknown')} fps")
        print(f"  Capture Start Time: {metadata.get('Capture Start Time', 'unknown')}")
        print(f"  Capture Start Frame: {metadata.get('Capture Start Frame', 'unknown')}")
        print(f"  Total Frames in Take: {metadata.get('Total Frames in Take', 'unknown')}")
        print(f"  Total Exported Frames: {metadata.get('Total Exported Frames', 'unknown')}")
        print(f"  Rotation Type: {metadata.get('Rotation Type', 'unknown')}")
        print(f"  Length Units: {metadata.get('Length Units', 'unknown')}")
        print(f"  Coordinate Space: {metadata.get('Coordinate Space', 'unknown')}")

        # Read CSV with pandas, skipping metadata rows
        # Row 0: metadata, Row 1: Type, Row 2: Name, Row 3: ID, Row 4: column names, Row 5+: data
        df = pd.read_csv(csv_path, skiprows=4, low_memory=False)

        # First column is usually Frame or Time
        first_col = df.columns[0]
        print(f"\n📊 CSV Structure:")
        print(f"  Total Rows (data): {len(df)}")
        print(f"  Total Columns: {len(df.columns)}")
        print(f"  First Column (time index): '{first_col}'")

        # Analyze first column (time reference)
        print(f"\n⏱️  Time Reference Column Analysis:")
        print(f"  Column Name: {first_col}")
        print(f"  First Value: {df.iloc[0, 0]}")
        print(f"  Last Value: {df.iloc[-1, 0]}")
        print(f"  Data Type: {df.iloc[:, 0].dtype}")

        # Check if it's frame index or timestamp
        # Skip NaN rows and find first valid value
        first_valid_idx = df.iloc[:, 0].first_valid_index()
        if first_valid_idx is not None:
            first_val = df.iloc[first_valid_idx, 0]
            if isinstance(first_val, (int, float)):
                if abs(first_val - int(first_val)) < 1e-6:  # Likely frame index
                    print(f"  Interpretation: Frame index (integer)")
                else:
                    print(f"  Interpretation: Timestamp (float seconds)")
        else:
            print(f"  Interpretation: Could not determine (all NaN)")

        # Read marker info from row 2 and 3
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [next(f) for _ in range(4)]

        marker_types = lines[1].split(',')
        marker_names = lines[2].split(',')
        marker_ids = lines[3].split(',')

        # Extract unique markers (each marker has X, Y, Z columns)
        unique_markers = []
        seen_names = set()
        for i in range(1, len(marker_names), 3):  # Skip first column, step by 3 (X,Y,Z)
            name = marker_names[i].strip()
            if name and name not in seen_names:
                marker_id = marker_ids[i].strip() if i < len(marker_ids) else ''
                unique_markers.append({
                    'name': name,
                    'id': marker_id,
                    'col_index': i
                })
                seen_names.add(name)

        print(f"\n🎯 Marker Information:")
        print(f"  Total Unique Markers: {len(unique_markers)}")
        print(f"  Columns per Marker: 3 (X, Y, Z)")
        print(f"  Total Marker Columns: {len(unique_markers) * 3}")

        # Show first few markers
        print(f"\n  First {min(5, len(unique_markers))} Markers:")
        for i, marker in enumerate(unique_markers[:5]):
            print(f"    {i+1}. {marker['name']} (ID: {marker['id']})")

        # Check for missing data
        print(f"\n🔍 Data Quality Check:")
        # Sample some marker data columns
        sample_cols = [col for col in df.columns[1:10]]  # First few data columns
        for col in sample_cols:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"  {col}: {null_pct:.1f}% missing")

        # Show sample data
        print(f"\n📝 Sample Data (first {max_rows_to_show} rows, first 12 columns):")
        print(df.iloc[:max_rows_to_show, :12].to_string(index=False))

        # Summary for sync implementation
        summary = {
            'csv_path': csv_path,
            'file_size_mb': os.path.getsize(csv_path) / (1024**2),
            'capture_fps': float(metadata.get('Capture Frame Rate', 0)),
            'export_fps': float(metadata.get('Export Frame Rate', 0)),
            'total_frames': int(metadata.get('Total Exported Frames', 0)),
            'capture_start_frame': int(metadata.get('Capture Start Frame', 0)),
            'num_markers': len(unique_markers),
            'time_column': first_col,
            'data_rows': len(df),
            'marker_names': [m['name'] for m in unique_markers[:10]],  # First 10
            'coordinate_space': metadata.get('Coordinate Space', 'unknown'),
            'length_units': metadata.get('Length Units', 'unknown')
        }

        return summary

    except Exception as e:
        print(f"❌ Error analyzing CSV: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def analyze_gopro_meta(json_path):
    """Analyze GoPro sync metadata JSON"""
    print_section(f"Analyzing GoPro Sync Meta: {Path(json_path).name}")

    if not os.path.exists(json_path):
        print(f"❌ JSON not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            meta = json.load(f)

        print("\n📹 GoPro Synchronization Metadata:")

        # Print structure
        if isinstance(meta, dict):
            for key, value in meta.items():
                if isinstance(value, dict):
                    print(f"\n  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

        return meta

    except Exception as e:
        print(f"❌ Error reading JSON: {str(e)}")
        return None


def generate_sync_implementation_guide(primecolor_info, mocap_info, gopro_meta):
    """Generate implementation recommendations based on analyzed data"""
    print_section("Synchronization Implementation Guide")

    print("\n🎬 Video Synchronization Strategy:")

    if primecolor_info:
        pc_fps = primecolor_info['fps']
        gopro_fps = 60  # Assumed from GoPro metadata

        print(f"\n  1. Frame Rate Handling:")
        print(f"     - PrimeColor FPS: {pc_fps:.3f}")
        print(f"     - GoPro FPS: {gopro_fps}")
        print(f"     - FPS Ratio: {pc_fps/gopro_fps:.3f}")

        if pc_fps > gopro_fps:
            print(f"     ⚠️  PrimeColor has higher FPS - consider two options:")
            print(f"         Option A: Keep original FPS, only align timeline")
            print(f"         Option B: Downsample to {gopro_fps} fps")
            print(f"           Method: ffmpeg -i input.mp4 -vf \"fps={gopro_fps}\" output.mp4")
        elif pc_fps < gopro_fps:
            print(f"     ⚠️  PrimeColor has lower FPS - keep as is, GoPro will have more frames")
        else:
            print(f"     ✓ Same FPS - no downsampling needed")

        print(f"\n  2. Video Alignment:")
        print(f"     - Method: QR code matching between iPad video and camera recordings")
        print(f"     - Output: time_offset (in frames or seconds)")
        print(f"     - Padding: Add black frames if PrimeColor starts later")
        print(f"     - Tool: ffmpeg with concat filter")

    if mocap_info:
        print(f"\n  3. Mocap CSV Synchronization:")
        print(f"     - CSV FPS: {mocap_info['capture_fps']}")
        print(f"     - Total Frames: {mocap_info['total_frames']}")
        print(f"     - Time Column: {mocap_info['time_column']}")

        if primecolor_info:
            fps_match = abs(mocap_info['capture_fps'] - primecolor_info['fps']) < 0.1
            print(f"     - FPS Match with PrimeColor: {'✓ Yes' if fps_match else '✗ No'}")

            if fps_match:
                print(f"     - Strategy: Apply same time_offset to CSV frame indices")
                print(f"       Example: csv['frame_synced'] = csv['frame'] + offset_frames")
            else:
                print(f"     - Strategy: Convert time offset, then adjust frame indices")
                print(f"       fps_ratio = {mocap_info['capture_fps']}/{primecolor_info['fps']} = {mocap_info['capture_fps']/primecolor_info['fps']:.3f}")

    print(f"\n  4. Recommended Implementation Order:")
    print(f"     Step 1: Sync GoPro cameras (existing code - sync_timecode.py)")
    print(f"     Step 2: Detect QR codes in GoPro synced videos")
    print(f"     Step 3: Detect QR codes in PrimeColor video")
    print(f"     Step 4: Calculate time mapping (offset, fps_ratio)")
    print(f"     Step 5: Align PrimeColor video (pad black frames)")
    print(f"     Step 6: Sync mocap CSV if exists")
    print(f"     Step 7: Verify sync quality (QR residuals, motion consistency)")

    print(f"\n  5. Output Structure:")
    print(f"     sync_output/")
    print(f"     ├── gopro_synced/")
    print(f"     │   ├── cam01.MP4 ... cam16.MP4")
    print(f"     │   └── meta_info.json")
    print(f"     ├── primecolor_synced/")
    print(f"     │   ├── video_synced.mp4")
    print(f"     │   └── sync_info.json (offset, fps_ratio, confidence)")
    print(f"     └── mocap_synced/")
    print(f"         ├── markers_synced.csv")
    print(f"         └── sync_info.json")


def main():
    """Main analysis routine"""
    print("=" * 80)
    print("  Multi-Camera Synchronization Data Analyzer")
    print("  Analyzing PrimeColor, GoPro, and Mocap data structures")
    print("=" * 80)

    # Define paths - UPDATE THESE
    primecolor_video = "/Volumes/FastACIS/csl-data/sync.avi"
    gopro_meta_json = None  # UPDATE if exists: "/path/to/gopro_synced/meta_info.json"
    mocap_csv = "/Volumes/FastACIS/csl-data/Take 2025-10-24 04.01.02 PM.csv"

    # Analysis results
    primecolor_info = None
    gopro_meta = None
    mocap_info = None

    # Analyze PrimeColor video
    if os.path.exists(primecolor_video):
        primecolor_info = analyze_video_with_ffprobe(primecolor_video)
    else:
        print(f"\n⚠️  PrimeColor video not found: {primecolor_video}")
        print("   Please update the 'primecolor_video' path in the script")

    # Analyze GoPro metadata
    if gopro_meta_json and os.path.exists(gopro_meta_json):
        gopro_meta = analyze_gopro_meta(gopro_meta_json)
    else:
        print(f"\n⚠️  GoPro meta JSON not found (optional)")
        print("   If you have synced GoPro videos, provide meta_info.json path")

    # Analyze Mocap CSV
    if os.path.exists(mocap_csv):
        mocap_info = analyze_mocap_csv(mocap_csv, max_rows_to_show=5)
    else:
        print(f"\n⚠️  Mocap CSV not found: {mocap_csv}")

    # Generate implementation guide
    if primecolor_info or mocap_info:
        generate_sync_implementation_guide(primecolor_info, mocap_info, gopro_meta)

    # Save results to JSON
    output_file = "sync/sync_analysis_results.json"
    results = {
        'primecolor': primecolor_info,
        'gopro_meta': gopro_meta,
        'mocap': mocap_info,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }

    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Analysis results saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {str(e)}")

    print("\n" + "=" * 80)
    print("  Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
