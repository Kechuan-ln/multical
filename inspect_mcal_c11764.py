#!/usr/bin/env python3
"""
Detailed inspection of C11764 camera in .mcal file.
Verify we're reading the correct camera (PrimeColor video camera, not OptiTrack mocap camera).
"""

import xml.etree.ElementTree as ET
import sys


def inspect_camera(mcal_path, camera_serial='C11764'):
    """Detailed inspection of camera parameters in .mcal file."""

    tree = ET.parse(mcal_path)
    root = tree.getroot()

    print("="*80)
    print(f"Searching for camera: {camera_serial}")
    print("="*80)
    print()

    # Find camera
    target_camera = None
    for camera in root.findall('.//Camera'):
        serial = camera.get('Serial')
        props = camera.find('.//Properties')

        if props is not None:
            camera_id = props.get('CameraID')

            if serial == camera_serial or camera_id == '13':
                target_camera = camera
                print(f"✓ Found camera!")
                print(f"  Serial: {serial}")
                print(f"  CameraID: {camera_id}")
                print()
                break

    if target_camera is None:
        print(f"❌ Camera {camera_serial} not found!")
        return

    # Extract Properties
    props = target_camera.find('.//Properties')
    if props is not None:
        print("PROPERTIES:")
        print("-"*80)
        for key, value in props.attrib.items():
            print(f"  {key:<30} = {value}")
        print()

    # Extract Attributes
    attrs = target_camera.find('.//Attributes')
    if attrs is not None:
        print("ATTRIBUTES:")
        print("-"*80)
        for key, value in attrs.attrib.items():
            print(f"  {key:<30} = {value}")
        print()

    # Extract Intrinsics - OptiTrack Internal Model
    intrinsic_internal = target_camera.find('.//Intrinsic')
    if intrinsic_internal is not None:
        print("INTRINSICS (OptiTrack Internal Model):")
        print("-"*80)
        print("  ⚠️  This is OptiTrack's internal distortion model (NOT standard OpenCV)")
        for key, value in intrinsic_internal.attrib.items():
            print(f"  {key:<30} = {value}")
        print()

    # Extract Intrinsics - Standard Camera Model (OpenCV compatible)
    intrinsic_standard = target_camera.find('.//IntrinsicStandardCameraModel')
    if intrinsic_standard is not None:
        print("INTRINSICS (Standard Camera Model - OpenCV compatible):")
        print("-"*80)
        print("  ✓ This should be used for OpenCV projection")

        fx = float(intrinsic_standard.get('HorizontalFocalLength'))
        fy = float(intrinsic_standard.get('VerticalFocalLength'))
        cx = float(intrinsic_standard.get('LensCenterX'))
        cy = float(intrinsic_standard.get('LensCenterY'))
        k1 = float(intrinsic_standard.get('k1'))
        k2 = float(intrinsic_standard.get('k2'))
        k3 = float(intrinsic_standard.get('k3'))
        p1 = float(intrinsic_standard.get('TangentialX'))
        p2 = float(intrinsic_standard.get('TangentialY'))

        print(f"  fx (HorizontalFocalLength)     = {fx:.6f} pixels")
        print(f"  fy (VerticalFocalLength)       = {fy:.6f} pixels")
        print(f"  cx (LensCenterX)               = {cx:.6f} pixels")
        print(f"  cy (LensCenterY)               = {cy:.6f} pixels")
        print(f"  k1 (radial distortion 1)       = {k1:.6f}")
        print(f"  k2 (radial distortion 2)       = {k2:.6f}")
        print(f"  k3 (radial distortion 3)       = {k3:.6f}")
        print(f"  p1 (tangential distortion 1)   = {p1:.6f}")
        print(f"  p2 (tangential distortion 2)   = {p2:.6f}")

        # Check if this looks like a video camera
        print()
        print("  Analysis:")
        image_width = attrs.get('ImagerPixelWidth') if attrs is not None else None
        image_height = attrs.get('ImagerPixelHeight') if attrs is not None else None

        if image_width and image_height:
            img_w = int(image_width)
            img_h = int(image_height)
            print(f"    Image size from Attributes: {img_w}x{img_h}")

            # Check if cx, cy are reasonable for this resolution
            if img_w == 1920 and img_h == 1080:
                print(f"    ✓ Resolution matches PrimeColor video (1920x1080)")
                expected_cx = img_w / 2
                expected_cy = img_h / 2
                print(f"    Expected principal point center: ({expected_cx:.1f}, {expected_cy:.1f})")
                print(f"    Actual principal point: ({cx:.1f}, {cy:.1f})")
                print(f"    Offset from center: ({cx - expected_cx:.1f}, {cy - expected_cy:.1f}) pixels")
            elif img_w == 1664 and img_h == 1088:
                print(f"    ⚠️  Resolution is OptiTrack mocap camera (1664x1088)")
                print(f"    ❌ THIS IS NOT THE VIDEO CAMERA!")
        print()

    # Extract Extrinsics
    extrinsic = target_camera.find('.//Extrinsic')
    if extrinsic is not None:
        print("EXTRINSICS (Camera pose in world coordinates):")
        print("-"*80)

        X = float(extrinsic.get('X'))
        Y = float(extrinsic.get('Y'))
        Z = float(extrinsic.get('Z'))

        print(f"  Camera Position (world frame):")
        print(f"    X = {X:.6f} meters")
        print(f"    Y = {Y:.6f} meters")
        print(f"    Z = {Z:.6f} meters")
        print()

        print(f"  Camera Orientation (rotation matrix, row-major):")
        for i in range(3):
            r0 = float(extrinsic.get(f'OrientMatrix{i*3+0}'))
            r1 = float(extrinsic.get(f'OrientMatrix{i*3+1}'))
            r2 = float(extrinsic.get(f'OrientMatrix{i*3+2}'))
            print(f"    [{r0:9.6f}, {r1:9.6f}, {r2:9.6f}]")
        print()

    # Summary
    print("="*80)
    print("VERIFICATION CHECKLIST:")
    print("="*80)

    if attrs is not None:
        img_w = int(attrs.get('ImagerPixelWidth', 0))
        img_h = int(attrs.get('ImagerPixelHeight', 0))

        if img_w == 1920 and img_h == 1080:
            print("✓ Image resolution: 1920x1080 (PrimeColor video camera)")
        elif img_w == 1664 and img_h == 1088:
            print("❌ Image resolution: 1664x1088 (OptiTrack mocap camera)")
            print("   THIS IS THE WRONG CAMERA!")
            print("   PrimeColor video should be 1920x1080")
        else:
            print(f"? Image resolution: {img_w}x{img_h} (unknown)")

    if intrinsic_standard is not None:
        cx = float(intrinsic_standard.get('LensCenterX'))
        cy = float(intrinsic_standard.get('LensCenterY'))

        if attrs is not None:
            img_w = int(attrs.get('ImagerPixelWidth', 0))
            img_h = int(attrs.get('ImagerPixelHeight', 0))

            if img_w > 0 and img_h > 0:
                if abs(cx - img_w/2) < img_w * 0.1 and abs(cy - img_h/2) < img_h * 0.1:
                    print("✓ Principal point is near image center (reasonable)")
                else:
                    print("⚠️  Principal point is far from image center (may be correct for some cameras)")

    if extrinsic is not None:
        Y = float(extrinsic.get('Y'))
        if Y > 2.0:
            print(f"✓ Camera Y position = {Y:.2f}m (camera is elevated, looking down - correct for video camera)")
        else:
            print(f"? Camera Y position = {Y:.2f}m (may be correct)")

    print()


if __name__ == "__main__":
    mcal_path = '/Volumes/FastACIS/GoPro/motion/mocap/Primecolor.mcal'

    if len(sys.argv) > 1:
        mcal_path = sys.argv[1]

    inspect_camera(mcal_path)
