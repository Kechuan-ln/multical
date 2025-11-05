#!/usr/bin/env python3
"""
检查QR码视频的内容和格式
"""
import cv2
import sys

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False

def inspect_qr_video(video_path, num_frames=10):
    """提取前N帧，检查QR码内容"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f}秒")
    print()

    frame_idx = 0
    sample_interval = max(1, total_frames // num_frames)

    print(f"采样间隔: 每{sample_interval}帧提取一次\n")

    last_qr_data = None
    qr_change_count = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps

        # 检测QR码
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] > 1080:
            scale = 1080.0 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        qr_codes = []
        if HAS_PYZBAR:
            try:
                detected = pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE])
                qr_codes = [obj.data.decode('utf-8') for obj in detected]
            except:
                pass

        if not qr_codes:
            try:
                detector = cv2.QRCodeDetector()
                data, _, _ = detector.detectAndDecode(gray)
                if data:
                    qr_codes = [data]
            except:
                pass

        if qr_codes:
            # 解析时间戳
            timestamps = []
            for qr_data in qr_codes:
                if '-' in qr_data:
                    parts = qr_data.split('-')
                    if len(parts) == 2:
                        try:
                            ts = int(parts[1])
                            timestamps.append(ts)
                        except:
                            pass

            if timestamps:
                print(f"Frame {frame_idx:5d} ({time_sec:6.2f}s): 检测到 {len(qr_codes)} 个QR码")
                print(f"  时间戳范围: {min(timestamps)} - {max(timestamps)}")
                print(f"  时间戳跨度: {(max(timestamps) - min(timestamps)) / 1e6:.3f}秒")

                # 检测QR码内容是否变化
                qr_signature = tuple(sorted(timestamps))
                if qr_signature != last_qr_data:
                    if last_qr_data is not None:
                        qr_change_count += 1
                        print(f"  ⚠️  QR码内容发生变化（第{qr_change_count}次）")
                    last_qr_data = qr_signature
            else:
                print(f"Frame {frame_idx:5d} ({time_sec:6.2f}s): 检测到QR码但无法解析时间戳")
        else:
            print(f"Frame {frame_idx:5d} ({time_sec:6.2f}s): 未检测到QR码")

        frame_idx += sample_interval

    cap.release()

    print(f"\n总结:")
    print(f"  QR码内容变化次数: {qr_change_count}")
    if qr_change_count > 0:
        avg_interval = duration / qr_change_count
        print(f"  平均切换间隔: {avg_interval:.2f}秒")
        print(f"  切换频率: {1.0/avg_interval:.2f} Hz")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python inspect_qr_video.py <video.mp4>")
        sys.exit(1)

    inspect_qr_video(sys.argv[1])
