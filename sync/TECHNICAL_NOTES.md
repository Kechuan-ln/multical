# 多相机同步系统 - 技术参考

**目标受众**: 开发者、维护者
**用户指南**: 见 [README.md](README.md)
**更新时间**: 2025-11-03

---

## 目录

1. [系统参数](#系统参数)
2. [数据结构](#数据结构)
3. [同步算法](#同步算法)
4. [错误处理和容错机制](#错误处理和容错机制) ⭐ 重要
5. [代码实现框架](#代码实现框架)
6. [测试数据](#测试数据)
7. [实施清单](#实施清单)

---

## 系统参数

### 硬件参数对比表

| 参数 | GoPro | PrimeColor | Mocap CSV |
|------|-------|------------|-----------|
| **FPS** | 60 | 120 | 120 |
| **分辨率** | 3840×2160 (4K) | 1920×1080 (FHD) | N/A |
| **编码格式** | H.264/HEVC | MJPEG | N/A |
| **容器格式** | .MP4 | .AVI | .CSV |
| **码率** | 变化 | 119.65 Mbps | N/A |
| **像素格式** | yuv420p | yuvj420p | N/A |
| **时间基准** | Timecode嵌入 | 无 (需QR) | Frame Index |
| **坐标系** | N/A | N/A | Global (mm) |

### FPS关系

```
PrimeColor FPS : GoPro FPS : Mocap FPS
    120       :     60     :    120
     2        :      1     :     2
```

**关键优势**:
- 整数倍关系（2:1）
- PrimeColor与Mocap FPS完全一致
- 高时间分辨率（8.33ms @ 120fps）

### 配置常量

```python
# sync/config.py

# 系统FPS
GOPRO_FPS = 60.0
PRIMECOLOR_FPS = 120.0
MOCAP_FPS = 120.0

# FPS比率
FPS_RATIO_PRIME_TO_GOPRO = 2.0  # PrimeColor / GoPro
FPS_RATIO_MOCAP_TO_PRIME = 1.0  # Mocap / PrimeColor

# 时间分辨率
TIME_RESOLUTION_GOPRO = 1/60.0        # 16.67 ms
TIME_RESOLUTION_PRIMECOLOR = 1/120.0  # 8.33 ms
TIME_RESOLUTION_MOCAP = 1/120.0       # 8.33 ms

# 同步精度目标
TARGET_SYNC_ERROR_FRAMES = 2      # 最大允许误差
TARGET_SYNC_ERROR_MS_GOPRO = 33.3  # @ 60fps
TARGET_SYNC_ERROR_MS_PRIME = 16.7  # @ 120fps

# PrimeColor视频参数
PRIMECOLOR_CONFIG = {
    'width': 1920,
    'height': 1080,
    'fps': 120,
    'codec': 'mjpeg',
    'container': 'avi',
    'pix_fmt': 'yuvj420p',
    'bitrate_mbps': 119.65
}

# 输出视频配置
OUTPUT_CONFIG = {
    'codec': 'libx264',
    'crf': 18,  # 高质量
    'preset': 'medium',
    'pix_fmt': 'yuv420p'
}

# QR检测配置
QR_DETECTION_CONFIG = {
    'scan_start_sec': 5,     # 跳过前5秒
    'scan_duration_sec': 60, # 扫描60秒
    'step_frames': 2,        # 每2帧检测一次
    'min_qr_matches': 10,    # 最少匹配数
    'max_error_frames': 2.0, # 最大残差
    'rmse_threshold': 1.0    # RMSE阈值
}

# Mocap CSV配置
MOCAP_CONFIG = {
    'csv_skiprows': 4,
    'frame_column': 'Frame',
    'time_column': 'Time (Seconds)',
    'coordinate_unit': 'Millimeters',
    'coordinate_space': 'Global',
    'fps': 120
}
```

---

## 数据结构

### Mocap CSV结构

**元数据行（第1行）**:
```
Format Version,1.25,Take Name,Take 2025-10-24 04.01.02 PM,
Capture Frame Rate,120.000000,Export Frame Rate,120.000000,
...
```

**标题行（第2-4行）**:
```
Row 2: Type,Marker,Marker,...
Row 3: Name,Unlabeled 1000,Unlabeled 1001,...
Row 4: ID,9F:D842698FB0A611F0,9F:D842698FB0A611F1,...
```

**列名行（第5行）**:
```
Frame,Time (Seconds),X,Y,Z,X,Y,Z,...
```

**数据结构**:
```python
import pandas as pd

# 读取CSV（跳过元数据）
df = pd.read_csv(csv_path, skiprows=4, low_memory=False)

# 列结构
# df.columns[0]: 'Frame' (或 'Unnamed: 0')
# df.columns[1]: 'Time (Seconds)'
# df.columns[2::3]: Marker X坐标
# df.columns[3::3]: Marker Y坐标
# df.columns[4::3]: Marker Z坐标

# 统计
num_markers = (len(df.columns) - 2) // 3
total_frames = len(df)
duration_sec = total_frames / 120.0
```

### GoPro meta_info.json

**预期结构**（来自`scripts/sync_timecode.py`）:
```json
{
  "reference_camera": "cam01",
  "gopro_fps": 60,
  "cameras": {
    "cam01": {
      "offset_seconds": 0.0,
      "duration_seconds": 300.0
    },
    "cam02": {
      "offset_seconds": 0.033,
      "duration_seconds": 299.967
    }
  }
}
```

### sync_info.json输出

**完整结构**:
```json
{
  "reference_camera": "gopro_cam01",
  "sync_method": "qr_anchor_video",
  "anchor_video": "/path/to/qr_anchor.mp4",

  "gopro": {
    "fps": 60,
    "num_cameras": 16,
    "sync_source": "official_qr_code"
  },

  "primecolor": {
    "fps": 120,
    "resolution": "1920x1080",
    "codec": "mjpeg",
    "offset_frames": 145,
    "offset_seconds": 1.208,
    "fps_ratio": 2.003,
    "qr_residuals": {
      "max_error_frames": 1.2,
      "rmse_frames": 0.6,
      "num_qr_matched": 45,
      "qr_range": [10, 1800]
    }
  },

  "mocap": {
    "synced": true,
    "fps": 120,
    "offset_frames": 145,
    "csv_path": "/path/to/mocap_synced.csv",
    "num_markers": 198,
    "total_frames": 9067
  },

  "verification": {
    "method": "dual_qr",
    "sync_quality": "excellent",
    "start_segment": {
      "offset_frames": 145,
      "rmse_frames": 0.6,
      "qr_matched": 45
    },
    "end_segment": {
      "offset_frames": 146,
      "rmse_frames": 0.7,
      "qr_matched": 42
    },
    "drift_frames": 1,
    "drift_ms_gopro": 16.7,
    "drift_ms_prime": 8.3,
    "is_linear": true,
    "fps_ratio_start": 2.003,
    "fps_ratio_end": 2.004,
    "fps_ratio_drift": 0.001
  }
}
```

---

## 同步算法

### 时间映射模型

**基本原理**:

对于任意QR码 `q`:
- GoPro在帧 `f_g` 检测到QR码 `q`
- PrimeColor在帧 `f_p` 检测到相同QR码 `q`
- Anchor视频定义QR码 `q` 对应时间 `t_q`

**线性模型**:
```
f_p = offset + fps_ratio × f_g
```

其中:
- `offset`: PrimeColor相对GoPro的帧偏移
- `fps_ratio`: PrimeColor FPS / GoPro FPS ≈ 2.0

### 最小二乘拟合

```python
import numpy as np
from scipy.optimize import least_squares

def fit_sync_params(gopro_frames, primecolor_frames):
    """
    拟合同步参数

    Args:
        gopro_frames: List[int] - GoPro帧号
        primecolor_frames: List[int] - PrimeColor帧号

    Returns:
        {
            'offset_frames': int,
            'fps_ratio': float,
            'max_error': float,
            'rmse': float
        }
    """
    def residual(params, f_g, f_p):
        offset, fps_ratio = params
        predicted = offset + fps_ratio * f_g
        return predicted - f_p

    # 初始猜测
    x0 = [0, PRIMECOLOR_FPS / GOPRO_FPS]

    # 最小二乘优化
    result = least_squares(
        residual,
        x0=x0,
        args=(np.array(gopro_frames), np.array(primecolor_frames))
    )

    offset, fps_ratio = result.x
    residuals = result.fun

    return {
        'offset_frames': int(round(offset)),
        'fps_ratio': float(fps_ratio),
        'max_error': float(np.max(np.abs(residuals))),
        'rmse': float(np.sqrt(np.mean(residuals**2))),
        'residuals': residuals.tolist()
    }
```

### 视频对齐算法

```python
def align_primecolor_video(input_video, output_video, offset_frames):
    """
    根据offset对齐PrimeColor视频

    offset > 0: PrimeColor开始更晚，需要在前面填充黑帧
    offset < 0: PrimeColor开始更早，需要裁剪开头
    offset = 0: 不需要调整
    """
    import subprocess

    if offset_frames > 0:
        # 填充黑帧
        black_duration = offset_frames / PRIMECOLOR_FPS

        # Step 1: 创建黑帧视频
        black_video = "/tmp/black_frames.mp4"
        cmd_black = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'color=c=black:s=1920x1080:r={PRIMECOLOR_FPS}',
            '-t', str(black_duration),
            '-pix_fmt', 'yuv420p',
            black_video
        ]
        subprocess.run(cmd_black, check=True)

        # Step 2: 拼接黑帧和原视频
        concat_list = "/tmp/concat_list.txt"
        with open(concat_list, 'w') as f:
            f.write(f"file '{black_video}'\n")
            f.write(f"file '{input_video}'\n")

        cmd_concat = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        subprocess.run(cmd_concat, check=True)

    elif offset_frames < 0:
        # 裁剪开头
        start_time = abs(offset_frames) / PRIMECOLOR_FPS
        cmd_trim = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', input_video,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        subprocess.run(cmd_trim, check=True)

    else:
        # offset = 0, 转码即可（MJPEG → H.264）
        cmd_copy = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        subprocess.run(cmd_copy, check=True)
```

### Mocap CSV同步

```python
def sync_mocap_csv(input_csv, output_csv, offset_frames):
    """
    同步Mocap CSV到GoPro时间基准

    由于Mocap FPS = PrimeColor FPS (both 120),
    直接应用相同的frame offset即可
    """
    import pandas as pd

    # 读取元数据行
    with open(input_csv, 'r') as f:
        metadata_lines = [f.readline() for _ in range(4)]

    # 读取数据
    df = pd.read_csv(input_csv, skiprows=4, low_memory=False)

    # 获取列名
    frame_col = df.columns[0]
    time_col = df.columns[1] if len(df.columns) > 1 else None

    # 应用偏移
    df[frame_col + '_synced'] = df[frame_col] + offset_frames

    if time_col and 'Time' in time_col:
        time_offset = offset_frames / MOCAP_FPS
        df[time_col + '_synced'] = df[time_col] + time_offset

    # 保存
    with open(output_csv, 'w') as f:
        # 写入元数据
        for line in metadata_lines:
            f.write(line)
        # 写入数据
        df.to_csv(f, index=False)
```

---

## 错误处理和容错机制

### 设计原则

**核心理念**: **渐进式降级（Graceful Degradation）**

系统应该尽可能完成任务，即使部分数据缺失或质量不佳。优先级：
1. ✅ **报告问题** - 详细记录所有异常
2. ✅ **继续处理** - 使用可用数据完成同步
3. ✅ **提供选项** - 让用户决定下一步
4. ❌ **避免中断** - 只在无法继续时才停止

---

### 场景1: GoPro官方同步失败

**问题描述**: 通过QR码检查发现GoPro官方同步存在问题（相机间offset不一致）

**检测方法**:
```python
def verify_gopro_sync_quality(gopro_qr_data, threshold_frames=2):
    """
    验证GoPro官方同步质量

    检查所有GoPro相机是否在相同帧号看到相同的QR码

    Returns:
        {
            'sync_quality': 'good' | 'poor',
            'max_offset_frames': int,
            'problem_cameras': List[str],
            'report': str
        }
    """
    # 选择参考相机（通常是cam01）
    reference_cam = 'cam01'
    if reference_cam not in gopro_qr_data:
        reference_cam = list(gopro_qr_data.keys())[0]

    ref_qr = gopro_qr_data[reference_cam]

    # 检查每个相机
    max_offset = 0
    problem_cameras = []
    camera_offsets = {}

    for cam_id, qr_data in gopro_qr_data.items():
        if cam_id == reference_cam:
            continue

        # 找到共同的QR码
        common_qrs = set(ref_qr.keys()) & set(qr_data.keys())
        if len(common_qrs) < 5:
            problem_cameras.append(f"{cam_id} (QR不足: {len(common_qrs)})")
            continue

        # 计算帧号偏移
        offsets = []
        for qr_num in common_qrs:
            ref_frame = ref_qr[qr_num] * 60  # 转换为帧号
            cam_frame = qr_data[qr_num] * 60
            offset = abs(cam_frame - ref_frame)
            offsets.append(offset)

        avg_offset = np.mean(offsets)
        max_offset_this_cam = np.max(offsets)

        camera_offsets[cam_id] = {
            'avg_offset': avg_offset,
            'max_offset': max_offset_this_cam
        }

        if max_offset_this_cam > threshold_frames:
            problem_cameras.append(f"{cam_id} (偏移: {max_offset_this_cam:.1f}帧)")
            max_offset = max(max_offset, max_offset_this_cam)

    # 生成报告
    if problem_cameras:
        quality = 'poor'
        report = (
            f"⚠️  GoPro官方同步存在问题！\n"
            f"参考相机: {reference_cam}\n"
            f"最大偏移: {max_offset:.1f} 帧\n"
            f"问题相机: {', '.join(problem_cameras)}\n"
        )
    else:
        quality = 'good'
        report = (
            f"✓ GoPro官方同步质量良好\n"
            f"所有相机偏移 < {threshold_frames} 帧\n"
        )

    return {
        'sync_quality': quality,
        'max_offset_frames': max_offset,
        'problem_cameras': problem_cameras,
        'camera_offsets': camera_offsets,
        'report': report
    }
```

**处理流程**:
```python
def handle_gopro_sync_failure(gopro_qr_data, verification_result):
    """
    处理GoPro官方同步失败

    提供两个选项：
    1. 使用我们的QR码重新同步GoPro
    2. 保存当前数据并继续
    """
    print("\n" + "=" * 80)
    print("GoPro同步验证失败")
    print("=" * 80)
    print(verification_result['report'])

    print("\n可选操作:")
    print("  [1] 使用QR码重新同步GoPro（推荐）")
    print("  [2] 保存当前数据并继续（风险：同步质量差）")
    print("  [3] 取消操作")

    choice = input("\n请选择 [1/2/3]: ").strip()

    if choice == '1':
        print("\n开始使用QR码重新同步GoPro...")
        # 调用QR同步函数
        return resync_gopro_with_qr(gopro_qr_data)

    elif choice == '2':
        print("\n⚠️  警告: 继续使用质量不佳的同步数据")
        # 保存验证报告
        with open('gopro_sync_warning.txt', 'w') as f:
            f.write(verification_result['report'])
        print("验证报告已保存: gopro_sync_warning.txt")
        return 'continue'

    else:
        print("\n操作已取消")
        return 'cancel'
```

**非交互模式** (用于自动化脚本):
```python
def verify_gopro_sync_auto(gopro_qr_data, auto_resync=False, force_continue=False):
    """
    自动模式：不询问用户

    Args:
        auto_resync: 如果True，自动重新同步
        force_continue: 如果True，忽略问题继续
    """
    result = verify_gopro_sync_quality(gopro_qr_data)

    if result['sync_quality'] == 'poor':
        if auto_resync:
            return resync_gopro_with_qr(gopro_qr_data)
        elif force_continue:
            # 记录警告
            log_warning(result['report'])
            return 'continue'
        else:
            # 默认：中止
            raise Exception(result['report'])

    return 'good'
```

---

### 场景2: QR码检测不足或缺失

**问题描述**: 某些GoPro相机QR码检测数量不足（< 10个）或完全没有检测到

**检测方法**:
```python
def analyze_qr_detection_coverage(gopro_qr_data, primecolor_qr_data, min_qr_count=10):
    """
    分析QR码检测覆盖率

    Returns:
        {
            'gopro_coverage': {
                'good_cameras': List[str],
                'insufficient_cameras': Dict[str, int],  # cam_id: qr_count
                'missing_cameras': List[str]
            },
            'primecolor_coverage': {
                'qr_count': int,
                'status': 'good' | 'insufficient' | 'missing'
            },
            'usable_for_sync': bool,
            'report': str
        }
    """
    good_cameras = []
    insufficient_cameras = {}
    missing_cameras = []

    # 检查每个GoPro相机
    for cam_id, qr_data in gopro_qr_data.items():
        qr_count = len(qr_data)

        if qr_count == 0:
            missing_cameras.append(cam_id)
        elif qr_count < min_qr_count:
            insufficient_cameras[cam_id] = qr_count
        else:
            good_cameras.append(cam_id)

    # 检查PrimeColor
    prime_qr_count = len(primecolor_qr_data)
    if prime_qr_count >= min_qr_count:
        prime_status = 'good'
    elif prime_qr_count > 0:
        prime_status = 'insufficient'
    else:
        prime_status = 'missing'

    # 判断是否可用于同步
    usable_for_sync = len(good_cameras) > 0 and prime_status != 'missing'

    # 生成报告
    report_lines = ["QR码检测覆盖率分析:", ""]

    if good_cameras:
        report_lines.append(f"✓ 良好相机 ({len(good_cameras)}个):")
        for cam_id in good_cameras:
            report_lines.append(f"  - {cam_id}: {len(gopro_qr_data[cam_id])} QR码")

    if insufficient_cameras:
        report_lines.append(f"\n⚠️  QR码不足 ({len(insufficient_cameras)}个):")
        for cam_id, count in insufficient_cameras.items():
            report_lines.append(f"  - {cam_id}: {count} QR码 (< {min_qr_count})")

    if missing_cameras:
        report_lines.append(f"\n❌ 无QR码检测 ({len(missing_cameras)}个):")
        for cam_id in missing_cameras:
            report_lines.append(f"  - {cam_id}: 0 QR码")

    report_lines.append(f"\nPrimeColor: {prime_qr_count} QR码 ({prime_status})")

    if usable_for_sync:
        report_lines.append(f"\n✓ 可用于同步（使用{len(good_cameras)}个相机）")
    else:
        report_lines.append(f"\n❌ 无法同步（可用相机不足或PrimeColor无QR码）")

    return {
        'gopro_coverage': {
            'good_cameras': good_cameras,
            'insufficient_cameras': insufficient_cameras,
            'missing_cameras': missing_cameras
        },
        'primecolor_coverage': {
            'qr_count': prime_qr_count,
            'status': prime_status
        },
        'usable_for_sync': usable_for_sync,
        'report': '\n'.join(report_lines)
    }
```

**处理流程**:
```python
def handle_insufficient_qr_detection(coverage_result):
    """
    处理QR码检测不足的情况

    策略：
    1. 报告问题
    2. 使用可用的相机继续同步
    3. 在sync_info.json中标记问题
    """
    print("\n" + "=" * 80)
    print("QR码检测覆盖率检查")
    print("=" * 80)
    print(coverage_result['report'])

    if not coverage_result['usable_for_sync']:
        print("\n❌ 错误: 无法继续同步")
        print("建议:")
        print("  1. 检查视频质量（光照、焦距、QR码清晰度）")
        print("  2. 调整QR检测参数（减小step_frames，增加scan_duration）")
        print("  3. 重新拍摄QR码片段")
        raise Exception("QR码检测失败，无法同步")

    # 有足够的相机可以继续
    insufficient = coverage_result['gopro_coverage']['insufficient_cameras']
    missing = coverage_result['gopro_coverage']['missing_cameras']

    if insufficient or missing:
        print("\n⚠️  警告: 部分相机QR码检测不足")
        print(f"将使用 {len(coverage_result['gopro_coverage']['good_cameras'])} 个相机进行同步")

        # 记录到日志
        warning_msg = (
            f"QR码检测不足的相机:\n"
            f"  - 不足: {list(insufficient.keys())}\n"
            f"  - 缺失: {missing}\n"
            f"这些相机将不参与PrimeColor同步参数计算"
        )
        log_warning(warning_msg)

    # 返回可用的相机列表
    return coverage_result['gopro_coverage']['good_cameras']
```

**修改同步计算** (只使用有效相机):
```python
def calculate_sync_mapping_robust(
    gopro_qr_data,
    primecolor_qr_data,
    valid_cameras=None  # 新增参数
):
    """
    健壮的同步映射计算

    只使用valid_cameras中的相机数据
    """
    if valid_cameras is None:
        valid_cameras = list(gopro_qr_data.keys())

    # 过滤：只使用有效相机
    filtered_gopro_qr = {
        cam_id: qr_data
        for cam_id, qr_data in gopro_qr_data.items()
        if cam_id in valid_cameras
    }

    if not filtered_gopro_qr:
        raise ValueError("没有可用的GoPro相机数据")

    # 选择参考相机（优先使用cam01，如果不在valid中则选第一个）
    if 'cam01' in filtered_gopro_qr:
        reference_cam = 'cam01'
    else:
        reference_cam = list(filtered_gopro_qr.keys())[0]

    print(f"使用参考相机: {reference_cam}")
    print(f"参与计算的相机: {', '.join(filtered_gopro_qr.keys())}")

    # 调用原始计算函数
    return calculate_sync_mapping(
        filtered_gopro_qr,
        primecolor_qr_data,
        reference_cam=reference_cam
    )
```

---

### 场景3: 结尾QR码缺失（双端验证失败）

**问题描述**: 录制结束时没有拍摄QR码，导致无法进行双端验证

**检测方法**:
```python
def check_dual_qr_availability(
    gopro_qr_data,
    primecolor_qr_data,
    video_duration_sec,
    end_segment_start=None  # 结束段开始时间（秒）
):
    """
    检查是否有结尾QR码可用于双端验证

    Args:
        end_segment_start: 结束段开始时间（如总时长-120秒）

    Returns:
        {
            'start_segment_available': bool,
            'end_segment_available': bool,
            'can_verify': bool,
            'missing_cameras': List[str],
            'report': str
        }
    """
    if end_segment_start is None:
        end_segment_start = max(0, video_duration_sec - 120)

    # 检查开始段（0-120秒）
    start_qr_count = {}
    for cam_id, qr_data in gopro_qr_data.items():
        count = sum(1 for t in qr_data.values() if t < 120)
        start_qr_count[cam_id] = count

    prime_start_count = sum(1 for t in primecolor_qr_data.values() if t < 120)

    start_available = any(c > 5 for c in start_qr_count.values()) and prime_start_count > 5

    # 检查结束段
    end_qr_count = {}
    missing_end_cameras = []

    for cam_id, qr_data in gopro_qr_data.items():
        count = sum(1 for t in qr_data.values() if t >= end_segment_start)
        end_qr_count[cam_id] = count
        if count < 5:
            missing_end_cameras.append(cam_id)

    prime_end_count = sum(1 for t in primecolor_qr_data.values() if t >= end_segment_start)

    end_available = any(c > 5 for c in end_qr_count.values()) and prime_end_count > 5

    # 是否可以验证
    can_verify = start_available and end_available

    # 生成报告
    report_lines = ["双端QR验证可用性检查:", ""]

    if start_available:
        report_lines.append(f"✓ 开始段QR码: 可用")
    else:
        report_lines.append(f"❌ 开始段QR码: 不足")

    if end_available:
        report_lines.append(f"✓ 结束段QR码: 可用")
    else:
        report_lines.append(f"❌ 结束段QR码: 不足")
        if missing_end_cameras:
            report_lines.append(f"  缺失相机: {', '.join(missing_end_cameras)}")

    if can_verify:
        report_lines.append(f"\n✓ 可以进行双端验证")
    else:
        report_lines.append(f"\n⚠️  无法进行双端验证")
        report_lines.append(f"将使用单端验证（仅开始段）")

    return {
        'start_segment_available': start_available,
        'end_segment_available': end_available,
        'can_verify': can_verify,
        'missing_cameras': missing_end_cameras,
        'start_qr_count': start_qr_count,
        'end_qr_count': end_qr_count,
        'report': '\n'.join(report_lines)
    }
```

**处理流程**:
```python
def perform_sync_verification(
    gopro_synced_path,
    primecolor_synced_path,
    anchor_video,
    dual_qr_check_result
):
    """
    执行同步验证（自动选择单端或双端）

    Returns:
        verification_result 或 None（如果无法验证）
    """
    print("\n" + "=" * 80)
    print("同步质量验证")
    print("=" * 80)
    print(dual_qr_check_result['report'])

    if not dual_qr_check_result['start_segment_available']:
        print("\n❌ 错误: 连开始段QR码都不足，无法验证")
        return None

    if dual_qr_check_result['can_verify']:
        # 双端验证
        print("\n执行双端QR验证...")
        result = verify_sync_with_dual_qr(
            gopro_synced_path,
            primecolor_synced_path,
            anchor_video
        )
    else:
        # 单端验证（降级）
        print("\n⚠️  执行单端QR验证（结尾QR码不足）...")
        result = verify_sync_with_single_qr(
            gopro_synced_path,
            primecolor_synced_path,
            anchor_video,
            segment='start'
        )

        # 标记验证方法降级
        result['verification_method'] = 'single_qr_degraded'
        result['warning'] = '结尾QR码不足，无法检测时间漂移'

    return result
```

**在sync_info.json中记录**:
```json
{
  "verification": {
    "method": "single_qr_degraded",
    "sync_quality": "unknown",
    "start_segment": {
      "offset_frames": 145,
      "rmse_frames": 0.6
    },
    "end_segment": null,
    "drift_detection": "unavailable",
    "warning": "结尾QR码不足，无法进行双端验证和时间漂移检测",
    "missing_end_cameras": ["cam02", "cam04"]
  }
}
```

---

### 完整错误处理流程

```python
def sync_multi_camera_with_error_handling(
    gopro_dir,
    primecolor_video,
    anchor_video,
    mocap_csv=None,
    output_dir=None,
    auto_mode=False  # 自动模式：不询问用户
):
    """
    带完整错误处理的多相机同步主函数
    """
    # 初始化报告
    sync_report = {
        'gopro_sync_verification': None,
        'qr_detection_coverage': None,
        'dual_qr_availability': None,
        'verification_result': None,
        'warnings': [],
        'errors': []
    }

    try:
        # Step 1: 检测QR码
        print("\n[1/6] 检测QR码...")
        gopro_qr = detect_qr_all_gopro(gopro_dir, anchor_video)
        primecolor_qr = detect_qr_primecolor(primecolor_video)

        # Step 2: 检查QR码覆盖率
        print("\n[2/6] 检查QR码检测覆盖率...")
        coverage_result = analyze_qr_detection_coverage(gopro_qr, primecolor_qr)
        sync_report['qr_detection_coverage'] = coverage_result
        print(coverage_result['report'])

        if not coverage_result['usable_for_sync']:
            raise Exception("QR码检测失败，无法继续")

        # 获取有效相机列表
        valid_cameras = coverage_result['gopro_coverage']['good_cameras']
        if coverage_result['gopro_coverage']['insufficient_cameras'] or \
           coverage_result['gopro_coverage']['missing_cameras']:
            sync_report['warnings'].append(
                f"部分相机QR码不足，仅使用{len(valid_cameras)}个相机"
            )

        # Step 3: 验证GoPro官方同步质量
        print("\n[3/6] 验证GoPro官方同步质量...")
        gopro_verification = verify_gopro_sync_quality(gopro_qr)
        sync_report['gopro_sync_verification'] = gopro_verification
        print(gopro_verification['report'])

        if gopro_verification['sync_quality'] == 'poor':
            if auto_mode:
                sync_report['warnings'].append(
                    "GoPro官方同步质量差，但继续处理（自动模式）"
                )
            else:
                # 交互模式：询问用户
                action = handle_gopro_sync_failure(gopro_qr, gopro_verification)
                if action == 'cancel':
                    return None
                elif action == 'resync':
                    # 重新同步GoPro
                    gopro_qr = resync_gopro_with_qr(gopro_qr)

        # Step 4: 计算同步映射（使用有效相机）
        print("\n[4/6] 计算同步参数...")
        sync_params = calculate_sync_mapping_robust(
            gopro_qr,
            primecolor_qr,
            valid_cameras=valid_cameras
        )

        print(f"  Offset: {sync_params['offset_frames']} 帧")
        print(f"  FPS Ratio: {sync_params['fps_ratio']:.4f}")
        print(f"  QR Matches: {sync_params['qr_matches']}")
        print(f"  RMSE: {sync_params['rmse']:.2f} 帧")

        # Step 5: 对齐视频
        print("\n[5/6] 对齐PrimeColor视频...")
        primecolor_synced = output_dir / "primecolor_synced.mp4"
        align_primecolor_video(
            primecolor_video,
            str(primecolor_synced),
            sync_params['offset_frames']
        )

        # Step 6: 验证同步质量
        print("\n[6/6] 验证同步质量...")

        # 检查双端QR可用性
        video_duration = get_video_duration(primecolor_video)
        dual_qr_check = check_dual_qr_availability(
            gopro_qr,
            primecolor_qr,
            video_duration
        )
        sync_report['dual_qr_availability'] = dual_qr_check

        # 执行验证
        verification_result = perform_sync_verification(
            gopro_dir,
            str(primecolor_synced),
            anchor_video,
            dual_qr_check
        )
        sync_report['verification_result'] = verification_result

        if verification_result and 'warning' in verification_result:
            sync_report['warnings'].append(verification_result['warning'])

        # 同步Mocap CSV
        if mocap_csv:
            print("\n同步Mocap CSV...")
            mocap_synced = output_dir / "mocap_synced.csv"
            sync_mocap_csv(mocap_csv, str(mocap_synced), sync_params['offset_frames'])

        # 保存完整报告
        save_sync_report(output_dir / 'sync_info.json', sync_report, sync_params)

        print("\n" + "=" * 80)
        print("✅ 同步完成！")
        print("=" * 80)

        if sync_report['warnings']:
            print(f"\n⚠️  警告 ({len(sync_report['warnings'])}个):")
            for w in sync_report['warnings']:
                print(f"  - {w}")

        return sync_report

    except Exception as e:
        sync_report['errors'].append(str(e))
        print(f"\n❌ 错误: {e}")

        # 保存错误报告
        if output_dir:
            save_sync_report(output_dir / 'sync_error.json', sync_report, None)

        raise
```

---

## 代码实现框架

### 模块1: QR检测 (detect_qr_all_cameras.py)

```python
#!/usr/bin/env python3
"""
多相机QR码检测模块
基于 sync_with_qr_anchor.py 扩展
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False

def detect_qr_fast(frame: np.ndarray) -> List[str]:
    """快速QR检测（pyzbar + OpenCV双引擎）"""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # 降采样加速
    if gray.shape[0] > 1080:
        scale = 1080.0 / gray.shape[0]
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)

    results = []

    # 优先pyzbar
    if HAS_PYZBAR:
        detected = pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE])
        if detected:
            for obj in detected:
                results.append(obj.data.decode('utf-8'))

    # 备用OpenCV
    if not results:
        detector = cv2.QRCodeDetector()
        data, vertices, _ = detector.detectAndDecode(gray)
        if data:
            results.append(data)

    return results

def scan_video_qr_segment(
    video_path: str,
    start_time: float = 0.0,
    duration: float = 60.0,
    frame_step: int = 2,
    prefix: str = ""
) -> Dict[int, float]:
    """
    扫描视频片段中的QR码

    Returns:
        {qr_frame_number: video_time_seconds, ...}
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps)
    end_frame = min(int((start_time + duration) * fps), total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    qr_detections = {}
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_step == 0:
            qr_codes = detect_qr_fast(frame)
            for qr_data in qr_codes:
                # 解析QR帧号
                if prefix and qr_data.startswith(prefix):
                    qr_data = qr_data[len(prefix):]
                try:
                    qr_frame_num = int(qr_data)
                    video_time = frame_idx / fps
                    if qr_frame_num not in qr_detections:
                        qr_detections[qr_frame_num] = video_time
                except:
                    pass

        frame_idx += 1

    cap.release()
    return qr_detections

def detect_qr_all_gopro(
    gopro_dir: str,
    anchor_video: str,
    scan_params: dict = None
) -> Dict[str, Dict[int, float]]:
    """
    检测所有GoPro视频中的QR码

    Returns:
        {
            'cam01': {qr_num: frame_time, ...},
            'cam02': {qr_num: frame_time, ...},
            ...
        }
    """
    if scan_params is None:
        scan_params = QR_DETECTION_CONFIG

    gopro_path = Path(gopro_dir)
    results = {}

    # 查找所有cam*文件夹
    for cam_folder in sorted(gopro_path.glob('cam*')):
        cam_id = cam_folder.name

        # 查找视频文件
        video_files = list(cam_folder.glob('*.MP4')) + list(cam_folder.glob('*.mp4'))
        if not video_files:
            continue

        video_path = str(video_files[0])
        print(f"检测 {cam_id}...")

        qr_data = scan_video_qr_segment(
            video_path,
            start_time=scan_params['scan_start_sec'],
            duration=scan_params['scan_duration_sec'],
            frame_step=scan_params['step_frames']
        )

        results[cam_id] = qr_data
        print(f"  ✓ 检测到 {len(qr_data)} 个QR码")

    return results

def detect_qr_primecolor(
    primecolor_video: str,
    scan_params: dict = None
) -> Dict[int, float]:
    """
    检测PrimeColor视频中的QR码

    Returns:
        {qr_num: frame_time, ...}
    """
    if scan_params is None:
        scan_params = QR_DETECTION_CONFIG

    print(f"检测PrimeColor视频...")
    qr_data = scan_video_qr_segment(
        primecolor_video,
        start_time=scan_params['scan_start_sec'],
        duration=scan_params['scan_duration_sec'],
        frame_step=scan_params['step_frames']
    )

    print(f"  ✓ 检测到 {len(qr_data)} 个QR码")
    return qr_data
```

### 模块2: 时间映射 (calculate_sync_mapping.py)

```python
#!/usr/bin/env python3
"""
时间映射计算模块
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict

def calculate_sync_mapping(
    gopro_qr_data: Dict[str, Dict[int, float]],
    primecolor_qr_data: Dict[int, float],
    gopro_fps: float = 60.0,
    primecolor_fps: float = 120.0,
    reference_cam: str = 'cam01'
) -> Dict:
    """
    计算同步映射参数

    Args:
        gopro_qr_data: {cam_id: {qr_num: video_time}}
        primecolor_qr_data: {qr_num: video_time}
        gopro_fps: GoPro FPS
        primecolor_fps: PrimeColor FPS
        reference_cam: 参考相机ID

    Returns:
        {
            'offset_frames': int,
            'fps_ratio': float,
            'qr_matches': int,
            'max_error': float,
            'rmse': float,
            'residuals': List[float]
        }
    """
    # 选择参考GoPro相机
    if reference_cam not in gopro_qr_data:
        reference_cam = list(gopro_qr_data.keys())[0]

    gopro_ref = gopro_qr_data[reference_cam]

    # 找到共同的QR码
    common_qrs = set(gopro_ref.keys()) & set(primecolor_qr_data.keys())

    if len(common_qrs) < 10:
        raise ValueError(f"QR匹配数量不足: {len(common_qrs)} < 10")

    # 转换时间到帧号
    gopro_frames = [gopro_ref[qr] * gopro_fps for qr in common_qrs]
    primecolor_frames = [primecolor_qr_data[qr] * primecolor_fps for qr in common_qrs]

    # 拟合线性模型
    def residual(params, f_g, f_p):
        offset, fps_ratio = params
        predicted = offset + fps_ratio * f_g
        return predicted - f_p

    result = least_squares(
        residual,
        x0=[0, primecolor_fps / gopro_fps],
        args=(np.array(gopro_frames), np.array(primecolor_frames))
    )

    offset, fps_ratio = result.x
    residuals = result.fun

    return {
        'offset_frames': int(round(offset)),
        'fps_ratio': float(fps_ratio),
        'qr_matches': len(common_qrs),
        'max_error': float(np.max(np.abs(residuals))),
        'rmse': float(np.sqrt(np.mean(residuals**2))),
        'residuals': residuals.tolist(),
        'reference_camera': reference_cam
    }
```

### 模块3: 主脚本 (sync_multi_camera_with_qr.py)

```python
#!/usr/bin/env python3
"""
多相机QR同步主脚本
"""

import argparse
import json
from pathlib import Path

from detect_qr_all_cameras import detect_qr_all_gopro, detect_qr_primecolor
from calculate_sync_mapping import calculate_sync_mapping
# from align_primecolor_video import align_video
# from sync_mocap_csv import sync_csv

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Camera Synchronization with QR Code'
    )
    parser.add_argument('--gopro-dir', required=True,
                       help='GoPro synced directory')
    parser.add_argument('--primecolor-video', required=True,
                       help='PrimeColor video path')
    parser.add_argument('--anchor-video', required=True,
                       help='QR anchor video path')
    parser.add_argument('--mocap-csv', default=None,
                       help='Optional: Mocap CSV path')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory')
    parser.add_argument('--scan-start', type=float, default=5,
                       help='QR scan start time (seconds)')
    parser.add_argument('--scan-duration', type=float, default=60,
                       help='QR scan duration (seconds)')
    parser.add_argument('--step-frames', type=int, default=2,
                       help='Frame step for QR detection')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # QR检测配置
    scan_params = {
        'scan_start_sec': args.scan_start,
        'scan_duration_sec': args.scan_duration,
        'step_frames': args.step_frames
    }

    print("=" * 80)
    print("Multi-Camera Synchronization")
    print("=" * 80)

    # Step 1: 检测QR码
    print("\n[1/5] Detecting QR codes...")
    gopro_qr = detect_qr_all_gopro(
        args.gopro_dir,
        args.anchor_video,
        scan_params
    )
    primecolor_qr = detect_qr_primecolor(
        args.primecolor_video,
        scan_params
    )

    # Step 2: 计算同步参数
    print("\n[2/5] Calculating sync parameters...")
    sync_params = calculate_sync_mapping(gopro_qr, primecolor_qr)

    print(f"  Offset: {sync_params['offset_frames']} frames")
    print(f"  FPS Ratio: {sync_params['fps_ratio']:.4f}")
    print(f"  QR Matches: {sync_params['qr_matches']}")
    print(f"  Max Error: {sync_params['max_error']:.2f} frames")
    print(f"  RMSE: {sync_params['rmse']:.2f} frames")

    # Step 3: 对齐PrimeColor视频
    print("\n[3/5] Aligning PrimeColor video...")
    output_video = output_dir / "primecolor_synced.mp4"
    # align_video(args.primecolor_video, str(output_video), sync_params['offset_frames'])
    print(f"  TODO: Implement video alignment")

    # Step 4: 同步Mocap CSV
    if args.mocap_csv:
        print("\n[4/5] Synchronizing Mocap CSV...")
        output_csv = output_dir / "mocap_synced.csv"
        # sync_csv(args.mocap_csv, str(output_csv), sync_params['offset_frames'])
        print(f"  TODO: Implement CSV sync")

    # Step 5: 保存元数据
    print("\n[5/5] Saving sync metadata...")
    metadata = {
        'reference_camera': sync_params.get('reference_camera', 'cam01'),
        'sync_method': 'qr_anchor_video',
        'anchor_video': args.anchor_video,
        'gopro': {
            'fps': 60,
            'num_cameras': len(gopro_qr)
        },
        'primecolor': {
            'fps': 120,
            'offset_frames': sync_params['offset_frames'],
            'offset_seconds': sync_params['offset_frames'] / 120.0,
            'fps_ratio': sync_params['fps_ratio'],
            'qr_residuals': {
                'max_error_frames': sync_params['max_error'],
                'rmse_frames': sync_params['rmse'],
                'num_qr_matched': sync_params['qr_matches']
            }
        }
    }

    with open(output_dir / 'sync_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Synchronization Complete!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Sync metadata: {output_dir / 'sync_info.json'}")

if __name__ == '__main__':
    main()
```

---

## 测试数据

### 已分析的数据

**PrimeColor视频**:
- 文件: `/Volumes/FastACIS/csl-data/sync.avi`
- FPS: 120
- 分辨率: 1920x1080
- 时长: 29.07秒
- 总帧数: 3488
- 文件大小: 414.53 MB

**Mocap CSV**:
- 文件: `/Volumes/FastACIS/csl-data/Take 2025-10-24 04.01.02 PM.csv`
- FPS: 120
- 时长: 75.56秒
- 总帧数: 9067
- Marker数量: ~200
- 文件大小: 13.5 MB

### 运行数据分析

```bash
cd /Volumes/FastACIS/annotation_pipeline

python sync/analyze_sync_data.py

# 输出: sync/sync_analysis_results.json
```

---

## 实施清单

### Phase 1: 数据收集 ✅

- [x] Mocap CSV结构分析
- [x] PrimeColor视频参数分析
- [x] FPS关系验证（2:1完美比率）
- [x] 数据分析脚本创建

### Phase 2: 代码开发 (进行中)

- [ ] **Module 1**: `detect_qr_all_cameras.py`
  - 扩展现有`sync_with_qr_anchor.py`
  - 支持多相机批量检测
  - 优化检测速度（pyzbar + OpenCV）

- [ ] **Module 2**: `calculate_sync_mapping.py`
  - 实现最小二乘拟合
  - QR匹配和残差分析
  - 验证FPS比率

- [ ] **Module 3**: `align_primecolor_video.py`
  - ffmpeg视频对齐
  - 黑帧填充/裁剪
  - MJPEG → H.264转码

- [ ] **Module 4**: `sync_mocap_csv.py`
  - CSV读取（跳过元数据）
  - 帧偏移应用
  - 保留元数据行

- [ ] **Module 5**: `sync_multi_camera_with_qr.py`
  - 主脚本整合
  - 参数解析
  - 进度输出

- [ ] **Module 6**: `verify_sync_quality.py`
  - 双端QR验证
  - 时间漂移检测
  - 质量报告生成

### Phase 3: 测试

- [ ] 单元测试各模块
- [ ] 端到端集成测试
- [ ] 验证同步精度（目标 < 2帧）
- [ ] 性能测试（处理时间）

### Phase 4: 文档

- [x] README.md（用户指南）
- [x] TECHNICAL_NOTES.md（技术参考）
- [ ] 使用示例和troubleshooting

---

## 依赖项

```bash
# Python依赖
pip install opencv-python opencv-contrib-python
pip install pyzbar  # 或 zbarlight
pip install numpy scipy pandas

# 系统依赖
# macOS:
brew install ffmpeg

# Ubuntu:
sudo apt-get install ffmpeg libzbar0
```

---

## 性能优化

### QR检测优化

```python
# 降采样加速
if frame.shape[0] > 1080:
    scale = 1080.0 / frame.shape[0]
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

# 增加步长（每N帧检测一次）
step_frames = 2  # 默认每2帧检测一次

# 优先使用pyzbar（比OpenCV快3-5倍）
HAS_PYZBAR = True
```

### 视频处理优化

```python
# MJPEG → H.264转码（减小文件大小）
ffmpeg_params = [
    '-c:v', 'libx264',
    '-preset', 'medium',  # 或 'fast' (更快但文件稍大)
    '-crf', '18',         # 高质量
    '-pix_fmt', 'yuv420p'
]

# 黑帧填充使用color filter（GPU加速）
color_filter = 'color=c=black:s=1920x1080:r=120'
```

---

## 常见错误和解决方法

### QR检测失败

**症状**: `QR匹配数量不足: N < 10`

**原因**:
- 光照不足
- QR码模糊
- 角度太斜
- 采样不足

**解决**:
```python
# 调整检测参数
QR_DETECTION_CONFIG = {
    'step_frames': 1,        # 减小步长（更密集）
    'scan_duration_sec': 120 # 增加扫描时长
}

# 增强对比度
frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
```

### FPS比率异常

**症状**: `fps_ratio` 不接近2.0（如1.95或2.10）

**原因**:
- 相机实际FPS与标称FPS不符
- 视频编码时FPS被改变

**解决**:
```bash
# 检查实际FPS
ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate video.mp4

# 使用检测到的FPS重新计算
```

### 时间漂移过大

**症状**: `drift_frames > 2`

**原因**:
- 相机FPS不稳定（59.97 vs 60.00）
- 长时间录制累积误差

**解决**:
- 考虑使用更稳定的相机
- 缩短单次录制时长
- 使用分段同步

---

## 参考文献

1. QR Code同步原理: [sync_with_qr_anchor.py](/Volumes/FastACIS/annotation_pipeline/sync/sync_with_qr_anchor.py)
2. GoPro Timecode同步: [scripts/sync_timecode.py](/Volumes/FastACIS/annotation_pipeline/scripts/sync_timecode.py)
3. Optitrack Motive CSV格式: 官方文档
4. 最小二乘法: scipy.optimize.least_squares文档

---

**版本**: 1.0
**最后更新**: 2025-11-03
**维护者**: Annotation Pipeline Team
