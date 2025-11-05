# 多相机系统完整工作流程文档

## 概述

本文档描述了**16 GoPro相机 + 2 PrimeColor相机 + Mocap系统**的完整标定与同步流程。

### 当前状态

#### ✅ 已完成
1. **16个GoPro相机的内参标定** - 存储在 `intrinsic_hyperoff_linear_60fps.json`
2. **GoPro视频硬件时间码同步系统** - 基于嵌入式timecode
3. **GoPro外参标定功能** - 基于ChArUco标定板
4. **完整的3D姿态估计pipeline** - 2D检测→三角化→3D重建

#### ⚠️ 待实现
1. **PrimeColor相机内参标定** - 需要ChArUco板拍摄
2. **PrimeColor与GoPro的外参标定** - 需要联合标定
3. **PrimeColor视频同步** - 目前无硬件timecode，需要其他方案
4. **Mocap系统外参** - 需要与相机系统的联合标定

---

## 第一阶段：GoPro相机系统（16相机）

### 1.1 硬件配置

- **相机数量**: 16台GoPro
- **相机编号**: cam1-cam18（部分编号）
- **拍摄设置**:
  - 分辨率: 4K (3840x2160)
  - 帧率: 60fps
  - 镜头模式: Linear（线性）
  - HyperSmooth: OFF（关闭）
- **同步方式**: 硬件timecode（嵌入到视频流）

### 1.2 内参标定（已完成）

#### 文件位置
```
/Volumes/FastACIS/annotation_pipeline/intrinsic_hyperoff_linear_60fps.json
```

#### 内参格式
```json
{
  "cameras": {
    "cam1": {
      "model": "standard",
      "image_size": [3840, 2160],
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "dist": [[k1, k2, p1, p2, k3]],
      "fov": {"horizontal": 93.2, "vertical": 61.5, "diagonal": 101.0},
      "rms": 0.41
    },
    ...共16个相机
  }
}
```

#### 质量指标
- **RMS误差**: 0.35-0.44像素（优秀）
- **FOV**: 水平93°, 垂直61°（符合GoPro Linear模式）

#### 如果你没有16个GoPro相机怎么办？

**问题**: 预存的 `intrinsic_hyperoff_linear_60fps.json` 包含16个相机（cam1-cam18），但你可能只有3个、5个或其他数量的相机。

**解决方案**: 使用 `filter_intrinsics.py` 从完整JSON中提取你需要的相机内参。

##### 方法A: 手动指定相机列表

```bash
# 假设你只有 cam1, cam2, cam4 三个相机
python filter_intrinsics.py \
  --input intrinsic_hyperoff_linear_60fps.json \
  --output intrinsic_my_cameras.json \
  --cameras cam1,cam2,cam4
```

**输出**: `intrinsic_my_cameras.json` 只包含指定的3个相机的内参。

##### 方法B: 从视频目录自动检测

```bash
# 自动检测 /Volumes/FastACIS/gopro/ex/ 目录中的相机
python filter_intrinsics.py \
  --input intrinsic_hyperoff_linear_60fps.json \
  --output intrinsic_my_cameras.json \
  --auto-detect /Volumes/FastACIS/gopro/ex/
```

**工作原理**:
1. 扫描目录中的 `cam*.MP4` 或 `cam*/` 文件夹
2. 自动提取相机名称列表（如 cam1, cam2, cam4）
3. 从完整内参JSON中过滤出这些相机

**输出示例**:
```
读取: intrinsic_hyperoff_linear_60fps.json

原始内参包含 16 个相机:
  cam1, cam10, cam11, cam12, cam15, cam16, cam17, cam18, cam2, cam3, ...

  ✓ 包含 cam1
  ✓ 包含 cam2
  ✓ 包含 cam4

保存到: intrinsic_my_cameras.json

✅ 完成！过滤后包含 3 个相机
  cam1, cam2, cam4
```

**注意事项**:
- ⚠️ 如果你的相机（如cam13）不在预存内参中，工具会显示警告
- ⚠️ 这种情况下需要单独为该相机标定内参（见下文）
- ✅ `filter_intrinsics.py` 也会自动过滤 `camera_base2cam`（外参）等其他字段

---

### 1.3 GoPro视频同步

#### 原理
使用GoPro嵌入的硬件timecode（格式：`HH:MM:SS:FF`）进行同步：
1. 提取每个视频的timecode和时长
2. 计算所有视频的公共时间窗口：`[max(start_times), min(end_times)]`
3. 用ffmpeg裁剪到同步时间段

#### 命令
```bash
cd /Volumes/FastACIS/annotation_pipeline

# 同步16个GoPro视频
python scripts/sync_timecode.py \
  --src_tag <recording_name> \
  --out_tag <recording_name>_synced \
  --fast_copy
```

#### 参数说明
- `--src_tag`: 输入视频目录（支持绝对路径或相对于 `PATH_ASSETS_VIDEOS`）
- `--out_tag`: 输出目录
- `--fast_copy`: 使用视频流复制（快速，1-2帧误差）；不加此参数则重新编码（慢但精确）
- `--stacked`: 可选，生成堆叠预览视频

#### 输入目录结构
```
<src_tag>/
├── cam1/video.MP4
├── cam2/video.MP4
├── ...
└── cam16/video.MP4
```

或平铺结构：
```
<src_tag>/
├── cam1.MP4
├── cam2.MP4
└── ...
```

#### 输出
```
<out_tag>/
├── cam1/video.MP4          # 同步后的视频
├── cam2/video.MP4
├── ...
├── meta_info.json          # 同步元数据
└── stacked_output.mp4      # (可选) 堆叠预览
```

**meta_info.json格式**:
```json
{
  "dir_src": "/path/to/source",
  "dir_out": "/path/to/output",
  "info_cam": {
    "cam1/video.MP4": {
      "src_timecode": "12:34:56:00",
      "src_duration": 120.5,
      "offset": 2.3,        // 需要裁剪的起始偏移（秒）
      "duration": 100.0,    // 同步窗口时长（秒）
      "fps": 60
    },
    ...
  }
}
```

#### 验证同步
```bash
# 方法1: 查看meta_info.json中的offset和duration
cat <out_tag>/meta_info.json

# 方法2: 提取帧图像，手动检查timecode显示是否对齐
python scripts/convert_video_to_images.py \
  --src_tag <recording_name>_synced \
  --cam_tags cam1,cam2,cam3 \
  --fps 1 \
  --duration 10
```

---

### 1.4 GoPro外参标定

#### 准备工作
1. **标定板**: ChArUco标定板（5x9或10x14格子）
2. **拍摄要求**:
   - 所有16个GoPro同时拍摄标定板
   - 标定板静止，移动相机或保持静止
   - 确保每个相机至少100帧清晰可见标定板
   - 标定板覆盖视野的不同位置和角度

#### 数据准备

**步骤1: 提取标定视频帧**
```bash
cd /Volumes/FastACIS/annotation_pipeline

# 如果标定视频已同步
python scripts/convert_video_to_images.py \
  --src_tag /Volumes/FastACIS/gopro/ex_synced\
  --cam_tags cam1,cam2,cam4\
  --fps 5 \
  --ss 2 \
  --duration 110
```

输出: `<calibration_recording>_synced/original/cam*/frame_*.png`

#### 外参标定命令

```bash
cd /Volumes/FastACIS/annotation_pipeline/multical

# 使用预存内参，只标定外参
python calibrate.py \
  --boards ./asset/charuco_b1_2.yaml \
  --image_path "/Volumes/FastACIS/gopro/ex_synced/original" \
  --calibration /Volumes/FastACIS/annotation_pipeline/intrinsic_my_cameras.json\
  --fix_intrinsic \
  --limit_images 1000 \
  --vis
```

#### 参数说明
- `--boards`: ChArUco板配置文件（定义板子几何参数）
- `--image_path`: 包含cam*/目录的路径（相对于 `PATH_ASSETS_VIDEOS`）
- `--calibration`: 内参JSON文件路径
- `--fix_intrinsic`: **关键！锁定内参，只优化外参**
- `--limit_images`: 每个相机使用的最大图像数量
- `--vis`: 生成可视化结果（检测的角点+3D坐标轴投影）

#### 输出文件

```
<calibration_recording>_synced/original/
├── calibration.json          # 最终标定结果
└── vis/
    ├── cam1/, cam2/, ...     # 可视化图像
    └── (角点检测+3D坐标轴投影)
```

**calibration.json格式**:
```json
{
  "cameras": {
    "cam1": {
      "K": [[...]],
      "dist": [[...]]
    },
    ...
  },
  "camera_base2cam": {          # 外参：相机相对位置
    "cam1": {
      "R": [[...], [...], [...]],  # 3x3旋转矩阵
      "T": [tx, ty, tz]            # 3D平移向量
    },
    ...
  }
}
```

#### 质量验证
```bash
# 1. 查看终端输出的RMS误差
# 应该看到: Final reprojection RMS=0.4-0.8 (期望 < 1.0像素)

# 2. 检查可视化结果
ls <calibration_recording>_synced/original/vis/cam*/
# 打开图像，检查：
# - ChArUco角点被正确检测（黄色圆圈）
# - 3D坐标轴投影正确（红=X，绿=Y，蓝=Z）
# - Z轴指向标定板内部

# 3. 计算FOV验证
python tool_scripts/intrinsics_to_fov.py
```

---

## 第二阶段：PrimeColor相机系统（2相机）

### 2.1 硬件配置（待确认）

- **相机数量**: 2台PrimeColor
- **拍摄设置**: 待确认（分辨率、帧率）
- **同步方式**: ⚠️ **问题：PrimeColor没有硬件timecode**

### 2.2 PrimeColor内参标定（待实现）

#### 所需材料
1. ChArUco标定板（与GoPro使用同一块）
2. 每个PrimeColor相机拍摄100+张不同角度的标定板图像

#### 标定步骤

**步骤1: 拍摄ChArUco板**
```bash
# 从PrimeColor视频提取帧
python scripts/convert_video_to_images.py \
  --src_tag /Volumes/FastACIS/csldata/video/exandin \
  --cam_tags primecolor \
  --fps 5 \
  --duration 130
```

**步骤2: 运行内参标定**
```bash
cd /Volumes/FastACIS/annotation_pipeline/multical

python intrinsic.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path /Volumes/FastACIS/csldata/video/exandin/original/ \
  --limit_images 300 \
  --limit_intrinsic 300 \
  --vis
```

**步骤3: 保存内参**
```bash
# 输出: intrinsic.json
# 将PrimeColor的内参合并到一个新的JSON文件
cp intrinsic.json ../primecolor_intrinsics.json
```

---

### 2.3 PrimeColor视频同步（缺失功能）

#### 问题
- PrimeColor视频**没有嵌入timecode**
- 当前 `scripts/sync_timecode.py` 无法处理PrimeColor

#### 解决方案（需要实现）

##### 方案A: 闪光灯/LED同步（推荐）
在拍摄开始时使用明显的视觉信号：
1. 拍摄时使用闪光灯或LED闪烁
2. 检测所有视频中闪光出现的帧
3. 将闪光帧作为t=0对齐

**实现步骤**:
```python
# 需要新建: scripts/sync_by_flash.py
# 功能:
# 1. 检测每个视频的最亮帧（闪光）
# 2. 计算帧偏移量
# 3. 用ffmpeg裁剪视频对齐
```

##### 方案B: 音频同步
如果PrimeColor录制了音频：
1. 拍摄时使用拍板或响指
2. 检测音频波形的峰值
3. 对齐音频峰值

**实现步骤**:
```python
# 需要新建: scripts/sync_by_audio.py
# 使用librosa库检测音频峰值
```

##### 方案C: 手动同步
1. 在视频编辑软件中手动对齐
2. 记录每个视频的偏移量和时长
3. 手动创建 `meta_info.json`
4. 用ffmpeg批量裁剪

---

## 第三阶段：GoPro与PrimeColor联合标定（待实现）

### 3.1 目标
获得**18个相机（16 GoPro + 2 PrimeColor）的统一外参矩阵**。

### 3.2 方案A: 联合ChArUco标定（推荐）

#### 要求
- 所有18个相机同时拍摄同一块ChArUco板
- 标定板在所有相机视野中可见
- 拍摄至少100-200帧

#### 步骤

**步骤1: 拍摄联合标定视频**
```bash
# 16个GoPro + 2个PrimeColor同时拍摄ChArUco板
# 注意: 需要先同步视频（见第二阶段）
```

**步骤2: 准备联合内参文件**
```bash
# 合并GoPro和PrimeColor的内参
cd /Volumes/FastACIS/annotation_pipeline

# 创建: combined_intrinsics.json
{
  "cameras": {
    "cam1": { ... },        // 16个GoPro
    ...
    "cam16": { ... },
    "primecolor1": { ... }, // 2个PrimeColor
    "primecolor2": { ... }
  }
}
```

**步骤3: 提取帧**
```bash
python scripts/convert_video_to_images.py \
  --src_tag <joint_calib_recording>_synced \
  --cam_tags cam1,...,cam16,primecolor1,primecolor2 \
  --fps 5 \
  --duration 60
```

**步骤4: 联合外参标定**
```bash
cd multical

python calibrate.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path "<joint_calib_recording>_synced/original" \
  --calibration ../combined_intrinsics.json \
  --fix_intrinsic \
  --limit_images 300 \
  --vis
```

**输出**: `calibration.json` 包含所有18个相机的外参

---

### 3.3 方案B: 分步标定 + 共同点配准

如果无法让所有相机同时看到ChArUco板：

#### 步骤1: 分别标定
```bash
# 1. GoPro系统外参（已完成）
# 输出: gopro_calibration.json

# 2. PrimeColor+部分GoPro外参
# 选择2-3个GoPro与PrimeColor重叠视野较大的位置
# 输出: primecolor_partial_calibration.json
```

#### 步骤2: 坐标系转换
```bash
# 需要实现: scripts/merge_calibrations.py
# 功能:
# 1. 找到两个标定中的共同相机（overlap GoPros）
# 2. 计算坐标系转换矩阵
# 3. 将PrimeColor外参转换到GoPro坐标系
# 4. 合并为统一的calibration.json
```

---

## 第四阶段：Mocap系统标定（待实现）

### 4.1 目标
获得**Mocap世界坐标系与相机坐标系的转换矩阵**。

### 4.2 问题
- Mocap输出的是3D标记点（marker）坐标
- 需要将Mocap坐标映射到相机坐标系

### 4.3 方案A: Marker棒标定（推荐）

#### 原理
使用带有已知几何的标记棒（rigid body）：
1. Mocap系统跟踪标记棒的3D位置
2. 相机拍摄标记棒上的标记点（如反光球）
3. 通过PnP算法求解坐标系转换

#### 步骤

**步骤1: 准备标记棒**
- 标记棒上至少4个非共面的标记点
- 精确测量标记点之间的3D几何关系
- 记录标记点在Mocap系统中的ID

**步骤2: 同步拍摄**
```bash
# 相机系统和Mocap系统同时记录
# 标记棒在相机视野内移动，覆盖不同位置和姿态
```

**步骤3: 提取数据**
```bash
# Mocap数据: 标记点的3D位置 (Nx3)
# 相机数据: 标记点的2D投影 (Nx2)

# 需要实现: scripts/extract_mocap_markers.py
# 从Mocap的.tak或.c3d文件提取标记点坐标
```

**步骤4: 标定**
```bash
# 需要实现: scripts/calibrate_mocap_camera.py
# 功能:
# 1. 匹配Mocap 3D点与相机2D投影
# 2. 使用PnP求解Mocap->Camera转换矩阵
# 3. 输出mocap_to_camera.json
```

**输出格式**:
```json
{
  "mocap_to_camera": {
    "R": [[...], [...], [...]],  // 旋转矩阵
    "T": [tx, ty, tz]            // 平移向量
  },
  "reprojection_error": 5.2      // 像素
}
```

---

### 4.4 方案B: ChArUco板 + Mocap标记

#### 原理
在ChArUco板上附加Mocap标记点：
1. 相机检测ChArUco角点
2. Mocap跟踪板上的标记点
3. 建立ChArUco坐标系与Mocap坐标系的关系

#### 优势
- 可以同时完成相机外参和Mocap标定
- 更高精度（ChArUco角点检测精度高）

---

## 第五阶段：同步所有系统（待实现）

### 5.1 时间同步方案

#### 当前状态
- ✅ 16 GoPro: 硬件timecode同步
- ❌ 2 PrimeColor: 无timecode，待实现
- ❌ Mocap系统: 需要与视频同步

#### 目标
所有系统共享统一的时间轴：`t=0` 对应同一物理时刻。

### 5.2 同步流程

#### 方案A: 分步同步（推荐）

```
第1步: GoPro内部同步（已实现）
  16 GoPro -> timecode同步 -> 16个对齐视频

第2步: PrimeColor同步（待实现）
  2 PrimeColor -> 闪光/音频同步 -> 2个对齐视频

第3步: GoPro-PrimeColor同步（待实现）
  方法: 在GoPro和PrimeColor同时可见的场景中使用闪光信号
  输出: 时间偏移量 (GoPro_t0 - PrimeColor_t0)

第4步: 视频-Mocap同步（待实现）
  方法:
  - 在拍摄开始时，标记棒做一个明显的动作（如快速抬起）
  - 在视频中手动标记该动作的帧号
  - 在Mocap数据中找到相同动作的时间戳
  输出: 时间偏移量 (Video_t0 - Mocap_t0)
```

#### 实现

**需要新建: scripts/sync_all_systems.py**
```python
# 功能:
# 1. 读取各系统的时间偏移量
# 2. 将所有视频和Mocap数据对齐到统一时间轴
# 3. 输出同步元数据 (sync_metadata.json)

# 输出格式:
{
  "reference_system": "gopro",
  "gopro_offset": 0.0,              # 参考系统，偏移为0
  "primecolor_offset": 2.3,         # 相对GoPro的秒数偏移
  "mocap_offset": -0.5,             # 相对GoPro的秒数偏移
  "common_duration": 120.0,         # 所有系统的公共时长
  "fps": {
    "gopro": 60,
    "primecolor": 30,
    "mocap": 120
  }
}
```

---

## 数据组织结构

### 标准目录结构

```
/Volumes/FastACIS/csltest1/
├── gopros/                          # 原始GoPro视频
│   ├── cam1/
│   │   ├── calibration.MP4          # 标定视频
│   │   ├── recording1.MP4           # 采集视频
│   │   └── recording2.MP4
│   ├── cam2/
│   │   └── ...
│   └── ...
│
├── primecolor/                      # 原始PrimeColor视频
│   ├── primecolor1/
│   │   ├── calibration.avi
│   │   └── recording1.avi
│   └── primecolor2/
│       └── ...
│
├── mocap/                           # Mocap数据
│   ├── calibration.tak
│   ├── recording1.tak
│   └── recording1.c3d
│
└── output/                          # 处理后的数据
    ├── gopro_synced/                # GoPro同步视频
    │   ├── cam1/, cam2/, ...
    │   └── meta_info.json
    │
    ├── primecolor_synced/           # PrimeColor同步视频
    │   ├── primecolor1/, primecolor2/
    │   └── meta_info.json
    │
    ├── calibration/                 # 标定结果
    │   ├── gopro_intrinsics.json
    │   ├── primecolor_intrinsics.json
    │   ├── gopro_extrinsics.json
    │   ├── combined_extrinsics.json # 18相机联合外参
    │   └── mocap_to_camera.json     # Mocap-相机转换
    │
    ├── sync_metadata.json           # 全局时间同步
    │
    └── recordings/                  # 采集数据处理
        ├── recording1/
        │   ├── original/            # 提取的帧图像
        │   │   ├── cam1/, cam2/, ...
        │   │   ├── primecolor1/, primecolor2/
        │   │   └── calibration.json
        │   ├── results/             # Pipeline结果
        │   │   ├── bbox/
        │   │   ├── vitpose/
        │   │   ├── triangulation/
        │   │   └── refined3d/
        │   └── mocap/               # 对齐的Mocap数据
        │       └── markers_synced.json
        └── recording2/
            └── ...
```

---

## 完整Pipeline命令总结

### 阶段1: GoPro标定与同步（已实现）

```bash
# 1. GoPro视频同步
python scripts/sync_timecode.py \
  --src_tag gopros \
  --out_tag output/gopro_synced \
  --fast_copy

# 2. 提取标定帧
python scripts/convert_video_to_images.py \
  --src_tag output/gopro_synced \
  --cam_tags cam1,cam2,...,cam16 \
  --fps 5 --duration 60

# 3. GoPro外参标定
cd multical
python calibrate.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path output/gopro_synced/original \
  --calibration ../intrinsic_hyperoff_linear_60fps.json \
  --fix_intrinsic --vis
```

### 阶段2: PrimeColor标定（待实现）

```bash
# 1. PrimeColor内参标定
cd multical
python intrinsic.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path <primecolor_calib>/original \
  --limit_images 300 --vis

# 2. PrimeColor视频同步（需要实现sync_by_flash.py）
python scripts/sync_by_flash.py \
  --src_tag primecolor \
  --out_tag output/primecolor_synced
```

### 阶段3: 联合标定（待实现）

```bash
# 1. 合并内参
# 手动合并或使用脚本

# 2. 联合外参标定
cd multical
python calibrate.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path output/joint_calib/original \
  --calibration combined_intrinsics.json \
  --fix_intrinsic --vis
```

### 阶段4: Mocap标定（待实现）

```bash
# 1. 提取Mocap标记点
python scripts/extract_mocap_markers.py \
  --tak_file mocap/calibration.tak \
  --output mocap_markers.json

# 2. Mocap-相机标定
python scripts/calibrate_mocap_camera.py \
  --mocap_markers mocap_markers.json \
  --camera_calib output/calibration/combined_extrinsics.json \
  --output mocap_to_camera.json
```

### 阶段5: 全局同步（待实现）

```bash
# 统一时间同步
python scripts/sync_all_systems.py \
  --gopro_meta output/gopro_synced/meta_info.json \
  --primecolor_meta output/primecolor_synced/meta_info.json \
  --mocap_file mocap/recording1.tak \
  --output sync_metadata.json
```

---

## 功能完成度检查表

### ✅ 已完成功能

| 功能 | 状态 | 文件/脚本 |
|------|------|-----------|
| GoPro内参标定 | ✅ 完成 | `intrinsic_hyperoff_linear_60fps.json` |
| GoPro timecode同步 | ✅ 完成 | `scripts/sync_timecode.py` |
| GoPro外参标定 | ✅ 完成 | `multical/calibrate.py` |
| 视频帧提取 | ✅ 完成 | `scripts/convert_video_to_images.py` |
| 2D姿态检测 | ✅ 完成 | `scripts/run_vitpose.py` |
| 3D三角化 | ✅ 完成 | `scripts/run_triangulation.py` |
| 3D姿态优化 | ✅ 完成 | `scripts/run_refinement.py` |

### ⚠️ 待实现功能

| 功能 | 优先级 | 说明 |
|------|--------|------|
| PrimeColor内参标定 | 🔴 高 | 需要拍摄ChArUco板 |
| PrimeColor视频同步 | 🔴 高 | 无timecode，需实现闪光/音频同步 |
| GoPro-PrimeColor联合外参 | 🔴 高 | 需要联合标定或分步配准 |
| Mocap数据提取 | 🟡 中 | 解析.tak/.c3d文件 |
| Mocap-相机标定 | 🟡 中 | PnP求解坐标系转换 |
| 全局时间同步 | 🟡 中 | 统一GoPro+PrimeColor+Mocap时间轴 |
| 标定结果合并工具 | 🟢 低 | 方便分步标定后合并 |
| 自动化标定流程 | 🟢 低 | 一键运行所有标定步骤 |

---

## 关键技术细节

### 坐标系定义

```
GoPro坐标系:
  - 原点: 第一个相机光心（或标定板中心）
  - X轴: 向右
  - Y轴: 向上
  - Z轴: 向前（远离相机）

Mocap坐标系:
  - 原点: 系统定义的世界原点
  - 坐标轴: 系统定义（通常Y向上）
  - 单位: 毫米或米

转换:
  point_camera = R @ point_mocap + T
```

### 时间对齐精度

| 系统 | 同步方法 | 理论精度 | 实际精度 |
|------|----------|----------|----------|
| GoPro (60fps) | 硬件timecode | 1/60秒 (16.7ms) | 1-2帧 (~33ms, fast_copy) |
| PrimeColor (30fps) | 闪光检测 | 1/30秒 (33.3ms) | 1-3帧 (~100ms) |
| Mocap (120Hz) | 运动标记 | 1/120秒 (8.3ms) | ~50-100ms（手动标记） |

### 标定质量指标

| 指标 | 优秀 | 良好 | 可接受 | 需重做 |
|------|------|------|--------|--------|
| 内参RMS (像素) | <0.5 | 0.5-1.0 | 1.0-2.0 | >2.0 |
| 外参RMS (像素) | <0.5 | 0.5-1.0 | 1.0-2.0 | >2.0 |
| 重投影误差 (像素) | <5 | 5-10 | 10-20 | >20 |
| Mocap-相机误差 (mm) | <10 | 10-20 | 20-50 | >50 |

---

## 故障排查

### 问题1: GoPro同步失败
```
错误: "Timecode not found"
原因: 视频未嵌入timecode
解决:
  1. 检查GoPro是否启用了timecode功能
  2. 使用专业版GoPro或外部timecode同步器
  3. 跳过同步，手动对齐视频
```

### 问题2: 标定RMS过大
```
错误: RMS > 2.0像素
原因:
  - 标定板检测不准确
  - 内参与实际相机设置不匹配
  - 标定板在某些帧中模糊
解决:
  1. 重新拍摄标定视频（确保静止、清晰）
  2. 检查相机设置是否与内参匹配
  3. 增加标定帧数量
  4. 过滤掉检测质量差的帧
```

### 问题3: PrimeColor无法同步
```
错误: 无法检测同步信号
解决:
  1. 确保闪光/拍板足够明显
  2. 手动标记同步帧号
  3. 使用音频峰值检测
```

### 问题4: Mocap-相机重投影误差大
```
错误: 重投影误差 > 50mm
原因:
  - 标记点ID匹配错误
  - 坐标系单位不一致
  - 标定数据时间不同步
解决:
  1. 检查标记点ID映射
  2. 统一单位（米或毫米）
  3. 确保标定时Mocap和相机同步采集
```

---

## 内参处理工具集

代码库提供了一系列工具用于处理相机内参JSON文件。这些工具位于项目根目录和 `tool_scripts/` 下。

### 1. filter_intrinsics.py - 过滤/子集化内参

**位置**: `/Volumes/FastACIS/annotation_pipeline/filter_intrinsics.py`

**功能**: 从完整的内参JSON中提取指定相机的子集。

**用途**:
- ✅ 你有16相机的内参，但只需要其中3个
- ✅ 自动匹配视频目录中的相机
- ✅ 同时过滤内参和外参（如果存在 `camera_base2cam`）

**命令示例**:
```bash
# 方法1: 手动指定相机
python filter_intrinsics.py \
  --input intrinsic_hyperoff_linear_60fps.json \
  --output intrinsic_3cams.json \
  --cameras cam1,cam2,cam4

# 方法2: 自动检测
python filter_intrinsics.py \
  --input intrinsic_hyperoff_linear_60fps.json \
  --output intrinsic_auto.json \
  --auto-detect /Volumes/FastACIS/gopro/ex/
```

**输出**:
- 新的JSON文件只包含指定的相机
- 保持原始数据结构（cameras, camera_base2cam等）
- 如果某相机不在原始JSON中，会显示警告

---

### 2. combine_intrinsic_json.py - 合并多个内参文件

**位置**: `tool_scripts/combine_intrinsic_json.py`

**功能**: 将多个单相机内参文件合并为一个多相机JSON。

**用途**:
- ✅ 你为每个相机单独标定了内参（如 `cam1/intrinsic.json`, `cam2/intrinsic.json`）
- ✅ 需要创建统一的多相机内参文件用于外参标定
- ✅ 自动计算并添加FOV信息
- ✅ 从log文件中提取RMS误差

**用法**:
```python
# 编辑脚本中的参数
dir_folder = '/path/to/camera/folders'
list_cams = ['cam1', 'cam2', 'cam3']
output_path = 'combined_intrinsic.json'

# 运行
python tool_scripts/combine_intrinsic_json.py
```

**目录结构要求**:
```
dir_folder/
├── cam1/
│   ├── intrinsic.json    # 单相机内参
│   └── log.txt           # 标定日志（包含RMS）
├── cam2/
│   ├── intrinsic.json
│   └── log.txt
└── cam3/
    ├── intrinsic.json
    └── log.txt
```

**输出示例**:
```json
{
  "cameras": {
    "cam1": {
      "model": "standard",
      "image_size": [3840, 2160],
      "K": [...],
      "dist": [...],
      "fov": {
        "horizontal": 93.2,
        "vertical": 61.5,
        "diagonal": 101.0
      },
      "rms": 0.41
    },
    "cam2": {...},
    "cam3": {...}
  }
}
```

---

### 3. intrinsics_to_fov.py - 计算FOV

**位置**: `tool_scripts/intrinsics_to_fov.py`

**功能**: 从内参K矩阵计算相机视场角（FOV）。

**用途**:
- ✅ 验证标定结果（与GoPro官方规格对比）
- ✅ 分析多个相机的FOV一致性
- ✅ 生成FOV统计报告

**命令**:
```bash
# 分析默认文件
python tool_scripts/intrinsics_to_fov.py

# 指定输入文件
python tool_scripts/intrinsics_to_fov.py \
  --input intrinsic_hyperoff_linear_60fps.json

# 保存分析结果
python tool_scripts/intrinsics_to_fov.py \
  --input intrinsic_hyperoff_linear_60fps.json \
  --output fov_analysis.json
```

**输出示例**:
```
Loading intrinsics from: intrinsic_hyperoff_linear_60fps.json
Found 16 cameras: ['cam1', 'cam10', 'cam11', ...]
================================================================================
Camera: cam1
  Image size: 3840 x 2160
  Focal lengths: fx = 1814.61, fy = 1812.65
  Principal point: cx = 1919.69, cy = 1079.33
  Field of View:
    Horizontal: 93.23°
    Vertical: 61.57°
    Diagonal: 101.06°
--------------------------------------------------
...

Summary Statistics:
  Horizontal FOV: 93.02° ± 0.35°
  Vertical FOV: 61.37° ± 0.35°
  Diagonal FOV: 100.81° ± 0.38°

  FOV Range - Horizontal: [92.29°, 93.61°]
  FOV Range - Vertical: [60.73°, 61.91°]
  FOV Range - Diagonal: [100.12°, 101.43°]
```

**用途场景**:
- 标定完成后验证FOV是否符合预期（GoPro Linear模式约93°×61°）
- 检查多相机FOV的一致性（标准差应很小）
- 发现异常相机（FOV偏差过大可能表示标定有问题）

---

### 4. fov_to_intrinsics.py - 从FOV生成内参

**位置**: `tool_scripts/fov_to_intrinsics.py`

**功能**: 从已知的FOV角度反向计算相机内参矩阵。

**用途**:
- ✅ 你知道相机的FOV规格但没有标定数据
- ✅ 快速生成近似内参用于测试
- ⚠️ 生成的内参**没有畸变系数**，仅用于无畸变或畸变校正后的图像

**命令**:
```bash
# 使用水平和垂直FOV
python tool_scripts/fov_to_intrinsics.py \
  --width 3840 --height 2160 \
  --fov-h 93.0 --fov-v 61.5

# 只用对角FOV（假设方形像素）
python tool_scripts/fov_to_intrinsics.py \
  --width 3840 --height 2160 \
  --fov-d 101.0

# 保存为JSON
python tool_scripts/fov_to_intrinsics.py \
  --width 3840 --height 2160 \
  --fov-h 93.0 --fov-v 61.5 \
  --output camera_intrinsics.json
```

**输出示例**:
```
Camera Intrinsic Parameters:
========================================
Image size: 3840 x 2160
Focal lengths: fx = 1820.45, fy = 1818.32
Principal point: cx = 1920.00, cy = 1080.00
Aspect ratio: 1.778

Field of View:
Horizontal: 93.00°
Vertical: 61.50°
Diagonal: 100.85°

Intrinsic Matrix (OpenCV format):
K = [[1820.450000, 0.000000, 1920.000000],
     [0.000000, 1818.320000, 1080.000000],
     [0.000000, 0.000000, 1.000000]]
```

**注意事项**:
- ⚠️ 生成的内参**不包含畸变校正**
- ⚠️ 主点默认为图像中心，可能与实际有偏差
- ✅ 适合快速原型测试或已畸变校正的图像
- ❌ 不推荐用于生产环境，应使用ChArUco标定

---

### 5. compare_calibrations.py - 对比两个标定文件

**位置**: `tool_scripts/compare_calibrations.py`

**功能**: 对比两个标定JSON文件的差异。

**用途**:
- ✅ 对比新旧标定结果
- ✅ 验证标定重复性
- ✅ 检查内参/外参的变化

---

## 典型工作流程示例

### 场景1: 你只有3个GoPro（不是16个）

```bash
# 步骤1: 从预存内参中提取你的相机
python filter_intrinsics.py \
  --input intrinsic_hyperoff_linear_60fps.json \
  --output intrinsic_3cams.json \
  --cameras cam1,cam2,cam4

# 步骤2: 验证FOV
python tool_scripts/intrinsics_to_fov.py \
  --input intrinsic_3cams.json

# 步骤3: 使用过滤后的内参进行外参标定
cd multical
python calibrate.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path /path/to/calib/images \
  --calibration ../intrinsic_3cams.json \
  --fix_intrinsic --vis
```

### 场景2: 混合相机（有些在预存内参中，有些不在）

```bash
# 假设：cam1, cam2在预存内参中；cam99是新相机

# 步骤1: 为新相机单独标定内参
cd multical
python intrinsic.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path cam99_images/ \
  --cameras cam99 \
  --vis

# 步骤2: 从预存内参中提取已有相机
cd ..
python filter_intrinsics.py \
  --input intrinsic_hyperoff_linear_60fps.json \
  --output intrinsic_cam12.json \
  --cameras cam1,cam2

# 步骤3: 手动合并JSON（或使用Python脚本）
# 将cam99的内参添加到intrinsic_cam12.json中

# 步骤4: 使用合并后的内参进行外参标定
cd multical
python calibrate.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path /path/to/calib/images \
  --calibration ../intrinsic_cam12_merged.json \
  --fix_intrinsic --vis
```

### 场景3: 为PrimeColor相机创建内参

```bash
# 步骤1: 标定PrimeColor内参
cd multical
python intrinsic.py \
  --boards ./asset/charuco_b3.yaml \
  --image_path primecolor_calib/original \
  --cameras primecolor1,primecolor2 \
  --vis

# 步骤2: 提取GoPro内参
cd ..
python filter_intrinsics.py \
  --input intrinsic_hyperoff_linear_60fps.json \
  --output intrinsic_gopro.json \
  --auto-detect /Volumes/FastACIS/gopro/ex/

# 步骤3: 使用combine_intrinsic_json.py合并（需修改脚本）
# 或手动合并JSON文件

# 步骤4: 验证合并结果
python tool_scripts/intrinsics_to_fov.py \
  --input combined_intrinsics.json
```

---

## 参考资料

### 内部文档
- [CLAUDE.md](CLAUDE.md) - Pipeline总体说明
- [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) - GoPro标定指南
- [README.md](README.md) - 项目README

### 外部资源
- [Multical文档](https://github.com/lambdaloop/multical) - ChArUco标定
- [GoPro Timecode](https://gopro.com/help/articles/question_answer/what-is-timecode) - GoPro时间码
- [OpenCV Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) - 相机标定
- [C3D文件格式](https://www.c3d.org/) - Mocap数据格式

---

## 联系与支持

如遇到问题：
1. 查看相关日志文件
2. 检查可视化结果
3. 参考故障排查章节
4. 查阅CLAUDE.md和README.md

---

**文档版本**: 1.1
**最后更新**: 2025-10-22
**适用Pipeline版本**: annotation_pipeline (current)

**更新日志**:
- v1.1 (2025-10-22): 添加内参处理工具集章节，包括filter_intrinsics.py、combine_intrinsic_json.py等工具的详细说明和使用场景
- v1.0 (2025-10-22): 初始版本，包含GoPro、PrimeColor、Mocap系统的完整标定与同步流程
