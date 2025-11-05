# Markers to Skeleton Conversion Guide

将标注好的mocap markers转换为17个关节点的skeleton（H36M格式）。

## 工作流程

### 步骤1: 标注Markers

首先使用交互式标注工具标注所有markers：

```bash
python annotate_mocap_markers.py --start_frame 2 --num_frames 200
```

在浏览器中标注所有需要的markers，标签会保存到 `marker_labels.csv`。

### 步骤2: 检查必需的Markers

转换需要以下markers（根据 `skeleton_config.json` 定义）：

**躯干 (Torso):**
- `Lpelvisf`, `Lpelvisb` - 左骨盆前后
- `Rpelvisf`, `Rpelvisb` - 右骨盆前后
- `Spine_Marker1`, `Spine_Marker2` - 脊柱标记
- `Neck_Marker1`, `Neck_Marker2` - 颈部标记

**头部 (Head):**
- `HeadF_L`, `HeadF_R`, `HeadB_L`, `HeadB_R` - 头部4个标记
- `FaceL`, `FaceR` - 脸部左右标记

**左臂 (Left Arm):**
- `LShoulder1`, `LShoulder2`, `LShoulder3` - 左肩3个标记
- `LElbow1`, `LElbow2`, `LElbow3` - 左肘3个标记
- `Lwristl1`, `Lwristl2` - 左腕2个标记

**右臂 (Right Arm):**
- `RShoulder1`, `RShoulder2`, `RShoulder3` - 右肩3个标记
- `RElbow1`, `RElbow2`, `RElbow3` - 右肘3个标记
- `Rwristr1`, `Rwristr2` - 右腕2个标记

**左腿 (Left Leg):**
- `Lkneel1`, `Lkneel2` - 左膝2个标记
- `Lanklel1`, `Lanklel2` - 左踝2个标记

**右腿 (Right Leg):**
- `Rkneer1`, `Rkneer2` - 右膝2个标记
- `Rankler1`, `Rankler2` - 右踝2个标记

### 步骤3: 转换为Skeleton

运行转换脚本：

```bash
python markers_to_skeleton.py \
  --mocap_csv /Volumes/FastACIS/csldata/csl/mocap.csv \
  --labels_csv marker_labels.csv \
  --config skeleton_config.json \
  --output_csv skeleton_joints.csv \
  --output_json skeleton_joints.json \
  --start_frame 2 \
  --end_frame 23374
```

#### 参数说明

- `--mocap_csv`: 原始mocap CSV文件路径
- `--labels_csv`: 标注的marker labels文件（由标注工具生成）
- `--config`: Skeleton配置文件（定义17个关节的计算公式）
- `--output_csv`: 输出的skeleton CSV文件
- `--output_json`: 输出的skeleton JSON文件
- `--start_frame`: 起始帧（默认2）
- `--end_frame`: 结束帧（默认处理所有帧）

## 17个关节定义（H36M格式）

按照配置文件定义，17个关节及其计算方法：

| ID | 关节名 | 计算公式 |
|----|--------|----------|
| 0 | Pelvis | (LHip + RHip) / 2 |
| 1 | LHip | (Lpelvisf + Lpelvisb) / 2 |
| 2 | RHip | (Rpelvisf + Rpelvisb) / 2 |
| 3 | Spine1 | (Spine_Marker1 + Spine_Marker2) / 2 |
| 4 | Neck | (Neck_Marker1 + Neck_Marker2) / 2 |
| 5 | Head | (HeadF_L + HeadF_R + HeadB_L + HeadB_R) / 4 |
| 6 | Jaw | (FaceL + FaceR) / 2 |
| 7 | LShoulder | (LShoulder1 + LShoulder2 + LShoulder3) / 3 |
| 8 | LElbow | (LElbow1 + LElbow2 + LElbow3) / 3 |
| 9 | LWrist | (Lwristl1 + Lwristl2) / 2 |
| 10 | RShoulder | (RShoulder1 + RShoulder2 + RShoulder3) / 3 |
| 11 | RElbow | (RElbow1 + RElbow2 + RElbow3) / 3 |
| 12 | RWrist | (Rwristr1 + Rwristr2) / 2 |
| 13 | LKnee | (Lkneel1 + Lkneel2) / 2 |
| 14 | LAnkle | (Lanklel1 + Lanklel2) / 2 |
| 15 | RKnee | (Rkneer1 + Rkneer2) / 2 |
| 16 | RAnkle | (Rankler1 + Rankler2) / 2 |

## Skeleton拓扑结构

父节点关系（用于骨架可视化）：

```
Pelvis (root)
├── LHip
│   └── LKnee
│       └── LAnkle
├── RHip
│   └── RKnee
│       └── RAnkle
└── Spine1
    └── Neck
        ├── Head
        │   └── Jaw
        ├── LShoulder
        │   └── LElbow
        │       └── LWrist
        └── RShoulder
            └── RElbow
                └── RWrist
```

Parents数组: `[-1, 0, 0, 0, 3, 4, 5, 4, 7, 8, 4, 10, 11, 1, 13, 2, 15]`

## 输出格式

### CSV格式 (skeleton_joints.csv)

```csv
Frame,Time,Pelvis_X,Pelvis_Y,Pelvis_Z,LHip_X,LHip_Y,LHip_Z,...
2,0.016667,-100.5,500.2,-1800.3,...
3,0.025000,-100.3,501.1,-1799.8,...
```

每一行包含：
- `Frame`: 帧号
- `Time`: 时间（秒）
- 17个关节，每个关节3列（X, Y, Z）

总共：2 + 17×3 = 53列

### JSON格式 (skeleton_joints.json)

```json
{
  "metadata": {
    "fps": 120.0,
    "start_frame": 2,
    "num_frames": 23373,
    "num_joints": 17,
    "joint_names": ["Pelvis", "LHip", ...],
    "coordinate_system": "Y-up (vertical), XZ-horizontal"
  },
  "frames": {
    "0": {
      "frame_num": 2,
      "time": 0.016667,
      "joints": {
        "Pelvis": [x, y, z],
        "LHip": [x, y, z],
        ...
      }
    },
    ...
  }
}
```

## 坐标系统

- **Y轴**: 向上（垂直/高度）
- **X轴**: 水平（地面）
- **Z轴**: 水平（地面）
- **单位**: 毫米 (mm)
- **原点**: Motive全局坐标系的原点

## 验证输出

转换完成后，脚本会显示：

```
✓ All required markers are labeled!

Computing skeleton joints...
  Computing Pelvis...
    ✓ Pelvis computed successfully
  Computing LHip...
    ✓ LHip computed successfully
  ...

Skeleton joints saved to: skeleton_joints.csv
  Frames: 23373
  Joints: 17/17

Skeleton joints saved to: skeleton_joints.json
```

如果某些markers缺失，会显示警告：

```
Warning: Joint 'LShoulder' missing markers: ['LShoulder3']
  ✗ LShoulder failed (missing markers)
```

## 自定义配置

如果你的marker命名不同，可以修改 `skeleton_config.json` 文件：

```json
{
  "joint_name": "LWrist",
  "joint_id": 9,
  "formula": "mean",
  "markers": ["Lwristl1", "Lwristl2"],  // 修改这里的marker名称
  "description": "Left wrist"
}
```

## 常见问题

### Q: 如何知道哪些markers缺失？

A: 运行转换脚本，它会自动检查并列出所有缺失的markers。

### Q: 某些帧的关节位置是NaN？

A: 这是正常的，当某些markers在该帧缺失时，计算出的关节位置会是NaN。你可以：
- 检查原始mocap数据中markers的可见性
- 使用插值填充缺失的关节位置

### Q: 如何可视化skeleton？

A: 可以使用输出的JSON文件，结合skeleton topology信息，绘制骨架线条。

### Q: 输出的skeleton能否用于其他格式（如SMPL）？

A: 这17个关节是H36M格式。如果要转换为其他格式（如SMPL的24个关节），需要额外的映射和计算。

## 下一步

完成skeleton转换后，你可以：

1. **3D可视化**: 创建skeleton的3D动画可视化
2. **动作分析**: 计算关节角度、速度、加速度
3. **姿态估计验证**: 与2D姿态估计结果对比
4. **数据集创建**: 将skeleton数据转换为训练数据格式
5. **运动学分析**: 计算步态参数、关节活动度等

## 文件清单

转换工具包含以下文件：

- `markers_to_skeleton.py` - 主转换脚本
- `skeleton_config.json` - Skeleton配置文件（17关节定义）
- `marker_labels.csv` - Marker标注结果（由标注工具生成）
- `skeleton_joints.csv` - 输出：Skeleton CSV格式
- `skeleton_joints.json` - 输出：Skeleton JSON格式
- `SKELETON_CONVERSION_README.md` - 本文档
