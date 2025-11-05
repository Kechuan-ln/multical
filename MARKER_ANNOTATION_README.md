# Mocap Marker Annotation Tool

交互式3D标注工具，用于为mocap markers添加语义标签。

## 功能特点

- ✅ 3D交互式可视化（可旋转、缩放）
- ✅ 点击marker进行选择
- ✅ 输入自定义标签名称（如 "Laxisl1"）
- ✅ 自动保存到CSV文件
- ✅ 实时显示已标注/未标注的markers
- ✅ 已标注的markers显示标签文字
- ✅ 帧滑块浏览不同时刻

## 安装依赖

在 `multical` conda环境中安装：

```bash
conda activate multical
pip install -r requirements_annotation.txt
```

需要的库：
- pandas
- numpy
- plotly
- dash

## 使用方法

### 基本用法

```bash
python annotate_mocap_markers.py --start_frame 2 --num_frames 200
```

### 完整参数

```bash
python annotate_mocap_markers.py \
  --csv /path/to/mocap.csv \
  --start_frame 2 \
  --num_frames 200 \
  --labels marker_labels.csv \
  --port 8050
```

### 参数说明

- `--csv`: mocap CSV文件路径（默认：`/Volumes/FastACIS/csldata/csl/mocap.csv`）
- `--start_frame`: 起始帧（默认：2）
- `--num_frames`: 加载的帧数（默认：200）
- `--labels`: 标签保存的CSV文件路径（默认：`marker_labels.csv`）
- `--port`: Web服务器端口（默认：8050）

## 使用流程

1. **启动工具**
   ```bash
   python annotate_mocap_markers.py --start_frame 2 --num_frames 200
   ```

2. **打开浏览器**
   - 工具会显示URL（通常是 `http://localhost:8050`）
   - 在浏览器中打开该URL

3. **标注markers**
   - 在3D视图中点击一个marker点
   - 右侧会显示选中的marker信息
   - 在"Label Name"输入框输入标签（例如：`Laxisl1`）
   - 点击"Set Label"按钮
   - 标签自动保存到CSV

4. **浏览帧**
   - 使用底部滑块切换帧
   - 标签会在所有帧中保持一致

5. **查看进度**
   - 标题显示已标注数量
   - 右侧显示所有已保存的标签列表
   - 已标注的markers用彩色显示，未标注的是灰色

## 输出文件

标签保存在CSV文件中（默认 `marker_labels.csv`），格式如下：

```csv
original_name,marker_id,label
Unlabeled 1216,784:73CB7475AD7A11F0,Laxisl1
Unlabeled 1217,787:73CB7475AD7A11F0,Laxisl2
...
```

### 字段说明

- `original_name`: 原始marker名称（Motive导出的名称）
- `marker_id`: Marker的唯一ID
- `label`: 你设置的标签名称

## 坐标系统

- **Y轴**: 垂直向上（高度）
- **X轴**: 水平（地面）
- **Z轴**: 水平（地面）
- **XZ平面**: 地面/水平面
- **颜色**: 已标注markers按Y值（高度）着色

## 交互控制

### 3D视图
- **旋转**: 点击并拖动
- **缩放**: 滚轮滚动
- **平移**: 右键拖动（或Shift+左键拖动）
- **选择marker**: 单击marker点

### 帧导航
- **滑块**: 拖动查看不同帧
- **标记**: 显示帧号

## 提示

1. **建议帧数**: 加载100-200帧足够标注所有markers
2. **保存时机**: 每次点击"Set Label"时自动保存
3. **重复标注**: 可以重新点击已标注的marker修改标签
4. **标签命名**: 建议使用有意义的名称（如身体部位名称）

## 示例标签

常见的人体marker标签示例：

- `Head_Top` - 头顶
- `L_Shoulder` - 左肩
- `R_Shoulder` - 右肩
- `L_Elbow` - 左肘
- `R_Elbow` - 右肘
- `L_Wrist` - 左腕
- `R_Wrist` - 右腕
- `L_Hip` - 左髋
- `R_Hip` - 右髋
- `L_Knee` - 左膝
- `R_Knee` - 右膝
- `L_Ankle` - 左踝
- `R_Ankle` - 右踝

或使用自定义命名系统（如 `Laxisl1`, `Laxisl2` 等）。

## 故障排除

### 问题: "Module not found: dash"
**解决**: 安装依赖
```bash
pip install dash
```

### 问题: 浏览器无法打开
**解决**:
1. 检查终端输出的URL
2. 尝试不同端口：`--port 8051`
3. 检查防火墙设置

### 问题: 点击marker没有反应
**解决**:
1. 确保点击在marker点上（不是空白处）
2. 查看右侧"Selected Marker"区域是否更新
3. 尝试点击其他marker

### 问题: 标签没有保存
**解决**:
1. 检查是否有写入权限
2. 查看终端输出的保存信息
3. 检查CSV文件是否生成

## 停止服务器

按 `Ctrl+C` 停止Web服务器。标签已自动保存到CSV文件。
