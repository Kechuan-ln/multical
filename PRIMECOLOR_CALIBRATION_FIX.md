# PrimeColor暗图像标定改进方案

## 问题诊断总结

### 检测率对比（当前状态）
- **GoPro cam4**: 2335角点 / 100帧 = **23.4点/帧** ✅
- **PrimeColor**: 557角点 / 100帧 = **5.6点/帧** ❌
- **理论最大**: ChArUco B1板 = 48角点/帧
- **检测率**: PrimeColor仅达到cam4的**24%**

### 根本原因
1. **亮度严重不足**：PrimeColor拍摄环境很暗
2. **对比度过低**：ChArUco板与背景对比度不足
3. **默认检测参数**：针对正常光照设计，不适合暗环境

### 当前标定质量
- RMS误差：1.402像素（勉强可接受）
- 可用帧：100帧中约39帧cam4有效，仅13帧primecolor有效
- 需改进：提高primecolor检测率至cam4水平

---

## 改进方案（三级优化）

### 🚀 方案1：优化ArUco检测参数（推荐优先）

**原理**：调整OpenCV的ArUco检测参数，提高暗环境敏感度

**实施步骤**：

1. **使用优化配置文件**
   ```bash
   # 已创建: multical/asset/charuco_b1_2_dark.yaml
   # 包含针对暗图像优化的参数
   ```

2. **修改标定命令**
   ```bash
   # 原命令（run_gopro_primecolor_calibration.py第74行）
   BOARD_CONFIG = "./asset/charuco_b1_2.yaml"

   # 改为
   BOARD_CONFIG = "./asset/charuco_b1_2_dark.yaml"
   ```

3. **重新运行标定**
   ```bash
   conda activate multical
   python run_gopro_primecolor_calibration.py
   ```

**关键参数说明**：
```yaml
aruco_params:
  adaptiveThreshWinSizeMax: 35       # 增大窗口（原23）
  adaptiveThreshConstant: 10         # 提高敏感度（原7）
  minMarkerPerimeterRate: 0.01       # 降低最小尺寸限制
  errorCorrectionRate: 0.8           # 提高容错率（原0.6）
  cornerRefinementMethod: 2          # 启用亚像素精度
  perspectiveRemovePixelPerCell: 8   # 提高透视校正分辨率
```

**预期改进**：检测率提升 **30-50%**

---

### ⚡ 方案2：图像预处理增强

**原理**：在检测前提升图像亮度和对比度

**实施步骤**：

#### 2.1 快速测试增强效果

```bash
# 对比不同增强方法
python enhance_dark_images.py --compare \
  "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor/frame_000000.png" \
  --output comparison.png
```

#### 2.2 批量增强primecolor图像

```bash
# 方法1: CLAHE（推荐，最稳定）
python enhance_dark_images.py \
  --input "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor" \
  --output "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor_enhanced" \
  --method clahe

# 方法2: Hybrid（最强，但可能引入噪声）
python enhance_dark_images.py \
  --input "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor" \
  --output "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor_enhanced_hybrid" \
  --method hybrid
```

#### 2.3 使用增强图像重新标定

修改 `calibrate_gopro_primecolor_extrinsics.py`:

```python
# 第269行修改为增强后的目录
primecolor_charuco = extrinsics_dir / 'primecolor_charuco_enhanced.mp4'

# 或直接在extract_sync_frames中增强
def extract_sync_frames(...):
    # 在保存前增强primecolor图像
    if cam_name == 'primecolor':
        from enhance_dark_images import enhance_dark_image
        frame = enhance_dark_image(frame, method='clahe')

    cv2.imwrite(output_path, frame)
```

**预期改进**：检测率提升 **40-70%**

---

### 🔥 方案3：组合优化（最佳效果）

**同时使用优化参数 + 图像增强**

```bash
# 1. 增强图像
python enhance_dark_images.py \
  --input primecolor_frames/ \
  --output primecolor_enhanced/ \
  --method clahe

# 2. 修改配置使用charuco_b1_2_dark.yaml

# 3. 重新标定
python run_gopro_primecolor_calibration.py
```

**预期改进**：检测率提升 **60-90%**，RMS降至 **<1.0像素**

---

## 验证和测试

### 测试检测率改进

```bash
# 测试单张图像（对比所有方法）
python test_charuco_detection.py \
  --image "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor/frame_000000.png" \
  --enhance clahe gamma hybrid

# 批量测试（前10张图像）
python test_charuco_detection.py \
  --dir "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor" \
  --limit 10 \
  --enhance clahe \
  --output detection_results.json
```

**输出示例**：
```
方法                           成功率      平均角点    平均Marker
--------------------------------------------------------------------------------
original_original              30.0%        5.6        8.2
original_clahe                 65.0%       18.3       22.1
optimized_original             45.0%       12.1       15.8
optimized_clahe                85.0%       28.5       31.4
```

### 验证标定质量

```bash
# 查看标定结果
cat calibration.json | python -m json.tool | grep -A 5 "rms"

# 检查可视化
open "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/vis/frames/primecolor/"

# 期望：
# - RMS < 1.0像素
# - primecolor检测帧数 > 30帧（原13帧）
# - 平均角点数 > 20点/帧（原5.6点/帧）
```

---

## 推荐实施流程

### 快速验证（15分钟）

```bash
cd /Volumes/FastACIS/annotation_pipeline

# 1. 测试当前检测率
python test_charuco_detection.py \
  --dir "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor" \
  --limit 5

# 2. 测试增强效果
python test_charuco_detection.py \
  --dir "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor" \
  --limit 5 \
  --enhance clahe

# 对比结果，选择最佳方法
```

### 完整优化（1小时）

```bash
# 1. 修改配置文件
# 编辑 run_gopro_primecolor_calibration.py:
#   BOARD_CONFIG = "./asset/charuco_b1_2_dark.yaml"

# 2. 增强primecolor视频（可选）
# 修改 calibrate_gopro_primecolor_extrinsics.py
# 在extract_sync_frames中添加图像增强

# 3. 重新运行完整标定流程
python run_gopro_primecolor_calibration.py

# 4. 验证改进效果
python test_charuco_detection.py \
  --dir "/Volumes/FastACIS/GoPro/gopro_primecolor_extrinsic /calibration_output/extrinsics/frames/primecolor" \
  --output validation_results.json
```

---

## 参数调优指南

如果上述方案效果仍不理想，可以手动微调参数：

### ArUco检测参数

在 `charuco_b1_2_dark.yaml` 中调整：

```yaml
aruco_params:
  # 【如果检测到的marker太少】
  adaptiveThreshWinSizeMax: 40        # 继续增大（默认35）
  adaptiveThreshConstant: 12          # 继续提高（默认10）
  minMarkerPerimeterRate: 0.005       # 进一步降低（默认0.01）

  # 【如果检测到很多错误marker】
  errorCorrectionRate: 0.6            # 降低容错率（默认0.8）
  minMarkerPerimeterRate: 0.02        # 提高最小尺寸

  # 【如果角点位置不准确】
  cornerRefinementWinSize: 7          # 增大窗口（默认5）
  cornerRefinementMaxIterations: 100  # 增加迭代（默认50）
```

### 图像增强参数

在 `enhance_dark_images.py` 中调整：

```python
# CLAHE方法（第30行）
clahe = cv2.createCLAHE(
    clipLimit=4.0,        # 增大可提升对比度（默认3.0）
    tileGridSize=(8, 8)   # 减小可处理更局部的对比度
)

# Gamma校正（第56行）
gamma = 1.8              # 增大可更亮（默认1.5）
```

---

## 长期改进建议

1. **改善拍摄环境**（根本解决）
   - 增加照明：使用补光灯
   - 调整相机设置：
     - 提高ISO（但注意噪声）
     - 降低快门速度（但注意运动模糊）
     - 开大光圈

2. **更换标定板**
   - 使用更大的标定板（当前B1，考虑定制）
   - 提高marker尺寸比例
   - 使用高对比度打印（激光打印，磨砂表面）

3. **多次拍摄合并**
   - 在不同光照条件下拍摄多组
   - 合并多次标定结果

4. **相机硬件检查**
   - 检查PrimeColor镜头是否有污渍
   - 确认传感器工作正常
   - 对比GoPro确认是否为相机问题

---

## 故障排除

### Q: 运行test_charuco_detection.py报错

```bash
# 确保安装依赖
pip install opencv-python opencv-contrib-python pyyaml tqdm

# 如果提示找不到multical
export PYTHONPATH="/Volumes/FastACIS/annotation_pipeline/multical:$PYTHONPATH"
```

### Q: 增强后图像检测率反而下降

可能过度增强导致噪声增加。尝试：
- 使用更保守的参数（降低clipLimit）
- 切换到`gamma`方法
- 使用`hybrid`但禁用锐化步骤

### Q: 标定RMS仍然 > 1.5像素

可能原因：
- primecolor和gopro同步误差
- 内参不准确
- 标定板打印质量问题
- 相机运动模糊

解决：
- 重新检查QR同步offset
- 重新标定primecolor内参
- 使用更多帧（增加EXTRINSIC_MAX_FRAMES）

---

## 预期结果

### 优化前（当前）
- PrimeColor检测率：5.6点/帧
- 有效帧数：13/100帧
- RMS误差：1.402像素

### 优化后（目标）
- PrimeColor检测率：**>20点/帧**（提升3.6倍）
- 有效帧数：**>50/100帧**（提升3.8倍）
- RMS误差：**<1.0像素**（降低30%）

---

## 文件清单

已创建的文件：
1. `multical/asset/charuco_b1_2_dark.yaml` - 优化的检测配置
2. `enhance_dark_images.py` - 图像增强工具
3. `test_charuco_detection.py` - 检测测试工具
4. `PRIMECOLOR_CALIBRATION_FIX.md` - 本文档

---

## 联系和反馈

如果遇到问题或需要进一步优化，请检查：
1. 图像原始亮度分布（`cv2.calcHist`）
2. ArUco marker检测日志
3. 标定板到相机的距离和角度
4. 是否有运动模糊或失焦

记录优化前后的具体数值以便对比改进效果。
