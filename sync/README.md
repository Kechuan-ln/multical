# 多相机同步系统使用指南

**系统**: 16× GoPro + 1× PrimeColor + Mocap
**同步方法**: QR Code Anchor Video
**更新时间**: 2025-11-03

---

## 📖 目录

1. [快速开始](#快速开始)
2. [系统概述](#系统概述)
3. [拍摄流程](#拍摄流程)
4. [后期处理](#后期处理)
5. [验证方法](#验证方法)
6. [常见问题](#常见问题)

---

## 快速开始

### 准备工作

1. **生成QR Anchor视频**（仅需一次）:
   ```bash
   python generate_qr_sync_video.py \
     --output qr_anchor.mp4 \
     --duration 300 \
     --fps 30
   ```

2. **拍摄时**:
   - 在iPad上循环播放QR anchor视频
   - 录制开始：让所有相机看到QR码 1-2分钟
   - 正常拍摄你的内容
   - 录制结束：再次让所有相机看到QR码 1-2分钟

3. **后期处理**:
   ```bash
   # Step 1: GoPro同步（已有工具）
   python scripts/sync_timecode.py \
     --src_tag gopro_raw \
     --out_tag gopro_synced

   # Step 2: 多相机同步（待实现）
   python sync/sync_multi_camera_with_qr.py \
     --gopro-dir data/gopro_synced \
     --primecolor-video data/primecolor.avi \
     --anchor-video qr_anchor.mp4 \
     --mocap-csv data/mocap.csv \
     --output-dir data/output
   ```

**输出**:
- `primecolor_synced.mp4` - 同步后的PrimeColor视频
- `mocap_synced.csv` - 同步后的Mocap数据
- `sync_info.json` - 同步元数据和质量报告

---

## 系统概述

### 硬件配置

| 设备 | 数量 | FPS | 分辨率 | 编码 |
|------|------|-----|--------|------|
| **GoPro** | 16 | 60 | 3840×2160 | H.264/MP4 |
| **PrimeColor** | 1 | 120 | 1920×1080 | MJPEG/AVI |
| **Mocap** | - | 120 | N/A | CSV (~200 markers) |
| **iPad** | 1 | - | - | QR视频播放器 |

### 关键发现 🎉

**完美的FPS关系**:
```
PrimeColor : GoPro : Mocap
   120    :   60   :  120
    2     :    1   :   2
```

**优势**:
- ✅ 整数倍关系 - 无需插值
- ✅ PrimeColor与Mocap FPS一致 - 同步简单
- ✅ 高时间分辨率（8.33ms @ 120fps）

### 同步策略

1. **GoPro间**: 官方QR码同步（已有）
2. **PrimeColor → GoPro**: 自定义iPad QR码同步（本系统）
3. **Mocap → PrimeColor**: 帧偏移同步（FPS相同）

---

## 拍摄流程

### 准备阶段

```bash
# 1. 生成QR anchor视频（5分钟，30fps）
python generate_qr_sync_video.py \
  --output qr_anchor.mp4 \
  --duration 300 \
  --fps 30
```

**说明**: 每帧包含唯一的QR码（如"000001", "000002"...），用于建立时间映射。

### 拍摄时间轴

```
┌─────────────────────────────────────────────────────┐
│ 录制时间轴                                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [0-120秒]                                          │
│  ▼ QR码同步段 (用于计算同步参数)                     │
│    - iPad全屏播放QR anchor视频                      │
│    - 确保所有相机（16 GoPro + 1 PrimeColor）清晰拍摄 │
│    - 光照充足，QR码清晰可见                         │
│                                                     │
│  [120秒 - 结束前120秒]                              │
│  ▼ 主要拍摄内容                                     │
│    - 正常拍摄你需要的内容                           │
│    - 无需显示QR码                                   │
│                                                     │
│  [结束前120秒 - 结束]                               │
│  ▼ QR码验证段 (用于验证同步质量)                     │
│    - iPad继续播放QR anchor视频                      │
│    - 所有相机再次拍摄QR码                           │
│    - 用于自动检测时间漂移                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 拍摄步骤

**Step 1: 开始录制**
1. 打开iPad，全屏播放 `qr_anchor.mp4`（循环模式）
2. 启动所有16个GoPro相机（已用官方QR同步）
3. 启动PrimeColor相机
4. 让所有相机同时看到iPad上的QR码，持续1-2分钟

**Step 2: 正常拍摄**
- 移开iPad，进行正常拍摄
- 可以是任意时长

**Step 3: 结束录制**
1. 再次把iPad放到所有相机视野中
2. 让所有相机看到QR码，持续1-2分钟
3. 停止所有相机录制

**注意事项**:
- ✅ QR码越清晰，同步精度越高
- ✅ 确保光照充足（避免QR码模糊）
- ✅ 建议QR码显示时间≥2分钟（更多数据点）
- ✅ iPad循环播放5分钟视频（足够覆盖开始和结束）

---

## 后期处理

### 完整工作流

```bash
# 工作目录
cd /Volumes/FastACIS/annotation_pipeline

# Step 1: GoPro官方同步（使用现有工具）
python scripts/sync_timecode.py \
  --src_tag gopro_raw \
  --out_tag gopro_synced \
  --stacked

# 输出: gopro_synced/ + meta_info.json
```

```bash
# Step 2: 多相机QR同步（主要脚本，待实现）
python sync/sync_multi_camera_with_qr.py \
  --gopro-dir data/gopro_synced \
  --primecolor-video data/primecolor_raw/sync.avi \
  --anchor-video data/qr_anchor.mp4 \
  --mocap-csv data/mocap.csv \
  --output-dir data/output

# 输出:
#   - primecolor_synced.mp4   (同步后视频)
#   - mocap_synced.csv        (同步后CSV)
#   - sync_info.json          (同步元数据)
```

### 输出说明

**sync_info.json** 示例:
```json
{
  "reference_camera": "gopro_cam01",
  "sync_method": "qr_anchor_video",
  "primecolor": {
    "fps": 120,
    "offset_frames": 145,
    "offset_seconds": 1.208,
    "fps_ratio": 2.003,
    "qr_residuals": {
      "max_error_frames": 1.2,
      "rmse_frames": 0.6,
      "num_qr_matched": 45
    }
  },
  "verification": {
    "sync_quality": "excellent",
    "drift_frames": 1,
    "drift_ms_gopro": 16.7,
    "drift_ms_prime": 8.3,
    "is_linear": true
  }
}
```

---

## 验证方法

### ⭐ 推荐：双端QR自动验证

这是最强大的验证方法，**完全自动化，无需人工观看视频**。

**工作原理**:
1. 使用开始段QR码计算同步参数 → 应用到整个视频
2. 使用结束段QR码重新计算参数 → 对比验证
3. 如果开始和结束的offset一致 → 同步成功，无时间漂移
4. 如果有差异 → 检测到时间漂移（相机FPS不稳定）

**质量标准**:

| 时间漂移 | 质量评级 | 说明 |
|---------|----------|------|
| ≤ 1帧 (< 17ms @ 60fps) | 优秀 ✓✓✓ | 完美同步 |
| ≤ 2帧 (< 33ms @ 60fps) | 良好 ✓✓ | 可接受 |
| > 2帧 (> 33ms) | 较差 ✗ | 需检查相机 |

**验证结果示例**:
```
✓✓✓ 同步质量: 优秀
时间漂移: 1 帧
  @ GoPro (60fps): 16.7 ms
  @ PrimeColor (120fps): 8.3 ms
✓ FPS稳定，比率漂移: 0.0023
```

### Level 2: 单端QR验证

如果只有开始段QR码（没有结束段）：

**检查指标**:
- QR匹配数量 ≥ 10
- 最大残差 < 2帧
- RMSE < 1帧
- FPS比率偏差 < 0.01

**局限**: 只验证开始时刻，不检测时间漂移。

### Level 3: 视觉验证（可选）

生成stacked对比视频：
```bash
ffmpeg -i data/gopro_synced/cam01/video.MP4 \
       -i data/output/primecolor_synced.mp4 \
       -filter_complex "[0:v]scale=960:540[v0];[1:v]scale=960:540[v1];[v0][v1]hstack" \
       -c:v libx264 -crf 20 verify_sync.mp4

# 播放验证
open verify_sync.mp4
```

**检查项**:
- [ ] 人体运动是否同步
- [ ] 无明显时间漂移

---

## 常见问题

### Q1: 为什么使用QR码而不是timecode？

**A**: PrimeColor相机不支持嵌入timecode，而QR码方案：
- ✅ 不依赖硬件功能
- ✅ 可视化验证（直接看QR码是否对齐）
- ✅ 支持不同FPS的相机（60fps vs 120fps）
- ✅ 可以独立验证GoPro官方同步的准确性

---

### Q2: 相机录制时间不同怎么办？

**A**: 完全没问题！QR同步会自动处理：
- GoPro可以先开始或晚开始
- PrimeColor可以提前停止或晚停止
- 只要有**重叠时段**包含QR码即可（建议1-2分钟）

---

### Q3: 需要所有相机看到相同的QR码吗？

**A**: 不需要！这是QR anchor方法的优势：
- 相机可以在不同时刻开始录制
- 看到不同的QR码（如Camera1看到#100，Camera2看到#150）
- 通过anchor视频的时间映射计算相对偏移
- 原理: 知道QR#100对应时间T1，QR#150对应T2，就能算出偏移

---

### Q4: QR码检测失败怎么办？

**可能原因和解决方法**:

| 问题 | 解决方法 |
|------|---------|
| 光照不足 | 增加灯光，避免反光 |
| QR码模糊 | 提高iPad亮度，调整距离 |
| 角度太斜 | 确保相机正对iPad |
| 采样不足 | 增加QR显示时间到2-3分钟 |

**调整检测参数**:
```python
# 在detect_qr_all_cameras.py中调整
QR_DETECTION_CONFIG = {
    'step_frames': 2,        # 减小到1（更密集采样）
    'scan_duration_sec': 120 # 增加到180（扫描更长时间）
}
```

---

### Q5: 同步误差多大算正常？

**A**: 参考标准：

| FPS | 1帧时间 | 2帧时间 | 目标 |
|-----|---------|---------|------|
| **60fps** | 16.7 ms | 33.3 ms | < 2帧 |
| **120fps** | 8.3 ms | 16.7 ms | < 2帧 |

- **优秀**: ≤ 1帧（基本完美）
- **良好**: ≤ 2帧（可接受，人眼难以察觉）
- **需改进**: > 2帧（检查相机稳定性）

---

### Q6: Mocap CSV如何同步？

**A**: 由于Mocap FPS (120) = PrimeColor FPS (120)，非常简单：
```python
# 应用相同的帧偏移
mocap_frame_synced = mocap_frame_original + offset_frames
```

**注意**: 如果Mocap和PrimeColor不是同时开始录制，需要额外处理。

---

### Q7: 能否不降采样PrimeColor到60fps？

**A**: 推荐**保持120fps**！
- ✅ 2:1整数倍关系，处理简单
- ✅ 保留高时间分辨率（8.33ms）
- ✅ 后期根据需要选择使用60fps或120fps
- ⚠️ 文件更大（但MJPEG→H.264转换后会减小）

---

### Q8: 如何验证GoPro官方同步的准确性？

**A**: 使用我们的QR码数据！
```python
# 检测所有16个GoPro视频中的QR码
gopro_qr_data = detect_qr_all_gopro(gopro_dir, anchor_video)

# 检查所有GoPro是否在相同帧号看到相同QR码
# 如果有偏差 → GoPro官方同步有问题
```

---

### Q9: 时间漂移是什么？为什么重要？

**A**: 时间漂移 = 开始段offset - 结束段offset

**原因**:
- 相机声称60fps，实际可能是59.97fps或60.03fps
- 长时间录制会累积误差

**检测**: 双端QR验证可以自动检测
- 漂移 ≤ 1帧 → 相机FPS非常稳定 ✓
- 漂移 > 2帧 → 相机FPS不稳定，考虑换相机 ✗

---

### Q10: 我能用这个系统同步更多相机吗？

**A**: 可以！系统设计支持任意数量：
- 代码会自动检测gopro_dir中的所有cam*文件夹
- 可以添加多个PrimeColor相机（修改参数即可）
- 唯一限制: 所有相机需要看到同一个QR anchor视频

---

## 容错机制 ⭐

系统采用**渐进式降级（Graceful Degradation）**策略，即使部分数据有问题也能完成同步。

### 自动处理的问题

**1. GoPro官方同步失败**
- **检测**: 通过QR码检查相机间偏移
- **处理**: 提供选项：用QR码重新同步 或 继续使用现有数据
- **报告**: 在`sync_info.json`中记录问题

**2. 部分相机QR码检测失败**
- **检测**: 某些相机QR码不足（< 10个）或完全缺失
- **处理**: 只用有效相机计算同步参数，继续处理
- **报告**: 列出问题相机，记录使用了哪些相机
- **不中断**: 只要有足够相机就继续

**3. 结尾QR码缺失**
- **检测**: 检查视频结尾是否有QR码
- **处理**: 自动降级为单端验证（只用开始段）
- **报告**: 说明无法检测时间漂移
- **不中断**: 仅警告，继续完成同步

### 验证报告示例

**完整同步**（一切正常）:
```
✓ 良好相机 (4个): cam01, cam02, cam03, cam04
✓ GoPro官方同步质量良好
✓ 双端QR验证可用
✓✓✓ 同步质量: 优秀
时间漂移: 1 帧
```

**残缺同步**（部分相机有问题）:
```
✓ 良好相机 (2个): cam01, cam03
⚠️  QR码不足 (1个): cam02: 7 QR码 (< 10)
❌ 无QR码检测 (1个): cam04: 0 QR码

⚠️  警告: 部分相机QR码检测不足
将使用 2 个相机进行同步

⚠️  结束段QR码: 不足
将使用单端验证（仅开始段）

✓ 同步完成（带警告）
```

### 详细技术文档

完整的错误处理逻辑和代码实现见：
- [TECHNICAL_NOTES.md - 错误处理和容错机制](TECHNICAL_NOTES.md#错误处理和容错机制)

---

## 下一步

### 当前状态
- [x] 系统设计完成
- [x] 数据分析完成
- [x] 文档完成
- [ ] **代码实现**（进行中）
- [ ] 测试验证

### 需要实现的脚本

详见 [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md) 的实现细节。

---

## 技术支持

**遇到问题？**
1. 查看 `sync_info.json` 中的 `qr_residuals` 和 `verification`
2. 阅读 [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md) 了解技术细节
3. 调整QR检测参数（光照、角度、采样密度）

**文件路径**:
- 本文档: `sync/README.md`
- 技术参考: `sync/TECHNICAL_NOTES.md`
- 数据分析: `sync/analyze_sync_data.py`
- QR生成: `generate_qr_sync_video.py`
- QR同步: `sync_with_qr_anchor.py`（两相机版本，参考）

---

**版本**: 2.0
**最后更新**: 2025-11-03
**适用系统**: 16× GoPro + 1× PrimeColor + Mocap CSV
