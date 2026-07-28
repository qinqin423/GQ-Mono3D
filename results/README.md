# GQ-Mono3D 实验结果

## 数据集

- **数据集**: SUN RGB-D
- **测试集**: SUNRGBD_test
- **类别数**: 10类

## 评测指标

- **3D mAP@0.25**: 3D检测平均精度（IoU阈值0.25）
- **3D mAP@0.50**: 3D检测平均精度（IoU阈值0.50）
- **FPS**: 推理速度（每秒帧数）

## 评测器

- **Omni3DEvaluator**: 使用Omni3D评测协议进行正式离线评测

## 实验结果

### Baseline (3D-MOOD)

| 指标 | 数值 |
|------|------|
| 3D mAP@0.25 | 0.5299 |
| 3D mAP@0.50 | 0.2350 |
| FPS | 4.50 |

### GQ-Mono3D (最佳mAP@0.50)

| 指标 | 数值 |
|------|------|
| 3D mAP@0.50 | 0.2676 |
| 3D mAP@0.25 | 0.5201 |
| FPS | 4.49 |

## 结果说明

1. **最佳结果对应checkpoint**: `best_map50.ckpt`
2. **结果一致性**: 0.5201 (mAP@0.25) 和 0.2676 (mAP@0.50) 来自同一个checkpoint
3. **数据可用性**:
   - Checkpoint文件因体积原因未包含在仓库中
   - SUN RGB-D数据集需从官方渠道获取，不包含在仓库中
4. **评测方式**: 所有结果由正式离线评测获得，使用Omni3DEvaluator

## 结果汇总

详细数值见 [summary.csv](evaluation/summary.csv)