# Phase 1 实验配置

## 实验目标
训练一个轻量级图片量测模型，能够在 CPU 上实时推理。

## 数据集
- 类型：合成几何图形 (rectangle, circle, triangle)
- 训练集：100 张
- 验证集：20 张
- 测试集：10 张

## 模型配置
- 架构：LightMeasureNet
- 参数量：~500K
- 输入尺寸：224x224

## 训练配置
- Epochs: 50
- Batch Size: 16
- Learning Rate: 0.001
- Optimizer: Adam
- LR Scheduler: ReduceLROnPlateau

## 预期性能
- 训练准确率：>90%
- 验证准确率：>85%
- 推理时间：<50ms (CPU)

## 运行命令

```bash
# 1. 生成测试数据
python src/generate_data.py --output_dir data/images

# 2. 训练模型
python src/train.py \
    --data_dir data/images/train \
    --model light \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001

# 3. 推理测试
python src/infer.py \
    --model_path models/best_acc.pth \
    --image data/images/test/img_0000.png \
    --show
```

## 下一步
- [ ] 验证训练流程
- [ ] 测试推理性能
- [ ] 记录实验结果
- [ ] 准备 Phase 2 (模型选择器)
