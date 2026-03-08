# 快速开始指南

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 生成测试数据

```bash
python src/generate_data.py --output_dir data/images
```

这会生成合成几何图形数据集用于测试。

## 3. 训练模型

```bash
python src/train.py --data_dir data/images/train --epochs 50
```

训练完成后，模型会保存在 `models/` 目录。

## 4. 推理测试

```bash
python src/infer.py --model_path models/best_acc.pth --image data/images/test/img_0000.png --show
```

## 5. 批量推理

```bash
python src/infer.py \
    --model_path models/best_acc.pth \
    --image_dir data/images/test \
    --output_dir results
```

## 常见问题

### Q: 训练很慢怎么办？
A: Phase 1 优先保证 CPU 兼容性。如需加速，可：
- 减少 `--epochs` 数量
- 减小 `--batch_size`
- 使用 `--model tiny` (更小的模型)

### Q: 如何用自己的数据训练？
A: 按以下结构组织数据：
```
data/
└── your_data/
    ├── class1/
    │   ├── img1.jpg
    │   └── ...
    └── class2/
        └── ...
```
然后运行：
```bash
python src/train.py --data_dir data/your_data
```

### Q: 推理速度不够快？
A: 尝试：
- 使用 `--model tiny` 架构
- 减小 `--input_size` (如 128x128)
- 导出为 ONNX 格式 (Phase 3)

## 项目结构

```
adaptive-model-system/
├── README.md
├── requirements.txt
├── src/
│   ├── models/        # 模型定义
│   ├── utils/         # 工具函数
│   ├── train.py       # 训练脚本
│   ├── infer.py       # 推理脚本
│   └── generate_data.py  # 数据生成
├── data/              # 数据集
├── models/            # 预训练模型
└── experiments/       # 实验配置
```

---

🚀 开始构建你的自适应智能模型系统！
