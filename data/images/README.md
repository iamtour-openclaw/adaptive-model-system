# 合成数据集

这是由 `generate_data.py` 自动生成的测试数据集。

## 结构

```
data/
├── train/  # 训练集
├── val/    # 验证集
└── test/   # 测试集
```

## 使用

```bash
# 训练
python src/train.py --data_dir data/train

# 推理
python src/infer.py --model_path models/best.pth --image data/test/img_0000.png
```
