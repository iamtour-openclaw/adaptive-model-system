# Adaptive Model System (AMS) 🧠

一个自适应智能模型系统 - 根据需求选择合适模型，并能自动进化优化，实现本地训推一体。

## 🎯 项目愿景

构建一个像人类一样学习和进化的 AI 系统：
- **自适应模型选择** - 根据任务复杂度自动选择小模型或基模型
- **自动进化** - 基于反馈持续优化模型性能
- **本地训推** - 训练和推理都在本机 CPU 上完成
- **渐进学习** - 从简单任务开始，逐步积累能力

## 📋 开发路线图

```
Phase 1: 图片量测模型 (当前阶段)
  └─ 基础训练 + 推理 pipeline
  └─ CPU 优化的轻量级 CNN

Phase 2: 模型选择器
  └─ 根据任务难度路由到不同模型
  └─ 模型性能评估系统

Phase 3: 自动进化系统
  └─ 基于性能反馈自动 retrain/微调
  └─ 超参数自动优化

Phase 4: 完整 AMS 系统
  └─ 统一接口 + 持续学习
  └─ 模型仓库管理
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch (CPU 版本)
- OpenCV
- NumPy

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
python src/train.py --data data/images --epochs 50
```

### 推理测试
```bash
python src/infer.py --model models/best.pth --image test.jpg
```

## 📁 项目结构

```
adaptive-model-system/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── models/          # 模型定义
│   ├── train.py         # 训练脚本
│   ├── infer.py         # 推理脚本
│   └── utils/           # 工具函数
├── data/                # 数据集
├── models/              # 预训练模型
└── experiments/         # 实验记录
```

## 🛠️ 技术栈

- **深度学习框架**: PyTorch (CPU 优化)
- **图像处理**: OpenCV
- **模型优化**: ONNX Runtime (可选)
- **自动调参**: Optuna (Phase 3)

## 📝 许可证

MIT License

## 👥 作者

老佛爷 & 小李子

---

_像人类一样学习和进化的 AI 系统_
