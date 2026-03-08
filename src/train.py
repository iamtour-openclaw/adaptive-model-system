"""
训练脚本 - Phase 1: 图片量测模型训练
支持 CPU 训练，自动保存最佳模型
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model
from src.utils import create_dataloader


def train_epoch(model, dataloader, criterion_cls, criterion_reg, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels, boxes) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        boxes = boxes.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        cls_outputs, reg_outputs = model(images)
        
        # 计算损失
        cls_loss = criterion_cls(cls_outputs, labels)
        reg_loss = criterion_reg(reg_outputs, boxes)
        loss = cls_loss + 0.5 * reg_loss  # 加权组合
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        
        _, predicted = cls_outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%',
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'reg_loss': total_reg_loss / len(dataloader),
        'acc': 100. * correct / total,
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion_cls, criterion_reg, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Evaluating')
    for images, labels, boxes in pbar:
        images = images.to(device)
        labels = labels.to(device)
        boxes = boxes.to(device)
        
        cls_outputs, reg_outputs = model(images)
        
        cls_loss = criterion_cls(cls_outputs, labels)
        reg_loss = criterion_reg(reg_outputs, boxes)
        loss = cls_loss + 0.5 * reg_loss
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        
        _, predicted = cls_outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%',
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'reg_loss': total_reg_loss / len(dataloader),
        'acc': 100. * correct / total,
    }


def save_checkpoint(model, optimizer, epoch, loss, acc, save_dir, filename='checkpoint.pth'):
    """保存检查点"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
        'timestamp': datetime.now().isoformat(),
    }
    
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"✓ 保存检查点：{save_path}")


def train(args):
    """主训练函数"""
    print("=" * 60)
    print("🚀 Adaptive Model System - 训练脚本")
    print("=" * 60)
    
    # 设备配置 (CPU only for Phase 1)
    device = torch.device('cpu')
    print(f"📦 使用设备：{device}")
    if torch.cuda.is_available():
        print("⚠️  检测到 GPU，但 Phase 1 优先使用 CPU 以确保兼容性")
    
    # 数据加载
    print(f"\n📁 加载数据：{args.data_dir}")
    train_loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        train=True,
        input_size=args.input_size,
        num_workers=0,  # CPU 模式
    )
    
    # 如果验证集存在则加载
    val_dir = args.data_dir.replace('train', 'val')
    val_loader = None
    if os.path.exists(val_dir):
        print(f"📁 加载验证集：{val_dir}")
        val_loader = create_dataloader(
            val_dir,
            batch_size=args.batch_size,
            train=False,
            input_size=args.input_size,
            num_workers=0,
        )
    
    # 模型
    print(f"\n🧠 创建模型：{args.model}")
    model = get_model(args.model, num_classes=args.num_classes, input_size=args.input_size)
    num_params = model.get_num_params()
    print(f"📊 参数量：{num_params:,}")
    model.to(device)
    
    # 损失函数
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    print(f"\n📈 开始训练...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print("-" * 60)
    
    best_acc = 0
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, device
        )
        
        # 验证
        val_metrics = None
        if val_loader:
            val_metrics = evaluate(
                model, val_loader, criterion_cls, criterion_reg, device
            )
        
        epoch_time = time.time() - start_time
        
        # 打印结果
        print(f"\n⏱️  Epoch {epoch} 耗时：{epoch_time:.1f}s")
        print(f"📊 训练集 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%")
        if val_metrics:
            print(f"📊 验证集 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}%")
        
        # 学习率调整
        if val_metrics:
            scheduler.step(val_metrics['loss'])
        
        # 保存最佳模型
        if val_metrics:
            if val_metrics['acc'] > best_acc:
                best_acc = val_metrics['acc']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics['loss'], val_metrics['acc'],
                    args.save_dir, 'best_acc.pth'
                )
                print(f"🏆 新的最佳准确率：{best_acc:.2f}%")
            
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics['loss'], val_metrics['acc'],
                    args.save_dir, 'best_loss.pth'
                )
                print(f"🏆 新的最低损失：{best_loss:.4f}")
        else:
            # 没有验证集时保存最后一个 epoch
            save_checkpoint(
                model, optimizer, epoch, train_metrics['loss'], train_metrics['acc'],
                args.save_dir, f'checkpoint_epoch_{epoch}.pth'
            )
    
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print(f"🏆 最佳准确率：{best_acc:.2f}%")
    print(f"🏆 最低损失：{best_loss:.4f}")
    print(f"💾 模型保存至：{args.save_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='AMS 图片量测模型训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/images',
                       help='训练数据目录')
    parser.add_argument('--input_size', type=int, default=224,
                       help='输入图像尺寸')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='分类类别数')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='light',
                       choices=['light', 'tiny'],
                       help='模型架构')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='models',
                       help='模型保存目录')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"⚠️  数据目录不存在：{args.data_dir}")
        print("📝 请创建数据目录并添加训练图像")
        print("\n目录结构示例:")
        print("  data/images/")
        print("  ├── class1/")
        print("  │   ├── img1.jpg")
        print("  │   └── ...")
        print("  └── class2/")
        sys.exit(1)
    
    train(args)


if __name__ == '__main__':
    main()
