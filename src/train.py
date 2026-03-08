"""
Training Script - Phase 1: Image Measurement Model
CPU-optimized training with automatic best model saving
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

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model
from src.utils import create_dataloader


def train_epoch(model, dataloader, criterion_cls, criterion_reg, optimizer, device):
    """Train for one epoch"""
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
        
        # Forward pass
        optimizer.zero_grad()
        cls_outputs, reg_outputs = model(images)
        
        # Compute loss
        cls_loss = criterion_cls(cls_outputs, labels)
        reg_loss = criterion_reg(reg_outputs, boxes)
        loss = cls_loss + 0.5 * reg_loss  # Weighted combination
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        
        _, predicted = cls_outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
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
    """Evaluate model"""
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
    """Save checkpoint"""
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
    print(f"[SAVE] Checkpoint saved: {save_path}")


def train(args):
    """Main training function"""
    print("=" * 60)
    print("[AMS] Adaptive Model System - Training Script")
    print("=" * 60)
    
    # Device configuration (CPU only for Phase 1)
    device = torch.device('cpu')
    print(f"[DEVICE] Using: {device}")
    if torch.cuda.is_available():
        print("[WARN] GPU detected, but Phase 1 uses CPU for compatibility")
    
    # Data loading
    print(f"\n[DATA] Loading data: {args.data_dir}")
    train_loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        train=True,
        input_size=args.input_size,
        num_workers=0,  # CPU mode
    )
    
    # Load validation set if exists
    val_dir = args.data_dir.replace('train', 'val')
    val_loader = None
    if os.path.exists(val_dir):
        print(f"[DATA] Loading validation: {val_dir}")
        val_loader = create_dataloader(
            val_dir,
            batch_size=args.batch_size,
            train=False,
            input_size=args.input_size,
            num_workers=0,
        )
    
    # Model
    print(f"\n[MODEL] Creating model: {args.model}")
    model = get_model(args.model, num_classes=args.num_classes, input_size=args.input_size)
    num_params = model.get_num_params()
    print(f"[STATS] Parameters: {num_params:,}")
    model.to(device)
    
    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n[TRAIN] Starting training...")
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
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, device
        )
        
        # Validation
        val_metrics = None
        if val_loader:
            val_metrics = evaluate(
                model, val_loader, criterion_cls, criterion_reg, device
            )
        
        epoch_time = time.time() - start_time
        
        # Print results
        print(f"\n[TIME] Epoch {epoch} took: {epoch_time:.1f}s")
        print(f"[STATS] Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%")
        if val_metrics:
            print(f"[STATS] Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}%")
        
        # Learning rate adjustment
        if val_metrics:
            scheduler.step(val_metrics['loss'])
        
        # Save best models
        if val_metrics:
            if val_metrics['acc'] > best_acc:
                best_acc = val_metrics['acc']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics['loss'], val_metrics['acc'],
                    args.save_dir, 'best_acc.pth'
                )
                print(f"[BEST] New best accuracy: {best_acc:.2f}%")
            
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics['loss'], val_metrics['acc'],
                    args.save_dir, 'best_loss.pth'
                )
                print(f"[BEST] New best loss: {best_loss:.4f}")
        else:
            # Save last epoch if no validation set
            save_checkpoint(
                model, optimizer, epoch, train_metrics['loss'], train_metrics['acc'],
                args.save_dir, f'checkpoint_epoch_{epoch}.pth'
            )
    
    print("\n" + "=" * 60)
    print("[OK] Training completed!")
    print(f"[BEST] Best accuracy: {best_acc:.2f}%")
    print(f"[BEST] Best loss: {best_loss:.4f}")
    print(f"[SAVE] Models saved to: {args.save_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='AMS Image Measurement Model Training')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/images',
                       help='Training data directory')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='light',
                       choices=['light', 'tiny'],
                       help='Model architecture')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Model save directory')
    
    args = parser.parse_args()
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        print(f"[WARN] Data directory not found: {args.data_dir}")
        print("[INFO] Please create data directory and add training images")
        print("\nExample structure:")
        print("  data/images/")
        print("  ├── class1/")
        print("  │   ├── img1.jpg")
        print("  │   └── ...")
        print("  └── class2/")
        sys.exit(1)
    
    train(args)


if __name__ == '__main__':
    main()
