"""
工具函数 - 数据加载、预处理、可视化等
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def get_transform(train=True, input_size=224):
    """
    获取数据变换
    
    Args:
        train: 是否为训练模式
        input_size: 输入尺寸
    
    Returns:
        transforms.Compose 对象
    """
    if train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])


class ImageMeasureDataset(Dataset):
    """
    图片量测数据集
    
    假设数据目录结构:
    data/
    ├── images/
    │   ├── class1/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── class2/
    └── annotations.txt (可选，包含边界框标注)
    """
    
    def __init__(self, root_dir, transform=None, has_annotations=False):
        """
        Args:
            root_dir: 数据根目录
            transform: 数据变换
            has_annotations: 是否有标注文件
        """
        self.root_dir = root_dir
        self.transform = transform
        self.has_annotations = has_annotations
        
        self.images = []
        self.labels = []
        self.boxes = []  # [x, y, w, h]
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        # 遍历所有类别目录
        for class_name in sorted(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            class_idx = hash(class_name) % 1000  # 简单哈希作为类别索引
            
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)
                
                # 假标注 (实际使用时需要真实标注)
                self.boxes.append([0.25, 0.25, 0.5, 0.5])  # 归一化的 [x, y, w, h]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        box = self.boxes[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 转换标注为 tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        box_tensor = torch.tensor(box, dtype=torch.float32)
        
        return image, label_tensor, box_tensor


def create_dataloader(data_dir, batch_size=16, train=True, input_size=224, num_workers=0):
    """
    创建数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        train: 是否训练模式
        input_size: 输入尺寸
        num_workers: 数据加载线程数 (CPU 建议 0)
    
    Returns:
        DataLoader 对象
    """
    transform = get_transform(train=train, input_size=input_size)
    dataset = ImageMeasureDataset(data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,  # CPU 模式建议设为 0
        pin_memory=False,  # CPU 模式不需要 pinned memory
    )
    
    return dataloader


def load_image(image_path, input_size=224):
    """
    加载单张图像用于推理
    
    Args:
        image_path: 图像路径
        input_size: 输入尺寸
    
    Returns:
        tensor: 处理后的图像 tensor
        image: PIL Image (用于可视化)
    """
    transform = get_transform(train=False, input_size=input_size)
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    
    # 添加 batch 维度
    tensor = tensor.unsqueeze(0)
    
    return tensor, image


def draw_bbox(image, box, class_name=None, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制边界框
    
    Args:
        image: PIL Image 或 numpy array
        box: [x, y, w, h] (归一化坐标 0-1)
        class_name: 类别名称
        color: 边框颜色 (BGR)
        thickness: 线宽
    
    Returns:
        numpy array with bbox drawn
    """
    # 转换为 numpy
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()
    
    h, w = img_np.shape[:2]
    
    # 归一化坐标转像素坐标
    x = int(box[0] * w)
    y = int(box[1] * h)
    bw = int(box[2] * w)
    bh = int(box[3] * h)
    
    # 绘制矩形
    cv2.rectangle(img_np, (x, y), (x + bw, y + bh), color, thickness)
    
    # 绘制标签
    if class_name:
        cv2.putText(img_np, class_name, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_np


if __name__ == '__main__':
    # 测试数据加载
    print("测试数据加载...")
    
    # 创建测试数据
    test_dir = "data/test"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"测试数据目录：{test_dir}")
    print("请添加一些测试图像到该目录后运行训练脚本")
