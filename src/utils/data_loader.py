"""
Utility functions - Data loading, preprocessing, visualization
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
    Get data transforms
    
    Args:
        train: Whether in training mode
        input_size: Input size
    
    Returns:
        transforms.Compose object
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
    Image measurement dataset
    
    Supports two structures:
    1. Class-based: data/class1/img1.jpg, data/class2/img2.jpg
    2. Flat: data/img1.jpg, data/img2.jpg (auto-assigns labels)
    """
    
    def __init__(self, root_dir, transform=None, has_annotations=False):
        """
        Args:
            root_dir: Data directory
            transform: Data transforms
            has_annotations: Whether annotation file exists
        """
        self.root_dir = root_dir
        self.transform = transform
        self.has_annotations = has_annotations
        
        self.images = []
        self.labels = []
        self.boxes = []  # [x, y, w, h]
        
        self._load_data()
    
    def _load_data(self):
        """Load data"""
        # Check if there are class subdirectories
        subdirs = [d for d in os.listdir(self.root_dir) 
                   if os.path.isdir(os.path.join(self.root_dir, d))]
        
        if subdirs:
            # Class-based structure
            for class_name in sorted(subdirs):
                class_dir = os.path.join(self.root_dir, class_name)
                class_idx = hash(class_name) % 1000
                
                for img_name in os.listdir(class_dir):
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        continue
                    
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                    self.boxes.append([0.25, 0.25, 0.5, 0.5])
        else:
            # Flat structure - treat all as one class
            print(f"[DATA] Flat structure detected, loading all images")
            for img_name in sorted(os.listdir(self.root_dir)):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                img_path = os.path.join(self.root_dir, img_name)
                self.images.append(img_path)
                self.labels.append(0)  # Single class
                self.boxes.append([0.25, 0.25, 0.5, 0.5])
        
        print(f"[DATA] Loaded {len(self.images)} images from {self.root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        box = self.boxes[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensors
        label_tensor = torch.tensor(label, dtype=torch.long)
        box_tensor = torch.tensor(box, dtype=torch.float32)
        
        return image, label_tensor, box_tensor


def create_dataloader(data_dir, batch_size=16, train=True, input_size=224, num_workers=0):
    """
    Create data loader
    
    Args:
        data_dir: Data directory
        batch_size: Batch size
        train: Whether training mode
        input_size: Input size
        num_workers: Data loading threads (0 for CPU)
    
    Returns:
        DataLoader object
    """
    transform = get_transform(train=train, input_size=input_size)
    dataset = ImageMeasureDataset(data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return dataloader


def load_image(image_path, input_size=224):
    """
    Load single image for inference
    
    Args:
        image_path: Image path
        input_size: Input size
    
    Returns:
        tensor: Processed image tensor
        image: PIL Image (for visualization)
    """
    transform = get_transform(train=False, input_size=input_size)
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor, image


def draw_bbox(image, box, class_name=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box on image
    
    Args:
        image: PIL Image or numpy array
        box: [x, y, w, h] (normalized 0-1)
        class_name: Class name
        color: Box color (BGR)
        thickness: Line width
    
    Returns:
        numpy array with bbox drawn
    """
    # Convert to numpy
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()
    
    h, w = img_np.shape[:2]
    
    # Normalized to pixel coordinates
    x = int(box[0] * w)
    y = int(box[1] * h)
    bw = int(box[2] * w)
    bh = int(box[3] * h)
    
    # Draw rectangle
    cv2.rectangle(img_np, (x, y), (x + bw, y + bh), color, thickness)
    
    # Draw label
    if class_name:
        cv2.putText(img_np, class_name, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_np


if __name__ == '__main__':
    # Test data loading
    print("[TEST] Testing data loading...")
    
    # Create test data
    test_dir = "data/test"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"[INFO] Test directory: {test_dir}")
    print("Add test images to run training")
