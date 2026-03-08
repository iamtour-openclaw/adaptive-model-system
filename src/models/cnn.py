"""
轻量级 CNN 模型 - 专为 CPU 优化的图片量测模型
Phase 1: 基础模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightMeasureNet(nn.Module):
    """
    轻量级图片量测网络
    特点:
    - 参数量小 (<1M)
    - CPU 友好 (避免复杂操作)
    - 适合边缘设备
    """
    
    def __init__(self, num_classes=10, input_size=224):
        super(LightMeasureNet, self).__init__()
        
        # 特征提取器 - 逐步降采样
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112
            
            # Block 2: 112 -> 56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56
            
            # Block 3: 56 -> 28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28
            
            # Block 4: 28 -> 14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14
        )
        
        # 自适应池化到固定尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        # 回归头 (用于量测)
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # 输出：x, y, width, height
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类和回归
        cls_out = self.classifier(x)
        reg_out = self.regressor(x)
        
        return cls_out, reg_out
    
    def get_num_params(self):
        """返回参数量"""
        return sum(p.numel() for p in self.parameters())


class TinyMeasureNet(nn.Module):
    """
    超轻量级模型 - 用于简单任务
    参数量 < 100K
    """
    
    def __init__(self, num_classes=10):
        super(TinyMeasureNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.classifier = nn.Linear(32 * 2 * 2, num_classes)
        self.regressor = nn.Linear(32 * 2 * 2, 4)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        cls_out = self.classifier(x)
        reg_out = self.regressor(x)
        
        return cls_out, reg_out
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def get_model(model_name='light', num_classes=10, input_size=224):
    """
    模型工厂函数
    
    Args:
        model_name: 'light' 或 'tiny'
        num_classes: 分类类别数
        input_size: 输入图像尺寸
    
    Returns:
        模型实例
    """
    models = {
        'light': LightMeasureNet,
        'tiny': TinyMeasureNet,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](num_classes=num_classes, input_size=input_size)


if __name__ == '__main__':
    # 测试模型
    print("测试 LightMeasureNet...")
    model = get_model('light', num_classes=10)
    print(f"参数量：{model.get_num_params():,}")
    
    x = torch.randn(1, 3, 224, 224)
    cls_out, reg_out = model(x)
    print(f"分类输出形状：{cls_out.shape}")
    print(f"回归输出形状：{reg_out.shape}")
    
    print("\n测试 TinyMeasureNet...")
    model = get_model('tiny', num_classes=10)
    print(f"参数量：{model.get_num_params():,}")
