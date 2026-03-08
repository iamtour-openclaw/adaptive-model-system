"""
示例数据生成器 - 创建测试数据集
用于快速验证训练流程
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_synthetic_image(width=224, height=224, num_shapes=3):
    """
    生成合成图像 (带几何图形)
    
    Args:
        width: 图像宽度
        height: 图像高度
        num_shapes: 图形数量
    
    Returns:
        PIL Image, list of boxes
    """
    # 创建空白图像
    image = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    boxes = []
    
    # 随机生成图形
    for _ in range(num_shapes):
        # 随机选择图形类型
        shape_type = np.random.choice(['rectangle', 'circle', 'triangle'])
        
        # 随机颜色
        color = tuple(np.random.randint(50, 200, 3).tolist())
        
        # 随机位置
        x = np.random.randint(20, width - 60)
        y = np.random.randint(20, height - 60)
        w = np.random.randint(30, 80)
        h = np.random.randint(30, 80)
        
        # 绘制图形
        if shape_type == 'rectangle':
            draw.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0, 128))
            boxes.append((x, y, w, h, 'rectangle'))
        
        elif shape_type == 'circle':
            draw.ellipse([x, y, x + w, y + h], fill=color, outline=(0, 0, 0, 128))
            boxes.append((x, y, w, h, 'circle'))
        
        elif shape_type == 'triangle':
            points = [
                (x + w // 2, y),
                (x, y + h),
                (x + w, y + h),
            ]
            draw.polygon(points, fill=color, outline=(0, 0, 0, 128))
            boxes.append((x, y, w, h, 'triangle'))
    
    return image, boxes


def create_dataset(output_dir, num_train=100, num_val=20, num_test=10):
    """
    创建完整数据集
    
    Args:
        output_dir: 输出目录
        num_train: 训练集数量
        num_val: 验证集数量
        num_test: 测试集数量
    """
    print("[DATA] Creating synthetic dataset...")
    print(f"   Train: {num_train} images")
    print(f"   Val: {num_val} images")
    print(f"   Test: {num_test} images")
    
    splits = {
        'train': num_train,
        'val': num_val,
        'test': num_test,
    }
    
    for split_name, num_images in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\n[DATA] Generating {split_name} set...")
        
        for i in range(num_images):
            # 生成图像
            image, boxes = generate_synthetic_image()
            
            # 保存图像
            img_path = os.path.join(split_dir, f'img_{i:04d}.png')
            image.save(img_path)
        
        print(f"[OK] {split_name} set done: {num_images} images")
    
    print(f"\n[OK] Dataset created successfully!")
    print(f"[INFO] Saved to: {output_dir}")
    
    # 创建说明文件
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# 合成数据集\n\n")
        f.write("这是由 `generate_data.py` 自动生成的测试数据集。\n\n")
        f.write("## 结构\n\n")
        f.write("```\n")
        f.write("data/\n")
        f.write("├── train/  # 训练集\n")
        f.write("├── val/    # 验证集\n")
        f.write("└── test/   # 测试集\n")
        f.write("```\n\n")
        f.write("## 使用\n\n")
        f.write("```bash\n")
        f.write("# 训练\n")
        f.write("python src/train.py --data_dir data/train\n")
        f.write("\n")
        f.write("# 推理\n")
        f.write("python src/infer.py --model_path models/best.pth --image data/test/img_0000.png\n")
        f.write("```\n")
    
    print(f"[INFO] README created: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description='生成合成测试数据集')
    
    parser.add_argument('--output_dir', type=str, default='data/images',
                       help='输出目录')
    parser.add_argument('--train', type=int, default=100,
                       help='训练集数量')
    parser.add_argument('--val', type=int, default=20,
                       help='验证集数量')
    parser.add_argument('--test', type=int, default=10,
                       help='测试集数量')
    
    args = parser.parse_args()
    
    create_dataset(
        args.output_dir,
        num_train=args.train,
        num_val=args.val,
        num_test=args.test,
    )


if __name__ == '__main__':
    main()
