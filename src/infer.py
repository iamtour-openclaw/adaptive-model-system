"""
推理脚本 - Phase 1: 图片量测模型推理
支持 CPU 推理，实时预测
"""

import os
import sys
import argparse
import time

import torch
import cv2
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import get_model
from src.utils import load_image, draw_bbox


def load_model(model_path, model_name='light', num_classes=10, input_size=224, device='cpu'):
    """
    加载模型
    
    Args:
        model_path: 模型文件路径
        model_name: 模型架构
        num_classes: 类别数
        input_size: 输入尺寸
        device: 设备
    
    Returns:
        模型实例
    """
    print(f"🧠 加载模型：{model_path}")
    
    model = get_model(model_name, num_classes=num_classes, input_size=input_size)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📊 训练 epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"📊 训练准确率：{checkpoint.get('acc', 'N/A'):.2f}%" if checkpoint.get('acc') else "")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功")
    return model


def predict(model, image_path, device='cpu', input_size=224, class_names=None):
    """
    单张图像预测
    
    Args:
        model: 模型实例
        image_path: 图像路径
        device: 设备
        input_size: 输入尺寸
        class_names: 类别名称列表
    
    Returns:
        dict: 预测结果
    """
    # 加载图像
    tensor, pil_image = load_image(image_path, input_size=input_size)
    tensor = tensor.to(device)
    
    # 推理
    with torch.no_grad():
        start_time = time.time()
        cls_output, reg_output = model(tensor)
        infer_time = time.time() - start_time
    
    # 解析结果
    cls_prob = torch.softmax(cls_output, dim=1)
    cls_conf, cls_pred = cls_prob.max(1)
    
    box = reg_output.squeeze().cpu().numpy()
    
    # 获取类别名称
    pred_class = cls_pred.item()
    class_name = class_names[pred_class] if class_names else f"Class {pred_class}"
    confidence = cls_conf.item() * 100
    
    result = {
        'class': pred_class,
        'class_name': class_name,
        'confidence': confidence,
        'box': box,
        'inference_time': infer_time,
    }
    
    return result


def predict_batch(model, image_paths, device='cpu', input_size=224, class_names=None):
    """
    批量预测
    
    Args:
        model: 模型实例
        image_paths: 图像路径列表
        device: 设备
        input_size: 输入尺寸
        class_names: 类别名称列表
    
    Returns:
        list: 预测结果列表
    """
    results = []
    
    for img_path in image_paths:
        result = predict(model, img_path, device, input_size, class_names)
        result['image_path'] = img_path
        results.append(result)
    
    return results


def visualize_result(image_path, result, output_path=None, show=True):
    """
    可视化预测结果
    
    Args:
        image_path: 图像路径
        result: 预测结果
        output_path: 输出路径 (可选)
        show: 是否显示
    """
    # 加载原始图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️  无法加载图像：{image_path}")
        return
    
    # 绘制边界框
    box = result['box']
    class_name = f"{result['class_name']} ({result['confidence']:.1f}%)"
    
    # OpenCV 使用 BGR
    image_with_bbox = draw_bbox(
        image, box, class_name, 
        color=(0, 255, 0), thickness=2
    )
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, image_with_bbox)
        print(f"💾 结果保存至：{output_path}")
    
    # 显示结果
    if show:
        cv2.imshow('Prediction', image_with_bbox)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image_with_bbox


def infer(args):
    """主推理函数"""
    print("=" * 60)
    print("🔍 Adaptive Model System - 推理脚本")
    print("=" * 60)
    
    # 设备配置
    device = torch.device('cpu')
    print(f"📦 使用设备：{device}")
    
    # 加载模型
    model = load_model(
        args.model_path,
        model_name=args.model,
        num_classes=args.num_classes,
        input_size=args.input_size,
        device=device,
    )
    
    # 加载类别名称 (如果有)
    class_names = None
    if args.class_names:
        with open(args.class_names, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"📋 加载类别名称：{len(class_names)} 个")
    
    # 单张图像推理
    if args.image:
        print(f"\n🖼️  推理图像：{args.image}")
        
        result = predict(
            model, args.image, device, 
            input_size=args.input_size,
            class_names=class_names,
        )
        
        print("\n📊 预测结果:")
        print(f"   类别：{result['class_name']}")
        print(f"   置信度：{result['confidence']:.2f}%")
        print(f"   边界框：{result['box']}")
        print(f"   推理时间：{result['inference_time']*1000:.2f}ms")
        
        # 可视化
        if args.output or args.show:
            visualize_result(
                args.image, result,
                output_path=args.output,
                show=args.show,
            )
    
    # 批量推理
    elif args.image_dir:
        print(f"\n📁 推理目录：{args.image_dir}")
        
        # 收集所有图像
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(image_extensions)
        ]
        
        print(f"📊 找到 {len(image_paths)} 张图像")
        
        # 批量预测
        results = predict_batch(
            model, image_paths, device,
            input_size=args.input_size,
            class_names=class_names,
        )
        
        # 统计
        print(f"\n📊 推理完成!")
        print(f"   总图像数：{len(results)}")
        avg_time = sum(r['inference_time'] for r in results) / len(results)
        print(f"   平均推理时间：{avg_time*1000:.2f}ms/张")
        
        # 保存结果
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            for result in results:
                output_path = os.path.join(
                    args.output_dir,
                    os.path.basename(result['image_path'])
                )
                visualize_result(
                    result['image_path'], result,
                    output_path=output_path,
                    show=False,
                )
            print(f"💾 结果保存至：{args.output_dir}")
    
    else:
        print("⚠️  请指定 --image 或 --image_dir")


def main():
    parser = argparse.ArgumentParser(description='AMS 图片量测模型推理')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径 (.pth)')
    parser.add_argument('--model', type=str, default='light',
                       choices=['light', 'tiny'],
                       help='模型架构')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='分类类别数')
    parser.add_argument('--input_size', type=int, default=224,
                       help='输入图像尺寸')
    
    # 输入参数
    parser.add_argument('--image', type=str,
                       help='单张图像路径')
    parser.add_argument('--image_dir', type=str,
                       help='图像目录 (批量推理)')
    
    # 输出参数
    parser.add_argument('--output', type=str,
                       help='输出图像路径 (单张)')
    parser.add_argument('--output_dir', type=str,
                       help='输出目录 (批量)')
    parser.add_argument('--show', action='store_true',
                       help='显示结果')
    
    # 其他参数
    parser.add_argument('--class_names', type=str,
                       help='类别名称文件 (每行一个)')
    
    args = parser.parse_args()
    
    # 检查输入
    if not args.image and not args.image_dir:
        print("⚠️  请指定 --image 或 --image_dir")
        print("\n使用示例:")
        print("  python src/infer.py --model_path models/best.pth --image test.jpg")
        print("  python src/infer.py --model_path models/best.pth --image_dir data/test/ --output_dir results/")
        sys.exit(1)
    
    if args.image and not os.path.exists(args.image):
        print(f"⚠️  图像不存在：{args.image}")
        sys.exit(1)
    
    if args.image_dir and not os.path.exists(args.image_dir):
        print(f"⚠️  目录不存在：{args.image_dir}")
        sys.exit(1)
    
    infer(args)


if __name__ == '__main__':
    main()
