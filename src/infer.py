"""
Inference Script - Phase 1: Image Measurement Model
CPU-optimized inference with real-time prediction
"""

import os
import sys
import argparse
import time

import torch
import cv2
import numpy as np
from PIL import Image

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models import get_model
from src.utils import load_image, draw_bbox


def load_model(model_path, model_name='light', num_classes=10, input_size=224, device='cpu'):
    """
    Load model
    
    Args:
        model_path: Model file path
        model_name: Model architecture
        num_classes: Number of classes
        input_size: Input size
        device: Device
    
    Returns:
        Model instance
    """
    print(f"[MODEL] Loading: {model_path}")
    
    model = get_model(model_name, num_classes=num_classes, input_size=input_size)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[STATS] Epoch: {checkpoint.get('epoch', 'N/A')}")
        if checkpoint.get('acc'):
            print(f"[STATS] Training accuracy: {checkpoint['acc']:.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"[OK] Model loaded")
    return model


def predict(model, image_path, device='cpu', input_size=224, class_names=None):
    """
    Single image prediction
    
    Args:
        model: Model instance
        image_path: Image path
        device: Device
        input_size: Input size
        class_names: List of class names
    
    Returns:
        dict: Prediction results
    """
    # Load image
    tensor, pil_image = load_image(image_path, input_size=input_size)
    tensor = tensor.to(device)
    
    # Inference
    with torch.no_grad():
        start_time = time.time()
        cls_output, reg_output = model(tensor)
        infer_time = time.time() - start_time
    
    # Parse results
    cls_prob = torch.softmax(cls_output, dim=1)
    cls_conf, cls_pred = cls_prob.max(1)
    
    box = reg_output.squeeze().cpu().numpy()
    
    # Get class name
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
    Batch prediction
    
    Args:
        model: Model instance
        image_paths: List of image paths
        device: Device
        input_size: Input size
        class_names: List of class names
    
    Returns:
        list: Prediction results
    """
    results = []
    
    for img_path in image_paths:
        result = predict(model, img_path, device, input_size, class_names)
        result['image_path'] = img_path
        results.append(result)
    
    return results


def visualize_result(image_path, result, output_path=None, show=True):
    """
    Visualize prediction results
    
    Args:
        image_path: Image path
        result: Prediction result
        output_path: Output path (optional)
        show: Whether to display
    """
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Cannot load image: {image_path}")
        return
    
    # Draw bounding box
    box = result['box']
    class_name = f"{result['class_name']} ({result['confidence']:.1f}%)"
    
    # OpenCV uses BGR
    image_with_bbox = draw_bbox(
        image, box, class_name, 
        color=(0, 255, 0), thickness=2
    )
    
    # Save result
    if output_path:
        cv2.imwrite(output_path, image_with_bbox)
        print(f"[SAVE] Result saved: {output_path}")
    
    # Display result
    if show:
        cv2.imshow('Prediction', image_with_bbox)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image_with_bbox


def infer(args):
    """Main inference function"""
    print("=" * 60)
    print("[AMS] Adaptive Model System - Inference Script")
    print("=" * 60)
    
    # Device configuration
    device = torch.device('cpu')
    print(f"[DEVICE] Using: {device}")
    
    # Load model
    model = load_model(
        args.model_path,
        model_name=args.model,
        num_classes=args.num_classes,
        input_size=args.input_size,
        device=device,
    )
    
    # Load class names (if provided)
    class_names = None
    if args.class_names:
        with open(args.class_names, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"[INFO] Loaded {len(class_names)} class names")
    
    # Single image inference
    if args.image:
        print(f"\n[IMG] Image: {args.image}")
        
        result = predict(
            model, args.image, device, 
            input_size=args.input_size,
            class_names=class_names,
        )
        
        print("\n[STATS] Prediction results:")
        print(f"   Class: {result['class_name']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Box: {result['box']}")
        print(f"   Inference time: {result['inference_time']*1000:.2f}ms")
        
        # Visualize
        if args.output or args.show:
            visualize_result(
                args.image, result,
                output_path=args.output,
                show=args.show,
            )
    
    # Batch inference
    elif args.image_dir:
        print(f"\n[DATA] Directory: {args.image_dir}")
        
        # Collect all images
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(image_extensions)
        ]
        
        print(f"[STATS] Found {len(image_paths)} images")
        
        # Batch prediction
        results = predict_batch(
            model, image_paths, device,
            input_size=args.input_size,
            class_names=class_names,
        )
        
        # Statistics
        print(f"\n[OK] Inference completed!")
        print(f"   Total images: {len(results)}")
        avg_time = sum(r['inference_time'] for r in results) / len(results)
        print(f"   Average time: {avg_time*1000:.2f}ms/image")
        
        # Save results
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
            print(f"[SAVE] Results saved: {args.output_dir}")
    
    else:
        print("[WARN] Please specify --image or --image_dir")


def main():
    parser = argparse.ArgumentParser(description='AMS Image Measurement Model Inference')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Model file path (.pth)')
    parser.add_argument('--model', type=str, default='light',
                       choices=['light', 'tiny'],
                       help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size')
    
    # Input parameters
    parser.add_argument('--image', type=str,
                       help='Single image path')
    parser.add_argument('--image_dir', type=str,
                       help='Image directory (batch inference)')
    
    # Output parameters
    parser.add_argument('--output', type=str,
                       help='Output image path (single)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory (batch)')
    parser.add_argument('--show', action='store_true',
                       help='Show results')
    
    # Other parameters
    parser.add_argument('--class_names', type=str,
                       help='Class names file (one per line)')
    
    args = parser.parse_args()
    
    # Check input
    if not args.image and not args.image_dir:
        print("[WARN] Please specify --image or --image_dir")
        print("\nExamples:")
        print("  python src/infer.py --model_path models/best.pth --image test.jpg")
        print("  python src/infer.py --model_path models/best.pth --image_dir data/test/ --output_dir results/")
        sys.exit(1)
    
    if args.image and not os.path.exists(args.image):
        print(f"[WARN] Image not found: {args.image}")
        sys.exit(1)
    
    if args.image_dir and not os.path.exists(args.image_dir):
        print(f"[WARN] Directory not found: {args.image_dir}")
        sys.exit(1)
    
    infer(args)


if __name__ == '__main__':
    main()
