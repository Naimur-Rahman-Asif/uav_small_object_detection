#!/usr/bin/env python
# main.py - Entry point for UAV Small Object Detection project

import argparse
import sys
import os
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(
        description='UAV Small Object Detection - YOLOv8 Enhanced',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python main.py train                          # Start training
    python main.py test --model weights/best_model.pth  # Test on dataset
    python main.py infer --model weights/best_model.pth --image test.jpg  # Infer on image
    python main.py validate --model weights/best_model.pth  # Validate on validation set
                '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                              help='Path to training config')
    train_parser.add_argument('--device', type=str, default='cuda',
                              help='Device to use (cuda/cpu)')
    
    # Testing command
    test_parser = subparsers.add_parser('test', help='Test on dataset')
    test_parser.add_argument('--model', type=str, default='weights/best_model.pth',
                             help='Path to model weights')
    test_parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                             help='Path to config')
    test_parser.add_argument('--device', type=str, default='cuda',
                             help='Device to use')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on image')
    infer_parser.add_argument('--model', type=str, default='weights/best_model.pth',
                              help='Path to model weights')
    infer_parser.add_argument('--image', type=str, required=True,
                              help='Path to input image')
    infer_parser.add_argument('--device', type=str, default='cuda',
                              help='Device to use')
    
    # Validation command
    val_parser = subparsers.add_parser('validate', help='Validate model')
    val_parser.add_argument('--model', type=str, default='weights/best_model.pth',
                            help='Path to model weights')
    val_parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                            help='Path to config')
    val_parser.add_argument('--device', type=str, default='cuda',
                            help='Device to use')
    
    # Comparison command
    comp_parser = subparsers.add_parser('compare', help='Compare models')
    
    args = parser.parse_args()
    
    # Change to project directory
    sys.path.insert(0, str(Path(__file__).parent))
    
    if args.command == 'train':
        print("Starting training...")
        from experiments.train import UAVTrainer
        from experiments.dataset import VisDroneDataset
        from torch.utils.data import DataLoader

        trainer = UAVTrainer(args.config)

        # Prepare datasets and dataloaders using trainer config
        data_root = Path('..') / 'data' if not Path('data').exists() else Path('data')
        train_dataset = VisDroneDataset(
            str(data_root / 'VisDrone' / 'train' / 'images'),
            str(data_root / 'VisDrone' / 'train' / 'annotations')
        )
        val_dataset = VisDroneDataset(
            str(data_root / 'VisDrone' / 'val' / 'images'),
            str(data_root / 'VisDrone' / 'val' / 'annotations')
        )

        def collate_fn(batch):
            images, targets = zip(*batch)
            images = torch.stack(images)
            return images, list(targets)

        train_loader = DataLoader(
            train_dataset,
            batch_size=trainer.config.get('batch_size', 16),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, trainer.config.get('batch_size', 16) // 2),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn
        )

        if len(train_loader) > 0:
            trainer.train(train_loader, val_loader)
        else:
            print("Error: No training data available. Ensure data/VisDrone is populated.")
    
    elif args.command == 'test':
        print("Running test...")
        from experiments.test import Tester
        from experiments.dataset import VisDroneDataset
        from torch.utils.data import DataLoader

        tester = Tester(model_path=args.model, config_path=args.config, device=args.device)

        # If user provided an image, run single-image inference (handled in 'infer')
        # Otherwise run on test dataset
        data_root = Path('..') / 'data' if not Path('data').exists() else Path('data')
        test_dataset = VisDroneDataset(
            str(data_root / 'VisDrone' / 'test' / 'images'),
            str(data_root / 'VisDrone' / 'test' / 'annotations')
        )

        def collate_fn(batch):
            images, targets = zip(*batch)
            images = torch.stack(images)
            return images, list(targets)

        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

        if len(test_loader) > 0:
            metrics = tester.test_dataset(test_loader)
            print("\nTest Results:")
            print(f"mAP@0.5: {metrics.get('map_50', 0):.4f}")
            print(f"mAP@0.5:0.95: {metrics.get('map_50_95', 0):.4f}")
        else:
            print("Error: No test data available. Ensure data/VisDrone/test is populated.")
    
    elif args.command == 'infer':
        print(f"Running inference on {args.image}...")
        from experiments.test import Tester
        tester = Tester(model_path=args.model, device=args.device)
        predictions, img_shape = tester.test_image(args.image)
        
        if predictions:
            print(f"Detected {len(predictions)} objects")
            for i, pred in enumerate(predictions):
                print(f"  Object {i+1}: Bounding box: {pred[:4]}, Confidence: {pred[4]:.3f}")
        else:
            print("No objects detected")
    
    elif args.command == 'validate':
        print("Running validation...")
        from experiments.validate import Validator
        from experiments.dataset import VisDroneDataset
        from torch.utils.data import DataLoader

        validator = Validator(model_path=args.model, config_path=args.config, device=args.device)

        data_root = Path('..') / 'data' if not Path('data').exists() else Path('data')
        val_dataset = VisDroneDataset(
            str(data_root / 'VisDrone' / 'val' / 'images'),
            str(data_root / 'VisDrone' / 'val' / 'annotations')
        )

        def collate_fn(batch):
            images, targets = zip(*batch)
            images = torch.stack(images)
            return images, list(targets)

        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

        if len(val_loader) > 0:
            metrics = validator.validate(val_loader)
            print("\nValidation Results:")
            print(f"mAP@0.5: {metrics.get('map_50', 0):.4f}")
            print(f"mAP@0.5:0.95: {metrics.get('map_50_95', 0):.4f}")
        else:
            print("Error: No validation data available. Ensure data/VisDrone/val is populated.")
    
    elif args.command == 'compare':
        print("Model comparison is a placeholder. Use experiments/comparison.py directly for custom comparisons.")
    
    else:
        # If no command provided, show help
        parser.print_help()

if __name__ == '__main__':
    main()
