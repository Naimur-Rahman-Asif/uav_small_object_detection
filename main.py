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
    comp_parser.add_argument('--input', type=str, default='results/runs',
                             help='Directory containing run JSON files')
    comp_parser.add_argument('--output', type=str, default='results',
                             help='Directory to save comparison outputs')
    comp_parser.add_argument('--metric', type=str, default='map_50_95',
                             help='Metric name for statistical report')
    comp_parser.add_argument('--baseline', type=str, default=None,
                             help='Baseline model name for delta analysis')

    # Ablation command
    abl_parser = subparsers.add_parser('ablation', help='Run ablation variants')
    abl_parser.add_argument('--base-config', type=str, default='configs/train_config.yaml',
                            help='Base training config for ablation variants')
    abl_parser.add_argument('--epochs', type=int, default=None,
                            help='Override epochs for all variants')
    abl_parser.add_argument('--only', type=str, default=None,
                            help='Run one variant only: baseline|plus_scale_assign|plus_tiny_weighting|full_method')
    abl_parser.add_argument('--out-dir', type=str, default='configs/generated_ablation',
                            help='Directory to write generated ablation configs')

    # Ablation table command
    ablt_parser = subparsers.add_parser('ablation-table', help='Generate journal-ready ablation table')
    ablt_parser.add_argument('--input', type=str, default='results/runs',
                             help='Directory containing run JSON files')
    ablt_parser.add_argument('--output', type=str, default='results',
                             help='Directory to save ablation table outputs')
    ablt_parser.add_argument('--baseline', type=str, default='YOLOv8-SAD-baseline',
                             help='Baseline model label for delta column')
    
    args = parser.parse_args()
    
    # Change to project directory
    sys.path.insert(0, str(Path(__file__).parent))
    
    if args.command == 'train':
        print("Starting training...")
        from experiments.train import UAVTrainer
        from experiments.dataset import VisDroneDataset
        from torch.utils.data import DataLoader

        trainer = UAVTrainer(args.config, device_override=args.device)

        # Prepare datasets and dataloaders using trainer config
        data_root = Path('..') / 'data' if not Path('data').exists() else Path('data')
        img_size = trainer.config.get('img_size', 640)
        num_workers = trainer.config.get('num_workers', 0)
        pin_memory = bool(trainer.config.get('pin_memory', False) and torch.cuda.is_available())
        drop_last = bool(trainer.config.get('drop_last', True))

        train_dataset = VisDroneDataset(
            str(data_root / 'VisDrone' / 'train' / 'images'),
            str(data_root / 'VisDrone' / 'train' / 'annotations'),
            img_size=img_size
        )
        val_dataset = VisDroneDataset(
            str(data_root / 'VisDrone' / 'val' / 'images'),
            str(data_root / 'VisDrone' / 'val' / 'annotations'),
            img_size=img_size
        )

        def collate_fn(batch):
            images, targets = zip(*batch)
            images = torch.stack(images)
            return images, list(targets)

        train_loader = DataLoader(
            train_dataset,
            batch_size=trainer.config.get('batch_size', 16),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=drop_last,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, trainer.config.get('batch_size', 16) // 2),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
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
        print("Running model comparison...")
        from experiments.comparison import ModelComparator

        comparator = ModelComparator(input_dir=args.input, output_dir=args.output)
        summary = comparator.aggregate()
        plot_path = comparator.plot_comparison()
        _, stats_path = comparator.statistical_analysis(metric=args.metric, baseline_model=args.baseline)

        print("\nComparison complete:")
        print(f"Summary CSV: {Path(args.output) / 'model_comparison.csv'}")
        print(f"LaTeX table: {Path(args.output) / 'comparison_table.tex'}")
        print(f"Plot: {plot_path}")
        print(f"Stats: {stats_path}")

        if len(summary) > 0:
            best = summary.iloc[0]
            print(f"Best model by mAP@0.5:0.95: {best['model']} ({best['map_50_95_mean']:.4f})")

    elif args.command == 'ablation':
        print("Running ablation study...")
        from experiments.ablation import build_variants, load_yaml, save_yaml, run_variant
        import copy

        base_path = Path(args.base_config)
        base_cfg = load_yaml(base_path)
        variants = build_variants(base_cfg)

        selected = [args.only] if args.only else list(variants.keys())
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        config_paths = []
        for name in selected:
            if name not in variants:
                raise ValueError(f"Unknown variant: {name}")
            cfg = copy.deepcopy(variants[name])
            if args.epochs is not None:
                cfg['epochs'] = int(args.epochs)
            cfg_path = out_dir / f"{name}.yaml"
            save_yaml(cfg_path, cfg)
            config_paths.append(cfg_path)

        for cfg_path in config_paths:
            run_variant(cfg_path)

        print("Ablation complete. Aggregate with:")
        print("python main.py compare --input results/runs --output results")

    elif args.command == 'ablation-table':
        print("Generating ablation table...")
        from experiments.ablation_table import build_ablation_table

        table_df, csv_path, tex_path = build_ablation_table(
            input_dir=args.input,
            output_dir=args.output,
            baseline=args.baseline,
        )

        print("Ablation table generated:")
        print(f"CSV: {csv_path}")
        print(f"LaTeX: {tex_path}")
        if len(table_df) > 0:
            print(table_df[['Method', 'mAP50_95', 'Delta_mAP50_95']].to_string(index=False))
    
    else:
        # If no command provided, show help
        parser.print_help()

if __name__ == '__main__':
    main()
