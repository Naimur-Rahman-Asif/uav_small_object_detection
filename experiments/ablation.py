# experiments/ablation.py
import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_variants(base_cfg):
    base_loss = copy.deepcopy(base_cfg.get('loss', {}))

    variants = {
        'baseline': {
            'use_scale_adaptive_assignment': False,
            'use_small_object_weighting': False,
            'use_tiny_neighbor_supervision': False,
            'small_object_boost': 1.0,
        },
        'plus_scale_assign': {
            'use_scale_adaptive_assignment': True,
            'use_small_object_weighting': False,
            'use_tiny_neighbor_supervision': False,
            'small_object_boost': 1.0,
        },
        'plus_tiny_weighting': {
            'use_scale_adaptive_assignment': True,
            'use_small_object_weighting': True,
            'use_tiny_neighbor_supervision': False,
            'small_object_boost': 2.0,
        },
        'full_method': {
            'use_scale_adaptive_assignment': True,
            'use_small_object_weighting': True,
            'use_tiny_neighbor_supervision': True,
            'small_object_boost': 2.0,
        },
    }

    out = {}
    for name, overrides in variants.items():
        cfg = copy.deepcopy(base_cfg)
        cfg['loss'] = copy.deepcopy(base_loss)
        cfg['loss'].update(overrides)
        cfg['model_label'] = f"YOLOv8-SAD-{name}"
        cfg['experiment_name'] = f"{base_cfg.get('experiment_name', 'visdrone')}_{name}"
        out[name] = cfg
    return out


def run_variant(config_path: Path):
    cmd = [sys.executable, 'main.py', 'train', '--config', str(config_path)]
    print(f"\n[ABLATION] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for config: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for loss novelty components')
    parser.add_argument('--base-config', type=str, default='configs/train_config.yaml', help='Base config path')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs for all variants')
    parser.add_argument('--only', type=str, default=None,
                        help='Run one variant only: baseline|plus_scale_assign|plus_tiny_weighting|full_method')
    parser.add_argument('--out-dir', type=str, default='configs/generated_ablation', help='Generated config dir')
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    base_cfg = load_yaml(base_config_path)
    variants = build_variants(base_cfg)

    selected = [args.only] if args.only else list(variants.keys())
    for key in selected:
        if key not in variants:
            raise ValueError(f"Unknown variant: {key}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []
    for name in selected:
        cfg = copy.deepcopy(variants[name])
        if args.epochs is not None:
            cfg['epochs'] = int(args.epochs)

        cfg_path = out_dir / f"{name}.yaml"
        save_yaml(cfg_path, cfg)
        generated_paths.append(cfg_path)

    print("Generated ablation configs:")
    for p in generated_paths:
        print(f" - {p}")

    for cfg_path in generated_paths:
        run_variant(cfg_path)

    print("\nAblation runs completed.")
    print("Run JSON outputs are saved by trainer under results/runs/ and can be aggregated via:")
    print("python main.py compare --input results/runs --output results")


if __name__ == '__main__':
    main()
