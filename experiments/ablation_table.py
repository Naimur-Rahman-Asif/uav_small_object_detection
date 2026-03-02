# experiments/ablation_table.py
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VARIANT_ORDER = [
    'YOLOv8-SAD-baseline',
    'YOLOv8-SAD-plus_scale_assign',
    'YOLOv8-SAD-plus_tiny_weighting',
    'YOLOv8-SAD-full_method',
]

VARIANT_FLAGS = {
    'YOLOv8-SAD-baseline': (0, 0, 0),
    'YOLOv8-SAD-plus_scale_assign': (1, 0, 0),
    'YOLOv8-SAD-plus_tiny_weighting': (1, 1, 0),
    'YOLOv8-SAD-full_method': (1, 1, 1),
}


def _read_runs(input_dir: Path) -> pd.DataFrame:
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    json_files = sorted(input_dir.glob('*.json'))
    if len(json_files) == 0:
        raise FileNotFoundError(f'No run JSON files found in {input_dir}')

    rows: List[Dict] = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data.setdefault('model', file.stem)
        rows.append(data)

    return pd.DataFrame(rows)


def _format_mean_std(mean_val: float, std_val: float, scale: float = 100.0) -> str:
    return f'{mean_val * scale:.2f}±{std_val * scale:.2f}'


def build_ablation_table(input_dir='results/runs', output_dir='results', baseline='YOLOv8-SAD-baseline'):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = _read_runs(input_path)

    # Keep only ablation variants
    ablation_df = df[df['model'].isin(VARIANT_ORDER)].copy()
    if ablation_df.empty:
        raise ValueError(
            'No ablation variants found in run JSON files. '
            'Expected model names like YOLOv8-SAD-baseline / plus_scale_assign / plus_tiny_weighting / full_method.'
        )

    required_metrics = ['map_50', 'map_50_95']
    for metric in required_metrics:
        if metric not in ablation_df.columns:
            raise ValueError(f'Missing required metric in run files: {metric}')

    optional_metrics = ['ap_small', 'precision', 'recall']

    agg_targets = {
        'map_50': ['mean', 'std'],
        'map_50_95': ['mean', 'std'],
    }
    for metric in optional_metrics:
        if metric in ablation_df.columns:
            agg_targets[metric] = ['mean', 'std']

    summary = ablation_df.groupby('model', as_index=False).agg(agg_targets)
    summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in summary.columns.values
    ]

    run_counts = ablation_df.groupby('model').size().reset_index(name='num_runs')
    summary = summary.merge(run_counts, on='model', how='left')

    # Ensure all known variants are present if available
    present_order = [name for name in VARIANT_ORDER if name in summary['model'].values]
    remaining = [name for name in summary['model'].tolist() if name not in present_order]
    final_order = present_order + remaining
    summary['order_key'] = summary['model'].apply(lambda x: final_order.index(x) if x in final_order else 999)
    summary = summary.sort_values('order_key').drop(columns=['order_key'])

    if baseline not in summary['model'].values:
        baseline = summary['model'].iloc[0]

    baseline_row = summary[summary['model'] == baseline].iloc[0]
    base_map5095 = float(baseline_row['map_50_95_mean'])

    # Best flags
    best_map50 = float(summary['map_50_mean'].max())
    best_map5095 = float(summary['map_50_95_mean'].max())
    best_aps = float(summary['ap_small_mean'].max()) if 'ap_small_mean' in summary.columns else None

    table_rows = []
    for _, row in summary.iterrows():
        model = row['model']
        sa, tw, ns = VARIANT_FLAGS.get(model, (None, None, None))

        map50_mean = float(row['map_50_mean'])
        map50_std = float(row.get('map_50_std', 0.0) if pd.notna(row.get('map_50_std', 0.0)) else 0.0)
        map5095_mean = float(row['map_50_95_mean'])
        map5095_std = float(row.get('map_50_95_std', 0.0) if pd.notna(row.get('map_50_95_std', 0.0)) else 0.0)

        ap_small_mean = float(row['ap_small_mean']) if 'ap_small_mean' in row and pd.notna(row['ap_small_mean']) else None
        ap_small_std = float(row['ap_small_std']) if 'ap_small_std' in row and pd.notna(row['ap_small_std']) else 0.0

        precision_mean = float(row['precision_mean']) if 'precision_mean' in row and pd.notna(row['precision_mean']) else None
        recall_mean = float(row['recall_mean']) if 'recall_mean' in row and pd.notna(row['recall_mean']) else None

        delta_map5095 = map5095_mean - base_map5095

        table_rows.append({
            'Method': model,
            'ScaleAssign': sa,
            'TinyWeight': tw,
            'NeighborSup': ns,
            'Runs': int(row['num_runs']),
            'mAP50': _format_mean_std(map50_mean, map50_std),
            'mAP50_95': _format_mean_std(map5095_mean, map5095_std),
            'AP_S': _format_mean_std(ap_small_mean, ap_small_std) if ap_small_mean is not None else '',
            'Delta_mAP50_95': f'{delta_map5095 * 100:+.2f}',
            'Precision': f'{precision_mean * 100:.2f}' if precision_mean is not None else '',
            'Recall': f'{recall_mean * 100:.2f}' if recall_mean is not None else '',
            'Best_mAP50': int(np.isclose(map50_mean, best_map50)),
            'Best_mAP50_95': int(np.isclose(map5095_mean, best_map5095)),
            'Best_AP_S': int(np.isclose(ap_small_mean, best_aps)) if (best_aps is not None and ap_small_mean is not None) else 0,
            'mAP50_mean_raw': map50_mean,
            'mAP50_95_mean_raw': map5095_mean,
            'ap_small_mean_raw': ap_small_mean if ap_small_mean is not None else np.nan,
        })

    table_df = pd.DataFrame(table_rows)

    csv_path = output_path / 'ablation_table1.csv'
    tex_path = output_path / 'ablation_table1.tex'

    table_df.to_csv(csv_path, index=False)

    display_cols = [
        'Method', 'ScaleAssign', 'TinyWeight', 'NeighborSup', 'Runs',
        'mAP50', 'mAP50_95', 'AP_S', 'Delta_mAP50_95', 'Precision', 'Recall'
    ]
    table_df[display_cols].to_latex(tex_path, index=False)

    return table_df, csv_path, tex_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate journal-ready ablation table')
    parser.add_argument('--input', type=str, default='results/runs', help='Directory with run JSON files')
    parser.add_argument('--output', type=str, default='results', help='Directory to save ablation table files')
    parser.add_argument('--baseline', type=str, default='YOLOv8-SAD-baseline', help='Baseline model name')
    args = parser.parse_args()

    table_df, csv_path, tex_path = build_ablation_table(
        input_dir=args.input,
        output_dir=args.output,
        baseline=args.baseline,
    )

    print('Ablation table generated successfully')
    print(f'CSV: {csv_path}')
    print(f'LaTeX: {tex_path}')
    print('\nPreview:')
    print(table_df[['Method', 'mAP50_95', 'Delta_mAP50_95']].to_string(index=False))
