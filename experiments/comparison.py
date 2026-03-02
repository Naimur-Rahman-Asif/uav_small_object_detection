# experiments/comparison.py
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ModelComparator:
    """
    Aggregate model results from JSON files and generate publication artifacts.

    Expected JSON format (one file per run):
    {
      "model": "YOLOv8-SAD",
      "seed": 42,
      "map_50": 0.42,
      "map_50_95": 0.27,
      "ap_small": 0.15,
      "precision": 0.55,
      "recall": 0.49,
      "inference_fps": 42.1,
      "parameters": 11234567,
      "gflops": 16.2
    }
    """

    def __init__(self, input_dir="results/runs", output_dir="results"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_runs: List[Dict] = []
        self.summary_df: pd.DataFrame = pd.DataFrame()

    def _read_json_files(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        files = sorted(self.input_dir.glob("*.json"))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No result JSON files found in {self.input_dir}. "
                "Save each run as a JSON file first."
            )

        runs = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            data.setdefault("model", file.stem)
            data.setdefault("seed", -1)
            data["source_file"] = str(file)
            runs.append(data)

        self.raw_runs = runs
        return runs

    def aggregate(self):
        runs = self._read_json_files()
        df = pd.DataFrame(runs)

        required_metrics = ["map_50", "map_50_95"]
        for metric in required_metrics:
            if metric not in df.columns:
                raise ValueError(f"Missing required metric '{metric}' in result files.")

        agg_targets = {
            "map_50": ["mean", "std"],
            "map_50_95": ["mean", "std"],
        }

        optional_metrics = ["ap_small", "precision", "recall", "inference_fps", "parameters", "gflops"]
        for metric in optional_metrics:
            if metric in df.columns:
                agg_targets[metric] = ["mean", "std"]

        grouped = df.groupby("model", as_index=False).agg(agg_targets)
        grouped.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in grouped.columns.values
        ]

        # Add run counts
        counts = df.groupby("model").size().reset_index(name="num_runs")
        grouped = grouped.merge(counts, on="model", how="left")

        grouped = grouped.sort_values("map_50_95_mean", ascending=False)
        self.summary_df = grouped

        # Save raw and summary tables
        df.to_csv(self.output_dir / "model_runs_raw.csv", index=False)
        grouped.to_csv(self.output_dir / "model_comparison.csv", index=False)
        grouped.to_latex(self.output_dir / "comparison_table.tex", index=False, float_format="%.4f")

        return grouped

    def plot_comparison(self):
        if self.summary_df.empty:
            raise RuntimeError("No summary available. Run aggregate() first.")

        df = self.summary_df.copy()
        models = df["model"].tolist()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # mAP bars
        x = np.arange(len(models))
        map50 = df["map_50_mean"].values
        map5095 = df["map_50_95_mean"].values

        axes[0, 0].bar(x - 0.2, map50, width=0.4, label="mAP@0.50")
        axes[0, 0].bar(x + 0.2, map5095, width=0.4, label="mAP@0.50:0.95")
        axes[0, 0].set_title("Detection Accuracy")
        axes[0, 0].set_ylabel("AP")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=30, ha="right")
        axes[0, 0].legend()

        # AP_small
        if "ap_small_mean" in df.columns:
            axes[0, 1].bar(models, df["ap_small_mean"].values)
            axes[0, 1].set_ylabel("AP_small")
            axes[0, 1].set_title("Small Object Performance")
            axes[0, 1].tick_params(axis="x", rotation=30)
        else:
            axes[0, 1].axis("off")

        # Speed vs accuracy
        if "inference_fps_mean" in df.columns:
            axes[1, 0].scatter(df["inference_fps_mean"], df["map_50_95_mean"])
            for _, row in df.iterrows():
                axes[1, 0].annotate(row["model"], (row["inference_fps_mean"], row["map_50_95_mean"]))
            axes[1, 0].set_xlabel("FPS")
            axes[1, 0].set_ylabel("mAP@0.50:0.95")
            axes[1, 0].set_title("Speed vs Accuracy")
        else:
            axes[1, 0].axis("off")

        # Params vs GFLOPs
        if "parameters_mean" in df.columns and "gflops_mean" in df.columns:
            params_m = df["parameters_mean"] / 1e6
            axes[1, 1].scatter(params_m, df["gflops_mean"])
            for _, row in df.iterrows():
                axes[1, 1].annotate(row["model"], (row["parameters_mean"] / 1e6, row["gflops_mean"]))
            axes[1, 1].set_xlabel("Parameters (M)")
            axes[1, 1].set_ylabel("GFLOPs")
            axes[1, 1].set_title("Model Complexity")
        else:
            axes[1, 1].axis("off")

        plt.tight_layout()
        out_path = self.output_dir / "model_comparison.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return out_path

    def statistical_analysis(self, metric="map_50_95", baseline_model=None):
        """
        Simple paired-like report by seed overlap (if available).
        Saves a CSV with mean/std and delta vs baseline.
        """
        if len(self.raw_runs) == 0:
            raise RuntimeError("No raw runs loaded. Run aggregate() first.")

        df = pd.DataFrame(self.raw_runs)
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not present in run files.")

        if baseline_model is None:
            baseline_model = self.summary_df.iloc[-1]["model"] if not self.summary_df.empty else df["model"].iloc[0]

        baseline_vals = df[df["model"] == baseline_model][metric].astype(float).values
        if len(baseline_vals) == 0:
            raise ValueError(f"Baseline model '{baseline_model}' has no runs.")

        rows = []
        for model, g in df.groupby("model"):
            vals = g[metric].astype(float).values
            row = {
                "model": model,
                "metric": metric,
                "num_runs": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "delta_vs_baseline": float(np.mean(vals) - np.mean(baseline_vals)),
            }
            rows.append(row)

        stats_df = pd.DataFrame(rows).sort_values("mean", ascending=False)
        stats_path = self.output_dir / f"statistical_analysis_{metric}.csv"
        stats_df.to_csv(stats_path, index=False)
        return stats_df, stats_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate model comparison results")
    parser.add_argument("--input", type=str, default="results/runs", help="Directory with run JSON files")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--metric", type=str, default="map_50_95", help="Metric for statistical report")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline model name")
    args = parser.parse_args()

    comparator = ModelComparator(input_dir=args.input, output_dir=args.output)
    summary = comparator.aggregate()
    plot_path = comparator.plot_comparison()
    stats_df, stats_path = comparator.statistical_analysis(metric=args.metric, baseline_model=args.baseline)

    print("Comparison complete")
    print(f"Summary table: {args.output}/model_comparison.csv")
    print(f"LaTeX table: {args.output}/comparison_table.tex")
    print(f"Plot: {plot_path}")
    print(f"Stats: {stats_path}")
    print("\nTop models:")
    print(summary[["model", "map_50_mean", "map_50_95_mean", "num_runs"]].head())
