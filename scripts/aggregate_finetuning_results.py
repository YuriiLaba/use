#!/usr/bin/env python3
"""Create a CSV summary from fine-tuning evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/finetuning")
    parser.add_argument("--output", default="results/finetuning_summary.csv")
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(Path(args.results_dir).glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        rows.append(
            {
                "experiment_name": metrics.get("experiment_name"),
                "pool_targets": metrics.get("pool_targets"),
                "seed": metrics.get("seed"),
                "train_data": metrics.get("train_data"),
                "wsd_accuracy": metrics.get("wsd", {}).get("accuracy"),
                "sts_pearson_cosine": metrics.get("sts", {}).get("pearson_cosine"),
                "sts_spearman_cosine": metrics.get("sts", {}).get("spearman_cosine"),
                "mteb_mean_main_score": metrics.get("mteb_metrics", {}).get(
                    "mteb/mean_main_score"
                ),
                "huggingface_url": metrics.get("huggingface_url"),
                "duration_seconds": metrics.get("duration_seconds"),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["experiment_name"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} results to {output_path}")


if __name__ == "__main__":
    main()
