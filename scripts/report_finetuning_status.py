#!/usr/bin/env python3
"""Print progress and the best completed fine-tuning run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/finetuning")
    parser.add_argument("--total-runs", type=int, required=True)
    parser.add_argument("--current", default=None)
    parser.add_argument("--event", choices=("started", "finished"), default=None)
    args = parser.parse_args()

    completed = []
    for metrics_path in sorted(Path(args.results_dir).glob("*/metrics.json")):
        try:
            with metrics_path.open("r", encoding="utf-8") as file:
                metrics = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        completed.append(metrics)

    valid_wsd = [
        metrics
        for metrics in completed
        if isinstance(metrics.get("wsd", {}).get("accuracy"), (int, float))
    ]
    best = max(
        valid_wsd,
        key=lambda metrics: metrics["wsd"]["accuracy"],
        default=None,
    )

    remaining = max(args.total_runs - len(completed), 0)
    prefix = "[STATUS]"
    if args.event:
        prefix += f" {args.event.upper()}"
    if args.current:
        prefix += f" {args.current}"

    print(
        f"{prefix} | completed={len(completed)}/{args.total_runs} "
        f"| remaining={remaining}"
    )

    if best is not None:
        print(
            "[STATUS] Best WSD so far: "
            f"{best.get('experiment_name')} | "
            f"accuracy={best['wsd']['accuracy']:.6f}"
        )
    else:
        print("[STATUS] Best WSD so far: no completed valid evaluation yet")


if __name__ == "__main__":
    main()
