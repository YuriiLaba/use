#!/usr/bin/env bash

set -u

PYTHON="./.venv/bin/python"
BENCHMARK="datasets_pre_defined/ukrainian_wsd_benchmark.jsonl"
OUTPUT="wsd_model_results.csv"
DEVICE="cpu"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --device" >&2
                exit 2
            fi
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -x "$PYTHON" ]]; then
    echo "Virtual-environment Python not found: $PYTHON" >&2
    echo "Create the environment first with: python3.11 -m venv .venv" >&2
    exit 1
fi

if [[ ! -f "$BENCHMARK" ]]; then
    echo "Benchmark file not found: $BENCHMARK" >&2
    exit 1
fi

echo "Starting WSD evaluation: $(date)"
echo "Benchmark: $BENCHMARK"
echo "Device: $DEVICE"

exec "$PYTHON" -m eval.eval_all_wsd_models \
    --benchmark-path "$BENCHMARK" \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    "${EXTRA_ARGS[@]}"
