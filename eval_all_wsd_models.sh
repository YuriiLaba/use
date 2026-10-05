#!/usr/bin/env bash

set -u

PYTHON="./.venv/bin/python"
CONFIG="${PIPELINE_CONFIG:-project_config.ini}"
BENCHMARK=""
OUTPUT=""
DEVICE=""
DEVICE_SET=0
DELETE_MODEL_AFTER_EVALUATION=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            cat <<'USAGE'
Usage: ./eval_all_wsd_models.sh [options]

Options:
  --config PATH                       Configuration file
  --device DEVICE                    cpu, cuda, or cuda:0
  --anchor-pooling sentence|target    Inference pooling (default: target)
  --models MODEL [MODEL ...]          Evaluate only the selected models
  --output PATH                      Override the output CSV
  --delete-model-after-evaluation    Delete each successfully evaluated model
  --help                             Show this help message
USAGE
            exit 0
            ;;
        --config)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --config" >&2
                exit 2
            fi
            CONFIG="$2"
            shift 2
            ;;
        --config=*)
            CONFIG="${1#*=}"
            shift
            ;;
        --device)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --device" >&2
                exit 2
            fi
            DEVICE="$2"
            DEVICE_SET=1
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            DEVICE_SET=1
            shift
            ;;
        --delete-model-after-evaluation)
            DELETE_MODEL_AFTER_EVALUATION=1
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

export PIPELINE_CONFIG="$CONFIG"

if [[ ! -x "$PYTHON" ]]; then
    echo "Virtual-environment Python not found: $PYTHON" >&2
    echo "Create the environment first with: python3.11 -m venv .venv" >&2
    exit 1
fi

BENCHMARK="$("$PYTHON" -m services.config --config "$CONFIG" --get paths benchmark)"
OUTPUT="$("$PYTHON" -m services.config --config "$CONFIG" --get evaluation wsd_output)"
if [[ "$DEVICE_SET" -eq 0 ]]; then
    DEVICE="$("$PYTHON" -m services.config --config "$CONFIG" --get evaluation device)"
fi

if [[ ! -f "$BENCHMARK" ]]; then
    echo "Benchmark file not found: $BENCHMARK" >&2
    exit 1
fi

echo "Starting WSD evaluation: $(date)"
echo "Benchmark: $BENCHMARK"
echo "Device: $DEVICE"

if [[ "$DELETE_MODEL_AFTER_EVALUATION" -eq 1 ]]; then
    echo "Model cleanup: enabled after successful evaluation"
    EXTRA_ARGS+=(--delete-model-after-evaluation)
else
    echo "Model cleanup: disabled"
fi

exec "$PYTHON" -m eval.eval_all_wsd_models \
    --benchmark-path "$BENCHMARK" \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    "${EXTRA_ARGS[@]}"
