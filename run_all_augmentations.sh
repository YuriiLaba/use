#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG="${PIPELINE_CONFIG:-$ROOT_DIR/project_config.ini}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

export PIPELINE_CONFIG="$CONFIG"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi

config_value() {
    "$PYTHON" -m services.config --config "$CONFIG" --get "$1" "$2"
}

MASK_GPU="$(config_value augmentation mask_gpu)"
MASK_DEFINITIONS_GPU="$(config_value augmentation mask_definitions_gpu)"
TRANSLATION_GPU="$(config_value augmentation translation_gpu)"
TRANSLATION_DEFINITIONS_GPU="$(config_value augmentation translation_definitions_gpu)"
COMBINED_GPU="$(config_value augmentation combined_gpu)"
COMBINED_DEFINITIONS_GPU="$(config_value augmentation combined_definitions_gpu)"

run_parallel() {
    local -a pids=()
    local status=0

    for command in "$@"; do
        eval "$command" &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done

    return "$status"
}

echo "Starting individual augmentations..."

run_parallel \
    "\"$PYTHON\" -m augment.dropout.dropout" \
    "\"$PYTHON\" -m augment.dropout.dropout_definitions" \
    "\"$PYTHON\" -m augment.token_shuffling.token_shuffling" \
    "\"$PYTHON\" -m augment.token_shuffling.token_shuffling_definitions" \
    "AUGMENT_GPU_ID=$MASK_GPU \"$PYTHON\" -m augment.mask.mask" \
    "AUGMENT_GPU_ID=$MASK_DEFINITIONS_GPU \"$PYTHON\" -m augment.mask.mask_definitions" \
    "AUGMENT_GPU_ID=$TRANSLATION_GPU \"$PYTHON\" -m augment.translation.augment_translation" \
    "AUGMENT_GPU_ID=$TRANSLATION_DEFINITIONS_GPU \"$PYTHON\" -m augment.translation.augment_translation_definitions"

echo "Individual augmentations completed."
echo "Starting combined augmentations..."

run_parallel \
    "AUGMENT_GPU_ID=$COMBINED_GPU \"$PYTHON\" -m augment.augment_all_together" \
    "AUGMENT_GPU_ID=$COMBINED_DEFINITIONS_GPU \"$PYTHON\" -m augment.augment_all_together_definitions"

echo "All augmentations completed successfully."
