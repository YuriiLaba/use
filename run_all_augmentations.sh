#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi

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
    "AUGMENT_GPU_ID=0 \"$PYTHON\" -m augment.mask.mask" \
    "AUGMENT_GPU_ID=1 \"$PYTHON\" -m augment.mask.mask_definitions" \
    "AUGMENT_GPU_ID=2 \"$PYTHON\" -m augment.translation.augment_translation" \
    "AUGMENT_GPU_ID=3 \"$PYTHON\" -m augment.translation.augment_translation_definitions"

echo "Individual augmentations completed."
echo "Starting combined augmentations..."

run_parallel \
    "AUGMENT_GPU_ID=0 \"$PYTHON\" -m augment.augment_all_together" \
    "AUGMENT_GPU_ID=1 \"$PYTHON\" -m augment.augment_all_together_definitions"

echo "All augmentations completed successfully."
