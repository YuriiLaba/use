#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
MODULE="local_datasets.semi_supervised_2.form_triplets"
CONFIG="${PIPELINE_CONFIG:-$ROOT_DIR/project_config.ini}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

export PIPELINE_CONFIG="$CONFIG"

config_value() {
    "$PYTHON" -m services.config --config "$CONFIG" --get "$1" "$2"
}

OUTPUT_DIR="$(config_value paths triplets_dir)"
COLLECTED_DATASET="$(config_value paths filtered_grouped)"
GENERATED_DATASET="$(config_value paths merged_dataset)"

DROPOUT="$(config_value augmentation dropout_output)"
DROPOUT_DEFINITIONS="$(config_value augmentation dropout_definitions_output)"
MASK="$(config_value augmentation mask_output)"
MASK_DEFINITIONS="$(config_value augmentation mask_definitions_output)"
SHUFFLING="$(config_value augmentation shuffling_output)"
SHUFFLING_DEFINITIONS="$(config_value augmentation shuffling_definitions_output)"
TRANSLATION="$(config_value augmentation translation_output)"
TRANSLATION_DEFINITIONS="$(config_value augmentation translation_definitions_output)"
COMBINED="$(config_value augmentation combined_output)"
COMBINED_DEFINITIONS="$(config_value augmentation combined_definitions_output)"

NATURAL_OUTPUT="$(config_value triplets natural_output)"
GENERATION_OUTPUT="$(config_value triplets generation_output)"
MASK_OUTPUT="$(config_value triplets mask_output)"
DROPOUT_OUTPUT="$(config_value triplets dropout_output)"
TRANSLATION_OUTPUT="$(config_value triplets translation_output)"
SHUFFLING_OUTPUT="$(config_value triplets shuffling_output)"
STOCHASTIC_OUTPUT="$(config_value triplets stochastic_output)"
ALL_COMBINED_OUTPUT="$(config_value triplets all_combined_output)"
TRIPLET_SEED="$(config_value triplets seed)"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

run_condition() {
    local name="$1"
    shift

    echo
    echo "========== Generating triplets: $name =========="
    "$PYTHON" -m "$MODULE" "$@"
    echo "========== Finished: $name =========="
}

mkdir -p "$OUTPUT_DIR"

# Natural: collected examples only, without LLM generation or augmentation.
require_file "$COLLECTED_DATASET"
run_condition "natural" \
    --dataset-path "$COLLECTED_DATASET" \
    --output-csv "$NATURAL_OUTPUT" \
    --no-augmented \
    --no-definitions-augmented \
    --seed "$TRIPLET_SEED"

# Generation: collected examples plus generated examples, without augmentation.
require_file "$GENERATED_DATASET"
run_condition "generation" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$GENERATION_OUTPUT" \
    --no-augmented \
    --no-definitions-augmented \
    --seed "$TRIPLET_SEED"

# Generation plus one individual augmentation technique.
run_condition "generation + dropout" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$DROPOUT_OUTPUT" \
    --augmentation-path "$DROPOUT" \
    --definitions-augmentation-path "$DROPOUT_DEFINITIONS" \
    --seed "$TRIPLET_SEED"

run_condition "generation + masking" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$MASK_OUTPUT" \
    --augmentation-path "$MASK" \
    --definitions-augmentation-path "$MASK_DEFINITIONS" \
    --seed "$TRIPLET_SEED"

run_condition "generation + token shuffling" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$SHUFFLING_OUTPUT" \
    --augmentation-path "$SHUFFLING" \
    --definitions-augmentation-path "$SHUFFLING_DEFINITIONS" \
    --seed "$TRIPLET_SEED"

run_condition "generation + back translation" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$TRANSLATION_OUTPUT" \
    --augmentation-path "$TRANSLATION" \
    --definitions-augmentation-path "$TRANSLATION_DEFINITIONS" \
    --seed "$TRIPLET_SEED"

# Stochastic combination produced by augment_all_together.py.
run_condition "generation + stochastic combination" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$STOCHASTIC_OUTPUT" \
    --augmentation-path "$COMBINED" \
    --definitions-augmentation-path "$COMBINED_DEFINITIONS" \
    --seed "$TRIPLET_SEED"

# All individual and stochastic-combination variants together.
run_condition "generation + all combined" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$ALL_COMBINED_OUTPUT" \
    --augmentation-path "$DROPOUT" \
    --augmentation-path "$MASK" \
    --augmentation-path "$SHUFFLING" \
    --augmentation-path "$TRANSLATION" \
    --augmentation-path "$COMBINED" \
    --definitions-augmentation-path "$DROPOUT_DEFINITIONS" \
    --definitions-augmentation-path "$MASK_DEFINITIONS" \
    --definitions-augmentation-path "$SHUFFLING_DEFINITIONS" \
    --definitions-augmentation-path "$TRANSLATION_DEFINITIONS" \
    --definitions-augmentation-path "$COMBINED_DEFINITIONS" \
    --seed "$TRIPLET_SEED"

echo
echo "All triplet datasets were generated in: $OUTPUT_DIR"
