#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
MODULE="local_datasets.semi_supervised_2.form_triplets"
OUTPUT_DIR="local_datasets/semi_supervised_2/triplets"

COLLECTED_DATASET="local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet_filtered.json"
GENERATED_DATASET="local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json"

DROPOUT="local_datasets/augmented/dropout/augmented_sentences.jsonl"
DROPOUT_DEFINITIONS="local_datasets/augmented/dropout/augmented_sentences_definitions.jsonl"
MASK="local_datasets/augmented/mask/augmented_sentences.jsonl"
MASK_DEFINITIONS="local_datasets/augmented/mask/augmented_sentences_definitions.jsonl"
SHUFFLING="local_datasets/augmented/token_shuffling/augmented_sentences.jsonl"
SHUFFLING_DEFINITIONS="local_datasets/augmented/token_shuffling/augmented_sentences_definitions.jsonl"
TRANSLATION="local_datasets/augmented/translation/augmented_sentences_translated_v3.jsonl"
TRANSLATION_DEFINITIONS="local_datasets/augmented/translation/augmented_sentences_translated_definitions.jsonl"
COMBINED="local_datasets/augmented/all_together/augmented_sentences_3.jsonl"
COMBINED_DEFINITIONS="local_datasets/augmented/all_together/augmented_sentences_definitions_3.jsonl"

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
    --output-csv "$OUTPUT_DIR/triplets_natural.csv" \
    --no-augmented \
    --no-definitions-augmented \
    --seed 42

# Generation: collected examples plus generated examples, without augmentation.
require_file "$GENERATED_DATASET"
run_condition "generation" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$OUTPUT_DIR/triplets_generation.csv" \
    --no-augmented \
    --no-definitions-augmented \
    --seed 42

# Generation plus one individual augmentation technique.
run_condition "generation + dropout" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$OUTPUT_DIR/triplets_generation_dropout.csv" \
    --augmentation-path "$DROPOUT" \
    --definitions-augmentation-path "$DROPOUT_DEFINITIONS" \
    --seed 42

run_condition "generation + masking" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$OUTPUT_DIR/triplets_generation_mask.csv" \
    --augmentation-path "$MASK" \
    --definitions-augmentation-path "$MASK_DEFINITIONS" \
    --seed 42

run_condition "generation + token shuffling" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$OUTPUT_DIR/triplets_generation_token_shuffling.csv" \
    --augmentation-path "$SHUFFLING" \
    --definitions-augmentation-path "$SHUFFLING_DEFINITIONS" \
    --seed 42

run_condition "generation + back translation" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$OUTPUT_DIR/triplets_generation_translation.csv" \
    --augmentation-path "$TRANSLATION" \
    --definitions-augmentation-path "$TRANSLATION_DEFINITIONS" \
    --seed 42

# Stochastic combination produced by augment_all_together.py.
run_condition "generation + stochastic combination" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$OUTPUT_DIR/triplets_generation_stochastic_combination.csv" \
    --augmentation-path "$COMBINED" \
    --definitions-augmentation-path "$COMBINED_DEFINITIONS" \
    --seed 42

# All individual and stochastic-combination variants together.
run_condition "generation + all combined" \
    --dataset-path "$GENERATED_DATASET" \
    --output-csv "$OUTPUT_DIR/triplets_generation_all_combined.csv" \
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
    --seed 42

echo
echo "All triplet datasets were generated in: $OUTPUT_DIR"
