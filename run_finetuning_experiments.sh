#!/usr/bin/env bash

# Run the eight non-baseline configurations from the paper table:
# 8 configurations x 2 pooling modes x 3 training seeds = 48 runs.
#
# Each GPU runs one independent training/evaluation job. With four RTX 6000
# GPUs, four jobs run concurrently. The default batch size remains 104 to match
# the paper; use --batch-size 208 for a throughput-oriented run on 48 GB cards.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG="services/trainer/fine_tuning_config.ini"
MODEL_ROOT="models/fine-tuned-models"
RESULTS_ROOT="results/finetuning"
BENCHMARK="datasets_pre_defined/ukrainian_wsd_benchmark.jsonl"
GPU_LIST="0,1,2,3"
BATCH_SIZE="104"
SEEDS="42,123,456"
SPLIT_SEED="42"
MTEB_NUM_PROC="1"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-}"
WANDB_PROJECT="${WANDB_PROJECT:-ucu-wsd-finetuning}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
HF_PRIVATE=0
NO_HF=0
NO_WANDB=0
DELETE_LOCAL=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            sed -n '2,12p' "$0"
            echo
            echo "Usage: $0 --hf-repo-prefix USERNAME/REPOSITORY [options]"
            echo "  --gpus 0,1,2,3         GPUs used for concurrent jobs"
            echo "  --batch-size 104       Per-GPU batch size (paper default)"
            echo "  --seeds 42,123,456     Training seeds"
            echo "  --split-seed 42        Fixed group split seed"
            echo "  --no-hf                 Do not upload to Hugging Face"
            echo "  --no-wandb              Do not log evaluation metrics to W&B"
            echo "  --delete-local-model    Delete checkpoints after successful HF upload"
            echo "  --force                 Re-run completed experiments"
            exit 0
            ;;
        --config)             CONFIG="$2"; shift 2 ;;
        --gpus)               GPU_LIST="$2"; shift 2 ;;
        --batch-size)         BATCH_SIZE="$2"; shift 2 ;;
        --seeds)              SEEDS="$2"; shift 2 ;;
        --split-seed)         SPLIT_SEED="$2"; shift 2 ;;
        --mteb-num-proc)      MTEB_NUM_PROC="$2"; shift 2 ;;
        --hf-repo-prefix)     HF_REPO_PREFIX="$2"; shift 2 ;;
        --wandb-project)      WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-entity)       WANDB_ENTITY="$2"; shift 2 ;;
        --hf-private)         HF_PRIVATE=1; shift ;;
        --no-hf)              NO_HF=1; shift ;;
        --no-wandb)           NO_WANDB=1; shift ;;
        --delete-local-model) DELETE_LOCAL=1; shift ;;
        --force)              FORCE=1; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi

if [[ "$NO_HF" -eq 0 && -z "$HF_REPO_PREFIX" ]]; then
    echo "Provide --hf-repo-prefix USERNAME/REPOSITORY, or use --no-hf." >&2
    exit 2
fi

if [[ "$DELETE_LOCAL" -eq 1 && "$NO_HF" -eq 1 ]]; then
    echo "--delete-local-model requires Hugging Face uploads; remove --no-hf." >&2
    exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
IFS=',' read -r -a TRAIN_SEEDS <<< "$SEEDS"

CONDITION_NAMES=(
    natural
    generation
    generation_mlm
    generation_dropout
    generation_back_translation
    generation_shuffling
    generation_stochastic
    generation_all_combined
)

CONDITION_FILES=(
    "local_datasets/semi_supervised_2/triplets/triplets_natural.csv"
    "local_datasets/semi_supervised_2/triplets/triplets_generation.csv"
    "local_datasets/semi_supervised_2/triplets/triplets_generation_mask.csv"
    "local_datasets/semi_supervised_2/triplets/triplets_generation_dropout.csv"
    "local_datasets/semi_supervised_2/triplets/triplets_generation_translation.csv"
    "local_datasets/semi_supervised_2/triplets/triplets_generation_token_shuffling.csv"
    "local_datasets/semi_supervised_2/triplets/triplets_generation_stochastic_combination.csv"
    "local_datasets/semi_supervised_2/triplets/triplets_generation_all_combined.csv"
)

for dataset in "${CONDITION_FILES[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Required triplet dataset not found: $dataset" >&2
        echo "Run ./generate_all_triplets.sh first." >&2
        exit 1
    fi
done

if [[ ! -f "$BENCHMARK" ]]; then
    echo "Benchmark not found: $BENCHMARK" >&2
    exit 1
fi

repo_id_for() {
    local experiment_name="$1"
    if [[ "$HF_REPO_PREFIX" == */* ]]; then
        local namespace="${HF_REPO_PREFIX%%/*}"
        local repository="${HF_REPO_PREFIX#*/}"
        echo "${namespace}/${repository}-${experiment_name}"
    else
        echo "${HF_REPO_PREFIX}-${experiment_name}"
    fi
}

run_one() {
    local gpu="$1"
    local condition="$2"
    local dataset="$3"
    local pool_targets="$4"
    local seed="$5"

    local experiment_name="${condition}_pt-${pool_targets}_seed-${seed}"
    local model_path="${MODEL_ROOT}/${experiment_name}_final"
    local result_dir="${RESULTS_ROOT}/${experiment_name}"
    local metrics_path="${result_dir}/metrics.json"
    local repo_id=""
    local retry_upload=0

    if [[ "$NO_HF" -eq 0 ]]; then
        repo_id="$(repo_id_for "$experiment_name")"
    fi

    if [[ "$FORCE" -eq 0 && -f "$metrics_path" ]]; then
        if [[ "$NO_HF" -eq 1 || "$(grep -c '"huggingface_url"' "$metrics_path" || true)" -gt 0 ]]; then
            echo "[GPU $gpu] Skipping completed run: $experiment_name"
            "$PYTHON" -m scripts.report_finetuning_status \
                --results-dir "$RESULTS_ROOT" \
                --total-runs "$TOTAL_RUNS" \
                --current "$experiment_name" \
                --event finished
            return 0
        elif [[ -f "$model_path/config.json" ]]; then
            retry_upload=1
            echo "[GPU $gpu] Retrying evaluation/upload without retraining: $experiment_name"
        fi
    fi

    echo
    echo "============================================================"
    echo "[GPU $gpu] Starting: $experiment_name"
    echo "[GPU $gpu] Dataset:  $dataset"
    echo "[GPU $gpu] Batch:    $BATCH_SIZE"
    echo "============================================================"

    "$PYTHON" -m scripts.report_finetuning_status \
        --results-dir "$RESULTS_ROOT" \
        --total-runs "$TOTAL_RUNS" \
        --current "$experiment_name" \
        --event started

    if [[ "$retry_upload" -eq 0 ]]; then
        local train_status=0
        local train_args=(
            "$PYTHON" -m services.trainer.trainer
            --config "$CONFIG"
            --device "cuda:${gpu}"
            --train-data "$dataset"
            --output-dir "$MODEL_ROOT"
            --model-name "$experiment_name"
            --run-name "$experiment_name"
            --pool-targets "$pool_targets"
            --seed "$seed"
            --split-seed "$SPLIT_SEED"
            --batch-size "$BATCH_SIZE"
            --wandb-project "$WANDB_PROJECT"
            --skip-final-wsd-eval
        )
        if [[ -n "$WANDB_ENTITY" ]]; then
            train_args+=(--wandb-entity "$WANDB_ENTITY")
        fi
        "${train_args[@]}" || train_status=$?

        if [[ "$train_status" -ne 0 || ! -f "$model_path/config.json" ]]; then
            echo "[GPU $gpu] Training failed or model was not saved: $experiment_name" >&2
            return 1
        fi
    fi

    local eval_args=(
        "$PYTHON" -m scripts.evaluate_finetuned_model
        --model-path "$model_path"
        --experiment-name "$experiment_name"
        --train-data "$dataset"
        --pool-targets "$pool_targets"
        --seed "$seed"
        --split-seed "$SPLIT_SEED"
        --benchmark-path "$BENCHMARK"
        --device "cuda:${gpu}"
        --results-dir "$result_dir"
        --mteb-num-proc "$MTEB_NUM_PROC"
        --wandb-project "$WANDB_PROJECT"
    )

    if [[ -n "$WANDB_ENTITY" ]]; then
        eval_args+=(--wandb-entity "$WANDB_ENTITY")
    fi

    if [[ "$NO_WANDB" -eq 1 ]]; then
        eval_args+=(--no-wandb)
    fi
    if [[ "$NO_HF" -eq 0 ]]; then
        eval_args+=(--hf-repo-id "$repo_id")
        if [[ "$DELETE_LOCAL" -eq 1 ]]; then
            eval_args+=(--delete-local-model)
        fi
        if [[ "$HF_PRIVATE" -eq 1 ]]; then
            eval_args+=(--hf-private)
        fi
    fi

    "${eval_args[@]}"
    echo "[GPU $gpu] Finished: $experiment_name"
    "$PYTHON" -m scripts.report_finetuning_status \
        --results-dir "$RESULTS_ROOT" \
        --total-runs "$TOTAL_RUNS" \
        --current "$experiment_name" \
        --event finished
}

mkdir -p "$MODEL_ROOT" "$RESULTS_ROOT"

PIDS=()
GPU_INDEX=0
TOTAL_RUNS=$(( ${#CONDITION_NAMES[@]} * 2 * ${#TRAIN_SEEDS[@]} ))
RUN_NUMBER=0

echo "Planned runs: $TOTAL_RUNS"
echo "GPUs: ${GPUS[*]}"
echo "Batch size: $BATCH_SIZE"

for index in "${!CONDITION_NAMES[@]}"; do
    condition="${CONDITION_NAMES[$index]}"
    dataset="${CONDITION_FILES[$index]}"

    for pool_targets in false true; do
        for seed in "${TRAIN_SEEDS[@]}"; do
            gpu="${GPUS[$GPU_INDEX]}"
            run_one "$gpu" "$condition" "$dataset" "$pool_targets" "$seed" &
            PIDS+=("$!")
            RUN_NUMBER=$((RUN_NUMBER + 1))
            GPU_INDEX=$(( (GPU_INDEX + 1) % ${#GPUS[@]} ))

            if [[ "${#PIDS[@]}" -ge "${#GPUS[@]}" ]]; then
                for pid in "${PIDS[@]}"; do
                    wait "$pid"
                done
                PIDS=()
                echo "Completed batch of ${#GPUS[@]} runs ($RUN_NUMBER/$TOTAL_RUNS planned)."
            fi
        done
    done
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

"$PYTHON" scripts/aggregate_finetuning_results.py \
    --results-dir "$RESULTS_ROOT" \
    --output results/finetuning_summary.csv

echo
echo "All fine-tuning experiments completed."
echo "Local models: $MODEL_ROOT"
echo "Local metrics: $RESULTS_ROOT"
echo "Summary: results/finetuning_summary.csv"
