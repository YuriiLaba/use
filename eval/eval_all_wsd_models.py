"""
Evaluate multiple Hugging Face models on the Ukrainian WSD benchmark.

Run from the repository root:

    python -m eval.eval_all_wsd_models

The evaluator is CPU-first for macOS. Individual model failures are recorded
in the output CSV and do not stop the remaining evaluations.
"""

import argparse
import gc
import logging
import os
import shutil
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import scan_cache_dir

from eval.eval_wsd import evaluate_wsd
from services.config import HOMONYM_BENCHMARK_PATH, get_list, get_value


logger = logging.getLogger(__name__)


MODELS = get_list("baseline_models", "wsd")


def delete_model_artifacts(model_name: str) -> bool:
    """Delete a local model directory or the matching Hugging Face cache entry."""
    local_path = Path(model_name)
    if local_path.exists():
        if local_path.is_dir():
            shutil.rmtree(local_path)
        else:
            local_path.unlink()
        logger.info("Deleted local model: %s", local_path)
        return True

    cache_dir = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    cache_info = scan_cache_dir(cache_dir=cache_dir)
    matching_revisions = [
        revision.commit_hash
        for repo in cache_info.repos
        if repo.repo_type == "model" and repo.repo_id == model_name
        for revision in repo.revisions
    ]

    if not matching_revisions:
        logger.warning("No cached files found for model: %s", model_name)
        return False

    strategy = cache_info.delete_revisions(*matching_revisions)
    logger.info(
        "Deleting cached model: %s | expected freed space: %s",
        model_name,
        strategy.expected_freed_size_str,
    )
    strategy.execute()
    return True


def evaluate_models(
    models, benchmark_path, device, output_path, delete_after_evaluation=False,
    anchor_pooling="target",
):
    if anchor_pooling not in ("sentence", "target"):
        raise ValueError("anchor_pooling must be 'sentence' or 'target'")
    results = []
    evaluation_started = time.perf_counter()

    logger.info("Starting WSD benchmark evaluation")
    logger.info("Benchmark: %s", benchmark_path)
    logger.info("Device: %s", device)
    logger.info("Anchor pooling: %s", anchor_pooling)
    logger.info("Models to evaluate: %d", len(models))

    for index, model_name in enumerate(models, start=1):
        logger.info("[%d/%d] Starting model: %s", index, len(models), model_name)
        started = time.perf_counter()

        result = {
            "model": model_name,
            "anchor_pooling": anchor_pooling,
            "accuracy": None,
            "status": "failed",
            "duration_seconds": None,
            "error": None,
            "deleted_after_evaluation": False,
        }

        try:
            accuracy = evaluate_wsd(
                model_path=model_name,
                model_tokenizer_path=model_name,
                verbose=False,
                benchmark_path=benchmark_path,
                device=device,
                anchor_pooling=anchor_pooling,
            )
            result["accuracy"] = float(accuracy)
            result["status"] = "ok"
            logger.info(
                "[%d/%d] Finished model: %s | accuracy=%.6f",
                index,
                len(models),
                model_name,
                accuracy,
            )
        except Exception as error:  # continue with the remaining models
            result["error"] = f"{type(error).__name__}: {error}"
            logger.exception(
                "[%d/%d] Failed model: %s | %s",
                index,
                len(models),
                model_name,
                result["error"],
            )
        finally:
            result["duration_seconds"] = round(time.perf_counter() - started, 2)
            results.append(result)

            logger.info(
                "[%d/%d] Duration: %.2f seconds",
                index,
                len(models),
                result["duration_seconds"],
            )

            # Release references and reduce memory pressure between models.
            gc.collect()

            if delete_after_evaluation and result["status"] == "ok":
                try:
                    result["deleted_after_evaluation"] = delete_model_artifacts(model_name)
                except Exception:
                    logger.exception("Could not delete model artifacts: %s", model_name)
            elif delete_after_evaluation:
                logger.warning(
                    "Keeping model because evaluation failed: %s", model_name
                )

    result_df = pd.DataFrame(results).sort_values(
        by=["status", "accuracy"],
        ascending=[True, False],
        na_position="last",
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    total_duration = time.perf_counter() - evaluation_started
    logger.info("Saved results to %s", output_path)
    logger.info(
        "Evaluation complete | total time=%.2f seconds (%.2f minutes)",
        total_duration,
        total_duration / 60,
    )
    print("\nFinal results:")
    print(result_df.to_string(index=False))
    print(
        f"\nTotal evaluation time: {total_duration:.2f} seconds "
        f"({total_duration / 60:.2f} minutes)"
    )
    return result_df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate multiple models on the Ukrainian WSD benchmark."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Model IDs or local model paths.",
    )
    parser.add_argument(
        "--benchmark-path",
        default=HOMONYM_BENCHMARK_PATH,
        help="Path to the benchmark JSONL file.",
    )
    parser.add_argument(
        "--device",
        default=get_value("evaluation", "device", "cpu"),
        help="Inference device. Defaults to the value in project_config.ini.",
    )
    parser.add_argument(
        "--output",
        default=get_value("evaluation", "wsd_output", "wsd_model_results.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--anchor-pooling", choices=("sentence", "target"), default="target",
        help="Pooling for benchmark sentences; definitions always use full-sentence mean pooling. Default: target.",
    )
    parser.add_argument(
        "--delete-model-after-evaluation",
        action="store_true",
        help="Delete the local model directory or Hugging Face cache entry after successful evaluation.",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_path}")

    evaluate_models(
        models=args.models,
        benchmark_path=str(benchmark_path),
        device=args.device,
        output_path=args.output,
        delete_after_evaluation=args.delete_model_after_evaluation,
        anchor_pooling=args.anchor_pooling,
    )


if __name__ == "__main__":
    main()
