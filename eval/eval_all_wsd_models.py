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
import time
from pathlib import Path

import pandas as pd

from eval.eval_wsd import evaluate_wsd
from services.config import HOMONYM_BENCHMARK_PATH


logger = logging.getLogger(__name__)


MODELS = [
    "lang-uk/electra-base-ukrainian-cased-discriminator",
    "ukr-models/xlm-roberta-base-uk",
    "google-bert/bert-base-multilingual-cased",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "benjamin/roberta-large-wechsel-ukrainian",
    "FacebookAI/xlm-roberta-base",
    "Goader/modern-liberta-large",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "lang-uk/ukr-paraphrase-multilingual-mpnet-base",
    "Qwen/Qwen3-Embedding-0.6B",
    "FacebookAI/xlm-roberta-large",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct",
]


def evaluate_models(models, benchmark_path, device, output_path):
    results = []
    evaluation_started = time.perf_counter()

    logger.info("Starting WSD benchmark evaluation")
    logger.info("Benchmark: %s", benchmark_path)
    logger.info("Device: %s", device)
    logger.info("Models to evaluate: %d", len(models))

    for index, model_name in enumerate(models, start=1):
        logger.info("[%d/%d] Starting model: %s", index, len(models), model_name)
        started = time.perf_counter()

        result = {
            "model": model_name,
            "accuracy": None,
            "status": "failed",
            "duration_seconds": None,
            "error": None,
        }

        try:
            accuracy = evaluate_wsd(
                model_path=model_name,
                model_tokenizer_path=model_name,
                verbose=False,
                benchmark_path=benchmark_path,
                device=device,
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

    result_df = pd.DataFrame(results).sort_values(
        by=["status", "accuracy"],
        ascending=[True, False],
        na_position="last",
    )
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
        default="cpu",
        help="Inference device. Defaults to cpu for macOS.",
    )
    parser.add_argument(
        "--output",
        default="wsd_model_results.csv",
        help="Output CSV path.",
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
    )


if __name__ == "__main__":
    main()
