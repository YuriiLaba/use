#!/usr/bin/env python3
"""Evaluate one fine-tuned model and publish its metrics.

The evaluator writes results locally, logs scalar metrics to Weights & Biases,
and optionally uploads the model directory (including its evaluation files) to
the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import evaluation

from eval.eval_wsd import evaluate_wsd
from services.config import HOMONYM_BENCHMARK_PATH

# Keep MTEB's cache inside the project so evaluation does not depend on a
# writable user-home directory (important on managed servers and containers).
os.environ.setdefault("MTEB_CACHE", str(Path("cache/mteb").resolve()))

import mteb
from mteb.cache import ResultCache


LOGGER = logging.getLogger(__name__)
STS_DATASET = "anikol12/STSB-UK"
# Exact MTEB task definitions reported in the paper.  In particular, the
# paper uses the original seven-label SIB200 task, not SIB200.v2.
MTEB_TASKS = [
    # Classification
    "SIB200Classification",
    "UkrFormalityClassification",
    # Clustering
    "SIB200ClusteringS2S",
    # Bitext mining
    "WebFAQBitextMiningQAs",
    "WebFAQBitextMiningQuestions",
    "NTREXBitextMining",
    "BibleNLPBitextMining",
    "FloresBitextMining",
    "Tatoeba",
    # Retrieval
    "BelebeleRetrieval",
    "WebFAQRetrieval",
]
MTEB_MODALITIES = ["text"]

# Restrict multilingual tasks to the Ukrainian subset used by the paper.
# Without this mapping, MTEB loads every language configuration in a task;
# some of those configurations are not present in the current HF snapshot
# and produce errors such as ``BuilderConfig 'nya_Latn' not found``.
MTEB_SUBSETS = {
    "SIB200Classification": ["ukr_Cyrl"],
    "UkrFormalityClassification": ["default"],
    "SIB200ClusteringS2S": ["ukr_Cyrl"],
    "WebFAQBitextMiningQAs": ["eng-ukr"],
    "WebFAQBitextMiningQuestions": ["eng-ukr"],
    "NTREXBitextMining": ["eng_Latn-ukr_Cyrl"],
    "BibleNLPBitextMining": ["eng_Latn-ukr_Cyrl"],
    "FloresBitextMining": ["eng_Latn-ukr_Cyrl"],
    "Tatoeba": ["ukr-eng"],
    "BelebeleRetrieval": ["ukr_Cyrl-ukr_Cyrl"],
    "WebFAQRetrieval": ["ukr"],
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=True)


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested MTEB scores into W&B-compatible scalar keys."""
    flattened: dict[str, float] = {}

    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(flatten_numeric(nested_value, nested_prefix))
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            flattened.update(flatten_numeric(nested_value, f"{prefix}/{index}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            flattened[prefix] = numeric_value

    return flattened


def evaluate_sts(model_path: str, device: str) -> dict[str, float]:
    dataset = load_dataset(STS_DATASET, split="train")
    sentences1 = dataset["sentence1"]
    sentences2 = dataset["sentence2"]
    scores = [
        1.0 if sentence1 == sentence2 else score
        for sentence1, sentence2, score in zip(
            sentences1, sentences2, dataset["score"]
        )
    ]

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1,
        sentences2,
        scores,
        show_progress_bar=True,
    )

    model = SentenceTransformer(model_path, device=device)
    raw_results = evaluator(model)
    del model
    gc.collect()
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "pearson_cosine": float(raw_results["pearson_cosine"]),
        "spearman_cosine": float(raw_results["spearman_cosine"]),
    }


def evaluate_mteb(
    model_path: str,
    results_dir: Path,
    device: str,
    num_proc: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    tasks = mteb.get_tasks(tasks=MTEB_TASKS)
    tasks = [
        task
        for task in tasks
        if task.metadata.modalities == MTEB_MODALITIES
    ]
    for task in tasks:
        requested_subsets = MTEB_SUBSETS[task.metadata.name]
        available_subsets = set(task.hf_subsets)
        missing_subsets = set(requested_subsets) - available_subsets
        if missing_subsets:
            raise ValueError(
                f"MTEB task {task.metadata.name} does not provide subset(s): "
                f"{sorted(missing_subsets)}. Available subsets include: "
                f"{sorted(available_subsets)[:10]}"
            )
        task.hf_subsets = requested_subsets

    LOGGER.info(
        "Evaluating %d paper-specified MTEB tasks on Ukrainian subsets",
        len(tasks),
    )

    model = SentenceTransformer(model_path, device=device)
    safe_model = safe_name(Path(model_path).name)
    cache = ResultCache(f"cache/mteb_{safe_model}")
    prediction_folder = results_dir / "mteb_predictions"
    mteb_results = mteb.evaluate(
        model,
        tasks=tasks,
        cache=cache,
        num_proc=num_proc,
        prediction_folder=str(prediction_folder),
    )

    raw_results: dict[str, Any] = {}
    wandb_metrics: dict[str, float] = {}
    main_scores: list[float] = []
    mteb_results_dir = results_dir / "mteb_results"

    for task_result in mteb_results.task_results:
        task_name = task_result.task_name
        scores = task_result.scores
        raw_results[task_name] = scores
        write_json(mteb_results_dir / f"{safe_name(task_name)}.json", scores)

        flattened = flatten_numeric(scores, f"mteb/{task_name}")
        wandb_metrics.update(flattened)

        for key, value in flatten_numeric(scores).items():
            if key.endswith("main_score"):
                main_scores.append(value)

    if main_scores:
        wandb_metrics["mteb/mean_main_score"] = sum(main_scores) / len(main_scores)

    del model
    gc.collect()
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()

    return raw_results, wandb_metrics


def build_model_card(
    model_dir: Path,
    experiment_name: str,
    train_data: str,
    pool_targets: bool,
    seed: int,
    split_seed: int,
    metrics: dict[str, Any],
) -> None:
    wsd = metrics.get("wsd", {}).get("accuracy")
    sts = metrics.get("sts", {})
    lines = [
        f"# {experiment_name}",
        "",
        "Fine-tuned Ukrainian word-sense-disambiguation model.",
        "",
        "## Training configuration",
        "",
        f"- Training data: `{train_data}`",
        f"- Target-token pooling: `{pool_targets}`",
        f"- Training seed: `{seed}`",
        f"- Validation split seed: `{split_seed}`",
        "- Base model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`",
        "",
        "## Evaluation",
        "",
        f"- WSD accuracy: `{wsd}`",
        f"- STS Pearson: `{sts.get('pearson_cosine')}`",
        f"- STS Spearman: `{sts.get('spearman_cosine')}`",
        "- Full MTEB task-level results are in `evaluation/mteb_results/`.",
        "",
    ]
    (model_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def repo_id_from_prefix(prefix: str, experiment_name: str) -> str:
    prefix = prefix.rstrip("/")
    if "/" in prefix:
        namespace, name = prefix.split("/", 1)
        return f"{namespace}/{name}-{experiment_name}"
    return f"{prefix}-{experiment_name}"


def upload_to_huggingface(model_dir: Path, repo_id: str, private: bool) -> str:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned model and evaluation results",
    )
    return f"https://huggingface.co/{repo_id}"


def delete_local_checkpoints(model_dir: Path) -> None:
    """Delete the final and best checkpoints after a successful upload."""
    checkpoint_dirs = [model_dir]
    if model_dir.name.endswith("_final"):
        checkpoint_dirs.append(
            model_dir.parent / f"{model_dir.name[:-len('_final')]}_best"
        )

    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            LOGGER.info("Deleted local checkpoint: %s", checkpoint_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--pool-targets", type=lambda value: value.lower() == "true", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--benchmark-path", default=HOMONYM_BENCHMARK_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--mteb-num-proc", type=int, default=1)
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "ucu-wsd-finetuning"),
    )
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument(
        "--delete-local-model",
        action="store_true",
        help="Delete final and best local checkpoints after successful Hugging Face upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.delete_local_model and not args.hf_repo_id:
        raise ValueError("--delete-local-model requires --hf-repo-id")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    model_dir = Path(args.model_path)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = model_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    metrics: dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "model_path": str(model_dir),
        "train_data": args.train_data,
        "pool_targets": args.pool_targets,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "benchmark_path": args.benchmark_path,
        "device": args.device,
    }

    LOGGER.info("Evaluating %s", args.experiment_name)
    LOGGER.info("Model path: %s", model_dir)

    try:
        wsd_accuracy = evaluate_wsd(
            model_path=str(model_dir),
            model_tokenizer_path=str(model_dir),
            verbose=False,
            benchmark_path=args.benchmark_path,
            device=args.device,
        )
        metrics["wsd"] = {"accuracy": float(wsd_accuracy), "status": "ok"}
    except Exception as error:
        LOGGER.exception("WSD evaluation failed")
        metrics["wsd"] = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }

    try:
        sts_metrics = evaluate_sts(str(model_dir), args.device)
        metrics["sts"] = {**sts_metrics, "status": "ok"}
    except Exception as error:
        LOGGER.exception("STS evaluation failed")
        metrics["sts"] = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }

    try:
        mteb_results, mteb_metrics = evaluate_mteb(
            str(model_dir),
            results_dir,
            args.device,
            args.mteb_num_proc,
        )
        metrics["mteb"] = {"status": "ok", "tasks": mteb_results}
        metrics["mteb_metrics"] = mteb_metrics
    except Exception as error:
        LOGGER.exception("MTEB evaluation failed")
        mteb_metrics = {}
        metrics["mteb"] = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }

    metrics["duration_seconds"] = round(time.perf_counter() - started, 2)
    write_json(results_dir / "metrics.json", metrics)
    write_json(evaluation_dir / "metrics.json", metrics)
    build_model_card(
        model_dir=model_dir,
        experiment_name=args.experiment_name,
        train_data=args.train_data,
        pool_targets=args.pool_targets,
        seed=args.seed,
        split_seed=args.split_seed,
        metrics=metrics,
    )

    wandb_metrics: dict[str, float] = {}
    if metrics.get("wsd", {}).get("status") == "ok":
        wandb_metrics["eval/wsd_accuracy"] = metrics["wsd"]["accuracy"]
    if metrics.get("sts", {}).get("status") == "ok":
        wandb_metrics.update(
            {
                "eval/sts_pearson_cosine": metrics["sts"]["pearson_cosine"],
                "eval/sts_spearman_cosine": metrics["sts"]["spearman_cosine"],
            }
        )
    wandb_metrics.update(mteb_metrics)

    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.experiment_name}-evaluation",
            group="fine-tuning-evaluation",
            config={
                "experiment_name": args.experiment_name,
                "train_data": args.train_data,
                "pool_targets": args.pool_targets,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "model_path": str(model_dir),
            },
        )
        run.log(wandb_metrics)
        run.finish()

    if args.hf_repo_id:
        hub_url = f"https://huggingface.co/{args.hf_repo_id}"
        metrics["huggingface_url"] = hub_url
        write_json(results_dir / "metrics.json", metrics)
        write_json(evaluation_dir / "metrics.json", metrics)
        build_model_card(
            model_dir=model_dir,
            experiment_name=args.experiment_name,
            train_data=args.train_data,
            pool_targets=args.pool_targets,
            seed=args.seed,
            split_seed=args.split_seed,
            metrics=metrics,
        )
        hub_url = upload_to_huggingface(
            model_dir=model_dir,
            repo_id=args.hf_repo_id,
            private=args.hf_private,
        )
        write_json(results_dir / "metrics.json", metrics)
        write_json(evaluation_dir / "metrics.json", metrics)
        LOGGER.info("Uploaded model to %s", hub_url)
        if args.delete_local_model:
            delete_local_checkpoints(model_dir)

    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
