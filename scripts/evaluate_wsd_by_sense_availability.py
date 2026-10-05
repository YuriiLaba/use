"""WSD-only analysis by pre-augmentation natural evidence, without retraining.

Keeps maximum-over-examples prediction and target-token inference unchanged.
Training PT is metadata, not an inference switch. The paper tables omit counts;
bucket sizes/exclusions and every prediction remain available for auditing.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import statistics
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
BUCKETS = ("0", "1-4", "5-19", ">=20")
LABELS = {
    "natural": "Natural",
    "generation": "Natural + Generation",
    "generation_mlm": "Natural + Generation + MLM",
    "generation_dropout": "Natural + Generation + Word deletion",
    "generation_back_translation": "Natural + Generation + Back-translation",
    "generation_shuffling": "Natural + Generation + Shuffling",
    "generation_stochastic": "Natural + Generation + Stochastic combination",
    "generation_all_combined": "Natural + Generation + Pooled augmentations",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bucket_for(count: int) -> str:
    if count < 0:
        raise ValueError("Natural sentence counts cannot be negative")
    return "0" if count == 0 else "1-4" if count < 5 else "5-19" if count < 20 else ">=20"


def sense_key(lemma: str, gloss) -> tuple[str, tuple[str, ...]]:
    gloss = [gloss] if isinstance(gloss, str) else gloss
    if not gloss or not all(isinstance(text, str) and text for text in gloss):
        raise ValueError(f"Invalid dictionary definition for {lemma!r}")
    # Match dictionary text exactly; do not conflate definitions by fuzzy matching.
    return lemma.strip().lower(), tuple(gloss)


def natural_counts(grouped: dict) -> dict:
    counts = {}
    for lemma, meanings in grouped.items():
        for meaning in meanings.values():
            key = sense_key(lemma, meaning["meaning"]["gloss"])
            if key in counts:
                raise ValueError(f"Duplicate natural definition: {key}")
            sentences = set()
            for record in meaning.get("sentences", []):
                if record.get("source", "natural") not in ("natural", "ubertext"):
                    raise ValueError("Use the filtered NATURAL pool, not merged/generated data")
                sentence = record["sentence"]
                if not isinstance(sentence, str) or not sentence.strip():
                    raise ValueError(f"Invalid natural sentence for {key}")
                sentences.add(sentence)
            counts[key] = len(sentences)
    return counts


def build_records(benchmark, counts: dict) -> list[dict]:
    records, seen = [], set()
    for _, row in benchmark.iterrows():
        key = sense_key(row["lemma"], row["gloss"])
        if key not in counts:
            raise ValueError(f"Benchmark definition is absent from the natural inventory: {key}")
        if key in seen:
            raise ValueError(f"Duplicate benchmark definition: {key}")
        seen.add(key)
        record_id = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()
        records.append({
            "record_id": record_id, "lemma": key[0], "gloss": list(key[1]),
            "natural_examples": counts[key], "bucket": bucket_for(counts[key]),
            "benchmark_examples": len(row["examples"]),
        })
    if not records:
        raise ValueError("The benchmark contains no valid records")
    return records


def attach_predictions(predictions, records: list[dict]) -> list[dict]:
    if len(predictions) != len(records):
        raise ValueError("Evaluation and benchmark record counts differ")
    output = []
    for (_, row), record in zip(predictions.iterrows(), records):
        if sense_key(row["lemma"], row["gloss"]) != sense_key(record["lemma"], record["gloss"]):
            raise ValueError("Evaluation record order or definition inventory changed")
        predicted = row["predicted_context"]
        scored = isinstance(predicted, (list, tuple)) and bool(predicted)
        output.append({
            **record, "predicted_gloss": list(predicted) if scored else None,
            "scored": scored,
            # Identical to prediction_accuracy: compare the first gloss text.
            "correct": predicted[0] == record["gloss"][0] if scored else None,
        })
    return output


def checkpoint_source(metrics: dict, models_dir: Path, experiment: str, prefix: str) -> dict:
    candidates = [Path(metrics.get("model_path") or "__missing_checkpoint__"),
                  models_dir / f"{experiment}_final"]
    for path in candidates:
        if (path / "config.json").is_file() and (
            list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin"))
        ):
            return {"kind": "local", "model_path": str(path.resolve())}
    url = metrics.get("huggingface_url")
    if url:
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        if parsed.hostname != "huggingface.co" or len(parts) != 2:
            raise ValueError(f"Unsupported model URL: {url}")
        repo = "/".join(parts)
    elif prefix and prefix.count("/") == 1:
        repo = f"{prefix}-{experiment}"
    else:
        raise FileNotFoundError(f"No final local checkpoint or HF repository for {experiment}")
    return {"kind": "huggingface", "model_path": repo}


def pin_source(source: dict) -> dict:
    source = dict(source)
    if source["kind"] == "huggingface":
        from huggingface_hub import HfApi
        source["revision"] = HfApi().model_info(source["model_path"]).sha
        if not source["revision"]:
            raise ValueError("Cannot pin the Hugging Face model to a commit")
    else:
        path = Path(source["model_path"])
        files = sorted(p for p in path.iterdir() if p.is_file() and (
            p.suffix in (".json", ".safetensors", ".bin", ".txt", ".model")
        ))
        source["file_sha256"] = {p.name: sha256_file(p) for p in files}
    return source


def delete_cached_checkpoint(source: dict) -> bool:
    """Remove only the evaluated Hub revision; never delete local checkpoints."""
    if source["kind"] != "huggingface":
        LOGGER.info("Keeping local checkpoint: %s", source["model_path"])
        return False
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    repos = frozenset(repo for repo in cache.repos
                      if repo.repo_type == "model" and repo.repo_id == source["model_path"])
    revision = source["revision"]
    if not any(r.commit_hash == revision for repo in repos for r in repo.revisions):
        LOGGER.info("No cached evaluated revision found: %s @ %s", source["model_path"], revision)
        return False
    # Mirrored repositories can share commit hashes. Scope deletion to this repo.
    strategy = replace(cache, repos=repos).delete_revisions(revision)
    LOGGER.info("Deleting cached checkpoint: %s @ %s | expected freed space: %s",
                source["model_path"], revision, strategy.expected_freed_size_str)
    strategy.execute()
    LOGGER.info("Deleted cached checkpoint: %s @ %s", source["model_path"], revision)
    return True


def summarize(runs: list[dict], records: list[dict], conditions, poolings, seeds):
    """Use the intersection of valid predictions across ALL requested runs."""
    expected = {(c, p, s) for c in conditions for p in poolings for s in seeds}
    lookup = {(r["condition"], r["pool_targets"], r["seed"]): r for r in runs}
    if len(lookup) != len(runs) or set(lookup) != expected:
        raise ValueError("Incomplete or duplicate runs; refusing to aggregate a partial experiment")
    ids = {r["record_id"] for r in records}
    common = set(ids)
    for run in runs:
        if {r["record_id"] for r in run["predictions"]} != ids:
            raise ValueError("Runs use different benchmark records")
        common &= {r["record_id"] for r in run["predictions"] if r["scored"]}
    if not common:
        raise ValueError("No commonly scored benchmark records")
    sizes = [{"bucket": b, "benchmark_records": sum(r["bucket"] == b for r in records),
              "common_scored_records": sum(r["bucket"] == b and r["record_id"] in common for r in records)}
             for b in BUCKETS]
    per_run, scores = [], {}
    for (condition, pooling, seed), run in lookup.items():
        for bucket in BUCKETS:
            selected = [r for r in run["predictions"]
                        if r["bucket"] == bucket and r["record_id"] in common]
            accuracy = 100 * sum(r["correct"] for r in selected) / len(selected) if selected else None
            scores[condition, pooling, seed, bucket] = accuracy
            per_run.append({"condition": condition, "pool_targets": pooling, "seed": seed,
                            "bucket": bucket, "accuracy_percent": accuracy,
                            "scored_records": len(selected)})
    summary, gains = [], []
    def stats(values):
        if any(v is None for v in values):
            return None, None
        return statistics.mean(values), statistics.stdev(values)
    for pooling in poolings:
        for bucket in BUCKETS:
            row = {"pool_targets": pooling, "bucket": bucket}
            for condition in conditions:
                mean, std = stats([scores[condition, pooling, seed, bucket] for seed in seeds])
                row[f"{condition}_mean_accuracy_percent"] = mean
                row[f"{condition}_std_accuracy_percent"] = std
            summary.append(row)
            comparisons = [("generation", "natural")] + [
                (c, "generation") for c in conditions if c not in ("natural", "generation")
            ]
            for augmented, reference in comparisons:
                if augmented not in conditions or reference not in conditions:
                    continue
                pairs = [(scores[augmented, pooling, s, bucket], scores[reference, pooling, s, bucket])
                         for s in seeds]
                mean, std = stats([a - b if a is not None and b is not None else None for a, b in pairs])
                gains.append({"pool_targets": pooling, "bucket": bucket,
                              "comparison": f"{augmented} minus {reference}",
                              "mean_gain_pp": mean, "std_gain_pp": std})
    details = {"scoring": "Intersection of valid records across all requested checkpoints",
               "benchmark_records": len(ids), "common_scored_records": len(common),
               "excluded_from_common_comparison": len(ids - common), "buckets": sizes,
               "runs": [{"experiment": r["experiment"],
                         "unscored_records": sum(not p["scored"] for p in r["predictions"]),
                         "overall_accuracy": r["overall_accuracy"],
                         "original_overall_accuracy": r.get("original_overall_accuracy")}
                        for r in runs]}
    return summary, per_run, gains, details


def latex_tables(summary: list[dict], conditions, poolings) -> str:
    output = []
    labels = {"0": "$n_g=0$", "1-4": "$1\\leq n_g\\leq4$",
              "5-19": "$5\\leq n_g\\leq19$", ">=20": "$n_g\\geq20$"}
    for pooling in poolings:
        pt = str(pooling).lower()
        training = "target-token" if pooling else "full-sentence"
        lines = [r"\begin{table}[!htbp]", r"\centering", r"\small",
                 r"\caption{WSD accuracy (\%) by pre-augmentation natural evidence, "
                 + f"using {training} anchor pooling during fine-tuning (PT={pt}). "
                 + r"All models use target-token pooling at inference and the same scored "
                 + r"dictionary-sense records. Values are mean $\pm$ sample standard deviation "
                 + r"over training seeds.}",
                 f"\\label{{tab:wsd-natural-evidence-pt-{pt}}}",
                 r"\begin{tabularx}{\textwidth}{@{}l" + "X" * len(conditions) + "@{}}",
                 r"\toprule",
                 r"\textbf{Natural examples} & " + " & ".join(
                     f"\\textbf{{{LABELS[c]}}}" for c in conditions) + r" \\", r"\midrule"]
        for row in summary:
            if row["pool_targets"] != pooling:
                continue
            cells = []
            for condition in conditions:
                mean, std = row[f"{condition}_mean_accuracy_percent"], row[f"{condition}_std_accuracy_percent"]
                cells.append(f"${mean:.2f} \\pm {std:.2f}$" if mean is not None else "--")
            lines.append(labels[row["bucket"]] + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
        output.append("\n".join(lines))
    return "\n\n".join(output) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.getenv("PIPELINE_CONFIG", str(ROOT / "project_config.ini")))
    for flag in ("benchmark-path", "natural-path", "results-dir", "models-dir", "output-dir", "device"):
        parser.add_argument(f"--{flag}")
    parser.add_argument("--conditions", nargs="+", choices=list(LABELS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--training-poolings", nargs="+", choices=("false", "true"))
    parser.add_argument("--resume", action="store_true", help="Reuse verified predictions for unchanged inputs/checkpoints")
    parser.add_argument("--delete-model-after-evaluation", action="store_true",
                        help="Delete the evaluated Hugging Face cache revision after saving predictions; keep local checkpoints")
    parser.add_argument("--plan-only", action="store_true", help="List checkpoints without downloading or running models")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    os.environ["PIPELINE_CONFIG"] = str(config_path)
    from services.config import load_config
    config = load_config(config_path)
    defaults = {
        "benchmark_path": config["paths"]["benchmark"],
        "natural_path": config["paths"]["filtered_grouped"],
        "results_dir": config["evaluation"]["results_dir"],
        "models_dir": config["paths"]["models_dir"],
        "output_dir": config.get("bucket_analysis", "output_dir", fallback="results/wsd_sense_availability"),
        "device": config.get("bucket_analysis", "device", fallback=config["evaluation"]["device"]),
    }
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    args.conditions = args.conditions or [v.strip() for v in config.get(
        "bucket_analysis", "conditions", fallback="natural,generation,generation_back_translation").split(",")]
    args.seeds = args.seeds or [int(v.strip()) for v in config["experiments"]["seeds"].split(",")]
    selected = args.training_poolings or config.get(
        "bucket_analysis", "training_poolings", fallback="false,true").split(",")
    if any(v.strip() not in ("false", "true") for v in selected):
        parser.error("training_poolings must contain only false and/or true")
    args.poolings = [v.strip() == "true" for v in selected]
    args.repo_prefix = config.get("huggingface", "repo_prefix", fallback="")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide at least two distinct training seeds for sample standard deviation")
    if not args.conditions or not set(args.conditions) <= set(LABELS) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Provide distinct supported conditions")
    if len(set(args.poolings)) != len(args.poolings):
        parser.error("Duplicate training pooling modes")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    plan = []
    for condition in args.conditions:
        for pooling in args.poolings:
            for seed in args.seeds:
                experiment = f"{condition}_pt-{str(pooling).lower()}_seed-{seed}"
                metrics_path = Path(args.results_dir) / experiment / "metrics.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if (metrics.get("pool_targets") != pooling or metrics.get("seed") != seed
                        or metrics.get("experiment_name", experiment) != experiment):
                    raise ValueError(f"Training metadata mismatch: {metrics_path}")
                plan.append({"experiment": experiment, "condition": condition,
                             "pool_targets": pooling, "seed": seed,
                             "source": checkpoint_source(metrics, Path(args.models_dir), experiment, args.repo_prefix),
                             "original_overall_accuracy": metrics.get("wsd", {}).get("accuracy")})
    if args.plan_only:
        for run in plan:
            print(f'{run["experiment"]}: {run["source"]["model_path"]}')
        print(f"{len(plan)} WSD-only runs; inference pooling=target; output={args.output_dir}")
        return
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")], force=True)
    from tqdm.auto import tqdm
    import torch
    from services.utils_data import read_homonym_benchmark
    from eval.eval_wsd import evaluate_wsd
    from services.config import PATH_TO_SOURCE_UDPIPE
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    records = build_records(read_homonym_benchmark(args.benchmark_path), natural_counts(
        json.loads(Path(args.natural_path).read_text(encoding="utf-8"))))
    code_files = ["eval/eval_wsd.py", "services/prediction_strategies.py", "services/poolings.py",
                  "services/utils_embedding_calculation_v2.py", "services/utils_model.py",
                  "services/word_sense_detector.py", "services/utils_data.py",
                  "scripts/evaluate_wsd_by_sense_availability.py"]
    identity = {"benchmark_sha256": sha256_file(Path(args.benchmark_path)),
                "natural_sha256": sha256_file(Path(args.natural_path)),
                "udpipe_sha256": sha256_file(Path(PATH_TO_SOURCE_UDPIPE)),
                "code_sha256": {f: sha256_file(ROOT / f) for f in code_files},
                "inference_pooling": "target", "prediction_strategy": "max_sim_across_all_examples",
                "device": args.device, "torch_version": torch.__version__,
                "transformers_version": version("transformers"), "numpy_version": version("numpy")}
    write_json(output / "bucket_membership.json", records)
    completed, failures = [], []
    for index, run in enumerate(tqdm(plan, desc="Checkpoints", unit="model"), start=1):
        name = run["experiment"]
        path = output / "predictions" / f"{name}.json"
        cleanup_source = None
        LOGGER.info("[%d/%d] %s | inference pooling=target", index, len(plan), name)
        try:
            manifest = {**identity, "source": pin_source(run["source"]),
                        "experiment": name, "condition": run["condition"],
                        "pool_targets": run["pool_targets"], "seed": run["seed"]}
            if path.exists():
                if not args.resume:
                    raise FileExistsError(f"{path} exists; use --resume or a different --output-dir")
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["manifest"] != manifest:
                    raise ValueError(f"Inputs/checkpoint changed for {name}; choose a new --output-dir")
                LOGGER.info("Reusing verified predictions: %s", name)
            else:
                import time
                start = time.perf_counter()
                source = manifest["source"]
                predictions = evaluate_wsd(
                    source["model_path"], verbose=False, benchmark_path=args.benchmark_path,
                    device=args.device, anchor_pooling="target", return_predictions=True,
                    model_revision=source.get("revision"),
                )
                scored = attach_predictions(predictions, records)
                valid = [r for r in scored if r["scored"]]
                if not valid:
                    raise ValueError(f"No valid predictions for {name}")
                accuracy = sum(r["correct"] for r in valid) / len(valid)
                saved = {**run, "manifest": manifest, "predictions": scored,
                         "overall_accuracy": accuracy,
                         "duration_seconds": time.perf_counter() - start}
                write_json(path, saved)
                previous = run["original_overall_accuracy"]
                if previous is not None and not math.isclose(accuracy, previous, abs_tol=1e-9):
                    LOGGER.warning("%s differs from original overall accuracy: %.6f vs %.6f",
                                   name, accuracy, previous)
                LOGGER.info("Finished %s | accuracy=%.4f%% | excluded=%d | %.1fs",
                            name, 100 * accuracy, len(scored) - len(valid), saved["duration_seconds"])
            if saved["predictions"] != [
                {**record, "predicted_gloss": p["predicted_gloss"],
                 "scored": p["scored"], "correct": p["correct"]}
                for record, p in zip(records, saved["predictions"])
            ] or len(saved["predictions"]) != len(records):
                raise ValueError(f"Cached prediction membership mismatch: {name}")
            completed.append(saved)
            if args.delete_model_after_evaluation:
                cleanup_source = manifest["source"]
        except Exception as error:
            LOGGER.exception("Failed checkpoint %s", name)
            failures.append({"experiment": name, "error": f"{type(error).__name__}: {error}"})
        finally:
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            if cleanup_source is not None:
                try:
                    delete_cached_checkpoint(cleanup_source)
                except Exception:
                    LOGGER.exception("Cache cleanup failed for %s; saved predictions remain valid", name)
    write_json(output / "failures.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} runs failed. Fix errors and rerun with --resume; no partial table generated.")
    summary, per_run, gains, details = summarize(completed, records, args.conditions, args.poolings, args.seeds)
    write_csv(output / "bucket_summary.csv", summary)
    write_csv(output / "per_run_bucket_metrics.csv", per_run)
    if gains:
        write_csv(output / "bucket_gains.csv", gains)
    write_json(output / "bucket_details.json", details)
    write_json(output / "manifest.json", {**identity, "conditions": args.conditions,
                                         "training_poolings": args.poolings, "seeds": args.seeds})
    (output / "bucket_tables.tex").write_text(latex_tables(summary, args.conditions, args.poolings), encoding="utf-8")
    LOGGER.info("Complete: %d/%d common records | results: %s",
                details["common_scored_records"], details["benchmark_records"], output)


if __name__ == "__main__":
    main()
