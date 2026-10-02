"""Zero-shot Ukrainian WSD evaluation of the eight LLMs from the paper.

Run from the repository root with ``python -m eval.eval_llm_wsd --device cuda:0``.
Each benchmark example is evaluated separately. CSV files are appended and
flushed after every batch/model, so earlier results survive a later failure.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download, scan_cache_dir
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    set_seed,
)

from services.config import get_int, get_list, get_value


LOGGER = logging.getLogger(__name__)
DEFAULT_MODELS = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0",
    "lapa-llm/lapa-v0.1.2-instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-VL-4B-Instruct",
    "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0",
    "google/gemma-3-12b-it",
    "Qwen/Qwen3-VL-8B-Instruct",
]
DEFAULT_PROMPT = Path(__file__).parent / "prompts/wsd_zero_shot_uk.txt"
PREDICTION_FIELDS = [
    "run_id", "model", "example_id", "lemma", "sentence", "candidate_meanings",
    "gold_sense", "predicted_sense", "correct", "status", "raw_answer",
]
SUMMARY_FIELDS = [
    "run_id", "model", "status", "accuracy", "accuracy_percent", "correct",
    "total_examples", "processed_examples", "invalid_answers", "duration_seconds",
    "device", "dtype", "batch_size", "max_new_tokens", "max_input_tokens", "seed", "model_revision",
    "benchmark_path", "benchmark_sha256", "prompt_sha256", "predictions_path",
    "torch_version", "transformers_version", "error",
]


@dataclass(frozen=True)
class Example:
    example_id: str
    lemma: str
    sentence: str
    meanings: tuple[tuple[str, ...], ...]
    gold_sense: int  # One-based, as in the prompt.


def text_list(value, field: str, line: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Benchmark line {line}: {field} must be a string or list of strings")
    return [item.strip() for item in value if item.strip()]


def read_examples(path: Path) -> list[Example]:
    """Build the full candidate inventory before expanding labeled sentences."""
    inventory: dict[str, list[tuple[str, ...]]] = {}
    rows = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            lemma = row["lemma"].strip().lower()
            gloss = tuple(text_list(row["gloss"], "gloss", line_number))
            examples = text_list(row["examples"], "examples", line_number)
            if not lemma or not gloss:
                skipped += 1
                continue
            meanings = inventory.setdefault(lemma, [])
            if gloss not in meanings:
                meanings.append(gloss)
            rows.append((line_number, lemma, gloss, examples))

    expanded = []
    for line_number, lemma, gloss, sentences in rows:
        meanings = tuple(inventory[lemma])
        if len(meanings) < 2:
            skipped += 1
            continue
        gold_sense = meanings.index(gloss) + 1
        for index, sentence in enumerate(sentences):
            expanded.append(Example(f"{line_number}:{index}", lemma, sentence, meanings, gold_sense))
    LOGGER.info("Loaded %d contextual examples | %d lemmas | skipped rows: %d",
                len(expanded), len(inventory), skipped)
    if not expanded:
        raise ValueError("Benchmark contains no examples with at least two candidate meanings")
    return expanded


def build_prompt(example: Example, template: str) -> str:
    glosses = "\n".join(
        f"{index}. {'; '.join(definitions)}"
        for index, definitions in enumerate(example.meanings, start=1)
    )
    return template.format(lemma=example.lemma, examples=example.sentence, glosses=glosses)


def parse_answer(answer: str, num_meanings: int) -> int | None:
    """Accept only a single numbered answer; explanations/ambiguous answers are invalid."""
    match = re.fullmatch(r"\s*([0-9]+)\s*[.)]?\s*", answer)
    if match:
        number = int(match.group(1))
        if 1 <= number <= num_meanings:
            return number
    return None


class TqdmLogHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    console_handler = TqdmLogHandler()
    console_handler.setLevel(logging.INFO)
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler], force=True)
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


@contextmanager
def csv_writer(path: Path, fields: list[str]):
    """Append runs without replacing previous results; verify the existing schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as handle:
            if next(csv.reader(handle)) != fields:
                raise ValueError(f"Existing CSV has a different schema: {path}")
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
            handle.flush()
        yield writer, handle


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Use cpu, cuda, cuda:N, or auto")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run on the CUDA server with a CUDA-enabled PyTorch build.")
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {device}")
        torch.cuda.set_device(device)
    return device


def resolve_dtype(value: str, device: torch.device):
    if value == "auto":
        value = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "float16" if device.type == "cuda" else "float32"
        )
    if value == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; use --dtype float16")
    return getattr(torch, value)


def load_text_tokenizer(model_id: str):
    """Load the model's text tokenizer and original chat template without image processors."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.chat_template:
        # Some multimodal repos (including Qwen3-VL) store their template in the
        # legacy processor JSON rather than tokenizer_config.json or a Jinja file.
        # Read that template directly instead of constructing AutoProcessor.
        model_path = Path(model_id)
        if model_path.is_dir():
            template_path = model_path / "chat_template.json"
        else:
            template_path = Path(hf_hub_download(model_id, filename="chat_template.json"))
        template = json.loads(template_path.read_text(encoding="utf-8")).get("chat_template")
        if not isinstance(template, (str, dict)) or not template:
            raise ValueError(f"No valid chat template in {template_path}")
        tokenizer.chat_template = template
        LOGGER.info("Loaded text chat template from %s", template_path)
    tokenizer.get_chat_template()  # Fail before loading weights if no default template is usable.
    return tokenizer


def load_model(model_id: str, device: torch.device, dtype):
    config = AutoConfig.from_pretrained(model_id)
    # Lapa and both MamayLM models are Gemma 3 multimodal architectures too.
    if config.model_type in {"qwen3_vl", "gemma3"}:
        model_class = AutoModelForImageTextToText
    else:
        model_class = AutoModelForCausalLM
    tokenizer = load_text_tokenizer(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"Tokenizer has neither padding nor EOS token: {model_id}")
        tokenizer.pad_token = tokenizer.eos_token
    LOGGER.info("Loading %s | architecture=%s | device=%s | dtype=%s",
                model_id, config.model_type, device, dtype)
    model = model_class.from_pretrained(
        model_id, config=config, dtype=dtype, device_map={"": str(device)},
        attn_implementation="sdpa",
    ).eval()
    return model, tokenizer


def generate_answers(model, tokenizer, prompts, device, max_new_tokens, max_input_tokens):
    rendered = []
    for prompt in prompts:
        rendered.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        ))
    inputs = tokenizer(rendered, padding=True, add_special_tokens=False, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    if input_length > max_input_tokens:
        raise ValueError(
            f"Prompt has {input_length} tokens, exceeding --max-input-tokens {max_input_tokens}. "
            "Increase the limit; prompts are never silently truncated."
        )
    # Text-only inference needs no image/video inputs or token-type IDs.
    inputs = {key: value.to(device) for key, value in inputs.items()
              if key in {"input_ids", "attention_mask"}}
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
            return_dict_in_generate=False, output_scores=False,
        )
    return tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=True)


def delete_downloaded_model(model_id: str) -> None:
    """Remove only this model's downloaded Hub revisions and associated blobs."""
    if Path(model_id).exists():
        LOGGER.info("Local model path %s: downloaded-model cache cleanup skipped", model_id)
        return
    cache_info = scan_cache_dir()
    matching_repos = frozenset(
        repo for repo in cache_info.repos
        if repo.repo_type == "model" and repo.repo_id == model_id
    )
    if not matching_repos:
        LOGGER.info("No downloaded cache entry found for %s", model_id)
        return
    # Restrict the cache view to this repository, even if another repository has
    # a revision with the same hash (for example, a mirrored model).
    model_cache = replace(cache_info, repos=matching_repos)
    revisions = [revision.commit_hash for repo in matching_repos for revision in repo.revisions]
    strategy = model_cache.delete_revisions(*revisions)
    LOGGER.info("Deleting downloaded model %s | expected freed space: %s",
                model_id, strategy.expected_freed_size_str)
    strategy.execute()
    LOGGER.info("Deleted downloaded model %s", model_id)


def evaluate_models(args, examples, template, device, dtype) -> bool:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    benchmark_hash = hashlib.sha256(Path(args.benchmark_path).read_bytes()).hexdigest()
    prompt_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    total_started = time.perf_counter()
    any_failed = False
    LOGGER.info("Run %s | models=%d | examples/model=%d | greedy decoding | seed=%d",
                run_id, len(args.models), len(examples), args.seed)
    LOGGER.info("Summary CSV: %s | predictions CSV: %s | log: %s",
                args.output, args.predictions_output, args.log_file)
    LOGGER.info("Delete downloaded models after successful evaluation: %s",
                args.delete_model_after_evaluation)
    with csv_writer(Path(args.output), SUMMARY_FIELDS) as (summary_writer, summary_file), \
            csv_writer(Path(args.predictions_output), PREDICTION_FIELDS) as (prediction_writer, prediction_file):
        for index, model_id in enumerate(args.models, start=1):
            started = time.perf_counter()
            processed = correct = invalid = 0
            model = tokenizer = None
            status, error, revision = "failed", "", ""
            batch_size = args.batch_size
            interrupted = False
            LOGGER.info("[%d/%d] Starting %s | models remaining after this: %d",
                        index, len(args.models), model_id, len(args.models) - index)
            try:
                set_seed(args.seed)
                model, tokenizer = load_model(model_id, device, dtype)
                revision = getattr(model.config, "_commit_hash", "") or ""
                with tqdm(total=len(examples), desc=f"[{index}/{len(args.models)}] {model_id}",
                          unit="example", dynamic_ncols=True) as progress:
                    while processed < len(examples):
                        batch = examples[processed:processed + batch_size]
                        prompts = [build_prompt(example, template) for example in batch]
                        try:
                            answers = generate_answers(
                                model, tokenizer, prompts, device,
                                args.max_new_tokens, args.max_input_tokens,
                            )
                        except torch.cuda.OutOfMemoryError:
                            if batch_size == 1:
                                raise
                            batch_size = max(1, batch_size // 2)
                            gc.collect()
                            torch.cuda.empty_cache()
                            LOGGER.warning("CUDA OOM: retrying the same examples with batch size %d", batch_size)
                            continue
                        if len(answers) != len(batch):
                            raise ValueError("Generation returned an unexpected number of answers")
                        for example, answer in zip(batch, answers):
                            prediction = parse_answer(answer, len(example.meanings))
                            is_correct = prediction == example.gold_sense
                            correct += int(is_correct)
                            invalid += int(prediction is None)
                            prediction_writer.writerow({
                                "run_id": run_id, "model": model_id,
                                "example_id": example.example_id, "lemma": example.lemma,
                                "sentence": example.sentence,
                                "candidate_meanings": json.dumps(example.meanings, ensure_ascii=False),
                                "gold_sense": example.gold_sense, "predicted_sense": prediction,
                                "correct": is_correct,
                                "status": "invalid_answer" if prediction is None else "ok",
                                "raw_answer": answer,
                            })
                            LOGGER.debug("%s | example=%s | lemma=%s | gold=%d | prediction=%s | raw=%r",
                                         model_id, example.example_id, example.lemma,
                                         example.gold_sense, prediction, answer)
                        prediction_file.flush()
                        previous = processed
                        processed += len(batch)
                        progress.update(len(batch))
                        progress.set_postfix(accuracy=f"{100 * correct / processed:.2f}%", invalid=invalid)
                        if processed // args.log_every > previous // args.log_every or processed == len(examples):
                            LOGGER.info("%s | %d/%d examples | accuracy=%.2f%% | invalid=%d",
                                        model_id, processed, len(examples), 100 * correct / processed, invalid)
                status = "ok"
            except KeyboardInterrupt:
                status, error, interrupted = "interrupted", "KeyboardInterrupt", True
                LOGGER.warning("Interrupted %s; completed predictions have been saved", model_id)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("[%d/%d] Failed %s after %d examples", index, len(args.models), model_id, processed)
            finally:
                duration = time.perf_counter() - started
                accuracy = correct / processed if processed else None
                summary_writer.writerow({
                    "run_id": run_id, "model": model_id, "status": status,
                    "accuracy": accuracy, "accuracy_percent": 100 * accuracy if accuracy is not None else None,
                    "correct": correct, "total_examples": len(examples),
                    "processed_examples": processed, "invalid_answers": invalid,
                    "duration_seconds": round(duration, 2), "device": str(device),
                    "dtype": str(dtype).removeprefix("torch."), "batch_size": batch_size,
                    "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens,
                    "seed": args.seed,
                    "model_revision": revision, "benchmark_path": str(args.benchmark_path),
                    "benchmark_sha256": benchmark_hash, "prompt_sha256": prompt_hash,
                    "predictions_path": str(args.predictions_output),
                    "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                    "error": error,
                })
                summary_file.flush()
                prediction_file.flush()
                model = tokenizer = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if args.delete_model_after_evaluation:
                    if status == "ok":
                        try:
                            delete_downloaded_model(model_id)
                        except Exception:
                            LOGGER.exception("Could not delete downloaded model %s", model_id)
                    else:
                        LOGGER.info("Keeping downloaded model %s because evaluation status is %s",
                                    model_id, status)
                any_failed |= status != "ok"
                LOGGER.info("[%d/%d] %s | status=%s | examples=%d/%d | duration=%.2fs",
                            index, len(args.models), model_id, status, processed, len(examples), duration)
            if interrupted:
                raise KeyboardInterrupt
    LOGGER.info("Finished all models | total time=%.2fs", time.perf_counter() - total_started)
    return not any_failed


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    known, _ = config_parser.parse_known_args(argv)
    def value(option, fallback):
        return get_value("llm_evaluation", option, fallback, config_path=known.config)
    def number(option, fallback):
        return get_int("llm_evaluation", option, fallback, config_path=known.config)
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--models", nargs="+", default=get_list(
        "llm_evaluation", "models", DEFAULT_MODELS, config_path=known.config))
    parser.add_argument("--benchmark-path", default=get_value(
        "paths", "benchmark", "datasets_pre_defined/ukrainian_wsd_benchmark.jsonl", config_path=known.config))
    parser.add_argument("--device", default=value("device", "cuda:0"))
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default=value("dtype", "auto"))
    parser.add_argument("--batch-size", type=int, default=number("batch_size", 4))
    parser.add_argument("--max-new-tokens", type=int, default=number("max_new_tokens", 16))
    parser.add_argument("--max-input-tokens", type=int, default=number("max_input_tokens", 4096))
    parser.add_argument("--seed", type=int, default=number("seed", 42))
    parser.add_argument("--log-every", type=int, default=number("log_every", 100))
    parser.add_argument("--max-examples", type=int, default=None, help="Evaluate the first N sentences for a smoke test.")
    parser.add_argument("--prompt-file", type=Path, default=value("prompt_file", str(DEFAULT_PROMPT)))
    parser.add_argument("--output", type=Path, default=value("output", "results/wsd_llm_results.csv"))
    parser.add_argument("--predictions-output", type=Path, default=value("predictions_output", "results/wsd_llm_predictions.csv"))
    parser.add_argument("--log-file", type=Path, default=value("log_file", "results/wsd_llm_eval.log"))
    parser.add_argument(
        "--delete-model-after-evaluation", action="store_true",
        help="Delete each downloaded model from the Hugging Face cache after its results are saved. Failed models are retained.",
    )
    args = parser.parse_args(argv)
    for option in ("batch_size", "max_new_tokens", "max_input_tokens", "log_every", "max_examples"):
        if getattr(args, option) is not None and getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    paths = [Path(args.output).resolve(), Path(args.predictions_output).resolve(), Path(args.log_file).resolve()]
    if len(set(paths)) != len(paths):
        parser.error("Summary, predictions, and log paths must be different")
    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    try:
        device = resolve_device(args.device)
        dtype = resolve_dtype(args.dtype, device)
        examples = read_examples(Path(args.benchmark_path))
        if args.max_examples:
            examples = examples[:args.max_examples]
        template = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        build_prompt(examples[0], template)  # Validate placeholders before loading weights.
        if not evaluate_models(args, examples, template, device, dtype):
            raise SystemExit(1)
    except KeyboardInterrupt:
        LOGGER.warning("Evaluation stopped by user")
        raise SystemExit(130)
    except Exception:
        LOGGER.exception("LLM WSD evaluation failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
