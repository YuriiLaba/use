"""
Assign benchmark meanings to collected sentences with sentence embeddings.

Run:
    python -m local_datasets.semi_supervised_2.assign_meaning_to_sentence
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from services.config import HOMONYM_BENCHMARK_PATH, UNIQUE_LEMMAS_WITH_SENTENCES_FILE
from services.utils_data import read_homonym_benchmark


logger = logging.getLogger(__name__)


BATCH_SIZE = 2048
TEMPERATURE = 0.05
CUT_OFF_PROBABILITY = 0.9
CUT_OFF_SIMILARITY = 0.6
EMBEDDER_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Keep this flat JSONL output for inspection and auditing of individual
# sentence-to-meaning assignments.
MEANINGS_PATH = "./local_datasets/semi_supervised_2/assigned_meanings_mpnet.jsonl"
LEMMAS_WITH_MEANINGS_AND_SENTENCES_PATH = (
    "./local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet.json"
)


@dataclass(frozen=True)
class AssignmentConfig:
    batch_size: int = BATCH_SIZE
    temperature: float = TEMPERATURE
    cutoff_probability: float = CUT_OFF_PROBABILITY
    cutoff_similarity: float = CUT_OFF_SIMILARITY


def resolve_device(requested_device: str) -> str:
    """Resolve ``auto`` without failing on CPU-only machines such as Macs."""
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sentences_by_lemma(path: str) -> dict[str, list[str]]:
    """Load JSONL sentence records and index them by lemma."""
    sentences_by_lemma: dict[str, list[str]] = {}

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            lemma = str(record.get("lemma", "")).strip().lower()
            sentences = record.get("sentences", [])

            if isinstance(sentences, str):
                sentences = [sentences]

            if not lemma:
                logger.warning("Skipping record %s without a lemma", line_number)
                continue

            valid_sentences = [
                sentence.strip()
                for sentence in sentences
                if isinstance(sentence, str) and sentence.strip()
            ]
            sentences_by_lemma.setdefault(lemma, []).extend(valid_sentences)

    return sentences_by_lemma


def _normalise_glosses(glosses) -> list[str]:
    if not glosses:
        return []

    if isinstance(glosses, str):
        glosses = [glosses]

    return [
        gloss.strip()
        for gloss in glosses
        if isinstance(gloss, str) and gloss.strip()
    ]


def _get_gloss_embeddings(
    glosses: list[str],
    embedder: SentenceTransformer,
    config: AssignmentConfig,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Return cached embeddings for glosses, encoding only missing values."""
    missing_glosses = [gloss for gloss in glosses if gloss not in cache]

    if missing_glosses:
        encoded = embedder.encode(
            missing_glosses,
            batch_size=min(config.batch_size, len(missing_glosses)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        encoded = np.asarray(encoded)
        if encoded.ndim == 1:
            encoded = encoded[None, :]

        for gloss, embedding in zip(missing_glosses, encoded):
            cache[gloss] = embedding

    return np.vstack([cache[gloss] for gloss in glosses])


def process_lemma(
    lemma: str,
    meanings,
    sentences: list[str],
    embedder: SentenceTransformer,
    config: AssignmentConfig,
    gloss_embedding_cache: dict[str, np.ndarray],
) -> tuple[list[dict], dict]:
    """Assign meanings to one lemma and return flat and grouped records."""
    grouped_meanings = {}
    meaning_records = []

    for meaning in meanings.itertuples(index=False):
        glosses = _normalise_glosses(meaning.gloss)
        if not glosses:
            logger.warning("Skipping empty gloss for lemma '%s'", lemma)
            continue

        meaning_key = glosses[0]
        if meaning_key in grouped_meanings:
            raise ValueError(
                f"Duplicate first gloss for lemma '{lemma}': {meaning_key!r}"
            )

        grouped_meanings[meaning_key] = {
            "meaning": {
                "gloss": glosses,
                "examples": meaning.examples,
            },
            "sentences": [],
        }
        meaning_records.append((meaning_key, glosses))

    if not meaning_records or not sentences:
        return [], grouped_meanings

    meaning_embeddings = np.vstack(
        [
            _get_gloss_embeddings(
                glosses, embedder, config, gloss_embedding_cache
            ).mean(axis=0)
            for _, glosses in meaning_records
        ]
    )

    sentence_embeddings = embedder.encode(
        sentences,
        batch_size=config.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarity_matrix = cosine_similarity(meaning_embeddings, sentence_embeddings)
    # Softmax is calculated across meanings for each sentence.
    probability_matrix = softmax(
        similarity_matrix / config.temperature,
        axis=0,
    )

    best_meaning_indices = np.argmax(probability_matrix, axis=0)
    best_probabilities = probability_matrix[
        best_meaning_indices, np.arange(len(sentences))
    ]
    best_similarities = similarity_matrix[
        best_meaning_indices, np.arange(len(sentences))
    ]

    flat_results = []
    for sentence_index, sentence in enumerate(sentences):
        probability = float(best_probabilities[sentence_index])
        similarity = float(best_similarities[sentence_index])

        if (
            probability < config.cutoff_probability
            or similarity < config.cutoff_similarity
        ):
            continue

        meaning_key = meaning_records[best_meaning_indices[sentence_index]][0]

        # Avoid storing a benchmark gloss as its own collected example.
        if meaning_key == sentence:
            continue

        assignment = {
            "lemma": lemma,
            "sentence": sentence,
            "similarity": similarity,
            "probability": probability,
            "assigned_meaning": meaning_key,
        }
        flat_results.append(assignment)
        grouped_meanings[meaning_key]["sentences"].append(
            {
                "sentence": sentence,
                "similarity": similarity,
                "probability": probability,
            }
        )

    return flat_results, grouped_meanings


def _ensure_parent_directory(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run_assignment(
    embedder: SentenceTransformer,
    config: AssignmentConfig,
    sentences_path: str = UNIQUE_LEMMAS_WITH_SENTENCES_FILE,
    benchmark_path: str = HOMONYM_BENCHMARK_PATH,
    meanings_path: str = MEANINGS_PATH,
    grouped_output_path: str = LEMMAS_WITH_MEANINGS_AND_SENTENCES_PATH,
) -> None:
    benchmark = read_homonym_benchmark(benchmark_path)
    if benchmark.empty:
        raise ValueError(f"No valid benchmark rows found in {benchmark_path}")

    sentences_by_lemma = load_sentences_by_lemma(sentences_path)
    grouped_benchmark = benchmark.groupby("lemma", sort=False)
    gloss_embedding_cache: dict[str, np.ndarray] = {}
    grouped_results = {}

    _ensure_parent_directory(meanings_path)
    _ensure_parent_directory(grouped_output_path)

    with open(meanings_path, "w", encoding="utf-8") as flat_file:
        for lemma, meanings in tqdm(
            grouped_benchmark,
            total=benchmark["lemma"].nunique(),
            desc="Processing lemmas",
        ):
            flat_results, grouped_meanings = process_lemma(
                lemma=lemma,
                meanings=meanings,
                sentences=sentences_by_lemma.get(lemma, []),
                embedder=embedder,
                config=config,
                gloss_embedding_cache=gloss_embedding_cache,
            )
            grouped_results[lemma] = grouped_meanings

            for result in flat_results:
                flat_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    with open(grouped_output_path, "w", encoding="utf-8") as grouped_file:
        json.dump(grouped_results, grouped_file, ensure_ascii=False, indent=2)

    logger.info(
        "Saved assignments to %s and grouped data to %s",
        meanings_path,
        grouped_output_path,
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Assign meanings to sentences")
    parser.add_argument(
        "--embedder-model",
        "--embedder_model",
        dest="embedder_model",
        default=EMBEDDER_MODEL,
        help="Sentence-transformer model used for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Embedding device; auto selects CUDA when available, otherwise CPU",
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--cutoff-probability",
        type=float,
        default=CUT_OFF_PROBABILITY,
    )
    parser.add_argument(
        "--cutoff-similarity",
        type=float,
        default=CUT_OFF_SIMILARITY,
    )
    parser.add_argument("--sentences-path", default=UNIQUE_LEMMAS_WITH_SENTENCES_FILE)
    parser.add_argument("--benchmark-path", default=HOMONYM_BENCHMARK_PATH)
    parser.add_argument("--meanings-path", default=MEANINGS_PATH)
    parser.add_argument(
        "--grouped-output-path",
        default=LEMMAS_WITH_MEANINGS_AND_SENTENCES_PATH,
    )
    parsed = parser.parse_args(args)

    if parsed.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if parsed.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 <= parsed.cutoff_probability <= 1:
        parser.error("--cutoff-probability must be between 0 and 1")
    if not -1 <= parsed.cutoff_similarity <= 1:
        parser.error("--cutoff-similarity must be between -1 and 1")

    return parsed


def main(args=None):
    options = parse_args(args)
    device = resolve_device(options.device)
    logger.info("Loading embedder '%s' on %s", options.embedder_model, device)

    embedder = SentenceTransformer(options.embedder_model, device=device)
    config = AssignmentConfig(
        batch_size=options.batch_size,
        temperature=options.temperature,
        cutoff_probability=options.cutoff_probability,
        cutoff_similarity=options.cutoff_similarity,
    )

    run_assignment(
        embedder=embedder,
        config=config,
        sentences_path=options.sentences_path,
        benchmark_path=options.benchmark_path,
        meanings_path=options.meanings_path,
        grouped_output_path=options.grouped_output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
