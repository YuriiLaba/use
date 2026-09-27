"""Central project configuration.

The active pipeline reads ``project_config.ini`` from the repository root.
Set ``PIPELINE_CONFIG`` to use another INI file, or pass a path explicitly to
``load_config``. The legacy constants below remain available to older modules.
"""

from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "project_config.ini"


def load_config(config_path: str | os.PathLike[str] | None = None) -> configparser.ConfigParser:
    selected_path = Path(
        config_path or os.getenv("PIPELINE_CONFIG", str(DEFAULT_CONFIG_PATH))
    )
    if not selected_path.is_absolute():
        selected_path = PROJECT_ROOT / selected_path

    parser = configparser.ConfigParser()
    if not parser.read(selected_path):
        raise FileNotFoundError(f"Configuration file not found: {selected_path}")
    return parser


def get_value(
    section: str,
    option: str,
    fallback=None,
    *,
    config_path: str | os.PathLike[str] | None = None,
):
    return load_config(config_path).get(section, option, fallback=fallback)


def get_list(
    section: str,
    option: str,
    fallback: list[str] | None = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    value = get_value(section, option, None, config_path=config_path)
    if value is None or not value.strip():
        return [] if fallback is None else fallback
    return [item.strip() for item in value.split(",") if item.strip()]


def get_bool(
    section: str,
    option: str,
    fallback: bool = False,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> bool:
    return load_config(config_path).getboolean(section, option, fallback=fallback)


def get_int(
    section: str,
    option: str,
    fallback: int = 0,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> int:
    return load_config(config_path).getint(section, option, fallback=fallback)


def _path(section: str, option: str, fallback: str) -> str:
    return str(get_value(section, option, fallback))


ACUTE = chr(0x301)
GRAVE = chr(0x300)
MINIMUM_POS_OCCURRENCE = 100
MINIMUM_GLOSS_OCCURRENCE = 300
FREQUENCY_QUANTILES = 10

PATH_TO_SOURCE_DATASET = _path(
    "collection", "source_news",
    "datasets_pre_defined/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2",
)
PATH_TO_SOURCE_UDPIPE = _path(
    "paths", "udpipe_model", "models/20180506.uk.mova-institute.udpipe"
)
PATH_TO_LEMMAS_OF_INTEREST = _path(
    "paths", "lemmas_file", "datasets_pre_defined/unique_lemmas_homonyms.txt"
)
PATH_TO_SAVE_GATHERED_DATASET = _path(
    "collection", "output_news",
    "local_datasets/raw_sentences/lemma_examples_news.json",
)
NUMBER_OF_EXAMPLES_TO_GATHER = get_int("collection", "number_of_examples", -1)
EMBEDDER_MODEL = _path(
    "models", "embedder_model",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
TRIPLET_PROCESSOR_BATCH_SIZE = 2000
PATH_TO_SAVE_TRIPLETS = "local_datasets/ubertext_triplets_6m_samples.csv"
UNIQUE_LEMMAS_WITH_SENTENCES_FILE = _path(
    "paths", "unique_sentences",
    "local_datasets/raw_sentences/unique_lemma_sentences.jsonl",
)
HOMONYM_BENCHMARK_PATH = _path(
    "paths", "benchmark", "datasets_pre_defined/ukrainian_wsd_benchmark.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a project configuration value")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--get", nargs=2, metavar=("SECTION", "OPTION"), required=True
    )
    args = parser.parse_args()
    print(get_value(args.get[0], args.get[1], "", config_path=args.config))


if __name__ == "__main__":
    main()
