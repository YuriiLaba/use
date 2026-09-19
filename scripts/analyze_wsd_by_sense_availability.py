"""Report how many assigned examples are available for each meaning."""

import argparse
import json


DEFAULT_INPUT_FILE = (
    "local_datasets/semi_supervised_2/"
    "lemmas_with_meanings_and_sentences_mpnet_filtered.json"
)
DEFAULT_MIN_SENTENCES = 5


def analyze(input_file: str, min_sentences: int) -> None:
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    total_definitions = 0
    definitions_under_threshold = 0

    for lemma, meanings in data.items():
        for meaning, meaning_data in meanings.items():
            total_definitions += 1
            sentence_count = len(meaning_data.get("sentences", []))

            if sentence_count < min_sentences:
                definitions_under_threshold += 1

    percentage = (
        definitions_under_threshold / total_definitions * 100
        if total_definitions
        else 0.0
    )
    print(f"{percentage:.2f}%")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Count definitions with insufficient assigned sentences."
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="Grouped JSON file to analyze.",
    )
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=DEFAULT_MIN_SENTENCES,
        help="Threshold below which a definition is considered underrepresented.",
    )
    parsed = parser.parse_args(args)

    if parsed.min_sentences < 1:
        parser.error("--min-sentences must be at least 1")

    return parsed


if __name__ == "__main__":
    options = parse_args()
    analyze(options.input_file, options.min_sentences)
