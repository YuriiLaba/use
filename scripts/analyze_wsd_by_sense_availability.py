"""Report how many assigned examples are available for each meaning."""

import argparse
import json
from collections import Counter


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
    definitions_with_zero_sentences = 0
    sentence_counts = Counter()
    definitions_needing_sentences = []

    for lemma, meanings in data.items():
        for meaning, meaning_data in meanings.items():
            total_definitions += 1
            sentence_count = len(meaning_data.get("sentences", []))
            sentence_counts[sentence_count] += 1

            if sentence_count < min_sentences:
                definitions_under_threshold += 1
                definitions_needing_sentences.append(
                    (lemma, meaning, sentence_count)
                )

            if sentence_count == 0:
                definitions_with_zero_sentences += 1

    print(f"Input file: {input_file}")
    print(f"Minimum required sentences: {min_sentences}")
    print(f"Total definitions: {total_definitions:,}")
    print(
        f"Definitions with fewer than {min_sentences} sentences: "
        f"{definitions_under_threshold:,}"
    )
    print(
        f"Definitions with zero sentences: "
        f"{definitions_with_zero_sentences:,}"
    )

    print("\nSentence-count distribution:")
    for count, number in sorted(sentence_counts.items()):
        print(f"{count} sentences: {number:,} definitions")

    print(
        f"\nDefinitions requiring generation "
        f"(showing the first 20 of {len(definitions_needing_sentences):,}):"
    )
    for lemma, meaning, count in definitions_needing_sentences[:20]:
        print(f"{lemma} | {meaning} | {count} sentences")


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
