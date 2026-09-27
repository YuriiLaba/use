"""
Process raw sentences to gather unique sentences for each lemma.

Run: python3 -m local_datasets.raw_sentences.process_raw_sentences
"""

import json
from glob import glob
from collections import defaultdict

from tqdm import tqdm

from services.config import UNIQUE_LEMMAS_WITH_SENTENCES_FILE, get_value


def main():
    """Gather unique sentences for each lemma from raw sentence files and save them to a JSONL file."""
    unique_sentences = defaultdict(set)
    raw_sentences_dir = get_value("paths", "raw_sentences_dir", "local_datasets/raw_sentences")
    sentences_files = list(sorted(glob(f"{raw_sentences_dir}/*.json")))

    if UNIQUE_LEMMAS_WITH_SENTENCES_FILE in sentences_files:
        sentences_files.remove(UNIQUE_LEMMAS_WITH_SENTENCES_FILE)

    # gather unique sentences
    for file_path in tqdm(
        sentences_files, desc="Gathering unique sentences", leave=False
    ):
        new_sentences_count = 0
        processed_sentences = 0

        with open(file_path, "r", encoding="utf-8") as file:
            for line in tqdm(
                file.readlines(), desc=f"Processing {file_path}", leave=False
            ):
                batch: dict = json.loads(line)

                for lemma, sentences in batch.items():
                    before_count = len(unique_sentences[lemma])

                    for sentence in sentences:
                        unique_sentences[lemma].add(sentence)

                    after_count = len(unique_sentences[lemma])

                    new_sentences_count += after_count - before_count
                    processed_sentences += len(sentences)

        print(
            f"\nNew unique sentences added from {file_path}: {new_sentences_count:,} out of {processed_sentences:,} processed sentences."
        )

    print(f"Total unique lemmas collected: {len(unique_sentences)}")

    total_sentences_count = sum(
        len(sentences) for sentences in unique_sentences.values()
    )
    print(f"Total sentences collected: {total_sentences_count:,}")

    with open(UNIQUE_LEMMAS_WITH_SENTENCES_FILE, "w", encoding="utf-8") as file:
        for lemma, sentences in unique_sentences.items():
            line = json.dumps(
                {"lemma": lemma, "sentences": list(sentences)}, ensure_ascii=False
            )
            file.write(line + "\n")


if __name__ == "__main__":
    main()
