"""
Run: python3 -m local_datasets.semi_supervised_2.form_triplets
"""

import csv
import json
import logging
import random
from collections import defaultdict

import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

from services.udpipe_model import UDPipeModel
from services.config import PATH_TO_SOURCE_UDPIPE
from services.utils_embedding_calculation_v2 import (
    _find_target_word_in_sentence,
    _find_target_word_in_tokenized_text,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

DATASET_PATH = (
    "local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json"
)
OUTPUT_CSV = (
    "local_datasets/semi_supervised_2/triplets_semi_supervised_all_augs_mixed_100.csv"
)
TOKENIZER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
AUGMENTATION_PATHS = (
    "local_datasets/augmented/token_shuffling/augmented_sentences.jsonl",
    "local_datasets/augmented/translation/augmented_sentences_translated_v3.jsonl",
    "local_datasets/augmented/dropout/augmented_sentences.jsonl",
    "local_datasets/augmented/mask/augmented_sentences.jsonl",
    # "local_datasets/augmented/all_together/augmented_sentences_3.jsonl",
)
DEFINITIONS_AUGMENTATION_PATHS = (
    "local_datasets/augmented/token_shuffling/augmented_sentences_definitions.jsonl",
    "local_datasets/augmented/translation/augmented_sentences_translated_definitions.jsonl",
    "local_datasets/augmented/dropout/augmented_sentences_definitions.jsonl",
    "local_datasets/augmented/mask/augmented_sentences_definitions.jsonl",
    # "local_datasets/augmented/all_together/augmented_sentences_definitions_3.jsonl",
)

MAX_SENTENCES_PER_MEANING = 100
USE_AUGMENTED = True
USE_DEFINITIONS_AUGMENTED = True

SCHEMA = [
    "lemma",
    "anchor",
    "positive",
    "negative",
    # "positives",
    "anchor_target_word_ids",
    "meaning_idx",
]


def get_recommended_number_of_sentences(
    n_sentences_per_meaning: int, index: int
) -> int:
    if n_sentences_per_meaning >= MAX_SENTENCES_PER_MEANING:
        return 1

    base = MAX_SENTENCES_PER_MEANING // n_sentences_per_meaning
    remainder = MAX_SENTENCES_PER_MEANING % n_sentences_per_meaning

    return base + 1 if index < remainder else base


def get_target_word_embedding_idx(udpipe_model, tokenizer, sentence: str, lemma: str):
    word_in_sentence = _find_target_word_in_sentence(udpipe_model, sentence, lemma)

    if not word_in_sentence:
        return None, None

    inputs = tokenizer(sentence, return_tensors="pt")
    word_positions = _find_target_word_in_tokenized_text(
        tokenizer, inputs, word_in_sentence
    )

    if not word_positions:
        return word_in_sentence, None

    return word_positions[0]


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SCHEMA)
        writer.writeheader()
        emitted_triplets = set()

        for lemma, meanings in tqdm(data.items(), total=len(data)):
            for meaning_idx, meaning_data in enumerate(meanings.values()):
                positives = set(meaning_data["meaning"]["gloss"])

                negatives_dicts = [
                    p for i, p in enumerate(meanings.values()) if meaning_idx != i
                ]
                negatives = set()
                for neg_dict in negatives_dicts:
                    negatives.update(neg_dict["meaning"]["gloss"])

                if USE_DEFINITIONS_AUGMENTED:
                    for positive in list(positives):
                        positives.update(
                            definitions_augmented_sentences.get(positive, [])
                        )

                    for negative in list(negatives):
                        negatives.update(
                            definitions_augmented_sentences.get(negative, [])
                        )

                positives = list(positives)
                negatives = list(negatives)

                sentences = meaning_data["sentences"]
                sentences = random.sample(
                    sentences,
                    min(len(sentences), MAX_SENTENCES_PER_MEANING),
                )

                for index, sentence in enumerate(sentences):
                    anchors = set([sentence["sentence"]])
                    probability = sentence.get("probability")  # currently unused

                    recommended_sentences_with_anchor = (
                        get_recommended_number_of_sentences(len(sentences), index)
                    )

                    if USE_AUGMENTED:
                        anchors_augmented = augmented_sentences.get(
                            sentence["sentence"]
                        )
                        if anchors_augmented is not None:
                            anchors.update(anchors_augmented)

                    anchors = list(anchors)

                    candidate_combinations = [
                        (anchor, positive, negative)
                        for anchor in anchors
                        for positive in positives
                        for negative in negatives
                        if (lemma, anchor, positive, negative) not in emitted_triplets
                    ]
                    random.shuffle(candidate_combinations)

                    max_sentences = min(
                        recommended_sentences_with_anchor,
                        len(candidate_combinations),
                    )

                    if max_sentences < recommended_sentences_with_anchor:
                        logging.warning(
                            "Not enough unique combinations for lemma '%s', "
                            "meaning index %s. Recommended: %s, available: %s.",
                            lemma,
                            meaning_idx,
                            recommended_sentences_with_anchor,
                            max_sentences,
                        )

                    for anchor, positive, negative in candidate_combinations[
                        :max_sentences
                    ]:
                        _, anchor_target_ids = get_target_word_embedding_idx(
                            udpipe_model, tokenizer, anchor, lemma
                        )

                        if anchor_target_ids is None:
                            continue

                        emitted_triplets.add(
                            (lemma, anchor, positive, negative)
                        )

                        row = {
                            "lemma": lemma,
                            "anchor": anchor,
                            "positive": positive,
                            "negative": negative,
                            # "positives": json.dumps(positives, ensure_ascii=False),
                            "anchor_target_word_ids": json.dumps(
                                anchor_target_ids, ensure_ascii=False
                            ),
                            "meaning_idx": meaning_idx,
                        }

                        writer.writerow(row)


if __name__ == "__main__":
    random.seed(42)

    if USE_AUGMENTED:
        augmented_sentences = defaultdict(list)

        logging.info("Loading augmented sentences...")
        for path in AUGMENTATION_PATHS:
            with open(path, "r", encoding="utf-8") as bt_file:
                for line in bt_file:
                    item = json.loads(line)
                    augmented_sentences[item["sentence"]].extend(item["augmented"])

    if USE_DEFINITIONS_AUGMENTED:
        definitions_augmented_sentences = defaultdict(list)

        logging.info("Loading definitions augmented sentences...")
        for path in DEFINITIONS_AUGMENTATION_PATHS:
            with open(path, "r", encoding="utf-8") as dbt_file:
                for line in dbt_file:
                    item = json.loads(line)
                    definitions_augmented_sentences[item["sentence"]].extend(
                        item["augmented"]
                    )

    logging.info("Loading NLP models...")
    spacy_nlp = spacy.load("uk_core_news_sm", enable=["lemmatizer"])
    udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    main()
