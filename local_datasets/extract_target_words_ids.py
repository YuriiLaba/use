"""
Module to extract target word ids from triplet dataset.

Run: python3 -m local_datasets.extract_target_words_ids
"""

import csv
from concurrent.futures import ProcessPoolExecutor

import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

from services.udpipe_model import UDPipeModel

WORKERS = 16

INPUT_CSV = "local_datasets/semi_supervised/semi_supervised_triplets_hard_mined.csv"
OUTPUT_CSV = "local_datasets/semi_supervised/semi_supervised_triplets_hard_mined_with_target_ids.csv"

TOKENIZER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
UDPIPE_MODEL = "models/20180506.uk.mova-institute.udpipe"

udpipe_model = UDPipeModel(UDPIPE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

spacy_nlp = spacy.load("uk_core_news_sm", enable=["lemmatizer"])


def ngrams(s: str, n: int = 3) -> set[str]:
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def char_dice(a: str, b: str, n: int = 3) -> float:
    """
    Character n-gram Sørensen–Dice coefficient.
    Designed for fuzzy lemma matching (e.g., Ukrainian morphology noise).
    """

    # normalize
    a = a.lower().strip()
    b = b.lower().strip()

    # exact match shortcut
    if a == b:
        return 1.0

    # guard against garbage
    if not a or not b:
        return 0.0

    # if too short, don't lie
    if min(len(a), len(b)) < n:
        return 0.0

    na = ngrams(a, n)
    nb = ngrams(b, n)

    if not na or not nb:
        return 0.0

    return 2 * len(na & nb) / (len(na) + len(nb))


def same_lemma(a: str, b: str) -> bool:
    if a in b or b in a:
        return True

    # length guard
    if min(len(a), len(b)) < 4:
        return False

    # character 3-gram Jaccard
    if char_dice(a, b) >= 0.5:
        return True

    return False


def _find_target_word_in_sentence(
    udpipe_model: UDPipeModel, input_text: str, target_word: str
):
    target_word = target_word.strip().lower()

    tokenized = udpipe_model.tokenize(target_word)
    udpipe_model.tag(tokenized[0])
    target_word_lemma = "".join([i.lemma.lower() for i in tokenized[0].words[1:]])

    tokenized = udpipe_model.tokenize(input_text)
    for tok_sent in tokenized:
        udpipe_model.tag(tok_sent)

        for word_index, w in enumerate(tok_sent.words[1:]):  # under 0 index is root
            token_lemma = w.lemma.lower()

            if same_lemma(target_word, token_lemma) or same_lemma(
                target_word_lemma, token_lemma
            ):
                return tok_sent.words[word_index + 1].form

    # use spacy as fallback
    target_word_lemma = "".join([i.lemma_.lower() for i in spacy_nlp(target_word)])

    doc = spacy_nlp(input_text)
    for token in doc:
        token_lemma = token.lemma_.lower()

        if same_lemma(target_word, token_lemma) or same_lemma(
            target_word_lemma, token_lemma
        ):
            return token.text

    return None


def normalize_word(word: str) -> str:
    return (
        word.replace("«", "")
        .replace("»", "")
        .replace('"', "")
        .replace("“", "")
        .replace("”", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(".", "")
        .strip()
    )


def _find_target_word_in_tokenized_text_new(tokenizer, tokenized_input_text, word: str):
    target_words_with_indexes = []

    word = word.strip().lower()
    current_word = ""
    start_index = 0

    # word_ids() is unavailable for slow Python tokenizers.  This helper only
    # needs token positions and decoded token text, so input IDs are enough.
    for index, input_id in enumerate(tokenized_input_text["input_ids"][0]):
        token = tokenizer.decode([input_id]).strip()

        # Remove subword prefix if present
        if token.startswith("##"):
            token = token.replace("##", "")

        token = token.replace("▁", "").lower()

        current_word += token
        current_word_normalized = normalize_word(current_word)

        if word.startswith(current_word_normalized):
            if word == current_word_normalized:
                end_index = index
                target_words_with_indexes.append(
                    (word, list(range(start_index, end_index + 1)))
                )

                current_word = ""
                start_index = index + 1

        else:
            current_word = token
            start_index = index
            current_word_normalized = normalize_word(current_word)

            if word.startswith(current_word_normalized):
                if word == current_word_normalized:
                    end_index = index
                    target_words_with_indexes.append(
                        (word, list(range(start_index, end_index + 1)))
                    )

                    current_word = ""
                    start_index = index + 1
            else:
                current_word = ""
                start_index = index + 1

    return target_words_with_indexes


def get_target_word_embedding_idx(udpipe_model, tokenizer, sentence: str, lemma: str):
    word_in_sentence = _find_target_word_in_sentence(udpipe_model, sentence, lemma)

    if not word_in_sentence:
        # print(f"Warning: could not find lemma '{lemma}' in sentence: {sentence}")
        return None, None

    inputs = tokenizer(sentence, return_tensors="pt")
    word_positions = _find_target_word_in_tokenized_text_new(
        tokenizer, inputs, word_in_sentence
    )

    if not word_positions:
        # print(
        #     f"Warning: could not find word '{word_in_sentence}' in sentence: {sentence}"
        # )
        return word_in_sentence, None

    return word_positions[0]


def process_row(row):
    anchor = row["anchor"]
    positive = row["positive"]
    negative = row["negative"]
    lemma = row["lemma"]

    _, anchored_ids = get_target_word_embedding_idx(
        udpipe_model, tokenizer, anchor, lemma
    )
    _, positive_ids = get_target_word_embedding_idx(
        udpipe_model, tokenizer, positive, lemma
    )
    _, negative_ids = get_target_word_embedding_idx(
        udpipe_model, tokenizer, negative, lemma
    )

    if None in (anchored_ids, positive_ids, negative_ids):
        return None

    row["anchor_target_word_ids"] = anchored_ids
    row["positive_target_word_ids"] = positive_ids
    row["negative_target_word_ids"] = negative_ids

    return row


def main():
    input_rows = 0
    output_rows = 0

    with open(INPUT_CSV, "r") as in_f, open(OUTPUT_CSV, "w", newline="") as out_f:
        reader = csv.DictReader(in_f)
        header = reader.fieldnames + [
            "anchor_target_word_ids",
            "positive_target_word_ids",
            "negative_target_word_ids",
        ]

        writer = csv.DictWriter(out_f, fieldnames=header)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = executor.map(process_row, reader, chunksize=256)

            for result in tqdm(futures):
                input_rows += 1

                if result is None:
                    continue

                writer.writerow(result)
                output_rows += 1

    print(f"Processed {input_rows} rows, wrote {output_rows} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
