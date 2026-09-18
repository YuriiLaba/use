"""
Run: python3 -m local_datasets.semi_supervised_2.assign_meaning_to_sentence
"""

import json
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from services.utils_data import read_homonym_benchmark
from services.config import (
    UNIQUE_LEMMAS_WITH_SENTENCES_FILE,
    HOMONYM_BENCHMARK_PATH,
)


## Global Variables
BATCH_SIZE = 2048

TEMPERATURE = 0.05
CUT_OFF_PROBABILITY = 0.9
CUT_OFF_SIMILARITY = 0.6
EMBEDDER_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

MEANINGS_PATH = "./local_datasets/semi_supervised_2/assigned_meanings_mpnet.jsonl"
LEMMAS_WITH_MEANINGS_AND_SENTENCES_PATH = (
    "./local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet.json"
)

embedder = None
data = None
sum = None

# load models
model_device = "cuda"  # change to "cpu" if you don't want GPU

# storage
lemmas_with_meanings_and_sentences = defaultdict(dict)


## Helpers
def process_lemma(lemma: str) -> list:
    # find and process lemma in SUM
    meanings = sum[sum["lemma"] == lemma]

    # add meanings to storage even if no sentences are found later
    for meaning in meanings.itertuples():
        meaning_examples = meaning.examples
        meaning_glosses = meaning.gloss

        lemmas_with_meanings_and_sentences[lemma][meaning_glosses[0]] = {
            "meaning": {
                "gloss": meaning_glosses,
                "examples": meaning_examples,
            },
            "sentences": [],
        }

    # make embeddings for meanings
    meanings_embeddings = []

    for i in meanings.itertuples():
        glosses = i.gloss if isinstance(i.gloss, (list, tuple)) else [i.gloss]
        # encode returns ndarray shape (n_glosses, dim)
        gloss_embs = embedder.encode(
            glosses,
            batch_size=min(BATCH_SIZE, max(1, len(glosses))),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # average embeddings if multiple gloss strings per meaning (same behaviour as before)
        mean_emb = np.mean(gloss_embs, axis=0)
        meanings_embeddings.append(mean_emb)

    meanings_embeddings = np.vstack(meanings_embeddings)

    # find lemma in unique sentences
    try:
        lemma_idx = next(i for i, item in enumerate(data) if item["lemma"] == lemma)
    except StopIteration:
        lemma_idx = None

    if lemma_idx is None:
        return []

    data_sentences = data[lemma_idx]["sentences"]

    train_set_embeddings = embedder.encode(
        data_sentences,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Display the top 5 closest sentences for each meaning
    similarity_matrix = cosine_similarity(meanings_embeddings, train_set_embeddings)
    prob_matrix = softmax(similarity_matrix / TEMPERATURE, axis=0)

    result = []
    n_meanings, n_sentences = prob_matrix.shape

    # best meaning per sentence
    best_meaning_idx = np.argmax(prob_matrix, axis=0)  # (n_sentences,)
    best_probs = prob_matrix[best_meaning_idx, np.arange(n_sentences)]
    best_sims = similarity_matrix[best_meaning_idx, np.arange(n_sentences)]

    for sent_idx in range(n_sentences):
        prob = best_probs[sent_idx]
        sim = best_sims[sent_idx]

        if prob < CUT_OFF_PROBABILITY or sim < CUT_OFF_SIMILARITY:
            continue

        meaning_idx = best_meaning_idx[sent_idx]
        meaning = meanings.iloc[meaning_idx]
        gloss_0 = meaning.gloss[0]

        sentence = data_sentences[sent_idx]

        if gloss_0 == sentence:
            continue  # skip identical sentence-meaning pairs

        result.append(
            {
                "lemma": lemma,
                "sentence": sentence,
                "similarity": float(sim),
                "probability": float(prob),
                "assigned_meaning": gloss_0,
            }
        )

        lemmas_with_meanings_and_sentences[lemma][gloss_0]["sentences"].append(
            {
                "sentence": sentence,
                "similarity": float(sim),
                "probability": float(prob),
            }
        )

    return result


def main():
    lemmas = sum["lemma"].unique()

    with open(MEANINGS_PATH, "w") as f:
        pbar = tqdm(lemmas, desc="Processing lemmas")

        for lemma in pbar:
            pbar.set_postfix({"current_lemma": lemma})
            lemma_results = process_lemma(lemma)
            lemma_results = [i for i in lemma_results if i]  # filter out empty results

            for res in lemma_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    with open(LEMMAS_WITH_MEANINGS_AND_SENTENCES_PATH, "w") as f:
        json.dump(lemmas_with_meanings_and_sentences, f, ensure_ascii=False, indent=2)


def parse_args():
    global EMBEDDER_MODEL, BATCH_SIZE

    parser = argparse.ArgumentParser(description="Assign meanings to sentences")
    parser.add_argument(
        "--embedder_model",
        type=str,
        default=EMBEDDER_MODEL,
        help="Sentence transformer model to use for embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for embedding computation",
    )
    args = parser.parse_args()

    EMBEDDER_MODEL = args.embedder_model
    BATCH_SIZE = args.batch_size


if __name__ == "__main__":
    parse_args()

    embedder = SentenceTransformer(EMBEDDER_MODEL, device=model_device)

    # load data
    with open(UNIQUE_LEMMAS_WITH_SENTENCES_FILE, "r") as f:
        data = [json.loads(line) for line in f]

    sum = read_homonym_benchmark(HOMONYM_BENCHMARK_PATH)

    main()
