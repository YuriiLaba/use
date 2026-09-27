import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from services.config import get_float, get_int, get_value

INPUT_FILE = get_value(
    "paths", "assigned_grouped",
    "local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet.json",
)
OUTPUT_FILE = get_value(
    "paths", "filtered_grouped",
    "local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet_filtered.json",
)
MODEL_NAME = get_value(
    "filtering", "model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
THRESHOLD = get_float("filtering", "threshold", 0.95)
BATCH_SIZE = get_int("filtering", "batch_size", 512)

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="Print sample matches without saving")
parser.add_argument("--show", type=int, default=20, help="Number of examples to print in dry-run mode")
args = parser.parse_args()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# Collect all example sentences globally
examples = []
for lemma, data in json_data.items():
    for meaning, meaning_data in data.items():
        examples.extend(meaning_data.get("meaning", {}).get("examples", []))

print(f"Total examples: {len(examples)}")

model = SentenceTransformer(MODEL_NAME)

print("Encoding examples...")
example_embeddings = model.encode(examples, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)

# Collect all candidate sentences with their location
all_sentences = []
locations = []  # (lemma, meaning, idx)
for lemma, data in json_data.items():
    for meaning, meaning_data in data.items():
        for idx, sent_obj in enumerate(meaning_data.get("sentences", [])):
            all_sentences.append(sent_obj["sentence"])
            locations.append((lemma, meaning, idx))

print(f"Total sentences to filter: {len(all_sentences)}")

print("Encoding sentences...")
sentence_embeddings = model.encode(all_sentences, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)

# Compute max cosine similarity to any example in batches to avoid OOM
print("Computing similarities...")
CHUNK = 2048
keep_mask = np.ones(len(all_sentences), dtype=bool)
for start in range(0, len(all_sentences), CHUNK):
    end = min(start + CHUNK, len(all_sentences))
    sims = cosine_similarity(sentence_embeddings[start:end], example_embeddings)
    max_sims = sims.max(axis=1)
    keep_mask[start:end] = max_sims < THRESHOLD
    if start % (CHUNK * 10) == 0:
        print(f"  {end}/{len(all_sentences)} processed, kept so far: {keep_mask[:end].sum()}")

n_removed = (~keep_mask).sum()
print(f"Keeping {keep_mask.sum()} / {len(all_sentences)} sentences (would remove {n_removed})")

if args.dry_run:
    print(f"\n--- Sample of {args.show} sentences that WOULD be deleted ---\n")
    removed_indices = np.where(~keep_mask)[0]
    sample_indices = removed_indices[:args.show]

    # Recompute max sim + closest example for the sample only
    sample_embeddings = sentence_embeddings[sample_indices]
    sims = cosine_similarity(sample_embeddings, example_embeddings)
    for rank, i in enumerate(sample_indices):
        lemma, meaning, _ = locations[i]
        sentence = all_sentences[i]
        best_example_idx = sims[rank].argmax()
        best_sim = sims[rank, best_example_idx]
        closest_example = examples[best_example_idx]
        print(f"[sim={best_sim:.3f}] [{lemma} / {meaning[:40]}]")
        print(f"  SENTENCE : {sentence}")
        print(f"  EXAMPLE  : {closest_example}")
        print()
    print("Dry run complete — no file written.")
else:
    # Build filtered dataset
    to_delete = set(int(i) for i in np.where(~keep_mask)[0])
    delete_map = {}
    for i in to_delete:
        lemma, meaning, idx = locations[i]
        delete_map.setdefault((lemma, meaning), set()).add(idx)

    for (lemma, meaning), bad_indices in delete_map.items():
        original = json_data[lemma][meaning]["sentences"]
        json_data[lemma][meaning]["sentences"] = [
            s for i, s in enumerate(original) if i not in bad_indices
        ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Saved filtered data to {OUTPUT_FILE}")
