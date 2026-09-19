"""
Run: python3 -m augment.translation.augment_translation
"""

import json
import logging
import os

from tqdm import tqdm
from torch.utils.data import DataLoader

from augment.translation.back_translator import (
    BackTranslator,
    HelsinkiCTranslateTranslator,
    # NLLB200CTranslateTranslator,
    # NLLB200TransformersTranslator,
)

from augment.common import TextDataset, ThreadedWriter, set_random_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)


INPUT_TEXTS_PATH = (
    "local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json"
)
OUTPUT_TEXTS_PATH = "local_datasets/augmented/translation/augmented_sentences_translated_v3.jsonl"

BATCH_SIZE = 256
NUM_WORKERS = 2
NUM_AUGMENTATIONS = 4
SEED = 42

# generating BATCH_SIZE x NUM_AUGMENTATIONS augmented per batch


def main():
    set_random_seed(SEED)
    # pivot = NLLB200CTranslateTranslator(
    #     "models/translators/nllb-200-3.3B",
    #     ["uk", "en"],
    #     device="cuda",
    #     device_index=[0, 1],
    # )

    pivot1 = HelsinkiCTranslateTranslator(
        "models/translators/opus-mt-zle-en-ct2",
        "Helsinki-NLP/opus-mt-tc-big-zle-en",
        device="cuda",
        device_index=[0],
    )

    pivot2 = HelsinkiCTranslateTranslator(
        "models/translators/opus-mt-en-zle-ct2",
        "Helsinki-NLP/opus-mt-tc-big-en-zle",
        device="cuda",
        device_index=[0],
    )

    translator = BackTranslator(
        pivot_models=[pivot1, pivot2], languages=[("uk", "en"), ("en", "uk")]
    )

    last_processed_sentence = None
    if os.path.exists(OUTPUT_TEXTS_PATH):
        with open(OUTPUT_TEXTS_PATH, "r", encoding="utf-8") as f:
            for line in reversed(list(f)):
                data = json.loads(line)

                if "original" in data:
                    last_processed_sentence = data["original"]
                    print(f"Resuming from sentence: {last_processed_sentence}")
                    break
    else:
        print("No existing output file found, starting fresh.")

    texts_dataset = TextDataset(INPUT_TEXTS_PATH, last_sentence=last_processed_sentence)
    dataloader = DataLoader(
        texts_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Open file for writing results immediately
    writer = ThreadedWriter(OUTPUT_TEXTS_PATH)

    try:
        for batch in tqdm(dataloader, desc="Augmenting"):
            augmented_texts = translator(batch["sentence"], n=NUM_AUGMENTATIONS)
            writer.write(batch, augmented_texts)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

# ct2-transformers-converter --model facebook/nllb-200-3.3B --output_dir ./models/translators/nllb-200-3.3B
