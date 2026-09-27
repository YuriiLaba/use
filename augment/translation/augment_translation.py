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
from services.config import get_int, get_value


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)


INPUT_TEXTS_PATH = get_value("augmentation", "input_dataset")
OUTPUT_TEXTS_PATH = get_value("augmentation", "translation_output")
BATCH_SIZE = get_int("augmentation", "batch_size", 256)
NUM_WORKERS = get_int("augmentation", "num_workers", 2)
NUM_AUGMENTATIONS = get_int("augmentation", "num_variants", 4)
SEED = get_int("augmentation", "seed", 42)
GPU_ID = int(os.getenv("AUGMENT_GPU_ID", get_value("augmentation", "translation_gpu", "0")))

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
        get_value("models", "translation_uk_en"),
        get_value("models", "translation_hf_uk_en"),
        device="cuda",
        device_index=[GPU_ID],
    )

    pivot2 = HelsinkiCTranslateTranslator(
        get_value("models", "translation_en_uk"),
        get_value("models", "translation_hf_en_uk"),
        device="cuda",
        device_index=[GPU_ID],
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
