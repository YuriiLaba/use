"""
Run: python3 -m augment.translation.augment_translation_definitions
"""

import os

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from augment.common import TextDataset, ThreadedWriter, set_random_seed

from augment.translation.back_translator import (
    BackTranslator,
    HelsinkiCTranslateTranslator,
    NLLB200CTranslateTranslator,
    NLLB200TransformersTranslator,
)

INPUT_TEXTS_PATH = "local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json"
OUTPUT_TEXTS_PATH = (
    "local_datasets/augmented/translation/augmented_sentences_translated_definitions.jsonl"
)

BATCH_SIZE = 256
NUM_WORKERS = 2
NUM_AUGMENTATIONS = 4
SEED = 42
GPU_ID = int(os.getenv("AUGMENT_GPU_ID", "0"))

# generating BATCH_SIZE x NUM_AUGMENTATIONS augmented per batch

def main():
    set_random_seed(SEED)
    pivot1 = HelsinkiCTranslateTranslator(
        "models/translators/opus-mt-zle-en-ct2",
        "Helsinki-NLP/opus-mt-tc-big-zle-en",
        device="cuda",
        device_index=[GPU_ID],
    )

    pivot2 = HelsinkiCTranslateTranslator(
        "models/translators/opus-mt-en-zle-ct2",
        "Helsinki-NLP/opus-mt-tc-big-en-zle",
        device="cuda",
        device_index=[GPU_ID],
    )

    translator = BackTranslator(
        pivot_models=[pivot1, pivot2], languages=[("uk", "en"), ("en", "uk")]
    )

    texts_dataset = TextDataset(INPUT_TEXTS_PATH, load_definitions=True)
    dataloader = DataLoader(
        texts_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Open file for writing results immediately
    writer = ThreadedWriter(OUTPUT_TEXTS_PATH, include_target_word_check=False)

    try:
        for batch in tqdm(dataloader, desc="Augmenting"):
            augmented_texts = translator(batch["sentence"], n=NUM_AUGMENTATIONS)
            writer.write(batch, augmented_texts)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

# ct2-transformers-converter --model facebook/nllb-200-3.3B --output_dir ./models/translators/nllb-200-3.3B
