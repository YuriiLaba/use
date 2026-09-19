"""
Run: python3 -m augment.augment_all_together
"""

import json
import logging
import os

from tqdm import tqdm
from torch.utils.data import DataLoader

from services.udpipe_model import UDPipeModel
from services.config import PATH_TO_SOURCE_UDPIPE

from augment.common import TextDataset, ThreadedWriter, set_random_seed
from augment.dropout.dropouter import Dropouter
from augment.token_shuffling.token_shuffler import TokenShuffler
from augment.mask.masker import Masker
from augment.translation.back_translator import (
    BackTranslator,
    HelsinkiCTranslateTranslator,
)
from augment.common import markov_process


INPUT_TEXTS_PATH = (
    "local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json"
)
OUTPUT_TEXTS_PATH = "local_datasets/augmented/all_together/augmented_sentences_3.jsonl"

BATCH_SIZE = 128
NUM_WORKERS = 2
NUM_AUGMENTATIONS = 9
MARKOV_P = 0.75
SEED = 42
GPU_ID = int(os.getenv("AUGMENT_GPU_ID", "0"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

def main():
    set_random_seed(SEED)
    texts_dataset = TextDataset(INPUT_TEXTS_PATH)
    dataloader = DataLoader(
        texts_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Open file for writing results immediately
    writer = ThreadedWriter(OUTPUT_TEXTS_PATH)

    udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)
    shuffler = Dropouter(udpipe_model)
    token_shuffler = TokenShuffler(udpipe_model)
    masker = Masker(
        "Goader/modern-liberta-large",
        udpipe_model,
        seed=SEED,
        batch_size=1024,
        device_index=GPU_ID,
    )
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

    all_augmenters = [shuffler, token_shuffler, masker, translator]

    selected_augmenters = []

    try:
        logging.info("Dataset augmentation began")

        for batch in tqdm(dataloader, desc="Augmenting"):
            augmented_texts = batch["sentence"]
            map_to_original = {text: text for text in augmented_texts}
            final_augmented_texts = None

            augmenters_list = []
            while augmenter := markov_process(
                all_augmenters, p=MARKOV_P, prev_augs_count=len(augmenters_list)
            ):
                augmenters_list.append(augmenter.__class__.__name__)
                new_augmented_texts = augmenter(
                    augmented_texts,
                    n=1 if len(augmenters_list) >= 2 else NUM_AUGMENTATIONS,
                )
                augmented_texts = []

                # Keep only descendants produced by the current/final stage.
                # Accumulate all branches that map back to the same original
                # sentence instead of replacing earlier branches.
                final_augmented_texts = {}
                for original, augmented_list in new_augmented_texts.items():
                    augmented_list = list(
                        filter(None, augmented_list)
                    )  # filter out empty values
                    augmented_texts.extend(augmented_list)

                    original = map_to_original[original]
                    final_augmented_texts.setdefault(original, []).extend(
                        augmented_list
                    )

                    map_to_original.update({aug: original for aug in augmented_list})

            writer.write(batch, final_augmented_texts)
            selected_augmenters.append(augmenters_list)

        logging.info("Dataset augmentation finished")
    finally:
        logging.info("Closing writer. Waiting for all data to be written...")
        writer.close()

        with open("selected_augmenters_log.json", "w") as f:
            json.dump(selected_augmenters, f, indent=4)


if __name__ == "__main__":
    main()
