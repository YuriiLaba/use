"""
Run: python3 -m augment.mask.mask_definitions
"""

import logging

from tqdm import tqdm
from torch.utils.data import DataLoader

from services.udpipe_model import UDPipeModel
from services.config import PATH_TO_SOURCE_UDPIPE

from augment.common import TextDataset, ThreadedWriter, set_random_seed
from augment.mask.masker import Masker

INPUT_TEXTS_PATH = (
    "local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json"
)
OUTPUT_TEXTS_PATH = (
    "local_datasets/augmented/mask/augmented_sentences_definitions.jsonl"
)

BATCH_SIZE = 256
NUM_WORKERS = 2
NUM_AUGMENTATIONS = 4
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)


def main():
    set_random_seed(SEED)
    texts_dataset = TextDataset(INPUT_TEXTS_PATH, load_definitions=True)
    dataloader = DataLoader(
        texts_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Open file for writing results immediately
    writer = ThreadedWriter(OUTPUT_TEXTS_PATH, include_target_word_check=False)

    udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)
    shuffler = Masker(
        "Goader/modern-liberta-large",
        udpipe=udpipe_model,
        rate=0.15,
        seed=SEED,
        batch_size=128,
    )

    try:
        logging.info("Dataset augmentation began")

        for batch in tqdm(dataloader, desc="Augmenting"):
            augmented_texts = shuffler(batch["sentence"], n=NUM_AUGMENTATIONS)
            writer.write(batch, augmented_texts)

        logging.info("Dataset augmentation finished")
    finally:
        logging.info("Closing writer. Waiting for all data to be written...")
        writer.close()


if __name__ == "__main__":
    main()
