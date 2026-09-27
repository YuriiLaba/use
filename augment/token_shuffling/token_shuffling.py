"""
Run: python3 -m augment.token_shuffling.token_shuffling
"""

import logging

from tqdm import tqdm
from torch.utils.data import DataLoader

from services.udpipe_model import UDPipeModel
from services.config import PATH_TO_SOURCE_UDPIPE, get_int, get_value

from augment.common import TextDataset, ThreadedWriter, set_random_seed
from augment.token_shuffling.token_shuffler import TokenShuffler

INPUT_TEXTS_PATH = get_value("augmentation", "input_dataset")
OUTPUT_TEXTS_PATH = get_value("augmentation", "shuffling_output")
BATCH_SIZE = get_int("augmentation", "batch_size", 256)
NUM_WORKERS = get_int("augmentation", "num_workers", 2)
NUM_AUGMENTATIONS = get_int("augmentation", "num_variants", 4)
SEED = get_int("augmentation", "seed", 42)

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
    shuffler = TokenShuffler(udpipe_model)

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
