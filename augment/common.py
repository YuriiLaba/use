import json
import os
import queue
import random
import threading

import numpy as np
import torch
from torch.utils.data import Dataset

from services.udpipe_model import UDPipeModel
from services.utils_embedding_calculation_v2 import _find_target_word_in_sentence
from services.config import PATH_TO_SOURCE_UDPIPE

from abc import ABC, abstractmethod


def set_random_seed(seed: int) -> None:
    """Seed the random number generators used by augmentation code."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Augmenter(ABC):
    @abstractmethod
    def __call__(self, sentences: list[str], n: int = 1) -> dict[str, list[str]]:
        pass


class TextDataset(Dataset):
    def __init__(
        self, texts_file: str, last_sentence: str = None, load_definitions=False
    ):
        with open(texts_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.texts = []

        if load_definitions:
            for lemma, meanings in data.items():
                for meaning in meanings.values():
                    self.texts.extend(
                        [
                            {"lemma": lemma, "sentence": gl}
                            for gl in meaning["meaning"]["gloss"]
                        ]
                    )
        else:
            for lemma, meanings in data.items():
                for meaning in meanings.values():
                    self.texts.extend(
                        [
                            {"lemma": lemma, "sentence": sentence["sentence"]}
                            for sentence in meaning["sentences"]
                        ]
                    )

            if last_sentence is not None:
                try:
                    for index, item in enumerate(self.texts):
                        if item["sentence"] == last_sentence:
                            last_sentence = index
                            self.texts = self.texts[last_sentence + 1 :]
                            break
                except ValueError:
                    pass  # last_sentence not found, keep all texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class ThreadedWriter:
    def __init__(self, filepath, include_target_word_check=True):
        filepath_dir = os.path.dirname(filepath)
        if not os.path.exists(filepath_dir):
            os.makedirs(filepath_dir)

        self.f = open(filepath, "w", encoding="utf-8")
        self.queue = queue.Queue()
        self.finished = False
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.include_target_word_check = include_target_word_check

        if include_target_word_check:
            self.udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)

        self.thread.start()

    def write(self, original_texts, batch_texts):
        self.queue.put((original_texts, batch_texts))

    def _write_loop(self):
        while not self.finished or not self.queue.empty():
            try:
                batch, augmented = self.queue.get(timeout=1)
                # Write chunk at once to minimize disk syscalls
                write_strs = []

                # reform the batch
                original_sentences_dict = {}
                for sentence, lemma in zip(batch["sentence"], batch["lemma"]):
                    try:
                        augmented_sentences = set(augmented[sentence])
                        original_sentences_dict[sentence] = {
                            "lemma": lemma,
                            "augmented": augmented_sentences,
                        }
                    except KeyError:
                        print(f"No augmented sentences for '{sentence}'")
                        original_sentences_dict[sentence] = {
                            "lemma": lemma,
                            "augmented": [],
                        }

                for sentence in original_sentences_dict:
                    lemma = original_sentences_dict[sentence]["lemma"]
                    augmented_texts = original_sentences_dict[sentence]["augmented"]

                    augmented_texts = [
                        i
                        for i in augmented_texts
                        if i.strip()
                        != sentence.strip()  # not the same as not-augmented sentence
                        and i.strip() != ""  # not empty
                        and "<unk>" not in i  # not containing unknown tokens
                    ]

                    if self.include_target_word_check:
                        augmented_texts = [
                            augmented_sentence
                            for augmented_sentence in augmented_texts
                            if _find_target_word_in_sentence(
                                self.udpipe_model, augmented_sentence, lemma
                            )
                            is not None
                        ]

                    write_strs.append(
                        json.dumps(
                            {"sentence": sentence, "augmented": augmented_texts},
                            ensure_ascii=False,
                        )
                    )

                self.f.write("\n".join(write_strs) + "\n")
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in writing thread: {e}")
                continue

        self.f.close()

    def close(self):
        self.finished = True
        self.thread.join()


def markov_process(
    augmenters: list[Augmenter], p=0.5, prev_augs_count=0, min_augs=1
) -> Augmenter | None:
    """
    Randomly selects one of the augmenters based on predefined probabilities.
    Returns the selected augmenter or None if no augmenter is selected.
    """
    if not augmenters:
        return None

    # allow stopping only after reaching minimum
    if prev_augs_count >= min_augs and random.random() < p:
        return None

    return random.choice(augmenters)
