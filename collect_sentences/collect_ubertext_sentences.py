"""
Collect sentences from the UberText 2.0 dataset that contain specified lemmas of interest.
The collected sentences are saved in a JSON file for further processing.

Run: python3 collect_sentences.collect_ubertext_sentences.py
"""

import json
import string
import argparse
import multiprocessing as mp
from collections import defaultdict

import smart_open
from tqdm import tqdm
from langdetect import detect

from services.config import (
    PATH_TO_SOURCE_DATASET,
    PATH_TO_SOURCE_UDPIPE,
    PATH_TO_SAVE_GATHERED_DATASET,
    PATH_TO_LEMMAS_OF_INTEREST,
    NUMBER_OF_EXAMPLES_TO_GATHER,
)

CHUNK_SIZE = 128
SAVE_EVERY_N_SENTENCES = 50_000

TOKENIZER = None  # udpipe or spacy
tokenizer_model = None


class CollectUberTextSentences:
    """
    Collect sentences from the UberText 2.0 dataset that contain specified lemmas of interest.
    """

    def __init__(
        self,
        path_to_ubertext: str,
        path_to_save_gathered_dataset: str,
        path_to_lemmas_of_interest: str,
        number_of_examples_to_gather: int,
    ):
        self.path_to_ubertext = path_to_ubertext
        self.path_to_save_gathered_dataset = path_to_save_gathered_dataset
        self.path_to_lemmas_of_interest = path_to_lemmas_of_interest
        self.number_of_examples_to_gather = number_of_examples_to_gather

        with open(self.path_to_lemmas_of_interest) as f:
            unique_lemmas = f.readlines()

        self.lemmas_of_interest = set([i.replace("\n", "") for i in unique_lemmas])
        self.current_batch = defaultdict(list)
        self.sentences_in_current_batch = 0
        self.total_collected_count = 0

    def _normalize_text_udpipe(self, line: str) -> set[str]:
        """
        Normalize text using UDPipe tokenizer and return a set of lemmas in the line.

        Args:
            line (str): The input text line to normalize.

        Returns:
            set[str]: A set of lemmas found in the input line.
        """
        global tokenizer_model
        tokens = tokenizer_model.tokenize(line)

        for tok_sent in tokens:
            tokenizer_model.tag(tok_sent)

        return {w.lemma for w in tok_sent.words[1:]}

    def _normalize_text_spacy(self, line: str) -> set[str]:
        """
        Normalize text using spacy tokenizer and return a set of lemmas in the line.

        Args:
            line (str): The input text line to normalize.

        Returns:
            set[str]: A set of lemmas found in the input line.
        """
        global tokenizer_model
        doc = tokenizer_model(line)
        return {token.lemma_ for token in doc}

    def _normalize_text(self, line: str) -> set[str]:
        """Normalize text based on the selected tokenizer and return a set of lemmas in the line.

        Args:
            line (str): The input text line to normalize.

        Raises:
            ValueError: If an unsupported tokenizer is specified.

        Returns:
            set[str]: A set of lemmas found in the input line.
        """
        global TOKENIZER
        if TOKENIZER == "udpipe":
            return self._normalize_text_udpipe(line)
        elif TOKENIZER == "spacy":
            return self._normalize_text_spacy(line)
        else:
            raise ValueError("Unsupported tokenizer specified.")

    def _append_batch_to_jsonl(self):
        """Append the current batch of collected sentences to the JSONL file."""
        if not self.current_batch:
            return

        print(
            f"\n[✓] Appending batch of {self.sentences_in_current_batch:,} sentences to disk..."
            f"Total collected so far: {self.total_collected_count:,}"
        )

        with open(self.path_to_save_gathered_dataset, "a", encoding="utf-8") as f:
            line = json.dumps(self.current_batch, ensure_ascii=False)
            f.write(line + "\n")

        self.current_batch.clear()
        self.sentences_in_current_batch = 0

    def _process_ubertext_line(self, line: str) -> tuple[set[str], str] | None:
        """Process a single line from the UberText dataset to check for lemmas of interest.

        Args:
            line (str): A single line from the UberText dataset.

        Returns:
            tuple[set[str], str] | None: A tuple containing the set of found lemmas and the original line
            if any lemmas of interest are found; otherwise, None.
        """
        line = line.replace("\n", "").replace("\xa0", " ").strip()
        line = " ".join(line.split())

        line_ = line.translate(
            str.maketrans("", "", string.punctuation)
        )  # remove punctuation from a processing line

        lint_split = line_.split(" ")
        if len(lint_split) <= 7 or len(lint_split) >= 16:
            return None

        if line.count("*") >= 4:
            return None

        if line.count("—") >= 5:
            return None

        if sum(c.isdigit() for c in line) >= 10:
            return None

        try:
            if detect(line) != "uk":
                return None
        except Exception as e:
            return None

        nomalize_line = self._normalize_text(line)
        intersection = self.lemmas_of_interest.intersection(nomalize_line)

        if len(intersection) > 0:
            return (intersection, line)

        return None

    def _collect_raw_lemma_examples_dataset(self):
        """Collect raw lemma examples dataset from the UberText 2.0 dataset."""
        with (
            mp.Pool(processes=mp.cpu_count() // 2) as pool,
            tqdm(
                total=self.number_of_examples_to_gather,
                desc="Collected sentences from UberText 2.0",
            ) as pbar,
        ):
            try:
                with smart_open.open(self.path_to_ubertext, encoding="utf-8") as file:
                    for result in tqdm(
                        pool.imap_unordered(
                            self._process_ubertext_line, file, chunksize=CHUNK_SIZE
                        ),
                        desc="Processed UberText 2.0 lines",
                    ):
                        if result is not None:
                            lemmas, sentence = result

                            for lemma in lemmas:
                                self.current_batch[lemma].append(sentence)

                            self.sentences_in_current_batch += 1
                            self.total_collected_count += 1
                            pbar.update(1)

                            # Trigger Batch Save
                            # fmt: off
                            if self.sentences_in_current_batch >= SAVE_EVERY_N_SENTENCES:
                                self._append_batch_to_jsonl()

                        # fmt: off
                        if self.total_collected_count >= self.number_of_examples_to_gather:
                            break

            except KeyboardInterrupt:
                print("\n[!] Manual stop. Saving remaining data...")
            finally:
                self._append_batch_to_jsonl()  # Final flush

    def collect_sentences(self):
        """
        Collect sentences containing lemmas of interest from the UberText 2.0 dataset and save them to a JSON file.
        """
        self._collect_raw_lemma_examples_dataset()
        print(f"[✓] Done. Total collected: {self.total_collected_count}")


def load_tokenizer_model(tokenizer_type: str):
    """Load the specified tokenizer model.

    Args:
        tokenizer_type (str): The type of tokenizer to load ("udpipe" or "spacy").

    Raises:
        ValueError: If an unsupported tokenizer is specified.
    """
    global tokenizer_model
    if tokenizer_type == "udpipe":
        from services.udpipe_model import UDPipeModel

        tokenizer_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)
    elif tokenizer_type == "spacy":
        import spacy

        tokenizer_model = spacy.load("uk_core_news_sm", enable=["lemmatizer"])
    else:
        raise ValueError("Unsupported tokenizer specified.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    global TOKENIZER, tokenizer_model, CHUNK_SIZE, SAVE_EVERY_N_SENTENCES
    parser = argparse.ArgumentParser(
        description="Collect sentences from UberText 2.0 containing specified lemmas."
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        default=PATH_TO_SOURCE_DATASET,
        help="Path to the UberText 2.0 dataset.",
    )
    parser.add_argument(
        "--save_dataset",
        type=str,
        default=PATH_TO_SAVE_GATHERED_DATASET,
        help="Path to save the gathered dataset.",
    )
    parser.add_argument(
        "--lemmas_file",
        type=str,
        default=PATH_TO_LEMMAS_OF_INTEREST,
        help="Path to the file containing lemmas of interest.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=NUMBER_OF_EXAMPLES_TO_GATHER,
        help="Number of examples to gather.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["udpipe", "spacy"],
        default="udpipe",
        help="Tokenizer to use: 'udpipe' or 'spacy'.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=CHUNK_SIZE,
        help="Chunk size for multiprocessing.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=SAVE_EVERY_N_SENTENCES,
        help="Number of sentences to collect before saving to disk.",
    )

    args = parser.parse_args()

    CHUNK_SIZE = args.chunk_size
    SAVE_EVERY_N_SENTENCES = args.save_every
    TOKENIZER = args.tokenizer
    load_tokenizer_model(TOKENIZER)

    return args


def main(args: argparse.Namespace):
    """Main function to initiate the sentence collection process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    collector = CollectUberTextSentences(
        args.source_dataset,
        args.save_dataset,
        args.lemmas_file,
        args.num_examples if args.num_examples > 0 else float("inf"),
    )
    collector.collect_sentences()


if __name__ == "__main__":
    main(parse_args())
