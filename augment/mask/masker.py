"""
Run: python3 -m augment.mask.masker
"""

import math
import random

import torch
from transformers import pipeline

from services.udpipe_model import UDPipeModel
from augment.common import Augmenter, set_random_seed


class Masker(Augmenter):
    def __init__(
        self,
        model_name_or_path: str,
        udpipe: UDPipeModel | str,
        rate=0.15,
        seed=42,
        batch_size=32,
        device_index=0,
    ):
        self.rate = rate
        self.batch_size = batch_size
        self.unmasker = pipeline(
            "fill-mask",
            model=model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device=device_index if torch.cuda.is_available() else -1,
        )
        self.mask_token = self.unmasker.tokenizer.mask_token
        self.udpipe_model = UDPipeModel(udpipe) if isinstance(udpipe, str) else udpipe

        set_random_seed(seed)

    def _unsmask_sentences(self, sentences: list[str]):
        processing_queue = sentences.copy()

        while True:
            # Identify indices that still need processing
            active_indices = [
                i for i, s in enumerate(processing_queue) if self.mask_token in s
            ]

            if not active_indices:
                break

            # Process in chunks of self.batch_size
            total_active = len(active_indices)

            for start_idx in range(0, total_active, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_active)
                batch_indices = active_indices[start_idx:end_idx]
                batch_texts = [processing_queue[i] for i in batch_indices]

                # Inference on the batch
                # Pipeline returns list of results (one per input string)
                results = self.unmasker(batch_texts, batch_size=len(batch_texts))

                # Update texts
                for idx_in_queue, result in zip(batch_indices, results):
                    if (
                        isinstance(result, list)
                        and len(result) > 0
                        and isinstance(result[0], list)
                    ):
                        first_mask_preds = result[0]
                    elif isinstance(result, list):
                        first_mask_preds = result
                    else:
                        first_mask_preds = [result]  # Should not happen in fill-mask

                    probs = torch.tensor([res["score"] for res in first_mask_preds])
                    probs = torch.softmax(probs, dim=0)
                    sampled_idx = torch.multinomial(probs, num_samples=1).item()
                    predicted_token = first_mask_preds[sampled_idx]["token_str"].strip()

                    processing_queue[idx_in_queue] = processing_queue[
                        idx_in_queue
                    ].replace(self.mask_token, predicted_token, 1)

        return processing_queue

    def _mask_sentence(self, sentence: str, n=1):
        tokenized = self.udpipe_model.tokenize(sentence)
        words = []
        for tok in tokenized:
            self.udpipe_model.tag(tok)
            words.extend(tok.words[1:])

        masked_sentences = []
        for _ in range(n):  # n distinct maskings
            masked_sentence = []

            for i, word in enumerate(words):
                # Preserve original spacing after the word, except for the first word
                if i > 0:
                    masked_sentence.append(words[i - 1].getSpacesAfter())

                # Don't mask punctuation and apply masking with the specified rate
                if word.upostag in {"PUNCT"} or random.random() > self.rate:
                    masked_sentence.append(word.form)
                else:
                    masked_sentence.append(self.mask_token)

            # n same maskings to be unsmasked differently
            masked_sentences.extend(["".join(masked_sentence)] * n)

        return masked_sentences

    def __call__(self, sentences: list[str], n: int = 1) -> dict[str, list[str]]:
        """Augment sentences by masking and unmasking words.

        Args:
            sentences (list[str]): batch of sentences to augment
            n (int, optional): number of augmentations per sentence. Defaults to 1.
        """
        n = int(math.sqrt(n))

        all_masked = []
        sentence_slices = {}
        
        for sentence in sentences:
            masked = self._mask_sentence(sentence, n)
            sentence_slices[sentence] = (len(all_masked), len(all_masked) + len(masked))
            all_masked.extend(masked)

        unmasked = self._unsmask_sentences(all_masked)

        return {s: unmasked[start:end] for s, (start, end) in sentence_slices.items()}


if __name__ == "__main__":
    from services.config import PATH_TO_SOURCE_UDPIPE

    udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)
    shuffler = Masker("Goader/modern-liberta-large", udpipe=udpipe_model, rate=0.15)

    # Example usage
    sentences = [
        "Він був дуже щасливий, коли отримав цю новину.",
        "Це був найкращий день у його житті!",
    ]
    augmented = shuffler(sentences, n=4)

    for original, variations in augmented.items():
        print(f"Original: {original}")

        for i, aug in enumerate(variations, 1):
            print(f"\tAugmentation {i}: {aug}")

        print()
