from transformers import AutoModel, AutoTokenizer
from services.poolings import PoolingStrategy
from services.utils_data import read_homonym_benchmark
from services.utils_embedding_calculation_v2 import (
    get_target_word_embedding,
    get_context_embedding,
)
from services.udpipe_model import UDPipeModel
from services.config import PATH_TO_SOURCE_UDPIPE

import torch
from torch.nn import functional as F
import numpy as np


def load_models(model_name: str, device: str):
    print("Loading UDPipe model...")
    udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)

    print("Loading fine-tuned model...")
    model = (
        AutoModel.from_pretrained(model_name, output_hidden_states=True)
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, udpipe_model


def load_dataset(path: str, target_lemma: str):
    print("Reading and processing evaluation dataset...")
    data = read_homonym_benchmark(path)
    data = data[data["lemma"] == target_lemma]
    return data


def compute_target_embedding(
    model,
    tokenizer,
    udpipe_model,
    pooling_strategy,
    lemma,
    sentence,
    device,
):
    print(f"Calculating embedding for '{lemma}' in sentence:\n{sentence}")
    embedding = get_target_word_embedding(
        model,
        tokenizer,
        udpipe_model,
        pooling_strategy,
        lemma,
        sentence,
        device,
    )
    return torch.tensor(np.array(embedding))


def find_best_meaning(
    model,
    tokenizer,
    data,
    pooling_strategy,
    target_embedding,
    device,
):
    print("Calculating similarities with each meaning...")

    max_sim = -1
    correct_context = None
    correct_gloss = None

    for index, row in data.iterrows():
        glosses = row["gloss"]
        max_sub_sim = -1

        for i, gloss in enumerate(glosses):
            print(f"Meaning #{index + 1}, gloss #{i + 1}: {gloss}")

            context_embedding = get_context_embedding(
                model, tokenizer, pooling_strategy, gloss, device
            )
            context_embedding = torch.tensor(np.array(context_embedding))

            similarity = F.cosine_similarity(
                target_embedding, context_embedding, dim=0
            ).item()

            print(f"Similarity: {similarity:.4f}")

            if similarity > max_sub_sim and similarity > max_sim:
                correct_gloss = gloss

            max_sub_sim = max(max_sub_sim, similarity)

        if max_sub_sim > max_sim:
            max_sim = max_sub_sim
            correct_context = glosses

        print()

    return correct_context, correct_gloss, max_sim


def main():
    SUM_PATH = "./datasets_pre_defined/homonym_dictionary_clean.jsonl"
    MODEL_NAME = "victormuryn/mpnet-use-markov-pt"
    TARGET_LEMMA = "коса"
    DEVICE = "cuda:0"
    SENTENCES = [
        "Якраз під старою вишнею стояла дівчина, хороша, як зоря ясна; руса коса нижче пояса",
        "Човен повернув за гострий ріг піскуватої коси і вступив у Чорне море",
    ]

    pooling_strategy = PoolingStrategy.mean_pooling

    model, tokenizer, udpipe_model = load_models(MODEL_NAME, DEVICE)
    data = load_dataset(SUM_PATH, TARGET_LEMMA)

    for sentence in SENTENCES:
        print()
        target_embedding = compute_target_embedding(
            model,
            tokenizer,
            udpipe_model,
            pooling_strategy,
            TARGET_LEMMA,
            sentence,
            DEVICE,
        )

        context, gloss, similarity = find_best_meaning(
            model,
            tokenizer,
            data,
            pooling_strategy,
            target_embedding,
            DEVICE,
        )

        print("\n=== RESULT ===")
        print(f"Most likely meaning: {context}")
        print(f'Gloss: "{gloss}"')
        print(f"Similarity: {similarity:.4f}")
        print()


if __name__ == "__main__":
    main()
