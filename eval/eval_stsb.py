import pandas as pd

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, evaluation
from services.config import get_list, get_value


model_name_or_path = get_value(
    "models", "base_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
benchmark_hf = get_value("evaluation", "sts_dataset", "anikol12/STSB-UK")

models = get_list("baseline_models", "wsd")


def main(model_paths=None, device=None):
    # load dataset
    eval_dataset = load_dataset(benchmark_hf, split="train")

    sentences1 = eval_dataset["sentence1"]
    sentences2 = eval_dataset["sentence2"]
    scores = [
        1.0 if s1 == s2 else s
        for s1, s2, s in zip(sentences1, sentences2, eval_dataset["score"])
    ]

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1, sentences2, scores, show_progress_bar=True
    )

    rows = []

    for model_name_or_path in (models if model_paths is None else model_paths):
        print(f"Evaluating {model_name_or_path}")

        model = SentenceTransformer(model_name_or_path, device=device)
        results = evaluator(model)

        rows.append(
            {
                "model": model_name_or_path,
                "pearson_cosine": results["pearson_cosine"] * 100,
                "spearman_cosine": results["spearman_cosine"] * 100,
            }
        )

    # ✅ build table
    df = (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values("spearman_cosine", ascending=False)
    )

    print("\nFinal table:")
    print(df.round(4))

    # optional save
    df.to_csv("sts_results.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Ukrainian STS-B correlations.")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--device",
        default=get_value("evaluation", "device", "cpu"),
        help="For example cuda:0 or cpu.",
    )
    args = parser.parse_args()
    main(args.models, args.device)
