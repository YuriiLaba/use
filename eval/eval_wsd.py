import logging
from functools import partial

from services.poolings import PoolingStrategy
from services.udpipe_model import UDPipeModel
from services.utils_results import results_reports
from services.utils_data import read_homonym_benchmark
from services.word_sense_detector import WordSenseDetector
from services.prediction_strategies import PredictionStrategy
from services.config import PATH_TO_SOURCE_UDPIPE, HOMONYM_BENCHMARK_PATH, get_value
from services.utils_results import prediction_accuracy

import torch
from transformers import AutoTokenizer, AutoModel


DEVICE = get_value("evaluation", "device", "cpu")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler("eval_wsd.log"), logging.StreamHandler()],
)

# Hugging Face Hub uses httpx/httpcore for model downloads. Their request-level
# INFO logs are noisy during batch evaluation; keep evaluator logs visible.
for _logger_name in (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "transformers",
    "urllib3",
):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def evaluate_wsd(
    model_path: str,
    model_tokenizer_path: str | None = None,
    verbose: bool = True,
    benchmark_path: str = HOMONYM_BENCHMARK_PATH,
    device: str = DEVICE,
    anchor_pooling: str = "target",
    return_predictions: bool = False,
    model_revision: str | None = None,
):
    """Return accuracy, or the same per-sense predictions for further analysis."""
    if anchor_pooling not in ("sentence", "target"):
        raise ValueError("anchor_pooling must be 'sentence' or 'target'")
    if model_tokenizer_path is None:
        model_tokenizer_path = model_path

    logger.info("Loading evaluation dataset...")
    data = read_homonym_benchmark(benchmark_path)

    logger.info("Loading fine-tuned model...")
    revision_kwargs = {"revision": model_revision} if model_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_tokenizer_path, trust_remote_code=True, **revision_kwargs
    )
    model = AutoModel.from_pretrained(
        model_path, output_hidden_states=True, trust_remote_code=True, **revision_kwargs
    )
    model = model.to(device).eval()

    logger.info("Loading UDPipe model...")
    udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)

    logger.info("Running Word Sense Detection | anchor pooling: %s", anchor_pooling)
    word_sense_detector = WordSenseDetector(
        pretrained_model=model,
        tokenizer=tokenizer,
        udpipe_model=udpipe_model,
        evaluation_dataset=data,
        pooling_strategy=PoolingStrategy.mean_pooling,
        prediction_strategy=partial(
            PredictionStrategy.max_sim_across_all_examples,
            anchor_pooling=anchor_pooling,
        ),
        device=torch.device(device),
    )
    evaluation_dataset_pd = word_sense_detector.run()
    scored_rows = len(evaluation_dataset_pd.dropna())
    logger.info(
        "Dictionary-sense rows | total=%d | scored=%d | excluded=%d",
        len(evaluation_dataset_pd), scored_rows,
        len(evaluation_dataset_pd) - scored_rows,
    )

    if verbose:
        results_reports(evaluation_dataset_pd, udpipe_model)

    if return_predictions:
        return evaluation_dataset_pd
    return prediction_accuracy(evaluation_dataset_pd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate dictionary-sense WSD accuracy.")
    parser.add_argument(
        "--model-path",
        default=get_value(
            "models", "base_model",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ),
    )
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--benchmark-path", default=HOMONYM_BENCHMARK_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--anchor-pooling", choices=("sentence", "target"), default="target",
        help="Pooling for benchmark sentences; definitions always use full-sentence mean pooling. Default: target.",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip POS/gloss reports and badly_predicted.csv; still print accuracy.",
    )
    args = parser.parse_args()
    accuracy = evaluate_wsd(
        args.model_path,
        args.tokenizer_path,
        verbose=not args.no_reports,
        benchmark_path=args.benchmark_path,
        device=args.device,
        anchor_pooling=args.anchor_pooling,
    )
    print(f"WSD accuracy (retained dictionary-sense rows): {accuracy:.6f}")
