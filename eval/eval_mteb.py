"""
Run: python3 -m eval.eval_mteb
"""

import logging
import os
import warnings
import simplejson as json
from tqdm import tqdm
from services.config import get_int, get_list, get_value

os.environ.setdefault(
    "MTEB_CACHE",
    os.path.abspath(get_value("evaluation", "mteb_cache", "cache/mteb")),
)

import mteb
from mteb.cache import ResultCache

from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ALLOWED_MODALITIES = get_list("mteb", "modalities", ["text"])
TASK_NAMES = get_list("mteb", "tasks")
NUM_PROC = get_int("evaluation", "mteb_num_proc", 1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


models = get_list("baseline_models", "wsd")
DEVICE = get_value("evaluation", "mteb_device", "cuda:0")


def main(model_paths=None, device=None):
    # get tasks that support ukrainian language
    ukrainian_tasks = mteb.get_tasks(tasks=TASK_NAMES)
    ukrainian_tasks = [
        task
        for task in ukrainian_tasks
        if task.metadata.modalities == ALLOWED_MODALITIES
    ]

    for task in ukrainian_tasks:
        subset_value = get_value("mteb_subsets", task.metadata.name, "")
        if subset_value:
            task.hf_subsets = [subset.strip() for subset in subset_value.split(",")]

    print(f"Found {len(ukrainian_tasks)} tasks that support Ukrainian language.")

    for model_name_or_path in tqdm(
        models if model_paths is None else model_paths, desc="Evaluating models"
    ):
        model = SentenceTransformer(
            model_name_or_path, device=DEVICE if device is None else device
        )

        # evaluate model on ukrainian tasks with caching
        cache = ResultCache(f"./cache/mteb_{model_name_or_path.replace('/', '_')}")
        results = mteb.evaluate(
            model,
            tasks=ukrainian_tasks,
            cache=cache,
            num_proc=NUM_PROC,
            prediction_folder=f"./eval/mteb_prediction/{model_name_or_path.replace('/', '_')}",
        )

        # save results to a file
        results_dir = f"./eval/mteb_results/{model_name_or_path.replace('/', '_')}"
        os.makedirs(results_dir, exist_ok=True)

        for result in results.task_results:
            result_file = os.path.join(results_dir, f"{result.task_name}_results.json")

            with open(result_file, "w") as f:
                json.dump(result.scores, f, indent=2, ignore_nan=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate text MTEB tasks supporting Ukrainian."
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--device", default=None, help="Overrides the script's CASE-specific GPU."
    )
    args = parser.parse_args()
    main(args.models, args.device)
