"""
Run: python3 -m services.trainer.trainer
"""

import os
import random
from ast import literal_eval
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import GroupShuffleSplit

import wandb
from tqdm.auto import tqdm

from services.trainer.data_factory import DataFactory
from services.trainer.training_config import TrainingConfig

from services.udpipe_model import UDPipeModel
from services.word_sense_detector import WordSenseDetector
from services.utils_results import prediction_accuracy
from services.poolings import PoolingStrategy
from services.prediction_strategies import PredictionStrategy
from services.trainer.utlis import report_gpu, AverageMeter
from services.trainer.losses import TripletLoss, NTXentLoss, MNRLoss
from services.config import PATH_TO_SOURCE_UDPIPE
from eval.eval_wsd import evaluate_wsd

from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Projection head for contrastive learning (SimCLR/NT-Xent style models)
class ContrastiveModel(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super().__init__()
        self.backbone = backbone  # This is your AutoModel
        self.hidden_dim = backbone.config.hidden_size

        # SimCLR/NT-Xent standard: MLP with ReLU
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, projection_dim),
        )

    def forward(self, input_ids, attention_mask):
        # 1. Get Transformer outputs
        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        # 2. Extract Mean Pooling (or token-level pooling)
        # Using your existing mean pooling logic here
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

        # 3. Project to the contrastive space
        projected = self.projection_head(pooled)
        return projected


class Trainer:
    def __init__(self, config, device=None):
        self.config = TrainingConfig.from_config(config) if isinstance(config, str) else config

        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        self.global_step = 0  # used for wandb logging steps
        self._init_logger()

        os.makedirs(self.config.path_to_save_fine_tuned_model, exist_ok=True)

        self._init_model()

        self._init_loss()
        self._setup_data()

        self.train_avg_meter = AverageMeter("train_loss")
        self.min_val_loss = float("inf")
        self.rounds_count = 0

        self._setup_optimizer()

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01
        )

        # Warmup logic: needs the total number of training steps
        if self.config.warmup_ratio > 0:
            total_steps = len(self.train_loader) * self.config.num_epochs
            warmup_steps = int(total_steps * self.config.warmup_ratio)

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            self.scheduler = None

    def _init_logger(self):
        if self.config.log_to_wandb:
            # gather params from the config section to pass to W&B
            params = {}
            for key in self.config.__dataclass_fields__.keys():
                params[key] = getattr(self.config, key)

            wandb_project_name = self.config.wandb_project_name
            wandb_entity = self.config.wandb_entity

            # initialize the run (expects WANDB_API_KEY env var or w&b login)
            init_kwargs = {"project": wandb_project_name, "config": params}
            if wandb_entity:
                init_kwargs["entity"] = wandb_entity

            # keep run name deterministic-ish
            run_name = self.config.wandb_run_name or datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            init_kwargs["name"] = run_name

            self.wandb_run = wandb.init(**{k: v for k, v in init_kwargs.items() if v})
            # record a few root-level fields as config (dataset sizes are filled later)
            self.wandb_run.config.update(
                {
                    "epochs": self.config.num_epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "early_stopping": self.config.early_stopping,
                },
                allow_val_change=True,
            )
            self.run_id = wandb.run.id
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run = None

    def _init_model(self):
        self.udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.config.model_to_fine_tune,
            output_hidden_states=True,
            trust_remote_code=True,
        ).to(self.device)

        layers_to_unfreeze = self.config.layers_to_unfreeze

        if layers_to_unfreeze > 0:
            # freeze model weights except n last layers and pooler
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.encoder.layer[-layers_to_unfreeze:].parameters():
                param.requires_grad = True

            for param in self.model.pooler.parameters():
                param.requires_grad = True

    def _init_loss(self):
        loss_type = self.config.loss_type
        if "triplet_loss" in loss_type:
            self.loss = TripletLoss(
                model=self.model,
                margin=0.2 if loss_type == "triplet_loss_cosine" else 1.0,
                p=2,
                loss_type=loss_type,
                pool_targets=self.config.pool_targets,
                use_both_poolings=self.config.use_both_poolings,
            )
        elif loss_type == "nt_xent_loss":
            self.loss = NTXentLoss(model=self.model, temperature=0.05)
        elif loss_type == "mnr_loss":
            self.loss = MNRLoss(
                model=self.model,
                temperature=0.05,
                pool_targets=self.config.pool_targets,
                use_both_poolings=self.config.use_both_poolings,
            )
        else:
            raise NotImplementedError(f"Loss {loss_type} is not implemented yet")

    def _setup_data(self):
        # 1. Load raw dataframes
        hf_dataset = getattr(self.config, "hf_dataset", None)
        hf_subset = getattr(self.config, "hf_subset", None)

        if hf_dataset and hf_subset:
            from datasets import load_dataset

            hf_token = os.environ.get("HF_TOKEN")
            ds = load_dataset(
                hf_dataset,
                hf_subset,
                token=hf_token,
                split="train",
            )
            df = ds.to_pandas()
        else:
            df = pd.read_csv(self.config.train_data_path)

        if "group_id" not in df.columns:
            raise ValueError(
                "Training data must contain 'group_id' for leakage-safe splitting. "
                "Regenerate the triplet CSV with the updated form_triplets.py."
            )

        if df["group_id"].isna().any():
            raise ValueError("Training data contains missing group_id values.")

        if df["group_id"].nunique() < 2:
            raise ValueError("At least two groups are required for train/eval splitting.")

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config.validation_split,
            random_state=self.config.split_seed,
        )
        train_indices, eval_indices = next(
            splitter.split(df, groups=df["group_id"])
        )
        train_df = df.iloc[train_indices]
        eval_df = df.iloc[eval_indices]

        train_groups = set(train_df["group_id"])
        eval_groups = set(eval_df["group_id"])
        overlap = train_groups & eval_groups
        if overlap:
            raise RuntimeError(
                f"Group leakage detected: {len(overlap)} groups occur in both "
                "training and validation data."
            )

        # 2. Get loaders from Factory
        # This keeps the Trainer class clean of dataset-specific logic
        self.train_loader, self.eval_loader = DataFactory.get_loaders(
            config=self.config,
            tokenizer=self.tokenizer,
            train_df=train_df,
            eval_df=eval_df,
        )

        if self.config.log_to_wandb:
            # update dataset sizes in wandb config
            self.wandb_run.config.update(
                {
                    "dataset/train": len(train_df),
                    "dataset/eval": len(eval_df),
                },
                allow_val_change=True,
            )

    def _calculate_wsd_accuracy(self, eval_data):
        word_sense_detector = WordSenseDetector(
            pretrained_model=self.model,
            udpipe_model=self.udpipe_model,
            evaluation_dataset=eval_data,
            tokenizer=self.tokenizer,
            pooling_strategy=PoolingStrategy.mean_pooling,
            prediction_strategy=PredictionStrategy.all_examples_to_one_embedding,
            device=self.device,
        )

        eval_data = word_sense_detector.run()
        return prediction_accuracy(eval_data)

    def _save_model(
        self, model, path_to_save_model, save_sentence_transformer_format=True
    ):
        try:
            os.makedirs(path_to_save_model, exist_ok=True)
            model.save_pretrained(path_to_save_model, from_pt=True)
            self.tokenizer.save_pretrained(path_to_save_model)

            if save_sentence_transformer_format:
                # Save in SentenceTransformer format for compatibility with eval scripts
                transformer = models.Transformer(path_to_save_model)
                pooling = models.Pooling(transformer.get_word_embedding_dimension())
                sentence_transformer_model = SentenceTransformer(
                    modules=[transformer, pooling]
                )
                sentence_transformer_model.save(path_to_save_model)
        except Exception as e:
            print(f"model not saved, error = {e}")

    def _model_stem(self) -> str:
        """Return a stable, human-readable model name for experiment runs."""
        return self.config.model_name or f"model_{self.run_id}"

    @torch.no_grad()
    def evaluate_epoch(self, epoch, batch_count):
        self.model.eval()

        eval_loss = 0
        eval_bar = tqdm(self.eval_loader, leave=True, desc="Triplet Eval")
        for eval_batch in eval_bar:
            with autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                eval_loss += self.loss(eval_batch)

        mean_eval_loss = eval_loss / len(self.eval_loader)

        report_gpu()

        if self.config.log_to_wandb:
            # log epoch-level eval metrics
            self.wandb_run.log(
                {
                    "eval/loss": mean_eval_loss,
                    "epoch": epoch,
                },
                step=self.global_step,
            )

        if eval_loss < self.min_val_loss:
            self.min_val_loss = eval_loss
            self.rounds_count = 0

            if batch_count > 0:
                # save the model and upload to W&B as artifact if enabled
                model_dir = (
                    f"{self.config.path_to_save_fine_tuned_model}/"
                    f"{self._model_stem()}_best"
                )
                self._save_model(self.model, model_dir)

        else:
            self.rounds_count += 1

        print(
            {
                "epoch": epoch,
                "batch_count": batch_count,
                "min_val_loss": self.min_val_loss,
                "rounds_count": self.rounds_count,
            }
        )

        if self.rounds_count >= self.config.early_stopping:
            print(
                f"Early stopping, model not improve WSD for {self.config.early_stopping}"
            )
            return True

        return False

    def train_epoch(self, epoch):
        train_bar = tqdm(self.train_loader, leave=True, desc=f"Train epoch: {epoch}")

        for batch_count, batch in enumerate(train_bar):
            self.model.train()
            self.optimizer.zero_grad()

            with autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                loss = self.loss(batch).to(self.device)

            self.scaler.scale(loss).backward()
            max_grad_norm = self.config.max_grad_norm
            if max_grad_norm and max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.train_avg_meter.update(
                loss.item(), self.config.batch_size
            )  # TODO: .detach().cpu()?

            if self.config.apply_warmup:
                self.scheduler.step()

            if self.config.log_to_wandb:
                acc_val, acc_avg = self.train_avg_meter()
                self.wandb_run.log(
                    {
                        "train/loss": acc_avg,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=self.global_step,
                )

            # increment global step for W&B alignment
            self.global_step += 1

            if batch_count > 0 and batch_count % self.config.num_batch_to_eval == 0:
                if self.evaluate_epoch(epoch, batch_count):
                    return True  # reach early stopping rounds

    def train(self):
        training_error = None
        try:
            # initial evaluation of the raw model
            for epoch in range(self.config.num_epochs):
                early_stop = self.evaluate_epoch(epoch=epoch, batch_count=0)
                report_gpu()

                early_stop = early_stop or self.train_epoch(epoch)

                if early_stop:
                    path_to_save_model = self.config.path_to_save_fine_tuned_model
                    model_name = (
                        f"{path_to_save_model}/{self._model_stem()}_"
                        f"{epoch}_early_stopped"
                    )
                    self._save_model(self.model, model_name)

                    break

                # path_to_save_model = self.config.path_to_save_fine_tuned_model
                # model_name = f"{path_to_save_model}/model_{self.run_id}_{epoch}"
                # self._save_model(self.model, model_name)
        except Exception as e:
            print(f"Training interrupted, error = {e}")
            training_error = e
        finally:
            path_to_save_model = self.config.path_to_save_fine_tuned_model
            model_name = f"{path_to_save_model}/{self._model_stem()}_final"
            self._save_model(self.model, model_name)

            try:
                if self.config.evaluate_final_wsd:
                    # final evaluation of the model
                    wsd_acc = evaluate_wsd(
                        model_path=model_name,
                        model_tokenizer_path=model_name,
                        verbose=True,
                        device=self.device,
                    )

                    # log final WSD accuracy to W&B if enabled
                    if self.config.log_to_wandb:
                        self.wandb_run.log(
                            {"test/wsd_acc": wsd_acc}, step=self.global_step
                        )
            except Exception as e:
                print(f"Final evaluation failed, error = {e}")
            finally:
                if self.config.log_to_wandb:
                    wandb.finish()

        if training_error is not None:
            raise training_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a model for WSD")
    parser.add_argument(
        "--config",
        type=str,
        default="services/trainer/fine_tuning_config.ini",
        help="Path to the training config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (e.g., 'cuda:0' or 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Override train_data_path from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for torch, numpy, and random.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override wandb_run_name from config.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override the W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Override the W&B entity/team. Empty uses the logged-in user's entity.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Stable model name used for the saved model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the root directory for saved fine-tuned models.",
    )
    parser.add_argument(
        "--skip-final-wsd-eval",
        action="store_true",
        help="Skip the trainer's final WSD evaluation; useful when an evaluation suite runs afterward.",
    )
    parser.add_argument(
        "--pool-targets",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=None,
        help="Override pool_targets from config (true/false).",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset repo id (e.g. 'victormuryn/wsd-training-dataset').",
    )
    parser.add_argument(
        "--hf-subset",
        type=str,
        default=None,
        help="Subset folder within the HF dataset (e.g. 'dropout_seed42').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size from config.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Fraction of groups reserved for validation.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Fixed seed used only for the group-based train/eval split.",
    )
    args = parser.parse_args()

    if args.validation_split is not None and not 0 < args.validation_split < 1:
        parser.error("--validation-split must be between 0 and 1")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = TrainingConfig.from_config(args.config)
    if args.train_data:
        config.train_data_path = args.train_data
    if args.run_name:
        config.wandb_run_name = args.run_name
    if args.wandb_project is not None:
        config.wandb_project_name = args.wandb_project
    if args.wandb_entity is not None:
        config.wandb_entity = args.wandb_entity
    if args.model_name:
        config.model_name = args.model_name
    if args.output_dir:
        config.path_to_save_fine_tuned_model = args.output_dir
    if args.skip_final_wsd_eval:
        config.evaluate_final_wsd = False
    if args.pool_targets is not None:
        config.pool_targets = args.pool_targets
    if args.hf_dataset:
        config.hf_dataset = args.hf_dataset
    if args.hf_subset:
        config.hf_subset = args.hf_subset
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.validation_split is not None:
        config.validation_split = args.validation_split
    if args.split_seed is not None:
        config.split_seed = args.split_seed

    model_trainer = Trainer(config, device=args.device)
    model_trainer.train()
