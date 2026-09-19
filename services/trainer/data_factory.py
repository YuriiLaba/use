import torch
from torch.utils.data import DataLoader
from services.trainer.datasets import TripletDataset, NTXentDataset, PairsDataset
from services.trainer.training_config import TrainingConfig


class DataFactory:
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2

    @staticmethod
    def get_loaders(config: TrainingConfig, tokenizer, train_df, eval_df):
        pool_targets = config.pool_targets or config.use_both_poolings

        # 1. Select Dataset based on Loss
        if "triplet_loss" in config.loss_type:
            train_ds = TripletDataset(
                anchor=train_df["anchor"].values,
                positive=train_df["positive"].values,
                negative=train_df["negative"].values,
                tokenizer=tokenizer,
                pool_targets=pool_targets,
                anchor_target_word_ids=train_df.get("anchor_target_word_ids"),
                positive_target_word_ids=train_df.get("positive_target_word_ids"),
                negative_target_word_ids=train_df.get("negative_target_word_ids"),
            )

            eval_ds = TripletDataset(
                anchor=eval_df["anchor"].values,
                positive=eval_df["positive"].values,
                negative=eval_df["negative"].values,
                tokenizer=tokenizer,
                pool_targets=pool_targets,
                anchor_target_word_ids=eval_df.get("anchor_target_word_ids"),
                positive_target_word_ids=eval_df.get("positive_target_word_ids"),
                negative_target_word_ids=eval_df.get("negative_target_word_ids"),
            )
        elif config.loss_type == "mnr_loss":
            train_ds = PairsDataset(
                anchor=train_df["anchor"].values,
                positive=train_df["positive"].values,
                tokenizer=tokenizer,
                pool_targets=pool_targets,
                anchor_target_word_ids=train_df.get("anchor_target_word_ids"),
            )

            eval_ds = PairsDataset(
                anchor=eval_df["anchor"].values,
                positive=eval_df["positive"].values,
                tokenizer=tokenizer,
                pool_targets=pool_targets,
                anchor_target_word_ids=eval_df.get("anchor_target_word_ids"),
            )
        elif config.loss_type == "nt_xent_loss":
            train_ds = NTXentDataset(
                view1=train_df["view1"].values,
                view2=train_df["view2"].values,
                tokenizer=tokenizer,
            )

            eval_ds = NTXentDataset(
                view1=eval_df["view1"].values,
                view2=eval_df["view2"].values,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Unsupported loss: {config.loss_type}")

        # 2. Handle Collate Logic (The messy part)
        collate_fn = DataFactory._get_collate_fn(config.loss_type, pool_targets)

        # 3. Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=DataFactory.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=DataFactory.PREFETCH_FACTOR,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=DataFactory.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=DataFactory.PREFETCH_FACTOR,
            collate_fn=collate_fn,
        )
        return train_loader, eval_loader

    @staticmethod
    def _get_collate_fn(loss_type, pool_targets):
        if not pool_targets:
            return None

        def triplet_collate(batch):
            return {
                "anchor_ids": torch.stack([item["anchor_ids"] for item in batch]),
                "anchor_mask": torch.stack([item["anchor_mask"] for item in batch]),
                "positive_ids": torch.stack([item["positive_ids"] for item in batch]),
                "positive_mask": torch.stack([item["positive_mask"] for item in batch]),
                "negative_ids": torch.stack([item["negative_ids"] for item in batch]),
                "negative_mask": torch.stack([item["negative_mask"] for item in batch]),
                "anchor_target_word_ids": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(item["anchor_target_word_ids"]) for item in batch],
                    batch_first=True,
                    padding_value=-1,
                ),
            }

        def mnr_collate(batch):
            return {
                "anchor_ids": torch.stack([item["anchor_ids"] for item in batch]),
                "anchor_mask": torch.stack([item["anchor_mask"] for item in batch]),
                "positive_ids": torch.stack([item["positive_ids"] for item in batch]),
                "positive_mask": torch.stack([item["positive_mask"] for item in batch]),
                "anchor_target_word_ids": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(item["anchor_target_word_ids"]) for item in batch],
                    batch_first=True,
                    padding_value=-1,
                ),
            }

        loss_type_to_collate = {
            "triplet_loss": triplet_collate,
            "triplet_loss_cosine": triplet_collate,
            "mnr_loss": mnr_collate,
        }

        return loss_type_to_collate.get(loss_type, None)
