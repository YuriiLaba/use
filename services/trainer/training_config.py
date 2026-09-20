import configparser
from dataclasses import dataclass, fields


@dataclass
class TrainingConfig:
    # wandb specific
    log_to_wandb: bool = True
    wandb_entity: str = "hellcaster-ukrainian-catholic-university"
    wandb_project_name: str = "СonEFUv2"
    wandb_run_name: str = None
    model_name: str | None = None
    evaluate_final_wsd: bool = True

    # paths
    path_to_save_fine_tuned_model: str = "models/fine-tuned-models"
    hf_dataset: str | None = None
    hf_subset: str | None = None

    # training
    train_data_path: str = (
        "local_datasets/semi_supervised_2/triplets_semi_supervised.csv"
    )
    loss_type: str = "triplet_loss"
    pool_targets: bool = True
    use_both_poolings: bool = False
    model_to_fine_tune: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    batch_size: int = 104
    num_batch_to_eval: int = 200
    validation_split: float = 0.01
    split_seed: int = 42

    # TRAINING_PARAMS
    layers_to_unfreeze: int = 0
    random_model_weights_reinitialization: bool = False
    number_of_layers_for_reinitialization: int = 3
    learning_rate: float = 2e-6
    apply_warmup: bool = True
    warmup_ratio: float = 0.1
    num_epochs: int = 2
    early_stopping: int = 15
    enable_gpu_parallel: bool = True
    max_grad_norm: float = 1.0

    @classmethod
    def from_config(cls, config_path: str) -> "TrainingConfig":
        parser = configparser.ConfigParser()
        parser.read(config_path)

        # Initialize instance with default values
        conf = cls()

        # Map of Python types to configparser getter methods
        type_getters = {
            bool: parser.getboolean,
            int: parser.getint,
            float: parser.getfloat,
            str: parser.get,
        }

        # Iterate over all defined fields in the dataclass
        for field in fields(cls):
            # We search all sections for the key
            for section in parser.sections():
                if parser.has_option(section, field.name):
                    # Determine the getter based on the type hint
                    getter = type_getters.get(field.type, parser.get)

                    try:
                        value = getter(section, field.name)
                        setattr(conf, field.name, value)
                    except ValueError as e:
                        print(
                            f"Warning: Could not parse {field.name} as {field.type}. Keeping default."
                        )

                    break  # Found the value, move to next field

        return conf
