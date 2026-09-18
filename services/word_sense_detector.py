import torch
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModel

tqdm.pandas(desc="WSD Eval")


class WordSenseDetector:
    def __init__(
        self,
        pretrained_model,
        udpipe_model,
        evaluation_dataset,
        pooling_strategy,
        prediction_strategy,
        device: torch.device = None,
        **kwargs
    ):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO create doc-string especially for describing prediction_strategy
        if isinstance(pretrained_model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                pretrained_model, output_hidden_states=True, trust_remote_code=True
            ).to(self.device)
        else:
            self.tokenizer = kwargs["tokenizer"]
            self.model = pretrained_model.to(self.device)

        self.udpipe_model = udpipe_model
        self.evaluation_dataset = evaluation_dataset
        self.prediction_strategy = prediction_strategy
        self.pooling_strategy = pooling_strategy
        # Cache gloss embeddings for the duration of one evaluation run.
        # The cache is instance-scoped so embeddings are never reused across
        # different models or detector instances.
        self.context_embedding_cache = {}
        # TODO create a WSD_logger and move there missing_target_word_in_sentence
        self.context_lookup = (
            self.evaluation_dataset.groupby("lemma")["gloss"].apply(list).to_dict()
        )

    def predict_word_sense(self, row):
        lemma = row["lemma"]
        examples = row["examples"]
        contexts = self.context_lookup[lemma]

        return self.prediction_strategy(
            lemma,
            examples,
            contexts,
            self.model,
            self.tokenizer,
            self.udpipe_model,
            self.pooling_strategy,
            self.device,
            self.context_embedding_cache,
        )

    def run(self):
        self.evaluation_dataset["predicted_context"] = (
            self.evaluation_dataset.progress_apply(self.predict_word_sense, axis=1)
        )
        return self.evaluation_dataset
