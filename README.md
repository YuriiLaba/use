# Ukrainian Word Sense Disambiguation

Research code for Ukrainian word-sense disambiguation using multilingual transformer models, Ukrainian dictionary senses, contextual examples, and contrastive fine-tuning.

## Requirements

- Python 3.10 or 3.11
- macOS or Linux
- 8 GB+ RAM for basic evaluation; more for collection and training
- Optional GPU for embedding, augmentation, and training workloads

The dependency versions are currently not pinned.

## Environment setup

Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-notebooks.txt
python -m spacy download uk_core_news_sm
```

If UDPipe is not available from the package index, install the local source archive:

```bash
python -m pip install ./models/ufal.udpipe-1.2.0.1.tar.gz
```

Verify the environment:

```bash
python scripts/reproduce/environment_report.py
python -m compileall -q .
```

## Required assets

Place these files in the following locations:

```text
models/20180506.uk.mova-institute.udpipe
datasets_pre_defined/ukrainian_wsd_benchmark.jsonl
datasets_pre_defined/unique_lemmas_homonyms.txt
```

The UDPipe Python package and the trained UDPipe model are separate assets. The package provides the API; the `.udpipe` file provides the Ukrainian tokenizer and tagger model.

The active benchmark is:

```text
datasets_pre_defined/ukrainian_wsd_benchmark.jsonl
```

It contains one sense record per JSONL line with:

```text
lemma, gloss, examples
```

## Build the lemma list

If the benchmark is the source of truth for target lemmas, generate the collector input file:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_json(
    "datasets_pre_defined/ukrainian_wsd_benchmark.jsonl",
    lines=True,
    encoding="utf-8",
)

df["lemma"].dropna().drop_duplicates().sort_values().to_csv(
    "datasets_pre_defined/unique_lemmas_homonyms.txt",
    index=False,
    header=False,
)
PY
```

## Download UberText corpora

On macOS, use `curl`:

```bash
cd datasets_pre_defined

curl -L -o ubertext.news.filter_rus_gcld+short.text_only.txt.bz2 \
  https://lang.org.ua/static/downloads/ubertext2.0/news/sentenced/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2

curl -L -o ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2 \
  https://lang.org.ua/static/downloads/ubertext2.0/fiction/sentenced/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2

curl -L -o ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2 \
  https://lang.org.ua/static/downloads/ubertext2.0/wikipedia/sentenced/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2

cd ..
```

## Collect sentences

The collector is CPU-based. It uses one worker by default because the UDPipe model is large. Use separate output files because collection outputs are appended to existing files.

For a smoke test:

```bash
python -m collect_sentences.collect_ubertext_sentences \
  --source_dataset datasets_pre_defined/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2 \
  --save_dataset local_datasets/raw_sentences/test.json \
  --num_examples 10 \
  --save_every 1 \
  --workers 1
```

For the three full corpora:

```bash
python -m collect_sentences.collect_ubertext_sentences \
  --source_dataset datasets_pre_defined/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2 \
  --save_dataset local_datasets/raw_sentences/lemma_examples_news.json \
  --num_examples -1 \
  --workers 1

python -m collect_sentences.collect_ubertext_sentences \
  --source_dataset datasets_pre_defined/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2 \
  --save_dataset local_datasets/raw_sentences/lemma_examples_fiction.json \
  --num_examples -1 \
  --workers 1

python -m collect_sentences.collect_ubertext_sentences \
  --source_dataset datasets_pre_defined/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2 \
  --save_dataset local_datasets/raw_sentences/lemma_examples_wikipedia.json \
  --num_examples -1 \
  --workers 1
```

Merge and deduplicate all JSON collection outputs:

```bash
python -m local_datasets.raw_sentences.process_raw_sentences
```

Output:

```text
local_datasets/raw_sentences/unique_lemma_sentences.jsonl
```

## Evaluate models

Evaluate one model:

```bash
python -m eval.eval_wsd \
  --model-path sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --tokenizer-path sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --benchmark-path datasets_pre_defined/ukrainian_wsd_benchmark.jsonl \
  --device cpu \
  --no-reports
```

Evaluate the configured model list:

```bash
./eval_all_wsd_models.sh
```

Results are written to:

```text
wsd_model_results.csv
```

The batch evaluator runs on CPU by default and continues if an individual model fails.

## Training pipeline

The full data pipeline is:

```text
UberText corpora
  → sentence collection
  → deduplication
  → meaning assignment
  → optional synthetic examples
  → augmentation
  → triplet generation
  → transformer fine-tuning
  → WSD evaluation
```

Assign collected sentences to dictionary meanings:

```bash
python -m local_datasets.semi_supervised_2.assign_meaning_to_sentence
```

Generate triplets after creating the required augmentation files:

```bash
python -m local_datasets.semi_supervised_2.form_triplets
```

Run a small local training smoke test:

```bash
python -m services.trainer.trainer \
  --config services/trainer/reviewer_config.ini \
  --train-data local_datasets/semi_supervised_2/triplets_semi_supervised_all_augs_mixed_100.csv \
  --pool-targets false \
  --device cpu \
  --batch-size 8 \
  --run-name onboarding-smoke-test
```

Models are saved under:

```text
models/fine-tuned-models/
```

## Notes

- Use `./.venv/bin/python` if the shell resolves `python3` to Homebrew’s system interpreter.
- Do not reuse collection output filenames because the collector appends to them.
- Keep `.DS_Store`, virtual environments, datasets, and model files out of Git.
- `pool_targets=True` requires target-token IDs in the training data and should be validated with a smoke test before large experiments.
