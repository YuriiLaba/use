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
find . -path './.venv' -prune -o -path './.git' -prune -o -name '*.py' -print0 \
  | xargs -0 python -m py_compile
python -c "import torch, transformers, sentence_transformers, pandas, ufal.udpipe; print('Core dependencies OK')"
```

## Required assets

Place these files in the following locations:

```text
models/20180506.uk.mova-institute.udpipe
datasets_pre_defined/ukrainian_wsd_benchmark.jsonl
datasets_pre_defined/unique_lemmas_homonyms.txt
```

The UDPipe Python package and the trained UDPipe model are separate assets. The package provides the API; the `.udpipe` file provides the Ukrainian tokenizer and tagger model.

## Install translation augmentation models

The translation and combined augmentation scripts use CTranslate2 models. These models are not downloaded automatically when the augmentation scripts start. Convert them once after installing the project dependencies.

Run from the repository root:

```bash
mkdir -p models/translators

./.venv/bin/ct2-transformers-converter \
  --model Helsinki-NLP/opus-mt-tc-big-zle-en \
  --output_dir models/translators/opus-mt-zle-en-ct2 \
  --quantization float16

./.venv/bin/ct2-transformers-converter \
  --model Helsinki-NLP/opus-mt-tc-big-en-zle \
  --output_dir models/translators/opus-mt-en-zle-ct2 \
  --quantization float16
```

The converter downloads the original Hugging Face checkpoints and saves the converted models locally at:

```text
models/translators/opus-mt-zle-en-ct2
models/translators/opus-mt-en-zle-ct2
```

These models are required for back-translation and combined augmentation. They are not required for evaluation, sentence collection, dropout, or token-shuffling augmentation.

The active benchmark is:

```text
datasets_pre_defined/ukrainian_wsd_benchmark.jsonl
```

It contains one sense record per JSONL line with:

```text
lemma, gloss, examples
```

## Download UberText corpora

On Linux, use `wget`:

```bash
cd datasets_pre_defined

wget -O ubertext.news.filter_rus_gcld+short.text_only.txt.bz2 \
  https://lang.org.ua/static/downloads/ubertext2.0/news/sentenced/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2

wget -O ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2 \
  https://lang.org.ua/static/downloads/ubertext2.0/fiction/sentenced/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2

wget -O ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2 \
  https://lang.org.ua/static/downloads/ubertext2.0/wikipedia/sentenced/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2

cd ..
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

To use a CUDA GPU on a Linux server, call the Python evaluator directly:

```bash
python -m eval.eval_all_wsd_models \
  --benchmark-path datasets_pre_defined/ukrainian_wsd_benchmark.jsonl \
  --device cuda:0 \
  --output wsd_model_results_gpu.csv
```

The root `eval_all_wsd_models.sh` launcher intentionally uses CPU by default for macOS.

## Collect sentences

The collector is CPU-based. The current implementation starts approximately half of the available CPU count as multiprocessing workers, and each worker may load the large UDPipe model. Use a Linux server with sufficient RAM for full-corpus collection when possible.

Use separate output files because collection outputs are appended to existing files.

The current collector version does not expose a `--workers` option. The previous macOS-safe worker initializer is not present in this checkout, so collection may stall on macOS when using the default `spawn` multiprocessing behavior. Run collection on Linux, or restore the worker-initializer change before running it locally on macOS.

For a smoke test:

```bash
python -m collect_sentences.collect_ubertext_sentences \
  --source_dataset datasets_pre_defined/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2 \
  --save_dataset local_datasets/raw_sentences/test.json \
  --num_examples 10 \
  --save_every 1
```

For the three full corpora:

```bash
python -m collect_sentences.collect_ubertext_sentences \
  --source_dataset datasets_pre_defined/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2 \
  --save_dataset local_datasets/raw_sentences/lemma_examples_news.json \
  --num_examples -1

python -m collect_sentences.collect_ubertext_sentences \
  --source_dataset datasets_pre_defined/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2 \
  --save_dataset local_datasets/raw_sentences/lemma_examples_fiction.json \
  --num_examples -1

python -m collect_sentences.collect_ubertext_sentences \
  --source_dataset datasets_pre_defined/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2 \
  --save_dataset local_datasets/raw_sentences/lemma_examples_wikipedia.json \
  --num_examples -1
```

Merge and deduplicate all JSON collection outputs:

```bash
python -m local_datasets.raw_sentences.process_raw_sentences
```

Output:

```text
local_datasets/raw_sentences/unique_lemma_sentences.jsonl
```

## Fine-tune the paper configurations

After generating the triplet CSV files with `./generate_all_triplets.sh`, run:

```bash
wandb login
huggingface-cli login

export WANDB_PROJECT=ucu-wsd-finetuning
# Set this only when logging to a W&B team/entity:
# export WANDB_ENTITY=your-wandb-team

./run_finetuning_experiments.sh \
  --gpus 0,1,2,3 \
  --hf-repo-prefix YOUR_HF_USERNAME/ucu-wsd \
  --delete-local-model
```

The runner excludes the two pretrained baseline rows and trains:

```text
8 training configurations × 2 pooling modes × 3 training seeds = 48 runs
```

Each GPU runs one independent experiment. Batch size defaults to `104` to match the paper. For a throughput-oriented run on 48 GB RTX 6000 cards, use `--batch-size 208`; this changes the optimization setup and is not an exact paper reproduction.

`--delete-local-model` removes the final and best checkpoints after a successful Hugging Face upload. A temporary local copy is still required during training, evaluation, and upload; if the upload fails, the local checkpoint is retained.

Models are saved locally under:

```text
models/fine-tuned-models/
```

Each run is evaluated on the Ukrainian WSD benchmark, STS-UK, and Ukrainian MTEB tasks. Metrics are saved under `results/finetuning/`, summarized in `results/finetuning_summary.csv`, logged to W&B, and uploaded with the model to a separate Hugging Face model repository.
