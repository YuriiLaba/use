# Improving Ukrainian Word Sense Disambiguation with Sense-Aware Sentence Embeddings

## Description

Research code accompanying the manuscript by Victor Muryn and Yurii Laba. The project adapts a multilingual sentence encoder for Ukrainian word sense disambiguation (WSD): selecting the dictionary meaning of an ambiguous word in context. It includes corpus collection, definition-based pseudo-labeling, data augmentation, contrastive fine-tuning, and evaluation on WSD and general sentence-embedding tasks.

## Dataset Information

- **Ukrainian WSD Benchmark:** 1,386 lemmas, 3,206 dictionary meanings, and 13,310 contextual examples. Definitions come from the Dictionary of Noun Homonyms in Contemporary Ukrainian; examples come from GRAC. Each JSONL record contains `lemma` (string), `gloss` (list of definitions), and `examples` (list of sentences). The benchmark is available at [doi:10.57967/hf/10571](https://doi.org/10.57967/hf/10571).
- **Training corpus:** the news, fiction, and Wikipedia sentence-level subsets of [UberText 2.0](https://lang.org.ua/en/ubertext/), using the compressed `filter_rus_gcld+short.text_only.txt.bz2` files.
- **Ukrainian UDPipe model:** `20180506.uk.mova-institute.udpipe`, archived on [Zenodo](https://doi.org/10.5281/zenodo.23205856), used for tokenization and lemmatization during lemma-based corpus retrieval.
- **Sentence-embedding evaluation:** [STS-UK](https://huggingface.co/datasets/anikol12/STSB-UK) and Ukrainian classification, clustering, bitext mining, and retrieval tasks from [MTEB](https://github.com/embeddings-benchmark/mteb).

The benchmark definitions supply the sense inventory for training; its example sentences are reserved for evaluation. Intermediate datasets are stored as JSON/JSONL, and training triplets as CSV with context, positive definition, negative definition, lemma, group ID, and target-token indices.

## Code Information

| Location | Purpose |
| --- | --- |
| `collect_sentences/` | Extract corpus sentences and generate examples for rare senses. |
| `local_datasets/semi_supervised_2/` | Assign senses, filter benchmark overlap, merge examples, and build triplets. |
| `augment/` | Back-translation, masking, dropout, token shuffling, and combined augmentation. |
| `services/` | WSD inference, pooling, data utilities, and model training. |
| `eval/` | WSD, zero-shot LLM, STS-UK, and MTEB evaluation. |
| `scripts/` | Experiment evaluation, result summaries, corpus-frequency plotting, and sense-availability analysis. |

Paths, models, thresholds, and experiment settings are defined in [project_config.ini](project_config.ini). Python modules use this file by default; set `PIPELINE_CONFIG` to use another configuration. Shell runners also accept `--config PATH`. Older experiments are kept in `local_datasets/archive/`.

## Usage Instructions

Run all commands from the repository root. The commands below follow the research workflow. `cuda:3` selects GPU index 3; adjust device arguments and the GPU indices in `project_config.ini` for your machine.

### 1. Set up the environment and data

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download uk_core_news_sm
```

Download the benchmark and the Ukrainian UDPipe model, and place them at:

```text
datasets_pre_defined/ukrainian_wsd_benchmark.jsonl
models/20180506.uk.mova-institute.udpipe
```

For corpus collection, also download the three UberText subsets listed above into `datasets_pre_defined/`. Their expected filenames are in `[collection]` in `project_config.ini`.

Create the target-lemma list from the benchmark:

```bash
python - <<'PYTHON'
import json
from pathlib import Path
records = Path('datasets_pre_defined/ukrainian_wsd_benchmark.jsonl').read_text(encoding='utf-8').splitlines()
lemmas = sorted({json.loads(line)['lemma'] for line in records if line.strip()})
Path('datasets_pre_defined/unique_lemmas_homonyms.txt').write_text('\n'.join(lemmas) + '\n', encoding='utf-8')
PYTHON
```

### 2. Evaluate the baselines

```bash
./eval_all_wsd_models.sh --device cuda:3 --delete-model-after-evaluation
./eval_all_llm_wsd_models.sh --device cuda:3 --delete-model-after-evaluation
```

These evaluate the encoder and zero-shot LLM lists in `project_config.ini`. Encoder results go to `wsd_model_results.csv`; LLM results, predictions, and metadata go to `results/wsd_llm_*`. The cleanup flag removes downloaded models after successful evaluation. The encoder evaluator also deletes explicitly supplied local model paths.

Encoder evaluation uses target-token pooling for contexts and sentence mean pooling for definitions. Accuracy is calculated over retained dictionary-sense rows; excluded rows are logged.

### 3. Prepare training data

Use Linux for full-corpus collection. Collect each subset, then merge and deduplicate the sentences:

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

python -m local_datasets.raw_sentences.process_raw_sentences
python -m local_datasets.semi_supervised_2.assign_meaning_to_sentence --device cuda
python -m local_datasets.semi_supervised_2.delete_similar_sentences
python -m scripts.analyze_wsd_by_sense_availability
```

Collection appends to its output files; use fresh files when restarting collection. The availability analysis runs after filtering and reports the percentage of definitions with fewer than five natural examples.

For sentence generation, copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The research command explicitly selects the generation model, overriding `[generation]`:

```bash
python -m collect_sentences.generate_sentences_4_absent_meanings --model gpt-5.6-luna
python -m local_datasets.semi_supervised_2.merge_collected_and_generated
```

Before back-translation, convert the two translation models:

```bash
mkdir -p models/translators

./.venv/bin/ct2-transformers-converter \
  --model Helsinki-NLP/opus-mt-tc-big-zle-en \
  --output_dir models/translators/opus-mt-zle-en-ct2 \
  --quantization float16 --force
./.venv/bin/ct2-transformers-converter \
  --model Helsinki-NLP/opus-mt-tc-big-en-zle \
  --output_dir models/translators/opus-mt-en-zle-ct2 \
  --quantization float16 --force
```

Set the augmentation GPU indices in `[augmentation]` for your machine, then run:

```bash
./run_all_augmentations.sh
./generate_all_triplets.sh
```

### 4. Fine-tune and evaluate

Set your Hugging Face repository prefix and W&B project in `project_config.ini`, then authenticate:

```bash
hf auth login
wandb login
./run_finetuning_experiments.sh --delete-local-model
```

This runs eight data configurations with two training pooling modes and three seeds (48 runs), using the configured GPUs. Each final checkpoint is evaluated on WSD, STS-UK, and Ukrainian MTEB tasks, logged to W&B, and uploaded to Hugging Face. `--delete-local-model` removes final and best local checkpoints only after a successful upload. Per-run results remain in the configured results directory; the summary is saved to `results/finetuning_summary_aug16.csv`.

The eight conditions are Natural, Natural + Generation, and generation combined with masking, word deletion (dropout), back-translation, shuffling, stochastic combination, or pooled augmentations (`all_combined`). The pooled condition samples from the individual and stochastic augmentation outputs.

### 5. Evaluate WSD by natural sense availability

```bash
./eval_wsd_by_sense_availability.sh --device cuda:3 --resume --delete-model-after-evaluation
```

This compares Natural, Natural + Generation, and Natural + Generation + Back-translation checkpoints across both training pooling modes and three seeds. Senses are grouped by their number of natural examples after filtering: 0, 1-4, 5-19, and 20 or more. The script loads local checkpoints or their Hugging Face copies, reuses verified predictions with `--resume`, and removes evaluated cache revisions with the cleanup flag. Tables, summaries, and predictions are saved in `results/wsd_sense_availability/`.

`bucket_tables.tex` contains the paper tables, `bucket_summary.csv` contains mean accuracy and sample standard deviation across seeds, and `bucket_gains.csv` contains paired augmentation gains. Comparisons use the same intersection of successfully scored records across all requested checkpoints.

### 6. Other analyses reported in the paper

**Corpus coverage** Lemma frequencies and coverage are calculated from `unique_lemma_sentences.jsonl` against the benchmark inventory. Definition coverage is calculated from the grouped datasets before and after benchmark-overlap filtering. `scripts.analyze_wsd_by_sense_availability` reports the percentage of definitions with fewer than five natural examples in the filtered dataset.

To render the top-15 lemma frequency figure:

```bash
python -m pip install matplotlib
python -m scripts.plot_ubertext_frequency
```

This requires `results/ubertext_coverage/statistics.json`, a separately prepared summary excluded from Git. The plot reads its `corpus.top_20` list of `lemma`/`count` records; the pipeline does not generate this summary automatically.

**Triplet counts and repeated runs** Table 8 counts training rows after the group-based split, excluding validation rows. For overall WSD, STS, and MTEB comparisons, group runs by data configuration and training pooling, then report mean and sample standard deviation over seeds 42, 123, and 456 (`ddof=1`). The fine-tuning summary CSV contains per-run scores; it does not calculate these grouped statistics. Per-task MTEB scores are in each run's `metrics.json` and `mteb_results/`.

**Sentence-level comparisons** The training runner evaluates fine-tuned checkpoints automatically. Evaluate the two pretrained comparison models separately:

```bash
python -m eval.eval_stsb --device cuda:3 \
  --models sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
           lang-uk/ukr-paraphrase-multilingual-mpnet-base
python -m eval.eval_mteb --device cuda:3 \
  --models sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
           lang-uk/ukr-paraphrase-multilingual-mpnet-base
```

STS correlations are saved to `sts_results.csv`; baseline MTEB scores are saved under `eval/mteb_results/`. Both use full-sentence embeddings, regardless of the checkpoint's training pooling. The current STS evaluators set the reference similarity to 1.0 for identical sentence pairs.

## Requirements

Python 3.10 or 3.11. Main dependencies include PyTorch, Transformers, Sentence Transformers, NumPy, pandas, SciPy, scikit-learn, spaCy, UDPipe, CTranslate2, OpenAI, and MTEB. The full list is in [requirements.txt](requirements.txt). Corpus-frequency plotting additionally requires Matplotlib. Dependency versions are not pinned.

CPU evaluation is supported. Training and GPU-based augmentation require CUDA; full-corpus processing requires substantial RAM and disk space. The paper's training runs used NVIDIA RTX 6000 Ada GPUs with 48 GB memory. Internet access is needed for model and evaluation-dataset downloads, and an OpenAI API key is needed for sentence generation. W&B logging and Hugging Face uploads are optional.

## Methodology

1. Extract sentences containing the benchmark lemmas from UberText 2.0 and deduplicate them.
2. Assign candidate senses by comparing sentence and definition embeddings. Retain assignments with probability at least 0.9 and cosine similarity at least 0.6.
3. Remove corpus sentences with cosine similarity of at least 0.95 to any benchmark example.
4. Generate examples for senses with fewer than five sentences, then apply the selected transformations to contexts and definitions.
5. Construct triplets: a context, its assigned definition, and a competing definition for the same lemma, with a budget of up to 100 triplets per definition. Keep each original context and its augmented variants in the same training or validation partition.
6. Fine-tune `paraphrase-multilingual-mpnet-base-v2` with cosine triplet loss, comparing sentence and target-token anchor pooling. Evaluate sense discrimination and general sentence-embedding performance.

Default training settings are batch size 104, learning rate `2e-6`, up to two epochs, and training seeds 42, 123, and 456. The manuscript provides the full experimental protocol.

## Citations

When using this code, cite the accompanying manuscript:

Muryn, V., and Laba, Y. *Improving Ukrainian word sense disambiguation with sense-aware sentence embeddings*. Manuscript prepared for submission to PeerJ.

Related resources:

- Laba, Y. (2026). *Ukrainian WSD Benchmark*. [doi:10.57967/hf/10571](https://doi.org/10.57967/hf/10571).
- Chaplynskyi, D. (2023). [Introducing UberText 2.0: A Corpus of Modern Ukrainian at Scale](https://aclanthology.org/2023.unlp-1.1/). UNLP, pp. 1-10.
- Laba, Y., Mudryi, V., Chaplynskyi, D., Romanyshyn, M., and Dobosevych, O. (2023). [Contextual Embeddings for Ukrainian: A Large Language Model Approach to Word Sense Disambiguation](https://aclanthology.org/2023.unlp-1.2/). UNLP, pp. 11-19.