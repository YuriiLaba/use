# [PREPARATION]
ACUTE = chr(0x301)
GRAVE = chr(0x300)

# [RESULTS]
MINIMUM_POS_OCCURRENCE = 100
MINIMUM_GLOSS_OCCURRENCE = 300
FREQUENCY_QUANTILES = 10

# [DATA MINING]
PATH_TO_SOURCE_DATASET = (
    "datasets_pre_defined/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2"
)
PATH_TO_SOURCE_UDPIPE = "models/20180506.uk.mova-institute.udpipe"
PATH_TO_LEMMAS_OF_INTEREST = "datasets_pre_defined/unique_lemmas_homonyms.txt"
PATH_TO_SAVE_GATHERED_DATASET = (
    "local_datasets/raw_sentences/lemma_examples_samples_udpipe_news.json"
)
NUMBER_OF_EXAMPLES_TO_GATHER = -1  # set to -1 to gather all available examples
EMBEDDER_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TRIPLET_PROCESSOR_BATCH_SIZE = 2000
PATH_TO_SAVE_TRIPLETS = "local_datasets/ubertext_triplets_6m_samples.csv"

UNIQUE_LEMMAS_WITH_SENTENCES_FILE = (
    "local_datasets/raw_sentences/unique_lemma_sentences.jsonl"
)

HOMONYM_BENCHMARK_PATH = "datasets_pre_defined/ukrainian_wsd_benchmark.jsonl"