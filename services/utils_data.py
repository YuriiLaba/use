import pandas as pd
import json
import pymorphy2
import os
import stanza
from services.config import (
    ACUTE,
    GRAVE,
)

def take_first_n_glosses(data, first_n_glosses):
    data = data.groupby("lemma").head(first_n_glosses)
    return data

def add_pos_tag(data_with_predictions, udpipe_model=None, engine="stanza"):
    if engine == "stanza":
        if os.path.exists("data/pos_precalculation.pkl"):
            pos_precalculation = pd.read_pickle("data/pos_precalculation.pkl")
            data_with_predictions.loc[:, "pos"] = data_with_predictions.lemma.replace(
                pos_precalculation["pos"]
            )
        else:
            nlp = stanza.Pipeline(
                lang="uk", processors="tokenize,mwt,pos", verbose=False
            )

            def get_pos_tag_udpipe(word, text):
                word = (
                    word.lower().replace(GRAVE, "").replace(ACUTE, "").replace("'", "’")
                )
                tokens = udpipe_model.tokenize(text)
                for tok_sent in tokens:
                    udpipe_model.tag(tok_sent)
                    for word_index, w in enumerate(tok_sent.words[1:]):
                        if w.lemma == word:
                            return w.upostag
                doc = nlp(word)
                return doc.sentences[0].words[0].upos

            data_with_predictions["pos"] = data_with_predictions.apply(
                lambda x: get_pos_tag_udpipe(x["lemma"], x["examples"][0]), axis=1
            )
            if not os.path.exists("data"):
                os.mkdir("data")
            data_with_predictions[["lemma", "pos"]].set_index("lemma").to_pickle(
                "data/pos_precalculation.pkl"
            )
    elif engine == "pymorphy":
        morph = pymorphy2.MorphAnalyzer(lang="uk")

        def get_pos_tag(row):
            p = morph.parse(row["lemma"])[0]
            return p.tag.POS

        data_with_predictions["pos"] = data_with_predictions.apply(get_pos_tag, axis=1)

    return data_with_predictions

def add_frequency_column(data, udpipe_model):
    dictionary = pd.read_pickle("data/frequents.pkl")

    if "pos" not in data.columns:
        data = add_pos_tag(data, udpipe_model)

    data["clear_lemma"] = (
        data.lemma.str.replace(GRAVE, "").str.replace(ACUTE, "").str.lower()
    )
    important_columns = data.columns

    data = data.merge(
        dictionary,
        how="left",
        left_on=["clear_lemma", "pos"],
        right_on=["lemma", "pos"],
        suffixes=("", "_y"),
    )

    data_not_merged = data[data.freq_in_corpus.isna()][important_columns].copy()
    data = data[data.freq_in_corpus.notna()]

    dictionary.drop_duplicates(subset=["lemma"], keep="last", inplace=True)
    data_not_merged = data_not_merged.merge(
        dictionary.drop(columns=["pos"]),
        how="left",
        left_on=["clear_lemma"],
        right_on=["lemma"],
        suffixes=("", "_y"),
    )

    data = pd.concat([data, data_not_merged])
    data.drop(
        columns=["clear_lemma", "lemma_y", "count", "doc_count", "pos_y"],
        inplace=True,
        errors="ignore",
    )

    data[["freq_by_pos", "freq_in_corpus", "doc_frequency"]] = data[
        ["freq_by_pos", "freq_in_corpus", "doc_frequency"]
    ].fillna(0)

    del dictionary, data_not_merged
    return data

def read_homonym_benchmark(path):
      rows = []
      with open(path, encoding="utf-8") as file:
          for line_number, line in enumerate(file, start=1):
              if not line.strip():
                  continue

              row = json.loads(line)

              lemma = str(row["lemma"]).strip().lower()
              gloss = row["gloss"]
              examples = row["examples"]

              if isinstance(gloss, str):
                  gloss = [gloss]

              if isinstance(examples, str):
                  examples = [examples]

              if not lemma or not gloss or not examples:
                  continue

              rows.append(
                  {
                      "lemma": lemma,
                      "gloss": gloss,
                      "examples": examples,
                  }
              )

      return pd.DataFrame(rows)