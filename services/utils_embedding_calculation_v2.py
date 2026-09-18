from services.utils_model import run_inference
from collections import Counter
from services.udpipe_model import UDPipeModel

import spacy


def ngrams(s: str, n: int = 3) -> set[str]:
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def char_dice(a: str, b: str, n: int = 3) -> float:
    """
    Character n-gram Sørensen–Dice coefficient.
    Designed for fuzzy lemma matching (e.g., Ukrainian morphology noise).
    """

    # normalize
    a = a.lower().strip()
    b = b.lower().strip()

    # exact match shortcut
    if a == b:
        return 1.0

    # guard against garbage
    if not a or not b:
        return 0.0

    # if too short, don't lie
    if min(len(a), len(b)) < n:
        return 0.0

    na = ngrams(a, n)
    nb = ngrams(b, n)

    if not na or not nb:
        return 0.0

    return 2 * len(na & nb) / (len(na) + len(nb))


def same_lemma(a: str, b: str) -> bool:
    if a in b or b in a:
        return True

    # length guard
    if min(len(a), len(b)) < 4:
        return False

    # character 3-gram Jaccard
    if char_dice(a, b) >= 0.5:
        return True

    return False


def lemmatize_word(word: str, model: UDPipeModel | spacy.Language) -> str:
    is_spacy = isinstance(model, spacy.Language)

    if is_spacy:
        doc = model(word)
        return "".join([t.lemma_.lower() for t in doc])
    else:
        tokenized = model.tokenize(word)
        model.tag(tokenized[0])
        return "".join([w.lemma.lower() for w in tokenized[0].words[1:]])


def iterate_sentence(text: str, model: UDPipeModel | spacy.Language):
    is_spacy = isinstance(model, spacy.Language)
    if is_spacy:
        doc = model(text)
        for token in doc:
            yield token.text, token.lemma_.lower()
    else:
        tokenized = model.tokenize(text)
        for sent in tokenized:
            model.tag(sent)
            for w in sent.words[1:]:  # skip root
                yield w.form, w.lemma.lower()


def _find_target_word_in_sentence(
    model: UDPipeModel | spacy.Language, input_text: str, target_word: str
):
    target_word = target_word.strip().lower()

    try:
        target_word_lemma = lemmatize_word(target_word, model)
    except Exception as e:
        print(f"Lemmatization failed for '{target_word}'")
        raise e

    for token_text, token_lemma in iterate_sentence(input_text, model):
        if same_lemma(target_word, token_lemma) or same_lemma(
            target_word_lemma, token_lemma
        ):
            return token_text

    return None


def normalize_word(word: str) -> str:
    return (
        word.replace("«", "")
        .replace("»", "")
        .replace('"', "")
        .replace("“", "")
        .replace("”", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(".", "")
        .strip()
    )


def _find_target_word_in_tokenized_text(tokenizer, tokenized_input_text, word: str):
    target_words_with_indexes = []

    word = word.strip().lower()
    current_word = ""
    start_index = 0

    zipped = zip(tokenized_input_text["input_ids"][0], tokenized_input_text.word_ids())
    for index, (input_id, word_id) in enumerate(zipped):
        token = tokenizer.decode([input_id]).strip()

        # Remove subword prefix if present
        if token.startswith("##"):
            token = token.replace("##", "")

        token = token.replace("▁", "").lower()

        current_word += token
        current_word_normalized = normalize_word(current_word)

        if word.startswith(current_word_normalized):
            if word == current_word_normalized:
                end_index = index
                target_words_with_indexes.append(
                    (word, list(range(start_index, end_index + 1)))
                )

                current_word = ""
                start_index = index + 1

        else:
            current_word = token
            start_index = index
            current_word_normalized = normalize_word(current_word)

            if word.startswith(current_word_normalized):
                if word == current_word_normalized:
                    end_index = index
                    target_words_with_indexes.append(
                        (word, list(range(start_index, end_index + 1)))
                    )

                    current_word = ""
                    start_index = index + 1
            else:
                current_word = ""
                start_index = index + 1

    return target_words_with_indexes


def get_target_word_embedding(
    model,
    tokenizer,
    udpipe_model,
    pooling_strategy,
    target_word,
    sentence_example,
    device,
):
    # TODO: remove it
    ACUTE = chr(0x301)
    GRAVE = chr(0x300)
    target_word_fixed = target_word.replace(GRAVE, "").replace(ACUTE, "")

    # TODO rename "word" for better understanding
    word = _find_target_word_in_sentence(
        udpipe_model, sentence_example, target_word_fixed
    )
    if word is None:
        # print("Can't find target word in sentence")
        return None

    tokenized_input_text, hidden_states = run_inference(
        model, tokenizer, sentence_example, device
    )
    target_word_indexes = _find_target_word_in_tokenized_text(
        tokenizer, tokenized_input_text, word
    )

    # TODO if len(target_word_indexes) != 1:
    if len(target_word_indexes) == 0:
        # print("Cant find target word in tokens")
        return None

    # TODO we drop all word if there are more that 2 target word occurence in example - its bad
    if len(target_word_indexes) > 1:
        # print("Skip for now")
        return None

    target_word_indexes = target_word_indexes[0][1]  # TODO: explain
    return pooling_strategy(
        hidden_states[target_word_indexes[0] : target_word_indexes[-1] + 1]
    )


def get_context_embedding(model, tokenizer, pooling_strategy, context, device):
    _, hidden_states_context = run_inference(model, tokenizer, context, device)
    return pooling_strategy(hidden_states_context)
