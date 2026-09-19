"""Generate missing Ukrainian WSD examples with the official OpenAI API."""

import argparse
import json
import os
import re
from typing import Any

from openai import OpenAI
from tqdm import tqdm


DEFAULT_INPUT_FILE = (
    "local_datasets/semi_supervised_2/"
    "lemmas_with_meanings_and_sentences_mpnet_filtered.json"
)
DEFAULT_OUTPUT_FILE = (
    "local_datasets/semi_supervised_2/generated_sentences.jsonl"
)
DEFAULT_MIN_SENTENCES = 5
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_SEED = 42


GENERATE_PROMPT = """Ти експерт з української мови, зокрема з лексикографії.

Створи {to_generate} нових, природних і граматично правильних українських
речень для слова «{lemma}», використовуючи значення нижче:
{glosses}

Не повторюй уже наявні приклади:
{existing_sentences}

Кожне речення повинно ілюструвати саме це значення слова.
Поверни лише маркований список речень, без пояснень.
Наприклад:
- Перше речення.
- Друге речення.

Нові речення:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate missing Ukrainian WSD examples with OpenAI."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=DEFAULT_MIN_SENTENCES,
        help="Generate examples until every meaning has this many sentences.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def sentence_text(item: Any) -> str:
    """Extract sentence text from either the current dict format or a string."""
    if isinstance(item, dict):
        return str(item.get("sentence", "")).strip()
    return str(item).strip()


def parse_generated_sentences(output: str, limit: int) -> list[str]:
    """Parse bullet/numbered lines and return unique non-empty sentences."""
    sentences = []
    seen = set()

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Accept -, *, •, and numbered lists such as "1." or "2)".
        sentence = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if sentence == line or not sentence:
            continue

        normalized = " ".join(sentence.split()).casefold()
        if normalized not in seen:
            seen.add(normalized)
            sentences.append(sentence)

        if len(sentences) >= limit:
            break

    return sentences


def generate_sentences(
    client: OpenAI,
    model: str,
    lemma: str,
    meaning_entry: dict[str, Any],
    to_generate: int,
    seed: int,
) -> list[str]:
    meaning = meaning_entry.get("meaning", {})
    glosses_list = [str(gloss).strip() for gloss in meaning.get("gloss", [])]
    glosses_list = [gloss for gloss in glosses_list if gloss]

    if not glosses_list:
        return []

    existing_sentences = [
        sentence_text(item) for item in meaning_entry.get("sentences", [])
    ]
    existing_sentences = [sentence for sentence in existing_sentences if sentence]

    glosses = "\n".join(f"- {gloss}" for gloss in glosses_list)
    existing = "\n".join(f"- {sentence}" for sentence in existing_sentences)
    if not existing:
        existing = "- Немає наявних прикладів."

    prompt = GENERATE_PROMPT.format(
        lemma=lemma,
        glosses=glosses,
        existing_sentences=existing,
        to_generate=to_generate,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        seed=seed,
    )
    output = response.choices[0].message.content or ""
    return parse_generated_sentences(output, to_generate)


def main() -> None:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running this script."
        )

    client = OpenAI()

    with open(args.input_file, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    meanings_to_generate = sum(
        1
        for meanings in data.values()
        for meaning_entry in meanings.values()
        if len(meaning_entry.get("sentences", [])) < args.min_sentences
    )

    print(f"Using OpenAI model: {args.model}")
    print(f"Meanings requiring generation: {meanings_to_generate}")
    print(f"Writing output to: {args.output_file}")

    # Write a fresh file so rerunning does not append duplicate records.
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        with tqdm(total=meanings_to_generate, desc="Generating sentences") as pbar:
            for lemma, meanings in data.items():
                generated_for_lemma = {}

                for meaning_id, meaning_entry in meanings.items():
                    current_count = len(meaning_entry.get("sentences", []))
                    to_generate = max(0, args.min_sentences - current_count)
                    if to_generate == 0:
                        continue

                    glosses = meaning_entry.get("meaning", {}).get("gloss", [])
                    if not glosses:
                        print(f"Skipping {lemma}/{meaning_id}: no gloss found")
                        pbar.update(1)
                        continue

                    print(
                        f"Generating {to_generate} sentence(s) for "
                        f"lemma={lemma!r}, meaning={meaning_id!r}"
                    )

                    new_sentences = generate_sentences(
                        client=client,
                        model=args.model,
                        lemma=lemma,
                        meaning_entry=meaning_entry,
                        to_generate=to_generate,
                        seed=args.seed,
                    )

                    generated_for_lemma[meaning_id] = {
                        "meaning": meaning_entry["meaning"],
                        "sentences": new_sentences,
                    }
                    print(f"Generated: {new_sentences}")
                    pbar.update(1)

                if generated_for_lemma:
                    output_file.write(
                        json.dumps(
                            {lemma: generated_for_lemma},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    output_file.flush()


if __name__ == "__main__":
    main()
