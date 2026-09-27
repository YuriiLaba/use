import json
from services.config import get_value

COLLECTED_SENTENCES_PATH = get_value(
    "paths", "filtered_grouped",
    "local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet_filtered.json",
)
GENERATED_SENTENCES_PATH = get_value(
    "paths", "generated_sentences",
    "local_datasets/semi_supervised_2/generated_sentences.jsonl",
)
OUTPUT_PATH = get_value(
    "paths", "merged_dataset",
    "local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json",
)


def main():
    with open(COLLECTED_SENTENCES_PATH, "r", encoding="utf-8") as collected_file:
        collected_data = json.load(collected_file)

    with open(GENERATED_SENTENCES_PATH, "r", encoding="utf-8") as generated_file:
        for line in generated_file:
            item = json.loads(line)

            for lemma, meanings in item.items():
                for meaning, meaning_values in meanings.items():
                    sentences = meaning_values["sentences"]

                    if lemma not in collected_data:
                        print(
                            f"WARNING: Lemma '{lemma}' from generated data not found in collected data."
                        )
                        continue

                    if meaning not in collected_data[lemma]:
                        print(
                            f"WARNING: Meaning ID '{meaning}' for lemma '{lemma}' not found in collected data."
                        )
                        continue

                    for sentence in sentences:
                        collected_data[lemma][meaning]["sentences"].append(
                            {
                                "sentence": sentence,
                                "source": "generated",
                                "similarity": None,
                                "probability": None,
                            }
                        )

                    if len(collected_data[lemma][meaning]["sentences"]) < 5:
                        print(
                            f"WARNING: Less than 5 sentences for lemma '{lemma}' meaning ID '{meaning}' after merging."
                        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        json.dump(collected_data, output_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
