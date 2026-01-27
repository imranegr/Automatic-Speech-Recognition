import json
from datasets import load_dataset
from .text_cleaning import remove_special_characters, remove_digits, normalize

DATASET_NAME = "atlasia/DODa-audio-dataset"
TEXT_COLUMN = "darija_Arab_new"

def extract_all_chars(batch):
    all_text = " ".join(t for t in batch["text"] if t is not None)
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def build_vocab(output_path="vocab.json"):
    raw_dataset = load_dataset(DATASET_NAME, split="train[:30%]")

    raw_dataset = raw_dataset.map(remove_special_characters)
    raw_dataset = raw_dataset.map(remove_digits)
    raw_dataset = raw_dataset.map(normalize)

    vocabs = raw_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=8,
        remove_columns=raw_dataset.column_names
    )

    vocab_list = list(set(c for vocab_item in vocabs["vocab"] for c in vocab_item))

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(output_path, "w", encoding="utf-8") as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)

    print("Vocab saved to:", output_path)


if __name__ == "__main__":
    build_vocab()
