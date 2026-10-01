"""
convert_ledgar_to_mc.py
Convert LEDGAR test.jsonl from classification format to four-choice multiple-choice format.

Each question:
  - 1 correct answer (original label)
  - 3 distractors (randomly sampled from other labels, independently per question)
  - Options are shuffled randomly; correct answer position is random
  - Assistant response is the correct option letter (A/B/C/D)

Usage:
  python convert_ledgar_to_mc.py \
      --input_jsonl data/LEDGAR/test.jsonl \
      --output_jsonl data/LEDGAR/test_mc.jsonl \
      --seed 42
"""

import argparse
import json
import random


PROMPT_TEMPLATE = (
    "Classify the following contract provision: {text}\n"
    "The topic of this contract provision is:\n"
    "{choices}\n"
    "Answer:"
)

LETTERS = ["A", "B", "C", "D"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl",  default="data/LEDGAR/test.jsonl")
    p.add_argument("--output_jsonl", default="data/LEDGAR/test_mc.jsonl")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load original test.jsonl and collect all labels
    examples = []
    all_labels = set()
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Extract original text from user prompt
            # Prompt format: "Classify the following contract provision: {text}\nLabel:"
            user_content = obj["messages"][0]["content"]
            text = user_content.replace("Classify the following contract provision: ", "").replace("\nLabel:", "").strip()
            label = obj.get("subset", "")
            examples.append({"text": text, "label": label})
            all_labels.add(label)

    all_labels = sorted(all_labels)
    print(f"Loaded: {len(examples):,} samples  |  labels: {len(all_labels)}")

    written = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for ex in examples:
            text       = ex["text"]
            true_label = ex["label"]

            # Sample 3 wrong options from other labels
            other_labels = [l for l in all_labels if l != true_label]
            wrong_labels = random.sample(other_labels, 3)

            # Shuffle all four options
            options = [true_label] + wrong_labels
            random.shuffle(options)

            # Find the letter corresponding to the correct answer
            correct_letter = LETTERS[options.index(true_label)]

            # Build choices string
            choices_str = "\n".join(
                f"{LETTERS[i]}) {options[i]}" for i in range(4)
            )

            user_content = PROMPT_TEMPLATE.format(
                text=text,
                choices=choices_str,
            )

            record = {
                "messages": [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": correct_letter},
                ],
                "dataset":        "ledgar_mc",
                "subset":         true_label,
                "correct_letter": correct_letter,
                "options":        {LETTERS[i]: options[i] for i in range(4)},
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done! Written {written:,} samples -> {args.output_jsonl}")
    print(f"\nExample:")
    with open(args.output_jsonl) as f:
        ex = json.loads(f.readline())
    print(ex["messages"][0]["content"])
    print(f"Answer: {ex['messages'][1]['content']}")


if __name__ == "__main__":
    main()