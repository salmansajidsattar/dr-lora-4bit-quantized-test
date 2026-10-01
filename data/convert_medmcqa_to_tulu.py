"""
convert_medmcqa_to_tulu.py
Convert the MedMCQA dataset to open-instruct/tulu messages format.

MedMCQA original format (one JSON per line):
{
    "question": "...",
    "exp": "...",          # explanation (may be empty)
    "cop": 1,              # correct option index: 1=opa, 2=opb, 3=opc, 4=opd
    "opa": "...",
    "opb": "...",
    "opc": "...",
    "opd": "...",
    "subject_name": "...",
    "topic_name": "...",
    "id": "...",
    "choice_type": "single"
}

Output format (one JSON per line):
{
    "messages": [
        {"role": "user",      "content": "<question>\\n\\nA. <opa>\\nB. <opb>\\nC. <opc>\\nD. <opd>"},
        {"role": "assistant", "content": "<answer_letter>. <answer_text>"}
    ],
    "dataset": "MedMCQA"
}

Usage:
  # Convert train split
  python convert_medmcqa_to_tulu.py \
      --input_path ./train.json \
      --output_path ./medmcqa_tulu_train.jsonl

  # Limit sample count (for quick debugging)
  python convert_medmcqa_to_tulu.py \
      --input_path ./train.json \
      --output_path ./medmcqa_tulu_1k.jsonl \
      --max_samples 1000

After conversion, update --dataset_mixer_list in your training script:
  --dataset_mixer_list /path/to/medmcqa_tulu_train.jsonl 1.0 \\
  --dataset_mixer_list_splits train \\
"""

import argparse
import json
import os
import sys

# Mapping from cop integer to option letter (MedMCQA uses 1-based integers)
COP_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}
LETTER_TO_OPT = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}


def build_user_content(example: dict) -> str:
    """
    Construct the user prompt from the question and four options.
    Format:
        <question>

        A. <opa>
        B. <opb>
        C. <opc>
        D. <opd>
    """
    question = example.get("question", "").strip()
    opa = example.get("opa", "").strip()
    opb = example.get("opb", "").strip()
    opc = example.get("opc", "").strip()
    opd = example.get("opd", "").strip()

    return (
        f"{question}\n\n"
        f"A. {opa}\n"
        f"B. {opb}\n"
        f"C. {opc}\n"
        f"D. {opd}"
    )


def build_assistant_content(example: dict) -> str:
    """
    Construct the assistant response from the correct option.
    Format: A. <answer_text>
    If the explanation (exp field) is non-empty, it is appended after the answer.
    """
    cop = example.get("cop")
    letter = COP_TO_LETTER.get(cop)
    if letter is None:
        return ""   # invalid cop; caller should skip this sample

    opt_key = LETTER_TO_OPT[letter]
    answer_text = example.get(opt_key, "").strip()
    answer = f"{letter}. {answer_text}"

    exp = example.get("exp", "").strip()
    if exp:
        answer = f"{answer}\n\n{exp}"

    return answer


def convert(example: dict) -> dict | None:
    """
    Convert a single MedMCQA sample to tulu messages format.
    Returns None if the sample should be skipped.
    """
    if not example.get("question", "").strip():
        return None
    if example.get("cop") not in COP_TO_LETTER:
        return None
    for opt in ("opa", "opb", "opc", "opd"):
        if not example.get(opt, "").strip():
            return None

    assistant_content = build_assistant_content(example)
    if not assistant_content:
        return None

    return {
        "messages": [
            {"role": "user",      "content": build_user_content(example)},
            {"role": "assistant", "content": assistant_content},
        ],
        "dataset": "MedMCQA",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert MedMCQA -> tulu messages format"
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the MedMCQA source json file (e.g. train.json).",
    )
    parser.add_argument(
        "--output_path", type=str, default="./medmcqa_tulu.jsonl",
        help="Output jsonl file path (default: ./medmcqa_tulu.jsonl).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to output (default: all).",
    )
    parser.add_argument(
        "--no_exp", action="store_true",
        help="Do not append the explanation (exp field) to the assistant response.",
    )
    args = parser.parse_args()

    # ── Load input file ───────────────────────────────────────────────────────
    if not os.path.exists(args.input_path):
        sys.exit(f"[Error] Input file not found: {args.input_path}")

    print(f"[1/3] Loading {args.input_path} ...")
    examples = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"      [Warning] JSON parse error, skipping line: {e}")

    print(f"      Loaded {len(examples)} samples")

    if args.max_samples is not None and args.max_samples < len(examples):
        examples = examples[: args.max_samples]
        print(f"      Truncated to {args.max_samples} samples")

    # ── Convert ───────────────────────────────────────────────────────────────
    print(f"[2/3] Converting (no_exp={args.no_exp})...")

    if args.no_exp:
        for ex in examples:
            ex["exp"] = ""

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    written = 0
    skipped = 0
    print(f"[3/3] Writing to {args.output_path} ...")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for example in examples:
            record = convert(example)
            if record is None:
                skipped += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone! Written {written} samples, skipped {skipped} invalid samples.")
    print(f"   Output file: {os.path.abspath(args.output_path)}")

    # ── Print next steps ───────────────────────────────────────────────────────
    abs_path = os.path.abspath(args.output_path)
    print(f"""
   ── Next step: update your training script ───────────────────────────────
   Replace --dataset_mixer_list with:

     --dataset_mixer_list {abs_path} 1.0 \\
     --dataset_mixer_list_splits train \\

   Recommended parameter adjustments for medical QA:
     --num_train_epochs 3          # MedMCQA train set ~182K samples; adjust as needed
     --learning_rate 2e-5          # keep unchanged
     --max_seq_length 1024         # MC samples are short; 512-1024 is usually sufficient
   ────────────────────────────────────────────────────────────────────────""")


if __name__ == "__main__":
    main()