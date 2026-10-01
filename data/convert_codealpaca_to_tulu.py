"""
convert_codealpaca_to_tulu.py
Convert sahil2801/CodeAlpaca-20k to open-instruct/tulu messages format.

Output format (one JSON per line):
{
    "messages": [
        {"role": "user",      "content": "<instruction>[\\n<input>]"},
        {"role": "assistant", "content": "<output>"}
    ],
    "dataset": "CodeAlpaca"
}

Usage:
  # Full conversion (~20K samples)
  python convert_codealpaca_to_tulu.py \
      --output_path ./codealpaca_tulu.jsonl

  # Limit sample count (for quick debugging)
  python convert_codealpaca_to_tulu.py \
      --max_samples 1000 \
      --output_path ./codealpaca_tulu_1k.jsonl

After conversion, update --dataset_mixer_list in your training script:
  --dataset_mixer_list /path/to/codealpaca_tulu.jsonl 1.0 \\
  --dataset_mixer_list_splits train \\
"""

import argparse
import json
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("[Error] Please install datasets first: pip install datasets")


def build_user_content(instruction: str, input_text: str) -> str:
    """
    CodeAlpaca has three fields: instruction, input, output.
    When input is non-empty, it is appended after instruction,
    consistent with the original Alpaca format.
    """
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def convert(example: dict) -> dict:
    user_content = build_user_content(
        example.get("instruction", ""),
        example.get("input", ""),
    )
    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": example.get("output", "").strip()},
        ],
        "dataset": "CodeAlpaca",
    }


def main():
    parser = argparse.ArgumentParser(description="Convert CodeAlpaca-20k -> tulu messages format")
    parser.add_argument(
        "--output_path", type=str, default="./codealpaca_tulu.jsonl",
        help="Output jsonl file path (default: ./codealpaca_tulu.jsonl).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to output (default: all).",
    )
    parser.add_argument(
        "--hf_cache_dir", type=str, default=None,
        help="HuggingFace dataset cache directory (optional).",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Which split to use (default: train).",
    )
    args = parser.parse_args()

    # ── Load dataset ───────────────────────────────────────────────────────────
    print(f"[1/3] Loading sahil2801/CodeAlpaca-20k from HuggingFace ({args.split} split)...")
    load_kwargs = {"split": args.split}
    if args.hf_cache_dir:
        load_kwargs["cache_dir"] = args.hf_cache_dir
    try:
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", **load_kwargs)
    except Exception as e:
        sys.exit(f"[Error] Failed to load dataset: {e}\n"
                 "Please check your network connection or specify a local cache with --hf_cache_dir.")

    print(f"[2/3] Full conversion: {len(dataset)} samples")

    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        print(f"      Truncated to {args.max_samples} samples")

    # ── Write output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print(f"[3/3] Writing to {args.output_path} ...")

    written = 0
    skipped = 0
    with open(args.output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            if not example.get("instruction", "").strip() or not example.get("output", "").strip():
                skipped += 1
                continue
            record = convert(example)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone! Written {written} samples, skipped {skipped} empty samples.")
    print(f"   Output file: {os.path.abspath(args.output_path)}")

    # ── Print next steps ───────────────────────────────────────────────────────
    abs_path = os.path.abspath(args.output_path)
    print(f"""
   ── Next step: update your training script ───────────────────────────────
   Replace --dataset_mixer_list with:

     --dataset_mixer_list {abs_path} 1.0 \\
     --dataset_mixer_list_splits train \\

   Recommended parameter adjustments for code generation:
     --num_train_epochs 3          # CodeAlpaca is small (20K); more epochs help
     --learning_rate 2e-5          # keep unchanged
     --max_seq_length 2048         # code samples can be long; increase accordingly
   ────────────────────────────────────────────────────────────────────────""")


if __name__ == "__main__":
    main()