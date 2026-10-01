"""
convert_metamathqa_to_tulu.py
Convert meta-math/MetaMathQA to open-instruct/tulu messages format,
with optional filtering to keep only the GSM8K subset.

Output format (one JSON per line):
{
    "messages": [
        {"role": "user",      "content": "<question>"},
        {"role": "assistant", "content": "<answer>"}
    ],
    "dataset": "MetaMathQA",
    "type": "GSM8K_FOBAR"
}

Usage:
  # Full conversion (~395K samples)
  python convert_metamathqa_to_tulu.py \
      --output_path ./metamathqa_tulu_full.jsonl

  # GSM8K subset only (recommended for gsm8k tasks, ~119K samples)
  python convert_metamathqa_to_tulu.py \
      --gsm8k_only \
      --output_path ./metamathqa_tulu_gsm8k.jsonl

  # Limit sample count (for quick debugging)
  python convert_metamathqa_to_tulu.py \
      --gsm8k_only \
      --max_samples 10000 \
      --output_path ./metamathqa_tulu_gsm8k_10k.jsonl

After conversion, update --dataset_mixer_list in your training script:
  --dataset_mixer_list /path/to/metamathqa_tulu_gsm8k.jsonl 1.0 \
  --dataset_mixer_list_splits train \
"""

import argparse
import json
import os
import sys

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    from datasets import load_dataset
except ImportError:
    sys.exit("[Error] Please install datasets first: pip install datasets")


# ── GSM8K-related type prefixes ───────────────────────────────────────────────
# Examples of MetaMathQA type field values:
#   GSM8K_Backward       GSM8K_FOBAR        GSM8K_Rephrased
#   GSM8K_SV             MATH_Backward      MATH_FOBAR  ...
GSM8K_TYPE_PREFIXES = ("GSM_",)  # [fix] real MetaMathQA types: GSM_AnsAug, GSM_Rephrased, GSM_SV, GSM_FOBAR


def is_gsm8k(example: dict) -> bool:
    """Check whether a sample belongs to the GSM8K series."""
    return any(example.get("type", "").startswith(p) for p in GSM8K_TYPE_PREFIXES)


def clean_response(response: str) -> str:
    """
    MetaMathQA responses typically end with "The answer is: X".
    We keep the full CoT reasoning chain here to help the model learn step-by-step reasoning.
    Truncate here if you only want the final answer.
    """
    return response.strip()


def convert(example: dict) -> dict:
    """Convert a single MetaMathQA sample to tulu messages format."""
    return {
        "messages": [
            {"role": "user",      "content": example["query"].strip()},
            {"role": "assistant", "content": clean_response(example["response"])},
        ],
        "dataset": "MetaMathQA",
        "type": example.get("type", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert MetaMathQA → tulu messages format")
    parser.add_argument(
        "--output_path", type=str, default="./metamathqa_tulu_gsm8k.jsonl",
        help="Output jsonl file path (default: ./metamathqa_tulu_gsm8k.jsonl)",
    )
    parser.add_argument(
        "--gsm8k_only", action="store_true",
        help="Keep only GSM8K-related samples (type starts with 'GSM8K')",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to output (default: all)",
    )
    parser.add_argument(
        "--hf_cache_dir", type=str, default=None,
        help="HuggingFace dataset cache directory (optional)",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Which split to use (default: train)",
    )
    args = parser.parse_args()

    # ── Load dataset ───────────────────────────────────────────────────────────
    print(f"[1/3] Loading meta-math/MetaMathQA from HuggingFace ({args.split} split)...")
    load_kwargs = {"split": args.split}
    if args.hf_cache_dir:
        load_kwargs["cache_dir"] = args.hf_cache_dir
    try:
        dataset = load_dataset("meta-math/MetaMathQA", **load_kwargs)
    except Exception as e:
        sys.exit(f"[Error] Failed to load dataset: {e}\n"
                 "Please check your network connection or specify a local cache with --hf_cache_dir.")

    # ── Filter ─────────────────────────────────────────────────────────────────
    if args.gsm8k_only:
        before = len(dataset)
        dataset = dataset.filter(is_gsm8k, num_proc=4)
        print(f"[2/3] GSM8K filtering: {before} → {len(dataset)} samples")
    else:
        print(f"[2/3] Full conversion: {len(dataset)} samples (no filtering)")

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
            if not example.get("query", "").strip() or not example.get("response", "").strip():
                skipped += 1
                continue
            record = convert(example)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone! Written {written} samples, skipped {skipped} empty samples.")
    print(f"   Output file: {os.path.abspath(args.output_path)}")

    # ── Print type distribution ────────────────────────────────────────────────
    from collections import Counter
    type_counter: Counter = Counter()
    with open(args.output_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            type_counter[obj.get("type", "unknown")] += 1
    print("\n   Type distribution (Top 10):")
    for t, cnt in type_counter.most_common(10):
        print(f"     {t:35s} {cnt:>7,}")

    # ── Print next steps ───────────────────────────────────────────────────────
    abs_path = os.path.abspath(args.output_path)
    print(f"""
   ── Next step: update your training script ───────────────────────────────
   Replace --dataset_mixer_list with:

     --dataset_mixer_list {abs_path} 1.0 \\
     --dataset_mixer_list_splits train \\

   Recommended parameter adjustments for math reasoning:
     --num_train_epochs 3          # 2-3 epochs is sufficient for MetaMathQA
     --learning_rate 2e-5          # keep unchanged
     --max_seq_length 1024         # MetaMathQA answers are short; reduce from 4096
   ────────────────────────────────────────────────────────────────────────""")


if __name__ == "__main__":
    main()